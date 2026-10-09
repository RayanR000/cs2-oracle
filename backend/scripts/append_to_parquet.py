#!/usr/bin/env python3
"""
Daily: append today's aggregator rows to the current year's Parquet files.

Writes two Parquet files (prices partitioned by month to stay well under
GitHub's 100MB-per-file limit; exchange-rates stays yearly, it's tiny):
  prices-YYYY-MM.parquet       — item_slug, day, source, mean_price, volume,
                                 ingested_at
  exchange-rates-YYYY.parquet  — Currency exchange rates (flat: currency, rate, day)

`ingested_at` is this run's wall clock, not `--date`: it records when the row
arrived, which is the whole point of having it beside a `day` that says what the
row describes. Backdating `--date` to re-export an old day therefore writes an
old `day` with a present-day arrival, which is the truth. Rows written before
2026-08-08 have none and never will. See `db/archive.py`.

`median_price`, `min_price` and `max_price` are deliberately NOT written. The
snapshot CSV carries one row per (item_slug, day, source) and has no median
column (`collectors/pipeline.py:276`), so all three were exact copies of
`mean_price` — 45% of the archive's bytes for no information. No reader ever
selected them. See docs/changelog/2026-08-06-price-archive-compaction.md.

`snapshots-YYYY-MM.parquet` is likewise retired: it was
`SELECT item_slug, day, source, mean_price AS price, volume` off the prices
file and nothing read it. `scripts/compact_price_archive.py` removed both.

Input: a snapshot CSV written by the aggregator (or Supabase + backfilled CSV for backward compat).

Usage:
    python scripts/append_to_parquet.py --date 2026-07-08 --out-dir ../archive \\
        --snapshot-csv /tmp/aggregator-snapshots-2026-07-08.csv
"""

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import engine
from db.archive import canonical_order


def _sum_observed(s: pd.Series):
    """Sum a volume group, keeping an all-absent group absent.

    A bare `.sum()` returns 0 for an all-NaN group, which is the whole bug this
    guards: the aggregator feeds carry no volume field, so every live row is
    *unknown*, and 0 would present that as an observed zero. A real zero never
    occurs -- a day with no sale produces an absent row, not a zero one -- so a
    0 here is always fabricated. `min_count=1` keeps one observed value enough
    to produce a sum and no observed values NA.
    """
    return s.sum(min_count=1)


FETCH_TODAY_SQL = """
    SELECT i.item_id AS item_slug,
           DATE(ph.timestamp) AS day,
           ph.price,
           ph.volume,
           ph.median_price,
           ph.source
    FROM price_history ph
    JOIN items i ON i.id = ph.item_id
    WHERE ph.timestamp >= :day_start AND ph.timestamp < :day_end
      AND ph.source IN ('aggregator_sync', 'steam_batch')
    ORDER BY i.item_id, ph.timestamp
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="UTC day to export (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--out-dir",
        default="../archive",
        help="Archive root; price-archive lives under this",
    )
    parser.add_argument(
        "--snapshot-csv",
        help="CSV of all-source snapshot prices (item_slug, day, source, price, volume)",
    )
    parser.add_argument(
        "--backfilled-csv",
        help="Backward compat: CSV of backfilled item Steam 24h prices (item_slug, day, price, volume)",
    )
    parser.add_argument(
        "--exchange-rates-csv",
        help="CSV of currency exchange rates (currency, rate, day)",
    )
    args = parser.parse_args()

    day_start = datetime.strptime(args.date, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)
    year = day_start.year
    # Current-year prices/snapshots are partitioned by month so the hot file
    # stays small (a day belongs to exactly one month). See
    # docs/specs/2026-07-25-monthly-parquet-partitioning-design.md
    ym = f"{year}-{day_start.month:02d}"
    out_dir = Path(args.out_dir) / "price-archive"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    snapshots_df = None
    legacy_frames = []

    if args.snapshot_csv:
        csv_path = Path(args.snapshot_csv)
        if csv_path.exists():
            snapshots_df = pd.read_csv(csv_path)
            if snapshots_df.empty:
                print(f"Warning: snapshot CSV {csv_path.name} is empty — no snapshots to write")
            else:
                snapshots_df["day"] = pd.to_datetime(snapshots_df["day"])
                print(f"Read {len(snapshots_df)} snapshot rows from {csv_path.name}")
        else:
            print(f"Warning: --snapshot-csv path does not exist: {csv_path} — skipping snapshot Parquet")
    else:
        # Legacy path: read from Supabase + backfilled CSV
        with engine.connect() as conn:
            snapshot_rows = conn.execute(
                text(FETCH_TODAY_SQL),
                {"day_start": day_start, "day_end": day_end},
            ).fetchall()

        if snapshot_rows:
            legacy_frames.append(
                pd.DataFrame(
                    snapshot_rows,
                    columns=["item_slug", "day", "price", "volume", "median_price", "source"],
                )
            )

        if args.backfilled_csv:
            csv_path = Path(args.backfilled_csv)
            if csv_path.exists():
                backfilled_df = pd.read_csv(csv_path)
                if not backfilled_df.empty:
                    backfilled_df["median_price"] = None
                    if "source" not in backfilled_df.columns:
                        backfilled_df["source"] = "aggregator_sync"
                    legacy_frames.append(backfilled_df)
                    print(f"Read {len(backfilled_df)} backfilled rows from {csv_path.name}")
                else:
                    print(f"Warning: backfilled CSV {csv_path.name} is empty")
            else:
                print(f"Warning: --backfilled-csv path does not exist: {csv_path} — skipping OHLCV Parquet")

    # ── Write prices-YYYY-MM.parquet (OHLCV, all sources) ──────────────────
    arrived = datetime.now(UTC).replace(tzinfo=None)

    if snapshots_df is not None and not snapshots_df.empty:
        daily = (
            snapshots_df.groupby(["item_slug", "day", "source"])
            .agg(
                mean_price=("price", "mean"),
                volume=("volume", _sum_observed),
            )
            .reset_index()
        )
        daily["volume"] = daily["volume"].astype("Int64")
        daily["day"] = pd.to_datetime(daily["day"])
        daily["ingested_at"] = arrived
        _append_parquet(out_dir / f"prices-{ym}.parquet", daily, PRICE_KEYS)
        print(f"Appended {len(daily)} OHLCV rows to prices-{ym}.parquet")

    if legacy_frames:
        df = pd.concat(legacy_frames, ignore_index=True)
        daily = (
            df.groupby(["item_slug", "day", "source"])
            .agg(
                mean_price=("price", "mean"),
                volume=("volume", _sum_observed),
            )
            .reset_index()
        )
        daily["volume"] = daily["volume"].astype("Int64")
        daily["day"] = pd.to_datetime(daily["day"])
        daily["ingested_at"] = arrived
        _append_parquet(out_dir / f"prices-{ym}.parquet", daily, PRICE_KEYS)
        print(f"Appended {len(daily)} OHLCV rows to prices-{ym}.parquet (legacy path)")

    # ── Write exchange-rates-YYYY.parquet ────────────────────────────
    if args.exchange_rates_csv:
        csv_path = Path(args.exchange_rates_csv)
        if csv_path.exists():
            rates_df = pd.read_csv(csv_path)
            if not rates_df.empty:
                rates_df["day"] = pd.to_datetime(rates_df["day"])
                _append_parquet(out_dir / f"exchange-rates-{year}.parquet", rates_df, ["currency", "day"])
                print(f"Appended {len(rates_df)} exchange rate rows to exchange-rates-{year}.parquet")
            else:
                print(f"Warning: exchange_rates CSV {csv_path.name} is empty")
        else:
            print(f"Warning: --exchange-rates-csv path does not exist: {csv_path} — skipping exchange-rates Parquet")

    if not legacy_frames and snapshots_df is None:
        print(f"No data found for {args.date}")
        sys.exit(0)

    print(f"Done: {args.date}")


#: Dedup key of the price files, in sort order: grouping a source's rows together
#: is what lets ZSTD compress the slugs and prices (see `_write_parquet`).
PRICE_KEYS = ["source", "item_slug", "day"]

#: Written as Parquet DATE, not TIMESTAMP. Every `day` in the archive is
#: midnight-truncated, so the time component was only ever a source of type
#: drift between files — `prices-2013..2025` hold TIMESTAMP and the monthly
#: files TIMESTAMP_NS, which a glob read has to reconcile at every call site.
DATE_COLUMNS = ("day",)


def _write_parquet(con, path: Path, relation: str, columns: list[str], order_by: list[str]):
    """COPY *relation* to *path* in canonical column order with DATE-typed day columns.

    `day` is cast to DATE because pandas has no date dtype: a datetime64 column
    always lands as TIMESTAMP. Rows are sorted on *order_by* and written ZSTD:
    `ORDER BY source, item_slug, day` with ZSTD measured 168 MB -> 104 MB across
    the price files, against SNAPPY unsorted (performance review 2026-10-08 §4).
    Readers never depend on row order. Written to a temp file and moved into
    place so a crash mid-write cannot truncate the month's archive.
    """
    ordered = canonical_order(columns)
    projection = ", ".join(f'CAST("{c}" AS DATE) AS "{c}"' if c in DATE_COLUMNS else f'"{c}"' for c in ordered)
    order = ", ".join(f'"{c}"' for c in order_by)
    tmp = path.with_suffix(".parquet.tmp")
    escaped = str(tmp).replace("'", "''")
    try:
        con.sql(
            f"COPY (SELECT {projection} FROM ({relation}) ORDER BY {order}) "
            f"TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _append_parquet(path: Path, new_data: pd.DataFrame, dedup_keys: list):
    """Append new_data to an existing Parquet file, deduplicating on dedup_keys.

    The merge runs in DuckDB: existing rows with no match in the batch, plus the
    batch. Until 2026-10-09 it round-tripped the whole month through pandas,
    which measured 6.4 s and 5.7 GB peak against 0.7 s and 0.46 GB for this
    anti-join, with an identical row set; peak memory neared 7 GB at month end.

    Semantics are the pandas version's `concat` + `drop_duplicates(keep="last")`:
    a batch row replaces a matching existing row wholesale, the last of any
    duplicates within the batch wins, and keys match NULL to NULL
    (`dropna=False`), hence `IS NOT DISTINCT FROM`.

    First arrival wins on `ingested_at`. Replacing the row wholesale is right for
    a corrected price and wrong for `ingested_at`: the value being corrected
    still became knowable on the day it first landed, and re-stamping it would
    date the whole month forward on any re-run. `least` skips NULL, so a row that
    predates the column takes the new timestamp rather than staying unknown.
    """
    con = duckdb.connect()
    try:
        new_cols = list(new_data.columns)
        # Position column: "last in the batch wins" needs the input order, which a
        # parallel scan does not guarantee to preserve.
        con.register("_new_raw", new_data.assign(_pos=range(len(new_data))))
        keys = ", ".join(f'"{k}"' for k in dedup_keys)
        day_casts = ", ".join(f'CAST("{c}" AS DATE) AS "{c}"' for c in DATE_COLUMNS if c in new_cols)
        replace = f" REPLACE ({day_casts})" if day_casts else ""
        con.sql(
            f"CREATE TEMP TABLE _new AS SELECT *{replace} FROM _new_raw "
            f"QUALIFY row_number() OVER (PARTITION BY {keys} ORDER BY _pos DESC) = 1"
        )
        con.unregister("_new_raw")

        if not path.exists():
            _write_parquet(con, path, "SELECT * EXCLUDE (_pos) FROM _new", new_cols, dedup_keys)
            print(f"  {path.name}: {len(new_data)} written (new file)")
            return

        escaped = str(path).replace("'", "''")
        con.sql(f"CREATE TEMP VIEW _old AS SELECT * FROM read_parquet('{escaped}')")
        old_cols = [r[0] for r in con.sql("DESCRIBE _old").fetchall()]

        # A key the file predates (`source`, before normalize_price_schema.py) reads
        # as NULL, the value the pandas version materialised for it. NULL, not a
        # label: `init_local_db.py` derives is_backfilled from the equivalence of
        # `source IS NULL` and `day < '2026-01-01'`.
        def old_key(k):
            return f'o."{k}"' if k in old_cols else "NULL"

        match = " AND ".join(f'{old_key(k)} IS NOT DISTINCT FROM n."{k}"' for k in dedup_keys)
        kept = f"SELECT o.* FROM _old o WHERE NOT EXISTS (SELECT 1 FROM _new n WHERE {match})"
        if "ingested_at" in new_cols and "ingested_at" in old_cols:
            first = f"least(n.ingested_at, (SELECT min(o.ingested_at) FROM _old o WHERE {match}))"
            incoming = f"SELECT n.* EXCLUDE (_pos) REPLACE ({first} AS ingested_at) FROM _new n"
        else:
            incoming = "SELECT n.* EXCLUDE (_pos) FROM _new n"
        columns = old_cols + [c for c in new_cols if c not in old_cols]
        _write_parquet(con, path, f"{kept} UNION ALL BY NAME {incoming}", columns, dedup_keys)
        total = con.sql(f"SELECT count(*) FROM read_parquet('{escaped}')").fetchone()[0]
        print(f"  {path.name}: {len(new_data)} appended, {total} total")
    finally:
        con.close()


if __name__ == "__main__":
    main()
