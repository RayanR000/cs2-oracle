#!/usr/bin/env python3
"""Record which already-published price rows a publish changed, dropped or back-filled.

The data repo keeps no history (each publish is an orphan commit + force-push), so a revised
day could never be diffed afterwards: the 2026-08-25 anchor audit could not tell whether a
day's prices had moved (docs/changelog/2026-08-25-centre-shrinks-to-zero-and-the-dollar-band-is-the-wedge.md).
Keeping the commits would cost ~70-100 MB a day, because Parquet does not delta-compress.
This logs the difference instead, at KB scale.

It compares `prices-YYYY-MM.parquet` as checked out (`--before-dir`, copied before the append)
with the same files after the day's writers ran (`--archive-dir`), restricted to the trailing
`--window-days`, and appends to `ops/price_revisions.parquet`:

- `changed`    key in both, `mean_price` or `volume` differs
- `removed`    key in the before copy, gone now (a stale-checkout publish does exactly this)
- `late_added` key only in the after copy, for a day on or before the newest day already
               published (the routine new-day rows are not revisions and are not logged)

A re-run for the same `--run-date` replaces its own rows, so it is idempotent. When one run
would log more than `--max-detail-rows` rows (a whole-archive backfill), it logs one
aggregate row per (day, source, kind) with `item_slug` NULL and `n_items` set, so the log
stays small.

Usage (from backend/):
    venv/bin/python -m scripts.log_price_revisions --before-dir /tmp/prices-before \\
        --archive-dir ../archive/price-archive --run-date 2026-09-30
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LOG_NAME = "price_revisions.parquet"
MONTHLY = "prices-????-??.parquet"
KEY = ("item_slug", "day", "source")

SCHEMA = pa.schema(
    [
        ("run_date", pa.date32()),
        ("detected_at", pa.timestamp("us")),
        ("item_slug", pa.string()),
        ("day", pa.date32()),
        ("source", pa.string()),
        ("kind", pa.string()),
        ("n_items", pa.int64()),
        ("prev_mean_price", pa.float64()),
        ("new_mean_price", pa.float64()),
        ("prev_volume", pa.int64()),
        ("new_volume", pa.int64()),
    ]
)


def _months(start: dt.date, end: dt.date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    return out


def _files(directory: Path, window_start: dt.date | None, run_date: dt.date) -> list[str]:
    if window_start is None:
        found = sorted(directory.glob(MONTHLY))
    else:
        found = [directory / f"prices-{ym}.parquet" for ym in _months(window_start, run_date)]
        found = [p for p in found if p.exists()]
    return [str(p) for p in found]


def _relation(con, files: list[str], name: str, window_start: dt.date | None) -> None:
    if not files:
        con.execute(
            f"CREATE TEMP VIEW {name} AS SELECT NULL::VARCHAR AS item_slug, NULL::DATE AS day, "
            "NULL::VARCHAR AS source, NULL::DOUBLE AS mean_price, NULL::BIGINT AS volume WHERE FALSE"
        )
        return
    where = "" if window_start is None else f"WHERE day >= DATE '{window_start}'"
    con.execute(
        f"CREATE TEMP VIEW {name} AS SELECT item_slug, CAST(day AS DATE) AS day, source, mean_price, "
        f"CAST(volume AS BIGINT) AS volume FROM read_parquet({files!r}, union_by_name=true) {where}"
    )


def diff_prices(before_dir, archive_dir, run_date: dt.date, window_days: int | None) -> pa.Table:
    """Row-level differences between two copies of the monthly price files, without run metadata."""
    before_dir, archive_dir = Path(before_dir), Path(archive_dir)
    window_start = None if window_days is None else run_date - dt.timedelta(days=window_days)
    con = duckdb.connect()
    _relation(con, _files(before_dir, window_start, run_date), "b", window_start)
    _relation(con, _files(archive_dir, window_start, run_date), "a", window_start)
    join = "b.item_slug = a.item_slug AND b.day = a.day AND b.source IS NOT DISTINCT FROM a.source"
    newest = con.execute("SELECT max(day) FROM b").fetchone()[0]
    # Both measures are compared NULL-safely: a price going NULL -> value is a revision.
    differs = (
        "(abs(b.mean_price - a.mean_price) > 1e-9 OR (b.mean_price IS NULL) != (a.mean_price IS NULL) "
        "OR b.volume IS DISTINCT FROM a.volume)"
    )
    cols = (
        "{s}.item_slug, {s}.day, {s}.source, b.mean_price AS prev_mean_price, a.mean_price AS new_mean_price, "
        "b.volume AS prev_volume, a.volume AS new_volume"
    )
    queries = [
        f"SELECT {cols.format(s='b')}, 'changed' AS kind FROM b JOIN a ON {join} WHERE {differs}",
        f"SELECT b.item_slug, b.day, b.source, b.mean_price, NULL::DOUBLE, b.volume, NULL::BIGINT, 'removed' "
        f"FROM b WHERE NOT EXISTS (SELECT 1 FROM a WHERE {join})",
    ]
    if newest is not None:
        queries.append(
            f"SELECT a.item_slug, a.day, a.source, NULL::DOUBLE, a.mean_price, NULL::BIGINT, a.volume, 'late_added' "
            f"FROM a WHERE a.day <= DATE '{newest}' AND NOT EXISTS (SELECT 1 FROM b WHERE {join})"
        )
    return con.execute(" UNION ALL ".join(queries)).to_arrow_table()


def _aggregate(rows: pa.Table) -> pa.Table:
    con = duckdb.connect()
    con.register("r", rows)
    return con.execute(
        "SELECT NULL::VARCHAR AS item_slug, day, source, kind, count(*)::BIGINT AS n_items, "
        "NULL::DOUBLE AS prev_mean_price, NULL::DOUBLE AS new_mean_price, "
        "NULL::BIGINT AS prev_volume, NULL::BIGINT AS new_volume FROM r GROUP BY day, source, kind"
    ).to_arrow_table()


def _stamp(rows: pa.Table, run_date: dt.date, detected_at: dt.datetime) -> pa.Table:
    n = rows.num_rows
    if "n_items" not in rows.schema.names:
        rows = rows.append_column("n_items", pa.array([1] * n, type=pa.int64()))
    rows = rows.append_column("run_date", pa.array([run_date] * n, type=pa.date32()))
    rows = rows.append_column("detected_at", pa.array([detected_at] * n, type=pa.timestamp("us")))
    return rows.select(SCHEMA.names).cast(SCHEMA)


def log_revisions(
    before_dir,
    archive_dir,
    run_date: dt.date,
    window_days: int | None = 14,
    max_detail_rows: int = 500_000,
    detected_at: dt.datetime | None = None,
) -> dict:
    """Diff the two copies and rewrite this run's rows in `ops/price_revisions.parquet`."""
    archive_dir = Path(archive_dir)
    rows = diff_prices(before_dir, archive_dir, run_date, window_days)
    detail = rows.num_rows
    if detail > max_detail_rows:
        rows = _aggregate(rows)
    new = _stamp(rows, run_date, detected_at or dt.datetime.now(dt.UTC).replace(tzinfo=None))

    path = archive_dir / "ops" / LOG_NAME
    if path.exists():
        old = pq.read_table(path).cast(SCHEMA)
        old = old.filter(pc.not_equal(old["run_date"], pa.scalar(run_date, pa.date32())))
        new = pa.concat_tables([old, new])
    elif new.num_rows == 0:
        return {"run_date": str(run_date), "revisions": 0, "logged_rows": 0, "wrote": False}

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(new, tmp, compression="zstd")
    os.replace(tmp, path)  # atomic: an interrupted write never truncates the log
    return {
        "run_date": str(run_date),
        "revisions": detail,
        "aggregated": detail > max_detail_rows,
        "logged_rows": new.num_rows,
        "wrote": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before-dir", required=True, type=Path, help="monthly price files copied before the append")
    ap.add_argument("--archive-dir", required=True, type=Path, help="the checked-out price-archive/ after the writers")
    ap.add_argument("--run-date", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--window-days", default="14", help="trailing days to diff, or 'all' (backfills)")
    ap.add_argument("--max-detail-rows", default=500_000, type=int)
    args = ap.parse_args(argv)

    window = None if args.window_days == "all" else int(args.window_days)
    report = log_revisions(args.before_dir, args.archive_dir, args.run_date, window, args.max_detail_rows)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
