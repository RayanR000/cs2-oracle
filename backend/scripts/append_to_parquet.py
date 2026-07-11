#!/usr/bin/env python3
"""
Daily: append today's aggregator rows to the current year's Parquet files.

Writes two Parquet files:
  prices-YYYY.parquet     — Steam daily OHLCV (from aggregator_sync rows)
  snapshots-YYYY.parquet  — All source snapshots (flat: item_slug, day, source, price, volume)

Input: a snapshot CSV written by the aggregator (or Supabase + backfilled CSV for backward compat).

Usage:
    python scripts/append_to_parquet.py --date 2026-07-08 --out-dir ../archive \\
        --snapshot-csv /tmp/aggregator-snapshots-2026-07-08.csv
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import engine


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
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
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
    args = parser.parse_args()

    day_start = datetime.strptime(args.date, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)
    year = day_start.year
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
            legacy_frames.append(pd.DataFrame(
                snapshot_rows,
                columns=["item_slug", "day", "price", "volume", "median_price", "source"],
            ))

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

    # ── Write prices-YYYY.parquet (OHLCV, all sources) ──────────────────
    if snapshots_df is not None and not snapshots_df.empty:
        daily = snapshots_df.groupby(["item_slug", "day", "source"]).agg(
            mean_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            median_price=("median_price", "mean") if "median_price" in snapshots_df.columns else ("price", "mean"),
            volume=("volume", "sum"),
        ).reset_index()
        daily["day"] = pd.to_datetime(daily["day"])
        _append_parquet(out_dir / f"prices-{year}.parquet", daily, ["item_slug", "day", "source"])
        print(f"Appended {len(daily)} OHLCV rows to prices-{year}.parquet")

    if legacy_frames:
        df = pd.concat(legacy_frames, ignore_index=True)
        daily = df.groupby(["item_slug", "day", "source"]).agg(
            mean_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            median_price=("median_price", "mean"),
            volume=("volume", "sum"),
        ).reset_index()
        daily["day"] = pd.to_datetime(daily["day"])
        _append_parquet(out_dir / f"prices-{year}.parquet", daily, ["item_slug", "day", "source"])
        print(f"Appended {len(daily)} OHLCV rows to prices-{year}.parquet (legacy path)")

    # ── Write snapshots-YYYY.parquet (all sources) ─────────────────────
    if snapshots_df is not None and not snapshots_df.empty:
        snap_path = out_dir / f"snapshots-{year}.parquet"
        out_cols = ["item_slug", "day", "source", "price", "volume"]
        snap_data = snapshots_df[out_cols].copy()
        _append_parquet(snap_path, snap_data, ["item_slug", "day", "source"])
        print(f"Appended {len(snap_data)} snapshot rows to snapshots-{year}.parquet")

    if not legacy_frames and snapshots_df is None:
        print(f"No data found for {args.date}")
        sys.exit(0)

    print(f"Done: {args.date}")


def _append_parquet(path: Path, new_data: pd.DataFrame, dedup_keys: list):
    """Append new_data to an existing Parquet file, deduplicating on dedup_keys."""
    con = duckdb.connect()
    try:
        if path.exists():
            existing = con.sql(f"SELECT * FROM read_parquet('{path}')").fetchdf()
            # Migrate old schema: add source column if missing
            if "source" not in existing.columns and "source" in new_data.columns:
                existing["source"] = "aggregator_sync"
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
            combined.to_parquet(path, index=False)
            print(f"  {path.name}: {len(new_data)} appended, {len(combined)} total")
        else:
            new_data.to_parquet(path, index=False)
            print(f"  {path.name}: {len(new_data)} written (new file)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
