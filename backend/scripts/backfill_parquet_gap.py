#!/usr/bin/env python3
"""
Backfill missing price history dates into prices-YYYY.parquet from Supabase.

The data-archive branch has daily prices from 2013 → 2026-03-29 (historical
STEAMCOMMUNITY export) and then spotty data from Jul 9 onward. The gap
(Mar 30 → Jul 8, 2026) may exist in Supabase price_history if daily syncs
were running before the parquet archive system existed.

Usage:
    python scripts/backfill_parquet_gap.py [--start 2026-03-30] [--end 2026-07-08]
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from database import engine


FETCH_RANGE_SQL = """
    SELECT i.item_id AS item_slug,
           DATE(ph.timestamp) AS day,
           ph.price,
           ph.volume,
           ph.median_price,
           ph.source
    FROM price_history ph
    JOIN items i ON i.id = ph.item_id
    WHERE ph.timestamp >= :start_date
      AND ph.timestamp < :end_date + INTERVAL '1 day'
      AND ph.source IN ('aggregator_sync', 'steam_batch')
    ORDER BY i.item_id, ph.timestamp
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-03-30", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-07-08", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="../archive", help="Archive root")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    out_dir = Path(args.out_dir) / "price-archive"

    print(f"Backfilling {args.start} → {args.end} from Supabase price_history...")

    with engine.connect() as conn:
        rows = conn.execute(
            text(FETCH_RANGE_SQL),
            {"start_date": start_date, "end_date": end_date},
        ).fetchall()

    if not rows:
        print("No data found in Supabase for this range. The gap is unrecoverable.")
        sys.exit(0)

    df = pd.DataFrame(rows, columns=["item_slug", "day", "price", "volume", "median_price", "source"])
    df["day"] = pd.to_datetime(df["day"])

    # Aggregate to daily OHLCV per item_slug, day, source
    daily = df.groupby(["item_slug", "day", "source"]).agg(
        mean_price=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        median_price=("median_price", "mean"),
        volume=("volume", "sum"),
    ).reset_index()

    days_found = sorted(daily["day"].dt.date.unique())
    print(f"Found {len(daily):,} rows across {len(days_found)} days in Supabase")
    print(f"  Date range: {days_found[0]} → {days_found[-1]}")

    # Group by year for output
    daily["year"] = daily["day"].dt.year
    for year, year_df in daily.groupby("year"):
        year = int(year)
        out_path = out_dir / f"prices-{year}.parquet"
        year_df = year_df.drop(columns=["year"])

        if args.dry_run:
            print(f"  Would append {len(year_df):,} rows to {out_path.name}")
        else:
            _append_parquet(out_path, year_df, ["item_slug", "day", "source"])
            print(f"  Appended {len(year_df):,} rows to {out_path.name}")

    print("Done.")


def _append_parquet(path: Path, new_data: pd.DataFrame, dedup_keys: list):
    con = duckdb.connect()
    try:
        if path.exists():
            existing = con.sql(f"SELECT * FROM read_parquet('{path}')").fetchdf()
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
