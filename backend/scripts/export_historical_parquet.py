#!/usr/bin/env python3
"""
One-time export: read csmarketapi.db STEAMCOMMUNITY data → year-split Parquet files.

Exports only STEAMCOMMUNITY-market data (Steam Community Market price basis).
MarketCSGO rows (~13.5% below Steam) are excluded.

Columns: item_slug, day, mean_price, median_price, volume
Writes to archive/price-archive/prices-YYYY.parquet.

Usage:
    python scripts/export_historical_parquet.py [--db-path csmarketapi.db] [--out-dir ../archive]
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default="csmarketapi.db",
        help="Path to the csmarketapi SQLite database",
    )
    parser.add_argument(
        "--out-dir",
        default="../archive",
        help="Output directory for Parquet files",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) / "price-archive"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite")

    try:
        con.execute(f"ATTACH '{db_path}' AS src (TYPE sqlite)")
        df = con.sql("""
            SELECT
                market_hash_name AS item_slug,
                day,
                COALESCE(mean_price, median_price) AS mean_price,
                median_price,
                volume
            FROM src.sales_history
            WHERE market = 'STEAMCOMMUNITY'
              AND median_price IS NOT NULL
        """).fetchdf()
    except Exception as e:
        print(f"SQLite query failed: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("No STEAMCOMMUNITY rows found")
        sys.exit(1)

    print(f"Read {len(df):,} STEAMCOMMUNITY rows from {db_path.name}")

    df["day"] = pd.to_datetime(df["day"])
    df["year"] = df["day"].dt.year

    total_rows = 0
    for year, year_df in df.groupby("year"):
        year = int(year)
        out_path = out_dir / f"prices-{year}.parquet"
        year_df = year_df.drop(columns=["year"])
        year_df.to_parquet(out_path, index=False)
        rows = len(year_df)
        total_rows += rows
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  {out_path.name}: {rows:,} rows, {size_mb:.1f} MB")

    con.close()
    print(f"\nExported {total_rows:,} total rows to {out_dir}")


if __name__ == "__main__":
    main()
