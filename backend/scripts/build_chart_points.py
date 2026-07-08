#!/usr/bin/env python3
"""
Build chart_points table from Parquet archive.

Reads year-split Parquet files, resolves item_slug → item_id via Supabase,
and upserts one daily close per item into chart_points.

Usage:
    python scripts/build_chart_points.py [--parquet-dir ../archive/price-archive]
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, utcnow_naive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-dir",
        default="../archive/price-archive",
        help="Directory containing prices-YYYY.parquet files",
    )
    parser.add_argument("--date", help="Only process this date (YYYY-MM-DD)")
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        print(f"ERROR: {parquet_dir} not found", file=sys.stderr)
        sys.exit(1)

    db = SessionLocal()

    # Build name → item_id mapping (parquet item_slug = display name)
    slug_map_rows = db.execute(text(
        "SELECT id, name FROM items WHERE is_backfilled = 1"
    )).fetchall()
    slug_to_id = {row.name: row.id for row in slug_map_rows}

    if not slug_to_id:
        print("No backfilled items found. Run the backfill migration first.")
        sys.exit(1)

    print(f"Loaded {len(slug_to_id)} backfilled item mappings (by display name)")

    con = duckdb.connect()
    parquet_files = sorted(parquet_dir.glob("prices-*.parquet"))

    if not parquet_files:
        print(f"ERROR: no Parquet files found in {parquet_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for pf in parquet_files:
        if args.date:
            date_filter = f"AND day = DATE '{args.date}'"
        else:
            date_filter = ""

        rows = con.sql(f"""
            SELECT item_slug, day, mean_price AS close
            FROM read_parquet('{pf}')
            WHERE 1=1 {date_filter}
        """).fetchall()
        all_rows.extend(rows)

    if not all_rows:
        print("No rows to process")
        return

    # Build list of (item_id, day, close) for upsert
    inserts = []
    slug_not_found = set()
    for slug, day, close in all_rows:
        item_id = slug_to_id.get(slug)
        if item_id is None:
            slug_not_found.add(slug)
            continue
        inserts.append({"item_id": item_id, "day": day, "close": float(close)})

    if slug_not_found:
        print(f"Warning: {len(slug_not_found)} slugs not in items table")

    if not inserts:
        print("No chart points to insert")
        return

    # Batch upsert into chart_points
    CHUNK_SIZE = 1000
    total_upserted = 0
    for i in range(0, len(inserts), CHUNK_SIZE):
        chunk = inserts[i:i + CHUNK_SIZE]
        db.execute(
            text("""
                INSERT INTO chart_points (item_id, day, close)
                VALUES (:item_id, :day, :close)
                ON CONFLICT (item_id, day) DO UPDATE SET close = EXCLUDED.close
            """),
            chunk,
        )
        db.commit()
        total_upserted += len(chunk)

    print(f"Upserted {total_upserted:,} chart points from {len(parquet_files)} Parquet files")

    con.close()
    db.close()


if __name__ == "__main__":
    main()
