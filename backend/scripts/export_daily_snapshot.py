#!/usr/bin/env python3
"""
Export one day's live-collected price rows to a gzipped CSV.

Used by the aggregator GitHub Actions workflow to archive each day's
collection to the data-archive branch. Snapshot-tier items (no CSMarketAPI
historical series) keep only their latest row in Supabase, so these daily
dumps are the only durable record of their price history.
"""

import argparse
import csv
import gzip
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database import engine

LIVE_ROWS_FOR_DAY = """
    SELECT i.item_id AS item_slug, i.name, ph.item_id AS supabase_item_id,
           ph.timestamp, ph.price, ph.volume, ph.median_price, ph.source
    FROM price_history ph
    JOIN items i ON i.id = ph.item_id
    WHERE ph.timestamp >= :day_start AND ph.timestamp < :day_end
      AND (ph.source IN ('aggregator_sync', 'steam_batch', 'synthetic_demo')
           OR ph.source LIKE 'historical_fallback:%')
    ORDER BY i.item_id, ph.timestamp
"""

HEADER = [
    "item_slug", "name", "supabase_item_id",
    "timestamp", "price", "volume", "median_price", "source",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="UTC day to export (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Archive root; file lands at <out-dir>/price-archive/YYYY/MM/prices-<date>.csv.gz",
    )
    args = parser.parse_args()

    day_start = datetime.strptime(args.date, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)

    out_path = (
        Path(args.out_dir) / "price-archive"
        / f"{day_start:%Y}" / f"{day_start:%m}" / f"prices-{args.date}.csv.gz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with engine.connect() as conn:
        rows = conn.execute(
            text(LIVE_ROWS_FOR_DAY),
            {"day_start": day_start, "day_end": day_end},
        ).fetchall()

    if not rows:
        print(f"ERROR: no live-collected rows found for {args.date}", file=sys.stderr)
        sys.exit(1)

    with gzip.open(out_path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    size_kb = out_path.stat().st_size / 1024
    print(f"exported {len(rows):,} rows for {args.date} to {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
