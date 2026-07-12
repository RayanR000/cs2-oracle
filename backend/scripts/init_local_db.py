#!/usr/bin/env python3
"""
Initialize local SQLite database from parquet archive + events file.

Populates:
- items table (from parquet slugs)
- events table (from cs2_events.json)
- All other tables (created empty)

Run from backend/ directory:
    python scripts/init_local_db.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, date, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, init_db, Item, Event
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("init_local_db")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"
EVENTS_FILE = Path(__file__).parent.parent / "data" / "cs2_events.json"


def populate_items(db):
    logger.info("Reading items from parquet archive...")
    import duckdb
    con = duckdb.connect()
    try:
        rows = con.sql(f"""
            SELECT DISTINCT item_slug
            FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet')
            ORDER BY item_slug
        """).fetchall()
    finally:
        con.close()

    total = len(rows)
    logger.info(f"Found {total:,} unique items in parquet")

    existing = {r[0] for r in db.query(Item.item_id).all()}
    to_insert = []
    for (slug,) in rows:
        if slug not in existing:
            to_insert.append(Item(
                item_id=slug,
                name=slug,
                type="skin",
                is_backfilled=1,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            ))

    if to_insert:
        db.add_all(to_insert)
        db.commit()
        logger.info(f"Inserted {len(to_insert)} new items")
    else:
        logger.info("No new items to insert")

    total_in_db = db.query(Item).count()
    logger.info(f"Total items in DB: {total_in_db}")


def populate_events(db):
    if not EVENTS_FILE.exists():
        logger.warning(f"Events file not found at {EVENTS_FILE}")
        return

    with open(EVENTS_FILE) as f:
        data = json.load(f)

    raw_events = data.get("events", [])
    logger.info(f"Found {len(raw_events)} events in {EVENTS_FILE}")

    existing = {(r.type, r.timestamp.date()) for r in db.query(Event).all()}
    to_insert = []
    for ev in raw_events:
        ev_type = ev.get("type", "update")
        ev_date = ev.get("date", "")
        try:
            parsed_date = datetime.strptime(ev_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.warning(f"Skipping event with bad date: {ev_date}")
            continue

        if (ev_type, parsed_date) not in existing:
            to_insert.append(Event(
                type=ev_type,
                timestamp=datetime.combine(parsed_date, datetime.min.time()),
                description=ev.get("description", ""),
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            ))

    if to_insert:
        db.add_all(to_insert)
        db.commit()
        logger.info(f"Inserted {len(to_insert)} new events")
    else:
        logger.info("No new events to insert")

    total_in_db = db.query(Event).count()
    logger.info(f"Total events in DB: {total_in_db}")


def main():
    logger.info("=" * 60)
    logger.info("INITIALIZING LOCAL DATABASE")
    logger.info("=" * 60)

    # Force SQLite for local dev
    import os
    os.environ["DATABASE_URL"] = "sqlite:///backend/cs2_market.db"

    init_db()
    db = SessionLocal()
    try:
        populate_items(db)
        populate_events(db)

        logger.info("\nFinal table counts:")
        for table in ["items", "events", "item_forecasts", "prediction_accuracy",
                       "forecast_outcomes", "accuracy_alerts", "price_history"]:
            try:
                count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                logger.info(f"  {table}: {count:,}")
            except Exception:
                logger.info(f"  {table}: <empty or does not exist>")
    finally:
        db.close()

    logger.info("\n✅ Local database initialized successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
