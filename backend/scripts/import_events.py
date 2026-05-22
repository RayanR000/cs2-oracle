#!/usr/bin/env python3
"""
Import CS2 events (game updates, majors, tournaments) into the database.
These events are used for time-series analysis to correlate price changes with game events.

Usage:
    python scripts/import_events.py [--file data/cs2_events.json] [--clear]

Options:
    --file: Path to JSON file with events (default: data/cs2_events.json)
    --clear: Clear existing events before importing
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, Event
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_events_from_file(filepath: str) -> list:
    """Load events from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('events', [])


def import_events(events: list, clear_existing: bool = False):
    """Import events into database."""
    db = SessionLocal()
    try:
        if clear_existing:
            logger.info("Clearing existing events...")
            db.query(Event).delete()
            db.commit()
            logger.info("✓ Events cleared")

        imported = 0
        skipped = 0

        for event_data in events:
            try:
                # Parse timestamp
                timestamp_str = event_data.get('date')
                timestamp = datetime.fromisoformat(timestamp_str)

                # Check if event already exists
                existing = db.query(Event).filter(
                    Event.timestamp == timestamp,
                    Event.type == event_data.get('type'),
                    Event.description == event_data.get('description')
                ).first()

                if existing:
                    logger.debug(f"Event already exists: {event_data['description']}")
                    skipped += 1
                    continue

                # Create new event
                event = Event(
                    type=event_data.get('type'),
                    timestamp=timestamp,
                    description=event_data.get('description')
                )

                db.add(event)
                imported += 1

                logger.info(f"+ {event_data['type']}: {event_data['description']} ({timestamp_str})")

            except Exception as e:
                logger.error(f"Error importing event {event_data}: {e}")
                db.rollback()
                continue

        db.commit()

        logger.info("\n" + "="*60)
        logger.info("IMPORT COMPLETE")
        logger.info("="*60)
        logger.info(f"✓ Imported: {imported} events")
        logger.info(f"⊘ Skipped: {skipped} events (already exist)")
        logger.info(f"Total in database: {db.query(Event).count()} events")

        return imported

    except SQLAlchemyError as db_error:
        logger.error(f"Database error: {db_error}", exc_info=True)
        db.rollback()
        return 0

    finally:
        db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Import CS2 events into database')
    parser.add_argument('--file', default='data/cs2_events.json', help='Path to events JSON file')
    parser.add_argument('--clear', action='store_true', help='Clear existing events before importing')

    args = parser.parse_args()

    # Find events file
    events_file = Path(__file__).parent.parent / args.file

    if not events_file.exists():
        logger.error(f"Events file not found: {events_file}")
        sys.exit(1)

    logger.info(f"Loading events from {events_file}...")
    events = load_events_from_file(str(events_file))
    logger.info(f"Loaded {len(events)} events")

    imported = import_events(events, clear_existing=args.clear)

    return 0 if imported > 0 or args.clear else 1


if __name__ == "__main__":
    sys.exit(main())
