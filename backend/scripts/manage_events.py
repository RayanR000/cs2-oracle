#!/usr/bin/env python3
"""
Manage CS2 events in the database.
Add, list, update, or delete events for time-series analysis.

Usage:
    python scripts/manage_events.py list [--type TYPE] [--recent N]
    python scripts/manage_events.py add --type TYPE --date DATE --description "DESC"
    python scripts/manage_events.py delete --id ID
    python scripts/manage_events.py clear
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, Event
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_events(event_type=None, recent=None):
    """List events from database."""
    db = SessionLocal()
    try:
        query = db.query(Event)

        if event_type:
            query = query.filter(Event.type == event_type)

        query = query.order_by(Event.timestamp.desc())

        if recent:
            events = query.limit(recent).all()
        else:
            events = query.all()

        if not events:
            print("No events found.")
            return

        print(f"\n{'ID':<5} {'Date':<12} {'Type':<15} {'Description':<50}")
        print("-" * 85)

        for event in events:
            print(f"{event.id:<5} {str(event.timestamp.date()):<12} {event.type:<15} {event.description:<50}")

        print(f"\nTotal: {len(events)} events\n")

    finally:
        db.close()


def add_event(event_type, date_str, description):
    """Add a new event."""
    db = SessionLocal()
    try:
        # Parse date
        try:
            timestamp = datetime.fromisoformat(date_str)
        except ValueError:
            logger.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
            return False

        # Check if event already exists
        existing = db.query(Event).filter(
            Event.timestamp == timestamp,
            Event.type == event_type,
            Event.description == description
        ).first()

        if existing:
            logger.warning(f"Event already exists (ID: {existing.id})")
            return False

        # Create new event
        event = Event(
            type=event_type,
            timestamp=timestamp,
            description=description
        )

        db.add(event)
        db.commit()

        logger.info(f"✓ Event added (ID: {event.id})")
        logger.info(f"  Type: {event_type}")
        logger.info(f"  Date: {timestamp}")
        logger.info(f"  Description: {description}")

        return True

    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        return False

    finally:
        db.close()


def delete_event(event_id):
    """Delete an event by ID."""
    db = SessionLocal()
    try:
        event = db.query(Event).filter(Event.id == event_id).first()

        if not event:
            logger.error(f"Event not found: {event_id}")
            return False

        db.delete(event)
        db.commit()

        logger.info(f"✓ Event deleted (ID: {event_id})")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        return False

    finally:
        db.close()


def clear_events():
    """Clear all events."""
    db = SessionLocal()
    try:
        count = db.query(Event).delete()
        db.commit()

        logger.info(f"✓ Cleared {count} events")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        return False

    finally:
        db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Manage CS2 events')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    list_parser = subparsers.add_parser('list', help='List events')
    list_parser.add_argument('--type', help='Filter by event type')
    list_parser.add_argument('--recent', type=int, help='Show only recent N events')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add new event')
    add_parser.add_argument('--type', required=True, help='Event type (major, game_update, case_drop, operation)')
    add_parser.add_argument('--date', required=True, help='Event date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    add_parser.add_argument('--description', required=True, help='Event description')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete event')
    delete_parser.add_argument('--id', type=int, required=True, help='Event ID')

    # Clear command
    subparsers.add_parser('clear', help='Clear all events (WARNING: destructive)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == 'list':
        list_events(event_type=args.type, recent=args.recent)
        return 0

    elif args.command == 'add':
        success = add_event(args.type, args.date, args.description)
        return 0 if success else 1

    elif args.command == 'delete':
        success = delete_event(args.id)
        return 0 if success else 1

    elif args.command == 'clear':
        confirm = input("Are you sure? This will delete all events. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            success = clear_events()
            return 0 if success else 1
        else:
            logger.info("Cancelled")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
