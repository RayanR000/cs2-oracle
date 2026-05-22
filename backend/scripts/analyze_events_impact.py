#!/usr/bin/env python3
"""
Analyze the impact of events on item prices.
Find correlations between game updates, tournaments, and price movements.

Usage:
    python scripts/analyze_events_impact.py [--days N] [--event-type TYPE]
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, Event, PriceHistory, Item
from sqlalchemy import text, func

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_events_with_price_impact(days=30, event_type=None):
    """
    Find events and their surrounding price movements.

    For each event, analyze:
    - Price changes in the week before
    - Price changes in the week after
    - Items with highest price volatility around event
    """
    db = SessionLocal()
    try:
        # Get events
        query = db.query(Event)
        if event_type:
            query = query.filter(Event.type == event_type)

        events = query.order_by(Event.timestamp).all()

        if not events:
            print("No events found.")
            return

        print(f"\n{'='*100}")
        print(f"EVENT IMPACT ANALYSIS ({len(events)} events)")
        print(f"{'='*100}\n")

        for event in events:
            print(f"📅 {event.timestamp.date()} | {event.type.upper()}")
            print(f"   {event.description}")

            # Analyze price movements around event
            before_start = event.timestamp - timedelta(days=7)
            after_end = event.timestamp + timedelta(days=7)

            # Get price movements before event
            result = db.execute(text(f'''
                SELECT
                    COUNT(DISTINCT item_id) as items_affected,
                    AVG(ABS(price)) as avg_price,
                    MAX(price) as max_price,
                    MIN(price) as min_price
                FROM price_history
                WHERE created_at BETWEEN '{before_start}' AND '{event.timestamp}'
            '''))

            before = result.fetchone()
            if before and before[0]:
                print(f"   ↪ Before: {int(before[0])} items, Avg price: ${before[1]:.2f}")

            # Get price movements after event
            result = db.execute(text(f'''
                SELECT
                    COUNT(DISTINCT item_id) as items_affected,
                    AVG(ABS(price)) as avg_price,
                    MAX(price) as max_price,
                    MIN(price) as min_price
                FROM price_history
                WHERE created_at BETWEEN '{event.timestamp}' AND '{after_end}'
            '''))

            after = result.fetchone()
            if after and after[0]:
                print(f"   ↪ After:  {int(after[0])} items, Avg price: ${after[1]:.2f}")

            # Show impact
            if before and after and before[0] and after[0]:
                price_change = ((after[1] - before[1]) / before[1] * 100) if before[1] > 0 else 0
                impact = "📈 UP" if price_change > 2 else "📉 DOWN" if price_change < -2 else "➡️  STABLE"
                print(f"   {impact} ({price_change:+.1f}% avg price change)")

            print()

    finally:
        db.close()


def show_event_timeline():
    """Show a timeline of all events."""
    db = SessionLocal()
    try:
        events = db.query(Event).order_by(Event.timestamp).all()

        if not events:
            print("No events found.")
            return

        print(f"\n{'='*100}")
        print(f"EVENT TIMELINE")
        print(f"{'='*100}\n")

        current_month = None
        for event in events:
            month = event.timestamp.strftime("%Y-%m")
            if month != current_month:
                current_month = month
                print(f"\n📆 {month}")

            icon = {
                'major': '🏆',
                'game_update': '⚙️',
                'case_drop': '📦',
                'operation': '🎮'
            }.get(event.type, '📌')

            print(f"  {icon} {event.timestamp.strftime('%d')}: {event.description}")

    finally:
        db.close()


def count_events_by_type():
    """Show event statistics."""
    db = SessionLocal()
    try:
        result = db.execute(text('''
            SELECT type, COUNT(*) as count
            FROM events
            GROUP BY type
            ORDER BY count DESC
        '''))

        print(f"\n{'='*60}")
        print(f"EVENT STATISTICS")
        print(f"{'='*60}\n")

        total = 0
        for event_type, count in result.fetchall():
            print(f"  {event_type:15}: {count:3} events")
            total += count

        print(f"\n  {'TOTAL':15}: {total:3} events\n")

    finally:
        db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze event impact on prices')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze (default: 30)')
    parser.add_argument('--event-type', help='Filter by event type')
    parser.add_argument('--timeline', action='store_true', help='Show event timeline')
    parser.add_argument('--stats', action='store_true', help='Show event statistics')

    args = parser.parse_args()

    if args.timeline:
        show_event_timeline()
    elif args.stats:
        count_events_by_type()
    else:
        get_events_with_price_impact(days=args.days, event_type=args.event_type)

    return 0


if __name__ == "__main__":
    sys.exit(main())
