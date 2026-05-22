#!/usr/bin/env python3
"""
Import historical price data from various sources.
Supports importing from CSV, JSON, or direct API sources.

Historical data sources:
1. BitSkins API - historical market data
2. CSGOFloat - has some historical data
3. CSV/JSON files - manually collected or from third-party services
4. Steam Community Market snapshots (if available)

Usage:
    python scripts/import_historical_prices.py --source csv --file data/historical_prices.csv
    python scripts/import_historical_prices.py --source json --file data/prices.json
    python scripts/import_historical_prices.py --source bitskins --date 2023-01-01 --end-date 2023-12-31
"""

import sys
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, PriceHistory, Item
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def import_from_csv(filepath: str) -> int:
    """
    Import prices from CSV file.
    Expected columns: item_name, timestamp, price, volume, source
    """
    db = SessionLocal()
    imported = 0
    errors = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Find item by name
                    item = db.query(Item).filter(Item.name == row['item_name']).first()

                    if not item:
                        logger.warning(f"Item not found: {row['item_name']}")
                        errors += 1
                        continue

                    # Parse timestamp
                    timestamp = datetime.fromisoformat(row['timestamp'])

                    # Check if price record already exists
                    existing = db.query(PriceHistory).filter(
                        PriceHistory.item_id == item.id,
                        PriceHistory.timestamp == timestamp
                    ).first()

                    if existing:
                        logger.debug(f"Price already exists: {item.name} at {timestamp}")
                        errors += 1
                        continue

                    # Create price record
                    price_record = PriceHistory(
                        item_id=item.id,
                        timestamp=timestamp,
                        price=float(row['price']),
                        volume=int(row.get('volume', 0)) if row.get('volume') else 0,
                        median_price=float(row.get('median_price', row['price'])),
                        source=row.get('source', 'historical_import')
                    )

                    db.add(price_record)
                    imported += 1

                    if imported % 1000 == 0:
                        db.commit()
                        logger.info(f"  → {imported} records imported...")

                except Exception as e:
                    logger.error(f"Error importing row {row}: {e}")
                    db.rollback()
                    errors += 1

        db.commit()
        return imported

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return 0

    finally:
        db.close()


def import_from_json(filepath: str) -> int:
    """
    Import prices from JSON file.
    Expected format:
    {
        "prices": [
            {
                "item_name": "AK-47 | Phantom Disruptor",
                "timestamp": "2023-01-15T10:00:00",
                "price": 3.50,
                "volume": 1200,
                "source": "historical"
            }
        ]
    }
    """
    db = SessionLocal()
    imported = 0
    errors = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prices = data.get('prices', [])

            for price_data in prices:
                try:
                    # Find item by name
                    item = db.query(Item).filter(Item.name == price_data['item_name']).first()

                    if not item:
                        logger.debug(f"Item not found: {price_data['item_name']}")
                        errors += 1
                        continue

                    # Parse timestamp
                    timestamp = datetime.fromisoformat(price_data['timestamp'])

                    # Check if price record already exists
                    existing = db.query(PriceHistory).filter(
                        PriceHistory.item_id == item.id,
                        PriceHistory.timestamp == timestamp
                    ).first()

                    if existing:
                        errors += 1
                        continue

                    # Create price record
                    price_record = PriceHistory(
                        item_id=item.id,
                        timestamp=timestamp,
                        price=float(price_data['price']),
                        volume=int(price_data.get('volume', 0)) if price_data.get('volume') else 0,
                        median_price=float(price_data.get('median_price', price_data['price'])),
                        source=price_data.get('source', 'historical_import')
                    )

                    db.add(price_record)
                    imported += 1

                    if imported % 1000 == 0:
                        db.commit()
                        logger.info(f"  → {imported} records imported...")

                except Exception as e:
                    logger.error(f"Error importing price data {price_data}: {e}")
                    db.rollback()
                    errors += 1

        db.commit()
        return imported

    except Exception as e:
        logger.error(f"Error reading JSON: {e}")
        return 0

    finally:
        db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Import historical price data',
        epilog='''
Examples:
  python import_historical_prices.py --source csv --file data/prices.csv
  python import_historical_prices.py --source json --file data/prices.json
        '''
    )

    parser.add_argument('--source', required=True, choices=['csv', 'json'],
                       help='Data source format')
    parser.add_argument('--file', required=True, help='Path to data file')

    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        sys.exit(1)

    logger.info(f"Importing historical prices from {args.source.upper()}: {filepath}")

    if args.source == 'csv':
        imported = import_from_csv(str(filepath))
    elif args.source == 'json':
        imported = import_from_json(str(filepath))

    print("\n" + "="*60)
    print("IMPORT COMPLETE")
    print("="*60)
    print(f"✓ Imported: {imported} price records")

    return 0


if __name__ == "__main__":
    sys.exit(main())
