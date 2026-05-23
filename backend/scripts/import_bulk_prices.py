#!/usr/bin/env python3
"""
Fast bulk import of historical price data using batch inserts.
Optimized for importing millions of records efficiently.

Usage:
    python scripts/import_bulk_prices.py --file data/consolidated_prices.csv --batch-size 5000
"""

import sys
import csv
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, PriceHistory, Item
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def bulk_import_prices(filepath: str, batch_size: int = 5000) -> int:
    """
    Import prices using batch inserts for maximum performance.

    Args:
        filepath: Path to CSV file
        batch_size: Number of records per batch insert

    Returns:
        Number of records imported
    """
    db = SessionLocal()

    try:
        # Pre-load all items into memory for fast lookups
        logger.info("Loading items into memory...")
        all_items = db.query(Item).all()
        item_map = {item.name: item.id for item in all_items}
        logger.info(f"Loaded {len(item_map)} items")

        imported = 0
        errors = 0
        batch = []
        last_log = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, 1):
                try:
                    # Skip if item not found
                    item_name = row['item_name']
                    item_id = item_map.get(item_name)
                    if not item_id:
                        errors += 1
                        continue

                    # Parse data
                    timestamp = datetime.fromisoformat(row['timestamp'])
                    price = float(row['price'])
                    volume = int(row.get('volume', 0)) if row.get('volume') else 0

                    if price <= 0:
                        errors += 1
                        continue

                    # Create record
                    price_record = PriceHistory(
                        item_id=item_id,
                        timestamp=timestamp,
                        price=price,
                        volume=volume,
                        median_price=price,
                        source=row.get('source', 'historical_import')
                    )

                    batch.append(price_record)
                    imported += 1

                    # Flush batch
                    if len(batch) >= batch_size:
                        try:
                            db.bulk_save_objects(batch, return_defaults=False)
                            db.commit()
                            last_log = imported
                            logger.info(f"  → {imported:,} records imported ({row_num:,} rows processed)")
                        except Exception as e:
                            logger.error(f"Batch insert failed at row {row_num}: {e}", exc_info=True)
                            db.rollback()
                            raise

                        batch = []

                    # Log progress every 100k rows even without batch flush
                    if imported - last_log >= 100000:
                        logger.info(f"  → {imported:,} records processed ({row_num:,} rows read)")
                        last_log = imported

                except Exception as e:
                    errors += 1
                    if errors % 10000 == 0:
                        logger.error(f"Error on row {row_num}: {e}")

        # Flush remaining batch
        if batch:
            try:
                db.bulk_save_objects(batch, return_defaults=False)
                db.commit()
                logger.info(f"  → Flushed final batch: {imported:,} records")
            except Exception as e:
                logger.error(f"Final batch insert failed: {e}", exc_info=True)
                db.rollback()
                raise

        logger.info("\n" + "="*60)
        logger.info("BULK IMPORT COMPLETE")
        logger.info("="*60)
        logger.info(f"✓ Records imported: {imported:,}")
        logger.info(f"✗ Records skipped: {errors:,}")

        return imported

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        db.rollback()
        return 0

    finally:
        db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fast bulk import of historical prices')
    parser.add_argument('--file', required=True, help='CSV file to import')
    parser.add_argument('--batch-size', type=int, default=5000, help='Records per batch (default: 5000)')

    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return 1

    logger.info(f"Starting bulk import from {filepath}")
    logger.info(f"Batch size: {args.batch_size}")

    imported = bulk_import_prices(str(filepath), args.batch_size)

    return 0 if imported > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
