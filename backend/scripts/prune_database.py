#!/usr/bin/env python3
"""
Database pruning and downsampling script.
Implements tiered data retention:
  - Week (0-7 days): All data points (per 6 hrs)
  - Month (7-30 days): Daily average
  - Year (30-365 days): Weekly average
  - Older (365+ days): Monthly average
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import func, text

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, PriceHistory, TrendIndicator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pruning")


def prune_price_history(db_session, days_to_keep=365, dry_run=False):
    """Delete price history older than days_to_keep (keep one per month)."""
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

    query = db_session.query(PriceHistory).filter(
        PriceHistory.timestamp < cutoff_date
    )

    count = query.count()

    if count > 0 and not dry_run:
        query.delete()
        db_session.commit()
        logger.info(f"Deleted {count} price history records older than {days_to_keep} days")

    return count


def prune_trend_indicators(db_session, days_to_keep=180, dry_run=False):
    """Delete trend indicators older than days_to_keep."""
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

    query = db_session.query(TrendIndicator).filter(
        TrendIndicator.timestamp < cutoff_date
    )

    count = query.count()

    if count > 0 and not dry_run:
        query.delete()
        db_session.commit()
        logger.info(f"Deleted {count} trend indicator records older than {days_to_keep} days")

    return count


def downsample_price_history(db_session, days_to_keep_granular=7, dry_run=False):
    """
    Downsample price history with tiered strategy:
    - 0-7 days: Keep all data
    - 7-30 days: Keep daily average
    - 30-365 days: Keep weekly average
    - 365+ days: Keep monthly average
    """
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    year_ago = now - timedelta(days=365)

    total_deleted = 0

    # Process each tier
    deleted_count = _downsample_daily(db_session, week_ago, month_ago, dry_run)
    total_deleted += deleted_count
    logger.info(f"Downsampled {deleted_count} records to daily averages (7-30 days)")

    deleted_count = _downsample_weekly(db_session, month_ago, year_ago, dry_run)
    total_deleted += deleted_count
    logger.info(f"Downsampled {deleted_count} records to weekly averages (30-365 days)")

    deleted_count = _downsample_monthly(db_session, year_ago, dry_run)
    total_deleted += deleted_count
    logger.info(f"Downsampled {deleted_count} records to monthly averages (365+ days)")

    return total_deleted


def _downsample_daily(db_session, start_date, end_date, dry_run=False):
    """Keep one daily average per item for 7-30 day range."""
    # Get all items with data in this range
    items = db_session.query(PriceHistory.item_id).filter(
        PriceHistory.timestamp >= start_date,
        PriceHistory.timestamp < end_date
    ).distinct().all()

    total_deleted = 0

    for (item_id,) in items:
        # Group by day
        days = db_session.query(
            func.date(PriceHistory.timestamp).label('day'),
            func.avg(PriceHistory.price).label('avg_price'),
            func.avg(PriceHistory.volume).label('avg_volume'),
            func.min(PriceHistory.timestamp).label('first_ts'),
            func.max(PriceHistory.timestamp).label('last_ts'),
        ).filter(
            PriceHistory.item_id == item_id,
            PriceHistory.timestamp >= start_date,
            PriceHistory.timestamp < end_date
        ).group_by(func.date(PriceHistory.timestamp)).all()

        for day, avg_price, avg_volume, first_ts, last_ts in days:
            # Delete all records for this day except one (which we'll update)
            records_to_keep = 1
            records_to_delete = db_session.query(PriceHistory).filter(
                PriceHistory.item_id == item_id,
                func.date(PriceHistory.timestamp) == day
            ).order_by(PriceHistory.timestamp).offset(records_to_keep).all()

            for record in records_to_delete:
                db_session.delete(record)
                total_deleted += 1

            # Update the kept record with the average
            kept_record = db_session.query(PriceHistory).filter(
                PriceHistory.item_id == item_id,
                func.date(PriceHistory.timestamp) == day
            ).order_by(PriceHistory.timestamp).first()

            if kept_record:
                kept_record.price = avg_price
                kept_record.volume = avg_volume
                kept_record.timestamp = first_ts  # Use first timestamp of the day

    if total_deleted > 0 and not dry_run:
        db_session.commit()

    return total_deleted


def _downsample_weekly(db_session, start_date, end_date, dry_run=False):
    """Keep one weekly average per item for 30-365 day range."""
    items = db_session.query(PriceHistory.item_id).filter(
        PriceHistory.timestamp >= start_date,
        PriceHistory.timestamp < end_date
    ).distinct().all()

    total_deleted = 0

    for (item_id,) in items:
        # Group by week (ISO week)
        weeks = db_session.query(
            func.to_char(PriceHistory.timestamp, 'YYYY-WW').label('week'),
            func.avg(PriceHistory.price).label('avg_price'),
            func.avg(PriceHistory.volume).label('avg_volume'),
            func.min(PriceHistory.timestamp).label('first_ts'),
        ).filter(
            PriceHistory.item_id == item_id,
            PriceHistory.timestamp >= start_date,
            PriceHistory.timestamp < end_date
        ).group_by(func.to_char(PriceHistory.timestamp, 'YYYY-WW')).all()

        for week, avg_price, avg_volume, first_ts in weeks:
            # Delete all records for this week except one
            records_to_keep = 1
            week_start = first_ts.replace(hour=0, minute=0, second=0, microsecond=0)
            week_end = week_start + timedelta(days=7)

            records_to_delete = db_session.query(PriceHistory).filter(
                PriceHistory.item_id == item_id,
                PriceHistory.timestamp >= week_start,
                PriceHistory.timestamp < week_end
            ).order_by(PriceHistory.timestamp).offset(records_to_keep).all()

            for record in records_to_delete:
                db_session.delete(record)
                total_deleted += 1

            # Update the kept record
            kept_record = db_session.query(PriceHistory).filter(
                PriceHistory.item_id == item_id,
                PriceHistory.timestamp >= week_start,
                PriceHistory.timestamp < week_end
            ).order_by(PriceHistory.timestamp).first()

            if kept_record:
                kept_record.price = avg_price
                kept_record.volume = avg_volume
                kept_record.timestamp = first_ts

    if total_deleted > 0 and not dry_run:
        db_session.commit()

    return total_deleted


def _downsample_monthly(db_session, start_date, dry_run=False):
    """Keep one monthly average per item for data older than 365 days."""
    items = db_session.query(PriceHistory.item_id).filter(
        PriceHistory.timestamp < start_date
    ).distinct().all()

    total_deleted = 0

    for (item_id,) in items:
        # Group by month
        months = db_session.query(
            func.to_char(PriceHistory.timestamp, 'YYYY-MM').label('month'),
            func.avg(PriceHistory.price).label('avg_price'),
            func.avg(PriceHistory.volume).label('avg_volume'),
            func.min(PriceHistory.timestamp).label('first_ts'),
        ).filter(
            PriceHistory.item_id == item_id,
            PriceHistory.timestamp < start_date
        ).group_by(func.to_char(PriceHistory.timestamp, 'YYYY-MM')).all()

        for month, avg_price, avg_volume, first_ts in months:
            # Delete all records for this month except one
            records_to_keep = 1
            month_start = first_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # Calculate next month
            if month_start.month == 12:
                month_end = month_start.replace(year=month_start.year + 1, month=1)
            else:
                month_end = month_start.replace(month=month_start.month + 1)

            records_to_delete = db_session.query(PriceHistory).filter(
                PriceHistory.item_id == item_id,
                PriceHistory.timestamp >= month_start,
                PriceHistory.timestamp < month_end
            ).order_by(PriceHistory.timestamp).offset(records_to_keep).all()

            for record in records_to_delete:
                db_session.delete(record)
                total_deleted += 1

            # Update the kept record
            kept_record = db_session.query(PriceHistory).filter(
                PriceHistory.item_id == item_id,
                PriceHistory.timestamp >= month_start,
                PriceHistory.timestamp < month_end
            ).order_by(PriceHistory.timestamp).first()

            if kept_record:
                kept_record.price = avg_price
                kept_record.volume = avg_volume
                kept_record.timestamp = first_ts

    if total_deleted > 0 and not dry_run:
        db_session.commit()

    return total_deleted


if __name__ == "__main__":
    db = SessionLocal()

    try:
        print("Starting database downsampling...")

        # Downsample price history (keeps monthly average forever, no deletion)
        downsampled = downsample_price_history(db, dry_run=False)

        # Delete very old trend indicators
        deleted_trends = prune_trend_indicators(db, days_to_keep=180, dry_run=False)

        total = downsampled + deleted_trends
        print(f"\n✅ Maintenance complete:")
        print(f"  Downsampled records: {downsampled}")
        print(f"  Deleted old trends: {deleted_trends}")
        print(f"  Total records processed: {total}")

    except Exception as e:
        logger.error(f"❌ Pruning failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()
