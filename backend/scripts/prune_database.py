#!/usr/bin/env python3
"""
Database pruning and downsampling script (optimized for performance).
Implements tiered data retention using batch SQL operations:
  - Week (0-7 days): All data points
  - Month (7-30 days): Daily average
  - Year (30-365 days): Weekly average
  - Older (365+ days): Monthly average (kept indefinitely)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pruning")


def prune_trend_indicators(db_session, days_to_keep=180, dry_run=False):
    """Delete trend indicators older than days_to_keep."""
    from database import TrendIndicator
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


def prune_daily_analysis(db_session, days_to_keep=90, dry_run=False):
    """Delete daily analysis records older than days_to_keep."""
    from database import DailyAnalysis
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

    query = db_session.query(DailyAnalysis).filter(
        DailyAnalysis.analysis_date < cutoff_date.date()
    )

    count = query.count()

    if count > 0 and not dry_run:
        query.delete()
        db_session.commit()
        logger.info(f"Deleted {count} daily analysis records older than {days_to_keep} days")

    return count


def prune_event_analyses(db_session, days_to_keep=365, dry_run=False):
    """Delete event impact and correlation records older than days_to_keep."""
    from database import EventImpact, EventCorrelation
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

    total = 0

    for model_cls, label in [(EventImpact, "event impacts"), (EventCorrelation, "event correlations")]:
        query = db_session.query(model_cls).filter(
            model_cls.created_at < cutoff_date
        )
        count = query.count()
        if count > 0 and not dry_run:
            query.delete()
            db_session.commit()
            logger.info(f"Deleted {count} {label} older than {days_to_keep} days")
        total += count

    return total


def prune_price_history(db_session, days_to_keep_granular=7, dry_run=False):
    """Backward-compatible alias for the current downsampling routine."""
    return downsample_price_history(
        db_session,
        days_to_keep_granular=days_to_keep_granular,
        dry_run=dry_run,
    )


def downsample_price_history(db_session, days_to_keep_granular=7, dry_run=False):
    """
    Downsample price history with tiered strategy.
    - 0-7 days: Keep all data
    - 7-30 days: Keep daily average (1 record per item per day)
    - 30-365 days: Keep weekly average (1 record per item per week)
    - 365+ days: Keep monthly average (1 record per item per month)
    """
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    year_ago = now - timedelta(days=365)

    total_deleted = 0

    # Tier 1: 7-30 days → Daily average
    deleted = _downsample_tier(
        db_session,
        week_ago,
        month_ago,
        "date(timestamp)::text",
        "daily averages (7-30 days)",
        dry_run
    )
    total_deleted += deleted

    # Tier 2: 30-365 days → Weekly average
    deleted = _downsample_tier(
        db_session,
        month_ago,
        year_ago,
        "to_char(timestamp, 'YYYY-WW')",
        "weekly averages (30-365 days)",
        dry_run
    )
    total_deleted += deleted

    # Tier 3: 365+ days → Monthly average
    deleted = _downsample_tier(
        db_session,
        year_ago,
        None,
        "to_char(timestamp, 'YYYY-MM')",
        "monthly averages (365+ days)",
        dry_run
    )
    total_deleted += deleted

    return total_deleted


def _downsample_tier(db_session, start_date, end_date, group_expr, desc, dry_run):
    """
    Delete records, keeping one per group with earliest timestamp.
    Uses per-item batching to avoid Supabase free-tier timeouts on large tiers.
    """

    try:
        db_session.execute(text("SET statement_timeout = '180000'"))
        db_session.execute(text("SET lock_timeout = '120000'"))
    except Exception:
        pass

    if end_date:
        raw_filter = f"timestamp >= '{start_date}' AND timestamp < '{end_date}'"
    else:
        raw_filter = f"timestamp < '{start_date}'"

    total_in_tier = db_session.execute(
        text(f"SELECT COUNT(*) FROM price_history WHERE {raw_filter}")
    ).scalar() or 0

    if total_in_tier == 0:
        logger.info(f"Downsampled 0 records to {desc}")
        return 0

    if dry_run:
        logger.info(f"Would downsample {total_in_tier:,} records to {desc}")
        return total_in_tier

    # Try single-pass first (works for small-medium tiers)
    try:
        delete_sql = f"""
        DELETE FROM price_history ph
        USING (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY item_id, ({group_expr})
                ORDER BY timestamp ASC
            ) AS rn
            FROM price_history
            WHERE {raw_filter}
        ) keep
        WHERE ph.id = keep.id AND keep.rn > 1
        """
        result = db_session.execute(text(delete_sql))
        db_session.commit()
        deleted = result.rowcount
        logger.info(f"Downsampled {deleted:,} records to {desc}")
        return deleted
    except Exception:
        db_session.rollback()

    # Fallback: per-item batching (handles large tiers without timeout)
    total_deleted = 0
    item_ids = db_session.execute(text(f"""
        SELECT DISTINCT item_id FROM price_history WHERE {raw_filter}
    """)).fetchall()
    item_ids = [r[0] for r in item_ids]
    logger.info(f"  Processing {len(item_ids)} items in batches of 20 for {desc}...")

    for i in range(0, len(item_ids), 20):
        batch = item_ids[i:i + 20]
        id_list = tuple(batch)
        delete_batch_sql = f"""
        DELETE FROM price_history ph
        USING (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY ({group_expr})
                ORDER BY timestamp ASC
            ) AS rn
            FROM price_history
            WHERE {raw_filter} AND item_id = ANY(:id_list)
        ) keep
        WHERE ph.id = keep.id AND keep.rn > 1
        """
        try:
            result = db_session.execute(
                text(delete_batch_sql), {'id_list': id_list}
            )
            db_session.commit()
            total_deleted += result.rowcount
        except Exception as e2:
            db_session.rollback()
            logger.error(f"  Batch error ({batch[0]}-{batch[-1]}): {e2}")

    logger.info(f"Downsampled {total_deleted:,} records to {desc}")
    return total_deleted


if __name__ == "__main__":
    db = SessionLocal()

    try:
        print("Starting database downsampling...")
        print("=" * 70)

        downsampled = downsample_price_history(db, dry_run=False)
        pruned_trends = prune_trend_indicators(db, dry_run=False)
        pruned_daily = prune_daily_analysis(db, days_to_keep=90, dry_run=False)
        pruned_events = prune_event_analyses(db, days_to_keep=365, dry_run=False)

        total = downsampled + pruned_trends + pruned_daily + pruned_events
        print(f"\n✅ Maintenance complete:")
        print(f"  Downsampled price records: {downsampled:,}")
        print(f"  Deleted old trend indicators: {pruned_trends:,}")
        print(f"  Deleted old daily analyses: {pruned_daily:,}")
        print(f"  Deleted old event records: {pruned_events:,}")
        print(f"  Total records processed: {total:,}")
        print("=" * 70)

    except Exception as e:
        logger.error(f"❌ Pruning failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()
