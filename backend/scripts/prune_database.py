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
    """Delete daily analysis records older than days_to_keep using raw SQL."""
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

    count = db_session.execute(
        text("SELECT COUNT(*) FROM daily_analysis WHERE analysis_date < :cutoff"),
        {"cutoff": cutoff_date.date()}
    ).scalar() or 0

    if count > 0 and not dry_run:
        db_session.execute(
            text("DELETE FROM daily_analysis WHERE analysis_date < :cutoff"),
            {"cutoff": cutoff_date.date()}
        )
        db_session.commit()
        logger.info(f"Deleted {count} daily analysis records older than {days_to_keep} days")

    return count


def prune_event_analyses(db_session, days_to_keep=365, dry_run=False):
    """Delete event impact and correlation records older than days_to_keep."""
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    total = 0

    for table, label in [("event_impacts", "event impacts"), ("event_correlations", "event correlations")]:
        count = db_session.execute(
            text(f"SELECT COUNT(*) FROM {table} WHERE created_at < :cutoff"),
            {"cutoff": cutoff_date}
        ).scalar() or 0

        if count > 0 and not dry_run:
            db_session.execute(
                text(f"DELETE FROM {table} WHERE created_at < :cutoff"),
                {"cutoff": cutoff_date}
            )
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
    # (start must be earlier than end: month_ago < week_ago)
    deleted = _downsample_tier(
        db_session,
        month_ago,
        week_ago,
        "CAST(date(timestamp) AS TEXT)",
        "daily averages (7-30 days)",
        dry_run
    )
    total_deleted += deleted

    # Tier 2: 30-365 days → Weekly average
    # (start must be earlier than end: year_ago < month_ago)
    deleted = _downsample_tier(
        db_session,
        year_ago,
        month_ago,
        "to_char(timestamp, 'IYYY-IW')",
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
    Keep one record per (item_id, time_bucket) that is closest to the
    median price for that bucket.  All other records in the bucket
    are deleted.  Using median (PERCENTILE_CONT) instead of mean
    makes this robust against outlier price spikes.
    """

    try:
        db_session.execute(text("SET statement_timeout = '180000'"))
        db_session.execute(text("SET lock_timeout = '120000'"))
    except Exception:
        pass

    if end_date:
        raw_filter = "timestamp >= :start_date AND timestamp < :end_date"
    else:
        raw_filter = "timestamp < :start_date"

    params = {"start_date": start_date}
    if end_date:
        params["end_date"] = end_date

    total_in_tier = db_session.execute(
        text(f"SELECT COUNT(*) FROM price_history WHERE {raw_filter}"),
        params,
    ).scalar() or 0

    if total_in_tier == 0:
        logger.info(f"Downsampled 0 records to {desc}")
        return 0

    if dry_run:
        logger.info(f"Would downsample {total_in_tier:,} records to {desc}")
        return total_in_tier

    # Keep the one row per (item_id, time_bucket) closest to the median price.
    # Uses a CTE to compute group medians (PERCENTILE_CONT), then ROW_NUMBER to
    # pick the nearest row.  Unlike the original code this does NOT nest
    # PERCENTILE_CONT inside OVER(), so PostgreSQL accepts it.
    def build_delete_sql(extra_filter=""):
        return f"""
        WITH bucket_medians AS (
            SELECT item_id, ({group_expr}) AS bucket,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS median_price
            FROM price_history
            WHERE {raw_filter} {extra_filter}
            GROUP BY item_id, ({group_expr})
        )
        DELETE FROM price_history ph
        USING (
            SELECT id FROM (
                SELECT ph2.id,
                    ROW_NUMBER() OVER (
                        PARTITION BY ph2.item_id, ({group_expr})
                        ORDER BY ABS(ph2.price - bm.median_price)
                    ) AS rn
                FROM price_history ph2
                JOIN bucket_medians bm
                  ON ph2.item_id = bm.item_id
                 AND ({group_expr}) = bm.bucket
            ) sub
            WHERE rn > 1
        ) keep
        WHERE ph.id = keep.id
        """

    # Single-pass: try full range first
    try:
        delete_sql = build_delete_sql()
        result = db_session.execute(text(delete_sql), params)
        db_session.commit()
        deleted = result.rowcount
        logger.info(f"Downsampled {deleted:,} records to {desc}")
        return deleted
    except Exception as e:
        db_session.rollback()
        logger.warning(f"Single-pass downsample failed, falling back to per-item batching: {e}")

    # Fallback: per-item batching (handles large tiers without timeout)
    total_deleted = 0
    item_ids = db_session.execute(
        text(f"SELECT DISTINCT item_id FROM price_history WHERE {raw_filter}"),
        params,
    ).fetchall()
    item_ids = [r[0] for r in item_ids]
    logger.info(f"  Processing {len(item_ids)} items in batches of 20 for {desc}...")

    for i in range(0, len(item_ids), 20):
        batch = item_ids[i:i + 20]
        id_list = list(batch)
        delete_batch_sql = build_delete_sql("AND item_id = ANY(:id_list)")
        try:
            batch_params = {**params, "id_list": id_list}
            result = db_session.execute(
                text(delete_batch_sql), batch_params
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
