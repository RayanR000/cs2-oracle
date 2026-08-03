#!/usr/bin/env python3
"""
One-shot migration: dump all operational data from Supabase → Parquet.

Writes to ``price-archive/ops/{table}.parquet`` for every table that has
been migrated off the relational database.

Idempotent — safe to re-run (upsert semantics via dedup keys).

Usage::

    python scripts/migrate_to_parquet.py [--table events]

Omit ``--table`` to migrate all tables.
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, utcnow_naive
from sqlalchemy import text
from db.parquet import append_table, ensure_ops_dir, read_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("migrate_to_parquet")

# A list/tuple is not hashable with pandas isnull, so keep the schema
# as simple column‑name lists + dedup key lists per table.

TABLES = {
    "events": {
        "query": "SELECT id, type, timestamp, description, created_at FROM events",
        "dedup": ["id"],
        "coerce_date": [],
    },
    "item_forecasts": {
        "query": (
            "SELECT id, item_id, forecast_date, horizon_days, "
            "price_low, price_mid, price_high, current_price, "
            "direction, confidence, model_version, created_at "
            "FROM item_forecasts"
        ),
        "dedup": ["item_id", "forecast_date", "horizon_days"],
        "coerce_date": ["forecast_date"],
    },
    "forecast_outcomes": {
        "query": (
            "SELECT id, forecast_id, item_id, forecast_date, horizon_days, "
            "target_date, current_price, predicted_price_low, predicted_price_mid, "
            "predicted_price_high, actual_price, direction_predicted, direction_actual, "
            "direction_correct, in_interval, abs_error, pct_error, model_version, evaluated_at "
            "FROM forecast_outcomes"
        ),
        "dedup": ["forecast_id"],
        "coerce_date": ["forecast_date", "target_date"],
    },
    "prediction_accuracy": {
        # price_tier is part of the identity of a row, not an attribute of it:
        # the table holds a row per price band, one for the >=$1 headline, and
        # one all-tiers aggregate, all sharing the other four key columns.
        # Omitting it here is what left the mirror unable to tell them apart
        # (migration 0019 added the column, 0020 put it in the unique key).
        "query": (
            "SELECT id, prediction_type, evaluation_date, horizon_days, "
            "model_version, price_tier, evaluation_window_days, sample_count, "
            "metrics, created_at "
            "FROM prediction_accuracy"
        ),
        "dedup": ["prediction_type", "evaluation_date", "horizon_days",
                  "model_version", "price_tier"],
        "coerce_date": ["evaluation_date"],
    },
    "supply_snapshots": {
        "query": (
            "SELECT item_id, snapshot_date, sell_listings, skinport_quantity, source, created_at "
            "FROM supply_snapshots"
        ),
        "dedup": ["item_id", "snapshot_date"],
        "coerce_date": ["snapshot_date"],
    },
    "social_mentions": {
        "query": (
            "SELECT item_id, source, post_id, subreddit, post_title, "
            "post_score, post_url, sentiment_score, mentioned_at, collected_at "
            "FROM social_mentions"
        ),
        "dedup": ["item_id", "source", "post_id"],
        "coerce_date": [],
    },
    "collection_runs": {
        "query": "SELECT * FROM collection_runs",
        "dedup": ["id"],
        "coerce_date": [],
    },
    "accuracy_alerts": {
        "query": "SELECT * FROM accuracy_alerts",
        "dedup": ["id"],
        "coerce_date": [],
    },
    "event_impacts_denorm": {
        "query": (
            "SELECT ei.event_id, ei.item_id, "
            "e.type AS event_type, e.description AS event_description, e.timestamp AS event_timestamp, "
            "ei.price_day_before, ei.price_day_1, ei.price_day_3, ei.price_day_7, "
            "ei.impact_pct_1day, ei.impact_pct_3day, ei.impact_pct_7day, "
            "ei.peak_impact_pct, ei.peak_impact_day, ei.duration_days, ei.z_score, "
            "ec.confidence_score "
            "FROM event_impacts ei "
            "JOIN events e ON e.id = ei.event_id "
            "LEFT JOIN event_correlations ec ON ec.event_id = ei.event_id AND ec.item_id = ei.item_id"
        ),
        "dedup": ["event_id", "item_id"],
        "coerce_date": ["event_timestamp"],
    },
}


def _coerce_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date
    return df


def migrate_table(table: str, config: dict):
    logger.info("Migrating %s …", table)

    db = SessionLocal()
    try:
        result = db.execute(text(config["query"]))
        col_names = result.keys()
        rows = result.fetchall()
    except Exception as e:
        logger.error("  Query failed for %s: %s", table, e)
        return 0
    finally:
        db.close()

    if not rows:
        logger.info("  No rows to migrate for %s", table)
        return 0

    data = [dict(zip(col_names, row)) for row in rows]

    # Coerce date columns to Python date objects
    df = pd.DataFrame(data)
    df = _coerce_dates(df, config["coerce_date"])

    # Convert datetimes to naive UTC for Parquet
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    logger.info("  Loaded %d rows from DB", len(df))

    # Merge into existing Parquet
    existing = read_table(table)
    if not existing.empty:
        n_before = len(existing)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=config["dedup"], keep="last")
        combined.to_parquet(ensure_ops_dir() / f"{table}.parquet", index=False)
        logger.info("  Merged: %d existing + %d new → %d total", n_before, len(df), len(combined))
    else:
        df.to_parquet(ensure_ops_dir() / f"{table}.parquet", index=False)
        logger.info("  Written: %d rows", len(df))

    return len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        choices=list(TABLES.keys()) + ["all"],
        default="all",
        help="Table to migrate (default: all)",
    )
    args = parser.parse_args()

    ensure_ops_dir()
    tables = TABLES if args.table == "all" else {args.table: TABLES[args.table]}
    total = 0
    for name, cfg in tables.items():
        total += migrate_table(name, cfg)

    logger.info("Done — %d total rows migrated across %d tables", total, len(tables))


if __name__ == "__main__":
    main()
