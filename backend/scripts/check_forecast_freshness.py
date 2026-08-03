#!/usr/bin/env python3
"""Fail when a forecast run did not persist today's forecasts.

Green CI is not evidence of collection — see the collector audit in
docs/changelog/2026-07-31-accuracy-work-closed.md, where three scheduled jobs
reported success while storing zero rows. ``item_forecasts`` accumulated only 5
distinct forecast dates in eight months for the same reason: nothing asserted
the output.

Both stores are checked because ``api/routes`` reads the Parquet mirror first
and falls back to the DB, so a DB row without a mirror row serves nothing.

Usage:
    python scripts/check_forecast_freshness.py
    python scripts/check_forecast_freshness.py --expected-date 2026-08-03
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func

from database import SessionLocal, ItemForecast
from db.parquet import ParquetQuery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def newest_forecast_date(dates):
    """Maximum non-null date, or None."""
    present = [d for d in dates if d is not None]
    if not present:
        return None
    return max(present)


def freshness_verdict(db_newest, parquet_newest, expected):
    """Decide whether both stores carry a forecast for *expected* or later."""
    problems = []
    if db_newest is None:
        problems.append("item_forecasts (DB) holds no forecasts at all")
    elif db_newest < expected:
        problems.append(
            f"item_forecasts (DB) newest forecast_date is {db_newest}, expected {expected}"
        )
    if parquet_newest is None:
        problems.append("the item_forecasts Parquet mirror holds no forecasts at all")
    elif parquet_newest < expected:
        problems.append(
            f"the item_forecasts Parquet mirror newest forecast_date is "
            f"{parquet_newest}, expected {expected}"
        )
    if problems:
        return False, "; ".join(problems)
    return True, f"Both stores carry forecasts for {expected} (DB {db_newest}, Parquet {parquet_newest})."


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _db_newest():
    db = SessionLocal()
    try:
        return _as_date(db.query(func.max(ItemForecast.forecast_date)).scalar())
    finally:
        db.close()


def _parquet_newest():
    with ParquetQuery("item_forecasts") as q:
        return _as_date(q.scalar("SELECT max(forecast_date) FROM item_forecasts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-date", default=None, help="ISO date; defaults to today (UTC).")
    args = parser.parse_args()

    expected = (
        datetime.fromisoformat(args.expected_date).date()
        if args.expected_date
        else datetime.utcnow().date()
    )

    ok, message = freshness_verdict(_db_newest(), _parquet_newest(), expected)
    if ok:
        logger.info(message)
        return 0
    logger.error(message)
    logger.error(
        "The forecast run reported success without persisting forecasts. The API "
        "serves the newest available forecast, so this degrades silently rather "
        "than erroring."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
