#!/usr/bin/env python3
"""Fail when a forecast run did not persist the current snapshot day's forecasts.

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

from collectors.snapshot_date import resolve_snapshot_date
from database import ItemForecast, SessionLocal
from db.parquet import ParquetQuery
from sqlalchemy import func

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
        problems.append(f"item_forecasts (DB) newest forecast_date is {db_newest}, expected {expected}")
    if parquet_newest is None:
        problems.append("the item_forecasts Parquet mirror holds no forecasts at all")
    elif parquet_newest < expected:
        problems.append(
            f"the item_forecasts Parquet mirror newest forecast_date is {parquet_newest}, expected {expected}"
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


def shadow_collection_readiness(production_dates, candidate_batches, horizons=(3, 7, 14, 30), window=3):
    """Whether shadow collection is keeping up with production.

    The three most recent production forecast dates each need a centre
    candidate batch at every horizon. A missing batch resets readiness and
    names the gap. Pure function over dates and (date, horizon, component)
    batch identities, so it is testable without a database.
    """
    recent = sorted({d for d in production_dates if d is not None})[-window:]
    if len(recent) < window:
        return {
            "shadow_collection_ready": False,
            "gap": f"only {len(recent)} production date(s), need {window}",
        }
    have = set(candidate_batches or ())
    for day in recent:
        for horizon in horizons:
            if (day, horizon, "centre") not in have:
                return {
                    "shadow_collection_ready": False,
                    "gap": f"missing centre batch for {day} h={horizon}",
                }
    return {"shadow_collection_ready": True, "gap": None}


def _shadow_batches():
    """Distinct (forecast_date, horizon, component) candidate batches, or None."""
    from database import ForecastCandidate

    db = SessionLocal()
    try:
        rows = db.query(
            ForecastCandidate.forecast_date, ForecastCandidate.horizon_days, ForecastCandidate.component
        ).distinct().all()
        return {(_as_date(r[0]), r[1], r[2]) for r in rows}
    finally:
        db.close()


def _recent_production_dates(limit=10):
    db = SessionLocal()
    try:
        rows = (
            db.query(ItemForecast.forecast_date)
            .distinct()
            .order_by(ItemForecast.forecast_date.desc())
            .limit(limit)
            .all()
        )
        return [_as_date(r[0]) for r in rows]
    finally:
        db.close()


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
    parser.add_argument(
        "--expected-date",
        default=None,
        help="ISO date; defaults to the current snapshot day (resolve_snapshot_date).",
    )
    args = parser.parse_args()

    # Not the wall-clock date. A forecast is stamped with the day of the price
    # dump it was anchored on, and `resolve_snapshot_date` is the single source
    # of truth for that (boundary 22:00 UTC, see collectors/snapshot_date.py).
    # Comparing against `utcnow().date()` failed every run between 00:00 and
    # 22:00 UTC on forecasts that were correctly dated -- the scheduled chain
    # only passed because the aggregator cron fires at 23:00, after the
    # boundary. Both sides must read the same rule or the gate reports a
    # persistence failure that did not happen.
    expected = datetime.fromisoformat(args.expected_date).date() if args.expected_date else resolve_snapshot_date()

    ok, message = freshness_verdict(_db_newest(), _parquet_newest(), expected)
    if ok:
        logger.info(message)
    else:
        logger.error(message)
        logger.error(
            "The forecast run reported success without persisting forecasts. The API "
            "serves the newest available forecast, so this degrades silently rather "
            "than erroring."
        )

    # Shadow collection health: loud but non-fatal. A fresh collection reads
    # INSUFFICIENT_EVIDENCE for weeks, which is expected, not broken. A
    # database without migration 0027 reports unknown rather than failing.
    try:
        readiness = shadow_collection_readiness(_recent_production_dates(), _shadow_batches())
    except Exception as e:
        logger.warning(f"Shadow collection status unknown ({e}); skipping the readiness check.")
        readiness = {"shadow_collection_ready": False, "gap": "unknown"}
    if readiness["shadow_collection_ready"]:
        logger.info("Shadow collection ready: three consecutive dates fully collected.")
    else:
        logger.warning(f"Shadow collection not ready: {readiness['gap']}.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
