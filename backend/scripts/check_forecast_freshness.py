#!/usr/bin/env python3
"""Fail when a forecast run did not persist the current snapshot day's forecasts.

Green CI is not evidence of collection — see the collector audit in
docs/changelog/2026-07-31-accuracy-work-closed.md, where three scheduled jobs
reported success while storing zero rows. ``item_forecasts`` accumulated only 5
distinct forecast dates in eight months for the same reason: nothing asserted
the output.

The verdict reads the monthly ops partitions (``db/ops_store.py``), the store every scoring
and serving reader uses since Phase 2. Postgres went read-only on 2026-10-30, so its newest
date is reported for comparison only: behind the partitions, or unreadable, is a ``::notice``.
Missing partitions are a stale verdict, not a crash.

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
from db import ops_store
from sqlalchemy import func

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def newest_forecast_date(dates):
    """Maximum non-null date, or None."""
    present = [d for d in dates if d is not None]
    if not present:
        return None
    return max(present)


def freshness_verdict(store_newest, expected):
    """Decide whether the ops partitions carry a forecast for *expected* or later."""
    if store_newest is None:
        return False, "the item_forecasts partitions hold no forecasts at all"
    if store_newest < expected:
        return False, f"the item_forecasts partitions' newest forecast_date is {store_newest}, expected {expected}"
    return True, f"The partitions carry forecasts for {expected} (newest {store_newest})."


def postgres_note(db_newest, store_newest):
    """A ``::notice`` when Postgres is unreadable or behind the partitions, else None. Never a failure."""
    if db_newest is None:
        return "::notice title=Postgres::item_forecasts (Postgres) could not be read; the verdict reads the partitions"
    if store_newest is not None and db_newest < store_newest:
        return (
            f"::notice title=Postgres::item_forecasts (Postgres) newest forecast_date is {db_newest}, "
            f"behind the partitions' {store_newest}; Postgres is read-only and the verdict reads the partitions"
        )
    return None


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


def _shadow_inputs(limit=10):
    """(recent production forecast dates, distinct candidate batches) from the partitions."""
    read = ops_store.reader(("item_forecasts", "forecast_candidates"))
    dates = read(
        f"SELECT DISTINCT forecast_date AS d FROM item_forecasts ORDER BY forecast_date DESC LIMIT {int(limit)}"
    )
    batches = read("SELECT DISTINCT forecast_date AS d, horizon_days AS h, component AS c FROM forecast_candidates")
    return [_as_date(r.d) for r in dates], {(_as_date(r.d), int(r.h), r.c) for r in batches}


def _db_newest():
    """Postgres's newest forecast date, for comparison only; None when it cannot be read."""
    try:
        db = SessionLocal()
        try:
            return _as_date(db.query(func.max(ItemForecast.forecast_date)).scalar())
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"item_forecasts (Postgres) could not be read: {e}")
        return None


def _store_newest():
    """The partitions' newest forecast date; None when they hold none (or none exist)."""
    try:
        rows = ops_store.reader(("item_forecasts",))("SELECT max(forecast_date) AS d FROM item_forecasts")
    except ops_store.MissingTable as e:
        logger.error(str(e))
        return None
    return _as_date(rows[0].d) if rows else None


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

    store_newest = _store_newest()
    db_newest = _db_newest()
    logger.info(f"Newest forecast_date: partitions {store_newest}, Postgres {db_newest} (comparison only).")
    note = postgres_note(db_newest, store_newest)
    if note:
        print(note, flush=True)
    ok, message = freshness_verdict(store_newest, expected)
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
    # store without candidate partitions reports unknown rather than failing.
    try:
        readiness = shadow_collection_readiness(*_shadow_inputs())
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
