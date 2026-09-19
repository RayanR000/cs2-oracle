#!/usr/bin/env python3
"""
Task runner for automated maintenance and collection.
Used by GitHub Actions to trigger specific pipeline tasks.
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.pipeline import DataPipeline
from database import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("task_runner")


def run_migrations(revision="head"):
    """Run Alembic migrations using the current Python interpreter."""
    cmd = [sys.executable, "-m", "alembic", "upgrade", revision]

    try:
        logger.info(f"Running migrations: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent.parent))
        return {"status": "success", "revision": revision, "returncode": result.returncode}
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error("Migration command failed. Install backend requirements and retry.")
        raise RuntimeError(f"Could not run migrations to {revision}: {e}") from e


# Row-count fields every task can return. This guard used to key on
# `items_collected` alone — a field only collectors/pipeline.py sets. Every other
# task could therefore return status "success" with zero rows and still exit 0,
# which is exactly how the Steam supply scraper (429s from 2026-07-16) and the
# Reddit collector (403s, never stored a single row) both stayed dead behind a
# green CI badge.
ROW_COUNT_FIELDS = (
    "items_collected",  # collectors/pipeline.py
    "total_records",  # scripts/backtest_accuracy.py
    "impacts_written",  # scripts/event_correlation_analysis.py
    "patterns_written",  # scripts/event_correlation_analysis.py
    "correlations_written",  # scripts/event_correlation_analysis.py
    "total_mentions",  # collectors/social_sentiment.py
    "inserted",  # collectors/social_sentiment.py
    "supply_rows",  # collectors/supply_depth.py
    "volume_rows",  # collectors/sales_volume.py
    "reddit_event_rows",  # collectors/reddit_events.py
    "case_panel_rows",  # scripts/build_case_panel.py
    "sticker_panel_rows",  # scripts/build_sticker_panel.py
    "forecasts_written",  # scripts/forecast_prices.py (production rows stay fatal at zero)
)

# Statuses that mean "this task legitimately had nothing to do", as opposed to
# "this task produced nothing because it is broken". Deliberately a tiny
# allowlist: every status not on it flows into the guards below, so a new
# no-op-shaped outcome has to be added here consciously rather than inheriting
# a green run. `no_events_in_window` is event_correlation_analysis.py's — see
# NO_EVENTS_STATUS there for why an empty event window is a calendar fact.
NO_OP_STATUSES = ("no_events_in_window",)


def check_results(task_name, results) -> None:
    """Exit non-zero unless every result reports real work or a known no-op.

    Extracted from `run_task` so it is testable without a DB session: `.env`
    here points at PRODUCTION Supabase and the engine binds at import, so a test
    that instantiated the runner would be querying prod.
    """
    results = [r for r in results if isinstance(r, dict)]

    # Pipeline methods catch their own exceptions and return status dicts; exit
    # non-zero so scheduled workflows report the failure instead of showing green
    # on a run that collected nothing.
    failures = [r for r in results if r.get("status") in ("failed", "error")]
    if failures:
        logger.error(f"❌ TASK '{task_name}' reported failure: {failures[0].get('error', failures[0])}")
        sys.exit(1)

    # A known no-op: loud, and exit 0. It has to be recognised BEFORE the
    # zero-row guard, because its counts are all legitimately zero.
    no_ops = [r for r in results if r.get("status") in NO_OP_STATUSES]
    for r in no_ops:
        logger.warning(
            f"⚠️  TASK '{task_name}' had nothing to do (status "
            f"{r.get('status')!r}) and wrote no rows — not a failure. "
            f"Result: {r}"
        )
    # Identity, not equality: two tasks in one run can return equal dicts, and
    # `!=` would drop both when only one was the no-op.
    results = [r for r in results if not any(r is n for n in no_ops)]

    def _zero_row(r) -> bool:
        """True if r reports row counts and every one of them is zero."""
        if r.get("status") != "success":
            return False
        counts = [r[f] for f in ROW_COUNT_FIELDS if isinstance(r.get(f), (int, float))]
        return bool(counts) and not any(counts)

    # Treat zero-row results as failures (all endpoints likely down). For
    # event_correlation specifically this is the guard that must STAY fatal:
    # "events exist but zero impacts were written" is the price_history bug
    # (2026-07-19 -> 2026-08-02, green the whole time), not an empty calendar.
    zero_rows = [r for r in results if _zero_row(r)]
    if zero_rows:
        logger.error(
            f"❌ TASK '{task_name}' completed with ZERO rows written — "
            "all upstream endpoints may be down or blocking this IP. "
            f"Result: {zero_rows[0]}"
        )
        sys.exit(1)

    # "skipped" passed both guards above: not a failure status, and no count
    # fields to inspect. pipeline.py returns it when there are no items to
    # update, which is itself a zero-row outcome worth surfacing.
    skipped = [r for r in results if r.get("status") == "skipped"]
    if skipped:
        logger.error(f"❌ TASK '{task_name}' was SKIPPED and wrote nothing: {skipped[0].get('reason', skipped[0])}")
        sys.exit(1)

    # Shadow incompleteness is loud but non-fatal: production > 0 with zero
    # shadow trips no guard above (the production count is non-zero), and a
    # fresh collection legitimately reports INSUFFICIENT_EVIDENCE for weeks.
    # Only an integrity conflict fails, and that raises inside the forecast
    # run itself rather than here.
    for r in results:
        if not isinstance(r, dict) or r.get("status") != "success":
            continue
        expected = r.get("candidate_batches_expected")
        written = r.get("candidate_batches_written")
        if (
            isinstance(expected, int)
            and isinstance(written, int)
            and expected > 0
            and written < expected
        ):
            logger.warning(
                f"⚠️  TASK '{task_name}' wrote production but shadow collection is "
                f"incomplete ({written}/{expected} batches). "
                f"Centre/ranking: {r.get('centre_candidates_written')}/{r.get('ranking_candidates_written')} rows. "
                "Reports will read INSUFFICIENT_EVIDENCE until the gate fills."
            )


def run_task(task_name):
    db = SessionLocal()
    pipeline = DataPipeline(db_session=db)

    try:
        start_time = datetime.now()
        result = None

        if task_name == "aggregate":
            logger.info("=" * 60)
            logger.info("TASK: Full Aggregator Scrape (All items)")
            logger.info("=" * 60)
            result = pipeline.run_full_aggregator_collection()

            if isinstance(result, dict):
                logger.info(f"✅ SUCCESS - Items collected: {result.get('items_collected', 'N/A')}")
                logger.info(f"  Errors: {result.get('errors', 0)}")
                logger.info(f"  Duration: {result.get('duration_seconds', 'N/A')}s")

            print(f"RESULT: {result}")

        elif task_name == "priority":
            logger.info("=" * 60)
            logger.info("TASK: Priority Aggregator Scrape (Top 2000)")
            logger.info("=" * 60)
            result = pipeline.run_priority_collection()
            print(f"RESULT: {result}")

        elif task_name == "migrate":
            logger.info("=" * 60)
            logger.info("TASK: Database Migration (Alembic upgrade head)")
            logger.info("=" * 60)
            result = run_migrations("head")
            print(f"RESULT: {result}")

        elif task_name == "backtest":
            logger.info("=" * 60)
            logger.info("TASK: Backtest Accuracy (All Types)")
            logger.info("=" * 60)
            from scripts.backtest_accuracy import run_backtest

            result = run_backtest()
            print(f"RESULT: {result}")

        elif task_name == "backtest_historical":
            logger.info("=" * 60)
            logger.info("TASK: Historical Walk-Forward Backtest")
            logger.info("=" * 60)
            from scripts.backtest_accuracy import run_backtest

            result = run_backtest(types=["historical"])
            print(f"RESULT: {result}")

        elif task_name == "event_correlation":
            logger.info("=" * 60)
            logger.info("TASK: Event Correlation Analysis")
            logger.info("=" * 60)
            from scripts.event_correlation_analysis import run_analysis

            result = run_analysis(days_back=90)
            print(f"RESULT: {result}")

        elif task_name == "walkforward_report":
            logger.info("=" * 60)
            logger.info("TASK: Walk-Forward Backtest Report (Per-Horizon)")
            logger.info("=" * 60)
            from scripts.archive.backtest_walkforward_report import run_walkforward_report

            result = run_walkforward_report()
            print(f"RESULT: {result}")

        elif task_name == "reddit_social":
            logger.info("=" * 60)
            logger.info("TASK: Reddit Social Sentiment Collection")
            logger.info("=" * 60)
            from collectors.social_sentiment import run as run_reddit_social

            result = run_reddit_social()
            print(f"RESULT: {result}")

        else:
            logger.error(f"Unknown task: {task_name}")
            sys.exit(1)

        check_results(task_name, (result,))

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total task time: {elapsed:.1f} seconds")

    except Exception as e:
        logger.error(f"❌ TASK FAILED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_task.py <task_name>")
        print(
            "Tasks: aggregate, priority, migrate, backtest, backtest_historical, walkforward_report, event_correlation, reddit_social"
        )
        sys.exit(1)

    run_task(sys.argv[1])
