#!/usr/bin/env python3
"""
Backtesting system for ML forecast accuracy.
Computes accuracy metrics by comparing past predictions against actual outcomes.

Analysis type:
  - forecast:  ML price forecasts (7d / 30d horizons) from live DB

Usage:
    python scripts/backtest_accuracy.py                  # resolve new + score
    python scripts/backtest_accuracy.py --rescore        # score frozen only
    python scripts/backtest_accuracy.py --reresolve      # re-read the archive
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, PredictionAccuracy
from sqlalchemy import text
from backtest.price_resolution import load_voted_prices, smoothed_prices
from backtest.scoring import (
    FLAT_TOLERANCE,
    HEADLINE_MIN_TIER,
    bootstrap_ci,
    direction_from_return,
    headline_records,
    price_tier,
    score_by_tier,
    score_cohort,
)

# Above this rate of mature forecasts that can't be resolved by the shared
# estimator, refuse to report a number rather than silently score a shrunken
# cohort — that silent shrinkage is how the pre-fix metric moved unnoticed.
MAX_UNRESOLVABLE_PCT = 10.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backtest_accuracy")


def _upsert_accuracy(db, rows):
    """Replace existing accuracy rows for the same key then insert new ones."""
    for row in rows:
        filters = {
            "prediction_type": row["prediction_type"],
            "evaluation_date": row["evaluation_date"],
        }
        if row.get("horizon_days") is not None:
            filters["horizon_days"] = row["horizon_days"]
        else:
            filters["horizon_days"] = None
        if row.get("model_version") is not None:
            filters["model_version"] = row["model_version"]
        else:
            filters["model_version"] = None
        filters["price_tier"] = row.get("price_tier")

        existing = db.query(PredictionAccuracy).filter_by(**filters).first()
        if existing:
            existing.sample_count = row["sample_count"]
            existing.metrics = row["metrics"]
            existing.evaluation_window_days = row.get("evaluation_window_days")
            existing.created_at = row["created_at"]
        else:
            db.add(PredictionAccuracy(**row))
    db.commit()

    if rows:
        from db.parquet import append_table
        append_table("prediction_accuracy", rows,
                     ["prediction_type", "evaluation_date", "horizon_days",
                      "model_version", "price_tier"])


# ---------------------------------------------------------------------------
# 1. Forecast backtesting
# ---------------------------------------------------------------------------

def _store_forecast_outcomes(db, outcomes, reresolve: bool = False) -> int:
    """Persist per-forecast outcomes. Insert-only unless *reresolve*.

    Resolved actuals are frozen: a forecast_id that already has a row keeps
    its base_price and actual_price forever. Re-running the backtest after the
    archive gains source rows for an old target date must not rewrite history
    — that silent rewriting is why the same 5,512 forecasts scored 61.76% on
    07-18 and 33.74% on 07-19. Metrics stay derived and are recomputed every
    run, so a *scoring* fix still lands without --reresolve.
    """
    if not outcomes:
        return 0
    from database import ForecastOutcome

    all_fids = [o["forecast_id"] for o in outcomes]
    existing_ids = set()
    for i in range(0, len(all_fids), 900):
        batch = all_fids[i:i + 900]
        rows = db.query(ForecastOutcome.forecast_id).filter(
            ForecastOutcome.forecast_id.in_(batch)
        ).all()
        existing_ids.update(r[0] for r in rows)

    if reresolve:
        stale = list(existing_ids)
        for i in range(0, len(stale), 900):
            batch = stale[i:i + 900]
            db.query(ForecastOutcome).filter(
                ForecastOutcome.forecast_id.in_(batch)
            ).delete(synchronize_session=False)
        to_write = outcomes
    else:
        to_write = [o for o in outcomes if o["forecast_id"] not in existing_ids]

    if not to_write:
        db.commit()
        logger.info(f"  All {len(outcomes):,} outcomes already resolved (frozen)")
        return 0

    resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
    for o in to_write:
        o["evaluated_at"] = resolved_at
        o["resolved_at"] = resolved_at

    db.bulk_insert_mappings(ForecastOutcome, to_write)
    db.commit()

    from db.parquet import append_table
    append_table("forecast_outcomes", to_write, ["forecast_id"])

    logger.info(
        f"  Resolved {len(to_write):,} new outcomes "
        f"({len(outcomes) - len(to_write):,} already frozen)"
    )
    return len(to_write)


def _records_from_frozen_outcomes(db, min_price=0):
    """Rebuild scoring records from stored outcomes, without the archive.

    `confidence` is not on ForecastOutcome, so it is joined back from
    item_forecasts.
    """
    rows = db.execute(text("""
        SELECT o.forecast_id, o.item_id, o.horizon_days, o.model_version,
               o.base_price, o.actual_price, o.predicted_price_low,
               o.predicted_price_mid, o.predicted_price_high,
               o.direction_predicted, o.direction_actual, o.direction_correct,
               o.in_interval, f.confidence
        FROM forecast_outcomes o
        LEFT JOIN item_forecasts f ON f.id = o.forecast_id
        WHERE o.base_price IS NOT NULL AND o.base_price > 0
    """)).fetchall()

    groups = defaultdict(list)
    for r in rows:
        if min_price > 0 and r.base_price < min_price:
            continue
        abs_error = abs(r.predicted_price_mid - r.actual_price)
        groups[(r.horizon_days, r.model_version or "unknown")].append({
            "abs_error": abs_error,
            "pct_error": abs(abs_error / r.base_price) * 100,
            "sq_error": (r.predicted_price_mid - r.actual_price) ** 2,
            "direction_correct": r.direction_correct,
            "predicted_direction": r.direction_predicted,
            "actual_direction": r.direction_actual,
            "in_interval": r.in_interval,
            "confidence": r.confidence or "low",
            "base_price": r.base_price,
            "actual_price": r.actual_price,
            "price_tier": price_tier(r.base_price),
            "item_id": r.item_id,
        })
    return groups


def backtest_forecasts(db, today=None, min_price=0, reresolve=False, rescore=False):
    """Compare mature ML forecasts against actual prices.

    Stores both aggregate accuracy metrics (prediction_accuracy) and
    per-forecast outcome records (forecast_outcomes).

    Metrics breakdown:
      - Point error:      MAE, RMSE, MAPE, wMAPE (dollar-weighted), MAPE by price tier
      - Direction:        directional_accuracy, baseline comp, bootstrap CI
      - Probabilistic:    interval_coverage, conf_gap_pp, conf_calibration_error
      - Skill:            skill_vs_baseline (Theil's U analog, <1 beats persistence)

    Parameters:
        min_price: Minimum current_price to include (filter out cheap items).
        reresolve: Overwrite already-frozen actuals with freshly resolved ones.
        rescore: Recompute metrics from stored (frozen) outcomes only — does
            not touch the archive at all.
    """
    today = today or date.today()
    logger.info("=" * 60)
    logger.info("BACKTEST: ML Forecasts")
    logger.info("=" * 60)

    if rescore:
        logger.info("  --rescore: scoring from frozen outcomes, archive not read")
        groups = _records_from_frozen_outcomes(db, min_price=min_price)
        results = []
        for (horizon, model_version), records in sorted(groups.items()):
            for tier, metrics, n in score_by_tier(records):
                results.append({
                    "prediction_type": "forecast",
                    "evaluation_date": today,
                    "horizon_days": horizon,
                    "model_version": model_version,
                    "price_tier": tier,
                    "evaluation_window_days": None,
                    "sample_count": n,
                    "metrics": metrics,
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                })
        if results:
            _upsert_accuracy(db, results)
            logger.info(f"  Stored {len(results)} forecast accuracy records")
        return results

    # Fetch all forecasts with a midpoint price, filter for maturity in Python
    rows = db.execute(text("""
        SELECT f.id, f.item_id, f.forecast_date, f.horizon_days,
               f.price_low, f.price_mid, f.price_high,
               f.current_price, f.direction, f.confidence, f.model_version
        FROM item_forecasts f
        WHERE f.price_mid IS NOT NULL
    """)).fetchall()

    # Filter for mature forecasts (forecast_date + horizon <= today)
    mature = []
    for r in rows:
        forecast_date = r.forecast_date if isinstance(r.forecast_date, date) else date.fromisoformat(r.forecast_date)
        maturity_date = forecast_date + timedelta(days=r.horizon_days)
        if maturity_date <= today:
            mature.append(r)

    if not mature:
        logger.info("  No mature forecasts to evaluate.")
        return []

    logger.info(f"  Found {len(mature)} mature forecasts to evaluate")

    # Group by horizon + model_version
    groups = defaultdict(list)
    for r in mature:
        key = (r.horizon_days, r.model_version or "unknown")
        groups[key].append(r)

    archive_dir = Path(__file__).parent.parent.parent / "price-archive"

    slug_rows = db.execute(text("SELECT id, item_id FROM items")).fetchall()
    id_to_slug = {r.id: r.item_id for r in slug_rows}

    results = []
    all_outcomes = []
    n_unresolvable = 0
    n_considered = 0

    for (horizon, model_version), forecasts in sorted(groups.items()):
        anchors = set()
        slugs = set()
        for f in forecasts:
            slug = id_to_slug.get(f.item_id)
            if slug is None:
                continue
            f_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            slugs.add(slug)
            anchors.add((slug, f_date))
            anchors.add((slug, f_date + timedelta(days=horizon)))

        if not anchors:
            logger.info(f"  [{horizon}d / {model_version}] No slug mappings")
            n_considered += len(forecasts)
            n_unresolvable += len(forecasts)
            continue

        anchor_dates = [a[1] for a in anchors]
        voted = load_voted_prices(archive_dir, sorted(slugs), min(anchor_dates), max(anchor_dates))
        prices = smoothed_prices(voted, anchors)
        logger.info(f"  Resolved {len(prices):,} of {len(anchors):,} anchors")

        # Per-forecast records for aggregation and bootstrap
        records = []
        for f in forecasts:
            n_considered += 1
            slug = id_to_slug.get(f.item_id)
            f_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            target_date = f_date + timedelta(days=horizon)

            base = prices.get((slug, f_date))
            actual = prices.get((slug, target_date))
            if base is None or actual is None or base <= 0 or actual <= 0:
                n_unresolvable += 1
                continue

            mid, low, high = f.price_mid, f.price_low, f.price_high
            if min_price > 0 and base < min_price:
                continue

            abs_error = abs(mid - actual)
            actual_ret = (actual - base) / base
            predicted_direction = f.direction or "flat"
            actual_direction = direction_from_return(actual_ret)
            direction_correct = 1 if predicted_direction == actual_direction else 0
            in_interval = None if (low is None or high is None) else (1 if low <= actual <= high else 0)
            pct_error = abs(abs_error / base) * 100

            records.append({
                "abs_error": abs_error,
                "pct_error": pct_error,
                "sq_error": (mid - actual) ** 2,
                "direction_correct": direction_correct,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "in_interval": in_interval,
                "confidence": f.confidence or "low",
                "base_price": base,
                "actual_price": actual,
                "price_tier": price_tier(base),
                "item_id": f.item_id,
            })

            all_outcomes.append({
                "forecast_id": f.id,
                "item_id": f.item_id,
                "forecast_date": f_date,
                "horizon_days": horizon,
                "target_date": target_date,
                # current_price is retained for reference only; nothing reads
                # it for scoring, and it is never synthesized. Written through
                # as-is (nullable) so it stays distinguishable from base_price,
                # which is always archive-resolved. Downstream consumers
                # (update_bias_corrections_from_outcomes, retro_bias_check,
                # tiered_breakdown) still read this column and must see the
                # real serving-time value or a genuine NULL, not a stand-in.
                "current_price": f.current_price,
                "base_price": base,
                "predicted_price_low": low,
                "predicted_price_mid": mid,
                "predicted_price_high": high,
                "actual_price": actual,
                "direction_predicted": predicted_direction,
                "direction_actual": actual_direction,
                "direction_correct": direction_correct,
                "in_interval": in_interval,
                "abs_error": round(abs_error, 4),
                "pct_error": pct_error,
                "model_version": model_version,
            })

        if not records:
            logger.info(f"  [{horizon}d / {model_version}] No valid comparisons")
            continue

        tiered = score_by_tier(records)
        if not tiered:
            logger.info(f"  [{horizon}d / {model_version}] No valid comparisons")
            continue

        for tier, metrics, n in tiered:
            results.append({
                "prediction_type": "forecast",
                "evaluation_date": today,
                "horizon_days": horizon,
                "model_version": model_version,
                "price_tier": tier,
                "evaluation_window_days": None,
                "sample_count": n,
                "metrics": metrics,
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            })

        head = headline_records(records)
        penny = [r for r in records if r["price_tier"] < HEADLINE_MIN_TIER]
        head_metrics, head_n = score_cohort(head)
        penny_metrics, penny_n = score_cohort(penny)

        if head_n:
            head_ci_lower = head_metrics["directional_accuracy_ci_lower"]
            head_ci_upper = head_metrics["directional_accuracy_ci_upper"]
            ci_str = ""
            if head_ci_lower is not None:
                ci_str = f" [CI: {head_ci_lower * 100:.1f}–{head_ci_upper * 100:.1f}]"
            logger.info(
                f"  [{horizon}d / {model_version}] >=$1: {head_n:,} samples — "
                f"MAE=${head_metrics['mae']:.2f} MAPE={head_metrics['mape']:.1f}% "
                f"DirAcc={head_metrics['directional_accuracy']:.1f}%{ci_str} "
                f"IntCov={head_metrics['interval_coverage']:.1f}% "
                f"ConfGap={head_metrics['conf_gap_pp']:.1f}pp "
                f"Skill={head_metrics['skill_vs_baseline']}"
            )
        if penny_n:
            logger.info(
                f"  [{horizon}d / {model_version}] <$1: {penny_n:,} samples — "
                f"DirAcc={penny_metrics['directional_accuracy']:.1f}% (tick-dominated)"
            )

    if n_considered:
        unresolvable_pct = n_unresolvable / n_considered * 100
        logger.info(f"  Unresolvable: {n_unresolvable:,}/{n_considered:,} ({unresolvable_pct:.1f}%)")
        if unresolvable_pct > MAX_UNRESOLVABLE_PCT:
            raise RuntimeError(
                f"{unresolvable_pct:.1f}% of mature forecasts could not be resolved "
                f"(cap {MAX_UNRESOLVABLE_PCT}%). Silent cohort shrinkage is how this "
                f"metric moved unnoticed before — refusing to report a number."
            )

    if results:
        _upsert_accuracy(db, results)
        logger.info(f"  Stored {len(results)} forecast accuracy records")

    if all_outcomes:
        _store_forecast_outcomes(db, all_outcomes, reresolve=reresolve)

    return results



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(types=None, min_price=0, update_bias=False, reresolve=False, rescore=False):
    db = SessionLocal()
    today = date.today()
    allowed = ["forecast"]
    types = types or allowed

    try:
        results = {}
        for t in types:
            if t == "forecast":
                results["forecast"] = backtest_forecasts(
                    db, today, min_price=min_price, reresolve=reresolve, rescore=rescore
                )
            else:
                logger.warning(f"Unknown backtest type: {t}")

        total = sum(len(v or []) for v in results.values())
        logger.info(f"\nBacktest complete: {total} accuracy records stored")

        if update_bias:
            logger.info("=" * 60)
            logger.info("Updating per-tier bias corrections from outcomes...")
            logger.info("=" * 60)
            try:
                from models.forecaster import ItemForecaster
                forecaster = ItemForecaster(db_session=db)
                forecaster.load_models()
                forecaster.update_bias_corrections_from_outcomes()
                logger.info("Bias corrections updated successfully")
            except Exception as e:
                logger.error(f"Bias update failed: {e}", exc_info=True)

        return {"status": "success", "total_records": total, "types": list(results.keys())}

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "message": str(e)}

    finally:
        db.close()


def main():
    args = list(sys.argv[1:])
    types = None
    min_price = 0.0
    update_bias = True
    reresolve = "--reresolve" in args
    rescore = "--rescore" in args
    i = 0
    while i < len(args):
        if args[i] == "--type" and i + 1 < len(args):
            types = [args[i + 1]]
            i += 2
        elif args[i] == "--min-price" and i + 1 < len(args):
            min_price = float(args[i + 1])
            i += 2
        elif args[i] == "--update-bias":
            update_bias = True
            i += 1
        else:
            i += 1
    if min_price:
        logger.info(f"Filtering items with current_price >= ${min_price:.2f}")
    if update_bias:
        logger.info("Bias correction update enabled")
    result = run_backtest(
        types, min_price=min_price, update_bias=update_bias,
        reresolve=reresolve, rescore=rescore,
    )
    print(f"RESULT: {json.dumps(result, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
