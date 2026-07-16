#!/usr/bin/env python3
"""
Backtesting system for ML forecast accuracy.
Computes accuracy metrics by comparing past predictions against actual outcomes.

Analysis type:
  - forecast:  ML price forecasts (7d / 30d horizons) from live DB

Usage:
    python scripts/backtest_accuracy.py
    python scripts/backtest_accuracy.py --type forecast
"""

import sys
import json
import math
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, PredictionAccuracy
from sqlalchemy import text

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

        existing = db.query(PredictionAccuracy).filter_by(**filters).first()
        if existing:
            existing.sample_count = row["sample_count"]
            existing.metrics = row["metrics"]
            existing.evaluation_window_days = row.get("evaluation_window_days")
            existing.created_at = row["created_at"]
        else:
            db.add(PredictionAccuracy(**row))
    db.commit()


def _load_actual_prices(db, item_ids, dates):
    """Load actual prices from price_history, averaged to daily closes."""
    if not item_ids or not dates:
        return {}

    min_date = min(dates)
    max_date = max(dates)

    from database import PriceHistory
    from datetime import datetime as dt

    start_dt = dt.combine(min_date, dt.min.time())
    end_dt = dt.combine(max_date, dt.max.time())

    rows = db.query(
        PriceHistory.item_id,
        PriceHistory.timestamp,
        PriceHistory.price,
    ).filter(
        PriceHistory.item_id.in_(list(item_ids)),
        PriceHistory.timestamp >= start_dt,
        PriceHistory.timestamp <= end_dt,
    ).order_by(PriceHistory.item_id, PriceHistory.timestamp).all()

    by_key = defaultdict(list)
    for r in rows:
        day = r.timestamp.date()
        by_key[(r.item_id, day)].append(r.price)

    return {k: sum(v) / len(v) for k, v in by_key.items()}


# ---------------------------------------------------------------------------
# 1. Forecast backtesting
# ---------------------------------------------------------------------------

def _store_forecast_outcomes(db, outcomes):
    """Bulk insert per-forecast outcome records.

    Replaces any existing outcome for the same forecast_id (so re-running
    backtest updates rather than duplicates).
    """
    if not outcomes:
        return
    from database import ForecastOutcome

    existing_ids = {
        r[0] for r in db.query(ForecastOutcome.forecast_id)
        .filter(ForecastOutcome.forecast_id.in_([o["forecast_id"] for o in outcomes]))
        .all()
    }

    to_insert = []
    to_update = []
    for o in outcomes:
        o["evaluated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
        if o["forecast_id"] in existing_ids:
            to_update.append(o)
        else:
            to_insert.append(ForecastOutcome(**o))

    # Update existing records
    if to_update:
        for o in to_update:
            db.query(ForecastOutcome).filter(
                ForecastOutcome.forecast_id == o["forecast_id"]
            ).update(o)

    # Insert new records
    if to_insert:
        db.add_all(to_insert)

    db.commit()
    logger.info(f"  Stored {len(outcomes)} forecast outcome records ({len(to_insert)} new, {len(to_update)} updated)")


def backtest_forecasts(db, today=None):
    """Compare mature ML forecasts against actual prices.

    Stores both aggregate accuracy metrics (prediction_accuracy) and
    per-forecast outcome records (forecast_outcomes).
    """
    today = today or date.today()
    logger.info("=" * 60)
    logger.info("BACKTEST: ML Forecasts")
    logger.info("=" * 60)

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

    results = []
    all_outcomes = []
    for (horizon, model_version), forecasts in sorted(groups.items()):
        target_dates = set()
        item_ids = set()
        for f in forecasts:
            f_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(f.forecast_date)
            target_dates.add(f_date + timedelta(days=horizon))
            item_ids.add(f.item_id)

        actual_prices = _load_actual_prices(db, item_ids, target_dates)

        errors = []
        directional_hits = 0
        directional_total = 0
        interval_hits = 0
        interval_total = 0
        confidence_buckets = defaultdict(lambda: {"hits": 0, "total": 0})

        for f in forecasts:
            f_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(f.forecast_date)
            target_date = f_date + timedelta(days=horizon)
            actual = actual_prices.get((f.item_id, target_date))
            if actual is None or actual <= 0:
                continue

            mid = f.price_mid
            low = f.price_low
            high = f.price_high
            current = f.current_price

            if mid is None or current is None or current <= 0:
                continue

            abs_error = abs(mid - actual)
            pct_error = abs(abs_error / actual) * 100 if actual > 0 else 0
            errors.append({
                "abs_error": abs_error,
                "pct_error": pct_error,
                "sq_error": (mid - actual) ** 2,
            })

            actual_direction = "up" if actual > current else "down" if actual < current else "flat"
            predicted_direction = f.direction or "flat"
            direction_correct = 1 if predicted_direction == actual_direction else 0
            if direction_correct:
                directional_hits += 1
            directional_total += 1

            in_interval = None
            if low is not None and high is not None:
                in_interval = 1 if low <= actual <= high else 0
                if in_interval:
                    interval_hits += 1
                interval_total += 1

            conf = f.confidence or "low"
            if direction_correct:
                confidence_buckets[conf]["hits"] += 1
            confidence_buckets[conf]["total"] += 1

            # Record per-forecast outcome
            all_outcomes.append({
                "forecast_id": f.id,
                "item_id": f.item_id,
                "forecast_date": f.forecast_date,
                "horizon_days": horizon,
                "target_date": target_date,
                "current_price": current,
                "predicted_price_low": low,
                "predicted_price_mid": mid,
                "predicted_price_high": high,
                "actual_price": actual,
                "direction_predicted": predicted_direction,
                "direction_actual": actual_direction,
                "direction_correct": direction_correct,
                "in_interval": in_interval,
                "abs_error": round(abs_error, 4),
                "pct_error": round(pct_error, 2),
                "model_version": model_version,
            })

        if not errors:
            logger.info(f"  [{horizon}d / {model_version}] No valid comparisons")
            continue

        n = len(errors)
        mae = sum(e["abs_error"] for e in errors) / n
        rmse = math.sqrt(sum(e["sq_error"] for e in errors) / n)
        mape = sum(e["pct_error"] for e in errors) / n
        directional_accuracy = (directional_hits / directional_total * 100) if directional_total > 0 else 0
        interval_coverage = (interval_hits / interval_total * 100) if interval_total > 0 else 0

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "directional_accuracy": round(directional_accuracy, 2),
            "interval_coverage": round(interval_coverage, 2),
            "confidence_accuracy_low": round(
                confidence_buckets["low"]["hits"] / confidence_buckets["low"]["total"] * 100, 2
            ) if confidence_buckets["low"]["total"] > 0 else 0,
            "confidence_accuracy_high": round(
                confidence_buckets["high"]["hits"] / confidence_buckets["high"]["total"] * 100, 2
            ) if confidence_buckets["high"]["total"] > 0 else 0,
        }

        results.append({
            "prediction_type": "forecast",
            "evaluation_date": today,
            "horizon_days": horizon,
            "model_version": model_version,
            "evaluation_window_days": None,
            "sample_count": n,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        })

        logger.info(
            f"  [{horizon}d / {model_version}] {n} samples — "
            f"MAE=${metrics['mae']:.2f} MAPE={metrics['mape']:.1f}% "
            f"DirAcc={metrics['directional_accuracy']:.1f}% "
            f"IntCov={metrics['interval_coverage']:.1f}%"
        )

    if results:
        _upsert_accuracy(db, results)
        logger.info(f"  Stored {len(results)} forecast accuracy records")

    if all_outcomes:
        _store_forecast_outcomes(db, all_outcomes)

    return results





# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(types=None):
    db = SessionLocal()
    today = date.today()
    allowed = ["forecast"]
    types = types or allowed

    try:
        results = {}
        for t in types:
            if t == "forecast":
                results["forecast"] = backtest_forecasts(db, today)
            else:
                logger.warning(f"Unknown backtest type: {t}")

        total = sum(len(v or []) for v in results.values())
        logger.info(f"\nBacktest complete: {total} accuracy records stored")
        return {"status": "success", "total_records": total, "types": list(results.keys())}

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "message": str(e)}

    finally:
        db.close()


def main():
    args = set(sys.argv[1:])
    types = None
    if "--type" in args:
        idx = sys.argv.index("--type") + 1
        if idx < len(sys.argv):
            types = [sys.argv[idx]]
        else:
            logger.error("--type requires: forecast")
            sys.exit(1)
    result = run_backtest(types)
    print(f"RESULT: {json.dumps(result, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
