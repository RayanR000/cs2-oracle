#!/usr/bin/env python3
"""
Backtesting system for all prediction and analysis types.
Computes accuracy metrics by comparing past predictions against actual outcomes.

Analysis types:
  - forecast:         ML price forecasts (7d / 30d horizons)
  - trend_direction:  Trend direction signals from daily_analysis
  - opportunity:      Opportunity signals from daily_analysis

Usage:
    python scripts/backtest_accuracy.py
    python scripts/backtest_accuracy.py --type forecast
    python scripts/backtest_accuracy.py --type trend_direction
    python scripts/backtest_accuracy.py --type opportunity
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


def _load_actual_prices_chartpoints(db, item_ids, dates):
    """Load actual close prices from chart_points for (item_id, date) pairs."""
    if not item_ids or not dates:
        return {}

    min_date = min(dates)
    max_date = max(dates)

    # Use SQLAlchemy ORM to avoid dialect issues with ANY()
    from database import ChartPoint
    rows = db.query(ChartPoint).filter(
        ChartPoint.item_id.in_(item_ids),
        ChartPoint.day >= min_date,
        ChartPoint.day <= max_date,
    ).all()

    return {(r.item_id, r.day): r.close for r in rows}


def _load_actual_prices_history(db, item_ids, dates):
    """Fallback: average price from price_history for missing dates."""
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
        ~PriceHistory.source.like('synthetic_demo'),
        ~PriceHistory.source.like('historical_fallback:%'),
    ).order_by(PriceHistory.item_id, PriceHistory.timestamp).all()

    by_key = defaultdict(list)
    for r in rows:
        day = r.timestamp.date()
        by_key[(r.item_id, day)].append(r.price)

    return {k: sum(v) / len(v) for k, v in by_key.items()}


def _load_actual_prices(db, item_ids, dates):
    """Load actual prices from chart_points, falling back to price_history."""
    prices = _load_actual_prices_chartpoints(db, item_ids, dates)
    if len(prices) < len(dates):
        fallback = _load_actual_prices_history(db, item_ids, dates)
        prices.update(fallback)
    return prices


# ---------------------------------------------------------------------------
# 1. Forecast backtesting
# ---------------------------------------------------------------------------

def backtest_forecasts(db, today=None):
    """Compare mature ML forecasts against actual prices."""
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
        maturity_date = r.forecast_date + timedelta(days=r.horizon_days)
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
    for (horizon, model_version), forecasts in sorted(groups.items()):
        target_dates = set()
        item_ids = set()
        for f in forecasts:
            target_dates.add(f.forecast_date + timedelta(days=horizon))
            item_ids.add(f.item_id)

        actual_prices = _load_actual_prices(db, item_ids, target_dates)

        errors = []
        directional_hits = 0
        directional_total = 0
        interval_hits = 0
        interval_total = 0
        confidence_buckets = defaultdict(lambda: {"hits": 0, "total": 0})

        for f in forecasts:
            target_date = f.forecast_date + timedelta(days=horizon)
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
            if predicted_direction == actual_direction:
                directional_hits += 1
            directional_total += 1

            if low is not None and high is not None:
                if low <= actual <= high:
                    interval_hits += 1
                interval_total += 1

            conf = f.confidence or "low"
            if predicted_direction == actual_direction:
                confidence_buckets[conf]["hits"] += 1
            confidence_buckets[conf]["total"] += 1

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
            "confidence_accuracy_medium": round(
                confidence_buckets["medium"]["hits"] / confidence_buckets["medium"]["total"] * 100, 2
            ) if confidence_buckets["medium"]["total"] > 0 else 0,
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
    return results


# ---------------------------------------------------------------------------
# 2. Trend direction backtesting
# ---------------------------------------------------------------------------

def backtest_trends(db, today=None):
    today = today or date.today()
    logger.info("=" * 60)
    logger.info("BACKTEST: Trend Directions")
    logger.info("=" * 60)

    eval_windows = [7, 14, 30]
    all_results = []

    for window_days in eval_windows:
        min_analysis_date = today - timedelta(days=window_days + 7)
        max_analysis_date = today - timedelta(days=window_days)

        rows = db.execute(text("""
            SELECT da.item_id, da.analysis_date, da.trend_direction,
                   da.current_price, da.momentum_7day, da.momentum_30day
            FROM daily_analysis da
            WHERE da.analysis_date BETWEEN :min_date AND :max_date
              AND da.trend_direction IS NOT NULL
              AND da.current_price IS NOT NULL
              AND da.current_price > 0
        """), {
            "min_date": min_analysis_date,
            "max_date": max_analysis_date,
        }).fetchall()

        if not rows:
            logger.info(f"  [{window_days}d window] No trend records to evaluate.")
            continue

        item_ids = {r.item_id for r in rows}
        target_dates = {r.analysis_date + timedelta(days=window_days) for r in rows}

        actual_prices = _load_actual_prices(db, item_ids, target_dates)

        confusion = {"up_up": 0, "up_down": 0, "up_flat": 0,
                      "down_up": 0, "down_down": 0, "down_flat": 0,
                      "flat_up": 0, "flat_down": 0, "flat_flat": 0}
        total = 0
        correct = 0
        total_return = 0.0
        return_count = 0

        for r in rows:
            target_date = r.analysis_date + timedelta(days=window_days)
            actual = actual_prices.get((r.item_id, target_date))
            if actual is None or actual <= 0:
                continue

            current = r.current_price
            if current <= 0:
                continue

            actual_move = ((actual - current) / current) * 100
            actual_direction = "up" if actual_move > 2 else "down" if actual_move < -2 else "flat"
            predicted = r.trend_direction or "flat"
            if predicted == "bullish":
                predicted = "up"
            elif predicted == "bearish":
                predicted = "down"

            key = f"{predicted}_{actual_direction}"
            if key in confusion:
                confusion[key] = confusion.get(key, 0) + 1

            if predicted == actual_direction:
                correct += 1
            total += 1
            total_return += actual_move
            return_count += 1

        if total == 0:
            continue

        accuracy = correct / total * 100
        avg_return = total_return / return_count if return_count > 0 else 0

        metrics = {
            "overall_accuracy": round(accuracy, 2),
            "avg_subsequent_return_pct": round(avg_return, 2),
            "avg_subsequent_return_days": window_days,
            "confusion_matrix": confusion,
        }

        all_results.append({
            "prediction_type": "trend_direction",
            "evaluation_date": today,
            "horizon_days": None,
            "model_version": None,
            "evaluation_window_days": window_days,
            "sample_count": total,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        })

        logger.info(
            f"  [{window_days}d window] {total} samples — "
            f"Acc={accuracy:.1f}% AvgRet={avg_return:+.2f}%"
        )

    if all_results:
        _upsert_accuracy(db, all_results)
        logger.info(f"  Stored {len(all_results)} trend accuracy records")
    return all_results


# ---------------------------------------------------------------------------
# 3. Opportunity backtesting
# ---------------------------------------------------------------------------

def backtest_opportunities(db, today=None):
    today = today or date.today()
    logger.info("=" * 60)
    logger.info("BACKTEST: Opportunity Signals")
    logger.info("=" * 60)

    eval_windows = [7, 14, 30]
    all_results = []

    for window_days in eval_windows:
        min_analysis_date = today - timedelta(days=window_days + 7)
        max_analysis_date = today - timedelta(days=window_days)

        rows = db.execute(text("""
            SELECT da.item_id, da.analysis_date, da.opportunity_score,
                   da.momentum_score, da.current_price,
                   da.trend_direction, da.price_stability
            FROM daily_analysis da
            WHERE da.analysis_date BETWEEN :min_date AND :max_date
              AND da.opportunity_score IS NOT NULL
              AND da.current_price IS NOT NULL
              AND da.current_price > 0
        """), {
            "min_date": min_analysis_date,
            "max_date": max_analysis_date,
        }).fetchall()

        if not rows:
            logger.info(f"  [{window_days}d window] No opportunity records.")
            continue

        item_ids = {r.item_id for r in rows}
        target_dates = {r.analysis_date + timedelta(days=window_days) for r in rows}
        actual_prices = _load_actual_prices(db, item_ids, target_dates)

        undervalued_hits = 0
        undervalued_total = 0
        overheated_hits = 0
        overheated_total = 0
        momentum_hits = 0
        momentum_total = 0
        total_return = 0.0
        return_count = 0

        for r in rows:
            target_date = r.analysis_date + timedelta(days=window_days)
            actual = actual_prices.get((r.item_id, target_date))
            if actual is None or actual <= 0:
                continue

            current = r.current_price
            if current <= 0:
                continue

            actual_return = ((actual - current) / current) * 100
            total_return += actual_return
            return_count += 1

            opportunity = r.opportunity_score or 0
            momentum = r.momentum_score or 0

            if opportunity <= -30:
                undervalued_total += 1
                if actual_return > 2:
                    undervalued_hits += 1
            if opportunity >= 30:
                overheated_total += 1
                if actual_return < -2:
                    overheated_hits += 1
            if abs(momentum) >= 40:
                momentum_total += 1
                if (momentum > 0 and actual_return > 3) or (momentum < 0 and actual_return < -3):
                    momentum_hits += 1

        if return_count == 0:
            continue

        avg_return = total_return / return_count

        metrics = {
            "avg_return_pct": round(avg_return, 2),
            "evaluation_window_days": window_days,
            "total_signals": undervalued_total + overheated_total + momentum_total,
            "undervalued": {
                "total": undervalued_total,
                "correct": undervalued_hits,
                "precision": round(undervalued_hits / undervalued_total * 100, 2) if undervalued_total > 0 else 0,
            },
            "overheated": {
                "total": overheated_total,
                "correct": overheated_hits,
                "precision": round(overheated_hits / overheated_total * 100, 2) if overheated_total > 0 else 0,
            },
            "momentum": {
                "total": momentum_total,
                "correct": momentum_hits,
                "precision": round(momentum_hits / momentum_total * 100, 2) if momentum_total > 0 else 0,
            },
        }

        all_results.append({
            "prediction_type": "opportunity",
            "evaluation_date": today,
            "horizon_days": None,
            "model_version": None,
            "evaluation_window_days": window_days,
            "sample_count": return_count,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        })

        logger.info(
            f"  [{window_days}d window] {return_count} samples — "
            f"AvgRet={avg_return:+.2f}% "
            f"UnderP={metrics['undervalued']['precision']:.1f}% "
            f"OverP={metrics['overheated']['precision']:.1f}%"
        )

    if all_results:
        _upsert_accuracy(db, all_results)
        logger.info(f"  Stored {len(all_results)} opportunity accuracy records")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(types=None):
    db = SessionLocal()
    today = date.today()
    all_types = ["forecast", "trend_direction", "opportunity"]
    types = types or all_types

    try:
        results = {}
        for t in types:
            if t == "forecast":
                results["forecast"] = backtest_forecasts(db, today)
            elif t == "trend_direction":
                results["trend_direction"] = backtest_trends(db, today)
            elif t == "opportunity":
                results["opportunity"] = backtest_opportunities(db, today)
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
            logger.error("--type requires: forecast, trend_direction, or opportunity")
            sys.exit(1)
    result = run_backtest(types)
    print(f"RESULT: {json.dumps(result, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
