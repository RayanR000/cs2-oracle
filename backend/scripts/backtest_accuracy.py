#!/usr/bin/env python3
"""
Backtesting system for all prediction and analysis types.
Computes accuracy metrics by comparing past predictions against actual outcomes.

Analysis types:
  - forecast:         ML price forecasts (7d / 30d horizons) from live DB
  - trend_direction:  Trend direction signals from daily_analysis
  - opportunity:      Opportunity signals from daily_analysis
  - historical:       Retroactive walk-forward evaluation of trend & opportunity
                      signals across 13 years of parquet archive data (2013-2026)

Usage:
    python scripts/backtest_accuracy.py
    python scripts/backtest_accuracy.py --type forecast
    python scripts/backtest_accuracy.py --type historical
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
# 4. Historical walk-forward backtesting (parquet archive)
# ---------------------------------------------------------------------------

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"


def _load_parquet_items(con):
    """Get item list with date range from parquet archive."""
    rows = con.sql("""
        SELECT item_slug,
               MIN(day) AS first_day,
               MAX(day) AS last_day,
               COUNT(*) AS row_count
        FROM read_parquet('{}/*.parquet')
        GROUP BY item_slug
        HAVING row_count >= 90
        ORDER BY row_count DESC
    """.format(ARCHIVE_DIR)).fetchall()
    return rows


def _compute_trend_at_date(prices, idx, ma_short=7, ma_long=30):
    """Compute trend direction at a historical point in the price series.

    Returns (direction: 'up'|'down'|'flat', ma_short_val, ma_long_val, current_price)
    """
    if idx < ma_long:
        return "flat", None, None, prices[idx][1]

    short_prices = [p[1] for p in prices[idx - ma_short + 1:idx + 1]]
    long_prices = [p[1] for p in prices[idx - ma_long + 1:idx + 1]]

    if len(short_prices) < ma_short or len(long_prices) < ma_long:
        return "flat", None, None, prices[idx][1]

    ma_s = sum(short_prices) / ma_short
    ma_l = sum(long_prices) / ma_long
    current = prices[idx][1]

    if ma_s > ma_l * 1.02:
        direction = "up"
    elif ma_s < ma_l * 0.98:
        direction = "down"
    else:
        direction = "flat"

    return direction, ma_s, ma_l, current


def _compute_opportunity_at_date(prices, idx, window=30):
    """Compute opportunity score at a historical point.

    Uses same logic as TrendAnalyzer.calculate_opportunity_score.
    """
    if idx < window:
        return 0, 0

    recent = [p[1] for p in prices[idx - window + 1:idx + 1]]
    if len(recent) < window:
        return 0, 0

    current = prices[idx][1]
    ma_30 = sum(recent) / window

    # Momentum: 7-day % change
    if idx >= 7:
        price_7d_ago = prices[idx - 7][1]
        momentum_7 = ((current - price_7d_ago) / price_7d_ago * 100) if price_7d_ago > 0 else 0
    else:
        momentum_7 = 0

    if idx >= 30:
        price_30d_ago = prices[idx - 30][1]
        momentum_30 = ((current - price_30d_ago) / price_30d_ago * 100) if price_30d_ago > 0 else 0
    else:
        momentum_30 = 0

    momentum_score = (momentum_7 + momentum_30) / 2
    momentum_score = max(-100, min(100, momentum_score))

    # Volatility
    if len(recent) >= 2:
        mean_p = sum(recent) / len(recent)
        if mean_p > 0:
            variance = sum((p - mean_p) ** 2 for p in recent) / len(recent)
            vol = (variance ** 0.5 / mean_p) * 100
        else:
            vol = 0
    else:
        vol = 0

    deviation = ((current - ma_30) / ma_30) * 100
    if deviation < -10:
        opportunity = 50 + (momentum_score * 0.5)
    elif deviation > 15:
        opportunity = -50 + (momentum_score * 0.3)
    else:
        opportunity = momentum_score * 0.7
    if vol > 10:
        opportunity *= 0.8
    opportunity = max(-100, min(100, opportunity))

    return opportunity, momentum_score


def backtest_historical(db, today=None):
    """Walk through the parquet archive simulating trend+opportunity signals."""
    today = today or date.today()
    logger.info("=" * 60)
    logger.info("HISTORICAL WALK-FORWARD BACKTEST (Parquet Archive)")
    logger.info("=" * 60)

    if not ARCHIVE_DIR.exists():
        logger.warning(f"  Parquet archive not found at {ARCHIVE_DIR}")
        return []

    import duckdb
    import numpy as np
    from datetime import datetime as dt

    con = duckdb.connect()
    try:
        items = _load_parquet_items(con)
        logger.info(f"  {len(items)} items with >= 90 days of history in archive")

        # Use up to 500 items with richest history for statistically meaningful results
        MAX_ITEMS = 500
        items = items[:MAX_ITEMS]
        logger.info(f"  Using top {len(items)} items by row count")

        eval_windows = [7, 14, 30]
        all_results = []

        for window_days in eval_windows:
            logger.info(f"  Evaluating {window_days}d forward window...")

            confusion = {"up_up": 0, "up_down": 0, "up_flat": 0,
                          "down_up": 0, "down_down": 0, "down_flat": 0,
                          "flat_up": 0, "flat_down": 0, "flat_flat": 0}
            trend_total = 0
            trend_correct = 0
            trend_return_total = 0.0
            trend_return_count = 0

            opp_undervalued_hits = 0
            opp_undervalued_total = 0
            opp_overheated_hits = 0
            opp_overheated_total = 0
            opp_momentum_hits = 0
            opp_momentum_total = 0
            opp_return_total = 0.0
            opp_return_count = 0

            for item_slug, first_day, last_day, row_count in items:
                # Load all prices for this item
                rows = con.sql("""
                    SELECT day, mean_price
                    FROM read_parquet('{}/*.parquet')
                    WHERE item_slug = ?
                    ORDER BY day
                """.format(ARCHIVE_DIR), params=[item_slug]).fetchall()

                if len(rows) < 90:
                    continue

                prices = [(r[0], float(r[1])) for r in rows if r[1] > 0]
                if len(prices) < 90:
                    continue

                # Walk through every 7th day starting from day 90
                step = 7
                n = len(prices)
                for idx in range(90, n - window_days, step):
                    signal_date = prices[idx][0]

                    # Trend direction
                    direction, ma_s, ma_l, current = _compute_trend_at_date(prices, idx)
                    if direction == "flat":
                        continue  # flat means no signal — skip for cleaner metrics

                    # Look forward
                    target_price = prices[idx + window_days][1]
                    if current <= 0 or target_price <= 0:
                        continue

                    actual_move = ((target_price - current) / current) * 100
                    actual_dir = "up" if actual_move > 2 else "down" if actual_move < -2 else "flat"

                    key = f"{direction}_{actual_dir}"
                    if key in confusion:
                        confusion[key] = confusion.get(key, 0) + 1

                    if direction == actual_dir:
                        trend_correct += 1
                    trend_total += 1
                    trend_return_total += actual_move
                    trend_return_count += 1

                    # Opportunity signals
                    opportunity, momentum = _compute_opportunity_at_date(prices, idx)

                    if opportunity <= -30:
                        opp_undervalued_total += 1
                        if actual_move > 2:
                            opp_undervalued_hits += 1
                    if opportunity >= 30:
                        opp_overheated_total += 1
                        if actual_move < -2:
                            opp_overheated_hits += 1
                    if abs(momentum) >= 40:
                        opp_momentum_total += 1
                        if (momentum > 0 and actual_move > 3) or (momentum < 0 and actual_move < -3):
                            opp_momentum_hits += 1

                    opp_return_total += actual_move
                    opp_return_count += 1

            # Store trend direction results
            if trend_total > 0:
                accuracy = trend_correct / trend_total * 100
                avg_ret = trend_return_total / trend_return_count if trend_return_count > 0 else 0
                metrics = {
                    "overall_accuracy": round(accuracy, 2),
                    "avg_subsequent_return_pct": round(avg_ret, 2),
                    "avg_subsequent_return_days": window_days,
                    "confusion_matrix": confusion,
                    "source": "historical_parquet",
                }
                all_results.append({
                    "prediction_type": "trend_direction",
                    "evaluation_date": today,
                    "horizon_days": None,
                    "model_version": "historical_walkforward",
                    "evaluation_window_days": window_days,
                    "sample_count": trend_total,
                    "metrics": metrics,
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                })
                logger.info(
                    f"    Trend [{window_days}d]: {trend_total:,} samples — "
                    f"Acc={accuracy:.1f}% AvgRet={avg_ret:+.2f}%"
                )

            # Store opportunity results
            if opp_return_count > 0:
                avg_ret = opp_return_total / opp_return_count
                metrics = {
                    "avg_return_pct": round(avg_ret, 2),
                    "evaluation_window_days": window_days,
                    "total_signals": opp_undervalued_total + opp_overheated_total + opp_momentum_total,
                    "source": "historical_parquet",
                    "undervalued": {
                        "total": opp_undervalued_total,
                        "correct": opp_undervalued_hits,
                        "precision": round(opp_undervalued_hits / opp_undervalued_total * 100, 2) if opp_undervalued_total > 0 else 0,
                    },
                    "overheated": {
                        "total": opp_overheated_total,
                        "correct": opp_overheated_hits,
                        "precision": round(opp_overheated_hits / opp_overheated_total * 100, 2) if opp_overheated_total > 0 else 0,
                    },
                    "momentum": {
                        "total": opp_momentum_total,
                        "correct": opp_momentum_hits,
                        "precision": round(opp_momentum_hits / opp_momentum_total * 100, 2) if opp_momentum_total > 0 else 0,
                    },
                }
                all_results.append({
                    "prediction_type": "opportunity",
                    "evaluation_date": today,
                    "horizon_days": None,
                    "model_version": "historical_walkforward",
                    "evaluation_window_days": window_days,
                    "sample_count": opp_return_count,
                    "metrics": metrics,
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                })
                logger.info(
                    f"    Opportunity [{window_days}d]: {opp_return_count:,} samples — "
                    f"AvgRet={avg_ret:+.2f}% "
                    f"UnderP={metrics['undervalued']['precision']:.1f}% "
                    f"OverP={metrics['overheated']['precision']:.1f}%"
                )

        if all_results:
            _upsert_accuracy(db, all_results)
            logger.info(f"  Stored {len(all_results)} historical accuracy records")
        return all_results

    finally:
        con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(types=None):
    db = SessionLocal()
    today = date.today()
    all_types = ["forecast", "trend_direction", "opportunity", "historical"]
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
            elif t == "historical":
                results["historical"] = backtest_historical(db, today)
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
            logger.error("--type requires: forecast, trend_direction, opportunity, or historical")
            sys.exit(1)
    result = run_backtest(types)
    print(f"RESULT: {json.dumps(result, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
