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

import numpy as np
import pandas as pd
from database import SessionLocal, PredictionAccuracy
from sqlalchemy import text
from models.forecaster import ItemForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backtest_accuracy")

FLAT_TOLERANCE = 0.005
PRICE_TIER_BOUNDS = [1.0, 5.0, 20.0, 100.0]
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95
BOOTSTRAP_RNG_SEED = 42


def _direction_from_return(ret: float) -> str:
    if ret > FLAT_TOLERANCE:
        return "up"
    if ret < -FLAT_TOLERANCE:
        return "down"
    return "flat"


def _price_tier(price: float) -> int:
    if price >= 100:
        return 4
    if price >= 20:
        return 3
    if price >= 5:
        return 2
    if price >= 1:
        return 1
    return 0


def _bootstrap_ci(values, n_resamples=N_BOOTSTRAP, ci=BOOTSTRAP_CI):
    if len(values) < 10:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    stats = np.empty(n_resamples)
    n = len(values)
    arr = np.array(values)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        stats[i] = np.mean(sample)
    alpha = (100 - ci) / 2
    lower = float(np.percentile(stats, alpha))
    upper = float(np.percentile(stats, 100 - alpha))
    return round(lower, 4), round(upper, 4)


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
    """Load actual prices from Parquet files (DuckDB).

    The DB price_history table only holds live aggregator data (recent dates).
    For full historical backtesting we read directly from the Parquet archive.

    Applies the same multi-source voting logic as the forecaster's training
    pipeline (median of sources within 2 std of consensus) so that backtest
    actuals consistently match the target prices the model was trained on.
    """
    if not item_ids or not dates:
        return {}

    logger.info("  Loading actual prices from Parquet archive...")
    archive_dir = Path(__file__).parent.parent.parent / "price-archive"
    if not archive_dir.exists():
        logger.warning("  No price-archive directory found")
        return {}

    import duckdb

    # Build item_id (int) -> slug (str) mapping from DB
    slug_rows = db.execute(
        text("SELECT id, item_id FROM items")
    ).fetchall()
    id_to_slug = {r.id: r.item_id for r in slug_rows}
    slug_to_id = {s: i for i, s in id_to_slug.items()}

    # Map the requested item_ids to slugs
    target_slugs = [id_to_slug.get(iid) for iid in item_ids if id_to_slug.get(iid)]
    if not target_slugs:
        logger.warning("  No slug mappings found for requested item IDs")
        return {}

    date_strs = sorted({d.isoformat() if isinstance(d, date) else str(d)[:10] for d in dates})

    con = duckdb.connect()
    try:
        pq_files = sorted([str(p) for p in archive_dir.glob("prices-*.parquet")])
        pq_queries = []
        for pqf in pq_files:
            cols = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()
            col_names = {r[0] for r in cols}
            if "source" in col_names:
                pq_queries.append(f"SELECT item_slug, CAST(day AS DATE) AS day, mean_price AS price, source, volume FROM read_parquet('{pqf}')")
            else:
                pq_queries.append(f"SELECT item_slug, CAST(day AS DATE) AS day, mean_price AS price, NULL::VARCHAR AS source, volume FROM read_parquet('{pqf}')")
        union_sql = " UNION ALL BY NAME ".join(pq_queries)

        # Load all data matching our slugs and dates
        slug_list = ", ".join(f"'{s.replace(chr(39), chr(39)+chr(39))}'" for s in target_slugs)
        date_list = ", ".join(f"'{d}'" for d in date_strs)
        rows = con.sql(f"""
            SELECT item_slug, day, price, source, volume
            FROM ({union_sql})
            WHERE item_slug IN ({slug_list})
              AND CAST(day AS DATE) IN ({date_list})
        """).fetchall()

        if not rows:
            logger.warning("  No matching Parquet rows found")
            return {}

        df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "source", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])

        n_before = len(df)
        df = ItemForecaster._apply_multi_source_voting(df)
        logger.info(f"  Loaded {len(df)} actual price points from Parquet "
                    f"(voted from {n_before} source-rows)")

        by_key = {}
        for _, row in df.iterrows():
            slug = row["item_id"]
            item_id = slug_to_id.get(slug)
            if item_id is not None and pd.notna(row["price"]):
                by_key[(item_id, row["date"])] = float(row["price"])

        return by_key
    finally:
        con.close()


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

    all_fids = [o["forecast_id"] for o in outcomes]
    existing_ids = set()
    for i in range(0, len(all_fids), 900):
        batch = all_fids[i:i+900]
        rows = db.query(ForecastOutcome.forecast_id).filter(
            ForecastOutcome.forecast_id.in_(batch)
        ).all()
        existing_ids.update(r[0] for r in rows)

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

    Metrics breakdown:
      - Point error:      MAE, RMSE, MAPE, wMAPE (dollar-weighted), MAPE by price tier
      - Direction:        directional_accuracy, baseline comp, bootstrap CI
      - Probabilistic:    interval_coverage, conf_gap_pp, conf_calibration_error
      - Skill:            skill_vs_baseline (Theil's U analog, <1 beats persistence)
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

        # Per-forecast records for aggregation and bootstrap
        records = []
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

            pred_ret = (mid - current) / current
            actual_ret = (actual - current) / current
            predicted_direction = _direction_from_return(pred_ret)
            actual_direction = _direction_from_return(actual_ret)
            direction_correct = 1 if predicted_direction == actual_direction else 0

            in_interval = None
            if low is not None and high is not None:
                in_interval = 1 if low <= actual <= high else 0

            conf = f.confidence or "low"
            tier = _price_tier(current)

            records.append({
                "abs_error": abs_error,
                "pct_error": abs(abs_error / current) * 100 if current > 0 else 0,
                "sq_error": (mid - actual) ** 2,
                "direction_correct": direction_correct,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "in_interval": in_interval,
                "confidence": conf,
                "current_price": current,
                "actual_price": actual,
                "price_tier": tier,
                "item_id": f.item_id,
            })

            # Record per-forecast outcome for DB storage
            fcast_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            all_outcomes.append({
                "forecast_id": f.id,
                "item_id": f.item_id,
                "forecast_date": fcast_date,
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
                "pct_error": abs(abs_error / current) * 100 if current > 0 else 0,
                "model_version": model_version,
            })

        if not records:
            logger.info(f"  [{horizon}d / {model_version}] No valid comparisons")
            continue

        n = len(records)
        # ------------------------------------------------------------------
        # 1. Standard point-error metrics
        # ------------------------------------------------------------------
        mae = sum(r["abs_error"] for r in records) / n
        rmse = math.sqrt(sum(r["sq_error"] for r in records) / n)
        mape = sum(r["pct_error"] for r in records) / n

        # ------------------------------------------------------------------
        # 2. Directional accuracy
        # ------------------------------------------------------------------
        directional_hits = sum(r["direction_correct"] for r in records)
        directional_accuracy = directional_hits / n * 100

        # ------------------------------------------------------------------
        # 3. Interval coverage
        # ------------------------------------------------------------------
        interval_records = [r for r in records if r["in_interval"] is not None]
        interval_total = len(interval_records)
        interval_hits = sum(r["in_interval"] for r in interval_records)
        interval_coverage = (interval_hits / interval_total * 100) if interval_total > 0 else 0

        # ------------------------------------------------------------------
        # 4. wMAPE (dollar-weighted MAPE)
        # ------------------------------------------------------------------
        total_actual = sum(r["actual_price"] for r in records)
        wmape = (sum(r["abs_error"] for r in records) / total_actual * 100) if total_actual > 0 else 0

        # ------------------------------------------------------------------
        # 5. MAPE by price tier
        # ------------------------------------------------------------------
        tier_errors = defaultdict(list)
        for r in records:
            tier_errors[r["price_tier"]].append(r["pct_error"])
        mape_by_tier = {}
        for tier_num, errors_list in sorted(tier_errors.items()):
            mape_by_tier[f"tier_{tier_num}"] = round(sum(errors_list) / len(errors_list), 2)

        # ------------------------------------------------------------------
        # 6. Naive baseline: persistence forecast
        #    Predicts current_price as the future price (zero change).
        #    Direction is always "flat" — so baseline directional accuracy
        #    is the proportion of items whose actual return is within ±FLAT_TOLERANCE.
        # ------------------------------------------------------------------
        baseline_hits = sum(1 for r in records if r["actual_direction"] == "flat")
        baseline_directional_accuracy = baseline_hits / n * 100

        baseline_mae = sum(abs(r["current_price"] - r["actual_price"]) for r in records) / n

        improvement_over_baseline_pp = round(directional_accuracy - baseline_directional_accuracy, 2)
        skill_vs_baseline = round(mae / baseline_mae, 4) if baseline_mae > 0 else None

        # ------------------------------------------------------------------
        # 7. Confidence calibration metrics
        # ------------------------------------------------------------------
        high_conf = [r for r in records if r["confidence"] == "high"]
        low_conf = [r for r in records if r["confidence"] == "low"]
        med_conf = [r for r in records if r["confidence"] == "medium"]

        high_dir_acc = sum(r["direction_correct"] for r in high_conf) / len(high_conf) * 100 if high_conf else 0
        low_dir_acc = sum(r["direction_correct"] for r in low_conf) / len(low_conf) * 100 if low_conf else 0

        conf_gap_pp = round(high_dir_acc - low_dir_acc, 2)

        high_interval_records = [r for r in high_conf if r["in_interval"] is not None]
        high_int_hits = sum(r["in_interval"] for r in high_interval_records)
        high_int_total = len(high_interval_records)
        conf_high_interval_cov = round(high_int_hits / high_int_total * 100, 2) if high_int_total > 0 else 0

        # Calibration error: |high_conf_dir_acc - target_accuracy (80%)|
        # The target_accuracy is the binary confidence target from forecaster._calibrate_confidence
        target_accuracy = 80.0
        conf_calibration_error = round(abs(high_dir_acc - target_accuracy), 2)

        # ------------------------------------------------------------------
        # 8. Bootstrap confidence intervals
        # ------------------------------------------------------------------
        dir_acc_vals = [r["direction_correct"] for r in records]
        mae_vals = [r["abs_error"] for r in records]

        dir_ci_lower, dir_ci_upper = _bootstrap_ci(dir_acc_vals)
        mae_ci_lower, mae_ci_upper = _bootstrap_ci(mae_vals)

        # ------------------------------------------------------------------
        # Assemble metrics dict
        # ------------------------------------------------------------------
        metrics = {
            # point error
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "wmape": round(wmape, 2),
            "mape_by_tier": mape_by_tier,
            # directional
            "directional_accuracy": round(directional_accuracy, 2),
            "interval_coverage": round(interval_coverage, 2),
            # baseline comparison
            "baseline_directional_accuracy": round(baseline_directional_accuracy, 2),
            "improvement_over_baseline_pp": improvement_over_baseline_pp,
            "baseline_mae": round(baseline_mae, 4),
            "skill_vs_baseline": skill_vs_baseline,
            # confidence calibration
            "conf_gap_pp": conf_gap_pp,
            "conf_high_interval_cov": conf_high_interval_cov,
            "conf_calibration_error": conf_calibration_error,
            # bootstrap CIs
            "directional_accuracy_ci_lower": dir_ci_lower,
            "directional_accuracy_ci_upper": dir_ci_upper,
            "mae_ci_lower": mae_ci_lower,
            "mae_ci_upper": mae_ci_upper,
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

        ci_str = ""
        if dir_ci_lower is not None:
            ci_str = f" [CI: {dir_ci_lower:.1f}–{dir_ci_upper:.1f}]"
        logger.info(
            f"  [{horizon}d / {model_version}] {n} samples — "
            f"MAE=${metrics['mae']:.2f} MAPE={metrics['mape']:.1f}% "
            f"wMAPE={metrics['wmape']:.1f}% "
            f"DirAcc={metrics['directional_accuracy']:.1f}%{ci_str} "
            f"IntCov={metrics['interval_coverage']:.1f}% "
            f"Gap={metrics['conf_gap_pp']:.1f}pp "
            f"Skill={metrics['skill_vs_baseline']}"
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
