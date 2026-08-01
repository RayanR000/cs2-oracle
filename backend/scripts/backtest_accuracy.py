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
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from database import SessionLocal, PredictionAccuracy
from sqlalchemy import text
from models.forecaster import ItemForecaster
from backtest.scoring import (
    FLAT_TOLERANCE,
    bootstrap_ci,
    direction_from_return,
    price_tier,
    score_cohort,
)

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

    if rows:
        from db.parquet import append_table
        append_table("prediction_accuracy", rows, ["prediction_type", "evaluation_date", "horizon_days", "model_version"])


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

    evaluated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # Delete existing records for these forecast_ids in batches (SQLite caps
    # bound variables at 999), then bulk-insert everything — replaces per-row
    # update loop (27k queries → a handful).
    existing_ids = list(existing_ids)
    for i in range(0, len(existing_ids), 900):
        batch = existing_ids[i:i+900]
        db.query(ForecastOutcome).filter(
            ForecastOutcome.forecast_id.in_(batch)
        ).delete(synchronize_session=False)

    for o in outcomes:
        o["evaluated_at"] = evaluated_at
    db.bulk_insert_mappings(ForecastOutcome, outcomes)

    db.commit()

    from db.parquet import append_table
    append_table("forecast_outcomes", outcomes, ["forecast_id"])

    logger.info(f"  Stored {len(outcomes)} forecast outcome records ({len(existing_ids)} replaced, {len(outcomes) - len(existing_ids)} new)")


def backtest_forecasts(db, today=None, min_price=0):
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

            if min_price > 0 and current < min_price:
                continue

            abs_error = abs(mid - actual)

            actual_ret = (actual - current) / current
            predicted_direction = f.direction or "flat"
            actual_direction = direction_from_return(actual_ret)
            direction_correct = 1 if predicted_direction == actual_direction else 0

            in_interval = None
            if low is not None and high is not None:
                in_interval = 1 if low <= actual <= high else 0

            conf = f.confidence or "low"
            tier = price_tier(current)

            records.append({
                "abs_error": abs_error,
                "pct_error": abs(abs_error / current) * 100 if current > 0 else 0,
                "sq_error": (mid - actual) ** 2,
                "direction_correct": direction_correct,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "in_interval": in_interval,
                "confidence": conf,
                "base_price": current,
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

        metrics, n = score_cohort(records)
        if n == 0:
            logger.info(f"  [{horizon}d / {model_version}] No valid comparisons")
            continue

        dir_ci_lower = metrics["directional_accuracy_ci_lower"]
        dir_ci_upper = metrics["directional_accuracy_ci_upper"]

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

def run_backtest(types=None, min_price=0, update_bias=False):
    db = SessionLocal()
    today = date.today()
    allowed = ["forecast"]
    types = types or allowed

    try:
        results = {}
        for t in types:
            if t == "forecast":
                results["forecast"] = backtest_forecasts(db, today, min_price=min_price)
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
    result = run_backtest(types, min_price=min_price, update_bias=update_bias)
    print(f"RESULT: {json.dumps(result, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
