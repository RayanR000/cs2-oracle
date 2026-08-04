#!/usr/bin/env python3
"""
Generate ML-based price forecasts for all CS2 items using LightGBM.
Trains quantile regression models (3d, 7d, 14d and 30d horizons) and writes
forecasts to the item_forecasts table.

Usage:
    python scripts/forecast_prices.py          # train + predict
    python scripts/forecast_prices.py --predict-only  # use saved models (no auto-retrain)
    python scripts/forecast_prices.py --train-only     # train models only, skip forecasts
    python scripts/forecast_prices.py --compare-regime  # A/B test regime vs global-only + backtest
    python scripts/forecast_prices.py --compare-ensemble # A/B test 3-member vs 6-member ensemble + backtest
"""

import sys
import os
import json
import math
import logging
from pathlib import Path
from datetime import datetime, date, timezone
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, ItemForecast, Item
from models.forecaster import ItemForecaster
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("forecast_prices")

MODEL_VERSION = "lgbm-v3"


def _model_age_days(forecaster) -> Optional[int]:
    """Days since the currently saved model was trained, or None if unknown."""
    meta_path = os.path.join(forecaster.model_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None
    trained_at = meta.get("trained_at")
    if not trained_at:
        return None
    try:
        trained = datetime.fromisoformat(trained_at)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - trained).days


def _write_forecasts_to_db(db, results, model_version, slug_to_id, today):
    """Write forecast results to the item_forecasts table. Returns count."""
    forecast_rows = []
    for _, row in results.iterrows():
        slug = str(row["item_id"])
        item_id = slug_to_id.get(slug)
        if item_id is None:
            logger.warning(f"  Skipping unknown slug: {slug}")
            continue
        current_price = row.get("current_price")
        forecasts = row.get("forecasts", {})

        for horizon, fcast in forecasts.items():
            forecast_rows.append({
                "item_id": item_id,
                "forecast_date": today,
                "horizon_days": horizon,
                "price_low": fcast.get("low"),
                "price_mid": fcast.get("mid"),
                "price_high": fcast.get("high"),
                "current_price": current_price,
                "direction": fcast.get("direction"),
                "confidence": fcast.get("confidence"),
                "model_version": model_version,
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            })

    if forecast_rows:
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        # Verify DB connection is still alive before the batch insert loop.
        # Training can take >1h, and the connection may have gone stale.
        try:
            db.execute(text("SELECT 1"))
        except Exception:
            logger.warning("  DB connection stale, reconnecting...")
            db.close()
            from database import SessionLocal
            db = SessionLocal()

        bind = db.get_bind()
        is_sqlite = bind is not None and bind.dialect.name == "sqlite"
        insert_stmt = sqlite_insert if is_sqlite else pg_insert
        table = ItemForecast.__table__
        batch_size = 90 if is_sqlite else 5000
        for i in range(0, len(forecast_rows), batch_size):
            batch = forecast_rows[i:i + batch_size]
            stmt = insert_stmt(table).values(batch)
            excluded = stmt.excluded
            update_cols = {
                col.name: getattr(excluded, col.name)
                for col in table.columns
                if col.name not in {"id", "item_id", "forecast_date", "horizon_days", "created_at"}
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["item_id", "forecast_date", "horizon_days"],
                set_=update_cols,
            )
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    db.execute(stmt)
                    db.commit()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"  Batch insert failed (attempt {attempt + 1}/{max_retries}), reconnecting and retrying: {e}")
                        db.rollback()
                        db.close()
                        from database import SessionLocal
                        db = SessionLocal()
                    else:
                        logger.error(f"  Batch insert failed after {max_retries} attempts: {e}")
                        db.rollback()
                        raise

        from db.parquet import append_table
        append_table("item_forecasts", forecast_rows, ["item_id", "forecast_date", "horizon_days"])

    return len(forecast_rows)


def run_forecast(train_only: bool = False, predict_only: bool = False,
                 compare_regime: bool = False,
                 compare_ensemble: bool = False,
                 update_bias: bool = False):
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db, prune_failed_groups=False)
        has_models = forecaster.load_models()

        force_retrain = os.environ.get("FORCE_RETRAIN") == "1"
        retrain_interval = int(os.environ.get("RETRAIN_INTERVAL_DAYS", "14"))

        do_train = False
        if train_only:
            do_train = True
        elif not predict_only:
            if not has_models:
                logger.info("No saved models found, training from scratch...")
                do_train = True
            elif force_retrain:
                logger.info("FORCE_RETRAIN set, retraining...")
                do_train = True
            else:
                age = _model_age_days(forecaster)
                # Retrain on model age only. Drift is not a valid trigger: it
                # reads a 1-2-forecast-date sample, so it tracks market
                # direction rather than model decay.
                if age is None or age >= retrain_interval:
                    reason = ("model stale" if (age is not None and age >= retrain_interval)
                              else "unknown age")
                    logger.info(f"Retraining ({reason}): age={age}, interval={retrain_interval}d")
                    do_train = True
                else:
                    logger.info(
                        f"Skipping retrain: model {age}d old (<{retrain_interval}d)"
                    )
        elif predict_only and has_models:
            # Drift is reported but does NOT trigger a retrain. The signal it
            # reads spans 1-2 distinct forecast dates, so it tracks market
            # direction rather than model decay, and the retrain it used to
            # trigger cost a measured 465s of every 835s daily run while
            # making the workflow's Monday-only retrain design fiction.
            # See docs/superpowers/specs/2026-08-04-remove-accidental-retrain-work-design.md
            allow_retrain = os.environ.get("ALLOW_DRIFT_RETRAIN") == "1"
            drifted_horizons = []
            for h in ItemForecaster.HORIZONS:
                drift_result = forecaster.check_concept_drift(horizon=h, sliding_window=7)
                if drift_result and drift_result.get("drifted"):
                    drifted_horizons.append(h)
            if drifted_horizons and allow_retrain:
                logger.warning(
                    f"Drift reported for horizons {drifted_horizons} and "
                    f"ALLOW_DRIFT_RETRAIN=1 — retraining before prediction."
                )
                do_train = True
            elif drifted_horizons:
                logger.warning(
                    f"Drift reported for horizons {drifted_horizons}. Not "
                    f"retraining: predict-only serves the scheduled model. Set "
                    f"ALLOW_DRIFT_RETRAIN=1 to retrain, or dispatch the "
                    f"workflow with mode=full."
                )

        if do_train:
            if has_models:
                logger.info("Saved models found, retraining...")
            forecaster.train(max_rows=700_000)
            has_models = True
            logger.info("Refreshing DB connection after training...")
            try:
                db.close()
            except Exception:
                pass
            db = SessionLocal()
            forecaster.db = db

        if train_only:
            logger.info("Train-only mode, skipping forecast generation.")
            return {"status": "success", "mode": "train_only"}

        if not has_models:
            logger.error("No models available for prediction.")
            return {"status": "error", "message": "No trained models"}

        # Map item slugs to integer IDs
        slug_rows = db.execute(
            text("SELECT id, item_id FROM items WHERE is_backfilled = 1")
        ).fetchall()
        slug_to_id = {r.item_id: r.id for r in slug_rows}
        logger.info(f"Loaded {len(slug_to_id)} slug->ID mappings from DB")
        override = os.environ.get("FORECAST_DATE_OVERRIDE")
        today = date.fromisoformat(override) if override else date.today()

        # Run A: prediction WITH regime-switching
        results = forecaster.predict()
        if results.empty:
            logger.warning("No forecast results generated.")
            return {"status": "empty", "forecast_count": 0}

        version_regime = f"{MODEL_VERSION}-regime"
        n_regime = _write_forecasts_to_db(db, results, version_regime, slug_to_id, today)
        logger.info(f"Wrote {n_regime} forecasts (regime mode) to item_forecasts table")

        # Update bias corrections from outcomes if requested
        if update_bias:
            logger.info("Updating per-tier bias corrections from outcomes...")
            try:
                forecaster.update_bias_corrections_from_outcomes()
            except Exception as e:
                logger.error(f"Bias update failed: {e}", exc_info=True)

        # Run B: prediction WITHOUT regime-switching (A/B comparison)
        if compare_regime:
            logger.info("=" * 60)
            logger.info("COMPARISON MODE: re-running with regime models disabled")
            logger.info("=" * 60)
            n_cleared = sum(len(v) for v in forecaster.regime_models.values())
            forecaster.regime_models.clear()
            logger.info(f"  Cleared {n_cleared} regime model groups")

            results_global = forecaster.predict()
            version_global = f"{MODEL_VERSION}-global-only"
            n_global = _write_forecasts_to_db(db, results_global, version_global, slug_to_id, today)
            logger.info(f"Wrote {n_global} forecasts (global-only mode) to item_forecasts table")

            # Run backtest on both model versions
            logger.info("=" * 60)
            logger.info("Running backtest on both model versions...")
            logger.info("=" * 60)
            try:
                from scripts.backtest_accuracy import backtest_forecasts
                bt_results = backtest_forecasts(db, today)
                logger.info(f"Backtest complete: {len(bt_results or [])} accuracy records")
            except Exception as e:
                logger.error(f"Backtest failed: {e}", exc_info=True)
                bt_results = []

            return {
                "status": "success",
                "items_regime": len(results),
                "forecasts_regime": n_regime,
                "items_global": len(results_global),
                "forecasts_global": n_global,
                "backtest_records": len(bt_results or []),
                "model_version_regime": version_regime,
                "model_version_global": version_global,
            }

        # Compare ensemble sizes: 3-member vs 6-member
        if compare_ensemble:
            logger.info("=" * 60)
            logger.info("COMPARISON MODE: comparing 3-member vs 6-member ensemble")
            logger.info("=" * 60)

            orig_n = ItemForecaster.N_ENSEMBLES
            orig_seeds = ItemForecaster.ENSEMBLE_SEEDS
            orig_ff = ItemForecaster.ENSEMBLE_FEATURE_FRACTIONS

            # Run A: 3-member ensemble
            logger.info("Training 3-member ensemble...")
            ItemForecaster.N_ENSEMBLES = 3
            ItemForecaster.ENSEMBLE_SEEDS = [42, 73, 91]
            ItemForecaster.ENSEMBLE_FEATURE_FRACTIONS = [0.6, 0.65, 0.7]
            forecaster_ens3 = ItemForecaster(
                db_session=db,
                model_dir=str(Path(__file__).parent.parent / "models" / "saved_models_ens3")
            )
            forecaster_ens3.train(max_rows=700_000)
            db.close()
            db = SessionLocal()
            results_ens3 = forecaster_ens3.predict()
            version_ens3 = f"{MODEL_VERSION}-ens3"
            n_ens3 = _write_forecasts_to_db(db, results_ens3, version_ens3, slug_to_id, today)
            logger.info(f"Wrote {n_ens3} forecasts (ens3 mode) to item_forecasts table")

            # Run B: 6-member ensemble
            logger.info("Training 6-member ensemble...")
            ItemForecaster.N_ENSEMBLES = orig_n
            ItemForecaster.ENSEMBLE_SEEDS = orig_seeds
            ItemForecaster.ENSEMBLE_FEATURE_FRACTIONS = orig_ff
            forecaster_ens6 = ItemForecaster(
                db_session=db,
                model_dir=str(Path(__file__).parent.parent / "models" / "saved_models_ens6")
            )
            forecaster_ens6.train(max_rows=700_000)
            db.close()
            db = SessionLocal()
            results_ens6 = forecaster_ens6.predict()
            version_ens6 = f"{MODEL_VERSION}-ens6"
            n_ens6 = _write_forecasts_to_db(db, results_ens6, version_ens6, slug_to_id, today)
            logger.info(f"Wrote {n_ens6} forecasts (ens6 mode) to item_forecasts table")

            # Backtest both
            logger.info("=" * 60)
            logger.info("Running backtest on both ensemble sizes...")
            logger.info("=" * 60)
            try:
                from scripts.backtest_accuracy import backtest_forecasts
                bt_results = backtest_forecasts(db, today)
                logger.info(f"Backtest complete: {len(bt_results or [])} accuracy records")
            except Exception as e:
                logger.error(f"Backtest failed: {e}", exc_info=True)
                bt_results = []

            return {
                "status": "success",
                "items_ens3": len(results_ens3),
                "forecasts_ens3": n_ens3,
                "items_ens6": len(results_ens6),
                "forecasts_ens6": n_ens6,
                "backtest_records": len(bt_results or []),
                "model_version_ens3": version_ens3,
                "model_version_ens6": version_ens6,
            }

        return {
            "status": "success",
            "items": len(results),
            "forecasts": n_regime,
            "model_version": version_regime,
        }

    except Exception as e:
        logger.error(f"Forecast failed: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "message": str(e)}

    finally:
        try:
            db.close()
        except Exception:
            pass


def main():
    args = set(sys.argv[1:])
    train_only = "--train-only" in args
    predict_only = "--predict-only" in args
    compare_regime = "--compare-regime" in args
    compare_ensemble = "--compare-ensemble" in args
    update_bias = "--update-bias" in args

    result = run_forecast(train_only=train_only, predict_only=predict_only,
                          compare_regime=compare_regime,
                          compare_ensemble=compare_ensemble,
                          update_bias=update_bias)
    print(f"RESULT: {result}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
