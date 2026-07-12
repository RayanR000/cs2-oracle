#!/usr/bin/env python3
"""
Generate ML-based price forecasts for all CS2 items using LightGBM.
Trains quantile regression models (3d, 7d, 14d and 30d horizons) and writes
forecasts to the item_forecasts table.

Usage:
    python scripts/forecast_prices.py          # train + predict
    python scripts/forecast_prices.py --predict-only  # use saved models (auto-retrain on drift)
    python scripts/forecast_prices.py --train-only     # train models only, skip forecasts
"""

import sys
import math
import logging
from pathlib import Path
from datetime import datetime, date, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, ItemForecast, Item
from models.forecaster import ItemForecaster
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("forecast_prices")

MODEL_VERSION = "lgbm-v1"


def run_forecast(train_only: bool = False, predict_only: bool = False):
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        has_models = forecaster.load_models()

        if not predict_only:
            if has_models:
                logger.info("Saved models found, retraining...")
            else:
                logger.info("No saved models found, training from scratch...")
            forecaster.train(max_rows=200_000)
            has_models = True
            # Training takes ~10 min and Supabase may drop idle connections.
            # Refresh the DB session before prediction.
            logger.info("Refreshing DB connection after training...")
            try:
                db.close()
            except Exception:
                pass  # stale connection, discard silently
            db = SessionLocal()
            forecaster.db = db

        if train_only:
            logger.info("Train-only mode, skipping forecast generation.")
            return {"status": "success", "mode": "train_only"}

        if not has_models:
            logger.error("No models available for prediction.")
            return {"status": "error", "message": "No trained models"}

        # Drift-triggered auto-retrain: check if any horizon has drifted.
        # When drift is detected outside of Monday (which always retrains),
        # we retrain immediately so predictions use a fresh model.
        if predict_only and has_models:
            drifted_horizons = []
            for h in ItemForecaster.HORIZONS:
                drift_result = forecaster.check_concept_drift(
                    horizon=h, sliding_window=7, threshold=60.0
                )
                if drift_result and drift_result.get("drifted"):
                    drifted_horizons.append(h)
            if drifted_horizons:
                logger.warning(
                    f"Drift detected for horizons {drifted_horizons} — "
                    f"triggering auto-retrain before prediction."
                )
                forecaster.train(max_rows=200_000)
                has_models = True
                try:
                    db.close()
                except Exception:
                    pass
                db = SessionLocal()
                forecaster.db = db

        results = forecaster.predict()

        if results.empty:
            logger.warning("No forecast results generated.")
            return {"status": "empty", "forecast_count": 0}

        # Map item slugs (strings from Parquet) to integer IDs from the DB.
        # Parquet uses items.item_id (the stable hash name) as the slug key.
        slug_rows = db.execute(
            text("SELECT id, item_id FROM items WHERE is_backfilled = 1")
        ).fetchall()
        slug_to_id = {r.item_id: r.id for r in slug_rows}
        logger.info(f"Loaded {len(slug_to_id)} slug→ID mappings from DB")

        # Write forecasts to DB
        today = date.today()
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
                    "model_version": MODEL_VERSION,
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                })

        # Bulk upsert in batches
        if forecast_rows:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            bind = db.get_bind()
            dialect_name = bind.dialect.name if bind is not None else "sqlite"
            insert_stmt = sqlite_insert if dialect_name == "sqlite" else pg_insert
            table = ItemForecast.__table__

            # SQLite has a default limit of 999 variables per query
            # (~90 rows with 11 columns). PostgreSQL handles 5000+.
            bind = db.get_bind()
            is_sqlite = bind is not None and bind.dialect.name == "sqlite"
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
                db.execute(stmt)
                db.commit()

        logger.info(f"✅ Wrote {len(forecast_rows)} forecasts to item_forecasts table")
        return {
            "status": "success",
            "items": len(results),
            "forecasts": len(forecast_rows),
            "model_version": MODEL_VERSION,
        }

    except Exception as e:
        logger.error(f"❌ Forecast failed: {e}", exc_info=True)
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

    result = run_forecast(train_only=train_only, predict_only=predict_only)
    print(f"RESULT: {result}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
