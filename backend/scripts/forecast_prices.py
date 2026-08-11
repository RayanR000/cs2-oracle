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
from models.forecaster import ItemForecaster, IncompatibleModelArtifact
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("forecast_prices")

MODEL_VERSION = "lgbm-v3"

# Rows kept BEFORE feature engineering, which is what decides how many whole
# item histories the model learns from. Raised 100_000 -> 1_200_000 on
# 2026-08-08 together with the price floor below, because the two are one
# setting: at the floor the served cohort is 926 items / 993,464 item-days, so
# this budget covers it outright and _stratified_item_subsample never runs.
# That is the whole point — the subsample's hardcoded seed=42 moves
# mean_classifier_acc_ge1 by sd 1.5-3.1pp across 8 draws, and no draw is the
# only setting that removes that variance rather than shrinking it. The
# 4.5x-wall-clock objection this default used to carry was priced as a daily
# cost; the workflow retrains Mondays only. Raising the floor means re-measuring
# the cohort's row count here. See
# docs/changelog/2026-08-08-training-price-floor-shipped.md.
# Env-configured to match SKIP_REGIMES/SKIP_CV rather than a flag, because this
# script parses argv as a plain set.
DEFAULT_TRAIN_FEATURE_ROWS = 1_200_000

# Rows per horizon AFTER feature engineering. A different knob, held at or above
# the feature budget so it stays non-binding — the per-horizon frame measured
# 958,289 rows on the >= $1 universe, so the old 700_000 would have started
# binding the moment the floor shipped and one change would have moved two
# things. Raised to keep the floor the only treatment, not to buy coverage.
TRAIN_HORIZON_MAX_ROWS = 1_200_000

# Median-price floor on the training universe, matching
# api/serving_policy.py::MIN_SERVED_PRICE_USD so the model trains on the cohort
# it serves. Inseparable from DEFAULT_TRAIN_FEATURE_ROWS above.
DEFAULT_TRAIN_MIN_MEDIAN_PRICE = 1.0

# Share of the directional classifier's training weight placed on the >= $1
# cohort. None keeps the pre-2026-08-06 behaviour exactly: no tier weighting,
# so the classifier trains on a frame that is ~83% sub-$1 while the product
# serves only >= $1 (api/serving_policy.py). Env-configured to match
# TRAIN_FEATURE_ROWS/SKIP_CV rather than a flag, because this script parses
# argv as a plain set. Raise it only against a measured gain in
# mean_classifier_acc_ge1 — see
# docs/superpowers/specs/2026-08-06-served-cohort-weighting-design.md.
DEFAULT_SERVED_COHORT_SHARE = None


def _served_cohort_share() -> Optional[float]:
    raw = os.environ.get("TRAIN_SERVED_COHORT_SHARE")
    if not raw:
        return DEFAULT_SERVED_COHORT_SHARE
    try:
        share = float(raw)
    except ValueError:
        logger.warning(
            f"TRAIN_SERVED_COHORT_SHARE={raw!r} is not a number; using "
            f"{DEFAULT_SERVED_COHORT_SHARE}"
        )
        return DEFAULT_SERVED_COHORT_SHARE
    if not 0.0 < share < 1.0:
        logger.warning(
            f"TRAIN_SERVED_COHORT_SHARE={share} is not in (0, 1); using "
            f"{DEFAULT_SERVED_COHORT_SHARE}"
        )
        return DEFAULT_SERVED_COHORT_SHARE
    logger.info(
        f"TRAIN_SERVED_COHORT_SHARE override: {share:.2f} of the direction "
        f"classifier's training weight goes to >= $1 rows (default "
        f"{DEFAULT_SERVED_COHORT_SHARE}) — expect mean_classifier_acc "
        f"(pooled) to fall and mean_classifier_acc_ge1 to be the metric that "
        f"matters"
    )
    return share


def _model_dir() -> Optional[str]:
    """Override where model artifacts are read from and written to.

    None keeps ItemForecaster's default (models/saved_models/), which is the
    deployed artifact. Point this at a scratch directory to run a training arm
    without clobbering production's model — there is no other guard: train()
    overwrites meta.json and the boosters in place.

    Env-configured for the same reason as TRAIN_FEATURE_ROWS: this script parses
    argv as a plain set, so a two-token flag does not fit.
    """
    raw = os.environ.get("FORECAST_MODEL_DIR")
    if not raw:
        return None
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"FORECAST_MODEL_DIR override: artifacts read/written under {path} "
        f"— the deployed model in models/saved_models/ is untouched"
    )
    return str(path)


def _train_horizons() -> Optional[list]:
    """`TRAIN_HORIZONS=30` or `TRAIN_HORIZONS=3,7` restricts which horizons train.

    Exists for the scheduled diagnostics job, which turns the CV diagnostic
    classifier back on (52% of a retrain) and so cannot do all four horizons
    inside the 30-minute cap. Horizons are independent — every piece of
    per-horizon state in ItemForecaster is dict-keyed by horizon — so one job per
    horizon is a valid decomposition.

    **The artifact it writes is partial**, holding boosters for the selected
    horizons only. Never point this at the deployed model directory: pair it with
    FORECAST_MODEL_DIR. Unparseable or empty values train every horizon, matching
    TRAIN_MIN_MEDIAN_PRICE's rule that a typo must not silently narrow the run.
    """
    raw = (os.environ.get("TRAIN_HORIZONS") or "").strip()
    if not raw:
        return None
    try:
        picked = [int(tok) for tok in raw.split(",") if tok.strip()]
    except ValueError:
        logger.warning(f"TRAIN_HORIZONS={raw!r} is not a comma-separated int "
                       f"list; training every horizon")
        return None
    unknown = [h for h in picked if h not in ItemForecaster.HORIZONS]
    if unknown or not picked:
        logger.warning(f"TRAIN_HORIZONS={raw!r} names unknown horizons "
                       f"{unknown or '[]'}; training every horizon")
        return None
    logger.info(f"TRAIN_HORIZONS override: training {picked} only — the saved "
                f"artifact will be PARTIAL and must not be served")
    return picked


def _train_feature_rows() -> int:
    raw = os.environ.get("TRAIN_FEATURE_ROWS")
    if not raw:
        return DEFAULT_TRAIN_FEATURE_ROWS
    try:
        budget = int(raw)
    except ValueError:
        logger.warning(
            f"TRAIN_FEATURE_ROWS={raw!r} is not an integer; using "
            f"{DEFAULT_TRAIN_FEATURE_ROWS:,}"
        )
        return DEFAULT_TRAIN_FEATURE_ROWS
    if budget <= 0:
        logger.warning(
            f"TRAIN_FEATURE_ROWS={budget} is not positive; using "
            f"{DEFAULT_TRAIN_FEATURE_ROWS:,}"
        )
        return DEFAULT_TRAIN_FEATURE_ROWS
    if budget != DEFAULT_TRAIN_FEATURE_ROWS:
        logger.info(
            f"TRAIN_FEATURE_ROWS override: {budget:,} rows "
            f"(default {DEFAULT_TRAIN_FEATURE_ROWS:,}) — expect training "
            f"wall-clock to move with it"
        )
    return budget


def _train_min_median_price() -> Optional[float]:
    """Median-price floor on the training universe, or None for no filter.

    Shipped at $1 on 2026-08-08 — step 7 of
    docs/research/2026-08-07-next-steps.md. The justification is
    **measurability, not accuracy**: with DEFAULT_TRAIN_FEATURE_ROWS the floor
    covers the served cohort with no subsample, so the item draw is removed by
    construction rather than shrunk. The +3.50pp at 30d that once also
    justified it re-derives to +1.642pp [-0.809, +4.505], null, inside a
    +/-3-4pp noise floor, and must not be cited
    (docs/changelog/2026-08-08-per-fold-price-filter-rederived.md).

    Set with TRAIN_FEATURE_ROWS, not instead of it: the floor decides *which*
    items the budget may buy, the budget decides how many. A floor under too
    small a budget re-introduces the draw over a smaller universe, which is
    worse than either alone.

    An unparseable value keeps the default floor rather than disabling it — a
    typo must not silently widen the training universe back to the pool. A
    value <= 0 is the deliberate escape hatch to the pre-2026-08-08 behaviour.
    """
    raw = os.environ.get("TRAIN_MIN_MEDIAN_PRICE")
    if not raw:
        floor = DEFAULT_TRAIN_MIN_MEDIAN_PRICE
    else:
        try:
            floor = float(raw)
        except ValueError:
            logger.warning(
                f"TRAIN_MIN_MEDIAN_PRICE={raw!r} is not a number; keeping the "
                f"default floor of ${DEFAULT_TRAIN_MIN_MEDIAN_PRICE:g}"
            )
            floor = DEFAULT_TRAIN_MIN_MEDIAN_PRICE
        else:
            if floor <= 0:
                logger.warning(
                    f"TRAIN_MIN_MEDIAN_PRICE={floor} disables the price floor: "
                    f"training on the pooled universe, where the row budget "
                    f"buys the pool's tier mix (44% stickers and graffiti at a "
                    f"$0.03 median) and the item draw returns"
                )
                return None
    logger.info(
        f"Training universe restricted to items with median price >= "
        f"${floor:g} (default ${DEFAULT_TRAIN_MIN_MEDIAN_PRICE:g})"
    )
    return floor


def _train_per_item_rows() -> bool:
    """Whether the per-horizon row cap draws an equal quota per item.

    Default False keeps the training set byte-identical to the pre-2026-08-07
    model, where the cap is a uniform draw and an item's share of the sample is
    its share of the rows. True spends the same budget on breadth instead: the
    paired harness measured that arm at +5.72pp at 30d against +3.50pp for the
    row-heavy one, ten times the rows losing at both horizons.

    Pairs with TRAIN_MIN_MEDIAN_PRICE and TRAIN_FEATURE_ROWS, and is the reason
    a wide floor need not cost a wide budget: the floor picks the universe,
    TRAIN_FEATURE_ROWS picks how many whole histories are engineered, and this
    decides how deeply each is drawn for the fit.

    Unrecognised values fall back to False rather than True — a typo must not
    silently change the training set.
    """
    raw = os.environ.get("TRAIN_PER_ITEM_ROWS", "").strip().lower()
    if not raw:
        return False
    if raw in {"1", "true", "yes", "on"}:
        logger.info(
            "TRAIN_PER_ITEM_ROWS override: the per-horizon row cap draws an "
            "equal quota per item instead of a uniform sample of rows"
        )
        return True
    if raw not in {"0", "false", "no", "off"}:
        logger.warning(
            f"TRAIN_PER_ITEM_ROWS={raw!r} is not a boolean; leaving per-item "
            f"row sampling off"
        )
    return False


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


def _write_forecasts_to_db(db, results, model_version, slug_to_id, today,
                           model_config=None):
    """Write forecast results to the item_forecasts table. Returns count.

    `model_version` is the served artifact's IDENTITY and nothing else.
    `model_config` names the serving configuration that produced the row and
    goes to the Parquet mirror only — the same split as `item_slug` below.

    Encoding the configuration in `model_version` is what F3 removed. The two
    were one string (`lgbm-v3-regime` / `lgbm-v3-global-only`), and
    `backtest/scoring.py::score_cohort` keys the cohort on it, so a config flag
    forked the date panel and no cohort could reach MIN_FORECAST_DATES. The
    suffix bought nothing in exchange: item_forecasts is unique on
    (item_id, forecast_date, horizon_days), so two configs writing the same day
    never coexisted as rows — the second overwrote the first and only relabelled
    it. See docs/changelog/2026-08-11-model-version-is-not-a-config.md.
    """
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

        # The Parquet mirror carries the slug as well as the surrogate id. The
        # price archive keys on item_slug, so without it a forecast cannot be
        # joined to its own price history without a round-trip to Supabase.
        # Mirror only: `forecast_rows` is the DB payload above and ItemForecast
        # has no such column.
        id_to_slug = {v: k for k, v in slug_to_id.items()}
        from db.parquet import append_table
        append_table(
            "item_forecasts",
            [{**r,
              "item_slug": id_to_slug.get(r["item_id"]),
              "model_config": model_config}
             for r in forecast_rows],
            ["item_id", "forecast_date", "horizon_days"],
        )

    return len(forecast_rows)


def run_forecast(train_only: bool = False, predict_only: bool = False,
                 compare_regime: bool = False,
                 update_bias: bool = False,
                 predict_smoke: bool = False):
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db, prune_failed_groups=False,
                                    served_cohort_share=_served_cohort_share(),
                                    model_dir=_model_dir())
        # Instance override, so the class default is untouched for anything else
        # importing ItemForecaster in this process.
        _horizons = _train_horizons()
        if _horizons is not None:
            forecaster.HORIZONS = _horizons
        # predict_only is a parameter, already known here -- no need to defer
        # this decision until do_train is computed below. A cache that
        # predates the current artifact scheme is exactly "no usable models"
        # for any mode that is willing to train, but predict-only has no
        # recovery path: serving from it would mean q_hat means something
        # other than what this code expects, which is the one thing this
        # guard exists to prevent.
        try:
            has_models = forecaster.load_models()
        except IncompatibleModelArtifact as e:
            if predict_only:
                raise
            logger.warning(
                f"Saved model cache is incompatible ({e}); ignoring it and "
                f"training from scratch."
            )
            has_models = False

        force_retrain = os.environ.get("FORCE_RETRAIN") == "1"
        retrain_interval = int(os.environ.get("RETRAIN_INTERVAL_DAYS", "14"))

        do_train = False
        if train_only or predict_smoke:
            # predict_smoke trains unconditionally, and must: its entire purpose
            # is to run the predict path under THIS run's flags. Left to the
            # branch below it would restore a cached artifact, skip training on
            # the age gate, and then predict with flags the artifact was not
            # trained under -- which is how run 31439896107 failed, with the
            # rank transform on from the environment and no cohort recorded in
            # the restored meta.
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
            forecaster.train(max_rows=TRAIN_HORIZON_MAX_ROWS,
                             max_feature_rows=_train_feature_rows(),
                             min_median_price=_train_min_median_price(),
                             per_item_row_sampling=_train_per_item_rows())
            has_models = True
            logger.info("Refreshing DB connection after training...")
            try:
                db.close()
            except Exception:
                pass
            db = SessionLocal()
            forecaster.db = db

        if predict_smoke:
            # Exercises the predict PATH and writes nothing. It exists because
            # the cross-sectional rank transform's serving branch
            # (_reference_cohort_mask -> reference_mask) cannot be reached from
            # CV: folds are already cohort-filtered, so they take the
            # reference_mask=None branch. The only other way to run it on real
            # data was a full price-forecast.yml dispatch, which writes
            # forecasts to prod under the SAME model_version production uses --
            # indistinguishable from real rows, and frozen into outcomes.
            #
            # It reports shape, not quality. Nothing here says the forecasts are
            # any good; a diagnostics run may hold one freshly trained horizon
            # beside three restored from cache.
            logger.info("Predict-smoke mode: running predict(), writing nothing.")
            results = forecaster.predict()
            n_items = int(results["item_id"].nunique()) if len(results) else 0
            # A row whose `forecasts` dict is empty reached the writer with
            # nothing to write -- the shape failure this mode is looking for.
            with_forecasts = int(
                sum(1 for f in results.get("forecasts", []) if f)
            ) if len(results) else 0
            logger.info(
                f"  predict-smoke: {len(results):,} rows, {n_items:,} items, "
                f"{with_forecasts:,} carrying a non-empty forecast dict"
            )
            return {"status": "success", "mode": "predict_smoke",
                    "rows": len(results), "items": n_items,
                    "with_forecasts": with_forecasts}

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

        # Read off the forecaster, not off which branch we are in: SKIP_REGIMES=1
        # (production since 2026-08-10) leaves regime_models empty, so the old
        # unconditional "-regime" label was describing a configuration that did
        # not run. `predict` prefers a regime model over the global one where it
        # has one, which is exactly why this has to be observed rather than
        # assumed.
        config_a = "regime" if forecaster.regime_models else "global-only"
        n_regime = _write_forecasts_to_db(db, results, MODEL_VERSION, slug_to_id,
                                          today, model_config=config_a)
        logger.info(f"Wrote {n_regime} forecasts ({config_a} config) to item_forecasts table")

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
            n_global = _write_forecasts_to_db(db, results_global, MODEL_VERSION,
                                              slug_to_id, today,
                                              model_config="global-only")
            logger.info(f"Wrote {n_global} forecasts (global-only config) to item_forecasts table")

            # Run backtest on both configs. They no longer score as separate
            # cohorts, and they never were separate rows: the second write
            # overwrote the first on the (item, date, horizon) key, so what is
            # stored for today is the global-only run either way.
            logger.info("=" * 60)
            logger.info("Running backtest...")
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
                "model_version": MODEL_VERSION,
                "model_config_a": config_a,
                "model_config_b": "global-only",
            }

        return {
            "status": "success",
            "items": len(results),
            "forecasts": n_regime,
            "model_version": MODEL_VERSION,
            "model_config": config_a,
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
    update_bias = "--update-bias" in args
    predict_smoke = "--predict-smoke" in args

    result = run_forecast(train_only=train_only, predict_only=predict_only,
                          compare_regime=compare_regime,
                          update_bias=update_bias,
                          predict_smoke=predict_smoke)
    print(f"RESULT: {result}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
