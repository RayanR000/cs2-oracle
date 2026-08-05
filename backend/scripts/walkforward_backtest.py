#!/usr/bin/env python3
"""
Walk-forward backtest using the current model's tuned parameters.

Loads items from the Parquet archive, engineers features via ItemForecaster,
and evaluates out-of-sample accuracy across walk-forward folds for all horizons.

Usage:
    python scripts/walkforward_backtest.py
    python scripts/walkforward_backtest.py --max-items 200 --horizons 3 7
    python scripts/walkforward_backtest.py --skip-db
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from database import SessionLocal, PredictionAccuracy
from models.forecaster import ItemForecaster
from backtest.scoring import HEADLINE_TIER, score_by_tier
from backtest.walkforward_records import fold_records

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("walkforward_backtest")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Walk-forward config
VAL_WINDOW_DAYS = 21
STEP_DAYS = 60
MAX_TRAIN_ROWS = 200_000
MIN_VAL_SAMPLES = 50
QUANTILES = [0.1, 0.5, 0.9]

# Seed for the per-fold boosters. scripts/compute_mde.py varies this to
# measure the gate's own noise floor: two runs of the same design differ only
# by seed, so the paired difference is the MDE.
FOLD_SEED = 42


def _load_parquet_items(con, backfilled_only=True):
    pq_files = sorted([str(p) for p in ARCHIVE_DIR.glob("prices-*.parquet")])
    pq_queries = []
    for pqf in pq_files:
        cols = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()]
        if "source" in cols:
            pq_queries.append(f"SELECT item_slug, CAST(day AS DATE) AS day, mean_price AS price, source, volume FROM read_parquet('{pqf}')")
        else:
            pq_queries.append(f"SELECT item_slug, CAST(day AS DATE) AS day, mean_price AS price, NULL::VARCHAR AS source, volume FROM read_parquet('{pqf}')")
    union_sql = " UNION ALL BY NAME ".join(pq_queries)

    where_clause = ""
    if backfilled_only:
        where_clause = "WHERE source = 'STEAMCOMMUNITY'"
    query = f"""
        SELECT item_slug,
               MIN(day) AS first_day,
               MAX(day) AS last_day,
               COUNT(*) AS row_count
        FROM ({union_sql})
        {where_clause}
        GROUP BY item_slug
        HAVING row_count >= 90
        ORDER BY row_count DESC
    """
    rows = con.sql(query).fetchall()
    return rows


def _load_all_prices(con, items):
    """Load prices for all requested items in a single scan.

    Previously issued one query per item (up to `max_items` full glob
    scans of the multi-year Parquet archive) — this batches them into
    one query with an IN-list, which the pq_files loop in
    `_load_parquet_items` already proved is the correct pattern here.
    """
    slugs = [item_slug for item_slug, *_ in items]
    if not slugs:
        return pd.DataFrame(columns=["item_id", "timestamp", "price", "volume", "date"])

    slug_list = ", ".join(f"'{s.replace(chr(39), chr(39) + chr(39))}'" for s in slugs)
    rows = con.sql(f"""
        SELECT item_slug AS item_id, CAST(day AS DATE) AS timestamp,
               mean_price AS price, volume
        FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet')
        WHERE item_slug IN ({slug_list})
        ORDER BY item_slug, day
    """).fetchall()
    df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df


# Boosting rounds for the per-fold directional classifier. Matches
# _fit_direction_classifier's own default so the gate's classifier is
# configured like production's.
DIRECTION_NUM_ROUNDS = 200


def _score_fold(*, item_ids, forecast_dates, base_prices, actual_returns_pct,
                mid_returns_pct, low_returns_pct, high_returns_pct,
                predicted_classes):
    """Records for one fold, for both estimators.

    Returns (classifier_records, median_sign_records). Both describe the same
    rows; only `predicted_direction` and `direction_correct` differ. The
    classifier set is what the pre-registered bar governs — production serves
    the classifier's call (forecaster.py:2981-2982) — and the median-sign set
    is reported alongside because the two have never been compared on the same
    folds.
    """
    shared = dict(
        item_ids=item_ids,
        forecast_dates=forecast_dates,
        base_prices=base_prices,
        actual_returns_pct=actual_returns_pct,
        mid_returns_pct=mid_returns_pct,
        low_returns_pct=low_returns_pct,
        high_returns_pct=high_returns_pct,
    )
    return (
        fold_records(**shared, predicted_classes=predicted_classes),
        fold_records(**shared, predicted_classes=None),
    )


def _aggregate_records(records):
    """Pool records across folds and score them with the clustered scorer.

    Replaces the old sample_count-weighted per-fold average, which treated
    every item-row inside a fold as an independent observation. Directional
    outcomes are clustered by date, so the effective sample size is the number
    of distinct dates — see backtest/scoring.py:MIN_FORECAST_DATES.

    Returns the >=$1 headline cohort's metrics, matching production's headline
    tier, with the per-tier rows attached under "by_tier".
    """
    if not records:
        return None
    scored = score_by_tier(records)
    headline = next((m for tier, m, _ in scored if tier == HEADLINE_TIER), None)
    all_tiers = next((m for tier, m, _ in scored if tier is None), None)
    out = dict(headline or all_tiers or {})
    out["by_tier"] = {
        ("all" if tier is None else "headline" if tier == HEADLINE_TIER else f"tier_{tier}"):
            {"directional_accuracy": m["directional_accuracy"], "sample_count": n}
        for tier, m, n in scored
    }
    return out


def _get_tuned_params(meta, horizon, q):
    hp = meta.get("tuned_params", {}).get(str(horizon), {})
    q_str = str(q)
    if q_str in hp:
        return hp[q_str].copy()
    return {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.03,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "min_data_in_leaf": 15,
        "bagging_freq": 5,
        "feature_fraction": 0.7,
    }


def _upsert_accuracy(db, rows):
    from database import PredictionAccuracy
    for row in rows:
        filters = {
            "prediction_type": row["prediction_type"],
            "evaluation_date": row["evaluation_date"],
            "horizon_days": row.get("horizon_days"),
            "model_version": row.get("model_version"),
        }
        existing = db.query(PredictionAccuracy).filter_by(**filters).first()
        if existing:
            existing.sample_count = row["sample_count"]
            existing.metrics = row["metrics"]
            existing.evaluation_window_days = row.get("evaluation_window_days")
            existing.created_at = row["created_at"]
        else:
            db.add(PredictionAccuracy(**row))
    db.commit()


def _build_horizons_report(results_by_horizon, return_records):
    """Assemble the report's `"horizons"` dict.

    `"records"` (Task 5's pairing input) is stripped unless `return_records`
    is set — the report is printed as JSON and the records are large, so the
    default must stay "excluded"; the flag is what makes them retrievable.
    """
    return {
        str(h): {k: v for k, v in m.items()
                 if return_records or k != "records"}
        for h, m in results_by_horizon.items()
    }


def run_walkforward(max_items=500, horizons=None, skip_db=False, return_records=False,
                     step_days: int = STEP_DAYS, fold_seed: int = FOLD_SEED):
    logger.info("=" * 60)
    logger.info("WALK-FORWARD BACKTEST")
    logger.info("=" * 60)

    import duckdb
    con = duckdb.connect()

    try:
        db = SessionLocal()
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        with open("models/saved_models/meta.json") as f:
            meta = json.load(f)

        items = _load_parquet_items(con, backfilled_only=False)[:max_items]
        logger.info(f"Loaded {len(items)} items from Parquet archive")

        all_prices = _load_all_prices(con, items)
        logger.info(f"All prices: {len(all_prices):,} rows, {all_prices['item_id'].nunique()} items")

        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)
        logger.info(f"Feature matrix: {len(df):,} rows, {len(df.columns)} cols")

        results_by_horizon = {}
        total_start = time.time()

        target_horizons = horizons or ItemForecaster.HORIZONS

        for horizon in target_horizons:
            logger.info(f"\n  === Evaluating {horizon}d horizon ===")

            tdf = forecaster.prepare_targets(df, horizon)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values(["item_id", "date"])

            if tdf.empty:
                logger.warning(f"    No valid targets for {horizon}d")
                continue

            dates = sorted(tdf["date"].unique())
            if len(dates) < 90:
                logger.warning(f"    Only {len(dates)} dates available, need >= 90")
                continue

            split_idx = len(dates) * 2 // 3
            logger.info(f"    {len(tdf):,} rows, {len(dates)} dates, split at idx {split_idx}")

            clf_records = []
            median_records = []

            for window_end in range(split_idx + 1, len(dates), step_days):
                train_dates = dates[:window_end]
                val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]

                if len(val_dates) < 7:
                    continue

                train_df = tdf[tdf["date"].isin(train_dates)]
                val_df = tdf[tdf["date"].isin(val_dates)]

                if len(val_df) < MIN_VAL_SAMPLES:
                    continue

                if len(train_df) > MAX_TRAIN_ROWS:
                    train_df = train_df.sort_values("date").tail(MAX_TRAIN_ROWS)

                feature_cols = [c for c in forecaster.feature_cols if c in tdf.columns]
                if not feature_cols:
                    exclude = {"item_id", "date", "timestamp", "price", "volume"}
                    exclude |= {f"target_{h}d" for h in forecaster.HORIZONS}
                    exclude |= {f"target_return_{h}d" for h in forecaster.HORIZONS}
                    feature_cols = [c for c in tdf.columns if c not in exclude
                                    and tdf[c].dtype in (np.float64, np.float32, np.int64, int, float)]

                if len(feature_cols) > 2:
                    corr = train_df[feature_cols].corr().abs()
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    to_drop = set()
                    for col in upper.columns:
                        if col in to_drop:
                            continue
                        highly_corr = upper[col][upper[col] > 0.95].index
                        to_drop.update(highly_corr)
                    feature_cols = [c for c in feature_cols if c not in to_drop]

                medians = train_df[feature_cols].median()
                X_train = train_df[feature_cols].fillna(medians)
                y_train = train_df[f"target_return_{horizon}d"]
                X_val = val_df[feature_cols].fillna(medians)
                y_val = val_df[f"target_return_{horizon}d"]

                preds = {}
                for q in QUANTILES:
                    params = _get_tuned_params(meta, horizon, q)
                    params.update({
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "verbosity": -1,
                        "random_state": fold_seed,
                        "n_jobs": -1,
                    })

                    dtrain = lgb.Dataset(X_train.values, y_train.values)
                    dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain)
                    model = lgb.train(
                        params, dtrain,
                        num_boost_round=200,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)]
                    )
                    preds[q] = model.predict(X_val.values)

                if len(preds) != 3:
                    continue

                low, high = ItemForecaster._fix_quantile_crossing(
                    preds[0.1], preds[0.5], preds[0.9]
                )

                # The estimator production actually serves. sigma_train/
                # sigma_val stay None so the fixed-band labels production
                # selects are used (forecaster.py:2984-2993).
                clf = forecaster._fit_direction_classifier(
                    X_train.values, y_train.values,
                    X_val.values, y_val.values,
                    _get_tuned_params(meta, horizon, 0.5).get("boosting_type", "gbdt"),
                    ItemForecaster._direction_tree_params(
                        {0.5: _get_tuned_params(meta, horizon, 0.5)}
                    ),
                    horizon=horizon,
                    sigma_train=None,
                    sigma_val=None,
                    num_boost_round=DIRECTION_NUM_ROUNDS,
                    random_state=fold_seed,
                )
                predicted_classes = clf.predict(X_val.values).argmax(axis=1)

                fold_clf, fold_median = _score_fold(
                    item_ids=val_df["item_id"].to_numpy(),
                    forecast_dates=val_df["date"].to_numpy(),
                    base_prices=val_df["price"].to_numpy(dtype=float),
                    actual_returns_pct=y_val.to_numpy(dtype=float),
                    mid_returns_pct=preds[0.5],
                    low_returns_pct=low,
                    high_returns_pct=high,
                    predicted_classes=predicted_classes,
                )
                clf_records.extend(fold_clf)
                median_records.extend(fold_median)

            if not clf_records:
                logger.warning(f"    No folds completed for {horizon}d")
                continue

            agg_clf = _aggregate_records(clf_records)
            agg_median = _aggregate_records(median_records)
            entry = {
                "classifier": agg_clf,
                "median_sign": agg_median,
                "sample_count": len(clf_records),
            }
            if return_records:
                entry["records"] = clf_records
            results_by_horizon[horizon] = entry

            lo = agg_clf["directional_accuracy_ci_clustered_lower"]
            hi = agg_clf["directional_accuracy_ci_clustered_upper"]
            ci_txt = f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "[insufficient dates]"
            logger.info(
                f"    {horizon}d: {len(clf_records):,} records, "
                f"{agg_clf['distinct_forecast_dates']} dates "
                f"(sufficient={agg_clf['date_coverage_sufficient']})"
            )
            logger.info(
                f"      classifier DirAcc={agg_clf['directional_accuracy']:.1f}% "
                f"clustered95={ci_txt}   "
                f"median-sign DirAcc={agg_median['directional_accuracy']:.1f}%"
            )
            logger.info(
                f"      MAE=${agg_clf['mae']:.2f}  MAPE={agg_clf['mape']:.1f}%  "
                f"IntCov={agg_clf['interval_coverage']:.1f}%"
            )

        total_elapsed = time.time() - total_start
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtest complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        logger.info(f"{'='*60}")

        if not skip_db:
            today = date.today()
            for horizon, entry in results_by_horizon.items():
                clf = entry["classifier"]
                _upsert_accuracy(db, [{
                    "prediction_type": "walkforward_backtest",
                    "evaluation_date": today,
                    "horizon_days": horizon,
                    # Bumped: the metric definition changed (3-label with a
                    # flat band, classifier-sourced direction, clustered CI),
                    # so these rows are NOT continuous with lgbm-v3-tuned.
                    "model_version": "lgbm-v3-clustered",
                    "evaluation_window_days": None,
                    "sample_count": entry["sample_count"],
                    "metrics": {
                        k: clf[k] for k in [
                            "mae", "rmse", "mape", "wmape",
                            "directional_accuracy",
                            "directional_accuracy_ci_clustered_lower",
                            "directional_accuracy_ci_clustered_upper",
                            "distinct_forecast_dates",
                            "date_coverage_sufficient",
                            "interval_coverage",
                        ]
                    } | {
                        "median_sign_directional_accuracy":
                            entry["median_sign"]["directional_accuracy"],
                        "by_tier": clf["by_tier"],
                    },
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                }])

        con.close()
        try:
            db.close()
        except Exception:
            pass

        report = {
            "test_date": str(date.today()),
            "total_items": len(items),
            "total_elapsed_seconds": round(total_elapsed, 1),
            "horizons": _build_horizons_report(results_by_horizon, return_records),
        }
        return report

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward backtest with tuned params")
    parser.add_argument("--max-items", type=int, default=500, help="Items to evaluate (default: 500)")
    parser.add_argument("--horizons", type=int, nargs="+", default=None, help="Horizons to test (default: all)")
    parser.add_argument("--skip-db", action="store_true", help="Skip writing to database")
    parser.add_argument("--step-days", type=int, default=STEP_DAYS,
                         help=f"Fold stride in days (default: {STEP_DAYS})")
    args = parser.parse_args()

    report = run_walkforward(max_items=args.max_items, horizons=args.horizons,
                              skip_db=args.skip_db, step_days=args.step_days)
    print(f"\nRESULT: {json.dumps(report, indent=2, default=str)}")
    return 0 if report.get("status") != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
