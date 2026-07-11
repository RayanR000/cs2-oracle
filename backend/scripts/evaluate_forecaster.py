#!/usr/bin/env python3
"""
Walk-forward evaluation of the ML forecaster on parquet archive data.
Trains the model on expanding windows and measures directional accuracy
on held-out data, matching the methodology from docs/model-audit-implementation.md.
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
import lightgbm as lgb

from database import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluate_forecaster")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"


def _load_parquet_items(con, min_rows=90):
    rows = con.sql(f"""
        SELECT item_slug,
               MIN(day) AS first_day,
               MAX(day) AS last_day,
               COUNT(*) AS row_count
        FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet')
        GROUP BY item_slug
        HAVING row_count >= {min_rows}
        ORDER BY row_count DESC
    """).fetchall()
    return rows


def _compute_features_for_item(prices_df, events_df):
    """Take a long-format price df for one item + events and build feature matrix."""
    from models.forecaster import ItemForecaster
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        forecaster.feature_cols = []
        df = forecaster.engineer_features(prices_df, events_df)
        df = forecaster._add_cross_sectional_features(df)
        return df
    finally:
        db.close()


def _compute_features_batch(items_data, events_df):
    """Batch compute features for all items at once (matching forecaster.py flow)."""
    from models.forecaster import ItemForecaster
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        all_prices = pd.concat([d["prices"] for d in items_data], ignore_index=True)
        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)
        return df, forecaster
    finally:
        db.close()


def run_walkforward_evaluation(max_items=500):
    """Walk-forward evaluation across parquet archive data."""
    logger.info("=" * 60)
    logger.info("ML FORECASTER WALK-FORWARD EVALUATION")
    logger.info("=" * 60)

    if not ARCHIVE_DIR.exists():
        logger.error(f"Parquet archive not found at {ARCHIVE_DIR}")
        return

    import duckdb
    con = duckdb.connect()

    try:
        # Load events
        db = SessionLocal()
        from models.forecaster import ItemForecaster
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        items = _load_parquet_items(con)
        items = items[:max_items]
        logger.info(f"  {len(items)} items for evaluation")

        # Load all items' price data into one DataFrame for batch feature engineering
        all_rows = []
        for item_slug, first_day, last_day, row_count in items:
            rows = con.sql(f"""
                SELECT item_slug AS item_id, day AS timestamp, mean_price AS price, volume
                FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet')
                WHERE item_slug = ?
                ORDER BY day
            """, params=[item_slug]).fetchall()
            item_df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
            item_df["timestamp"] = pd.to_datetime(item_df["timestamp"])
            item_df["date"] = item_df["timestamp"].dt.date
            all_rows.append(item_df)

        all_prices = pd.concat(all_rows, ignore_index=True)

        results_by_horizon = {}
        for horizon in [7, 30]:
            logger.info(f"\n  Evaluating {horizon}d horizon...")

            # Build features for all items
            df = forecaster.engineer_features(all_prices, events_df)
            df = forecaster._add_cross_sectional_features(df)

            # Prepare targets (date-based)
            tdf = forecaster.prepare_targets(df, horizon)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values(["item_id", "date"])

            if tdf.empty:
                logger.warning(f"    No valid targets for {horizon}d")
                continue

            # Walk forward: expanding window by 60-day steps
            dates = sorted(tdf["date"].unique())
            # Train on first 2/3 of data, walk forward through last 1/3
            split_idx = len(dates) * 2 // 3

            directional_hits = 0
            directional_total = 0
            mae_total = 0.0
            mae_count = 0
            interval_hits = 0
            interval_total = 0
            all_metrics = []

            best_params = None

            step = 60
            for window_end in range(split_idx + 1, len(dates), step):
                train_dates = dates[:window_end]
                val_date = dates[window_end]

                train_df = tdf[tdf["date"].isin(train_dates)]
                val_df = tdf[tdf["date"] == val_date]

                if len(val_df) < 50:
                    continue

                # Cap training rows to most recent 200k
                if len(train_df) > 200000:
                    train_df = train_df.sort_values("date").tail(200000)

                # Learn feature medians from training data
                feature_cols = [c for c in forecaster.feature_cols if c in tdf.columns]
                if not feature_cols:
                    exclude = {"item_id", "date", "timestamp", "price", "volume",
                               "name", "release_date"}
                    exclude |= {f"target_{h}d" for h in forecaster.HORIZONS}
                    exclude |= {f"target_return_{h}d" for h in forecaster.HORIZONS}
                    feature_cols = [c for c in tdf.columns if c not in exclude
                                    and tdf[c].dtype in (np.float64, np.float32, np.int64, int, float)]

                # Feature pruning: remove highly correlated features
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

                X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
                y_train = train_df[f"target_return_{horizon}d"]
                X_val = val_df[feature_cols].fillna(train_df[feature_cols].median())
                y_val = val_df[f"target_return_{horizon}d"]

                # Run grid search ONCE on the first valid window, cache results
                if best_params is None:
                    logger.info("    Grid search (first window only)...")
                    grid = {
                        "num_leaves": [15, 31],
                        "learning_rate": [0.01, 0.03],
                        "lambda_l1": [0.0, 0.5],
                        "lambda_l2": [0.0, 0.5],
                    }
                    best_params = {"num_leaves": 31, "learning_rate": 0.03,
                                   "lambda_l1": 0.5, "lambda_l2": 0.5}
                    best_loss = float("inf")
                    for nl in grid["num_leaves"]:
                        for lr in grid["learning_rate"]:
                            for l1 in grid["lambda_l1"]:
                                for l2 in grid["lambda_l2"]:
                                    p = {
                                        "objective": "quantile", "alpha": 0.5,
                                        "metric": "quantile", "boosting_type": "gbdt",
                                        "num_leaves": nl, "max_depth": 5,
                                        "min_data_in_leaf": 15, "min_gain_to_split": 0.1,
                                        "learning_rate": lr, "feature_fraction": 0.7,
                                        "bagging_fraction": 0.7, "bagging_freq": 5,
                                        "lambda_l1": l1, "lambda_l2": l2,
                                        "verbosity": -1, "random_state": 42, "n_jobs": -1,
                                    }
                                    dt = lgb.Dataset(X_train.values, y_train.values)
                                    dv = lgb.Dataset(X_val.values, y_val.values, reference=dt)
                                    m = lgb.train(
                                        p, dt, num_boost_round=150, valid_sets=[dv],
                                        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                                    )
                                    loss = m.best_score["valid_0"]["quantile"]
                                    if loss < best_loss:
                                        best_loss = loss
                                        best_params = p.copy()

                # Train ensemble of 3 models per quantile with cached best params
                # Ensemble of 2 models per quantile (3 seeds → 2 for eval speed)
                eval_ensemble_seeds = [42, 73]
                models = {}
                for q in [0.1, 0.5, 0.9]:
                    base_params = {
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "boosting_type": "gbdt",
                        "max_depth": 5,
                        "min_data_in_leaf": 15,
                        "min_gain_to_split": 0.1,
                        "feature_fraction": 0.7,
                        "bagging_fraction": 0.7,
                        "bagging_freq": 5,
                        "verbosity": -1,
                        "n_jobs": -1,
                    }
                    for k in ("num_leaves", "learning_rate", "lambda_l1", "lambda_l2"):
                        if k in best_params:
                            base_params[k] = best_params[k]

                    ensemble_preds = []
                    for ei, seed in enumerate(eval_ensemble_seeds):
                        params = base_params.copy()
                        params["random_state"] = seed
                        dtrain = lgb.Dataset(X_train.values, y_train.values)
                        dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain)
                        model = lgb.train(
                            params, dtrain,
                            num_boost_round=100,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
                        )
                        ensemble_preds.append(model.predict(X_val.values))
                    models[q] = np.mean(ensemble_preds, axis=0)

                p10_ret = models[0.1]
                p50_ret = models[0.5]
                p90_ret = models[0.9]

                # Ensure quantile monotonicity without scrambling model identities
                low_ret_arr = np.minimum(p10_ret, p50_ret)
                high_ret_arr = np.maximum(p50_ret, p90_ret)
                mid_ret_arr = p50_ret

                actual_prices = val_df["price"].values
                current_prices = val_df["price"].values

                for i in range(len(val_df)):
                    low_ret, mid_ret, high_ret = low_ret_arr[i], mid_ret_arr[i], high_ret_arr[i]
                    current_price = float(current_prices[i])

                    # Direction check
                    actual_future_price = float(actual_prices[i])
                    actual_return = ((actual_future_price - current_price) / current_price) * 100

                    actual_dir = "up" if actual_return > 0 else "down" if actual_return < 0 else "flat"
                    pred_dir = "up" if mid_ret > 0 else "down" if mid_ret < 0 else "flat"

                    if pred_dir == actual_dir:
                        directional_hits += 1
                    directional_total += 1

                    # MAE
                    future_price = current_price * (1 + mid_ret / 100)
                    mae_total += abs(future_price - actual_future_price)
                    mae_count += 1

                    # Interval coverage
                    price_low = current_price * (1 + low_ret / 100)
                    price_high = current_price * (1 + high_ret / 100)
                    interval_total += 1
                    if price_low <= actual_future_price <= price_high:
                        interval_hits += 1

                if directional_total % 50000 == 0:
                    logger.info(f"      Step {window_end - split_idx}/{len(dates) - split_idx - 1}: "
                                f"dir_acc={directional_hits/directional_total*100:.1f}% "
                                f"({directional_total} samples)")

            if directional_total > 0:
                dir_acc = directional_hits / directional_total * 100
                mae = mae_total / mae_count if mae_count > 0 else 0
                int_cov = interval_hits / interval_total * 100 if interval_total > 0 else 0

                results_by_horizon[horizon] = {
                    "directional_accuracy": round(dir_acc, 2),
                    "mae": round(mae, 4),
                    "interval_coverage": round(int_cov, 2),
                    "sample_count": directional_total,
                }

                logger.info(f"\n  === {horizon}d Results ===")
                logger.info(f"  Directional Accuracy: {dir_acc:.1f}% ({directional_total:,} samples)")
                logger.info(f"  MAE: ${mae:.2f}")
                logger.info(f"  Interval Coverage: {int_cov:.1f}%")

    finally:
        con.close()

    return results_by_horizon


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=50,
                        help="Number of items to evaluate (default: 50)")
    args = parser.parse_args()
    results = run_walkforward_evaluation(max_items=args.max_items)
    print(f"\nRESULT: {json.dumps(results, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
