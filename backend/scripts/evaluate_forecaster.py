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


def _load_parquet_items(con, min_rows=90, backfilled_only=False):
    where_clause = ""
    if backfilled_only:
        where_clause = f"""
            WHERE item_slug IN (
                SELECT DISTINCT item_slug
                FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet')
                WHERE source = 'STEAMCOMMUNITY'
            )
        """
    rows = con.sql(f"""
        SELECT item_slug,
               MIN(day) AS first_day,
               MAX(day) AS last_day,
               COUNT(*) AS row_count
        FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet')
        {where_clause}
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

        items = _load_parquet_items(con, backfilled_only=True)
        items = items[:max_items]
        logger.info(f"  {len(items)} items for evaluation (backfilled only)")

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
        for horizon in ItemForecaster.HORIZONS:
            logger.info(f"\n  Evaluating {horizon}d horizon...")

            # Build features for all items (must match build_training_data order)
            df = forecaster.engineer_features(all_prices, events_df)
            df = forecaster._add_weapon_type_cross_sectional_features(df)
            df = forecaster._add_cross_sectional_features(df)
            df = forecaster._add_player_count_features(df)

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
            per_fold = []  # list of dicts: {fold_id, dir_acc, mae, int_cov, n}

            VAL_WINDOW_DAYS = 21
            step = 60
            for window_end in range(split_idx + 1, len(dates), step):
                train_dates = dates[:window_end]
                val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue

                train_df = tdf[tdf["date"].isin(train_dates)]
                val_df = tdf[tdf["date"].isin(val_dates)]

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

                # Use fixed tuned params (same defaults as forecaster.py train())
                # No grid search in eval mode — this is a measurement run.
                models = {}
                for q in [0.1, 0.5, 0.9]:
                    params = {
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "boosting_type": "gbdt",
                        "num_leaves": 31,
                        "max_depth": 5,
                        "min_data_in_leaf": 15,
                        "min_gain_to_split": 0.1,
                        "learning_rate": 0.03,
                        "feature_fraction": 0.7,
                        "bagging_fraction": 0.7,
                        "bagging_freq": 5,
                        "lambda_l1": 0.5,
                        "lambda_l2": 0.5,
                        "verbosity": -1,
                        "random_state": 42,
                        "n_jobs": -1,
                    }
                    dtrain = lgb.Dataset(X_train.values, y_train.values)
                    dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain)
                    model = lgb.train(
                        params, dtrain,
                        num_boost_round=100,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)]
                    )
                    models[q] = model.predict(X_val.values)

                p10_ret = models[0.1]
                p50_ret = models[0.5]
                p90_ret = models[0.9]

                # Ensure quantile monotonicity without scrambling model identities.
                # For crossing items, impute the average half-width from well-behaved items.
                crossing_mask_eval = (p10_ret > p50_ret) | (p50_ret > p90_ret)
                non_crossing_eval = ~crossing_mask_eval
                low_ret_arr = np.minimum(p10_ret, p50_ret)
                high_ret_arr = np.maximum(p50_ret, p90_ret)
                if non_crossing_eval.any():
                    avg_hw = np.mean([
                        np.mean(p50_ret[non_crossing_eval] - p10_ret[non_crossing_eval]),
                        np.mean(p90_ret[non_crossing_eval] - p50_ret[non_crossing_eval]),
                    ])
                    if avg_hw > 0:
                        low_ret_arr[crossing_mask_eval] = p50_ret[crossing_mask_eval] - avg_hw
                        high_ret_arr[crossing_mask_eval] = p50_ret[crossing_mask_eval] + avg_hw
                low_ret_arr = np.minimum(low_ret_arr, p50_ret)
                high_ret_arr = np.maximum(high_ret_arr, p50_ret)
                mid_ret_arr = p50_ret

                current_prices = val_df["price"].values
                actual_returns = y_val.values

                fold_hits = 0
                fold_total = 0
                fold_mae = 0.0
                fold_int_hits = 0
                fold_int_total = 0
                for i in range(len(val_df)):
                    low_ret, mid_ret, high_ret = low_ret_arr[i], mid_ret_arr[i], high_ret_arr[i]
                    current_price = float(current_prices[i])
                    actual_return = float(actual_returns[i])

                    # Future price for MAE / interval coverage
                    actual_future_price = current_price * (1 + actual_return / 100)

                    actual_dir = "up" if actual_return > 0 else "down" if actual_return < 0 else "flat"
                    pred_dir = "up" if mid_ret > 0 else "down" if mid_ret < 0 else "flat"

                    if pred_dir == actual_dir:
                        directional_hits += 1
                        fold_hits += 1
                    directional_total += 1
                    fold_total += 1

                    # MAE
                    future_price = current_price * (1 + mid_ret / 100)
                    abs_err = abs(future_price - actual_future_price)
                    mae_total += abs_err
                    fold_mae += abs_err
                    mae_count += 1

                    # Interval coverage
                    price_low = current_price * (1 + low_ret / 100)
                    price_high = current_price * (1 + high_ret / 100)
                    fold_int_total += 1
                    interval_total += 1
                    if price_low <= actual_future_price <= price_high:
                        interval_hits += 1
                        fold_int_hits += 1

                per_fold.append({
                    "fold": len(per_fold) + 1,
                    "val_start": str(val_dates[0]),
                    "val_end": str(val_dates[-1]),
                    "dir_acc": round(fold_hits / fold_total * 100, 1) if fold_total > 0 else 0,
                    "mae": round(fold_mae / fold_total, 4) if fold_total > 0 else 0,
                    "int_cov": round(fold_int_hits / fold_int_total * 100, 1) if fold_int_total > 0 else 0,
                    "n": fold_total,
                })

                if directional_total % 50000 == 0:
                    logger.info(f"      Step {window_end - split_idx}/{len(dates) - split_idx - 1}: "
                                f"dir_acc={directional_hits/directional_total*100:.1f}% "
                                f"({directional_total} samples)")

            if directional_total > 0:
                dir_acc = directional_hits / directional_total * 100
                mae = mae_total / mae_count if mae_count > 0 else 0
                int_cov = interval_hits / interval_total * 100 if interval_total > 0 else 0

                # Per-fold stats
                fold_accs = [f["dir_acc"] for f in per_fold]
                baseline_2class = 50.0
                result = {
                    "directional_accuracy": round(dir_acc, 2),
                    "mae": round(mae, 4),
                    "interval_coverage": round(int_cov, 2),
                    "sample_count": directional_total,
                    "effective_baseline": baseline_2class,
                    "fold_count": len(per_fold),
                    "fold_mean_dir_acc": round(np.mean(fold_accs), 1) if fold_accs else 0,
                    "fold_std_dir_acc": round(np.std(fold_accs), 1) if len(fold_accs) > 1 else 0,
                    "fold_min_dir_acc": round(min(fold_accs), 1) if fold_accs else 0,
                    "fold_max_dir_acc": round(max(fold_accs), 1) if fold_accs else 0,
                    "per_fold": per_fold,
                }
                # Improve-over-baseline (2-class, since "flat" is essentially never the actual direction)
                result["improvement_over_baseline_pp"] = round(dir_acc - baseline_2class, 1)

                results_by_horizon[horizon] = result

                logger.info(f"\n  === {horizon}d Results ===")
                logger.info(f"  Directional Accuracy: {dir_acc:.1f}% ({directional_total:,} samples)")
                logger.info(f"  Effective baseline: {baseline_2class:.0f}% (2-class: flat is never actual)")
                logger.info(f"  Improvement: {result['improvement_over_baseline_pp']:.1f}pp")
                logger.info(f"  Per-fold: mean={result['fold_mean_dir_acc']:.1f}% "
                            f"sd={result['fold_std_dir_acc']:.1f}% "
                            f"range=[{result['fold_min_dir_acc']:.1f}%, {result['fold_max_dir_acc']:.1f}%]")
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
