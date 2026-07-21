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
import math
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
    dfs = []
    for item_slug, *_ in items:
        rows = con.sql("""
            SELECT item_slug AS item_id, CAST(day AS DATE) AS timestamp,
                   mean_price AS price, volume
            FROM read_parquet('""" + str(ARCHIVE_DIR) + """/prices-*.parquet')
            WHERE item_slug = ?
            ORDER BY day
        """, params=[item_slug]).fetchall()
        df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _compute_metrics(y_true, y_low, y_mid, y_high, current_prices):
    n = len(y_true)
    if n == 0:
        return None

    abs_errors = [abs(y_mid[i] - y_true[i]) for i in range(n)]
    pct_errors = [abs(abs_errors[i] / y_true[i]) * 100 if y_true[i] > 0 else 0 for i in range(n)]

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in abs_errors) / n)
    mape = sum(pct_errors) / n

    total_actual = sum(y_true)
    wmape = (sum(abs_errors) / total_actual * 100) if total_actual > 0 else 0

    dir_hits = 0
    for i in range(n):
        actual_dir = "up" if y_true[i] > current_prices[i] else "down" if y_true[i] < current_prices[i] else "flat"
        pred_dir = "up" if y_mid[i] > current_prices[i] else "down" if y_mid[i] < current_prices[i] else "flat"
        if pred_dir == actual_dir:
            dir_hits += 1
    dir_acc = dir_hits / n * 100

    int_hits = sum(1 for i in range(n) if y_low[i] is not None and y_high[i] is not None and y_low[i] <= y_true[i] <= y_high[i])
    int_cov = int_hits / n * 100

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "wmape": round(wmape, 2),
        "directional_accuracy": round(dir_acc, 2),
        "interval_coverage": round(int_cov, 2),
        "sample_count": n,
    }


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


def run_walkforward(max_items=500, horizons=None, skip_db=False):
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

            fold_results = []

            for window_end in range(split_idx + 1, len(dates), STEP_DAYS):
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

                X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
                y_train = train_df[f"target_return_{horizon}d"]
                X_val = val_df[feature_cols].fillna(train_df[feature_cols].median())
                y_val = val_df[f"target_return_{horizon}d"]

                preds = {}
                for q in QUANTILES:
                    params = _get_tuned_params(meta, horizon, q)
                    params.update({
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "verbosity": -1,
                        "random_state": 42,
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

                if len(preds) == 3:
                    low, high = ItemForecaster._fix_quantile_crossing(
                        preds[0.1], preds[0.5], preds[0.9]
                    )

                    current_prices = val_df["price"].values
                    actual_returns = y_val.values
                    actual_prices = current_prices * (1 + actual_returns / 100)
                    mid_prices = current_prices * (1 + preds[0.5] / 100)
                    low_prices = current_prices * (1 + low / 100)
                    high_prices = current_prices * (1 + high / 100)

                    metrics = _compute_metrics(actual_prices, low_prices, mid_prices, high_prices, current_prices)
                    if metrics:
                        fold_results.append(metrics)

            if not fold_results:
                logger.warning(f"    No folds completed for {horizon}d")
                continue

            n_folds = len(fold_results)
            total_n = sum(f["sample_count"] for f in fold_results)
            agg = {
                "mae": sum(f["mae"] * f["sample_count"] for f in fold_results) / total_n,
                "rmse": sum(f["rmse"] * f["sample_count"] for f in fold_results) / total_n,
                "mape": sum(f["mape"] * f["sample_count"] for f in fold_results) / total_n,
                "wmape": sum(f["wmape"] * f["sample_count"] for f in fold_results) / total_n,
                "directional_accuracy": sum(f["directional_accuracy"] * f["sample_count"] for f in fold_results) / total_n,
                "interval_coverage": sum(f["interval_coverage"] * f["sample_count"] for f in fold_results) / total_n,
                "sample_count": total_n,
                "fold_count": n_folds,
            }
            results_by_horizon[horizon] = agg

            logger.info(f"    {horizon}d: {n_folds} folds, {total_n} samples")
            logger.info(f"      DirAcc={agg['directional_accuracy']:.1f}%  MAE=${agg['mae']:.2f}  MAPE={agg['mape']:.1f}%  IntCov={agg['interval_coverage']:.1f}%")

        total_elapsed = time.time() - total_start
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtest complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        logger.info(f"{'='*60}")

        if not skip_db:
            today = date.today()
            for horizon, metrics in results_by_horizon.items():
                row = {
                    "prediction_type": "walkforward_backtest",
                    "evaluation_date": today,
                    "horizon_days": horizon,
                    "model_version": "lgbm-v3-tuned",
                    "evaluation_window_days": None,
                    "sample_count": metrics["sample_count"],
                    "metrics": {k: metrics[k] for k in ["mae", "rmse", "mape", "wmape", "directional_accuracy", "interval_coverage", "fold_count"]},
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                }
                _upsert_accuracy(db, [row])

        con.close()
        try:
            db.close()
        except Exception:
            pass

        report = {
            "test_date": str(date.today()),
            "total_items": len(items),
            "total_elapsed_seconds": round(total_elapsed, 1),
            "horizons": {str(h): m for h, m in results_by_horizon.items()},
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
    args = parser.parse_args()

    report = run_walkforward(max_items=args.max_items, horizons=args.horizons, skip_db=args.skip_db)
    print(f"\nRESULT: {json.dumps(report, indent=2, default=str)}")
    return 0 if report.get("status") != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
