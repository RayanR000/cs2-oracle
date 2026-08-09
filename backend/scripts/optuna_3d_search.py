#!/usr/bin/env python3
"""
Run Optuna HP search for the 3d horizon with a proper trial budget (50 trials)
and report the chosen hyperparameters. Uses Parquet archive data + DB metadata
(matches the forecaster.py train() flow).

The goal: see whether Optuna converges to depth >= 5 when given enough trials,
or whether it still picks shallow params (confirming the signal-ceiling hypothesis).
"""

import sys
import json
import logging
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from database import SessionLocal
from models.forecaster import ItemForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optuna_3d_search")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"
HORIZON = 3
N_TRIALS = 50


def main():
    logger.info("=" * 60)
    logger.info(f"OPTUNA SEARCH: 3d horizon, {N_TRIALS} trials")
    logger.info("=" * 60)

    # 1. Load data via ItemForecaster (same flow as train())
    import duckdb
    con = duckdb.connect()
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        # Load ~200 backfilled items from Parquet
        items = con.sql(f"""
            SELECT item_slug,
                   MIN(day) AS first_day,
                   MAX(day) AS last_day,
                   COUNT(*) AS row_count
            FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet')
            GROUP BY item_slug
            HAVING row_count >= 90
            ORDER BY row_count DESC
            LIMIT 200
        """).fetchall()
        logger.info(f"Loaded {len(items)} items")

        all_rows = []
        for item_slug, *_ in items:
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
        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)
        logger.info(f"Feature matrix: {df.shape[0]} rows, {len(forecaster.feature_cols)} features")

        # 2. Build 3d targets
        tdf = forecaster.prepare_targets(df, HORIZON)
        tdf = tdf.dropna(subset=[f"target_return_{HORIZON}d"]).copy()
        tdf = tdf.sort_values(["item_id", "date"])
        logger.info(f"Target rows: {len(tdf):,}")

        # 3. Infer feature columns (engineer_features doesn't set self.feature_cols;
        #    same fallback logic as evaluate_forecaster.py)
        feature_cols = [c for c in forecaster.feature_cols if c in tdf.columns]
        if not feature_cols:
            exclude = {"item_id", "date", "timestamp", "price", "volume",
                       "name", "release_date"}
            exclude |= {f"target_{h}d" for h in forecaster.HORIZONS}
            exclude |= {f"target_return_{h}d" for h in forecaster.HORIZONS}
            feature_cols = [c for c in tdf.columns if c not in exclude
                            and tdf[c].dtype in (np.float64, np.float32, np.int64, int, float)]
        logger.info(f"Using {len(feature_cols)} features")

        # 4. Train/val split: hold out last 60 days for validation (wider than
        #    the 21-day default to give Optuna more signal to search on).
        max_date = tdf["date"].max()
        split_date = max_date - timedelta(days=60)
        train_set = tdf[tdf["date"] <= split_date]
        val_set = tdf[tdf["date"] > split_date]
        logger.info(f"Train: {len(train_set):,} rows, Val: {len(val_set):,} rows")

        # Cap training rows
        if len(train_set) > 200000:
            train_set = train_set.sort_values("date").tail(200000)
        X_train = train_set[feature_cols].fillna(train_set[feature_cols].median())
        y_train = train_set[f"target_return_{HORIZON}d"]
        X_val = val_set[feature_cols].fillna(train_set[feature_cols].median())
        y_val = val_set[f"target_return_{HORIZON}d"]

        # Also prune highly correlated features (same as forecaster.py train())
        if len(feature_cols) > 2:
            corr = train_set[feature_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                if col in to_drop:
                    continue
                highly_corr = upper[col][upper[col] > 0.95].index
                to_drop.update(highly_corr)
            feature_cols = [c for c in feature_cols if c not in to_drop]
            logger.info(f"After pruning: {len(feature_cols)} features")

        # 5. Run Optuna search for each quantile
        results = {}
        for q in [0.1, 0.5, 0.9]:
            logger.info(f"\n  Searching p{int(q*100)} ({N_TRIALS} trials)...")
            best_params = forecaster._optuna_search_params(
                X_train, y_train, X_val, y_val,
                val_dates=val_set["date"],
                quantile=q, boosting_type="gbdt", n_trials=N_TRIALS,
            )
            logger.info(f"  Best params p{int(q*100)}: {best_params}")
            results[f"p{int(q*100)}"] = best_params

        # 5. Report
        print(f"\n{'='*70}")
        print(f"OPTUNA SEARCH RESULTS: 3d horizon, {N_TRIALS} trials")
        print(f"{'='*70}")
        for q_label, params in results.items():
            print(f"\n  {q_label}:")
            for k, v in params.items():
                print(f"    {k}: {v}")

        out = {"horizon": HORIZON, "n_trials": N_TRIALS, "results": results}
        out_path = Path(__file__).parent.parent / "optuna_3d_result.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"\nSaved to {out_path}")

        return results
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
