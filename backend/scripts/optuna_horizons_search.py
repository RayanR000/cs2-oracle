#!/usr/bin/env python3
"""
Run Optuna HP search for 7d/14d/30d horizons with 50 trials each and report
the chosen hyperparameters. Uses Parquet archive data + DB metadata.

Purpose: confirm the 15-trial default isn't hiding local optima at longer
horizons the way it did for 3d. If 50-trial results are consistent with
current 15-trial choices (especially depth), we can close this out.
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
logger = logging.getLogger("optuna_horizons_search")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"
N_TRIALS = 50
HORIZONS = [7, 14, 30]


def main():
    logger.info("=" * 60)
    logger.info(f"OPTUNA SEARCH: horizons {HORIZONS}, {N_TRIALS} trials each")
    logger.info("=" * 60)

    import duckdb
    con = duckdb.connect()
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

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

        all_results = {}
        for horizon in HORIZONS:
            boosting = forecaster.BOOSTING_TYPE
            logger.info(f"\n{'='*60}")
            logger.info(f"HORIZON {horizon}d (boosting={boosting})")
            logger.info(f"{'='*60}")

            tdf = forecaster.prepare_targets(df, horizon)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values(["item_id", "date"])
            logger.info(f"Target rows: {len(tdf):,}")

            feature_cols = [c for c in forecaster.feature_cols if c in tdf.columns]
            if not feature_cols:
                exclude = {"item_id", "date", "timestamp", "price", "volume",
                           "name", "release_date"}
                exclude |= {f"target_{h}d" for h in forecaster.HORIZONS}
                exclude |= {f"target_return_{h}d" for h in forecaster.HORIZONS}
                feature_cols = [c for c in tdf.columns if c not in exclude
                                and tdf[c].dtype in (np.float64, np.float32, np.int64, int, float)]

            max_date = tdf["date"].max()
            split_date = max_date - timedelta(days=90)
            train_set = tdf[tdf["date"] <= split_date]
            val_set = tdf[tdf["date"] > split_date]

            if len(train_set) > 200000:
                train_set = train_set.sort_values("date").tail(200000)
            X_train = train_set[feature_cols].fillna(train_set[feature_cols].median())
            y_train = train_set[f"target_return_{horizon}d"]
            X_val = val_set[feature_cols].fillna(train_set[feature_cols].median())
            y_val = val_set[f"target_return_{horizon}d"]

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
                logger.info(f"After correlation pruning: {len(feature_cols)} features")

            logger.info(f"Train: {len(X_train):,} rows, Val: {len(X_val):,} rows, "
                        f"Features: {len(feature_cols)}")

            horizon_results = {}
            for q in [0.1, 0.5, 0.9]:
                logger.info(f"\n  Searching p{int(q*100)} ({N_TRIALS} trials)...")
                best_params = forecaster._optuna_search_params(
                    X_train, y_train, X_val, y_val,
                    val_dates=val_set["date"],
                    quantile=q, boosting_type=boosting, n_trials=N_TRIALS,
                )
                logger.info(f"  Best params p{int(q*100)}: {best_params}")
                horizon_results[f"p{int(q*100)}"] = best_params

            all_results[str(horizon)] = horizon_results

            print(f"\n  --- {horizon}d RESULTS ({N_TRIALS} trials) ---")
            for q_label, params in horizon_results.items():
                print(f"    {q_label}:")
                for k, v in params.items():
                    print(f"      {k}: {v}")

        out = {"horizons": HORIZONS, "n_trials": N_TRIALS, "results": all_results}
        out_path = Path(__file__).parent.parent / "optuna_horizons_result.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"\nSaved to {out_path}")

        print(f"\n{'='*70}")
        print(f"OPTUNA SEARCH RESULTS: horizons {HORIZONS}, {N_TRIALS} trials")
        print(f"{'='*70}")
        for h_label, h_res in all_results.items():
            print(f"\n  Horizon {h_label}d:")
            for q_label, params in h_res.items():
                print(f"    {q_label}:")
                for k, v in params.items():
                    print(f"      {k}: {v}")

        return all_results
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
