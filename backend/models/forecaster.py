"""
LightGBM-based price forecaster for CS2 items.
Trains quantile regression models for 7d and 30d horizons,
using price history, technical indicators, events, and item metadata.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sqlalchemy import text

logger = logging.getLogger(__name__)


class ItemForecaster:
    HORIZONS = [7, 30]
    QUANTILES = [0.1, 0.5, 0.9]
    MIN_HISTORY_DAYS = 30
    TRAIN_SPLIT_DATE = "2026-06-15"

    def __init__(self, db_session, model_dir: str = None):
        self.db = db_session
        self.model_dir = model_dir or str(Path(__file__).parent / "saved_models")
        self.models: Dict[Tuple[int, float], lgb.Booster] = {}
        self.feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _now(self):
        return datetime.now(timezone.utc)

    def fetch_price_history(self, days_back: int = 365) -> pd.DataFrame:
        logger.info(f"Fetching price history (last {days_back}d)...")
        cutoff = self._now() - timedelta(days=days_back)
        rows = self.db.execute(text("""
            SELECT item_id, timestamp, price, volume
            FROM price_history
            WHERE timestamp >= :cutoff
              AND source NOT LIKE 'synthetic_demo'
              AND source NOT LIKE 'historical_fallback:%'
            ORDER BY item_id, timestamp
        """), {"cutoff": cutoff}).fetchall()
        df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        logger.info(f"  {len(df):,} rows, {df.item_id.nunique():,} items")
        return df

    def fetch_daily_analysis(self, days_back: int = 30) -> pd.DataFrame:
        cutoff = (self._now() - timedelta(days=days_back)).date()
        rows = self.db.execute(text("""
            SELECT item_id, analysis_date, current_price,
                   ma_7day, ma_30day, ma_90day,
                   momentum_7day, momentum_30day, volatility,
                   trend_direction, momentum_score, opportunity_score,
                   trading_volume_trend, price_stability
            FROM daily_analysis
            WHERE analysis_date >= :cutoff
            ORDER BY item_id, analysis_date
        """), {"cutoff": cutoff}).fetchall()
        df = pd.DataFrame(rows, columns=[
            "item_id", "analysis_date", "current_price",
            "ma_7day", "ma_30day", "ma_90day",
            "momentum_7day", "momentum_30day", "volatility",
            "trend_direction", "momentum_score", "opportunity_score",
            "trading_volume_trend", "price_stability"
        ])
        df["analysis_date"] = pd.to_datetime(df["analysis_date"])
        logger.info(f"  daily_analysis: {len(df):,} rows")
        return df

    def fetch_events(self) -> pd.DataFrame:
        rows = self.db.execute(text("""
            SELECT id, type, timestamp, description
            FROM events
            ORDER BY timestamp
        """)).fetchall()
        df = pd.DataFrame(rows, columns=["id", "type", "timestamp", "description"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        logger.info(f"  events: {len(df)}")
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering price features...")
        df = df.sort_values(["item_id", "date"]).copy()

        # Sort within each item group
        grouped = df.groupby("item_id")

        # Lag prices
        for lag in [1, 3, 7, 14, 30]:
            df[f"price_lag_{lag}d"] = grouped["price"].shift(lag)

        # Returns
        for lag in [1, 3, 7, 14, 30]:
            col = f"price_lag_{lag}d"
            df[f"return_{lag}d"] = (df["price"] - df[col]) / df[col].replace(0, np.nan) * 100

        # Rolling statistics
        for window in [7, 14, 30]:
            df[f"price_mean_{window}d"] = grouped["price"].transform(
                lambda x: x.rolling(window, min_periods=3).mean()
            )
            df[f"price_std_{window}d"] = grouped["price"].transform(
                lambda x: x.rolling(window, min_periods=3).std()
            )
            df[f"price_min_{window}d"] = grouped["price"].transform(
                lambda x: x.rolling(window, min_periods=3).min()
            )
            df[f"price_max_{window}d"] = grouped["price"].transform(
                lambda x: x.rolling(window, min_periods=3).max()
            )

        # Z-score vs 30d rolling
        mean_30 = df["price_mean_30d"]
        std_30 = df["price_std_30d"].replace(0, np.nan)
        df["price_zscore_30d"] = (df["price"] - mean_30) / std_30

        # Volume features
        if "volume" in df.columns and df["volume"].notna().any():
            df["volume_lag_1d"] = grouped["volume"].shift(1)
            df["volume_lag_7d"] = grouped["volume"].shift(7)
            df["volume_mean_7d"] = grouped["volume"].transform(
                lambda x: x.rolling(7, min_periods=2).mean()
            )
            df["volume_change_1d"] = df["volume"] / df["volume_lag_1d"].replace(0, np.nan)
            df["volume_change_7d"] = df["volume"] / df["volume_lag_7d"].replace(0, np.nan)
        else:
            for col in ["volume_lag_1d", "volume_lag_7d", "volume_mean_7d",
                        "volume_change_1d", "volume_change_7d"]:
                df[col] = np.nan

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(df["date"])
        df["day_of_week"] = dates.dt.dayofweek
        df["month"] = dates.dt.month
        df["quarter"] = dates.dt.quarter
        df["day_of_year"] = dates.dt.dayofyear
        df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        return df

    def _add_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        if events_df.empty:
            df["days_since_last_event"] = 999
            df["events_next_30d"] = 0
            return df

        for event_type in ["major", "operation", "case_drop", "update"]:
            type_events = events_df[events_df["type"] == event_type].sort_values("date")

            if type_events.empty:
                df[f"days_since_{event_type}"] = 999
                df[f"events_next_30d_{event_type}"] = 0
                continue

            # Map: for each item date, find days since last event
            dates = pd.to_datetime(df["date"])
            event_dates = pd.to_datetime(type_events["date"].unique())

            # Compute days since last event
            sorted_events = np.sort(event_dates)
            all_dates = dates.values
            indices = np.searchsorted(sorted_events, all_dates) - 1

            # Handle items before first event
            valid = indices >= 0
            days_since = np.full(len(dates), 999, dtype=float)
            if valid.any():
                last_event_dates = sorted_events[indices[valid]]
                days_since[valid] = (all_dates[valid] - last_event_dates).astype('timedelta64[D]').astype(float)
            df[f"days_since_{event_type}"] = np.clip(days_since, 0, 999)

            # Count events in next 30 days (vectorized)
            left = np.searchsorted(sorted_events, all_dates, side="right")
            right = np.searchsorted(sorted_events, all_dates + np.timedelta64(30, "D"), side="right")
            df[f"events_next_30d_{event_type}"] = right - left

        return df

    def engineer_features(self, price_df: pd.DataFrame,
                          events_df: pd.DataFrame) -> pd.DataFrame:
        # Resample to one row per item per day before feature engineering.
        # Raw price_history has multiple rows per day (collection runs every 6h).
        # Without resampling, "lag_1d" is really ~6h and "mean_7d" covers ~2 days.
        if "date" in price_df.columns:
            daily = price_df.groupby(["item_id", "date"], as_index=False).agg(
                price=("price", "mean"),
                volume=("volume", "sum"),
            )
        else:
            daily = price_df
        df = self._compute_price_features(daily)
        df = self._add_temporal_features(df)
        df = self._add_event_features(df, events_df)
        return df

    # ------------------------------------------------------------------
    # Target preparation
    # ------------------------------------------------------------------

    def prepare_targets(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        logger.info(f"Preparing {horizon}d targets...")
        df = df.sort_values(["item_id", "date"]).copy()
        df[f"target_{horizon}d"] = df.groupby("item_id")["price"].transform(
            lambda x: x.shift(-horizon)
        )
        df[f"target_return_{horizon}d"] = (
            (df[f"target_{horizon}d"] - df["price"]) / df["price"].replace(0, np.nan) * 100
        )
        return df

    # ------------------------------------------------------------------
    # Build training dataset
    # ------------------------------------------------------------------

    def build_training_data(self, days_back: int = 365) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        price_df = self.fetch_price_history(days_back=days_back)
        events_df = self.fetch_events()
        da_df = self.fetch_daily_analysis(days_back=30)

        df = self.engineer_features(price_df, events_df)

        # Merge daily_analysis features
        if not da_df.empty:
            da_df["analysis_date"] = pd.to_datetime(da_df["analysis_date"])
            da_df.rename(columns={"current_price": "da_price"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.merge(da_df, left_on=["item_id", "date"],
                          right_on=["item_id", "analysis_date"], how="left")
        else:
            for col in ["ma_7day", "ma_30day", "ma_90day", "momentum_7day",
                        "momentum_30day", "volatility", "trend_direction",
                        "momentum_score", "opportunity_score",
                        "trading_volume_trend", "price_stability"]:
                df[col] = np.nan

        # One-hot trend_direction
        if "trend_direction" in df.columns:
            dummies = pd.get_dummies(df["trend_direction"], prefix="trend")
            for col in ["trend_up", "trend_down", "trend_flat"]:
                if col not in dummies.columns:
                    dummies[col] = 0
            df = pd.concat([df.drop(columns=["trend_direction"]), dummies], axis=1)

        # Define feature columns (exclude leakage-prone and target columns)
        exclude = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date", "analysis_date", "da_price"}
        exclude |= {f"target_{h}d" for h in self.HORIZONS}
        exclude |= {f"target_return_{h}d" for h in self.HORIZONS}

        self.feature_cols = [c for c in df.columns if c not in exclude
                             and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]

        # Prepare targets for each horizon
        targets = {}
        for h in self.HORIZONS:
            tdf = self.prepare_targets(df, h)
            targets[h] = tdf

        logger.info(f"Feature matrix: {len(df):,} rows, {len(self.feature_cols)} features")
        logger.info(f"Features: {self.feature_cols}")

        return df, targets

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _get_feature_importance(self, model: lgb.Booster) -> pd.DataFrame:
        importance = model.feature_importance(importance_type="gain")
        fi = pd.DataFrame({"feature": self.feature_cols, "importance": importance})
        fi = fi.sort_values("importance", ascending=False).head(20)
        return fi

    def _sample_training_data(self, targets: pd.DataFrame,
                               horizon: int, max_rows: int = 200_000) -> pd.DataFrame:
        """Sample training data, prioritizing recent data."""
        train = targets.dropna(subset=[f"target_{horizon}d"]).copy()
        before_split = train[pd.to_datetime(train["date"]) < self.TRAIN_SPLIT_DATE]
        after_split = train[pd.to_datetime(train["date"]) >= self.TRAIN_SPLIT_DATE]

        sampled = []
        if len(after_split) > 0:
            sampled.append(after_split)
        remaining = max_rows - len(after_split)
        if remaining > 0 and len(before_split) > 0:
            sampled.append(before_split.sample(min(len(before_split), remaining), random_state=42))

        result = pd.concat(sampled, ignore_index=True) if sampled else train
        logger.info(f"  Training samples for {horizon}d: {len(result):,}")
        return result

    def train(self, max_rows: int = 200_000):
        logger.info("=" * 60)
        logger.info("TRAINING LIGHTGBM FORECASTER")
        logger.info("=" * 60)

        df, targets_by_horizon = self.build_training_data(days_back=365)

        for horizon in self.HORIZONS:
            tdf = targets_by_horizon[horizon]

            for q in self.QUANTILES:
                logger.info(f"Training {horizon}d p{int(q*100)} model...")
                train_df = self._sample_training_data(tdf, horizon, max_rows)

                train_df = train_df.sort_values("date")
                train_dates = pd.to_datetime(train_df["date"])
                split_idx = int(len(train_df) * 0.8)

                train_set = train_df.iloc[:split_idx]
                val_set = train_df.iloc[split_idx:]

                X_train = train_set[self.feature_cols].fillna(0)
                y_train = train_set[f"target_{horizon}d"]
                X_val = val_set[self.feature_cols].fillna(0)
                y_val = val_set[f"target_{horizon}d"]

                params = {
                    "objective": "quantile",
                    "alpha": q,
                    "metric": "quantile",
                    "boosting_type": "gbdt",
                    "num_leaves": 63,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbosity": -1,
                    "random_state": 42,
                    "n_jobs": -1,
                }

                dtrain = lgb.Dataset(X_train, y_train)
                dval = lgb.Dataset(X_val, y_val, reference=dtrain)
                model = lgb.train(
                    params, dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )

                self.models[(horizon, q)] = model
                fi = self._get_feature_importance(model)
                logger.info(f"  Top features: {fi['feature'].head(5).tolist()}")

        self.save_models()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, item_ids: List[int] = None) -> pd.DataFrame:
        logger.info("Generating forecasts...")

        price_df = self.fetch_price_history(days_back=90)
        events_df = self.fetch_events()

        df = self.engineer_features(price_df, events_df)

        # Merge daily_analysis
        da_df = self.fetch_daily_analysis(days_back=3)
        if not da_df.empty:
            da_df["analysis_date"] = pd.to_datetime(da_df["analysis_date"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.merge(da_df, left_on=["item_id", "date"],
                          right_on=["item_id", "analysis_date"], how="left")

        if "trend_direction" in df.columns:
            dummies = pd.get_dummies(df["trend_direction"], prefix="trend")
            for col in ["trend_up", "trend_down", "trend_flat"]:
                if col not in dummies.columns:
                    dummies[col] = 0
            df = pd.concat([df.drop(columns=["trend_direction"]), dummies], axis=1)

        # Align features with training columns (add missing, drop extras)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_cols + [c for c in df.columns if c not in self.feature_cols]]

        # Get latest feature row per item
        df = df.sort_values(["item_id", "date"])
        latest_rows = df.groupby("item_id").last().reset_index()

        if item_ids:
            latest_rows = latest_rows[latest_rows["item_id"].isin(item_ids)]

        X_batch = latest_rows[self.feature_cols].fillna(0)

        results = []
        for horizon in self.HORIZONS:
            preds = {}
            for q in self.QUANTILES:
                key = (horizon, q)
                if key in self.models:
                    preds[q] = self.models[key].predict(X_batch)

            if len(preds) == 3:
                lows = np.round(preds[0.1], 2)
                mids = np.round(preds[0.5], 2)
                highs = np.round(preds[0.9], 2)

                for i, (_, row) in enumerate(latest_rows.iterrows()):
                    item_id = row["item_id"]
                    current_price = row["price"]
                    prices = sorted([float(lows[i]), float(mids[i]), float(highs[i])])

                    results.append({
                        "item_id": item_id,
                        "current_price": current_price,
                        "forecasts": {
                            horizon: {
                                "low": prices[0],
                                "mid": prices[1],
                                "high": prices[2],
                                "direction": "up" if prices[1] > current_price else "down" if prices[1] < current_price else "flat",
                                "confidence": self._compute_confidence(prices[1], prices[0], prices[2], current_price),
                            }
                        },
                        "generated_at": self._now(),
                    })

        # Aggregate results: one row per item with all horizons
        agg = {}
        for r in results:
            iid = r["item_id"]
            if iid not in agg:
                agg[iid] = {
                    "item_id": iid,
                    "current_price": r["current_price"],
                    "forecasts": {},
                    "generated_at": r["generated_at"],
                }
            agg[iid]["forecasts"].update(r["forecasts"])

        result_df = pd.DataFrame(list(agg.values()))
        logger.info(f"  Forecasts generated for {len(result_df)} items")
        return result_df

    def predict_single(self, item_id: int) -> Dict[str, Any]:
        results = self.predict(item_ids=[item_id])
        if results.empty:
            return {}
        return results.iloc[0].to_dict()

    @staticmethod
    def _compute_confidence(mid: float, low: float, high: float, current: float) -> str:
        if mid == 0 or current == 0:
            return "low"
        range_pct = (high - low) / mid
        change_pct = abs(mid - current) / current
        if range_pct < 0.1 and change_pct > 0.03:
            return "high"
        elif range_pct < 0.2:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        for (horizon, q), model in self.models.items():
            path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}.txt")
            model.save_model(path)

        # Save feature columns
        meta = {"feature_cols": self.feature_cols, "trained_at": str(self._now())}
        with open(os.path.join(self.model_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        logger.info(f"Models saved to {self.model_dir}")

    def load_models(self):
        meta_path = os.path.join(self.model_dir, "meta.json")
        if not os.path.exists(meta_path):
            logger.warning(f"No saved models found in {self.model_dir}")
            return False

        with open(meta_path) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]

        for horizon in self.HORIZONS:
            for q in self.QUANTILES:
                path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}.txt")
                if os.path.exists(path):
                    self.models[(horizon, q)] = lgb.Booster(model_file=path)

        logger.info(f"Loaded {len(self.models)} models from {self.model_dir}")
        return len(self.models) > 0

    def has_models(self) -> bool:
        return len(self.models) > 0
