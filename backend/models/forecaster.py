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
    # Prediction eligibility is looser than training: the live aggregator
    # series is still young, and 14 daily points is enough for the lag/rolling
    # features to be non-degenerate.
    PREDICT_MIN_HISTORY_DAYS = 14
    # Walk-forward validation split: most recent N days are held out.
    # A relative split stays valid as data accumulates (a fixed date would
    # eventually leave the validation set covering all new data).
    VALIDATION_WINDOW_DAYS = 21
    N_ENSEMBLES = 3
    ENSEMBLE_SEEDS = [42, 73, 91]
    PRUNE_CORRELATION_THRESHOLD = 0.95

    def __init__(self, db_session, model_dir: str = None):
        self.db = db_session
        self.model_dir = model_dir or str(Path(__file__).parent / "saved_models")
        self.models: Dict[Tuple[int, float], lgb.Booster] = {}
        self.feature_cols: List[str] = []
        self.confidence_thresholds: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _now(self):
        return datetime.now(timezone.utc)

    def fetch_price_history(self, days_back: int = 365) -> pd.DataFrame:
        logger.info(f"Fetching price history (last {days_back}d)...")

        archive_dir = Path(__file__).parent.parent.parent / "price-archive"
        if archive_dir.exists() and days_back > 14:
            import duckdb
            cutoff = (self._now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            con = duckdb.connect()
            try:
                rows = con.sql("""
                    SELECT item_slug, day, mean_price AS price, volume
                    FROM read_parquet('{}/prices-*.parquet')
                    WHERE day >= ?
                    ORDER BY item_slug, day
                """.format(archive_dir), params=[cutoff]).fetchall()
                df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["date"] = df["timestamp"].dt.date
                logger.info(f"  {len(df):,} rows (Parquet), {df.item_id.nunique():,} items")
                return df
            finally:
                con.close()

        cutoff = self._now() - timedelta(days=days_back)
        rows = self.db.execute(text("""
            SELECT item_id, date(timestamp) AS day, AVG(price) AS price, SUM(volume) AS volume
            FROM price_history
            WHERE timestamp >= :cutoff
              AND source NOT LIKE 'synthetic_demo'
              AND source NOT LIKE 'historical_fallback:%'
            GROUP BY item_id, date(timestamp)
            ORDER BY item_id, day
        """), {"cutoff": cutoff}).fetchall()
        df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        logger.info(f"  {len(df):,} rows, {df.item_id.nunique():,} items")
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
        grouped = df.groupby("item_id")

        # Lag prices
        for lag in [1, 3, 7, 14, 30]:
            df[f"price_lag_{lag}d"] = grouped["price"].shift(lag)

        # Returns
        for lag in [1, 3, 7, 14, 30]:
            col = f"price_lag_{lag}d"
            df[f"return_{lag}d"] = (df["price"] - df[col]) / df[col].replace(0, np.nan) * 100

        # Rolling statistics (min_periods=1 so items with short history get partial estimates)
        for window in [7, 14, 20, 30]:
            roll = grouped["price"].rolling(window, min_periods=1)
            df[f"price_mean_{window}d"] = roll.mean().reset_index(level=0, drop=True)
            df[f"price_std_{window}d"] = roll.std().reset_index(level=0, drop=True)
            df[f"price_min_{window}d"] = roll.min().reset_index(level=0, drop=True)
            df[f"price_max_{window}d"] = roll.max().reset_index(level=0, drop=True)

        # Z-score vs 30d rolling
        mean_30 = df["price_mean_30d"]
        std_30 = df["price_std_30d"].replace(0, np.nan)
        df["price_zscore_30d"] = (df["price"] - mean_30) / std_30

        # Price acceleration (2nd derivative)
        df["price_accel_7d"] = df["return_7d"] - df["return_7d"].groupby(df["item_id"]).shift(7)

        # Log returns (stationary, scale-invariant)
        df["log_return_1d"] = np.log(df["price"] / df["price_lag_1d"].replace(0, np.nan))
        df["log_return_7d"] = np.log(df["price"] / df["price_lag_7d"].replace(0, np.nan))

        # Price autocorrelation proxy (direction agreement between lag-1 and lag-7 returns)
        df["autocorr_1d"] = df["return_1d"] * df["return_1d"].groupby(df["item_id"]).shift(1)
        df["autocorr_7d"] = df["return_7d"] * df["return_7d"].groupby(df["item_id"]).shift(7)

        # =====================================================================
        # Bollinger Bands (20-day)
        # =====================================================================
        bb_mid = df["price_mean_20d"]
        bb_std = df["price_std_20d"].replace(0, np.nan)
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_pct_b"] = ((df["price"] - df["bb_lower"]) / bb_range).clip(-2, 2)
        df["bb_width"] = (bb_range / bb_mid.replace(0, np.nan))

        # =====================================================================
        # RSI (14-day)
        # =====================================================================
        price_change = grouped["price"].diff()
        gain = price_change.clip(lower=0)
        loss = (-price_change).clip(lower=0)
        avg_gain = gain.groupby(df["item_id"]).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
        avg_loss = loss.groupby(df["item_id"]).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].clip(0, 100)

        # =====================================================================
        # MACD
        # =====================================================================
        ema_12 = grouped["price"].transform(lambda x: x.ewm(span=12, min_periods=12).mean())
        ema_26 = grouped["price"].transform(lambda x: x.ewm(span=26, min_periods=26).mean())
        df["macd_line"] = ema_12 - ema_26
        df["macd_signal"] = df.groupby("item_id")["macd_line"].transform(
            lambda x: x.ewm(span=9, min_periods=9).mean()
        )
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # =====================================================================
        # Support / Resistance distances
        # =====================================================================
        df["distance_to_support"] = ((df["price"] - df["price_min_30d"]).replace(0, np.nan) /
                                      df["price_min_30d"].replace(0, np.nan) * 100)
        df["distance_to_resistance"] = ((df["price_max_30d"] - df["price"]).replace(0, np.nan) /
                                         df["price"].replace(0, np.nan) * 100)
        df["high_low_range_30d"] = ((df["price_max_30d"] - df["price_min_30d"]).replace(0, np.nan) /
                                     df["price_min_30d"].replace(0, np.nan) * 100)

        # =====================================================================
        # Volume features
        # =====================================================================
        has_volume = "volume" in df.columns and df["volume"].notna().any()
        df["volume_missing"] = (1 if not has_volume else
                                df["volume"].isna().astype(int))

        if has_volume:
            df["volume_lag_1d"] = grouped["volume"].shift(1)
            df["volume_lag_7d"] = grouped["volume"].shift(7)
            df["volume_mean_7d"] = grouped["volume"].rolling(
                7, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            df["volume_mean_30d"] = grouped["volume"].rolling(
                30, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            df["volume_std_30d"] = grouped["volume"].rolling(
                30, min_periods=1
            ).std().reset_index(level=0, drop=True)

            # Log-ratio volume change (avoids division-by-zero issues)
            vol_lag_1 = df["volume_lag_1d"].replace(0, np.nan)
            vol_lag_7 = df["volume_lag_7d"].replace(0, np.nan)
            df["volume_log_change_1d"] = np.log(df["volume"] / vol_lag_1)
            df["volume_log_change_7d"] = np.log(df["volume"] / vol_lag_7)

            # Volume z-score vs 30d
            vol_std_30 = df["volume_std_30d"].replace(0, np.nan)
            df["volume_zscore_30d"] = ((df["volume"] - df["volume_mean_30d"]) / vol_std_30)

            # Volume-price confirmation
            df["volume_price_conf_7d"] = (df["return_7d"] *
                                          (df["volume_log_change_7d"] > 0).astype(int))
            df["volume_price_conf_1d"] = (df["return_1d"] *
                                          (df["volume_log_change_1d"] > 0).astype(int))
        else:
            for col in ["volume_lag_1d", "volume_lag_7d", "volume_mean_7d",
                        "volume_mean_30d", "volume_std_30d",
                        "volume_log_change_1d", "volume_log_change_7d",
                        "volume_zscore_30d", "volume_price_conf_7d",
                        "volume_price_conf_1d"]:
                df[col] = np.nan

        # Boolean indicators for features with frequent missingness
        df["rsi_missing"] = df["rsi_14"].isna().astype(int)
        df["macd_missing"] = df["macd_line"].isna().astype(int)

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(df["date"])
        df["day_of_week"] = dates.dt.dayofweek
        df["month"] = dates.dt.month
        df["quarter"] = dates.dt.quarter
        df["day_of_year"] = dates.dt.dayofyear
        df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        if "item_id" in df.columns:
            item_first_date = df.groupby("item_id")["date"].transform("min")
            df["item_age_days"] = (pd.to_datetime(df["date"]) - pd.to_datetime(item_first_date)).dt.days
        else:
            df["item_age_days"] = 0
        return df

    def _add_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        event_types = ["major", "operation", "case_drop", "update", "game_update"]

        # Decay constants: days until ~37% effect remaining (learnable per type)
        decay_constants = {
            "major": 60,
            "operation": 21,
            "case_drop": 14,
            "update": 7,
            "game_update": 7,
        }

        if events_df.empty:
            for event_type in event_types:
                df[f"event_decay_{event_type}"] = 0.0
                df[f"events_next_30d_{event_type}"] = 0
                df[f"event_density_30d_{event_type}"] = 0
                df[f"event_density_90d_{event_type}"] = 0
            return df

        for event_type in event_types:
            type_events = events_df[events_df["type"] == event_type].sort_values("date")
            decay_tau = decay_constants.get(event_type, 30)

            if type_events.empty:
                df[f"event_decay_{event_type}"] = 0.0
                df[f"events_next_30d_{event_type}"] = 0
                df[f"event_density_30d_{event_type}"] = 0
                df[f"event_density_90d_{event_type}"] = 0
                continue

            dates = pd.to_datetime(df["date"])
            event_dates = pd.to_datetime(type_events["date"].unique())
            sorted_events = np.sort(event_dates)
            all_dates = dates.values

            # Exponential decay of most recent event: exp(-days_since / tau)
            indices = np.searchsorted(sorted_events, all_dates) - 1
            valid = indices >= 0
            decay_val = np.zeros(len(dates), dtype=float)
            if valid.any():
                last_event_dates = sorted_events[indices[valid]]
                days_since = (all_dates[valid] - last_event_dates).astype('timedelta64[D]').astype(float)
                decay_val[valid] = np.exp(-days_since / decay_tau)
            df[f"event_decay_{event_type}"] = decay_val

            # Count events in next 30 days (vectorized)
            left = np.searchsorted(sorted_events, all_dates, side="right")
            right = np.searchsorted(sorted_events, all_dates + np.timedelta64(30, "D"), side="right")
            df[f"events_next_30d_{event_type}"] = right - left

            # Event density: number of events in recent windows
            past_30 = np.searchsorted(sorted_events, all_dates, side="right") - np.searchsorted(
                sorted_events, all_dates - np.timedelta64(30, "D"), side="right"
            )
            past_90 = np.searchsorted(sorted_events, all_dates, side="right") - np.searchsorted(
                sorted_events, all_dates - np.timedelta64(90, "D"), side="right"
            )
            df[f"event_density_30d_{event_type}"] = past_30
            df[f"event_density_90d_{event_type}"] = past_90

        return df

    def _add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-level and category-level context features."""
        logger.info("Adding cross-sectional features...")
        df = df.copy()

        # Market return: mean return across all items per date
        for lag in [1, 7, 14, 30]:
            ret_col = f"return_{lag}d"
            if ret_col not in df.columns:
                continue
            market_col = f"market_return_{lag}d"
            df[market_col] = df.groupby("date")[ret_col].transform("mean")
            df[f"item_return_vs_market_{lag}d"] = df[ret_col] - df[market_col]

        # Market volatility: mean of individual item volatilities per date
        if "price_std_30d" in df.columns:
            df["market_volatility_30d"] = df.groupby("date")["price_std_30d"].transform("mean")

        # Market volume: compute daily market mean, then rolling 30d of that
        if "volume" in df.columns and df["volume"].notna().any():
            daily_vol = df.groupby("date")["volume"].mean().to_frame("daily_market_vol")
            daily_vol = daily_vol.sort_index()
            daily_vol["market_volume_mean_30d"] = daily_vol["daily_market_vol"].rolling(
                30, min_periods=1
            ).mean()
            df = df.merge(
                daily_vol[["market_volume_mean_30d"]],
                left_on="date", right_index=True, how="left"
            )
            df["item_volume_vs_market_30d"] = (
                df["volume"] / df["market_volume_mean_30d"].replace(0, np.nan)
            )

        # Market regime: bull/bear/range based on median 30d market return
        if "market_return_30d" in df.columns:
            market_ret_median = df.groupby("date")["market_return_30d"].transform("median")
            df["market_regime_bull"] = (market_ret_median > 5).astype(int)
            df["market_regime_bear"] = (market_ret_median < -5).astype(int)
            df["market_regime_range"] = ((market_ret_median >= -5) & (market_ret_median <= 5)).astype(int)

        return df

    def _prune_features(self, df: pd.DataFrame) -> List[str]:
        """Remove highly correlated features to reduce noise and multicollinearity.

        Identifies feature pairs with correlation > PRUNE_CORRELATION_THRESHOLD
        and drops one from each pair, keeping features with lower index (earlier
        in the list).
        """
        if len(self.feature_cols) < 2:
            return self.feature_cols

        corr = df[self.feature_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            highly_correlated = upper[col][upper[col] > self.PRUNE_CORRELATION_THRESHOLD].index
            to_drop.update(highly_correlated)

        pruned = [c for c in self.feature_cols if c not in to_drop]
        if pruned != self.feature_cols:
            logger.info(
                f"  Pruned {len(self.feature_cols) - len(pruned)} features "
                f"(corr>{self.PRUNE_CORRELATION_THRESHOLD}): "
                f"{len(pruned)} remaining"
            )
        return pruned

    def _grid_search_params(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Simple grid search over key hyperparameters for quantile regression.

        Searches over num_leaves, learning_rate, lambda_l1, lambda_l2 using
        validation quantile loss to pick the best combination.
        """
        grid = {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.03, 0.05],
            "lambda_l1": [0.0, 0.5, 1.0],
            "lambda_l2": [0.0, 0.5, 1.0],
        }

        best_params = None
        best_loss = float("inf")

        n_trials = 0
        for nl in grid["num_leaves"]:
            for lr in grid["learning_rate"]:
                for l1 in grid["lambda_l1"]:
                    for l2 in grid["lambda_l2"]:
                        n_trials += 1
                        params = {
                            "objective": "quantile",
                            "alpha": 0.5,
                            "metric": "quantile",
                            "boosting_type": "gbdt",
                            "num_leaves": nl,
                            "max_depth": 5,
                            "min_data_in_leaf": 15,
                            "min_gain_to_split": 0.1,
                            "learning_rate": lr,
                            "feature_fraction": 0.7,
                            "bagging_fraction": 0.7,
                            "bagging_freq": 5,
                            "lambda_l1": l1,
                            "lambda_l2": l2,
                            "verbosity": -1,
                            "random_state": 42,
                            "n_jobs": -1,
                        }
                        dtrain = lgb.Dataset(X_train, y_train)
                        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
                        model = lgb.train(
                            params, dtrain,
                            num_boost_round=200,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                        )
                        loss = model.best_score["valid_0"]["quantile"]
                        if loss < best_loss:
                            best_loss = loss
                            best_params = params.copy()

        logger.info(
            f"  Grid search ({n_trials} combos): best loss={best_loss:.6f} "
            f"params={best_params}"
        )
        return best_params

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

        # Date-based target lookup: find the price exactly `horizon` calendar
        # days later, rather than shifting rows. Row-based shift gives
        # incorrect horizons when item data has gaps.
        df["_date_dt"] = pd.to_datetime(df["date"])
        df["_target_dt"] = df["_date_dt"] + pd.Timedelta(days=horizon)
        df["_target_date"] = df["_target_dt"].dt.date

        future = df[["item_id", "_target_date", "price"]].rename(
            columns={"_target_date": "date", "price": f"target_{horizon}d"}
        )
        df = df.merge(future, on=["item_id", "date"], how="left")
        df = df.drop(columns=["_date_dt", "_target_dt", "_target_date"])

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

        df = self.engineer_features(price_df, events_df)

        # Add cross-sectional (market-regime) features
        df = self._add_cross_sectional_features(df)

        # Define feature columns (exclude metadata and target columns)
        exclude = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date"}
        exclude |= {f"target_{h}d" for h in self.HORIZONS}
        exclude |= {f"target_return_{h}d" for h in self.HORIZONS}

        self.feature_cols = [c for c in df.columns if c not in exclude
                             and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]

        # Prune highly correlated features to reduce noise
        self.feature_cols = self._prune_features(df)

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

    def train(self, max_rows: int = 200_000):
        logger.info("=" * 60)
        logger.info("TRAINING LIGHTGBM FORECASTER (ensemble, HP search, walk-forward)")
        logger.info("=" * 60)

        df, targets_by_horizon = self.build_training_data(days_back=365)

        for horizon in self.HORIZONS:
            tdf = targets_by_horizon[horizon]

            # Drop NaN targets (use percentage return as primary target)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values("date")

            # Proper temporal walk-forward split (using actual data dates):
            # Hold out the last VALIDATION_WINDOW_DAYS of calendar data.
            max_date = pd.to_datetime(tdf["date"].max())
            split_date = max_date - timedelta(days=self.VALIDATION_WINDOW_DAYS)
            train_set = tdf[pd.to_datetime(tdf["date"]) < split_date]
            val_set = tdf[pd.to_datetime(tdf["date"]) >= split_date]

            # Cap training size (keep most recent data)
            if len(train_set) > max_rows:
                train_set = train_set.tail(max_rows)

            if len(val_set) < 100:
                logger.warning(f"  Validation set for {horizon}d has only {len(val_set)} rows; "
                               "using last 20% of training data as fallback.")
                split_idx = int(len(tdf) * 0.8)
                train_set = tdf.iloc[:split_idx]
                val_set = tdf.iloc[split_idx:]

            logger.info(f"  {horizon}d: {len(train_set)} train, {len(val_set)} val")

            # Per-feature median imputation (learned from train, applied to both)
            train_features = train_set[self.feature_cols]
            feature_medians = train_features.median()
            X_train = train_features.fillna(feature_medians)
            y_train = train_set[f"target_return_{horizon}d"]

            X_val = val_set[self.feature_cols].fillna(feature_medians)
            y_val = val_set[f"target_return_{horizon}d"]

            # Grid search for best hyperparameters
            logger.info(f"  Grid searching hyperparameters for {horizon}d...")
            best_params = self._grid_search_params(X_train, y_train, X_val, y_val)

            for q in self.QUANTILES:
                logger.info(f"Training {horizon}d p{int(q*100)} ensemble ({self.N_ENSEMBLES} members)...")

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
                # Merge grid search results into base params
                if best_params:
                    for k in ("num_leaves", "learning_rate", "lambda_l1", "lambda_l2"):
                        if k in best_params:
                            base_params[k] = best_params[k]
                else:
                    base_params["num_leaves"] = 31
                    base_params["learning_rate"] = 0.03
                    base_params["lambda_l1"] = 0.5
                    base_params["lambda_l2"] = 0.5

                ensemble_models = []
                for ei in range(self.N_ENSEMBLES):
                    params = base_params.copy()
                    params["random_state"] = self.ENSEMBLE_SEEDS[ei]

                    dtrain = lgb.Dataset(X_train, y_train)
                    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
                    model = lgb.train(
                        params, dtrain,
                        num_boost_round=1000,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                    ensemble_models.append(model)

                self.models[(horizon, q)] = ensemble_models
                fi = self._get_feature_importance(ensemble_models[0])
                logger.info(f"  Top features: {fi['feature'].head(5).tolist()}")

            # Calibrate confidence thresholds from validation set
            self._calibrate_confidence(X_val, y_val, val_set, horizon)

        self.save_models()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, item_ids: List[int] = None) -> pd.DataFrame:
        logger.info("Generating forecasts...")

        price_df = self.fetch_price_history(days_back=365)

        # Skip items without a real recent series: snapshot-tier items keep
        # only a single latest row, and a "forecast" from one data point is
        # a meaningless constant that would still be written to the DB.
        day_counts = price_df.groupby("item_id")["date"].nunique()
        eligible = day_counts[day_counts >= self.PREDICT_MIN_HISTORY_DAYS].index
        skipped = price_df["item_id"].nunique() - len(eligible)
        price_df = price_df[price_df["item_id"].isin(eligible)]
        logger.info(
            f"  {len(eligible):,} items with >= {self.PREDICT_MIN_HISTORY_DAYS} days of history "
            f"({skipped:,} skipped)"
        )

        events_df = self.fetch_events()

        df = self.engineer_features(price_df, events_df)

        # Add cross-sectional features (same as training)
        df = self._add_cross_sectional_features(df)

        # Align features with training columns (add missing, drop extras)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_cols + [c for c in df.columns if c not in self.feature_cols]]

        # Get latest feature row per item
        df = df.sort_values(["item_id", "date"])
        latest_rows = df.groupby("item_id").last().reset_index()

        if item_ids:
            latest_rows = latest_rows[latest_rows["item_id"].isin(item_ids)]

        # Median imputation (use predefined medians, or compute from this batch)
        feature_medians = latest_rows[self.feature_cols].median()
        X_batch = latest_rows[self.feature_cols].fillna(feature_medians)

        item_id_arr = latest_rows["item_id"].to_numpy()
        current_price_arr = latest_rows["price"].to_numpy()
        generated_at = self._now()

        # One row per item, filled in horizon by horizon.
        agg = {
            iid: {
                "item_id": iid,
                "current_price": float(cur),
                "forecasts": {},
                "generated_at": generated_at,
            }
            for iid, cur in zip(item_id_arr, current_price_arr)
        }

        for horizon in self.HORIZONS:
            preds = {}
            for q in self.QUANTILES:
                key = (horizon, q)
                if key in self.models:
                    ensemble = self.models[key]
                    if isinstance(ensemble, list):
                        # Average ensemble predictions
                        all_preds = []
                        for m in ensemble:
                            all_preds.append(m.predict(X_batch))
                        preds[q] = np.mean(all_preds, axis=0)
                    else:
                        # Backwards compat: single model
                        preds[q] = ensemble.predict(X_batch)

            if len(preds) != 3:
                continue

            # Models predict percentage returns. Convert back to price levels.
            # preds are return percentages (e.g., 5.0 means +5%).
            p10_ret = preds[0.1]
            p50_ret = preds[0.5]
            p90_ret = preds[0.9]

            # Ensure quantile monotonicity without scrambling model identities.
            # The median model (0.5) is kept as-is; low/high are clamped to
            # guarantee p10 <= p50 <= p90.
            low_ret_arr = np.minimum(p10_ret, p50_ret)
            high_ret_arr = np.maximum(p50_ret, p90_ret)
            mid_ret_arr = p50_ret

            # Diagnostic: log crossing rate
            crossing_rate = np.mean((p10_ret > p50_ret) | (p50_ret > p90_ret))
            if crossing_rate > 0.01:
                logger.warning(f"  Quantile crossing rate: {crossing_rate:.3f}")

            for i, iid in enumerate(item_id_arr):
                low_ret, mid_ret, high_ret = (float(low_ret_arr[i]),
                                               float(mid_ret_arr[i]),
                                               float(high_ret_arr[i]))
                current_price = float(current_price_arr[i])

                # Convert return predictions to price levels
                price_low = round(current_price * (1 + low_ret / 100), 2)
                price_mid = round(current_price * (1 + mid_ret / 100), 2)
                price_high = round(current_price * (1 + high_ret / 100), 2)

                agg[iid]["forecasts"][horizon] = {
                    "low": price_low,
                    "mid": price_mid,
                    "high": price_high,
                    "direction": "up" if mid_ret > 0 else "down" if mid_ret < 0 else "flat",
                    "confidence": self._compute_confidence(price_mid, price_low, price_high, current_price),
                }

        result_df = pd.DataFrame([r for r in agg.values() if r["forecasts"]])
        logger.info(f"  Forecasts generated for {len(result_df)} items")
        return result_df

    def predict_single(self, item_id: int) -> Dict[str, Any]:
        results = self.predict(item_ids=[item_id])
        if results.empty:
            return {}
        return results.iloc[0].to_dict()

    def _get_ensemble_prediction(self, horizon, q, X):
        """Get averaged prediction from an ensemble or single model."""
        key = (horizon, q)
        if key not in self.models:
            return None
        ensemble = self.models[key]
        if isinstance(ensemble, list):
            preds = np.mean([m.predict(X) for m in ensemble], axis=0)
        else:
            preds = ensemble.predict(X)
        return preds

    def _calibrate_confidence(self, X_val, y_val, val_set, horizon):
        """Calibrate confidence thresholds from validation set predictions.

        Binary confidence (high/low only):
        - "high" confidence achieves >= 75% directional accuracy
        - everything else is "low"
        """
        p50_pred = self._get_ensemble_prediction(horizon, 0.5, X_val.values)
        p10_pred = self._get_ensemble_prediction(horizon, 0.1, X_val.values)
        p90_pred = self._get_ensemble_prediction(horizon, 0.9, X_val.values)
        if p50_pred is None or p10_pred is None or p90_pred is None:
            return

        current_prices = val_set["price"].values
        actual_returns = y_val.values

        records = []
        for i in range(len(val_set)):
            mid_ret = float(p50_pred[i])
            low_ret = float(min(p10_pred[i], p50_pred[i]))
            high_ret = float(max(p50_pred[i], p90_pred[i]))
            curr = float(current_prices[i])
            actual_ret = float(actual_returns[i])

            mid_price = curr * (1 + mid_ret / 100)
            low_price = curr * (1 + low_ret / 100)
            high_price = curr * (1 + high_ret / 100)

            if mid_price == 0 or curr == 0:
                continue

            range_pct = (high_price - low_price) / mid_price
            change_pct = abs(mid_price - curr) / curr
            actual_dir = "up" if actual_ret > 0 else "down" if actual_ret < 0 else "flat"
            pred_dir = "up" if mid_ret > 0 else "down" if mid_ret < 0 else "flat"
            hit = 1.0 if pred_dir == actual_dir else 0.0

            records.append({
                "range_pct": range_pct,
                "change_pct": change_pct,
                "hit": hit,
            })

        if len(records) < 50:
            return

        df = pd.DataFrame(records)

        # Find optimal range_pct threshold for high confidence (binary)
        thresholds = sorted(df["range_pct"].quantile([i / 20 for i in range(1, 20)]).unique())

        best_high_threshold = 0.15
        best_high_acc = 0.0

        for t in thresholds:
            subset = df[df["range_pct"] < t]
            if len(subset) >= max(50, len(df) * 0.05):
                acc = subset["hit"].mean()
                if acc >= 0.75 and acc > best_high_acc:
                    best_high_threshold = t
                    best_high_acc = acc

        # Find change_pct threshold that further improves high confidence
        best_change_threshold = 0.0
        best_change_acc = best_high_acc
        if best_high_threshold > 0:
            high_set = df[df["range_pct"] < best_high_threshold]
            if len(high_set) > 0:
                change_thresholds = sorted(high_set["change_pct"].quantile(
                    [i / 10 for i in range(1, 10)]).unique())
                for ct in change_thresholds:
                    subset = high_set[high_set["change_pct"] > ct]
                    if len(subset) >= max(20, len(high_set) * 0.3):
                        acc = subset["hit"].mean()
                        if acc > best_change_acc:
                            best_change_threshold = ct
                            best_change_acc = acc

                self.confidence_thresholds = {
                    "high_range": best_high_threshold,
                    "high_change": best_change_threshold,
                    "high_accuracy": round(best_change_acc * 100, 1),
                }
        else:
            self.confidence_thresholds = {
                "high_range": 0.15,
                "high_change": 0.0,
                "high_accuracy": 0.0,
            }

        logger.info(
            f"  Calibrated (binary): high_range={self.confidence_thresholds['high_range']:.3f} "
            f"(acc={self.confidence_thresholds['high_accuracy']:.1f}%)"
        )

    def _compute_confidence(self, mid: float, low: float, high: float, current: float) -> str:
        """Binary confidence: high (narrow prediction interval) or low."""
        if mid == 0 or current == 0:
            return "low"
        range_pct = (high - low) / mid
        change_pct = abs(mid - current) / current

        # Use calibrated thresholds if available, fall back to sensible defaults
        high_range = self.confidence_thresholds.get("high_range", 0.15)
        high_change = self.confidence_thresholds.get("high_change", 0.0)

        if range_pct < high_range and change_pct > high_change:
            return "high"
        return "low"

    # ------------------------------------------------------------------
    # Concept drift monitoring
    # ------------------------------------------------------------------

    def check_concept_drift(self, horizon: int = 7, sliding_window: int = 7,
                             threshold: float = 60.0) -> Optional[Dict]:
        """Check if recent prediction accuracy has dropped below threshold.

        Queries the last `sliding_window` days of forecast backtest results
        and compares directional accuracy against the threshold. Logs an
        alert to the accuracy_alerts table if drift is detected.
        """
        from database import PredictionAccuracy, AccuracyAlert
        from sqlalchemy import desc

        cutoff = (self._now() - timedelta(days=sliding_window * 2)).strftime("%Y-%m-%d")
        records = self.db.execute(text("""
            SELECT evaluation_date, metrics
            FROM prediction_accuracy
            WHERE prediction_type = 'forecast'
              AND horizon_days = :horizon
              AND evaluation_date >= :cutoff
            ORDER BY evaluation_date DESC
            LIMIT :limit
        """), {"horizon": horizon, "cutoff": cutoff, "limit": sliding_window}).fetchall()

        if not records:
            return None

        accuracies = []
        for r in records:
            m = r.metrics if isinstance(r.metrics, dict) else json.loads(r.metrics)
            if "directional_accuracy" in m:
                accuracies.append(m["directional_accuracy"])

        if len(accuracies) < 3:
            return None

        recent_avg = sum(accuracies) / len(accuracies)
        logger.info(f"  Drift check ({horizon}d, {len(accuracies)} windows): "
                     f"avg_acc={recent_avg:.1f}%, threshold={threshold:.1f}%")

        if recent_avg >= threshold:
            # Resolve any open alert
            open_alert = self.db.query(AccuracyAlert).filter(
                AccuracyAlert.prediction_type == "forecast",
                AccuracyAlert.horizon_days == horizon,
                AccuracyAlert.resolved_at.is_(None),
            ).first()
            if open_alert:
                open_alert.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
                self.db.commit()
                logger.info(f"  Drift resolved: accuracy back to {recent_avg:.1f}%")
            return {"drifted": False, "accuracy": recent_avg, "threshold": threshold}

        # Trigger alert
        alert = AccuracyAlert(
            prediction_type="forecast",
            horizon_days=horizon,
            sliding_window_days=sliding_window,
            current_accuracy=round(recent_avg, 2),
            threshold_accuracy=threshold,
            sample_count=len(accuracies),
            triggered_at=datetime.now(timezone.utc).replace(tzinfo=None),
            details={"window_accuracies": accuracies},
        )
        self.db.add(alert)
        self.db.commit()
        logger.warning(f"  DRIFT DETECTED ({horizon}d): accuracy={recent_avg:.1f}% "
                        f"below threshold={threshold:.1f}%")
        return {"drifted": True, "accuracy": recent_avg, "threshold": threshold}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        for (horizon, q), ensemble in self.models.items():
            if isinstance(ensemble, list):
                for ei, model in enumerate(ensemble):
                    path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}_e{ei}.txt")
                    model.save_model(path)
            else:
                path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}.txt")
                ensemble.save_model(path)

        # Save feature columns and calibration thresholds
        thresholds = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in self.confidence_thresholds.items()}
        meta = {
            "feature_cols": self.feature_cols,
            "trained_at": str(self._now()),
            "confidence_thresholds": thresholds,
            "n_ensembles": self.N_ENSEMBLES,
        }
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
        self.confidence_thresholds = meta.get("confidence_thresholds", {})
        n_ensembles = meta.get("n_ensembles", 1)

        for horizon in self.HORIZONS:
            for q in self.QUANTILES:
                if n_ensembles > 1:
                    ensemble = []
                    for ei in range(n_ensembles):
                        path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}_e{ei}.txt")
                        if os.path.exists(path):
                            ensemble.append(lgb.Booster(model_file=path))
                    if ensemble:
                        self.models[(horizon, q)] = ensemble
                else:
                    path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}.txt")
                    if os.path.exists(path):
                        self.models[(horizon, q)] = lgb.Booster(model_file=path)

        logger.info(f"Loaded {len(self.models)} model groups from {self.model_dir}")
        return len(self.models) > 0

    def has_models(self) -> bool:
        return len(self.models) > 0
