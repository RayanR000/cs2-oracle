"""
LightGBM-based price forecaster for CS2 items.
Trains quantile regression models for 3d, 7d, 14d and 30d horizons,
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
from models.item_parser import parse_item_name

logger = logging.getLogger(__name__)

RNG = np.random.RandomState(42)


def _feature_group(name: str) -> str:
    if any(name.startswith(p) for p in ("price_", "return_", "log_return_", "bb_",
                                         "rsi_", "macd_", "vol_", "trend_",
                                         "price_accel_", "autocorr_", "support_",
                                         "volume_", "vol_price_")):
        return "price_technicals"
    if name.startswith("players_"):
        return "player_counts"
    # Cross-sectional supply-side features (check before identity to avoid
    # prefix collision — e.g., "weapon_type_group_return_7d" starts with
    # "weapon_type_" but belongs to supply_side, not item_identity).
    if any(name.startswith(p) for p in ("wt_group_return_", "wt_volatility_", "wt_volume_",
                                         "item_return_vs_weapon_type_",
                                         "item_volume_vs_weapon_type_",
                                         "rarity_market_regime_", "quality_rarity_")):
        return "supply_side"
    if any(name.startswith(p) for p in ("is_", "quality_rank", "rarity_", "weapon_type_")):
        return "item_identity"
    if name.startswith("type_"):
        return "item_metadata"
    if any(name.startswith(p) for p in ("day_", "month_", "quarter_", "week_",
                                         "item_age", "weekend")):
        return "temporal"
    if name.startswith("event_"):
        return "events"
    if any(name.startswith(p) for p in ("market_", "market_regime_")):
        return "cross_sectional"
    return "other"


class ItemForecaster:
    HORIZONS = [3, 7, 14, 30]
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
    # Expanding-window cross-validation
    CV_STEP_DAYS = 200        # Days between each fold's validation window
    CV_MIN_TRAIN_DAYS = 200   # Minimum unique dates before first validation fold

    def __init__(self, db_session, model_dir: str = None, prune_failed_groups: bool = True):
        self.db = db_session
        self.model_dir = model_dir or str(Path(__file__).parent / "saved_models")
        self.models: Dict[Tuple[int, float], lgb.Booster] = {}
        self.feature_cols: List[str] = []
        self.prune_failed_groups = prune_failed_groups
        # Per-horizon confidence thresholds: {horizon: {"high_range": ..., "high_change": ..., "high_accuracy": ...}}
        self.confidence_thresholds: Dict[int, Dict[str, float]] = {}
        self.feature_medians: pd.Series = pd.Series(dtype=np.float64)
        # Expanding-window CV results per horizon: {horizon: {fold_count, fold_accs, per_fold, ...}}
        self.cv_results: Dict[int, Dict] = {}

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _now(self):
        return datetime.now(timezone.utc)

    def fetch_price_history(self, days_back: int = 365,
                            backfilled_only: bool = False) -> pd.DataFrame:
        logger.info(f"Fetching price history (last {days_back}d)...")

        archive_dir = Path(__file__).parent.parent.parent / "price-archive"
        if archive_dir.exists() and days_back > 14:
            import duckdb
            cutoff = (self._now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            con = duckdb.connect()
            try:
                backfilled_slugs = None
                if backfilled_only:
                    backfilled_slugs = {
                        r[0] for r in con.sql(f"""
                            SELECT DISTINCT item_slug
                            FROM read_parquet('{archive_dir}/prices-*.parquet')
                        """).fetchall()
                    }
                rows = con.sql(f"""
                    SELECT item_slug, day, mean_price AS price, volume
                    FROM read_parquet('{archive_dir}/prices-*.parquet')
                    WHERE day >= ?
                    ORDER BY item_slug, day
                """, params=[cutoff]).fetchall()
                if backfilled_slugs is not None:
                    rows = [r for r in rows if r[0] in backfilled_slugs]
                df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["date"] = df["timestamp"].dt.date
                logger.info(f"  {len(df):,} rows (Parquet), {df.item_id.nunique():,} items")
                if backfilled_only:
                    logger.info(f"  Filtered to STEAMCOMMUNITY-backfilled items only")
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
        if hasattr(self, "_events_cache") and self._events_cache is not None:
            return self._events_cache
        try:
            rows = self.db.execute(text("""
                SELECT id, type, timestamp, description
                FROM events
                ORDER BY timestamp
            """)).fetchall()
        except Exception:
            logger.warning("  DB connection lost, reconnecting...")
            from database import SessionLocal
            self.db = SessionLocal()
            rows = self.db.execute(text("""
                SELECT id, type, timestamp, description
                FROM events
                ORDER BY timestamp
            """)).fetchall()
        df = pd.DataFrame(rows, columns=["id", "type", "timestamp", "description"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        self._events_cache = df
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
        for lag in [1, 3, 7, 14, 30, 60]:
            df[f"price_lag_{lag}d"] = grouped["price"].shift(lag)

        # Returns
        for lag in [1, 3, 7, 14, 30, 60]:
            col = f"price_lag_{lag}d"
            df[f"return_{lag}d"] = (df["price"] - df[col]) / df[col].replace(0, np.nan) * 100

        # Winsorize extreme returns (>500%) which are likely data artifacts
        for lag in [1, 3, 7, 14, 30, 60]:
            col = f"return_{lag}d"
            if col in df.columns:
                df[col] = df[col].clip(-500, 500)

        # Rolling statistics (min_periods=1 so items with short history get partial estimates)
        for window in [7, 14, 20, 30, 60]:
            roll = grouped["price"].rolling(window, min_periods=1)
            df[f"price_mean_{window}d"] = roll.mean().reset_index(level=0, drop=True)
            df[f"price_std_{window}d"] = roll.std().reset_index(level=0, drop=True)
            df[f"price_min_{window}d"] = roll.min().reset_index(level=0, drop=True)
            df[f"price_max_{window}d"] = roll.max().reset_index(level=0, drop=True)

        # Z-score vs 30d rolling
        mean_30 = df["price_mean_30d"]
        std_30 = df["price_std_30d"].replace(0, np.nan)
        df["price_zscore_30d"] = (df["price"] - mean_30) / std_30

        # Long-term volatility regime: ratio of 60d to 30d volatility
        # Values > 1 mean volatility is rising, < 1 mean it's falling.
        vol_30 = df["price_std_30d"].replace(0, np.nan)
        vol_60 = df["price_std_60d"].replace(0, np.nan)
        df["vol_regime_60_30"] = vol_60 / vol_30

        # Trend divergence: ratio of 30d return to 60d return.
        # Shows whether short-term momentum agrees with long-term trend.
        df["trend_divergence_30_60"] = (df.get("return_30d", pd.Series(np.nan, index=df.index)) /
                                        df.get("return_60d", pd.Series(np.nan, index=df.index)).replace(0, np.nan))

        # Price acceleration (2nd derivative)
        df["price_accel_7d"] = df["return_7d"] - df["return_7d"].groupby(df["item_id"]).shift(7)

        # Log price (scale-invariant)
        df["price_log"] = np.log(df["price"].clip(lower=0.01))

        # Price tier (categorical price level buckets)
        price = df["price"]
        df["price_tier"] = 0
        df.loc[price >= 1, "price_tier"] = 1
        df.loc[price >= 5, "price_tier"] = 2
        df.loc[price >= 20, "price_tier"] = 3
        df.loc[price >= 100, "price_tier"] = 4
        df["price_tier"] = df["price_tier"].astype(int)

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
            df["volume_mean_60d"] = grouped["volume"].rolling(
                60, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            df["volume_std_60d"] = grouped["volume"].rolling(
                60, min_periods=1
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
                        "volume_mean_60d", "volume_std_60d",
                        "volume_log_change_1d", "volume_log_change_7d",
                        "volume_zscore_30d", "volume_price_conf_7d",
                        "volume_price_conf_1d"]:
                df[col] = np.nan

        # Boolean indicators for features with frequent missingness
        df["rsi_missing"] = df["rsi_14"].isna().astype(int)
        df["macd_missing"] = df["macd_line"].isna().astype(int)

        return df

    def _fetch_item_metadata(self) -> pd.DataFrame:
        if hasattr(self, "_item_meta_cache") and self._item_meta_cache is not None:
            return self._item_meta_cache
        try:
            rows = self.db.execute(text("""
                SELECT item_id, type FROM items
            """)).fetchall()
            df = pd.DataFrame(rows, columns=["item_id", "type"])
        except Exception:
            self._item_meta_cache = pd.DataFrame(columns=["item_id", "type"])
            return self._item_meta_cache
        self._item_meta_cache = df
        logger.info(f"  item metadata: {len(df)} items loaded")
        return df

    def _fetch_supply_metadata(self) -> pd.DataFrame:
        """Load supply-side metadata (rarity, weapon_type) from Parquet or DB.

        Tries price-archive/item-metadata.parquet first, then falls back
        to the items table in the database.
        """
        if hasattr(self, "_supply_meta_cache") and self._supply_meta_cache is not None:
            return self._supply_meta_cache

        archive_dir = Path(__file__).parent.parent.parent / "price-archive"
        meta_path = archive_dir / "item-metadata.parquet"

        if meta_path.exists():
            try:
                df = pd.read_parquet(meta_path)
                df = df.rename(columns={"item_slug": "item_id"})
                logger.info(f"  supply metadata: {len(df)} items loaded from Parquet")
                self._supply_meta_cache = df
                return df
            except Exception as e:
                logger.warning(f"  Failed to load supply metadata from Parquet: {e}")

        try:
            rows = self.db.execute(text("""
                SELECT item_id, rarity, rarity_rank, weapon_type FROM items
            """)).fetchall()
            df = pd.DataFrame(rows, columns=["item_id", "rarity", "rarity_rank", "weapon_type"])
            logger.info(f"  supply metadata: {len(df)} items loaded from DB")
        except Exception:
            logger.warning("  Could not fetch supply metadata from DB; using empty DataFrame")
            df = pd.DataFrame(columns=["item_id", "rarity", "rarity_rank", "weapon_type"])

        self._supply_meta_cache = df
        return df

    def _add_supply_side_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add supply-side features: rarity flags, weapon type flags,
        and weapon-type cross-sectional features.

        Estimated impact: +3-6pp directional accuracy.
        """
        logger.info("Adding supply-side features...")
        df = df.copy()

        meta = self._fetch_supply_metadata()
        if meta.empty:
            return df

        df = df.merge(meta, on="item_id", how="left")

        # ── Identity features (static per item) ──
        # Rarity ordinal (NaN → 0 for missing)
        df["rarity_ordinal"] = df["rarity_rank"].fillna(0).astype(int)

        # Rarity one-hot dummies
        rarity_cats = ["base", "consumer", "industrial", "milspec",
                       "restricted", "classified", "covert",
                       "high_grade", "remarkable", "exotic", "extraordinary"]
        for cat in rarity_cats:
            col = f"rarity_{cat}"
            df[col] = ((df["rarity"] == cat).astype(int))

        # Weapon type one-hot dummies
        weapon_cats = ["rifle", "pistol", "smg", "shotgun", "sniper",
                       "machinegun", "knife", "glove", "case",
                       "sticker", "graffiti", "musickit", "charm",
                       "agent", "patch", "collectible", "equipment",
                       "key", "pass", "tool", "tag", "gift"]
        for cat in weapon_cats:
            col = f"weapon_type_{cat}"
            df[col] = ((df["weapon_type"] == cat).astype(int))

        return df

    def _add_weapon_type_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weapon-type group cross-sectional features.

        These parallel the market-level cross-sectional features but are
        computed per weapon_type group (rifle, pistol, smg, etc.).
        """
        logger.info("Adding weapon-type cross-sectional features...")
        df = df.copy()

        if "weapon_type" not in df.columns:
            logger.warning("  weapon_type not available; skipping cross-sectional features")
            return df

        # Compute per-date per-weapon-type aggregates
        for lag in [1, 7, 14, 30]:
            ret_col = f"return_{lag}d"
            if ret_col not in df.columns:
                continue
            group_col = f"wt_group_return_{lag}d"
            df[group_col] = df.groupby(["date", "weapon_type"])[ret_col].transform("mean")
            df[f"item_return_vs_weapon_type_{lag}d"] = df[ret_col] - df[group_col]

        # Weapon-type volatility
        if "price_std_30d" in df.columns:
            df["wt_volatility_30d"] = df.groupby(["date", "weapon_type"])["price_std_30d"].transform("mean")

        # Weapon-type volume
        if "volume" in df.columns and df["volume"].notna().any():
            daily_vol = df.groupby(["date", "weapon_type"])["volume"].mean().reset_index()
            daily_vol = daily_vol.sort_values(["weapon_type", "date"])
            daily_vol["wt_volume_30d"] = daily_vol.groupby("weapon_type")["volume"].transform(
                lambda x: x.rolling(30, min_periods=1).mean()
            )
            df = df.merge(
                daily_vol[["date", "weapon_type", "wt_volume_30d"]],
                on=["date", "weapon_type"], how="left"
            )

        return df

    def _fetch_player_counts(self) -> pd.DataFrame:
        if hasattr(self, "_player_counts_cache") and self._player_counts_cache is not None:
            return self._player_counts_cache
        archive_dir = Path(__file__).parent.parent.parent / "price-archive"
        pc_path = archive_dir / "player-counts-*.parquet"
        if not archive_dir.exists() or not list(archive_dir.glob("player-counts-*.parquet")):
            logger.warning("No player-counts Parquet files found — using empty DataFrame")
            self._player_counts_cache = pd.DataFrame(columns=["day", "mean_players", "peak_players", "reading_count"])
            return self._player_counts_cache
        try:
            import duckdb
            con = duckdb.connect()
            rows = con.sql(f"""
                SELECT day, mean_players, peak_players, min_players, reading_count, last_players
                FROM read_parquet('{pc_path}')
                ORDER BY day
            """).fetchall()
            con.close()
            df = pd.DataFrame(rows, columns=["day", "mean_players", "peak_players", "min_players", "reading_count", "last_players"])
            df["day"] = pd.to_datetime(df["day"])
            df["date"] = df["day"].dt.date
            logger.info(f"  player counts: {len(df)} days loaded")
            self._player_counts_cache = df
            return df
        except Exception as e:
            logger.warning(f"Failed to load player counts from Parquet: {e}")
            self._player_counts_cache = pd.DataFrame(columns=["day", "mean_players", "peak_players", "reading_count"])
            return self._player_counts_cache

    def _add_player_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pc = self._fetch_player_counts()
        if pc.empty:
            for col in ["players_mean", "players_peak", "players_min",
                         "players_last", "players_readings",
                         "players_change_1d", "players_change_7d",
                         "players_z_score_30d", "players_ma7"]:
                df[col] = 0.0
            return df
        df = df.merge(pc, on="date", how="left")
        df["players_mean"] = df["mean_players"].fillna(0).astype(float)
        df["players_peak"] = df["peak_players"].fillna(0).astype(float)
        df["players_min"] = df["min_players"].fillna(0).astype(float)
        df["players_last"] = df["last_players"].fillna(0).astype(float)
        df["players_readings"] = df["reading_count"].fillna(0).astype(float)
        df = df.drop(columns=["mean_players", "peak_players", "min_players",
                              "last_players", "reading_count", "day"], errors="ignore")
        df = df.sort_values("date")
        df["players_change_1d"] = df["players_mean"] - df["players_mean"].shift(1)
        df["players_change_7d"] = df["players_mean"] - df["players_mean"].shift(7)
        df["players_ma7"] = df["players_mean"].rolling(7, min_periods=1).mean()
        rolling_std = df["players_mean"].rolling(30, min_periods=1).std().replace(0, np.nan)
        df["players_z_score_30d"] = ((df["players_mean"] - df["players_mean"].rolling(30, min_periods=1).mean()) / rolling_std).fillna(0)
        df["players_mean_ratio_7d"] = (df["players_mean"] / df["players_ma7"].replace(0, np.nan)).fillna(1.0)
        logger.info("  player count features added")
        return df

    def _add_item_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        meta = self._fetch_item_metadata()
        if meta.empty:
            return df
        df = df.merge(meta, on="item_id", how="left")
        df["type"] = df["type"].fillna("unknown")
        type_dummies = pd.get_dummies(df["type"], prefix="item_type").astype(int)
        for t in ["skin", "sticker", "case", "graffiti", "musickit", "unknown"]:
            col = f"item_type_{t}"
            if col not in type_dummies.columns:
                type_dummies[col] = 0
        df = pd.concat([df, type_dummies], axis=1)
        df = df.drop(columns=["type"])
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(df["date"])
        dow = dates.dt.dayofweek
        month = dates.dt.month
        doy = dates.dt.dayofyear
        df["day_of_week"] = dow
        df["month"] = month
        df["quarter"] = dates.dt.quarter
        df["day_of_year"] = doy
        df["is_weekend"] = (dow >= 5).astype(int)

        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 366)
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

        # Add relevance-weighted event signals
        # Different item types respond differently to each event type.
        has_identity = all(c in df.columns for c in
                           ["is_sticker", "is_case", "is_glove", "is_knife"])
        if has_identity:
            is_skin = (
                1 - df["is_sticker"] - df["is_case"] - df["is_music_kit"]
                - df["is_graffiti"] - df["is_charm"] - df["is_patch"]
                - df["is_capsule"]
            ).clip(lower=0).astype(float)

            relevance_map = {
                "major": (
                    df["is_sticker"].astype(float) * 1.0
                    + df["is_case"].astype(float) * 0.3
                    + df["is_capsule"].astype(float) * 0.6
                ),
                "operation": (
                    df["is_case"].astype(float) * 1.0
                    + is_skin * 0.3
                ),
                "case_drop": (
                    df["is_case"].astype(float) * 1.0
                    + is_skin * 0.5
                ),
                "update": is_skin * 0.5,
                "game_update": is_skin * 0.3,
            }
            for et in event_types:
                raw_col = f"event_decay_{et}"
                if raw_col in df.columns:
                    weight = relevance_map.get(et, pd.Series(1.0, index=df.index))
                    df[f"event_decay_{et}_weighted"] = df[raw_col] * weight

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

        # Market regime: refined 5-state regime with duration tracking
        if "market_return_30d" in df.columns:
            market_ret_median = df.groupby("date")["market_return_30d"].transform("median")
            df["market_regime_crash"] = (market_ret_median < -10).astype(int)
            df["market_regime_bear"] = ((market_ret_median >= -10) & (market_ret_median < -3)).astype(int)
            df["market_regime_range"] = ((market_ret_median >= -3) & (market_ret_median <= 3)).astype(int)
            df["market_regime_bull"] = ((market_ret_median > 3) & (market_ret_median <= 10)).astype(int)
            df["market_regime_mania"] = (market_ret_median > 10).astype(int)

            # Market regime duration: consecutive days in same regime
            regime_cols = ["market_regime_crash", "market_regime_bear",
                           "market_regime_range", "market_regime_bull",
                           "market_regime_mania"]
            combined = pd.DataFrame(index=df.index, dtype=int)
            combined["regime_id"] = 0
            for i, col in enumerate(regime_cols):
                if col in df.columns:
                    combined.loc[df[col] == 1, "regime_id"] = i + 1
            # Count consecutive same-regime days per item
            regime_changes = (combined["regime_id"] != combined["regime_id"].groupby(df["item_id"]).shift(1)).astype(int)
            df["market_regime_duration_days"] = regime_changes.groupby(df["item_id"]).cumsum().groupby(
                [df["item_id"], regime_changes.cumsum()]
            ).cumcount() + 1

        # Market return percentile vs rolling 365-day history
        if "market_return_30d" in df.columns:
            def _pct_rank_365(series):
                return series.rolling(365, min_periods=30).apply(
                    lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1),
                    raw=False
                )
            df["market_return_30d_percentile"] = df.groupby("item_id")["market_return_30d"].transform(
                lambda x: x.rolling(365, min_periods=30).apply(
                    lambda s: (s.iloc[-1] > s[:-1]).sum() / max(len(s) - 1, 1),
                    raw=False
                )
            )

        return df

    def _add_item_identity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add item identity features (is_stattrak, is_knife, quality_rank, etc.).

        Fetches item names from the DB and parses them into structured identity
        features. Items not found in the DB get default (0) values.
        """
        logger.info("Adding item identity features...")
        df = df.copy()

        # Fetch item_id → (name, type) mapping from the DB
        try:
            items_rows = self.db.execute(text("""
                SELECT item_id, name, type FROM items
            """)).fetchall()
            item_map = {r.item_id: r for r in items_rows}
        except Exception:
            logger.warning("  Could not fetch item names from DB; using default identity features")
            identity_cols = [
                "is_stattrak", "is_souvenir", "is_knife", "is_glove",
                "is_sticker", "is_case", "is_capsule", "is_agent",
                "is_music_kit", "is_graffiti", "is_charm", "is_patch",
                "quality_rank",
            ]
            for col in identity_cols:
                df[col] = 0
            return df

        # Build identity features for each unique item
        identity_cache = {}
        for item_id in df["item_id"].unique():
            item_row = item_map.get(str(item_id))
            if item_row is None:
                identity_cache[item_id] = {
                    "is_stattrak": 0, "is_souvenir": 0, "is_knife": 0,
                    "is_glove": 0, "is_sticker": 0, "is_case": 0,
                    "is_capsule": 0, "is_agent": 0, "is_music_kit": 0,
                    "is_graffiti": 0, "is_charm": 0, "is_patch": 0,
                    "quality_rank": 0,
                }
                continue

            name = item_row.name
            db_type = item_row.type
            parsed = parse_item_name(name) if name else {}

            use_type = db_type or "skin"
            identity_cache[item_id] = {
                "is_stattrak": int(parsed.get("is_stattrak", False)),
                "is_souvenir": int(parsed.get("is_souvenir", False)),
                "is_knife": int(parsed.get("is_knife", False)),
                "is_glove": int(parsed.get("is_glove", False)),
                "is_sticker": int(use_type == "sticker" or parsed.get("is_sticker", False)),
                "is_case": int(use_type == "case" or parsed.get("is_case", False)),
                "is_capsule": int(parsed.get("is_capsule", False)),
                "is_agent": int(parsed.get("is_agent", False)),
                "is_music_kit": int(use_type == "musickit" or parsed.get("is_music_kit", False)),
                "is_graffiti": int(use_type == "graffiti" or parsed.get("is_graffiti", False)),
                "is_charm": int(parsed.get("is_charm", False)),
                "is_patch": int(parsed.get("is_patch", False)),
                "quality_rank": int(parsed.get("quality_rank", 0)),
            }

        # Map identity features onto the dataframe
        if not identity_cache:
            logger.warning("  No identity features computed (empty cache)")
            identity_cols = [
                "is_stattrak", "is_souvenir", "is_knife", "is_glove",
                "is_sticker", "is_case", "is_capsule", "is_agent",
                "is_music_kit", "is_graffiti", "is_charm", "is_patch",
                "quality_rank",
            ]
            for col in identity_cols:
                df[col] = 0
            return df

        identity_df = pd.DataFrame.from_dict(identity_cache, orient="index")
        identity_df.index.name = "item_id"
        identity_df = identity_df.reset_index()

        df = df.merge(identity_df, on="item_id", how="left")

        for col in identity_cache[next(iter(identity_cache))].keys():
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)

        logger.info("  Added item identity features")
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

        # Find all pairs with correlation above threshold using the stacked
        # (melted) correlation matrix. Each unordered pair appears twice in
        # the stack; we deduplicate by index order.
        stacked = corr.stack()
        high_pairs = stacked[
            (stacked > self.PRUNE_CORRELATION_THRESHOLD)
            & (stacked.index.get_level_values(0) != stacked.index.get_level_values(1))
        ]

        to_drop = set()
        feature_index = {name: i for i, name in enumerate(self.feature_cols)}
        for feat_a, feat_b in high_pairs.index:
            if feat_a in to_drop or feat_b in to_drop:
                continue
            # Keep the earlier feature (lower index), drop the later one
            if feature_index[feat_a] < feature_index[feat_b]:
                to_drop.add(feat_b)
            else:
                to_drop.add(feat_a)

        pruned = [c for c in self.feature_cols if c not in to_drop]
        if pruned != self.feature_cols:
            logger.info(
                f"  Pruned {len(self.feature_cols) - len(pruned)} features "
                f"(corr>{self.PRUNE_CORRELATION_THRESHOLD}): "
                f"{len(pruned)} remaining"
            )
        return pruned

    def _validate_feature_groups(
        self, X_val: np.ndarray, y_val: np.ndarray,
        feature_names: List[str], horizon: int = 7,
        n_shuffles: int = 20, min_drop_pp: float = 0.5,
    ) -> Dict[str, Dict]:
        """Validate feature groups via permutation importance.

        For each feature group, shuffles its columns on the validation set
        and measures the directional accuracy drop vs the unshuffled baseline.
        Groups with drops below `min_drop_pp` are flagged as non-contributing.

        Uses the p50 (median) quantile model for the given horizon.

        Returns:
            {group_name: {"drop_pp": float, "base_acc": float,
                          "shuffled_acc": float, "passed": bool,
                          "feature_count": int, "features": [str]}}
        """
        model_key = (horizon, 0.5)
        if model_key not in self.models:
            return {}

        model = self.models[model_key]
        if isinstance(model, list):
            model = model[0]

        groups: Dict[str, List[str]] = {}
        for i, name in enumerate(feature_names):
            g = _feature_group(name)
            groups.setdefault(g, []).append(name)

        col_to_idx = {name: i for i, name in enumerate(feature_names)}
        group_indices: Dict[str, List[int]] = {}
        for g, feats in groups.items():
            idxs = [col_to_idx[f] for f in feats if f in col_to_idx]
            if idxs:
                group_indices[g] = idxs

        p50_idx = np.squeeze(model.predict(X_val))
        base_acc = np.mean((p50_idx > 0) == (y_val > 0)) * 100

        results = {}
        for group, idxs in group_indices.items():
            shuffled_accs = []
            for _ in range(n_shuffles):
                X_shuf = X_val.copy()
                for i in idxs:
                    RNG.shuffle(X_shuf[:, i])
                p50_shuf = np.squeeze(model.predict(X_shuf))
                acc = np.mean((p50_shuf > 0) == (y_val > 0)) * 100
                shuffled_accs.append(acc)

            mean_shuf = np.mean(shuffled_accs)
            drop_pp = base_acc - mean_shuf
            passed = drop_pp >= min_drop_pp
            results[group] = {
                "drop_pp": round(drop_pp, 2),
                "base_acc": round(base_acc, 2),
                "shuffled_acc": round(mean_shuf, 2),
                "passed": passed,
                "feature_count": len(idxs),
                "features": groups[group],
            }

            status = "PASS" if passed else "WARN"
            logger.info(
                f"  [feat group] {group}: {drop_pp:+.2f}pp when shuffled "
                f"({base_acc:.1f}% -> {mean_shuf:.1f}%) "
                f"[{status}]"
            )

        return results

    def _compute_cv_splits(self, sorted_dates):
        """Compute expanding-window CV fold boundaries.

        Returns list of (train_date_list, val_date_list) tuples.
        Each fold trains on an expanding window and validates on a fixed-width
        window of VALIDATION_WINDOW_DAYS at the end.
        """
        val_window = self.VALIDATION_WINDOW_DAYS  # 21 days
        step = self.CV_STEP_DAYS  # 120 days
        min_train = self.CV_MIN_TRAIN_DAYS

        folds = []
        for end in range(min_train, len(sorted_dates) - val_window + 1, step):
            train_d = sorted_dates[:end]
            val_d = sorted_dates[end:end + val_window]
            if len(val_d) >= 7:
                folds.append((list(train_d), list(val_d)))
        return folds

    def _optuna_search_params(self, X_train, y_train, X_val, y_val,
                               quantile: float = 0.5) -> Dict[str, Any]:
        """Bayesian hyperparameter search via Optuna.

        Searches over 6 key params using TPE pruning, with early
        termination of unpromising trials (median pruner).
        """
        import optuna

        n_trials = 15

        def objective(trial):
            params = {
                "objective": "quantile",
                "alpha": quantile,
                "metric": "quantile",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "n_jobs": -1,
                "random_state": 42,
                "num_leaves": trial.suggest_int("num_leaves", 15, 63, step=8),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0, step=0.5),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0, step=0.5),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 30, step=5),
                "min_gain_to_split": 0.1,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "bagging_freq": 5,
            }
            dtrain = lgb.Dataset(X_train, y_train)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain)
            model = lgb.train(
                params, dtrain,
                num_boost_round=200,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            return model.best_score["valid_0"]["quantile"]

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_trial
        best_params = {
            "num_leaves": best.params["num_leaves"],
            "learning_rate": best.params["learning_rate"],
            "lambda_l1": best.params["lambda_l1"],
            "lambda_l2": best.params["lambda_l2"],
            "max_depth": best.params["max_depth"],
            "min_data_in_leaf": best.params["min_data_in_leaf"],
        }

        logger.info(
            f"  Optuna search ({n_trials} trials): best loss={best.value:.6f} "
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
        df = self._add_item_identity_features(df)
        df = self._add_event_features(df, events_df)
        df = self._add_item_metadata_features(df)
        df = self._add_supply_side_features(df)
        return df

    # ------------------------------------------------------------------
    # Target preparation
    # ------------------------------------------------------------------

    def prepare_targets(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        logger.info(f"Preparing {horizon}d targets...")
        df = df.sort_values(["item_id", "date"])

        # Date-based target lookup: find the price exactly `horizon` calendar
        # days later, rather than shifting rows. Row-based shift gives
        # incorrect horizons when item data has gaps.
        df["_date_dt"] = pd.to_datetime(df["date"])

        # Shift each row's price BACKWARD by horizon so it becomes the
        # target for the row `horizon` days earlier. E.g., the row at
        # date=Jan 8 with price=P becomes target=P for the row at date=Jan 1.
        future = df[["item_id", "_date_dt", "price"]].copy()
        future.columns = ["item_id", "date", f"target_{horizon}d"]
        future["date"] = future["date"] - pd.Timedelta(days=horizon)
        future["date"] = future["date"].dt.date

        df = df.merge(future, on=["item_id", "date"], how="left")
        df = df.drop(columns=["_date_dt"])

        df[f"target_return_{horizon}d"] = (
            (df[f"target_{horizon}d"] - df["price"]) / df["price"].replace(0, np.nan) * 100
        )
        return df

    # ------------------------------------------------------------------
    # Build training dataset
    # ------------------------------------------------------------------

    def build_training_data(self, days_back: int = 365,
                            backfilled_only: bool = False) -> pd.DataFrame:
        price_df = self.fetch_price_history(days_back=days_back, backfilled_only=backfilled_only)
        events_df = self.fetch_events()

        df = self.engineer_features(price_df, events_df)

        # Add cross-sectional (market-regime) features
        # Add weapon-type cross-sectional features (supply-side group)
        df = self._add_weapon_type_cross_sectional_features(df)

        df = self._add_cross_sectional_features(df)

        # Add player count features
        df = self._add_player_count_features(df)

        # Define feature columns (exclude metadata and target columns)
        exclude = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date"}
        exclude |= {f"target_{h}d" for h in self.HORIZONS}
        exclude |= {f"target_return_{h}d" for h in self.HORIZONS}

        self.feature_cols = [c for c in df.columns if c not in exclude
                             and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]

        # Prune highly correlated features to reduce noise
        self.feature_cols = self._prune_features(df)

        # Downcast features to float32 to halve feature matrix memory
        for col in self.feature_cols:
            if col in df.columns and df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        logger.info(f"Feature matrix: {len(df):,} rows, {len(self.feature_cols)} features")
        logger.info(f"Features: {self.feature_cols}")

        return df

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

        df = self.build_training_data(days_back=730, backfilled_only=True)

        for horizon in self.HORIZONS:
            tdf = self.prepare_targets(df, horizon)

            # Drop NaN targets (use percentage return as primary target)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values("date")

            for attempt in range(2):
                if attempt == 1:
                    logger.info(f"  Retry {horizon}d — pruned feature groups")

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

                if attempt == 0:
                    logger.info(f"  {horizon}d: {len(train_set)} train, {len(val_set)} val")

                # Replace INF with NaN before imputation (division-by-zero artifacts)
                X_train_pre = train_set[self.feature_cols].replace([np.inf, -np.inf], np.nan)
                feature_medians = X_train_pre.median()
                self.feature_medians = feature_medians
                X_train = X_train_pre.fillna(feature_medians)
                y_train = train_set[f"target_return_{horizon}d"]

                X_val = val_set[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(feature_medians)
                y_val = val_set[f"target_return_{horizon}d"]

                per_quantile_params = {}
                for q in self.QUANTILES:
                    logger.info(f"  Searching hyperparams for {horizon}d p{int(q*100)} (Optuna)...")
                    best_params = self._optuna_search_params(X_train, y_train, X_val, y_val, quantile=q)
                    logger.info(f"Training {horizon}d p{int(q*100)} ensemble ({self.N_ENSEMBLES} members)...")

                    base_params = {
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "boosting_type": "gbdt",
                        "min_gain_to_split": 0.1,
                        "feature_fraction": 0.7,
                        "bagging_fraction": 0.7,
                        "bagging_freq": 5,
                        "verbosity": -1,
                        "n_jobs": -1,
                    }
                    # Merge Optuna results into base params
                    if best_params:
                        for k in ("num_leaves", "learning_rate", "lambda_l1", "lambda_l2",
                                  "max_depth", "min_data_in_leaf"):
                            if k in best_params:
                                base_params[k] = best_params[k]
                    else:
                        base_params["num_leaves"] = 31
                        base_params["learning_rate"] = 0.03
                        base_params["lambda_l1"] = 0.5
                        base_params["lambda_l2"] = 0.5
                        base_params["max_depth"] = 5
                        base_params["min_data_in_leaf"] = 15

                    per_quantile_params[q] = dict(base_params)

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

                # Expanding-window CV evaluation using the best hyperparams
                oof_records, cv_metrics = self._cv_evaluate_horizon(tdf, horizon, per_quantile_params)

                # Calibrate confidence thresholds on pooled OOF predictions from CV
                # (more robust than a single 21-day holdout)
                if oof_records:
                    records_df = pd.DataFrame(oof_records)
                    logger.info(f"  Calibrating on {len(records_df)} pooled OOF predictions "
                                f"({len(cv_metrics)} folds)")
                    self._calibrate_confidence(horizon=horizon, records_df=records_df)
                else:
                    logger.warning(f"  CV produced no OOF records; falling back to single-split calibration")
                    self._calibrate_confidence(horizon=horizon, X_val=X_val, y_val=y_val, val_set=val_set)

                # Log CV fold-level metrics
                if cv_metrics:
                    fold_accs = [m["directional_accuracy"] for m in cv_metrics]
                    mean_acc = np.mean(fold_accs)
                    std_acc = np.std(fold_accs)
                    logger.info(f"  CV ({len(cv_metrics)} folds): "
                                f"mean={mean_acc:.1f}% sd={std_acc:.1f}% "
                                f"range=[{min(fold_accs):.1f}%, {max(fold_accs):.1f}%]")
                    self.cv_results[horizon] = {
                        "fold_count": len(cv_metrics),
                        "per_fold": cv_metrics,
                        "mean_dir_acc": round(mean_acc, 1),
                        "std_dir_acc": round(std_acc, 1) if len(fold_accs) > 1 else 0,
                        "min_dir_acc": round(min(fold_accs), 1),
                        "max_dir_acc": round(max(fold_accs), 1),
                    }

                # Validate feature groups: permutation test on the held-out set
                need_retrain = False
                try:
                    X_val_np = X_val.values if hasattr(X_val, "values") else X_val
                    y_val_np = y_val.values if hasattr(y_val, "values") else y_val
                    fv = self._validate_feature_groups(
                        X_val_np, y_val_np, self.feature_cols,
                        horizon=horizon, n_shuffles=20, min_drop_pp=0.5,
                    )
                    self.cv_results[horizon]["feature_validation"] = fv
                    failed = [g for g, r in fv.items() if not r["passed"]]
                    if failed:
                        if self.prune_failed_groups and attempt == 0:
                            failed_cols = set()
                            for g in failed:
                                for f in fv[g]["features"]:
                                    failed_cols.add(f)
                            pre_count = len(self.feature_cols)
                            self.feature_cols = [c for c in self.feature_cols
                                                  if c not in failed_cols]
                            logger.warning(
                                f"  Pruned {len(failed_cols)} features from non-causal groups "
                                f"{failed} ({pre_count} -> {len(self.feature_cols)}). Retraining."
                            )
                            need_retrain = True
                        else:
                            logger.warning(
                                f"  Feature groups with no causal signal: {failed}. "
                                f"({'Auto-prune disabled' if not self.prune_failed_groups else 'Already re-trained.'})"
                            )
                except Exception as e:
                    logger.warning(f"  Feature validation skipped: {e}")

                if not need_retrain:
                    break

            del tdf

        del df

        self.save_models()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, item_ids: List[int] = None) -> pd.DataFrame:
        logger.info("Generating forecasts...")

        price_df = self.fetch_price_history(days_back=365, backfilled_only=True)

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

        # Add weapon-type cross-sectional features (same as training)
        df = self._add_weapon_type_cross_sectional_features(df)

        # Add cross-sectional features (same as training)
        df = self._add_cross_sectional_features(df)

        # Add player count features (same as training)
        df = self._add_player_count_features(df)

        # Align features with training columns (add missing, drop extras)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_cols + [c for c in df.columns if c not in self.feature_cols]]

        # Use median of last 3 days for current_price to filter spike artifacts.
        # Prediction features still come from the latest row (rolling windows
        # inside the features already provide smoothing), but the base price
        # used to convert percentage returns into dollar amounts is more
        # robust when averaged over a short window.
        df = df.sort_values(["item_id", "date"])

        # Compute the median price over the last 3 days per item
        last_3 = df.groupby("item_id").tail(3)
        smoothed_price = last_3.groupby("item_id")["price"].median().to_frame("_smoothed_price")

        # Detect outliers: log when latest price deviates > 10% from 3d median
        latest_rows = df.groupby("item_id").last().reset_index()
        latest_rows = latest_rows.merge(smoothed_price, left_on="item_id", right_index=True, how="left")
        outlier_mask = (latest_rows["_smoothed_price"] > 0) & (
            abs(latest_rows["price"] - latest_rows["_smoothed_price"]) / latest_rows["_smoothed_price"] > 0.10
        )
        n_outliers = outlier_mask.sum()
        if n_outliers:
            logger.warning(f"  {n_outliers} items have latest price >10% from 3d median — using smoothed price")
        latest_rows["price"] = latest_rows["_smoothed_price"].fillna(latest_rows["price"])
        latest_rows = latest_rows.drop(columns=["_smoothed_price"])

        if item_ids:
            latest_rows = latest_rows[latest_rows["item_id"].isin(item_ids)]

        # Replace INF with NaN (division-by-zero artifacts), then median imputation
        latest_clean = latest_rows[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        if not self.feature_medians.empty:
            medians_aligned = self.feature_medians.reindex(self.feature_cols)
            medians_aligned = medians_aligned.where(medians_aligned.notna(), latest_clean.median())
            X_batch = latest_clean.fillna(medians_aligned)
        else:
            feature_medians = latest_clean.median()
            X_batch = latest_clean.fillna(feature_medians)

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
                all_preds = []

                # LGB predictions
                key = (horizon, q)
                if key in self.models:
                    ensemble = self.models[key]
                    if isinstance(ensemble, list):
                        for m in ensemble:
                            all_preds.append(m.predict(X_batch))
                    else:
                        all_preds.append(ensemble.predict(X_batch))

                if all_preds:
                    preds[q] = np.mean(all_preds, axis=0)

            if len(preds) != 3:
                continue

            # Models predict percentage returns. Convert back to price levels.
            # preds are return percentages (e.g., 5.0 means +5%).
            p10_ret = preds[0.1]
            p50_ret = preds[0.5]
            p90_ret = preds[0.9]

            # Ensure quantile monotonicity without scrambling model identities.
            # For items where quantiles cross (p10 > p50 or p50 > p90), we
            # preserve a non-zero interval width by imputing the average
            # half-width from well-behaved items. This avoids collapsing the
            # prediction interval to a point (which clamping alone would do).
            crossing_mask = (p10_ret > p50_ret) | (p50_ret > p90_ret)
            non_crossing = ~crossing_mask

            low_ret_arr = np.where(crossing_mask, p50_ret, p10_ret)
            high_ret_arr = np.where(crossing_mask, p50_ret, p90_ret)

            if non_crossing.any():
                # Average half-width of non-crossing items (in percentage points)
                avg_half_width = np.mean([
                    np.mean(p50_ret[non_crossing] - p10_ret[non_crossing]),
                    np.mean(p90_ret[non_crossing] - p50_ret[non_crossing]),
                ])
                # Apply symmetric interval to crossing items using the average width
                if avg_half_width > 0:
                    low_ret_arr[crossing_mask] = p50_ret[crossing_mask] - avg_half_width
                    high_ret_arr[crossing_mask] = p50_ret[crossing_mask] + avg_half_width

            low_ret_arr = np.minimum(low_ret_arr, p50_ret)
            high_ret_arr = np.maximum(high_ret_arr, p50_ret)
            mid_ret_arr = p50_ret

            # Diagnostic: log crossing rate
            crossing_rate = np.mean(crossing_mask)
            if crossing_rate > 0.01:
                logger.warning(f"  Quantile crossing rate: {crossing_rate:.3f} "
                               f"(imputed avg_half_width={avg_half_width:.2f}pp)")

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
                    "confidence": self._compute_confidence(price_mid, price_low, price_high,
                                                            current_price, horizon=horizon),
                }

        result_df = pd.DataFrame([r for r in agg.values() if r["forecasts"]])
        if not result_df.empty:
            result_df = self._sanitize_forecasts(result_df)
        logger.info(f"  Forecasts generated for {len(result_df)} items")
        return result_df

    def _sanitize_forecasts(self, result_df: pd.DataFrame) -> pd.DataFrame:
        for h in self.HORIZONS:
            for key in ["low", "mid", "high"]:
                vals = np.array([r["forecasts"].get(h, {}).get(key, np.nan)
                                 for r in result_df.to_dict("records")])
                mask_bad = ~np.isfinite(vals) | (vals <= 0)
                if mask_bad.any():
                    current_prices = result_df["current_price"].values
                    vals[mask_bad] = current_prices[mask_bad]
                    for i in np.where(mask_bad)[0]:
                        cf = result_df.iloc[i]["forecasts"].get(h)
                        if cf is None:
                            continue
                        cf[key] = float(vals[i])
                        cf["direction"] = "flat"
                        cf["confidence"] = "low"
        if "volume" in result_df.columns:
            zero_vol = result_df["volume"].fillna(0) == 0
            if zero_vol.any():
                for h in self.HORIZONS:
                    for i in np.where(zero_vol.values)[0]:
                        cf = result_df.iloc[i]["forecasts"].get(h)
                        if cf and cf.get("confidence") == "high":
                            cf["confidence"] = "low"
        return result_df

    def predict_single(self, item_id: int) -> Dict[str, Any]:
        results = self.predict(item_ids=[item_id])
        if results.empty:
            return {}
        return results.iloc[0].to_dict()

    def _get_ensemble_prediction(self, horizon, q, X):
        """Get averaged prediction from LGB ensemble."""
        all_preds = []
        key = (horizon, q)

        # LGB
        if key in self.models:
            ensemble = self.models[key]
            if isinstance(ensemble, list):
                all_preds.extend(m.predict(X) for m in ensemble)
            else:
                all_preds.append(ensemble.predict(X))

        if not all_preds:
            return None
        return np.mean(all_preds, axis=0)

    def _cv_evaluate_horizon(self, tdf, horizon, per_quantile_params):
        """Run expanding-window CV for a single horizon.

        Trains a single model (no ensemble) per fold using the best hyperparams
        found by Optuna, collects OOF predictions, and returns pooled records
        for confidence calibration plus fold-level metrics.

        Args:
            tdf: DataFrame with features + targets (from prepare_targets).
            horizon: Horizon in days.
            per_quantile_params: Dict {q: base_params} with best HP merged.

        Returns:
            (oof_records, fold_metrics) where oof_records is a list of dicts
            with range_pct/change_pct/hit for calibration, and fold_metrics
            is a list of per-fold accuracy dicts.
        """
        sorted_dates = sorted(tdf["date"].unique())
        splits = self._compute_cv_splits(sorted_dates)
        if len(splits) < 2:
            logger.warning(f"  Not enough data for CV ({len(splits)} fold{'s' if splits else 's'}); skipping")
            return [], []

        oof_records = []
        fold_metrics = []

        for fold_id, (train_dates, val_dates) in enumerate(splits):
            train_df = tdf[tdf["date"].isin(train_dates)]
            val_df = tdf[tdf["date"].isin(val_dates)]

            if len(val_df) < 50:
                continue

            fold_p50 = None
            fold_p10 = None
            fold_p90 = None

            X_train_pre = train_df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
            fold_medians = X_train_pre.median()
            X_train = X_train_pre.fillna(fold_medians)
            y_train = train_df[f"target_return_{horizon}d"]

            X_val = val_df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(fold_medians)
            y_val = val_df[f"target_return_{horizon}d"]

            for q in self.QUANTILES:
                lgb_preds = []

                # LGB
                params = per_quantile_params.get(q, {}).copy()
                params["objective"] = "quantile"
                params["alpha"] = q
                params["metric"] = "quantile"
                params["verbosity"] = -1
                params["n_jobs"] = -1
                params["random_state"] = 42

                dtrain = lgb.Dataset(X_train, y_train)
                dval = lgb.Dataset(X_val, y_val, reference=dtrain)
                model = lgb.train(
                    params, dtrain,
                    num_boost_round=200,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                lgb_preds.append(model.predict(X_val))

                pred = lgb_preds[0]
                if q == 0.5:
                    fold_p50 = pred
                elif q == 0.1:
                    fold_p10 = pred
                elif q == 0.9:
                    fold_p90 = pred

            if fold_p50 is None or fold_p10 is None or fold_p90 is None:
                continue

            # Quantile crossing fix (same logic as _calibrate_confidence)
            crossing_mask = (fold_p10 > fold_p50) | (fold_p50 > fold_p90)
            non_crossing = ~crossing_mask
            low_pred = np.minimum(fold_p10, fold_p50)
            high_pred = np.maximum(fold_p90, fold_p50)
            if non_crossing.any():
                avg_hw = np.mean([
                    np.mean(fold_p50[non_crossing] - fold_p10[non_crossing]),
                    np.mean(fold_p90[non_crossing] - fold_p50[non_crossing]),
                ])
                if avg_hw > 0:
                    low_pred[crossing_mask] = fold_p50[crossing_mask] - avg_hw
                    high_pred[crossing_mask] = fold_p50[crossing_mask] + avg_hw
            low_pred = np.minimum(low_pred, fold_p50)
            high_pred = np.maximum(high_pred, fold_p50)

            current_prices = val_df["price"].values
            actual_returns = y_val.values

            # Fold-level directional accuracy
            fold_hits = 0
            for i in range(len(val_df)):
                actual_ret = float(actual_returns[i])
                mid_ret = float(fold_p50[i])
                actual_dir = "up" if actual_ret > 0 else "down" if actual_ret < 0 else "flat"
                pred_dir = "up" if mid_ret > 0 else "down" if mid_ret < 0 else "flat"
                if pred_dir == actual_dir:
                    fold_hits += 1

            fold_acc = round(fold_hits / len(val_df) * 100, 1)
            fold_metrics.append({
                "fold": fold_id + 1,
                "train_start": str(train_dates[0]),
                "train_end": str(train_dates[-1]),
                "val_start": str(val_dates[0]),
                "val_end": str(val_dates[-1]),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "directional_accuracy": fold_acc,
            })

            # Build per-row records for pooled calibration
            for i in range(len(val_df)):
                mid_ret = float(fold_p50[i])
                low_ret = float(low_pred[i])
                high_ret = float(high_pred[i])
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

                oof_records.append({
                    "range_pct": range_pct,
                    "change_pct": change_pct,
                    "hit": hit,
                })

        return oof_records, fold_metrics

    def _calibrate_confidence(self, horizon, X_val=None, y_val=None, val_set=None,
                               records_df=None):
        """Calibrate confidence thresholds from validation set predictions.

        Binary confidence (high/low only):
        - Finds a range_pct threshold where high-confidence predictions achieve
          >= target_accuracy (default 80%) while maximizing coverage.
        - A change_pct floor prevents marking near-zero-move predictions as
          "high" confidence (they're correct but uninformative).

        If records_df is provided (from CV OOF predictions), it skips the
        prediction-computation step and uses the pre-built records directly.
        """
        target_accuracy = 0.80
        min_coverage_pct = 0.05

        if records_df is not None:
            df = records_df
        else:
            p50_pred = self._get_ensemble_prediction(horizon, 0.5, X_val.values)
            p10_pred = self._get_ensemble_prediction(horizon, 0.1, X_val.values)
            p90_pred = self._get_ensemble_prediction(horizon, 0.9, X_val.values)
            if p50_pred is None or p10_pred is None or p90_pred is None:
                return

            current_prices = val_set["price"].values
            actual_returns = y_val.values

            # Apply the same crossing-aware monotonicity correction for calibration
            crossing_mask_cal = (p10_pred > p50_pred) | (p50_pred > p90_pred)
            non_crossing_cal = ~crossing_mask_cal
            low_pred = np.minimum(p10_pred, p50_pred)
            high_pred = np.maximum(p90_pred, p50_pred)
            if non_crossing_cal.any():
                avg_half_width_cal = np.mean([
                    np.mean(p50_pred[non_crossing_cal] - p10_pred[non_crossing_cal]),
                    np.mean(p90_pred[non_crossing_cal] - p50_pred[non_crossing_cal]),
                ])
                if avg_half_width_cal > 0:
                    low_pred[crossing_mask_cal] = p50_pred[crossing_mask_cal] - avg_half_width_cal
                    high_pred[crossing_mask_cal] = p50_pred[crossing_mask_cal] + avg_half_width_cal
            low_pred = np.minimum(low_pred, p50_pred)
            high_pred = np.maximum(high_pred, p50_pred)

            records = []
            for i in range(len(val_set)):
                mid_ret = float(p50_pred[i])
                low_ret = float(low_pred[i])
                high_ret = float(high_pred[i])
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

        # Find the widest range_pct threshold where accuracy >= target.
        # Wider threshold = more items get "high" confidence → better coverage.
        # This is the opposite of the old approach (which maximized accuracy).
        thresholds = sorted(df["range_pct"].quantile([i / 20 for i in range(1, 20)]).unique())

        best_high_threshold = 0.15
        best_high_acc = 0.0
        best_high_coverage = 0

        for t in thresholds:
            subset = df[df["range_pct"] < t]
            if len(subset) >= max(50, len(df) * min_coverage_pct):
                acc = subset["hit"].mean()
                coverage = len(subset)
                # Pick the threshold with the most coverage that meets the target
                if acc >= target_accuracy and coverage > best_high_coverage:
                    best_high_threshold = t
                    best_high_acc = acc
                    best_high_coverage = coverage

        # If no threshold meets target_accuracy, fall back to the highest accuracy
        # that still covers at least 5% of items.
        if best_high_coverage == 0:
            for t in thresholds:
                subset = df[df["range_pct"] < t]
                if len(subset) >= max(50, len(df) * min_coverage_pct):
                    acc = subset["hit"].mean()
                    coverage = len(subset)
                    if coverage > best_high_coverage or (coverage == best_high_coverage and acc > best_high_acc):
                        best_high_threshold = t
                        best_high_acc = acc
                        best_high_coverage = coverage

        # Find change_pct threshold that filters near-zero-move predictions
        # (which are trivially correct but uninformative).
        best_change_threshold = 0.0
        best_change_coverage = best_high_coverage
        if best_high_coverage > 0:
            high_set = df[df["range_pct"] < best_high_threshold]
            if len(high_set) > 0:
                change_thresholds = sorted(high_set["change_pct"].quantile(
                    [i / 10 for i in range(1, 10)]).unique())
                for ct in change_thresholds:
                    subset = high_set[high_set["change_pct"] > ct]
                    if len(subset) >= max(20, len(high_set) * 0.3):
                        acc = subset["hit"].mean()
                        coverage = len(subset)
                        # Accept the change_pct threshold if accuracy stays >= target
                        if acc >= target_accuracy and coverage >= best_high_coverage * 0.5:
                            best_change_threshold = ct
                            best_change_coverage = coverage

        self.confidence_thresholds[horizon] = {
            "high_range": best_high_threshold,
            "high_change": best_change_threshold,
            "high_accuracy": round(best_high_acc * 100, 1),
        }

        th = self.confidence_thresholds[horizon]
        logger.info(
            f"  Calibrated (binary): high_range={th['high_range']:.3f} "
            f"(acc={th['high_accuracy']:.1f}%, "
            f"coverage={best_high_coverage / len(df) * 100:.1f}%)"
        )

    def _compute_confidence(self, mid: float, low: float, high: float, current: float,
                             horizon: int = 7) -> str:
        """Binary confidence: high (tight interval, non-trivial move) or low."""
        if mid == 0 or current == 0:
            return "low"
        range_pct = (high - low) / mid
        change_pct = abs(mid - current) / current

        # Use per-horizon calibrated thresholds if available, fall back to sensible defaults
        h_thresholds = self.confidence_thresholds.get(horizon, {})
        high_range = h_thresholds.get("high_range", 0.15)
        high_change = h_thresholds.get("high_change", 0.0)

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

        # Save feature columns, calibration thresholds, and imputation medians
        thresholds_serial = {}
        for horizon, th in self.confidence_thresholds.items():
            thresholds_serial[str(horizon)] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in th.items()
            }
        medians = self.feature_medians.to_dict() if not self.feature_medians.empty else {}
        medians_serial = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                          for k, v in medians.items()}
        # Compute feature importance for each horizon (average over quantiles/ensembles)
        feature_importance = {}
        for horizon in self.HORIZONS:
            all_importances = {}
            for q in self.QUANTILES:
                key = (horizon, q)
                if key not in self.models:
                    continue
                ensemble = self.models[key]
                if isinstance(ensemble, list):
                    for m in ensemble:
                        fi = self._get_feature_importance(m)
                        for _, row in fi.iterrows():
                            all_importances.setdefault(row["feature"], []).append(float(row["importance"]))
                else:
                    fi = self._get_feature_importance(ensemble)
                    for _, row in fi.iterrows():
                        all_importances.setdefault(row["feature"], []).append(float(row["importance"]))
            if all_importances:
                avg = {f: sum(v) / len(v) for f, v in all_importances.items()}
                sorted_fi = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:20]
                feature_importance[str(horizon)] = [{"feature": f, "importance": round(v, 4)} for f, v in sorted_fi]

        # Serialize CV results (convert int keys to str for JSON)
        cv_serial = {}
        for h_str, cvdata in self.cv_results.items():
            cv_serial[str(h_str)] = cvdata

        meta = {
            "feature_cols": self.feature_cols,
            "trained_at": str(self._now()),
            "confidence_thresholds": thresholds_serial,
            "feature_medians": medians_serial,
            "n_ensembles": self.N_ENSEMBLES,
            "feature_importance": feature_importance,
            "cv_results": cv_serial,
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
        self.feature_medians = pd.Series(meta.get("feature_medians", {}), dtype=np.float64)

        # Load per-horizon confidence thresholds (backward compat: treat flat dict as 7d)
        raw_thresholds = meta.get("confidence_thresholds", {})
        self.confidence_thresholds = {}
        if raw_thresholds:
            first_key = next(iter(raw_thresholds))
            if isinstance(first_key, str) and first_key.lstrip("-").isdigit():
                # New nested format: {"7": {"high_range": ..., ...}, "30": {...}}
                for h_str, th in raw_thresholds.items():
                    self.confidence_thresholds[int(h_str)] = th
            else:
                # Legacy flat format: {"high_range": ..., ...} — assign to all horizons
                for h in self.HORIZONS:
                    self.confidence_thresholds[h] = dict(raw_thresholds)
        n_ensembles = meta.get("n_ensembles", 1)

        for horizon in self.HORIZONS:
            for q in self.QUANTILES:
                # Load LGB models
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

        total_groups = len(self.models)
        logger.info(f"Loaded {total_groups} model groups from {self.model_dir}")
        return total_groups > 0

    def has_models(self) -> bool:
        return len(self.models) > 0
