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
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
from sqlalchemy import text
from models.item_parser import parse_item_name

# Residual stacking (Ridge on LightGBM residuals)
_sklearn_available = False
try:
    from sklearn.linear_model import Ridge
    _sklearn_available = True
except ImportError:
    Ridge = None  # type: ignore

logger = logging.getLogger(__name__)

RNG = np.random.RandomState(42)
DIRECTION_FLAT_TOLERANCE_PCT = 0.5


def _gpu_available() -> bool:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _feature_group(name: str) -> str:
    if any(name.startswith(p) for p in ("price_", "return_", "log_return_", "bb_",
                                         "rsi_", "macd_", "vol_", "trend_",
                                         "price_accel_", "autocorr_", "support_",
                                         "volume_", "vol_price_")):
        return "price_technicals"
    if name.startswith("supply_"):
        return "supply_depth"
    if any(name.startswith(p) for p in ("is_", "quality_rank", "rarity_")):
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
    REGIMES = ["bear", "range", "bull"]
    REGIME_RETURN_THRESHOLD_BEAR = -3.0   # market_return_30d < -3% → bear
    REGIME_RETURN_THRESHOLD_BULL = 3.0    # market_return_30d > 3% → bull
    N_ENSEMBLES = 3
    ENSEMBLE_SEEDS = [42, 73, 91]
    ENSEMBLE_FEATURE_FRACTIONS = [0.6, 0.7, 0.8]
    MAX_BIN = 63
    # Per-horizon boosting configuration.
    # GBDT used for all horizons (feature-ablation study showed 14d/30d
    # carry stronger signal than 3d/7d — DART's dropout overhead is not
    # justified). DART left available for future experiments.
    WEAK_HORIZONS = [14, 30]
    BOOSTING_TYPE_MAP = {3: "gbdt", 7: "gbdt", 14: "gbdt", 30: "gbdt"}
    DART_PARAMS = {
        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,
        "xgboost_dart_mode": False,
        "uniform_drop": False,
    }
    DART_NUM_BOOST_ROUND = 500
    # Horizon-specific feature exclusions based on ablation study
    # (2026-07-19-feature-contribution-by-horizon.md):
    # - Cross-sectional features actively harm 14d (−0.9pp) and 30d (−3.5pp)
    # - Event features harm 30d (−3.0pp) but help 14d (+2.0pp)
    HORIZON_EXCLUDED_GROUPS = {
        14: ["cross_sectional"],
        30: ["cross_sectional", "events"],
    }
    # Residual stacking: train a Ridge regression on LightGBM residuals
    # after ensemble training to correct systematic bias. Applied only
    # to weak horizons by default.
    STACK_RESIDUALS = True
    RESIDUAL_ALPHA = 5.0
    # Weight given to the previous day's forecast when smoothing/blending
    # current predictions to reduce daily direction flip-flopping.
    FORECAST_BLEND_WEIGHT = 0.15
    PRUNE_CORRELATION_THRESHOLD = 0.95
    # Expanding-window cross-validation
    CV_STEP_DAYS = 200        # Days between each fold's validation window
    CV_MIN_TRAIN_DAYS = 200   # Minimum unique dates before first validation fold

    ENGINEERED_CACHE_NAME = "engineered_data.parquet"

    def __init__(self, db_session, model_dir: str = None, prune_failed_groups: bool = True):
        self.db = db_session
        self.model_dir = model_dir or str(Path(__file__).parent / "saved_models")
        self.models: Dict[Tuple[int, float], lgb.Booster] = {}
        self.regime_models: Dict[Tuple[str, int, float], list] = {}
        self.regime_feature_cols: Dict[Tuple[int, str], List[str]] = {}
        self.feature_cols: List[str] = []
        self.residual_models: Dict[Tuple[int, float], Any] = {}
        self.prune_failed_groups = prune_failed_groups
        self.tuned_params: Dict[int, Dict[float, Dict[str, Any]]] = {}
        # Per-horizon confidence thresholds: {horizon: {"high_range": ..., "high_change": ..., "high_accuracy": ...}}
        self.confidence_thresholds: Dict[int, Dict[str, float]] = {}
        self.feature_medians: pd.Series = pd.Series(dtype=np.float64)
        # Conformal quantile calibration adjustment per horizon (in percentage-return space).
        # Maps horizon -> q_hat, the (1-α)(1+1/n) quantile of nonconformity scores
        # from CV out-of-fold predictions. Applied as: low -= q_hat, high += q_hat.
        self.conformal_calibration: Dict[int, float] = {}
        # Expanding-window CV results per horizon: {horizon: {fold_count, fold_accs, per_fold, ...}}
        self.cv_results: Dict[int, Dict] = {}
        # Event decay constants (grid-searchable per event type)
        self.horizon_feature_cols: Dict[int, List[str]] = {}
        self.event_decay_constants: Dict[str, float] = {
            "major": 60,
            "operation": 21,
            "case_drop": 14,
            "update": 7,
            "game_update": 7,
        }

    # ------------------------------------------------------------------
    # Regime switching
    # ------------------------------------------------------------------

    def _assign_regime_label(self, market_return_30d: float) -> str:
        """Assign a regime label based on market_return_30d.

        Thresholds:
          bear:  market_return_30d < -3%
          range: -3% <= market_return_30d <= 3%
          bull:  market_return_30d > 3%
        """
        if market_return_30d < self.REGIME_RETURN_THRESHOLD_BEAR:
            return "bear"
        elif market_return_30d <= self.REGIME_RETURN_THRESHOLD_BULL:
            return "range"
        else:
            return "bull"

    def _assign_regime_labels(self, df: pd.DataFrame) -> pd.Series:
        """Assign a regime label to each row based on market_return_30d.

        Returns a Series of regime strings ("bear", "range", "bull")
        indexed like df. Rows without market_return_30d get "range".
        """
        if "market_return_30d" not in df.columns:
            return pd.Series("range", index=df.index)
        labels = df["market_return_30d"].apply(
            lambda x: self._assign_regime_label(x) if pd.notna(x) else "range"
        )
        return labels

    def _detect_current_regime(self, df: pd.DataFrame) -> str:
        """Detect the current market regime from the latest engineered data.

        Uses the most recent row's market_return_30d (if available).
        Falls back to 'range' when undetermined.
        """
        if "market_return_30d" not in df.columns:
            return "range"
        latest_rets = df["market_return_30d"].dropna()
        if latest_rets.empty:
            return "range"
        return self._assign_regime_label(latest_rets.iloc[-1])

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
                    try:
                        slug_rows = self.db.execute(text("""
                            SELECT item_id FROM items WHERE is_backfilled = 1
                        """)).fetchall()
                        backfilled_slugs = {r[0] for r in slug_rows}
                        logger.info(f"  Backfilled items filter: {len(backfilled_slugs)} items from DB")
                    except Exception as e:
                        logger.warning(f"  Could not fetch backfilled items from DB, using all: {e}")
                        backfilled_slugs = {
                            r[0] for r in con.sql("""
                                SELECT DISTINCT item_slug
                                FROM read_parquet(?)
                            """, params=[str(archive_dir / "prices-*.parquet")]).fetchall()
                        }
                # Load Parquet files, handling schema mismatch (older files lack 'source' column)
                pq_files = sorted([str(p) for p in archive_dir.glob("prices-*.parquet")])
                pq_queries = []
                for pqf in pq_files:
                    cols = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()
                    col_names = {r[0] for r in cols}
                    if "source" in col_names:
                        pq_queries.append(f"SELECT * FROM read_parquet('{pqf}')")
                    else:
                        pq_queries.append(f"SELECT *, NULL::VARCHAR AS source FROM read_parquet('{pqf}')")
                union_sql = " UNION ALL BY NAME ".join(pq_queries)

                # Filter to backfilled slugs via a temp table JOIN (handles special chars safely)
                if backfilled_slugs is not None:
                    con.sql("CREATE TEMP TABLE _backfilled (slug VARCHAR)")
                    con.executemany("INSERT INTO _backfilled VALUES (?)",
                                    [(s,) for s in backfilled_slugs])
                    logger.info(f"  Filtering to {len(backfilled_slugs)} backfilled items via temp table")

                slug_join = "JOIN _backfilled b ON sub.item_slug = b.slug" if backfilled_slugs is not None else ""

                rows = con.sql(f"""
                    SELECT item_slug, day, mean_price AS price, volume, source
                    FROM ({union_sql}) sub
                    {slug_join}
                    WHERE day >= ?
                      AND (source IS NULL OR source NOT LIKE 'historical_fallback:%')
                    ORDER BY item_slug, day, source
                """, params=[cutoff]).fetchall()
                logger.info(f"  DuckDB query returned {len(rows):,} rows")
                # Column order MUST match the SELECT above
                # (item_slug, day, mean_price, volume, source).
                df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume", "source"])
                del rows  # free memory
                logger.info(f"  DataFrame created, converting types...")
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["date"] = df["timestamp"].dt.date
                # Some Parquet years store mean_price/volume as VARCHAR; the
                # glob union then coerces the whole column to string. Force
                # numeric so multi-source voting (np.median) works.
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                df = df.dropna(subset=["price"])
                logger.info(f"  After parsing: {len(df):,} rows, {df.item_id.nunique():,} items")
                n_before = len(df)
                n_sources_before = df["source"].nunique() if "source" in df.columns else 1
                logger.info(f"  Applying multi-source voting ({n_sources_before} sources)...")
                df = self._apply_multi_source_voting(df)
                n_after = len(df)
                logger.info(f"  {n_after:,} rows (Parquet, voted from {n_before:,} rows "
                            f"across {n_sources_before} sources), "
                            f"{df.item_id.nunique():,} items")
                if backfilled_only:
                    logger.info(f"  Filtered to STEAMCOMMUNITY-backfilled items only")
                return df
            finally:
                con.close()

        cutoff = self._now() - timedelta(days=days_back)
        rows = self.db.execute(text("""
            SELECT item_id, date(timestamp) AS day, source, AVG(price) AS price, SUM(volume) AS volume
            FROM price_history
            WHERE timestamp >= :cutoff
              AND source NOT LIKE 'synthetic_demo'
              AND source NOT LIKE 'historical_fallback:%'
            GROUP BY item_id, date(timestamp), source
            ORDER BY item_id, day
        """), {"cutoff": cutoff}).fetchall()
        if not rows:
            logger.info("  No DB price history rows found")
            return pd.DataFrame(columns=["item_id", "timestamp", "price", "volume", "date"])
        df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume", "source"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        n_before = len(df)
        n_sources_before = df["source"].nunique() if "source" in df.columns else 1
        df = self._apply_multi_source_voting(df)
        n_after = len(df)
        logger.info(f"  {n_after:,} rows (DB, voted from {n_before:,} rows "
                    f"across {n_sources_before} sources), "
                    f"{df.item_id.nunique():,} items")
        return df

    @staticmethod
    def _apply_multi_source_voting(df: pd.DataFrame) -> pd.DataFrame:
        """Apply multi-source outlier voting to get a single consensus price per item per day.

        For each (item_id, date) with >= 3 sources:
        - Compute median and std of source prices
        - Reject sources > 2 std from the median consensus
        - Use median of remaining sources
        For < 3 sources: use simple median.

        This is a data quality improvement — not a new feature dimension.
        Reducing noise in the target price improves ALL downstream features
        (lags, returns, rolling stats, Bollinger, RSI, MACD, volume features).
        """
        if "source" not in df.columns:
            unique_rows = df.groupby(["item_id", "date"]).size()
            already_single = not (unique_rows > 1).any()
            if already_single:
                return df
            return df.groupby(["item_id", "date"], as_index=False).agg(
                price=("price", "mean"),
                volume=("volume", "sum"),
            )

        # Speedup: split into single-source (≤1 row per item/date) and multi-source groups.
        # Single-source rows use fast groupby agg; multi-source uses the vote function.
        # This avoids calling a Python function millions of times.
        item_date_counts = df.groupby(["item_id", "date"], as_index=False).size()
        multi_groups = item_date_counts[item_date_counts["size"] > 1]
        if multi_groups.empty:
            return df.drop(columns=["source"], errors="ignore")

        multi_keys = multi_groups[["item_id", "date"]].drop_duplicates()
        is_multi = df.set_index(["item_id", "date"]).index.isin(
            multi_keys.set_index(["item_id", "date"]).index
        )

        single_df = df[~is_multi].copy()
        multi_df = df[is_multi].copy()

        def vote(group):
            prices = group["price"].values
            n_sources = len(prices)

            if n_sources >= 3:
                consensus = np.median(prices)
            else:
                consensus = np.median(prices)
                return pd.Series({
                    "price": consensus,
                    "volume": group["volume"].sum() if "volume" in group.columns else 0,
                })

            median = consensus
            std = np.std(prices, ddof=0)
            if std > 0:
                mask = np.abs(prices - median) <= 2.0 * std
                kept = mask.sum()
                if kept >= 1:
                    consensus = np.median(prices[mask])
                else:
                    consensus = median
            else:
                consensus = median

            return pd.Series({
                "price": consensus,
                "volume": group["volume"].sum() if "volume" in group.columns else 0,
            })

        # Fast path: single-source rows
        if len(single_df) > 0:
            result_single = single_df.groupby(["item_id", "date"], as_index=False).agg(
                price=("price", "mean"),
                volume=("volume", "sum"),
            )
        else:
            result_single = pd.DataFrame(columns=["item_id", "date", "price", "volume"])

        # Slow path: multi-source rows (small subset, typically <2% of groups)
        if len(multi_df) > 0:
            result_multi = multi_df.groupby(["item_id", "date"], as_index=False).apply(
                vote
            ).reset_index(drop=True)
        else:
            result_multi = pd.DataFrame(columns=["item_id", "date", "price", "volume"])

        result = pd.concat([result_single, result_multi], ignore_index=True)
        return result

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
        # MACD (vectorized — no lambda transforms)
        # =====================================================================
        df = df.sort_values(["item_id", "date"])
        # Compute EMAs per group using expanding EWMA on sorted data
        ewm12 = df.groupby("item_id")["price"].ewm(span=12, min_periods=12, adjust=False).mean()
        ewm26 = df.groupby("item_id")["price"].ewm(span=26, min_periods=26, adjust=False).mean()
        df["macd_line"] = ewm12.reset_index(level=0, drop=True) - ewm26.reset_index(level=0, drop=True)
        df["macd_signal"] = (
            df.groupby("item_id")["macd_line"]
            .ewm(span=9, min_periods=9, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
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
        """Add supply-side features: rarity ordinal and one-hot dummies.

        Permutation test confirmed strong causal signal (+10-12pp across all
        horizons) from rarity features. Weapon-type one-hot and cross-sectional
        features were removed — they showed zero causal signal.
        """
        logger.info("Adding supply-side features...")
        df = df.copy()

        meta = self._fetch_supply_metadata()
        if meta.empty:
            return df

        df = df.merge(meta, on="item_id", how="left")

        # Rarity ordinal (NaN → 0 for missing)
        df["rarity_ordinal"] = df["rarity_rank"].fillna(0).astype(int)

        # Rarity one-hot dummies
        rarity_cats = ["base", "consumer", "industrial", "milspec",
                       "restricted", "classified", "covert",
                       "high_grade", "remarkable", "exotic", "extraordinary"]
        for cat in rarity_cats:
            col = f"rarity_{cat}"
            df[col] = ((df["rarity"] == cat).astype(int))

        return df



    # ── Supply-depth features (sell_listings, skinport_quantity) ──────

    def _fetch_supply_snapshots(self) -> pd.DataFrame:
        """Load daily supply snapshots from DB.

        Returns DataFrame with columns:
          item_id, date, sell_listings, skinport_quantity
        """
        if hasattr(self, "_supply_snap_cache") and self._supply_snap_cache is not None:
            return self._supply_snap_cache

        try:
            rows = self.db.execute(text("""
                SELECT i.item_id, ss.snapshot_date AS date,
                       ss.sell_listings, ss.skinport_quantity
                FROM supply_snapshots ss
                JOIN items i ON i.id = ss.item_id
                ORDER BY i.item_id, ss.snapshot_date
            """)).fetchall()
            df = pd.DataFrame(rows, columns=["item_id", "date", "sell_listings", "skinport_quantity"])
            if df.empty:
                logger.info("  supply snapshots: empty")
                self._supply_snap_cache = df
                return df
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["sell_listings"] = df["sell_listings"].fillna(0).astype(int)
            df["skinport_quantity"] = df["skinport_quantity"].fillna(0).astype(int)
            logger.info(f"  supply snapshots: {len(df):,} rows, {df.item_id.nunique():,} items")
            self._supply_snap_cache = df
            return df
        except Exception as e:
            logger.warning(f"  Failed to load supply snapshots: {e}")
            self._supply_snap_cache = pd.DataFrame()
            return self._supply_snap_cache

    def _add_supply_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add supply-depth features: listing count, change, ratio.

        Uses steam sell_listings as the primary signal, with
        skinport_quantity as a secondary source where available.

        Features:
          supply_listings_log        — log(1 + sell_listings)
          supply_listings_zscore     — z-score vs item's own history
          supply_change_7d           — % change in sell_listings (7d)
          supply_skinport_qty_log    — log(1 + skinport_quantity)
          supply_to_volume_ratio     — sell_listings / (volume_30d + 1)
        """
        snap = self._fetch_supply_snapshots()
        if snap.empty:
            for col in ["supply_listings_log", "supply_listings_zscore",
                        "supply_change_7d", "supply_skinport_qty_log",
                        "supply_to_volume_ratio"]:
                df[col] = 0.0
            return df

        df = df.merge(snap, on=["item_id", "date"], how="left")
        df["sell_listings"] = df["sell_listings"].fillna(0).astype(int)
        df["skinport_quantity"] = df["skinport_quantity"].fillna(0).astype(int)

        # Log transform (scale-invariant, handles right skew)
        df["supply_listings_log"] = np.log1p(df["sell_listings"]).astype(np.float32)
        df["supply_skinport_qty_log"] = np.log1p(df["skinport_quantity"]).astype(np.float32)

        # Z-score vs item's own history (30d rolling)
        df = df.sort_values(["item_id", "date"])
        grouped = df.groupby("item_id")["sell_listings"]
        rolling_mean = grouped.transform(lambda x: x.rolling(30, min_periods=1).mean())
        rolling_std = grouped.transform(lambda x: x.rolling(30, min_periods=1).std().replace(0, np.nan))
        df["supply_listings_zscore"] = (
            (df["sell_listings"] - rolling_mean) / rolling_std
        ).fillna(0).astype(np.float32)

        # 7-day change in listing count
        df["supply_change_7d"] = (
            (df["sell_listings"] - grouped.shift(7))
            / grouped.shift(7).replace(0, np.nan) * 100
        ).fillna(0).astype(np.float32)

        # Supply-to-volume ratio: listings / trailing 30d volume
        vol_col = None
        for candidate in ["volume_30d", "volume_mean_30d", "volume"]:
            if candidate in df.columns:
                vol_col = candidate
                break
        if vol_col:
            df["supply_to_volume_ratio"] = (
                df["sell_listings"] / (df[vol_col].fillna(0).replace(0, 1) + 1)
            ).astype(np.float32)
        else:
            df["supply_to_volume_ratio"] = 0.0

        # Drop intermediate raw columns
        df = df.drop(columns=["sell_listings", "skinport_quantity"], errors="ignore")

        logger.info("  supply depth features added")
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

        decay_constants = self.event_decay_constants

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
        # Uses rolling rank (fast Cython) instead of rolling+apply (slow Python loop).
        if "market_return_30d" in df.columns:
            df["market_return_30d_percentile"] = df.groupby("item_id")["market_return_30d"].transform(
                lambda x: x.rolling(365, min_periods=30).rank(pct=True)
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
        significance_level: float = 0.05,
    ) -> Dict[str, Dict]:
        """Validate feature groups via permutation importance with
        statistical significance gating.

        For each feature group, shuffles its columns on the validation set
        and measures the directional accuracy drop vs the unshuffled baseline.
        A group is kept only if it passes BOTH:

          1. **Statistical significance** — the p-value (fraction of shuffled
             trials where accuracy >= baseline accuracy) is below
             `significance_level`. This ensures the group's signal isn't
             attributable to noise.
          2. **Practical significance** — the accuracy drop exceeds
             `min_drop_pp` (default 0.5pp). This ensures the effect is large
             enough to matter for forecasting.

        Uses the p50 (median) quantile model for the given horizon.

        Returns:
            {group_name: {"drop_pp": float, "base_acc": float,
                          "shuffled_acc": float, "p_value": float,
                          "passed": bool, "feature_count": int,
                          "features": [str]}}
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

            shuffled_arr = np.array(shuffled_accs)
            mean_shuf = float(np.mean(shuffled_arr))
            drop_pp = base_acc - mean_shuf
            p_value = float(np.mean(shuffled_arr >= base_acc))
            passed = bool(p_value < significance_level and drop_pp >= min_drop_pp)

            results[group] = {
                "drop_pp": round(drop_pp, 2),
                "base_acc": round(float(base_acc), 2),
                "shuffled_acc": round(mean_shuf, 2),
                "p_value": round(p_value, 4),
                "passed": passed,
                "feature_count": len(idxs),
                "features": groups[group],
            }

            status = "PASS" if passed else "WARN"
            logger.info(
                f"  [feat group] {group}: {drop_pp:+.2f}pp when shuffled "
                f"({base_acc:.1f}% -> {mean_shuf:.1f}%) "
                f"p={p_value:.4f} [{status}]"
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
                               quantile: float = 0.5,
                               boosting_type: str = "gbdt") -> Dict[str, Any]:
        """Bayesian hyperparameter search via Optuna.

        Searches over 6 key params using TPE pruning, with early
        termination of unpromising trials (median pruner).

        Args:
            boosting_type: "gbdt" or "dart". DART uses dropout on trees
                to reduce overfitting, useful for noisy longer horizons.
        """
        import optuna

        n_trials = 15

        # Build the binned Dataset once and reuse across all trials. Only tree
        # params (num_leaves, learning_rate, ...) vary between trials; the data
        # and its binning (max_bin) are constant, so there's no need to re-bin.
        ds_params = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
        dtrain = lgb.Dataset(X_train, y_train, params=ds_params)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, params=ds_params)

        def objective(trial):
            params = {
                "feature_pre_filter": False,
                "objective": "quantile",
                "alpha": quantile,
                "metric": "quantile",
                "boosting_type": boosting_type,
                "verbosity": -1,
                "n_jobs": -1,
                "random_state": 42,
                "max_bin": self.MAX_BIN,
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
            # DART-specific hyperparameters
            if boosting_type == "dart":
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3, log=False)
                params["max_drop"] = trial.suggest_int("max_drop", 10, 100, step=10)
                params["skip_drop"] = trial.suggest_float("skip_drop", 0.2, 0.8, log=False)
            model = lgb.train(
                params, dtrain,
                num_boost_round=200,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            return model.best_score["valid_0"]["quantile"]

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
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
        # Winsorize extreme returns at ±500% to prevent API corruption artifacts
        # from polluting gradient estimates. The audit found 11,044 jumps >1000%,
        # 84% of which revert the next day (definitive corruption).
        winsorized = df[f"target_return_{horizon}d"].clip(-500.0, 500.0)
        n_clipped = (winsorized != df[f"target_return_{horizon}d"]).sum()
        if n_clipped:
            logger.info(
                f"  Winsorized {n_clipped} extreme targets for {horizon}d "
                f"(±500% clip)"
            )
            df[f"target_return_{horizon}d"] = winsorized
        return df

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _train_ensemble_member(params: dict, dtrain: lgb.Dataset,
                                dval: lgb.Dataset) -> lgb.Booster:
        """Train a single ensemble member. Static for joblib compatibility."""
        return lgb.train(
            params, dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

    @staticmethod
    def _filter_dead_items(price_df: pd.DataFrame) -> pd.DataFrame:
        """Remove items that never meaningfully move (dead items at Steam floor).

        An item is 'dead' if its max price is <= $0.05 (Steam floor) AND its
        lifetime price range is < 5%. These items constitute ~41% of training
        rows but provide zero predictive signal — they dilute the model and
        waste gradient steps on constant targets.

        Also filters items where mean_price <= 0 (division-by-zero safety).
        """
        pre = len(price_df)
        price_df = price_df[price_df["price"] > 0].copy()
        if pre != len(price_df):
            logger.info(f"  Removed {pre - len(price_df)} rows with zero/negative price")

        item_stats = price_df.groupby("item_id")["price"].agg(["min", "max"])
        dead_mask = (
            (item_stats["max"] <= 0.05)
            & ((item_stats["max"] - item_stats["min"]) / item_stats["min"] < 0.05)
        )
        dead_items = set(item_stats[dead_mask].index)
        if not dead_items:
            return price_df

        filtered = price_df[~price_df["item_id"].isin(dead_items)].copy()
        removed_rows = len(price_df) - len(filtered)
        logger.info(
            f"  Filtered {len(dead_items)} dead items "
            f"({removed_rows:,} rows, {removed_rows / max(len(price_df), 1) * 100:.1f}%)"
        )
        return filtered

    def _flag_corrupt_items(self, price_df: pd.DataFrame,
                            jump_threshold: float = 500.0,
                            max_jumps: int = 10) -> set:
        """Identify items with frequent extreme price jumps (API corruption).

        Counts how many times each item's daily price jumps exceed
        ``jump_threshold`` percent. Items with more than ``max_jumps`` such
        events are flagged as corrupt and excluded from training.

        The audit found 84% of >1000% jumps revert the next day — definitive
        API corruption, not real market movement. 905 items are affected,
        151 with 10+ events.
        """
        pdf = price_df.sort_values(["item_id", "date"]).copy()
        pdf["_prev"] = pdf.groupby("item_id")["price"].shift(1)
        pdf["_pct"] = (pdf["price"] - pdf["_prev"]) / pdf["_prev"].replace(0, np.nan) * 100
        is_first = pdf.groupby("item_id").cumcount() == 0
        pdf.loc[is_first, "_pct"] = 0.0

        bad = pdf.groupby("item_id")["_pct"].apply(
            lambda s: int((s.abs() > jump_threshold).sum())
        )
        bad_items = set(bad[bad > max_jumps].index)
        if bad_items:
            logger.info(
                f"  Flagged {len(bad_items)} corrupt items "
                f"(>{max_jumps} jumps >{jump_threshold:.0f}%)"
            )
        return bad_items

    def _stratified_item_subsample(self, price_df: pd.DataFrame,
                                   max_rows: int, seed: int = 42,
                                   exclude_items: set = None) -> pd.DataFrame:
        """Subsample whole-item histories to bound the row count *before*
        feature engineering, while preserving per-item time-series continuity
        and the full calendar window.

        The old approach capped rows via ``train_set.tail(max_rows)`` *after*
        feature engineering. As the archive grew, that kept only the most
        recent ~51 calendar days (dropping 93% of voted rows) which silently
        disabled expanding-window CV and made the weekly retrain OOM because
        ``engineer_features`` still ran on all ~2.9M rows.

        This selects entire item histories (not individual rows) so lag/rolling
        features stay valid, stratified by rarity so rare items (knives, gloves)
        are retained proportionally, and keeps every calendar date intact so CV
        has enough distinct dates.
        """
        if exclude_items:
            pre = len(price_df)
            price_df = price_df[~price_df["item_id"].isin(exclude_items)].copy()
            if len(price_df) != pre:
                logger.info(f"  Excluded {pre - len(price_df)} corrupt-item rows")

        total_rows = len(price_df)
        if total_rows <= max_rows or "item_id" not in price_df.columns:
            return price_df

        rows_per_item = price_df.groupby("item_id").size()
        n_items = len(rows_per_item)
        avg_rows = total_rows / max(n_items, 1)
        target_items = max(1, int(max_rows / max(avg_rows, 1.0)))
        if target_items >= n_items:
            return price_df

        meta = self._fetch_supply_metadata()
        rarity_map = {}
        if meta is not None and not meta.empty and "rarity" in meta.columns:
            rarity_map = dict(zip(meta["item_id"], meta["rarity"].fillna("unknown")))

        items = pd.DataFrame({"item_id": rows_per_item.index})
        items["rarity"] = items["item_id"].map(rarity_map).fillna("unknown")

        rng = np.random.RandomState(seed)
        selected: List = []
        for _rarity, group in items.groupby("rarity"):
            frac = len(group) / n_items
            k = min(len(group), max(1, int(round(target_items * frac))))
            selected.extend(group["item_id"].sample(n=k, random_state=rng).tolist())

        selected_set = set(selected)
        out = price_df[price_df["item_id"].isin(selected_set)].copy()
        logger.info(
            f"  Stratified subsample: {len(selected_set):,}/{n_items:,} items, "
            f"{len(out):,}/{total_rows:,} rows (budget {max_rows:,}); "
            f"full calendar window preserved"
        )
        return out

    def build_training_data(self, days_back: int = 365,
                             backfilled_only: bool = False,
                             max_feature_rows: int = 400_000) -> pd.DataFrame:
        _t0 = datetime.now()
        price_df = self.fetch_price_history(days_back=days_back, backfilled_only=backfilled_only)
        logger.info(f"  fetch_price_history took {(datetime.now() - _t0).total_seconds():.0f}s")
        price_df = self._filter_dead_items(price_df)
        corrupt_items = self._flag_corrupt_items(price_df)

        # Distribution-shift guard: exclude incomplete 2026 data.
        # Run BEFORE stratified subsample so the subsample budget isn't wasted
        # on 2026 rows, and so the 7d validation window doesn't land on sparse
        # mid-2026 data (~352 items, single calendar day) where permutation
        # tests produce pure noise.
        if "date" in price_df.columns:
            dates_2026 = pd.DatetimeIndex(price_df["date"]).year == 2026
            n_2026 = dates_2026.sum()
            if n_2026 > 0:
                price_df = price_df[~dates_2026].copy()
                logger.info(f"  Excluded {n_2026:,} incomplete 2026 rows")

        if max_feature_rows:
            price_df = self._stratified_item_subsample(
                price_df, max_feature_rows, exclude_items=corrupt_items
            )
        else:
            price_df = price_df[~price_df["item_id"].isin(corrupt_items)].copy()

        _t1 = datetime.now()
        events_df = self.fetch_events()
        logger.info(f"  fetch_events took {(datetime.now() - _t1).total_seconds():.0f}s")

        _t2 = datetime.now()
        df = self.engineer_features(price_df, events_df)
        logger.info(f"  engineer_features took {(datetime.now() - _t2).total_seconds():.0f}s, "
                    f"result: {len(df):,} rows, {len(df.columns)} cols")
        del price_df, events_df

        # Add cross-sectional (market-regime) features
        _t3 = datetime.now()
        df = self._add_cross_sectional_features(df)
        logger.info(f"  cross_sectional_features took {(datetime.now() - _t3).total_seconds():.0f}s")

        # Add supply depth features (sell_listings, skinport_quantity)
        df = self._add_supply_depth_features(df)

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
    # Feature cache for predict speed
    # ------------------------------------------------------------------

    @property
    def _engineered_cache_path(self) -> str:
        return os.path.join(self.model_dir, self.ENGINEERED_CACHE_NAME)

    def _save_engineered_cache(self, df: pd.DataFrame):
        """Save the fully-engineered feature DataFrame to Parquet cache."""
        path = self._engineered_cache_path
        cache_df = df.copy()
        cache_df.attrs["_cache_date"] = str(date.today())
        logger.info(f"  Saving engineered feature cache ({len(cache_df):,} rows) to {path}")
        cache_df.to_parquet(path, index=False)

    def _load_engineered_cache(self) -> Optional[pd.DataFrame]:
        """Load cached engineered features. Returns None if cache is missing or stale."""
        path = self._engineered_cache_path
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            cache_date_str = df.attrs.get("_cache_date", "")
            logger.info(f"  Loaded engineered feature cache from {path} "
                        f"({len(df):,} rows, cache_date={cache_date_str})")

            # Check staleness: if cache is older than 3 days, trigger refresh
            if cache_date_str:
                try:
                    cache_date = date.fromisoformat(cache_date_str)
                    days_stale = (date.today() - cache_date).days
                    if days_stale > 3:
                        logger.info(f"  Cache is {days_stale} days stale (>3), will refresh")
                        return None
                except (ValueError, TypeError):
                    pass

            # For legacy caches without attrs (pre-cache-refactor), fall back to
            # DuckDB freshness check against the Parquet archive.
            if not cache_date_str:
                try:
                    import duckdb
                    archive_dir = Path(__file__).parent.parent.parent / "price-archive"
                    if archive_dir.exists():
                        con = duckdb.connect()
                        pq_files = sorted([str(p) for p in archive_dir.glob("prices-*.parquet")])
                        if pq_files:
                            latest_archive = con.sql(
                                "SELECT MAX(day) FROM read_parquet(?)",
                                params=[pq_files[-1]]
                            ).fetchone()[0]
                            con.close()
                            cache_max_date = df["date"].max() if "date" in df.columns else None
                            if cache_max_date is not None and latest_archive is not None:
                                archive_max = pd.to_datetime(latest_archive).date()
                                cache_max = pd.to_datetime(cache_max_date).date() if not isinstance(cache_max_date, date) else cache_max_date
                                days_diff = (archive_max - cache_max).days
                                if days_diff > 3:
                                    logger.info(f"  Cache max_date={cache_max} < archive max_date={archive_max} ({days_diff}d diff), refreshing")
                                    return None
                except Exception:
                    pass

            return df
        except Exception as e:
            logger.warning(f"  Failed to load feature cache: {e}")
            return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _get_feature_importance(self, model: lgb.Booster) -> pd.DataFrame:
        importance = model.feature_importance(importance_type="gain")
        feature_names = model.feature_name()
        if len(feature_names) != len(importance):
            feature_names = self.feature_cols[:len(importance)]
        fi = pd.DataFrame({"feature": feature_names, "importance": importance})
        fi = fi.sort_values("importance", ascending=False).head(20)
        return fi

    @staticmethod
    def _compute_sample_weights(tdf: pd.DataFrame, horizon: int) -> Optional[np.ndarray]:
        """Compute sample weights proportional to item price variance.

        Items that move more get higher gradient weight; flat/dead items
        get down-weighted. Uses 30-day rolling std of daily returns as the
        weight signal, clipped to [0.1, 99th percentile].

        This prevents the ~41% of historically flat items from dominating
        the loss even after dead-item filtering removes the extreme cases.
        """
        if tdf.empty or "price" not in tdf.columns:
            return None
        vol = tdf.groupby("item_id", group_keys=False)["price"].transform(
            lambda x: x.pct_change().rolling(30, min_periods=5).std()
        ).fillna(1.0).values
        vol = np.clip(vol, 0.1, np.percentile(vol, 99))
        vol = vol / max(np.mean(vol), 1e-8)
        return vol.astype(np.float32)

    def train(self, max_rows: int = 700_000):
        logger.info("=" * 60)
        logger.info("TRAINING LIGHTGBM FORECASTER (ensemble, HP search, walk-forward)")
        logger.info("=" * 60)

        _train_start = datetime.now()
        df = self.build_training_data(days_back=1460, backfilled_only=True)
        self._save_engineered_cache(df)

        self.horizon_feature_cols = {}

        for hi, horizon in enumerate(self.HORIZONS, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"HORIZON {horizon}d ({hi}/{len(self.HORIZONS)})")
            logger.info(f"{'='*60}")
            _hz_start = datetime.now()

            tdf = self.prepare_targets(df, horizon)

            # Drop NaN targets (use percentage return as primary target)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values("date")

            # Assign regime labels for regime-switching training
            tdf["_regime"] = self._assign_regime_labels(tdf)

            for attempt in range(2):
                if attempt == 1:
                    logger.info(f"  Retry {horizon}d — pruned feature groups")

                # Proper temporal walk-forward split (using actual data dates):
                # Hold out the last VALIDATION_WINDOW_DAYS of calendar data.
                max_date = pd.to_datetime(tdf["date"].max())
                split_date = max_date - timedelta(days=self.VALIDATION_WINDOW_DAYS)
                train_set = tdf[pd.to_datetime(tdf["date"]) < split_date]
                val_set = tdf[pd.to_datetime(tdf["date"]) >= split_date]

                # Safety guard only: the calendar window is already bounded by
                # the stratified item subsample in build_training_data(). Sample
                # randomly (never tail()) so we don't truncate the calendar
                # window, which would silently disable expanding-window CV.
                if len(train_set) > max_rows:
                    train_set = train_set.sample(
                        n=max_rows, random_state=42).sort_values("date")

                val_dates = val_set["date"].nunique() if "date" in val_set.columns else 0
                if len(val_set) < 2000 or val_dates < 7:
                    logger.warning(
                        f"  Validation set for {horizon}d has only {len(val_set)} rows "
                        f"({val_dates} distinct dates); using last 20% of training data as fallback."
                    )
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

                # Sample weights: down-weight historically flat items so the
                # model focuses on items with meaningful price movement.
                train_weights = self._compute_sample_weights(train_set, horizon)
                val_weights = self._compute_sample_weights(val_set, horizon)

                # Build the binned Dataset once per horizon and reuse it across
                # all quantiles and ensemble members. X/y and binning are
                # identical for every quantile (only the objective's alpha
                # changes), so rebuilding per fit just re-bins the same matrix.
                ds_params = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
                dtrain_kw = dict(params=ds_params)
                if train_weights is not None:
                    dtrain_kw["weight"] = train_weights
                dval_kw = dict(params=ds_params)
                if val_weights is not None:
                    dval_kw["weight"] = val_weights
                dtrain = lgb.Dataset(X_train, y_train, **dtrain_kw)
                dval = lgb.Dataset(X_val, y_val, reference=dtrain, **dval_kw)

                per_quantile_params = {}

                # Determine boosting type per horizon: DART for weak/long
                # horizons, GBDT for short horizons. DART's dropout
                # regularization helps reduce overfitting on noisy
                # longer-range signals.
                boosting_type = self.BOOSTING_TYPE_MAP.get(horizon, "gbdt")
                dart_msg = " (DART)" if boosting_type == "dart" else ""
                logger.info(f"  Boosting type for {horizon}d: {boosting_type}{dart_msg}")

                cached_hp = self.tuned_params.get(horizon, {})
                reuse_hp = (os.environ.get("FORCE_HP_SEARCH") != "1"
                            and all(q in cached_hp for q in self.QUANTILES))
                if reuse_hp:
                    logger.info(f"  Reusing cached HP for {horizon}d (Optuna skipped)...")
                    for q in self.QUANTILES:
                        bp = dict(cached_hp[q])
                        bp["max_bin"] = self.MAX_BIN
                        bp["feature_pre_filter"] = False
                        bp["device"] = "cuda" if _gpu_available() else "cpu"
                        bp["boosting_type"] = boosting_type
                        per_quantile_params[q] = bp
                else:
                    for q in self.QUANTILES:
                        logger.info(f"  Searching hyperparams for {horizon}d p{int(q*100)} (Optuna, {boosting_type})...")
                        best_params = self._optuna_search_params(
                            X_train, y_train, X_val, y_val, quantile=q,
                            boosting_type=boosting_type,
                        )

                        device = "cuda" if _gpu_available() else "cpu"
                        base_params = {
                            "device": device,
                            "feature_pre_filter": False,
                            "objective": "quantile",
                            "alpha": q,
                            "metric": "quantile",
                            "boosting_type": boosting_type,
                            "min_gain_to_split": 0.1,
                            "feature_fraction": 0.7,
                            "bagging_fraction": 0.7,
                            "bagging_freq": 5,
                            "max_bin": self.MAX_BIN,
                            "verbosity": -1,
                            "n_jobs": -1,
                        }
                        # Merge Optuna results into base params
                        if best_params:
                            merge_keys = ["num_leaves", "learning_rate", "lambda_l1",
                                          "lambda_l2", "max_depth", "min_data_in_leaf"]
                            if boosting_type == "dart":
                                merge_keys += ["drop_rate", "max_drop", "skip_drop"]
                            for k in merge_keys:
                                if k in best_params:
                                    base_params[k] = best_params[k]
                        else:
                            base_params["num_leaves"] = 31
                            base_params["learning_rate"] = 0.03
                            base_params["lambda_l1"] = 0.5
                            base_params["lambda_l2"] = 0.5
                            base_params["max_depth"] = 5
                            base_params["min_data_in_leaf"] = 15

                        # Add DART-specific defaults if not set by Optuna
                        if boosting_type == "dart":
                            for k, v in self.DART_PARAMS.items():
                                base_params.setdefault(k, v)

                        per_quantile_params[q] = dict(base_params)
                    self.tuned_params[horizon] = {
                        q: dict(per_quantile_params[q]) for q in self.QUANTILES
                    }

                # Train ensemble models (sequential — LightGBM internal threading handles CPU)
                for q in self.QUANTILES:
                    pq = per_quantile_params[q]
                    logger.info(f"  Training {horizon}d p{int(q*100)} ensemble ({self.N_ENSEMBLES} members)...")
                    _ens_start = datetime.now()
                    ensemble_models = []
                    for ei in range(self.N_ENSEMBLES):
                        params = pq.copy()
                        params["random_state"] = self.ENSEMBLE_SEEDS[ei]
                        params["feature_fraction"] = self.ENSEMBLE_FEATURE_FRACTIONS[ei]
                        model = self._train_ensemble_member(params, dtrain, dval)
                        ensemble_models.append(model)

                    self.models[(horizon, q)] = ensemble_models
                    _ens_elapsed = (datetime.now() - _ens_start).total_seconds()
                    fi = self._get_feature_importance(ensemble_models[0])
                    logger.info(f"  Done in {_ens_elapsed:.0f}s — Top features: {fi['feature'].head(5).tolist()}")

                    # Residual stacking: train Ridge regression on ensemble
                    # residuals to correct systematic bias patterns.
                    if self.STACK_RESIDUALS and _sklearn_available and horizon in self.WEAK_HORIZONS:
                        ensemble_pred = np.mean(
                            [m.predict(X_val) for m in ensemble_models], axis=0
                        )
                        residual = y_val.values - ensemble_pred
                        residual_model = Ridge(alpha=self.RESIDUAL_ALPHA, random_state=42)
                        residual_model.fit(X_val.values, residual)
                        self.residual_models[(horizon, q)] = residual_model
                        r2 = np.corrcoef(residual, residual_model.predict(X_val.values))[0, 1] ** 2
                        logger.info(f"  Residual model (Ridge α={self.RESIDUAL_ALPHA}) "
                                     f"trained for {horizon}d p{int(q*100)}: "
                                     f"R²={r2:.4f}")
                    elif self.STACK_RESIDUALS and not _sklearn_available and horizon in self.WEAK_HORIZONS:
                        logger.warning(f"  sklearn not available — skipping residual stacking for {horizon}d")

                # Train regime-specific models (optional: SKIP_REGIMES=1 to skip)
                if os.environ.get("SKIP_REGIMES") == "1":
                    logger.info(f"  Regime models skipped (SKIP_REGIMES=1)")
                else:
                    for regime in self.REGIMES:
                        if regime == "global":
                            continue
                        r_train = train_set[train_set["_regime"] == regime]
                        r_val = val_set[val_set["_regime"] == regime]
                        MIN_REGIME_TRAIN = 500
                        MIN_REGIME_VAL = 50
                        if len(r_train) < MIN_REGIME_TRAIN or len(r_val) < MIN_REGIME_VAL:
                            logger.info(f"  Skipping {regime} regime ({len(r_train)} train, {len(r_val)} val — "
                                        f"below minimum)")
                            continue

                        # Use global HP params (reuse Optuna results from global)
                        r_X_train = r_train[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(feature_medians)
                        r_y_train = r_train[f"target_return_{horizon}d"]
                        r_X_val = r_val[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(feature_medians)
                        r_y_val = r_val[f"target_return_{horizon}d"]

                        r_train_weights = self._compute_sample_weights(r_train, horizon)
                        r_val_weights = self._compute_sample_weights(r_val, horizon)

                        r_ds_params = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
                        r_dtrain_kw = dict(params=r_ds_params)
                        if r_train_weights is not None:
                            r_dtrain_kw["weight"] = r_train_weights
                        r_dval_kw = dict(params=r_ds_params)
                        if r_val_weights is not None:
                            r_dval_kw["weight"] = r_val_weights
                        r_dtrain = lgb.Dataset(r_X_train, r_y_train, **r_dtrain_kw)
                        r_dval = lgb.Dataset(r_X_val, r_y_val, reference=r_dtrain, **r_dval_kw)

                        logger.info(f"  Training {regime} regime models ({horizon}d, "
                                    f"{len(r_train):,} train, {len(r_val):,} val)...")
                        for q in self.QUANTILES:
                            pq = per_quantile_params[q]
                            r_ensemble = []
                            for ei in range(self.N_ENSEMBLES):
                                params = pq.copy()
                                params["random_state"] = self.ENSEMBLE_SEEDS[ei]
                                params["feature_fraction"] = self.ENSEMBLE_FEATURE_FRACTIONS[ei]
                                model = self._train_ensemble_member(params, r_dtrain, r_dval)
                                r_ensemble.append(model)
                            self.regime_models[(regime, horizon, q)] = r_ensemble

                        self.regime_feature_cols[(horizon, regime)] = list(self.feature_cols)
                        logger.info(f"  {regime} regime models for {horizon}d done "
                                    f"({len(r_train)} train, {len(r_val)} val)")

                # Expanding-window CV evaluation using the best hyperparams
                if os.environ.get("SKIP_CV") == "1":
                    logger.info(f"  CV skipped (SKIP_CV=1); calibrating from single holdout")
                    oof_records, cv_metrics, nc_scores = [], [], []
                else:
                    oof_records, cv_metrics, nc_scores = self._cv_evaluate_horizon(tdf, horizon, per_quantile_params)

                # Calibrate confidence thresholds on pooled OOF predictions from CV
                # (more robust than a single 21-day holdout)
                if oof_records:
                    records_df = pd.DataFrame(oof_records)
                    logger.info(f"  Calibrating on {len(records_df)} pooled OOF predictions "
                                f"({len(cv_metrics)} folds)")
                    self._calibrate_confidence(horizon=horizon, records_df=records_df)

                    # Conformal quantile calibration: compute the (1-α)(1+1/n)
                    # quantile of nonconformity scores from pooled OOF predictions.
                    # This adjustment factor widens prediction intervals so that
                    # [p10 - q_hat, p90 + q_hat] achieves ~(1-2α) empirical coverage.
                    if nc_scores:
                        nc_arr = np.array(nc_scores)
                        n_cal = len(nc_arr)
                        alpha = 0.10  # target 90% coverage for the conformal interval
                        q_level = (1.0 - alpha) * (1.0 + 1.0 / max(n_cal, 1))
                        q_level = min(q_level, 0.999)
                        q_hat = float(np.quantile(nc_arr, q_level))
                        self.conformal_calibration[horizon] = q_hat
                        logger.info(f"  CQR calibration: q_hat={q_hat:.4f}pp "
                                    f"(n={n_cal}, α={alpha}, target coverage={(1-alpha)*100:.0f}%)")
                elif os.environ.get("SKIP_CV") == "1":
                    logger.info("  CV skipped (SKIP_CV=1); fallback to single-split calibration")
                    self._calibrate_confidence(horizon=horizon, X_val=X_val, y_val=y_val, val_set=val_set)
                else:
                    raise RuntimeError(
                        f"CV produced no OOF records for {horizon}d horizon "
                        f"but SKIP_CV is not set. This indicates a bug — "
                        f"_cv_evaluate_horizon should have raised."
                    )

                # Log CV fold-level metrics
                fold_accs = [m["directional_accuracy"] for m in cv_metrics]
                mean_acc = float(np.mean(fold_accs)) if fold_accs else float("nan")
                std_acc = float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0
                if cv_metrics:
                    logger.info(f"  CV ({len(cv_metrics)} folds): "
                                f"mean={mean_acc:.1f}% sd={std_acc:.1f}% "
                                f"range=[{min(fold_accs):.1f}%, {max(fold_accs):.1f}%]")
                self.cv_results[horizon] = {
                    "fold_count": len(cv_metrics),
                    "per_fold": cv_metrics,
                    "mean_dir_acc": round(mean_acc, 1) if fold_accs else 0,
                    "std_dir_acc": round(std_acc, 1) if len(fold_accs) > 1 else 0,
                    "min_dir_acc": round(min(fold_accs), 1) if fold_accs else 0,
                    "max_dir_acc": round(max(fold_accs), 1) if fold_accs else 0,
                }

                # Validate feature groups: permutation test on the held-out set.
                # Skip entirely when the validation window is thin (same threshold
                # as the temporal-split floor above) — permutation tests on <2000
                # rows or <7 distinct dates are pure noise and cause false-positive
                # pruning that collapses 14d/30d models to ~4 features.
                # The significance_level parameter (0.05) gates pruning further:
                # a group must pass BOTH the statistical significance test (p < α)
                # AND the practical significance test (drop_pp >= 0.5) to be kept.
                # This prevents noisy-but-spurious correlations from surviving
                # on marginal windows without fully skipping the check.
                val_dates = val_set["date"].nunique() if "date" in val_set.columns else 0
                need_retrain = False
                if len(val_set) < 2000 or val_dates < 7:
                    logger.info(
                        f"  Skipping feature-group validation ({len(val_set)} rows, "
                        f"{val_dates} dates — below minimum threshold)"
                    )
                else:
                    try:
                        X_val_np = X_val.values if hasattr(X_val, "values") else X_val
                        y_val_np = y_val.values if hasattr(y_val, "values") else y_val
                        fv = self._validate_feature_groups(
                            X_val_np, y_val_np, self.feature_cols,
                            horizon=horizon, n_shuffles=20, min_drop_pp=0.5,
                            significance_level=0.05,
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
                                # Safety net: if all features were pruned, fall
                                # back to a minimal core set so LightGBM doesn't
                                # crash with 0 columns.
                                if not self.feature_cols and pre_count > 0:
                                    safe = ["price_log", "price_lag_1d", "price_lag_3d",
                                            "price_return_1d", "price_return_3d",
                                            "price_return_7d", "price_std_7d"]
                                    self.feature_cols = [c for c in safe if c in tdf.columns]
                                    if not self.feature_cols:
                                        self.feature_cols = tdf.select_dtypes(include=[np.number]).columns[:1].tolist()
                                    logger.warning(
                                        f"  All features pruned — falling back to "
                                        f"{len(self.feature_cols)} core features as safety net"
                                    )
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

            base_features = list(self.feature_cols)
            excluded_groups = self.HORIZON_EXCLUDED_GROUPS.get(horizon, [])
            if excluded_groups:
                horizon_features = [
                    c for c in base_features
                    if _feature_group(c) not in excluded_groups
                ]
                logger.info(
                    f"  {horizon}d: excluded {len(base_features) - len(horizon_features)} features "
                    f"from groups {excluded_groups} "
                    f"({len(base_features)} -> {len(horizon_features)} features)"
                )
                self.horizon_feature_cols[horizon] = horizon_features
            else:
                self.horizon_feature_cols[horizon] = base_features

            _hz_elapsed = (datetime.now() - _hz_start).total_seconds()
            logger.info(f"  Horizon {horizon}d done in {_hz_elapsed:.0f}s")
            if self.cv_results.get(horizon):
                cv = self.cv_results[horizon]
                logger.info(f"  CV summary: mean={cv.get('mean_dir_acc', '?'):}% "
                            f"std={cv.get('std_dir_acc', '?'):}% "
                            f"range=[{cv.get('min_dir_acc', '?'):}%, {cv.get('max_dir_acc', '?'):}%]")
            del tdf

        del df

        _train_elapsed = (datetime.now() - _train_start).total_seconds()
        self.save_models()
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETE in {_train_elapsed:.0f}s ({_train_elapsed/60:.1f}min)")
        logger.info(f"{'='*60}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_quantile_crossing(low: np.ndarray, mid: np.ndarray,
                                high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enforce low <= mid <= high via isotonic regression (PAV for 3 points).

        For each item where quantiles cross, projects [low, mid, high] onto
        the non-decreasing constraint using the Pool-Adjacent-Violators
        algorithm. This preserves item-level interval width as much as
        possible, unlike a global average-half-width imputation.

        Returns (low_fixed, high_fixed) arrays with low_fixed <= mid <= high_fixed.
        """
        v0, v1, v2 = low.copy().astype(np.float64), mid.copy().astype(np.float64), high.copy().astype(np.float64)

        # Pattern A: v0 > v1 (first two cross)
        cross_01 = v0 > v1
        if cross_01.any():
            pool_01 = (v0[cross_01] + v1[cross_01]) * 0.5
            v0[cross_01] = pool_01
            v1[cross_01] = pool_01

        # Pattern B: after fixing A, check v1 > v2 (last two cross)
        cross_12 = v1 > v2
        if cross_12.any():
            pool_12 = (v1[cross_12] + v2[cross_12]) * 0.5
            v1[cross_12] = pool_12
            v2[cross_12] = pool_12

        # Pattern C: after fixing B, check again if A was re-broken (pooled
        # v1,v2 < original v0). Only possible when all three crossed.
        recross_01 = v0 > v1
        if recross_01.any():
            pool_all = (v0[recross_01] + v1[recross_01] + v2[recross_01]) / 3.0
            v0[recross_01] = pool_all
            v1[recross_01] = pool_all
            v2[recross_01] = pool_all

        return v0, v2

    def _blend_returns_with_prior(self, low_ret_arr: np.ndarray, mid_ret_arr: np.ndarray,
                                  high_ret_arr: np.ndarray, prior: Dict[str, np.ndarray],
                                  weight: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Blend current return-space predictions toward the prior day's.

        Returns the (low, mid, high) arrays after exponentially smoothing with
        the previous forecast. Items without a prior are left untouched.
        """
        mask = prior["mask"]
        if not mask.any() or weight <= 0:
            return low_ret_arr, mid_ret_arr, high_ret_arr
        mid_ret_arr = np.where(mask, (1 - weight) * mid_ret_arr + weight * prior["mid_ret"], mid_ret_arr)
        low_ret_arr = np.where(mask, (1 - weight) * low_ret_arr + weight * prior["low_ret"], low_ret_arr)
        high_ret_arr = np.where(mask, (1 - weight) * high_ret_arr + weight * prior["high_ret"], high_ret_arr)
        return low_ret_arr, mid_ret_arr, high_ret_arr

    def _fetch_prior_forecasts(self, item_ids: np.ndarray, horizon: int) -> Dict[str, np.ndarray]:
        """Fetch the most recent prior-day forecast per item for blending.

        Returns return-space predictions (percentage return vs the prior
        ``current_price``) aligned to ``item_ids``, with NaN where no prior
        forecast exists. Used by ``predict()`` to smooth daily flip-flopping.
        """
        result = {
            "mask": np.zeros(len(item_ids), dtype=bool),
            "low_ret": np.full(len(item_ids), np.nan),
            "mid_ret": np.full(len(item_ids), np.nan),
            "high_ret": np.full(len(item_ids), np.nan),
        }
        if self.db is None:
            return result
        today = date.today()
        try:
            rows = self.db.execute(text("""
                SELECT item_id, price_low, price_mid, price_high, current_price, forecast_date
                FROM item_forecasts
                WHERE horizon_days = :h AND forecast_date < :today
            """), {"h": horizon, "today": today}).fetchall()
        except Exception as e:
            logger.warning(f"  Prior-forecast fetch failed ({e}); skipping blend.")
            return result
        if not rows:
            return result

        # Keep the latest forecast_date seen per item.
        best: Dict[int, tuple] = {}
        for r in rows:
            iid = int(r.item_id)
            if iid not in best or r.forecast_date > best[iid][4]:
                best[iid] = (r.price_low, r.price_mid, r.price_high, r.current_price, r.forecast_date)

        # Map Parquet string slugs → integer DB IDs
        try:
            slug_rows = self.db.execute(
                text("SELECT id, item_id FROM items WHERE is_backfilled = 1")
            ).fetchall()
            slug_to_id = {r.item_id: r.id for r in slug_rows}
        except Exception:
            slug_to_id = {}

        id_to_idx = {}
        for i, slug in enumerate(item_ids):
            iid = slug_to_id.get(str(slug))
            if iid is not None:
                id_to_idx[iid] = i
        for iid, vals in best.items():
            idx = id_to_idx.get(iid)
            if idx is None:
                continue
            low, mid, high, cur, _ = vals
            if not cur or cur <= 0 or mid is None:
                continue
            result["mask"][idx] = True
            cur_f = float(cur)
            result["mid_ret"][idx] = (float(mid) / cur_f - 1.0) * 100.0
            result["low_ret"][idx] = (float(low) / cur_f - 1.0) * 100.0 if low is not None else result["mid_ret"][idx]
            result["high_ret"][idx] = (float(high) / cur_f - 1.0) * 100.0 if high is not None else result["mid_ret"][idx]
        return result

    def predict(self, item_ids: List[int] = None) -> pd.DataFrame:
        logger.info("Generating forecasts...")

        # Try to load cached engineered features first (major speedup)
        df = self._load_engineered_cache()

        if df is not None:
            logger.info(f"  Using cached engineered features ({len(df):,} rows)")
        else:
            logger.info("  No usable cache found — running full feature engineering")
            price_df = self.fetch_price_history(days_back=1460, backfilled_only=True)

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

            # Add supply depth features (same as training)
            df = self._add_supply_depth_features(df)

            # Save to cache for next predict run
            self._save_engineered_cache(df)

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

        # Build the full feature matrix with the union of all per-horizon feature
        # sets (each horizon may have pruned different features).
        all_feature_cols = sorted(set().union(
            *[set(cols) for cols in self.horizon_feature_cols.values()]
        )) if self.horizon_feature_cols else self.feature_cols

        latest_clean = latest_rows[all_feature_cols].replace([np.inf, -np.inf], np.nan)
        if not self.feature_medians.empty:
            medians_aligned = self.feature_medians.reindex(all_feature_cols)
            medians_aligned = medians_aligned.where(medians_aligned.notna(), latest_clean.median())
            X_batch = latest_clean.fillna(medians_aligned)
        else:
            X_batch = latest_clean.fillna(latest_clean.median())

        item_id_arr = latest_rows["item_id"].to_numpy()
        current_price_arr = latest_rows["price"].to_numpy()
        generated_at = self._now()

        # Detect current market regime for regime-aware model selection
        current_regime = self._detect_current_regime(df)
        has_regime_models = any(r == current_regime for r, _, _ in self.regime_models)
        if current_regime in self.REGIMES:
            if has_regime_models:
                logger.info(f"  Current market regime: {current_regime} (using regime models)")
            else:
                logger.info(f"  Current market regime: {current_regime} (no regime models trained, using global)")
        else:
            logger.info(f"  Current market regime: {current_regime} (using global models)")

        # Track regime vs global model usage for diagnostics
        regime_count = 0
        global_count = 0

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
            h_features = self.horizon_feature_cols.get(horizon, self.feature_cols)
            X_horizon = X_batch[h_features]
            preds = {}
            for q in self.QUANTILES:
                all_preds = []

                # Prefer regime-specific model, fall back to global
                regime_key = (current_regime, horizon, q)
                use_regime = (current_regime in self.REGIMES
                              and regime_key in self.regime_models)
                if use_regime:
                    ensemble = self.regime_models[regime_key]
                    if isinstance(ensemble, list):
                        for m in ensemble:
                            all_preds.append(m.predict(X_horizon))
                    else:
                        all_preds.append(ensemble.predict(X_horizon))
                    regime_count += 1
                else:
                    key = (horizon, q)
                    if key in self.models:
                        ensemble = self.models[key]
                        if isinstance(ensemble, list):
                            for m in ensemble:
                                all_preds.append(m.predict(X_horizon))
                        else:
                            all_preds.append(ensemble.predict(X_horizon))
                        global_count += 1

                if all_preds:
                    preds[q] = np.mean(all_preds, axis=0)

                    # Residual stacking correction: add Ridge prediction
                    # to correct systematic bias patterns learned during
                    # training. Only applies to weak horizons where the
                    # residual model was trained.
                    residual_key = (horizon, q)
                    if residual_key in self.residual_models:
                        res_correction = self.residual_models[residual_key].predict(X_horizon.values)
                        preds[q] = preds[q] + res_correction

            if len(preds) != 3:
                continue

            # Models predict percentage returns. Convert back to price levels.
            # preds are return percentages (e.g., 5.0 means +5%).
            p10_ret = preds[0.1]
            p50_ret = preds[0.5]
            p90_ret = preds[0.9]

            # Fix quantile crossing via isotonic regression (PAV for 3 points).
            # Preserves item-level interval width instead of global imputation.
            low_ret_arr, high_ret_arr = self._fix_quantile_crossing(
                p10_ret, p50_ret, p90_ret)
            mid_ret_arr = p50_ret

            # Conformal calibration: widen intervals by the CQR adjustment factor
            # learned from CV out-of-fold residuals. This pushes empirical coverage
            # toward nominal (80% for [p10, p90] intervals).
            q_hat = self.conformal_calibration.get(horizon, 0.0)
            if q_hat > 0:
                low_ret_arr = low_ret_arr - q_hat
                high_ret_arr = high_ret_arr + q_hat

            # Forecast blending / directional smoothing: blend the current
            # return-space predictions with the previous day's forecast for the
            # same item+horizon. Reduces daily direction flip-flopping. No-ops
            # when no prior forecast exists (first run / retrain).
            prior = self._fetch_prior_forecasts(item_id_arr, horizon)
            low_ret_arr, mid_ret_arr, high_ret_arr = self._blend_returns_with_prior(
                low_ret_arr, mid_ret_arr, high_ret_arr, prior, self.FORECAST_BLEND_WEIGHT)

            # Diagnostic: log crossing rate
            crossing_mask = (p10_ret > p50_ret) | (p50_ret > p90_ret)
            crossing_rate = np.mean(crossing_mask)
            if crossing_rate > 0.01:
                logger.warning(f"  Quantile crossing rate: {crossing_rate:.3f} "
                               f"(corrected via PAV isotonic regression)")

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
                    "direction": "up" if mid_ret > DIRECTION_FLAT_TOLERANCE_PCT else "down" if mid_ret < -DIRECTION_FLAT_TOLERANCE_PCT else "flat",
                    "confidence": self._compute_confidence(price_mid, price_low, price_high,
                                                            current_price, horizon=horizon),
                }

        result_df = pd.DataFrame([r for r in agg.values() if r["forecasts"]])
        if not result_df.empty:
            result_df = self._sanitize_forecasts(result_df)

        total_used = regime_count + global_count
        if total_used > 0:
            pct = regime_count / total_used * 100
            logger.info(f"  Regime model usage: {regime_count}/{total_used} "
                        f"({pct:.1f}%) regime, {global_count}/{total_used} "
                        f"({100-pct:.1f}%) global")

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
                        if key == "mid":
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
            (oof_records, fold_metrics, nonconformity_scores) where oof_records
            is a list of dicts with range_pct/change_pct/hit for calibration,
            fold_metrics is a list of per-fold accuracy dicts, and
            nonconformity_scores is a list of CQR scores for conformal calibration.
        """
        sorted_dates = sorted(tdf["date"].unique())
        splits = self._compute_cv_splits(sorted_dates)
        if len(splits) < 2:
            raise RuntimeError(
                f"CV produced {len(splits)} fold{'s' if splits else 's'} "
                f"(need >=2). Cannot evaluate {horizon}d horizon — "
                f"check that training data has enough distinct dates "
                f"({len(sorted_dates)} available)."
            )

        oof_records = []
        fold_metrics = []
        nonconformity_scores = []

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

            # Sample weights (same as main training loop)
            train_w = self._compute_sample_weights(train_df, horizon)
            val_w = self._compute_sample_weights(val_df, horizon)

            # Build the fold's binned Dataset once; reuse across all quantiles.
            ds_params = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
            dtrain_kw = dict(params=ds_params)
            if train_w is not None:
                dtrain_kw["weight"] = train_w
            dval_kw = dict(params=ds_params)
            if val_w is not None:
                dval_kw["weight"] = val_w
            dtrain = lgb.Dataset(X_train, y_train, **dtrain_kw)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain, **dval_kw)

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

            # Fix quantile crossing via isotonic regression (same as predict()).
            low_pred, high_pred = self._fix_quantile_crossing(
                fold_p10, fold_p50, fold_p90)
            current_prices = val_df["price"].values
            actual_returns = y_val.values

            # Conformal nonconformity scores: max(Q_low - y, y - Q_high) in % return space.
            # Measures how far the actual return falls outside the prediction interval.
            fold_scores = np.maximum(
                low_pred - actual_returns,
                actual_returns - high_pred,
            )
            fold_scores = np.clip(fold_scores, 0.0, None)  # inside interval → score 0
            nonconformity_scores.extend(fold_scores[~np.isnan(fold_scores)].tolist())

            # Fold-level directional accuracy
            fold_hits = 0
            for i in range(len(val_df)):
                actual_ret = float(actual_returns[i])
                mid_ret = float(fold_p50[i])
                actual_dir = "up" if actual_ret > DIRECTION_FLAT_TOLERANCE_PCT else "down" if actual_ret < -DIRECTION_FLAT_TOLERANCE_PCT else "flat"
                pred_dir = "up" if mid_ret > DIRECTION_FLAT_TOLERANCE_PCT else "down" if mid_ret < -DIRECTION_FLAT_TOLERANCE_PCT else "flat"
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
                actual_dir = "up" if actual_ret > DIRECTION_FLAT_TOLERANCE_PCT else "down" if actual_ret < -DIRECTION_FLAT_TOLERANCE_PCT else "flat"
                pred_dir = "up" if mid_ret > DIRECTION_FLAT_TOLERANCE_PCT else "down" if mid_ret < -DIRECTION_FLAT_TOLERANCE_PCT else "flat"
                hit = 1.0 if pred_dir == actual_dir else 0.0

                oof_records.append({
                    "range_pct": range_pct,
                    "change_pct": change_pct,
                    "hit": hit,
                })

        if not fold_metrics:
            raise RuntimeError(
                f"CV evaluated zero folds for {horizon}d horizon — "
                f"every validation window had <50 rows or all quantile predictions "
                f"failed. Need at least one usable fold. Training data has "
                f"{len(sorted_dates)} distinct dates and {len(tdf)} rows."
            )

        return oof_records, fold_metrics, nonconformity_scores

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

            # Fix quantile crossing via isotonic regression (same as predict / CV).
            low_pred, high_pred = self._fix_quantile_crossing(
                p10_pred, p50_pred, p90_pred)

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

        # Save regime-specific models
        for (regime, horizon, q), ensemble in self.regime_models.items():
            if isinstance(ensemble, list):
                for ei, model in enumerate(ensemble):
                    path = os.path.join(
                        self.model_dir,
                        f"lgb_{horizon}d_q{int(q*100)}_{regime}_e{ei}.txt"
                    )
                    model.save_model(path)
            else:
                path = os.path.join(
                    self.model_dir,
                    f"lgb_{horizon}d_q{int(q*100)}_{regime}.txt"
                )
                ensemble.save_model(path)

        # Save residual stacking models (Ridge coefficients for weak horizons)
        if _sklearn_available and self.residual_models:
            import joblib
            for (horizon, q), res_model in self.residual_models.items():
                path = os.path.join(
                    self.model_dir,
                    f"residual_{horizon}d_q{int(q*100)}.pkl"
                )
                joblib.dump(res_model, path)
            logger.info(f"  Saved {len(self.residual_models)} residual stacking models")

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

        # Serialize tuned per-quantile hyperparameters so subsequent retrains
        # can reuse them and skip the Optuna search (Tier-1 speedup).
        tuned_serial = {}
        for h, qd in self.tuned_params.items():
            tuned_serial[str(h)] = {str(q): dict(params) for q, params in qd.items()}

        # Serialize regime feature cols: {"horizon_regime": [cols]}
        regime_cols_serial = {}
        for (h, regime), cols in self.regime_feature_cols.items():
            regime_cols_serial[f"{h}_{regime}"] = cols

        # Track which regimes actually have trained models
        trained_regimes = list(set(reg for (reg, h, q) in self.regime_models.keys()))

        meta = {
            "feature_cols": self.feature_cols,
            "horizon_feature_cols": {
                str(h): cols for h, cols in self.horizon_feature_cols.items()
            },
            "regime_feature_cols": regime_cols_serial,
            "trained_regimes": trained_regimes,
            "regime_threshold_bear": self.REGIME_RETURN_THRESHOLD_BEAR,
            "regime_threshold_bull": self.REGIME_RETURN_THRESHOLD_BULL,
            "trained_at": str(self._now()),
            "confidence_thresholds": thresholds_serial,
            "conformal_calibration": {str(h): v for h, v in self.conformal_calibration.items()},
            "feature_medians": medians_serial,
            "n_ensembles": self.N_ENSEMBLES,
            "ensemble_seeds": self.ENSEMBLE_SEEDS,
            "ensemble_feature_fractions": self.ENSEMBLE_FEATURE_FRACTIONS,
            "training_window_days": 1460,
            "feature_importance": feature_importance,
            "cv_results": cv_serial,
            "tuned_params": tuned_serial,
            "residual_models": [
                [h, q] for (h, q) in self.residual_models.keys()
            ],
        }
        def _json_default(o):
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(os.path.join(self.model_dir, "meta.json"), "w") as f:
            json.dump(meta, f, default=_json_default)

        logger.info(f"Models saved to {self.model_dir}")

    def load_models(self):
        meta_path = os.path.join(self.model_dir, "meta.json")
        if not os.path.exists(meta_path):
            logger.warning(f"No saved models found in {self.model_dir}")
            return False

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupt meta.json ({e}); ignoring saved models and retraining.")
            return False
        self.feature_cols = meta["feature_cols"]
        self.feature_medians = pd.Series(meta.get("feature_medians", {}), dtype=np.float64)

        # Restore cached tuned hyperparameters (skips Optuna on retrain when present).
        self.tuned_params = {}
        raw_tp = meta.get("tuned_params", {})
        for h_str, qd in raw_tp.items():
            try:
                h = int(h_str)
            except (ValueError, TypeError):
                continue
            self.tuned_params[h] = {}
            for q_str, params in qd.items():
                try:
                    q = float(q_str)
                except (ValueError, TypeError):
                    continue
                self.tuned_params[h][q] = params

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
        # Restore conformal calibration adjustment factors
        raw_cc = meta.get("conformal_calibration", {})
        self.conformal_calibration = {}
        for h_str, q_hat in raw_cc.items():
            try:
                self.conformal_calibration[int(h_str)] = float(q_hat)
            except (ValueError, TypeError):
                continue

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

        # Load regime-specific models
        trained_regimes = meta.get("trained_regimes", [])
        raw_regime_fc = meta.get("regime_feature_cols", {})
        self.regime_feature_cols = {}
        for key_str, cols in raw_regime_fc.items():
            parts = key_str.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    h = int(parts[0])
                    regime = parts[1]
                    self.regime_feature_cols[(h, regime)] = cols
                except ValueError:
                    continue

        for regime in trained_regimes:
            for horizon in self.HORIZONS:
                for q in self.QUANTILES:
                    ensemble = []
                    for ei in range(n_ensembles):
                        path = os.path.join(
                            self.model_dir,
                            f"lgb_{horizon}d_q{int(q*100)}_{regime}_e{ei}.txt"
                        )
                        if os.path.exists(path):
                            ensemble.append(lgb.Booster(model_file=path))
                    if ensemble:
                        self.regime_models[(regime, horizon, q)] = ensemble

        # Load residual stacking models (Ridge on LGB residuals for weak horizons)
        if _sklearn_available:
            import joblib
            res_model_list = meta.get("residual_models", [])
            for entry in res_model_list:
                try:
                    h, q = int(entry[0]), float(entry[1])
                except (ValueError, TypeError):
                    continue
                rpath = os.path.join(
                    self.model_dir,
                    f"residual_{h}d_q{int(q*100)}.pkl"
                )
                if os.path.exists(rpath):
                    self.residual_models[(h, q)] = joblib.load(rpath)
            if self.residual_models:
                logger.info(f"  Loaded {len(self.residual_models)} residual stacking models")

        # Build per-horizon feature sets from the loaded models.
        # Each model internally stores the feature names it was trained with.
        self.horizon_feature_cols = {}
        for (horizon, q), ensemble in self.models.items():
            if horizon in self.horizon_feature_cols:
                continue
            model = ensemble[0] if isinstance(ensemble, list) else ensemble
            self.horizon_feature_cols[horizon] = model.feature_name()

        total_groups = len(self.models) + len(self.regime_models)
        if self.regime_models:
            regimes_found = set(r for (r, h, q) in self.regime_models.keys())
            logger.info(f"Loaded {len(self.models)} global + {len(self.regime_models)} regime "
                        f"model groups ({regimes_found}) from {self.model_dir}")
        else:
            logger.info(f"Loaded {total_groups} model groups from {self.model_dir}")
        return total_groups > 0

    def has_models(self) -> bool:
        return len(self.models) > 0
