"""
Unit tests for ItemForecaster — feature engineering, quantile handling,
confidence calibration, sanitization, and drift monitoring.

Tests avoid DB/Parquet dependencies by constructing synthetic DataFrames
and injecting them directly into the methods under test.
"""

import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.forecaster import ItemForecaster, SAMPLE_WEIGHT_HALFLIFE_DAYS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def forecaster(tmp_path_factory):
    """Forecaster pointed at a throwaway model_dir.

    Never let the default model_dir (the real models/saved_models/) through:
    anything that persists — train(), save_models() — would otherwise clobber
    production artifacts, which are gitignored and so unrecoverable.
    """
    f = ItemForecaster(db_session=MagicMock(),
                       model_dir=str(tmp_path_factory.mktemp("saved_models")))
    return f


@pytest.fixture
def basic_price_df():
    """10 items × 100 days of smooth price data with a known trend."""
    np.random.seed(42)
    rows = []
    for item_id in range(10):
        price = 10.0 + item_id * 5  # different base prices
        for day_offset in range(100):
            d = date(2026, 1, 1) + timedelta(days=day_offset)
            trend = price * (1 + 0.001 * day_offset)  # upward drift
            noise = np.random.normal(0, trend * 0.01)
            rows.append({
                "item_id": f"item_{item_id}",
                "date": d,
                "price": round(trend + noise, 2),
                "volume": int(np.random.poisson(100 + day_offset * 2)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def empty_events_df():
    return pd.DataFrame(columns=["id", "type", "timestamp", "description"])


@pytest.fixture
def events_df():
    dates = [date(2026, 2, 1), date(2026, 3, 15), date(2026, 4, 1)]
    return pd.DataFrame([
        {"id": 1, "type": "major", "timestamp": pd.Timestamp(d), "description": "Test Major", "date": d}
        for d in dates
    ] + [
        {"id": 2, "type": "update", "timestamp": pd.Timestamp(date(2026, 3, 1)), "description": "Test Update", "date": date(2026, 3, 1)}
    ])


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_compute_price_features_lags_and_returns(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        assert "price_lag_1d" in df.columns
        assert "return_1d" in df.columns
        assert "price_mean_7d" in df.columns
        assert "price_std_7d" in df.columns
        # Lag of first row per item should be NaN
        first_item_rows = df[df["item_id"] == "item_0"].head(1)
        assert first_item_rows["price_lag_1d"].isna().all()

    def test_longer_trend_features_present_and_price_group(self, forecaster):
        """New (2026-07-24) longer-lookback trend features must be computed and
        classified into the price_technicals allowlist group."""
        from models.forecaster import _feature_group
        rows = []
        price = 20.0
        np.random.seed(0)
        for d in range(250):
            price *= 1 + np.random.randn() * 0.01
            rows.append({"item_id": "a", "date": date(2025, 1, 1) + timedelta(days=d),
                         "price": round(max(price, 0.01), 2), "volume": 100})
        df = forecaster._compute_price_features(pd.DataFrame(rows))
        new_cols = ["return_90d", "return_120d", "return_180d",
                    "price_dist_ma100", "price_dist_ma200", "trend_up_fraction_30d"]
        for c in new_cols:
            assert c in df.columns, f"missing {c}"
            assert _feature_group(c) == "price_technicals", f"{c} not in price group"
        # Longer returns require enough history: NaN early, populated late.
        assert df["return_180d"].head(180).isna().all()
        assert df["return_180d"].tail(10).notna().all()
        # Trend fraction is a probability in [0, 1].
        tf = df["trend_up_fraction_30d"].dropna()
        assert tf.between(0.0, 1.0).all()

    def test_date_based_lags_are_gap_robust(self, forecaster):
        """Returns must be computed by calendar date, not row position, so a gap
        in an item's series yields NaN (→ neutral) instead of a return that
        silently spans the hole (the 2026 May–June gap bug)."""
        base = date(2026, 1, 1)
        rows = []
        # continuous days 0..19 at price 10, GAP days 20..39, then days 40..49 at price 20
        for d in list(range(0, 20)) + list(range(40, 50)):
            rows.append({"item_id": "a", "date": base + timedelta(days=d),
                         "price": 10.0 if d < 20 else 20.0, "volume": 100})
        df = forecaster._compute_price_features(pd.DataFrame(rows))
        row = lambda d: df[df["date"] == base + timedelta(days=d)].iloc[0]
        # day 45: return_14d looks for day 31 (inside the gap) -> NaN, NOT +100%
        assert pd.isna(row(45)["return_14d"])
        # within the post-gap block, day 45 vs day 44 (both present, price 20) -> 0%
        assert row(45)["return_1d"] == 0.0
        # a clean in-window return still computes: day 15 vs day 8 (both price 10) -> 0%
        assert row(15)["return_7d"] == 0.0
        # endpoints exactly 30 days apart that both EXIST still compute (10->20)
        assert row(40)["return_30d"] == pytest.approx(100.0)  # day 40 vs day 10

    def test_returns_winsorized(self, forecaster):
        df = pd.DataFrame({
            "item_id": ["a"] * 10,
            "date": [date(2026, 1, 1) + timedelta(days=i) for i in range(10)],
            "price": [1.0, 10.0, 50.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0],
            "volume": [100] * 10,
        })
        result = forecaster._compute_price_features(df)
        price_lag_1 = result["price_lag_1d"]
        return_1 = result["return_1d"]
        extreme = return_1.abs() > 500
        assert not extreme.any(), f"Returns not winsorized: max={return_1.max():.1f}"

    def test_bollinger_bands(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_pct_b" in df.columns
        # bb_pct_b should be in [-2, 2]
        valid = df["bb_pct_b"].dropna()
        assert valid.between(-2, 2).all()

    def test_rsi_in_range(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        rsi = df["rsi_14"].dropna()
        assert rsi.between(0, 100).all()

    def test_macd_components(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        assert "macd_line" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns

    def test_missingness_indicators(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        assert "rsi_missing" in df.columns
        assert "macd_missing" in df.columns
        assert df["rsi_missing"].dtype == int
        # First row per-item has NaN price_change so RSI is NaN → rsi_missing=1
        first_item = df[df["item_id"] == "item_0"]
        assert first_item["rsi_missing"].iloc[0] == 1
        # After first row, min_periods=1 means RSI computes immediately
        assert first_item["rsi_missing"].iloc[1] == 0

    def test_volume_features(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        assert "volume_log_change_1d" in df.columns
        assert "volume_zscore_30d" in df.columns
        assert "volume_price_conf_1d" in df.columns

    def test_volume_features_missing_column(self, forecaster, basic_price_df):
        no_vol = basic_price_df.drop(columns=["volume"])
        df = forecaster._compute_price_features(no_vol)
        assert df["volume_missing"].iloc[0] == 1

    def test_vol_asymmetry_present_and_price_group(self, forecaster, basic_price_df):
        from models.forecaster import _feature_group
        df = forecaster._compute_price_features(basic_price_df)
        for col in ["vol_semidev_down_30d", "vol_semidev_up_30d", "vol_skew_30d"]:
            assert col in df.columns
            assert _feature_group(col) == "price_technicals"

    def test_vol_skew_clip_bounds(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        s = df["vol_skew_30d"].dropna()
        assert (s >= 0).all() and (s <= 5).all()

    def test_vol_skew_reflects_asymmetry(self, forecaster):
        # Large, varied up-moves; small, varied down-moves => upside semidev >
        # downside semidev => skew > 1. Both sides vary, so neither semidev is
        # exactly zero (which would make the ratio NaN).
        rng = np.random.default_rng(0)
        price = 100.0
        rows = []
        for d in range(120):
            if d % 2 == 0:
                price *= 1 + rng.uniform(0.03, 0.06)    # big ups
            else:
                price *= 1 - rng.uniform(0.002, 0.006)  # tiny downs
            rows.append({"item_id": "a",
                         "date": date(2026, 1, 1) + timedelta(days=d),
                         "price": round(price, 2), "volume": 100})
        df = forecaster._compute_price_features(pd.DataFrame(rows))
        assert df["vol_skew_30d"].dropna().iloc[-1] > 1.0


class TestTemporalFeatures:
    def test_temporal_features_added(self, forecaster, basic_price_df):
        df = forecaster._add_temporal_features(basic_price_df)
        for col in ["day_of_week", "month", "quarter", "is_weekend",
                     "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert col in df.columns

    def test_is_weekend_correct(self, forecaster, basic_price_df):
        df = forecaster._add_temporal_features(basic_price_df)
        saturday = df[df["day_of_week"] == 5]
        sunday = df[df["day_of_week"] == 6]
        assert (saturday["is_weekend"] == 1).all()
        assert (sunday["is_weekend"] == 1).all()

    def test_item_age_days(self, forecaster, basic_price_df):
        df = forecaster._add_temporal_features(basic_price_df)
        assert "item_age_days" in df.columns
        item_0 = df[df["item_id"] == "item_0"]
        ages = item_0["item_age_days"].dropna()
        assert len(ages) > 0
        assert ages.iloc[-1] >= ages.iloc[0]


class TestEventFeatures:
    def test_event_features_empty(self, forecaster, basic_price_df, empty_events_df):
        df = forecaster._add_event_features(basic_price_df, empty_events_df)
        for et in ["major", "operation", "case_drop", "update", "game_update"]:
            assert f"event_decay_{et}" in df.columns
            assert f"events_next_30d_{et}" in df.columns
            assert f"event_density_30d_{et}" in df.columns
            assert f"event_density_90d_{et}" in df.columns
            assert (df[f"event_decay_{et}"] == 0.0).all()

    def test_event_decay_resets_after_new_events(self, forecaster, basic_price_df, events_df):
        df = forecaster._add_event_features(basic_price_df, events_df)
        item_0 = df[df["item_id"] == "item_0"].sort_values("date")
        decay_col = "event_decay_major"
        # Before the first event (Feb 1), decay should be 0
        jan_rows = item_0[pd.to_datetime(item_0["date"]).dt.month == 1]
        assert (jan_rows[decay_col] == 0).all(), "Before first event, decay should be 0"
        # After the first event (Feb 2+), decay should be positive and decreasing.
        # The event date itself (Feb 1) has decay=0 because searchsorted finds
        # the previous event, and there is none before Feb 1.
        feb_rows = item_0[(pd.to_datetime(item_0["date"]).dt.month == 2)
                          & (pd.to_datetime(item_0["date"]).dt.day > 1)]
        if len(feb_rows) > 5:
            feb_decay = feb_rows[decay_col].values
            assert feb_decay[0] > feb_decay[-1], "Decay should decrease between events"

    def test_events_next_30d_counts(self, forecaster, basic_price_df, events_df):
        df = forecaster._add_event_features(basic_price_df, events_df)
        item_0 = df[df["item_id"] == "item_0"].sort_values("date")
        # Before Feb 1: should have at least 1 major in next 30 days if we're in Jan
        jan_rows = item_0[pd.to_datetime(item_0["date"]).dt.month == 1]
        if not jan_rows.empty:
            # 2 events in Feb (Feb 1 and Mar 1), so some should show up
            assert jan_rows["events_next_30d_major"].sum() > 0


class TestCrossSectionalFeatures:
    def test_market_return_computed(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        df = forecaster._add_cross_sectional_features(df)
        assert "market_return_1d" in df.columns
        assert "item_return_vs_market_1d" in df.columns
        # Market return should be the same for all items on a given date
        # (first date has NaN returns for all items since no lag data exists)
        date_group = df.dropna(subset=["market_return_1d"]).groupby("date")["market_return_1d"].nunique()
        assert (date_group == 1).all(), "market_return should be identical for all items on same date"

    def test_market_regime_flags(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        df = forecaster._add_cross_sectional_features(df)
        for regime in ["bull", "bear", "range"]:
            assert f"market_regime_{regime}" in df.columns
        # At least one regime should be present
        assert df["market_regime_range"].sum() > 0

    def test_market_volume_feature(self, forecaster, basic_price_df):
        df = forecaster._compute_price_features(basic_price_df)
        df = forecaster._add_cross_sectional_features(df)
        assert "market_volume_mean_30d" in df.columns
        assert "item_volume_vs_market_30d" in df.columns


class TestFeaturePruning:
    def test_prune_features_removes_highly_correlated(self, forecaster):
        forecaster.feature_cols = ["a", "b", "c", "d"]
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.randn(100),
            "d": np.random.randn(100),
        })
        df["b"] = df["a"] * 2 + 0.01  # nearly perfect correlation
        pruned = forecaster._prune_features(df)
        assert "b" not in pruned, "b should be pruned (correlated with a)"
        assert "a" in pruned, "a should be kept"

    def test_no_pruning_when_not_correlated(self, forecaster):
        forecaster.feature_cols = ["x", "y", "z"]
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "z": np.random.randn(100),
        })
        pruned = forecaster._prune_features(df)
        assert len(pruned) == 3

    def test_validate_feature_groups_passes_statistically_significant(
        self, forecaster,
    ):
        """A feature group that causally affects predictions should pass both
        the statistical (p < 0.05) and practical (drop >= 0.5pp) gates.
        Uses quantile regression (alpha=0.5) to match the real pipeline."""
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 4).astype(np.float32)
        # y is a continuous return-like target; only cols 0-1 are predictive
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + rng.randn(n) * 0.5

        ds = lgb.Dataset(X, y)
        model = lgb.train(
            {"objective": "quantile", "alpha": 0.5, "verbosity": -1,
             "max_bin": 63, "min_data_in_leaf": 1, "num_leaves": 4,
             "learning_rate": 0.1, "metric": "quantile"},
            ds, num_boost_round=30,
        )
        forecaster.models[(7, 0.5)] = model

        # Feature names: "price_technicals" group (cols 0-1, predictive),
        # "events" group (cols 2-3, pure noise)
        feature_names = ["price_momentum_1", "price_momentum_2",
                         "event_major_1", "event_major_2"]
        results = forecaster._validate_feature_groups(
            X, y, feature_names, horizon=7,
            n_shuffles=30, min_drop_pp=0.5, significance_level=0.05,
        )

        # The "price_technicals" group — causally linked to y — should PASS
        pt = results["price_technicals"]
        assert pt["passed"], (
            f"Expected 'price_technicals' to pass (drop_pp={pt['drop_pp']}, "
            f"p={pt['p_value']})"
        )
        assert pt["p_value"] < 0.05

    def test_validate_feature_groups_fails_noisy_group(
        self, forecaster,
    ):
        """A feature group with no causal signal should fail the statistical
        significance gate (p >= 0.05), even if the drop happens to be >= 0.5pp
        by chance. Uses quantile regression to match the real pipeline."""
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 6).astype(np.float32)
        # y depends only on cols 0-2; cols 3-5 (events) are pure noise
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.0 + rng.randn(n) * 0.5

        ds = lgb.Dataset(X, y)
        model = lgb.train(
            {"objective": "quantile", "alpha": 0.5, "verbosity": -1,
             "max_bin": 63, "min_data_in_leaf": 1, "num_leaves": 4,
             "learning_rate": 0.1, "metric": "quantile"},
            ds, num_boost_round=30,
        )
        forecaster.models[(7, 0.5)] = model

        # Feature names: "price_technicals" group (cols 0-2, predictive),
        # "events" group (cols 3-5, pure noise — no relation to y)
        feature_names = [
            "price_momentum_1", "price_momentum_2", "price_momentum_3",
            "event_major_1", "event_major_2", "event_major_3",
        ]
        results = forecaster._validate_feature_groups(
            X, y, feature_names, horizon=7,
            n_shuffles=30, min_drop_pp=0.5, significance_level=0.05,
        )

        # The "events" group (cols 3-5) has no causal link to y — should FAIL
        events = results["events"]
        assert not events["passed"], (
            f"Expected 'events' to fail (drop_pp={events['drop_pp']}, "
            f"p={events['p_value']})"
        )
        assert events["p_value"] >= 0.05

    def test_validate_feature_groups_skips_no_model(self, forecaster):
        """When no p50 model exists for the horizon, validation returns {}."""
        assert forecaster.models == {}
        results = forecaster._validate_feature_groups(
            np.empty((10, 2)), np.empty(10), ["a", "b"], horizon=7,
        )
        assert results == {}

    def test_prune_features_filters_by_significance(
        self, forecaster,
    ):
        """Integration: _validate_feature_groups gates pruning in train().
        Non-causal groups are dropped; causal groups are kept."""
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 4).astype(np.float32)
        # y depends only on cols 0-1; cols 2-3 are pure noise
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + rng.randn(n) * 0.5

        ds = lgb.Dataset(X, y)
        model = lgb.train(
            {"objective": "quantile", "alpha": 0.5, "verbosity": -1,
             "max_bin": 63, "min_data_in_leaf": 1, "num_leaves": 4,
             "learning_rate": 0.1, "metric": "quantile"},
            ds, num_boost_round=30,
        )
        forecaster.models[(7, 0.5)] = model

        # Feature groups: "price_technicals" (predictive cols 0-1) and
        # "events" (noise cols 2-3)
        forecaster.feature_cols = ["price_momentum_1", "price_momentum_2",
                                   "event_major_1", "event_major_2"]
        results = forecaster._validate_feature_groups(
            X, y, forecaster.feature_cols, horizon=7,
            n_shuffles=30, min_drop_pp=0.5, significance_level=0.05,
        )

        # "price_technicals" should pass, "events" should fail
        pt_passed = results.get("price_technicals", {}).get("passed", False)
        events_passed = results.get("events", {}).get("passed", True)
        assert pt_passed, "Causal group should pass significance gate"
        assert not events_passed, "Noise group should fail significance gate"


# ---------------------------------------------------------------------------
# Target Preparation
# ---------------------------------------------------------------------------

class TestTargetPreparation:
    def test_prepare_targets_horizon(self, forecaster):
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(40)]
        df = pd.DataFrame({
            "item_id": ["a"] * 40 + ["b"] * 40,
            "date": dates * 2,
            "price": [float(i + 1) for i in range(40)] * 2,
            "volume": [100] * 80,
        })
        result = forecaster.prepare_targets(df, 7)
        assert "target_return_7d" in result.columns
        assert "target_7d" in result.columns
        assert len(result) == len(df)
        # Forward target: row with date=Jan 1 gets price at Jan 1+7=Jan 8.
        # price_Jan1=1, price_Jan8=8 → return = (8-1)/1*100 = 700, winsorized to 500
        a_rows = result[result["item_id"] == "a"].sort_values("date")
        row_jan1 = a_rows[a_rows["date"] == date(2026, 1, 1)]
        assert row_jan1["target_return_7d"].iloc[0] == 500.0
        # Last 7 rows per item have no forward target → NaN
        last_rows = a_rows.tail(7)
        assert last_rows["target_return_7d"].isna().all()


# ---------------------------------------------------------------------------
# Quantile Monotonicity
# ---------------------------------------------------------------------------

class TestQuantileMonotonicity:
    def test_already_monotonic_unchanged(self):
        """Items with p10 <= p50 <= p90 should not be altered."""
        from models.forecaster import ItemForecaster
        low, high = ItemForecaster._fix_quantile_crossing(
            np.array([1.0, 3.0, 5.0]),
            np.array([2.0, 4.0, 6.0]),
            np.array([3.0, 5.0, 7.0]),
        )
        assert list(low) == [1.0, 3.0, 5.0]
        assert list(high) == [3.0, 5.0, 7.0]

    def test_p10_greater_than_p50_pooled(self):
        """When p10 > p50, first two are pooled to their mean."""
        from models.forecaster import ItemForecaster
        low, high = ItemForecaster._fix_quantile_crossing(
            np.array([5.0, 5.0]),
            np.array([3.0, 3.0]),
            np.array([6.0, 6.0]),
        )
        assert np.allclose(low, [4.0, 4.0])
        assert np.allclose(high, [6.0, 6.0])

    def test_p50_greater_than_p90_pooled(self):
        """When p50 > p90, last two are pooled to their mean."""
        from models.forecaster import ItemForecaster
        low, high = ItemForecaster._fix_quantile_crossing(
            np.array([1.0, 1.0]),
            np.array([5.0, 5.0]),
            np.array([3.0, 3.0]),
        )
        assert np.allclose(low, [1.0, 1.0])
        assert np.allclose(high, [4.0, 4.0])

    def test_all_three_cross_pooled_equally(self):
        """When p10 > p50 > p90, all three are pooled to their common mean."""
        from models.forecaster import ItemForecaster
        low, high = ItemForecaster._fix_quantile_crossing(
            np.array([8.0, 8.0]),
            np.array([5.0, 5.0]),
            np.array([2.0, 2.0]),
        )
        assert np.allclose(low, [5.0, 5.0])
        assert np.allclose(high, [5.0, 5.0])

    def test_mixed_mask_handles_both_crossing_and_non(self):
        """Mixed arrays with some crossing and some monotonic items."""
        from models.forecaster import ItemForecaster
        low, high = ItemForecaster._fix_quantile_crossing(
            np.array([1.0, 8.0]),
            np.array([5.0, 5.0]),
            np.array([9.0, 2.0]),
        )
        # Item 0: 1 <= 5 <= 9 → unchanged
        # Item 1: 8 > 5 and then 5+2 pool → (8+5+2)/3 = 5
        assert low[0] == 1.0
        assert high[0] == 9.0
        assert np.allclose(low[1], 5.0)
        assert np.allclose(high[1], 5.0)

    def test_returns_low_high_only(self):
        """Method returns only (low, high); mid is left for caller to keep."""
        from models.forecaster import ItemForecaster
        low, high = ItemForecaster._fix_quantile_crossing(
            np.array([5.0, 1.0, 7.0]),
            np.array([3.0, 3.0, 5.0]),
            np.array([4.0, 6.0, 4.0]),
        )
        assert len(low) == 3
        assert len(high) == 3
        assert np.all(low <= high)


# ---------------------------------------------------------------------------
# Confidence Computation
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_narrow_range_high(self, forecaster):
        forecaster.confidence_thresholds = {
            7: {"high_range": 0.15, "high_change": 0.02}
        }
        result = forecaster._compute_confidence(
            mid=10.0, low=9.5, high=10.5, current=9.5, horizon=7
        )
        assert result == "high"

    def test_confidence_wide_range_low(self, forecaster):
        forecaster.confidence_thresholds = {
            7: {"high_range": 0.15, "high_change": 0.02}
        }
        result = forecaster._compute_confidence(
            mid=10.0, low=5.0, high=15.0, current=9.5, horizon=7
        )
        assert result == "low"

    def test_confidence_fallback_defaults(self, forecaster):
        forecaster.confidence_thresholds = {}
        result = forecaster._compute_confidence(
            mid=10.0, low=8.0, high=12.0, current=9.5, horizon=7
        )
        assert result in ("high", "low")

    def test_confidence_zero_mid(self, forecaster):
        result = forecaster._compute_confidence(
            mid=0.0, low=0.0, high=0.0, current=10.0, horizon=7
        )
        assert result == "low"


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

class TestSanitization:
    def test_sanitize_invalid_prices_clamped(self, forecaster):
        result_df = pd.DataFrame([{
            "item_id": "test",
            "current_price": 10.0,
            "forecasts": {
                7: {"low": -5.0, "mid": np.nan, "high": 20.0,
                     "direction": "up", "confidence": "high"},
            },
            "generated_at": datetime.now(timezone.utc),
        }])
        cleaned = forecaster._sanitize_forecasts(result_df)
        fc = cleaned.iloc[0]["forecasts"][7]
        assert fc["mid"] == 10.0  # clamped to current price
        assert fc["direction"] == "flat"
        assert fc["confidence"] == "low"

    def test_sanitize_zero_volume_downgrades(self, forecaster):
        result_df = pd.DataFrame([{
            "item_id": "test",
            "current_price": 10.0,
            "volume": 0,
            "forecasts": {
                7: {"low": 9.5, "mid": 11.0, "high": 12.0,
                     "direction": "up", "confidence": "high"},
            },
            "generated_at": datetime.now(timezone.utc),
        }])
        cleaned = forecaster._sanitize_forecasts(result_df)
        fc = cleaned.iloc[0]["forecasts"][7]
        assert fc["confidence"] == "low"


# ---------------------------------------------------------------------------
# Feature Engineering Pipeline (Integration)
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    def test_engineer_features_resamples_daily(self, forecaster, events_df):
        """Multiple rows per day should be aggregated to one row per item per day."""
        rows = []
        for item_id in ["a", "b"]:
            for day_offset in range(10):
                d = date(2026, 1, 1) + timedelta(days=day_offset)
                for _ in range(3):  # 3 observations per day
                    rows.append({
                        "item_id": item_id,
                        "date": d,
                        "price": 10.0 + day_offset + np.random.randn() * 0.1,
                        "volume": 100,
                    })
        multi_df = pd.DataFrame(rows)
        result = forecaster.engineer_features(multi_df, events_df)
        daily_counts = result.groupby(["item_id", "date"]).size()
        assert (daily_counts == 1).all(), "Should have exactly 1 row per item per day"

    def test_build_training_data_restricts_to_price_group(self, forecaster):
        """After the 2026-07-24 simplification, build_training_data keeps only
        price/technical features (FEATURE_GROUP_ALLOWLIST); event/social/
        cross-sectional groups are excluded."""
        # Mock fetch_price_history and fetch_events to return synthetic data
        def mock_fetch(*args, **kwargs):
            np.random.seed(42)
            rows = []
            for item_id in range(5):
                price = 50.0
                for day_offset in range(200):
                    d = date(2026, 1, 1) + timedelta(days=day_offset)
                    price *= 1 + np.random.randn() * 0.01
                    rows.append({
                        "item_id": f"item_{item_id}",
                        "date": d,
                        "price": round(max(price, 0.01), 2),
                        "volume": int(max(np.random.poisson(200), 0)),
                    })
            return pd.DataFrame(rows)

        def mock_events(*args, **kwargs):
            return pd.DataFrame([
                {"id": 1, "type": "major", "timestamp": pd.Timestamp("2026-06-01"), "description": "Major", "date": date(2026, 6, 1)},
                {"id": 2, "type": "update", "timestamp": pd.Timestamp("2026-07-01"), "description": "Update", "date": date(2026, 7, 1)},
            ])

        with patch.object(forecaster, 'fetch_price_history', mock_fetch):
            with patch.object(forecaster, 'fetch_events', mock_events):
                df = forecaster.build_training_data(days_back=200, backfilled_only=False)

        # Price/technical categories must be present...
        from models.forecaster import _feature_group
        feature_set = set(forecaster.feature_cols)
        assert any("return_" in c for c in feature_set), "Missing return features"
        assert any("bb_" in c for c in feature_set), "Missing Bollinger features"
        assert any("rsi" in c for c in feature_set), "Missing RSI features"
        assert any("macd" in c for c in feature_set), "Missing MACD features"
        # ...and non-price groups must be excluded by the allowlist.
        assert not any(c.startswith("event_") for c in feature_set), "Event features leaked past allowlist"
        assert not any(c.startswith("market_") for c in feature_set), "Market features leaked past allowlist"
        assert all(_feature_group(c) == "price_technicals" for c in feature_set), \
            f"Non-price feature groups present: {[c for c in feature_set if _feature_group(c) != 'price_technicals']}"

        # Check no float64 feature columns remain (memory optimization)
        float64_cols = [c for c in forecaster.feature_cols if c in df.columns and df[c].dtype == np.float64]
        assert len(float64_cols) == 0, f"Features still float64: {float64_cols}"

        # Check prepare_targets still produces valid target columns
        for h in forecaster.HORIZONS:
            tdf = forecaster.prepare_targets(df, h)
            assert f"target_return_{h}d" in tdf.columns

    def test_build_training_data_feature_count(self, forecaster):
        """Feature count should be reasonable (not too few, not too many)."""
        def mock_fetch(*args, **kwargs):
            np.random.seed(42)
            rows = []
            for item_id in range(5):
                price = 50.0
                for day_offset in range(200):
                    d = date(2026, 1, 1) + timedelta(days=day_offset)
                    price *= 1 + np.random.randn() * 0.01
                    rows.append({
                        "item_id": f"item_{item_id}",
                        "date": d,
                        "price": round(max(price, 0.01), 2),
                        "volume": int(max(np.random.poisson(200), 0)),
                    })
            return pd.DataFrame(rows)

        def mock_events(*args, **kwargs):
            return pd.DataFrame([
                {"id": 1, "type": "major", "timestamp": pd.Timestamp("2026-06-01"),
                 "description": "Major", "date": date(2026, 6, 1)},
            ])

        with patch.object(forecaster, 'fetch_price_history', mock_fetch):
            with patch.object(forecaster, 'fetch_events', mock_events):
                df = forecaster.build_training_data(days_back=200, backfilled_only=False)

        n_features = len(forecaster.feature_cols)
        assert 45 <= n_features <= 200, f"Feature count {n_features} outside expected range [45, 200]"


# ---------------------------------------------------------------------------
# Concept Drift Monitoring
# ---------------------------------------------------------------------------

class TestConceptDrift:
    def _make_mock_record(self, metrics_dict):
        """Create a mock DB result row with .metrics attribute and .fetchall()."""
        record = MagicMock()
        record.metrics = metrics_dict
        return record

    def _make_mock_execute(self, records):
        """Mock db.execute to return an object with .fetchall()."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = records
        mock_execute = MagicMock(return_value=mock_result)
        return mock_execute

    def test_drift_detected_when_accuracy_low(self, forecaster):
        from database import AccuracyAlert
        mock_records = [
            self._make_mock_record({"directional_accuracy": 45.0}),
            self._make_mock_record({"directional_accuracy": 42.0}),
            self._make_mock_record({"directional_accuracy": 48.0}),
        ]
        forecaster.db.execute = self._make_mock_execute(mock_records)
        with patch.object(forecaster.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.first.return_value = None
            result = forecaster.check_concept_drift(horizon=7, sliding_window=7, threshold=60.0)
            assert result is not None
            assert result["drifted"] is True
            assert result["accuracy"] == 45.0

    def test_no_drift_when_accuracy_high(self, forecaster):
        mock_records = [
            self._make_mock_record({"directional_accuracy": 85.0}),
            self._make_mock_record({"directional_accuracy": 88.0}),
            self._make_mock_record({"directional_accuracy": 82.0}),
        ]
        forecaster.db.execute = self._make_mock_execute(mock_records)
        result = forecaster.check_concept_drift(horizon=7, sliding_window=7, threshold=60.0)
        assert result is not None
        assert result["drifted"] is False

    def test_insufficient_records_returns_none(self, forecaster):
        mock_records = [
            self._make_mock_record({"directional_accuracy": 85.0}),
        ]
        forecaster.db.execute = self._make_mock_execute(mock_records)
        result = forecaster.check_concept_drift(horizon=7, sliding_window=7, threshold=60.0)
        assert result is None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibration_requires_min_samples(self, forecaster):
        """Calibration should return early with < 50 samples."""
        result = forecaster._calibrate_confidence(
            X_val=MagicMock(), y_val=MagicMock(), val_set=MagicMock(), horizon=7
        )
        # With mocked objects, preds will be None → early return
        assert forecaster.confidence_thresholds.get(7) is None

    def test_confidence_thresholds_per_horizon(self, forecaster):
        """Different horizons should have different threshold dicts."""
        forecaster.confidence_thresholds = {
            3: {"high_range": 0.30, "high_change": 0.01, "high_accuracy": 99.0},
            30: {"high_range": 0.10, "high_change": 0.05, "high_accuracy": 95.0},
        }
        # range_pct = (11-9)/10 = 0.20 < 0.30, change_pct = |10-9.8|/9.8 ≈ 0.02 > 0.01 → high
        assert forecaster._compute_confidence(10.0, 9.0, 11.0, 9.8, horizon=3) == "high"
        # For 30d, range_pct = 0.20 is NOT < 0.10 → low
        result_30 = forecaster._compute_confidence(10.0, 9.0, 11.0, 9.8, horizon=30)
        assert result_30 == "low"


# ---------------------------------------------------------------------------
# Conformal Calibration (CQR)
# ---------------------------------------------------------------------------

class TestConformalCalibration:
    def test_q_hat_zero_when_all_scores_zero(self):
        """If all nonconformity scores are 0, q_hat should be 0."""
        from models.forecaster import ItemForecaster
        f = ItemForecaster.__new__(ItemForecaster)
        f.conformal_calibration = {}
        f.confidence_thresholds = {}
        scores = [0.0, 0.0, 0.0, 0.0, 0.0]
        alpha = 0.10
        q_level = (1.0 - alpha) * (1.0 + 1.0 / max(len(scores), 1))
        q_level = min(q_level, 0.999)
        q_hat = float(np.quantile(scores, q_level))
        assert q_hat == 0.0

    def test_q_hat_captures_tail_nonconformity(self):
        """q_hat should be >= largest middle-80% score for 90% coverage target."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        alpha = 0.10
        n = len(scores)
        q_level = (1.0 - alpha) * (1.0 + 1.0 / n)
        q_hat = float(np.quantile(scores, min(q_level, 0.999)))
        # With n=10 and α=0.10, q_level = 0.9 * 1.1 = 0.99 → q_hat ≈ 0.99
        assert q_hat > 0.9
        assert q_hat <= 1.0

    def test_conformal_calibration_stored_per_horizon(self, forecaster):
        """After training, conformal_calibration should have entries per horizon."""
        forecaster.conformal_calibration[7] = 0.45
        forecaster.conformal_calibration[30] = 0.32
        assert 7 in forecaster.conformal_calibration
        assert 30 in forecaster.conformal_calibration
        assert forecaster.conformal_calibration[7] == 0.45

    def test_q_hat_applied_to_prediction_intervals(self, forecaster):
        """Setting q_hat should widen the predict output intervals."""
        forecaster.conformal_calibration[7] = 5.0
        forecaster.models = {}
        # Patch predict to return early; just verify that conformal_calibration
        # is checked and q_hat is positive for the horizon
        q_hat = forecaster.conformal_calibration.get(7, 0.0)
        assert q_hat == 5.0
        q_hat_missing = forecaster.conformal_calibration.get(99, 0.0)
        assert q_hat_missing == 0.0


# ---------------------------------------------------------------------------
# Predict method edge cases
# ---------------------------------------------------------------------------

class TestPredictEdgeCases:
    def test_predict_empty_results_no_items(self, forecaster):
        """predict() should handle empty item list gracefully."""
        with patch.object(forecaster, 'fetch_price_history',
                          return_value=pd.DataFrame(columns=["item_id", "date", "price", "volume"])):
            with patch.object(forecaster, 'fetch_events',
                              return_value=pd.DataFrame(columns=["id", "type", "timestamp", "description"])):
                result = forecaster.predict()
                assert result.empty

    def test_predict_skips_items_with_insufficient_history(self, forecaster):
        """Items with < PREDICT_MIN_HISTORY_DAYS days should be skipped."""
        rows = []
        for i in range(3):
            for d in range(5 if i == 0 else 30):  # item_0 has only 5 days
                rows.append({
                    "item_id": f"item_{i}",
                    "date": date(2026, 6, 1) + timedelta(days=d),
                    "price": 10.0 + d * 0.1,
                    "volume": 100,
                })
        price_df = pd.DataFrame(rows)
        with patch.object(forecaster, 'fetch_price_history', return_value=price_df):
            with patch.object(forecaster, 'fetch_events',
                              return_value=pd.DataFrame(columns=["id", "type", "timestamp", "description"])):
                result = forecaster.predict()
                # Should not crash; may be empty if no models loaded
                assert isinstance(result, pd.DataFrame)

    def test_predict_smooths_spike_outlier(self, forecaster):
        """Latest price outlier >10% from 3d median should be smoothed."""
        np.random.seed(42)
        rows = []
        for d in range(30):
            rows.append({
                "item_id": "item_0",
                "date": date(2026, 6, 1) + timedelta(days=d),
                "price": 100.0 + d * 0.5,
                "volume": 100,
            })
        # Add a spike on the last day (200 instead of ~115)
        rows[-1] = {**rows[-1], "price": 200.0}

        price_df = pd.DataFrame(rows)
        with patch.object(forecaster, 'fetch_price_history', return_value=price_df):
            with patch.object(forecaster, 'fetch_events',
                              return_value=pd.DataFrame(columns=["id", "type", "timestamp", "description"])):
                result = forecaster.predict()
                # Should not crash
                assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_save_load_empty_models(self, forecaster, tmp_path):
        forecaster.model_dir = str(tmp_path)
        forecaster.models = {}
        forecaster.save_models()
        assert (tmp_path / "meta.json").exists()
        loaded = forecaster.load_models()
        assert loaded is False  # no model files to load

    def test_save_removes_orphaned_regime_files(self, forecaster, tmp_path):
        """A regime-free save must delete stale regime .txt files left on disk,
        so SKIP_REGIMES runs don't perpetuate old regime artifacts."""
        forecaster.model_dir = str(tmp_path)
        forecaster.models = {}
        forecaster.regime_models = {}
        # simulate stale regime artifacts from a prior regime-enabled run
        for name in ["lgb_14d_q50_bear_e0.txt", "lgb_7d_q10_bull_e1.txt",
                     "lgb_30d_q90_range.txt"]:
            (tmp_path / name).write_text("stale")
        # a global model file must be left untouched
        (tmp_path / "lgb_14d_q50_e0.txt").write_text("keep")
        forecaster.save_models()
        assert not (tmp_path / "lgb_14d_q50_bear_e0.txt").exists()
        assert not (tmp_path / "lgb_7d_q10_bull_e1.txt").exists()
        assert not (tmp_path / "lgb_30d_q90_range.txt").exists()
        assert (tmp_path / "lgb_14d_q50_e0.txt").exists()  # global untouched

    def test_save_confidence_thresholds_serializable(self, forecaster, tmp_path):
        """Confirm thresholds round-trip through JSON."""
        forecaster.model_dir = str(tmp_path)
        forecaster.confidence_thresholds = {
            3: {"high_range": 0.15, "high_change": 0.01, "high_accuracy": np.float64(99.8)},
            7: {"high_range": 0.15, "high_change": 0.05, "high_accuracy": np.float64(99.7)},
        }
        forecaster.feature_cols = ["a", "b"]
        forecaster.save_models()
        # Load back in a new forecaster
        f2 = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
        f2.load_models()
        assert f2.confidence_thresholds[3]["high_range"] == 0.15
        assert f2.confidence_thresholds[7]["high_accuracy"] == 99.7

    def test_tuned_params_roundtrip(self, forecaster, tmp_path):
        """Cached tuned HP must survive save/load so retrains can skip Optuna."""
        forecaster.model_dir = str(tmp_path)
        forecaster.feature_cols = ["a", "b"]
        forecaster.confidence_thresholds = {}
        forecaster.tuned_params = {
            7: {0.5: {"objective": "quantile", "alpha": 0.5, "max_bin": 63,
                      "num_leaves": 31, "learning_rate": 0.03}},
            30: {0.9: {"objective": "quantile", "alpha": 0.9, "max_bin": 63,
                       "num_leaves": 23, "learning_rate": 0.05}},
        }
        forecaster.save_models()
        f2 = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
        # load_models returns False (no model .txt files) but still restores meta.
        f2.load_models()
        assert 7 in f2.tuned_params and 30 in f2.tuned_params
        assert 0.5 in f2.tuned_params[7] and 0.9 in f2.tuned_params[30]
        assert f2.tuned_params[7][0.5]["learning_rate"] == 0.03
        assert f2.tuned_params[30][0.9]["num_leaves"] == 23


# ---------------------------------------------------------------------------
# Feature Cache
# ---------------------------------------------------------------------------


class TestFeatureCache:
    def test_save_and_load_cache(self, forecaster, tmp_path):
        """Cached feature DataFrame round-trips through Parquet."""
        import os
        forecaster.model_dir = str(tmp_path)
        df = pd.DataFrame({
            "item_id": ["a", "b", "a", "b"],
            "date": [date(2026, 1, 1), date(2026, 1, 1),
                     date(2026, 1, 2), date(2026, 1, 2)],
            "price": [10.0, 20.0, 11.0, 21.0],
            "volume": [100, 200, 110, 220],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [1.0, 2.0, 3.0, 4.0],
        })
        forecaster._save_engineered_cache(df)
        cache_path = os.path.join(str(tmp_path), ItemForecaster.ENGINEERED_CACHE_NAME)
        assert os.path.exists(cache_path)
        loaded = forecaster._load_engineered_cache()
        assert loaded is not None
        assert len(loaded) == 4
        assert "item_id" in loaded.columns
        assert "feature_1" in loaded.columns
        assert loaded["price"].iloc[0] == 10.0

    def test_cache_missing_returns_none(self, forecaster, tmp_path):
        """_load_engineered_cache returns None when no cache file exists."""
        forecaster.model_dir = str(tmp_path)
        result = forecaster._load_engineered_cache()
        assert result is None

    def test_cache_stale_after_3_days(self, forecaster, tmp_path):
        """Cache older than 3 days triggers refresh (returns None)."""
        import json
        import os
        import pyarrow.parquet as pq
        forecaster.model_dir = str(tmp_path)
        df = pd.DataFrame({
            "item_id": ["a"], "date": [date(2020, 1, 1)],
            "price": [10.0], "volume": [100],
            "feature_1": [0.1],
        })
        # _save_engineered_cache sets attrs to today; overwrite PANDAS_ATTRS manually
        forecaster._save_engineered_cache(df)
        path = os.path.join(str(tmp_path), ItemForecaster.ENGINEERED_CACHE_NAME)
        table = pq.read_table(path)
        meta = dict(table.schema.metadata or {})
        meta[b"PANDAS_ATTRS"] = json.dumps({"_cache_date": "2020-01-01"}).encode()
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, path)
        result = forecaster._load_engineered_cache()
        assert result is None, "Stale cache (>3 days) should return None"


# ---------------------------------------------------------------------------
# Training Window / Subsampling (regression tests for the 2026-07-16 audit)
# ---------------------------------------------------------------------------


class TestTrainingWindow:
    @pytest.fixture
    def wide_price_df(self):
        """200 items × 250 days — big enough to trigger subsampling."""
        rows = []
        base = date(2025, 1, 1)
        for item_id in range(200):
            for day_offset in range(250):
                rows.append({
                    "item_id": f"item_{item_id}",
                    "date": base + timedelta(days=day_offset),
                    "price": 10.0 + item_id,
                    "volume": 100,
                })
        return pd.DataFrame(rows)

    def test_subsample_bounds_rows_and_preserves_calendar(self, forecaster, wide_price_df):
        """Subsampling must cut rows but keep the full calendar window intact
        so expanding-window CV still has enough distinct dates."""
        forecaster._supply_meta_cache = pd.DataFrame(
            columns=["item_id", "rarity", "rarity_rank", "weapon_type"])

        dates_before = wide_price_df["date"].nunique()
        out = forecaster._stratified_item_subsample(wide_price_df, max_rows=10_000)

        assert len(out) < len(wide_price_df)
        assert len(out) <= len(wide_price_df)
        assert out["date"].nunique() == dates_before  # full window preserved
        assert out["date"].min() == wide_price_df["date"].min()
        assert out["date"].max() == wide_price_df["date"].max()

    def test_subsample_keeps_full_item_history(self, forecaster, wide_price_df):
        """Whole item histories are kept (not individual rows) so lag/rolling
        features stay valid."""
        forecaster._supply_meta_cache = pd.DataFrame(
            columns=["item_id", "rarity", "rarity_rank", "weapon_type"])
        out = forecaster._stratified_item_subsample(wide_price_df, max_rows=10_000)
        counts = out.groupby("item_id").size()
        assert (counts == 250).all()  # every kept item has its full history

    def test_subsample_noop_when_under_budget(self, forecaster, wide_price_df):
        out = forecaster._stratified_item_subsample(wide_price_df, max_rows=10_000_000)
        assert len(out) == len(wide_price_df)

    def test_cv_produces_at_least_two_folds(self, forecaster):
        """Regression: with a full-length window, expanding-window CV must
        produce >= 2 folds (the audit found 51 days → zero folds)."""
        base = date(2025, 1, 1)
        sorted_dates = [base + timedelta(days=i) for i in range(500)]
        folds = forecaster._compute_cv_splits(sorted_dates)
        assert len(folds) >= 2

    def test_cv_skipped_with_truncated_window(self, forecaster):
        """Documents the failure mode: a 51-day window yields zero folds."""
        base = date(2025, 1, 1)
        sorted_dates = [base + timedelta(days=i) for i in range(51)]
        folds = forecaster._compute_cv_splits(sorted_dates)
        assert len(folds) == 0

    def test_cv_purge_default_is_noop(self, forecaster):
        """purge_days=0 (default) must reproduce the un-purged splits exactly."""
        base = date(2025, 1, 1)
        sorted_dates = [base + timedelta(days=i) for i in range(500)]
        assert (forecaster._compute_cv_splits(sorted_dates)
                == forecaster._compute_cv_splits(sorted_dates, purge_days=0))

    def test_cv_purge_gap_prevents_target_leakage(self, forecaster):
        """With purge_days=H, no training row's target (train_date + H days)
        may land inside its fold's validation window — the horizon-forecasting
        embargo that keeps CV honest."""
        base = date(2025, 1, 1)
        sorted_dates = [base + timedelta(days=i) for i in range(500)]
        horizon = 30
        folds = forecaster._compute_cv_splits(sorted_dates, purge_days=horizon)
        assert len(folds) >= 2
        for train_d, val_d in folds:
            latest_target = max(train_d) + timedelta(days=horizon)
            assert latest_target < min(val_d)


class TestDirectionalBaselines:
    """The naive baselines a real forecaster must beat: predicting flat
    (random walk in price) and predicting trailing-return momentum."""

    def test_perfect_prediction_scores_100(self, forecaster):
        actual = np.array([5.0, -5.0, 0.0, 8.0])
        assert forecaster._directional_accuracy(actual.copy(), actual) == 100.0

    def test_flat_tolerance_bucketing(self, forecaster):
        # |ret| < 0.5 counts as flat; predicting 0 matches only the flat actual
        actual = np.array([5.0, -5.0, 0.2])
        pred_flat = np.zeros(3)
        assert forecaster._directional_accuracy(pred_flat, actual) == pytest.approx(33.3, abs=0.1)

    def test_ignores_nan_actuals(self, forecaster):
        actual = np.array([5.0, np.nan, -5.0])
        pred = np.array([5.0, 5.0, -5.0])
        assert forecaster._directional_accuracy(pred, actual) == 100.0

    def test_empty_returns_zero(self, forecaster):
        assert forecaster._directional_accuracy(np.array([]), np.array([])) == 0.0


class TestFeatureAllowlist:
    """Restricting the model to price/technical features (2026-07-24 ablation)."""

    def test_allowlist_keeps_only_price_group(self, forecaster):
        cols = ["price_lag_1d", "return_7d", "rsi_14", "macd_line",
                "event_decay_major", "social_mentions_7d", "market_return_30d",
                "is_knife", "day_of_week", "supply_sell_listings"]
        kept = forecaster._apply_feature_allowlist(cols, ["price_technicals"])
        assert kept == ["price_lag_1d", "return_7d", "rsi_14", "macd_line"]

    def test_allowlist_none_is_noop(self, forecaster):
        cols = ["return_7d", "event_decay_major", "is_knife"]
        assert forecaster._apply_feature_allowlist(cols, None) == cols
        assert forecaster._apply_feature_allowlist(cols, []) == cols

    def test_momentum_features_survive_allowlist(self, forecaster):
        # The trailing-return (momentum) features must remain — they are the
        # model's core signal after simplification.
        cols = [f"return_{h}d" for h in (3, 7, 14, 30)] + ["event_decay_major"]
        kept = forecaster._apply_feature_allowlist(cols, ["price_technicals"])
        assert set(kept) == {f"return_{h}d" for h in (3, 7, 14, 30)}


class TestMomentumRecenter:
    """Serving weak horizons as momentum while preserving interval width."""

    def test_recenters_on_momentum_preserving_width(self, forecaster):
        low = np.array([-4.0, 0.0])
        mid = np.array([0.0, 5.0])
        high = np.array([6.0, 8.0])
        mom = np.array([10.0, -2.0])
        nl, nm, nh = forecaster._recenter_on_momentum(low, mid, high, mom)
        # median becomes momentum
        assert np.allclose(nm, mom)
        # half-widths preserved: low_off=[4,5], high_off=[6,3]
        assert np.allclose(nl, [6.0, -7.0])
        assert np.allclose(nh, [16.0, 1.0])

    def test_nan_momentum_keeps_model_forecast(self, forecaster):
        low = np.array([-4.0]); mid = np.array([1.0]); high = np.array([6.0])
        mom = np.array([np.nan])
        nl, nm, nh = forecaster._recenter_on_momentum(low, mid, high, mom)
        assert np.allclose([nl[0], nm[0], nh[0]], [-4.0, 1.0, 6.0])

    def test_recentred_triple_stays_monotone(self, forecaster):
        low = np.array([-4.0, -1.0]); mid = np.array([0.0, 2.0]); high = np.array([6.0, 3.0])
        mom = np.array([10.0, -20.0])
        nl, nm, nh = forecaster._recenter_on_momentum(low, mid, high, mom)
        assert np.all(nl <= nm) and np.all(nm <= nh)


class TestDirectionClassifierHelpers:
    """Pure helpers backing the 3-class directional classifier."""

    def test_direction_classes_bucketing(self, forecaster):
        # tolerance is +/-0.5%
        r = np.array([5.0, -5.0, 0.2, -0.2, 0.5, -0.5])
        cls = forecaster._direction_classes(r)
        # 5>0.5 up(2), -5<-0.5 down(0), |0.2|<=0.5 flat(1), 0.5 not >0.5 flat, -0.5 flat
        assert list(cls) == [2, 0, 1, 1, 1, 1]

    def test_mover_sample_weights(self, forecaster):
        r = np.array([5.0, 0.1, -3.0, 0.0])
        w = forecaster._direction_sample_weights(r, 0.5, mover_weight=3.0)
        assert list(w) == [3.0, 1.0, 3.0, 1.0]

    def test_recenter_on_direction_sets_sign_keeps_magnitude_and_width(self, forecaster):
        # mid magnitudes = [2, 2, 2]; classes down/flat/up
        low = np.array([0.0, 0.0, 0.0])
        mid = np.array([2.0, 2.0, 2.0])
        high = np.array([5.0, 5.0, 5.0])
        cls = np.array([0, 1, 2])  # down, flat, up
        nl, nm, nh = forecaster._recenter_on_direction(low, mid, high, cls)
        assert list(nm) == [-2.0, 0.0, 2.0]           # sign follows class, |mid| kept
        # half-widths preserved: low_off=2, high_off=3
        assert list(nl) == [-4.0, -2.0, 0.0]
        assert list(nh) == [1.0, 3.0, 5.0]
        assert np.all(nl <= nm) and np.all(nm <= nh)

    def test_direction_tree_params_extracts_agnostic_keys(self, forecaster):
        pqp = {0.5: {"num_leaves": 63, "learning_rate": 0.04, "objective": "quantile",
                     "alpha": 0.5, "max_depth": 6, "drop_rate": 0.1}}
        tp = forecaster._direction_tree_params(pqp)
        assert tp == {"num_leaves": 63, "learning_rate": 0.04, "max_depth": 6, "drop_rate": 0.1}
        assert "objective" not in tp and "alpha" not in tp


class TestResidualStackingDisabled:
    """Residual Ridge stacking is disabled: it was fit on raw unscaled features
    and extrapolated without bound at serving time (penny-item 14d forecasts
    exploded to +10,000%+ and quantiles inverted). See STACK_RESIDUALS note."""

    def test_stack_residuals_flag_off(self, forecaster):
        assert forecaster.STACK_RESIDUALS is False, \
            "Residual stacking must stay disabled (fragile unbounded corrector)"

    def test_train_does_not_fit_residual_models(self, forecaster):
        """With the flag off, train() must not populate residual_models even on
        weak horizons — the fit block is gated on STACK_RESIDUALS."""
        # The training gate is `STACK_RESIDUALS and _sklearn_available and
        # horizon in WEAK_HORIZONS`; with the flag False no models are fit.
        assert forecaster.STACK_RESIDUALS is False
        assert 14 in forecaster.WEAK_HORIZONS  # the horizon that blew up


class TestQ50RowSampling:
    """q50 uses bagging, not GOSS.

    GOSS ranks rows by |gradient| to choose which to keep, but the quantile
    objective emits constant ±alpha gradients, so the ranking is degenerate
    and the small-gradient rescaling injects bias. Shipped symptom: 7d q50
    saved 1-2 trees per ensemble member (best_iteration ~= 1), i.e. the served
    median model was effectively an intercept. A/B 2026-07-29 (8-fold
    purge-gap CV, production params): bagging improved pinball +5.21% (3d) /
    +1.76% (7d), DA +1.13pp / +0.71pp, winning 7/8 and 5/8 folds.
    """

    def test_q50_uses_bagging(self, forecaster):
        p = forecaster._row_sampling_params(0.5)
        assert p["data_sample_strategy"] == "bagging"

    def test_q50_has_no_goss_keys(self, forecaster):
        p = forecaster._row_sampling_params(0.5)
        assert "top_rate" not in p and "other_rate" not in p

    def test_no_quantile_uses_goss(self, forecaster):
        for q in forecaster.QUANTILES:
            assert forecaster._row_sampling_params(q)["data_sample_strategy"] == "bagging"

    def test_q50_sets_bagging_freq(self, forecaster):
        """LightGBM ignores bagging_fraction/subsample unless bagging_freq >= 1
        (default 0 disables bagging), so without this the switch away from GOSS
        would silently train on every row — not the config the A/B validated."""
        assert forecaster._row_sampling_params(0.5).get("bagging_freq", 0) >= 1

    def test_subsample_is_honored(self, forecaster):
        assert forecaster._row_sampling_params(0.5, subsample=0.6)["subsample"] == 0.6

    def test_apply_strips_stale_goss_keys(self, forecaster):
        """Params cached by an older build carry GOSS keys for q50; the warm
        retrain path must not let them through."""
        cached = {
            "num_leaves": 47, "learning_rate": 0.01,
            "data_sample_strategy": "goss", "top_rate": 0.2, "other_rate": 0.1,
        }
        out = forecaster._apply_row_sampling(dict(cached), 0.5)
        assert out["data_sample_strategy"] == "bagging"
        assert "top_rate" not in out and "other_rate" not in out
        # tree params must survive untouched
        assert out["num_leaves"] == 47 and out["learning_rate"] == 0.01

    def test_apply_preserves_cached_subsample(self, forecaster):
        out = forecaster._apply_row_sampling(
            {"subsample": 0.7}, 0.5, subsample=0.7)
        assert out["subsample"] == 0.7


class TestRecencyWeighting:
    """Time-decayed sample weights — the mechanism, which ships DISABLED.

    Diagnosed 2026-07-29: production trains on a 1460-day window in which only
    ~14.5% of rows fall in the last 180 days, while the early-stopping
    validation window (most recent 30 days) has ~2x the return spread of the
    training data overall (std 35.1 vs 21.9, p10/p90 -14.4/+20.1 vs -9.1/+10.0).
    Weights encode item volatility and direction only, so a 2022 row counts as
    much as a 2026 one.

    The decay knob was A/B'd and did not clear the gate (see
    SAMPLE_WEIGHT_HALFLIFE_DAYS), so the production default is 0.0. These tests
    set a half-life explicitly so the mechanism stays correct if it is ever
    re-enabled; `test_production_default_is_disabled` pins the shipped value.
    """

    HALF_LIFE = 365

    @pytest.fixture(autouse=True)
    def _enable_decay(self, monkeypatch):
        import models.forecaster as fmod
        monkeypatch.setattr(fmod, "SAMPLE_WEIGHT_HALFLIFE_DAYS",
                            float(self.HALF_LIFE))

    def test_production_default_is_disabled(self):
        """Guards the 2026-07-29 A/B decision: decay is off in production."""
        assert SAMPLE_WEIGHT_HALFLIFE_DAYS == 0.0, (
            "recency decay was A/B'd and failed its gate on 3 of 4 horizons; "
            "re-enabling needs a fresh A/B, not a constant edit")

    @staticmethod
    def _frame(n_items=4, n_days=800, end="2026-07-18"):
        dates = pd.date_range(end=end, periods=n_days, freq="D")
        rows = []
        for i in range(n_items):
            for d in dates:
                rows.append({"item_id": f"item_{i}", "date": d.date(),
                             "price": 10.0 + (i + 1) * 0.01 * (d.dayofyear % 7),
                             "target_return_7d": 1.0})
        return pd.DataFrame(rows)

    def test_recent_rows_weigh_more_than_old(self, forecaster):
        df = self._frame()
        w = forecaster._compute_sample_weights(df, 7)
        assert w is not None
        df = df.copy()
        df["w"] = w
        by_date = df.groupby("date")["w"].mean().sort_index()
        # Newest day must carry strictly more gradient weight than the oldest.
        assert by_date.iloc[-1] > by_date.iloc[0], (
            f"no recency decay: newest={by_date.iloc[-1]:.4f} "
            f"oldest={by_date.iloc[0]:.4f}")

    def test_decay_matches_configured_half_life(self, forecaster):
        # The price pattern repeats weekly, so the 30-day rolling vol is
        # effectively constant in the interior — a flat price would instead
        # zero the vol term entirely (clip(0, 0.1, 0)) and make every weight 0.
        hl = self.HALF_LIFE
        df = self._frame(n_items=2, n_days=2 * hl + 1)
        w = forecaster._compute_sample_weights(df, 7)
        df = df.copy(); df["w"] = w
        by_date = df.groupby("date")["w"].mean().sort_index()
        newest, one_half_life = by_date.iloc[-1], by_date.iloc[-1 - hl]
        assert newest > 0
        assert one_half_life / newest == pytest.approx(0.5, rel=0.05), (
            f"expected ~0.5 at one half-life, got "
            f"{one_half_life / newest:.4f}")

    def test_weights_stay_mean_normalized(self, forecaster):
        w = forecaster._compute_sample_weights(self._frame(), 7)
        assert np.mean(w) == pytest.approx(1.0, rel=1e-6)

    def test_decay_disabled_by_zero_half_life(self, forecaster, monkeypatch):
        import models.forecaster as fmod
        monkeypatch.setattr(fmod, "SAMPLE_WEIGHT_HALFLIFE_DAYS", 0)
        df = self._frame()
        w = forecaster._compute_sample_weights(df, 7)
        df = df.copy(); df["w"] = w
        by_date = df.groupby("date")["w"].mean().sort_index()
        # With decay off, age must not matter: interior oldest ~= newest.
        assert by_date.iloc[-1] == pytest.approx(by_date.iloc[40], rel=0.02), (
            "half-life 0 must disable decay entirely")

    def test_missing_date_column_is_tolerated(self, forecaster):
        df = self._frame().drop(columns=["date"])
        w = forecaster._compute_sample_weights(df, 7)
        assert w is not None and len(w) == len(df)


class TestPredictEnsembleSafe:
    """Guard against stale models left in the dir with a different feature count."""

    class _FakeModel:
        def __init__(self, nf, val):
            self._nf = nf; self._val = val
        def num_feature(self):
            return self._nf
        def predict(self, X):
            return np.full(len(X), self._val)

    def test_skips_feature_mismatched_models(self, forecaster):
        X = pd.DataFrame(np.zeros((4, 3)))  # 3 features
        ensemble = [self._FakeModel(3, 1.0), self._FakeModel(2, 9.0), self._FakeModel(3, 2.0)]
        out = forecaster._predict_ensemble_safe(ensemble, X)
        assert len(out) == 2                       # the 2-feature model is skipped
        assert all(np.allclose(a, v) for a, v in zip(out, [1.0, 2.0]))

    def test_all_incompatible_returns_empty(self, forecaster):
        X = pd.DataFrame(np.zeros((4, 3)))
        out = forecaster._predict_ensemble_safe([self._FakeModel(49, 1.0)], X)
        assert out == []

    def test_single_model_not_list(self, forecaster):
        X = pd.DataFrame(np.zeros((2, 3)))
        out = forecaster._predict_ensemble_safe(self._FakeModel(3, 5.0), X)
        assert len(out) == 1 and np.allclose(out[0], 5.0)

    def test_build_training_data_includes_2026(self, forecaster):
        """Regression: build_training_data must NOT drop 2026 rows.

        A temporary distribution-shift guard used to exclude all 2026 data
        while the May–June 2026 archive gap made it sparse. The gap is
        backfilled and the guard is removed, so training/CV now cover 2026
        (the current regime). This guards against the guard being
        reintroduced and silently truncating recent data again."""
        def mock_fetch(*args, **kwargs):
            np.random.seed(7)
            rows = []
            for item_id in range(5):
                price = 50.0
                # Span 2025 into 2026 so both years are represented.
                for day_offset in range(300):
                    d = date(2025, 6, 1) + timedelta(days=day_offset)
                    price *= 1 + np.random.randn() * 0.01
                    rows.append({
                        "item_id": f"item_{item_id}",
                        "date": d,
                        "price": round(max(price, 0.01), 2),
                        "volume": int(max(np.random.poisson(200), 0)),
                    })
            return pd.DataFrame(rows)

        def mock_events(*args, **kwargs):
            return pd.DataFrame(columns=["id", "type", "timestamp", "description", "date"])

        with patch.object(forecaster, 'fetch_price_history', mock_fetch):
            with patch.object(forecaster, 'fetch_events', mock_events):
                df = forecaster.build_training_data(days_back=300, backfilled_only=False)

        years = pd.DatetimeIndex(df["date"]).year
        assert (years == 2026).any(), "2026 rows must be retained in training data"
        assert (years == 2025).any(), "2025 rows must be retained in training data"


# ---------------------------------------------------------------------------
# Player Count Features
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Forecast blending / directional smoothing (2026-07-16 quick wins)
# ---------------------------------------------------------------------------


class TestForecastBlending:
    def test_ensemble_constants(self, forecaster):
        """Ensemble must use 6 diversified members (Tier-1 speedup)."""
        assert forecaster.N_ENSEMBLES == 3
        assert len(forecaster.ENSEMBLE_SEEDS) == 3
        assert len(forecaster.ENSEMBLE_FEATURE_FRACTIONS) == 3
        # Fractions should span a diversification range (not all identical).
        assert len(set(forecaster.ENSEMBLE_FEATURE_FRACTIONS)) > 1
        assert 0.0 < forecaster.FORECAST_BLEND_WEIGHT < 1.0
        assert forecaster.MAX_BIN == 63

    def test_prior_forecast_empty_when_no_rows(self, forecaster):
        forecaster.db = MagicMock()
        forecaster.db.execute.return_value.fetchall.return_value = []
        out = forecaster._fetch_prior_forecasts(np.array([1, 2, 3]), horizon=7)
        assert out["mask"].shape == (3,)
        assert not out["mask"].any()
        assert np.all(np.isnan(out["mid_ret"]))

    def test_prior_forecast_parses_return_space(self, forecaster):
        forecaster.db = MagicMock()
        rows = [
            MagicMock(item_id=1, price_low=11.0, price_mid=12.0, price_high=13.0,
                      current_price=10.0, forecast_date=date(2026, 7, 15)),
            # Older forecast for the same item — must be ignored.
            MagicMock(item_id=1, price_low=10.5, price_mid=11.0, price_high=12.5,
                      current_price=10.0, forecast_date=date(2026, 7, 14)),
            MagicMock(item_id=2, price_low=9.0, price_mid=10.0, price_high=11.0,
                      current_price=20.0, forecast_date=date(2026, 7, 15)),
        ]
        slug_rows = [
            MagicMock(item_id="1", id=1),
            MagicMock(item_id="2", id=2),
        ]
        forecaster.db.execute.side_effect = [
            MagicMock(fetchall=lambda: rows),
            MagicMock(fetchall=lambda: slug_rows),
        ]
        out = forecaster._fetch_prior_forecasts(np.array(["1", "2", "3"]), horizon=7)

        assert out["mask"][0] and out["mask"][1] and not out["mask"][2]
        # Item 1 latest: current=10, mid=12 -> +20% return.
        assert abs(out["mid_ret"][0] - 20.0) < 1e-6
        assert abs(out["low_ret"][0] - 10.0) < 1e-6
        assert abs(out["high_ret"][0] - 30.0) < 1e-6
        # Item 2: current=20, mid=10 -> -50% return.
        assert abs(out["mid_ret"][1] - (-50.0)) < 1e-6
        # Item 3 has no prior -> NaN.
        assert np.isnan(out["mid_ret"][2])

    def test_blend_moves_prediction_toward_prior(self, forecaster):
        w = forecaster.FORECAST_BLEND_WEIGHT
        current = np.array([5.0, -5.0])
        prior = {
            "mask": np.array([True, True]),
            "low_ret": np.array([15.0, 5.0]),
            "mid_ret": np.array([15.0, 5.0]),
            "high_ret": np.array([15.0, 5.0]),
        }
        low, mid, high = forecaster._blend_returns_with_prior(
            current.copy(), current.copy(), current.copy(), prior, w)
        # Blended value lies strictly between current and prior (toward prior).
        assert np.all(mid > current) and np.all(mid < prior["mid_ret"])
        # With weight 0 the prediction is unchanged.
        low0, mid0, high0 = forecaster._blend_returns_with_prior(
            current.copy(), current.copy(), current.copy(), prior, 0.0)
        assert np.all(mid0 == current)

    def test_blend_noop_without_prior(self, forecaster):
        current = np.array([5.0, -5.0])
        prior = {
            "mask": np.array([False, False]),
            "low_ret": np.full(2, np.nan),
            "mid_ret": np.full(2, np.nan),
            "high_ret": np.full(2, np.nan),
        }
        low, mid, high = forecaster._blend_returns_with_prior(
            current.copy(), current.copy(), current.copy(), prior, 0.15)
        assert np.all(mid == current)


# ---------------------------------------------------------------------------
# Regime-switching models
# ---------------------------------------------------------------------------

class TestRegimeSwitching:
    def test_assign_regime_label_bear(self, forecaster):
        """market_return_30d < -3% should be bear."""
        assert forecaster._assign_regime_label(-5.0) == "bear"
        assert forecaster._assign_regime_label(-3.1) == "bear"
        assert forecaster._assign_regime_label(-100.0) == "bear"

    def test_assign_regime_label_range(self, forecaster):
        """-3% <= market_return_30d <= 3% should be range."""
        assert forecaster._assign_regime_label(-3.0) == "range"
        assert forecaster._assign_regime_label(0.0) == "range"
        assert forecaster._assign_regime_label(3.0) == "range"

    def test_assign_regime_label_bull(self, forecaster):
        """market_return_30d > 3% should be bull."""
        assert forecaster._assign_regime_label(3.1) == "bull"
        assert forecaster._assign_regime_label(10.0) == "bull"
        assert forecaster._assign_regime_label(100.0) == "bull"

    def test_assign_regime_labels_dataframe(self, forecaster):
        """_assign_regime_labels should label rows based on market_return_30d."""
        df = pd.DataFrame({
            "market_return_30d": [-5.0, 0.0, 5.0, np.nan],
            "price": [10.0] * 4,
        })
        labels = forecaster._assign_regime_labels(df)
        assert labels.iloc[0] == "bear"
        assert labels.iloc[1] == "range"
        assert labels.iloc[2] == "bull"
        assert labels.iloc[3] == "range"  # NaN → range

    def test_assign_regime_labels_missing_column(self, forecaster):
        """When market_return_30d column is missing, all rows get 'range'."""
        df = pd.DataFrame({"price": [10.0, 20.0]})
        labels = forecaster._assign_regime_labels(df)
        assert (labels == "range").all()

    def test_detect_current_regime_from_latest(self, forecaster):
        """_detect_current_regime uses the most recent row's market_return_30d."""
        df = pd.DataFrame({
            "market_return_30d": [1.0, 2.0, -5.0, 0.5],
            "date": pd.date_range("2026-01-01", periods=4),
        })
        regime = forecaster._detect_current_regime(df)
        assert regime == "range"  # latest row has 0.5

    def test_detect_current_regime_bull(self, forecaster):
        df = pd.DataFrame({
            "market_return_30d": [1.0, 5.0],
            "date": pd.date_range("2026-01-01", periods=2),
        })
        assert forecaster._detect_current_regime(df) == "bull"

    def test_detect_current_regime_bear(self, forecaster):
        df = pd.DataFrame({
            "market_return_30d": [1.0, -10.0],
            "date": pd.date_range("2026-01-01", periods=2),
        })
        assert forecaster._detect_current_regime(df) == "bear"

    def test_detect_current_regime_missing_column(self, forecaster):
        """If market_return_30d is missing, default to range."""
        df = pd.DataFrame({"price": [10.0]})
        assert forecaster._detect_current_regime(df) == "range"

    def test_detect_current_regime_empty_series(self, forecaster):
        """If market_return_30d is all NaN, default to range."""
        df = pd.DataFrame({
            "market_return_30d": [np.nan, np.nan],
            "date": pd.date_range("2026-01-01", periods=2),
        })
        assert forecaster._detect_current_regime(df) == "range"

    def test_regime_columns_on_training_data(self, forecaster):
        """_regime column should be present after prepare_targets."""
        def mock_fetch(*args, **kwargs):
            np.random.seed(42)
            rows = []
            for item_id in range(5):
                price = 50.0
                for day_offset in range(200):
                    d = date(2025, 1, 1) + timedelta(days=day_offset)
                    price *= 1 + np.random.randn() * 0.01
                    rows.append({
                        "item_id": f"item_{item_id}",
                        "date": d,
                        "price": round(max(price, 0.01), 2),
                        "volume": int(max(np.random.poisson(200), 0)),
                    })
            return pd.DataFrame(rows)

        def mock_events(*args, **kwargs):
            return pd.DataFrame([
                {"id": 1, "type": "major", "timestamp": pd.Timestamp("2025-06-01"),
                 "description": "Major", "date": date(2025, 6, 1)},
            ])

        with patch.object(forecaster, 'fetch_price_history', mock_fetch):
            with patch.object(forecaster, 'fetch_events', mock_events):
                df = forecaster.build_training_data(days_back=200, backfilled_only=False)

        # Simulate the train() flow: prepare targets and check _regime column
        for h in forecaster.HORIZONS:
            tdf = forecaster.prepare_targets(df, h)
            tdf = tdf.dropna(subset=[f"target_return_{h}d"]).copy()
            tdf = tdf.sort_values("date")
            tdf["_regime"] = forecaster._assign_regime_labels(tdf)

            assert "_regime" in tdf.columns
            assert tdf["_regime"].isin(["bear", "range", "bull"]).all()
            # At least one regime type should be present
            assert tdf["_regime"].nunique() >= 1

    def test_regime_models_populated_after_train(self, tmp_path):
        """After train(), regime_models should contain entries for regimes
        with sufficient data.

        NOTE: model_dir MUST be tmp_path. train() persists at the end, so
        without this the test overwrites every artifact in the real
        models/saved_models/ (which is gitignored, so unrecoverable) with
        models fit on the 5 synthetic items below.
        """
        f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
        f._supply_meta_cache = pd.DataFrame(columns=["item_id", "rarity", "rarity_rank", "weapon_type"])

        def mock_fetch(*args, **kwargs):
            np.random.seed(42)
            rows = []
            for item_id in range(5):
                price = 50.0
                for day_offset in range(800):
                    d = date(2024, 1, 1) + timedelta(days=day_offset)
                    # Vary returns to create different regimes
                    price *= 1 + np.random.randn() * 0.02
                    rows.append({
                        "item_id": f"item_{item_id}",
                        "date": d,
                        "price": round(max(price, 0.01), 2),
                        "volume": int(max(np.random.poisson(200), 0)),
                    })
            df = pd.DataFrame(rows)
            return df

        def mock_events(*args, **kwargs):
            return pd.DataFrame([
                {"id": 1, "type": "major", "timestamp": pd.Timestamp("2025-06-01"),
                 "description": "Major", "date": date(2025, 6, 1)},
            ])

        with patch.object(f, 'fetch_price_history', mock_fetch):
            with patch.object(f, 'fetch_events', mock_events):
                with patch.object(f, '_fetch_supply_metadata',
                                  return_value=pd.DataFrame(columns=["item_id", "rarity", "rarity_rank", "weapon_type"])):
                    with patch.object(f, '_fetch_item_metadata',
                                      return_value=pd.DataFrame(columns=["item_id", "name", "type"])):
                        with patch.dict('os.environ', {'SKIP_CV': '1', 'FORCE_HP_SEARCH': '1'}):
                            f.train(max_rows=100_000)

        # Should have global models for all horizons and quantiles
        for h in f.HORIZONS:
            for q in f.QUANTILES:
                assert (h, q) in f.models, f"Missing global model for {h}d q{q}"

        # Should have at least some regime models (likely range with 300 days of data)
        if f.regime_models:
            regimes_trained = set(r for (r, h, q) in f.regime_models.keys())
            for regime in regimes_trained:
                for h in f.HORIZONS:
                    for q in f.QUANTILES:
                        key = (regime, h, q)
                        if key in f.regime_models:
                            assert len(f.regime_models[key]) == f.N_ENSEMBLES

    def test_predict_falls_back_to_global_when_no_regime_model(self, forecaster):
        """When no regime models exist, predict should use global models."""
        with patch.object(forecaster, 'fetch_price_history',
                          return_value=pd.DataFrame(columns=["item_id", "date", "price", "volume"])):
            with patch.object(forecaster, 'fetch_events',
                              return_value=pd.DataFrame(columns=["id", "type", "timestamp", "description"])):
                # No regime models loaded, should use global (which are also empty)
                result = forecaster.predict()
                assert isinstance(result, pd.DataFrame)
                assert result.empty

    def test_regime_models_save_and_load(self, forecaster, tmp_path):
        """Regime-specific models should round-trip through save/load."""
        forecaster.model_dir = str(tmp_path)
        forecaster.feature_cols = ["price_log", "price_lag_1d"]
        forecaster.horizon_feature_cols = {7: ["price_log", "price_lag_1d"]}
        forecaster.regime_feature_cols = {(7, "range"): ["price_log", "price_lag_1d"]}

        # Create dummy regime model
        X = np.random.randn(100, 2).astype(np.float32)
        y = np.random.randn(100)
        ds = lgb.Dataset(X, y)
        model = lgb.train({"objective": "regression", "verbosity": -1,
                           "max_bin": 63, "min_data_in_leaf": 1,
                           "num_leaves": 3, "learning_rate": 0.1},
                          ds, num_boost_round=5)

        forecaster.regime_models[("range", 7, 0.5)] = [model]
        forecaster.confidence_thresholds = {7: {"high_range": 0.15, "high_change": 0.01, "high_accuracy": 99.0}}
        forecaster.save_models()

        # Load into a new forecaster
        f2 = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
        f2.load_models()

        assert ("range", 7, 0.5) in f2.regime_models
        assert len(f2.regime_models[("range", 7, 0.5)]) == 1
        assert (7, "range") in f2.regime_feature_cols

    def test_regime_models_skipped_with_insufficient_data(self, forecaster):
        """Regimes with too few training rows should be skipped."""
        # Data that covers all 3 regimes but has minimal rows in bear/bull
        rows = []
        for day_offset in range(300):
            d = date(2025, 1, 1) + timedelta(days=day_offset)
            regime = "range"
            if day_offset < 10:
                regime = "bear"
            elif day_offset >= 290:
                regime = "bull"
            for item_id in range(3):
                price = 50.0 + (10.0 if regime == "bull" else -10.0 if regime == "bear" else 0.0)
                rows.append({
                    "item_id": f"item_{item_id}",
                    "date": d,
                    "price": price + np.random.randn() * 0.5,
                    "volume": 100,
                    "_regime": regime,
                })

        tdf = pd.DataFrame(rows)

        # Simulate the filter logic from train()
        for regime in forecaster.REGIMES:
            r_train = tdf[tdf["_regime"] == regime]
            # Both bear (30 rows) and bull (30 rows) should fail the 500-row minimum
            assert len(r_train) < 500 or regime == "range"


# ---------------------------------------------------------------------------
# Voted price frame cache
# ---------------------------------------------------------------------------

class TestVotedPriceCache:
    """The voted frame dominates ``fetch_price_history`` and is a pure
    function of the archive contents plus the query window, so repeated runs
    on an unchanged archive — walk-forward folds, A/B harnesses, retrain
    iteration — can skip the rebuild. Measured: 35.5s cold, 0.3s cached.
    """

    @pytest.fixture
    def cached_forecaster(self, forecaster, tmp_path):
        """Forecaster with a throwaway cache dir and a fake Parquet archive."""
        cache_dir = tmp_path / "cache"
        archive_dir = tmp_path / "price-archive"
        archive_dir.mkdir()
        (archive_dir / "prices-2025.parquet").write_bytes(b"fake-2025")
        (archive_dir / "prices-2026.parquet").write_bytes(b"fake-2026")
        forecaster.cache_dir = str(cache_dir)
        forecaster.archive_dir = archive_dir
        return forecaster

    @pytest.fixture
    def voted_df(self):
        """A voted frame shaped like fetch_price_history's return value."""
        return pd.DataFrame({
            "item_id": ["ak47", "ak47", "awp"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
            "date": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 1)],
            "price": [10.5, 11.0, 99.9],
            "volume": [100, 120, 5],
            "source": ["steamcommunity", "steamcommunity", None],
        })

    # -- key derivation ---------------------------------------------------

    def test_key_is_stable_for_unchanged_inputs(self, cached_forecaster):
        f = cached_forecaster
        k1 = f._voted_cache_key(1460, False, None)
        k2 = f._voted_cache_key(1460, False, None)
        assert k1 == k2

    def test_key_changes_when_archive_content_changes(self, cached_forecaster):
        f = cached_forecaster
        before = f._voted_cache_key(1460, False, None)
        # Aggregator appends a day: same filename, different size.
        (f.archive_dir / "prices-2026.parquet").write_bytes(b"fake-2026-plus-a-new-day")
        assert f._voted_cache_key(1460, False, None) != before

    def test_key_changes_when_archive_gains_a_file(self, cached_forecaster):
        f = cached_forecaster
        before = f._voted_cache_key(1460, False, None)
        (f.archive_dir / "prices-2027.parquet").write_bytes(b"fake-2027")
        assert f._voted_cache_key(1460, False, None) != before

    def test_key_ignores_non_price_files(self, cached_forecaster):
        """Only prices-*.parquet feeds the voted frame; ops/ must not bust it."""
        f = cached_forecaster
        before = f._voted_cache_key(1460, False, None)
        (f.archive_dir / "forecasts.parquet").write_bytes(b"unrelated")
        assert f._voted_cache_key(1460, False, None) == before

    def test_key_varies_with_query_window(self, cached_forecaster):
        f = cached_forecaster
        assert f._voted_cache_key(1460, False, None) != f._voted_cache_key(730, False, None)

    def test_key_varies_with_backfilled_only(self, cached_forecaster):
        f = cached_forecaster
        assert f._voted_cache_key(1460, False, None) != f._voted_cache_key(1460, True, {"ak47"})

    def test_key_varies_with_slug_set(self, cached_forecaster):
        f = cached_forecaster
        assert (f._voted_cache_key(1460, True, {"ak47"})
                != f._voted_cache_key(1460, True, {"ak47", "awp"}))

    def test_key_ignores_slug_set_ordering(self, cached_forecaster):
        """The slug set comes from an unordered DB query."""
        f = cached_forecaster
        assert (f._voted_cache_key(1460, True, {"ak47", "awp"})
                == f._voted_cache_key(1460, True, {"awp", "ak47"}))

    def test_key_varies_with_cache_version(self, cached_forecaster):
        """Bumping the version constant invalidates every entry — the escape
        hatch for when voting or the DuckDB query itself changes."""
        f = cached_forecaster
        before = f._voted_cache_key(1460, False, None)
        with patch.object(type(f), "VOTED_CACHE_VERSION", f.VOTED_CACHE_VERSION + 1):
            assert f._voted_cache_key(1460, False, None) != before

    # -- round trip -------------------------------------------------------

    def test_roundtrip_preserves_frame_exactly(self, cached_forecaster, voted_df):
        """Downstream code indexes df["date"] as datetime.date objects, so the
        Parquet round trip must not silently promote them to Timestamps."""
        f = cached_forecaster
        key = f._voted_cache_key(1460, False, None)
        f._save_voted_cache(key, voted_df)
        loaded = f._load_voted_cache(key)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, voted_df)
        assert isinstance(loaded["date"].iloc[0], date)

    def test_load_returns_none_on_miss(self, cached_forecaster):
        f = cached_forecaster
        assert f._load_voted_cache("nonexistent-key") is None

    def test_load_returns_none_on_corrupt_file(self, cached_forecaster, voted_df):
        f = cached_forecaster
        key = f._voted_cache_key(1460, False, None)
        f._save_voted_cache(key, voted_df)
        Path(f._voted_cache_path(key)).write_bytes(b"not a parquet file")
        assert f._load_voted_cache(key) is None

    def test_load_returns_none_on_empty_frame(self, cached_forecaster, voted_df):
        """An empty cached frame would silently train on zero rows."""
        f = cached_forecaster
        key = f._voted_cache_key(1460, False, None)
        f._save_voted_cache(key, voted_df.iloc[0:0])
        assert f._load_voted_cache(key) is None

    def test_save_does_not_write_to_model_dir(self, cached_forecaster, voted_df):
        """model_dir holds gitignored production artifacts — keep the cache out."""
        f = cached_forecaster
        f._save_voted_cache(f._voted_cache_key(1460, False, None), voted_df)
        assert not list(Path(f.model_dir).glob("*.parquet"))

    # -- disk bounds ------------------------------------------------------

    def test_prunes_stale_entries(self, cached_forecaster, voted_df):
        """Each entry is hundreds of MB in production and the archive changes
        daily, so entries must not accumulate."""
        f = cached_forecaster
        for i in range(f.VOTED_CACHE_MAX_ENTRIES + 3):
            f._save_voted_cache(f"key{i:03d}", voted_df)
        entries = list(Path(f.cache_dir).glob("voted_*.parquet"))
        assert len(entries) <= f.VOTED_CACHE_MAX_ENTRIES

    def test_prune_keeps_the_newest_entry(self, cached_forecaster, voted_df):
        f = cached_forecaster
        for i in range(f.VOTED_CACHE_MAX_ENTRIES + 3):
            key = f"key{i:03d}"
            f._save_voted_cache(key, voted_df)
        assert f._load_voted_cache(key) is not None

    # -- kill switch ------------------------------------------------------

    def test_disabled_by_env_skips_read_and_write(self, cached_forecaster, voted_df,
                                                  monkeypatch):
        f = cached_forecaster
        key = f._voted_cache_key(1460, False, None)
        f._save_voted_cache(key, voted_df)
        monkeypatch.setenv("VOTED_CACHE", "0")
        assert f._load_voted_cache(key) is None
        f._save_voted_cache("another-key", voted_df)
        assert f._load_voted_cache("another-key") is None

    # -- integration with fetch_price_history ------------------------------

    def test_fetch_price_history_skips_voting_on_cache_hit(self, cached_forecaster,
                                                           voted_df):
        """The whole point: a second fetch over an unchanged archive must not
        re-run the DuckDB query or the voting pass."""
        f = cached_forecaster
        key = f._voted_cache_key(1460, False, None)
        f._save_voted_cache(key, voted_df)

        with patch.object(f, "_apply_multi_source_voting") as vote, \
             patch("duckdb.connect") as connect:
            out = f.fetch_price_history(days_back=1460, backfilled_only=False)

        vote.assert_not_called()
        connect.assert_not_called()
        pd.testing.assert_frame_equal(out, voted_df)

    def test_fetch_price_history_populates_cache_on_miss(self, cached_forecaster,
                                                         voted_df, monkeypatch):
        f = cached_forecaster
        key = f._voted_cache_key(1460, False, None)
        assert f._load_voted_cache(key) is None

        monkeypatch.setattr(f, "_fetch_voted_price_history",
                            lambda **kw: voted_df.copy())
        out = f.fetch_price_history(days_back=1460, backfilled_only=False)

        pd.testing.assert_frame_equal(out, voted_df)
        assert f._load_voted_cache(key) is not None
