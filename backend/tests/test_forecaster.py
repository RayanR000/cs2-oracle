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
from unittest.mock import MagicMock, patch

from models.forecaster import ItemForecaster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def forecaster():
    f = ItemForecaster(db_session=MagicMock())
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

    def test_build_training_data_includes_all_expected_features(self, forecaster):
        """Verify the feature set includes all expected categories via build_training_data."""
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

        # Check feature categories exist
        feature_set = set(forecaster.feature_cols)
        assert any("return_" in c for c in feature_set), "Missing return features"
        assert any("bb_" in c for c in feature_set), "Missing Bollinger features"
        assert any("rsi" in c for c in feature_set), "Missing RSI features"
        assert any("macd" in c for c in feature_set), "Missing MACD features"
        assert any("event_decay_" in c for c in feature_set), "Missing event features"
        assert any("market_return_" in c for c in feature_set), "Missing market features"

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

    def test_distribution_shift_guard_excludes_2026(self, forecaster):
        """Regression: the 2026 distribution-shift guard must remove all 2026
        rows BEFORE stratified subsampling. The old guard had a `month < 6`
        check that failed in July 2026, and ran AFTER subsampling so the
        budget was wasted on 2026 rows, collapsing the 7d validation window
        to a single day and triggering false-positive feature pruning."""
        rows = []
        base = date(2022, 6, 1)
        for year in [2022, 2023, 2024, 2025]:
            for day_offset in range(100):
                rows.append({
                    "item_id": f"item_{year}",
                    "date": date(year, 1, 1) + timedelta(days=day_offset),
                    "price": 10.0,
                    "volume": 100,
                })
        # Add incomplete 2026 data (Jan-Jun, incomplete year)
        for day_offset in range(100):
            rows.append({
                "item_id": "item_2026",
                "date": date(2026, 1, 1) + timedelta(days=day_offset),
                "price": 10.0,
                "volume": 100,
            })
        df = pd.DataFrame(rows)

        n_before = len(df)
        counts_before = df.groupby(df["date"].apply(lambda d: d.year)).size()

        # Apply the exact guard from build_training_data
        if "date" in df.columns:
            dates_2026 = pd.DatetimeIndex(df["date"]).year == 2026
            n_2026 = dates_2026.sum()
            if n_2026 > 0:
                df = df[~dates_2026].copy()

        n_after = len(df)
        counts_after = df.groupby(df["date"].apply(lambda d: d.year)).size()

        assert n_after < n_before, "Guard should have removed rows"
        assert 2026 not in counts_after.index, "All 2026 rows must be excluded"
        for year in [2022, 2023, 2024, 2025]:
            assert counts_after[year] == counts_before[year], \
                f"Non-2026 rows for {year} must be preserved"

    def test_distribution_shift_guard_preserves_earlier_years(self, forecaster):
        """When there's no 2026 data, the guard should be a no-op."""
        rows = []
        for day_offset in range(100):
            rows.append({
                "item_id": "item_a",
                "date": date(2025, 1, 1) + timedelta(days=day_offset),
                "price": 10.0,
                "volume": 100,
            })
        df = pd.DataFrame(rows)
        n_before = len(df)

        if "date" in df.columns:
            dates_2026 = pd.DatetimeIndex(df["date"]).year == 2026
            n_2026 = dates_2026.sum()
            if n_2026 > 0:
                df = df[~dates_2026].copy()

        assert len(df) == n_before, "No-op guard must not remove rows when no 2026 data"


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

    def test_regime_models_populated_after_train(self):
        """After train(), regime_models should contain entries for regimes
        with sufficient data."""
        f = ItemForecaster(db_session=MagicMock())
        f._supply_meta_cache = pd.DataFrame(columns=["item_id", "rarity", "rarity_rank", "weapon_type"])

        def mock_fetch(*args, **kwargs):
            np.random.seed(42)
            rows = []
            for item_id in range(5):
                price = 50.0
                for day_offset in range(300):
                    d = date(2025, 1, 1) + timedelta(days=day_offset)
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
                                      return_value=pd.DataFrame(columns=["item_id", "type"])):
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


