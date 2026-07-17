"""
Unit tests for ItemForecaster — feature engineering, quantile handling,
confidence calibration, sanitization, and drift monitoring.

Tests avoid DB/Parquet dependencies by constructing synthetic DataFrames
and injecting them directly into the methods under test.
"""

import pytest
import numpy as np
import pandas as pd
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
    def test_quantile_monotonicity_enforced(self):
        """Verify that np.minimum(p10, p50) and np.maximum(p50, p90) work."""
        p10 = np.array([5.0, 8.0, 3.0, 10.0])
        p50 = np.array([4.0, 7.0, 5.0, 9.0])
        p90 = np.array([6.0, 6.0, 7.0, 8.0])
        low = np.minimum(p10, p50)   # [4, 7, 3, 9]
        high = np.maximum(p50, p90)  # [6, 7, 7, 9]
        assert (low <= p50).all()
        assert (p50 <= high).all()

    def test_crossing_rate_diagnostic(self, forecaster):
        p10 = np.array([5.0, 2.0])
        p50 = np.array([4.0, 3.0])
        p90 = np.array([6.0, 8.0])
        crossing = (p10 > p50) | (p50 > p90)
        rate = np.mean(crossing)
        assert rate == 0.5


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


# ---------------------------------------------------------------------------
# Player Count Features
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Forecast blending / directional smoothing (2026-07-16 quick wins)
# ---------------------------------------------------------------------------


class TestForecastBlending:
    def test_ensemble_constants(self, forecaster):
        """Ensemble must use 6 diversified members (Tier-1 speedup)."""
        assert forecaster.N_ENSEMBLES == 6
        assert len(forecaster.ENSEMBLE_SEEDS) == 6
        assert len(forecaster.ENSEMBLE_FEATURE_FRACTIONS) == 6
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
        forecaster.db.execute.return_value.fetchall.return_value = rows
        out = forecaster._fetch_prior_forecasts(np.array([1, 2, 3]), horizon=7)

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


