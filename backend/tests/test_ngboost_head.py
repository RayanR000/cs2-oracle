"""Tests for the NGBoost distributional head (pure module)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ngboost_mod = pytest.importorskip("ngboost", reason="ngboost not installed (probabilistic extra)")

from models.ngboost_head import (
    SUPPORTED_HORIZONS,
    NGBoostPrediction,
    predict_ngboost,
    train_ngboost,
)


def _synthetic_data(n: int = 500, n_features: int = 10, seed: int = 42):
    """Generate synthetic regression data matching the project's feature matrix format."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Simulate a noisy return target with some signal
    y = 0.3 * X["feat_0"] - 0.2 * X["feat_1"] + rng.randn(n) * 0.5
    return X, y.values


@pytest.fixture
def train_val_data():
    """Provide train/val split of synthetic data."""
    X, y = _synthetic_data(n=600, n_features=10)
    split = 450
    return X.iloc[:split], y[:split], X.iloc[split:], y[split:]


class TestTrainNGBoost:
    """Training produces valid results."""

    def test_train_h7(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        result = train_ngboost(X_train, y_train, X_val, y_val, horizon=7, n_estimators=20)
        assert result.horizon == 7
        assert result.n_train == 450
        assert result.n_val == 150
        assert np.isfinite(result.train_nll)
        assert np.isfinite(result.val_nll)

    def test_train_h14(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        result = train_ngboost(X_train, y_train, X_val, y_val, horizon=14, n_estimators=20)
        assert result.horizon == 14

    def test_unsupported_horizon_raises(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        for h in [3, 30, 1, 60]:
            with pytest.raises(ValueError, match="only supports horizons"):
                train_ngboost(X_train, y_train, X_val, y_val, horizon=h, n_estimators=5)

    def test_nan_targets_filtered(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        # Inject NaNs into training targets
        y_train_nan = y_train.copy()
        y_train_nan[0:10] = np.nan
        result = train_ngboost(X_train, y_train_nan, X_val, y_val, horizon=7, n_estimators=10)
        assert result.n_train == 440  # 450 - 10 NaNs
        assert result.n_val == 150

    def test_numpy_features_accepted(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        result = train_ngboost(
            X_train.values, y_train, X_val.values, y_val, horizon=7, n_estimators=10
        )
        assert result.n_train == 450


class TestPredictNGBoost:
    """Predictions have mean and std."""

    def test_predictions_shape(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        result = train_ngboost(X_train, y_train, X_val, y_val, horizon=7, n_estimators=20)
        pred = predict_ngboost(result.model, X_val, horizon=7)
        assert isinstance(pred, NGBoostPrediction)
        assert len(pred.mean) == 150
        assert len(pred.std) == 150

    def test_predictions_are_finite(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        result = train_ngboost(X_train, y_train, X_val, y_val, horizon=14, n_estimators=20)
        pred = predict_ngboost(result.model, X_val, horizon=14)
        assert np.all(np.isfinite(pred.mean))
        assert np.all(np.isfinite(pred.std))

    def test_std_is_positive(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        result = train_ngboost(X_train, y_train, X_val, y_val, horizon=7, n_estimators=20)
        pred = predict_ngboost(result.model, X_val, horizon=7)
        assert np.all(pred.std > 0)

    def test_unsupported_horizon_raises(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        result = train_ngboost(X_train, y_train, X_val, y_val, horizon=7, n_estimators=10)
        with pytest.raises(ValueError, match="only supports horizons"):
            predict_ngboost(result.model, X_val, horizon=3)


class TestSupportedHorizons:
    """Only h=7 and h=14 are supported."""

    def test_supported_horizons_are_7_and_14(self):
        assert SUPPORTED_HORIZONS == frozenset({7, 14})
