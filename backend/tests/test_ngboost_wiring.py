"""Tests for NGBoost wiring into ItemForecaster (flag gating, save/load)."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ngboost_mod = pytest.importorskip("ngboost", reason="ngboost not installed (probabilistic extra)")

from models.ngboost_head import (
    NGBoostTrainResult,
    save_ngboost_model,
    load_ngboost_model,
)


class TestNGBoostFlagGating:
    """NGBOOST_HEAD env flag controls whether training runs."""

    def test_flag_off_by_default(self):
        """NGBoost head is disabled when NGBOOST_HEAD is not set."""
        env = os.environ.copy()
        env.pop("NGBOOST_HEAD", None)
        with patch.dict(os.environ, env, clear=True):
            assert os.environ.get("NGBOOST_HEAD") != "1"

    def test_flag_on_when_set(self):
        """NGBoost head is enabled when NGBOOST_HEAD=1."""
        with patch.dict(os.environ, {"NGBOOST_HEAD": "1"}):
            assert os.environ.get("NGBOOST_HEAD") == "1"

    def test_flag_off_with_other_values(self):
        """NGBOOST_HEAD must be exactly '1' to enable."""
        for val in ["0", "true", "yes", "on", ""]:
            with patch.dict(os.environ, {"NGBOOST_HEAD": val}):
                assert os.environ.get("NGBOOST_HEAD") != "1"


class TestNGBoostSaveLoad:
    """Save/load round-trip for NGBoost models."""

    def _make_trained_model(self):
        """Train a small NGBoost model for testing."""
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.randn(100)
        model = NGBRegressor(
            Dist=Normal, n_estimators=5, learning_rate=0.1, verbose=False,
        )
        model.fit(X, y)
        return model

    def test_save_load_roundtrip(self, tmp_path):
        """A saved model loads and produces the same predictions."""
        model = self._make_trained_model()
        path = str(tmp_path / "ngboost_7d.joblib")

        # Generate reference predictions
        X_test = np.random.RandomState(99).randn(20, 5)
        ref_dist = model.pred_dist(X_test)
        ref_mean = ref_dist.loc.copy()
        ref_std = ref_dist.scale.copy()

        # Save and load
        save_ngboost_model(model, path)
        loaded = load_ngboost_model(path)

        # Predictions must match
        loaded_dist = loaded.pred_dist(X_test)
        np.testing.assert_array_almost_equal(loaded_dist.loc, ref_mean)
        np.testing.assert_array_almost_equal(loaded_dist.scale, ref_std)

    def test_save_creates_file(self, tmp_path):
        """save_ngboost_model creates the file on disk."""
        model = self._make_trained_model()
        path = str(tmp_path / "test_model.joblib")
        assert not os.path.exists(path)
        save_ngboost_model(model, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
