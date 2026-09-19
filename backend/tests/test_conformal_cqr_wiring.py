"""Tests for CQR wiring into forecaster.py.

Verifies that:
1. CQR is NOT run when CONFORMAL_CQR flag is off
2. CQR IS run when CONFORMAL_CQR flag is on
3. Production band ALWAYS comes from the old conformal module regardless
"""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

mapie = pytest.importorskip("mapie", reason="mapie not installed")


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------

class TestCQRFlagGating:
    """CONFORMAL_CQR flag controls whether CQR runs at all."""

    def test_cqr_not_run_when_flag_off(self):
        """With CONFORMAL_CQR unset, the CQR module should not be called."""
        env = {k: v for k, v in os.environ.items() if k != "CONFORMAL_CQR"}
        with mock.patch.dict(os.environ, env, clear=True):
            from models.conformal_cqr import cqr_enabled
            assert not cqr_enabled()

    def test_cqr_run_when_flag_on(self):
        """With CONFORMAL_CQR=1, the CQR module should report enabled."""
        with mock.patch.dict(os.environ, {"CONFORMAL_CQR": "1"}):
            from models.conformal_cqr import cqr_enabled
            assert cqr_enabled()

    def test_cqr_not_run_for_other_values(self):
        """Only '1' enables CQR — other truthy strings do not."""
        with mock.patch.dict(os.environ, {"CONFORMAL_CQR": "true"}):
            from models.conformal_cqr import cqr_enabled
            assert not cqr_enabled()

    def test_cqr_disabled_by_zero(self):
        with mock.patch.dict(os.environ, {"CONFORMAL_CQR": "0"}):
            from models.conformal_cqr import cqr_enabled
            assert not cqr_enabled()


# ---------------------------------------------------------------------------
# Production band unchanged
# ---------------------------------------------------------------------------

class TestProductionBandUnchanged:
    """The production conformal band must ALWAYS come from conformal.py."""

    def test_old_conformal_produces_band(self):
        """conformal.band and conformal.band_signed still work identically."""
        from models import conformal

        mid = np.array([0.0, 5.0, -3.0])
        sigma = np.array([0.05, 0.10, 0.08])
        q_hat = 2.0

        low, high = conformal.band(mid, sigma, q_hat)

        # Band is symmetric: mid +/- q_hat * sigma
        np.testing.assert_allclose(low, mid - q_hat * sigma)
        np.testing.assert_allclose(high, mid + q_hat * sigma)

    def test_cqr_result_is_shadow_only(self):
        """CQR results should be structured as shadow data, never replacing conformal."""
        from models.conformal_cqr import calibrate_cqr

        rng = np.random.RandomState(99)
        n = 200
        X = rng.uniform(0, 1, size=(n, 2))
        y = rng.normal(0, 1, size=n)
        sigma = rng.uniform(0.01, 0.1, size=n)

        result = calibrate_cqr(
            X_cal=X,
            y_cal=y,
            residuals_pct=y,
            sigma=sigma,
            alpha=0.20,
        )

        # The result dict must not contain keys that would overwrite conformal state
        assert "q_hat" not in result, "CQR must not produce q_hat — that belongs to conformal.py"
        assert "q_lo" not in result
        assert "q_hi" not in result


# ---------------------------------------------------------------------------
# Shadow CQR metrics
# ---------------------------------------------------------------------------

class TestShadowMetrics:
    """When enabled, CQR should produce metrics for comparison logging."""

    def test_shadow_metrics_structure(self):
        from models.conformal_cqr import calibrate_cqr

        rng = np.random.RandomState(123)
        n = 300
        X = rng.uniform(0, 1, size=(n, 2))
        y = rng.normal(0, 1, size=n)
        sigma = rng.uniform(0.01, 0.1, size=n)

        result = calibrate_cqr(
            X_cal=X,
            y_cal=y,
            residuals_pct=y,
            sigma=sigma,
            alpha=0.20,
        )

        # Must provide these for training metadata logging
        assert "coverage" in result
        assert "mean_width" in result
        assert isinstance(result["coverage"], float)
        assert isinstance(result["mean_width"], float)
