"""Tests for MAPIE Conformalized Quantile Regression (CQR).

Uses synthetic data only. mapie is an optional dependency — tests skip
gracefully when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

mapie = pytest.importorskip("mapie", reason="mapie not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng():
    return np.random.RandomState(42)


@pytest.fixture()
def synthetic_calibration(rng):
    """Synthetic calibration data mimicking the conformal module's inputs.

    Returns (X_cal, residuals_pct, sigma, y_cal) where:
    - X_cal: 2-column feature matrix (price, volatility)
    - residuals_pct: y - y_hat in percentage-return space
    - sigma: per-item volatility scale (positive)
    - y_cal: actual target values
    """
    n = 500
    price = rng.uniform(1, 200, size=n)
    volatility = rng.uniform(0.01, 0.30, size=n)
    X_cal = np.column_stack([price, volatility])

    # Make residuals heteroscedastic: high-volatility items have wider errors
    residuals_pct = rng.normal(0, 1 + 5 * volatility)
    sigma = volatility + 0.01  # ensure positive
    y_cal = residuals_pct  # in return space, target is the residual itself

    return X_cal, residuals_pct, sigma, y_cal


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    """CQR intervals should cover approximately the target 80%."""

    def test_coverage_approximately_at_target(self, synthetic_calibration, rng):
        from models.conformal_cqr import calibrate_cqr

        X_cal, residuals_pct, sigma, y_cal = synthetic_calibration

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        assert "coverage" in result
        # CQR has a theoretical guarantee of >= 1 - alpha.
        # On finite data we allow some slack.
        assert result["coverage"] >= 0.70, (
            f"Coverage {result['coverage']:.3f} is far below 80% target"
        )
        assert result["coverage"] <= 0.95, (
            f"Coverage {result['coverage']:.3f} is suspiciously high"
        )

    def test_coverage_with_different_alpha(self, synthetic_calibration):
        from models.conformal_cqr import calibrate_cqr

        X_cal, residuals_pct, sigma, y_cal = synthetic_calibration

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.10,
        )

        # 90% target: should be higher than the 80% case
        assert result["coverage"] >= 0.80


# ---------------------------------------------------------------------------
# Adaptive intervals
# ---------------------------------------------------------------------------

class TestAdaptiveIntervals:
    """CQR intervals should vary with prediction difficulty."""

    def test_interval_widths_vary(self, synthetic_calibration):
        from models.conformal_cqr import calibrate_cqr

        X_cal, residuals_pct, sigma, y_cal = synthetic_calibration

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        widths = result["upper"] - result["lower"]
        assert widths.std() > 0, "Interval widths are constant — CQR should be adaptive"

    def test_wider_intervals_for_volatile_items(self, rng):
        """Items with high sigma should get wider CQR intervals on average."""
        from models.conformal_cqr import calibrate_cqr

        n = 600
        price = rng.uniform(1, 200, size=n)
        volatility = np.concatenate([
            rng.uniform(0.01, 0.05, size=n // 2),  # calm
            rng.uniform(0.20, 0.50, size=n // 2),  # volatile
        ])
        X_cal = np.column_stack([price, volatility])
        residuals_pct = rng.normal(0, 1 + 10 * volatility)
        sigma = volatility + 0.01
        y_cal = residuals_pct

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        widths = result["upper"] - result["lower"]
        calm_widths = widths[:n // 2]
        volatile_widths = widths[n // 2:]

        assert volatile_widths.mean() > calm_widths.mean(), (
            "Volatile items should have wider CQR intervals than calm items"
        )


# ---------------------------------------------------------------------------
# Valid bounds
# ---------------------------------------------------------------------------

class TestValidBounds:
    """CQR must produce valid bounds: low < mid < high."""

    def test_lower_below_upper(self, synthetic_calibration):
        from models.conformal_cqr import calibrate_cqr

        X_cal, residuals_pct, sigma, y_cal = synthetic_calibration

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        assert np.all(result["lower"] <= result["upper"]), (
            "Some lower bounds exceed upper bounds"
        )

    def test_result_keys(self, synthetic_calibration):
        from models.conformal_cqr import calibrate_cqr

        X_cal, residuals_pct, sigma, y_cal = synthetic_calibration

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        required_keys = {"lower", "upper", "coverage", "mean_width"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_all_values_finite(self, synthetic_calibration):
        from models.conformal_cqr import calibrate_cqr

        X_cal, residuals_pct, sigma, y_cal = synthetic_calibration

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        assert np.all(np.isfinite(result["lower"]))
        assert np.all(np.isfinite(result["upper"]))
        assert np.isfinite(result["coverage"])
        assert np.isfinite(result["mean_width"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: constant predictions, extreme values, small sets."""

    def test_constant_predictions(self, rng):
        """When all residuals are identical, CQR should still produce valid bounds."""
        from models.conformal_cqr import calibrate_cqr

        n = 200
        X_cal = rng.uniform(0, 1, size=(n, 2))
        residuals_pct = np.zeros(n)
        sigma = np.ones(n) * 0.05
        y_cal = np.zeros(n)

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        assert np.all(result["lower"] <= result["upper"])
        assert np.all(np.isfinite(result["lower"]))

    def test_extreme_values(self, rng):
        """Large residuals should not crash the calibration."""
        from models.conformal_cqr import calibrate_cqr

        n = 300
        X_cal = rng.uniform(0, 1, size=(n, 2))
        residuals_pct = rng.normal(0, 100, size=n)  # very large residuals
        sigma = rng.uniform(0.01, 0.5, size=n)
        y_cal = residuals_pct

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        assert np.all(np.isfinite(result["lower"]))
        assert np.all(np.isfinite(result["upper"]))

    def test_small_calibration_set(self, rng):
        """With few calibration points, should still work (or return a fallback)."""
        from models.conformal_cqr import calibrate_cqr

        n = 30  # small
        X_cal = rng.uniform(0, 1, size=(n, 2))
        residuals_pct = rng.normal(0, 1, size=n)
        sigma = rng.uniform(0.01, 0.1, size=n)
        y_cal = residuals_pct

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        assert "lower" in result
        assert "upper" in result
        assert np.all(result["lower"] <= result["upper"])

    def test_mean_width_positive(self, synthetic_calibration):
        from models.conformal_cqr import calibrate_cqr

        X_cal, residuals_pct, sigma, y_cal = synthetic_calibration

        result = calibrate_cqr(
            X_cal=X_cal,
            y_cal=y_cal,
            residuals_pct=residuals_pct,
            sigma=sigma,
            alpha=0.20,
        )

        assert result["mean_width"] > 0
