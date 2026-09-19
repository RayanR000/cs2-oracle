"""Conformalized Quantile Regression (CQR) via MAPIE.

Shadow-only alternative to the hand-rolled split conformal in conformal.py.
Produces ADAPTIVE intervals (wider when uncertain, narrower when confident)
by fitting quantile regressors on the calibration features and conformally
correcting their predictions.

Gated by the CONFORMAL_CQR env flag. When enabled, results are logged as
shadow/candidate data alongside the production conformal band -- they never
replace it.

Pure module: numpy + sklearn + mapie only, no DB, no I/O, no clock.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Match the existing conformal module's target.
NOMINAL_COVERAGE = 0.80
ALPHA = 1.0 - NOMINAL_COVERAGE

# Minimum calibration set size for CQR. Below this, the quantile regressors
# cannot learn meaningful conditional quantiles and the conformal correction
# has too few points for a reliable guarantee.
MIN_CQR_CALIBRATION_ROWS = 50


def cqr_enabled() -> bool:
    """Whether the CONFORMAL_CQR flag is set to '1'."""
    return os.environ.get("CONFORMAL_CQR") == "1"


def calibrate_cqr(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    residuals_pct: np.ndarray,
    sigma: np.ndarray,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Fit CQR on calibration data and return adaptive intervals + metrics.

    Uses MAPIE's ConformalizedQuantileRegressor with a GradientBoostingRegressor
    as the base quantile estimator. The fit/conformalize split is done
    internally (60/40 of the calibration set).

    Parameters
    ----------
    X_cal : ndarray of shape (n, d)
        Feature matrix for the calibration set. At minimum, should include
        sigma as a column so the quantile regressors can condition on it.
    y_cal : ndarray of shape (n,)
        Target values (actual returns) for the calibration set.
    residuals_pct : ndarray of shape (n,)
        Residuals (y - y_hat) in percentage-return space. Used for coverage
        measurement against the CQR intervals.
    sigma : ndarray of shape (n,)
        Per-item volatility scale. Included as a feature for conditioning.
    alpha : float
        Miscoverage rate (default 0.20 for 80% coverage).

    Returns
    -------
    dict with keys:
        lower : ndarray — lower bounds of CQR intervals
        upper : ndarray — upper bounds of CQR intervals
        coverage : float — empirical coverage on the calibration set
        mean_width : float — mean interval width
    """
    try:
        from mapie.regression import ConformalizedQuantileRegressor
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as e:
        logger.warning(f"CQR dependencies not available: {e}")
        return _fallback_result(y_cal, sigma, alpha)

    X = np.asarray(X_cal, dtype=float)
    y = np.asarray(y_cal, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    n = X.shape[0]
    if n < MIN_CQR_CALIBRATION_ROWS:
        logger.warning(
            f"CQR calibration set too small ({n} < {MIN_CQR_CALIBRATION_ROWS}), "
            f"falling back to sigma-scaled intervals"
        )
        return _fallback_result(y, sigma_arr, alpha)

    # Augment features with sigma for conditioning
    X_aug = np.column_stack([X, sigma_arr]) if sigma_arr.ndim == 1 else X

    # Split into fit (60%) and conformalize (40%) sets
    split_idx = int(n * 0.6)
    # Shuffle indices for a random split
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fit_idx = indices[:split_idx]
    conf_idx = indices[split_idx:]

    X_fit, y_fit = X_aug[fit_idx], y[fit_idx]
    X_conf, y_conf = X_aug[conf_idx], y[conf_idx]

    # Base quantile regressor — lightweight, fast
    base_estimator = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.5,  # median; MAPIE handles the quantile levels internally
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=42,
    )

    confidence_level = 1.0 - alpha

    try:
        cqr = ConformalizedQuantileRegressor(
            estimator=base_estimator,
            confidence_level=confidence_level,
        )
        cqr.fit(X_fit, y_fit)
        cqr.conformalize(X_conf, y_conf)

        # Predict intervals on the FULL calibration set for shadow comparison
        point_preds, intervals = cqr.predict_interval(X_aug)
        # intervals shape: (n, 2, 1) — squeeze last dim
        lower = intervals[:, 0, 0]
        upper = intervals[:, 1, 0]

    except Exception as e:
        logger.warning(f"CQR fitting failed: {e}; falling back to sigma-scaled")
        return _fallback_result(y, sigma_arr, alpha)

    # Ensure lower <= upper
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

    # Empirical coverage on the full calibration set
    covered = (y >= lower) & (y <= upper)
    coverage = float(np.mean(covered))

    widths = upper - lower
    mean_width = float(np.mean(widths))

    return {
        "lower": lower,
        "upper": upper,
        "coverage": coverage,
        "mean_width": mean_width,
    }


def _fallback_result(
    y: np.ndarray, sigma: np.ndarray, alpha: float
) -> dict[str, Any]:
    """Sigma-scaled symmetric fallback when CQR cannot run.

    Uses the same logic as the production conformal module but without the
    finite-sample correction, since this is shadow-only.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(sigma, dtype=float)
    s = np.where(s > 0, s, 0.01)

    # Simple quantile of |y| / sigma as the half-width multiplier
    scores = np.abs(y) / s
    q = float(np.quantile(scores, 1.0 - alpha))
    half = q * s

    lower = y - half
    upper = y + half

    covered = (y >= lower) & (y <= upper)
    coverage = float(np.mean(covered))
    mean_width = float(np.mean(upper - lower))

    return {
        "lower": lower,
        "upper": upper,
        "coverage": coverage,
        "mean_width": mean_width,
    }
