"""Locally-weighted split conformal band.

Replaces 24 p10/p90 quantile GBMs whose empirical coverage was 39-48% against
a nominal target the old code stated two ways (forecaster.py:3088 said 80%,
:3092 set alpha=0.10 for 90%). Nominal is pinned at 80% here.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.conformal import (
    ALPHA,
    NOMINAL_COVERAGE,
    band,
    calibrate,
    sigma_bounds,
    sigma_from_columns,
)


def test_nominal_coverage_is_pinned_at_eighty_percent():
    assert NOMINAL_COVERAGE == 0.80
    assert ALPHA == pytest.approx(0.20)


def test_sigma_is_a_coefficient_of_variation():
    # price_std_60d is in dollars; the band lives in return space, so sigma
    # must be scale-free or a $5000 knife and a $1 case get the same width.
    sigma = sigma_from_columns(
        price_std_60d=np.array([10.0, 0.10]),
        price=np.array([100.0, 1.0]),
        floor=0.001, cap=10.0,
    )
    assert sigma == pytest.approx([0.10, 0.10])


def test_sigma_is_clipped_to_the_bounds():
    sigma = sigma_from_columns(
        price_std_60d=np.array([0.0, 1000.0]),
        price=np.array([100.0, 1.0]),
        floor=0.01, cap=2.0,
    )
    assert sigma == pytest.approx([0.01, 2.0])


def test_sigma_falls_back_for_missing_or_zero_history():
    # PREDICT_MIN_HISTORY_DAYS = 14, so eligible items can have no 60d std.
    # A NaN reaching forecast_low would surface in the UI.
    sigma = sigma_from_columns(
        price_std_60d=np.array([np.nan, 0.0]),
        price=np.array([100.0, 50.0]),
        floor=0.02, cap=2.0,
        fallback=0.35,
    )
    assert sigma == pytest.approx([0.35, 0.35])
    assert np.all(np.isfinite(sigma))


def test_sigma_falls_back_when_price_is_nonpositive():
    sigma = sigma_from_columns(
        price_std_60d=np.array([1.0]),
        price=np.array([0.0]),
        floor=0.02, cap=2.0, fallback=0.35,
    )
    assert sigma == pytest.approx([0.35])


def test_sigma_bounds_are_the_first_and_ninety_ninth_percentiles():
    raw = np.concatenate([np.linspace(0.01, 1.0, 1000), [np.nan, np.inf]])
    floor, cap = sigma_bounds(raw)
    assert floor == pytest.approx(np.percentile(np.linspace(0.01, 1.0, 1000), 1))
    assert cap == pytest.approx(np.percentile(np.linspace(0.01, 1.0, 1000), 99))
    assert floor > 0


def test_calibrate_achieves_nominal_coverage_on_its_own_calibration_set():
    rng = np.random.default_rng(0)
    n = 5000
    sigma = rng.uniform(0.05, 0.5, size=n)
    # Heteroscedastic residuals: spread proportional to sigma.
    residuals = rng.normal(scale=sigma * 10.0, size=n)
    q_hat = calibrate(residuals, sigma, ALPHA)
    low, high = band(np.zeros(n), sigma, q_hat)
    covered = np.mean((residuals >= low) & (residuals <= high))
    assert covered == pytest.approx(NOMINAL_COVERAGE, abs=0.02)


def test_calibrate_is_scale_invariant_in_sigma():
    # Doubling sigma halves q_hat, leaving the band unchanged. This is the
    # property that makes the normalization meaningful rather than cosmetic.
    rng = np.random.default_rng(1)
    residuals = rng.normal(scale=1.0, size=2000)
    sigma = np.full(2000, 0.2)
    q1 = calibrate(residuals, sigma, ALPHA)
    q2 = calibrate(residuals, sigma * 2, ALPHA)
    assert q1 == pytest.approx(q2 * 2, rel=1e-9)


def test_band_width_varies_with_sigma():
    mid = np.zeros(3)
    sigma = np.array([0.1, 0.2, 0.4])
    low, high = band(mid, sigma, q_hat=5.0)
    widths = high - low
    assert widths[1] == pytest.approx(widths[0] * 2)
    assert widths[2] == pytest.approx(widths[0] * 4)


def test_band_is_ordered_and_centred_on_the_median():
    mid = np.array([3.0, -2.0])
    low, high = band(mid, np.array([0.2, 0.2]), q_hat=4.0)
    assert np.all(low <= mid) and np.all(mid <= high)
    assert (low + high) / 2 == pytest.approx(mid)


def test_calibrate_rejects_an_empty_calibration_set():
    with pytest.raises(ValueError, match="empty calibration set"):
        calibrate(np.array([]), np.array([]), ALPHA)


def test_calibrate_ignores_nonfinite_scores():
    residuals = np.array([1.0, 2.0, np.nan, 3.0])
    sigma = np.array([0.1, 0.1, 0.1, 0.0])
    q_hat = calibrate(residuals, sigma, ALPHA)
    assert np.isfinite(q_hat)
