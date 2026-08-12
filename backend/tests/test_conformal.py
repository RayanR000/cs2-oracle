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


def test_calibrate_achieves_held_out_marginal_coverage():
    """Split conformal's actual distribution-free guarantee: calibrate q_hat
    on one half of exchangeable data, measure coverage on a disjoint held-out
    half, and it lands near the nominal level.

    This test deliberately does NOT discriminate sigma-normalization: an
    earlier version of this suite calibrated and measured coverage on the
    SAME set, which is a tautology (np.quantile inverting its own CDF, ~0.80
    coverage whether sigma is correct, inverted, or a constant). Splitting
    into disjoint calibration/test halves closes that hole for the marginal
    number, but marginal coverage is still ~80% with or without
    normalization by construction of split conformal — see the next test,
    which partitions the held-out set by sigma and is the one that actually
    tells a locally-weighted implementation apart from an unnormalized one.
    """
    rng = np.random.default_rng(0)
    n = 8000
    sigma = rng.uniform(0.05, 0.5, size=n)
    # Heteroscedastic residuals: spread proportional to sigma.
    residuals = rng.normal(scale=sigma * 10.0, size=n)

    idx = rng.permutation(n)
    cal_idx, test_idx = idx[: n // 2], idx[n // 2:]

    q_hat = calibrate(residuals[cal_idx], sigma[cal_idx], ALPHA)
    low, high = band(np.zeros(test_idx.size), sigma[test_idx], q_hat)
    covered = (residuals[test_idx] >= low) & (residuals[test_idx] <= high)
    assert np.mean(covered) == pytest.approx(NOMINAL_COVERAGE, abs=0.03)


def test_calibrate_gives_conditional_coverage_across_volatility_strata():
    """The discriminating test: sigma-normalization equalizes coverage across
    volatility levels, not just on average.

    Measured directly against this module (n=8000, 50/50 calibration/test
    split via the same rng stream, residual spread proportional to sigma):
    with normalization the low- vs high-volatility coverage gap on the
    held-out half is ~0.007 (79.1% vs 79.8%); replacing sigma with a constant
    (sigma_from_columns bypassed, no normalization) blows the gap out to
    ~0.307 (94.8% vs 64.1%), while marginal coverage stays ~79.4% in BOTH
    cases. That last fact is exactly why the previous test cannot tell these
    apart and this one is required.
    """
    rng = np.random.default_rng(0)
    n = 8000
    sigma = rng.uniform(0.05, 0.5, size=n)
    residuals = rng.normal(scale=sigma * 10.0, size=n)

    idx = rng.permutation(n)
    cal_idx, test_idx = idx[: n // 2], idx[n // 2:]
    sigma_test = sigma[test_idx]
    residuals_test = residuals[test_idx]

    q_hat = calibrate(residuals[cal_idx], sigma[cal_idx], ALPHA)
    low, high = band(np.zeros(test_idx.size), sigma_test, q_hat)
    covered = (residuals_test >= low) & (residuals_test <= high)

    median_sigma = np.median(sigma_test)
    low_vol = sigma_test <= median_sigma
    high_vol = ~low_vol
    gap = abs(np.mean(covered[low_vol]) - np.mean(covered[high_vol]))
    assert gap < 0.05


def _coverage_by_sigma_decile(sigma, residuals, denom_cal, denom_test, rng):
    """Held-out coverage per sigma decile, calibrating on half and scoring on half.

    `denom_*` is what the nonconformity score divides by, which is the whole
    subject: production passes `sigma`, and the remedy under test passes
    `sigma**beta`.
    """
    n = sigma.size
    idx = rng.permutation(n)
    cal, test = idx[: n // 2], idx[n // 2:]
    q_hat = calibrate(residuals[cal], denom_cal[cal], ALPHA)
    covered = np.abs(residuals[test]) <= q_hat * denom_test[test]

    s = sigma[test]
    edges = np.quantile(s, np.linspace(0, 1, 11)[1:-1])
    dec = np.searchsorted(edges, s, side="right")
    per = np.array([covered[dec == d].mean() for d in range(10)])
    return float(covered.mean()), per


def test_sigma_normalization_fails_when_the_elasticity_is_not_one():
    """CHARACTERIZATION of a measured defect, not a property being asserted.

    `test_calibrate_gives_conditional_coverage_across_volatility_strata` above
    builds residuals at `scale = sigma * 10.0` — elasticity **exactly 1.0 by
    construction** — so it passes whatever the archive does and cannot fail on
    this axis. That blind spot was named in
    `docs/changelog/2026-08-12-conformal-basis-follows-serving.md`; this test
    closes it.

    On ~500K item-days of the >=$1 cohort, `d log|residual| / d log sigma` is
    **0.408 / 0.401 / 0.363 / 0.327** at 3/7/14/30d fitted walk-forward, not
    1.000, and it is not the sigma clip (excluding every clipped row moves the
    whole-panel fit from 0.389 to 0.395 at h=3 and 0.348 to 0.369 at h=30, and
    only 1.2% / 1.0% of rows sit at the floor / cap). Level-matched to 80%
    marginal, coverage then runs
    **62 -> 95%** across sigma deciles at h=3 and **58 -> 98%** at h=30.
    `docs/changelog/2026-08-12-the-band-is-tilted-in-sigma.md`.

    Direction matters and is easy to get backwards: the score is
    `|r| / sigma ∝ sigma**(beta - 1)`, so with `beta < 1` it FALLS as sigma
    rises — low-sigma rows are UNDER-covered and high-sigma rows over-covered.

    Marginal coverage is unaffected, which is why nothing caught this for so
    long, and it is asserted here so a future fix cannot trade one for the other.
    """
    rng = np.random.default_rng(20260812)
    n = 40_000
    beta = 0.4
    sigma = rng.uniform(0.02, 0.5, size=n)
    residuals = rng.normal(scale=(sigma ** beta) * 3.0, size=n)

    marginal, per = _coverage_by_sigma_decile(
        sigma, residuals, sigma, sigma, np.random.default_rng(1))

    # The marginal guarantee is untouched -- this defect is invisible to it.
    assert marginal == pytest.approx(NOMINAL_COVERAGE, abs=0.02)
    # And conditional coverage is badly tilted, in the measured direction.
    assert per[0] < NOMINAL_COVERAGE - 0.10
    assert per[-1] > NOMINAL_COVERAGE + 0.10
    assert per[-1] - per[0] > 0.20

    # The remedy, expressed with no change to this module: normalize by
    # sigma**beta instead of sigma. `calibrate` and `band` already take the
    # denominator as an argument, so the exponent is a fitted CONSTANT the
    # artifact would have to persist -- not new math.
    marginal_b, per_b = _coverage_by_sigma_decile(
        sigma, residuals, sigma ** beta, sigma ** beta,
        np.random.default_rng(1))
    assert marginal_b == pytest.approx(NOMINAL_COVERAGE, abs=0.02)
    assert per_b.max() - per_b.min() < 0.10


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
