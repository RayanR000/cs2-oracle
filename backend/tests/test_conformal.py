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
    BETA_NEUTRAL,
    NOMINAL_COVERAGE,
    band,
    band_signed,
    calibrate,
    calibrate_signed,
    resolve_scale,
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


# --- signed quantiles: the two-sided band that recentres itself -------------


def test_band_signed_with_a_symmetric_pair_equals_the_symmetric_band():
    """The from_meta fallback for a pre-signed artifact is (-q_hat, +q_hat),
    and it must reproduce today's band to the bit -- otherwise loading an old
    model silently changes the served geometry."""
    mid = np.array([1.0, -2.0, 3.5])
    sigma = np.array([0.1, 0.2, 0.4])
    q_hat = 5.0
    lo_a, hi_a = band(mid, sigma, q_hat)
    lo_b, hi_b = band_signed(mid, sigma, -q_hat, q_hat)
    assert np.array_equal(lo_a, lo_b)
    assert np.array_equal(hi_a, hi_b)


def test_calibrate_signed_is_narrower_than_the_absolute_band_on_biased_residuals():
    """The whole point. When the q50 residual is off-centre (its median is not
    zero -- what DIRECTION_UPWEIGHT=1.5 produces, P(actual<mid)=0.60-0.71), the
    absolute quantile is dominated by the fat tail and the symmetric band is
    inflated. Two signed quantiles centre on the residual's own median and are
    strictly narrower at the same nominal coverage."""
    rng = np.random.default_rng(0)
    n = 8000
    sigma = np.full(n, 0.2)
    residuals = rng.normal(loc=2.0, scale=1.0, size=n)   # biased upward

    q_hat = calibrate(residuals, sigma, ALPHA)
    lo_s, hi_s = band(np.zeros(n), sigma, q_hat)
    width_symmetric = float((hi_s - lo_s)[0])

    q_lo, q_hi = calibrate_signed(residuals, sigma, ALPHA)
    lo, hi = band_signed(np.zeros(n), sigma, q_lo, q_hi)
    width_signed = float((hi - lo)[0])

    assert width_signed < width_symmetric * 0.9
    cov = float(((residuals >= lo) & (residuals <= hi)).mean())
    assert cov >= NOMINAL_COVERAGE - 0.02


def test_calibrate_signed_achieves_held_out_marginal_coverage():
    """Split conformal's two-sided guarantee: calibrate the pair on one half of
    exchangeable, heteroscedastic, off-centre data and it covers near nominal on
    a disjoint half. The offset is proportional to sigma so the signed scores are
    constant across the volatility range, isolating the two-sided level maths."""
    rng = np.random.default_rng(3)
    n = 8000
    sigma = rng.uniform(0.05, 0.5, size=n)
    residuals = rng.normal(loc=sigma * 3.0, scale=sigma * 10.0, size=n)

    idx = rng.permutation(n)
    cal_idx, test_idx = idx[: n // 2], idx[n // 2:]

    q_lo, q_hi = calibrate_signed(residuals[cal_idx], sigma[cal_idx], ALPHA)
    lo, hi = band_signed(np.zeros(test_idx.size), sigma[test_idx], q_lo, q_hi)
    covered = (residuals[test_idx] >= lo) & (residuals[test_idx] <= hi)
    assert np.mean(covered) == pytest.approx(NOMINAL_COVERAGE, abs=0.03)


def test_calibrate_signed_returns_ordered_quantiles_and_the_band_cannot_cross():
    rng = np.random.default_rng(7)
    residuals = rng.normal(loc=1.5, scale=2.0, size=5000)
    sigma = np.full(5000, 0.3)
    q_lo, q_hi = calibrate_signed(residuals, sigma, ALPHA)
    assert q_lo <= q_hi
    lo, hi = band_signed(np.array([4.0, -1.0]), np.array([0.3, 0.3]), q_lo, q_hi)
    assert np.all(lo <= hi)


def test_band_signed_width_scales_with_sigma():
    """band_signed goes through resolve_scale like band, so a doubled sigma
    doubles the width -- the property that keeps the pair a matched set with
    beta and the learned scale."""
    lo, hi = band_signed(np.zeros(3), np.array([0.1, 0.2, 0.4]),
                         q_lo=-4.0, q_hi=6.0)
    widths = hi - lo
    assert widths[1] == pytest.approx(widths[0] * 2)
    assert widths[2] == pytest.approx(widths[0] * 4)


def test_calibrate_signed_rejects_an_empty_calibration_set():
    with pytest.raises(ValueError, match="empty calibration set"):
        calibrate_signed(np.array([]), np.array([]), ALPHA)


# --- resolve_scale: the learned alternative to sigma ------------------------


def test_resolve_scale_returns_sigma_when_nothing_is_learned():
    """The default path must be byte-identical to what shipped before the
    learned scale existed."""
    sig = np.array([0.02, 0.07, 0.5])
    assert np.array_equal(resolve_scale(sig), sig)
    assert np.array_equal(resolve_scale(sig, BETA_NEUTRAL),
                          sig)


def test_resolve_scale_serves_a_learned_scale_as_is():
    """`beta` does not apply to a fitted scale: the exponent corrects sigma's
    over-reaction and an estimate fitted on the residuals has none."""
    learned = np.array([1.0, 2.0, 3.0])
    out = resolve_scale(np.array([9.0, 9.0, 9.0]), learned=learned)
    assert np.array_equal(out, learned)


def test_stacking_a_learned_scale_on_an_exponent_raises():
    """Two alternative denominators, not two layers. Applying both would
    re-tilt the band in the opposite direction -- the failure measured in
    2026-08-12-served-sigma-profile.md -- so there is no sensible reading of
    the request and it must not silently pick one."""
    with pytest.raises(ValueError, match="alternative denominators"):
        resolve_scale(np.array([0.07]), beta=0.35,
                                learned=np.array([1.0]))


def test_calibrate_and_band_round_trip_through_a_learned_scale():
    """q_hat calibrated against a learned scale, then served against the same
    scale, covers at nominal. The pairing is the correctness condition."""
    rng = np.random.default_rng(11)
    n = 20_000
    s = rng.uniform(0.5, 4.0, n)
    resid = rng.normal(scale=s)

    q = calibrate(resid, sigma=None, learned_scale=s)
    lo, hi = band(np.zeros(n), sigma=None, q_hat=q, learned_scale=s)
    covered = (resid >= lo) & (resid <= hi)
    assert covered.mean() == pytest.approx(NOMINAL_COVERAGE, abs=0.01)


def test_a_learned_q_hat_served_against_sigma_is_not_a_degraded_band():
    """It is an unrelated one. This is the failure the artifact's matched-pair
    persistence exists to prevent, and it is gross rather than subtle."""
    rng = np.random.default_rng(12)
    n = 20_000
    s = rng.uniform(0.5, 4.0, n)
    sigma = s ** 2.0 / 50.0          # a different variable, different units
    resid = rng.normal(scale=s)

    q_learned = calibrate(resid, sigma=None, learned_scale=s)
    lo, hi = band(np.zeros(n), sigma, q_hat=q_learned)   # mismatched
    covered = ((resid >= lo) & (resid <= hi)).mean()
    assert abs(covered - NOMINAL_COVERAGE) > 0.15, (
        f"a mismatched scale covered {covered:.3f}, close enough to nominal to "
        f"go unnoticed; the pairing guard rests on this being obvious"
    )
