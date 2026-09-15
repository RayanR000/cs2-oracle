"""The attribution instrument's arithmetic, pinned.

`scripts/attribute_marginal_coverage.py` answers how much of the band's marginal
over-coverage a shift in the served `sigma` distribution buys. The whole answer
rests on two properties of `CoverageCurve`, and both are asserted here on
synthetic data with a KNOWN elasticity rather than on the archive:

1. At `beta = 1` -- the exponent split conformal assumes -- coverage does not
   depend on `sigma`, so no `sigma`-mix shift can move marginal coverage. This is
   the pre-registered placebo. If it ever moves, the instrument has a path to
   marginal coverage that is not the tilt and every figure it produced is void.
2. At `beta < 1` coverage RISES with `sigma`, so an upward shift over-covers.
   That is the mechanism being sized.

See `docs/research/2026-08-12-marginal-coverage-attribution-preregistration.md`.
"""

from __future__ import annotations

import numpy as np
import pytest
from models import conformal
from scripts.archive.attribute_marginal_coverage import (
    CoverageCurve,
    empirical_decile_coverage,
    shifted_sigma,
    solve_k,
)

FLOOR, CAP = 1e-4, 1e4  # wide, so the clip is not what the test measures


def _panel(elasticity: float, n: int = 60_000, seed: int = 20260812):
    """`|resid| = sigma ** elasticity * lognormal noise`, by construction."""
    rng = np.random.default_rng(seed)
    sigma = np.exp(rng.normal(np.log(0.07), 0.6, n))
    resid = (sigma**elasticity) * np.exp(rng.normal(0.0, 0.5, n)) * 100.0
    return resid, sigma


def test_placebo_beta_one_makes_marginal_coverage_shift_invariant():
    """The pre-registered placebo. `beta = 1` -> flat curve -> zero shift effect."""
    resid, sigma = _panel(elasticity=1.0)
    q_hat = conformal.calibrate(resid, sigma)
    flat = CoverageCurve(resid, sigma, q_hat, beta=1.0)

    base = flat.marginal(sigma)
    for k in (0.5, 0.9, 1.3, 2.0, 5.0):
        moved = flat.marginal(shifted_sigma(sigma, k, FLOOR, CAP))
        assert abs(moved - base) < 5e-4, f"beta=1 moved at k={k}"


def test_a_sub_unit_elasticity_makes_high_sigma_over_covered():
    """`beta < 1` -> coverage monotone increasing in sigma, and an up-shift over-covers."""
    resid, sigma = _panel(elasticity=0.4)
    q_hat = conformal.calibrate(resid, sigma)
    curve = CoverageCurve(resid, sigma, q_hat)

    assert curve.beta == pytest.approx(0.4, abs=0.02)
    # Marginal coverage is on target by construction -- which is exactly why this
    # defect is invisible to the guarantee the module advertises.
    assert curve.marginal(sigma) == pytest.approx(0.80, abs=0.01)

    quiet, loud = np.quantile(sigma, [0.05, 0.95])
    assert curve(np.array([quiet]))[0] < 0.80 < curve(np.array([loud]))[0]

    up = curve.marginal(shifted_sigma(sigma, 1.3, FLOOR, CAP))
    down = curve.marginal(shifted_sigma(sigma, 0.77, FLOOR, CAP))
    assert up > 0.80 > down


def test_solve_k_inverts_the_shift():
    """`solve_k` returns the shift that produces a given marginal coverage."""
    resid, sigma = _panel(elasticity=0.4)
    q_hat = conformal.calibrate(resid, sigma)
    curve = CoverageCurve(resid, sigma, q_hat)

    k = solve_k(curve, sigma, 0.87, FLOOR, CAP)
    assert np.isfinite(k) and k > 1.0
    assert curve.marginal(shifted_sigma(sigma, k, FLOOR, CAP)) == pytest.approx(0.87, abs=1e-3)


def test_solve_k_reports_an_unreachable_target_rather_than_a_number():
    """A target no shift can reach must come back NaN, not a silent bound.

    The inverse leg's whole use is to say "the mechanism cannot produce this",
    so returning a clamped `k` would turn a refutation into a false attribution.
    """
    resid, sigma = _panel(elasticity=1.0)  # flat curve: nothing is reachable
    q_hat = conformal.calibrate(resid, sigma)
    flat = CoverageCurve(resid, sigma, q_hat, beta=1.0)
    assert np.isnan(solve_k(flat, sigma, 0.95, FLOOR, CAP))


def test_the_curve_reproduces_empirical_decile_coverage():
    """The pre-registered validity bar: MAE over deciles <= 2pp.

    The curve is parametric in the mean and empirical in the noise; if that form
    could not reproduce the bins it is fitted on, the extrapolated shift figures
    would be unreadable.
    """
    resid, sigma = _panel(elasticity=0.4)
    q_hat = conformal.calibrate(resid, sigma)
    curve = CoverageCurve(resid, sigma, q_hat)

    emp, dec = empirical_decile_coverage(resid, sigma, q_hat)
    c_all = curve(sigma)
    mod = np.array([c_all[dec == k].mean() for k in range(10)])
    assert float(np.mean(np.abs(mod - emp))) < 0.02, f"{mod} vs {emp}"


def test_the_decile_comparison_averages_over_rows_not_the_median_sigma():
    """Why `empirical_decile_coverage` returns the index and not a median sigma.

    `c` is steep inside the extreme deciles, so evaluating it at the decile's
    median sigma is not its mean over that decile. The gap is a Jensen artifact
    of the comparison, not a defect in the model, and reading it as validity
    error would fail the instrument on data whose elasticity is known exactly.
    """
    resid, sigma = _panel(elasticity=0.4)
    q_hat = conformal.calibrate(resid, sigma)
    curve = CoverageCurve(resid, sigma, q_hat)

    emp, dec = empirical_decile_coverage(resid, sigma, q_hat)
    c_all = curve(sigma)
    by_row = np.array([c_all[dec == k].mean() for k in range(10)])
    at_median = curve(np.array([np.median(sigma[dec == k]) for k in range(10)]))

    assert abs(by_row[0] - emp[0]) < abs(at_median[0] - emp[0])
    assert float(np.mean(np.abs(at_median - emp))) > float(np.mean(np.abs(by_row - emp)))
