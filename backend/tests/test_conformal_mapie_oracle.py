"""`conformal.calibrate` checked against MAPIE's split-conformal regressor.

A hand-rolled conformal quantile is the kind of code that is silently off by
one order statistic. MAPIE (scikit-learn-contrib, BSD-3) is the reference
implementation: with a prefit constant-zero estimator its half-width is exactly
the ceil((n+1)(1-alpha))-th smallest |residual|.

Ours reads the same finite-sample level through `np.quantile`'s linear
interpolation, so it lands between that order statistic and the next one up.
That is the conservative side: coverage is never below the guarantee, and the
excess is at most one order statistic, which vanishes at production n (~10^5).
The test pins exactly that relationship, so a change to the level or the
interpolation that dropped below the guarantee fails here.

Skips when MAPIE is absent (it lives in the `probabilistic` extra).
"""

from __future__ import annotations

import numpy as np
import pytest
from models.conformal import ALPHA, calibrate

mapie_regression = pytest.importorskip("mapie.regression")
sklearn_dummy = pytest.importorskip("sklearn.dummy")


def _mapie_half_width(residuals: np.ndarray, alpha: float) -> float:
    n = residuals.size
    X = np.zeros((n, 1))
    est = sklearn_dummy.DummyRegressor(strategy="constant", constant=0.0).fit(X, np.zeros(n))
    reg = mapie_regression.SplitConformalRegressor(est, confidence_level=1.0 - alpha, prefit=True)
    reg.conformalize(X, residuals)
    _, intervals = reg.predict_interval(np.zeros((1, 1)))
    return float(intervals[0, 1, 0])


@pytest.mark.parametrize("n", [50, 137, 1_000, 20_000])
def test_q_hat_is_mapie_or_at_most_one_order_statistic_above(n):
    rng = np.random.default_rng(n)
    res = rng.standard_t(4, size=n)

    ours = calibrate(res, np.ones(n))
    reference = _mapie_half_width(res, ALPHA)

    s = np.sort(np.abs(res))
    k = int(np.ceil((n + 1) * (1.0 - ALPHA)))
    assert reference == pytest.approx(s[k - 1])  # the oracle is what we think it is
    assert reference <= ours <= s[min(k, n - 1)]


def test_sigma_scaling_matches_mapie_on_normalised_residuals():
    """With beta = 1 the score is |r| / sigma, so our q_hat must equal MAPIE's on
    the pre-normalised residuals (up to the same one-order-statistic slack)."""
    rng = np.random.default_rng(7)
    n = 2_000
    sigma = rng.uniform(0.02, 0.3, n)
    res = rng.standard_normal(n) * sigma

    ours = calibrate(res, sigma)
    reference = _mapie_half_width(res / sigma, ALPHA)

    s = np.sort(np.abs(res / sigma))
    k = int(np.ceil((n + 1) * (1.0 - ALPHA)))
    assert reference <= ours <= s[min(k, n - 1)]
