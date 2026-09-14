"""WACI harness helpers: level-match monotonicity, fit/apply shapes, placebo separation.

What is silent when it breaks: a level-match that does not actually land on
80% (the whole verdict is level-matched), a WACI fit that collapses to the
pooled pair without saying so, and a placebo that is accidentally identical
to the treatment.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import conformal
from scripts.measure_waci import (
    _level_match,
    apply_fit,
    fit_s0,
    fit_waci,
)


class _FakePanel:
    """Minimal Panel surface the fit functions read: resid/sigma over a
    date-sorted array with embargo arithmetic."""

    def __init__(self, resid, sigma, dates, embargo=16):
        self.resid = np.asarray(resid, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.dates = np.asarray(dates)
        self.unique_dates = np.unique(self.dates)
        self.embargo = embargo

    def upto(self, day):
        return int(np.searchsorted(self.dates, day, side="right"))

    def since(self, day):
        return int(np.searchsorted(self.dates, day, side="left"))

    def rows_on(self, day):
        return slice(self.since(day), self.upto(day))

    def avail(self, day):
        return day - np.timedelta64(self.embargo, "D")


def _panel(n_dates=60, per_date=200, seed=0):
    rng = np.random.default_rng(seed)
    resid, sigma, dates = [], [], []
    base = np.datetime64("2024-01-01")
    for d in range(n_dates):
        day = base + np.timedelta64(d, "D")
        s = rng.lognormal(-2.6, 0.6, per_date)
        r = rng.normal(0, s)
        resid.append(r)
        sigma.append(s)
        dates.append(np.full(per_date, day))
    return _FakePanel(np.concatenate(resid), np.concatenate(sigma), np.concatenate(dates))


def test_level_match_lands_on_target():
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 1, 5000)
    q_lo = np.full(5000, -1.0)
    q_hi = np.full(5000, 1.0)
    c = _level_match(scores, q_lo, q_hi)
    got = float(np.mean((scores >= c * q_lo) & (scores <= c * q_hi)))
    assert abs(got - 0.80) < 0.005


def test_level_match_scales_asymmetric_pair():
    rng = np.random.default_rng(2)
    scores = rng.normal(0.5, 1.0, 8000)
    q_lo = np.full(8000, -0.8)
    q_hi = np.full(8000, 1.5)
    c = _level_match(scores, q_lo, q_hi)
    assert c > 0 and np.isfinite(c)
    got = float(np.mean((scores >= c * q_lo) & (scores <= c * q_hi)))
    assert abs(got - 0.80) < 0.005


def test_s0_returns_pooled_signed_pair():
    p = _panel()
    day = np.datetime64("2024-03-01")
    fit = fit_s0(p, day)
    assert fit is not None
    q_lo, q_hi = fit
    assert q_lo < 0 < q_hi


def test_waci_returns_per_bin_dict_with_spread():
    p = _panel()
    day = np.datetime64("2024-03-01")
    fit = fit_waci(p, day)
    assert isinstance(fit, dict)
    assert len(fit["bin_q_lo"]) == 10
    # Tilted data (resid ~ sigma): bins must disagree, or the arm is a
    # no-op wearing a Mondrian costume.
    assert (fit["bin_q_hi"].max() - fit["bin_q_hi"].min()) > 0


def test_waci_falls_back_to_global_on_thin_history():
    p = _panel(n_dates=3, per_date=10)
    day = np.datetime64("2024-01-10")
    fit = fit_waci(p, day)
    # 30 rows < 10 bins * 10: documented fallback, single-bin dict.
    assert fit is None or len(fit["bin_q_hi"]) == 1


def test_apply_fit_broadcasts_pooled_and_looks_up_waci():
    p = _panel()
    day = np.datetime64("2024-03-01")
    s0 = fit_s0(p, day)
    w = fit_waci(p, day)
    s = np.array([0.03, 0.10, 0.25])
    lo0, hi0 = apply_fit(s0, s)
    assert (lo0 == lo0[0]).all() and (hi0 == hi0[0]).all()
    lo1, hi1 = apply_fit(w, s)
    assert lo1.shape == s.shape and hi1.shape == s.shape
    assert np.all(lo1 <= hi1)


def test_placebo_uses_same_machinery():
    p = _panel()
    day = np.datetime64("2024-03-01")
    a = fit_waci(p, day)
    b = fit_waci(p, day, shuffle=True)
    assert isinstance(b, dict) and len(b["bin_q_hi"]) == 10
    # Same rows in, same shapes out; values differ because the scale was
    # destroyed (overwhelmingly likely on 12k shuffled rows).
    assert not np.array_equal(a["bin_q_hi"], b["bin_q_hi"])


def test_conformal_waci_lookup_is_monotone_in_scale():
    params = {
        "bin_centres": np.array([0.03, 0.07, 0.15]),
        "bin_q_lo": np.array([-2.0, -2.0, -2.0]),
        "bin_q_hi": np.array([1.0, 2.0, 4.0]),
        "fallback_q_lo": -2.0,
        "fallback_q_hi": 2.0,
    }
    _, hi = conformal.waci_lookup(np.array([0.03, 0.07, 0.15]), params)
    assert hi[0] < hi[1] < hi[2]
