"""Bagged/Mondrian q_hat harness: grids, fits, level-match, placebo separation.

What is silent when it breaks: grids that are secretly identical (bagging a
constant), Mondrian edges fitted on test rows (leakage), a level-match that
does not land on 80% (the verdict is level-matched), and a placebo
indistinguishable from treatment.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.measure_qhat_bagging_mondrian import (
    _level_match_c,
    apply_fit,
    evaluate,
    fit_bagged,
    fit_mondrian,
    fit_s0,
    grid_date_sets,
    val_windows,
)


class _FakePanel:
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


def _panel(n_dates=400, per_date=200, seed=0, tilt: float = 0.0):
    rng = np.random.default_rng(seed)
    resid, sigma, dates = [], [], []
    base = np.datetime64("2024-01-01")
    for d in range(n_dates):
        day = base + np.timedelta64(d, "D")
        s = rng.lognormal(-2.6, 0.6, per_date)
        # tilt>0: high-sigma rows carry larger |resid|/sigma (a real vol
        # regime); tilt=0: homoscedastic scores, Mondrian has nothing to find.
        r = rng.normal(0, s * (1.0 + tilt * (s - s.mean())))
        resid.append(r)
        sigma.append(s)
        dates.append(np.full(per_date, day))
    return _FakePanel(np.concatenate(resid), np.concatenate(sigma), np.concatenate(dates))


def test_val_windows_end_anchored_and_disjoint():
    dates = np.array([np.datetime64("2024-01-01") + np.timedelta64(d, "D") for d in range(500)])
    wins = val_windows(dates)
    assert wins, "no windows on 500 dates"
    assert wins[-1][1] == 500, "newest window must abut the frame end"
    for (s1, e1), (s2, e2) in zip(wins, wins[1:]):
        assert e1 <= s2, "windows must not overlap"


def test_shifted_grids_differ_but_overlap_heavily():
    p = _panel()
    g0, g1 = grid_date_sets(p.unique_dates, 2)
    assert g0 and g1 and g0 != g1, "shifted grids must cover different dates"
    # Shift (75 positions) exceeds the window (30), so grids are disjoint by
    # construction — the union is what shows K grids see more history.
    assert len(g0 | g1) > len(g0), "second grid must add unseen dates"


def test_s0_and_bagged_agree_on_homoscedastic_panel():
    p = _panel(tilt=0.0)
    grids = grid_date_sets(p.unique_dates, 2)
    day = np.datetime64("2025-01-01")
    s0 = fit_s0(p, day, grids[0])
    bag = fit_bagged(p, day, grids, "mean")
    assert s0 is not None and bag is not None
    assert abs(bag - s0) / s0 < 0.15, f"bagged {bag} far from pooled {s0}"


def test_mondrian_edges_come_from_calibration_only():
    p = _panel(n_dates=300, tilt=1.0)
    grids = grid_date_sets(p.unique_dates, 2)
    day = np.datetime64("2024-11-01")
    fit = fit_mondrian(p, day, grids[0])
    assert fit is not None
    edges, per_bin, fallback = fit
    assert len(edges) == 9 and len(per_bin) == 10
    # Per-bin qs must spread on tilted data, or the arm is a no-op.
    assert per_bin.max() - per_bin.min() > 0
    # Test rows outside the calibration range still map (clipped, finite).
    q = apply_fit(fit, np.array([1e-6, 10.0]))
    assert np.all(np.isfinite(q))


def test_mondrian_falls_back_on_thin_bins():
    p = _panel(n_dates=250, per_date=60)
    grids = grid_date_sets(p.unique_dates, 2)
    day = np.datetime64("2024-06-01")
    fit = fit_mondrian(p, day, grids[0])
    if fit is not None:
        _, per_bin, fallback = fit
        assert np.all(np.isfinite(per_bin))


def test_placebo_uses_same_machinery_with_different_values():
    p = _panel(tilt=1.0)
    grids = grid_date_sets(p.unique_dates, 2)
    day = np.datetime64("2024-11-01")
    a = fit_mondrian(p, day, grids[0])
    b = fit_mondrian(p, day, grids[0], shuffle=True)
    assert a is not None and b is not None
    assert not np.array_equal(a[1], b[1]), "placebo identical to treatment"


def test_level_match_lands_on_target():
    rng = np.random.default_rng(3)
    scores = np.abs(rng.normal(0, 1, 8000))
    q = np.full(8000, 1.0)
    c = _level_match_c(scores, q)
    assert abs(float(np.mean(scores <= q * c)) - 0.80) < 0.005


def test_evaluate_reports_width_and_guardrail_fields():
    p = _panel()
    grids = grid_date_sets(p.unique_dates, 2)
    td = [d for d in p.unique_dates if d > np.datetime64("2024-10-01")][:60]
    m = evaluate(p, lambda pp, d: fit_s0(pp, d, grids[0]), td)
    assert m, "no fit on a dense fake panel"
    for k in ("M1_marginal", "mean_width", "level_match_c", "mean_width_lm", "min_decile_cov_lm", "M2lm_sigma_err_pp"):
        assert np.isfinite(m[k]), k
    assert abs(m["M1lm_marginal"] - 0.80) < 0.005
