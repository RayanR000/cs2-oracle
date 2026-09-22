"""The ACI instrument: equivalence to the harness, no lookahead, placebo, bar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from models import conformal
from scripts.archive import measure_aci as aci
from scripts.archive.measure_qhat_bagging_mondrian import (
    CalibPanel,
    calib_mask_for,
    grid_date_sets,
    pooled_qhat,
)
from scripts.archive.measure_qhat_bagging_mondrian import (
    test_dates_for_panel as walk_forward_dates,
)

N_DATES = 1_000
ROWS_PER_DATE = 120


def _panel(horizon: int = 3, regime: bool = False, seed: int = 0, block: int = 30, vol: float = 3.0) -> CalibPanel:
    """Synthetic scored panel. `regime` alternates `block`-date calm/volatile
    blocks (residuals x `vol`) that sigma does NOT see -- the time
    non-exchangeability ACI targets."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=N_DATES, freq="D")
    date = np.repeat(days, ROWS_PER_DATE)
    sigma = rng.uniform(0.5, 2.0, size=date.size)
    scale = np.ones(date.size)
    if regime:
        phase = (np.arange(N_DATES) // block) % 2
        scale = np.repeat(np.where(phase == 1, vol, 1.0), ROWS_PER_DATE)
    resid = rng.normal(size=date.size) * sigma * scale
    frame = pd.DataFrame({"date": date, "item_id": np.tile(np.arange(ROWS_PER_DATE), N_DATES), "resid": resid, "sigma": sigma})
    return CalibPanel(frame, horizon)


def _grid(p: CalibPanel) -> frozenset:
    return grid_date_sets(p.unique_dates, 1)[0]


def test_quantile_levels_equal_production_calibrate():
    rng = np.random.default_rng(1)
    resid, sigma = rng.normal(size=5_000), rng.uniform(0.5, 2.0, size=5_000)
    scores = np.abs(resid) / sigma
    alphas = [0.005, 0.1, 0.2, 0.35, 0.6]
    got = aci.quantile_levels(scores, alphas)
    want = [conformal.calibrate(resid, sigma, alpha=a) for a in alphas]
    assert got == pytest.approx(want, rel=0, abs=1e-12)


def test_embargoed_calib_equals_the_harness_s0_fit():
    p = _panel()
    grid = _grid(p)
    calib = aci.EmbargoedCalib(p, grid)
    checked = 0
    for day in walk_forward_dates(p)[::40]:
        want = pooled_qhat(p, calib_mask_for(p, grid, day))
        got = calib.quantiles(day, [aci.ALPHA_STAR])
        if want is None:
            assert got is None
            continue
        assert got[0] == pytest.approx(want, abs=1e-12)
        checked += 1
    assert checked > 3


def test_no_arm_reads_an_outcome_before_it_resolves():
    """Rewriting every residual not yet resolved at day t leaves t's q unchanged."""
    p = _panel(horizon=7)
    dates = walk_forward_dates(p)
    t = dates[-60]
    unresolved = p.dates > t - np.timedelta64(7, "D")

    def run(panel):
        calib = aci.EmbargoedCalib(panel, _grid(panel))
        s0, qmaps, _ = aci.run_aci(panel, calib, dates)
        return s0[t], {g: qmaps[g][t] for g in aci.GAMMAS}, aci.run_roll(panel, dates)[t]

    before = run(p)
    p.resid = np.where(unresolved, p.resid * 50.0, p.resid)
    after = run(p)
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert after[2] == before[2]


def test_placebo_equals_aci_when_every_err_is_identical():
    """With nothing to permute, the placebo must reproduce ACI exactly --
    it differs only in err ORDER, never in schedule, gamma or values."""
    p = _panel()
    dates = walk_forward_dates(p)
    calib = aci.EmbargoedCalib(p, _grid(p))
    gamma = 0.02
    _, _, events = aci.run_aci(p, calib, dates, gammas=(gamma,))
    flat = [(d, 0.3) for d, _ in events[gamma]]

    # Replay ACI itself with the constant err, then the placebo on the same events.
    alpha, k, want = aci.ALPHA_STAR, 0, {}
    for day in dates:
        while k < len(flat) and flat[k][0] <= day:
            alpha = aci._clip(alpha + gamma * (aci.ALPHA_STAR - flat[k][1]))
            k += 1
        qs = calib.quantiles(day, [alpha])
        if qs is not None:
            want[day] = qs[0]
    assert aci.run_placebo(p, calib, dates, gamma, flat) == want


def test_aci_widens_in_volatile_blocks_and_narrows_in_calm_ones():
    """ACI's q tracks the regime S0 cannot see: compare the settled second half
    of each 30-date block."""
    p = _panel(regime=True)
    dates = walk_forward_dates(p)
    s0, qmaps, _ = aci.run_aci(p, aci.EmbargoedCalib(p, _grid(p)), dates, gammas=(0.05,))
    q = qmaps[0.05]
    first = p.unique_dates[0]
    pos = {d: int((d - first) / np.timedelta64(1, "D")) for d in q}
    settled = [d for d in q if pos[d] % 30 >= 15]
    volatile = [q[d] for d in settled if (pos[d] // 30) % 2 == 1]
    calm = [q[d] for d in settled if (pos[d] // 30) % 2 == 0]
    assert np.mean(volatile) > 1.5 * np.mean(calm)
    s0_v = [s0[d] for d in settled if (pos[d] // 30) % 2 == 1]
    s0_c = [s0[d] for d in settled if (pos[d] // 30) % 2 == 0]
    assert np.mean(s0_v) < 1.1 * np.mean(s0_c)  # the pooled fit does not move


def test_aci_beats_pooled_on_date_error_under_regime_shift():
    """The instrument can see the effect it tests for when the effect exists
    (synthetic: 14.8pp -> 6.9pp at gamma=0.05)."""
    p = _panel(regime=True, block=90, vol=2.0)
    dates = walk_forward_dates(p)
    s0, qmaps, _ = aci.run_aci(p, aci.EmbargoedCalib(p, _grid(p)), dates, gammas=(0.05,))
    common = [d for d in dates if d in s0]
    assert aci.score(p, qmaps[0.05], common)["M2lm_cond_err_pp"] < aci.score(p, s0, common)["M2lm_cond_err_pp"] - 3.0


def _m(err=5.0, m1=0.80, p10=0.70, width=1.0, dec=0.75):
    return {"M2lm_cond_err_pp": err, "M1_marginal": m1, "p10_date_cov_lm": p10, "mean_width_lm": width, "min_decile_cov_lm": dec}


def test_judge_applies_every_preregistered_condition():
    s0 = _m(err=8.0, m1=0.92, p10=0.60)
    good = _m(err=6.5, m1=0.81, p10=0.64, width=1.04, dec=0.72)
    assert all(aci.judge(good, s0, [_m(err=7.0), _m(err=7.5)]).values())

    assert not aci.judge(_m(err=6.5, m1=0.84, p10=0.64), s0, [])["c1_marginal"]
    assert not aci.judge(_m(err=7.5, p10=0.64), s0, [])["c2_date_err"]  # < 1pp better
    assert not aci.judge(good, s0, [_m(err=6.0)])["c2_date_err"]  # a control beats it
    assert not aci.judge(_m(err=6.5, p10=0.62), s0, [])["c3_p10"]
    assert not aci.judge(_m(err=6.5, p10=0.64, width=1.06), s0, [])["c4_width"]
    assert not aci.judge(_m(err=6.5, p10=0.64, dec=0.69), s0, [])["c5_decile"]


def test_void_conditions():
    s0 = _m(err=8.0)
    assert aci.void_reasons(150, s0, _m(err=7.5)) == []
    assert len(aci.void_reasons(99, s0, _m(err=7.5))) == 1
    assert "lacks the defect" in aci.void_reasons(150, _m(err=2.5), _m(err=2.5))[0]
    assert "machinery" in aci.void_reasons(150, s0, _m(err=6.9))[0]
