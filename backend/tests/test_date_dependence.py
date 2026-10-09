"""Date-overlap sensitivity: the robust intervals widen exactly where dates overlap."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from backtest.date_dependence import calendar_acf, hac_t_ci, iid_ci, moving_block_ci, sensitivity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _dates(n, start=date(2026, 9, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def _ma(n, q, seed=0):
    """MA(q) with equal weights: the overlap of (q+1)-day forward windows."""
    e = np.random.default_rng(seed).normal(size=n + q)
    return np.convolve(e, np.ones(q + 1) / (q + 1), mode="valid")


def test_calendar_acf_skips_gaps_rather_than_pairing_neighbours():
    # Four two-day runs separated by gaps. Lag 1 pairs only within a run, where
    # the values are equal, so it reads +1; a positional lag would also pair
    # each run's end with the next run's start, where the sign flips.
    d, x = [], []
    for run, v in enumerate([1.0, -1.0, 1.0, -1.0]):
        start = date(2026, 9, 1) + timedelta(days=10 * run)
        d += [start, start + timedelta(days=1)]
        x += [v, v]
    acf1, n1 = calendar_acf(d, x, 1)
    assert n1 == 4
    assert acf1 == pytest.approx(1.0)
    assert calendar_acf(d, x, 2) == (None, 0)


def test_iid_matches_the_frozen_instruments_bootstrap():
    """The h=7 read's `paired_date_diff` and this `iid_ci` are the same draw."""
    import outside_baseline as ob

    rng = np.random.default_rng(3)
    d = _dates(20)
    df = pd.DataFrame(
        [{"forecast_date": dd, "a": rng.normal(), "b": rng.normal()} for dd in d for _ in range(5)]
    )
    est, lo, hi = ob.paired_date_diff(df, "a", "b")
    per = df.assign(_d=df["a"] - df["b"]).groupby("forecast_date")["_d"].mean().to_numpy()
    assert iid_ci(per, 0.90, 2000, ob.RNG_SEED) == pytest.approx((lo, hi))
    assert per.mean() == pytest.approx(est)


def test_the_pid_bootstrap_is_reproduced_too():
    import measure_conformal_pid as pid

    rng = np.random.default_rng(5)
    a, b = rng.uniform(0.6, 0.95, 30), rng.uniform(0.6, 0.95, 30)
    _, lo, hi = pid.paired_bootstrap(a, b)
    diff = (np.abs(a - 0.8) - np.abs(b - 0.8)) * 100
    assert iid_ci(diff, 0.95, pid.N_BOOT, pid.BOOT_SEED) == pytest.approx((lo, hi))


def test_overlapping_dates_widen_the_robust_intervals():
    x = _ma(400, 6)
    iid = iid_ci(x, 0.90, 2000, 0)
    mbb = moving_block_ci(x, 7, 0.90, 2000, 0)
    hac = hac_t_ci(x, 6, 0.90)
    w = iid[1] - iid[0]
    assert (mbb[1] - mbb[0]) / w > 1.8
    assert (hac[1] - hac[0]) / w > 1.8


def test_independent_dates_leave_them_close_to_iid():
    x = np.random.default_rng(1).normal(size=400)
    iid = iid_ci(x, 0.90, 2000, 0)
    w = iid[1] - iid[0]
    for ci in (moving_block_ci(x, 7, 0.90, 2000, 0), hac_t_ci(x, 6, 0.90)):
        assert 0.75 < (ci[1] - ci[0]) / w < 1.3


def test_fragile_when_only_the_iid_interval_clears_zero():
    x = _ma(20, 6, seed=11)
    x = x - x.mean()
    lo_centered, _ = iid_ci(x, 0.90, 2000, 0)
    x = x - lo_centered * 1.05  # shift so the iid lower bound sits just above zero
    s = sensitivity(_dates(20), x, horizon=7, level=0.90, n_boot=2000, seed=0)
    assert s["sign"]["iid"] == "above_zero"
    assert s["sign"]["moving_block"] == "spans_zero" and s["sign"]["hac"] == "spans_zero"
    assert s["fragile"] is True


def test_not_fragile_when_every_interval_agrees():
    x = np.random.default_rng(2).normal(loc=1.0, size=40)
    s = sensitivity(_dates(40), x, horizon=3, level=0.90, n_boot=2000, seed=0)
    assert set(s["sign"].values()) == {"above_zero"}
    assert s["fragile"] is False


def test_sensitivity_sorts_dates_and_refuses_tiny_panels():
    d = _dates(10)
    x = np.arange(10, dtype=float)
    a = sensitivity(d, x, 3, 0.9, 500, 0)
    b = sensitivity(d[::-1], x[::-1], 3, 0.9, 500, 0)
    assert a == b
    with pytest.raises(ValueError):
        sensitivity(d[:2], x[:2], 3, 0.9, 500, 0)


# ------------------------------------------------------------------ the runner


def test_the_pid_row_is_refused_before_its_read(capsys):
    import date_dependence_sensitivity as dds

    assert dds.main(["pid"], today=date(2026, 10, 22)) == 2
    assert "not read before 2026-10-23" in capsys.readouterr().out


def test_an_outside_baseline_window_is_refused_until_it_has_20_dates(monkeypatch, tmp_path, capsys):
    import date_dependence_sensitivity as dds
    import outside_baseline as ob

    panel = pd.DataFrame(
        {
            "item_slug": "a",
            "horizon_days": 7,
            "forecast_date": [date(2026, 9, 20) + timedelta(days=i) for i in range(12)],
        }
    )
    monkeypatch.setattr(ob, "load_panel", lambda _d: panel)
    monkeypatch.setattr(ob, "build_baselines", lambda *a, **k: pytest.fail("peeked: built baselines"))
    argv = ["outside-baseline", "--archive-dir", str(tmp_path), "--horizon", "7", "--since", "2026-09-20"]
    assert dds.main(argv) == 2
    assert "would be a peek" in capsys.readouterr().out
