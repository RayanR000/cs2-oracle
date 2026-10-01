"""Synthetic tests for the conformal-PID served-panel instrument.

No DB and no served data: the prereg forbids reading post-09-15 h=3 outcomes before
2026-10-23, so everything here runs on constructed panels.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scripts import measure_conformal_pid as mcp

D0 = date(2026, 9, 7)


def _frame(dates, r_by_date, mult=1.0, width=0.1, **extra):
    """A served-outcome frame whose rows have base-band score r (mid 100, half 10*mult)."""
    rows = []
    for d, rs in zip(dates, r_by_date):
        half = 100 * width * mult
        for r in rs:
            rows.append(
                {
                    "forecast_date": d,
                    "horizon_days": 3,
                    "price_tier": 1,
                    "predicted_price_low": 100 - half,
                    "predicted_price_mid": 100.0,
                    "predicted_price_high": 100 + half,
                    "actual_price": 100 + r * 100 * width,  # r_stored = r / mult, r_base = r
                    "band_multiplier": mult,
                    **extra,
                }
            )
    return pd.DataFrame(rows)


def _panel(scales, n=400, seed=0):
    """One date per scale; r_base ~ |N(0,1)| * scale, so m* = 1.2816 * scale covers 80%."""
    rng = np.random.default_rng(seed)
    dates = [D0 + timedelta(days=i) for i in range(len(scales))]
    rs = [np.abs(rng.standard_normal(n)) * s for s in scales]
    return mcp.build_panel(_frame(dates, rs), window=(D0, D0 + timedelta(days=len(scales))))


# ------------------------------------------------------------------ panel / replay


def test_r_base_undoes_the_served_multiplier():
    """A band served at m=0.5 is half as wide; r_base must still be the base-band score."""
    df = _frame([D0], [[0.3, 0.9]], mult=0.5)
    p = mcp.build_panel(df, window=(D0, D0))
    np.testing.assert_allclose(p.r_base[0], [0.3, 0.9])
    np.testing.assert_allclose(p.width[0], [0.1, 0.1])  # base relative half-width


def test_sub_dollar_rows_and_other_horizons_are_dropped():
    df = pd.concat(
        [
            _frame([D0], [[0.1]]),
            _frame([D0], [[0.2]]).assign(price_tier=0),
            _frame([D0], [[0.3]]).assign(horizon_days=7),
        ]
    )
    p = mcp.build_panel(df, window=(D0, D0))
    np.testing.assert_allclose(p.r_base[0], [0.1])


def test_resolution_lag_is_horizon_plus_one():
    assert mcp.resolved_by(D0) == D0 + timedelta(days=4)


# ------------------------------------------------------------------ controllers


def test_p_step_is_eta_times_miss_gap():
    ctl = mcp.PID(integrate=False)
    ctl.update(0.30)  # 10pp over alpha
    assert ctl.served == pytest.approx(1.0 + 0.5 * 0.10)
    ctl.update(0.10)
    assert ctl.served == pytest.approx(1.0)


def test_integrator_matches_the_reference_saturation_fn_log():
    """KI * tan(x * log(t+1) / (Csat * (t+1))), as in core/methods.py."""
    s, k = 2.0, 20
    assert mcp.saturation_log(s, k) == pytest.approx(0.3 * math.tan(2.0 * math.log(21) / 21))
    assert mcp.saturation_log(s, k, ki=0) == 0.0
    assert mcp.saturation_log(1e6, 1) == math.inf


def test_served_value_is_clamped_and_counted():
    p = _panel([5.0] * 30)  # every date misses at m <= 2
    arm = mcp.run_online(p, True, "QI")
    assert arm["m"].max() == pytest.approx(2.0)
    assert arm["clamp_hits"] > 0


def test_no_update_uses_an_unresolved_date():
    """The first 4 dates are served at 1.0 whatever they score: nothing has resolved yet."""
    p = _panel([3.0] * 10)
    arm = mcp.run_online(p, True, "QI")
    np.testing.assert_array_equal(arm["m"][:4], 1.0)
    assert arm["m"][4] > 1.0
    assert arm["schedule"][:5] == [0, 0, 0, 0, 1]


def test_each_err_is_scored_at_the_multiplier_served_that_date():
    """Closed loop: a date's miss rate is its band's, i.e. at the m served ON that date, not
    the controller's value when the outcome resolves."""
    p = _panel([0.9] * 15)  # m* ~ 1.15: m climbs from 1.0 without reaching the clamp
    arm = mcp.run_online(p, True, "QI")
    n = len(arm["errs"])
    assert n >= 5
    # the multiplier moved across the scored dates, so "served then" != "current"
    assert len(set(np.round(arm["m"][:n], 6))) > 1
    assert arm["m"].max() < mcp.CLAMP[1]
    for j, e in enumerate(arm["errs"]):
        assert e == pytest.approx(mcp.miss_rate(p.r_base[j], arm["m"][j]))


def test_placebo_is_a_permutation_on_the_same_schedule():
    p = _panel([0.5] * 10 + [1.5] * 10 + [0.5] * 10)
    qi = mcp.run_online(p, True, "QI")
    pl = mcp.run_placebo(p, qi)
    assert sorted(pl["errs"]) == pytest.approx(sorted(qi["errs"]))
    assert pl["errs"] != pytest.approx(qi["errs"])


def test_batch_waits_for_the_gate_then_refits_only_on_mondays():
    p = _panel([1.0] * 30)
    b = mcp.run_batch(p, min_dates=8)
    # 8 dates resolved first on D0+7+4 = D0+11.
    first = 11
    np.testing.assert_array_equal(b["m"][:first], 1.0)
    assert b["m"][first] != 1.0
    changes = [i for i in range(first + 1, 30) if b["m"][i] != b["m"][i - 1]]
    assert all(p.dates[i].weekday() == 0 for i in changes)


def test_batch_refit_is_the_production_estimator():
    """B's factor must equal factors_from_panel on the same resolved rows."""
    from models import served_recalibration

    rng = np.random.default_rng(3)
    dates = [D0 + timedelta(days=i) for i in range(12)]
    rs = [np.abs(rng.standard_normal(300)) * 0.6 for _ in dates]
    df = _frame(dates, rs, mult=0.5)
    p = mcp.build_panel(df, window=(D0, dates[-1]))
    b = mcp.run_batch(p, min_dates=8)
    expect = served_recalibration.factors_from_panel(df[df["forecast_date"] <= dates[7]], [3], min_dates=8)[3]
    assert b["m"][11] == pytest.approx(expect)


# ------------------------------------------------------------------ closed loop


def test_pid_tracks_a_regime_switch_that_the_batch_factor_lags():
    """Blocks of calm/volatile dates: an online arm should cut the date error vs F1 and B."""
    scales = ([0.5] * 12 + [1.5] * 12) * 2
    p = _panel(scales, seed=1)
    s = {
        k: mcp.score(p, a)
        for k, a in {"F1": mcp.run_fixed(p), "B": mcp.run_batch(p), "QI": mcp.run_online(p, True, "QI")}.items()
    }
    assert s["QI"]["primary_pp"] < s["F1"]["primary_pp"]
    assert s["QI"]["primary_pp"] < s["B"]["primary_pp"]


def test_pid_converges_toward_nominal_on_a_stationary_panel():
    p = _panel([0.6] * 60, seed=2)
    qi = mcp.run_online(p, True, "QI")
    tail = mcp.score(
        mcp.Panel(p.dates[30:], p.r_base[30:], p.width[30:], p.blend_delta[30:]),
        {"name": "t", "m": qi["m"][30:], "clamp_hits": 0},
    )
    assert tail["m1_pp"] == pytest.approx(80.0, abs=5.0)


# ------------------------------------------------------------------ scoring / judge


def test_decile_coverage_splits_rows_by_base_width():
    p = mcp.build_panel(
        pd.concat([_frame([D0], [[0.5] * 50], width=w) for w in np.linspace(0.05, 0.5, 10)]), window=(D0, D0)
    )
    cov = mcp.decile_coverage(p, np.array([1.0]))
    np.testing.assert_allclose(cov, 100.0)


def test_paired_bootstrap_sign_and_ci():
    a = np.full(30, 0.80)
    b = np.full(30, 0.90)
    mean, lo, hi = mcp.paired_bootstrap(a, b)
    assert mean == pytest.approx(-10.0)
    assert lo == pytest.approx(-10.0) and hi == pytest.approx(-10.0)


def _s(primary, p10=70.0, m1=80.0, mult=0.6, dec=None):
    return {
        "primary_pp": primary,
        "p10_pp": p10,
        "m1_pp": m1,
        "mean_mult": mult,
        "decile_cov_pp": np.full(10, 80.0) if dec is None else dec,
    }


def test_judge_pass_fail_unresolved():
    b, pl = _s(6.0, p10=60.0), _s(9.0)
    assert mcp.judge(_s(4.0, p10=65.0), b, pl, (-2.0, -3.0, -1.0))["verdict"] == "PASS"
    assert mcp.judge(_s(4.0, p10=65.0), b, pl, (-2.0, -4.0, 0.5))["verdict"] == "UNRESOLVED"
    assert mcp.judge(_s(5.5, p10=65.0), b, pl, (-0.5, -1.0, -0.1))["verdict"] == "FAIL"  # margin < 1pp
    assert mcp.judge(_s(4.0, p10=65.0, mult=0.7), b, pl, (-2.0, -3.0, -1.0))["verdict"] == "FAIL"  # width
    worse = np.full(10, 80.0)
    worse[3] = 74.0
    assert mcp.judge(_s(4.0, p10=65.0, dec=worse), b, pl, (-2.0, -3.0, -1.0))["verdict"] == "FAIL"


def test_void_reasons():
    b, pl = _s(6.0), _s(9.0)
    assert mcp.void_reasons(30, b, pl, "FAIL", ["FAIL", "FAIL"]) == []
    assert any("scoreable" in r for r in mcp.void_reasons(20, b, pl, "FAIL", ["FAIL", "FAIL"]))
    assert any("time variation" in r for r in mcp.void_reasons(30, _s(2.0), pl, "FAIL", ["FAIL", "FAIL"]))
    assert any("placebo" in r for r in mcp.void_reasons(30, b, _s(4.5), "FAIL", ["FAIL", "FAIL"]))
    assert any("blend" in r for r in mcp.void_reasons(30, b, pl, "PASS", ["PASS", "FAIL"]))


def test_blend_bound_perturbs_r_base_both_ways():
    df = _frame([D0], [[0.5]], prior_width_ratio=1.2, prior_multiplier=1.0)
    p = mcp.build_panel(df, window=(D0, D0))
    assert p.blend_delta[0][0] == pytest.approx(0.15 * 1.0 * 0.2)
    assert mcp.perturbed(p, +1).r_base[0][0] == pytest.approx(0.5 * 1.03)
    assert mcp.perturbed(p, -1).r_base[0][0] == pytest.approx(0.5 * 0.97)


def test_attach_priors_links_each_row_to_its_previous_forecast():
    fc = pd.DataFrame(
        {
            "item_id": [1, 1],
            "forecast_date": [D0, D0 + timedelta(days=1)],
            "predicted_price_low": [80.0, 90.0],
            "predicted_price_mid": [100.0, 100.0],
            "predicted_price_high": [120.0, 110.0],
            "band_multiplier": [None, None],
        }
    )
    out = pd.DataFrame(
        {
            "item_id": [1],
            "forecast_date": [D0 + timedelta(days=1)],
            "horizon_days": [3],
            "tier_price": [10.0],
            "predicted_price_low": [90.0],
            "predicted_price_mid": [100.0],
            "predicted_price_high": [110.0],
            "actual_price": [105.0],
            "band_multiplier": [None],
        }
    )
    df = mcp.attach_priors(out, fc)
    assert df["prior_width_ratio"].iloc[0] == pytest.approx(2.0)  # 0.2 / 0.1
    assert df["prior_multiplier"].iloc[0] == pytest.approx(1.0)


def test_evaluate_runs_end_to_end_on_a_synthetic_window():
    p = _panel(([0.5] * 10 + [1.2] * 10) * 2, seed=4)
    res = mcp.evaluate(p)
    assert set(res["scores"]) == {"F1", "B", "Q", "QI", "P_QI"}
    assert res["judgement"]["verdict"] in {"PASS", "FAIL", "UNRESOLVED"}
    assert "only" not in " ".join(res["void"]) or len(p.dates) < mcp.MIN_SCORE_DATES


def test_main_refuses_to_read_prod_before_the_read_date():
    assert mcp.main([], today=date(2026, 10, 22)) == 2


def test_load_prod_panel_runs_against_the_real_schema():
    """Both queries against tables built from the ORM. The prior-row read selected
    `predicted_price_*` from item_forecasts, whose columns are `price_low/mid/high`,
    so the single 10-23 read would have raised before scoring anything."""
    import datetime as dt

    from database import Base, ForecastOutcome, Item, ItemForecast
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(
        Item(
            id=1,
            item_id="AK-47 | Redline (Field-Tested)",
            name="AK-47 | Redline (Field-Tested)",
            type="skin",
            is_backfilled=1,
        )
    )
    for fid, fd in ((1, dt.date(2026, 9, 10)), (2, dt.date(2026, 9, 11))):
        s.add(
            ItemForecast(
                id=fid,
                item_id=1,
                forecast_date=fd,
                horizon_days=3,
                price_low=9.0,
                price_mid=10.0,
                price_high=11.0,
                current_price=10.0,
                band_multiplier=1.0,
            )
        )
        s.add(
            ForecastOutcome(
                forecast_id=fid,
                item_id=1,
                forecast_date=fd,
                horizon_days=3,
                target_date=fd + dt.timedelta(days=3),
                base_price=10.0,
                current_price=10.0,
                predicted_price_low=9.0,
                predicted_price_mid=10.0,
                predicted_price_high=11.0,
                actual_price=10.5,
                direction_correct=0,
                abs_error=0.5,
            )
        )
    s.commit()

    panel = mcp.load_prod_panel(s)

    assert len(panel) == 2
    assert panel.sort_values("forecast_date")["prior_width_ratio"].tolist() == [1.0, 1.0]
    engine.dispose()
