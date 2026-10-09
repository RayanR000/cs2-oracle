"""Synthetic tests for the direction-reopening instrument.

No DB and no served data: the prereg (`docs/research/2026-10-08-direction-reopening-preregistration.md`)
forbids scoring any forecast date >= 2026-10-05 before a horizon has 20 usable dates, so everything
here runs on constructed panels.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest
from backtest.directional_test import pesaran_timmermann
from scripts import measure_direction_reopen as mdr

START = date(2026, 10, 5)


def _rows(d, pdad, pdau, puad, puau):
    """One forecast date's records from the 2x2 of (predicted, actual) counts.

    pdad = predicted down / actual down, pdau = predicted down / actual up, and so on."""
    out = []
    for pred, act, n in (("down", "down", pdad), ("down", "up", pdau), ("up", "down", puad), ("up", "up", puau)):
        out += [
            {
                "forecast_date": d,
                "predicted_direction": pred,
                "actual_direction": act,
                "direction_correct": pred == act,
            }
            for _ in range(n)
        ]
    return out


def _dates(n, start=START):
    return [start + timedelta(days=i) for i in range(n)]


def _panel(counts_by_date):
    """counts_by_date: {date: (pdad, pdau, puad, puau)} -> flat record list."""
    return [r for d, c in counts_by_date.items() for r in _rows(d, *c)]


# A date where the call is informative AND beats always-down: actual 70% down, DA 90%.
GOOD = [(26, 2, 2, 10), (25, 3, 3, 9), (27, 1, 1, 11), (26, 2, 2, 10)]
# Informative but loses to always-down: actual 70% down, the model says "down" on only a few rows.
INFORMATIVE_BUT_LOSES = [(8, 0, 20, 12), (7, 0, 21, 12), (9, 0, 19, 12), (8, 0, 20, 12)]
# Anti-informative: the mirror of GOOD.
PERVERSE = [(2, 10, 26, 2), (3, 9, 25, 3), (1, 11, 27, 1), (2, 10, 26, 2)]


def _cycle(patterns, n_dates=20, start=START):
    return {d: patterns[i % len(patterns)] for i, d in enumerate(_dates(n_dates, start))}


# ------------------------------------------------------------------ window


def test_window_excludes_dates_before_the_start_boundary():
    recs = _panel({date(2026, 10, 4): GOOD[0], **_cycle(GOOD, 20)})
    window, dates = mdr.select_window(recs)
    assert date(2026, 10, 4) not in dates
    assert dates[0] == START
    assert all(r["forecast_date"] >= START for r in window)


def test_window_is_the_first_twenty_usable_dates_and_ignores_later_ones():
    recs = _panel(_cycle(GOOD, 25))
    window, dates = mdr.select_window(recs)
    assert len(dates) == 20
    assert dates == _dates(20)
    assert {r["forecast_date"] for r in window} == set(dates)


def test_a_date_under_the_row_floor_is_skipped_and_does_not_count():
    thin = _rows(START + timedelta(days=1), 5, 0, 0, 5)  # 10 rows < PT_MIN_ROWS_PER_DATE
    recs = _panel({START: GOOD[0]}) + thin + _panel(_cycle(GOOD, 25, START + timedelta(days=2)))
    _, dates = mdr.select_window(recs)
    assert START + timedelta(days=1) not in dates
    assert len(dates) == 20


def test_fewer_than_twenty_usable_dates_returns_what_exists():
    window, dates = mdr.select_window(_panel(_cycle(GOOD, 12)))
    assert len(dates) == 12
    assert len({r["forecast_date"] for r in window}) == 12


def test_iso_string_dates_are_read_like_dates():
    recs = _panel(_cycle(GOOD, 20))
    for r in recs:
        r["forecast_date"] = r["forecast_date"].isoformat()
    _, dates = mdr.select_window(recs)
    assert dates == _dates(20)


# ------------------------------------------------------------------ frozen constants


def test_no_drift_on_the_frozen_scorer_constants():
    assert mdr.constants_drift() == []


def test_a_changed_scorer_constant_is_reported_by_name(monkeypatch):
    monkeypatch.setitem(mdr.FROZEN_CONSTANTS, "PT_T_HURDLE", 99.0)
    assert mdr.constants_drift() == ["PT_T_HURDLE"]


# ------------------------------------------------------------------ DA vs the runnable baseline


def test_da_vs_down_reports_the_gap_in_points():
    recs = _panel(_cycle(INFORMATIVE_BUT_LOSES, 20))  # 20/40 hits, 28/40 actually down
    out = mdr.da_vs_down(recs)
    assert out["da_pp"] == pytest.approx(50.0)
    assert out["down_pp"] == pytest.approx(70.0)
    assert out["diff_pp"] == pytest.approx(-20.0)


def test_da_vs_down_bootstrap_brackets_a_stable_gap():
    out = mdr.da_vs_down(_panel(_cycle(GOOD, 20)))
    assert out["ci_lo"] <= out["diff_pp"] <= out["ci_hi"]
    assert out["ci_lo"] > 0  # GOOD beats always-down on every date


def _varied(n_dates=20):
    """Per-date hit counts spread widely, so the bootstrap CI sits on a fine lattice and a
    different seed (or none) cannot land on the same bounds by coincidence."""
    counts = {}
    for i, d in enumerate(_dates(n_dates)):
        a, b, c = 10 + (i * 5) % 17, (i * 3) % 7, (i * 2) % 5
        counts[d] = (a, b, c, 40 - a - b - c)
    return _panel(counts)


def test_da_vs_down_bootstrap_is_reproducible_for_a_seed_and_differs_across_seeds():
    recs = _varied()
    first, again = mdr.da_vs_down(recs, seed=1), mdr.da_vs_down(recs, seed=1)
    other = mdr.da_vs_down(recs, seed=2)
    assert (first["ci_lo"], first["ci_hi"]) == (again["ci_lo"], again["ci_hi"])
    assert (first["ci_lo"], first["ci_hi"]) != (other["ci_lo"], other["ci_hi"])


def test_the_default_bootstrap_seed_is_the_prereg_seed():
    recs = _varied()
    assert mdr.da_vs_down(recs) == mdr.da_vs_down(recs, seed=0)


# ------------------------------------------------------------------ drop-one-date guard


def test_drop_one_removes_the_largest_excess_date_and_rescores_the_rest():
    counts = _cycle(GOOD, 20)
    big = START + timedelta(days=7)
    counts[big] = (28, 0, 0, 12)  # a perfect date: the largest excess in the window
    recs = _panel(counts)
    out = mdr.drop_one_date(recs)
    assert out["dropped_date"] == big
    rest = [r for r in recs if r["forecast_date"] != big]
    expected = pesaran_timmermann(rest, 20)
    assert out["pt_n_dates"] == 19
    assert out["t"] == pytest.approx(expected["pt_t_stat"])
    assert out["excess_pp"] == pytest.approx(expected["pt_excess_pp"])


# ------------------------------------------------------------------ the verdict


def _pt(excess, t):
    return {"pt_excess_pp": excess, "pt_t_stat": t, "pt_n_dates": 20}


def _da(diff):
    return {"diff_pp": diff}


def _drop(excess, t):
    return {"excess_pp": excess, "t": t}


@pytest.mark.parametrize(
    ("pt", "da", "drop", "expected"),
    [
        (_pt(1.4, 3.0), _da(0.0), _drop(1.2, 2.0), "PASS"),  # every bar met exactly at its boundary
        (_pt(1.4, 3.5), _da(5.0), _drop(1.2, 2.5), "PASS"),
        (_pt(1.4, 3.5), _da(-0.1), _drop(1.2, 2.5), "INFORMATION_WITHOUT_USABLE_CALL"),
        (_pt(1.4, 3.5), _da(5.0), _drop(0.2, 1.9), "UNRESOLVED"),  # one date carries it
        (_pt(1.4, 3.5), _da(5.0), _drop(-0.2, 2.5), "UNRESOLVED"),  # sign flips without that date
        (_pt(1.0, 2.99), _da(5.0), _drop(1.0, 2.5), "UNRESOLVED"),  # just under the hurdle
        (_pt(0.9, 2.0), _da(5.0), _drop(0.9, 2.0), "UNRESOLVED"),  # 2.0 is the floor of unresolved
        (_pt(0.5, 1.99), _da(5.0), _drop(0.5, 2.5), "NULL"),
        (_pt(-1.5, -4.0), _da(5.0), _drop(-1.5, -3.0), "NULL"),
        (_pt(0.0, 0.0), _da(5.0), _drop(0.0, 0.0), "NULL"),
        (_pt(None, None), _da(5.0), _drop(None, None), "VOID"),  # degenerate: no variation to test
    ],
)
def test_verdict_follows_the_prereg_table(pt, da, drop, expected):
    assert mdr.judge(pt, da, drop) == expected


def test_achieved_mde_is_the_excess_needed_to_clear_the_hurdle():
    assert mdr.achieved_mde_pp(_pt(1.0, 2.0)) == pytest.approx(1.5)  # se 0.5pp, hurdle 3.0
    assert mdr.achieved_mde_pp(_pt(None, None)) is None
    assert mdr.achieved_mde_pp(_pt(0.0, 0.0)) is None


# ------------------------------------------------------------------ evaluate one horizon


def test_a_horizon_without_twenty_dates_computes_no_statistic():
    out = mdr.evaluate_horizon(_panel(_cycle(GOOD, 19)), 3)
    assert out == {"horizon": 3, "status": "NOT_READY", "usable_dates": 19, "need": 20}


def test_ready_horizon_that_beats_always_down_passes():
    out = mdr.evaluate_horizon(_panel(_cycle(GOOD, 25)), 3)
    assert out["status"] == "READ"
    assert out["verdict"] == "PASS"
    assert out["pt"]["pt_n_dates"] == 20
    assert out["da"]["diff_pp"] > 0
    assert out["window"] == (START, START + timedelta(days=19))


def test_ready_horizon_with_information_but_a_losing_call_stays_withheld():
    out = mdr.evaluate_horizon(_panel(_cycle(INFORMATIVE_BUT_LOSES, 20)), 7)
    assert out["verdict"] == "INFORMATION_WITHOUT_USABLE_CALL"
    assert out["da"]["diff_pp"] < 0


def test_ready_horizon_with_a_perverse_call_is_null():
    out = mdr.evaluate_horizon(_panel(_cycle(PERVERSE, 20)), 3)
    assert out["verdict"] == "NULL"


def test_report_only_horizons_get_numbers_but_never_a_verdict():
    out = mdr.evaluate_horizon(_panel(_cycle(GOOD, 20)), 14)
    assert out["status"] == "READ"
    assert out["verdict"] == "REPORT_ONLY"
    assert out["pt"]["pt_t_stat"] is not None


# ------------------------------------------------------------------ main(): the refusal guard


def _loader(by_horizon):
    return lambda: by_horizon


def test_main_refuses_and_prints_no_statistic_while_no_horizon_is_ready(caplog):
    data = {3: _panel(_cycle(GOOD, 19)), 7: _panel(_cycle(GOOD, 11)), 14: [], 30: []}
    with caplog.at_level(logging.INFO):
        code = mdr.main([], records_loader=_loader(data))
    text = caplog.text
    assert code == 3
    assert "h=3" in text and "19/20" in text
    assert "h=7" in text and "11/20" in text
    # Nothing from the outcomes may leak: no PT, DA, or verdict.
    for leak in ("excess", "t=", "DA", "PASS", "NULL", "UNRESOLVED"):
        assert leak not in text


def test_main_reads_a_ready_horizon_and_refuses_only_the_one_that_is_not(caplog):
    # The prereg reads per horizon: h=3 is due ~10-29, h=7 ~11-02.
    data = {3: _panel(_cycle(GOOD, 20)), 7: _panel(_cycle(GOOD, 11)), 14: [], 30: []}
    with caplog.at_level(logging.INFO):
        code = mdr.main([], records_loader=_loader(data))
    h7 = [line for line in caplog.text.splitlines() if "h=7" in line]
    assert code == 3  # a primary horizon is still outstanding
    assert "PASS" in caplog.text  # h=3 was read
    assert h7 and all(("PASS" not in line and "excess" not in line) for line in h7)
    assert any("11/20" in line for line in h7)


def test_main_reads_once_both_primary_horizons_are_ready(caplog):
    data = {3: _panel(_cycle(GOOD, 20)), 7: _panel(_cycle(INFORMATIVE_BUT_LOSES, 20)), 14: [], 30: []}
    with caplog.at_level(logging.INFO):
        code = mdr.main([], records_loader=_loader(data))
    assert code == 0
    assert "PASS" in caplog.text
    assert "INFORMATION_WITHOUT_USABLE_CALL" in caplog.text


def test_main_is_void_when_a_scorer_constant_drifted(monkeypatch, caplog):
    monkeypatch.setitem(mdr.FROZEN_CONSTANTS, "FLAT_TOLERANCE", 0.01)
    data = {3: _panel(_cycle(GOOD, 20)), 7: _panel(_cycle(GOOD, 20)), 14: [], 30: []}
    with caplog.at_level(logging.INFO):
        code = mdr.main([], records_loader=_loader(data))
    assert code == 4
    assert "FLAT_TOLERANCE" in caplog.text
    assert "PASS" not in caplog.text
