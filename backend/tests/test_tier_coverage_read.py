"""The price-tier coverage read (docs/research/2026-11-22-tier-conditional-coverage-preregistration.md).

Synthetic panels with a known answer: a real gradient must PASS, a null must KILL, and a gradient
that is only the width tilt must come out NOT_NEW. The harness has to be right before the real
panel is read once.
"""

from __future__ import annotations

from collections import namedtuple
from datetime import timedelta

import numpy as np
import pytest
from scripts import tier_coverage_read as tcr

Row = namedtuple("Row", ["forecast_date", "current_price", "lo", "mid", "hi", "in_interval"])
PRICE = {1: 2.0, 2: 10.0, 3: 50.0}


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(tcr, "N_BOOT", 300)
    monkeypatch.setattr(tcr, "N_PLACEBO", 60)


def _panel(p_hit, width_of_tier, *, n_dates=20, per_cell=150, start=tcr.WINDOW_START, seed=1):
    """Rows for n_dates consecutive dates. p_hit(tier, width) -> P(in interval); width_of_tier(tier, rng) -> width."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_dates):
        d = start + timedelta(days=k)
        for tier in (1, 2, 3):
            for _ in range(per_cell):
                w = width_of_tier(tier, rng)
                hit = rng.random() < p_hit(tier, w)
                rows.append(Row(d, PRICE[tier], 10 * (1 - w / 2), 10.0, 10 * (1 + w / 2), hit))
    return rows


def _reader(rows):
    return lambda sql: rows


def _uniform_width(tier, rng):
    return rng.uniform(0.10, 0.30)


def test_a_real_tier_gradient_passes():
    rows = _panel(lambda t, w: {1: 0.76, 2: 0.85, 3: 0.94}[t], _uniform_width)
    res = tcr.read_horizon(_reader(rows), 3)
    assert not res["refused"] and res["dates"] == 20
    assert res["g"] == pytest.approx(18.0, abs=4.0)
    assert res["gw"] >= 4.0
    assert res["verdict"] == "PASS"


def test_no_gradient_is_killed():
    rows = _panel(lambda t, w: 0.82, _uniform_width)
    res = tcr.read_horizon(_reader(rows), 7)
    assert res["verdict"] == "KILL"
    assert res["g_ci"][0] <= 0 <= res["g_ci"][1]


def test_a_gradient_that_is_only_the_width_tilt_is_not_new():
    """Tier 3 has wider bands and coverage rises with width; within a width tercile tier does nothing."""

    def width(tier, rng):
        return (
            rng.uniform(0.05, 0.25) if tier == 1 else rng.uniform(0.25, 0.45) if tier == 3 else rng.uniform(0.15, 0.35)
        )

    rows = _panel(lambda t, w: 0.70 + 0.5 * (w - 0.05), width)
    res = tcr.read_horizon(_reader(rows), 3)
    assert res["g"] >= tcr.BAR_G_PP  # the raw gradient is big
    assert res["verdict"] == "NOT_NEW"


def test_h14_is_reported_not_judged():
    rows = _panel(lambda t, w: {1: 0.76, 2: 0.85, 3: 0.94}[t], _uniform_width)
    assert tcr.read_horizon(_reader(rows), 14)["verdict"] == "REPORT_ONLY"


def test_h30_is_descriptive_only():
    rows = _panel(lambda t, w: 0.9, _uniform_width)
    res = tcr.read_horizon(_reader(rows), 30)
    assert res["verdict"] == "REPORT_ONLY" and "g" not in res
    assert set(res["table"].index) == {1, 2, 3}


def test_a_short_window_is_refused_and_computes_nothing():
    rows = _panel(lambda t, w: 0.9, _uniform_width, n_dates=19)
    res = tcr.read_horizon(_reader(rows), 3)
    assert res == {"horizon": 3, "refused": True, "dates": 19, "need": 20}
    assert "REFUSED" in tcr.render(res)


def test_dates_before_the_window_never_count():
    pre = _panel(lambda t, w: 0.5, _uniform_width, n_dates=30, start=tcr.WINDOW_START - timedelta(days=30), seed=2)
    assert tcr.read_horizon(_reader(pre), 3)["refused"] is True


def test_the_window_is_the_first_20_dates_so_later_dates_cannot_move_it():
    first = _panel(lambda t, w: {1: 0.76, 2: 0.85, 3: 0.94}[t], _uniform_width)
    later = _panel(lambda t, w: 0.1, _uniform_width, n_dates=10, start=tcr.WINDOW_START + timedelta(days=20), seed=3)
    a = tcr.read_horizon(_reader(first), 3)
    b = tcr.read_horizon(_reader(first + later), 3)
    assert b["dates"] == 20
    assert (a["g"], a["gw"], a["verdict"]) == (b["g"], b["gw"], b["verdict"])


def test_rows_under_a_dollar_or_with_no_outcome_are_dropped_by_the_query_contract():
    """The SQL filters them; the reader contract is that it is asked for only scoreable served rows."""
    seen = []

    def read(sql):
        seen.append(sql)
        return []

    tcr.read_horizon(read, 3)
    assert "in_interval IS NOT NULL" in seen[0] and "current_price >= 1" in seen[0]
    assert "horizon_days = 3" in seen[0]


def test_a_non_null_placebo_voids_the_run():
    skewed = np.full(60, 5.0)
    assert tcr.verdict(18.0, 12.0, 24.0, 15.0, 8.0, skewed) == "VOID"


def test_verdict_bars():
    null = np.zeros(60) + np.linspace(-1, 1, 60)
    assert tcr.verdict(18.0, 12.0, 24.0, 15.0, 8.0, null) == "PASS"
    assert tcr.verdict(7.9, 5.0, 11.0, 6.0, 2.0, null) == "KILL"  # under +8pp
    assert tcr.verdict(18.0, -1.0, 24.0, 15.0, 8.0, null) == "KILL"  # interval spans zero
    assert tcr.verdict(10.0, 2.0, 18.0, 6.0, 2.0, null) == "UNDERPOWERED"  # lower bound <= +3pp
    assert tcr.verdict(18.0, 12.0, 24.0, 2.0, 0.5, null) == "NOT_NEW"  # width control fails
    assert tcr.verdict(18.0, 12.0, 24.0, float("nan"), float("nan"), null) == "NOT_NEW"


def test_overall_needs_both_primary_horizons():
    assert tcr.overall({3: "PASS", 7: "PASS"}) == "PASS"
    assert tcr.overall({3: "PASS", 7: "KILL"}) == "KILL"
    assert tcr.overall({3: "PASS", 7: "UNDERPOWERED"}) == "UNDERPOWERED"
    assert tcr.overall({3: "PASS", 7: "NOT_NEW"}) == "NOT_NEW"
    assert tcr.overall({3: "PASS"}) == "INCOMPLETE"
    assert tcr.overall({3: "VOID", 7: "PASS"}) == "VOID"
