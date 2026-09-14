"""Friction constants: the round trip and the spread a call has to clear.

These are the numbers that decide whether a forecast implies a trade at all. The
spread table is a STATED APPROXIMATION — the measurement's bands do not line up
with ``price_tier``'s cuts — so these tests pin the mapping RULE, not just the
values, because a silently re-pointed band would move every actionable number
downstream without failing anything.
"""

from __future__ import annotations

import math

import pytest
from backtest.friction import (
    DEFAULT_VENUE,
    ROUND_TRIP_COST,
    SPREAD_BY_TIER,
    SPREAD_SOURCE_BAND,
    actionable_threshold,
)
from backtest.scoring import price_tier


def test_round_trip_costs_are_the_measured_venue_figures():
    assert ROUND_TRIP_COST["csfloat"] == 0.020
    assert ROUND_TRIP_COST["dmarket"] == 0.020
    assert ROUND_TRIP_COST["skinport"] == 0.087
    assert ROUND_TRIP_COST["steam"] == 0.161


def test_csfloat_is_the_default_because_it_is_the_cheapest_round_trip():
    assert DEFAULT_VENUE == "csfloat"
    assert ROUND_TRIP_COST[DEFAULT_VENUE] == min(ROUND_TRIP_COST.values())


def test_every_price_tier_has_a_spread_entry():
    """A new tier without a spread must fail here, not KeyError at score time."""
    tiers = {price_tier(p) for p in (0.5, 1, 5, 20, 100, 1000, 50_000)}
    assert tiers == set(SPREAD_BY_TIER), "price_tier and SPREAD_BY_TIER disagree"
    assert set(SPREAD_BY_TIER) == set(SPREAD_SOURCE_BAND)


def test_spread_is_monotone_decreasing_in_price():
    """35.5% sub-$1 down to 5.2% at $1000+. Expensive items are the liquid ones."""
    values = [SPREAD_BY_TIER[t] for t in sorted(SPREAD_BY_TIER)]
    assert values == sorted(values, reverse=True)


def test_spread_values_are_the_measured_band_medians():
    assert SPREAD_BY_TIER[0] == 0.355  # <$1
    assert SPREAD_BY_TIER[1] == 0.211  # $1-10
    assert SPREAD_BY_TIER[2] == 0.173  # $10-50
    assert SPREAD_BY_TIER[3] == 0.173  # $10-50
    assert SPREAD_BY_TIER[4] == 0.108  # $50-500
    assert SPREAD_BY_TIER[5] == 0.052  # $1000+


def test_each_tier_borrowed_its_nearest_source_band_by_log_geometric_midpoint():
    """The mapping RULE, reproduced independently of the table.

    The source bands are not the tier cuts, so each tier takes the band whose
    geometric midpoint is nearest its own in log price. Pinning this stops a
    later editor from re-pointing a band by eye.
    """
    source_mid = {
        "<$1": 0.5,  # open at the bottom
        "$1-10": math.sqrt(1 * 10),
        "$10-50": math.sqrt(10 * 50),
        "$50-500": math.sqrt(50 * 500),
        "$1000+": 2000.0,  # open at the top
    }
    tier_mid = {
        0: 0.5,
        1: math.sqrt(1 * 5),
        2: math.sqrt(5 * 20),
        3: math.sqrt(20 * 100),
        4: math.sqrt(100 * 1000),
        5: 2000.0,
    }
    for tier, mid in tier_mid.items():
        nearest = min(
            source_mid,
            key=lambda b: abs(math.log(source_mid[b]) - math.log(mid)),
        )
        assert SPREAD_SOURCE_BAND[tier] == nearest, f"tier {tier} borrowed the wrong band"


def test_actionable_threshold_is_the_round_trip_plus_the_tier_spread():
    assert actionable_threshold(5) == pytest.approx(0.020 + 0.052)
    assert actionable_threshold(1) == pytest.approx(0.020 + 0.211)
    assert actionable_threshold(5, venue="steam") == pytest.approx(0.161 + 0.052)


def test_the_cheapest_actionable_bar_is_still_over_seven_percent():
    """The point of the whole metric: even the most liquid tier at the cheapest
    venue needs a >7% predicted move before a call implies a trade."""
    assert min(actionable_threshold(t) for t in SPREAD_BY_TIER) > 0.07


def test_unknown_venue_raises_rather_than_defaulting():
    with pytest.raises(KeyError):
        actionable_threshold(1, venue="buff163")
