"""The ranking floor is the lower bound of the headline tier (moved from test_serving_policy.py
when the API was retired, Phase 4 PR 5)."""

from backtest.scoring import HEADLINE_MIN_TIER, MIN_SERVED_PRICE_USD, price_tier


def test_floor_is_the_lower_bound_of_the_headline_tier():
    assert price_tier(MIN_SERVED_PRICE_USD) == HEADLINE_MIN_TIER


def test_just_below_the_floor_is_outside_the_headline_tier():
    assert price_tier(MIN_SERVED_PRICE_USD - 0.01) < HEADLINE_MIN_TIER
