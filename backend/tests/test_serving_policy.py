"""The served universe and the measured universe must stay the same population.

The headline accuracy number is computed over price_tier >= HEADLINE_MIN_TIER
(scoring.py), while the API surfaces had no price floor at all — 84% of the
forecasts they ranked were sub-$1 items where one cent is a 20% move. The
number quoted and the list displayed described different populations. These
tests pin the floor to the headline tier so the two cannot drift apart again.
"""

from __future__ import annotations

from api.serving_policy import (
    MIN_SERVED_PRICE_USD,
    meets_price_floor,
    price_floor_clause,
    tradeability,
)
from backtest.friction import actionable_threshold
from backtest.scoring import HEADLINE_MIN_TIER, price_tier
from database import ItemForecast
from sqlalchemy.dialects import postgresql


def _sql(clause) -> str:
    return str(
        clause.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


class TestFloorMatchesHeadlineTier:
    def test_floor_is_the_lower_bound_of_the_headline_tier(self):
        """If either constant moves alone, this fails — that is the point."""
        assert price_tier(MIN_SERVED_PRICE_USD) == HEADLINE_MIN_TIER

    def test_just_below_the_floor_is_outside_the_headline_tier(self):
        assert price_tier(MIN_SERVED_PRICE_USD - 0.01) < HEADLINE_MIN_TIER


class TestPriceFloorClause:
    def test_clause_compiles_to_a_greater_or_equal_comparison(self):
        sql = _sql(price_floor_clause(ItemForecast.current_price))
        assert "item_forecasts.current_price >= 1.0" in sql


class TestTradeability:
    """Sub-$1 items are served but flagged not economically tradeable: the
    35.5% sub-$1 spread plus the round trip dwarfs any few-percent forecast
    edge. The flag carries the estimated round-trip cost so the claim is legible
    rather than a bare boolean.
    """

    def test_sub_dollar_item_is_not_tradeable(self):
        assert tradeability(0.09).tradeable is False

    def test_at_the_floor_is_tradeable(self):
        assert tradeability(MIN_SERVED_PRICE_USD).tradeable is True

    def test_above_the_floor_is_tradeable(self):
        assert tradeability(5.0).tradeable is True

    def test_cost_is_round_trip_plus_the_item_tier_spread(self):
        # tier 0 at the cheapest venue: 2.0% round trip + 35.5% spread = 37.5%
        assert tradeability(0.09).est_roundtrip_cost_pct == 37.5

    def test_cost_tracks_the_price_tier(self):
        for price in (0.09, 1.0, 5.0, 250.0):
            expected = round(actionable_threshold(price_tier(price)) * 100, 1)
            assert tradeability(price).est_roundtrip_cost_pct == expected

    def test_none_price_is_not_tradeable_with_no_cost(self):
        result = tradeability(None)
        assert result.tradeable is False
        assert result.est_roundtrip_cost_pct is None


class TestMeetsPriceFloor:
    def test_at_the_floor_passes(self):
        assert meets_price_floor(1.0) is True

    def test_below_the_floor_fails(self):
        assert meets_price_floor(0.99) is False

    def test_none_fails_rather_than_raising(self):
        assert meets_price_floor(None) is False

    def test_zero_fails(self):
        assert meets_price_floor(0.0) is False
