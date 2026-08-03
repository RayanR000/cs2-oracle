"""The served universe and the measured universe must stay the same population.

The headline accuracy number is computed over price_tier >= HEADLINE_MIN_TIER
(scoring.py), while the API surfaces had no price floor at all — 84% of the
forecasts they ranked were sub-$1 items where one cent is a 20% move. The
number quoted and the list displayed described different populations. These
tests pin the floor to the headline tier so the two cannot drift apart again.
"""
from __future__ import annotations

from sqlalchemy.dialects import postgresql

from api.serving_policy import (
    MIN_SERVED_PRICE_USD,
    meets_price_floor,
    price_floor_clause,
)
from backtest.scoring import HEADLINE_MIN_TIER, price_tier
from database import ItemForecast


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


class TestMeetsPriceFloor:
    def test_at_the_floor_passes(self):
        assert meets_price_floor(1.0) is True

    def test_below_the_floor_fails(self):
        assert meets_price_floor(0.99) is False

    def test_none_fails_rather_than_raising(self):
        assert meets_price_floor(None) is False

    def test_zero_fails(self):
        assert meets_price_floor(0.0) is False
