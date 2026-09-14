"""A served PredictionOut carries the tradeability of its own current_price.

Every item is served a forecast regardless of price; the sub-$1 tier is flagged
not economically tradeable rather than withheld. Deriving the fields on the
schema keeps every route that builds a PredictionOut correct by construction.
"""

from __future__ import annotations

from api.schemas import PredictionOut
from api.serving_policy import tradeability


def _pred(price: float) -> PredictionOut:
    return PredictionOut(
        item_id=1,
        item_name="Test Item",
        current_price=price,
        forecast_low=price * 0.9,
        forecast_mid=price,
        forecast_high=price * 1.1,
        forecast_period="7_days",
        trend_direction="neutral",
    )


class TestPredictionCarriesTradeability:
    def test_sub_dollar_forecast_is_flagged_not_tradeable(self):
        assert _pred(0.09).tradeable is False

    def test_dollar_and_above_forecast_is_tradeable(self):
        assert _pred(5.0).tradeable is True

    def test_est_cost_matches_the_serving_policy_helper(self):
        assert _pred(0.09).est_roundtrip_cost_pct == tradeability(0.09).est_roundtrip_cost_pct
        assert _pred(5.0).est_roundtrip_cost_pct == tradeability(5.0).est_roundtrip_cost_pct
