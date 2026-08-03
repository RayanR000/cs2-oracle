"""What the product is willing to put in front of a user.

The one threshold here exists because sub-$1 items are ~72% of the forecast
universe and one cent there is a 20% move, so their up/flat/down label is
dominated by tick quantisation rather than by anything the model knows
(``backtest/scoring.py:209-213``). Ranking those items by percentage move —
which every opportunities surface did — promotes rounding artifacts to the top
of the list.

The floor is deliberately equal to the lower bound of ``HEADLINE_MIN_TIER``, so
the population the product shows is the population the headline accuracy figure
describes. ``tests/test_serving_policy.py`` fails if the two ever diverge.

It is a convention, not a derivation: the sharp break in the tier evidence is
nearer $0.50 (27.6% actual-flat below it against ~1% above). Matching the
headline is what earns the number its meaning.
"""
from __future__ import annotations

from sqlalchemy.sql.elements import ColumnElement

MIN_SERVED_PRICE_USD = 1.0


def price_floor_clause(column) -> ColumnElement:
    """SQL-side floor, for query filters."""
    return column >= MIN_SERVED_PRICE_USD


def meets_price_floor(price: float | None) -> bool:
    """Python-side floor, for rows already in memory."""
    if price is None:
        return False
    return price >= MIN_SERVED_PRICE_USD
