"""What a predicted move has to clear before it implies a trade.

Two constants, no logic beyond adding them. Kept separate from ``scoring`` so
the numbers have one home and a citation: a threshold silently edited inside a
metric function is a number nobody can audit.

**Both tables are measurements, and the spread table is a measurement AT THE
WRONG CUTS.** The bid-ask spread was measured on BUFF163's paired
``starting_at`` / ``highest_order`` over n = 22,449 items in bands
(<$1, $1-10, $10-50, $50-500, $1000+), which do not line up with
``scoring.price_tier``'s cuts (1, 5, 20, 100, 1000). Each tier therefore borrows
the band whose geometric midpoint is nearest its own in log price, and
``SPREAD_SOURCE_BAND`` records which one it borrowed. Do not read these as
measured per tier.

The honest replacement is a per-item ``buff_spread_rel`` frozen onto
``forecast_outcomes`` at resolution time. It is not built yet: the bid feed
starts 2026-07-11, so it would be NULL on almost every stored outcome, and this
module reads nothing but its own constants by design.
"""

from __future__ import annotations

# Round trip to buy and sell one item at each venue, as a fraction of price.
# Steam's 16.1% is the fee stack; the cash venues are the seller commission.
# Sourced from docs/research/2026-08-07-cs2-forecasting-research.md section 11.
ROUND_TRIP_COST = {
    "csfloat": 0.020,
    "dmarket": 0.020,
    "skinport": 0.087,
    "steam": 0.161,
}

# The cheapest round trip, so the actionable metric is reported against the most
# favourable venue available. Any harsher venue only shrinks n_actionable, and a
# metric that fails at the best venue fails everywhere.
DEFAULT_VENUE = "csfloat"

# Median relative bid-ask spread, by price_tier. See the module docstring: the
# values are measured, the tier assignment is a nearest-band rule.
#
# Spread tightens MONOTONICALLY with price — 35.5% sub-$1 to 5.2% at $1000+ —
# which inverts retail intuition: expensive items are the liquid ones, and
# sub-$1 items are midpoints of a book nobody could transact in. This is also
# the reason a DA pooled across tiers is uninterpretable.
SPREAD_BY_TIER = {
    0: 0.355,  # < $1
    1: 0.211,  # $1 - 5
    2: 0.173,  # $5 - 20
    3: 0.173,  # $20 - 100
    4: 0.108,  # $100 - 1000
    5: 0.052,  # >= $1000
}

# Which measured band each tier's spread came from. Present so the borrowing is
# visible in the data rather than buried in a comment, and so a test can pin the
# mapping rule independently of the values above.
SPREAD_SOURCE_BAND = {
    0: "<$1",
    1: "$1-10",
    2: "$10-50",
    3: "$10-50",
    4: "$50-500",
    5: "$1000+",
}


def actionable_threshold(price_tier: int, venue: str = DEFAULT_VENUE) -> float:
    """The ``|r_hat|`` a forecast must exceed to imply a trade, as a fraction.

    ``RT_venue + s_i``: the round trip plus the spread you cross to take it. At
    CSFloat this runs 37.5% at tier 0 down to 7.2% at tier 5 — so even the most
    liquid cohort at the cheapest venue needs a >7% predicted move.

    Raises KeyError on an unknown venue or tier rather than falling back to a
    default. A silent fallback here would report a threshold nobody chose, and
    every actionable number downstream would inherit it.
    """
    return ROUND_TRIP_COST[venue] + SPREAD_BY_TIER[price_tier]
