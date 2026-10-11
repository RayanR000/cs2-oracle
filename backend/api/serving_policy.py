"""What the product ranks, and how it labels what it serves.

Two rules. The price floor below, and the clean-anchor gate on the ranked
surfaces (``anchor_clean_clause`` / ``meets_anchor_gate``).

**The floor is a RANKING floor, not a serving floor.** A per-item forecast is
generated and served for every eligible item regardless of price (``predict``
filters only on history depth); the floor gates the ranked ``/opportunities``
and ``/trending`` surfaces and matches the headline tier. Served sub-$1 items
carry ``tradeability`` (below) so the forecast is honest about the fee wall.

The floor exists because sub-$1 items are ~72% of the forecast universe and one
cent there is a 20% move, so their up/flat/down label is dominated by tick
quantisation rather than by anything the model knows
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

from typing import NamedTuple

from backtest.friction import actionable_threshold
from backtest.scoring import price_tier
from sqlalchemy import or_
from sqlalchemy.sql.elements import ColumnElement

MIN_SERVED_PRICE_USD = 1.0

#: The horizons a model is trained and served for (mirrors
#: ``ItemForecaster.HORIZONS``, duplicated here so ``api/`` need not import the
#: heavy forecaster module). Any other horizon has no forecasts, so a ranking
#: request for it is rejected rather than served as a silently-empty list.
SERVED_HORIZONS = (3, 7, 14, 30)

#: How far behind the calendar a forecast's anchor day may sit before a ranked
#: surface treats it as stale. Forecasts are stamped with the day the band was
#: anchored on (the newest archived day), which lags `date.today()` because the
#: daily dump lands ~22:00 UTC; this window absorbs that lag plus a missed run or
#: two. Mirrors the archive's own staleness span (MAX_WINDOW_SPAN_DAYS).
MAX_ARCHIVE_LAG_DAYS = 7


class Tradeability(NamedTuple):
    """Whether acting on a served forecast can clear its trading costs.

    Every item is served a forecast (the floor above governs only the ranked
    surfaces and the headline tier), but sub-$1 items sit on a book with a
    35.5% median spread and a Steam fee that climbs past 60% at the cheapest
    prices, so any few-percent forecast edge is economically dead there. This
    labels that honestly instead of withholding the forecast.

    ``est_roundtrip_cost_pct`` is the round trip at the cheapest venue plus the
    item's tier spread (``backtest/friction.py::actionable_threshold``) — the
    move a forecast must beat to imply a trade — as a percentage. ``None`` when
    no price is known.
    """

    tradeable: bool
    est_roundtrip_cost_pct: float | None


def tradeability(price: float | None) -> Tradeability:
    """Tradeability of a served forecast at ``price``.

    Tradeable is the same $1 line as the ranked/headline floor: below it the
    cost hurdle dwarfs any edge. The cost estimate is reported for every price
    so the boolean is legible rather than a bare cutoff.
    """
    if price is None:
        return Tradeability(tradeable=False, est_roundtrip_cost_pct=None)
    cost_pct = round(actionable_threshold(price_tier(price)) * 100, 1)
    return Tradeability(
        tradeable=price >= MIN_SERVED_PRICE_USD,
        est_roundtrip_cost_pct=cost_pct,
    )


def price_floor_clause(column) -> ColumnElement:
    """SQL-side floor, for query filters."""
    return column >= MIN_SERVED_PRICE_USD


def meets_price_floor(price: float | None) -> bool:
    """Python-side floor, for rows already in memory."""
    if price is None:
        return False
    return price >= MIN_SERVED_PRICE_USD


def anchor_clean_clause(column) -> ColumnElement:
    """SQL-side clean-anchor gate, for query filters.

    Keeps NULL. See ``meets_anchor_gate`` for why, and note that a bare
    ``column.is_(True)`` would drop every row written before the column existed
    -- SQL's three-valued logic makes that the silent default.
    """
    return or_(column.is_(None), column.is_(True))


def meets_anchor_gate(anchor_clean: bool | None) -> bool:
    """Whether a forecast may appear on a RANKED surface.

    The model orders items well where the anchor quote equals its own local
    median and not at all where it does not: served rank IC +0.1321 / +0.1562 /
    +0.1747 at 3/7/14d on the clean cohort at 4 CI anchors of 4, against -0.2014
    (0 of 4) at h=3 on the rest and no distinguishable signal at the other
    horizons (`docs/changelog/2026-08-11-clean-anchor-confirmed-in-ci.md`).
    ``/opportunities`` ranks by predicted return, which is the ordering rank IC
    measures, so this is the surface that evidence covers.

    **It gates ranking only.** A per-item lookup still serves its forecast and
    carries the flag: the finding is about ordering, and refusing a forecast
    someone asked for by name goes past what was measured.

    **NULL passes.** It means "not recorded", which is every row written before
    the disclosure shipped. Dropping those would empty the ranked surfaces the
    moment the migration landed and refill them only after the next forecast
    run -- an outage produced by adding a column.
    """
    if anchor_clean is None:
        return True
    return bool(anchor_clean)


#: The served ``trend_direction`` is WITHHELD at every horizon, 2026-09-10.
#:
#: Measured on the first backtest panel after the leg-window resolver fix
#: (2a0ace3; run 34427386566). Pesaran-Timmermann on the >=$1 ``lgbm-v3%``
#: cohort is negative at every horizon -- -1.62pp (p=0.032, 17 dates), -0.66pp
#: (p=0.193, 23), -0.54pp (p=0.617, 23), -1.31pp (p=0.118, 8) at 3/7/14/30d --
#: directional accuracy is below a coin flip everywhere (43.8/48.2/48.9/45.3%),
#: and a constant "down" call beats the model at every horizon (55.0/58.1/65.2%
#: down-rate). Two cells are significant-negative: h=3 at >=$1 and h=7 at the
#: >=$20 floor (-1.89pp, p=0.037). Nothing is positive at any floor, so the
#: call is anti-informative, not merely null -- and consumers act on it.
#:
#: Unlike ``ANOMALY_SERVED_HORIZONS`` this is not a per-horizon tuple: no
#: horizon earned disclosure, so there is no partial gate to keep in sync.
#: This mirrors the 2026-08-12 ``confidence`` withdrawal -- the ``direction``
#: column is still written by the forecaster and still scored, it is simply no
#: longer disclosed. Republish only once a panel shows positive PT.
DIRECTION_DISCLOSED = False


def served_direction(raw: str | None, horizon: int | None = None) -> str:
    """The direction label to serve, given the model's stored ``raw`` call.

    Always ``"neutral"`` while ``DIRECTION_DISCLOSED`` is False. ``raw`` and
    ``horizon`` are accepted so callers read as a policy lookup rather than a
    constant, and so republishing is a change here and not in every route.
    """
    if not DIRECTION_DISCLOSED:
        return "neutral"
    return {"up": "bullish", "down": "bearish"}.get(raw or "", "neutral")
