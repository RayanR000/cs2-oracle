"""Steam's trailing-window means must not vote against point-in-time asks.

aggregator_steam_7d/30d/90d are MA(7)/MA(30)/MA(90) of the SALE price
(collectors/csgotrader_aggregator.py:294-312). Measured against genuine
third-party asks (buff163/csfloat/csmoney/skinport/youpin -- the Steam-derived
aggregator_sync and aggregator_steam_17mafo excluded from the comparison
group), the trailing window sits ~32% ABOVE them (median ratio 1.316, a
premium on 76.6% of item-days) -- the Steam cash-out fee wedge every
Steam-derived feed carries. So unlike BID_SOURCES, excluding these pulls the
consensus DOWN, not up: the defect is not the LEVEL, it is the TIME basis. A
trailing 90-day mean barely moves when live asks move, so it damps the
consensus and mechanically manufactures mean-reversion in the resulting
returns -- which matters acutely here because a reversal effect is the signal
this project is currently trying to validate.

These three sources exist only 2026-07-11 to 2026-08-08, so it is labels and
stored A/B verdicts from 2026-07-11 onward that sit downstream of this, not
"every label from 2026-03 onward".

Measured over 2026, >=$1, universe-filtered: excluding them costs 650
item-days of 2,589,787 that have no other ask source, but moves the voted
median on 16.89% of item-days (median -8.47%) and flips 6.17% of
consecutive-day return directions.

This is NOT a staleness fix -- aggregator_sync and aggregator_steam_17mafo are
last_24h FALLING BACK to these same windows on exactly the illiquid items.
There is no point-in-time Steam price in this archive at all.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _rows(*specs):
    return pd.DataFrame([
        {"item_id": i, "date": pd.Timestamp(d), "price": p, "source": s}
        for i, d, p, s in specs
    ])


def test_the_three_windows_are_named():
    assert ItemForecaster.TRAILING_WINDOW_SOURCES == frozenset({
        "aggregator_steam_7d", "aggregator_steam_30d", "aggregator_steam_90d"})


def test_they_are_separate_from_bid_sources():
    """A bid is the wrong side of the book; an MA is the wrong time basis.
    Overloading BID_SOURCES would make the next reader think these are bids."""
    assert not (ItemForecaster.TRAILING_WINDOW_SOURCES
                & set(ItemForecaster.BID_SOURCES))


def test_trailing_windows_do_not_move_the_median(tmp_path):
    """Two asks and three MA legs: the vote must be the asks' median alone."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-07-11", 10.0, "aggregator_buff163"),
        ("a", "2026-07-11", 10.4, "aggregator_csfloat"),
        ("a", "2026-07-11", 7.0, "aggregator_steam_7d"),
        ("a", "2026-07-11", 6.5, "aggregator_steam_30d"),
        ("a", "2026-07-11", 6.0, "aggregator_steam_90d"),
    ))
    assert out["price"].iloc[0] == 10.2


def test_a_trailing_only_item_day_is_dropped_not_zeroed(tmp_path):
    """650 item-days of 2,589,787 have no other source. They must vanish, not
    become a 0 or a NaN price."""
    out = _f(tmp_path)._apply_multi_source_voting(
        _rows(("a", "2026-07-11", 7.0, "aggregator_steam_7d")))
    assert len(out) == 0


def test_exclusion_is_null_safe(tmp_path):
    """Invariant 2: a bare NOT IN against a NULL source drops 13 years of
    prices. Every pre-2026 row has source IS NULL."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2020-01-01", 10.0, None),
        ("b", "2020-01-01", 20.0, None),
    ))
    assert len(out) == 2


def test_17mafo_is_not_caught_by_the_exclusion(tmp_path):
    """The task's own highest-severity risk: for 2026-04-16 -> 07-10,
    aggregator_steam_17mafo is the ONLY ask source in the entire archive --
    buff163/csfloat/youpin have a full ~86-day collection outage across
    exactly that span. The exclusion must be exact frozenset membership, never
    a prefix match: a `str.startswith("aggregator_steam")` "simplification"
    would leave TRAILING_WINDOW_SOURCES untouched, pass every other test here,
    and silently delete effectively all 2026 price data for those 86 days.
    This item-day must survive and vote alone."""
    out = _f(tmp_path)._apply_multi_source_voting(
        _rows(("a", "2026-05-01", 10.0, "aggregator_steam_17mafo")))
    assert len(out) == 1
    assert out["price"].iloc[0] == 10.0
    assert out["n_ask_sources"].iloc[0] == 1


def test_n_ask_sources_excludes_trailing_windows(tmp_path):
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-07-11", 10.0, "aggregator_buff163"),
        ("a", "2026-07-11", 7.0, "aggregator_steam_7d"),
    ))
    assert out["n_ask_sources"].iloc[0] == 1


def test_cache_version_bumped():
    assert ItemForecaster.VOTED_CACHE_VERSION >= 6
