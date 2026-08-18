"""Steam's clean spot (`aggregator_steam_spot`) must not vote.

`aggregator_steam_spot` is Steam's point-in-time `last_24h` written WITHOUT the
trailing-window fallback that `aggregator_sync` carries (pipeline.py's steam
branch, from 2026-08-17). It exists as the Steam leg for the cross-venue basis
feature -- docs/research/2026-08-16-cross-venue-basis-steam-buff.md.

It must be excluded from the consensus vote for a reason distinct from both
BID_SOURCES (wrong side of the book) and TRAILING_WINDOW_SOURCES (wrong time
basis): Steam ALREADY votes through `aggregator_sync`, so a voting
`aggregator_steam_spot` would cast a SECOND Steam ballot and double-count the
venue against the third-party asks. Kept in the archive, read directly by the
basis sidecar builder; never voted.
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


def test_the_spot_source_is_named():
    assert ItemForecaster.STEAM_SPOT_SOURCES == frozenset({"aggregator_steam_spot"})


def test_it_is_separate_from_bid_and_trailing_sources():
    """Three different exclusion reasons; overloading either set would tell the
    next reader the spot is a bid or a trailing mean, which it is not."""
    assert not (ItemForecaster.STEAM_SPOT_SOURCES & set(ItemForecaster.BID_SOURCES))
    assert not (ItemForecaster.STEAM_SPOT_SOURCES
                & set(ItemForecaster.TRAILING_WINDOW_SOURCES))


def test_spot_does_not_move_the_median(tmp_path):
    """Two asks and a spot leg: the vote is the asks' median alone."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-08-17", 10.0, "aggregator_buff163"),
        ("a", "2026-08-17", 10.4, "aggregator_csfloat"),
        ("a", "2026-08-17", 6.0, "aggregator_steam_spot"),
    ))
    assert out["price"].iloc[0] == 10.2


def test_spot_does_not_double_count_steam_against_sync(tmp_path):
    """The whole point: Steam votes through aggregator_sync. With sync present,
    the spot must not add a second Steam ballot -- the vote is sync + the one
    third-party ask, and n_ask_sources counts two, not three."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-08-17", 12.0, "aggregator_sync"),
        ("a", "2026-08-17", 10.0, "aggregator_buff163"),
        ("a", "2026-08-17", 6.0, "aggregator_steam_spot"),
    ))
    assert out["n_ask_sources"].iloc[0] == 2
    assert out["price"].iloc[0] == 11.0  # median(12, 10), spot excluded


def test_a_spot_only_item_day_is_dropped_not_zeroed(tmp_path):
    """An item-day whose only row is the spot leg must vanish, not become a 0 or
    NaN price -- it is not an ask."""
    out = _f(tmp_path)._apply_multi_source_voting(
        _rows(("a", "2026-08-17", 6.0, "aggregator_steam_spot")))
    assert len(out) == 0


def test_exclusion_is_null_safe(tmp_path):
    """Invariant 2: a bare NOT IN against a NULL source drops 13 years of
    prices. Every pre-2026 row has source IS NULL and must keep voting."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2020-01-01", 10.0, None),
        ("b", "2020-01-01", 20.0, None),
    ))
    assert len(out) == 2


def test_cache_version_bumped():
    assert ItemForecaster.VOTED_CACHE_VERSION >= 7
