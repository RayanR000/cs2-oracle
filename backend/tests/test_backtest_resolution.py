from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backtest.price_resolution import (
    MAX_WINDOW_SPAN_DAYS,
    SMOOTH_WINDOW,
    load_voted_prices,
    smoothed_prices,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["item_id", "date", "price"])


def test_flat_series_gives_exactly_the_flat_price():
    """The symmetry property: a flat series must resolve to its own value on
    both legs, so actual_ret is exactly 0.0. The old code could not do this."""
    rows = [("ak", date(2026, 7, d), 2.0) for d in range(1, 11)]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 4)), ("ak", date(2026, 7, 10))})

    base = out[("ak", date(2026, 7, 4))]
    actual = out[("ak", date(2026, 7, 10))]
    assert base == 2.0
    assert actual == 2.0
    assert (actual - base) / base == 0.0


def test_uses_median_of_last_three_observations_at_or_before_anchor():
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 2), 10.0),   # spike
        ("ak", date(2026, 7, 3), 2.0),
        ("ak", date(2026, 7, 4), 3.0),    # after the anchor — must be ignored
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 3))})
    # median(1.0, 10.0, 2.0) == 2.0 — the spike is filtered, 07-04 excluded
    assert out[("ak", date(2026, 7, 3))] == 2.0


def test_lookback_is_row_based_not_calendar_based():
    """Observations on 07-01, 07-05, 07-09 are the last 3 rows even though
    they span 8 calendar days — but that exceeds the staleness cap."""
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 5), 2.0),
        ("ak", date(2026, 7, 9), 3.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 9))})
    assert ("ak", date(2026, 7, 9)) not in out  # span 8d > 7d cap


def test_staleness_cap_rejects_scattered_observations():
    rows = [
        ("ak", date(2026, 5, 1), 1.0),
        ("ak", date(2026, 6, 1), 2.0),
        ("ak", date(2026, 7, 1), 3.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 1))})
    assert out == {}


def test_fewer_than_three_observations_resolve_within_the_span():
    rows = [
        ("ak", date(2026, 7, 8), 4.0),
        ("ak", date(2026, 7, 9), 6.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 9))})
    assert out[("ak", date(2026, 7, 9))] == 5.0  # median(4.0, 6.0)


def test_anchor_before_any_observation_is_unresolvable():
    rows = [("ak", date(2026, 7, 9), 4.0)]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 1))})
    assert out == {}


def test_items_are_independent():
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 2), 1.0),
        ("m4", date(2026, 7, 1), 50.0),
        ("m4", date(2026, 7, 2), 50.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 2)), ("m4", date(2026, 7, 2))})
    assert out[("ak", date(2026, 7, 2))] == 1.0
    assert out[("m4", date(2026, 7, 2))] == 50.0


def test_span_is_measured_across_the_window_not_from_the_anchor():
    """The span check measures from the oldest to newest observation in the
    selected window, not from the anchor to the oldest observation. This matters
    when the anchor has no exact observation and sits well after the window.
    Observations tightly clustered within 2 days should resolve even if the
    anchor is 7+ days later."""
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 2), 2.0),
        ("ak", date(2026, 7, 3), 3.0),
    ]
    # Anchor on 07-10 has no observation; selected window is 07-01 to 07-03 (2d span).
    # Correct: span = 07-03 - 07-01 = 2d < 7d cap → resolves.
    # Incorrect: span = 07-10 - 07-01 = 9d > 7d cap → unresolvable.
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 10))})
    assert ("ak", date(2026, 7, 10)) in out
    assert out[("ak", date(2026, 7, 10))] == 2.0  # median(1.0, 2.0, 3.0)


def test_constants_match_the_codebase_staleness_convention():
    from collectors.pipeline import FALLBACK_MAX_AGE_DAYS

    assert SMOOTH_WINDOW == 3
    assert MAX_WINDOW_SPAN_DAYS == FALLBACK_MAX_AGE_DAYS


def test_missing_archive_raises_rather_than_returning_empty(tmp_path):
    """A green run with zero actuals is the failure shape 324cfff removed.
    Resolution must fail loudly instead."""
    with pytest.raises(FileNotFoundError, match="price archive"):
        load_voted_prices(tmp_path / "nope", ["ak"], date(2026, 7, 1), date(2026, 7, 9))


def test_loads_and_votes_multi_source_rows(tmp_path):
    archive = tmp_path / "price-archive"
    archive.mkdir()
    pd.DataFrame(
        {
            "item_slug": ["ak", "ak", "ak"],
            "day": pd.to_datetime([date(2026, 7, 1)] * 3),
            "mean_price": [2.0, 2.1, 90.0],  # 90.0 is the outlier source
            "volume": [10, 10, 10],
            "source": ["a", "b", "c"],
        }
    ).to_parquet(archive / "prices-2026.parquet")

    out = load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 1))

    assert list(out.columns) == ["item_id", "date", "price"]
    assert len(out) == 1  # one row per item-day after voting
    assert out.iloc[0]["price"] < 10.0  # the 90.0 source was voted out


def test_window_dates_before_the_range_are_loaded(tmp_path):
    """Resolving an anchor needs the observations *before* it, so the loader
    must reach back past min_date by the staleness cap."""
    archive = tmp_path / "price-archive"
    archive.mkdir()
    days = pd.to_datetime([date(2026, 6, 28), date(2026, 6, 29), date(2026, 7, 1)])
    pd.DataFrame(
        {
            "item_slug": ["ak"] * 3,
            "day": days,
            "mean_price": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
            "source": ["a", "a", "a"],
        }
    ).to_parquet(archive / "prices-2026.parquet")

    out = load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 1))
    assert len(out) == 3
