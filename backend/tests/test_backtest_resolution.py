from __future__ import annotations

from datetime import date

import pandas as pd

from backtest.price_resolution import (
    MAX_WINDOW_SPAN_DAYS,
    SMOOTH_WINDOW,
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
