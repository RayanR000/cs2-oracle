"""Serving's base price must obey the same staleness bound as the backtest.

``predict()`` anchored its dollar conversion on ``df.groupby("item_id").tail(3)``
— the last three *rows*, with no calendar bound. For a sparsely observed item
those three rows can span months, so the "3-day median" silently became a median
of prices from unrelated periods, stamped as today's value.

That is the price-laundering shape ``db5bddb`` removed from the collector's
historical fallback and that ``backtest.price_resolution.MAX_WINDOW_SPAN_DAYS``
bounds on both scoring legs. Serving was the remaining copy without the bound,
which is why ``item_forecasts.current_price`` and the archive-resolved
``base_price`` diverge by a median 3.6% and a 90th percentile of 35%.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS
from models.forecaster import ItemForecaster


def _frame(rows):
    return pd.DataFrame(
        [{"item_id": i, "date": pd.Timestamp(d), "price": p} for i, d, p in rows]
    )


ANCHOR = pd.Timestamp(date(2026, 7, 17))


class TestAnchorRespectsTheSpanBound:
    def test_dense_item_is_the_three_day_median(self):
        df = _frame([
            ("ak", date(2026, 7, 15), 10.0),
            ("ak", date(2026, 7, 16), 12.0),
            ("ak", date(2026, 7, 17), 11.0),
        ])
        out = ItemForecaster._smoothed_anchor_prices(df, ANCHOR)
        assert out["ak"] == pytest.approx(11.0)

    def test_observations_beyond_the_span_are_excluded(self):
        """The regression. Two stale prices must not vote on today's anchor."""
        stale = ANCHOR - pd.Timedelta(days=MAX_WINDOW_SPAN_DAYS + 10)
        df = _frame([
            ("ak", stale.date(), 100.0),
            ("ak", (stale + pd.Timedelta(days=1)).date(), 100.0),
            ("ak", date(2026, 7, 17), 10.0),
        ])
        out = ItemForecaster._smoothed_anchor_prices(df, ANCHOR)
        # Unbounded tail(3) would return median(100, 100, 10) = 100.0
        assert out["ak"] == pytest.approx(10.0)

    def test_a_single_in_window_observation_is_used_unsmoothed(self):
        """Serving cannot drop items, so a thin window degrades to no smoothing."""
        stale = ANCHOR - pd.Timedelta(days=MAX_WINDOW_SPAN_DAYS + 5)
        df = _frame([
            ("ak", stale.date(), 99.0),
            ("ak", date(2026, 7, 17), 3.0),
        ])
        out = ItemForecaster._smoothed_anchor_prices(df, ANCHOR)
        assert out["ak"] == pytest.approx(3.0)

    def test_item_with_nothing_in_window_falls_back_to_its_latest(self):
        """Degrade, never drop: the item still gets a forecast."""
        stale = ANCHOR - pd.Timedelta(days=MAX_WINDOW_SPAN_DAYS + 30)
        df = _frame([
            ("ak", stale.date(), 42.0),
            ("ak", (stale + pd.Timedelta(days=2)).date(), 44.0),
        ])
        out = ItemForecaster._smoothed_anchor_prices(df, ANCHOR)
        assert "ak" in out
        assert out["ak"] == pytest.approx(44.0), "must be the latest, not the median"

    def test_only_the_three_most_recent_in_window_observations_vote(self):
        df = _frame([
            ("ak", date(2026, 7, 13), 1.0),
            ("ak", date(2026, 7, 14), 1.0),
            ("ak", date(2026, 7, 15), 5.0),
            ("ak", date(2026, 7, 16), 6.0),
            ("ak", date(2026, 7, 17), 7.0),
        ])
        out = ItemForecaster._smoothed_anchor_prices(df, ANCHOR)
        assert out["ak"] == pytest.approx(6.0)

    def test_items_are_smoothed_independently(self):
        stale = ANCHOR - pd.Timedelta(days=MAX_WINDOW_SPAN_DAYS + 3)
        df = _frame([
            ("ak", date(2026, 7, 16), 10.0),
            ("ak", date(2026, 7, 17), 12.0),
            ("awp", stale.date(), 500.0),
            ("awp", date(2026, 7, 17), 20.0),
        ])
        out = ItemForecaster._smoothed_anchor_prices(df, ANCHOR)
        assert out["ak"] == pytest.approx(11.0)
        assert out["awp"] == pytest.approx(20.0)

    def test_matches_the_backtest_resolver_on_the_same_observations(self):
        """The whole point: one staleness convention across serving and scoring."""
        from backtest.price_resolution import resolve_anchors

        rows = [
            ("ak", date(2026, 7, 15), 3.0),
            ("ak", date(2026, 7, 16), 3.2),
            ("ak", date(2026, 7, 17), 3.4),
        ]
        # resolve_anchors compares against plain dates; the serving frame
        # carries Timestamps. Same observations, each in its native form.
        voted = pd.DataFrame(
            [{"item_id": i, "date": d, "price": p} for i, d, p in rows]
        )
        served = ItemForecaster._smoothed_anchor_prices(_frame(rows), ANCHOR)
        resolved = resolve_anchors(voted, [("ak", ANCHOR.date())])
        assert served["ak"] == pytest.approx(resolved[("ak", ANCHOR.date())].price)
