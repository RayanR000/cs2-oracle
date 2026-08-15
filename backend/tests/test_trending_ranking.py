"""The trending list must not rank by confidence.

``_build_trending`` ordered by ``desc(confidence_order)`` first, so the
anti-predictive high-confidence subset occupied the top of the list before
predicted return was considered at all. The ``"medium"`` tier it ranked between
high and low has never been emitted once — 0 rows in the entire forecast
history.

See docs/superpowers/specs/2026-08-03-served-forecast-surface-design.md.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import api.routes.items as items_mod


class TestTrendingDoesNotRankByConfidence:
    def test_confidence_order_is_gone(self):
        source = inspect.getsource(items_mod._build_trending)
        assert "confidence_order" not in source

    def test_dead_medium_tier_is_gone(self):
        source = Path(items_mod.__file__).read_text()
        assert '"medium"' not in source

    def test_trending_applies_the_price_floor(self):
        source = inspect.getsource(items_mod._build_trending)
        assert "price_floor_clause" in source

    def test_trending_freshness_is_a_window_not_an_exact_today(self):
        """`forecast_date` is the day the band was anchored on -- the newest
        archived day, usually today-1 because the dump lands ~22:00 UTC -- not
        the wall clock of the run. A `== today` freshness pin empties the
        trending list after every normal daily run; the guard has to be a
        bounded recency window."""
        source = inspect.getsource(items_mod._build_trending)
        assert "forecast_date == today" not in source, (
            "an exact-today pin goes empty once forecast_date is the anchor day")
        assert "forecast_date >=" in source, (
            "the freshness guard must be a bounded window, not dropped entirely")
