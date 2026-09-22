"""Explanation copy must not leak directional or confidence claims.

The direction signal is withdrawn (DIRECTION_DISCLOSED = False) and confidence
was anti-predictive, so `_build_trend_explanation` returns a single range-based
sentence with no arguments.
"""

from __future__ import annotations

import inspect

from api.routes.items import _build_trend_explanation


class TestExplanationCopy:
    def test_no_confidence_mention(self):
        assert "onfidence" not in _build_trend_explanation()

    def test_no_directional_language(self):
        text = _build_trend_explanation()
        for word in ("upward", "downward", "bullish", "bearish"):
            assert word not in text.lower()

    def test_range_based_language(self):
        assert "range" in _build_trend_explanation().lower()

    def test_takes_no_arguments(self):
        params = list(inspect.signature(_build_trend_explanation).parameters)
        assert params == []
