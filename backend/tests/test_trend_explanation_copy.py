"""Explanation copy must not state a confidence level.

"Confidence is high" was printed for forecasts realizing 4-31% directional
accuracy against 25-55% for the forecasts labelled low. The claim is not merely
unhelpful, it is backwards, so the sentence is removed rather than reworded.
The ``confidence`` field stays on the response schema — the frontend consumes
it and removing a field is a breaking change — it is simply no longer narrated.

See docs/superpowers/specs/2026-08-03-served-forecast-surface-design.md.
"""
from __future__ import annotations

import inspect

import pytest

from api.routes.items import _build_trend_explanation


class TestExplanationCopy:
    @pytest.mark.parametrize("direction", ["bullish", "bearish", "neutral"])
    def test_no_direction_mentions_confidence(self, direction):
        text = _build_trend_explanation(direction, 12.50)
        assert "onfidence" not in text

    def test_bullish_still_describes_the_direction(self):
        assert "upward" in _build_trend_explanation("bullish", 12.50)

    def test_bearish_still_describes_the_direction(self):
        assert "downward" in _build_trend_explanation("bearish", 12.50)

    def test_neutral_still_describes_the_direction(self):
        assert "stable" in _build_trend_explanation("neutral", 12.50)

    def test_signature_no_longer_takes_confidence(self):
        params = list(inspect.signature(_build_trend_explanation).parameters)
        assert "confidence" not in params
