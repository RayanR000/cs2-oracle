"""The directional classifier's inherited up/down prior, and its correction.

The classifier trains multiclass with no class_weight, so whatever up/down
skew its 1460-day window carried is learned and then applied at serve time
regardless of current market state. That is the mechanism behind "predicts
down 57-87% of the time regardless of date" in
docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md.

Two asymmetries live in this classifier and only one is a bug: up-vs-down is
inherited and unwanted; mover-vs-flat is deliberate (DIRECTION_MOVER_WEIGHT_MAP
up-weights movers 3x). Every correction here is zero-sum over down/up and
leaves the flat mass alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT, ItemForecaster

THR = DIRECTION_FLAT_TOLERANCE_PCT  # 0.5
MOVER = 3.0


class TestDirectionClassPrior:
    def test_balanced_returns_give_a_balanced_prior(self):
        r = np.array([5.0, -5.0, 5.0, -5.0])
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert prior[0] == pytest.approx(0.5)
        assert prior[2] == pytest.approx(0.5)
        assert prior[1] == pytest.approx(0.0)

    def test_prior_is_weighted_by_mover_weight_not_raw_counts(self):
        # 1 up mover (w=3), 3 flat rows (w=1 each) -> total weight 6.
        r = np.array([5.0, 0.1, 0.1, 0.1])
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert prior[2] == pytest.approx(3.0 / 6.0)   # not 1/4
        assert prior[1] == pytest.approx(3.0 / 6.0)
        assert prior[0] == pytest.approx(0.0)

    def test_down_skew_is_reflected_in_the_prior(self):
        r = np.array([-5.0, -5.0, -5.0, 5.0])
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert prior[0] > prior[2]

    def test_prior_sums_to_one(self):
        rng = np.random.default_rng(0)
        r = rng.normal(-0.3, 4.0, size=5000)
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert sum(prior.values()) == pytest.approx(1.0)

    def test_non_finite_returns_are_dropped(self):
        r = np.array([5.0, -5.0, np.nan, np.inf, -np.inf])
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert sum(prior.values()) == pytest.approx(1.0)
        assert prior[0] == pytest.approx(0.5)

    def test_empty_input_is_not_estimable(self):
        assert ItemForecaster._direction_class_prior(np.array([]), THR, MOVER) == {}

    def test_all_non_finite_is_not_estimable(self):
        assert ItemForecaster._direction_class_prior(
            np.array([np.nan, np.nan]), THR, MOVER) == {}

    def test_zero_total_weight_is_not_estimable(self):
        # When mover_weight=0.0, all movers get weight 0. If all returns are
        # movers (|r| > threshold), then total weight is 0 and not estimable.
        assert ItemForecaster._direction_class_prior(
            np.array([5.0, -5.0]), 0.0, 0.0) == {}
