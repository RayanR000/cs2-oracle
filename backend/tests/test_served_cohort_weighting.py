"""Served-cohort weighting for the directional classifier.

The training frame is ~83% sub-$1 rows while api/serving_policy.py serves only
>= $1 (MIN_SERVED_PRICE_USD == the lower bound of HEADLINE_MIN_TIER), so
untouched the classifier spends most of its capacity on a population nobody
sees. `served_cohort_share` scales the >= $1 rows so they carry a chosen share
of the total training weight.

The knob is a *share*, not a raw multiplier, because a multiplier's meaning
drifts with the frame's tier composition and that composition varies fold to
fold. These tests pin the share semantics, the control arm's byte-identity with
the old weights, and the degenerate partitions.

See docs/superpowers/specs/2026-08-06-served-cohort-weighting-design.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from backtest.scoring import HEADLINE_MIN_TIER
from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT, ItemForecaster

THR = DIRECTION_FLAT_TOLERANCE_PCT  # 0.5
MOVER = 3.0


def _served_weight_share(weights, tiers) -> float:
    w = np.asarray(weights, dtype=float)
    served = np.asarray(tiers) >= HEADLINE_MIN_TIER
    return float(w[served].sum() / w.sum())


class TestControlArmIsUnchanged:
    """share=None must reproduce the pre-2026-08-06 weights exactly, or the
    A/B has no valid control."""

    def test_share_none_matches_the_mover_only_weights(self):
        r = np.array([5.0, -5.0, 0.1, -0.2, 9.0])
        tiers = np.array([0, 1, 0, 2, 0])
        base = ItemForecaster._direction_sample_weights(r, THR, MOVER)
        with_none = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=tiers, served_share=None)
        np.testing.assert_array_equal(base, with_none)

    def test_tiers_without_a_share_is_a_no_op(self):
        r = np.array([5.0, -5.0, 0.1])
        base = ItemForecaster._direction_sample_weights(r, THR, MOVER)
        got = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=np.array([1, 1, 1]), served_share=None)
        np.testing.assert_array_equal(base, got)

    def test_a_share_without_tiers_is_a_no_op(self):
        r = np.array([5.0, -5.0, 0.1])
        base = ItemForecaster._direction_sample_weights(r, THR, MOVER)
        got = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=None, served_share=0.5)
        np.testing.assert_array_equal(base, got)

    def test_the_base_weight_array_is_not_mutated_in_place(self):
        # _direction_sample_weights builds its own array, but the multiplier
        # helper takes a caller-owned one; scaling it in place would corrupt a
        # shared buffer.
        base = np.array([1.0, 3.0, 1.0, 3.0])
        tiers = np.array([0, 0, 1, 1])
        before = base.copy()
        ItemForecaster._served_cohort_multiplier(base, tiers, 0.5)
        np.testing.assert_array_equal(base, before)


class TestTheShareIsHit:
    def test_half_share_puts_half_the_weight_on_served_rows(self):
        r = np.array([5.0, -5.0, 0.1, -0.2, 9.0, 0.0])
        tiers = np.array([0, 0, 0, 0, 1, 1])
        w = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=tiers, served_share=0.5)
        assert _served_weight_share(w, tiers) == pytest.approx(0.5)

    @pytest.mark.parametrize("share", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_arbitrary_shares_are_hit_exactly(self, share):
        rng = np.random.RandomState(0)
        r = rng.normal(0, 5, 500)
        tiers = (rng.random_sample(500) < 0.17).astype(int)  # ~the real mix
        w = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=tiers, served_share=share)
        assert _served_weight_share(w, tiers) == pytest.approx(share)

    def test_the_share_is_exact_after_composing_with_the_mover_weight(self):
        # The served rows here are all flat (weight 1) and the non-served rows
        # all movers (weight 3). A multiplier derived from row COUNTS would
        # land at 0.5 of the rows but not 0.5 of the weight; only one derived
        # from the mover weights hits the share.
        r = np.array([0.0, 0.1, 9.0, -9.0])
        tiers = np.array([1, 1, 0, 0])
        w = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=tiers, served_share=0.5)
        assert _served_weight_share(w, tiers) == pytest.approx(0.5)
        # and it is emphatically not the count-based answer
        served = tiers >= HEADLINE_MIN_TIER
        assert w[served][0] == pytest.approx(3.0)

    def test_tier_2_and_above_count_as_served(self):
        # HEADLINE_MIN_TIER is a floor, not an equality: $5/$20/$100 items are
        # served too.
        r = np.array([9.0, 9.0, 9.0, 9.0])
        tiers = np.array([0, 1, 2, 4])
        w = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=tiers, served_share=0.75)
        assert _served_weight_share(w, tiers) == pytest.approx(0.75)
        assert w[1] == w[2] == w[3]

    def test_a_share_below_the_current_one_down_weights_served_rows(self):
        # Legitimate and allowed: the knob is a target, not a ratchet.
        r = np.array([9.0, 9.0, 9.0, 9.0])
        tiers = np.array([1, 1, 1, 0])
        w = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=tiers, served_share=0.25)
        assert _served_weight_share(w, tiers) == pytest.approx(0.25)
        assert w[0] < MOVER


class TestDegeneratePartitions:
    def test_no_served_rows_leaves_weights_untouched(self):
        r = np.array([5.0, -5.0, 0.1])
        tiers = np.array([0, 0, 0])
        base = ItemForecaster._direction_sample_weights(r, THR, MOVER)
        got = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=tiers, served_share=0.5)
        np.testing.assert_array_equal(base, got)

    def test_all_served_rows_leaves_weights_untouched(self):
        # The share is already 1.0; no finite multiplier moves it to 0.5, and
        # scaling every row uniformly changes nothing LightGBM can see.
        r = np.array([5.0, -5.0, 0.1])
        tiers = np.array([1, 2, 3])
        base = ItemForecaster._direction_sample_weights(r, THR, MOVER)
        got = ItemForecaster._direction_sample_weights(
            r, THR, MOVER, tiers=tiers, served_share=0.5)
        np.testing.assert_array_equal(base, got)

    def test_multiplier_is_one_when_a_partition_is_empty(self):
        assert ItemForecaster._served_cohort_multiplier(
            np.array([1.0, 1.0]), np.array([0, 0]), 0.5) == 1.0
        assert ItemForecaster._served_cohort_multiplier(
            np.array([1.0, 1.0]), np.array([1, 1]), 0.5) == 1.0

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_a_share_outside_the_open_unit_interval_is_rejected(self, bad):
        with pytest.raises(ValueError, match="served_share"):
            ItemForecaster._served_cohort_multiplier(
                np.array([1.0, 1.0]), np.array([0, 1]), bad)


class TestConstructorValidation:
    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 2.0])
    def test_an_out_of_range_share_is_rejected_at_construction(self, bad):
        with pytest.raises(ValueError, match="served_cohort_share"):
            ItemForecaster(db_session=None, served_cohort_share=bad)

    def test_none_is_the_default(self):
        assert ItemForecaster(db_session=None).served_cohort_share is None

    def test_a_valid_share_is_stored(self):
        f = ItemForecaster(db_session=None, served_cohort_share=0.5)
        assert f.served_cohort_share == 0.5


class TestClassPriorTracksTheWeighting:
    """_direction_class_prior promises to report the distribution the objective
    actually sees. If the classifier gains a weight the prior ignores, the
    diagnostic silently describes a model that was never trained."""

    def test_the_prior_moves_when_served_rows_are_up_weighted(self):
        # Served rows are all "up", non-served all "down". Up-weighting the
        # served cohort must shift the prior toward up.
        r = np.array([9.0, 9.0, -9.0, -9.0, -9.0, -9.0])
        tiers = np.array([1, 1, 0, 0, 0, 0])
        flat = ItemForecaster._direction_class_prior(r, THR, MOVER)
        tilted = ItemForecaster._direction_class_prior(
            r, THR, MOVER, tiers=tiers, served_share=0.8)
        assert flat[2] == pytest.approx(1 / 3)
        assert tilted[2] == pytest.approx(0.8)
        assert tilted[0] == pytest.approx(0.2)

    def test_the_prior_is_unchanged_without_a_share(self):
        r = np.array([9.0, -9.0, 0.0])
        tiers = np.array([1, 0, 1])
        assert (ItemForecaster._direction_class_prior(r, THR, MOVER)
                == ItemForecaster._direction_class_prior(
                    r, THR, MOVER, tiers=tiers, served_share=None))

    def test_tiers_are_filtered_by_the_same_finite_mask_as_returns(self):
        # A caller passing the raw columns must not have them misalign when
        # _direction_class_prior drops non-finite returns.
        r = np.array([np.nan, 9.0, 9.0, -9.0, -9.0, -9.0, -9.0])
        tiers = np.array([1, 1, 1, 0, 0, 0, 0])
        got = ItemForecaster._direction_class_prior(
            r, THR, MOVER, tiers=tiers, served_share=0.8)
        assert got[2] == pytest.approx(0.8)
        assert got[0] == pytest.approx(0.2)


class TestClassifierFitAcceptsTiers:
    def test_the_classifier_trains_with_served_cohort_weighting(self):
        rng = np.random.RandomState(7)
        n = 400
        X = rng.normal(0, 1, (n, 3))
        # Give the served rows a learnable signal the others lack, so a
        # weighted fit is not degenerate.
        tiers = (rng.random_sample(n) < 0.2).astype(int)
        y = np.where(tiers == 1, X[:, 0] * 10, rng.normal(0, 10, n))

        f = ItemForecaster(db_session=None, served_cohort_share=0.5)
        booster = f._fit_direction_classifier(
            X, y, None, None, "gbdt",
            {"num_leaves": 7, "learning_rate": 0.1, "max_depth": 3,
             "min_data_in_leaf": 5},
            horizon=7, num_boost_round=10, tier_train=tiers)
        assert booster.predict(X).shape == (n, 3)

    def test_tier_train_none_still_trains(self):
        rng = np.random.RandomState(7)
        X = rng.normal(0, 1, (200, 3))
        y = rng.normal(0, 10, 200)
        f = ItemForecaster(db_session=None, served_cohort_share=0.5)
        booster = f._fit_direction_classifier(
            X, y, None, None, "gbdt",
            {"num_leaves": 7, "learning_rate": 0.1, "max_depth": 3,
             "min_data_in_leaf": 5},
            horizon=7, num_boost_round=10, tier_train=None)
        assert booster.predict(X).shape == (200, 3)
