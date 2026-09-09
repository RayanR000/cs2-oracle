"""The exceedance-metadata A/B's two pure readers.

Neither is exercised by running the harness — a bug in either changes the
verdict silently, and the harness costs minutes per fold to run, so they are
tested directly. `paired_fold_deltas` is where an underpowered read turns into
a false verdict; `_score` is where a single-class fold does.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.exceedance_meta_ab import paired_fold_deltas, _score  # noqa: E402


def _folds(**by_fold):
    return [{"fold": f, "heldout_auc": v} for f, v in by_fold.items()]


class TestPairedFoldDeltas:
    def test_pairs_on_fold_index_not_position(self):
        # The arms can drop different folds (a fold with <50 val rows, or a
        # degenerate head). Pairing by list position would then subtract
        # unrelated folds and report a difference that is really fold variance.
        base = [{"fold": 0, "heldout_auc": 0.60},
                {"fold": 1, "heldout_auc": 0.50},
                {"fold": 2, "heldout_auc": 0.70}]
        arm = [{"fold": 0, "heldout_auc": 0.62},
               {"fold": 2, "heldout_auc": 0.72}]
        d = paired_fold_deltas(base, arm, "heldout_auc")
        assert d["n_folds"] == 2
        assert d["mean"] == pytest.approx(0.02)
        assert d["wins"] == 2

    def test_a_single_paired_fold_is_not_a_result(self):
        base = [{"fold": 0, "heldout_auc": 0.60}]
        arm = [{"fold": 0, "heldout_auc": 0.99}]
        assert paired_fold_deltas(base, arm, "heldout_auc") is None

    def test_none_metric_folds_are_dropped_not_zeroed(self):
        # A single-class fold scores None. Coercing it to 0.0 would pull the
        # mean toward "no effect" and inflate the fold count.
        base = [{"fold": 0, "heldout_auc": 0.60},
                {"fold": 1, "heldout_auc": 0.50},
                {"fold": 2, "heldout_auc": 0.55}]
        arm = [{"fold": 0, "heldout_auc": 0.65},
               {"fold": 1, "heldout_auc": None},
               {"fold": 2, "heldout_auc": 0.60}]
        d = paired_fold_deltas(base, arm, "heldout_auc")
        assert d["n_folds"] == 2
        assert d["mean"] == pytest.approx(0.05)

    def test_interval_excludes_zero_only_when_consistent(self):
        base = [{"fold": i, "heldout_auc": 0.50} for i in range(5)]
        steady = [{"fold": i, "heldout_auc": 0.55} for i in range(5)]
        assert paired_fold_deltas(base, steady, "heldout_auc")["excludes_zero"]

        noisy = [{"fold": i, "heldout_auc": v} for i, v in
                 enumerate([0.80, 0.20, 0.75, 0.25, 0.55])]
        assert not paired_fold_deltas(base, noisy, "heldout_auc")["excludes_zero"]

    def test_sign_is_arm_minus_baseline(self):
        base = [{"fold": i, "heldout_auc": 0.60} for i in range(3)]
        worse = [{"fold": i, "heldout_auc": 0.55} for i in range(3)]
        assert paired_fold_deltas(base, worse, "heldout_auc")["mean"] < 0


class TestScore:
    def test_single_class_slice_scores_none(self):
        y = np.ones(50)
        auc, ll, n = _score(y, np.full(50, 0.7))
        assert auc is None and ll is None and n == 50

    def test_too_few_rows_scores_none(self):
        y = np.array([0, 1] * 5)
        auc, ll, _ = _score(y, np.linspace(0.1, 0.9, 10))
        assert auc is None and ll is None

    def test_nan_labels_are_dropped_before_scoring(self):
        y = np.array([0.0, 1.0] * 15 + [np.nan] * 10)
        p = np.concatenate([np.tile([0.2, 0.8], 15), np.full(10, 0.5)])
        auc, ll, n = _score(y, p)
        assert n == 30
        assert auc == pytest.approx(1.0)

    def test_a_perfect_ranking_scores_auc_one(self):
        y = np.array([0] * 20 + [1] * 20, dtype=float)
        p = np.concatenate([np.full(20, 0.1), np.full(20, 0.9)])
        auc, ll, n = _score(y, p)
        assert auc == pytest.approx(1.0)
        assert ll < 0.2 and n == 40

    def test_probabilities_at_the_bounds_do_not_produce_inf_logloss(self):
        # LightGBM can emit exactly 0.0 or 1.0. An unclipped log loss is inf
        # there, and one inf fold poisons the whole paired mean.
        y = np.array([0] * 20 + [1] * 20, dtype=float)
        p = np.concatenate([np.zeros(20), np.ones(20)])
        _, ll, _ = _score(y, p)
        assert np.isfinite(ll)


class TestFoldTally:
    """Direction-aware fold counting.

    Log loss improves when the delta is NEGATIVE. Reporting `wins` (delta > 0)
    for both metrics reads a 22-of-26 calibration improvement as 4/26 — which
    is how a real result gets written up as a null.
    """

    def test_auc_counts_positive_folds(self):
        from scripts.exceedance_meta_ab import fold_tally
        assert fold_tally("heldout_auc", {"wins": 20, "n_folds": 26}) == "better 20/26"

    def test_logloss_counts_negative_folds(self):
        from scripts.exceedance_meta_ab import fold_tally
        assert fold_tally("heldout_logloss",
                          {"wins": 4, "n_folds": 26}) == "better 22/26"
