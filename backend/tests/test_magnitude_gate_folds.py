"""Fold splitting for the magnitude band gate.

`magnitude_vs_climatology.py` answers a width question, and the one way it
could answer it wrongly in the model's FAVOUR is by letting an eval window see
dates the booster or the climatology table was fitted on. The repo has been
burned by exactly this shape before -- every band arm so far has been
CV-positive and serving-negative -- so the split boundaries are pinned here
rather than eyeballed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "archive"))

import itertools

from magnitude_vs_climatology import _folds


def _dates(n: int) -> np.ndarray:
    start = np.datetime64("2025-05-22")
    return start + np.arange(n).astype("timedelta64[D]")


class TestFoldCount:
    def test_one_fold_reproduces_the_single_split(self):
        got = _folds(_dates(100), n_folds=1)
        assert len(got) == 1

    def test_n_folds_yields_n_windows(self):
        assert len(_folds(_dates(200), n_folds=4)) == 4

    def test_too_few_dates_yields_no_folds_rather_than_a_degenerate_one(self):
        assert _folds(_dates(3), n_folds=4) == []


class TestNoLeakage:
    @pytest.mark.parametrize("n_folds", [1, 2, 3, 5])
    def test_every_fit_date_precedes_every_eval_date(self, n_folds):
        for fit_end, cal_end, ev_end in _folds(_dates(300), n_folds=n_folds):
            assert fit_end <= cal_end <= ev_end

    def test_eval_windows_do_not_overlap(self):
        got = _folds(_dates(300), n_folds=4)
        for (_, prev_cal, prev_ev), (_, nxt_cal, _) in itertools.pairwise(got):
            # the next fold's eval starts at its own cal_end, which must be at
            # or after the previous fold's eval end -- windows tile, never overlap
            assert nxt_cal >= prev_ev
            assert prev_cal < prev_ev

    def test_each_fold_grows_its_fit_window(self):
        """Walk-forward, not a rolling window: later folds train on more history,
        which is what production would actually do."""
        got = _folds(_dates(300), n_folds=4)
        fits = [f for f, _, _ in got]
        assert fits == sorted(fits)
        assert fits[0] < fits[-1]


class TestBoundariesAreRealDates:
    def test_cuts_are_drawn_from_the_supplied_dates(self):
        d = _dates(120)
        for cuts in _folds(d, n_folds=3):
            for c in cuts:
                assert c in d
