"""q_hat fold-design dispersion harness: stats, bars, void conditions.

What is silent when it breaks: a dispersion read that pools horizons, a
bar-1 pass counted on void horizons, a placebo gap that divides by a ~zero
CV, and an arm declared winner without the replay proviso (bar 3 is Phase B,
not scored here).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.measure_qhat_dispersion import _rel_gap, dispersion_stats, evaluate_bars


def _stats_for(cvs: dict) -> dict:
    """{h: {arm: {'cv': v}}} shape evaluate_bars consumes."""
    return {h: {arm: {"cv": v} for arm, v in arms.items()} for h, arms in cvs.items()}


class TestDispersionStats:
    def test_ratio_and_cv(self):
        s = dispersion_stats([100.0, 120.0, 140.0])
        assert s["n_folds"] == 3
        assert s["max_min_ratio"] == 1.4
        assert s["cv"] == float(np.std([100.0, 120.0, 140.0], ddof=1) / 120.0)

    def test_thin_panel_is_void_not_zero(self):
        s = dispersion_stats([100.0])
        assert s["n_folds"] == 1
        assert s["cv"] is None
        assert s["max_min_ratio"] is None

    def test_none_and_nan_fold_qhats_dropped(self):
        s = dispersion_stats([100.0, None, float("nan"), 120.0])
        assert s["n_folds"] == 2
        assert s["max_min_ratio"] == 1.2

    def test_horizons_never_pooled(self):
        # Two horizons with identical pooled mean but different spread must
        # not be merged: the helper scores one list per call.
        a = dispersion_stats([100.0, 100.0, 100.0, 160.0])
        b = dispersion_stats([115.0, 115.0, 115.0, 115.0])
        assert a["cv"] > b["cv"]


class TestRelGap:
    def test_normal(self):
        assert _rel_gap(0.20, 0.22) == pytest.approx(0.10)

    def test_missing_is_none(self):
        assert _rel_gap(None, 0.2) is None
        assert _rel_gap(0.2, None) is None

    def test_two_degenerate_cvs_are_stable(self):
        assert _rel_gap(0.0, 0.0) == 0.0

    def test_degenerate_control_vs_live_placebo_is_inf(self):
        assert _rel_gap(0.0, 0.2) == float("inf")


class TestEvaluateBars:
    def _cvs(self, ctrl=0.20, pbo=0.21, dense=0.20, sparse=0.20):
        return _stats_for(
            {h: {"control": ctrl, "placebo": pbo, "dense": dense, "sparse": sparse} for h in (3, 7, 14, 30)}
        )

    def test_null_everywhere(self):
        v = evaluate_bars(self._cvs())
        assert not v["void"]
        assert not v["arms"]["dense"]["pass"]
        assert not v["arms"]["sparse"]["pass"]

    def test_dense_pass(self):
        # 65% below control at 3 horizons, equal at the 4th: wins 3, worse 0.
        v = evaluate_bars(self._cvs(dense=0.07))
        assert not v["void"]
        assert v["arms"]["dense"]["pass"]

    def test_one_worse_horizon_vetoes(self):
        stats = self._cvs(dense=0.07)
        stats[30]["dense"]["cv"] = 0.25  # +25% vs control
        v = evaluate_bars(stats)
        assert v["arms"]["dense"]["wins"] == 3
        assert v["arms"]["dense"]["worse"] == 1
        assert not v["arms"]["dense"]["pass"]

    def test_two_wins_is_not_enough(self):
        stats = self._cvs()
        stats[3]["dense"]["cv"] = 0.07
        stats[7]["dense"]["cv"] = 0.07
        v = evaluate_bars(stats)
        assert not v["arms"]["dense"]["pass"]

    def test_placebo_instability_voids_the_probe(self):
        stats = self._cvs()
        stats[14]["placebo"]["cv"] = 0.30  # 50% gap
        v = evaluate_bars(stats)
        assert v["void"]
        assert not v["placebo_gap"][14]["stable"]
        assert v["placebo_gap"][3]["stable"]

    def test_void_horizon_is_never_a_win(self):
        stats = self._cvs(dense=0.07)
        stats[30]["dense"]["cv"] = None
        v = evaluate_bars(stats)
        assert v["arms"]["dense"]["wins"] == 3
        assert v["arms"]["dense"]["scored"] == 3
        assert v["arms"]["dense"]["pass"]
