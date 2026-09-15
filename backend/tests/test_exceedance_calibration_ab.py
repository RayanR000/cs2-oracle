"""The exceedance calibration harness pins its grid and scores held-out only.

Two failure modes from the band programme are designed out in advance (see
`docs/research/2026-09-13-exceedance-calibration-preregistration.md`):

- UNPINNED GRID: real-OOF per-fold q_hat spread is 3.4-3.6x, so between-fold
  regime movement dwarfs between-arm movement. Every arm must draw the same
  training rows per fold — the seed may depend on the fold, never the arm.
- IN-SAMPLE MAP: the anomaly modulator fitted its width map on in-sample
  predicted p and applied it out-of-sample. The isotonic map here must be
  fitted date-disjoint from the inner train slice with the H+13 embargo at the
  inner boundary, and scored only on the val window.

These tests pin the mechanism, not the verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.forecaster import ItemForecaster, embargo_days
from scripts.archive.ab_test_item_metadata import _stratified_sample
from scripts.exceedance_calibration_ab import ARMS, P_CLIP, _brier, _clip


def _train_frame(n_dates=40, n_items=4):
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    items = [f"item_{i}" for i in range(n_items)]
    return pd.DataFrame([{"item_id": it, "date": d, "price": 10.0} for d in dates for it in items])


def test_arms_cover_both_bars():
    """The shipping test needs gbm_cal vs global_rate; the layer test needs
    gbm_cal vs gbm_inner. A refactor dropping either arm silently voids the
    prereg."""
    assert set(ARMS) == {"gbm", "gbm_inner", "gbm_cal", "item_rate", "global_rate"}


def test_stratified_sample_is_pinned_across_arms():
    """Same fold index -> identical rows, however many arms draw. The seed may
    depend on the fold, never the arm."""
    df = _train_frame(n_dates=30, n_items=6)
    items = sorted(df["item_id"].unique())
    first = _stratified_sample(df, items, 60, 3)
    for _ in range(3):  # three arms drawing "independently"
        other = _stratified_sample(df, items, 60, 3)
        pd.testing.assert_frame_equal(first, other)


def test_inner_split_is_date_disjoint_and_embargoed():
    from scripts.anomaly_calibration_ab import inner_calibration_split

    df = _train_frame(n_dates=60)
    horizon = 7
    inner, calib = inner_calibration_split(df, horizon=horizon, calib_frac=0.25)
    assert calib["date"].min() > inner["date"].max()
    gap_days = (calib["date"].min() - inner["date"].max()).days
    assert gap_days >= embargo_days(horizon)


def test_split_refuses_when_the_embargo_eats_the_inner_train():
    from scripts.anomaly_calibration_ab import inner_calibration_split

    inner, _ = inner_calibration_split(_train_frame(n_dates=12), horizon=30, calib_frac=0.25)
    assert inner is None


def test_clip_matches_the_served_range():
    out = _clip(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert out.min() >= P_CLIP[0]
    assert out.max() <= P_CLIP[1]
    assert out[2] == 0.5


def test_clip_matches_what_exceedance_probability_serves():
    """Behavioural: a stub head emitting out-of-range p through production's
    own disclosed path (no calibrator -> raw, clipped) must equal the
    harness clip. If the served clip moves, this fails."""

    class _Head:
        def feature_name(self):
            return ["f"]

        def predict(self, X):
            return np.array([-1.0, 0.0, 0.5, 1.0, 2.0])

    f = ItemForecaster(db_session=None)
    f.exceedance_models = {7: _Head()}
    f.feature_medians = pd.Series(dtype=float)
    served = f.exceedance_probability(7, pd.DataFrame({"f": [0.0] * 5}))

    assert served is not None
    np.testing.assert_allclose(served, _clip(_Head().predict(None)))


def test_brier_is_mean_squared_error():
    assert _brier(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
    assert _brier(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == 0.25
    assert _brier(np.array([]), np.array([])) is None
