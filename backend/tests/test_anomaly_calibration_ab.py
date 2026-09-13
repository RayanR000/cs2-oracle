"""The 30d anomaly calibration harness fits its map out-of-sample.

The whole point of the harness is whether isotonic calibration earns h=30 its
disclosure. That verdict is only worth reading if the map is fitted on rows the
head did not train on, with the same H+13 embargo the outer split uses — a map
fitted on in-sample probabilities flatters itself and would ship a 30d
`anomaly_p` on a measurement that never happened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.forecaster import ItemForecaster, embargo_days  # noqa: E402
from scripts.anomaly_calibration_ab import (  # noqa: E402
    P_CLIP, _clip, inner_calibration_split)


def _train_frame(n_dates=40, n_items=3):
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    return pd.DataFrame([
        {"item_id": f"item_{i}", "date": d, "price": 10.0, "y": 0.0}
        for d in dates for i in range(n_items)
    ])


def test_split_holds_out_the_last_calib_frac_of_dates():
    df = _train_frame(n_dates=40)
    inner, calib = inner_calibration_split(df, horizon=1, calib_frac=0.25)

    assert calib["date"].min() > inner["date"].max()
    assert calib["date"].nunique() == 10  # the last 25% of 40 dates


def test_inner_train_is_embargoed_by_horizon_plus_13():
    """Rows whose label resolves inside the calibration slice must be dropped —
    otherwise the map is fitted on outcomes the head already saw."""
    df = _train_frame(n_dates=60)
    horizon = 7
    inner, calib = inner_calibration_split(df, horizon=horizon, calib_frac=0.25)

    calib_start = calib["date"].min()
    gap_days = (calib_start - inner["date"].max()).days
    assert gap_days >= embargo_days(horizon), (
        f"only {gap_days}d between inner-train and the calibration slice; "
        f"the embargo is {embargo_days(horizon)}d")


def test_h30_embargo_is_wider_than_h3():
    """The embargo scales with the horizon, so 30d loses more of its own fold."""
    df = _train_frame(n_dates=120)
    inner_3, calib_3 = inner_calibration_split(df, horizon=3, calib_frac=0.25)
    inner_30, calib_30 = inner_calibration_split(df, horizon=30, calib_frac=0.25)

    assert calib_3["date"].min() == calib_30["date"].min()  # same cut
    assert inner_30["date"].max() < inner_3["date"].max()


def test_split_refuses_a_fold_too_short_to_hold_anything_out():
    inner, calib = inner_calibration_split(_train_frame(n_dates=5),
                                           horizon=30, calib_frac=0.25)
    assert inner is None and calib is None


def test_split_refuses_when_the_embargo_eats_the_inner_train():
    """A fold whose inner slice is entirely inside the embargo yields no map,
    rather than one fitted on the handful of rows that survive."""
    inner, calib = inner_calibration_split(_train_frame(n_dates=12),
                                           horizon=30, calib_frac=0.25)
    assert inner is None


def test_clip_matches_the_served_range():
    """Arms are scored on production's own clip, so a featureless 0.0 cannot
    score an infinite log loss the GBM never risks."""
    out = _clip(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert out.min() >= P_CLIP[0]
    assert out.max() <= P_CLIP[1]
    assert out[2] == 0.5


def test_clip_matches_what_anomaly_probability_actually_serves():
    """Behavioural, not textual: drive production's own serving method with a
    stub head that emits out-of-range probabilities, and require the harness's
    clip to reproduce it exactly. If the served clip moves, this fails."""
    class _Head:
        def feature_name(self):
            return ["f"]

        def predict(self, X):
            return np.array([-1.0, 0.0, 0.5, 1.0, 2.0])

    f = ItemForecaster(db_session=None)
    f.anomaly_models = {3: _Head()}
    f.feature_medians = pd.Series(dtype=float)
    served = f.anomaly_probability(3, pd.DataFrame({"f": [0.0] * 5}))

    assert served is not None
    np.testing.assert_allclose(served, _clip(_Head().predict(None)))
