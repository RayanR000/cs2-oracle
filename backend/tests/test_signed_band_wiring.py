"""The served band is two signed conformal offsets, with a symmetric fallback.

`docs/superpowers/specs/2026-08-19-signed-conformal-quantile-design.md`. `conformal.py`
grew `calibrate_signed`/`band_signed`; these guard the forecaster seam that persists the
`(q_lo, q_hi)` pair and serves it, plus the symmetric fallback that keeps an artifact
without the pair byte-identical to the old band. They do not run a full train (~5 min).
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np

from models.forecaster import ItemForecaster


def _forecaster(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_fresh_forecaster_exposes_empty_signed_offset_stores(tmp_path):
    f = _forecaster(tmp_path)
    assert f.conformal_q_lo == {}
    assert f.conformal_q_hi == {}


def test_band_offsets_falls_back_to_the_symmetric_pair(tmp_path):
    """No stored pair (every artifact before this change) must reproduce the old
    symmetric band exactly: offsets (-q_hat, +q_hat)."""
    f = _forecaster(tmp_path)
    f.conformal_calibration = {7: 3.5}
    assert f.band_offsets(7) == (-3.5, 3.5)


def test_band_offsets_returns_the_stored_signed_pair(tmp_path):
    f = _forecaster(tmp_path)
    f.conformal_calibration = {7: 3.5}
    f.conformal_q_lo = {7: -1.2}
    f.conformal_q_hi = {7: 4.1}
    assert f.band_offsets(7) == (-1.2, 4.1)


def test_band_offsets_ignores_a_nonfinite_stored_pair(tmp_path):
    """A hand-edited or degenerate pair must not push a NaN into a half-width;
    fall back to the symmetric band the q_hat guarantees is finite."""
    f = _forecaster(tmp_path)
    f.conformal_calibration = {7: 3.5}
    f.conformal_q_lo = {7: np.nan}
    f.conformal_q_hi = {7: 4.1}
    assert f.band_offsets(7) == (-3.5, 3.5)


def test_signed_pair_round_trips_through_save_and_load(tmp_path):
    f = _forecaster(tmp_path)
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.conformal_q_lo = {7: -1.2, 14: -0.8}
    f.conformal_q_hi = {7: 4.1, 14: 3.3}
    f.feature_cols = ["f", "g"]
    import pandas as pd
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f.save_models()

    g = _forecaster(tmp_path)
    g.load_models()
    assert g.band_offsets(7) == (-1.2, 4.1)
    assert g.band_offsets(14) == (-0.8, 3.3)


def test_predict_serves_the_signed_band_and_no_longer_recentres_on_direction():
    """The range stance: the served interval comes from the signed pair, and the
    3-class classifier no longer moves the mid after calibration."""
    src = inspect.getsource(ItemForecaster.predict)
    assert "band_signed(" in src
    assert "self._recenter_on_direction(" not in src
