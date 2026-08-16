"""Wiring of the served-coverage factor into ItemForecaster: accessor, persistence, and the
serve/train seams. The estimator itself is covered by test_served_recalibration.py; here we
guard that predict() scales q_hat by the factor, training computes and stores it, and it
round-trips through meta.json (no-op when absent). Real end-to-end proof waits for the data gate.
"""
from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from models.forecaster import ItemForecaster
from models.served_recalibration import FACTOR_MAX, FACTOR_MIN


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_fresh_forecaster_has_an_empty_factor_map(tmp_path):
    assert _f(tmp_path).served_coverage_factor == {}


def test_accessor_returns_stored_factor_and_defaults_to_one(tmp_path):
    f = _f(tmp_path)
    f.served_coverage_factor = {7: 0.85}
    assert f.served_qhat_multiplier(7) == 0.85
    assert f.served_qhat_multiplier(14) == 1.0            # absent horizon -> no-op


def test_accessor_guards_nonfinite_and_reclamps(tmp_path):
    f = _f(tmp_path)
    f.served_coverage_factor = {3: float("nan"), 7: 9.0, 14: -1.0}
    assert f.served_qhat_multiplier(3) == 1.0             # NaN -> no-op
    assert f.served_qhat_multiplier(7) == FACTOR_MAX      # re-clamped on read (defense in depth)
    assert f.served_qhat_multiplier(14) == FACTOR_MIN


def test_factor_round_trips_through_meta_json(tmp_path):
    f = _f(tmp_path)
    f.served_coverage_factor = {3: 0.9, 7: 1.1}
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    assert json.loads((tmp_path / "meta.json").read_text())["served_coverage_factor"] == {
        "3": 0.9, "7": 1.1}

    g = _f(tmp_path)
    g.load_models()
    assert g.served_coverage_factor[3] == 0.9
    assert g.served_qhat_multiplier(7) == 1.1


def test_artifact_without_the_key_loads_as_no_op(tmp_path):
    f = _f(tmp_path)
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    del meta["served_coverage_factor"]
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    g = _f(tmp_path)
    g.load_models()
    assert g.served_coverage_factor == {}
    assert g.served_qhat_multiplier(3) == 1.0


def test_predict_scales_qhat_by_the_multiplier(_stub=None):
    src = inspect.getsource(ItemForecaster.predict)
    assert "served_qhat_multiplier(" in src


def test_training_computes_and_stores_the_factor(_stub=None):
    src = inspect.getsource(ItemForecaster.train)
    assert "served_coverage_factors(" in src
    assert "self.served_coverage_factor" in src
