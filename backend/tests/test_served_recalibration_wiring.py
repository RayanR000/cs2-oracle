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

import models.served_recalibration as sr
from models.forecaster import ItemForecaster
from models.served_recalibration import (
    CLIMATOLOGY_SERVING_START,
    FACTOR_MAX,
    FACTOR_MIN,
    SIGNED_BAND_SERVING_START,
    served_coverage_factors,
)


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


def test_feedback_is_dormant_when_the_cutover_is_unset(_stub=None):
    """With no cutover (since=None), served_coverage_factors returns {} without ever reading the
    panel — the whole store would be pre-signed-band geometry. A session whose every attribute
    access raises proves the panel is not touched. (The shipped default is now a real date; the
    dormancy is a property of since=None, exercised here explicitly.)"""
    class _Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"panel was read ({name}) while feedback should be dormant")

    assert served_coverage_factors(_Exploding(), [3, 7, 14, 30], since=None) == {}


def test_shipped_cutover_is_a_parseable_date(_stub=None):
    """The deployed defaults must be valid ISO dates (or None) — a typo here would silently
    filter every row out and keep the feedback dormant forever."""
    import numpy as np
    for start in (SIGNED_BAND_SERVING_START, CLIMATOLOGY_SERVING_START):
        if start is not None:
            np.datetime64(start)                              # raises on a malformed date


def test_geometry_floor_takes_the_latest_cutover(monkeypatch):
    """Two band-geometry cutovers -> the panel floors to the LATER, so the factor is fit only on
    rows served under the current band shape. A climatology cutover after the signed band wins;
    one before it (or unset) leaves the signed-band floor in force."""
    monkeypatch.setattr(sr, "SIGNED_BAND_SERVING_START", "2026-08-19")

    monkeypatch.setattr(sr, "CLIMATOLOGY_SERVING_START", "2026-09-15")
    assert sr._geometry_floor() == "2026-09-15"

    monkeypatch.setattr(sr, "CLIMATOLOGY_SERVING_START", "2026-08-01")
    assert sr._geometry_floor() == "2026-08-19"

    monkeypatch.setattr(sr, "CLIMATOLOGY_SERVING_START", None)
    assert sr._geometry_floor() == "2026-08-19"


def test_geometry_floor_is_none_only_when_no_cutover_is_set(monkeypatch):
    monkeypatch.setattr(sr, "SIGNED_BAND_SERVING_START", None)
    monkeypatch.setattr(sr, "CLIMATOLOGY_SERVING_START", None)
    assert sr._geometry_floor() is None


def test_default_since_floors_the_panel_to_the_latest_cutover(monkeypatch):
    """served_coverage_factors (no explicit since) reads the panel at _geometry_floor(), not the
    signed-band start alone, once a later climatology cutover is set."""
    monkeypatch.setattr(sr, "SIGNED_BAND_SERVING_START", "2026-08-19")
    monkeypatch.setattr(sr, "CLIMATOLOGY_SERVING_START", "2026-09-15")

    seen = {}

    def _capture(session, horizons, *, since=None):
        seen["since"] = since
        import pandas as pd
        return pd.DataFrame(columns=list(sr.PANEL_COLUMNS))

    monkeypatch.setattr(sr, "_load_panel", _capture)
    served_coverage_factors(object(), [3, 7, 14, 30])
    assert seen["since"] == "2026-09-15"
