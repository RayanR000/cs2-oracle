"""Isotonic calibration of the exceedance head (served as `move_odds`).

The head is a raw LightGBM binary output with no probability-calibration
layer: replay reliability reads ECE <1.3pp at h3/h7 but ~3.8pp at h30, so the
served number is uncalibrated where it is most needed. These guard the fix:

- the isotonic map (PAV, pure numpy) is monotone, bounded, and improves (or
  ties) in-sample Brier — the identity map is itself monotone, so a correct
  PAV fit can never lose to raw p;
- Brier / reliability-table / ECE helpers do the documented arithmetic;
- the DISCLOSED path serves calibrated p while the BAND denominator stays on
  raw p (its q_hat is dimensionally tied to it — a matched pair);
- the map + report round-trip through meta.json, and an artifact predating
  the key serves raw;
- the CV loop computes OOF exceed_p whenever EITHER head flag is on (with
  EXCEEDANCE_HEAD=1 alone — production's configuration — the old
  scale-only gate left no OOF pairs, so the head served uncalibrated with no
  Brier anywhere);
- the fit refuses below MIN_EXCEEDANCE_CALIBRATION_ROWS and EXCEEDANCE_CALIBRATE=0
  serves raw.
"""
from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def _forecaster(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _miscalibrated_pairs(n=4000, seed=0):
    """Overconfident raw p against a lower realised rate, with real spread."""
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(2, 2, n), 1e-3, 1.0)
    # True rate is a shrunk version of p: the head exaggerates.
    true_rate = 0.15 + 0.5 * p
    y = (rng.uniform(size=n) < true_rate).astype(float)
    return p, y


def _records(p, y):
    return pd.DataFrame({"exceed_p": np.asarray(p, dtype=float),
                         "exceed_y": np.asarray(y, dtype=float)})


def _exc_head(f, cols=("f", "g"), seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    x = rng.normal(size=n)
    y = (x > 0.5).astype(float)
    X = pd.DataFrame({cols[0]: x, cols[1]: rng.normal(size=n)})
    return f._fit_exceedance_classifier(
        X, y, boosting_type="gbdt", tree_params={}, horizon=7,
        tier_train=np.full(n, 2), num_boost_round=60)


# --- the isotonic fit -----------------------------------------------------

def test_isotonic_fit_is_monotone_bounded_and_within_range(tmp_path):
    f = _forecaster(tmp_path)
    p, y = _miscalibrated_pairs()
    xs, ys = f._isotonic_fit(np.clip(p, 1e-3, 1.0), y)
    assert len(xs) >= 2
    assert bool(np.all(np.diff(xs) > 0))
    assert bool(np.all(np.diff(ys) >= 0))
    assert float(ys.min()) >= 0.0 and float(ys.max()) <= 1.0
    assert float(xs.min()) >= 1e-3 and float(xs.max()) <= 1.0


def test_fit_improves_brier_and_reports_both_legs(tmp_path):
    f = _forecaster(tmp_path)
    p, y = _miscalibrated_pairs()
    meta = f._fit_exceedance_calibrator(7, _records(p, y), out_of_sample=True)
    assert meta is not None
    assert meta["method"] == "isotonic"
    assert meta["out_of_sample"] is True
    assert meta["n_rows"] == len(p)
    # Guaranteed: the identity map is monotone, so PAV can never lose in sample.
    assert meta["brier_cal"] <= meta["brier_raw"] + 1e-12
    assert meta["ece_cal_pp"] <= meta["ece_raw_pp"] + 2.0
    assert meta["reliability_raw"] and meta["reliability_cal"]
    first = meta["reliability_cal"][0]
    assert set(first) == {"lo", "hi", "n", "pred", "realized"}


def test_fit_refuses_below_min_rows(tmp_path):
    f = _forecaster(tmp_path)
    p = np.linspace(0.1, 0.9, 50)
    y = (p > 0.5).astype(float)
    assert f._fit_exceedance_calibrator(7, _records(p, y), True) is None
    assert 7 not in f.exceedance_calibrators
    assert 7 not in f.exceedance_calibration_meta


def test_fit_returns_none_without_pairs(tmp_path):
    f = _forecaster(tmp_path)
    recs = pd.DataFrame({"residual_pct": [1.0], "sigma": [0.07]})
    assert f._fit_exceedance_calibrator(7, recs, True) is None


def test_flag_off_fits_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("EXCEEDANCE_CALIBRATE", "0")
    f = _forecaster(tmp_path)
    p, y = _miscalibrated_pairs()
    assert f._fit_exceedance_calibrator(7, _records(p, y), True) is None
    assert f.exceedance_calibrate_enabled() is False


# --- the helpers ----------------------------------------------------------

def test_brier_is_mean_squared_error():
    p = np.array([0.9, 0.1, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    assert ItemForecaster.exceedance_brier_score(p, y) == pytest.approx(
        (0.81 + 0.01 + 0.25) / 3)
    assert np.isnan(ItemForecaster.exceedance_brier_score([], []))


def test_reliability_bins_are_fixed_width():
    p = np.array([0.05, 0.15, 0.95])
    y = np.array([0.0, 0.0, 0.0])
    rows = ItemForecaster.exceedance_reliability_table(p, y, n_bins=5)
    assert [(r["lo"], r["hi"]) for r in rows] == [(0.0, 0.2), (0.8, 1.0)]
    assert rows[0]["n"] == 2 and rows[0]["pred"] == pytest.approx(0.10)
    assert rows[0]["realized"] == pytest.approx(0.0)
    assert ItemForecaster.exceedance_reliability_table([], []) == []


def test_ece_is_count_weighted():
    rows = [{"n": 1, "pred": 0.9, "realized": 0.0},
            {"n": 3, "pred": 0.1, "realized": 0.0}]
    assert ItemForecaster.exceedance_ece(rows) == pytest.approx(
        (1 * 0.9 + 3 * 0.1) / 4)
    assert np.isnan(ItemForecaster.exceedance_ece([]))


# --- serving: calibrated disclosed, raw band -------------------------------

def test_disclosed_path_calibrates_while_band_stays_raw(tmp_path, monkeypatch):
    """The matched pair: q_hat is tied to RAW p, so band_scale must never see
    the calibrated one, while the disclosed exceed_p must."""
    monkeypatch.setenv("CLIMATOLOGY_SCALE", "0")
    monkeypatch.delenv("LEARNED_SCALE", raising=False)
    monkeypatch.delenv("SIGMA_EXPONENT", raising=False)
    f = _forecaster(tmp_path)
    head = _exc_head(f)
    f.exceedance_models = {7: head}
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f._artifact_exceedance_scale = True
    # A shrink toward 0.2 everywhere: calibrated != raw at every p.
    f.exceedance_calibrators = {
        7: {"xs": [0.001, 0.5, 1.0], "ys": [0.001, 0.2, 0.6]}}

    rng = np.random.default_rng(1)
    rows = pd.DataFrame({"f": rng.normal(size=40), "g": rng.normal(size=40)})
    sigma = np.full(40, 0.07)
    raw = np.clip(head.predict(rows[head.feature_name()]), 1e-3, 1.0)

    scale = f.band_scale(7, rows, sigma)
    np.testing.assert_allclose(scale, sigma * np.sqrt(raw))

    disclosed = f.exceedance_probability(7, rows)
    np.testing.assert_allclose(
        disclosed, np.clip(np.interp(raw, [0.001, 0.5, 1.0],
                                     [0.001, 0.2, 0.6]), 1e-3, 1.0))
    assert not np.allclose(disclosed, raw)


def test_calibrate_flag_off_serves_raw(tmp_path, monkeypatch):
    monkeypatch.setenv("EXCEEDANCE_CALIBRATE", "0")
    f = _forecaster(tmp_path)
    head = _exc_head(f)
    f.exceedance_models = {7: head}
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f.exceedance_calibrators = {
        7: {"xs": [0.001, 1.0], "ys": [0.5, 0.5]}}
    rows = pd.DataFrame({"f": [0.1, -0.4], "g": [0.2, 0.3]})
    raw = np.clip(head.predict(rows[head.feature_name()]), 1e-3, 1.0)
    np.testing.assert_allclose(f.exceedance_probability(7, rows), raw)


# --- persistence -----------------------------------------------------------

def test_map_and_report_round_trip_through_meta_json(tmp_path, monkeypatch):
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
    f = _forecaster(tmp_path)
    f.feature_cols = ["f", "g"]
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    p, y = _miscalibrated_pairs()
    meta = f._fit_exceedance_calibrator(7, _records(p, y), True)
    assert meta is not None
    f.save_models()
    stored = json.loads((tmp_path / "meta.json").read_text())[
        "exceedance_calibration"]
    assert "7" in stored and stored["7"]["meta"]["method"] == "isotonic"

    g = _forecaster(tmp_path)
    g.load_models()
    assert g.exceedance_calibrators[7] == f.exceedance_calibrators[7]
    assert g.exceedance_calibration_meta[7]["brier_cal"] == pytest.approx(
        meta["brier_cal"])


def test_artifact_without_a_map_serves_raw(tmp_path, monkeypatch):
    """Pre-calibration artifacts carry no key: move_odds is what it always
    was (raw), never an error and never a fabricated correction."""
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
    f = _forecaster(tmp_path)
    f.feature_cols = ["f", "g"]
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.exceedance_models = {7: _exc_head(f)}
    f.save_models()
    path = tmp_path / "meta.json"
    meta = json.loads(path.read_text())
    del meta["exceedance_calibration"]
    path.write_text(json.dumps(meta))

    g = _forecaster(tmp_path)
    g.load_models()
    assert g.exceedance_calibrators == {}
    rows = pd.DataFrame({"f": [0.1], "g": [0.2]})
    raw = np.clip(
        g.exceedance_models[7].predict(rows[["f", "g"]]), 1e-3, 1.0)
    np.testing.assert_allclose(g.exceedance_probability(7, rows), raw)


# --- the OOF gate (source guard; real proof is a training run) --------------

def test_cv_loop_computes_oof_whenever_a_head_flag_is_on():
    """With EXCEEDANCE_HEAD=1 alone (production: disclosed move_odds on a
    climatology band) the old scale-only gate fit no per-fold head, so the
    records carried no OOF p and the calibrator had nothing honest to fit on."""
    src = inspect.getsource(ItemForecaster._cv_evaluate_horizon)
    assert "exceedance_head_enabled()" in src
    assert "_fit_exceedance_classifier(" in src
    assert "exceed_y=" in src


def test_record_builder_carries_the_label():
    src = inspect.getsource(ItemForecaster._conformal_records)
    assert "exceed_y" in src
