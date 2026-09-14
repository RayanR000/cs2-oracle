"""EXCEEDANCE_SCALE: the band divides by `sigma * sqrt(p_exceed)` instead of `sigma`.

Task 2 of docs/superpowers/plans/2026-08-16-exceedance-band-scale-phase2-plan.md. The
exceedance probability is passed to conformal as a `learned_scale` array, so beta stays
neutral (resolve_scale forbids a learned scale beside beta != 1) and q_hat is calibrated
against the SAME scale it is served against — a matched pair, exactly like SIGMA_EXPONENT's
q_hat/beta. These guard the flag, the mutual exclusion, the matched calibrate/serve identity,
and the meta round-trip. The offline A/B is Task 3, run from the controller.
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models import conformal
from models.forecaster import ItemForecaster


def _forecaster(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _records(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sigma = np.clip(np.abs(rng.normal(0.07, 0.03, n)), 0.02, 0.65)
    resid = np.abs(rng.normal(0, 5.0, n))
    p = np.clip(rng.uniform(0, 1, n), 1e-3, 1.0)
    return pd.DataFrame(
        {
            "residual_pct": resid,
            "sigma": sigma,
            "mid_ret": rng.normal(0, 2.0, n),
            "exceed_p": p,
        }
    )


def _exc_head(f, cols=("f", "g"), seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    x = rng.normal(size=n)
    y = (x > 0.5).astype(float)
    X = pd.DataFrame({cols[0]: x, cols[1]: rng.normal(size=n)})
    return f._fit_exceedance_classifier(
        X, y, boosting_type="gbdt", tree_params={}, horizon=7, tier_train=np.full(n, 2), num_boost_round=60
    )


# --- the flag -------------------------------------------------------------


def test_flag_reads_the_environment(monkeypatch):
    monkeypatch.delenv("EXCEEDANCE_SCALE", raising=False)
    assert ItemForecaster.exceedance_scale_enabled() is False
    monkeypatch.setenv("EXCEEDANCE_SCALE", "1")
    assert ItemForecaster.exceedance_scale_enabled() is True


def test_exceedance_scale_is_mutually_exclusive_with_learned_and_exponent(tmp_path, monkeypatch):
    f = _forecaster(tmp_path)
    recs = _records()
    monkeypatch.setenv("EXCEEDANCE_SCALE", "1")
    monkeypatch.setenv("LEARNED_SCALE", "1")
    monkeypatch.delenv("SIGMA_EXPONENT", raising=False)
    with pytest.raises(RuntimeError, match="EXCEEDANCE_SCALE"):
        f._calibrate_conformal(7, recs.copy())

    monkeypatch.delenv("LEARNED_SCALE", raising=False)
    monkeypatch.setenv("SIGMA_EXPONENT", "1")
    with pytest.raises(RuntimeError, match="EXCEEDANCE_SCALE"):
        f._calibrate_conformal(7, recs.copy())


# --- the matched calibrate scale ------------------------------------------


def test_calibrate_uses_sigma_times_sqrt_p_and_keeps_beta_neutral(tmp_path, monkeypatch):
    f = _forecaster(tmp_path)
    recs = _records()
    monkeypatch.setenv("EXCEEDANCE_SCALE", "1")
    monkeypatch.setenv("CLIMATOLOGY_SCALE", "0")  # default-on since 2026-08-19; mutually exclusive
    monkeypatch.delenv("LEARNED_SCALE", raising=False)
    monkeypatch.delenv("SIGMA_EXPONENT", raising=False)

    q_hat = f._calibrate_conformal(7, recs.copy())

    expected_scale = recs["sigma"].to_numpy() * np.sqrt(np.clip(recs["exceed_p"].to_numpy(), 1e-3, 1.0))
    expected_q = conformal.calibrate(
        recs["residual_pct"].to_numpy(),
        recs["sigma"].to_numpy(),
        conformal.ALPHA,
        conformal.BETA_NEUTRAL,
        learned_scale=expected_scale,
    )
    assert q_hat == pytest.approx(expected_q)
    assert f.conformal_beta[7] == conformal.BETA_NEUTRAL


def test_calibrate_falls_back_to_sigma_when_no_exceed_p_column(tmp_path, monkeypatch):
    """Flag on but the records carry no OOF probability -> sigma band, loudly."""
    f = _forecaster(tmp_path)
    recs = _records().drop(columns=["exceed_p"])
    monkeypatch.setenv("EXCEEDANCE_SCALE", "1")
    monkeypatch.setenv("CLIMATOLOGY_SCALE", "0")  # default-on since 2026-08-19; mutually exclusive
    monkeypatch.delenv("LEARNED_SCALE", raising=False)
    monkeypatch.delenv("SIGMA_EXPONENT", raising=False)

    q_hat = f._calibrate_conformal(7, recs.copy())
    plain = conformal.calibrate(
        recs["residual_pct"].to_numpy(), recs["sigma"].to_numpy(), conformal.ALPHA, conformal.BETA_NEUTRAL
    )
    assert q_hat == pytest.approx(plain)


# --- serving --------------------------------------------------------------


def test_band_scale_serves_sigma_times_sqrt_p_when_the_artifact_says_on(tmp_path):
    f = _forecaster(tmp_path)
    head = _exc_head(f)
    f.exceedance_models = {7: head}
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f._artifact_exceedance_scale = True

    rng = np.random.default_rng(1)
    rows = pd.DataFrame({"f": rng.normal(size=40), "g": rng.normal(size=40)})
    sigma = np.full(40, 0.07)
    scale = f.band_scale(7, rows, sigma)
    p = head.predict(rows[head.feature_name()])
    np.testing.assert_allclose(scale, sigma * np.sqrt(np.clip(p, 1e-3, 1.0)))


def test_band_scale_ignores_env_when_the_artifact_says_off(tmp_path, monkeypatch):
    """A plain-sigma artifact reloaded with EXCEEDANCE_SCALE=1 in the env must NOT
    apply the scale — its q_hat was calibrated at plain sigma (matched-pair safety)."""
    monkeypatch.setenv("EXCEEDANCE_SCALE", "1")
    f = _forecaster(tmp_path)
    f.exceedance_models = {7: _exc_head(f)}
    f._artifact_exceedance_scale = False
    rows = pd.DataFrame({"f": [0.1], "g": [0.2]})
    assert f.band_scale(7, rows, np.array([0.07])) is None


# --- persistence ----------------------------------------------------------


def test_flag_round_trips_through_meta_json(tmp_path, monkeypatch):
    monkeypatch.setenv("EXCEEDANCE_SCALE", "1")
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
    f = _forecaster(tmp_path)
    f.feature_cols = ["f", "g"]
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    assert json.loads((tmp_path / "meta.json").read_text())["exceedance_scale"] is True

    monkeypatch.delenv("EXCEEDANCE_SCALE", raising=False)
    g = _forecaster(tmp_path)
    g.load_models()
    assert g._artifact_exceedance_scale is True
    assert g._exceedance_scale_served() is True
    assert ItemForecaster.exceedance_scale_enabled() is False


# --- OOF honesty (source guard; real proof is the Task 3 controller run) --


def test_cv_loop_fits_a_per_fold_exceedance_head_for_calibration():
    """q_hat must be calibrated on OUT-OF-FOLD p_exceed: the served head trains on
    rows that overlap the calibration folds, so re-using it in-sample would shrink
    the scale and under-cover. The CV loop fits a per-fold head and passes its held-out
    predictions into the records."""
    src = inspect.getsource(ItemForecaster._cv_evaluate_horizon)
    assert "exceed_p=" in src
    assert "_fit_exceedance_classifier(" in src
