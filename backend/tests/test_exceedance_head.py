"""EXCEEDANCE_HEAD (Phase A): the exceedance probability P(move clears cost) is
trained and SERVED as a disclosed per-item output, independent of whether the band
uses it as a width scale.

Before this, the head was fit only under EXCEEDANCE_SCALE and its probability was
consumed as a band denominator and thrown away — so with the (winning) climatology
scale live, no `exceed_p` was produced at all. Phase A decouples the head's
computation from the band-scale flag and emits `exceed_p` on the predict record.
The probability is a MAGNITUDE signal, never a directional call (invariant 4).

These guard: the flag, the flag-independent serving accessor, the training gate,
that band_scale reuses the one accessor, and that predict emits the field. Scope:
docs/plans/2026-08-16-exceedance-band-scale-phase2-plan.md.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from models.forecaster import ItemForecaster

from tests._source import method_closure_source


def _forecaster(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


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


def test_head_flag_reads_the_environment(monkeypatch):
    monkeypatch.setenv("EXCEEDANCE_HEAD", "0")
    assert ItemForecaster.exceedance_head_enabled() is False
    monkeypatch.setenv("EXCEEDANCE_HEAD", "1")
    assert ItemForecaster.exceedance_head_enabled() is True


# --- the flag-independent serving accessor --------------------------------


def test_exceedance_probability_is_clipped_and_flag_independent(tmp_path, monkeypatch):
    """The served probability comes from the loaded head regardless of BOTH scale
    flags: it is a disclosed output, not the band denominator. Clipped to (1e-3, 1]."""
    monkeypatch.setenv("EXCEEDANCE_SCALE", "0")
    monkeypatch.setenv("EXCEEDANCE_HEAD", "0")
    f = _forecaster(tmp_path)
    f.exceedance_models = {7: _exc_head(f)}
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f._artifact_exceedance_scale = False  # band is NOT on the exceedance scale

    rng = np.random.default_rng(1)
    rows = pd.DataFrame({"f": rng.normal(size=40), "g": rng.normal(size=40)})
    p = f.exceedance_probability(7, rows)

    expected = np.clip(f.exceedance_models[7].predict(rows[["f", "g"]]), 1e-3, 1.0)
    np.testing.assert_allclose(p, expected)
    assert p.min() >= 1e-3 and p.max() <= 1.0


def test_exceedance_probability_none_without_head(tmp_path):
    """No head for the horizon (degenerate horizon / pre-Phase-2 artifact) -> None,
    so the caller emits a null field rather than a fabricated probability."""
    f = _forecaster(tmp_path)
    f.exceedance_models = {}
    f.feature_medians = pd.Series(dtype=float)
    rows = pd.DataFrame({"f": [0.1], "g": [0.2]})
    assert f.exceedance_probability(7, rows) is None


def test_exceedance_probability_fills_missing_features_with_medians(tmp_path):
    """A served row missing a head feature is filled from feature_medians, not left
    NaN — mirrors the band_scale path's reindex/fillna."""
    f = _forecaster(tmp_path)
    f.exceedance_models = {7: _exc_head(f)}
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    rows = pd.DataFrame({"f": [0.5]})  # "g" absent
    p = f.exceedance_probability(7, rows)
    assert p is not None and np.isfinite(p).all()


# --- band_scale reuses the one accessor -----------------------------------


def test_band_scale_reuses_exceedance_probability(tmp_path):
    src = inspect.getsource(ItemForecaster.band_scale)
    assert "exceedance_probability(" in src


# --- training gate --------------------------------------------------------


def test_training_fits_head_under_head_flag_without_scale():
    """The production head must train when EXCEEDANCE_HEAD is set even if the band
    uses another scale — otherwise no artifact carries a head to serve exceed_p from."""
    src = method_closure_source(ItemForecaster, "_train_horizon_inline")
    assert "exceedance_head_enabled" in src


# --- predict emits the field ----------------------------------------------


def test_predict_emits_exceed_p_on_forecast_record():
    src = method_closure_source(ItemForecaster, "predict")
    assert "exceedance_probability(" in src
    assert '"exceed_p"' in src
