"""FEATURE_NATIVE_NAN: leave NaN features for LightGBM's native handling instead
of median-imputing them.

The imputation this replaces fills a missing feature with the TRAINING
cross-sectional median. For the long-return features that median is strongly
positive (return_180d ~6.3, 120d ~4.2, 90d ~2.8), so a newly-eligible /
short-history item is served a coherent multi-month uptrend it never had — a
bullish prior on the q50 centre for exactly the growing backfilled universe.
This is deep-model-review §10.4.

Like naive_init_score, TRAIN and SERVE are a matched pair: a booster trained on
imputed frames never learned a NaN default-direction, so serving must follow the
artifact, not the environment. These guard the flag, the impute helper (off =
byte-identical fillna, on = pass-through), and the artifact-over-environment
serving rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from models.forecaster import ItemForecaster


def _mk(monkeypatch, env=None):
    if env is None:
        monkeypatch.delenv("FEATURE_NATIVE_NAN", raising=False)
    else:
        monkeypatch.setenv("FEATURE_NATIVE_NAN", env)
    return ItemForecaster(db_session=MagicMock())


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("FEATURE_NATIVE_NAN", raising=False)
    assert ItemForecaster.feature_native_nan_enabled() is False


def test_flag_on(monkeypatch):
    monkeypatch.setenv("FEATURE_NATIVE_NAN", "1")
    assert ItemForecaster.feature_native_nan_enabled() is True


def test_impute_off_is_fillna(monkeypatch):
    """Off: byte-identical to X.fillna(medians)."""
    f = _mk(monkeypatch, env=None)
    X = pd.DataFrame({"return_180d": [np.nan, 1.0], "return_1d": [0.5, np.nan]})
    medians = pd.Series({"return_180d": 6.3, "return_1d": 0.05})
    out = f._impute_features(X, medians)
    expected = X.fillna(medians)
    pd.testing.assert_frame_equal(out, expected)
    # the bullish prior is present when off
    assert out.loc[0, "return_180d"] == 6.3


def test_impute_on_passes_nan_through(monkeypatch):
    """On: NaN is preserved for LightGBM; no bullish prior injected."""
    f = _mk(monkeypatch, env="1")
    X = pd.DataFrame({"return_180d": [np.nan, 1.0]})
    medians = pd.Series({"return_180d": 6.3})
    out = f._impute_features(X, medians)
    assert np.isnan(out.loc[0, "return_180d"])


def test_serving_follows_artifact_not_env(monkeypatch):
    """A model trained with the flag ON must keep passing NaN through even if the
    serving process has the env var unset (and vice versa)."""
    monkeypatch.delenv("FEATURE_NATIVE_NAN", raising=False)
    f = ItemForecaster(db_session=MagicMock())
    # artifact says it was trained native-nan; env says off
    f._artifact_feature_native_nan = True
    assert f._feature_native_nan_served() is True

    f._artifact_feature_native_nan = False
    monkeypatch.setenv("FEATURE_NATIVE_NAN", "1")
    assert f._feature_native_nan_served() is False


def test_serving_falls_back_to_env_when_artifact_silent(monkeypatch):
    """An older artifact with no flag stored falls back to the environment."""
    f = ItemForecaster(db_session=MagicMock())
    f._artifact_feature_native_nan = None
    monkeypatch.setenv("FEATURE_NATIVE_NAN", "1")
    assert f._feature_native_nan_served() is True
    monkeypatch.delenv("FEATURE_NATIVE_NAN", raising=False)
    assert f._feature_native_nan_served() is False


def test_served_helper_uses_artifact_when_served_true(monkeypatch):
    """_impute_features(served=True) reads the artifact flag; served=False reads
    the environment."""
    monkeypatch.delenv("FEATURE_NATIVE_NAN", raising=False)
    f = ItemForecaster(db_session=MagicMock())
    f._artifact_feature_native_nan = True
    X = pd.DataFrame({"return_180d": [np.nan]})
    medians = pd.Series({"return_180d": 6.3})
    # served path follows the (native) artifact -> NaN preserved
    assert np.isnan(f._impute_features(X, medians, served=True).loc[0, "return_180d"])
    # train path follows the (off) environment -> imputed
    assert f._impute_features(X, medians, served=False).loc[0, "return_180d"] == 6.3
