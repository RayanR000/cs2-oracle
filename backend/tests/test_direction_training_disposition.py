"""Routine production training must not fit, persist, restore, or require
the three-class direction classifier.

The model remains reproducible offline through scripts/direction_benchmark.py
while the API continues returning neutral through the serving policy.
"""

import inspect

import models.forecaster as fc_module


def test_routine_train_never_fits_direction():
    src = inspect.getsource(fc_module.ItemForecaster._train_horizon_inline)
    assert "_fit_direction_classifier" not in src


def test_save_never_persists_direction():
    src = inspect.getsource(fc_module.ItemForecaster.save_models)
    assert "direction_models" not in src


def test_load_never_restores_direction():
    src = inspect.getsource(fc_module.ItemForecaster.load_models)
    assert "direction_models" not in src


def test_predict_never_consults_direction():
    src = inspect.getsource(fc_module.ItemForecaster.predict)
    assert "direction_models" not in src


def test_legacy_artifact_loads_direction_free(tmp_path):
    """A legacy directory containing clf_7d.txt loads with no direction state."""
    from unittest.mock import MagicMock

    import pandas as pd

    f = fc_module.ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {}
    f.save_models()
    (tmp_path / "clf_7d.txt").write_text("legacy direction booster (not a real model)")
    g = fc_module.ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    g.load_models()
    assert g.direction_models == {}


def test_api_direction_remains_neutral():
    from api.serving_policy import DIRECTION_DISCLOSED, served_direction

    assert DIRECTION_DISCLOSED is False
    assert served_direction("up", 7) == "neutral"
