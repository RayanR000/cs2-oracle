"""CV trains the centre on the objective production serves.

`CENTRE_OBJECTIVE` (default `30:regression`) sets the served centre's loss per
horizon, but the CV fold loop forced `objective="quantile"` at every horizon.
The OOF residuals behind `q_hat` and the calibrators therefore came from an MAE
model while the served 30d centre was trained on MSE, so the band was
calibrated on a model nobody served (performance review 2026-10-08).
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import lightgbm as lgb
import models.forecaster as fc
import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster


@pytest.fixture(autouse=True)
def _ml_heads_off(monkeypatch):
    """The minimal frame below has no exceedance/anomaly targets."""
    monkeypatch.setenv("EXCEEDANCE_HEAD", "0")
    monkeypatch.setenv("ANOMALY_GBM", "0")


def _forecaster(tmp_path):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.QUANTILES = [0.5]
    f.feature_cols = ["feat_a"]
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 15
    f.VALIDATION_WINDOW_DAYS = 10
    return f


def _tdf(horizon=3, n_items=20, n_dates=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for d in range(n_dates):
            feat = rng.normal()
            rows.append(
                {
                    "item_id": f"i{i}",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "price": 5.0,
                    "price_tier": 1,
                    "price_std_60d": 1.0,
                    "feat_a": feat,
                    "return_1d": rng.normal(),
                    f"target_return_{horizon}d": feat + rng.normal(scale=0.5),
                }
            )
    return pd.DataFrame(rows)


def _cv_objectives(tmp_path, monkeypatch, objective_map, horizon=3):
    """Run CV and return the objective params of every centre fit it made."""
    monkeypatch.setattr(fc, "CENTRE_OBJECTIVE_MAP", objective_map)
    seen = []
    real_train = lgb.train

    def spy(params, *args, **kwargs):
        seen.append({k: params.get(k) for k in ("objective", "alpha", "metric")})
        return real_train(params, *args, **kwargs)

    monkeypatch.setattr(fc.lgb, "train", spy)
    _forecaster(tmp_path)._cv_evaluate_horizon(_tdf(horizon), horizon, {0.5: {}})
    assert seen, "CV trained no centre model"
    return seen


def test_cv_trains_a_regression_centre_where_serving_does(tmp_path, monkeypatch):
    seen = _cv_objectives(tmp_path, monkeypatch, {3: "regression"})
    assert all(p["objective"] == "regression" and p["metric"] == "l2" for p in seen), seen
    assert all(p["alpha"] is None for p in seen), "a stray quantile alpha rode along"


def test_cv_trains_a_huber_centre_where_serving_does(tmp_path, monkeypatch):
    seen = _cv_objectives(tmp_path, monkeypatch, {3: "huber"})
    assert all(p == {"objective": "huber", "alpha": 1.0, "metric": "huber"} for p in seen), seen


def test_an_unmentioned_horizon_keeps_the_quantile_centre(tmp_path, monkeypatch):
    seen = _cv_objectives(tmp_path, monkeypatch, {30: "regression"})
    assert all(p == {"objective": "quantile", "alpha": 0.5, "metric": "quantile"} for p in seen), seen


@pytest.mark.parametrize("objective", ["quantile", "regression", "huber"])
def test_cv_and_the_production_fit_share_one_definition(objective, monkeypatch):
    """Every site that sets the centre's loss reads the same helper, so the
    next site cannot drift the way the CV loop did."""
    monkeypatch.setattr(fc, "CENTRE_OBJECTIVE_MAP", {7: objective})
    params = ItemForecaster._centre_objective_params(7, 0.5)
    expected = {
        "quantile": {"objective": "quantile", "alpha": 0.5, "metric": "quantile"},
        "regression": {"objective": "regression", "metric": "l2"},
        "huber": {"objective": "huber", "alpha": 1.0, "metric": "huber"},
    }[objective]
    assert params == expected
