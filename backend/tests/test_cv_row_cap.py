"""CV folds must respect a row cap the way the production split does.

`max_rows` was applied only in `_build_production_split`; `_cv_evaluate_horizon`
took the whole expanding window every fold, so nine folds per horizon summed to
4.2x the training frame and the conformal CV phase was 50.4% of an 872s retrain.

These tests pin the cap and, more importantly, pin what it must NOT touch: the
fold count, the validation rows, and the OOF record count are the sample size of
q_hat, mean_rank_ic and the PT statistic.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def _forecaster(tmp_path, feature_cols):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.QUANTILES = [0.5]
    f.feature_cols = list(feature_cols)
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 15
    f.VALIDATION_WINDOW_DAYS = 10
    return f


def _frame(n_items=40, n_dates=120, horizon=3, seed=7):
    """A tdf in the shape _cv_evaluate_horizon consumes, big enough that a
    small cap actually binds on the later expanding-window folds."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-01")
    for item in range(n_items):
        for d in range(n_dates):
            rows.append({
                "item_id": f"item-{item}",
                "date": start + pd.Timedelta(days=d),
                # _cv_evaluate_horizon reads val_df["price"] to build the
                # conformal records; it is not a feature.
                "price": 10.0 + rng.normal(scale=0.5),
                "f0": rng.normal(),
                "f1": rng.normal(),
                "price_tier": 2,
                f"target_return_{horizon}d": rng.normal(scale=0.05),
            })
    return pd.DataFrame(rows)


def test_cap_default_is_300k():
    assert ItemForecaster.CV_MAX_TRAIN_ROWS == 300_000


def test_env_overrides_the_cap(tmp_path):
    f = _forecaster(tmp_path, ["f0", "f1"])
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "1234"}):
        assert f._cv_max_train_rows() == 1234


def test_fold_train_rows_are_capped(tmp_path):
    """Every fold's training frame is at or under the cap."""
    f = _forecaster(tmp_path, ["f0", "f1"])
    seen = []
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "500"}):
        with patch.object(ItemForecaster, "_record_cv_fold_train_rows",
                          side_effect=seen.append, create=True):
            f._cv_evaluate_horizon(_frame(), 3, {0.5: {"objective": "quantile"}})
    assert seen, "no folds ran"
    assert max(seen) <= 500


def test_cap_does_not_change_fold_count_or_oof_rows(tmp_path):
    """The cap thins training rows only. Fold count and OOF record count are the
    sample size of q_hat, rank IC and PT — if either moves, the cap hit the
    wrong axis."""
    frame = _frame()
    params = {0.5: {"objective": "quantile", "num_leaves": 7, "verbosity": -1}}

    f_uncapped = _forecaster(tmp_path / "a", ["f0", "f1"])
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "100000000"}):
        oof_a, folds_a = f_uncapped._cv_evaluate_horizon(frame, 3, params)[:2]

    f_capped = _forecaster(tmp_path / "b", ["f0", "f1"])
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "500"}):
        oof_b, folds_b = f_capped._cv_evaluate_horizon(frame, 3, params)[:2]

    assert len(folds_a) == len(folds_b)
    assert len(oof_a) == len(oof_b)


def test_only_train_is_thinned_never_val(tmp_path):
    """Thinning val would move the evaluation cohort, which is the artifact
    pairing exists to remove."""
    frame = _frame()
    params = {0.5: {"objective": "quantile", "num_leaves": 7, "verbosity": -1}}
    f = _forecaster(tmp_path, ["f0", "f1"])
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "500"}):
        _, fold_metrics = f._cv_evaluate_horizon(frame, 3, params)[:2]
    # Every fold's validation frame is the full 10-day window x 40 items.
    for m in fold_metrics:
        assert m["n_val"] == 400
        # And the cap did bind on train, so this is not a vacuous pass.
        assert m["n_train"] <= 500
