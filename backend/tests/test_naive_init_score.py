"""Tests for N1: fitting the q50 on top of `-return_1d` via LightGBM `init_score`.

The measured gap this closes is rank IC, not DA: the model loses to ranking by
minus yesterday's return at all four horizons (-0.0159 / -0.0371 / -0.0433 /
-0.0091, `docs/changelog/2026-08-10-post-revote-retrain.md`), and that baseline
uses no hindsight. With the offset the model fits the RESIDUAL to the naive
predictor instead of competing with it.

Two failure modes are worth more than the rest, and both have a test here:

1. **The offset is applied in training but not on the predict path.** The served
   mid then silently loses the baseline and is a residual quoted as a level.
   `test_a_signal_free_model_reproduces_the_naive_baselines_ranking` pins the
   floor property end to end through a real fit; `test_predict_adds_the_offset_
   back_to_the_served_mid` pins the serving arithmetic directly.
2. **The instrument turns itself on, or a served artifact disagrees with the
   environment about whether it is on.** Same hazard as the tier-lead and
   rank-transform instruments — see `tests/test_tier_lead_and_xs_rank.py`.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster


def _fc():
    return ItemForecaster.__new__(ItemForecaster)


# ----------------------------------------------------------------------
# Gating
# ----------------------------------------------------------------------


def test_the_instrument_is_off_by_default(monkeypatch):
    monkeypatch.delenv("NAIVE_INIT_SCORE", raising=False)
    assert ItemForecaster.naive_init_score_enabled() is False


def test_only_the_exact_flag_value_enables(monkeypatch):
    for value in ("0", "", "true", "yes", "TRUE", "2"):
        monkeypatch.setenv("NAIVE_INIT_SCORE", value)
        assert ItemForecaster.naive_init_score_enabled() is False
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    assert ItemForecaster.naive_init_score_enabled() is True


# ----------------------------------------------------------------------
# The offset itself
# ----------------------------------------------------------------------


def test_offset_is_minus_return_1d(monkeypatch):
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    df = pd.DataFrame({"return_1d": [2.0, -3.5, 0.0]})
    assert _fc()._naive_offset(df).tolist() == [-2.0, 3.5, 0.0]


def test_offset_is_in_percent_units_like_the_target(monkeypatch):
    """`target_return_{h}d` is `(target - price) / price * 100` (prepare_targets)
    and `return_{lag}d` is built the same way, so the offset needs no rescaling.
    An offset in fractions against a percent target would be 100x too small and
    the read would look like a null result rather than a units bug.
    """
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    # A 5% move up yesterday.
    df = pd.DataFrame({"return_1d": [5.0]})
    assert _fc()._naive_offset(df)[0] == pytest.approx(-5.0)


def test_offset_is_none_when_the_instrument_is_off(monkeypatch):
    monkeypatch.delenv("NAIVE_INIT_SCORE", raising=False)
    df = pd.DataFrame({"return_1d": [2.0]})
    assert _fc()._naive_offset(df) is None


def test_missing_and_infinite_values_become_a_zero_offset(monkeypatch):
    """NaN reaches LightGBM as a NaN init_score and poisons every prediction.

    Zero is the right fill: it means "this row has no baseline view", which
    leaves the model's own output as the whole forecast for that row.
    """
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    df = pd.DataFrame({"return_1d": [np.nan, np.inf, -np.inf, 4.0]})
    off = _fc()._naive_offset(df)
    assert off.tolist() == [0.0, 0.0, 0.0, -4.0]
    assert np.isfinite(off).all()


def test_a_missing_column_raises_rather_than_silently_disabling(monkeypatch):
    """The whole point of N1 is a floor, and a silent None removes it.

    `return_1d` is in every allowlisted feature set, so its absence means the
    frame is wrong — which must stop the run, not quietly train a model without
    the offset it is being measured with.
    """
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    with pytest.raises(RuntimeError, match="return_1d"):
        _fc()._naive_offset(pd.DataFrame({"other": [1.0]}))


def test_offset_length_and_order_match_the_frame(monkeypatch):
    """init_score is positional against the Dataset's rows."""
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    df = pd.DataFrame({"return_1d": [1.0, 2.0, 3.0]}, index=[7, 99, 3])
    off = _fc()._naive_offset(df)
    assert off.tolist() == [-1.0, -2.0, -3.0]


# ----------------------------------------------------------------------
# Artifact over environment on the serving path
# ----------------------------------------------------------------------


def test_served_flag_follows_the_artifact_over_the_environment(monkeypatch):
    fc = _fc()
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    fc._artifact_naive_init = False
    assert fc._naive_init_score_served() is False

    monkeypatch.delenv("NAIVE_INIT_SCORE", raising=False)
    fc._artifact_naive_init = True
    assert fc._naive_init_score_served() is True


def test_no_artifact_falls_back_to_the_environment(monkeypatch):
    fc = _fc()
    fc._artifact_naive_init = None
    monkeypatch.delenv("NAIVE_INIT_SCORE", raising=False)
    assert fc._naive_init_score_served() is False
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    assert fc._naive_init_score_served() is True


def test_served_offset_ignores_the_environment_when_an_artifact_says_off(monkeypatch):
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    fc = _fc()
    fc._artifact_naive_init = False
    assert fc._naive_offset_served(pd.DataFrame({"return_1d": [1.0]})) is None


def test_the_flag_round_trips_through_meta_json(tmp_path, monkeypatch):
    """A warm retrain restores meta.json; predict must follow what it says.

    Saved with the instrument on and loaded with the environment off, the served
    flag has to come back True — otherwise the next daily run serves a residual
    model as a level model.
    """
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["return_1d"]
    f.feature_medians = pd.Series({"return_1d": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["naive_init_score"] is True

    monkeypatch.delenv("NAIVE_INIT_SCORE", raising=False)
    g = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    g.load_models()
    assert g._artifact_naive_init is True
    assert g._naive_init_score_served() is True
    assert ItemForecaster.naive_init_score_enabled() is False


def test_an_artifact_written_before_the_flag_existed_loads(tmp_path, monkeypatch):
    """Absence must read as None ("does not say"), not as False."""
    monkeypatch.delenv("NAIVE_INIT_SCORE", raising=False)
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["return_1d"]
    f.feature_medians = pd.Series({"return_1d": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()

    meta = json.loads((tmp_path / "meta.json").read_text())
    del meta["naive_init_score"]
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    g = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    g.load_models()
    assert g._artifact_naive_init is None


# ----------------------------------------------------------------------
# The serving arithmetic
# ----------------------------------------------------------------------


def test_predict_adds_the_offset_back_to_the_served_mid(monkeypatch):
    """`p50 + offset`, on the same rows, in the same order.

    Pinned as arithmetic rather than through `predict()` so the assertion is
    exact: a booster fitted on residuals emits residuals, and the level the
    product serves is the booster's output plus the baseline it was fitted over.
    """
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    fc = _fc()
    fc._artifact_naive_init = True
    rows = pd.DataFrame({"return_1d": [3.0, -1.0, 0.0]})
    residual = np.array([0.5, 0.5, -2.0])

    served = residual + fc._naive_offset_served(rows)
    assert served.tolist() == pytest.approx([-2.5, 1.5, -2.0])


# ----------------------------------------------------------------------
# The floor property, end to end through a real fit
# ----------------------------------------------------------------------


def _cv_frame(n_items=30, n_dates=90, horizon=3, seed=3):
    """A tdf `_cv_evaluate_horizon` consumes, with SIGNAL-FREE features.

    Every feature is constant, so LightGBM cannot split and each tree emits one
    leaf value: the prediction is `offset + c`. Within-date ranks of `offset + c`
    are the ranks of the offset, so the model's rank IC must equal the naive
    baseline's exactly. That is the floor N1 buys, and it is only reachable if
    the offset is applied on BOTH the training and the prediction side.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    rows = []
    for item in range(n_items):
        for d in range(n_dates):
            rows.append(
                {
                    "item_id": f"item-{item}",
                    "date": start + pd.Timedelta(d, unit="D"),
                    "price": 10.0,
                    "constant_a": 1.0,
                    "constant_b": 0.0,
                    # The offset column: varies across items within a date, which is
                    # what makes a within-date rank IC defined at all.
                    "return_1d": float(rng.normal()),
                    # >= HEADLINE_MIN_TIER so the served cohort is non-empty.
                    "price_tier": 2,
                    f"target_return_{horizon}d": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


def _cv_forecaster(tmp_path):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.QUANTILES = [0.5]
    # `return_1d` stays OUT of the feature set here: it is the offset column, and
    # a booster able to split on it would order items by itself and make the
    # assertion below untestable. `_minus_return_1d` reads the frame, not X.
    f.feature_cols = ["constant_a", "constant_b"]
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 15
    f.VALIDATION_WINDOW_DAYS = 10
    return f


def _run_cv(tmp_path, monkeypatch, horizon=3):
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "0")
    f = _cv_forecaster(tmp_path)
    params = {0.5: {"num_leaves": 7, "learning_rate": 0.05, "verbosity": -1}}
    _, fold_metrics, _, _ = f._cv_evaluate_horizon(_cv_frame(horizon=horizon), horizon, params)
    return fold_metrics


def test_a_signal_free_model_reproduces_the_naive_baselines_ranking(tmp_path, monkeypatch):
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    fold_metrics = _run_cv(tmp_path, monkeypatch)

    assert fold_metrics, "CV produced no folds"
    for fold in fold_metrics:
        assert fold["naive_rank_ic"] is not None
        assert fold["rank_ic"] == pytest.approx(fold["naive_rank_ic"], abs=1e-9), (
            f"fold {fold['fold']}: a model with no usable features must rank "
            f"exactly like -return_1d once the offset is added back "
            f"({fold['rank_ic']} vs {fold['naive_rank_ic']}) — the offset is "
            f"probably missing from the prediction side"
        )


def _train_frame(n_items=10, n_dates=140, seed=5):
    """A frame in the shape `build_training_data` returns."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-01")
    for item in range(n_items):
        base = 10.0 * (item + 1)
        price = base
        for d in range(n_dates):
            price = max(price * (1.0 + rng.normal(0.001, 0.02)), 0.5)
            rows.append(
                {
                    "item_id": f"item_{item}",
                    "date": (start + pd.Timedelta(d, unit="D")).date(),
                    "price": price,
                    "price_std_60d": abs(rng.normal(base * 0.05, base * 0.01)),
                    "return_1d": float(rng.normal(scale=2.0)),
                    "feat_a": rng.normal(),
                    "feat_b": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("skip_regimes", [True, False])
def test_a_real_train_runs_with_the_instrument_on(tmp_path, monkeypatch, skip_regimes):
    """Every `lgb.Dataset` on the training path has to accept the offset.

    There are four of them (global, regime, CV, Optuna) plus the holdout
    calibration fallback, and a missed one is not a crash — it is a booster
    trained on a different target than its siblings. `skip_regimes=False`
    exercises the regime branch, which `predict` PREFERS over the global model.
    """
    monkeypatch.setenv("NAIVE_INIT_SCORE", "1")
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "0")
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.N_ENSEMBLES = 1
    f.SKIP_HP_HORIZONS = list(f.HORIZONS)
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 25
    f.VALIDATION_WINDOW_DAYS = 10

    df = _train_frame()

    def fake_build(*a, **kw):
        f.feature_cols = ["feat_a", "feat_b"]
        f._base_feature_cols = list(f.feature_cols)
        return df.copy()

    monkeypatch.setattr(f, "build_training_data", fake_build)
    monkeypatch.setattr(f, "save_models", lambda *a, **kw: None)
    if skip_regimes:
        monkeypatch.setenv("SKIP_REGIMES", "1")
    else:
        monkeypatch.delenv("SKIP_REGIMES", raising=False)

    f.train()

    assert f.models, "training produced no models"
    # The band still has to be calibrated: q_hat is fitted on residuals to the
    # offset mid, so a missing offset on the CV prediction side would move it.
    for h in f.HORIZONS:
        assert f.conformal_calibration.get(h) is not None
        assert np.isfinite(f.conformal_calibration[h])


def test_without_the_instrument_the_same_model_has_no_ranking_at_all(tmp_path, monkeypatch):
    """The control for the test above, so it cannot pass vacuously.

    With no offset and no usable features every prediction in a fold is the same
    number, so within-date Spearman is undefined and rank IC is None.
    """
    monkeypatch.delenv("NAIVE_INIT_SCORE", raising=False)
    fold_metrics = _run_cv(tmp_path, monkeypatch)

    assert fold_metrics, "CV produced no folds"
    assert all(f["rank_ic"] is None for f in fold_metrics), (
        "a constant prediction cannot have a within-date rank IC; if this "
        "passes a value, the fixture is no longer signal-free"
    )
