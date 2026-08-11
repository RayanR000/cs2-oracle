"""Shape guarantees for the 8-model forecaster.

The band now comes from conformal calibration around a single median model,
not from 24 p10/p90 GBMs. These tests guard the properties that made that
safe: item-varying width, a finite band for short-history items, ordering
without the crossing fix, and an artifact that cannot be loaded by the wrong
code version.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import inspect
import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from models import conformal
from models.conformal import ALPHA
from models.forecaster import ItemForecaster


def test_sigma_clip_defaults_are_present_and_finite():
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    assert set(f.sigma_clip) == {"floor", "cap", "fallback"}
    assert 0 < f.sigma_clip["floor"] < f.sigma_clip["cap"]
    assert np.isfinite(f.sigma_clip["fallback"])


def test_alpha_matches_the_pinned_nominal_coverage():
    assert ALPHA == pytest.approx(0.20)


def test_sigma_for_rows_uses_persisted_clip_bounds():
    """Sigma must be clipped with the bounds q_hat was calibrated against.

    A bare std/price at a use site would let a serving row sit outside the
    clip range q_hat saw, which silently voids the coverage guarantee.
    """
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.sigma_clip = {"floor": 0.02, "cap": 0.30, "fallback": 0.11}

    rows = pd.DataFrame({
        # std/price = 0.001 (below floor), 0.5 (above cap), 0.1 (in range)
        "price_std_60d": [0.1, 50.0, 1.0],
        "price": [100.0, 100.0, 10.0],
    })
    sigma = f._sigma_for_rows(rows)
    assert sigma.tolist() == [0.02, 0.30, 0.1]


def test_sigma_for_rows_never_returns_nan():
    """A NaN reaching forecast_low would surface in the UI.

    PREDICT_MIN_HISTORY_DAYS = 14 means eligible items can carry a NaN or 0
    std, and price can be 0 in a corrupt archive row.
    """
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.sigma_clip = {"floor": 0.02, "cap": 0.30, "fallback": 0.11}

    rows = pd.DataFrame({
        "price_std_60d": [np.nan, 0.0, 1.0],
        "price": [100.0, 100.0, 0.0],
    })
    sigma = f._sigma_for_rows(rows)
    assert np.isfinite(sigma).all()
    assert sigma.tolist() == [0.11, 0.11, 0.11]


def test_sigma_for_rows_tolerates_a_missing_std_column():
    """Frames without price_std_60d fall back rather than KeyError."""
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.sigma_clip = {"floor": 0.02, "cap": 0.30, "fallback": 0.11}

    sigma = f._sigma_for_rows(pd.DataFrame({"price": [100.0, 50.0]}))
    assert sigma.tolist() == [0.11, 0.11]


# ---------------------------------------------------------------------------
# Two-pass calibration: q_hat first, then the width it implies
# ---------------------------------------------------------------------------


def _records(n=400, seed=0):
    """Pooled OOF records in the shape _cv_evaluate_horizon emits."""
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.05, 0.25, n)
    resid = rng.normal(0.0, 1.0, n) * sigma * 20.0
    mid = rng.normal(0.0, 2.0, n)
    return pd.DataFrame({
        "mid_ret": mid,
        "residual_pct": resid,
        "sigma": sigma,
        "change_pct": np.abs(mid) / 100.0,
        "hit": rng.integers(0, 2, n).astype(float),
    })


def test_calibrate_conformal_stores_q_hat_and_adds_numeric_range_pct():
    """`_calibrate_confidence` thresholds on range_pct and cannot take a None.

    The width is only knowable after q_hat, so it is derived in a second pass;
    this asserts the second pass actually happens and leaves a finite column.
    """
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.conformal_calibration = {}
    # `__init__` sets this; `__new__` does not. _calibrate_conformal reads it to
    # decide whether a q50-centred q_hat is about to be recentred out from under.
    f.direction_models = {}

    df = _records()
    q_hat = f._calibrate_conformal(7, df)

    assert f.conformal_calibration[7] == q_hat
    assert q_hat > 0
    assert "range_pct" in df.columns
    assert df["range_pct"].notna().all()
    assert np.isfinite(df["range_pct"].to_numpy()).all()
    assert (df["range_pct"] > 0).all()


def test_range_pct_varies_with_item_sigma():
    """The whole point of normalizing: width is item-specific, not global."""
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.conformal_calibration = {}
    # `__init__` sets this; `__new__` does not. _calibrate_conformal reads it to
    # decide whether a q50-centred q_hat is about to be recentred out from under.
    f.direction_models = {}

    df = _records()
    f._calibrate_conformal(7, df)
    corr = np.corrcoef(df["sigma"], df["range_pct"])[0, 1]
    assert corr > 0.9


def test_calibrated_band_covers_about_the_nominal_rate():
    """q_hat * sigma should bracket ~80% of the OOF residuals it was fit on."""
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.conformal_calibration = {}
    # `__init__` sets this; `__new__` does not. _calibrate_conformal reads it to
    # decide whether a q50-centred q_hat is about to be recentred out from under.
    f.direction_models = {}

    df = _records(n=2000, seed=7)
    q_hat = f._calibrate_conformal(3, df)
    half = q_hat * df["sigma"].to_numpy()
    covered = (np.abs(df["residual_pct"].to_numpy()) <= half).mean()
    assert covered == pytest.approx(1.0 - ALPHA, abs=0.02)


def test_calibrated_records_are_accepted_by_calibrate_confidence():
    """End of the chain: the records must actually fit thresholds, not crash."""
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.conformal_calibration = {}
    # `__init__` sets this; `__new__` does not. _calibrate_conformal reads it to
    # decide whether a q50-centred q_hat is about to be recentred out from under.
    f.direction_models = {}
    f.confidence_thresholds = {}

    df = _records(n=1000, seed=3)
    f._calibrate_conformal(7, df)
    f._calibrate_confidence(horizon=7, records_df=df)

    th = f.confidence_thresholds[7]
    assert np.isfinite(th["high_range"])
    # high_accuracy is the discriminating assertion: the silent-default path
    # leaves high_range at the hardcoded 0.15 (finite and > 0, so those two
    # assertions alone would pass) with high_accuracy at 0.0.
    assert th["high_accuracy"] > 0


# ---------------------------------------------------------------------------
# The blocker: every CV fold must reach the record loop with QUANTILES == [0.5]
# ---------------------------------------------------------------------------


def _cv_frame(n_items=8, n_dates=80, horizon=3, seed=11):
    """Synthetic tdf in the shape _cv_evaluate_horizon consumes."""
    rng = np.random.default_rng(seed)
    rows = []
    for item in range(n_items):
        base = 10.0 * (item + 1)
        for d in range(n_dates):
            price = base * (1.0 + 0.002 * d) + rng.normal(0, base * 0.02)
            rows.append({
                "item_id": f"item_{item}",
                "date": date(2025, 1, 1) + timedelta(days=d),
                "price": max(price, 0.5),
                "price_std_60d": abs(rng.normal(base * 0.05, base * 0.01)),
                "feat_a": rng.normal(),
                "feat_b": rng.normal(),
                f"target_return_{horizon}d": rng.normal(0, 3.0),
            })
    return pd.DataFrame(rows)


def _cv_forecaster(tmp_path, quantiles):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.QUANTILES = quantiles
    f.feature_cols = ["feat_a", "feat_b"]
    # Shrink the expanding window so a synthetic frame yields >= 2 folds.
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 15
    f.VALIDATION_WINDOW_DAYS = 10
    return f


@pytest.mark.parametrize("quantiles", [[0.5], [0.1, 0.5, 0.9]])
def test_every_cv_fold_produces_oof_records(tmp_path, quantiles):
    """The load-bearing guard.

    A guard on fold_p10/fold_p90 would `continue` on every fold once
    QUANTILES == [0.5], leaving oof_records empty. Training would then look
    successful while conformal_calibration stayed {} and predict() could not
    build a band. Parametrized so this holds both before and after Task 11
    collapses the quantile grid.
    """
    f = _cv_forecaster(tmp_path, quantiles)
    tdf = _cv_frame(horizon=3)

    result = f._cv_evaluate_horizon(tdf, 3, {q: {} for q in quantiles})
    assert len(result) == 4, (
        "return tuple must be "
        "(oof_records, fold_metrics, pt_records, pt_records_clf)")
    oof_records, fold_metrics, pt_records, _ = result

    assert len(fold_metrics) >= 2
    assert oof_records, "no OOF records — calibration would be skipped"
    # One record per validation row of every fold that ran.
    assert len(oof_records) == sum(m["n_val"] for m in fold_metrics)

    for key in ("mid_ret", "residual_pct", "sigma", "change_pct", "hit"):
        assert key in oof_records[0]
    assert all(np.isfinite(r["sigma"]) and r["sigma"] > 0 for r in oof_records)
    assert all(np.isfinite(r["residual_pct"]) for r in oof_records)


def test_cv_records_calibrate_end_to_end_with_a_median_only_grid(tmp_path):
    """CV -> q_hat -> range_pct -> confidence thresholds, median model only."""
    f = _cv_forecaster(tmp_path, [0.5])
    tdf = _cv_frame(horizon=3)

    oof_records = f._cv_evaluate_horizon(tdf, 3, {0.5: {}})[0]
    records_df = pd.DataFrame(oof_records)
    q_hat = f._calibrate_conformal(3, records_df)
    f._calibrate_confidence(horizon=3, records_df=records_df)

    assert f.conformal_calibration[3] == q_hat
    assert q_hat > 0
    assert records_df["range_pct"].notna().all()
    assert np.isfinite(f.confidence_thresholds[3]["high_range"])


# ---------------------------------------------------------------------------
# F1: q_hat must be calibrated around the mid predict() actually serves
#
# `predict` builds the band around the q50 mid and then recentres the mid on
# the directional classifier's call, preserving half-widths. A q_hat fitted on
# residuals to the q50 mid therefore describes a band that is never served:
# on every row where the classifier disagrees with the q50's sign, the served
# centre is displaced by up to 2*|mid| while the width is unchanged.
# ---------------------------------------------------------------------------


def _served_centre_inputs(n=3000, seed=5, disagree_share=0.4):
    """q50 mids, honest actuals, and a classifier that disagrees on a share.

    `actual` is generated around the **q50** mid, so the recentring is pure
    displacement — which is the worst case for coverage and the one production
    is in.
    """
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.05, 0.25, n)
    mid = rng.normal(0.0, 4.0, n)
    actual = mid + rng.normal(0.0, 1.0, n) * sigma * 20.0
    price = np.full(n, 50.0)

    q50_class = ItemForecaster._direction_classes(mid)
    served_class = q50_class.copy()
    flip = rng.random(n) < disagree_share
    # 0 <-> 2 (down <-> up), 1 -> 1: a flat call pins the served mid to zero.
    served_class[flip] = 2 - q50_class[flip]
    return mid, actual, sigma, price, served_class


def test_conformal_records_measure_the_residual_against_the_served_centre():
    """The residual q_hat is fitted on must be to the recentred mid."""
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)

    mid, actual, sigma, price, served_class = _served_centre_inputs(n=200)
    records = pd.DataFrame(f._conformal_records(
        mid, actual, sigma, price, direction_class=served_class))
    assert len(records) == len(mid), "no row should be dropped in this fixture"

    _, served_mid, _ = ItemForecaster._recenter_on_direction(
        mid, mid, mid, served_class)
    assert records["residual_pct"].to_numpy() == pytest.approx(
        actual - served_mid)
    # The served centre is carried so the calibration can describe itself, and
    # it must differ from the q50 mid — otherwise the fixture proves nothing.
    assert records["served_mid_ret"].to_numpy() == pytest.approx(served_mid)
    assert (records["served_mid_ret"] != records["mid_ret"]).any()


def test_conformal_records_leave_the_confidence_columns_on_the_q50_mid():
    """`_calibrate_confidence` fits the no-classifier fallback path.

    That path serves the q50-derived direction and mid, so `hit`, `change_pct`
    and `mid_ret` must stay on the q50 centre even when the served centre is
    supplied. Only `residual_pct` moves.
    """
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)

    mid, actual, sigma, price, served_class = _served_centre_inputs(n=200)
    plain = pd.DataFrame(f._conformal_records(mid, actual, sigma, price))
    served = pd.DataFrame(f._conformal_records(
        mid, actual, sigma, price, direction_class=served_class))

    for col in ("mid_ret", "sigma", "change_pct", "hit"):
        assert served[col].to_numpy() == pytest.approx(plain[col].to_numpy())
    assert "served_mid_ret" not in plain.columns


def test_served_band_covers_nominal_on_a_held_out_split():
    """The falsifiable coverage check.

    Fits q_hat on one half and measures coverage of the band **as predict()
    assembles it** — `conformal.band` around the q50 mid, then
    `_recenter_on_direction` — on the other half. Measuring coverage against
    `residual_pct` on the rows q_hat was fitted on is >= nominal by
    construction and cannot fail.
    """
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.conformal_calibration = {}
    # `__init__` sets this; `__new__` does not. _calibrate_conformal reads it to
    # decide whether a q50-centred q_hat is about to be recentred out from under.
    f.direction_models = {}

    mid, actual, sigma, price, served_class = _served_centre_inputs(n=4000)
    half = len(mid) // 2
    fit, ev = slice(0, half), slice(half, None)

    records = pd.DataFrame(f._conformal_records(
        mid[fit], actual[fit], sigma[fit], price[fit],
        direction_class=served_class[fit]))
    q_hat = f._calibrate_conformal(7, records)

    low, high = conformal.band(mid[ev], sigma[ev], q_hat)
    s_low, _, s_high = ItemForecaster._recenter_on_direction(
        low, mid[ev], high, served_class[ev])
    covered = ((actual[ev] >= s_low) & (actual[ev] <= s_high)).mean()
    assert covered >= conformal.NOMINAL_COVERAGE - 0.02, (
        f"served band covers {covered:.1%} against a "
        f"{conformal.NOMINAL_COVERAGE:.0%} target")


def test_calibration_records_which_centre_it_fitted():
    """A band calibrated on the wrong centre must be visible, not silent.

    The served centre needs an out-of-fold direction call, which
    CV_DIAGNOSTIC_CLASSIFIER=0 does not produce — so production can legitimately
    end up on the q50 centre, and the artifact has to say so.
    """
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.conformal_calibration = {}
    # `__init__` sets this; `__new__` does not. _calibrate_conformal reads it to
    # decide whether a q50-centred q_hat is about to be recentred out from under.
    f.direction_models = {}

    mid, actual, sigma, price, served_class = _served_centre_inputs(n=600)
    f._calibrate_conformal(7, pd.DataFrame(f._conformal_records(
        mid, actual, sigma, price, direction_class=served_class)))
    f._calibrate_conformal(14, pd.DataFrame(f._conformal_records(
        mid, actual, sigma, price)))

    assert f.conformal_centre == {7: "served", 14: "q50"}


def test_calibrating_on_the_q50_centre_warns_when_a_classifier_will_recentre(caplog):
    """The defect must be loud in forecast.log, not inferable from meta.json.

    `predict` recentres whenever `direction_models[horizon]` exists, so that is
    exactly the condition under which a q50-centred q_hat is wrong.
    """
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.conformal_calibration = {}
    # A classifier for 7d exists, so predict() WILL recentre this horizon's mid.
    f.direction_models = {7: object()}

    mid, actual, sigma, price, _ = _served_centre_inputs(n=600)
    with caplog.at_level(logging.WARNING):
        f._calibrate_conformal(7, pd.DataFrame(
            f._conformal_records(mid, actual, sigma, price)))
    assert any("recentre" in r.message for r in caplog.records), caplog.text

    # And it must stay quiet when the two centres agree, or the warning is noise.
    caplog.clear()
    _, _, _, _, served_class = _served_centre_inputs(n=600)
    with caplog.at_level(logging.WARNING):
        f._calibrate_conformal(7, pd.DataFrame(f._conformal_records(
            mid, actual, sigma, price, direction_class=served_class)))
    assert not caplog.records, caplog.text


def test_cv_records_carry_the_served_centre_when_the_classifier_runs(tmp_path, monkeypatch):
    """The wiring: CV's own out-of-fold call is what q_hat needs.

    Without this the fix is inert on the production path — `_conformal_records`
    would accept a `direction_class` nobody passes.
    """
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "1")
    f = _cv_forecaster(tmp_path, [0.5])
    oof_records = f._cv_evaluate_horizon(_cv_frame(horizon=3), 3, {0.5: {}})[0]

    assert oof_records
    assert all("served_mid_ret" in r for r in oof_records)
    q_hat = f._calibrate_conformal(3, pd.DataFrame(oof_records))
    assert q_hat > 0
    assert f.conformal_centre[3] == "served"


def test_cv_records_fall_back_to_the_q50_centre_without_the_classifier(tmp_path, monkeypatch):
    """CV_DIAGNOSTIC_CLASSIFIER=0 is production's default and cannot crash.

    It produces no out-of-fold direction call, so the centre is honestly q50 —
    recorded as such rather than silently assumed to be served.
    """
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "0")
    f = _cv_forecaster(tmp_path, [0.5])
    oof_records = f._cv_evaluate_horizon(_cv_frame(horizon=3), 3, {0.5: {}})[0]

    assert oof_records
    assert not any("served_mid_ret" in r for r in oof_records)
    f._calibrate_conformal(3, pd.DataFrame(oof_records))
    assert f.conformal_centre[3] == "q50"


def test_save_then_load_round_trips_the_conformal_centre(tmp_path):
    """Which centre a served band was calibrated on has to survive predict-only.

    A predict-only run never recalibrates, so disk is its only route to this
    field — and reading a coverage number without it is what let an 80%-labelled
    band serve 34.6-61.8%.
    """
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["feat_a", "feat_b"]
    f.conformal_calibration = {3: 1.1, 7: 2.2}
    f.conformal_centre = {3: "served", 7: "q50"}
    f.save_models()

    g = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    g.load_models()
    assert g.conformal_centre == {3: "served", 7: "q50"}
    assert all(isinstance(h, int) for h in g.conformal_centre)


def test_load_models_tolerates_an_artifact_with_no_conformal_centre(tmp_path):
    """Provenance, not a serving input: absence means unknown, not fail-closed.

    Every artifact written before this field existed lacks it, and predict()
    reads the band from `conformal_calibration` alone.
    """
    import json

    (tmp_path / "meta.json").write_text(json.dumps({
        "model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION,
        "feature_cols": ["feat_a", "feat_b"],
        "sigma_clip": {"floor": 0.01, "cap": 2.0, "fallback": 0.15},
        "conformal_calibration": {"3": 1.1},
        "n_ensembles": 1,
    }))
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.load_models()
    assert f.conformal_centre == {}


# ---------------------------------------------------------------------------
# q_hat must be fitted, and fitted on data the model was not selected against
#
# Two separate guarantees:
#  1. Every horizon gets a q_hat. There used to be branches (SKIP_CV=1, warm
#     retrain) that finished a horizon without one, which leaves predict() with
#     no band.
#  2. It is fitted on out-of-fold CV records. X_val is the early-stopping dval
#     and the Optuna scoring set, so residuals measured there are optimistically
#     small and q_hat comes out biased low — the band would under-cover.
# ---------------------------------------------------------------------------


def _train_frame(n_items=10, n_dates=140, seed=5):
    """Minimal frame in the shape build_training_data returns."""
    rng = np.random.default_rng(seed)
    rows = []
    for item in range(n_items):
        base = 10.0 * (item + 1)
        price = base
        for d in range(n_dates):
            price = max(price * (1.0 + rng.normal(0.001, 0.02)), 0.5)
            rows.append({
                "item_id": f"item_{item}",
                "date": date(2025, 1, 1) + timedelta(days=d),
                "price": price,
                "price_std_60d": abs(rng.normal(base * 0.05, base * 0.01)),
                "feat_a": rng.normal(),
                "feat_b": rng.normal(),
            })
    return pd.DataFrame(rows)


def _fast_forecaster(tmp_path, warm=False):
    """A real ItemForecaster shrunk enough to train inside a unit test.

    Only cost knobs are touched. QUANTILES and N_ENSEMBLES keep their production
    values on the class; the instance overrides here are test-local and do not
    change what ships.

    `warm=True` seeds cached HP for every horizon/quantile, which is what makes
    reuse_hp — and therefore a warm retrain — true.
    """
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.N_ENSEMBLES = 1
    f.SKIP_HP_HORIZONS = list(f.HORIZONS)   # no Optuna
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 25
    f.VALIDATION_WINDOW_DAYS = 10
    if warm:
        f.tuned_params = {
            h: {q: {"num_leaves": 15, "learning_rate": 0.05, "max_depth": 4,
                    "min_data_in_leaf": 10, "objective": "quantile",
                    "alpha": q, "metric": "quantile", "verbosity": -1}
                for q in f.QUANTILES}
            for h in f.HORIZONS
        }
    return f


def _run_train(f, monkeypatch, df, skip_regimes=True):
    """Drive the real train() over a synthetic frame.

    build_training_data is the only thing stubbed, so train()'s own sigma-clip
    measurement and every per-horizon calibration branch run for real.

    `skip_regimes=False` exercises the regime branch, which is the one a warm
    retrain used to bypass entirely.
    """
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


def test_ci_workflow_does_not_skip_cv():
    """SKIP_CV=1 in CI is what put an in-sample q_hat behind the served band.

    Skipping CV routes calibration to the single holdout, which is also the
    early-stopping dval and the Optuna scoring set, so q_hat is biased low and
    the band under-covers. This pins the workflow so the flag cannot come back
    as a CI-minutes optimization.
    """
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "price-forecast.yml")
    spec = yaml.safe_load(wf.read_text())

    def env_keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "env" and isinstance(v, dict):
                    yield from v
                else:
                    yield from env_keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from env_keys(v)

    assert "SKIP_CV" not in set(env_keys(spec))
    # The run: blocks are strings, so an inline `SKIP_CV=1 python …` would slip
    # past the env walk above.
    assert "SKIP_CV" not in wf.read_text().replace("# SKIP_CV", "")


def test_ci_restores_the_model_cache_on_training_runs_and_forces_the_retrain():
    """The two halves of this have to move together.

    Gating the restore to predict-only left `tuned_params` empty on every
    training run, so Optuna re-searched from scratch — 692.9s of run
    31356483719. But un-gating it alone is worse than the cost: `full` reaches
    the age gate in forecast_prices.py, which retrains on model age (14d) only,
    so Monday's weekly retrain would find a 7-day-old restored artifact, read it
    as fresh, and silently serve a predict-only run. FORCE_RETRAIN is what keeps
    `full` a retrain. Pinned together so neither can be reverted on its own.
    """
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "price-forecast.yml")
    spec = yaml.safe_load(wf.read_text())
    steps = spec["jobs"]["forecast"]["steps"]

    restore = next(s for s in steps if s.get("name") == "Restore trained models")
    assert "if" not in restore, (
        "the model cache restore is conditional again; a training run with an "
        "empty saved_models/ cannot reuse hyperparameters"
    )

    run_step = next(s for s in steps if s.get("name") == "Run ML price forecasting")
    force = str(run_step["env"]["FORCE_RETRAIN"])
    assert "== 'full'" in force and "'1'" in force, (
        f"FORCE_RETRAIN is {force!r}; `full` must force the retrain or the "
        f"restored artifact trips the age gate and Monday trains nothing"
    )


def test_ci_skips_regime_models_so_the_artifact_does_not_depend_on_its_runner():
    """SKIP_REGIMES=1 in CI, deliberately, and it IS a served change.

    Two things this pins. First, that CI and the documented local retrain agree:
    local passes SKIP_REGIMES=1, so while CI trained regimes the served artifact
    differed by where it was built. Second, that the flag is set in `env` rather
    than inlined into the run: block, so the walk in
    test_ci_workflow_does_not_skip_cv-style tests can see it.

    Note what this is NOT. It is not a cost saving with no served effect --
    predict() prefers the regime model whenever _detect_current_regime matches,
    and `range` (the only regime that ever qualifies) covers 818K-893K of ~985K
    rows, so dropping it moves the mid for ~90% of the cohort onto the global
    model. The justification is that the deployed regime set has included 1-tree
    and 3-tree boosters (docs/research/2026-08-08-model-review.md item 7), not
    that the ~183s is free.

    If a measured read ever shows the regime models help, remove the flag --
    do not weaken test_warm_retrain_still_trains_regime_models, which pins a
    different thing: that regime training is not coupled to the warm-retrain
    gate. Both must stay true.
    """
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "price-forecast.yml")
    spec = yaml.safe_load(wf.read_text())
    run_step = next(s for s in spec["jobs"]["forecast"]["steps"]
                    if s.get("name") == "Run ML price forecasting")
    assert str(run_step["env"]["SKIP_REGIMES"]) == "1"


def test_warm_retrain_still_trains_regime_models(tmp_path, monkeypatch, caplog):
    """A warm retrain may not silently drop the regime models.

    predict() *prefers* the regime model over the global one whenever the
    detected regime matches, so an artifact without them serves a different
    mid. Warm retrain used to skip them as a cost saving, which was invisible
    while warm retrains only ran locally — and becomes a change to production's
    forecast now that CI restores the model cache on training runs.
    """
    f = _fast_forecaster(tmp_path, warm=True)
    with caplog.at_level("INFO", logger="models.forecaster"):
        _run_train(f, monkeypatch, _train_frame(), skip_regimes=False)

    assert "Regime models skipped (warm retrain)" not in caplog.text, (
        "regime training was skipped because the retrain was warm"
    )
    # The branch has to have been *entered*: on a frame this small every regime
    # falls below MIN_REGIME_TRAIN, so its per-regime rejection is the evidence.
    entered = ("below minimum" in caplog.text) or bool(f.regime_models)
    assert entered, "the regime branch never ran on a warm retrain"


def test_warm_retrain_does_not_carry_stale_regime_models_forward(
        tmp_path, monkeypatch):
    """Refitting has to clear the restored artifact's regime models first.

    On a warm run `self.regime_models` arrives populated from load_models(). A
    regime that no longer clears the row minimums is `continue`d, so without an
    explicit clear its *previous* model survives and is re-persisted next to
    freshly trained global models — two training runs mixed in one artifact.
    """
    f = _fast_forecaster(tmp_path, warm=True)
    sentinel = ("bear", f.HORIZONS[0], 0.5)
    f.regime_models[sentinel] = ["stale-ensemble"]

    _run_train(f, monkeypatch, _train_frame(), skip_regimes=False)

    assert sentinel not in f.regime_models or (
        f.regime_models[sentinel] != ["stale-ensemble"]), (
        "a restored regime model survived a retrain of its horizon"
    )


def test_engineered_cache_write_is_suppressible(tmp_path, monkeypatch):
    """ENGINEERED_CACHE=0 must skip the write, not just redirect it.

    CI excludes this ~2GB frame from both the cache save and the restore, so
    every run writes one nothing will ever read. Nothing reads it back in
    process either — predict() only consults it before engineering.
    """
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    path = Path(f._engineered_cache_path)

    monkeypatch.setenv("ENGINEERED_CACHE", "0")
    f._save_engineered_cache(df)
    assert not path.exists(), "cache was written despite ENGINEERED_CACHE=0"

    # And the default still writes, round-trip intact.
    monkeypatch.delenv("ENGINEERED_CACHE", raising=False)
    f._save_engineered_cache(df)
    assert path.exists()
    assert len(pd.read_parquet(path)) == 2


@pytest.mark.parametrize("raw,expected", [
    ("30", [30]),
    ("3,7", [3, 7]),
    (" 7 , 30 ", [7, 30]),
    ("", None),            # unset -> every horizon
    ("   ", None),
    ("nonsense", None),    # a typo must not narrow the run
    ("5", None),           # not a real horizon
    ("3,999", None),       # one bad member poisons the whole list
])
def test_train_horizons_override_parses_or_falls_back(raw, expected, monkeypatch):
    """A typo must train everything, never a silent subset.

    Same rule as TRAIN_MIN_MEDIAN_PRICE: an unparseable value keeps the wider
    behaviour, because a partial artifact that looks complete is the dangerous
    outcome.
    """
    from scripts.forecast_prices import _train_horizons

    monkeypatch.setenv("TRAIN_HORIZONS", raw)
    assert _train_horizons() == expected


def test_diagnostics_workflow_cannot_promote_its_artifact():
    """The diagnostics job scores the served classifier — it must not serve.

    The only route from a run to production is the `forecast-models-` cache key,
    which the daily predict run restores by prefix. This job restores it and must
    never save it, or a single-horizon artifact silently becomes the deployed
    model. Also pins the classifier on: without it the whole workflow is a slower
    copy of the daily run.
    """
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "model-diagnostics.yml")
    spec = yaml.safe_load(wf.read_text())
    steps = spec["jobs"]["diagnose"]["steps"]

    saves = [s for s in steps
             if "cache/save" in str(s.get("uses", ""))
             or (str(s.get("uses", "")).startswith("actions/cache@")
                 and "saved_models" in str(s.get("with", {}).get("path", "")))]
    assert not saves, (
        f"the diagnostics job writes a model cache ({[s.get('name') for s in saves]}); "
        f"a partial artifact would be promoted to production by the daily predict run"
    )

    env = next(s for s in steps
               if s.get("name") == "Run CV with the served classifier scored")["env"]
    # Default-ON, and it cannot use the `&& '1' || '0'` idiom the arms use: a
    # scheduled event carries no inputs, so that form would resolve to '0' and
    # silently turn the Sunday run into the q50-centred band. The fallback form
    # `inputs.x || '1'` is what keeps absence meaning on.
    expr = str(env["CV_DIAGNOSTIC_CLASSIFIER"])
    assert "inputs.cv_diagnostic_classifier" in expr and "'1'" in expr, expr
    # The guard must compare against the WORD. `== '0'` would be `0 == 0` on a
    # scheduled run, where the absent input casts to 0 -- flipping the Sunday
    # control to the treatment. A bare `inputs.x || '1'` instead rests on '0'
    # being truthy, which fails as a no-op arm that reads like a control.
    assert "== 'q50'" in expr, expr
    assert "== '0'" not in expr, expr
    assert "FORECAST_MODEL_DIR" not in env, (
        "FORECAST_MODEL_DIR would point load_models() at an empty directory, so "
        "the job would measure a freshly-tuned model instead of production's"
    )


def test_diagnostics_arms_default_to_the_control():
    """The scheduled Sunday diagnostics run must stay the control arm.

    A scheduled event carries no inputs, so `inputs.tier_lead` is falsy there and
    the expression has to resolve to '0'. If either flag ever defaults on, the
    weekly series silently becomes a treatment arm and every number in it stops
    being comparable to the ones before it -- which is exactly the failure the
    repo already has with pre-873148b A/B verdicts.
    """
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "model-diagnostics.yml")
    spec = yaml.safe_load(wf.read_text())

    inputs = spec[True]["workflow_dispatch"]["inputs"]
    for name in ("tier_lead", "cross_sectional_rank", "naive_init_score",
                 "label_smoothed_anchor", "serve_outlier_gated_anchor",
                 "force_hp_search"):
        assert inputs[name]["default"] is False, (
            f"{name} defaults on; the Sunday scheduled run would stop being a control"
        )

    env = next(s for s in spec["jobs"]["diagnose"]["steps"]
               if s.get("name") == "Run CV with the served classifier scored")["env"]
    # The `&& '1' || '0'` form is what makes a missing input resolve to '0'
    # rather than to an empty string, which ItemForecaster would read as off
    # anyway -- but only because the gate tests for exactly "1". Pin the shape so
    # the two cannot drift apart.
    for key, inp in (("TIER_LEAD_FEATURE", "tier_lead"),
                     ("CROSS_SECTIONAL_RANK", "cross_sectional_rank"),
                     ("NAIVE_INIT_SCORE", "naive_init_score"),
                     ("LABEL_SMOOTHED_ANCHOR", "label_smoothed_anchor"),
                     # Serving-only, so it reaches nothing this step trains --
                     # but predict_smoke runs predict() from here, and an arm
                     # that leaked into the scheduled run would move the served
                     # `current_price` with nothing in meta.json to say so.
                     ("SERVE_OUTLIER_GATED_ANCHOR", "serve_outlier_gated_anchor"),
                     ("FORCE_HP_SEARCH", "force_hp_search")):
        expr = str(env[key])
        assert f"inputs.{inp}" in expr and "'1'" in expr and "'0'" in expr, (
            f"{key} is {expr!r}; it must resolve to '0' when the input is absent"
        )


def test_the_conformal_centre_knobs_default_to_todays_behaviour():
    """Two knobs added 2026-08-11 to settle how production should get a correct
    band. Both must leave the scheduled Sunday run exactly as it was.

    `cv_diagnostic_classifier` gates the out-of-fold direction call `q_hat` is
    centred on, so switching it off is what reproduces the DEFECT — the paired
    control the served-centre read could not produce. `replay_disable` reaches
    `predict` through REPLAY_DISABLE and is how the cheap alternative gets read:
    with `recenter` off, the served mid IS the q50 mid, and the q50-centred
    `q_hat` is already correct for it.
    """
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "model-diagnostics.yml")
    spec = yaml.safe_load(wf.read_text())
    inputs = spec[True]["workflow_dispatch"]["inputs"]

    # A word-valued choice, not a boolean and not '1'/'0': a default-on flag has
    # no safe numeric spelling in a GitHub expression, because an absent input
    # casts to 0 and would equal '0'.
    assert inputs["cv_diagnostic_classifier"]["default"] == "served"
    assert inputs["cv_diagnostic_classifier"]["options"] == ["served", "q50"]
    assert inputs["replay_disable"]["default"] == ""

    steps = spec["jobs"]["diagnose"]["steps"]
    replay = next(s for s in steps if s.get("name") == "Replay the serving path")
    assert "inputs.replay_disable" in str(replay["env"]["REPLAY_DISABLE"])

    # And it must NOT reach training: REPLAY_DISABLE names serving transforms,
    # so a training step that honoured it would report a skip it never made.
    train = next(s for s in steps
                 if s.get("name") == "Run CV with the served classifier scored")
    assert "REPLAY_DISABLE" not in train["env"]


def test_ci_suppresses_the_engineered_cache_write():
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "price-forecast.yml")
    spec = yaml.safe_load(wf.read_text())
    run_step = next(s for s in spec["jobs"]["forecast"]["steps"]
                    if s.get("name") == "Run ML price forecasting")
    assert str(run_step["env"]["ENGINEERED_CACHE"]) == "0"


def test_the_holdout_fallback_still_fits_q_hat_for_every_horizon(
        tmp_path, monkeypatch):
    """The degraded path must degrade, not disappear.

    SKIP_CV=1 is now a local-only speedup, and a horizon with too few distinct
    dates takes the same leg. It under-covers — but every served horizon still
    needs *a* q_hat, or predict() has no band at all.
    """
    f = _fast_forecaster(tmp_path)
    monkeypatch.setenv("SKIP_CV", "1")
    _run_train(f, monkeypatch, _train_frame())

    assert set(f.conformal_calibration) == set(f.HORIZONS), (
        f"horizons with no fitted q_hat: "
        f"{sorted(set(f.HORIZONS) - set(f.conformal_calibration))}"
    )
    assert all(v > 0 for v in f.conformal_calibration.values())
    # And the confidence thresholds must be on the conformal scale, which
    # means they were fitted rather than left at the hardcoded default.
    for h in f.HORIZONS:
        assert f.confidence_thresholds[h]["high_accuracy"] > 0


def test_warm_retrain_calibrates_from_cv_not_the_holdout(tmp_path, monkeypatch):
    """The production steady state: cached HP present, no SKIP_CV.

    A warm retrain reuses hyperparameters — that is its purpose — but it also
    refits the models, so q_hat has to be refitted too, and against records the
    models were not selected on. Warm retrain used to skip CV outright and
    calibrate from X_val, i.e. from the early-stopping and Optuna scoring set.

    `cv_results[h]["fold_count"]` is the observable seam: it is 0 when CV was
    skipped and >= 2 when the out-of-fold records are real.
    """
    f = _fast_forecaster(tmp_path, warm=True)
    monkeypatch.delenv("SKIP_CV", raising=False)
    stale = {h: 999.0 for h in f.HORIZONS}
    f.conformal_calibration = dict(stale)

    _run_train(f, monkeypatch, _train_frame())

    for h in f.HORIZONS:
        assert f.cv_results[h]["fold_count"] >= 2, (
            f"{h}d calibrated with {f.cv_results[h]['fold_count']} CV folds — "
            f"q_hat was fitted on the early-stopping holdout, not out-of-fold"
        )
    assert set(f.conformal_calibration) == set(f.HORIZONS)
    assert all(v > 0 for v in f.conformal_calibration.values())
    assert f.conformal_calibration != stale, (
        "q_hat was inherited from the previous run instead of refitted "
        "against the models this run trained"
    )


def test_train_measures_sigma_clip_from_the_training_frame(tmp_path, monkeypatch):
    """The clip q_hat is calibrated against must come from the data, not the
    class defaults, or serving clips differently than calibration did.

    Measured in train() before the horizon loop, so the calibration branch is
    irrelevant here — SKIP_CV=1 only to keep the fixture cheap.
    """
    f = _fast_forecaster(tmp_path, warm=True)
    monkeypatch.setenv("SKIP_CV", "1")
    defaults = dict(f.sigma_clip)
    _run_train(f, monkeypatch, _train_frame())

    assert set(f.sigma_clip) == {"floor", "cap", "fallback"}
    assert f.sigma_clip != defaults
    assert 0 < f.sigma_clip["floor"] < f.sigma_clip["cap"]
    assert (f.sigma_clip["floor"] <= f.sigma_clip["fallback"]
            <= f.sigma_clip["cap"])


# ---------------------------------------------------------------------------
# Artifact version: a pre-rewrite cache must fail loudly, not load tolerantly
# ---------------------------------------------------------------------------


def test_artifact_version_constant_exists():
    assert isinstance(ItemForecaster.MODEL_ARTIFACT_VERSION, int)
    assert ItemForecaster.MODEL_ARTIFACT_VERSION >= 2


def test_loading_a_pre_rewrite_artifact_raises(tmp_path):
    import json

    from models.forecaster import IncompatibleModelArtifact

    model_dir = tmp_path / "saved_models"
    model_dir.mkdir()
    # A pre-rewrite meta.json: CQR floats, no artifact version, no sigma clip.
    (model_dir / "meta.json").write_text(json.dumps({
        "conformal_calibration": {"3": 4.21, "7": 6.02},
        "n_ensembles": 3,
        "quantiles": [0.1, 0.5, 0.9],
    }))

    f = ItemForecaster.__new__(ItemForecaster)
    with pytest.raises(IncompatibleModelArtifact, match="artifact version"):
        ItemForecaster._check_artifact_version(f, json.loads(
            (model_dir / "meta.json").read_text()
        ))


def test_current_artifact_version_is_accepted():
    f = ItemForecaster.__new__(ItemForecaster)
    meta = {"model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION}
    # Must not raise.
    ItemForecaster._check_artifact_version(f, meta)


def test_load_models_raises_before_reading_any_other_field(tmp_path):
    """The version check must run through the real load_models() entry
    point, not just be reachable in isolation -- and it must fire even
    though every other field in this artifact (feature_cols, n_ensembles)
    is well-formed. An artifact this shape is exactly what the repo's own
    pre-rewrite saved_models/meta.json looks like.
    """
    import json

    from models.forecaster import IncompatibleModelArtifact

    (tmp_path / "meta.json").write_text(json.dumps({
        "feature_cols": ["feat_a", "feat_b"],
        "conformal_calibration": {"3": 1.567, "7": 1.673, "14": 4.440, "30": 7.419},
        "n_ensembles": 3,
    }))
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    with pytest.raises(IncompatibleModelArtifact, match="artifact version"):
        f.load_models()


def test_load_models_raises_on_missing_sigma_clip_even_at_current_version(tmp_path):
    """_check_artifact_version passing does not license a tolerant default
    for a field it didn't itself check. A current-version artifact missing
    sigma_clip is corrupt, not old, and must still raise.
    """
    import json

    (tmp_path / "meta.json").write_text(json.dumps({
        "model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION,
        "feature_cols": ["feat_a", "feat_b"],
        "conformal_calibration": {"3": 1.1, "7": 2.2, "14": 3.3, "30": 4.4},
        "n_ensembles": 1,
    }))
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    with pytest.raises(KeyError, match="sigma_clip"):
        f.load_models()


def test_load_models_raises_on_missing_conformal_calibration_even_at_current_version(tmp_path):
    import json

    (tmp_path / "meta.json").write_text(json.dumps({
        "model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION,
        "feature_cols": ["feat_a", "feat_b"],
        "sigma_clip": {"floor": 0.01, "cap": 2.0, "fallback": 0.15},
        "n_ensembles": 1,
    }))
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    with pytest.raises(KeyError, match="conformal_calibration"):
        f.load_models()


def test_save_then_load_round_trips_conformal_and_sigma_clip(tmp_path):
    """The only path a predict-only run has to these fields is a round trip
    through disk -- it never recalibrates. save_models must persist exactly
    what load_models restores, with no coercion drift (e.g. int horizon keys
    staying int, not becoming str or float on the way back).
    """
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["feat_a", "feat_b"]
    f.conformal_calibration = {3: 1.1, 7: 2.2, 14: 3.3, 30: 4.4}
    f.sigma_clip = {"floor": 0.023, "cap": 0.91, "fallback": 0.156}
    f.save_models()

    g = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    g.load_models()

    assert g.conformal_calibration == f.conformal_calibration
    assert all(isinstance(h, int) for h in g.conformal_calibration)
    assert g.sigma_clip == pytest.approx(f.sigma_clip)


def _tiny_booster():
    """A real one-tree Booster, so this exercises the actual file format."""
    import lightgbm as lgb
    rng = np.random.RandomState(0)
    X = rng.rand(60, 2)
    y = X[:, 0] * 2.0
    ds = lgb.Dataset(X, y)
    return lgb.train({"objective": "quantile", "alpha": 0.5, "verbosity": -1,
                      "num_leaves": 2, "min_data_in_leaf": 5}, ds,
                     num_boost_round=1)


def test_single_member_ensemble_round_trips_through_disk(tmp_path):
    """A one-member ensemble must load back, not silently vanish.

    train() always stores self.models[(h, q)] as a LIST, so save_models writes
    the member as `lgb_3d_q50_e0.txt` whatever N_ENSEMBLES is. load_models
    branched on `n_ensembles > 1` and looked for an unsuffixed
    `lgb_3d_q50.txt` in the single-member case -- a filename save_models never
    produces. With N_ENSEMBLES == 1 that left self.models EMPTY after a load,
    so a predict-only run served nothing. Train-only runs could not catch it:
    the boosters are still in memory there.
    """
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["feat_a", "feat_b"]
    f.conformal_calibration = {3: 1.0}
    f.models[(3, 0.5)] = [_tiny_booster()]
    f.save_models()

    g = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    g.load_models()

    assert (3, 0.5) in g.models, f"models empty after load; on disk: " \
        f"{sorted(p.name for p in tmp_path.glob('*.txt'))}"
    members = g.models[(3, 0.5)]
    assert isinstance(members, list) and len(members) == 1


def test_load_ignores_stale_extra_ensemble_members_on_disk(tmp_path):
    """The 40-model artifact left lgb_*_e1/_e2 files behind on real deploys.

    save_models only purges orphaned *regime* files, so those stale members
    persist in models/saved_models. Loading must be governed by the member
    count, not by whatever happens to be on disk, or the minimal model would
    quietly serve a 3-member ensemble again.
    """
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["feat_a", "feat_b"]
    f.conformal_calibration = {3: 1.0}
    f.models[(3, 0.5)] = [_tiny_booster()]
    f.save_models()
    # Simulate leftovers from the pre-rewrite grid.
    for ei in (1, 2):
        _tiny_booster().save_model(str(tmp_path / f"lgb_3d_q50_e{ei}.txt"))

    g = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    g.load_models()

    assert len(g.models[(3, 0.5)]) == 1


def test_saved_meta_records_the_current_artifact_version(tmp_path):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["feat_a"]
    f.conformal_calibration = {3: 1.0}
    f.save_models()

    import json
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["model_artifact_version"] == ItemForecaster.MODEL_ARTIFACT_VERSION
    assert meta["sigma_clip"] == f.sigma_clip


# ---------------------------------------------------------------------------
# scripts/forecast_prices.py: the caller must recover from an incompatible
# cache when the run is willing to train, and must not paper over it in
# predict-only, which has no recovery path of its own.
# ---------------------------------------------------------------------------

def _fake_incompatible_forecast_env(monkeypatch, tmp_path):
    """Same shape as test_drift_retrain_guard.py's fixture, but load_models
    raises IncompatibleModelArtifact instead of returning True -- this is
    what the real ItemForecaster does when meta.json predates
    MODEL_ARTIFACT_VERSION, which is exactly the state of the checked-in
    backend/models/saved_models/meta.json today.
    """
    import scripts.forecast_prices as fp
    from models.forecaster import IncompatibleModelArtifact

    class FakeForecaster:
        HORIZONS = [3, 7, 14, 30]
        train_called = False

        def __init__(self, *a, **kw):
            self.model_dir = str(tmp_path)
            self.db = None
            type(self).instance = self

        def load_models(self):
            raise IncompatibleModelArtifact(
                "saved model artifact version None != expected 2"
            )

        def check_concept_drift(self, horizon=7, sliding_window=7, threshold=None):
            return None

        def train(self, *a, **kw):
            type(self).train_called = True

        def predict(self):
            return pd.DataFrame()

    fake_db = MagicMock()
    fake_db.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr(fp, "ItemForecaster", FakeForecaster)
    monkeypatch.setattr(fp, "SessionLocal", lambda: fake_db)
    monkeypatch.delenv("ALLOW_DRIFT_RETRAIN", raising=False)
    monkeypatch.delenv("FORCE_RETRAIN", raising=False)
    return FakeForecaster


def test_incompatible_artifact_triggers_a_full_retrain_not_a_crash(monkeypatch, tmp_path):
    import scripts.forecast_prices as fp

    fake = _fake_incompatible_forecast_env(monkeypatch, tmp_path)
    fp.run_forecast()
    assert fake.train_called is True, (
        "A rejected cache must read as 'no usable models' for a mode that is "
        "willing to train, not crash the run."
    )


def test_incompatible_artifact_triggers_retrain_in_train_only_mode(monkeypatch, tmp_path):
    import scripts.forecast_prices as fp

    fake = _fake_incompatible_forecast_env(monkeypatch, tmp_path)
    result = fp.run_forecast(train_only=True)
    assert fake.train_called is True
    assert result["status"] == "success"


def test_incompatible_artifact_fails_the_run_in_predict_only_mode(monkeypatch, tmp_path):
    """predict-only has no recovery path: serving from a cache whose q_hat
    means something else is the exact failure this guard exists to prevent,
    so the incompatibility must re-raise rather than being silently treated
    as absent.

    run_forecast has a pre-existing catch-all (`except Exception as e:
    ... return {"status": "error", ...}`) wrapping the whole function body,
    which every other failure mode in this pipeline already surfaces
    through -- main() turns any non-"success" status into a non-zero exit.
    Re-raising out of the inner try/except therefore surfaces here as that
    same error status, not as a raw exception escaping run_forecast(). That
    is the existing "fail loudly" idiom for this pipeline: this must NOT
    fall through to do_train/predict and must NOT report "success".
    """
    import scripts.forecast_prices as fp

    fake = _fake_incompatible_forecast_env(monkeypatch, tmp_path)
    result = fp.run_forecast(predict_only=True)
    assert result["status"] == "error"
    assert "artifact version" in result["message"]
    assert fake.train_called is False


# ---------------------------------------------------------------------------
# predict(): the served band comes from conformal, not from p10/p90 models
# ---------------------------------------------------------------------------


def test_predict_no_longer_calls_the_crossing_fix():
    import inspect

    src = inspect.getsource(ItemForecaster.predict)
    assert "_fix_quantile_crossing" not in src, (
        "a symmetric band around the median cannot cross; the crossing fix "
        "survives only for walkforward's baseline arm"
    )


def test_crossing_fix_still_exists_for_the_harness():
    # scripts/walkforward_backtest.py uses it for the baseline arm, as do
    # evaluate_forecaster.py and the ab_test_* scripts. Deleting it breaks the
    # baseline the minimal model is measured against.
    assert hasattr(ItemForecaster, "_fix_quantile_crossing")


def test_predict_never_indexes_the_p10_or_p90_prediction():
    """No unreachable p10/p90 branch may survive in predict().

    Once QUANTILES == [0.5] (Task 11) any `preds[0.1]` would be a KeyError, and
    a diagnostic that reads p10_ret/p90_ret is just as fatal as the band did.
    """
    import inspect

    src = inspect.getsource(ItemForecaster.predict)
    for dead in ("preds[0.1]", "preds[0.9]", "p10_ret", "p90_ret",
                 "len(preds) != 3"):
        assert dead not in src, f"predict() still references {dead}"


def test_band_from_conformal_varies_by_item_and_is_finite():
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.sigma_clip = {"floor": 0.01, "cap": 2.0, "fallback": 0.25}

    rows = pd.DataFrame({
        "price": [100.0, 100.0, 50.0],
        # third row: no 60d history, the short-history case
        "price_std_60d": [5.0, 20.0, np.nan],
    })
    sigma = ItemForecaster._sigma_for_rows(f, rows)
    assert np.all(np.isfinite(sigma))
    assert sigma[1] > sigma[0]          # more volatile item, wider sigma
    assert sigma[2] == pytest.approx(0.25)   # fallback, not NaN

    from models.conformal import band
    low, high = band(np.zeros(3), sigma, q_hat=3.0)
    widths = high - low
    assert widths[1] > widths[0]
    assert np.all(np.isfinite(widths))
    assert np.all(low <= high)


# --- behavioural: drive the real predict() over a synthetic catalogue -------

class _StubBooster:
    """Minimal LightGBM Booster stand-in for _predict_ensemble_safe.

    `num_feature` is checked against the live matrix width before predict is
    called, so it has to be honest about the column count.
    """

    def __init__(self, n_features: int, value: float):
        self._n = n_features
        self._value = value

    def num_feature(self):
        return self._n

    def predict(self, X):
        return np.full(len(X), float(self._value))


class _StubClassifier:
    """3-class direction model returning one fixed probability row per item."""

    def __init__(self, probs_row):
        self._row = np.asarray(probs_row, dtype=float)

    def predict(self, X):
        return np.tile(self._row, (len(X), 1))


# Two items at the same price level with an order of magnitude between their
# volatilities, plus a cheap item as volatile as the second. Deterministic:
# the sigma ordering asserted below must not depend on a seed.
_PREDICT_ITEMS = {"calm100": (100.0, 1.0), "wild100": (100.0, 20.0),
                  "wild5": (5.0, 1.0)}


def _predict_price_frame(n_dates=70):
    rows = []
    for d in range(n_dates):
        day = date(2026, 6, 1) + timedelta(days=d)
        for iid, (base, amp) in _PREDICT_ITEMS.items():
            rows.append({
                "item_id": iid,
                "date": day,
                "price": base + (amp if d % 2 else -amp),
                "volume": 100,
            })
    return pd.DataFrame(rows)


_MID_RET = 4.0          # the stub median model's return, in percent
_Q_HAT = 10.0           # sigma is a ratio and mid is in percent, so q_hat
                        # absorbs the factor of 100 — see conformal.calibrate


def _predict_forecaster(tmp_path, q_hat=_Q_HAT, horizons=None,
                        classifier=None):
    """A forecaster wired for predict() with a stub median model per horizon.

    `db=None` disables prior-forecast blending, and bias corrections /
    thresholds are left empty, so the served band is exactly what the conformal
    step produced.
    """
    f = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    f.feature_cols = ["price", "price_std_60d"]
    f.horizon_feature_cols = {}
    f.conformal_calibration = ({h: q_hat for h in f.HORIZONS}
                               if horizons is None
                               else {h: q_hat for h in horizons})
    f.models = {(h, 0.5): [_StubBooster(len(f.feature_cols), _MID_RET)]
                for h in f.HORIZONS}
    if classifier is not None:
        f.direction_models = {h: classifier for h in f.HORIZONS}
    return f


def _run_predict(f):
    """Call the real predict() with only the two data fetches stubbed."""
    empty_events = pd.DataFrame(columns=["id", "type", "timestamp",
                                         "description"])
    with patch.object(f, "fetch_price_history",
                      return_value=_predict_price_frame()), \
            patch.object(f, "fetch_events", return_value=empty_events):
        return f.predict()


def _forecast_rows(result, horizon):
    """{item_id: forecast dict} for one horizon."""
    return {r["item_id"]: r["forecasts"][horizon]
            for r in result.to_dict("records")}


def test_predict_serves_a_band_centred_on_the_median(tmp_path):
    """The served interval is symmetric about the median, in price space."""
    f = _predict_forecaster(tmp_path)
    result = _run_predict(f)
    assert not result.empty

    for h in f.HORIZONS:
        for iid, fc in _forecast_rows(result, h).items():
            assert fc["low"] <= fc["mid"] <= fc["high"], (h, iid, fc)
            # round(, 2) on each leg, so allow one cent of asymmetry.
            assert (fc["high"] - fc["mid"]) == pytest.approx(
                fc["mid"] - fc["low"], abs=0.011), (h, iid, fc)


def test_predict_half_width_is_q_hat_times_the_clipped_sigma(tmp_path):
    """The width must be q_hat * sigma, with sigma from _sigma_for_rows.

    Recomputing std/price at the use site, or skipping the clip, would void the
    coverage guarantee q_hat was fitted under; this pins the exact arithmetic.
    """
    f = _predict_forecaster(tmp_path)
    seen = {}
    real = f._sigma_for_rows

    def recording(rows):
        sigma = real(rows)
        seen["sigma"] = dict(zip(rows["item_id"], sigma))
        seen["price"] = dict(zip(rows["item_id"], rows["price"]))
        return sigma

    with patch.object(f, "_sigma_for_rows", side_effect=recording):
        result = _run_predict(f)

    assert seen, "predict() did not route sigma through _sigma_for_rows"
    for h in f.HORIZONS:
        for iid, fc in _forecast_rows(result, h).items():
            base = seen["price"][iid]
            expected_half = base * (_Q_HAT * seen["sigma"][iid]) / 100.0
            assert (fc["high"] - fc["mid"]) == pytest.approx(
                expected_half, abs=0.011), (h, iid, fc)
            assert fc["mid"] == pytest.approx(
                base * (1 + _MID_RET / 100.0), abs=0.011)


def test_predict_band_width_scales_with_item_volatility(tmp_path):
    """The item-level variation the p10/p90 models used to supply."""
    f = _predict_forecaster(tmp_path)
    result = _run_predict(f)
    fcs = _forecast_rows(result, 7)

    def rel_width(iid):
        return (fcs[iid]["high"] - fcs[iid]["low"]) / fcs[iid]["mid"]

    assert rel_width("wild100") > rel_width("calm100")
    # Same volatility, ten times cheaper: normalization makes the relative
    # width comparable and the absolute width scale with price.
    assert rel_width("wild5") == pytest.approx(rel_width("wild100"), rel=0.5)
    assert (fcs["wild100"]["high"] - fcs["wild100"]["low"]) > (
        fcs["wild5"]["high"] - fcs["wild5"]["low"])


def test_predict_refuses_to_serve_a_horizon_with_no_q_hat(tmp_path):
    """Refuse rather than fabricate: a zero-width or default band would be
    served to users as if it were calibrated."""
    f = _predict_forecaster(tmp_path, horizons=[])
    with pytest.raises(RuntimeError, match="conformal calibration"):
        _run_predict(f)


def test_predict_still_serves_the_classifier_direction(tmp_path):
    """The direction call is the classifier's, not a threshold on mid_ret."""
    up = _StubClassifier([0.05, 0.05, 0.90])
    f = _predict_forecaster(tmp_path, classifier=up)
    result = _run_predict(f)

    for h in f.HORIZONS:
        for iid, fc in _forecast_rows(result, h).items():
            assert fc["direction"] == "up", (h, iid, fc)
            assert fc["confidence"] == "high"
            assert fc["low"] <= fc["mid"] <= fc["high"]

    down = _StubClassifier([0.90, 0.05, 0.05])
    f2 = _predict_forecaster(tmp_path, classifier=down)
    result2 = _run_predict(f2)
    for h in f2.HORIZONS:
        for iid, fc in _forecast_rows(result2, h).items():
            assert fc["direction"] == "down", (h, iid, fc)
            # _recenter_on_direction flips the median but keeps half-widths.
            assert fc["low"] <= fc["mid"] <= fc["high"]


# ---------------------------------------------------------------------------
# The conformal band is unbounded below, so sanitization must keep it ordered
#
# _sanitize_forecasts clamps low/mid/high to current_price INDEPENDENTLY per
# leg. Under the old [p10, p90] + percentage-point widening, low_ret never got
# near -100 and the per-leg clamp was effectively unreachable. The conformal
# band has no such bound: low_ret = mid_ret - q_hat * sigma with sigma free to
# sit at the persisted 99th-percentile cap. So the ordering invariant has to be
# enforced structurally, and specifically for volatile items — the ones this
# design exists to serve better.
# ---------------------------------------------------------------------------


def _one_row_result(cur, low, mid, high, horizon=7, item_id="itm"):
    return pd.DataFrame([{
        "item_id": item_id,
        "current_price": float(cur),
        "forecasts": {horizon: {"low": float(low), "mid": float(mid),
                                "high": float(high), "direction": "down",
                                "confidence": "high"}},
        "generated_at": datetime.now(timezone.utc),
    }])


def test_a_conformal_band_wide_enough_to_go_negative_stays_ordered(tmp_path):
    """The reported repro, composed through the real band construction.

    cur=100, mid_ret=-20, sigma=2.0 (the default cap), q_hat=60 gives
    low_ret=-140 -> a low PRICE of -40, which the per-leg clamp lifts to
    current_price=100 while mid stays at 80. That served low > mid.
    """
    from models.conformal import band

    f = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    cur, mid_ret, sigma, q_hat = 100.0, -20.0, 2.0, 60.0
    low_ret, high_ret = band(np.array([mid_ret]), np.array([sigma]), q_hat)

    assert low_ret[0] == pytest.approx(-140.0)   # below -100%: negative price
    to_price = lambda r: round(cur * (1 + r / 100.0), 2)
    result = _one_row_result(cur, to_price(low_ret[0]), to_price(mid_ret),
                             to_price(high_ret[0]))
    assert result.iloc[0]["forecasts"][7]["low"] == pytest.approx(-40.0)

    fc = f._sanitize_forecasts(result).iloc[0]["forecasts"][7]
    assert fc["low"] <= fc["mid"] <= fc["high"], fc
    assert fc["low"] > 0, "a served price must be positive"
    assert fc["mid"] == pytest.approx(80.0), "the valid median must survive"


@pytest.mark.parametrize("mid_ret", [-95.0, -60.0, -20.0, 0.0, 25.0, 300.0])
@pytest.mark.parametrize("sigma", [0.01, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("q_hat", [1.0, 15.0, 60.0, 200.0])
def test_ordering_holds_for_any_band_width(tmp_path, mid_ret, sigma, q_hat):
    """Property check across the reachable (mid_ret, sigma, q_hat) space.

    sigma spans the default clip range including the cap; q_hat spans from a
    tight band to one far wider than anything calibration should produce.
    """
    from models.conformal import band

    f = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    cur = 100.0
    low_ret, high_ret = band(np.array([mid_ret]), np.array([sigma]), q_hat)
    to_price = lambda r: round(cur * (1 + r / 100.0), 2)
    result = _one_row_result(cur, to_price(low_ret[0]), to_price(mid_ret),
                             to_price(high_ret[0]))

    fc = f._sanitize_forecasts(result).iloc[0]["forecasts"][7]
    assert fc["low"] <= fc["mid"] <= fc["high"], (mid_ret, sigma, q_hat, fc)
    assert fc["low"] > 0, (mid_ret, sigma, q_hat, fc)
    assert np.isfinite([fc["low"], fc["mid"], fc["high"]]).all()


def test_sanitization_cannot_be_relied_on_being_given_an_ordered_band(tmp_path):
    """The guard must not assume its input is ordered.

    Every leg here is a positive, finite price, so no per-leg clamp fires — the
    band is simply inverted on arrival. Sanitization is the last layer before
    serving, so it has to hold for any upstream input, not just for the band
    predict() happens to build today.
    """
    f = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    result = _one_row_result(cur=100.0, low=120.0, mid=90.0, high=70.0)
    fc = f._sanitize_forecasts(result).iloc[0]["forecasts"][7]
    assert fc["low"] <= fc["mid"] <= fc["high"], fc
    assert fc["low"] > 0


def test_quantiles_collapse_to_the_median_only():
    assert ItemForecaster.QUANTILES == [0.5]


def test_ensemble_is_a_single_member():
    assert ItemForecaster.N_ENSEMBLES == 1
    assert len(ItemForecaster.ENSEMBLE_SEEDS) == 1
    assert len(ItemForecaster.ENSEMBLE_FEATURE_FRACTIONS) == 1


def test_dart_is_gone_from_the_forecaster():
    """No dart branch, constant or per-horizon selector may come back.

    A source-level check rather than a config assertion, because the config that
    selected DART (BOOSTING_TYPE_MAP) and the constant that sized its runs
    (DART_NUM_BOOST_ROUND) are both deleted — there is no longer a value to
    assert on, only the absence of the mechanism. DART cost 78% of pre-rewrite
    training time and lost to GBDT at 14d by 3.14pp when finally measured.
    """
    import inspect

    assert not hasattr(ItemForecaster, "BOOSTING_TYPE_MAP")
    assert not hasattr(ItemForecaster, "DART_NUM_BOOST_ROUND")
    assert ItemForecaster.BOOSTING_TYPE == "gbdt"

    src = inspect.getsource(inspect.getmodule(ItemForecaster))
    offenders = [ln for ln in src.splitlines()
                 if "dart" in ln.lower() and not ln.lstrip().startswith("#")]
    assert not offenders, f"dart re-entered non-comment code: {offenders}"


def test_trained_model_count_is_eight():
    # 4 median GBMs + 4 directional classifiers. Guards accidental
    # re-expansion of the quantile/ensemble grid.
    expected = len(ItemForecaster.HORIZONS) * len(ItemForecaster.QUANTILES) \
        * ItemForecaster.N_ENSEMBLES
    assert expected == 4
    assert expected + len(ItemForecaster.HORIZONS) == 8


def test_residual_stacking_is_gone():
    assert not hasattr(ItemForecaster, "STACK_RESIDUALS")
    assert not hasattr(ItemForecaster, "RESIDUAL_ALPHA")


def test_dart_params_are_gone():
    assert not hasattr(ItemForecaster, "DART_PARAMS")


# ---------------------------------------------------------------------------
# Invariant #4 must describe the SERVED classifier, not only the q50 sign
#
# The 2026-08-10 model-diagnostics run (31416199251) scored the classifier's
# accuracy but reported "edge vs constant call" and the PT verdict from the
# quantile-median sign, under a heading that implied the served signal. These
# pin the fix: both signals are published, and the served one is never
# silently substituted by the other.
# ---------------------------------------------------------------------------


def test_direction_records_from_classes_maps_the_class_encoding():
    """0=down, 1=flat, 2=up, matching _direction_classes."""
    recs = ItemForecaster._direction_records_from_classes(
        [0, 1, 2], [0, 2, 2], ["2026-01-01"] * 3)

    assert [r["predicted_direction"] for r in recs] == ["down", "flat", "up"]
    assert [r["actual_direction"] for r in recs] == ["down", "up", "up"]
    assert [r["direction_correct"] for r in recs] == [True, False, True]
    assert all(r["forecast_date"] == "2026-01-01" for r in recs)


def test_classifier_pt_records_are_empty_when_the_diagnostic_is_off(
        tmp_path, monkeypatch):
    """CV_DIAGNOSTIC_CLASSIFIER=0 must yield NO served-side records.

    Not a fallback to the quantile sign — absent. A PT verdict that changes
    which signal it describes based on an env var is the defect being fixed.
    """
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "0")
    f = _cv_forecaster(tmp_path, [0.5])

    _, _, pt_records, pt_records_clf = f._cv_evaluate_horizon(
        _cv_frame(horizon=3), 3, {0.5: {}})

    assert pt_records, "quantile-sign records should still be built"
    assert pt_records_clf == []


def test_classifier_pt_records_are_built_when_the_diagnostic_is_on(
        tmp_path, monkeypatch):
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "1")
    f = _cv_forecaster(tmp_path, [0.5])

    _, fold_metrics, pt_records, pt_records_clf = f._cv_evaluate_horizon(
        _cv_frame(horizon=3), 3, {0.5: {}})

    assert pt_records_clf, "served classifier produced no PT records"
    # One record per validation row, same as the quantile-sign stream.
    assert len(pt_records_clf) == len(pt_records)
    assert len(pt_records_clf) == sum(m["n_val"] for m in fold_metrics)
    assert {r["predicted_direction"] for r in pt_records_clf} <= {
        "down", "flat", "up"}
    # The outcomes are a property of the labels, so both streams must agree on
    # them row-for-row. That is what lets constant_call be shared.
    assert ([r["actual_direction"] for r in pt_records_clf]
            == [r["actual_direction"] for r in pt_records])


def test_cv_results_publish_both_invariant_4_signals():
    """Source-level guard on the aggregation block inside `train`.

    The block is not reachable without a full train, so this asserts the
    contract on the keys it writes: two distinct signal-labelled names, and no
    path that lets the served verdict fall back to the quantile sign.
    """
    src = inspect.getsource(ItemForecaster._train_horizon_inline)

    # Both signals published, under names that say which is which.
    assert '"invariant_4_signal": "quantile_sign"' in src
    assert '"edge_vs_constant_call_classifier": edge_vs_constant_clf' in src
    assert '"pt_classifier": pt_clf' in src

    # pt_clf is None when the diagnostic classifier did not run -- never `pt`.
    assert "pt_clf = (pesaran_timmermann(pt_records_clf, MIN_FORECAST_DATES)" in src
    assert "pt_clf = pt" not in src
    assert "pt_clf or pt" not in src
    assert "edge_vs_constant_clf or edge_vs_constant" not in src


def test_predict_smoke_writes_nothing():
    """The point of the mode: exercise predict() without touching the DB.

    A price-forecast.yml dispatch reaches the same code and then calls
    _write_forecasts_to_db under MODEL_VERSION + "-regime" -- the same string
    production writes -- so its rows are indistinguishable from real ones and
    are frozen into outcomes. This mode must return before any writer.
    """
    import inspect
    from scripts import forecast_prices

    src = inspect.getsource(forecast_prices.run_forecast)
    smoke = src.split("if predict_smoke:")[1].split("if train_only:")[0]
    assert "forecaster.predict()" in smoke, "the mode does not run predict()"
    assert "_write_forecasts_to_db" not in smoke
    assert "return" in smoke, "the mode must return before the writer below it"
    # And it has to be positioned ahead of the writer, not merely avoid calling
    # it: falling through would write.
    assert src.index("if predict_smoke:") < src.index("_write_forecasts_to_db")


def test_predict_smoke_is_off_unless_asked():
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "model-diagnostics.yml")
    spec = yaml.safe_load(wf.read_text())
    assert spec[True]["workflow_dispatch"]["inputs"]["predict_smoke"]["default"] is False

    run_step = next(s for s in spec["jobs"]["diagnose"]["steps"]
                    if s.get("name") == "Run CV with the served classifier scored")
    # The scheduled run carries no inputs, so the comparison is false and the
    # mode stays --train-only.
    assert "--train-only" in run_step["run"]
    assert "inputs.predict_smoke" in run_step["run"]


def test_predict_smoke_always_trains():
    """It must not predict from a restored artifact.

    The mode exists to run the predict path under the current run's flags. A
    cached artifact would be restored, the age gate would skip training, and the
    smoke would then measure a model trained under different flags -- or, if the
    flag is on and the artifact predates it, refuse to serve at all. That is
    exactly how run 31439896107 failed.
    """
    from scripts import forecast_prices

    src = inspect.getsource(forecast_prices.run_forecast)
    decision = src.split("do_train = False")[1].split("if do_train:")[0]
    assert "if train_only or predict_smoke:" in decision, (
        "predict_smoke no longer forces training; it would predict from "
        "whatever the model cache happened to restore"
    )
