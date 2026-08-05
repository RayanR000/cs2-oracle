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

import numpy as np
import pandas as pd
import pytest
import yaml

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

    df = _records()
    f._calibrate_conformal(7, df)
    corr = np.corrcoef(df["sigma"], df["range_pct"])[0, 1]
    assert corr > 0.9


def test_calibrated_band_covers_about_the_nominal_rate():
    """q_hat * sigma should bracket ~80% of the OOF residuals it was fit on."""
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.conformal_calibration = {}

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
    assert len(result) == 2, "return tuple must be (oof_records, fold_metrics)"
    oof_records, fold_metrics = result

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

    oof_records, _ = f._cv_evaluate_horizon(tdf, 3, {0.5: {}})
    records_df = pd.DataFrame(oof_records)
    q_hat = f._calibrate_conformal(3, records_df)
    f._calibrate_confidence(horizon=3, records_df=records_df)

    assert f.conformal_calibration[3] == q_hat
    assert q_hat > 0
    assert records_df["range_pct"].notna().all()
    assert np.isfinite(f.confidence_thresholds[3]["high_range"])


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

    Only cost knobs are touched. QUANTILES, N_ENSEMBLES and BOOSTING_TYPE_MAP
    keep their production values on the class; the instance overrides here are
    test-local and do not change what ships.

    `warm=True` seeds cached HP for every horizon/quantile, which is what makes
    reuse_hp — and therefore a warm retrain — true.
    """
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.N_ENSEMBLES = 1
    f.DART_NUM_BOOST_ROUND = 25
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


def _run_train(f, monkeypatch, df):
    """Drive the real train() over a synthetic frame.

    build_training_data is the only thing stubbed, so train()'s own sigma-clip
    measurement and every per-horizon calibration branch run for real.
    """
    def fake_build(*a, **kw):
        f.feature_cols = ["feat_a", "feat_b"]
        f._base_feature_cols = list(f.feature_cols)
        return df.copy()

    monkeypatch.setattr(f, "build_training_data", fake_build)
    monkeypatch.setattr(f, "save_models", lambda *a, **kw: None)
    monkeypatch.setenv("SKIP_REGIMES", "1")
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


def test_no_horizon_uses_dart():
    assert set(ItemForecaster.BOOSTING_TYPE_MAP.values()) == {"gbdt"}


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
