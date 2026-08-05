"""Shape guarantees for the 8-model forecaster.

The band now comes from conformal calibration around a single median model,
not from 24 p10/p90 GBMs. These tests guard the properties that made that
safe: item-varying width, a finite band for short-history items, ordering
without the crossing fix, and an artifact that cannot be loaded by the wrong
code version.
"""
from __future__ import annotations

from datetime import date, timedelta
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
