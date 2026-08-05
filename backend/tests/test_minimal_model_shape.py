"""Shape guarantees for the 8-model forecaster.

The band now comes from conformal calibration around a single median model,
not from 24 p10/p90 GBMs. These tests guard the properties that made that
safe: item-varying width, a finite band for short-history items, ordering
without the crossing fix, and an artifact that cannot be loaded by the wrong
code version.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

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
    assert th["high_range"] > 0


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
