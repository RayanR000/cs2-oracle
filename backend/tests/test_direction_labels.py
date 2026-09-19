import numpy as np
import pandas as pd
from models.forecaster import DIRECTION_LABEL_VOL_COL, ItemForecaster


def test_direction_threshold_scales_with_sqrt_horizon():
    # sigma = 1.0% daily, k = 1.0 -> threshold = sqrt(h)% (before clamp)
    sigma = np.array([1.0, 1.0])
    t3 = ItemForecaster._direction_threshold(sigma, horizon=3, k=1.0, floor=0.0, cap=100.0)
    np.testing.assert_allclose(t3, np.sqrt(3.0), rtol=1e-6)
    t30 = ItemForecaster._direction_threshold(sigma, horizon=30, k=1.0, floor=0.0, cap=100.0)
    np.testing.assert_allclose(t30, np.sqrt(30.0), rtol=1e-6)


def test_direction_threshold_applies_floor_and_cap():
    sigma = np.array([0.0, 1000.0])  # degenerate low and high vol
    t = ItemForecaster._direction_threshold(sigma, horizon=7, k=1.0, floor=0.2, cap=15.0)
    assert t[0] == 0.2  # floored
    assert t[1] == 15.0  # capped


def test_direction_threshold_accepts_scalar_sigma():
    t = ItemForecaster._direction_threshold(2.0, horizon=1, k=0.5, floor=0.0, cap=100.0)
    assert t.shape == (1,)
    np.testing.assert_allclose(t, 1.0, rtol=1e-6)


def test_direction_classes_scalar_matches_legacy():
    r = np.array([1.0, -1.0, 0.2, -0.2, 0.5, -0.5])
    got = ItemForecaster._direction_classes(r, 0.5)
    # >0.5 -> up(2), <-0.5 -> down(0), else flat(1); boundary 0.5 is flat
    assert got.tolist() == [2, 0, 1, 1, 1, 1]


def test_direction_classes_default_is_legacy_half_pct():
    r = np.array([0.6, -0.6, 0.0])
    assert ItemForecaster._direction_classes(r).tolist() == [2, 0, 1]


def test_direction_classes_per_row_threshold():
    r = np.array([1.0, 1.0])
    thr = np.array([0.5, 2.0])  # same return, different bands
    assert ItemForecaster._direction_classes(r, thr).tolist() == [2, 1]


def test_direction_sample_weights_per_row_threshold():
    r = np.array([1.0, 1.0])
    thr = np.array([0.5, 2.0])
    w = ItemForecaster._direction_sample_weights(r, thr, mover_weight=3.0)
    assert w.tolist() == [3.0, 1.0]


def test_label_vol_column_is_trailing_and_grouped():
    fc = ItemForecaster(db_session=None)
    # two items; smooth-trend item -> low vol, noisy item -> higher vol
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    calm = pd.DataFrame({"item_id": "calm", "date": dates, "price": np.linspace(10.0, 12.0, n)})
    # ~1% daily oscillation -> should land on a PERCENT scale (~0.5-3), not ~0.01
    noisy_price = 10.0 * (1 + 0.01 * np.sin(np.arange(n)))
    noisy = pd.DataFrame({"item_id": "noisy", "date": dates, "price": noisy_price})
    df = pd.concat([calm, noisy], ignore_index=True)
    out = fc._compute_price_features(df)

    assert DIRECTION_LABEL_VOL_COL in out.columns

    calm_vol = out[out["item_id"] == "calm"][DIRECTION_LABEL_VOL_COL].iloc[-1]
    noisy_vol = out[out["item_id"] == "noisy"][DIRECTION_LABEL_VOL_COL].iloc[-1]
    assert noisy_vol > calm_vol

    # Units check: ~1% daily moves should yield a vol on the order of ~1
    # (percent units), not ~0.01 (raw log-return fraction units).
    assert 0.3 <= noisy_vol <= 3.0

    # No leakage: first row per item has no trailing window -> NaN
    first_calm = out[out["item_id"] == "calm"][DIRECTION_LABEL_VOL_COL].iloc[0]
    assert np.isnan(first_calm)


def test_fit_classifier_signature_accepts_horizon_and_sigma():
    import inspect

    sig = inspect.signature(ItemForecaster._fit_direction_classifier)
    params = list(sig.parameters)
    assert "horizon" in params
    assert "sigma_train" in params and "sigma_val" in params


def test_fit_classifier_vol_scaling_changes_labels():
    # With a large per-row sigma, movers should collapse to flat, shrinking
    # the number of up/down training labels vs the fixed-0.5% baseline.
    fc = ItemForecaster(db_session=None)
    rng = np.random.RandomState(0)
    pd.DataFrame({"f0": rng.randn(400), "f1": rng.randn(400)})
    y = rng.randn(400) * 2.0  # returns in percent, spread around 0
    big_sigma = np.full(400, 50.0)  # huge vol -> band hits cap 15% -> most flat
    legacy = fc._direction_classes(y)  # fixed 0.5%
    thr = fc._direction_threshold(big_sigma, horizon=3, k=fc.DIRECTION_VOL_MULTIPLIER_MAP[3], floor=0.2, cap=15.0)
    scaled = fc._direction_classes(y, thr)
    assert (scaled == 1).sum() > (legacy == 1).sum()


def test_sweep_grid_and_eval_are_importable():
    import importlib

    mod = importlib.import_module("scripts.archive.ab_test_direction_labels")
    # grid constants exist and are non-empty
    assert len(mod.K_GRID) >= 2
    assert len(mod.MOVER_WEIGHT_GRID) >= 2
    # eval uses the fixed yardstick: a helper that scores preds vs fixed labels
    assert hasattr(mod, "score_fixed_yardstick")


def test_sweep_max_folds_and_grids():
    import importlib

    mod = importlib.import_module("scripts.archive.ab_test_direction_labels")
    assert isinstance(mod.MAX_FOLDS, int)
    assert mod.K_GRID == [0.25, 0.5, 1.0]
    assert mod.MOVER_WEIGHT_GRID == [3.0, 5.0, 8.0]


def test_score_fixed_yardstick_ignores_training_threshold():
    from scripts.archive.ab_test_direction_labels import score_fixed_yardstick

    actual_returns = np.array([1.0, -1.0, 0.1])  # up, down, flat @0.5%
    pred_cls = np.array([2, 0, 1])  # all correct vs fixed band
    acc, movers_acc = score_fixed_yardstick(pred_cls, actual_returns)
    assert acc == 1.0
    assert movers_acc == 1.0  # only the two movers, both correct
