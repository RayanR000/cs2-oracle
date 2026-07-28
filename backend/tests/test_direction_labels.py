import numpy as np
from models.forecaster import ItemForecaster


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
    assert t[0] == 0.2   # floored
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
