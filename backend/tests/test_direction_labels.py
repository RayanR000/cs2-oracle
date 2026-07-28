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
