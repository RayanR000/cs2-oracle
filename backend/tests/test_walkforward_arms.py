"""The gate must be able to score cheap baselines on identical folds."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import scripts.walkforward_backtest as wf


def test_run_walkforward_accepts_an_arm():
    assert "arm" in inspect.signature(wf.run_walkforward).parameters


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        wf.run_walkforward(arm="lstm", max_items=1, skip_db=True)


def test_naive_arm_predicts_the_trailing_return():
    X = np.zeros((3, 2))
    trailing = np.array([1.5, -2.0, 0.0])
    mid, low, high, classes = wf._naive_predict(trailing)
    assert mid == pytest.approx(trailing)
    # 0=down, 1=flat, 2=up with the 0.5% flat band.
    assert list(classes) == [2, 0, 1]
    assert np.all(low <= mid) and np.all(mid <= high)


def test_ridge_arm_returns_aligned_arrays():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 4))
    y_train = X_train[:, 0] * 2.0 + rng.normal(scale=0.1, size=200)
    X_val = rng.normal(size=(40, 4))
    mid, low, high, classes = wf._ridge_predict(X_train, y_train, X_val)
    assert mid.shape == (40,)
    assert low.shape == (40,) and high.shape == (40,)
    assert classes.shape == (40,)
    assert set(np.unique(classes)) <= {0, 1, 2}
