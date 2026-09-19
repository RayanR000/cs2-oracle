"""Regression test for the vectorized fold directional accuracy (PR 4, Task 4.1).

`models/forecaster.py` used to bucket each fold row in a Python loop; it now
calls `models.direction.directional_accuracy`. Both use the same 3-class
up/flat/down bucketing at +/-DIRECTION_FLAT_TOLERANCE_PCT, in the same units
the trainer labels in: PERCENT (5.0 == 5%), not fractions. A 0.05-style
fraction input would sit entirely inside the +/-0.5 flat band and score a
vacuous 100% — the fixtures below are percent-scale on purpose.
"""

import numpy as np
from models.direction import DIRECTION_FLAT_TOLERANCE_PCT, directional_accuracy


def test_directional_accuracy_matches_manual():
    """Vectorized directional_accuracy matches the manual loop output."""
    pred = np.array([5.0, -2.0, 0.0, 10.0, -3.0])
    actual = np.array([3.0, -1.0, 1.0, -5.0, -8.0])
    result = directional_accuracy(pred, actual)
    # Manual: up/down match for indices 0,1,4; mismatch 2,3 -> 60%
    assert abs(result - 60.0) < 0.1


def test_directional_accuracy_matches_fold_loop():
    """Byte-for-byte agreement with the loop forecaster.py used to run."""
    rng = np.random.default_rng(7)
    pred = rng.normal(0, 5, size=200)
    actual = rng.normal(0, 5, size=200)

    fold_hits = 0
    for i in range(len(actual)):
        actual_ret = float(actual[i])
        mid_ret = float(pred[i])
        actual_dir = (
            "up"
            if actual_ret > DIRECTION_FLAT_TOLERANCE_PCT
            else "down"
            if actual_ret < -DIRECTION_FLAT_TOLERANCE_PCT
            else "flat"
        )
        pred_dir = (
            "up"
            if mid_ret > DIRECTION_FLAT_TOLERANCE_PCT
            else "down"
            if mid_ret < -DIRECTION_FLAT_TOLERANCE_PCT
            else "flat"
        )
        if pred_dir == actual_dir:
            fold_hits += 1
    expected = round(fold_hits / len(actual) * 100, 1)

    assert directional_accuracy(pred, actual) == expected


def test_directional_accuracy_empty():
    assert directional_accuracy([], []) == 0.0
