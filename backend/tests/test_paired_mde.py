"""Paired MDE for arm-vs-arm comparison on shared folds."""
from __future__ import annotations

from datetime import date

import pytest

from backtest.paired_mde import paired_da_difference


def _rec(item, day, correct):
    return {
        "item_id": item,
        "forecast_date": date(2026, 1, day),
        "direction_correct": correct,
    }


def _arm(pattern):
    """pattern: {day: [(item, correct), ...]}"""
    return [_rec(item, day, correct)
            for day, entries in pattern.items()
            for item, correct in entries]


def test_identical_arms_have_zero_mean_difference():
    pattern = {d: [(f"i{i}", i % 2) for i in range(10)] for d in range(1, 6)}
    out = paired_da_difference(_arm(pattern), _arm(pattern))
    assert out["mean_diff_pp"] == pytest.approx(0.0)
    assert out["n_paired"] == 50
    assert out["n_dates"] == 5


def test_uniformly_better_arm_reports_positive_difference():
    a = {d: [(f"i{i}", 0) for i in range(10)] for d in range(1, 6)}
    b = {d: [(f"i{i}", 1) for i in range(10)] for d in range(1, 6)}
    out = paired_da_difference(_arm(a), _arm(b))
    # b is arm 2; the difference is reported as b - a.
    assert out["mean_diff_pp"] == pytest.approx(100.0)


def test_only_rows_present_in_both_arms_are_paired():
    a = _arm({1: [("x", 1), ("y", 1)]})
    b = _arm({1: [("x", 0)]})
    out = paired_da_difference(a, b)
    assert out["n_paired"] == 1


def test_mde_is_the_half_width_of_the_difference_interval():
    a = {d: [(f"i{i}", 1) for i in range(10)] for d in range(1, 8)}
    b = {d: [(f"i{i}", 1 if d % 2 else 0) for i in range(10)] for d in range(1, 8)}
    out = paired_da_difference(_arm(a), _arm(b))
    expected = (out["ci_upper_pp"] - out["ci_lower_pp"]) / 2
    assert out["mde_pp"] == pytest.approx(expected)
    assert out["mde_pp"] > 0


def test_single_date_yields_no_interval():
    a = _arm({1: [(f"i{i}", 1) for i in range(10)]})
    b = _arm({1: [(f"i{i}", 0) for i in range(10)]})
    out = paired_da_difference(a, b)
    # One date carries no information about between-date variation.
    assert out["ci_lower_pp"] is None
    assert out["ci_upper_pp"] is None
    assert out["mde_pp"] is None


def test_no_overlap_raises():
    a = _arm({1: [("x", 1)]})
    b = _arm({2: [("y", 1)]})
    with pytest.raises(ValueError, match="no paired records"):
        paired_da_difference(a, b)
