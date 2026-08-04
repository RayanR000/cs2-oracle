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


def test_bootstrap_is_seeded_deterministically():
    """Regression pin on bootstrap outputs under BOOTSTRAP_RNG_SEED = 42.

    This dataset is crafted to produce a non-zero MDE under the seeded
    bootstrap. If the resampling logic changes, this hard-coded output
    will alert us. Changing the bootstrap constants or the resampling
    strategy will change this output.
    """
    # Create test data with varying differences across dates
    # Date 1: all +1, Date 2: mixed +0, Date 3: all -1, Date 4: mixed -1
    a = [_rec(f"i{i}", 1, 1) for i in range(5)] + \
        [_rec(f"i{i}", 2, 1) for i in range(5)] + \
        [_rec(f"i{i}", 3, 1) for i in range(5)] + \
        [_rec(f"i{i}", 4, 1) for i in range(5)]

    b = [_rec(f"i{i}", 1, 1) for i in range(5)] + \
        [_rec(f"i{i}", 2, 1 if i < 2 else 0) for i in range(5)] + \
        [_rec(f"i{i}", 3, 0) for i in range(5)] + \
        [_rec(f"i{i}", 4, 0 if i < 2 else 1) for i in range(5)]

    out = paired_da_difference(a, b)
    # These are recorded outputs under BOOTSTRAP_RNG_SEED = 42
    assert out["mean_diff_pp"] == pytest.approx(-50.0, abs=0.01)
    assert out["ci_lower_pp"] == pytest.approx(-85.0, abs=0.1)
    assert out["ci_upper_pp"] == pytest.approx(-15.0, abs=0.1)
    assert out["mde_pp"] == pytest.approx(35.0, abs=0.1)


def test_varying_differences_across_dates_discriminates_resampling_axis():
    """Verify the resampling unit is dates, not rows.

    Constructs a fixture where per-date group means *vary* while values are
    homogeneous *within* each date:
    - date 1: 5 records, all differences +1 (A=0, B=1)
    - date 2: 10 records, all differences 0 (A=0, B=0)
    - date 3: 20 records, all differences -1 (A=1, B=0)

    Under date-level resampling (correct): bootstrap draws 3 dates with
    replacement, each contributing its fixed value, so the sampling
    distribution is wide (CI = [-100, +100], MDE = 100).

    Under row-level resampling (bug): bootstrap draws 35 rows with
    replacement, giving a narrower distribution (CI ≈ [-65.71, -17.14],
    MDE ≈ 24.29 pp).

    This fixture discriminates because the two resampling schemes yield
    materially different sampling distributions on this data. The
    mean_diff_pp also exercises count-weighted pooling: it should be
    the weighted average (5*1 + 10*0 + 20*(-1)) / (5+10+20) = -42.857%,
    not the unweighted mean of date means (which would be 0%).
    """
    a = [_rec(f"i{i}", 1, 0) for i in range(5)] + \
        [_rec(f"i{i}", 2, 0) for i in range(10)] + \
        [_rec(f"i{i}", 3, 1) for i in range(20)]

    b = [_rec(f"i{i}", 1, 1) for i in range(5)] + \
        [_rec(f"i{i}", 2, 0) for i in range(10)] + \
        [_rec(f"i{i}", 3, 0) for i in range(20)]

    out = paired_da_difference(a, b)

    # Count-weighted mean: (5*1 + 10*0 + 20*(-1)) / 35 = -15/35 ≈ -42.857%
    # Also verifies pooling is count-weighted, not a naive date-mean average
    assert out["mean_diff_pp"] == pytest.approx(-42.857, abs=0.01)

    # Date-level resampling yields wide CI; row-level would yield ~24.3pp MDE
    assert out["ci_lower_pp"] == pytest.approx(-100.0, abs=0.1)
    assert out["ci_upper_pp"] == pytest.approx(100.0, abs=0.1)
    assert out["mde_pp"] == pytest.approx(100.0, abs=0.1)

    # Discriminating bound: if resampling regressed to row-level, MDE ≈ 24.3pp
    assert out["mde_pp"] > 50.0
