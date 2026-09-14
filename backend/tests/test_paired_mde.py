"""Paired MDE for arm-vs-arm comparison on shared folds.

The tests below that pass `cluster_key="forecast_date"` do so deliberately:
their fixtures predate `fold_id` and they pin the date-grain resampling
behaviour. Production callers cluster on `fold_id` — see the fold-clustering
block at the bottom of this file and the `paired_mde` module docstring.
"""

from __future__ import annotations

from datetime import date

import pytest
from backtest.paired_mde import paired_da_difference


def _rec(item, day, correct, fold=None):
    r = {
        "item_id": item,
        "forecast_date": date(2026, 1, day),
        "direction_correct": correct,
    }
    if fold is not None:
        r["fold_id"] = fold
    return r


def _arm(pattern):
    """pattern: {day: [(item, correct), ...]}"""
    return [_rec(item, day, correct) for day, entries in pattern.items() for item, correct in entries]


def test_identical_arms_have_zero_mean_difference():
    pattern = {d: [(f"i{i}", i % 2) for i in range(10)] for d in range(1, 6)}
    out = paired_da_difference(_arm(pattern), _arm(pattern), cluster_key="forecast_date")
    assert out["mean_diff_pp"] == pytest.approx(0.0)
    assert out["n_paired"] == 50
    assert out["n_dates"] == 5


def test_uniformly_better_arm_reports_positive_difference():
    a = {d: [(f"i{i}", 0) for i in range(10)] for d in range(1, 6)}
    b = {d: [(f"i{i}", 1) for i in range(10)] for d in range(1, 6)}
    out = paired_da_difference(_arm(a), _arm(b), cluster_key="forecast_date")
    # b is arm 2; the difference is reported as b - a.
    assert out["mean_diff_pp"] == pytest.approx(100.0)


def test_only_rows_present_in_both_arms_are_paired():
    a = _arm({1: [("x", 1), ("y", 1)]})
    b = _arm({1: [("x", 0)]})
    out = paired_da_difference(a, b, cluster_key="forecast_date")
    assert out["n_paired"] == 1


def test_mde_is_the_half_width_of_the_difference_interval():
    a = {d: [(f"i{i}", 1) for i in range(10)] for d in range(1, 8)}
    b = {d: [(f"i{i}", 1 if d % 2 else 0) for i in range(10)] for d in range(1, 8)}
    out = paired_da_difference(_arm(a), _arm(b), cluster_key="forecast_date")
    expected = (out["ci_upper_pp"] - out["ci_lower_pp"]) / 2
    assert out["mde_pp"] == pytest.approx(expected)
    assert out["mde_pp"] > 0


def test_single_date_yields_no_interval():
    a = _arm({1: [(f"i{i}", 1) for i in range(10)]})
    b = _arm({1: [(f"i{i}", 0) for i in range(10)]})
    out = paired_da_difference(a, b, cluster_key="forecast_date")
    # One date carries no information about between-date variation.
    assert out["ci_lower_pp"] is None
    assert out["ci_upper_pp"] is None
    assert out["mde_pp"] is None


def test_no_overlap_raises():
    a = _arm({1: [("x", 1)]})
    b = _arm({2: [("y", 1)]})
    with pytest.raises(ValueError, match="no paired records"):
        paired_da_difference(a, b, cluster_key="forecast_date")


def test_bootstrap_is_seeded_deterministically():
    """Regression pin on bootstrap outputs under BOOTSTRAP_RNG_SEED = 42.

    This dataset is crafted to produce a non-zero MDE under the seeded
    bootstrap. If the resampling logic changes, this hard-coded output
    will alert us. Changing the bootstrap constants or the resampling
    strategy will change this output.
    """
    # Create test data with varying differences across dates
    # Date 1: all +1, Date 2: mixed +0, Date 3: all -1, Date 4: mixed -1
    a = (
        [_rec(f"i{i}", 1, 1) for i in range(5)]
        + [_rec(f"i{i}", 2, 1) for i in range(5)]
        + [_rec(f"i{i}", 3, 1) for i in range(5)]
        + [_rec(f"i{i}", 4, 1) for i in range(5)]
    )

    b = (
        [_rec(f"i{i}", 1, 1) for i in range(5)]
        + [_rec(f"i{i}", 2, 1 if i < 2 else 0) for i in range(5)]
        + [_rec(f"i{i}", 3, 0) for i in range(5)]
        + [_rec(f"i{i}", 4, 0 if i < 2 else 1) for i in range(5)]
    )

    out = paired_da_difference(a, b, cluster_key="forecast_date")
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
    a = (
        [_rec(f"i{i}", 1, 0) for i in range(5)]
        + [_rec(f"i{i}", 2, 0) for i in range(10)]
        + [_rec(f"i{i}", 3, 1) for i in range(20)]
    )

    b = (
        [_rec(f"i{i}", 1, 1) for i in range(5)]
        + [_rec(f"i{i}", 2, 0) for i in range(10)]
        + [_rec(f"i{i}", 3, 0) for i in range(20)]
    )

    out = paired_da_difference(a, b, cluster_key="forecast_date")

    # Count-weighted mean: (5*1 + 10*0 + 20*(-1)) / 35 = -15/35 ≈ -42.857%
    # Also verifies pooling is count-weighted, not a naive date-mean average
    assert out["mean_diff_pp"] == pytest.approx(-42.857, abs=0.01)

    # Date-level resampling yields wide CI; row-level would yield ~24.3pp MDE
    assert out["ci_lower_pp"] == pytest.approx(-100.0, abs=0.1)
    assert out["ci_upper_pp"] == pytest.approx(100.0, abs=0.1)
    assert out["mde_pp"] == pytest.approx(100.0, abs=0.1)

    # Discriminating bound: if resampling regressed to row-level, MDE ≈ 24.3pp
    assert out["mde_pp"] > 50.0


# --------------------------------------------------------------------------
# Fold clustering (2026-08-07). The bug this replaces made intervals too NARROW,
# so these tests assert the direction of the correction, not just that it runs.
# --------------------------------------------------------------------------


def _folded(n_folds, dates_per_fold, rows_per_date, diff_by_fold):
    """Two arms whose difference is constant within a fold and varies across.

    This is the real data's shape: one trained model per fold, many correlated
    dates inside it. Date-grain resampling sees `n_folds * dates_per_fold`
    "independent" units where only `n_folds` exist.
    """
    a, b, day = [], [], 1
    for f in range(n_folds):
        for _ in range(dates_per_fold):
            for i in range(rows_per_date):
                a.append(_rec(f"i{i}", day, 0, fold=f))
                b.append(_rec(f"i{i}", day, diff_by_fold[f], fold=f))
            day += 1
    return a, b


def test_fold_clustering_is_the_default():
    a, b = _folded(4, 3, 5, [1, 0, 1, 0])
    out = paired_da_difference(a, b)
    assert out["cluster_key"] == "fold_id"
    assert out["n_clusters"] == 4
    assert out["n_dates"] == 12


def test_clustering_on_folds_is_wider_than_clustering_on_dates():
    """The correction must widen the interval, or it does not fix the bug.

    Same records, same pairing, only the resampling grain differs. Dates inside
    a fold carry identical differences here, so date-grain resampling is
    counting the same information 3x over and reports a falsely tight interval.
    """
    a, b = _folded(6, 3, 5, [1, 0, 1, 0, 1, 0])
    folded = paired_da_difference(a, b)
    dated = paired_da_difference(a, b, cluster_key="forecast_date")
    assert folded["n_clusters"] == 6
    assert dated["n_clusters"] == 18
    assert folded["mde_pp"] > dated["mde_pp"]


def test_pairing_grain_is_unchanged_by_the_cluster_grain():
    """Only the interval moves; the point estimate must not."""
    a, b = _folded(5, 3, 4, [1, 0, 1, 0, 1])
    folded = paired_da_difference(a, b)
    dated = paired_da_difference(a, b, cluster_key="forecast_date")
    assert folded["mean_diff_pp"] == pytest.approx(dated["mean_diff_pp"])
    assert folded["n_paired"] == dated["n_paired"]


def test_missing_fold_id_raises_rather_than_falling_back_to_dates():
    """Silent fallback would reinstate the under-dispersion bug."""
    a = [_rec("x", 1, 1), _rec("x", 2, 1)]
    b = [_rec("x", 1, 0), _rec("x", 2, 0)]
    with pytest.raises(ValueError, match="missing cluster key"):
        paired_da_difference(a, b)


def test_a_none_fold_id_is_treated_as_missing():
    """`fold_records` defaults fold_id to None; that must not become a cluster."""
    a = [_rec("x", 1, 1, fold=None) | {"fold_id": None}]
    b = [_rec("x", 1, 0, fold=None) | {"fold_id": None}]
    with pytest.raises(ValueError, match="missing cluster key"):
        paired_da_difference(a, b)


def test_single_fold_spanning_many_dates_yields_no_interval():
    """One trained model is one observation, however many dates it scored.

    Under the old date grain this returned a confident interval off 10 dates.
    """
    a, b = _folded(1, 10, 5, [1])
    out = paired_da_difference(a, b)
    assert out["n_dates"] == 10
    assert out["n_clusters"] == 1
    assert out["mde_pp"] is None
