"""Regression tests for the vectorized bootstrap CIs (PR 4, Task 4.2).

`bootstrap_ci` / `block_bootstrap_ci` draw from a module-fixed seed, so
"deterministic" here means two consecutive calls return identical bounds.
Both return (None, None) below their minimum data floors (<10 values,
<2 clusters) rather than fabricating an interval.
"""

import numpy as np

from backtest.scoring import block_bootstrap_ci, bootstrap_ci


def test_bootstrap_ci_deterministic():
    arr = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )
    lo, hi = bootstrap_ci(arr, n_resamples=1000)
    # Re-run — same fixed seed, must match exactly
    lo2, hi2 = bootstrap_ci(arr, n_resamples=1000)
    assert lo == lo2 and hi == hi2
    # Sanity: the interval brackets the sample mean
    assert lo <= float(np.mean(arr)) <= hi


def test_bootstrap_ci_too_few_values():
    lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0]), n_resamples=100)
    assert lo is None and hi is None


def test_block_bootstrap_ci_deterministic():
    sums = np.array([10.0, 20.0, 30.0, 40.0])
    counts = np.array([5, 10, 15, 20])
    values = np.repeat(sums / counts, counts.astype(int))
    clusters = np.repeat(["a", "b", "c", "d"], counts.astype(int))
    lo, hi = block_bootstrap_ci(values, clusters, n_resamples=1000)
    lo2, hi2 = block_bootstrap_ci(values, clusters, n_resamples=1000)
    assert lo == lo2 and hi == hi2


def test_block_bootstrap_ci_single_cluster():
    values = [1, 0, 1, 0] * 50
    clusters = ["only-date"] * len(values)
    lo, hi = block_bootstrap_ci(values, clusters, n_resamples=100)
    assert lo is None and hi is None


def test_block_bootstrap_ci_empty():
    lo, hi = block_bootstrap_ci([], [], n_resamples=100)
    assert lo is None and hi is None
