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


def test_ridge_arm_requires_the_scaler_to_fit_ill_scaled_features():
    """A same-scale fixture (like the test above) can't tell a scaled Ridge
    from an unscaled one -- StandardScaler is nearly a no-op on N(0,1) data.
    This fixture uses four columns spanning six orders of magnitude (1,
    1000, 0.001, 1), with the *small*-scale column (index 2) carrying the
    large coefficient. That combination is decisive: measured against the
    real `_ridge_predict`, corr(prediction, noise-free truth) is ~1.0000
    with the StandardScaler present and ~0.3365 with it removed. The 0.90
    bound below only passes when the scaler is doing its job.
    """
    rng = np.random.default_rng(0)
    n_train, n_val = 200, 40
    X_train = np.column_stack([
        rng.normal(0, 1, n_train),
        rng.normal(0, 1000, n_train),
        rng.normal(0, 0.001, n_train),
        rng.normal(0, 1, n_train),
    ])
    y_train = X_train[:, 0] * 2.0 + X_train[:, 2] * 5000.0 + rng.normal(0, 0.1, n_train)
    X_val = np.column_stack([
        rng.normal(0, 1, n_val),
        rng.normal(0, 1000, n_val),
        rng.normal(0, 0.001, n_val),
        rng.normal(0, 1, n_val),
    ])
    y_val_true = X_val[:, 0] * 2.0 + X_val[:, 2] * 5000.0

    mid, low, high, classes = wf._ridge_predict(X_train, y_train, X_val)

    corr = np.corrcoef(mid, y_val_true)[0, 1]
    assert corr > 0.90, (
        f"corr={corr:.4f}; expected ~1.0000 with StandardScaler applied "
        "(measured ~0.3365 with the scaler removed) -- if this drops below "
        "0.90 the scaler was likely removed from _ridge_predict"
    )


def test_ridge_arm_tolerates_non_finite_features():
    """LightGBM accepts +-inf; sklearn's Ridge/StandardScaler raise on it.

    `price_log` is log(price), so a zero price in the archive yields -inf and
    the column survives the gate's `fillna(medians)` (fillna does not touch
    infinities). Arm C died on real data with "Input X contains infinity or a
    value too large for dtype('float64')" after ~5 minutes, having produced no
    records, while arms A/B/D completed -- so the baseline silently dropped out
    of the comparison rather than failing loudly up front.
    """
    rng = np.random.RandomState(0)
    X_train = rng.rand(80, 3)
    X_train[3, 1] = -np.inf
    X_train[7, 2] = np.inf
    y_train = rng.rand(80)
    X_val = rng.rand(10, 3)
    X_val[2, 0] = -np.inf

    mid, low, high, classes = wf._ridge_predict(X_train, y_train, X_val)

    assert np.isfinite(mid).all()
    assert len(mid) == len(X_val)
    assert np.all(low <= mid) and np.all(mid <= high)


def test_item_selection_is_deterministic_under_row_count_ties():
    """The 60-item universe must be reproducible across runs.

    `ORDER BY row_count DESC` alone leaves ties broken by whatever order
    DuckDB happens to produce. Measured consequence: arms A and B drew
    universes differing by 2 of 60 items, so only 96.66% of
    (item_id, forecast_date) keys paired. That is not a 3.3% data loss -- the
    gate's feature list falls back to every numeric column, including
    cross-sectional features computed ACROSS the universe, and the >0.95
    correlation prune is data-dependent, so swapping two items perturbs every
    row. paired_da_difference raises only on ZERO overlap, so this drift is
    otherwise silent.
    """
    import duckdb
    src = inspect.getsource(wf._load_parquet_items)
    assert "ORDER BY row_count DESC, item_slug" in src, (
        "row_count needs a tie-break on item_slug or the universe is not "
        "reproducible between runs"
    )

    con = duckdb.connect()
    # Every slug has an identical row_count, so ordering is decided entirely
    # by the tie-break.
    con.sql("""
        CREATE TABLE t AS
        SELECT * FROM (VALUES
            ('zeta', DATE '2024-01-01'), ('alpha', DATE '2024-01-01'),
            ('mu', DATE '2024-01-01'), ('beta', DATE '2024-01-01')
        ) AS v(item_slug, day)
    """)
    rows = con.sql("""
        SELECT item_slug, COUNT(*) AS row_count FROM t
        GROUP BY item_slug ORDER BY row_count DESC, item_slug
    """).fetchall()
    assert [r[0] for r in rows] == ["alpha", "beta", "mu", "zeta"]
