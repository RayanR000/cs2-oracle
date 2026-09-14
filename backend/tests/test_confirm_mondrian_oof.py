"""Mondrian OOF confirm helpers: fold selection, OOF panel, default params.

What is silent when it breaks: a test-fold pick that leaves no calibration
history, an OOF panel that drops the date join (fold is then the only
partition and the embargo is unenforced), and default params that drift
from the trainer's SKIP_HP block.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.confirm_mondrian_oof import (
    DEFAULT_Q50,
    default_q50_params,
    last_test_folds,
    oof_panel,
)


def _oof(n_folds=6, per_fold=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    base = np.datetime64("2024-01-01")
    for f in range(n_folds):
        # Folds as disjoint 30-day windows ~150 days apart, like production.
        start = base + np.timedelta64(f * 150, "D")
        for i in range(per_fold):
            day = start + np.timedelta64(int(i % 30), "D")
            s = float(rng.lognormal(-2.6, 0.6))
            rows.append(
                {
                    "residual_pct": float(rng.normal(0, s * 150)),
                    "sigma": s,
                    "fold": f,
                    "date": day,
                    "row_index": len(rows),
                }
            )
    return pd.DataFrame(rows)


def test_last_test_folds_are_last_and_oldest_first():
    oof = _oof()
    got = last_test_folds(oof, 3)
    assert got == [3, 4, 5], got


def test_last_test_folds_skips_thin_folds():
    oof = _oof()
    oof = oof[~(oof["fold"] == 5)].copy()
    thin = oof[oof["fold"] == 4].head(10).copy()
    oof = pd.concat([oof[oof["fold"] != 4], thin])
    got = last_test_folds(oof, 3, min_rows=1000)
    assert got[-1] != 4, got
    assert got == sorted(got)


def test_last_test_folds_raises_when_nothing_held_out():
    oof = _oof(n_folds=1)
    try:
        last_test_folds(oof, 3)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on a single fold")


def test_oof_panel_sorts_dates_and_drops_nonfinite():
    oof = _oof()
    oof.loc[oof.index[:5], "sigma"] = np.nan
    oof.loc[oof.index[5:8], "residual_pct"] = np.inf
    p = oof_panel(oof, 14)
    assert len(p.resid) == len(oof) - 8
    assert bool(np.all(np.diff(p.dates).astype(int) >= 0)), "dates unsorted"
    assert p.embargo == 14 + 13, p.embargo


def test_default_params_match_skip_hp_block():
    for k, v in DEFAULT_Q50.items():
        assert k in ("num_leaves", "learning_rate", "lambda_l1", "lambda_l2", "max_depth", "min_data_in_leaf"), k
    assert DEFAULT_Q50 == {
        "num_leaves": 47,
        "learning_rate": 0.01,
        "lambda_l1": 0.0,
        "lambda_l2": 1.5,
        "max_depth": 5,
        "min_data_in_leaf": 15,
    }


def test_default_q50_params_shape():
    from models.forecaster import ItemForecaster

    fc = ItemForecaster.__new__(ItemForecaster)
    p = default_q50_params(fc)
    assert p["objective"] == "quantile" and p["alpha"] == 0.5
    assert p["boosting_type"] == ItemForecaster.BOOSTING_TYPE
    assert p["max_bin"] == ItemForecaster.MAX_BIN
    for k in DEFAULT_Q50:
        assert p[k] == DEFAULT_Q50[k]
