import numpy as np
import pandas as pd
import pytest

from scripts.replay_lambdarank import _anchor_metrics, MIN_TIED_ROWS


def _synthetic_anchor(n, signal, rng):
    """One anchor date with `n` items; `signal` scales how well score predicts
    the realised outcome. Returns (val_df, lr, q50, naive, outcomes, tied)."""
    anchor = pd.Timestamp("2026-05-01")
    item_id = np.arange(n)
    current = rng.uniform(2, 50, size=n)  # all >= $1 floor
    true_rank = rng.normal(size=n)
    lr = true_rank + 0.05 * rng.normal(size=n)
    q50 = 0.3 * true_rank + rng.normal(size=n)  # weaker orderer
    naive = rng.normal(size=n)
    realised = current * (1.0 + signal * true_rank * 0.01)
    val_df = pd.DataFrame({"item_id": item_id, "current": current,
                           "date": anchor})
    # Outcome rows land inside the h=7 resolve window (anchor, anchor+7].
    out_day = anchor + pd.Timedelta(days=7)
    outcomes = pd.DataFrame({"item_id": item_id,
                             "day": np.repeat(out_day, n),
                             "price": realised})
    tied = pd.Series(True, index=item_id)  # everyone tied in the synthetic
    tied.index.name = "item_id"
    return val_df, lr, q50, naive, outcomes, tied


def test_returns_none_below_min_tied(monkeypatch):
    rng = np.random.default_rng(0)
    val_df, lr, q50, naive, outcomes, tied = _synthetic_anchor(
        MIN_TIED_ROWS - 1, signal=1.0, rng=rng)
    out = _anchor_metrics(pd.Timestamp("2026-05-01"), 7, val_df, lr, q50,
                          naive, outcomes, floor=1.0, tied=tied)
    assert out is None


def test_ranker_beats_q50_on_a_planted_signal():
    rng = np.random.default_rng(1)
    val_df, lr, q50, naive, outcomes, tied = _synthetic_anchor(
        300, signal=1.0, rng=rng)
    out = _anchor_metrics(pd.Timestamp("2026-05-01"), 7, val_df, lr, q50,
                          naive, outcomes, floor=1.0, tied=tied)
    assert out is not None
    assert out["n_tied"] == 300
    assert out["lr_ic"] > out["q50_ic"]   # lr orders the planted signal better
    assert out["ls_spread"] > 0           # top decile outperforms bottom
    assert len(out["pt_records"]) == 300
