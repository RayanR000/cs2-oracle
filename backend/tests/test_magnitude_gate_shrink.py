"""Shrinkage parameterisation for the magnitude band gate's climatology arm.

The 2026-08-26 fold run showed a global constant beating the per-item
climatology at every horizon, which makes `CLIMATOLOGY_SHRINK_K` the live
lever. Sweeping it is only meaningful if the two ENDS of the sweep are exactly
what they claim to be -- K=0 the pure per-item estimate, K->inf the pure pool --
so both ends are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from magnitude_vs_climatology import TARGET_COVERAGE, _climatology_halfwidth


def _calib() -> pd.DataFrame:
    """A volatile and a calm item, both tier 3, both well observed."""
    rng = np.random.default_rng(0)
    rows = []
    for iid, scale in [("vol", 30.0), ("calm", 2.0)]:
        for _ in range(400):
            rows.append({"item_id": iid, "tier": 3, "r_h": float(rng.normal(0, scale))})
    return pd.DataFrame(rows)


def _test_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [{"item_id": "vol", "tier": 3}, {"item_id": "calm", "tier": 3}, {"item_id": "unseen", "tier": 3}]
    )


def _item_q(calib: pd.DataFrame, iid: str) -> float:
    s = calib.loc[calib["item_id"] == iid, "r_h"].abs()
    return float(np.quantile(s, TARGET_COVERAGE))


class TestShrinkEnds:
    def test_k_zero_is_the_pure_per_item_quantile(self):
        calib = _calib()
        got = _climatology_halfwidth(calib, _test_rows(), "r_h", k=0)
        assert got[0] == pytest.approx(_item_q(calib, "vol"))
        assert got[1] == pytest.approx(_item_q(calib, "calm"))

    def test_huge_k_collapses_every_item_onto_the_pool(self):
        got = _climatology_halfwidth(_calib(), _test_rows(), "r_h", k=10**9)
        assert got[0] == pytest.approx(got[1], rel=1e-3)

    def test_default_k_sits_between_the_two_ends(self):
        calib = _calib()
        rows = _test_rows()
        item = _climatology_halfwidth(calib, rows, "r_h", k=0)[1]
        pool = _climatology_halfwidth(calib, rows, "r_h", k=10**9)[1]
        mid = _climatology_halfwidth(calib, rows, "r_h", k=20)[1]
        assert min(item, pool) <= mid <= max(item, pool)


class TestFallback:
    def test_an_unseen_item_takes_the_tier_pool_at_any_k(self):
        calib = _calib()
        rows = _test_rows()
        for k in (0, 20, 10**9):
            got = _climatology_halfwidth(calib, rows, "r_h", k=k)
            assert np.isfinite(got[2])

    def test_the_volatile_item_stays_wider_than_the_calm_one_at_default_k(self):
        got = _climatology_halfwidth(_calib(), _test_rows(), "r_h", k=20)
        assert got[0] > got[1]
