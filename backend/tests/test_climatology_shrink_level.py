"""How hard the climatology band pools a WELL-OBSERVED item.

`CLIMATOLOGY_SHRINK_K` was validated at 20 over a 5..100 sweep
(docs/research/2026-08-19-climatology-vs-gbm-band.md). A 2026-08-26 walk-forward
sweep that ran the grid PAST 100 found the optimum on a broad 240..640 plateau:
at matched 80% coverage K=320 is 6.5 / 7.4 / 9.7 / 15.4% NARROWER than K=20 at
h=3/7/14/30, winning 4 of 4 folds at every horizon with every worst-fold ratio
below 1. The old sweep simply stopped short of the optimum.

This pins the CONSEQUENCE rather than the number: at the served K, an item with
200 calibration observations must sit nearer its tier pool than its own noisy
dispersion estimate. That is false at K=20 (weight 0.91 on the item) and true
at K=320 (weight 0.38), so this test discriminates the decision instead of
restating the constant.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.forecaster import ItemForecaster  # noqa: E402

TCOL = "target_return_7d"
N_OBS = 200


@pytest.fixture
def fc(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(),
                          model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _frame() -> pd.DataFrame:
    """One volatile item against a calm tier-3 pool, all well observed."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(N_OBS):
        rows.append({"item_id": "vol", "price": 50.0,
                     TCOL: float(rng.normal(0, 30.0))})
    for j in range(20):
        for _ in range(N_OBS):
            rows.append({"item_id": f"calm{j}", "price": 50.0,
                         TCOL: float(rng.normal(0, 2.0))})
    return pd.DataFrame(rows)


def _raw_item_q(df: pd.DataFrame, iid: str) -> float:
    s = df.loc[df["item_id"] == iid, TCOL].abs()
    return float(np.quantile(s, 0.80))


class TestWellObservedItemsArePooled:
    def test_a_200_observation_item_lands_nearer_its_tier_pool(self, fc):
        df = _frame()
        table, tier_pool, _ = fc._build_climatology_table(df, TCOL)
        served = table["vol"]
        raw = _raw_item_q(df, "vol")
        pool = tier_pool[3]
        assert abs(served - pool) < abs(served - raw), (
            f"served={served:.3f} sits nearer its own estimate {raw:.3f} than "
            f"the tier pool {pool:.3f}; K is too small to pool a 200-obs item")

    def test_the_shrink_weight_on_200_observations_is_below_a_half(self):
        w = N_OBS / (N_OBS + ItemForecaster.CLIMATOLOGY_SHRINK_K)
        assert w < 0.5


class TestPoolingStillPreservesOrdering:
    def test_a_volatile_item_stays_wider_than_a_calm_one(self, fc):
        """Pooling harder must not flatten the cross-item signal entirely --
        that is what makes this shrinkage rather than the pooled constant."""
        table, _, _ = fc._build_climatology_table(_frame(), TCOL)
        assert table["vol"] > table["calm0"]
