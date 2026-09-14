"""Shelved-and-unused features are not just dropped from training — their
computation is skipped on the production path.

The volume pipeline (``_compute_volume_features`` + its sidecar read) and the
2026-07-26 price primitives (``vol_semidev_*``, ``vol_skew_30d``,
``rsi_divergence_7d``, ``rsi_price_divergence_7d``, ``macd_hist_slope_7d``) are
all in ``SHELVED_FEATURES``: the trainer never fits on them. On the production
frame build (``skip_unused_groups=True``, allowlist ``["price_technicals"]``)
their only live consumers -- ``supply_to_volume_ratio`` (``supply_depth``) and
``item_volume_vs_market_30d`` (``cross_sectional``) -- are themselves skipped,
so computing them is pure waste (see docs finding #1/#2).

These tests pin the contract: skip the compute where it is provably unused,
recompute it byte-identically everywhere it is still consumed (the A/B
harnesses and the predict path, which call ``engineer_features`` with
``skip_unused_groups=False``; the ``VOLUME_FEATURES=1`` reinstate path; and any
allowlist that admits a consuming group).
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster

VOLUME_COLS = frozenset(ItemForecaster.VOLUME_FEATURE_NAMES)
PRIMITIVE_COLS = frozenset(
    {
        "vol_semidev_down_30d",
        "vol_semidev_up_30d",
        "vol_skew_30d",
        "rsi_divergence_7d",
        "rsi_price_divergence_7d",
        "macd_hist_slope_7d",
    }
)
_EVENTS = pd.DataFrame(columns=["date", "event_type", "name"])


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _series(n_items=2, n_days=120, start=date(2024, 1, 1)):
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_items):
        price = 10.0 + i
        for d in range(n_days):
            price *= 1.0 + rng.normal(0.0005, 0.02)
            rows.append(
                {
                    "item_id": f"item-{i}",
                    "date": start + timedelta(days=d),
                    "price": price,
                    "volume": float(rng.integers(1, 100)),
                }
            )
    return pd.DataFrame(rows)


def test_prod_frame_skips_the_shelved_volume_and_primitive_compute(forecaster):
    """skip_unused_groups=True with the default allowlist: waste is not built."""
    df = forecaster.engineer_features(_series(), _EVENTS, skip_unused_groups=True)
    assert VOLUME_COLS.isdisjoint(df.columns), (
        "volume pipeline computed on the prod path though every column is "
        "shelved and its consumers skipped: "
        f"{sorted(VOLUME_COLS & set(df.columns))}"
    )
    assert PRIMITIVE_COLS.isdisjoint(df.columns), (
        f"shelved price primitives computed on the prod path: {sorted(PRIMITIVE_COLS & set(df.columns))}"
    )
    # The one allowlisted group is unaffected.
    assert "return_7d" in df.columns


def test_ab_and_predict_path_still_computes_them(forecaster):
    """skip_unused_groups=False (A/B harnesses, predict): byte-identical build."""
    df = forecaster.engineer_features(_series(), _EVENTS, skip_unused_groups=False)
    assert set(df.columns) >= VOLUME_COLS, (
        f"volume columns missing on the full-frame path the A/B harness reads: {sorted(VOLUME_COLS - set(df.columns))}"
    )
    assert set(df.columns) >= PRIMITIVE_COLS, (
        f"shelved primitives missing on the full-frame path: {sorted(PRIMITIVE_COLS - set(df.columns))}"
    )


def test_volume_features_flag_reinstates_the_compute(forecaster, monkeypatch):
    """VOLUME_FEATURES=1 is the documented reinstate path — must recompute."""
    monkeypatch.setenv("VOLUME_FEATURES", "1")
    df = forecaster.engineer_features(_series(), _EVENTS, skip_unused_groups=True)
    assert set(df.columns) >= VOLUME_COLS, (
        f"VOLUME_FEATURES=1 did not reinstate the volume compute: {sorted(VOLUME_COLS - set(df.columns))}"
    )


def test_admitting_a_consuming_group_keeps_volume_computed(forecaster, monkeypatch):
    """supply_depth in the allowlist ⇒ supply_to_volume_ratio needs volume."""
    monkeypatch.setattr(ItemForecaster, "FEATURE_GROUP_ALLOWLIST", ["price_technicals", "supply_depth"])
    df = forecaster.engineer_features(_series(), _EVENTS, skip_unused_groups=True)
    assert set(df.columns) >= VOLUME_COLS, (
        f"volume compute skipped while a consuming group is admitted: {sorted(VOLUME_COLS - set(df.columns))}"
    )
