"""Tests for the gated order-book feature group (default-off A/B instrument).

The group derives ASK-SIDE proxies from the free supply panel's lis-skins
ladder — no paid book feed. Bid-side quantities (spread, imbalance, impact)
have no free source and are not represented here.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from models.forecaster import ItemForecaster, _feature_group  # noqa: E402


class _StubForecaster(ItemForecaster):
    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
        self._orderbook_cache = None


def _supply_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _ladder_row(slug="a", day=date(2026, 9, 8), source="lis_skins", **kw):
    base = {"item_slug": slug, "snapshot_day": day, "source": source,
            "listing_count": 100, "p05_ask": 10.0, "p25_ask": 10.5,
            "depth_5pct": 30.0, "depth_10pct": 60.0,
            "age_median_days": 5.0, "inflow_24h": 8}
    base.update(kw)
    return base


def test_feature_group_routes_ob_prefix():
    assert _feature_group("ob_ladder_slope") == "orderbook"
    assert _feature_group("ob_present") == "orderbook"
    assert _feature_group("supply_change_7d") == "supply_depth"


def test_group_is_gated_off_by_default(monkeypatch):
    monkeypatch.delenv("ORDERBOOK_FEATURES", raising=False)
    assert _StubForecaster._orderbook_enabled() is False
    assert "orderbook" in _StubForecaster(Path("."))._skipped_feature_groups()


def test_group_admits_with_flag(monkeypatch):
    monkeypatch.setenv("ORDERBOOK_FEATURES", "1")
    assert _StubForecaster._orderbook_enabled() is True
    assert "orderbook" not in _StubForecaster(Path("."))._skipped_feature_groups()


def test_empty_book_is_nan_not_zero(tmp_path):
    fc = _StubForecaster(tmp_path)
    df = pd.DataFrame({"item_id": ["a"], "date": [date(2026, 9, 8)],
                       "volume_30d": [100.0]})
    out = fc._compute_orderbook_features(df)
    assert out["ob_present"].tolist() == [0]
    assert out["ob_ladder_slope"].isna().all()
    assert out["ob_turnover"].isna().all()


def test_non_ladder_sources_are_ignored(tmp_path):
    """Scalar feeds have no ladder columns worth of signal; only lis_skins feeds the group."""
    pd.DataFrame([_ladder_row(source="waxpeer")]).to_parquet(
        tmp_path / "supply-2026-09.parquet", index=False)
    fc = _StubForecaster(tmp_path)
    df = pd.DataFrame({"item_id": ["a"], "date": [date(2026, 9, 8)]})
    out = fc._compute_orderbook_features(df)
    assert out["ob_present"].tolist() == [0]


def test_ladder_proxies_and_turnover(tmp_path):
    pd.DataFrame([
        _ladder_row(day=date(2026, 9, 7), listing_count=90),
        _ladder_row(day=date(2026, 9, 8), listing_count=100),
    ]).to_parquet(tmp_path / "supply-2026-09.parquet", index=False)
    fc = _StubForecaster(tmp_path)
    df = pd.DataFrame({"item_id": ["a", "a"],
                       "date": [date(2026, 9, 7), date(2026, 9, 8)],
                       "volume_30d": [200.0, 200.0]})
    out = fc._compute_orderbook_features(df).sort_values("date").reset_index(drop=True)
    assert out["ob_present"].tolist() == [1, 1]
    assert out["ob_ladder_slope"].iloc[0] == 0.05
    assert out["ob_depth_concentration"].iloc[0] == 0.5
    assert out["ob_turnover"].iloc[0] == 200.0 / 30.0
    # Churn is NaN on the first day (no anchor), real on the second.
    assert pd.isna(out["ob_churn"].iloc[0])
    assert out["ob_churn"].iloc[1] > 0
