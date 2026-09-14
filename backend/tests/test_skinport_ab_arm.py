"""Skinport live-volume A/B arm: the loader must fail loudly on a stalled feed."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from scripts.ab_test_volume_features import _load_skinport_live_volume  # noqa: E402


def test_loader_raises_with_no_volume_files(tmp_path):
    con = duckdb.connect()
    try:
        with pytest.raises(RuntimeError, match="no volume-"):
            _load_skinport_live_volume(con, ["some-item"], archive_dir=tmp_path)
    finally:
        con.close()


def test_loader_reads_sales_24h(tmp_path):
    pd.DataFrame(
        [
            {
                "item_slug": "some-item",
                "day": date(2026, 8, 9),
                "source": "skinport_sales",
                "sales_24h": 7,
                "sales_7d": 40,
                "sales_30d": 150,
                "sales_90d": 400,
                "median_30d": 10.0,
            }
        ]
    ).to_parquet(tmp_path / "volume-2026-08.parquet", index=False)
    con = duckdb.connect()
    try:
        df = _load_skinport_live_volume(con, ["some-item"], archive_dir=tmp_path)
    finally:
        con.close()
    assert df.set_index("item_id").loc["some-item", "live_volume"] == 7
