"""Phase 4 PR 1: the forecast write goes to the partitions only, keeps ids on a re-run, and fails loudly.

The old-vs-new parity (22,516 rows x 3 nights, 0 column diffs) was run with scripts/forecast_write_parity.py
at the commit before the wiring; see PR #137."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from db import ops_partitions
from scripts import forecast_prices as fp

NIGHT = date.today() - timedelta(days=3)  # an open month


def test_a_rerun_of_the_same_night_keeps_every_id_and_new_keys_take_fresh_ones(tmp_path, monkeypatch):
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    frows = [
        {
            "item_id": i,
            "forecast_date": NIGHT,
            "horizon_days": 3,
            "price_low": 1.0,
            "price_mid": 2.0,
            "price_high": 3.0,
            "current_price": 2.0,
            "direction": "up",
            "confidence": "high",
            "model_version": "v",
            "anchor_clean": True,
            "anchor_wedge_pct": 0.0,
            "exceed_p": None,
            "anomaly_p": None,
            "band_multiplier": 1.0,
            "created_at": pd.Timestamp("2026-10-20 03:00:00"),
        }
        for i in (1, 2)
    ]
    slugs = {1: "a", 2: "b", 3: "c"}
    fp._write_forecasts_partitions(frows, slugs, "regime")
    first = _ids(tmp_path)
    fp._write_forecasts_partitions(frows, slugs, "regime")  # the same night again
    assert _ids(tmp_path) == first
    fp._write_forecasts_partitions([{**frows[0], "item_id": 3}], slugs, "regime")
    third = _ids(tmp_path)
    assert {k: v for k, v in third.items() if k in first} == first
    assert third[(3, NIGHT, 3)] > max(first.values())
    assert min(third.values()) >= ops_partitions.READ_ONLY_ID_BASE


def _ids(root):
    import duckdb

    files = ", ".join(f"'{f}'" for f in sorted((root / "item_forecasts").glob("*.parquet")))
    rows = duckdb.sql(f"SELECT item_id, forecast_date, horizon_days, id FROM read_parquet([{files}])").fetchall()
    return {(i, d, h): int(x) for i, d, h, x in rows}


def test_two_batches_on_one_night_both_survive(tmp_path, monkeypatch):
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    slugs = {1: "a", 2: "b"}
    fp._write_forecasts_partitions([_row(1)], slugs, "regime")
    fp._write_forecasts_partitions([_row(2)], slugs, "global-only")
    assert set(_ids(tmp_path)) == {(1, NIGHT, 3), (2, NIGHT, 3)}


def test_a_partition_failure_raises_instead_of_losing_the_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    monkeypatch.setattr(ops_partitions, "upsert", _boom)
    with pytest.raises(RuntimeError, match="disk full"):
        fp._write_forecasts_partitions([_row(1)], {1: "a"}, "regime")


def test_a_closed_month_is_refused_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    with pytest.raises(ops_partitions.PartitionError, match="closed"):
        fp._write_forecasts_partitions([{**_row(1), "forecast_date": date(2026, 8, 11)}], {1: "a"}, "regime")


def _boom(*a, **k):
    raise RuntimeError("disk full")


def _row(item_id):
    return {
        "item_id": item_id,
        "forecast_date": NIGHT,
        "horizon_days": 3,
        "price_low": 1.0,
        "price_mid": 2.0,
        "price_high": 3.0,
        "current_price": 2.0,
        "direction": "up",
        "confidence": "high",
        "model_version": "v",
        "anchor_clean": True,
        "anchor_wedge_pct": 0.0,
        "exceed_p": None,
        "anomaly_p": None,
        "band_multiplier": 1.0,
        "created_at": pd.Timestamp("2026-10-20 03:00:00"),
    }
