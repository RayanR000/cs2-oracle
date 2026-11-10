"""Phase 4 PR 1: the partitions-only forecast writer matches the dual-write, and keeps ids on a re-run."""

from __future__ import annotations

from datetime import date

import pandas as pd
from db import ops_partitions
from scripts import forecast_prices as fp
from scripts import forecast_write_parity as parity

NIGHT = date(2026, 10, 20)


def _night(tmp_path, n_items=4):
    rows = []
    for item in range(1, n_items + 1):
        for h in (3, 7, 14, 30):
            rows.append(
                {
                    "id": 10_000 + item * 10 + h,
                    "item_id": item,
                    "forecast_date": NIGHT,
                    "horizon_days": h,
                    "price_low": 8.0 + item,
                    "price_mid": 10.0 + item,
                    "price_high": 12.0 + item,
                    "current_price": 10.0 + item,
                    "direction": "up",
                    "confidence": "high",
                    "model_version": "lgbm-test",
                    "created_at": pd.Timestamp("2026-10-20 03:00:00"),
                    "anchor_clean": True,
                    "anchor_wedge_pct": 0.5,
                    "exceed_p": 0.2,
                    "anomaly_p": 0.1,
                    "band_multiplier": 0.55,
                    "model_config": "regime",
                    "item_slug": f"item_{item}",
                }
            )
    ops_partitions.upsert("item_forecasts", pd.DataFrame(rows), allow_closed=True, root=tmp_path)
    return tmp_path


def test_the_new_writer_is_identical_to_the_dual_write_on_a_real_shaped_night(tmp_path):
    report = parity.run(_night(tmp_path), NIGHT)
    assert report["identical"], report
    assert report["old_rows"] == report["new_rows"] == 16


def test_compare_flags_a_changed_price_and_a_missing_row():
    base = pd.DataFrame(
        {
            "id": [1, 2],
            "item_id": [1, 2],
            "forecast_date": [NIGHT, NIGHT],
            "horizon_days": [3, 3],
            "price_mid": [10.0, 11.0],
            "created_at": [pd.NaT, pd.NaT],
        }
    )
    changed = base.assign(price_mid=[10.0, 11.5])
    assert parity.compare(base, changed)["diffs"] == {"price_mid": 1}
    assert "_keys" in parity.compare(base, base.iloc[:1])["diffs"]


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
