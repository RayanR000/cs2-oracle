"""scripts/ops_sync_from_postgres.py: fill ids and repair dates from Postgres."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from tests.test_ops_readback import D, _seed, _session

TODAY = date(2026, 10, 12)
TABLES = ["item_forecasts", "forecast_outcomes", "forecast_candidates", "forecast_candidate_outcomes"]


def _mirror_only_rows(tmp_path):
    """The 10-11 state: rows with no id, one extra column Postgres lacks."""
    from db.ops_partitions import upsert

    frame = pd.DataFrame(
        [
            {
                "id": None,
                "item_id": 1,
                "forecast_date": D,
                "horizon_days": 7,
                "price_mid": 10.0,
                "model_config": "global-only",
            },
            {
                "id": None,
                "item_id": 1,
                "forecast_date": D,
                "horizon_days": 3,
                "price_mid": 9.0,
                "model_config": "global-only",
            },
        ]
    )
    upsert("item_forecasts", frame, root=tmp_path, today=TODAY)


def test_dry_run_writes_nothing(tmp_path):
    from scripts.ops_sync_from_postgres import sync

    db = _session()
    _seed(db)
    _mirror_only_rows(tmp_path)
    before = (tmp_path / "item_forecasts" / "2026-10.parquet").read_bytes()
    plan = sync(db, [D], tables=TABLES, apply=False, root=tmp_path, today=TODAY)
    assert plan["item_forecasts"] == 2
    assert (tmp_path / "item_forecasts" / "2026-10.parquet").read_bytes() == before
    assert not (tmp_path / "forecast_outcomes").exists()


def test_apply_fills_ids_keeps_mirror_columns_and_adds_nothing_postgres_lacks(tmp_path):
    from scripts.ops_sync_from_postgres import sync

    db = _session()
    _seed(db)
    _mirror_only_rows(tmp_path)
    sync(db, [D], tables=TABLES, apply=True, root=tmp_path, today=TODAY)
    got = duckdb.sql(
        f"SELECT * FROM read_parquet('{tmp_path}/item_forecasts/2026-10.parquet') ORDER BY horizon_days"
    ).df()
    assert got["id"].tolist() == [11, 10]
    assert got["model_config"].tolist() == ["global-only", "global-only"]
    outcomes = duckdb.sql(f"SELECT id FROM read_parquet('{tmp_path}/forecast_outcomes/2026-10.parquet')").df()
    assert outcomes["id"].tolist() == [20]
