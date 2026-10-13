"""Shadow candidates dual-write to forecast_candidates partitions (Phase 1.2)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import duckdb

from tests.test_candidate_store import _record, _session


def _part(tmp_path, day):
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{tmp_path}/forecast_candidates/{day:%Y-%m}.parquet') ORDER BY id"
    ).df()


def test_candidate_partitions_carry_postgres_ids(tmp_path, monkeypatch):
    from database import ForecastCandidate
    from db.candidate_store import write_candidate_batches

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    records = [replace(_record(item_id=1), forecast_date=day), replace(_record(item_id=2), forecast_date=day)]
    write_candidate_batches(db, records)
    ids = sorted(c.id for c in db.query(ForecastCandidate).all())
    assert _part(tmp_path, day)["id"].tolist() == ids


def test_a_retry_rewrites_nothing_new(tmp_path, monkeypatch):
    from db.candidate_store import write_candidate_batches

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    records = [replace(_record(item_id=1), forecast_date=day)]
    write_candidate_batches(db, records)
    write_candidate_batches(db, records)
    assert len(_part(tmp_path, day)) == 1


def test_read_only_candidates_land_with_partition_ids(tmp_path, monkeypatch):
    from db.candidate_store import write_candidate_partitions_read_only

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    write_candidate_partitions_read_only([replace(_record(item_id=1), forecast_date=day)])
    from db.ops_partitions import READ_ONLY_ID_BASE

    got = _part(tmp_path, day)
    assert got["id"].tolist() == [READ_ONLY_ID_BASE + 1]
    assert got["created_at"].notna().all()
