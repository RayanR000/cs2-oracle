from datetime import date, datetime, timedelta

import duckdb
import pytest
from db.ops_partitions import READ_ONLY_ID_BASE

DAY = date.today() - timedelta(days=3)  # an open month: the partitions refuse closed ones


def _stored(tmp_path):
    files = sorted(tmp_path.glob("forecast_candidates/*.parquet"))
    if not files:
        return duckdb.sql("SELECT 1 AS id WHERE false").df()
    lst = ", ".join(f"'{f}'" for f in files)
    return duckdb.sql(f"SELECT * FROM read_parquet([{lst}]) ORDER BY id").df()


def _record(item_id=1, component="centre", candidate_name="last_price", fingerprint="a" * 64, **overrides):
    from models.candidate_predictions import CandidateRecord

    payload = dict(
        item_id=item_id,
        forecast_date=DAY,
        horizon_days=7,
        component=component,
        candidate_name=candidate_name,
        candidate_version="v1",
        centre_price=40.0 if component == "centre" else None,
        predicted_price_low=36.0 if component == "centre" else None,
        predicted_price_high=50.0 if component == "centre" else None,
        score=0.7 if component == "ranking" else None,
        anchor_price=40.0,
        feature_cutoff_at=datetime.combine(DAY, datetime.min.time()),
        artifact_version="a123",
        config_fingerprint=fingerprint,
    )
    if component == "ranking" and candidate_name == "last_price":
        payload["candidate_name"] = "lambdarank_v1"
    payload.update(overrides)
    return CandidateRecord(**payload)


@pytest.fixture(autouse=True)
def _ops(tmp_path, monkeypatch):
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)


def test_new_writes_persist_with_counts(tmp_path):
    from db.candidate_store import write_candidate_batches

    records = [_record(item_id=1), _record(item_id=2), _record(item_id=1, component="ranking")]
    result = write_candidate_batches(records)
    assert result.centre_written == 2
    assert result.ranking_written == 1
    assert result.batches_expected == 2
    assert result.batches_written == 2
    stored = _stored(tmp_path)
    assert len(stored) == 3
    assert stored["id"].is_unique and stored["id"].min() > READ_ONLY_ID_BASE
    assert stored["created_at"].notna().all()


def test_identical_retry_is_idempotent(tmp_path):
    from db.candidate_store import write_candidate_batches

    records = [_record(item_id=1), _record(item_id=1, component="ranking")]
    first = write_candidate_batches(records)
    assert first.batches_written == 2
    ids = _stored(tmp_path)["id"].tolist()
    second = write_candidate_batches(records)
    assert second.centre_written == 0
    assert second.ranking_written == 0
    assert second.batches_written == 2
    assert _stored(tmp_path)["id"].tolist() == ids


def test_conflicting_fingerprint_raises(tmp_path):
    from db.candidate_store import CandidateFingerprintConflict, write_candidate_batches

    write_candidate_batches([_record(item_id=1)])
    with pytest.raises(CandidateFingerprintConflict):
        write_candidate_batches([_record(item_id=1, fingerprint="b" * 64)])
    assert len(_stored(tmp_path)) == 1


def test_centre_and_ranking_batches_counted_separately():
    from db.candidate_store import write_candidate_batches

    result = write_candidate_batches([_record(item_id=1), _record(item_id=2, component="ranking")])
    assert result.batches_expected == 2
    assert result.batches_written == 2
    assert result.centre_written == 1
    assert result.ranking_written == 1


def test_a_partition_failure_raises(monkeypatch):
    from db import ops_partitions
    from db.candidate_store import write_candidate_batches

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ops_partitions, "upsert", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_candidate_batches([_record(item_id=1)])


def test_a_closed_month_is_refused_loudly():
    from datetime import date as _date

    from db.candidate_store import write_candidate_batches
    from db.ops_partitions import PartitionError

    with pytest.raises(PartitionError, match="closed"):
        write_candidate_batches([_record(item_id=1, forecast_date=_date(2026, 8, 11))])
