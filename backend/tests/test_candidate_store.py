from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker


def _session():
    from database import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _record(item_id=1, component="centre", candidate_name="last_price", fingerprint="a" * 64, **overrides):
    from models.candidate_predictions import CandidateRecord

    payload = dict(
        item_id=item_id,
        forecast_date=date(2026, 9, 18),
        horizon_days=7,
        component=component,
        candidate_name=candidate_name,
        candidate_version="v1",
        centre_price=40.0 if component == "centre" else None,
        predicted_price_low=36.0 if component == "centre" else None,
        predicted_price_high=50.0 if component == "centre" else None,
        score=0.7 if component == "ranking" else None,
        anchor_price=40.0,
        feature_cutoff_at=datetime(2026, 9, 18, 12, 0, 0),
        artifact_version="a123",
        config_fingerprint=fingerprint,
    )
    if component == "ranking" and candidate_name == "last_price":
        payload["candidate_name"] = "lambdarank_v1"
    payload.update(overrides)
    return CandidateRecord(**payload)


def test_new_writes_persist_with_counts():
    from database import ForecastCandidate
    from db.candidate_store import write_candidate_batches

    session = _session()
    records = [_record(item_id=1), _record(item_id=2), _record(item_id=1, component="ranking")]
    result = write_candidate_batches(session, records)
    assert result.centre_written == 2
    assert result.ranking_written == 1
    assert result.batches_expected == 2
    assert result.batches_written == 2
    assert session.query(ForecastCandidate).count() == 3


def test_identical_retry_is_idempotent():
    from database import ForecastCandidate
    from db.candidate_store import write_candidate_batches

    session = _session()
    records = [_record(item_id=1), _record(item_id=1, component="ranking")]
    first = write_candidate_batches(session, records)
    assert first.batches_written == 2
    second = write_candidate_batches(session, records)
    assert second.centre_written == 0
    assert second.ranking_written == 0
    assert second.batches_written == 2
    assert session.query(ForecastCandidate).count() == 2


def test_conflicting_fingerprint_raises():
    from db.candidate_store import CandidateFingerprintConflict, write_candidate_batches

    session = _session()
    write_candidate_batches(session, [_record(item_id=1)])
    with pytest.raises(CandidateFingerprintConflict):
        write_candidate_batches(session, [_record(item_id=1, fingerprint="b" * 64)])


def test_mixed_valid_invalid_group_leaves_zero_rows():
    from database import ForecastCandidate
    from db.candidate_store import write_candidate_batches

    session = _session()
    bad = _record(item_id=2, score=0.5)  # centre carrying a score violates the payload check
    with pytest.raises(IntegrityError):
        write_candidate_batches(session, [_record(item_id=1), bad])
    session.rollback()
    assert (
        session.query(ForecastCandidate)
        .filter(ForecastCandidate.forecast_date == date(2026, 9, 18))
        .count()
        == 0
    )


def test_centre_and_ranking_batches_counted_separately():
    from db.candidate_store import write_candidate_batches

    session = _session()
    result = write_candidate_batches(
        session, [_record(item_id=1), _record(item_id=2, component="ranking")]
    )
    assert result.batches_expected == 2
    assert result.batches_written == 2
    assert result.centre_written == 1
    assert result.ranking_written == 1
