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


def _centre(**overrides):
    from database import ForecastCandidate

    payload = dict(
        item_id=1,
        forecast_date=date(2026, 9, 18),
        horizon_days=7,
        component="centre",
        candidate_name="last_price",
        candidate_version="v1",
        centre_price=40.0,
        predicted_price_low=36.0,
        predicted_price_high=50.0,
        score=None,
        anchor_price=40.0,
        feature_cutoff_at=datetime(2026, 9, 18),
        config_fingerprint="a" * 64,
    )
    payload.update(overrides)
    return ForecastCandidate(**payload)


def _ranking(**overrides):
    from database import ForecastCandidate

    payload = dict(
        item_id=1,
        forecast_date=date(2026, 9, 18),
        horizon_days=7,
        component="ranking",
        candidate_name="lambdarank_v1",
        candidate_version="v1",
        centre_price=None,
        predicted_price_low=None,
        predicted_price_high=None,
        score=0.7,
        anchor_price=40.0,
        feature_cutoff_at=datetime(2026, 9, 18),
        config_fingerprint="a" * 64,
    )
    payload.update(overrides)
    return ForecastCandidate(**payload)


def test_valid_centre_row_persists():
    session = _session()
    session.add(_centre())
    session.commit()
    assert session.query(type(_centre())).count() == 1


def test_valid_ranking_row_persists():
    session = _session()
    session.add(_ranking())
    session.commit()


def test_invalid_horizon_rejected():
    session = _session()
    session.add(_centre(horizon_days=5))
    with pytest.raises(IntegrityError):
        session.commit()


def test_missing_component_payload_rejected():
    session = _session()
    row = _centre(
        centre_price=None,
        predicted_price_low=None,
        predicted_price_high=None,
        score=None,
    )
    session.add(row)
    with pytest.raises(IntegrityError):
        session.commit()


def test_invalid_centre_ordering_rejected():
    session = _session()
    session.add(_centre(centre_price=100.0, predicted_price_low=120.0, predicted_price_high=130.0))
    with pytest.raises(IntegrityError):
        session.commit()


def test_duplicate_identity_rejected():
    from database import ForecastCandidate

    session = _session()
    session.add(_centre())
    session.commit()
    session.add(_centre())
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    assert session.query(ForecastCandidate).count() == 1


def test_one_to_one_outcome():
    from database import ForecastCandidateOutcome

    session = _session()
    row = _centre()
    session.add(row)
    session.commit()
    session.add(
        ForecastCandidateOutcome(
            candidate_id=row.id,
            base_price=40.0,
            actual_price=42.0,
            resolved_at=datetime(2026, 9, 25),
            absolute_error=2.0,
            percentage_error=5.0,
            in_interval=True,
            resolution_version="v1",
        )
    )
    session.commit()
