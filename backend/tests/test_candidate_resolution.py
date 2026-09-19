from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

D = date(2026, 9, 1)
T = datetime(2026, 9, 8, 12, 0, 0)


def _db():
    from database import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _prod(session, item_id=5, base=40.0, actual=42.0, resolved_at=T):
    from database import ForecastOutcome

    row = ForecastOutcome(
        forecast_id=999,
        item_id=item_id,
        forecast_date=D,
        horizon_days=7,
        target_date=date(2026, 9, 8),
        current_price=40.0,
        predicted_price_mid=41.0,
        actual_price=actual,
        base_price=base,
        direction_predicted="flat",
        direction_actual="flat",
        direction_correct=1,
        in_interval=1,
        abs_error=1.0,
        pct_error=2.5,
        model_version="lgbm-v3",
        evaluated_at=T,
        resolved_at=resolved_at,
    )
    session.add(row)
    session.commit()
    return row


def _cand(session, item_id=5, component="centre", name="last_price", centre=41.0, low=36.0, high=50.0):
    from database import ForecastCandidate

    row = ForecastCandidate(
        item_id=item_id,
        forecast_date=D,
        horizon_days=7,
        component=component,
        candidate_name=name,
        candidate_version="v1",
        centre_price=centre if component == "centre" else None,
        predicted_price_low=low if component == "centre" else None,
        predicted_price_high=high if component == "centre" else None,
        score=0.7 if component == "ranking" else None,
        anchor_price=40.0,
        feature_cutoff_at=T,
        artifact_version="6",
        config_fingerprint="a" * 64,
    )
    session.add(row)
    session.commit()
    return row


def _mature(monkeypatch):
    import backtest.candidate_resolution as cr

    monkeypatch.setattr(cr, "archive_max_day", lambda archive_dir: date(2026, 10, 1))
    monkeypatch.setattr(
        cr, "load_voted_prices", lambda *a, **k: (_ for _ in ()).throw(AssertionError("archive must not be read"))
    )


def test_shared_legs_are_byte_identical(monkeypatch):
    import backtest.candidate_resolution as cr
    from database import ForecastCandidateOutcome

    session = _db()
    _prod(session)
    cand = _cand(session)
    _mature(monkeypatch)

    stats = cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir="unused")

    out = session.query(ForecastCandidateOutcome).one()
    assert out.candidate_id == cand.id
    assert out.base_price == 40.0
    assert out.actual_price == 42.0
    assert out.resolved_at == T
    assert out.absolute_error == 1.0
    assert out.percentage_error == 2.5
    assert out.in_interval is True
    assert stats["resolved"] == 1


def test_ranking_rows_store_legs_with_null_metrics(monkeypatch):
    import backtest.candidate_resolution as cr
    from database import ForecastCandidateOutcome

    session = _db()
    _prod(session)
    _cand(session, component="ranking", name="lambdarank_v1")
    _mature(monkeypatch)

    cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir="unused")

    out = session.query(ForecastCandidateOutcome).one()
    assert out.base_price == 40.0
    assert out.actual_price == 42.0
    assert out.absolute_error is None
    assert out.percentage_error is None
    assert out.in_interval is None


def test_unresolvable_without_production_or_archive(monkeypatch):
    import backtest.candidate_resolution as cr
    from database import ForecastCandidateOutcome

    session = _db()
    _cand(session)
    monkeypatch.setattr(cr, "archive_max_day", lambda archive_dir: date(2026, 10, 1))
    monkeypatch.setattr(cr, "archive_covered_days", lambda archive_dir: set())

    stats = cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir="unused")

    assert session.query(ForecastCandidateOutcome).count() == 0
    assert stats["unresolvable"] == 1
