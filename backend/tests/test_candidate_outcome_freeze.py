from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.test_candidate_resolution import D, T, _cand, _db, _prod


class _Legs:
    def __init__(self, price, oldest):
        self.price = price
        self.oldest_observation = oldest


def _stub_archive(monkeypatch, base=40.0, actual=45.0):
    import backtest.candidate_resolution as cr

    monkeypatch.setattr(cr, "archive_max_day", lambda archive_dir: date(2026, 10, 1))
    monkeypatch.setattr(cr, "archive_covered_days", lambda archive_dir: {D, date(2026, 9, 8)})
    monkeypatch.setattr(cr, "load_voted_prices", lambda *a, **k: None)
    monkeypatch.setattr(cr, "classify_chronic", lambda *a, **k: None)
    monkeypatch.setattr(cr, "classify_archive_gap", lambda *a, **k: None)
    monkeypatch.setattr(cr, "classify_base_gap", lambda *a, **k: None)

    def _anchors(voted, anchors):
        return {(s, d): _Legs(base if d == D else actual, D + timedelta_days(1)) for s, d in anchors}

    monkeypatch.setattr(cr, "resolve_anchors", _anchors)
    return cr


def timedelta_days(n):
    from datetime import timedelta

    return timedelta(days=n)


def _with_slug(session):
    from database import Item

    session.add(Item(id=5, item_id="slug-5", name="n", type="skin"))
    session.commit()


def test_frozen_legs_do_not_move(monkeypatch):
    import backtest.candidate_resolution as cr
    from database import ForecastCandidateOutcome

    session = _db()
    _prod(session)
    _cand(session)
    _with_slug(session)
    _stub_archive(monkeypatch)

    cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir="unused")
    from database import ForecastOutcome

    session.query(ForecastOutcome).delete()
    _prod(session, actual=99.0)
    cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir="unused")

    out = session.query(ForecastCandidateOutcome).one()
    assert out.actual_price == 42.0


def test_reresolve_moves_legs(monkeypatch):
    import backtest.candidate_resolution as cr
    from database import ForecastCandidateOutcome

    session = _db()
    _prod(session)
    _cand(session)
    _with_slug(session)
    _stub_archive(monkeypatch)

    cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir="unused")
    from database import ForecastOutcome

    session.query(ForecastOutcome).delete()
    _prod(session, actual=45.0)
    stats = cr.resolve_candidate_outcomes(
        session, today=date(2026, 10, 1), archive_dir="unused", reresolve=True
    )

    out = session.query(ForecastCandidateOutcome).one()
    assert out.actual_price == 45.0
    assert stats["reresolved"] == 1


def test_reresolve_rejects_production_mismatch(monkeypatch):
    import pytest

    import backtest.candidate_resolution as cr

    session = _db()
    _prod(session, actual=45.0)
    _cand(session)
    _with_slug(session)
    _stub_archive(monkeypatch, actual=46.0)

    with pytest.raises(RuntimeError, match="mismatch"):
        cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir="unused", reresolve=True)


def test_rescore_refreshes_derived_without_the_archive(monkeypatch):
    import backtest.candidate_resolution as cr
    from database import ForecastCandidateOutcome

    session = _db()
    _prod(session)
    _cand(session)
    monkeypatch.setattr(cr, "archive_max_day", lambda archive_dir: date(2026, 10, 1))
    cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir="unused")

    out = session.query(ForecastCandidateOutcome).one()
    out.absolute_error = -999.0
    session.commit()

    def _no_archive(*a, **k):
        raise AssertionError("rescore must not read the archive")

    monkeypatch.setattr(cr, "archive_max_day", _no_archive)
    monkeypatch.setattr(cr, "archive_covered_days", _no_archive)
    monkeypatch.setattr(cr, "load_voted_prices", _no_archive)
    stats = cr.resolve_candidate_outcomes(session, today=date(2026, 10, 1), archive_dir=None, rescore=True)

    out = session.query(ForecastCandidateOutcome).one()
    assert out.absolute_error == 1.0
    assert out.actual_price == 42.0
    assert stats["rescored"] == 1
