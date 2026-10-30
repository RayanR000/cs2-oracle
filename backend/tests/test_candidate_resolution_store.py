"""Candidate resolution reads and writes the ops partitions (Phase 2 PR 2).

docs/specs/2026-10-13-ops-readers-switch-design.md § PR 2 and its 2026-10-30 amendment: since
Postgres went read-only the shadow candidates land only in the partitions, so the plan reads them
there and the write lands in the partitions when Postgres cannot take it.
"""

from datetime import date

import pandas as pd
import pytest
from db import ops_partitions, ops_store

from tests._ops_mirror import mirror_session_to_ops
from tests.test_candidate_outcome_freeze import _stub_archive, _with_slug
from tests.test_candidate_resolution import D, T, _cand, _db, _prod

TODAY = date(2026, 10, 1)
HIGH = ops_partitions.READ_ONLY_ID_BASE


@pytest.fixture(autouse=True)
def _open_months(monkeypatch):
    """The fixtures are September 2026, which closes on 2026-11-12. The production writers do not
    pass allow_closed, so without this the partition assertions would fail from that date."""
    real = ops_partitions.upsert
    monkeypatch.setattr(
        ops_partitions, "upsert", lambda table, frame, **kw: real(table, frame, **{"allow_closed": True, **kw})
    )


def _plan(read, **kw):
    import backtest.candidate_resolution as cr

    return cr.plan_candidate_outcomes(read, today=TODAY, archive_dir="unused", **kw)


def _store():
    import backtest.candidate_resolution as cr

    return ops_store.reader(cr.CANDIDATE_TABLES)


def _norm(rows):
    frame = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    for c in ("_part", "resolved_at"):
        frame[c] = pd.to_datetime(frame[c])
    return frame


def _partition_candidate(cid, item_id=5):
    ops_partitions.upsert(
        "forecast_candidates",
        pd.DataFrame(
            [
                {
                    "id": cid,
                    "item_id": item_id,
                    "forecast_date": D,
                    "horizon_days": 7,
                    "component": "centre",
                    "candidate_name": "partition_only",
                    "candidate_version": "v1",
                    "centre_price": 41.0,
                    "predicted_price_low": 36.0,
                    "predicted_price_high": 50.0,
                    "anchor_price": 40.0,
                    "config_fingerprint": "b" * 64,
                }
            ]
        ),
        allow_closed=True,
    )


def test_plan_is_identical_from_both_stores(monkeypatch):
    _stub_archive(monkeypatch)
    session = _db()
    _with_slug(session)
    _prod(session)
    _cand(session)
    _cand(session, component="ranking", name="rank")
    _cand(session, item_id=6, name="archive_leg")  # no production row, no slug: unresolvable
    mirror_session_to_ops(session)

    store_rows, store_stats = _plan(_store())
    pg_rows, pg_stats = _plan(ops_store.pg_reader(session))

    assert len(store_rows) == 2
    pd.testing.assert_frame_equal(_norm(store_rows), _norm(pg_rows), check_dtype=False)
    assert store_stats == pg_stats


def test_plan_resolves_partition_only_candidates(monkeypatch):
    _stub_archive(monkeypatch)
    session = _db()
    _with_slug(session)
    _prod(session, base=40.0, actual=42.0)
    mirror_session_to_ops(session)
    _partition_candidate(HIGH + 3)

    rows, stats = _plan(_store())

    assert [(r["candidate_id"], r["base_price"], r["actual_price"]) for r in rows] == [(HIGH + 3, 40.0, 42.0)]
    assert pd.Timestamp(rows[0]["resolved_at"]) == pd.Timestamp(T)
    assert stats["resolved"] == 1


def test_plan_skips_candidates_frozen_in_the_partitions(monkeypatch):
    _stub_archive(monkeypatch)
    session = _db()
    _with_slug(session)
    _prod(session)
    cand = _cand(session)
    mirror_session_to_ops(session)
    ops_partitions.upsert(
        "forecast_candidate_outcomes",
        pd.DataFrame([{"candidate_id": cand.id, "base_price": 40.0, "actual_price": 42.0, "_part": D}]),
        allow_closed=True,
    )

    rows, stats = _plan(_store())

    assert rows == []
    assert stats["skipped_frozen"] == 1


def test_plan_ignores_pg_deleted_rows(monkeypatch):
    _stub_archive(monkeypatch, base=40.0, actual=45.0)
    session = _db()
    _with_slug(session)
    _prod(session, base=40.0, actual=42.0)
    _cand(session)
    mirror_session_to_ops(session)
    (prod,) = ops_store.reader(("forecast_outcomes",))("SELECT * FROM forecast_outcomes")
    ops_partitions.upsert(
        "forecast_outcomes", pd.DataFrame([{**prod._asdict(), "pg_deleted": True}]), allow_closed=True
    )

    rows, _ = _plan(_store())

    # The deleted production legs (42.0) are not used; the archive's are.
    assert [r["actual_price"] for r in rows] == [45.0]
    assert rows[0]["resolved_at"] is None


def test_plan_mismatch_raises(monkeypatch):
    _stub_archive(monkeypatch, base=40.0, actual=45.0)
    session = _db()
    _with_slug(session)
    _prod(session, base=40.0, actual=42.0)
    _cand(session)
    mirror_session_to_ops(session)

    with pytest.raises(RuntimeError, match="mismatch"):
        _plan(_store(), reresolve=True)


# --- Task 3: the write step ---------------------------------------------------


def _resolve(session, **kw):
    import backtest.candidate_resolution as cr

    return cr.resolve_candidate_outcomes(session, today=TODAY, archive_dir="unused", **kw)


def _partition_outcomes():
    return {int(r.candidate_id): r for r in _store()("SELECT * FROM forecast_candidate_outcomes")}


def _pg_outcome_ids(session):
    from database import ForecastCandidateOutcome

    return {o.candidate_id for o in session.query(ForecastCandidateOutcome).all()}


def _mixed_night(monkeypatch):
    """One ordinary candidate and one partition-only candidate, both mature, sharing production legs."""
    _stub_archive(monkeypatch)
    session = _db()
    _with_slug(session)
    _prod(session)
    cand = _cand(session)
    mirror_session_to_ops(session)
    _partition_candidate(HIGH + 3)
    return session, cand


def _unwritable(session, monkeypatch, pgcode):
    from sqlalchemy.exc import InternalError

    from tests.test_read_only_postgres import _PgError

    def commit():
        raise InternalError("INSERT ...", {}, _PgError("rejected", pgcode))

    monkeypatch.setattr(session, "commit", commit)


def test_read_only_lands_every_row_in_the_partitions(monkeypatch, capsys):
    session, cand = _mixed_night(monkeypatch)
    _unwritable(session, monkeypatch, "25006")

    stats = _resolve(session)

    assert set(_partition_outcomes()) == {cand.id, HIGH + 3}
    assert _pg_outcome_ids(session) == set()
    assert stats["resolved"] == 2
    assert "::warning title=Postgres read-only::" in capsys.readouterr().out


def test_disk_full_is_treated_as_unwritable(monkeypatch):
    session, cand = _mixed_night(monkeypatch)
    _unwritable(session, monkeypatch, "53100")

    _resolve(session)

    assert set(_partition_outcomes()) == {cand.id, HIGH + 3}


def test_partition_only_candidates_skip_postgres(monkeypatch):
    session, cand = _mixed_night(monkeypatch)

    _resolve(session)

    assert _pg_outcome_ids(session) == {cand.id}
    outcomes = _partition_outcomes()
    assert set(outcomes) == {cand.id, HIGH + 3}
    assert outcomes[HIGH + 3].actual_price == 42.0


def test_rerun_is_idempotent(monkeypatch):
    session, _ = _mixed_night(monkeypatch)

    _resolve(session)
    second = _resolve(session)

    assert len(_store()("SELECT * FROM forecast_candidate_outcomes")) == 2
    assert second["skipped_frozen"] == 2


def test_read_twice_logs_outcomes_per_date(monkeypatch, caplog):
    import logging

    session, _ = _mixed_night(monkeypatch)

    with caplog.at_level(logging.INFO, logger=ops_store.logger.name):
        _resolve(session)

    assert "read twice candidate outcomes per forecast date" in caplog.text


def test_rescore_reads_the_partitions(monkeypatch):
    _stub_archive(monkeypatch)
    session = _db()
    _with_slug(session)
    cand = _cand(session, centre=41.0)
    mirror_session_to_ops(session)
    ops_partitions.upsert(
        "forecast_candidate_outcomes",
        pd.DataFrame(
            [
                {
                    "candidate_id": cand.id,
                    "base_price": 40.0,
                    "actual_price": 42.0,
                    "absolute_error": 99.0,  # stale
                    "resolution_version": "v1",
                    "_part": D,
                }
            ]
        ),
        allow_closed=True,
    )

    assert _resolve(session, rescore=True) == {"rescored": 1}
    assert _partition_outcomes()[cand.id].absolute_error == pytest.approx(1.0)
