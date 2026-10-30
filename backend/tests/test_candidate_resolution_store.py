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
