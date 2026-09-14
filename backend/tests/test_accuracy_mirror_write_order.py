"""The accuracy mirror is written BEFORE the DB commit, not after.

``_upsert_accuracy`` committed the DB and *then* appended to Parquet. Every
2026-08-05 backtest failure therefore landed the DB rows and skipped the mirror,
then exited 1 — and the API reads Parquet first, falling back to the DB only on
``None`` or an exception, so a present-but-stale mirror WINS over a current DB.
Each failed run widened that divergence silently.

This is the same discipline ``_flush_verdict_refresh`` and the ``--reresolve``
path already document and rely on: Parquet first, DB second. A crash between
them leaves the mirror briefly ahead of the DB, which the next run reconciles,
rather than leaving the served copy permanently behind.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import pytest
from database import Base, PredictionAccuracy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class _Recorder:
    """Records the order of the two writes."""

    def __init__(self):
        self.calls = []

    def commit(self):
        self.calls.append("commit")

    def query(self, *a, **k):
        return self

    def filter_by(self, **k):
        return self

    def first(self):
        return None

    def add(self, obj):
        self.calls.append("add")


ROW = {
    "prediction_type": "forecast",
    "evaluation_date": "2026-08-05",
    "horizon_days": 7,
    "model_version": "lgbm-v3",
    "price_tier": 1,
    "sample_count": 10,
    "metrics": {"mae": 1.0},
    "created_at": "2026-08-05",
}


def _patch(monkeypatch, append):
    import db.parquet
    import scripts.backtest_accuracy as ba

    monkeypatch.setattr(db.parquet, "append_table", append)
    # PredictionAccuracy is constructed with **row; keep it inert.
    monkeypatch.setattr(ba, "PredictionAccuracy", lambda **kw: object())
    return ba


def test_parquet_is_appended_before_the_db_commit(monkeypatch):
    db = _Recorder()

    def append(table, rows, keys):
        db.calls.append("append")

    ba = _patch(monkeypatch, append)
    ba._upsert_accuracy(db, [dict(ROW)])

    assert "append" in db.calls, "the mirror was never written"
    assert db.calls.index("append") < db.calls.index("commit"), (
        f"DB committed before the mirror was written: {db.calls}"
    )


def test_a_failing_mirror_write_does_not_leave_a_committed_db(monkeypatch):
    """The regression: the run exits 1 with the DB ahead of the served copy."""
    db = _Recorder()

    def append(table, rows, keys):
        db.calls.append("append")
        raise RuntimeError("Conversion Error: Could not convert string 'None'")

    ba = _patch(monkeypatch, append)
    with pytest.raises(RuntimeError):
        ba._upsert_accuracy(db, [dict(ROW)])

    assert "commit" not in db.calls, (
        "the DB was committed even though the mirror write failed, so the DB "
        f"and the served Parquet copy now disagree: {db.calls}"
    )


# ---------------------------------------------------------------------------
# The same path against a real session and a real Parquet file. The tests above
# mock both stores to observe ordering; this one checks the reorder did not break
# the ORM upsert it sits in front of, and that both stores actually land.
# ---------------------------------------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)()
    engine.dispose()


def _row(tier, mae):
    return {
        "prediction_type": "forecast",
        "evaluation_date": date(2026, 8, 5),
        "horizon_days": 7,
        "model_version": "lgbm-v3",
        "price_tier": tier,
        "evaluation_window_days": None,
        "sample_count": 100,
        "metrics": {"mae": mae, "skill_vs_baseline": None},
        "created_at": datetime(2026, 8, 5, 12, 0, 0),
    }


def test_both_stores_land_and_agree(session, tmp_path, monkeypatch):
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    import scripts.backtest_accuracy as ba

    # The all-tiers aggregate (NULL tier) and the >=$1 headline together, since
    # the NULL key is what the dedup used to mishandle.
    ba._upsert_accuracy(session, [_row(None, 1.0), _row(-1, 2.0)])

    stored = {r.price_tier: r.metrics for r in session.query(PredictionAccuracy).all()}
    assert stored.keys() == {None, -1}
    assert stored[None]["mae"] == 1.0

    mirror = pd.read_parquet(tmp_path / "prediction_accuracy.parquet")
    assert len(mirror) == 2
    by_tier = {
        (None if pd.isna(r.price_tier) else int(r.price_tier)): json.loads(r.metrics) for r in mirror.itertuples()
    }
    assert by_tier[None]["mae"] == 1.0
    assert by_tier[-1]["mae"] == 2.0
    assert by_tier[None]["skill_vs_baseline"] is None


def test_a_rerun_replaces_in_both_stores(session, tmp_path, monkeypatch):
    """The daily run rescores the same evaluation_date; neither store may grow."""
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    import scripts.backtest_accuracy as ba

    ba._upsert_accuracy(session, [_row(None, 1.0), _row(-1, 2.0)])
    ba._upsert_accuracy(session, [_row(None, 9.0), _row(-1, 8.0)])

    assert session.query(PredictionAccuracy).count() == 2
    rows = {r.price_tier: r.metrics["mae"] for r in session.query(PredictionAccuracy)}
    assert rows == {None: 9.0, -1: 8.0}

    mirror = pd.read_parquet(tmp_path / "prediction_accuracy.parquet")
    assert len(mirror) == 2, f"the mirror duplicated instead of replacing:\n{mirror}"
    maes = sorted(json.loads(m)["mae"] for m in mirror["metrics"])
    assert maes == [8.0, 9.0], "the mirror kept the stale metrics"


def test_the_caller_s_row_dicts_are_not_mutated(session, tmp_path, monkeypatch):
    """The mirror serialises metrics to JSON; the DB's JSON column wants the dict.

    Both stores are handed the SAME row dicts, so a serialisation that mutated
    them in place would write a JSON string into the DB column too.
    """
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    import scripts.backtest_accuracy as ba

    rows = [_row(None, 1.0)]
    ba._upsert_accuracy(session, rows)

    assert isinstance(rows[0]["metrics"], dict), "the caller's dict was stringified"
    assert isinstance(session.query(PredictionAccuracy).one().metrics, dict)
