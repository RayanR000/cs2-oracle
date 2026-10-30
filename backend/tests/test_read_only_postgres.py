"""A read-only Postgres skips the DB write loudly and the Parquet output still lands.

Prod Supabase is over its free-tier cap (986 MB vs 500 MB on 2026-10-11) and can be
put in read-only mode, where writes fail with SQLSTATE 25006. Before this, the
forecast run wrote Postgres first and raised, so the Parquet mirror never saw the
day's forecasts; the Aggregator's run-record commit and migration step would have
failed the chain before Price Forecast ran at all. See db/read_only.py.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from db.read_only import READ_ONLY_SQLSTATE, is_read_only_error
from sqlalchemy.exc import DBAPIError, InternalError


class _PgError(Exception):
    def __init__(self, msg, pgcode=None):
        super().__init__(msg)
        self.pgcode = pgcode


def _read_only_exc():
    orig = _PgError("cannot execute INSERT in a read-only transaction", READ_ONLY_SQLSTATE)
    return InternalError("INSERT ...", {}, orig)


# --- detection -------------------------------------------------------------


def test_detects_sqlstate_on_the_wrapped_driver_error():
    assert is_read_only_error(_read_only_exc())


def test_detects_sqlstate_through_the_cause_chain():
    try:
        try:
            raise _read_only_exc()
        except InternalError as inner:
            raise RuntimeError("batch failed") from inner
    except RuntimeError as outer:
        assert is_read_only_error(outer)


def test_detects_the_message_without_a_code():
    assert is_read_only_error(Exception("ERROR: cannot execute UPDATE in a read-only transaction"))


@pytest.mark.parametrize(
    "exc",
    [
        Exception("connection reset"),
        DBAPIError("INSERT ...", {}, _PgError("duplicate key", "23505")),
        None,
    ],
)
def test_other_errors_are_not_read_only(exc):
    assert not is_read_only_error(exc)


# --- forecast write --------------------------------------------------------


def _results():
    return pd.DataFrame(
        [
            {
                "item_id": "ak_1",
                "current_price": 10.0,
                "anchor_clean": True,
                "anchor_wedge_pct": 0.0,
                "anchor_date": date(2026, 10, 10),
                "forecasts": {3: {"low": 9.0, "mid": 10.0, "high": 11.0, "direction": "flat", "confidence": 0.5}},
            }
        ]
    )


class _DB:
    def __init__(self, error, events):
        self.error = error
        self.events = events
        self.rolled_back = False

    def execute(self, stmt, *a, **k):
        if stmt == "stmt":
            self.events.append("db")
            raise self.error
        return None

    def commit(self):
        return None

    def rollback(self):
        self.rolled_back = True

    def close(self):
        return None

    def get_bind(self):
        return type("_Bind", (), {"dialect": type("_D", (), {"name": "postgresql"})()})()


def _drive(db, events):
    from scripts.forecast_prices import _write_forecasts_to_db

    class _Inspector:
        def get_columns(self, name):
            return [
                {"name": c} for c in ("anchor_clean", "anchor_wedge_pct", "exceed_p", "anomaly_p", "band_multiplier")
            ]

    with (
        patch("db.parquet.append_table", side_effect=lambda *a, **k: events.append("parquet")),
        patch("sqlalchemy.inspect", return_value=_Inspector()),
        patch("sqlalchemy.dialects.postgresql.insert") as ins,
        patch("database.SessionLocal", return_value=db),
    ):
        ins.return_value.values.return_value.on_conflict_do_update.return_value = "stmt"
        return _write_forecasts_to_db(db, _results(), "lgbm-v3", {"ak_1": 1}, date(2026, 10, 11))


def test_read_only_forecast_write_lands_parquet_and_does_not_raise(capsys):
    events = []
    db = _DB(_read_only_exc(), events)
    n = _drive(db, events)
    assert n == 1
    assert events == ["parquet", "db"]  # mirror first, one DB attempt, no retries
    assert db.rolled_back
    assert "::warning title=Postgres read-only::" in capsys.readouterr().out


def test_other_forecast_write_errors_still_raise_after_the_mirror():
    events = []
    db = _DB(RuntimeError("connection reset"), events)
    with pytest.raises(RuntimeError):
        _drive(db, events)
    assert events[0] == "parquet"
    assert events.count("db") == 3  # the existing retry loop is unchanged


# --- shadow candidates -----------------------------------------------------


def _shadow(error):
    from scripts.forecast_prices import _write_shadow_candidates

    record = MagicMock(item_id="ak_1", forecast_date=date(2026, 10, 10))
    forecaster = MagicMock(pending_candidates=[record])
    db = MagicMock()
    with (
        patch("db.candidate_store.write_candidate_batches", side_effect=error),
        patch("dataclasses.replace", side_effect=lambda r, **k: r),
    ):
        return _write_shadow_candidates(db, forecaster, {"ak_1": 1}), db


def test_read_only_shadow_write_is_skipped():
    counts, db = _shadow(_read_only_exc())
    assert counts == (0, 0, 0, 0)
    db.rollback.assert_called_once()


def test_other_shadow_write_errors_still_raise():
    with pytest.raises(DBAPIError):
        _shadow(DBAPIError("INSERT ...", {}, _PgError("conflict", "23505")))


# --- drift alert -----------------------------------------------------------


def test_read_only_drift_alert_commit_is_skipped():
    from models.forecaster import ItemForecaster

    f = ItemForecaster.__new__(ItemForecaster)
    f.db = MagicMock()
    f.db.commit.side_effect = _read_only_exc()
    f._commit_alert("the accuracy_alerts insert")
    f.db.rollback.assert_called_once()


def test_other_drift_alert_errors_still_raise():
    from models.forecaster import ItemForecaster

    f = ItemForecaster.__new__(ItemForecaster)
    f.db = MagicMock()
    f.db.commit.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        f._commit_alert("the accuracy_alerts insert")


# --- migration step --------------------------------------------------------


def test_migrate_skips_on_a_read_only_database():
    from scripts import run_task

    with (
        patch.object(run_task, "_postgres_read_only", return_value=True),
        patch.object(run_task.subprocess, "run") as run,
    ):
        result = run_task.run_migrations("head")
    run.assert_not_called()
    assert result == {"status": "success", "revision": "head", "skipped_read_only": True}


def test_migrate_runs_when_writable():
    from scripts import run_task

    with (
        patch.object(run_task, "_postgres_read_only", return_value=False),
        patch.object(run_task.subprocess, "run", return_value=MagicMock(returncode=0)) as run,
    ):
        assert run_task.run_migrations("head")["status"] == "success"
    run.assert_called_once()


# --- out of disk counts as unwritable (53100, Supabase DiskFull) ------------


def test_disk_full_pgcode_matches():
    from db.read_only import DISK_FULL_SQLSTATE, is_disk_full_error

    assert is_disk_full_error(InternalError("INSERT ...", {}, _PgError("could not extend file", DISK_FULL_SQLSTATE)))


def test_disk_full_message_matches_through_the_cause_chain():
    from db.read_only import is_disk_full_error
    from sqlalchemy.exc import OperationalError

    try:
        try:
            raise OperationalError("SELECT ...", {}, _PgError('could not write to file "x": No space left on device'))
        except OperationalError as inner:
            raise RuntimeError("batch failed") from inner
    except RuntimeError as outer:
        assert is_disk_full_error(outer)


def test_unrelated_error_is_writable():
    from db.read_only import is_unwritable_error

    assert not is_unwritable_error(ValueError("bad row"))


def test_read_only_is_unwritable():
    from db.read_only import is_unwritable_error

    assert is_unwritable_error(_read_only_exc())
