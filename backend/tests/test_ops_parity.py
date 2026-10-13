"""scripts/ops_parity.py: Postgres and the ops partitions agree per table and forecast_date."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tests.test_ops_readback import D, _seed, _session

TODAY = date(2026, 10, 12)


def _mirror_postgres(db, tmp_path):
    from db import ops_partitions, ops_readback

    ops_partitions.upsert("item_forecasts", ops_readback.forecasts(db, {D}), root=tmp_path, today=TODAY)
    ops_partitions.upsert("forecast_outcomes", ops_readback.outcomes(db, dates=[D]), root=tmp_path, today=TODAY)
    ops_partitions.upsert("forecast_candidates", ops_readback.candidates(db, {D}), root=tmp_path, today=TODAY)
    ops_partitions.upsert(
        "forecast_candidate_outcomes", ops_readback.candidate_outcomes(db, dates=[D]), root=tmp_path, today=TODAY
    )


def test_identical_stores_pass(tmp_path):
    from scripts.ops_parity import run_parity

    db = _session()
    _seed(db)
    _mirror_postgres(db, tmp_path)
    assert run_parity(db, root=tmp_path) == []


def test_pg_deleted_rows_do_not_count(tmp_path):
    from db.ops_partitions import upsert
    from scripts.ops_parity import run_parity

    db = _session()
    _seed(db)
    _mirror_postgres(db, tmp_path)
    upsert(
        "item_forecasts",
        pd.DataFrame([{"id": None, "item_id": 9, "forecast_date": D, "horizon_days": 7, "pg_deleted": True}]),
        root=tmp_path,
        today=TODAY,
    )
    assert run_parity(db, root=tmp_path) == []


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        ("DELETE FROM item_forecasts WHERE id = 11", "n"),
        ("UPDATE item_forecasts SET id = 99 WHERE id = 11", "id_sum"),
        ("UPDATE forecast_outcomes SET in_interval = 0", "n_in_interval"),
    ],
)
def test_a_missing_row_a_wrong_id_and_a_changed_verdict_fail(tmp_path, mutate, field):
    from scripts.ops_parity import run_parity
    from sqlalchemy import text

    db = _session()
    _seed(db)
    db.execute(text("UPDATE forecast_outcomes SET in_interval = 1"))
    db.commit()
    _mirror_postgres(db, tmp_path)
    db.execute(text(mutate))
    db.commit()
    got = run_parity(db, root=tmp_path)
    assert any(m.field == field and m.forecast_date == D for m in got), got


def test_streak_counts_consecutive_clean_nights(tmp_path, monkeypatch):
    from scripts.ops_parity import Mismatch, clean_streak, record

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    record([Mismatch("item_forecasts", D, "n", 1, 0)], date(2026, 10, 12))
    record([], date(2026, 10, 13))
    record([], date(2026, 10, 14))
    assert clean_streak() == 2


def test_closed_month_divergence_is_reported_apart_and_does_not_fail_the_night(tmp_path, monkeypatch):
    from scripts.ops_parity import Mismatch, clean_streak, failing, record

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    closed = Mismatch("forecast_outcomes", date(2026, 8, 5), "n_in_interval", 3, 2)
    open_ = Mismatch("forecast_outcomes", date(2026, 10, 1), "n", 1, 0)
    today = date(2026, 10, 14)
    assert failing([closed], today) == []
    assert failing([closed, open_], today) == [open_]
    record([closed], today)
    assert clean_streak() == 1


def test_repeated_ids_in_the_partitions_fail(tmp_path):
    from db.ops_partitions import upsert
    from scripts.ops_parity import failing, run_parity

    db = _session()
    _seed(db)
    _mirror_postgres(db, tmp_path)
    upsert(
        "item_forecasts",
        pd.DataFrame([{"id": 10, "item_id": 5, "forecast_date": D, "horizon_days": 7}]),
        root=tmp_path,
        today=TODAY,
    )
    got = [m for m in run_parity(db, root=tmp_path) if m.field == "duplicate_ids"]
    assert [(m.table, m.partitions) for m in got] == [("item_forecasts", 1)]
    assert failing(got, TODAY) == got
