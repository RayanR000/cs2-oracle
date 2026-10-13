"""Outcome writers dual-write forecast_outcomes partitions (Phase 1.2)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _db(day):
    from database import Base, ForecastOutcome, Item, ItemForecast

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(Item(id=1, item_id="ak_1", name="AK 1", type="rifle"))
    db.add(ItemForecast(id=10, item_id=1, forecast_date=day, horizon_days=3, price_mid=10.0))
    db.add(
        ForecastOutcome(
            id=20,
            forecast_id=10,
            item_id=1,
            forecast_date=day,
            horizon_days=3,
            target_date=datetime(2026, 10, 18),
            predicted_price_mid=10.0,
            actual_price=11.0,
            abs_error=1.0,
            in_interval=1,
        )
    )
    db.commit()
    return db


def _part(tmp_path, day):
    return duckdb.sql(f"SELECT * FROM read_parquet('{tmp_path}/forecast_outcomes/{day:%Y-%m}.parquet')").df()


def test_outcome_partitions_carry_postgres_ids(tmp_path, monkeypatch):
    from scripts.backtest_accuracy import _outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    assert _outcome_partitions(_db(day), [10]) is True
    got = _part(tmp_path, day)
    assert got["id"].tolist() == [20]
    assert got["item_slug"].tolist() == ["ak_1"]


def test_reresolve_replaces_dropped_forecasts(tmp_path, monkeypatch):
    from database import ForecastOutcome
    from scripts.backtest_accuracy import _outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    db = _db(day)
    _outcome_partitions(db, [10])
    db.query(ForecastOutcome).delete()
    db.commit()
    assert _outcome_partitions(db, [], drop=[10]) is True
    assert _part(tmp_path, day).empty


def test_a_closed_month_refresh_is_refused_and_logged(tmp_path, monkeypatch, capsys):
    from scripts.backtest_accuracy import _outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    old = date(2026, 7, 20)
    assert _outcome_partitions(_db(old), [10]) is False
    assert "::error title=Ops partition write failed::forecast_outcomes" in capsys.readouterr().out
    assert not (tmp_path / "forecast_outcomes" / "2026-07.parquet").exists()


def test_the_verdict_refresh_writes_its_mirror_rows_after_its_commit(monkeypatch):
    import scripts.backtest_accuracy as ba

    events = []

    class _DB:
        def execute(self, *a, **k):
            events.append("db")

        def commit(self):
            events.append("commit")

    monkeypatch.setattr("db.parquet.append_table", lambda *a, **k: events.append("mirror"))
    monkeypatch.setattr(ba, "_refreshed_outcome_partitions", lambda rows: events.append(("partitions", len(rows))))
    ba._flush_verdict_refresh(_DB(), [{"forecast_id": 10}], [{"forecast_id": 10}])
    assert events == ["mirror", "db", "commit", ("partitions", 1)]


def test_refreshed_rows_keep_their_stored_id_without_a_read_back(tmp_path, monkeypatch):
    from scripts.backtest_accuracy import _outcome_partitions, _refreshed_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    _outcome_partitions(_db(day), [10])
    row = {
        "forecast_id": 10,
        "item_id": 1,
        "forecast_date": day,
        "horizon_days": 3,
        "in_interval": 0,
        "item_slug": "ak_1",
    }
    assert _refreshed_outcome_partitions([row]) is True
    got = _part(tmp_path, day)
    assert got["id"].tolist() == [20]
    assert got["in_interval"].tolist() == [0]
