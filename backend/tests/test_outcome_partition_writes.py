"""The backtest's forecast_outcomes writes land in the monthly partitions (Phase 4: only there)."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
from db import ops_partitions


def _row(day, forecast_id=10, **kw):
    return {
        "forecast_id": forecast_id,
        "item_id": 1,
        "forecast_date": day,
        "horizon_days": 3,
        "target_date": day + timedelta(days=3),
        "predicted_price_mid": 10.0,
        "actual_price": 11.0,
        "abs_error": 1.0,
        "in_interval": 1,
        "item_slug": "ak_1",
        **kw,
    }


def _part(tmp_path, day):
    return duckdb.sql(f"SELECT * FROM read_parquet('{tmp_path}/forecast_outcomes/{day:%Y-%m}.parquet')").df()


def test_new_outcomes_get_ids_from_the_read_only_base_and_keep_their_slug(tmp_path, monkeypatch):
    from scripts.backtest_accuracy import _write_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    assert _write_outcome_partitions([_row(day)], "forecast_outcomes") is True
    got = _part(tmp_path, day)
    assert got["id"].tolist() == [ops_partitions.READ_ONLY_ID_BASE + 1]
    assert got["item_slug"].tolist() == ["ak_1"]


def test_reresolve_removes_a_forecast_that_no_longer_resolves(tmp_path, monkeypatch):
    from scripts.backtest_accuracy import _store_forecast_outcomes, _write_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    _write_outcome_partitions([_row(day)], "forecast_outcomes")
    assert _store_forecast_outcomes([], reresolve=True, considered_ids={10}, id_to_slug={}) == 0
    assert _part(tmp_path, day).empty


def test_reresolve_keeps_the_stored_id_of_a_rewritten_row(tmp_path, monkeypatch):
    from scripts.backtest_accuracy import _store_forecast_outcomes, _write_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    _write_outcome_partitions([_row(day)], "forecast_outcomes")
    (stored_id,) = _part(tmp_path, day)["id"].tolist()
    revised = {k: v for k, v in _row(day, actual_price=12.0).items() if k != "item_slug"}
    assert _store_forecast_outcomes([revised], reresolve=True, considered_ids={10}, id_to_slug={1: "ak_1"}) == 1
    got = _part(tmp_path, day)
    assert got["id"].tolist() == [stored_id]
    assert got["actual_price"].tolist() == [12.0]


def test_a_frozen_outcome_is_not_rewritten_on_the_daily_path(tmp_path, monkeypatch):
    from scripts.backtest_accuracy import _store_forecast_outcomes, _write_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    _write_outcome_partitions([_row(day)], "forecast_outcomes")
    revised = {k: v for k, v in _row(day, actual_price=99.0).items() if k != "item_slug"}
    assert _store_forecast_outcomes([revised], id_to_slug={1: "ak_1"}) == 0
    assert _part(tmp_path, day)["actual_price"].tolist() == [11.0]


def test_a_closed_month_write_is_refused_and_logged(tmp_path, monkeypatch, capsys):
    from scripts.backtest_accuracy import _write_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    old = date(2026, 7, 20)
    assert _write_outcome_partitions([_row(old)], "forecast_outcomes") is False
    assert "::error title=Ops partition write failed::forecast_outcomes" in capsys.readouterr().out
    assert not (tmp_path / "forecast_outcomes" / "2026-07.parquet").exists()


def test_the_verdict_refresh_writes_the_mirror_then_the_partitions(monkeypatch):
    import scripts.backtest_accuracy as ba

    events = []
    monkeypatch.setattr("db.parquet.append_table", lambda *a, **k: events.append("mirror"))
    monkeypatch.setattr(ba, "_refreshed_outcome_partitions", lambda rows: events.append(("partitions", len(rows))))
    ba._flush_verdict_refresh([{"forecast_id": 10}])
    assert events == ["mirror", ("partitions", 1)]


def test_refreshed_rows_keep_their_stored_id(tmp_path, monkeypatch):
    from scripts.backtest_accuracy import _refreshed_outcome_partitions, _write_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    _write_outcome_partitions([_row(day)], "forecast_outcomes")
    (stored_id,) = _part(tmp_path, day)["id"].tolist()
    assert _refreshed_outcome_partitions([_row(day, in_interval=0)]) is True
    got = _part(tmp_path, day)
    assert got["id"].tolist() == [stored_id]
    assert got["in_interval"].tolist() == [0]
