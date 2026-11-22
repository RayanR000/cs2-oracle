"""Candidate outcomes land in the partitions, routed by their candidate's forecast_date."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb


def _row(cand_day, resolved_at=None):
    return {
        "candidate_id": 30,
        "base_price": 10.0,
        "actual_price": 11.0,
        "resolved_at": resolved_at,
        "resolution_version": "r1",
        "_part": cand_day,
    }


def _read(tmp_path, day):
    path = tmp_path / "forecast_candidate_outcomes" / f"{day:%Y-%m}.parquet"
    return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()


def test_routed_by_the_candidate_s_date_not_the_resolution_date(tmp_path, monkeypatch):
    from backtest.candidate_resolution import _write_candidate_outcomes

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    cand_day = date.today().replace(day=1) - timedelta(days=1)  # last day of the previous month
    assert _write_candidate_outcomes([_row(cand_day)]) is True
    got = _read(tmp_path, cand_day)
    assert got["candidate_id"].tolist() == [30]
    assert "_part" not in got.columns
    assert got["resolved_at"].notna().all(), "a planned row with no resolved_at is stamped"


def test_a_planned_resolved_at_is_kept(tmp_path, monkeypatch):
    from backtest.candidate_resolution import _write_candidate_outcomes

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=3)
    when = datetime(2026, 1, 2, 3, 4, 5)
    _write_candidate_outcomes([_row(day, resolved_at=when)])
    assert _read(tmp_path, day)["resolved_at"].tolist() == [when]


def test_nothing_planned_writes_nothing(tmp_path, monkeypatch):
    from backtest.candidate_resolution import _write_candidate_outcomes

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    assert _write_candidate_outcomes([]) is True
    assert not (tmp_path / "forecast_candidate_outcomes").exists()
