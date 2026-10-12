"""Candidate outcomes dual-write, routed by their candidate's forecast_date (Phase 1.2)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _db(cand_day):
    from database import Base, ForecastCandidate, ForecastCandidateOutcome

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(
        ForecastCandidate(
            id=30,
            item_id=1,
            forecast_date=cand_day,
            horizon_days=7,
            component="centre",
            candidate_name="last_price",
            candidate_version="v1",
            config_fingerprint="a" * 64,
            feature_cutoff_at=datetime(2026, 10, 1),
            artifact_version="x",
            anchor_price=10.0,
            centre_price=10.0,
            predicted_price_low=9.0,
            predicted_price_high=11.0,
        )
    )
    db.add(
        ForecastCandidateOutcome(
            candidate_id=30, base_price=10.0, actual_price=11.0, resolved_at=datetime.now(), resolution_version="r1"
        )
    )
    db.commit()
    return db


def test_routed_by_the_candidate_s_date_not_the_resolution_date(tmp_path, monkeypatch):
    from backtest.candidate_resolution import _candidate_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    cand_day = date.today().replace(day=1) - timedelta(days=1)  # last day of the previous month
    assert _candidate_outcome_partitions(_db(cand_day), {30}) is True
    path = tmp_path / "forecast_candidate_outcomes" / f"{cand_day:%Y-%m}.parquet"
    got = duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()
    assert got["candidate_id"].tolist() == [30]
    assert "_part" not in got.columns


def test_nothing_touched_writes_nothing(tmp_path, monkeypatch):
    from backtest.candidate_resolution import _candidate_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    assert _candidate_outcome_partitions(_db(date.today()), set()) is True
    assert not (tmp_path / "forecast_candidate_outcomes").exists()
