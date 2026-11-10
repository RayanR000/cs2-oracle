"""The ranking-transfer and centre-promotion reports load their frame from the ops partitions."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from database import Base, ForecastCandidate, ForecastCandidateOutcome, ForecastOutcome, Item, ItemForecast
from db import ops_store
from scripts import centre_promotion_report as centre
from scripts import ranking_transfer_report as ranking
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests._ops_mirror import mirror_session_to_ops

D0 = date(2026, 10, 1)
RESOLVED = datetime(2026, 10, 12, 12, 0, 0)
TABLES = ("item_forecasts", "forecast_outcomes", "forecast_candidates", "forecast_candidate_outcomes")


@pytest.fixture()
def read():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    cid = 0
    for item in (1, 2):
        s.add(Item(id=item, item_id=f"item_{item}", name=f"item {item}", type="skin"))
    for k in range(3):
        fd = D0 + timedelta(days=k)
        for item in (1, 2):
            fid = 100 * item + k
            s.add(
                ItemForecast(
                    id=fid,
                    item_id=item,
                    forecast_date=fd,
                    horizon_days=7,
                    price_low=36.0,
                    price_mid=40.0,
                    price_high=44.0,
                    current_price=40.0,
                )
            )
            s.add(
                ForecastOutcome(
                    id=1000 + fid,
                    forecast_id=fid,
                    item_id=item,
                    forecast_date=fd,
                    horizon_days=7,
                    target_date=fd + timedelta(days=7),
                    base_price=40.0,
                    current_price=40.0,
                    predicted_price_low=36.0,
                    predicted_price_mid=40.0,
                    predicted_price_high=44.0,
                    actual_price=42.0,
                    resolved_at=RESOLVED,
                    in_interval=1,
                    direction_correct=0,
                    abs_error=2.0,
                )
            )
            for component, name in (("centre", "last_price"), ("ranking", "lambdarank_v1")):
                cid += 1
                s.add(
                    ForecastCandidate(
                        id=cid,
                        item_id=item,
                        forecast_date=fd,
                        horizon_days=7,
                        component=component,
                        candidate_name=name,
                        candidate_version="v1",
                        centre_price=41.0 if component == "centre" else None,
                        predicted_price_low=37.0 if component == "centre" else None,
                        predicted_price_high=45.0 if component == "centre" else None,
                        score=0.5 if component == "ranking" else None,
                        anchor_price=40.0,
                        feature_cutoff_at=RESOLVED,
                        config_fingerprint="a" * 64,
                    )
                )
                s.add(
                    ForecastCandidateOutcome(
                        candidate_id=cid,
                        base_price=40.0,
                        actual_price=42.0,
                        resolved_at=RESOLVED,
                        resolution_version="v1",
                    )
                )
    s.commit()
    mirror_session_to_ops(s)
    yield ops_store.reader(TABLES)
    engine.dispose()


def test_ranking_frame_comes_from_the_partitions(read):
    frame = ranking.load_horizon_frame(read, 7)
    assert len(frame["rows"]) == 6
    assert frame["observed_batches"] == 3 and frame["expected_batches"] == 3
    assert all(r["legs_match"] and r["rank_score"] == 0.5 for r in frame["rows"])


def test_centre_frame_comes_from_the_partitions(read):
    frame = centre.load_horizon_frame(read, 7)
    assert len(frame["candidate"]) == 6 and len(frame["production"]) == 6
    assert frame["observed_batches"] == 3 and frame["expected_batches"] == 3
    assert {r["centre_price"] for r in frame["candidate"]} == {41.0}


def test_an_empty_horizon_loads_as_zero_batches(read):
    assert ranking.load_horizon_frame(read, 30) == {"rows": [], "expected_batches": 0, "observed_batches": 0}
