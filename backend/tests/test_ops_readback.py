"""db/ops_readback.py: committed rows re-read from Postgres, with the ids Postgres issued."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

D = date(2026, 10, 11)


def _session():
    from database import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed(db):
    from database import ForecastCandidate, ForecastCandidateOutcome, ForecastOutcome, Item, ItemForecast

    db.add(Item(id=1, item_id="ak_1", name="AK 1", type="rifle"))
    db.add_all(
        [
            ItemForecast(id=10, item_id=1, forecast_date=D, horizon_days=7, price_mid=10.0),
            ItemForecast(id=11, item_id=1, forecast_date=D, horizon_days=3, price_mid=9.0),
        ]
    )
    db.add(
        ForecastOutcome(
            id=20,
            forecast_id=10,
            item_id=1,
            forecast_date=D,
            horizon_days=7,
            target_date=datetime(2026, 10, 18),
            predicted_price_mid=10.0,
            actual_price=11.0,
            abs_error=1.0,
        )
    )
    db.add(
        ForecastCandidate(
            id=30,
            item_id=1,
            forecast_date=D,
            horizon_days=7,
            component="centre",
            candidate_name="last_price",
            candidate_version="v1",
            config_fingerprint="a" * 64,
            feature_cutoff_at=datetime(2026, 10, 11),
            artifact_version="x",
            anchor_price=10.0,
            centre_price=10.0,
            predicted_price_low=9.0,
            predicted_price_high=11.0,
        )
    )
    db.add(
        ForecastCandidateOutcome(
            candidate_id=30,
            base_price=10.0,
            actual_price=11.0,
            resolved_at=datetime(2026, 10, 18),
            resolution_version="r1",
        )
    )
    db.commit()


def test_forecasts_carry_ids_and_slug_and_filter_to_keys():
    from db import ops_readback

    db = _session()
    _seed(db)
    got = ops_readback.forecasts(db, {D}, {(1, D, 7)})
    assert got["id"].tolist() == [10]
    assert got["item_slug"].tolist() == ["ak_1"]


def test_outcomes_candidates_and_candidate_outcomes():
    from db import ops_readback

    db = _session()
    _seed(db)
    assert ops_readback.outcomes(db, forecast_ids=[10])["id"].tolist() == [20]
    assert ops_readback.outcomes(db, dates=[D])["item_slug"].tolist() == ["ak_1"]
    assert ops_readback.candidates(db, {D})["id"].tolist() == [30]
    co = ops_readback.candidate_outcomes(db, candidate_ids=[30])
    assert co["candidate_id"].tolist() == [30]
    assert str(co["_part"].iloc[0])[:10] == "2026-10-11"
    assert ops_readback.items(db, ["ak_1"])["id"].tolist() == [1]


def test_a_column_prod_lacks_is_not_selected():
    from db import ops_readback

    db = _session()
    _seed(db)
    db.execute(text("ALTER TABLE item_forecasts DROP COLUMN band_multiplier"))
    db.commit()
    got = ops_readback.forecasts(db, {D})
    assert "band_multiplier" not in got.columns
    assert len(got) == 2


def test_nothing_committed_is_an_empty_frame():
    from db import ops_readback

    db = _session()
    assert ops_readback.forecasts(db, {D}).empty
    assert ops_readback.outcomes(db, forecast_ids=[]).empty
