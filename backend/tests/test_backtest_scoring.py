from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, ForecastOutcome, PredictionAccuracy


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


def test_forecast_outcome_has_freeze_columns(session):
    row = ForecastOutcome(
        forecast_id=1,
        item_id=1,
        forecast_date=date(2026, 7, 1),
        horizon_days=3,
        target_date=date(2026, 7, 4),
        current_price=1.0,
        base_price=1.05,
        predicted_price_mid=1.1,
        actual_price=1.2,
        direction_correct=1,
        abs_error=0.1,
        resolved_at=datetime(2026, 7, 4, 9, 0, 0),
    )
    session.add(row)
    session.commit()

    stored = session.query(ForecastOutcome).one()
    assert stored.base_price == 1.05
    assert stored.resolved_at == datetime(2026, 7, 4, 9, 0, 0)


def test_prediction_accuracy_has_price_tier(session):
    row = PredictionAccuracy(
        prediction_type="forecast",
        evaluation_date=date(2026, 8, 1),
        horizon_days=3,
        model_version="lgbm-v3-regime",
        price_tier=1,
        sample_count=10,
        metrics={"directional_accuracy": 55.0},
        created_at=datetime(2026, 8, 1, 9, 0, 0),
    )
    session.add(row)
    session.commit()

    assert session.query(PredictionAccuracy).one().price_tier == 1


from backtest.scoring import (
    FLAT_TOLERANCE,
    direction_from_return,
    price_tier,
    score_cohort,
)


def _record(**overrides):
    base = {
        "abs_error": 0.10,
        "pct_error": 10.0,
        "sq_error": 0.01,
        "direction_correct": 1,
        "predicted_direction": "up",
        "actual_direction": "up",
        "in_interval": 1,
        "confidence": "high",
        "base_price": 1.00,
        "actual_price": 1.10,
        "price_tier": 1,
        "item_id": 1,
    }
    base.update(overrides)
    return base


def test_direction_from_return_respects_flat_tolerance():
    assert direction_from_return(FLAT_TOLERANCE * 2) == "up"
    assert direction_from_return(-FLAT_TOLERANCE * 2) == "down"
    assert direction_from_return(0.0) == "flat"
    assert direction_from_return(FLAT_TOLERANCE) == "flat"  # boundary is inclusive-flat


def test_price_tier_boundaries():
    assert price_tier(0.99) == 0
    assert price_tier(1.0) == 1
    assert price_tier(5.0) == 2
    assert price_tier(20.0) == 3
    assert price_tier(100.0) == 4


def test_score_cohort_is_pure_and_repeatable():
    records = [_record(item_id=i) for i in range(20)]
    first, n_first = score_cohort(records)
    second, n_second = score_cohort(records)
    assert first == second
    assert n_first == n_second == 20


def test_score_cohort_does_not_mutate_its_input():
    records = [_record(item_id=i) for i in range(20)]
    snapshot = [dict(r) for r in records]
    score_cohort(records)
    assert records == snapshot


def test_score_cohort_computes_directional_accuracy():
    records = [_record(direction_correct=1) for _ in range(6)]
    records += [_record(direction_correct=0, predicted_direction="down") for _ in range(4)]
    metrics, n = score_cohort(records)
    assert n == 10
    assert metrics["directional_accuracy"] == 60.0


def test_score_cohort_uses_base_price_for_the_persistence_baseline():
    """baseline_mae is |base - actual|, the error a predict-no-change model
    would make. It must read base_price, not the retired current_price."""
    records = [_record(base_price=1.00, actual_price=1.50) for _ in range(10)]
    metrics, _ = score_cohort(records)
    assert metrics["baseline_mae"] == 0.5
