"""`_load_panel` must read columns `forecast_outcomes` actually has.

The estimator selected `price_tier` from `forecast_outcomes`, but that column lives on
`prediction_accuracy` (migration 0019). Postgres raised UndefinedColumn on every prod run and
`served_coverage_factors` swallows a read failure by design, so the feedback looked "data-blocked"
when it was schema-blocked. These tests run the real query against a schema built from the ORM
metadata, so a column that does not exist on the table fails here instead of silently in prod.
"""
import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, ForecastOutcome
from backtest.scoring import HEADLINE_MIN_TIER
from models.served_recalibration import PANEL_COLUMNS, _load_panel


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)()
    engine.dispose()


def _outcome(**kw):
    base = dict(
        forecast_id=1, item_id=1, forecast_date=dt.date(2026, 8, 21), horizon_days=7,
        target_date=dt.date(2026, 8, 28), base_price=10.0, current_price=10.0,
        predicted_price_low=9.0, predicted_price_mid=10.0, predicted_price_high=11.0,
        actual_price=10.5, direction_correct=0, abs_error=0.5,
    )
    base.update(kw)
    return ForecastOutcome(**base)


def test_load_panel_runs_against_the_real_forecast_outcomes_schema(session):
    session.add(_outcome())
    session.commit()

    panel = _load_panel(session, [7])

    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert len(panel) == 1


@pytest.mark.parametrize("price, tier", [(0.99, 0), (1.0, 1), (25.0, 3), (1500.0, 5)])
def test_price_tier_is_derived_from_the_serve_time_price(session, price, tier):
    session.add(_outcome(base_price=price, current_price=price))
    session.commit()

    assert int(_load_panel(session, [7])["price_tier"].iloc[0]) == tier


def test_null_base_price_falls_back_to_current_price(session):
    session.add(_outcome(base_price=None, current_price=150.0))
    session.commit()

    assert int(_load_panel(session, [7])["price_tier"].iloc[0]) == 4


def test_sub_dollar_rows_are_below_the_headline_cohort(session):
    session.add(_outcome(base_price=0.50, current_price=0.50))
    session.commit()

    assert int(_load_panel(session, [7])["price_tier"].iloc[0]) < HEADLINE_MIN_TIER


def test_empty_panel_still_carries_the_derived_column(session):
    panel = _load_panel(session, [7])

    assert panel.empty
    assert list(panel.columns) == list(PANEL_COLUMNS)
