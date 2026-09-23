"""`_load_panel` must read columns `forecast_outcomes` actually has.

The estimator selected `price_tier` from `forecast_outcomes`, but that column lives on
`prediction_accuracy` (migration 0019). Postgres raised UndefinedColumn on every prod run and
`served_coverage_factors` swallows a read failure by design, so the feedback looked "data-blocked"
when it was schema-blocked. These tests run the real query against a schema built from the ORM
metadata, so a column that does not exist on the table fails here instead of silently in prod.
"""

import datetime as dt

import numpy as np
import pytest
from backtest.scoring import HEADLINE_MIN_TIER
from database import Base, ForecastOutcome
from models.served_recalibration import PANEL_COLUMNS, _load_panel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)()
    engine.dispose()


def _outcome(**kw):
    base = dict(
        forecast_id=1,
        item_id=1,
        forecast_date=dt.date(2026, 8, 21),
        horizon_days=7,
        target_date=dt.date(2026, 8, 28),
        base_price=10.0,
        current_price=10.0,
        predicted_price_low=9.0,
        predicted_price_mid=10.0,
        predicted_price_high=11.0,
        actual_price=10.5,
        direction_correct=0,
        abs_error=0.5,
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


# --- band_multiplier: the multiplier each row was SERVED at, joined from item_forecasts ------


def _forecast(session, fid, *, fd, mult, h=7):
    from database import ItemForecast

    session.add(
        ItemForecast(
            id=fid,
            item_id=1,
            forecast_date=fd,
            horizon_days=h,
            price_low=9.0,
            price_mid=10.0,
            price_high=11.0,
            current_price=10.0,
            band_multiplier=mult,
        )
    )


def test_band_multiplier_is_joined_from_the_served_forecast(session):
    _forecast(session, 1, fd=dt.date(2026, 9, 21), mult=0.5385)
    session.add(_outcome(forecast_id=1, forecast_date=dt.date(2026, 9, 21)))
    session.commit()

    assert _load_panel(session, [7])["band_multiplier"].iloc[0] == pytest.approx(0.5385)


def test_unrecorded_multiplier_is_one_before_first_activation_and_unknown_after(session):
    _forecast(session, 1, fd=dt.date(2026, 9, 10), mult=None)
    _forecast(session, 2, fd=dt.date(2026, 9, 21), mult=None)
    session.add(_outcome(forecast_id=1, forecast_date=dt.date(2026, 9, 10)))
    session.add(_outcome(forecast_id=2, forecast_date=dt.date(2026, 9, 21)))
    session.commit()

    m = _load_panel(session, [7]).sort_values("forecast_date")["band_multiplier"].to_numpy()
    assert m[0] == 1.0
    assert np.isnan(m[1])


def test_outcome_whose_forecast_row_is_gone_is_treated_as_unrecorded(session):
    session.add(_outcome(forecast_id=999, forecast_date=dt.date(2026, 9, 21)))
    session.commit()

    assert np.isnan(_load_panel(session, [7])["band_multiplier"].iloc[0])


def test_missing_column_degrades_to_unrecorded_rather_than_failing_the_read(session):
    # The prod migration may lag the code. A failed read returns {} -> multiplier 1.0 -> the
    # band snaps to full width, the exact failure class of the 2026-08-18 UndefinedColumn bug.
    from sqlalchemy import text

    session.add(_outcome(forecast_id=1, forecast_date=dt.date(2026, 9, 10)))
    session.add(_outcome(forecast_id=2, forecast_date=dt.date(2026, 9, 21)))
    session.commit()
    session.execute(text("ALTER TABLE item_forecasts DROP COLUMN band_multiplier"))
    session.commit()

    m = _load_panel(session, [7]).sort_values("forecast_date")["band_multiplier"].to_numpy()
    assert m[0] == 1.0
    assert np.isnan(m[1])
