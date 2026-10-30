"""The served-coverage feedback panel reads the ops partitions (Phase 2 PR 4).

docs/specs/2026-10-13-ops-readers-switch-design.md. Postgres is only the read-twice side: the
factor is computed from the partitions, and a dead Postgres changes nothing served.
"""

import datetime as dt
import logging

import pandas as pd
import pytest
from database import Base, ForecastOutcome, Item, ItemForecast
from db import ops_partitions, ops_store
from models import served_recalibration as sr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests._ops_mirror import mirror_session_to_ops

SINCE = "2026-08-01"


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Item(id=1, item_id="ak", name="ak", type="skin", is_backfilled=1))
    s.commit()
    yield s
    engine.dispose()


def _seed(session, n_dates=3, mult=0.5385):
    """One forecast + outcome per date at h=7, every row >= $1 with a usable band."""
    for k in range(n_dates):
        fd = dt.date(2026, 9, 21) + dt.timedelta(days=k)
        session.add(
            ItemForecast(
                id=k + 1,
                item_id=1,
                forecast_date=fd,
                horizon_days=7,
                price_low=9.0,
                price_mid=10.0,
                price_high=11.0,
                current_price=10.0,
                band_multiplier=None if k == 0 else mult,
            )
        )
        session.add(
            ForecastOutcome(
                forecast_id=k + 1,
                item_id=1,
                forecast_date=fd,
                horizon_days=7,
                target_date=fd + dt.timedelta(days=7),
                base_price=10.0,
                current_price=10.0,
                predicted_price_low=9.0,
                predicted_price_mid=10.0,
                predicted_price_high=11.0,
                actual_price=10.2 + 0.1 * k,
                direction_correct=1,
                abs_error=0.2,
            )
        )
    session.commit()


def _sorted(panel):
    # SQLite hands a raw-SQL DATE back as a string; Postgres and DuckDB hand back a date.
    panel = panel.assign(forecast_date=pd.to_datetime(panel["forecast_date"]))
    return panel.sort_values(["forecast_date", "actual_price"]).reset_index(drop=True)


def test_store_panel_equals_postgres_panel(session):
    _seed(session)
    mirror_session_to_ops(session)

    store = sr._load_panel(ops_store.reader(), [7], since=SINCE)
    pg = sr._load_panel(ops_store.pg_reader(session), [7], since=SINCE)

    assert len(store) == 3
    pd.testing.assert_frame_equal(_sorted(store), _sorted(pg), check_dtype=False)


def test_panel_joins_partition_only_forecast_multiplier(session):
    mirror_session_to_ops(session)
    fid = ops_partitions.READ_ONLY_ID_BASE + 1
    fd = dt.date(2026, 10, 30)
    ops_partitions.upsert(
        "item_forecasts",
        pd.DataFrame(
            [
                {
                    "id": fid,
                    "item_id": 1,
                    "forecast_date": fd,
                    "horizon_days": 7,
                    "price_mid": 10.0,
                    "band_multiplier": 0.5385,
                }
            ]
        ),
        allow_closed=True,
    )
    ops_partitions.upsert(
        "forecast_outcomes",
        pd.DataFrame(
            [
                {
                    "id": fid + 1,
                    "forecast_id": fid,
                    "item_id": 1,
                    "forecast_date": fd,
                    "horizon_days": 7,
                    "base_price": 10.0,
                    "predicted_price_low": 9.0,
                    "predicted_price_mid": 10.0,
                    "predicted_price_high": 11.0,
                    "actual_price": 10.5,
                }
            ]
        ),
        allow_closed=True,
    )

    panel = sr._load_panel(ops_store.reader(), [7])

    assert len(panel) == 1
    assert panel["band_multiplier"].iloc[0] == pytest.approx(0.5385)


def test_panel_hides_pg_deleted_outcomes(session):
    _seed(session, n_dates=2)
    mirror_session_to_ops(session)
    (deleted,) = ops_store.reader(("forecast_outcomes",))("SELECT * FROM forecast_outcomes WHERE forecast_id = 2")
    row = {**deleted._asdict(), "pg_deleted": True}
    ops_partitions.upsert("forecast_outcomes", pd.DataFrame([row]), allow_closed=True)

    panel = sr._load_panel(ops_store.reader(), [7])

    assert list(panel["forecast_date"]) == [dt.date(2026, 9, 21)]


def test_missing_store_returns_no_factor_and_names_the_store(session, caplog):
    _seed(session)  # Postgres has the rows; the partitions are empty

    with caplog.at_level(logging.WARNING, logger=sr.logger.name):
        assert sr.served_coverage_factors(session, [7], min_dates=1, since=SINCE) == {}

    assert "ops partitions" in caplog.text
    assert "MissingTable" in caplog.text


def test_factors_survive_a_failing_postgres_read(session, monkeypatch):
    _seed(session)
    mirror_session_to_ops(session)
    working = sr.served_coverage_factors(session, [7], min_dates=1, since=SINCE)
    assert 7 in working

    def dead(_session):
        def read(sql):
            raise RuntimeError("cannot execute SELECT in a read-only transaction")

        return read

    monkeypatch.setattr(ops_store, "pg_reader", dead)
    rollbacks = []
    monkeypatch.setattr(session, "rollback", lambda: rollbacks.append(1))

    assert sr.served_coverage_factors(session, [7], min_dates=1, since=SINCE) == working
    assert rollbacks


def test_read_twice_logs_factor_per_horizon(session, caplog):
    _seed(session)
    mirror_session_to_ops(session)

    with caplog.at_level(logging.INFO, logger=ops_store.logger.name):
        sr.served_coverage_factors(session, [7], min_dates=1, since=SINCE)

    assert "read twice served-coverage factor per horizon: 1 keys, 0 differ" in caplog.text


def test_postgres_side_does_not_repeat_the_factor_log(session, caplog):
    _seed(session)
    mirror_session_to_ops(session)

    with caplog.at_level(logging.INFO, logger=sr.logger.name):
        sr.served_coverage_factors(session, [7], min_dates=1, since=SINCE)

    assert caplog.text.count("7d served-coverage factor") == 1
