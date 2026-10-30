"""tests/_ops_mirror.py: a test's SQLite rows, copied into the hermetic ops partitions."""

from __future__ import annotations

from datetime import date

import pytest
from database import Base, ForecastOutcome, Item, ItemForecast
from db import ops_store
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests._ops_mirror import mirror_session_to_ops


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)()
    engine.dispose()


def _seed(session):
    session.add(Item(id=1, item_id="ak", name="ak", type="skin"))
    session.add(ItemForecast(id=10, item_id=1, forecast_date=date(2026, 7, 1), horizon_days=3, price_mid=3.0))
    session.add(
        ForecastOutcome(
            id=20,
            forecast_id=10,
            item_id=1,
            forecast_date=date(2026, 7, 1),
            horizon_days=3,
            target_date=date(2026, 7, 4),
            predicted_price_mid=3.0,
            actual_price=3.1,
            abs_error=0.1,
        )
    )


def test_seeded_rows_read_back_through_the_store(session):
    _seed(session)  # not committed: pending rows count
    mirror_session_to_ops(session)
    rows = ops_store.reader()(
        "SELECT o.id, f.price_mid, i.item_id FROM forecast_outcomes o JOIN item_forecasts f ON f.id = o.forecast_id JOIN items i ON i.id = o.item_id"
    )
    assert [(r.id, r.price_mid, r.item_id) for r in rows] == [(20, 3.0, "ak")]


def test_an_empty_table_is_an_empty_view_not_a_missing_one(session):
    mirror_session_to_ops(session)
    assert ops_store.reader()("SELECT count(*) AS n FROM forecast_outcomes")[0].n == 0


def test_a_second_mirror_replaces_rather_than_merges(session):
    _seed(session)
    session.commit()
    mirror_session_to_ops(session)
    session.query(ForecastOutcome).delete()
    session.commit()
    mirror_session_to_ops(session)
    assert ops_store.reader()("SELECT count(*) AS n FROM forecast_outcomes")[0].n == 0


def test_it_refuses_a_directory_outside_pytest_tmp(session, tmp_path):
    from pathlib import Path

    with pytest.raises(RuntimeError, match="pytest"):
        mirror_session_to_ops(session, root=Path("/var/not-a-test-dir/ops"))


@pytest.mark.usefixtures("ops_mirror")
def test_the_fixture_mirrors_before_every_connect(session):
    _seed(session)
    session.commit()
    assert len(ops_store.reader(("items",))("SELECT id FROM items")) == 1
    session.add(Item(id=2, item_id="awp", name="awp", type="skin"))
    session.commit()
    assert len(ops_store.reader(("items",))("SELECT id FROM items")) == 2
