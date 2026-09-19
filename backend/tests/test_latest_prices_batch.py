"""Batch _latest_prices tests (PR1 Task 1.1)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from api.routes.items import _latest_prices
from database import Base, Item, PriceHistory
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    now = datetime.utcnow()
    for i in range(1, 4):
        session.add(Item(id=i, item_id=f"test-item-{i}", name=f"Test Item {i}", type="skin"))
    session.commit()
    # Two price points per item; latest must win.
    prices = {1: (10.0, 11.5), 2: (20.0, 19.0), 3: (5.0, 7.25)}
    for item_id, (old, new) in prices.items():
        session.add(
            PriceHistory(item_id=item_id, timestamp=now - timedelta(days=2), price=old, source="steam")
        )
        session.add(
            PriceHistory(item_id=item_id, timestamp=now - timedelta(days=1), price=new, source="steam")
        )
    session.commit()
    yield session
    session.close()
    engine.dispose()


def test_latest_prices_empty(db):
    assert _latest_prices(db, []) == {}


def test_latest_prices_multi_item_returns_latest(db):
    result = _latest_prices(db, [1, 2, 3])
    assert result == {1: 11.5, 2: 19.0, 3: 7.25}
    for v in result.values():
        assert isinstance(v, float)
