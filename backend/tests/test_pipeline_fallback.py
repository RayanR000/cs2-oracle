from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database as database_module
import collectors.csgotrader_aggregator as aggregator_module
from collectors.pipeline import DataPipeline
from database import Item, PriceHistory, CollectionRun


class FakeAggregator:
    def collect_batch_items(self, item_names):
        return {}

    def find_source_key_candidates(self, name, limit=5):
        return []


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    database_module.engine = test_engine
    database_module.SessionLocal = test_session_factory
    database_module.Base.metadata.create_all(bind=test_engine)
    yield


def test_full_aggregator_collection_recovers_exact_item_from_history(monkeypatch):
    db = database_module.SessionLocal()
    try:
        item = Item(
            item_id="sticker-yekindar-holo-shanghai-2024",
            name="Sticker | YEKINDAR (Holo) | Shanghai 2024",
            type="sticker",
            release_date=datetime(2024, 1, 1),
        )
        db.add(item)
        db.commit()

        db.add(
            PriceHistory(
                item_id=item.id,
                timestamp=datetime.utcnow() - timedelta(days=1),
                price=22.5,
                volume=7,
                source="steam_batch",
            )
        )
        db.commit()

        monkeypatch.setattr(aggregator_module, "CSGOTraderAggregator", FakeAggregator)

        pipeline = DataPipeline(db_session=db)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "success"
        assert result["items_collected"] == 1
        assert result["errors"] == 0
        assert result["missing_name_sample"] == []

        latest_row = (
            db.query(PriceHistory)
            .filter(PriceHistory.item_id == item.id)
            .order_by(PriceHistory.timestamp.desc(), PriceHistory.id.desc())
            .first()
        )
        assert latest_row is not None
        assert latest_row.source == "historical_fallback:steam_batch"
        assert latest_row.price == 22.5

        run = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
        assert run is not None
        assert run.successful == 1
        assert run.failed == 0
        assert run.source_breakdown == {
            "aggregator": 0,
            "historical_fallback": 1,
        }
    finally:
        db.close()


def test_full_aggregator_collection_does_not_stack_fallback_prefix(monkeypatch):
    db = database_module.SessionLocal()
    try:
        item = Item(
            item_id="sticker-pilot-holo",
            name="Sticker | P1L0T (Holo)",
            type="sticker",
            release_date=datetime(2024, 1, 1),
        )
        db.add(item)
        db.commit()

        db.add(
            PriceHistory(
                item_id=item.id,
                timestamp=datetime.utcnow() - timedelta(days=1),
                price=1.25,
                volume=3,
                source="historical_fallback:steam_batch",
            )
        )
        db.commit()

        monkeypatch.setattr(aggregator_module, "CSGOTraderAggregator", FakeAggregator)

        pipeline = DataPipeline(db_session=db)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "success"

        latest_row = (
            db.query(PriceHistory)
            .filter(PriceHistory.item_id == item.id)
            .order_by(PriceHistory.timestamp.desc(), PriceHistory.id.desc())
            .first()
        )
        assert latest_row is not None
        assert latest_row.source == "historical_fallback:steam_batch"
        assert len(latest_row.source) < 50
    finally:
        db.close()
