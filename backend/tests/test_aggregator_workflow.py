from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database as database_module
import collectors.csgotrader_aggregator as aggregator_module
from collectors.pipeline import DataPipeline
from database import Item, PriceHistory, CollectionRun


class FakeAggregator:
    def __init__(self, price_data=None):
        self._price_data = price_data or {}

    def collect_batch_items(self, item_names):
        results = {}
        for name in item_names:
            if name in self._price_data:
                price = self._price_data[name]
                results[name] = (float(price), 42, datetime.now(timezone.utc).replace(tzinfo=None))
        return results

    def find_source_key_candidates(self, name, limit=5):
        return []


@pytest.fixture(autouse=True)
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
    # Drop all tables between tests for isolation
    database_module.Base.metadata.drop_all(bind=test_engine)


def seed_items(db, items_data):
    """Helper to seed items and return them."""
    items = []
    for data in items_data:
        item = Item(**data)
        db.add(item)
        items.append(item)
    db.commit()
    for item in items:
        db.refresh(item)
    return items


class TestHappyPath:
    """Simulate the exact flow that the GitHub Actions aggregator-update.yml
    workflow executes:
      1. run_task.py aggregate
      2. DataPipeline.run_full_aggregator_collection()
      3. CSGOTraderAggregator.collect_batch_items()
      4. DB insert into price_history + recording CollectionRun
    """

    def test_full_workflow_collects_items_and_records_run(self, monkeypatch):
        db = database_module.SessionLocal()
        try:
            items = seed_items(db, [
                {"item_id": "ak-47-redline-field-tested", "name": "AK-47 | Redline (Field-Tested)", "type": "skin"},
                {"item_id": "stattrak-usp-s-cortex-factory-new", "name": "StatTrak USPS | Cortex (Factory New)", "type": "skin"},
                {"item_id": "sticker-s1mple-holo-shanghai-2024", "name": "Sticker | s1mple (Holo) | Shanghai 2024", "type": "sticker"},
                {"item_id": "skeleton-knife-night-stained-field-tested", "name": "Skeleton Knife | Night Stained (Field-Tested)", "type": "skin"},
                {"item_id": "operation-riptide-case", "name": "Operation Riptide Case", "type": "case"},
            ])

            fake_prices = {
                "AK-47 | Redline (Field-Tested)": 22.50,
                "Sticker | s1mple (Holo) | Shanghai 2024": 1.20,
                "Skeleton Knife | Night Stained (Field-Tested)": 250.00,
                "Operation Riptide Case": 0.50,
            }
            monkeypatch.setattr(
                aggregator_module, "CSGOTraderAggregator",
                lambda: FakeAggregator(price_data=fake_prices),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "success"
            assert result["items_collected"] >= 1
            assert result["total_items"] == len(items)
            assert result["errors"] >= 0

            history_rows = db.query(PriceHistory).all()
            assert len(history_rows) > 0
            for row in history_rows:
                assert row.source == "aggregator_sync"
                assert row.price > 0
                assert row.volume == 42

            run_record = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
            assert run_record is not None
            assert run_record.status == "completed"
            assert run_record.total_items == len(items)
            assert run_record.duration_seconds > 0
        finally:
            db.close()

    def test_missing_item_falls_back_to_historical_price(self, monkeypatch):
        """When the aggregator cannot match an item, the pipeline
        recovers from the last non-aggregator price history."""
        db = database_module.SessionLocal()
        try:
            [item] = seed_items(db, [
                {"item_id": "glock-18-royal-legion-minimal-wear", "name": "Glock-18 | Royal Legion (Minimal Wear)", "type": "skin"},
            ])

            db.add(PriceHistory(
                item_id=item.id,
                timestamp=datetime.utcnow() - timedelta(days=1),
                price=3.75,
                volume=10,
                source="steam_batch",
            ))
            db.commit()

            monkeypatch.setattr(
                aggregator_module, "CSGOTraderAggregator",
                lambda: FakeAggregator(price_data={}),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "success"
            assert result["items_collected"] == 1
            assert result["errors"] == 0

            latest = (
                db.query(PriceHistory)
                .filter(PriceHistory.item_id == item.id)
                .order_by(PriceHistory.timestamp.desc())
                .first()
            )
            assert latest is not None
            assert latest.source == "historical_fallback:steam_batch"
            assert latest.price == 3.75
        finally:
            db.close()

    def test_no_items_in_database_returns_skipped(self, monkeypatch):
        """When the DB has zero items, the pipeline should skip."""
        db = database_module.SessionLocal()
        try:
            monkeypatch.setattr(
                aggregator_module, "CSGOTraderAggregator",
                lambda: FakeAggregator(price_data={"AK-47 | Redline": 22.50}),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "skipped"
            assert result["reason"] == "no_items"
        finally:
            db.close()

    def test_aggregator_exception_records_failed_run(self, monkeypatch):
        """If the aggregator raises during collection, a failed
        CollectionRun is recorded."""
        db = database_module.SessionLocal()
        try:
            [item] = seed_items(db, [
                {"item_id": "test-item-blows-up", "name": "Weapon | Will Explode", "type": "skin"},
            ])

            class ExplodingAggregator:
                def collect_batch_items(self, item_names):
                    raise RuntimeError("simulated network failure")
                def find_source_key_candidates(self, name, limit=5):
                    return []

            monkeypatch.setattr(
                aggregator_module, "CSGOTraderAggregator",
                lambda: ExplodingAggregator(),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "failed"
            assert "simulated network failure" in result["error"]

            failed_run = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
            assert failed_run is not None
            assert failed_run.status == "failed"
            assert failed_run.error_message is not None
        finally:
            db.close()

    def test_all_items_collected_when_all_match(self, monkeypatch):
        db = database_module.SessionLocal()
        try:
            names = [
                "AK-47 | Redline (Field-Tested)",
                "M4A4 | Howl (Factory New)",
                "AWP | Dragon Lore (Battle-Scarred)",
                "Desert Eagle | Code Red (Minimal Wear)",
                "USP-S | Kill Confirmed (Field-Tested)",
            ]
            items = seed_items(db, [
                {"item_id": f"test-{i}", "name": names[i], "type": "skin"}
                for i in range(5)
            ])
            fake_prices = {name: float(i + 1) * 10.0 for i, name in enumerate(names)}
            monkeypatch.setattr(
                aggregator_module, "CSGOTraderAggregator",
                lambda: FakeAggregator(price_data=fake_prices),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "success"
            assert result["items_collected"] == 5
            assert result["total_items"] == 5
            assert result["errors"] == 0
        finally:
            db.close()

    def test_collect_batch_items_returns_tuple_with_price_volume_and_timestamp(self):
        """Verify the raw aggregator returns the expected (price, volume, timestamp) shape."""
        aggregator = aggregator_module.CSGOTraderAggregator()
        aggregator._price_cache = {"Sticker | test (Holo)": 5.50}

        results = aggregator.collect_batch_items(["Sticker | test (Holo)"])
        assert "Sticker | test (Holo)" in results
        result = results["Sticker | test (Holo)"]
        assert len(result) == 3
        assert result[0] == 5.50
        assert isinstance(result[1], int)
        assert isinstance(result[2], datetime)