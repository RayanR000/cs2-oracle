from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

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


def test_full_aggregator_collection_recovers_from_previous_aggregator_sync(monkeypatch):
    db = database_module.SessionLocal()
    try:
        item = Item(
            item_id="sticker-lux-shanghai-2024",
            name="Sticker | lux | Shanghai 2024",
            type="sticker",
            release_date=datetime(2024, 1, 1),
        )
        db.add(item)
        db.commit()

        db.add(
            PriceHistory(
                item_id=item.id,
                timestamp=datetime.utcnow() - timedelta(days=1),
                price=4.2,
                volume=11,
                source="aggregator_sync",
            )
        )
        db.commit()

        monkeypatch.setattr(aggregator_module, "CSGOTraderAggregator", FakeAggregator)

        pipeline = DataPipeline(db_session=db)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "success"
        assert result["items_collected"] >= 1
        assert result["errors"] >= 0

        latest_row = (
            db.query(PriceHistory)
            .filter(PriceHistory.item_id == item.id)
            .order_by(PriceHistory.timestamp.desc(), PriceHistory.id.desc())
            .first()
        )
        assert latest_row is not None
        assert latest_row.source == "historical_fallback:aggregator_sync"
        assert latest_row.price == 4.2
    finally:
        db.close()


def test_missing_name_report_groups_by_conservative_pattern():
    pipeline = DataPipeline(db_session=None)
    item_map = {
        "Sticker | YEKINDAR (Holo) | Shanghai 2024": [SimpleNamespace(type="skin")],
        "Sticker Slab | Team Liquid (Gold) | Budapest 2025": [SimpleNamespace(type="other")],
        "StatTrak™ M249 | Hypnosis (Factory New)": [SimpleNamespace(type="skin")],
        "★ Skeleton Knife | Damascus Steel (Well-Worn)": [SimpleNamespace(type="skin")],
        "SCAR-20 | Wild Berry (Well-Worn)": [SimpleNamespace(type="skin")],
        "SG 553 | Berry Gel Coat (Well-Worn)": [SimpleNamespace(type="skin")],
        "Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage": [SimpleNamespace(type="skin")],
        "Souvenir Charm | Austin 2025 Highlight | Spinx Quadra Kill": [SimpleNamespace(type="skin")],
        "Masterminds 2 Music Kit Box": [SimpleNamespace(type="case")],
        "Plain Missing Item": [SimpleNamespace(type="other")],
    }

    report = pipeline._build_missing_name_report(list(item_map.keys()), item_map)

    assert report["total_missing_names"] == 10
    assert report["total_missing_rows"] == 10
    assert [bucket["key"] for bucket in report["buckets"]] == [
        "sticker_items",
        "skin_variant_items",
        "container_items",
        "other_items",
    ]
    assert report["buckets"][0]["sample"] == [
        "Sticker | YEKINDAR (Holo) | Shanghai 2024",
        "Sticker Slab | Team Liquid (Gold) | Budapest 2025",
    ]
    assert report["buckets"][0]["count"] == 2
    assert report["buckets"][1]["sample"] == [
        "StatTrak™ M249 | Hypnosis (Factory New)",
        "★ Skeleton Knife | Damascus Steel (Well-Worn)",
        "SCAR-20 | Wild Berry (Well-Worn)",
        "SG 553 | Berry Gel Coat (Well-Worn)",
        "Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage",
    ]
    assert [bucket["key"] for bucket in report["buckets"][1]["subgroups"]] == [
        "knife_items",
        "stattrak_souvenir_items",
        "wear_suffix_items",
    ]
    assert report["buckets"][1]["subgroups"][0]["sample"] == ["★ Skeleton Knife | Damascus Steel (Well-Worn)"]
    assert report["buckets"][1]["subgroups"][1]["sample"] == [
        "StatTrak™ M249 | Hypnosis (Factory New)",
        "Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage",
        "Souvenir Charm | Austin 2025 Highlight | Spinx Quadra Kill",
    ]
    assert report["buckets"][1]["subgroups"][2]["sample"] == [
        "SCAR-20 | Wild Berry (Well-Worn)",
        "SG 553 | Berry Gel Coat (Well-Worn)",
    ]
    assert [bucket["key"] for bucket in report["buckets"][1]["subgroups"][1]["subgroups"]] == [
        "stattrak_items",
        "souvenir_charm_items",
    ]
    assert report["buckets"][1]["subgroups"][1]["subgroups"][0]["sample"] == [
        "StatTrak™ M249 | Hypnosis (Factory New)"
    ]
    assert report["buckets"][1]["subgroups"][1]["subgroups"][1]["sample"] == [
        "Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage",
        "Souvenir Charm | Austin 2025 Highlight | Spinx Quadra Kill",
    ]
