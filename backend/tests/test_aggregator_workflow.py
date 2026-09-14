from __future__ import annotations

from datetime import UTC, datetime, timedelta

import collectors.csgotrader_aggregator as aggregator_module
import database as database_module
import pytest
from collectors.pipeline import DataPipeline
from database import CollectionRun, Item, PriceHistory
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class FakeAggregator:
    _raw_sources: dict = {}

    def __init__(self, price_data=None):
        self._price_data = price_data or {}

    def collect_batch_items(self, item_names):
        results = {}
        now = datetime.now(UTC).replace(tzinfo=None)
        for name in item_names:
            if name in self._price_data:
                price = float(self._price_data[name])
                results[name] = {"steam": (price, 42, now)}
        return results

    def fetch_exchange_rates(self):
        return None

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
            items = seed_items(
                db,
                [
                    {"item_id": "ak-47-redline-field-tested", "name": "AK-47 | Redline (Field-Tested)", "type": "skin"},
                    {
                        "item_id": "stattrak-usp-s-cortex-factory-new",
                        "name": "StatTrak USPS | Cortex (Factory New)",
                        "type": "skin",
                    },
                    {
                        "item_id": "sticker-s1mple-holo-shanghai-2024",
                        "name": "Sticker | s1mple (Holo) | Shanghai 2024",
                        "type": "sticker",
                    },
                    {
                        "item_id": "skeleton-knife-night-stained-field-tested",
                        "name": "Skeleton Knife | Night Stained (Field-Tested)",
                        "type": "skin",
                    },
                    {"item_id": "operation-riptide-case", "name": "Operation Riptide Case", "type": "case"},
                ],
            )

            fake_prices = {
                "AK-47 | Redline (Field-Tested)": 22.50,
                "Sticker | s1mple (Holo) | Shanghai 2024": 1.20,
                "Skeleton Knife | Night Stained (Field-Tested)": 250.00,
                "Operation Riptide Case": 0.50,
            }
            monkeypatch.setattr(
                aggregator_module,
                "CSGOTraderAggregator",
                lambda: FakeAggregator(price_data=fake_prices),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "success"
            assert result["items_collected"] >= 1
            assert result["total_items"] == len(items)
            assert result["errors"] >= 0

            run_record = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
            assert run_record is not None
            assert run_record.status == "completed"
            assert run_record.total_items == len(items)
            assert run_record.duration_seconds > 0
        finally:
            db.close()

    def test_snapshot_csv_is_named_from_the_snapshot_date_not_the_clock(self, monkeypatch):
        """The archive lost whole days to a wall-clock filename.

        `agg_date` came from the pipeline's own `datetime.utcnow()`, and the
        workflow's append step read `date -u +%F` from a *separate* clock one
        step later. A cron firing at 23:5x two nights running stamped the same
        date and lost the day between; a run straddling midnight would have had
        the two steps disagree on the filename outright.

        Both sides now resolve through collectors.snapshot_date, so pinning that
        value pins every path this run produces.
        """
        monkeypatch.setenv("AGGREGATOR_SNAPSHOT_DATE", "2026-07-27")
        db = database_module.SessionLocal()
        try:
            seed_items(
                db,
                [
                    {"item_id": "ak-47-redline-field-tested", "name": "AK-47 | Redline (Field-Tested)", "type": "skin"},
                ],
            )
            monkeypatch.setattr(
                aggregator_module,
                "CSGOTraderAggregator",
                lambda: FakeAggregator(price_data={"AK-47 | Redline (Field-Tested)": 22.50}),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "success"
            assert result["snapshot_csv_path"].endswith("aggregator-snapshots-2026-07-27.csv")
        finally:
            db.close()

    def test_exchange_rates_csv_also_uses_the_snapshot_date(self, monkeypatch):
        """The rates branch reads the same resolved day.

        FakeAggregator returns None for rates, so this second CSV path was never
        exercised by any test — and it is a *separate* reference to the resolved
        day further down the function. Worth its own case: a NameError there
        would only ever have surfaced in production.
        """
        monkeypatch.setenv("AGGREGATOR_SNAPSHOT_DATE", "2026-07-30")

        class RatesAggregator(FakeAggregator):
            def fetch_exchange_rates(self):
                return {"EUR": 0.92, "GBP": 0.79}

        db = database_module.SessionLocal()
        try:
            seed_items(
                db,
                [
                    {"item_id": "ak-47-redline-field-tested", "name": "AK-47 | Redline (Field-Tested)", "type": "skin"},
                ],
            )
            monkeypatch.setattr(
                aggregator_module,
                "CSGOTraderAggregator",
                lambda: RatesAggregator(price_data={"AK-47 | Redline (Field-Tested)": 22.50}),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "success"
            assert result["exchange_rates_csv_path"].endswith("exchange-rates-2026-07-30.csv")
        finally:
            db.close()

    def test_missing_item_falls_back_to_historical_price(self, monkeypatch):
        """When the aggregator cannot match an item, the pipeline
        recovers from the last non-aggregator price history."""
        db = database_module.SessionLocal()
        try:
            [item] = seed_items(
                db,
                [
                    {
                        "item_id": "glock-18-royal-legion-minimal-wear",
                        "name": "Glock-18 | Royal Legion (Minimal Wear)",
                        "type": "skin",
                    },
                ],
            )

            db.add(
                PriceHistory(
                    item_id=item.id,
                    timestamp=datetime.utcnow() - timedelta(days=1),
                    price=3.75,
                    volume=10,
                    source="steam_batch",
                )
            )
            db.commit()

            monkeypatch.setattr(
                aggregator_module,
                "CSGOTraderAggregator",
                lambda: FakeAggregator(price_data={}),
            )

            pipeline = DataPipeline(db_session=db)
            result = pipeline.run_full_aggregator_collection()

            assert result["status"] == "success"
            assert result["items_collected"] == 1
            assert result["errors"] == 0
        finally:
            db.close()

    def test_no_items_in_database_returns_skipped(self, monkeypatch):
        """When the DB has zero items, the pipeline should skip."""
        db = database_module.SessionLocal()
        try:
            monkeypatch.setattr(
                aggregator_module,
                "CSGOTraderAggregator",
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
            [item] = seed_items(
                db,
                [
                    {"item_id": "test-item-blows-up", "name": "Weapon | Will Explode", "type": "skin"},
                ],
            )

            class ExplodingAggregator:
                def collect_batch_items(self, item_names):
                    raise RuntimeError("simulated network failure")

                def find_source_key_candidates(self, name, limit=5):
                    return []

            monkeypatch.setattr(
                aggregator_module,
                "CSGOTraderAggregator",
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
            items = seed_items(db, [{"item_id": f"test-{i}", "name": names[i], "type": "skin"} for i in range(5)])
            fake_prices = {name: float(i + 1) * 10.0 for i, name in enumerate(names)}
            monkeypatch.setattr(
                aggregator_module,
                "CSGOTraderAggregator",
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

    def test_collect_batch_items_returns_sources_dict(self):
        """Verify the raw aggregator returns the expected sources dict."""
        aggregator = aggregator_module.CSGOTraderAggregator()
        aggregator._raw_sources = {
            "steam": {"Sticker | test (Holo)": {"last_24h": 5.50, "last_7d": 4.50, "last_30d": 4.00, "last_90d": 3.50}}
        }

        results = aggregator.collect_batch_items(["Sticker | test (Holo)"])
        assert "Sticker | test (Holo)" in results
        sources = results["Sticker | test (Holo)"]
        assert sources is not None
        assert "steam" in sources
        price, vol, ts = sources["steam"]
        assert price == 5.50
        assert vol == 0
        assert isinstance(ts, datetime)
        assert "steam_7d" in sources
        assert sources["steam_7d"][0] == 4.50
        assert "steam_30d" in sources
        assert sources["steam_30d"][0] == 4.00
        assert "steam_90d" in sources
        assert sources["steam_90d"][0] == 3.50
