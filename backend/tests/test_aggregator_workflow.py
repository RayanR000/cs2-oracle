from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar

import collectors.csgotrader_aggregator as aggregator_module
import pytest
from collectors.pipeline import DataPipeline


class FakeAggregator:
    _raw_sources: ClassVar[dict] = {}

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


@pytest.fixture
def ops(tmp_path):
    """The ops dir the pipeline reads its catalog from (``AGGREGATOR_OPS_DIR`` in CI)."""
    return tmp_path / "ops"


def seed_items(ops, items_data):
    """Write the catalog to ``items.parquet`` and return its rows, ids 1..n."""
    import pandas as pd
    from db import ops_partitions

    rows = [{"id": i, "is_backfilled": 0, **data} for i, data in enumerate(items_data, start=1)]
    ops_partitions.upsert("items", pd.DataFrame(rows), root=ops)
    return [SimpleNamespace(**r) for r in rows]


class TestHappyPath:
    """Simulate the exact flow that the GitHub Actions aggregator-update.yml
    workflow executes:
      1. run_task.py aggregate
      2. DataPipeline.run_full_aggregator_collection()
      3. CSGOTraderAggregator.collect_batch_items()
      4. snapshot CSV for the Parquet archive; the catalog is items.parquet and
         nothing touches a database (Phase 4, PRs 2a and 2b)
    """

    def test_full_workflow_collects_items_without_a_run_record(self, monkeypatch, ops):
        items = seed_items(
            ops,
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

        pipeline = DataPipeline(ops_root=ops)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "success"
        assert result["items_collected"] >= 1
        assert result["total_items"] == len(items)
        assert result["errors"] >= 0

        assert result["duration_seconds"] > 0

    def test_snapshot_csv_is_named_from_the_snapshot_date_not_the_clock(self, monkeypatch, ops):
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
        seed_items(
            ops,
            [
                {"item_id": "ak-47-redline-field-tested", "name": "AK-47 | Redline (Field-Tested)", "type": "skin"},
            ],
        )
        monkeypatch.setattr(
            aggregator_module,
            "CSGOTraderAggregator",
            lambda: FakeAggregator(price_data={"AK-47 | Redline (Field-Tested)": 22.50}),
        )

        pipeline = DataPipeline(ops_root=ops)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "success"
        assert result["snapshot_csv_path"].endswith("aggregator-snapshots-2026-07-27.csv")

    def test_exchange_rates_csv_also_uses_the_snapshot_date(self, monkeypatch, ops):
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

        seed_items(
            ops,
            [
                {"item_id": "ak-47-redline-field-tested", "name": "AK-47 | Redline (Field-Tested)", "type": "skin"},
            ],
        )
        monkeypatch.setattr(
            aggregator_module,
            "CSGOTraderAggregator",
            lambda: RatesAggregator(price_data={"AK-47 | Redline (Field-Tested)": 22.50}),
        )

        pipeline = DataPipeline(ops_root=ops)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "success"
        assert result["exchange_rates_csv_path"].endswith("exchange-rates-2026-07-30.csv")

    def test_missing_item_counts_as_an_error(self, monkeypatch, ops):
        """An item the aggregator cannot match is an error (no historical fallback)."""
        seed_items(
            ops,
            [
                {
                    "item_id": "glock-18-royal-legion-minimal-wear",
                    "name": "Glock-18 | Royal Legion (Minimal Wear)",
                    "type": "skin",
                },
            ],
        )

        monkeypatch.setattr(
            aggregator_module,
            "CSGOTraderAggregator",
            lambda: FakeAggregator(price_data={}),
        )

        pipeline = DataPipeline(ops_root=ops)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "success"
        assert result["items_collected"] == 0
        assert result["errors"] == 1

    def test_a_missing_catalog_fails_the_run(self, monkeypatch, ops):
        """No items.parquet under the ops dir (say AGGREGATOR_OPS_DIR points nowhere) is a
        failure, not a quiet run that collects nothing."""
        monkeypatch.setattr(
            aggregator_module,
            "CSGOTraderAggregator",
            lambda: FakeAggregator(price_data={"AK-47 | Redline": 22.50}),
        )

        pipeline = DataPipeline(ops_root=ops)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "failed"
        assert "no partition files for items" in result["error"]

    def test_aggregator_exception_fails_the_run(self, monkeypatch, ops):
        """If the aggregator raises during collection, the run reports failure
        (run_task exits non-zero on it) and writes no run record."""
        [_item] = seed_items(
            ops,
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

        pipeline = DataPipeline(ops_root=ops)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "failed"
        assert "simulated network failure" in result["error"]

        assert result["duration_seconds"] >= 0

    def test_all_items_collected_when_all_match(self, monkeypatch, ops):
        names = [
            "AK-47 | Redline (Field-Tested)",
            "M4A4 | Howl (Factory New)",
            "AWP | Dragon Lore (Battle-Scarred)",
            "Desert Eagle | Code Red (Minimal Wear)",
            "USP-S | Kill Confirmed (Field-Tested)",
        ]
        _items = seed_items(ops, [{"item_id": f"test-{i}", "name": names[i], "type": "skin"} for i in range(5)])
        fake_prices = {name: float(i + 1) * 10.0 for i, name in enumerate(names)}
        monkeypatch.setattr(
            aggregator_module,
            "CSGOTraderAggregator",
            lambda: FakeAggregator(price_data=fake_prices),
        )

        pipeline = DataPipeline(ops_root=ops)
        result = pipeline.run_full_aggregator_collection()

        assert result["status"] == "success"
        assert result["items_collected"] == 5
        assert result["total_items"] == 5
        assert result["errors"] == 0

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
