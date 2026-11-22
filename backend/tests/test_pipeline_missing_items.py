from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import collectors.csgotrader_aggregator as aggregator_module
import pandas as pd
from collectors.pipeline import DataPipeline
from db import ops_partitions


class FakeAggregator:
    _raw_sources: ClassVar[dict] = {}

    def collect_batch_items(self, item_names):
        return {}

    def fetch_exchange_rates(self):
        return None

    def find_source_key_candidates(self, name, limit=5):
        return []


def test_a_missing_item_counts_as_an_error(monkeypatch, tmp_path):
    """No historical fallback: an item missing from the day's dumps is an error.

    The fallback read Postgres `price_history`, frozen since 2026-07-11, so it
    recovered nothing in production; Phase 4 deleted it.
    """
    name = "Sticker | YEKINDAR (Holo) | Shanghai 2024"
    ops = tmp_path / "ops"
    ops_partitions.upsert(
        "items",
        pd.DataFrame([{"id": 1, "item_id": "sticker-yekindar", "name": name, "type": "sticker", "is_backfilled": 1}]),
        root=ops,
    )
    monkeypatch.setattr(aggregator_module, "CSGOTraderAggregator", FakeAggregator)

    result = DataPipeline(ops_root=ops).run_full_aggregator_collection()

    assert result["status"] == "success"
    assert result["items_collected"] == 0
    assert result["errors"] == 1
    assert result["missing_name_sample"] == [name]
    assert "fallback_items_collected" not in result


def test_missing_name_report_groups_by_conservative_pattern():
    pipeline = DataPipeline()
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
    assert report["buckets"][1]["subgroups"][1]["subgroups"][0]["sample"] == ["StatTrak™ M249 | Hypnosis (Factory New)"]
    assert report["buckets"][1]["subgroups"][1]["subgroups"][1]["sample"] == [
        "Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage",
        "Souvenir Charm | Austin 2025 Highlight | Spinx Quadra Kill",
    ]
