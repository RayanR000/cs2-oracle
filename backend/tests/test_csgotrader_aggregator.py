from __future__ import annotations

from datetime import datetime

from collectors.csgotrader_aggregator import CSGOTraderAggregator


def test_collect_batch_items_matches_sticker_without_event_suffix(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {"Sticker | noway (Holo)": 12.34},
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["Sticker | noway (Holo) | Shanghai 2024"])

    assert results["Sticker | noway (Holo) | Shanghai 2024"][0] == 12.34
    assert results["Sticker | noway (Holo) | Shanghai 2024"][1] == 0
    assert isinstance(results["Sticker | noway (Holo) | Shanghai 2024"][2], datetime)


def test_collect_batch_items_still_prefers_exact_sticker_match(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {
            "Sticker | noway (Holo)": 12.34,
            "Sticker | noway (Holo) | Shanghai 2024": 56.78,
        },
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["Sticker | noway (Holo) | Shanghai 2024"])

    assert results["Sticker | noway (Holo) | Shanghai 2024"][0] == 56.78
