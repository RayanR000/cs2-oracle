from __future__ import annotations

from collectors.csgotrader_aggregator import CSGOTraderAggregator


def test_collect_batch_items_does_not_cross_match_sticker_event_suffix_without_base_key(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {"Sticker | noway (Holo)": 12.34},
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["Sticker | noway (Holo) | Shanghai 2024"])

    assert "Sticker | noway (Holo) | Shanghai 2024" not in results


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


def test_collect_batch_items_matches_sticker_without_finish_suffix(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {"Sticker | YEKINDAR | Shanghai 2024": 33.33},
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["Sticker | YEKINDAR (Holo) | Shanghai 2024"])

    assert results["Sticker | YEKINDAR (Holo) | Shanghai 2024"][0] == 33.33


def test_collect_batch_items_does_not_cross_match_sticker_event_suffix(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {"Sticker | YEKINDAR (Holo) | Paris 2023": 22.5},
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["Sticker | YEKINDAR (Holo) | Shanghai 2024"])

    assert "Sticker | YEKINDAR (Holo) | Shanghai 2024" not in results


def test_collect_batch_items_does_not_cross_match_sticker_without_quality(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {"Sticker | YEKINDAR | Paris 2023": 33.33},
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["Sticker | YEKINDAR (Holo) | Shanghai 2024"])

    assert "Sticker | YEKINDAR (Holo) | Shanghai 2024" not in results


def test_collect_batch_items_does_not_cross_match_sticker_quality(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {"Sticker | Liazz (Glitter) | Paris 2023": 44.44},
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["Sticker | Liazz (Holo) | Shanghai 2024"])

    assert "Sticker | Liazz (Holo) | Shanghai 2024" not in results


def test_collect_batch_items_matches_stattrak_without_trademark_symbol(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {"StatTrak M249 | Hypnosis (Factory New)": 12.5},
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["StatTrak™ M249 | Hypnosis (Factory New)"])

    assert results["StatTrak™ M249 | Hypnosis (Factory New)"][0] == 12.5


def test_collect_batch_items_matches_knife_without_leading_star(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {"Skeleton Knife | Damascus Steel (Well-Worn)": 88.8},
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["★ Skeleton Knife | Damascus Steel (Well-Worn)"])

    assert results["★ Skeleton Knife | Damascus Steel (Well-Worn)"][0] == 88.8


def test_collect_batch_items_prefers_exact_starred_knife_match(monkeypatch):
    aggregator = CSGOTraderAggregator()
    monkeypatch.setattr(
        aggregator,
        "fetch_all_prices",
        lambda: {
            "Skeleton Knife | Damascus Steel (Well-Worn)": 88.8,
            "★ Skeleton Knife | Damascus Steel (Well-Worn)": 99.9,
        },
    )
    aggregator._price_cache = {}

    results = aggregator.collect_batch_items(["★ Skeleton Knife | Damascus Steel (Well-Worn)"])

    assert results["★ Skeleton Knife | Damascus Steel (Well-Worn)"][0] == 99.9
