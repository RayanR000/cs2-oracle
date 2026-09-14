from __future__ import annotations

from datetime import datetime

from collectors.csgotrader_aggregator import CSGOTraderAggregator


def _make_steam_source(data: dict) -> dict:
    """Wrap flat price dict into steam _raw_sources format with full fields."""
    wrapped = {}
    for name, price in data.items():
        wrapped[name] = {"last_24h": price, "last_7d": price, "last_30d": price, "last_90d": price}
    return wrapped


def _assert_sources(results, name, expected_price, source_key="steam"):
    """Assert that results[name] has source_key with the expected price."""
    assert name in results
    sources = results[name]
    assert sources is not None
    assert source_key in sources
    price, vol, ts = sources[source_key]
    assert price == expected_price
    if source_key == "steam":
        assert vol == 0
    assert isinstance(ts, datetime)


def test_collect_batch_items_does_not_cross_match_sticker_event_suffix_without_base_key(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {"steam": _make_steam_source({"Sticker | noway (Holo)": 12.34})}

    results = aggregator.collect_batch_items(["Sticker | noway (Holo) | Shanghai 2024"])

    assert "Sticker | noway (Holo) | Shanghai 2024" not in results


def test_collect_batch_items_still_prefers_exact_sticker_match(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {
        "steam": _make_steam_source(
            {
                "Sticker | noway (Holo)": 12.34,
                "Sticker | noway (Holo) | Shanghai 2024": 56.78,
            }
        )
    }

    results = aggregator.collect_batch_items(["Sticker | noway (Holo) | Shanghai 2024"])

    _assert_sources(results, "Sticker | noway (Holo) | Shanghai 2024", 56.78)


def test_collect_batch_items_matches_sticker_without_finish_suffix(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {"steam": _make_steam_source({"Sticker | YEKINDAR | Shanghai 2024": 33.33})}

    results = aggregator.collect_batch_items(["Sticker | YEKINDAR (Holo) | Shanghai 2024"])

    _assert_sources(results, "Sticker | YEKINDAR (Holo) | Shanghai 2024", 33.33)


def test_collect_batch_items_does_not_cross_match_sticker_event_suffix(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {"steam": _make_steam_source({"Sticker | YEKINDAR (Holo) | Paris 2023": 22.5})}

    results = aggregator.collect_batch_items(["Sticker | YEKINDAR (Holo) | Shanghai 2024"])

    assert "Sticker | YEKINDAR (Holo) | Shanghai 2024" not in results


def test_collect_batch_items_does_not_cross_match_sticker_without_quality(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {"steam": _make_steam_source({"Sticker | YEKINDAR | Paris 2023": 33.33})}

    results = aggregator.collect_batch_items(["Sticker | YEKINDAR (Holo) | Shanghai 2024"])

    assert "Sticker | YEKINDAR (Holo) | Shanghai 2024" not in results


def test_collect_batch_items_does_not_cross_match_sticker_quality(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {"steam": _make_steam_source({"Sticker | Liazz (Glitter) | Paris 2023": 44.44})}

    results = aggregator.collect_batch_items(["Sticker | Liazz (Holo) | Shanghai 2024"])

    assert "Sticker | Liazz (Holo) | Shanghai 2024" not in results


def test_collect_batch_items_matches_stattrak_without_trademark_symbol(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {"steam": _make_steam_source({"StatTrak M249 | Hypnosis (Factory New)": 12.5})}

    results = aggregator.collect_batch_items(["StatTrak™ M249 | Hypnosis (Factory New)"])

    _assert_sources(results, "StatTrak™ M249 | Hypnosis (Factory New)", 12.5)


def test_collect_batch_items_matches_knife_without_leading_star(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {"steam": _make_steam_source({"Skeleton Knife | Damascus Steel (Well-Worn)": 88.8})}

    results = aggregator.collect_batch_items(["★ Skeleton Knife | Damascus Steel (Well-Worn)"])

    _assert_sources(results, "★ Skeleton Knife | Damascus Steel (Well-Worn)", 88.8)


def test_collect_batch_items_prefers_exact_starred_knife_match(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {
        "steam": _make_steam_source(
            {
                "Skeleton Knife | Damascus Steel (Well-Worn)": 88.8,
                "★ Skeleton Knife | Damascus Steel (Well-Worn)": 99.9,
            }
        )
    }

    results = aggregator.collect_batch_items(["★ Skeleton Knife | Damascus Steel (Well-Worn)"])

    _assert_sources(results, "★ Skeleton Knife | Damascus Steel (Well-Worn)", 99.9)


def test_collect_batch_items_matches_souvenir_charm_without_charm_word(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {
        "steam": _make_steam_source({"Souvenir | Austin 2025 Highlight | Almost 500 Damage": 17.5})
    }

    results = aggregator.collect_batch_items(["Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage"])

    _assert_sources(results, "Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage", 17.5)


def test_collect_batch_items_does_not_cross_match_souvenir_charm_event(monkeypatch):
    aggregator = CSGOTraderAggregator()
    aggregator._raw_sources = {
        "steam": _make_steam_source({"Souvenir | Paris 2023 Highlight | Almost 500 Damage": 17.5})
    }

    results = aggregator.collect_batch_items(["Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage"])

    assert "Souvenir Charm | Austin 2025 Highlight | Almost 500 Damage" not in results
