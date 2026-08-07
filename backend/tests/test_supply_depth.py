"""Tests for the supply-depth collector.

These cover the parsing and aggregation layer, which is pure — no network. The
failure mode being guarded against throughout is *silent* wrongness: a payload
shape change, a string-typed count, or a dead feed presenting as an empty one.
This repo has shipped that bug twice (the Steam supply scraper's 16 green days
storing nothing, the Reddit collector's 403s), so the tests assert that bad input
raises or drops rather than zero-filling.
"""
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from collectors.supply_depth import (
    DEDUP_KEYS,
    SUPPLY_COLUMNS,
    SupplyFeedError,
    aggregate_lis_skins,
    collect,
    parse_bitskins,
    parse_market_csgo,
    parse_skinport,
    parse_waxpeer,
    supply_parquet_path,
    write_supply_rows,
)

DAY = date(2026, 8, 6)
NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


# ── Scalar feed parsers ───────────────────────────────────────────────────────

def test_skinport_parses_quantity():
    rows = parse_skinport(
        [{"market_hash_name": "AK-47 | Redline (Field-Tested)", "quantity": 412, "min_price": 7.62}],
        DAY, NOW,
    )
    assert list(rows.columns) == SUPPLY_COLUMNS
    assert rows.iloc[0]["listing_count"] == 412
    assert rows.iloc[0]["source"] == "skinport"
    assert rows.iloc[0]["min_ask"] == pytest.approx(7.62)


def test_market_csgo_coerces_string_volume():
    """market.csgo.com ships `volume` as a string.

    Treating it as unparseable would silently drop the entire feed while every
    other column still looked healthy.
    """
    rows = parse_market_csgo(
        {"items": [{"market_hash_name": "Kilowatt Case", "volume": "1873", "price": "0.42"}]},
        DAY, NOW,
    )
    assert rows.iloc[0]["listing_count"] == 1873
    assert rows.iloc[0]["min_ask"] == pytest.approx(0.42)


def test_waxpeer_normalises_millicent_price():
    """Waxpeer quotes `min` in thousandths; `min_ask` must mean USD everywhere."""
    rows = parse_waxpeer(
        {"items": [{"name": "Kilowatt Case", "count": 900, "min": 420}]}, DAY, NOW
    )
    assert rows.iloc[0]["min_ask"] == pytest.approx(0.42)


def test_bitskins_normalises_millicent_price():
    rows = parse_bitskins(
        {"list": [{"name": "Kilowatt Case", "quantity": 55, "price_min": 420}]}, DAY, NOW
    )
    assert rows.iloc[0]["min_ask"] == pytest.approx(0.42)
    assert rows.iloc[0]["listing_count"] == 55


def test_unparseable_count_is_dropped_not_zero_filled():
    """A count of 0 is a real observation (nothing for sale).

    Coercing a parse failure into 0 would fabricate the very signal this
    collector exists to measure, so bad rows must disappear instead.
    """
    rows = parse_skinport(
        [
            {"market_hash_name": "Good", "quantity": 5},
            {"market_hash_name": "Bad", "quantity": "not-a-number"},
            {"market_hash_name": "Zero", "quantity": 0},
        ],
        DAY, NOW,
    )
    slugs = set(rows["item_slug"])
    assert slugs == {"Good", "Zero"}
    assert rows.set_index("item_slug").loc["Zero", "listing_count"] == 0


def test_duplicate_names_keep_deepest():
    """Feeds ship duplicate market_hash_names; a dupe must not halve a count."""
    rows = parse_skinport(
        [
            {"market_hash_name": "Dupe", "quantity": 10},
            {"market_hash_name": "Dupe", "quantity": 90},
        ],
        DAY, NOW,
    )
    assert len(rows) == 1
    assert rows.iloc[0]["listing_count"] == 90


def test_missing_slug_is_skipped():
    rows = parse_skinport([{"quantity": 5}, {"market_hash_name": "", "quantity": 7}], DAY, NOW)
    assert rows.empty


# ── lis-skins ladder aggregation ──────────────────────────────────────────────

def _listing(name, price, created_at, lid):
    return {"name": name, "price": price, "created_at": created_at, "id": lid}


def test_ladder_aggregates_quantiles_and_depth():
    listings = [_listing("Item", p, "2026-08-01T00:00:00Z", i) for i, p in enumerate([10, 10.2, 10.4, 11, 20])]
    rows = aggregate_lis_skins(listings, DAY, NOW)
    r = rows.iloc[0]

    assert r["listing_count"] == 5
    assert r["min_ask"] == pytest.approx(10.0)
    assert r["median_ask"] == pytest.approx(10.4)
    # Anchor is p05 = 10.04; within 5% is <= 10.542 -> 10, 10.2, 10.4
    assert r["depth_5pct"] == 3
    # within 10% is <= 11.044 -> adds 11
    assert r["depth_10pct"] == 4


def test_depth_anchor_resists_a_single_mispriced_listing():
    """358 items show a lowest ask >10x the archive price (worst case 1,154x).

    Depth measured from the raw minimum would let one bad listing redefine
    "near the bottom of the book"; the trimmed p05 anchor must not move much.
    """
    normal = [_listing("Item", 10.0 + i * 0.01, "2026-08-01T00:00:00Z", i) for i in range(100)]
    with_outlier = normal + [_listing("Item", 0.001, "2026-08-01T00:00:00Z", 999)]

    base = aggregate_lis_skins(normal, DAY, NOW).iloc[0]
    poisoned = aggregate_lis_skins(with_outlier, DAY, NOW).iloc[0]

    assert poisoned["min_ask"] == pytest.approx(0.001)      # raw min IS moved
    assert poisoned["depth_5pct"] >= base["depth_5pct"] * 0.5  # depth is not destroyed


def test_age_and_inflow_from_created_at():
    listings = [
        _listing("Item", 10, "2026-08-06T06:00:00Z", 1),   # <1 day old
        _listing("Item", 11, "2026-08-01T00:00:00Z", 2),   # ~6 days
        _listing("Item", 12, "2026-07-07T00:00:00Z", 3),   # ~31 days
    ]
    r = aggregate_lis_skins(listings, DAY, NOW).iloc[0]
    assert r["inflow_24h"] == 1
    assert r["age_median_days"] == pytest.approx(6.0, abs=0.5)
    assert r["age_p90_days"] > r["age_median_days"]


def test_listing_id_digest_is_order_independent():
    """The digest exists to test whether `created_at` moves for persisting
    listings. It must depend on the id set, not on payload ordering."""
    a = [_listing("Item", 10, "2026-08-01T00:00:00Z", 1), _listing("Item", 11, "2026-08-01T00:00:00Z", 2)]
    b = list(reversed(a))
    assert (
        aggregate_lis_skins(a, DAY, NOW).iloc[0]["listing_id_digest"]
        == aggregate_lis_skins(b, DAY, NOW).iloc[0]["listing_id_digest"]
    )


def test_listing_created_after_snapshot_is_not_aged_zero():
    """A negative age means the timestamp disagrees with the snapshot boundary.

    Clipping to 0 would fabricate a real-looking value at a meaningful edge, the
    same mistake the forecaster's item-age handling explicitly avoids.
    """
    listings = [
        _listing("Item", 10, "2026-08-09T00:00:00Z", 1),   # after the day ends
        _listing("Item", 11, "2026-08-01T00:00:00Z", 2),
    ]
    r = aggregate_lis_skins(listings, DAY, NOW).iloc[0]
    assert r["listing_count"] == 2          # still counted as supply
    assert r["age_median_days"] == pytest.approx(6.0, abs=0.5)  # but not aged


def test_ladder_skips_items_with_no_usable_price():
    rows = aggregate_lis_skins(
        [_listing("NoPrice", None, "2026-08-01T00:00:00Z", 1)], DAY, NOW
    )
    assert rows.empty


# ── Persistence ───────────────────────────────────────────────────────────────

def test_write_is_idempotent_per_day_and_source(tmp_path):
    rows = parse_skinport([{"market_hash_name": "Item", "quantity": 5}], DAY, NOW)
    write_supply_rows(rows, tmp_path, DAY)

    updated = parse_skinport([{"market_hash_name": "Item", "quantity": 9}], DAY, NOW)
    write_supply_rows(updated, tmp_path, DAY)

    stored = pd.read_parquet(supply_parquet_path(tmp_path, DAY))
    assert len(stored) == 1, "re-running a day must replace, not duplicate"
    assert stored.iloc[0]["listing_count"] == 9, "the later run must win"


def test_write_preserves_other_sources_same_day(tmp_path):
    write_supply_rows(parse_skinport([{"market_hash_name": "Item", "quantity": 5}], DAY, NOW), tmp_path, DAY)
    write_supply_rows(
        parse_waxpeer({"items": [{"name": "Item", "count": 7}]}, DAY, NOW), tmp_path, DAY
    )
    stored = pd.read_parquet(supply_parquet_path(tmp_path, DAY))
    assert set(stored["source"]) == {"skinport", "waxpeer"}
    assert len(stored) == 2


def test_write_partitions_by_month(tmp_path):
    assert supply_parquet_path(tmp_path, date(2026, 8, 6)).name == "supply-2026-08.parquet"
    assert supply_parquet_path(tmp_path, date(2026, 9, 1)).name == "supply-2026-09.parquet"


# ── The zero-row guard contract ───────────────────────────────────────────────

def test_collect_raises_when_every_feed_fails(tmp_path, monkeypatch):
    """All feeds down is indistinguishable from a network block.

    `scripts/run_task.py`'s guard is the only thing between a dead collector and
    a green badge, so this must never return a success dict.
    """
    import collectors.supply_depth as sd

    monkeypatch.setattr(
        sd, "fetch_feed",
        lambda feed, day, now, session=None: sd.FeedResult(feed.source, error="boom"),
    )
    monkeypatch.setattr(
        sd, "fetch_lis_skins",
        lambda day, now, session=None: sd.FeedResult("lis_skins", error="boom"),
    )

    with pytest.raises(SupplyFeedError, match="all 5 supply feeds failed"):
        collect(tmp_path, snapshot_day=DAY)


def test_collect_survives_one_dead_feed_but_reports_it(tmp_path, monkeypatch):
    """Losing one marketplace must not cost the day, but must stay visible."""
    import collectors.supply_depth as sd

    def fake_fetch(feed, day, now, session=None):
        if feed.source == "bitskins":
            return sd.FeedResult("bitskins", error="HTTP 503")
        return sd.FeedResult(
            feed.source,
            rows=parse_skinport([{"market_hash_name": "Item", "quantity": 3}], day, now).assign(
                source=feed.source
            ),
            raw_items=1,
        )

    monkeypatch.setattr(sd, "fetch_feed", fake_fetch)
    monkeypatch.setattr(
        sd, "fetch_lis_skins",
        lambda day, now, session=None: sd.FeedResult("lis_skins", error="skipped"),
    )

    summary = collect(tmp_path, snapshot_day=DAY)
    assert summary["status"] == "success"
    assert summary["supply_rows"] > 0
    assert "bitskins" in summary["feeds_failed"]
    assert "bitskins" not in summary["feeds_ok"]


def test_snapshot_day_matches_the_aggregator_not_the_wall_clock(tmp_path, monkeypatch):
    """A supply row is only useful joined to the price row from the same run.

    The price side labels its day from the CSGOTrader dump boundary (~21:40 UTC),
    so a run just after midnight stamps the *previous* day. If supply used the
    wall clock it would stamp D+1 against prices D and the join would miss every
    item — silently, since a missing join reads as "no supply data for this item"
    rather than as an error.
    """
    import collectors.supply_depth as sd

    monkeypatch.setenv("AGGREGATOR_SNAPSHOT_DATE", "2026-08-06")
    monkeypatch.setattr(
        sd, "fetch_feed",
        lambda feed, day, now, session=None: sd.FeedResult(
            feed.source,
            rows=parse_skinport([{"market_hash_name": "I", "quantity": 1}], day, now).assign(
                source=feed.source
            ),
        ),
    )
    monkeypatch.setattr(
        sd, "fetch_lis_skins",
        lambda day, now, session=None: sd.FeedResult("lis_skins", error="skipped"),
    )

    summary = collect(tmp_path, dry_run=True)
    assert summary["snapshot_day"] == "2026-08-06"


def test_collect_reports_row_count_field_for_the_guard(tmp_path, monkeypatch):
    """`supply_rows` is the name registered in run_task.ROW_COUNT_FIELDS."""
    import collectors.supply_depth as sd
    from scripts.run_task import ROW_COUNT_FIELDS

    assert "supply_rows" in ROW_COUNT_FIELDS

    monkeypatch.setattr(
        sd, "fetch_feed",
        lambda feed, day, now, session=None: sd.FeedResult(
            feed.source,
            rows=parse_skinport([{"market_hash_name": "I", "quantity": 1}], day, now).assign(
                source=feed.source
            ),
        ),
    )
    monkeypatch.setattr(
        sd, "fetch_lis_skins",
        lambda day, now, session=None: sd.FeedResult("lis_skins", error="skipped"),
    )
    summary = collect(tmp_path, snapshot_day=DAY)
    assert summary["supply_rows"] == 4
