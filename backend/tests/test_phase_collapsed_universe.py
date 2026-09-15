"""A name that prices several assets at once must leave the item universe.

`market_hash_name` encodes weapon + finish + wear + StatTrak/Souvenir, so every
Doppler and Gamma Doppler name collapses its phases (Ruby / Sapphire / Black
Pearl / Emerald / P1-P4) into one series. Measured 2026-08-07 against the BUFF
dump (46,041 items): 29 base names cover 181 distinct `paint_index` assets, the
median max/min phase ratio *within one name* is 3.25x, 87.3% carry >2x internal
dispersion, and the quoted headline is the cheapest phase 95.5% of the time.

That last number is the defect: the series steps whenever which phase is
cheapest changes, which is a level shift with no asset repricing — a return the
model is asked to predict and a label the backtest scores, both fabricated by
composition. See docs/research/2026-08-07-cs2-forecasting-research.md 25.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import duckdb
import pandas as pd
import pytest
from models.forecaster import (
    PHASE_COLLAPSED_SLUG_PATTERNS,
    ItemForecaster,
    is_phase_collapsed,
    phase_collapsed_sql_filter,
)
from models.item_parser import is_phase_collapsed as _canonical_predicate

# Real archive keys. `item_slug` holds the market_hash_name for almost every
# row, plus a handful of lowercase-hyphenated duplicate keys for the same item
# (six of the Doppler names have one) — both spellings have to go.
COLLAPSED = [
    "★ Gut Knife | Doppler (Factory New)",
    "★ Huntsman Knife | Gamma Doppler (Factory New)",
    "★ StatTrak™ M9 Bayonet | Doppler (Minimal Wear)",  # $1,261-$29,685, one name
    "Glock-18 | Gamma Doppler (Minimal Wear)",
    "gut-knife-doppler-factory-new",  # duplicate slug key for the first entry
]

# Single-asset names. The last two are the measured false positives: two
# stickers borrow the word without collapsing anything, and they are 396 of the
# 47,081 rows the bare pattern matches.
SINGLE_ASSET = [
    "AK-47 | Redline (Field-Tested)",
    "★ Karambit | Fade (Factory New)",
    "★ StatTrak™ M9 Bayonet | Marble Fade (Factory New)",
    "Sticker | Doppler Poison Frog (Foil)",
    "Sticker Slab | Doppler Poison Frog (Foil)",
]


def _archive(tmp_path, slugs, days=3):
    """A one-file price archive holding `days` days for each slug."""
    archive = tmp_path / "price-archive"
    archive.mkdir()
    start = date(2026, 7, 10)
    rows = []
    for slug in slugs:
        for i in range(days):
            rows.append((slug, start + timedelta(days=i), 100.0 + i, 1, "aggregator_csfloat"))
    pd.DataFrame(
        {
            "item_slug": [r[0] for r in rows],
            "day": pd.to_datetime([r[1] for r in rows]),
            "mean_price": [r[2] for r in rows],
            "volume": [r[3] for r in rows],
            "source": [r[4] for r in rows],
        }
    ).to_parquet(archive / "prices-2026.parquet")
    return archive


# -- the predicate ---------------------------------------------------------


@pytest.mark.parametrize("slug", COLLAPSED)
def test_phase_collapsed_names_are_recognised(slug):
    assert is_phase_collapsed(slug)


@pytest.mark.parametrize("slug", SINGLE_ASSET)
def test_single_asset_names_are_kept(slug):
    assert not is_phase_collapsed(slug)


def test_forecaster_reexports_the_one_predicate():
    """The rule lives in item_parser so the API can use it without importing
    LightGBM; forecaster re-exports it so archive readers keep taking their
    universe rules from one module."""
    assert is_phase_collapsed is _canonical_predicate


def test_pattern_list_is_not_empty():
    """Guards against the constant being emptied and every test below passing
    for the wrong reason — the bid exclusion needed the same guard."""
    assert "doppler" in PHASE_COLLAPSED_SLUG_PATTERNS


def test_sql_filter_is_null_safe(tmp_path):
    """A bare NOT LIKE over a NULL slug evaluates to NULL, which drops the row.
    `source IS NULL` selects 13 years of the archive here; the same class of
    mistake on the key would delete rows nobody asked to delete."""
    con = duckdb.connect()
    try:
        con.sql("CREATE TABLE t (item_slug VARCHAR)")
        con.sql("INSERT INTO t VALUES (NULL)")
        kept = con.sql(f"SELECT count(*) FROM t WHERE {phase_collapsed_sql_filter()}").fetchone()[0]
    finally:
        con.close()

    assert kept == 1


def test_predicate_and_sql_filter_agree(tmp_path):
    """One rule, two dialects: a Python-side check that disagreed with the SQL
    the readers actually run would mean the tests below prove nothing about
    production."""
    con = duckdb.connect()
    try:
        con.sql("CREATE TABLE t (item_slug VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?)", [(s,) for s in COLLAPSED + SINGLE_ASSET])
        kept = {r[0] for r in con.sql(f"SELECT item_slug FROM t WHERE {phase_collapsed_sql_filter()}").fetchall()}
    finally:
        con.close()

    assert kept == set(SINGLE_ASSET)


# -- the training and serving read -----------------------------------------


def test_training_read_drops_them(tmp_path):
    """`_fetch_voted_price_history` is the one archive read behind both
    `train()` and `predict()`, so filtering there removes the names from the
    training universe *and* stops the product forecasting them."""
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path / "saved_models"))
    f.archive_dir = _archive(tmp_path, COLLAPSED + SINGLE_ASSET)
    f._now = lambda: datetime(2026, 7, 20, tzinfo=UTC)

    out = f._fetch_voted_price_history(days_back=30, backfilled_only=False)

    assert set(out["item_id"]) == set(SINGLE_ASSET)


def test_voted_cache_is_invalidated(tmp_path):
    """The cache key covers the archive and the window but cannot see the
    query, so a surviving pre-change frame would train the next model on the
    old universe."""
    assert ItemForecaster.VOTED_CACHE_VERSION >= 3


# -- the published gate's own loaders --------------------------------------


def test_gate_price_loader_drops_them(tmp_path, monkeypatch):
    """`walkforward_backtest` never calls `fetch_price_history`; it globs the
    archive itself, which is why the bid exclusion needed a second fix here."""
    from scripts.archive import walkforward_backtest as wf

    archive = _archive(tmp_path, COLLAPSED + SINGLE_ASSET)
    monkeypatch.setattr(wf, "ARCHIVE_DIR", archive)

    con = duckdb.connect()
    try:
        out = wf._load_all_prices(con, [(s,) for s in COLLAPSED + SINGLE_ASSET])
    finally:
        con.close()

    assert set(out["item_id"]) == set(SINGLE_ASSET)


def test_gate_item_selection_drops_them(tmp_path, monkeypatch):
    """Selection has to agree with the price loader, or the `max_items` budget
    is spent on items that then come back with no rows."""
    from scripts.archive import walkforward_backtest as wf

    archive = _archive(tmp_path, COLLAPSED + SINGLE_ASSET, days=120)
    monkeypatch.setattr(wf, "ARCHIVE_DIR", archive)

    con = duckdb.connect()
    try:
        rows = wf._load_parquet_items(con, backfilled_only=False)
    finally:
        con.close()

    assert {r[0] for r in rows} == set(SINGLE_ASSET)


# -- the ranked product surfaces -------------------------------------------


def test_opportunities_drop_them():
    """`opportunities` takes each item's *latest* forecast with no date bound,
    so an item that stops being forecast keeps its final row on the ranked
    lists forever. The six real names are $200-500 knives — they rank."""
    from api.routes import opportunities as opp

    class _Item:
        # `item_id` is a non-nullable column and `_load_items` also screens the
        # phantom duplicate keys on it, so the stub has to carry it. Every
        # inserter but the two phantom writers sets it to the market_hash_name.
        def __init__(self, id, name):
            self.id, self.name, self.item_id = id, name, name

    items = [
        _Item(1, "★ Gut Knife | Doppler (Factory New)"),
        _Item(2, "AK-47 | Redline (Field-Tested)"),
        _Item(3, "Sticker | Doppler Poison Frog (Foil)"),
    ]

    class _Query:
        def filter(self, *a, **k):
            return self

        def all(self):
            return items

    class _DB:
        def query(self, *a, **k):
            return _Query()

    loaded = opp._load_items([1, 2, 3], _DB())

    assert set(loaded) == {2, 3}
