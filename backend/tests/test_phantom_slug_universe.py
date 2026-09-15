"""A phantom slug key is a second copy of an item already in the universe.

`items.item_id` is the archive's `item_slug` verbatim. Two writers keyed rows on
something other than the `market_hash_name` every other inserter uses —
`migrate_historical_data.py:345` wrote `slugify(name)` (3,145 rows) and a
since-deleted `real_data_collector.py` wrote `f"steam_{...}"` (4 rows) — and the
aggregator matches its price lookup on `name`, so both copies receive every
day's price. Measured 2026-08-08 over the whole archive: **3,149 phantom keys,
786,408 rows (3.6%)**, and **all 3,149 pair 1:1 onto a correctly-keyed row** —
32,623 of 32,645 same-day same-source pairs are identical to the cent (99.93%).

The defect is not the wasted rows, it is validation. A CV or A/B split that
partitions by item can put `gut-knife-doppler-factory-new` in train and
`★ Gut Knife | Doppler (Factory New)` in test. Those are the same price series,
so the fold reports a score for an item it memorised.

See docs/changelog/2026-08-06-steam-listing-backfill-and-phantom-items.md.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import duckdb
import pandas as pd
import pytest
from models.forecaster import ItemForecaster
from models.item_parser import (
    archive_universe_sql_filter,
    is_phantom_slug,
    phantom_slug_sql_filter,
)

# Real archive keys. The first four are `slugify(market_hash_name)`; the last
# two are the four-row `steam_` form, which holds '_' and '|' and so does NOT
# match the slug regex — both arms are needed.
#
# Deliberately no Doppler name on either list: those leave the universe under
# `phase_collapsed_sql_filter` regardless, so a Doppler pair would pass these
# tests without the phantom rule doing any of the work.
PHANTOM = [
    "ak-47-redline-field-tested",
    "sealed-graffiti-popdog-battle-green",
    "kukri-knife-fade-factory-new",
    "stattrak-usp-s-flashback-field-tested",
    "steam_sticker_|_sico_|_rio_2022",
    "steam_sealed_graffiti_|_popdog_(tiger_orange)",
]

# The correctly-keyed rows. Every phantom above is a copy of one of these.
REAL = [
    "AK-47 | Redline (Field-Tested)",
    "Sealed Graffiti | Popdog (Battle Green)",
    "★ Kukri Knife | Fade (Factory New)",
    "StatTrak™ USP-S | Flashback (Field-Tested)",
    "Sticker | sico | Rio 2022",
    "Sealed Graffiti | Popdog (Tiger Orange)",
    # A single-token name still carries a capital or a space, which is what
    # keeps it clear of the slug regex.
    "Glove Case",
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


def _kept(slugs, where):
    con = duckdb.connect()
    try:
        con.sql("CREATE TABLE t (item_slug VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?)", [(s,) for s in slugs])
        return {r[0] for r in con.sql(f"SELECT item_slug FROM t WHERE {where}").fetchall()}
    finally:
        con.close()


# -- the predicate ---------------------------------------------------------


@pytest.mark.parametrize("slug", PHANTOM)
def test_phantom_keys_are_recognised(slug):
    assert is_phantom_slug(slug)


@pytest.mark.parametrize("slug", REAL)
def test_correctly_keyed_names_are_kept(slug):
    assert not is_phantom_slug(slug)


def test_sql_filter_is_null_safe():
    """A bare NOT regexp over a NULL slug evaluates to NULL, which drops the
    row. The bid exclusion and the phase-collapse rule both carry this guard
    because getting it wrong silently deletes 13 years of prices."""
    con = duckdb.connect()
    try:
        con.sql("CREATE TABLE t (item_slug VARCHAR)")
        con.sql("INSERT INTO t VALUES (NULL)")
        kept = con.sql(f"SELECT count(*) FROM t WHERE {phantom_slug_sql_filter()}").fetchone()[0]
    finally:
        con.close()

    assert kept == 1


def test_predicate_and_sql_filter_agree():
    """One rule, two dialects. A Python check that disagreed with the SQL the
    readers actually run would mean the tests below prove nothing."""
    assert _kept(PHANTOM + REAL, phantom_slug_sql_filter()) == set(REAL)


def test_universe_filter_carries_the_rule():
    """Invariant 2 in backend/AGENTS.md: a loader that globs the archive gets
    every universe rule from this one call. A harness that had to add the
    phantom exclusion separately is a harness that will forget to."""
    assert _kept(PHANTOM + REAL, archive_universe_sql_filter(source_column=None)) == set(REAL)


# -- the training and serving read -----------------------------------------


def test_training_read_drops_the_phantom_and_keeps_the_twin(tmp_path):
    """`_fetch_voted_price_history` is the one archive read behind both
    `train()` and `predict()`. Dropping the phantom there removes the duplicate
    series from the training universe without losing the item: every phantom
    has a correctly-keyed twin, which is why this is a de-duplication and not a
    universe reduction."""
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path / "saved_models"))
    f.archive_dir = _archive(tmp_path, PHANTOM + REAL)
    f._now = lambda: datetime(2026, 7, 20, tzinfo=UTC)

    out = f._fetch_voted_price_history(days_back=30, backfilled_only=False)

    assert set(out["item_id"]) == set(REAL)


def test_voted_cache_is_invalidated():
    """The cache key covers the archive fingerprint and the window but cannot
    see the query, so a surviving pre-change frame would train the next model
    on the duplicated universe."""
    assert ItemForecaster.VOTED_CACHE_VERSION >= 4


def test_opportunities_drop_the_phantom_and_keep_the_twin():
    """The ranked surfaces take each item's *latest* forecast with no date
    bound, so both copies sit there permanently: 3,149 of the 8,691 forecast
    items are phantoms (2026-08-08), i.e. 36% of the served universe is a
    second copy of something already listed.

    The phantom's `name` holds the **real** market_hash_name — that is what
    makes it repairable, and it is also why the phase-collapse check on `name`
    does not catch it. The key is `item_id`."""
    from api.routes import opportunities as opp

    class _Item:
        def __init__(self, id, item_id, name):
            self.id, self.item_id, self.name = id, item_id, name

    items = [
        # The phantom and its twin: same name, different key.
        _Item(1, "ak-47-redline-field-tested", "AK-47 | Redline (Field-Tested)"),
        _Item(2, "AK-47 | Redline (Field-Tested)", "AK-47 | Redline (Field-Tested)"),
        _Item(3, "Glove Case", "Glove Case"),
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


def test_purge_script_shares_the_one_predicate():
    """`purge_phantom_items.py` decides what to delete from production. If its
    notion of a phantom drifted from the reader's, the archive and the model
    would disagree about which items exist."""
    from scripts.archive.purge_phantom_items import is_mangled_key

    assert is_mangled_key is is_phantom_slug
