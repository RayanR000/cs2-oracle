"""New-item discovery: what the aggregator adds to the catalog, and what it must not."""

import datetime as dt

import pytest
from collectors import new_item_discovery as nid

DAY = dt.date(2026, 11, 20)
NEW = "M4A1-S | Brand New Finish (Factory New)"


def _sources(prices: dict[str, int]) -> dict[str, dict]:
    """`{name: n_sources}` -> raw dumps where `name` is priced in that many sources."""
    shapes = {
        "skinport": lambda p: {"starting_at": p},
        "csfloat": lambda p: {"price": p},
        "csmoney": lambda p: {"price": p},
        "youpin": lambda p: {"price": p},
    }
    out: dict[str, dict] = {s: {} for s in shapes}
    for name, n in prices.items():
        for source in list(shapes)[:n]:
            out[source][name] = shapes[source](12.5)
    return out


def test_baseline_and_seed_ship_and_do_not_overlap():
    baseline, seed = nid._baseline(), nid._seed()

    assert len(baseline) > 40_000
    assert 400 <= len(seed) <= 529
    assert not (baseline & set(seed))
    assert "AK-47 | AUTOEXEC (Field-Tested)" in seed


def test_a_name_priced_in_two_sources_and_unknown_everywhere_is_new():
    assert nid.find_new_items(_sources({NEW: 2}), set(), DAY) == [(NEW, DAY)]


def test_one_priced_source_is_not_enough():
    assert nid.find_new_items(_sources({NEW: 1}), set(), DAY) == []


def test_a_null_priced_ghost_is_not_new():
    raw = {"steam": {NEW: {"last_24h": None, "last_7d": None}}, "csfloat": {NEW: {"price": None}}}

    assert nid.find_new_items(raw, set(), DAY) == []


def test_catalog_names_are_never_reinserted():
    assert nid.find_new_items(_sources({NEW: 3}), {NEW}, DAY) == []


def test_a_baseline_name_is_not_new_even_when_priced():
    old = next(iter(nid._baseline()))

    assert nid.find_new_items(_sources({old: 4}), set(), DAY) == []


def test_seed_names_carry_their_first_archive_day_not_today():
    name, first_day = next(iter(nid._seed().items()))

    assert nid.find_new_items(_sources({name: 2}), set(), DAY) == [(name, first_day)]


def test_universe_rules_apply():
    doppler = "★ Karambit | Doppler (Factory New)"
    phantom = "ak-47-redline-field-tested"

    assert nid.find_new_items(_sources({doppler: 3, phantom: 3, "#CSGO_crate_x": 3}), set(), DAY) == []


@pytest.mark.parametrize(
    "name, kind",
    [
        ("AK-47 | Redline (Field-Tested)", "skin"),
        ("Sticker | s1mple (Gold) | Cologne 2026", "sticker"),
        ("Sealed Graffiti | Lightbulb (Monarch Blue)", "graffiti"),
        ("Cologne 2026 Sticker Capsule", "case"),
        ("Music Kit | Beartooth, Aggressive", "musickit"),
    ],
)
def test_item_type_uses_the_catalog_vocabulary(name, kind):
    assert nid.item_type(name) == kind


# --- writing to items.parquet, the catalog's only store (Phase 4 PR 2b) ---------

OTHER = "USP-S | Another New Finish (Factory New)"
REDLINE = "AK-47 | Redline (Field-Tested)"


def _catalog(ops):
    from db import ops_store

    return {r.item_id: r for r in ops_store.reader(("items",), root=ops)("SELECT * FROM items")}


@pytest.fixture()
def ops(tmp_path):
    """An ops dir holding one established item, with a Postgres-issued id."""
    import pandas as pd
    from db import ops_partitions

    d = tmp_path / "ops"
    ops_partitions.upsert("items", pd.DataFrame([{"id": 1, "item_id": REDLINE, "name": REDLINE}]), root=d)
    return d


def test_discover_writes_young_rows_to_the_partitions_and_is_idempotent(ops):
    from db.ops_partitions import READ_ONLY_ID_BASE

    assert nid.discover_new_items(_sources({NEW: 2}), DAY, ops_root=ops) == 1
    assert nid.discover_new_items(_sources({NEW: 2}), DAY, ops_root=ops) == 0

    row = _catalog(ops)[NEW]
    assert row.id == READ_ONLY_ID_BASE + 1
    assert (row.name, row.type, row.is_backfilled, row.is_trainable) == (NEW, "skin", 0, 0)
    assert row.release_date == dt.datetime(2026, 11, 20)


def test_a_catalog_name_is_never_reinserted(ops):
    assert nid.discover_new_items(_sources({REDLINE: 3}), DAY, ops_root=ops) == 0
    assert len(_catalog(ops)) == 1


def test_the_callers_catalog_is_the_known_set(ops):
    """The pipeline passes the rows it already read; discovery does not re-read them."""
    from types import SimpleNamespace

    known = [SimpleNamespace(id=7, name=NEW, item_id=NEW)]

    assert nid.discover_new_items(_sources({NEW: 2}), DAY, ops_root=ops, catalog=known) == 0


def test_a_flood_of_new_names_inserts_nothing(ops, monkeypatch):
    monkeypatch.setattr(nid, "MAX_NEW_PER_RUN", 2)
    flood = {f"Glock-18 | Finish {i} (Factory New)": 2 for i in range(3)}

    assert nid.discover_new_items(_sources(flood), DAY, ops_root=ops) == 0
    assert len(_catalog(ops)) == 1


def test_a_discovered_item_keeps_its_release_date(ops):
    nid.discover_new_items(_sources({NEW: 2}), DAY, ops_root=ops)

    assert nid.discover_new_items(_sources({NEW: 2}), DAY + dt.timedelta(days=1), ops_root=ops) == 0
    assert _catalog(ops)[NEW].release_date == dt.datetime(2026, 11, 20)


def test_ids_continue_upward_across_nights(ops):
    from db.ops_partitions import READ_ONLY_ID_BASE

    nid.discover_new_items(_sources({NEW: 2}), DAY, ops_root=ops)
    nid.discover_new_items(_sources({NEW: 2, OTHER: 2}), DAY + dt.timedelta(days=1), ops_root=ops)

    cat = _catalog(ops)
    assert (cat[NEW].id, cat[OTHER].id) == (READ_ONLY_ID_BASE + 1, READ_ONLY_ID_BASE + 2)
    assert cat[REDLINE].id == 1


def test_already_discovered_items_do_not_count_against_the_flood_cap(ops, monkeypatch):
    nid.discover_new_items(_sources({NEW: 2, OTHER: 2}), DAY, ops_root=ops)
    monkeypatch.setattr(nid, "MAX_NEW_PER_RUN", 1)
    third = "Glock-18 | Third New Finish (Factory New)"

    assert nid.discover_new_items(_sources({NEW: 2, OTHER: 2, third: 2}), DAY, ops_root=ops) == 1
    assert third in _catalog(ops)


def test_an_empty_ops_dir_starts_ids_above_every_postgres_id(tmp_path):
    from db.ops_partitions import READ_ONLY_ID_BASE

    assert nid.discover_new_items(_sources({NEW: 2}), DAY, ops_root=tmp_path / "empty") == 1
    assert _catalog(tmp_path / "empty")[NEW].id == READ_ONLY_ID_BASE + 1
