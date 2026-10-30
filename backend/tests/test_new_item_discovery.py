"""New-item discovery: what the aggregator adds to the catalog, and what it must not."""

import datetime as dt

import pytest
from collectors import new_item_discovery as nid
from database import Base, Item
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

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


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Item(item_id="AK-47 | Redline (Field-Tested)", name="AK-47 | Redline (Field-Tested)", type="skin"))
    s.commit()
    yield s
    engine.dispose()


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


def test_discover_inserts_young_rows_and_is_idempotent(session):
    assert nid.discover_new_items(session, _sources({NEW: 2}), DAY) == 1
    assert nid.discover_new_items(session, _sources({NEW: 2}), DAY) == 0

    row = session.query(Item).filter_by(item_id=NEW).one()
    assert (row.name, row.type, row.is_backfilled, row.is_trainable) == (NEW, "skin", 0, 0)
    assert row.release_date == dt.datetime(2026, 11, 20)


def test_a_flood_of_new_names_inserts_nothing(session, monkeypatch):
    monkeypatch.setattr(nid, "MAX_NEW_PER_RUN", 2)
    flood = {f"Glock-18 | Finish {i} (Factory New)": 2 for i in range(3)}

    assert nid.discover_new_items(session, _sources(flood), DAY) == 0
    assert session.query(Item).count() == 1


# --- a Postgres that cannot take the insert (Phase 2 PR 5) -------------------

OTHER = "USP-S | Another New Finish (Factory New)"


class _PgError(Exception):
    def __init__(self, msg, pgcode=None):
        super().__init__(msg)
        self.pgcode = pgcode


def _refuse_inserts(session, monkeypatch):
    from db.read_only import READ_ONLY_SQLSTATE
    from sqlalchemy.exc import InternalError

    real = session.execute

    def execute(stmt, *a, **k):
        if "INSERT" in str(stmt):
            raise InternalError("INSERT ...", {}, _PgError("read-only transaction", READ_ONLY_SQLSTATE))
        return real(stmt, *a, **k)

    monkeypatch.setattr(session, "execute", execute)


def _catalog(ops):
    from db import ops_store

    return {r.item_id: r for r in ops_store.reader(("items",), root=ops)("SELECT * FROM items")}


@pytest.fixture()
def ops(tmp_path):
    """An ops dir holding the session's one established item, as the nightly sync leaves it."""
    import pandas as pd
    from db import ops_partitions

    d = tmp_path / "ops"
    ops_partitions.upsert(
        "items",
        pd.DataFrame(
            [{"id": 1, "item_id": "AK-47 | Redline (Field-Tested)", "name": "AK-47 | Redline (Field-Tested)"}]
        ),
        root=d,
    )
    return d


def test_a_read_only_postgres_lands_new_items_in_the_partitions(session, ops, monkeypatch, capsys):
    from db.ops_partitions import READ_ONLY_ID_BASE

    _refuse_inserts(session, monkeypatch)
    assert nid.discover_new_items(session, _sources({NEW: 2}), DAY, ops_root=ops) == 1

    row = _catalog(ops)[NEW]
    assert row.id == READ_ONLY_ID_BASE + 1
    assert (row.name, row.type, row.is_backfilled, row.is_trainable) == (NEW, "skin", 0, 0)
    assert row.release_date == dt.datetime(2026, 11, 20)
    assert "::warning" in capsys.readouterr().out
    assert session.query(Item).count() == 1


def test_a_partition_only_item_is_not_rediscovered_and_keeps_its_release_date(session, ops, monkeypatch):
    _refuse_inserts(session, monkeypatch)
    nid.discover_new_items(session, _sources({NEW: 2}), DAY, ops_root=ops)
    later = DAY + dt.timedelta(days=1)

    assert nid.discover_new_items(session, _sources({NEW: 2}), later, ops_root=ops) == 0
    assert _catalog(ops)[NEW].release_date == dt.datetime(2026, 11, 20)


def test_partition_ids_continue_upward_across_nights(session, ops, monkeypatch):
    from db.ops_partitions import READ_ONLY_ID_BASE

    _refuse_inserts(session, monkeypatch)
    nid.discover_new_items(session, _sources({NEW: 2}), DAY, ops_root=ops)
    nid.discover_new_items(session, _sources({NEW: 2, OTHER: 2}), DAY + dt.timedelta(days=1), ops_root=ops)

    cat = _catalog(ops)
    assert (cat[NEW].id, cat[OTHER].id) == (READ_ONLY_ID_BASE + 1, READ_ONLY_ID_BASE + 2)
    assert cat["AK-47 | Redline (Field-Tested)"].id == 1


def test_partition_only_items_do_not_count_against_the_flood_cap(session, ops, monkeypatch):
    _refuse_inserts(session, monkeypatch)
    nid.discover_new_items(session, _sources({NEW: 2, OTHER: 2}), DAY, ops_root=ops)
    monkeypatch.setattr(nid, "MAX_NEW_PER_RUN", 1)
    third = "Glock-18 | Third New Finish (Factory New)"

    assert nid.discover_new_items(session, _sources({NEW: 2, OTHER: 2, third: 2}), DAY, ops_root=ops) == 1
    assert third in _catalog(ops)


def test_a_writable_postgres_leaves_the_partitions_to_the_nightly_sync(session, ops):
    assert nid.discover_new_items(session, _sources({NEW: 2}), DAY, ops_root=ops) == 1
    assert session.query(Item).filter_by(item_id=NEW).count() == 1
    assert NEW not in _catalog(ops)


def test_without_an_ops_dir_a_read_only_postgres_inserts_nothing(session, monkeypatch):
    _refuse_inserts(session, monkeypatch)
    assert nid.discover_new_items(session, _sources({NEW: 2}), DAY) == 0


def test_the_nightly_items_sync_keeps_partition_only_items(session, ops, monkeypatch):
    from scripts.ops_sync_from_postgres import sync

    with monkeypatch.context() as m:
        _refuse_inserts(session, m)
        nid.discover_new_items(session, _sources({NEW: 2}), DAY, ops_root=ops)
    sync(session, [], tables=["items"], apply=True, root=ops)

    cat = _catalog(ops)
    assert NEW in cat and "AK-47 | Redline (Field-Tested)" in cat
