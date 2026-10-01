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
