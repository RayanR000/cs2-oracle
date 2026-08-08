"""`load_targets` in scripts/backfill_steam_listing_history.py.

The bug this pins: the loader globbed the archive raw, so DuckDB narrowed the
whole read to `prices-2013.parquet`'s schema. When the 2026-08-08 normalisation
renamed the price column, the loader's `HAVING MAX(median_price)` kept working
against a column that was no longer there — right up until `--min-price` was
next passed, and then it raised `BinderException` on the script's own
documented usage line. Nothing failed in between.

So the tests are behavioural on both halves: `--min-price` must filter, and the
pre-2026 file must not decide the schema.
"""

from datetime import date

import pandas as pd
import pytest

from scripts import backfill_steam_listing_history as mod


@pytest.fixture
def archive(tmp_path):
    """Two files with different schemas — the shape the glob read hid.

    `prices-2013.parquet` predates `source` and sorts first, so it is the file
    a bare `read_parquet('prices-*.parquet')` takes its schema from. It also
    supplies the `is_backfilled` gate: an item with a pre-2026 row is one
    training already has, and is not a backfill target.
    """
    pd.DataFrame([
        {"item_slug": "AK-47 | Redline (Field-Tested)",
         "day": date(2013, 8, 14), "mean_price": 5.0, "volume": 1},
    ]).to_parquet(tmp_path / "prices-2013.parquet", index=False)

    pd.DataFrame([
        # gated: has pre-2026 history, so training is not dropping it
        {"item_slug": "AK-47 | Redline (Field-Tested)", "day": date(2026, 8, 7),
         "source": "aggregator_sync", "mean_price": 6.0, "volume": 1},
        # a target, above any sane floor
        {"item_slug": "Glock-18 | Fade (Factory New)", "day": date(2026, 8, 7),
         "source": "aggregator_sync", "mean_price": 400.0, "volume": 1},
        # a target, below a $1 floor
        {"item_slug": "P250 | Sand Dune (Well-Worn)", "day": date(2026, 8, 7),
         "source": "aggregator_sync", "mean_price": 0.10, "volume": 1},
        # phase-collapsed: one name over every Doppler phase, out of universe
        {"item_slug": "★ StatTrak™ M9 Bayonet | Doppler (Minimal Wear)",
         "day": date(2026, 8, 7), "source": "aggregator_sync",
         "mean_price": 1200.0, "volume": 1},
        # priced only by a BUFF bid, which does not vote
        {"item_slug": "MP9 | Bulldozer (Factory New)", "day": date(2026, 8, 7),
         "source": "aggregator_buff163_buy", "mean_price": 50.0, "volume": 1},
        # mangled slug key — a duplicate of an already-gated item
        {"item_slug": "sealed-graffiti-popdog-battle-green",
         "day": date(2026, 8, 7), "source": "aggregator_sync",
         "mean_price": 5.0, "volume": 1},
    ]).to_parquet(tmp_path / "prices-2026-08.parquet", index=False)
    return tmp_path


@pytest.fixture(autouse=True)
def _point_at_fixture(archive, monkeypatch):
    monkeypatch.setattr(mod, "PRICE_ARCHIVE_DIR", archive)


def test_min_price_filters_rather_than_raising():
    """The documented `--min-price 1.0` usage line."""
    assert mod.load_targets(1.0, None) == ["Glock-18 | Fade (Factory New)"]


def test_no_floor_keeps_the_cheap_item():
    assert mod.load_targets(None, None) == [
        "Glock-18 | Fade (Factory New)",
        "P250 | Sand Dune (Well-Worn)",
    ]


def test_the_2013_schema_does_not_narrow_the_read():
    """`source` is absent from the first file and present in the second.

    Under the glob, `source` vanished from the relation entirely, so the
    universe filter had no column to apply and the bid-priced item counted.
    """
    assert "MP9 | Bulldozer (Factory New)" not in mod.load_targets(None, None)


def test_phase_collapsed_and_mangled_names_are_not_targets():
    targets = mod.load_targets(None, None)
    assert "★ StatTrak™ M9 Bayonet | Doppler (Minimal Wear)" not in targets
    assert "sealed-graffiti-popdog-battle-green" not in targets


def test_the_gate_excludes_items_with_pre_2026_history():
    assert "AK-47 | Redline (Field-Tested)" not in mod.load_targets(None, None)
