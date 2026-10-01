"""Add newly released items to the catalog from the aggregator's own dumps.

The monthly CSMarketAPI backfill was the only thing that added items, and that
access is gone (2026-09-30). The csgotrader.app dumps the aggregator already
downloads list every tradeable name, so a release shows up there on its own:
the 2026-07-08 release was 529/529 present while 0/529 were in the catalog.
This compares each run's dumps against the catalog and inserts what is new.

What counts as new:

- **Priced in >= 2 sources** this run (primary price > 0). The dumps carry ~60
  Steam-only names with every price null (``AWP | Doodle Lore``) that Steam's
  own market search does not return; one priced source is not enough evidence.
- **Not in the catalog**, by ``name`` or ``item_id``.
- **Not in the 2026-09-30 baseline** (``data/discovery_baseline_2026-09-30.txt.gz``):
  every name priced anywhere that day, plus never-priced names that ByMykel dates
  before 2026-07 or that carry a pre-2026 event year. Without it the first run
  would insert the ~15K old names the catalog leaves out on purpose. Over the 10
  runs to 2026-09-29 the dumps' name counts moved by +2, so a name that appears
  after the baseline is a release, not a flicker.
- **Not phase-collapsed or a phantom key** -- the same universe rules every
  archive reader applies (``models/item_parser.py``).

``release_date`` is set to the first day the archive priced the item, which is
what `models/serve_universe.py` counts its 60 days from. For a new name that is
this run's snapshot day. The 2026-07-08 release predates this code and has 43+
days of history already, so its true first days ship as a seed file
(``data/discovery_seed_2026-07-08_release.tsv``) and are inserted on first sight.

Inserted rows are ``is_backfilled = 0``: they never join the train universe and
are served at h=3 only (`models/serve_universe.py`).
"""

from __future__ import annotations

import gzip
import logging
from datetime import date, datetime
from functools import cache
from pathlib import Path

from models.item_parser import is_phantom_slug, is_phase_collapsed, parse_item_name
from sqlalchemy import text

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASELINE_PATH = DATA_DIR / "discovery_baseline_2026-09-30.txt.gz"
SEED_PATH = DATA_DIR / "discovery_seed_2026-07-08_release.tsv"

#: A new name needs a positive primary price in at least this many sources.
MIN_PRICED_SOURCES = 2

#: More than this many non-seed names in one run is not a release; it is the
#: dumps changing shape (a new source, a renamed key format). Insert nothing and
#: say so. A Major's sticker drop is a few hundred names.
MAX_NEW_PER_RUN = 1500


@cache
def _baseline() -> frozenset[str]:
    with gzip.open(BASELINE_PATH, "rt", encoding="utf-8") as f:
        return frozenset(line.rstrip("\n") for line in f if line.strip())


@cache
def _seed() -> dict[str, date]:
    out = {}
    for line in SEED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, day = line.split("\t")
        out[name] = date.fromisoformat(day)
    return out


def item_type(name: str) -> str:
    """The catalog's ``type`` vocabulary: skin, sticker, graffiti, case, musickit."""
    p = parse_item_name(name)
    if p["is_sticker"]:
        return "sticker"
    if p["is_graffiti"]:
        return "graffiti"
    if p["is_case"] or p["is_capsule"]:
        return "case"
    if p["is_music_kit"]:
        return "musickit"
    return "skin"


def priced_source_counts(raw_sources: dict[str, dict]) -> dict[str, int]:
    """Per name, how many sources quote a positive primary price this run."""
    from collectors.csgotrader_aggregator import CSGOTraderAggregator

    counts: dict[str, int] = {}
    for source, data in raw_sources.items():
        for name, info in data.items():
            p = CSGOTraderAggregator._extract_primary_price(source, info)
            if p is not None and p > 0:
                counts[name] = counts.get(name, 0) + 1
    return counts


def find_new_items(raw_sources: dict[str, dict], known: set[str], snapshot_day: date) -> list[tuple[str, date]]:
    """``[(name, release_date)]`` to insert. Pure: no DB."""
    baseline, seed = _baseline(), _seed()
    out = []
    for name, n in priced_source_counts(raw_sources).items():
        if n < MIN_PRICED_SOURCES or name in known or name.startswith("#"):
            continue
        if name in seed:
            out.append((name, seed[name]))
        elif name not in baseline and not is_phase_collapsed(name) and not is_phantom_slug(name):
            out.append((name, snapshot_day))
    return sorted(out)


def discover_new_items(session, raw_sources: dict[str, dict], snapshot_day: date) -> int:
    """Insert newly released items. Returns how many were inserted."""
    if not raw_sources:
        return 0
    known: set[str] = set()
    for r in session.execute(text("SELECT name, item_id FROM items")).fetchall():
        known.add(r.name)
        known.add(r.item_id)
    new = find_new_items(raw_sources, known, snapshot_day)
    seed = _seed()
    n_fresh = sum(1 for name, _ in new if name not in seed)
    if n_fresh > MAX_NEW_PER_RUN:
        logger.error(
            "New-item discovery found %s new names in one run (cap %s) -- the dumps changed "
            "shape rather than a release landing. Inserted nothing. Sample: %s",
            n_fresh,
            MAX_NEW_PER_RUN,
            [n for n, _ in new if n not in seed][:10],
        )
        return 0
    if not new:
        return 0
    now = datetime.utcnow()
    session.execute(
        text(
            "INSERT INTO items (item_id, name, type, release_date, created_at, updated_at, "
            "is_backfilled, is_trainable) "
            "VALUES (:item_id, :name, :type, :release_date, :now, :now, 0, 0)"
        ),
        [
            {
                "item_id": name,
                "name": name,
                "type": item_type(name),
                "release_date": datetime(d.year, d.month, d.day),
                "now": now,
            }
            for name, d in new
        ],
    )
    session.commit()
    logger.info(
        "New-item discovery: inserted %s items (%s from the 2026-07-08 seed, %s new this run). Sample: %s",
        len(new),
        len(new) - n_fresh,
        n_fresh,
        [n for n, _ in new][:10],
    )
    return len(new)
