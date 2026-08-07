"""Purge the 3,149 phantom `items` rows and the archive rows they generate.

## What the phantoms are

Two writers put malformed values in `items.item_id`, where every other inserter
uses `item_id = market_hash_name`:

  * `migrate_historical_data.py:345` -- `"item_id": slugify(mn)`, 3,145 rows,
    e.g. `sealed-graffiti-popdog-battle-green`.
  * a since-deleted `real_data_collector.py` -- `f"steam_{...}"`, 4 rows.

Crucially both wrote the **real** `market_hash_name` into `items.name`
(`migrate_historical_data.py:346`). That is what makes this repairable: the
phantom's identity is not lost, so it can be matched to the correctly-keyed row
by name without inverting the lossy `slugify`.

## Why it leaks daily

The aggregator keys its external price lookup on `name`, not `item_id`
(`collectors/pipeline.py:124-128`), then writes one `price_history` row per
matched item (`:162-169`). Two rows share a name, so both receive every day's
price. The archive's `item_slug` is `items.item_id` verbatim
(`append_to_parquet.py:33`, `export_daily_snapshot.py:24`,
`pipeline.py:255-280`), so the phantoms surface there as separate items --
~9% of recent archive writes. Fixing the writers changes nothing; the bad rows
already exist and are re-read every run.

## Safety invariant

A phantom is deleted only once its **keeper** -- the correctly-keyed row holding
the same name -- has been positively identified. Anything unpaired is reported
and left in place, because for an unpaired row the phantom is the only copy.

`--dry-run` is the DEFAULT here, inverting the repo convention, because this
mutates production and is not reversible.
"""
import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("purge_phantom_items")

# Same predicate the Steam backfill uses to exclude these keys from its target
# list (`backfill_steam_listing_history.py:195-199`). The slug regex does NOT
# match the `steam_` form, which holds '_' and '|', so both arms are needed.
_SLUG_KEY = re.compile(r"^[a-z0-9][a-z0-9\-]*$")


def is_mangled_key(item_id: str) -> bool:
    return bool(_SLUG_KEY.match(item_id)) or item_id.startswith("steam_")


def slugify(name: str) -> str:
    """Verbatim copy of `migrate_historical_data.py:56-59`.

    Kept here as a cross-check only: it is lossy (case, and every run of
    non-alphanumerics collapses to one '-'), so it is not used to recover names.
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# Every table with a FOREIGN KEY onto items.id, ordered so a child is always
# deleted before its parent. There is no ON DELETE anywhere in the schema
# (verified across database.py and all of migrations/versions/), so Postgres
# applies NO ACTION and a bare DELETE FROM items raises while any child lives.
# forecast_outcomes also FKs item_forecasts.id, so it must precede that table.
CHILD_TABLES: Tuple[str, ...] = (
    "forecast_outcomes",
    "item_forecasts",
    "price_history",
    "supply_snapshots",
    "social_mentions",
    "event_impacts",
    "event_patterns",
    "event_correlations",
)


@dataclass
class Phantom:
    id: int
    item_id: str
    name: str
    keeper_id: Optional[int] = None
    keeper_item_id: Optional[str] = None
    reason: str = "paired"
    slug_confirms: bool = False


def pair_phantoms(
    rows: Iterable[Sequence],
) -> Tuple[List[Phantom], List[Phantom]]:
    """Split `items` rows into (deletable phantoms, unresolved phantoms).

    `rows` yields (id, item_id, name). Returns phantoms that have an identified
    keeper, and phantoms that do not and must be left alone.
    """
    rows = [(int(r[0]), str(r[1]), str(r[2])) for r in rows]
    # Index the correctly-keyed rows by the name they claim.
    keepers = {
        name: (rid, item_id)
        for rid, item_id, name in rows
        if not is_mangled_key(item_id)
    }

    paired: List[Phantom] = []
    unresolved: List[Phantom] = []
    for rid, item_id, name in rows:
        if not is_mangled_key(item_id):
            continue
        # `init_local_db.populate_items` mirrors archive keys back as
        # `name = item_id`. Such a row has lost its link to the real identity.
        if name == item_id:
            unresolved.append(
                Phantom(rid, item_id, name, reason="name-overwritten"))
            continue
        keeper = keepers.get(name)
        if keeper is None:
            unresolved.append(Phantom(rid, item_id, name, reason="no-keeper"))
            continue
        keeper_id, keeper_item_id = keeper
        paired.append(Phantom(
            rid, item_id, name,
            keeper_id=keeper_id,
            keeper_item_id=keeper_item_id,
            reason="paired",
            slug_confirms=slugify(name) == item_id,
        ))
    return paired, unresolved


def purge_archive_frame(frame):
    """Drop rows whose `item_slug` is a mangled key. Returns (kept, n_dropped)."""
    mask = frame["item_slug"].astype(str).map(is_mangled_key)
    return frame[~mask].copy(), int(mask.sum())


# ----------------------------------------------------------------------
# Database side
# ----------------------------------------------------------------------

def _require_postgres(engine) -> None:
    """Mirror of init_local_db's guard, inverted.

    That script refuses to run against anything but SQLite by inspecting the
    engine that actually got built rather than trusting env vars. This one is
    the reverse: `config.py` reads `.env` relative to the CWD, so running from
    a different directory silently rebinds to the local SQLite file and would
    "purge" the wrong database while reporting success.
    """
    backend = engine.url.get_backend_name()
    if backend != "postgresql":
        raise SystemExit(
            f"Refusing to run: engine is bound to '{backend}', not postgresql. "
            "This script is for production. Run it from backend/ so config.py "
            "picks up backend/.env."
        )


def purge_database(apply: bool) -> int:
    from sqlalchemy import text
    from database import engine

    _require_postgres(engine)
    logger.info("Engine: %s", str(engine.url).split("@")[-1])

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, item_id, name FROM items")).fetchall()
        logger.info("Scanned %s items rows", f"{len(rows):,}")

        paired, unresolved = pair_phantoms(rows)
        logger.info("Phantoms with an identified keeper: %s", f"{len(paired):,}")
        logger.info("Phantoms left alone (unresolved):   %s", f"{len(unresolved):,}")

        unconfirmed = [p for p in paired if not p.slug_confirms]
        if unconfirmed:
            logger.warning(
                "%s paired by name but slugify(name) != item_id -- review these:",
                f"{len(unconfirmed):,}")
            for p in unconfirmed[:10]:
                logger.warning("    id=%s %r name=%r", p.id, p.item_id, p.name)

        for p in unresolved[:10]:
            logger.info("  UNRESOLVED (%s) id=%s %r", p.reason, p.id, p.item_id)
        if len(unresolved) > 10:
            logger.info("  ... and %s more", f"{len(unresolved) - 10:,}")

        if not paired:
            logger.info("Nothing to delete.")
            return 0

        ids = [p.id for p in paired]
        logger.info("Child rows referencing the %s phantoms:", f"{len(ids):,}")
        total_children = 0
        for table in CHILD_TABLES:
            n = conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE item_id = ANY(:ids)"),
                {"ids": ids},
            ).scalar_one()
            total_children += n
            logger.info("    %-22s %s", table, f"{n:,}")
        logger.info("    %-22s %s", "TOTAL", f"{total_children:,}")

    if not apply:
        logger.info("")
        logger.info("DRY RUN -- nothing written. Re-run with --apply to execute.")
        return 0

    # One transaction: a partial purge would leave the archive and the DB
    # disagreeing about which items exist.
    with engine.begin() as conn:
        for table in CHILD_TABLES:
            res = conn.execute(
                text(f"DELETE FROM {table} WHERE item_id = ANY(:ids)"),
                {"ids": ids},
            )
            logger.info("Deleted %s from %s", f"{res.rowcount:,}", table)
        res = conn.execute(
            text("DELETE FROM items WHERE id = ANY(:ids)"), {"ids": ids})
        logger.info("Deleted %s from items", f"{res.rowcount:,}")
    logger.info("Database purge complete.")
    return len(ids)


# ----------------------------------------------------------------------
# Archive side
# ----------------------------------------------------------------------

def purge_archive(archive_dir: Path, apply: bool) -> int:
    """Rewrite each prices- parquet with phantom rows removed.

    Follows the read-filter-overwrite idiom of `append_to_parquet.py`.
    Only files that actually contain phantoms are rewritten.

    `snapshots-*.parquet` used to be purged alongside; it was retired as a pure
    duplicate of prices-* by `scripts/compact_price_archive.py`.
    """
    import pandas as pd

    targets = sorted(archive_dir.glob("prices-*.parquet"))
    if not targets:
        logger.warning("No parquet files found under %s", archive_dir)
        return 0

    total_dropped = 0
    for path in targets:
        frame = pd.read_parquet(path)
        if "item_slug" not in frame.columns:
            logger.info("%-34s no item_slug column, skipped", path.name)
            continue
        kept, dropped = purge_archive_frame(frame)
        total_dropped += dropped
        if dropped == 0:
            logger.info("%-34s clean", path.name)
            continue
        phantom_items = frame["item_slug"][
            frame["item_slug"].astype(str).map(is_mangled_key)].nunique()
        logger.info(
            "%-34s %s rows / %s items %s",
            path.name, f"{dropped:,}", f"{phantom_items:,}",
            "WOULD DROP" if not apply else "DROPPED",
        )
        if apply:
            kept.to_parquet(path, index=False)

    logger.info("Archive phantom rows: %s", f"{total_dropped:,}")
    if not apply:
        logger.info("DRY RUN -- no parquet rewritten. Re-run with --apply.")
    return total_dropped


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually write. Omitted, the script only reports (the default).")
    ap.add_argument(
        "--target", choices=["db", "archive", "both"], default="db",
        help="Which side to purge. Default 'db' -- that alone stops the daily "
             "leak, since the aggregator re-reads items every run.")
    ap.add_argument(
        "--archive-dir", type=Path, default=Path("../price-archive"),
        help="Directory holding prices-*.parquet.")
    args = ap.parse_args(argv)

    if args.target in ("db", "both"):
        logger.info("=== items table ===")
        purge_database(apply=args.apply)
    if args.target in ("archive", "both"):
        logger.info("=== price archive ===")
        purge_archive(args.archive_dir, apply=args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
