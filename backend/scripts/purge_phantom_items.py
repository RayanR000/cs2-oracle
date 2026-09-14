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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("purge_phantom_items")

# The one definition, shared with the archive readers. What this script deletes
# from production and what `archive_universe_sql_filter` drops from a training
# read have to be the same set, or the model and the database disagree about
# which items exist. Also the predicate the Steam backfill uses to keep these
# keys out of its target list (`backfill_steam_listing_history.py:195-199`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.item_parser import is_phantom_slug as is_mangled_key  # noqa: E402


def slugify(name: str) -> str:
    """Verbatim copy of `migrate_historical_data.py:56-59`.

    Kept here as a cross-check only: it is lossy (case, and every run of
    non-alphanumerics collapses to one '-'), so it is not used to recover names.
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# Every table with a FOREIGN KEY onto items.id. There is no ON DELETE anywhere
# in the schema, so Postgres applies NO ACTION and a bare DELETE FROM items
# raises while any child lives.
#
# This list is read from the live catalog rather than hardcoded, because the
# obvious way to build it -- reading database.py and migrations/versions/ --
# gives the WRONG answer against production. Prod's schema is behind the
# migrations: `daily_analysis` was dropped by 0015 but still exists there and
# still holds a FK, so a purge on 2026-08-09 deleted all 97,520 child rows and
# then failed on the final DELETE FROM items (the transaction rolled back).
# `trend_indicators` (0010) and `chart_points` (0012) are dropped the same way
# and may survive the same way. Only the database knows what is really there.
CHILD_FK_SQL = """
    SELECT ch.relname AS child_table, a.attname AS child_column
    FROM pg_constraint c
    JOIN pg_class ch ON ch.oid = c.conrelid
    JOIN pg_class p ON p.oid = c.confrelid
    JOIN pg_namespace n ON n.oid = ch.relnamespace
    JOIN pg_attribute a
      ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
    WHERE c.contype = 'f'
      AND p.relname = 'items'
      AND n.nspname = 'public'
"""

# Edges among those tables, so one that references another is emptied first --
# forecast_outcomes FKs item_forecasts.id and must precede it.
CHILD_EDGE_SQL = """
    SELECT ch.relname AS child_table, p.relname AS parent_table
    FROM pg_constraint c
    JOIN pg_class ch ON ch.oid = c.conrelid
    JOIN pg_class p ON p.oid = c.confrelid
    WHERE c.contype = 'f'
"""


def order_child_tables(
    tables: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> list[str]:
    """Delete order: a table is emitted before anything it references.

    *edges* is (child, parent) meaning child holds a FK onto parent. A table
    nothing else still-pending references is safe to empty now.
    """
    remaining = set(tables)
    edges = [(c, p) for c, p in edges if c != p]
    order: list[str] = []
    while remaining:
        referenced = {p for c, p in edges if c in remaining and p in remaining}
        ready = sorted(t for t in remaining if t not in referenced)
        if not ready:
            # A cycle cannot be resolved by ordering; emit the rest and let
            # Postgres raise rather than guessing.
            ready = sorted(remaining)
        order.extend(ready)
        remaining -= set(ready)
    return order


def discover_child_tables(conn) -> list[tuple[str, str]]:
    """(table, fk_column) for every table referencing items.id, in delete order."""
    from sqlalchemy import text

    cols = {r[0]: r[1] for r in conn.execute(text(CHILD_FK_SQL)).fetchall()}
    edges = [(r[0], r[1]) for r in conn.execute(text(CHILD_EDGE_SQL)).fetchall()]
    return [(t, cols[t]) for t in order_child_tables(cols, edges)]


@dataclass
class Phantom:
    id: int
    item_id: str
    name: str
    keeper_id: int | None = None
    keeper_item_id: str | None = None
    reason: str = "paired"
    slug_confirms: bool = False


def pair_phantoms(
    rows: Iterable[Sequence],
) -> tuple[list[Phantom], list[Phantom]]:
    """Split `items` rows into (deletable phantoms, unresolved phantoms).

    `rows` yields (id, item_id, name). Returns phantoms that have an identified
    keeper, and phantoms that do not and must be left alone.
    """
    rows = [(int(r[0]), str(r[1]), str(r[2])) for r in rows]
    # Index the correctly-keyed rows by the name they claim.
    keepers = {name: (rid, item_id) for rid, item_id, name in rows if not is_mangled_key(item_id)}

    paired: list[Phantom] = []
    unresolved: list[Phantom] = []
    for rid, item_id, name in rows:
        if not is_mangled_key(item_id):
            continue
        # `init_local_db.populate_items` mirrors archive keys back as
        # `name = item_id`. Such a row has lost its link to the real identity.
        if name == item_id:
            unresolved.append(Phantom(rid, item_id, name, reason="name-overwritten"))
            continue
        keeper = keepers.get(name)
        if keeper is None:
            unresolved.append(Phantom(rid, item_id, name, reason="no-keeper"))
            continue
        keeper_id, keeper_item_id = keeper
        paired.append(
            Phantom(
                rid,
                item_id,
                name,
                keeper_id=keeper_id,
                keeper_item_id=keeper_item_id,
                reason="paired",
                slug_confirms=slugify(name) == item_id,
            )
        )
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
    from database import engine
    from sqlalchemy import text

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
            logger.warning("%s paired by name but slugify(name) != item_id -- review these:", f"{len(unconfirmed):,}")
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
        child_tables = discover_child_tables(conn)
        logger.info(
            "Child rows referencing the %s phantoms (%s FK tables, from the live catalog):",
            f"{len(ids):,}",
            len(child_tables),
        )
        total_children = 0
        for table, col in child_tables:
            n = conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE {col} = ANY(:ids)"),
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
        # Re-discovered inside the write transaction: the count above ran on a
        # separate connection, and the list has to match what is deleted.
        for table, col in discover_child_tables(conn):
            res = conn.execute(
                text(f"DELETE FROM {table} WHERE {col} = ANY(:ids)"),
                {"ids": ids},
            )
            logger.info("Deleted %s from %s", f"{res.rowcount:,}", table)
        res = conn.execute(text("DELETE FROM items WHERE id = ANY(:ids)"), {"ids": ids})
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
        phantom_items = frame["item_slug"][frame["item_slug"].astype(str).map(is_mangled_key)].nunique()
        logger.info(
            "%-34s %s rows / %s items %s",
            path.name,
            f"{dropped:,}",
            f"{phantom_items:,}",
            "WOULD DROP" if not apply else "DROPPED",
        )
        if apply:
            kept.to_parquet(path, index=False)

    logger.info("Archive phantom rows: %s", f"{total_dropped:,}")
    if not apply:
        logger.info("DRY RUN -- no parquet rewritten. Re-run with --apply.")
    return total_dropped


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--apply", action="store_true", help="Actually write. Omitted, the script only reports (the default)."
    )
    ap.add_argument(
        "--target",
        choices=["db", "archive", "both"],
        default="db",
        help="Which side to purge. Default 'db' -- that alone stops the daily "
        "leak, since the aggregator re-reads items every run.",
    )
    ap.add_argument(
        "--archive-dir", type=Path, default=Path("../price-archive"), help="Directory holding prices-*.parquet."
    )
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
