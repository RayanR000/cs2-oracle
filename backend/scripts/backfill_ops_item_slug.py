#!/usr/bin/env python3
"""
One-off: stamp `item_slug` onto the ops tables that already have rows.

`price-archive/*.parquet` keys on `item_slug` (the market_hash_name).
`price-archive/ops/*.parquet` keys on `item_id`, the Postgres surrogate. Nothing
in the archive maps one to the other, so joining a stored forecast to its own
price history needed a round-trip to production Supabase — the network hop the
Parquet store exists to remove.

`forecast_prices.py` and `backtest_accuracy.py` now write the slug on every new
row. This fills in the rows written before that, reading the mapping from
`items` (a read-only `SELECT id, item_id`).

Idempotent: a row that already carries a slug is left alone, so a second run
reports zero filled. Rows whose `item_id` has no `items` row keep a NULL slug —
that is a referential break between the ops table and `items`, not something
this script can invent a value for, and it is reported rather than hidden.

**Run it against the archive you mean.** The local `price-archive/` is not the
canonical store; only `aggregator-update.yml` writes the `cs2-oracle-data` repo.

Usage:
    venv/bin/python scripts/backfill_ops_item_slug.py --archive-dir ../price-archive
    venv/bin/python scripts/backfill_ops_item_slug.py --archive-dir ../price-archive --apply
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("backfill_ops_item_slug")

#: Every ops table keyed on the surrogate `item_id`. `event_impacts_denorm`
#: carries one too and is included: leaving one table un-joinable defeats the
#: point of making the others joinable.
DEFAULT_TABLES = ("item_forecasts", "forecast_outcomes", "event_impacts_denorm")


def load_id_to_slug() -> dict:
    """`items.id` -> `items.item_id`. Read-only; the DB is never written."""
    from database import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        return {r.id: r.item_id for r in db.execute(text("SELECT id, item_id FROM items")).fetchall()}
    finally:
        db.close()


def backfill_table(con, path: Path, id_to_slug: dict, apply: bool) -> tuple:
    """Fill `item_slug` on one ops file. Returns (filled, unmapped, total)."""
    cols = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
    if "item_id" not in cols:
        logger.info("%-24s no item_id column — skipped", path.name)
        return (0, 0, 0)

    has_slug = "item_slug" in cols
    total = con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    todo = (
        total
        if not has_slug
        else con.sql(f"SELECT count(*) FROM read_parquet('{path}') WHERE item_slug IS NULL").fetchone()[0]
    )

    if not todo:
        logger.info("%-24s all %s row(s) already carry a slug", path.name, f"{total:,}")
        return (0, 0, total)

    mapping = pd.DataFrame({"_map_id": list(id_to_slug.keys()), "_map_slug": list(id_to_slug.values())})
    con.register("_slug_map", mapping)

    unmapped = con.sql(f"""
        SELECT count(*) FROM read_parquet('{path}') o
        LEFT JOIN _slug_map m ON m._map_id = o.item_id
        WHERE m._map_slug IS NULL
          {"AND o.item_slug IS NULL" if has_slug else ""}
    """).fetchone()[0]

    logger.info(
        "%-24s %s of %s row(s) to fill%s",
        path.name,
        f"{todo:,}",
        f"{total:,}",
        f"; {unmapped:,} have no items row and stay NULL" if unmapped else "",
    )
    if not apply:
        con.unregister("_slug_map")
        return (todo - unmapped, unmapped, total)

    # Every original column, in its original order, with item_slug appended
    # (or coalesced onto, on a re-run over a partially filled file).
    kept = [c for c in cols if c != "item_slug"]
    slug_expr = "COALESCE(o.item_slug, m._map_slug)" if has_slug else "m._map_slug"
    projection = ", ".join(f'o."{c}"' for c in kept) + f", {slug_expr} AS item_slug"

    tmp = path.with_suffix(".parquet.tmp")
    escaped = str(tmp).replace("'", "''")
    try:
        con.sql(f"""
            COPY (
                SELECT {projection}
                FROM read_parquet('{path}') o
                LEFT JOIN _slug_map m ON m._map_id = o.item_id
            ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        after = con.sql(f"SELECT count(*) FROM read_parquet('{tmp}')").fetchone()[0]
        if after != total:
            # A LEFT JOIN cannot drop rows, but it CAN multiply them if the
            # mapping has a duplicate id. Better to refuse than to double the
            # served outcome table.
            raise RuntimeError(
                f"{path.name}: expected {total:,} rows, got {after:,} — the id "
                f"mapping is not unique. Refusing to replace."
            )
        os.replace(tmp, path)
    finally:
        con.unregister("_slug_map")
        tmp.unlink(missing_ok=True)

    filled = con.sql(f"SELECT count(*) FROM read_parquet('{path}') WHERE item_slug IS NOT NULL").fetchone()[0]
    logger.info("%-24s -> %s of %s row(s) now carry a slug", "", f"{filled:,}", f"{total:,}")
    return (todo - unmapped, unmapped, total)


def backfill(archive_dir: Path, tables: Sequence[str], id_to_slug: dict, apply: bool) -> int:
    """Backfill every named ops table. Returns rows filled."""
    ops_dir = archive_dir / "ops"
    if not ops_dir.is_dir():
        logger.error("No ops/ directory under %s", archive_dir)
        return 0

    filled = 0
    con = duckdb.connect()
    try:
        for table in tables:
            path = ops_dir / f"{table}.parquet"
            if not path.exists():
                logger.info("%-24s absent — skipped", f"{table}.parquet")
                continue
            n, _, _ = backfill_table(con, path, id_to_slug, apply)
            filled += n
    finally:
        con.close()

    logger.info("%s row(s) %s", f"{filled:,}", "filled" if apply else "would be filled")
    return filled


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--archive-dir", type=Path, default=Path("../price-archive"), help="Archive root; ops/ lives under it."
    )
    ap.add_argument(
        "--tables",
        nargs="+",
        default=list(DEFAULT_TABLES),
        help=f"Ops tables to stamp (default: {' '.join(DEFAULT_TABLES)}).",
    )
    ap.add_argument("--apply", action="store_true", help="Actually write. Omitted, the script only reports.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not (args.archive_dir / "ops").is_dir():
        logger.error("No ops/ under %s — wrong --archive-dir?", args.archive_dir)
        return 1

    id_to_slug = load_id_to_slug()
    if not id_to_slug:
        # Every row would land NULL and the run would report success having
        # achieved nothing — the silent-no-op shape this repo keeps hitting.
        logger.error("items table returned no rows; refusing to write NULLs")
        return 1
    logger.info("Loaded %s id -> slug mapping(s)", f"{len(id_to_slug):,}")

    backfill(args.archive_dir, args.tables, id_to_slug, apply=args.apply)
    if not args.apply:
        logger.info("DRY RUN — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
