#!/usr/bin/env python3
"""
One-off: give every `prices-*.parquet` the same schema.

The archive grew three kinds of drift, and a plain glob read hides all of them
rather than failing on them (see `db/archive.py` for the demonstration):

1. **`source` is missing from the 13 pre-2026 yearly files.** DuckDB narrows a
   glob to the first file's schema, so `SELECT *` over the whole archive
   returns four columns and no error — `source` silently does not exist.
2. **`day` is `TIMESTAMP` in the yearly files, `TIMESTAMP_NS` in the monthly
   ones.** Both are midnight-truncated (verified: zero rows across all 21 files
   have a time component), so the distinction carries no information.
3. **`prices-2026-08.parquet` orders its columns differently** from the months
   before it.
4. **No file written before 2026-08-08 has `ingested_at`.** That one is not
   drift — the column did not exist. It is materialised as a typed NULL for the
   same reason `source` is, and a NULL there means "arrival unknown", never
   "arrived on `day`". See `db/archive.py`.

This rewrites each file into `db/archive.CANONICAL_PRICE_COLUMNS` order with
`day` as `DATE`, materialising absent columns as **typed NULLs**.

`source` stays NULL for pre-2026 rows on purpose. That series predates the
column, and `scripts/init_local_db.py` documents `source IS NULL` and
`day < '2026-01-01'` as selecting the same 5,542 backfilled items. Stamping a
literal label would break that equivalence for no gain — materialising the
column as NULL preserves it exactly while making the column addressable.

`min_price`/`max_price` are preserved where they exist. `compact_price_archive.py`
kept them on `prices-2026-03/04` because those two hold real intraday range, so
they are appended after the canonical columns rather than dropped.

Every rewrite is fingerprint-verified before it replaces the original, and the
script is idempotent: a second run reports every file as already normalized.

**The local `price-archive/` is not the canonical archive.** Running this here
changes nothing in production — only `aggregator-update.yml` writes the
`cs2-oracle-data` repo. Dispatch that workflow with `normalize_schema = true`.

Usage:
    venv/bin/python scripts/normalize_price_schema.py --archive-dir ../price-archive
    venv/bin/python scripts/normalize_price_schema.py --archive-dir ../price-archive --apply
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.archive import (  # noqa: E402
    CANONICAL_PRICE_COLUMNS,
    COLUMN_TYPES,
    RANGE_PRICE_COLUMNS,
    canonical_order,
)

logger = logging.getLogger("normalize_price_schema")

DAY_TYPE = "DATE"


def target_columns(present: Sequence[str]) -> list[str]:
    """The column list `present` should be rewritten to, in order.

    Canonical columns first, then whichever range columns the file actually
    carries. Any other column the file holds is preserved on the end rather
    than silently dropped — dropping columns is `compact_price_archive.py`'s
    job and it has its own evidence for each one.
    """
    missing = [c for c in CANONICAL_PRICE_COLUMNS if c not in present]
    return canonical_order(list(present) + missing)


def _describe(con, path: Path) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in con.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]


def needs_rewrite(present: Sequence[tuple[str, str]]) -> bool:
    """True if the file's columns or `day` type differ from canonical."""
    names = [c for c, _ in present]
    types = dict(present)
    return names != target_columns(names) or types.get("day") != DAY_TYPE


def _assert_midnight(con, path: Path, day_type: str) -> None:
    """Refuse to cast a `day` that carries a time component.

    A silent truncation here would collapse two observations onto one day and
    the fingerprint would not catch it — the row count is unchanged and the
    sums are unchanged. Nothing in the archive has ever had one, so this is a
    guard against a future writer, not a known case.
    """
    if day_type == DAY_TYPE:
        return
    n = con.sql(
        f"SELECT count(*) FROM read_parquet('{path}') "
        f"WHERE day <> date_trunc('day', day)").fetchone()[0]
    if n:
        raise RuntimeError(
            f"{path.name}: {n:,} rows have a non-midnight `day`; casting to "
            f"DATE would truncate them. Refusing to rewrite.")


def _fingerprint(con, relation: str, cols: Sequence[str]) -> tuple:
    """Row count plus an order-independent checksum over the data we keep.

    Sums are taken in DECIMAL, not DOUBLE: DuckDB aggregates in parallel and
    float addition is not associative, so two reads of the *same* data differ
    in the last digits. Matches `compact_price_archive._fingerprint`.

    `day` is cast to DATE on both sides so the TIMESTAMP -> DATE change itself
    does not register as a difference — the non-midnight guard above is what
    proves that cast lossless.
    """
    parts = ["count(*)", "count(DISTINCT item_slug)",
             "min(CAST(day AS DATE))", "max(CAST(day AS DATE))"]
    if "mean_price" in cols:
        parts += ["sum(mean_price::DECIMAL(24,8))", "count(mean_price)"]
    if "volume" in cols:
        parts += ["sum(volume::DECIMAL(24,0))", "count(volume)"]
    if "source" in cols:
        # Counted, not summed: this is what proves a NULLed-in source column
        # stayed NULL and a populated one kept every label.
        parts += ["count(source)", "count(DISTINCT source)"]
    if "ingested_at" in cols:
        # Same reasoning as `source`. A rewrite must not invent an arrival
        # time for a row that has none, nor drop one from a row that has.
        parts += ["count(ingested_at)", "max(ingested_at)"]
    for c in RANGE_PRICE_COLUMNS:
        if c in cols:
            parts.append(f"sum({c}::DECIMAL(24,8))")
    return con.sql(f"SELECT {', '.join(parts)} FROM {relation}").fetchone()


def normalize_file(con, path: Path, apply: bool) -> bool:
    """Rewrite one prices file into canonical shape. True if it changed."""
    present = _describe(con, path)
    names = [c for c, _ in present]
    types = dict(present)

    if not needs_rewrite(present):
        logger.info("%-24s already canonical", path.name)
        return False

    want = target_columns(names)
    added = [c for c in want if c not in names]
    reordered = names != want and not added
    logger.info(
        "%-24s %s%s%s",
        path.name,
        f"add {','.join(added)}; " if added else "",
        "reorder columns; " if reordered or added else "",
        f"day {types.get('day')} -> {DAY_TYPE}" if types.get("day") != DAY_TYPE else "",
    )
    if not apply:
        return True

    _assert_midnight(con, path, types.get("day", DAY_TYPE))

    # Fingerprint the columns that survive; `want` is a superset of `names`
    # except for the columns we are about to materialise as NULL.
    checked = [c for c in want if c in names]
    before = _fingerprint(con, f"read_parquet('{path}')", checked)

    projection = ", ".join(
        f"CAST(day AS {DAY_TYPE}) AS day" if c == "day"
        # A materialised column is NULLed to the type `prices_relation` will
        # read it back as. It used to be VARCHAR unconditionally, which was
        # right only as long as `source` was the sole column ever missing;
        # `ingested_at` is a TIMESTAMP, and a VARCHAR NULL in its place makes
        # the migrated file disagree with the reader's own CAST.
        else (f'"{c}"' if c in names
              else f"NULL::{COLUMN_TYPES[c]} AS {c}")
        for c in want
    )
    tmp = path.with_suffix(".parquet.tmp")
    try:
        # DuckDB preserves insertion order, so the physical row order (and the
        # item clustering some files have) survives the rewrite.
        con.sql(
            f"COPY (SELECT {projection} FROM read_parquet('{path}')) "
            f"TO '{tmp}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
        )
        after = _fingerprint(con, f"read_parquet('{tmp}')", checked)
        if after != before:
            raise RuntimeError(
                f"{path.name}: fingerprint changed, refusing to replace "
                f"(before={before} after={after})")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

    logger.info("%-24s -> %s", "", ", ".join(want))
    return True


def normalize_archive(archive_dir: Path, apply: bool) -> int:
    """Normalize every prices file. Returns the number that needed it."""
    targets = sorted(archive_dir.glob("prices-*.parquet"))
    if not targets:
        logger.warning("No prices-*.parquet found under %s", archive_dir)
        return 0

    changed = 0
    con = duckdb.connect()
    try:
        for path in targets:
            if normalize_file(con, path, apply):
                changed += 1
    finally:
        con.close()

    logger.info("%d of %d file(s) %s", changed, len(targets),
                "rewritten" if apply else "need rewriting")
    return changed


def verify(archive_dir: Path) -> bool:
    """True if a plain `SELECT *` over the glob now sees every column."""
    con = duckdb.connect()
    try:
        cols = [r[0] for r in con.sql(
            f"DESCRIBE SELECT * FROM "
            f"read_parquet('{archive_dir}/prices-*.parquet')").fetchall()]
    finally:
        con.close()
    ok = cols[:len(CANONICAL_PRICE_COLUMNS)] == list(CANONICAL_PRICE_COLUMNS)
    logger.info("Plain glob read now sees: %s  [%s]",
                ", ".join(cols), "OK" if ok else "STILL NARROWED")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--archive-dir", type=Path, default=Path("../price-archive"),
        help="Directory holding prices-*.parquet.")
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually write. Omitted, the script only reports (the default).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.archive_dir.is_dir():
        logger.error("Archive dir does not exist: %s", args.archive_dir)
        return 1

    # A wrong --archive-dir is the likely CI failure mode, and normalize_archive
    # treats "no files" as nothing to do. Without this, a typo'd path exits 0
    # having normalized nothing. Same guard as compact_price_archive.py.
    if not any(args.archive_dir.glob("prices-*.parquet")):
        logger.error(
            "No prices-*.parquet under %s — wrong --archive-dir?", args.archive_dir)
        return 1

    normalize_archive(args.archive_dir, apply=args.apply)
    if args.apply:
        verify(args.archive_dir)
    else:
        logger.info("DRY RUN — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
