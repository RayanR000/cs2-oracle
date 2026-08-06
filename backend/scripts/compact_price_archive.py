#!/usr/bin/env python3
"""
One-off: strip redundant columns from `prices-*.parquet` and retire `snapshots-*`.

Two independent reclaims, both dry-run unless `--apply` is passed:

1. **Redundant price columns.** `median_price` is a structural duplicate of
   `mean_price`: the aggregator's snapshot CSV header is
   `item_slug,day,source,price,volume` (`collectors/pipeline.py:276`), so
   `append_to_parquet.py`'s `median_price=("median_price","mean")` aggregation
   always falls through to its `("price","mean")` branch. `min_price`/`max_price`
   collapse the same way — one CSV row per `(item_slug, day, source)` means
   min == max == mean by construction. Measured: `median_price == mean_price` in
   20,755,907 of 20,755,907 rows.

   `prices-2026-03` and `prices-2026-04` are the exception and are left intact:
   they predate the current one-row-per-key CSV and hold 530,111 / 778,786 rows
   where `min != max`. That is real intraday range, so those two files keep all
   eight columns.

   `volume` is NOT dropped even though it is identically 0 since 2026-05 —
   `backtest/price_resolution.py:249` selects it.

2. **`snapshots-*.parquet`.** A strict projection of `prices-*`
   (`SELECT item_slug, day, source, mean_price AS price, volume`) that no code
   reads. Verified: every snapshot row joins to `prices-*` on
   `(item_slug, day, source)` with equal price and volume, all six months.

Usage:
    venv/bin/python scripts/compact_price_archive.py --archive-dir ../price-archive
    venv/bin/python scripts/compact_price_archive.py --archive-dir ../price-archive --apply
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import duckdb

logger = logging.getLogger("compact_price_archive")

# Dropped from every prices file: a structural duplicate of mean_price.
ALWAYS_DROP = ("median_price",)

# Dropped except where they carry real range data (see module docstring).
RANGE_COLS = ("min_price", "max_price")
RANGE_KEEP = frozenset({"prices-2026-03.parquet", "prices-2026-04.parquet"})


def columns_to_drop(filename: str, present: Sequence[str]) -> list:
    """Which of `present` should be dropped from the file named `filename`."""
    drop = [c for c in ALWAYS_DROP if c in present]
    if filename not in RANGE_KEEP:
        drop += [c for c in RANGE_COLS if c in present]
    return drop


def _describe(con, path: Path) -> list:
    return [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]


def _fingerprint(con, path: Path, cols: Sequence[str]) -> tuple:
    """Row count plus an order-independent checksum over the columns we keep.

    Sums are taken in DECIMAL, not DOUBLE: DuckDB aggregates in parallel and
    float addition is not associative, so `sum(mean_price::DOUBLE)` differs in
    the last few digits between two reads of the *same* data. DECIMAL addition
    is exact, so a difference here is a real difference.
    """
    parts = ["count(*)", "count(DISTINCT item_slug)", "min(day)", "max(day)"]
    if "mean_price" in cols:
        parts.append("sum(mean_price::DECIMAL(24,8))")
        parts.append("count(mean_price)")
    if "volume" in cols:
        parts.append("sum(volume::DECIMAL(24,0))")
    for c in ("min_price", "max_price"):
        if c in cols:
            parts.append(f"sum({c}::DECIMAL(24,8))")
    return con.sql(f"SELECT {', '.join(parts)} FROM read_parquet('{path}')").fetchone()


def compact_columns(archive_dir: Path, apply: bool) -> int:
    """Rewrite each prices file without its redundant columns. Returns bytes saved."""
    targets = sorted(archive_dir.glob("prices-*.parquet"))
    if not targets:
        logger.warning("No prices-*.parquet found under %s", archive_dir)
        return 0

    saved = 0
    con = duckdb.connect()
    try:
        for path in targets:
            present = _describe(con, path)
            drop = columns_to_drop(path.name, present)
            if not drop:
                logger.info("%-24s %d cols, nothing to drop", path.name, len(present))
                continue

            keep = [c for c in present if c not in drop]
            before_bytes = path.stat().st_size
            before = _fingerprint(con, path, keep)

            logger.info(
                "%-24s drop %-34s %.1f MB",
                path.name, ",".join(drop), before_bytes / 1e6,
            )
            if not apply:
                continue

            tmp = path.with_suffix(".parquet.tmp")
            col_list = ", ".join(f'"{c}"' for c in keep)
            # DuckDB preserves insertion order, so the existing physical row
            # order (and the item clustering it gives some files) survives.
            con.sql(
                f"COPY (SELECT {col_list} FROM read_parquet('{path}')) "
                f"TO '{tmp}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
            )
            after = _fingerprint(con, tmp, keep)
            if after != before:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{path.name}: fingerprint changed, refusing to replace "
                    f"(before={before} after={after})"
                )
            os.replace(tmp, path)
            after_bytes = path.stat().st_size
            saved += before_bytes - after_bytes
            logger.info(
                "%-24s -> %.1f MB (saved %.1f MB), %s rows verified",
                path.name, after_bytes / 1e6,
                (before_bytes - after_bytes) / 1e6, f"{before[0]:,}",
            )
    finally:
        con.close()

    logger.info("Column compaction saved %.1f MB", saved / 1e6)
    return saved


def absorb_orphan_snapshot_rows(archive_dir: Path, apply: bool) -> int:
    """Copy snapshot rows that prices-* is missing into prices-*. Returns row count.

    Retiring snapshots-* is only lossless once these are absorbed. In the real
    archive there are 131 such rows, all on 2026-07-11 (the CSV -> Parquet
    cutover day) across 129 slugs that appear on other days — a partial write,
    not junk data.
    """
    absorbed = 0
    con = duckdb.connect()
    try:
        for snap in sorted(archive_dir.glob("snapshots-*.parquet")):
            prices = archive_dir / snap.name.replace("snapshots-", "prices-")
            if not prices.exists():
                continue
            keep = _describe(con, prices)
            orphan_sql = f"""
                SELECT s.item_slug, s.day, s.source, s.price, s.volume
                FROM read_parquet('{snap}') s
                LEFT JOIN read_parquet('{prices}') p
                  ON p.item_slug = s.item_slug AND p.day = s.day
                 AND p.source = s.source
                WHERE p.item_slug IS NULL
            """
            n = con.sql(f"SELECT count(*) FROM ({orphan_sql})").fetchone()[0]
            if not n:
                continue
            days = con.sql(
                f"SELECT min(day), max(day) FROM ({orphan_sql})").fetchone()
            logger.info(
                "%-28s %s orphan rows (%s..%s) %s",
                snap.name, f"{n:,}", days[0], days[1],
                "ABSORBED" if apply else "WOULD ABSORB",
            )
            absorbed += n
            if not apply:
                continue

            # Project the orphans onto the prices schema. mean_price is the
            # snapshot price; the dropped range columns are not resurrected.
            proj = []
            for c in keep:
                if c == "mean_price":
                    proj.append("o.price AS mean_price")
                elif c in ("item_slug", "day", "source", "volume"):
                    proj.append(f"o.{c}")
                else:
                    proj.append(f"NULL AS {c}")
            tmp = prices.with_suffix(".parquet.tmp")
            col_list = ", ".join(f'"{c}"' for c in keep)
            con.sql(
                f"COPY (SELECT {col_list} FROM read_parquet('{prices}') "
                f"UNION ALL BY NAME "
                f"SELECT {', '.join(proj)} FROM ({orphan_sql}) o) "
                f"TO '{tmp}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
            )
            before = con.sql(
                f"SELECT count(*) FROM read_parquet('{prices}')").fetchone()[0]
            after = con.sql(
                f"SELECT count(*) FROM read_parquet('{tmp}')").fetchone()[0]
            if after != before + n:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{prices.name}: expected {before + n} rows, got {after}")
            os.replace(tmp, prices)
            logger.info("%-28s -> %s now %s rows", "", prices.name, f"{after:,}")
    finally:
        con.close()

    if absorbed:
        logger.info("Absorbed %s orphan snapshot rows", f"{absorbed:,}")
    return absorbed


def _verify_snapshot_is_derivable(con, archive_dir: Path, snap: Path) -> Optional[str]:
    """Return None if `snap` is fully reproducible from the matching prices file."""
    prices = archive_dir / snap.name.replace("snapshots-", "prices-")
    if not prices.exists():
        return f"no matching {prices.name}"
    row = con.sql(f"""
        SELECT count(*),
               count(p.item_slug),
               sum(CASE WHEN p.mean_price = s.price
                         AND p.volume IS NOT DISTINCT FROM s.volume
                        THEN 1 ELSE 0 END)
        FROM read_parquet('{snap}') s
        LEFT JOIN read_parquet('{prices}') p
          ON p.item_slug = s.item_slug AND p.day = s.day AND p.source = s.source
    """).fetchone()
    total, matched, equal = row
    if matched != total or equal != total:
        return f"{total - matched} unmatched, {total - equal} divergent vs {prices.name}"
    return None


def drop_snapshots(archive_dir: Path, apply: bool) -> int:
    """Delete snapshots-*.parquet after proving each is derivable. Returns bytes freed."""
    targets = sorted(archive_dir.glob("snapshots-*.parquet"))
    if not targets:
        logger.info("No snapshots-*.parquet found — already retired")
        return 0

    freed = 0
    con = duckdb.connect()
    try:
        for path in targets:
            problem = _verify_snapshot_is_derivable(con, archive_dir, path)
            size = path.stat().st_size
            if problem:
                logger.warning("%-28s KEPT — %s", path.name, problem)
                continue
            logger.info(
                "%-28s %.1f MB %s",
                path.name, size / 1e6, "DELETED" if apply else "WOULD DELETE",
            )
            if apply:
                path.unlink()
                freed += size
    finally:
        con.close()

    logger.info("Snapshot retirement freed %.1f MB", freed / 1e6)
    return freed


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--archive-dir", type=Path, default=Path("../price-archive"),
        help="Directory holding prices-*.parquet / snapshots-*.parquet.")
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually write. Omitted, the script only reports (the default).")
    ap.add_argument(
        "--skip-columns", action="store_true", help="Leave prices-* columns alone.")
    ap.add_argument(
        "--skip-snapshots", action="store_true", help="Leave snapshots-* in place.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.archive_dir.is_dir():
        logger.error("Archive dir does not exist: %s", args.archive_dir)
        return 1

    # A wrong --archive-dir is the likely CI failure mode, and every function
    # below treats "no files" as nothing to do. Without this, a typo'd path
    # exits 0 having silently compacted nothing.
    if not any(args.archive_dir.glob("prices-*.parquet")):
        logger.error(
            "No prices-*.parquet under %s — wrong --archive-dir?", args.archive_dir)
        return 1

    if not args.skip_columns:
        logger.info("=== redundant price columns ===")
        compact_columns(args.archive_dir, apply=args.apply)
    if not args.skip_snapshots:
        logger.info("=== orphan snapshot rows ===")
        absorb_orphan_snapshot_rows(args.archive_dir, apply=args.apply)
        logger.info("=== snapshots-*.parquet ===")
        drop_snapshots(args.archive_dir, apply=args.apply)

    if not args.apply:
        logger.info("DRY RUN — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
