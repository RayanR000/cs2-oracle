#!/usr/bin/env python3
"""
One-off: seed the monthly ops partitions from the 2026-10-11 Supabase backup.

Phase 1, step 1 of docs/specs/2026-10-11-postgres-to-parquet-history-design.md.
The single-file mirror (`ops/item_forecasts.parquet`, `ops/forecast_outcomes.parquet`)
has `item_forecasts` only from 08-04 and never had the shadow tables or `items`, so
the history it lacks exists only in Postgres and in the backup. This writes the union
into the target layout:

    ops/item_forecasts/YYYY-MM.parquet               (by forecast_date)
    ops/forecast_outcomes/YYYY-MM.parquet            (by forecast_date)
    ops/forecast_candidates/YYYY-MM.parquet          (by forecast_date)
    ops/forecast_candidate_outcomes/YYYY-MM.parquet  (by the candidate's forecast_date)
    ops/items.parquet

The single-file mirror is READ, never written: every current reader, including the
frozen instruments, keeps reading exactly what it read before.

MERGE RULE (2026-10-11). Where both stores hold a row, POSTGRES wins, whole-row.
The criterion is "keep what the frozen reads see", and every frozen instrument that
reads forecasts or verdicts reads Postgres; `outside_baseline` reads the mirror, but
only outcome price columns, which agree in both stores on every shared row. The
mirror is not a faithful copy where it matters, measured against the backup:

* ~28k forecasts dated 08-05, 08-24, 08-26 and 09-08 carry a DIFFERENT band in the
  mirror (up to 48x apart, usually a later `created_at`): a re-run the mirror kept
  and Postgres did not. Every scored outcome for the first three dates matches the
  Postgres band and none matches the mirror's; 09-08 splits 6,238 / 4,933.
* 9,847 outcome verdicts (07-18 to 08-07) were rewritten in Postgres alone by the
  2026-08-11 21:10 verdict refresh after e4b57e9 moved `in_interval` to return
  space; the mirror still holds the retired dollar-basis verdict.
* Elsewhere they differ only by float8 round-tripping (<=5e-16).

The backup supplies what the mirror lacks: the surrogate `id` that outcomes key
on, `item_slug` for rows the mirror never had (from `items`), and those rows.

Rows the mirror has and Postgres deleted (25,192 forecasts dated 08-04/08-05, and
9,447 outcomes of 07-29 and 07-31 h=7 and 08-04 h=3 whose forecasts are gone;
measured 2026-10-11, cause unknown) are KEPT, with `pg_deleted = true`: a deletion
can be applied later, a dropped row cannot be recovered. A deleted forecast's `id`
is recovered from a mirror outcome that names it. A row dated after the backup is
newer, not deleted, and is never flagged.

GUARDS. A table whose target already exists is refused (closed months are written
once); `--overwrite` exists for a re-seed and is never used by CI. Each table is
checked for duplicate keys and for row count = |mirror keys U backup keys| before
anything is moved into place, and each lands with one `os.replace` of its directory.

Usage:
    venv/bin/python scripts/backfill_ops_partitions.py --backup-dir DIR --archive-dir ../archive/price-archive
    venv/bin/python scripts/backfill_ops_partitions.py --backup-dir DIR --archive-dir ... --apply
"""

import argparse
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import duckdb

logger = logging.getLogger("backfill_ops_partitions")

#: Partitioned tables: (natural key, sort order). The key is what a row IS; the
#: dedup/union and the duplicate check both run on it.
FORECAST_KEY = ["item_id", "forecast_date", "horizon_days"]
CANDIDATE_KEY = [*FORECAST_KEY, "component", "candidate_name", "candidate_version"]


class BackfillError(RuntimeError):
    pass


def _cols(con, relation: str) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]


def _merge_sql(con, mirror: str, backup: str, key: list[str], extra_select: str = "") -> str:
    """SELECT the full outer union of *mirror* and *backup* on *key*, backup winning.

    Column order is the backup's (the Postgres schema), then mirror-only columns.
    A column only one side has comes from that side; a column both have comes from
    the backup whenever the backup has the row, even when its value is NULL, so a
    row is never stitched from two versions.
    """
    m_cols, b_cols = _cols(con, mirror), _cols(con, backup)
    on = " AND ".join(f"m.{k} = b.{k}" for k in key)

    def pick(c):
        if c in key:
            return f"COALESCE(b.{c}, m.{c}) AS {c}"
        if c in m_cols and c in b_cols:
            return f"CASE WHEN b._in_b THEN b.{c} ELSE m.{c} END AS {c}"
        side = "m" if c in m_cols else "b"
        return f"{side}.{c} AS {c}"

    out = [c for c in b_cols if c != "_in_b"] + [c for c in m_cols if c not in b_cols and c != "_in_m"]
    return f"SELECT {', '.join(pick(c) for c in out)}{extra_select} FROM {mirror} m FULL OUTER JOIN {backup} b ON {on}"


def _check(con, table: str, key: list[str], expected: int) -> int:
    n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    dups = con.execute(
        f"SELECT count(*) FROM (SELECT 1 FROM {table} GROUP BY {', '.join(key)} HAVING count(*) > 1)"
    ).fetchone()[0]
    if dups:
        raise BackfillError(f"{table}: {dups} duplicate keys on {key}")
    if n != expected:
        raise BackfillError(f"{table}: {n} rows, expected |mirror U backup| = {expected}")
    return n


def build(con, backup_dir: Path, ops_dir: Path) -> dict[str, str]:
    """Create one DuckDB table per output; return {output name: sort clause}."""

    def src(path: Path) -> str:
        return f"read_parquet('{path}')"

    b = {
        t: src(backup_dir / f"{t}.parquet")
        for t in ("item_forecasts", "forecast_outcomes", "forecast_candidates", "forecast_candidate_outcomes", "items")
    }
    for t in ("item_forecasts", "forecast_outcomes"):
        if not (ops_dir / f"{t}.parquet").exists():
            raise BackfillError(f"mirror {ops_dir / t}.parquet is missing; is --archive-dir the data checkout?")

    # Normalise forecast_date to DATE (the mirror writes TIMESTAMP_NS at midnight, Postgres DATE).
    con.execute(
        f"CREATE TEMP VIEW m_fc AS SELECT * REPLACE (CAST(forecast_date AS DATE) AS forecast_date), "
        f"true AS _in_m FROM {src(ops_dir / 'item_forecasts.parquet')}"
    )
    con.execute(
        f"CREATE TEMP VIEW m_fo AS SELECT * REPLACE (CAST(forecast_date AS DATE) AS forecast_date), "
        f"true AS _in_m FROM {src(ops_dir / 'forecast_outcomes.parquet')}"
    )
    con.execute(f"CREATE TEMP VIEW b_fc AS SELECT *, true AS _in_b FROM {b['item_forecasts']}")
    con.execute(f"CREATE TEMP VIEW b_fo AS SELECT *, true AS _in_b FROM {b['forecast_outcomes']}")
    cutoff = con.execute("SELECT max(forecast_date) FROM b_fc").fetchone()[0]
    logger.info("backup forecast cutoff: %s", cutoff)

    # --- item_forecasts -------------------------------------------------------
    # A mirror-only row has no Postgres id; recover it from an outcome that names it.
    con.execute(
        "CREATE TEMP VIEW fo_ids AS SELECT item_id, forecast_date, horizon_days, "
        "any_value(forecast_id) AS forecast_id FROM m_fo GROUP BY ALL"
    )
    sql = _merge_sql(
        con,
        "m_fc",
        "b_fc",
        FORECAST_KEY,
        extra_select=f", COALESCE(m._in_m AND b.id IS NULL AND m.forecast_date <= DATE '{cutoff}', false) AS pg_deleted",
    )
    con.execute(f"CREATE TEMP TABLE out_item_forecasts AS {sql}")
    # id: the backup's, else the one an outcome recorded for a deleted forecast.
    con.execute(
        "UPDATE out_item_forecasts f SET id = r.forecast_id FROM fo_ids r WHERE f.id IS NULL "
        "AND r.item_id = f.item_id AND r.forecast_date = f.forecast_date AND r.horizon_days = f.horizon_days"
    )
    expected = con.execute(
        f"SELECT count(*) FROM (SELECT {', '.join(FORECAST_KEY)} FROM m_fc UNION SELECT {', '.join(FORECAST_KEY)} FROM b_fc)"
    ).fetchone()[0]
    _check(con, "out_item_forecasts", FORECAST_KEY, expected)
    ids = con.execute("SELECT count(*) - count(DISTINCT id), count(*) - count(id) FROM out_item_forecasts").fetchone()
    if ids[0] - ids[1] > 0:
        raise BackfillError(f"item_forecasts: {ids[0] - ids[1]} repeated ids")
    if ids[1]:
        logger.warning(
            "item_forecasts: %d mirror-only rows have no id (no Postgres row and no outcome naming one)", ids[1]
        )

    # --- forecast_outcomes ----------------------------------------------------
    sql = _merge_sql(
        con,
        "m_fo",
        "b_fo",
        ["forecast_id"],
        extra_select=(
            f", COALESCE(m._in_m AND f.id IS NULL AND m.forecast_date <= DATE '{cutoff}', false) AS pg_deleted"
        ),
    )
    sql += " LEFT JOIN (SELECT id FROM b_fc) f ON f.id = m.forecast_id"
    con.execute(f"CREATE TEMP TABLE out_forecast_outcomes AS {sql}")
    expected = con.execute(
        "SELECT count(*) FROM (SELECT forecast_id FROM m_fo UNION SELECT forecast_id FROM b_fo)"
    ).fetchone()[0]
    _check(con, "out_forecast_outcomes", ["forecast_id"], expected)

    # Backup-only rows carry no slug; every mirror writer must (rules/archive-reads).
    for t in ("out_item_forecasts", "out_forecast_outcomes"):
        con.execute(
            f"UPDATE {t} SET item_slug = i.item_id FROM {b['items']} i "
            f"WHERE {t}.item_slug IS NULL AND i.id = {t}.item_id"
        )
        missing = con.execute(f"SELECT count(*) FROM {t} WHERE item_slug IS NULL").fetchone()[0]
        if missing:
            logger.warning("%s: %d rows have no slug (item_id not in items)", t, missing)

    # --- shadow tables and items: never mirrored, backup only ----------------
    con.execute(f"CREATE TEMP TABLE out_forecast_candidates AS SELECT * FROM {b['forecast_candidates']}")
    _check(
        con,
        "out_forecast_candidates",
        ["id"],
        con.execute(f"SELECT count(*) FROM {b['forecast_candidates']}").fetchone()[0],
    )
    orphans = con.execute(
        f"SELECT count(*) FROM {b['forecast_candidate_outcomes']} o ANTI JOIN out_forecast_candidates c ON o.candidate_id = c.id"
    ).fetchone()[0]
    if orphans:
        raise BackfillError(f"forecast_candidate_outcomes: {orphans} rows name no candidate; cannot partition them")
    con.execute(
        f"CREATE TEMP TABLE out_forecast_candidate_outcomes AS SELECT o.*, c.forecast_date AS _part "
        f"FROM {b['forecast_candidate_outcomes']} o JOIN out_forecast_candidates c ON o.candidate_id = c.id"
    )
    _check(
        con,
        "out_forecast_candidate_outcomes",
        ["candidate_id"],
        con.execute(f"SELECT count(*) FROM {b['forecast_candidate_outcomes']}").fetchone()[0],
    )
    con.execute(f"CREATE TEMP TABLE out_items AS SELECT * FROM {b['items']}")
    _check(con, "out_items", ["id"], con.execute(f"SELECT count(*) FROM {b['items']}").fetchone()[0])

    return {
        "item_forecasts": ", ".join(FORECAST_KEY),
        "forecast_outcomes": "forecast_date, item_id, horizon_days, forecast_id",
        "forecast_candidates": ", ".join(CANDIDATE_KEY),
        "forecast_candidate_outcomes": "_part, candidate_id",
        "items": "id",
    }


def _part_col(table: str) -> str:
    return "_part" if table == "forecast_candidate_outcomes" else "forecast_date"


def plan(con, outputs: dict[str, str]) -> dict[str, list[tuple[str, int]]]:
    out = {}
    for t in outputs:
        if t == "items":
            out[t] = [("", con.execute("SELECT count(*) FROM out_items").fetchone()[0])]
            continue
        col = _part_col(t)
        out[t] = con.execute(
            f"SELECT strftime({col}, '%Y-%m') AS month, count(*) FROM out_{t} GROUP BY 1 ORDER BY 1"
        ).fetchall()
    return out


def write(con, outputs: dict[str, str], parts: dict, ops_dir: Path, overwrite: bool) -> None:
    targets = {t: ops_dir / ("items.parquet" if t == "items" else t) for t in outputs}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not overwrite:
        raise BackfillError(f"refusing to overwrite existing target(s): {existing}")

    for t, order in outputs.items():
        final = targets[t]
        tmp = final.with_name(f"{final.name}.{os.getpid()}.tmp")
        if t == "items":
            con.execute(
                f"COPY (SELECT * FROM out_items ORDER BY {order}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        else:
            tmp.mkdir()
            col = _part_col(t)
            drop = " EXCLUDE (_part)" if col == "_part" else ""
            for month, n in parts[t]:
                path = tmp / f"{month}.parquet"
                con.execute(
                    f"COPY (SELECT *{drop} FROM out_{t} WHERE strftime({col}, '%Y-%m') = '{month}' ORDER BY {order}) "
                    f"TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
                )
                got = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
                if got != n:
                    raise BackfillError(f"{t}/{month}: wrote {got} rows, planned {n}")
        if final.exists():
            shutil.rmtree(final) if final.is_dir() else final.unlink()
        os.replace(tmp, final)
        logger.info("wrote %s", final)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--backup-dir", type=Path, required=True, help="The 2026-10-11 backup (release supabase-backup-2026-10-11)."
    )
    ap.add_argument("--archive-dir", type=Path, required=True, help="The data checkout's price-archive/ directory.")
    ap.add_argument("--apply", action="store_true", help="Actually write. Omitted, the script only reports.")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing targets. Re-seed only; never in CI.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ops_dir = args.archive_dir / "ops"
    con = duckdb.connect()
    try:
        outputs = build(con, args.backup_dir, ops_dir)
        parts = plan(con, outputs)
        for t, months in parts.items():
            logger.info("%s: %s", t, ", ".join(f"{m or 'all'}={n}" for m, n in months))
        for t in ("item_forecasts", "forecast_outcomes"):
            n = con.execute(f"SELECT count(*) FILTER (WHERE pg_deleted) FROM out_{t}").fetchone()[0]
            logger.info("%s: %d rows kept with pg_deleted = true", t, n)
        if args.apply:
            write(con, outputs, parts, ops_dir, args.overwrite)
        else:
            logger.info("dry run; pass --apply to write")
    except BackfillError as e:
        logger.error("%s", e)
        return 1
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
