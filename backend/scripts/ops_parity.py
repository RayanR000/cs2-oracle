#!/usr/bin/env python3
"""
Nightly parity: Postgres and the monthly ops partitions must agree (Phase 1.2).

docs/specs/2026-10-11-postgres-to-parquet-history-design.md. Per table and forecast_date both
stores compute the same summary: row count, count(id), sum(id) and sums of the value columns.
Postgres returns only those aggregate rows (a few KB a night), so this costs no egress worth
counting. `pg_deleted` partition rows are excluded: Postgres deleted them on purpose.

Counts and id sums must match exactly; float sums within 1e-9 relative (the float8 noise
measured on 2026-10-11). The verdict refresh's rewrites show up as verdict-sum deltas.

Each night appends one row per table to ops/parity_runs.parquet, so "seven clean nights in a
row" is counted, not remembered. Exit 1 on any mismatch.

A Postgres that is read-only or out of disk (Supabase over its free-tier cap, 2026-10-29) skips
the check with a warning and records nothing: it can no longer take the nightly writes, so the
partitions are the current store and a mismatch against it means nothing.
"""

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pandas as pd
from db.ops_partitions import SPECS, closes_on, month_files, ops_dir
from sqlalchemy import text

logger = logging.getLogger("ops_parity")

SUMMARY = {
    "item_forecasts": (
        "count(*) AS n, count(id) AS n_id, sum(id) AS id_sum, sum(price_low) AS price_low, "
        "sum(price_mid) AS price_mid, sum(price_high) AS price_high, sum(current_price) AS current_price, "
        "sum(CASE WHEN direction = 'up' THEN 1 ELSE 0 END) AS n_up, "
        "sum(CASE WHEN direction = 'down' THEN 1 ELSE 0 END) AS n_down"
    ),
    "forecast_outcomes": (
        "count(*) AS n, count(id) AS n_id, sum(id) AS id_sum, sum(forecast_id) AS forecast_id_sum, "
        "sum(actual_price) AS actual_price, sum(base_price) AS base_price, sum(abs_error) AS abs_error, "
        "sum(pct_error) AS pct_error, sum(CAST(direction_correct AS INTEGER)) AS n_direction_correct, "
        "sum(CAST(in_interval AS INTEGER)) AS n_in_interval"
    ),
    "forecast_candidates": (
        "count(*) AS n, count(id) AS n_id, sum(id) AS id_sum, sum(centre_price) AS centre_price, "
        "sum(predicted_price_low) AS predicted_price_low, sum(predicted_price_high) AS predicted_price_high, "
        "sum(score) AS score"
    ),
    "forecast_candidate_outcomes": (
        "count(*) AS n, count(candidate_id) AS n_id, sum(candidate_id) AS id_sum, "
        "sum(actual_price) AS actual_price, sum(absolute_error) AS absolute_error, "
        "sum(percentage_error) AS percentage_error, sum(CAST(in_interval AS INTEGER)) AS n_in_interval"
    ),
}

_CO = "(SELECT o.*, c.forecast_date FROM {o} o JOIN {c} c ON c.id = o.candidate_id) r"

PG_REL = {
    "item_forecasts": "item_forecasts r",
    "forecast_outcomes": "forecast_outcomes r",
    "forecast_candidates": "forecast_candidates r",
    "forecast_candidate_outcomes": _CO.format(o="forecast_candidate_outcomes", c="forecast_candidates"),
}


@dataclass(frozen=True)
class Mismatch:
    table: str
    forecast_date: date | None  # None: a whole-table check
    field: str
    postgres: float | None
    partitions: float | None


def _files(table, root) -> str | None:
    files = month_files(table, root)
    if not files:
        return None
    return "read_parquet([" + ", ".join(f"'{f}'" for f in files) + "], union_by_name = true)"


def _parquet_rel(table, root) -> str | None:
    src = _files(table, root)
    if src is None:
        return None
    if table == "forecast_candidate_outcomes":
        cands = _files("forecast_candidates", root)
        return None if cands is None else _CO.format(o=src, c=cands)
    live = " WHERE NOT coalesce(pg_deleted, false)" if "pg_deleted" in SPECS[table].columns else ""
    return f"(SELECT * FROM {src}{live}) r"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    return df.set_index("forecast_date").astype(float)


def _summaries(db, table, root) -> tuple[pd.DataFrame, pd.DataFrame]:
    sql = "SELECT forecast_date, {s} FROM {rel} GROUP BY forecast_date"
    res = db.execute(text(sql.format(s=SUMMARY[table], rel=PG_REL[table])))
    pg = pd.DataFrame(res.fetchall(), columns=list(res.keys()))
    rel = _parquet_rel(table, root)
    pq = pd.DataFrame(columns=pg.columns) if rel is None else duckdb.sql(sql.format(s=SUMMARY[table], rel=rel)).df()
    return _normalise(pg), _normalise(pq)


def _exact(field: str) -> bool:
    return field.startswith("n") or field.endswith("_sum")


def _value(df, d, field) -> float:
    if df.empty or d not in df.index:
        return 0.0
    v = df.at[d, field]
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def compare(table, pg: pd.DataFrame, pq: pd.DataFrame) -> list[Mismatch]:
    fields = list(pg.columns if not pg.empty else pq.columns)
    dates = sorted(set(pg.index if not pg.empty else []) | set(pq.index if not pq.empty else []))
    out = []
    for d in dates:
        for f in fields:
            a, b = _value(pg, d, f), _value(pq, d, f)
            same = a == b if _exact(f) else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
            if not same:
                out.append(Mismatch(table, d, f, a, b))
    return out


def _duplicate_ids(table, root) -> int:
    """Rows sharing an id across the whole table: an aggregate per date cannot see two rows
    on different dates carrying one id (a read-only night's ids meeting Postgres's)."""
    src = _files(table, root)
    if src is None or "id" not in SPECS[table].columns:
        return 0
    return duckdb.sql(f"SELECT count(id) - count(DISTINCT id) FROM {src}").fetchone()[0]


def run_parity(db, root=None) -> list[Mismatch]:
    out = []
    for table in SUMMARY:
        out += compare(table, *_summaries(db, table, root))
        dup = _duplicate_ids(table, root)
        if dup:
            out.append(Mismatch(table, None, "duplicate_ids", 0.0, float(dup)))
    return out


def _closed(m: Mismatch, today: date) -> bool:
    return m.forecast_date is not None and closes_on(m.forecast_date.strftime("%Y-%m")) <= today


def failing(mismatches: list[Mismatch], today: date) -> list[Mismatch]:
    """The mismatches that fail the night: those in open months.

    A closed month is immutable in the partitions and released, so a later Postgres change there
    (a verdict refresh after a scoring change) can never be synced without breaking the release
    check. It is reported every night, apart, and does not stall the clean-night count.
    """
    return [m for m in mismatches if not _closed(m, today)]


def record(mismatches: list[Mismatch], run_date: date, root=None) -> None:
    from db.parquet import append_table

    now = datetime.now(UTC).replace(tzinfo=None)
    bad = failing(mismatches, run_date)
    rows = [
        {
            "run_date": run_date,
            "table": t,
            "n_mismatches": sum(1 for m in bad if m.table == t),
            "n_closed_mismatches": sum(1 for m in mismatches if m.table == t) - sum(1 for m in bad if m.table == t),
            "ok": not any(m.table == t for m in bad),
            "run_at": now,
        }
        for t in SUMMARY
    ]
    append_table("parity_runs", rows, ["run_date", "table"])


def clean_streak(root=None) -> int:
    path = ops_dir(root) / "parity_runs.parquet"
    if not path.exists():
        return 0
    nights = duckdb.sql(
        f"SELECT run_date, bool_and(ok) AS ok FROM read_parquet('{path}') GROUP BY 1 ORDER BY 1 DESC"
    ).fetchall()
    streak = 0
    for _, ok in nights:
        if not ok:
            break
        streak += 1
    return streak


DISK_FULL_SQLSTATE = "53100"


def _dead_postgres_reason(db) -> str | None:
    """Why Postgres cannot be compared against tonight, or None when it can."""
    if str(db.execute(text("SHOW transaction_read_only")).scalar()).lower() == "on":
        return "Postgres is read-only"
    return None


def _is_disk_full(exc: BaseException) -> bool:
    pgcode = getattr(getattr(exc, "orig", None), "pgcode", None)
    return pgcode == DISK_FULL_SQLSTATE or "No space left on device" in str(exc)


def _skip(reason: str) -> int:
    msg = f"{reason}; parity not checked and no night recorded. The partitions are the current store."
    logger.warning(msg)
    print(f"::warning title=Ops parity skipped::{msg}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from database import SessionLocal

    db = SessionLocal()
    try:
        reason = _dead_postgres_reason(db)
        if reason:
            return _skip(reason)
        try:
            mismatches = run_parity(db)
        except Exception as e:
            if not _is_disk_full(e):
                raise
            return _skip("Postgres is out of disk")
    finally:
        db.close()
    today = datetime.now(UTC).date()
    record(mismatches, today)
    bad = failing(mismatches, today)
    closed = [m for m in mismatches if m not in bad]
    for m in bad[:50]:
        logger.error("%s %s %s: postgres=%s partitions=%s", m.table, m.forecast_date, m.field, m.postgres, m.partitions)
    if len(bad) > 50:
        logger.error("... and %d more", len(bad) - 50)
    if closed:
        months = sorted({f"{m.table} {m.forecast_date:%Y-%m}" for m in closed if m.forecast_date})
        print(f"::warning title=Ops parity (closed months)::Postgres differs in closed month(s): {months}", flush=True)
    streak = clean_streak()
    logger.info(
        "parity: %d open-month mismatches, %d in closed months; clean nights in a row: %d (Phase 2 starts at 3)",
        len(bad),
        len(closed),
        streak,
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
