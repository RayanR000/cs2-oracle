"""Read the monthly ops partitions as if they were the Postgres tables (Phase 2).

docs/specs/2026-10-13-ops-readers-switch-design.md. `connect` opens an in-memory DuckDB
connection with one view per table, named and typed like the Postgres table, so a reader's
SQL runs unchanged. `reader` wraps that as a `sql -> rows` function, the same shape as
`lambda sql: db.execute(text(sql)).fetchall()` on Postgres, which is how a switched reader
also computes its Postgres value for `compare`.

The view's file list is fixed when the connection opens, so `reader` opens a fresh one per
call: a read after tonight's partition writes sees them.

A table with no files raises instead of returning an empty view. An empty feedback panel snaps
the factor to 1.0 with no error, and an empty scoring panel publishes nothing.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from collections.abc import Callable, Iterable

import duckdb

from db import ops_partitions

logger = logging.getLogger(__name__)

TABLES = ("item_forecasts", "forecast_outcomes", "forecast_candidates", "forecast_candidate_outcomes", "items")


class MissingTable(RuntimeError):
    """A reader asked for a table, or a required column, that the partitions do not hold."""


def _view_sql(con, table: str, files) -> str:
    spec = ops_partitions.SPECS[table]
    src = f"read_parquet({ops_partitions._sql_list(files)}, union_by_name = true)"
    present = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
    # Projected from the schema, not SELECT *: a column an older month lacks reads NULL instead
    # of vanishing from the multi-file read (.claude/rules/archive-reads.md).
    cols = ", ".join(
        f"CAST({c} AS {t}) AS {c}" if c in present else f"CAST(NULL AS {t}) AS {c}"
        for c, t in spec.columns.items()
        if c != "pg_deleted"
    )
    where = " WHERE pg_deleted IS NOT TRUE" if "pg_deleted" in present else ""
    return f"CREATE VIEW {table} AS SELECT {cols} FROM {src}{where}"


def connect(root=None, tables: Iterable[str] = TABLES) -> duckdb.DuckDBPyConnection:
    """An in-memory connection with a view per table in *tables*; raises MissingTable for one with no files."""
    tables = tuple(tables)
    unknown = [t for t in tables if t not in ops_partitions.SPECS]
    if unknown:
        raise ValueError(f"ops_store: not an ops table: {unknown}")
    con = duckdb.connect()
    try:
        for t in tables:
            files = ops_partitions.month_files(t, root)
            if not files:
                raise MissingTable(f"ops_store: no partition files for {t} under {ops_partitions.ops_dir(root)}")
            con.execute(_view_sql(con, t, files))
    except BaseException:
        con.close()
        raise
    return con


def fetch(con, sql: str) -> list:
    """Run *sql* and return named tuples, so `r.base_price` reads as on a SQLAlchemy row."""
    cur = con.execute(sql)
    row = namedtuple("Row", [d[0] for d in cur.description])
    return [row(*r) for r in cur.fetchall()]


def reader(tables: Iterable[str] = TABLES, root=None) -> Callable[[str], list]:
    """A `sql -> rows` function over *tables*, opening a fresh connection per call."""
    tables = tuple(tables)

    def read(sql: str) -> list:
        con = connect(root, tables)
        try:
            return fetch(con, sql)
        finally:
            con.close()

    return read


def check_columns(read: Callable[[str], list], table: str, columns: Iterable[str]) -> None:
    """Raise MissingTable when *table* has rows and one of *columns* is NULL on all of them."""
    columns = list(columns)
    counts = ", ".join(f"count({c}) AS {c}" for c in columns)
    (row,) = read(f"SELECT count(*) AS n_rows, {counts} FROM {table}")
    if not row.n_rows:
        return
    empty = [c for c in columns if getattr(row, c) == 0]
    if empty:
        raise MissingTable(f"ops_store: {', '.join(f'{table}.{c}' for c in empty)} is NULL on all {row.n_rows:,} rows")


_STALE_NOTE = (
    "The partitions' value is used. On a night Postgres was read-only, Postgres is the stale store "
    "and this is expected; on any other night it blocks the next reader switch until explained."
)


def _differ(a, b, tol: float) -> bool:
    if a is None or b is None:
        return a is not b
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        return abs(float(a) - float(b)) > tol
    return a != b


def _diffs(pg_value, parquet_value, tol: float) -> list[tuple]:
    if isinstance(pg_value, dict) and isinstance(parquet_value, dict):
        keys = sorted(set(pg_value) | set(parquet_value), key=str)
        return [(k, pg_value.get(k), parquet_value.get(k)) for k in keys if _differ(pg_value.get(k), parquet_value.get(k), tol)]
    return [("value", pg_value, parquet_value)] if _differ(pg_value, parquet_value, tol) else []


def compare(name: str, pg: Callable[[], object], parquet_value, *, tol: float = 0.0, db=None) -> bool:
    """Log a switched reader's Parquet value beside Postgres's; warn on a difference. Never raises.

    *pg* is called here, so a failed Postgres read is a warning too; *db* is rolled back after
    one, or the caller's next statement would hit an aborted transaction. Removed in Phase 4.
    """
    try:
        pg_value = pg()
    except Exception as e:
        if db is not None:
            try:
                db.rollback()
            except Exception:
                logger.exception("read twice %s: rollback after the failed Postgres read also failed", name)
        logger.warning("read twice %s: Postgres read failed (%s); the partitions' value is used", name, e)
        print(f"::warning title=Read twice: {name}::Postgres read failed ({type(e).__name__}); the partitions' value is used", flush=True)
        return False
    diffs = _diffs(pg_value, parquet_value, tol)
    if isinstance(parquet_value, dict):
        logger.info("read twice %s: %d keys, %d differ", name, len(set(pg_value or {}) | set(parquet_value)), len(diffs))
    else:
        logger.info("read twice %s: postgres=%r partitions=%r", name, pg_value, parquet_value)
    if not diffs:
        return True
    if isinstance(parquet_value, dict):
        shown = "; ".join(f"{k}: postgres={a!r} partitions={b!r}" for k, a, b in diffs[:10])
        more = f" (+{len(diffs) - 10} more)" if len(diffs) > 10 else ""
    else:
        shown, more = f"postgres={pg_value!r} partitions={parquet_value!r}", ""
    logger.warning("read twice %s differs: %s%s", name, shown, more)
    print(f"::warning title=Read twice: {name}::{shown}{more}. {_STALE_NOTE}", flush=True)
    return False
