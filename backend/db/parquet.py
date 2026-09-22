"""DuckDB-backed Parquet store for operational data.

Each table in the system gets a single Parquet file under
``price-archive/ops/{table_name}.parquet``.  The store uses DuckDB for
columnar reads and supports:

* ``append()`` — concat-and-dedup (the file is fully rewritten, which is
  fine since every ops table is <100 MiB).
* ``query()`` — read + filter via a DuckDB connection returned as a context
  manager so callers can run SQL directly against ``read_parquet(...)``.

Denormalised tables (e.g. *event_impacts* which joins 3 tables on the
read side) are written as a single pre-joined file so that API read
paths are single-file queries with no runtime join overhead.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent / "price-archive"
OPS_DIR = ARCHIVE_ROOT / "ops"


def ensure_ops_dir() -> Path:
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    return OPS_DIR


def _table_path(table: str) -> Path:
    ensure_ops_dir()
    return OPS_DIR / f"{table}.parquet"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0 and hasattr(non_null.iloc[0], "isoformat"):
                with contextlib.suppress(ValueError, TypeError):
                    df[col] = pd.to_datetime(df[col])
    return df


# ---------------------------------------------------------------------------
# Nested values
# ---------------------------------------------------------------------------

# NESTED VALUES ARE STORED AS JSON TEXT. This is a store-wide invariant, not a
# per-table choice, and it exists because DuckDB infers the SQL type of a pandas
# object column FROM THE BATCH'S CONTENTS. Handed a column of Python dicts it
# produced three different types for the same `prediction_accuracy.metrics`
# column, measured 2026-08-05:
#
#   * STRUCT  — every dict in the batch had identical keys and value types,
#   * MAP(VARCHAR, DOUBLE) — a nested dict's key set differed between rows,
#     which `score_by_tier` guarantees (`mape_by_tier` carries only the tiers a
#     cohort actually has),
#   * VARCHAR holding a Python `repr` — it could reconcile neither.
#
# The file on disk holds ONE type, so the append has to cast whatever was
# inferred today into it, and raises when they disagree. Because the inference
# is data-dependent, so is the failure: the 2026-08-05 00:07 run passed and the
# 10:32 run failed on identical code, one raising `Could not convert string
# 'None' to DOUBLE` and the next `Type VARCHAR ... can't be cast to STRUCT(...)`.
#
# `_project`'s schema-drift handling cannot reach this. It NULLs out whole
# columns a side lacks, and `metrics` is ONE column — drift *inside* its struct
# is invisible to it, and always was.
#
# JSON text removes the inference step entirely: one stable scalar type, no
# frozen field list, and a new metric costs nothing. The cost is that inner
# fields need `metrics->>'$.mae'` rather than `metrics.mae`; DuckDB reads JSON
# natively, so they stay queryable.
_NESTED_TYPE_PREFIXES = ("STRUCT", "MAP")


def _is_nested_type(sql_type: str) -> bool:
    """True if *sql_type* is a DuckDB nested type, i.e. a pre-JSON column."""
    return sql_type == "JSON" or sql_type.endswith("[]") or sql_type.startswith(_NESTED_TYPE_PREFIXES)


def _json_default(value):
    """Convert what json.dumps cannot, and refuse what we don't understand.

    Deliberately not ``default=str``: stringifying an unrecognised value would
    turn a numpy float into the string ``"1.0"``, which round-trips as a string
    and silently corrupts the metric. numpy scalars and dates are converted;
    anything else raises.
    """
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if hasattr(value, "isoformat"):  # date / datetime
        return value.isoformat()
    raise TypeError(
        f"{type(value).__name__} is not JSON-serialisable and has no known "
        f"conversion; add one to _json_default rather than stringifying it."
    )


def _jsonify_nested(df: pd.DataFrame) -> pd.DataFrame:
    """Serialise dict/list-valued columns to JSON text.

    Does not mutate *df*, and does not mutate the values inside it — callers
    hand the SAME row dicts to the DB, whose columns really are JSON-typed and
    want the original objects. ``_upsert_accuracy`` does exactly that.

    Keys are sorted so that identical metrics serialise to identical bytes and a
    re-write of an unchanged row is a no-op rather than a diff. Ordering is
    otherwise cosmetic here: the value is opaque text on disk and comes back as
    a dict.
    """

    def _convert(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, default=_json_default)
        # A partially-populated nested column arrives with NaN or None in the
        # rows that have no value; both mean SQL NULL. Anything else is already
        # scalar (an already-serialised JSON string, typically) and is left be.
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return value

    nested = [c for c in df.columns if df[c].dtype == object and any(isinstance(v, (dict, list)) for v in df[c])]
    if not nested:
        return df
    df = df.copy()
    for c in nested:
        df[c] = [_convert(v) for v in df[c]]
    return df


def _append_parquet(path: Path, new_data: pd.DataFrame, dedup_keys: list[str]):
    """Append new_data to an existing Parquet file, deduplicating on dedup_keys.

    Uses DuckDB-native operations to avoid loading the full file into Python
    memory. The dedup is performed within DuckDB's engine via anti-join.

    SCHEMA. The output schema is the UNION of the file's columns and
    *new_data*': it widens, and it never narrows — the same rule
    ``replace_rows`` follows, for the same reason. An intersection silently
    drops a column the served file predates, and the dropped column is then
    also missing from ``dedup_keys``, so rows that differ only in that column
    become indistinguishable. That is not academic: ``prediction_accuracy``
    gained ``price_tier`` in migration 0019 and the unique constraint gained it
    in 0020, but the mirror predated both, so every write intersected it away
    and the served file carried six unlabelled rows per (horizon, model) —
    four price bands, the >=$1 headline, and the tick-dominated tier 0 — with
    no way for a reader to tell which was which.

    Missing columns are NULL on whichever side lacks them: a column only
    *new_data* has is NULL on the surviving existing rows, and a column only
    the file has is NULL on the appended rows. File column order is preserved,
    with added columns appended, so the caller's dict ordering does not get to
    reshuffle the served file.

    NESTED VALUES. Dict- and list-valued columns are written as JSON text, and a
    file that still holds them as a DuckDB nested type is converted in the same
    single rewrite — see the _jsonify_nested comment for why the nested types
    cannot be appended to at all. The migration is done here rather than by a
    script because ``price-archive/`` is gitignored: the served mirror exists
    only where the job runs, so no operator can be relied on to migrate it
    before the next unattended daily run.
    """
    new_data = _jsonify_nested(new_data)
    new_data = _coerce_dates(new_data)
    if not path.exists():
        new_data.to_parquet(path, index=False)
        return

    con = duckdb.connect()
    tmp = _tmp_path(path)
    try:
        con.register("_new", new_data)
        described = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        existing_cols = [r[0] for r in described]
        existing_nested = {r[0] for r in described if _is_nested_type(r[1])}
        new_cols = list(new_data.columns)
        # Union, file order first, so an existing served file keeps its layout.
        union_cols = existing_cols + [c for c in new_cols if c not in existing_cols]

        def _project(cols_present, alias, nested=frozenset()):
            """Select union_cols from a relation, NULLing what it lacks.

            Columns in *nested* are still a DuckDB nested type on this side and
            are converted to JSON text, so both sides of the UNION agree on
            VARCHAR and the file lands fully migrated.
            """

            def _one(c):
                if c not in cols_present:
                    return f"NULL AS {c}"
                if c in nested:
                    return f"CAST(to_json({alias}.{c}) AS VARCHAR) AS {c}"
                return f"{alias}.{c}"

            return ", ".join(_one(c) for c in union_cols)

        # Dedup on every key both sides carry. A key absent from the file is
        # still meaningful: those existing rows predate the column, so they
        # cannot match a new row on it and must survive.
        #
        # IS NOT DISTINCT FROM, not `=`: a NULL in a dedup key is a key VALUE
        # here, not "unknown". prediction_accuracy uses a NULL price_tier as the
        # discriminator for the all-tiers aggregate — the row every accuracy
        # endpoint serves by default — and horizon_days and model_version are
        # nullable too. Under `=` those rows compared NULL rather than TRUE, so
        # the anti-join kept the stale copy and the file gained one duplicate
        # aggregate row PER RUN, leaving "the latest row" arbitrary among them.
        # The DB side was never affected (filter_by(price_tier=None) emits
        # IS NULL), so this was one more way for the two stores to disagree.
        #
        # Latent until the mirror gained price_tier: while the column was absent
        # from the file it dropped out of usable_keys entirely and dedup ran on
        # the four non-NULL keys, which happened to be correct.
        usable_keys = [k for k in dedup_keys if k in new_cols and k in existing_cols]
        dedup_conditions = " AND ".join(f"_existing.{k} IS NOT DISTINCT FROM _new.{k}" for k in usable_keys)
        if not dedup_conditions:
            dedup_conditions = "1=0"

        con.execute(f"""
            COPY (
                SELECT {_project(new_cols, "_new")} FROM _new
                UNION ALL
                SELECT {_project(existing_cols, "_existing", existing_nested)}
                FROM read_parquet('{path}') _existing
                WHERE NOT EXISTS (
                    SELECT 1 FROM _new
                    WHERE {dedup_conditions}
                )
            ) TO '{tmp}' (FORMAT PARQUET)
        """)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        con.close()

    os.replace(tmp, path)


def append_monthly(
    out_dir: Path | str,
    prefix: str,
    df: pd.DataFrame,
    dedup_keys: list[str],
    day_col: str = "day",
):
    """Append rows to per-month Parquet files ``{prefix}-{YYYY-MM}.parquet``.

    This is the canonical partitioner for the price archive: current-era
    price/snapshot data is split by month so no single file approaches
    GitHub's 100 MB-per-file limit and the daily rewrite stays small. Each row
    is routed to the file for its own ``day_col`` month, so a frame spanning
    several months fans out correctly. Readers glob ``{prefix}-*.parquet``, so
    the monthly split is transparent to them (frozen pre-2026 yearly files,
    e.g. ``prices-2025.parquet``, still match that glob and coexist).
    """
    if df is None or df.empty:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df[day_col] = pd.to_datetime(df[day_col])
    for period, group in df.groupby(df[day_col].dt.strftime("%Y-%m")):
        _append_parquet(
            out_dir / f"{prefix}-{period}.parquet",
            group.reset_index(drop=True),
            dedup_keys,
        )


def read_table(table: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return all rows from *table* as a DataFrame."""
    path = _table_path(table)
    if not path.exists():
        return pd.DataFrame()
    con = duckdb.connect()
    try:
        col_spec = ", ".join(columns) if columns else "*"
        return con.sql(f"SELECT {col_spec} FROM read_parquet('{path}')").fetchdf()
    finally:
        con.close()


def append_table(table: str, rows: list[dict] | pd.DataFrame, dedup_keys: list[str]):
    """Append rows, deduplicating on *dedup_keys*."""
    if isinstance(rows, list):
        rows = pd.DataFrame(rows)
    if rows.empty:
        return
    path = _table_path(table)
    _append_parquet(path, rows, dedup_keys)


def replace_rows(
    table: str,
    key_column: str,
    delete_keys: Iterable,
    new_rows: list[dict] | pd.DataFrame | None = None,
):
    """Drop every row whose *key_column* is in *delete_keys*, then add *new_rows*.

    ``append_table`` can only append-with-dedup, so it has no way to express
    "this key no longer has a row at all". Re-resolving a cohort needs exactly
    that: a forecast that used to resolve and no longer does must end with NO
    mirrored row, not a stale one.

    Delete and insert are ONE rewrite, not two. The rows carried by *new_rows*
    are removed by key as well, so passing a key in both sets is a replace.

    CRASH SAFETY. The rewrite goes to a sibling temp file and is moved into
    place with ``os.replace``, which is atomic within a directory: a concurrent
    reader — and a process killed at any point — sees either the whole
    pre-state or the whole post-state of the table, never a half-written file
    and never a file with the deletes applied but the inserts missing.
    ``_append_parquet`` COPYs over the live path and does not have this
    property; it is left alone because appending is additive, while a delete
    that is interrupted mid-COPY would destroy rows nothing can rebuild.

    SCHEMA. The output schema is the UNION of the file's columns and
    *new_rows*': it widens, and it never narrows.

    * A column in *new_rows* that the file lacks is ADDED, NULL (typed from the
      incoming frame) on every surviving row. ``_append_parquet``'s
      intersection would have discarded it. That is not academic — the
      production ``forecast_outcomes`` mirror has 18 columns and no
      ``base_price``, while the re-resolve write carries 20. Under an
      intersection a ``--reresolve`` would rewrite the whole 65k-row served
      file *without the frozen actuals*, which are the entire point of the
      backfill.
    * A column the file has and *new_rows* lacks cannot be preserved on the
      rows being rewritten, so it raises rather than silently blanking them.
      Surviving rows always keep every column they had.

    File column order is preserved, with added columns appended — the caller's
    dict ordering does not get to reshuffle the served file.
    """
    if isinstance(new_rows, list) or new_rows is None:
        new_rows = pd.DataFrame(new_rows or [])

    keys = list(dict.fromkeys(delete_keys or []))
    if not new_rows.empty and key_column in new_rows.columns:
        keys = list(dict.fromkeys(keys + list(new_rows[key_column])))

    path = _table_path(table)

    if not path.exists():
        # Nothing to delete from; this degenerates to a first write.
        if new_rows.empty:
            return
        _atomic_write(path, _coerce_dates(new_rows))
        return

    if new_rows.empty and not keys:
        return

    new_rows = _jsonify_nested(new_rows)
    new_rows = _coerce_dates(new_rows)
    tmp = _tmp_path(path)
    con = duckdb.connect()
    try:
        described = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        existing_cols = [r[0] for r in described]
        # Still a DuckDB nested type on disk: converted to JSON text as the rows
        # are rewritten, the same migration _append_parquet performs.
        nested = {r[0] for r in described if _is_nested_type(r[1])}

        def _existing_col(c):
            if c in nested:
                return f"CAST(to_json(_existing.{c}) AS VARCHAR) AS {c}"
            return f"_existing.{c}"

        if new_rows.empty:
            out_cols = existing_cols
            existing_select = ", ".join(_existing_col(c) for c in existing_cols)
            new_select = ""
        else:
            missing = [c for c in existing_cols if c not in new_rows.columns]
            if missing:
                raise ValueError(
                    f"replace_rows('{table}'): incoming rows are missing "
                    f"column(s) {missing} that the file has. The rewritten "
                    f"rows would silently lose them. Supply the whole row."
                )
            con.register("_new", new_rows)
            new_types = {r[0]: r[1] for r in con.execute("DESCRIBE SELECT * FROM _new").fetchall()}
            added = [c for c in new_rows.columns if c not in existing_cols]
            out_cols = existing_cols + added
            new_select = f"SELECT {', '.join(out_cols)} FROM _new UNION ALL "
            existing_select = ", ".join(
                _existing_col(c) if c in existing_cols else f"CAST(NULL AS {new_types[c]}) AS {c}" for c in out_cols
            )

        if keys:
            con.register("_del", pd.DataFrame({"_k": keys}))
            keep = f"WHERE NOT EXISTS (SELECT 1 FROM _del WHERE _del._k = _existing.{key_column})"
        else:
            keep = ""

        con.execute(f"""
            COPY (
                {new_select}
                SELECT {existing_select} FROM read_parquet('{path}') _existing
                {keep}
            ) TO '{tmp}' (FORMAT PARQUET)
        """)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        con.close()

    os.replace(tmp, path)


def _tmp_path(path: Path) -> Path:
    """A writer-unique sibling temp name.

    Unique per writer, not just per table: a shared ``<table>.parquet.tmp``
    lets two concurrent writers interleave their COPYs into one file, and lets
    one writer's error path unlink the other's temp out from under it. Single
    writer today; the suffix removes the class rather than relying on that.
    """
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")


def _atomic_write(path: Path, df: pd.DataFrame):
    tmp = _tmp_path(path)
    try:
        df.to_parquet(tmp, index=False)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# DuckDB connection context — for ad-hoc queries in API routes
# ---------------------------------------------------------------------------


class ParquetQuery:
    """Wraps a DuckDB connection that reads from an ops table.

    Usage::

        with ParquetQuery("events") as con:
            df = con.sql("SELECT * FROM events WHERE type = 'major'").fetchdf()
    """

    def __init__(self, table: str):
        self._table = table
        self._con: duckdb.DuckDBPyConnection | None = None
        self._path: Path | None = None

    def __enter__(self):
        if not re.match(r"^[a-z_][a-z0-9_]*$", self._table):
            raise ValueError(f"Invalid table name: {self._table!r}")
        path = _table_path(self._table)
        if not path.exists():
            self._path = None
            return self
        self._path = path
        self._con = duckdb.connect()
        self._con.sql(f"CREATE VIEW {self._table} AS SELECT * FROM read_parquet('{path}')")
        return self

    def __exit__(self, *args):
        if self._con:
            self._con.close()

    @property
    def con(self):
        if self._con is None:
            msg = f"Table '{self._table}' not found at {self._path}"
            raise FileNotFoundError(msg)
        return self._con

    def df(self) -> pd.DataFrame:
        if self._con is None:
            return pd.DataFrame()
        return self._con.sql(f"SELECT * FROM {self._table}").fetchdf()

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        if self._con is None:
            return pd.DataFrame()
        return self._con.sql(sql, params=params if params else {}).fetchdf()

    def scalar(self, sql: str, params: dict | None = None):
        if self._con is None:
            return None
        r = self._con.sql(sql, params=params if params else {}).fetchone()
        return r[0] if r else None


def delete_table(table: str, key_filters: dict[str, Any]):
    """Delete rows matching *key_filters* and rewrite the file."""
    path = _table_path(table)
    if not path.exists():
        return
    df = read_table(table)
    if df.empty:
        return
    mask = pd.Series(True, index=df.index)
    for col, val in key_filters.items():
        if col in df.columns:
            mask &= df[col] == val
    df = df[~mask]
    _atomic_write(path, df)
