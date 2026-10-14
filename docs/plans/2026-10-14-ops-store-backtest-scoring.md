# Ops Store + Backtest Scoring (Phase 2, PR 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtest Accuracy reads the forecast record from the monthly Parquet partitions instead of Postgres, keeps scoring on a night Postgres is read-only, and logs every number it now reads from Parquet beside the Postgres value.

**Architecture:** A new `db/ops_store.py` opens in-memory DuckDB views over the partitions, named and typed like the Postgres tables, and returns rows with attribute access. Every backtest reader takes a `read(sql) -> rows` function, so the same reader runs on the store (the value used) and on Postgres (the read-twice comparison, removed in Phase 4). The backtest's writes catch SQLSTATE 25006 and land their rows in the partitions with ids from `READ_ONLY_ID_BASE`.

**Tech Stack:** Python 3.13 (3.11 in CI), DuckDB 1.5, SQLAlchemy, pandas, pytest, GitHub Actions.

**Spec:** `docs/specs/2026-10-13-ops-readers-switch-design.md`, sections "The read module", "Read twice", "PR sequence and gates" and "PR 1". Parent: `docs/specs/2026-10-11-postgres-to-parquet-history-design.md`.

## Global Constraints

- Storage only: no change to forecasts, bands, scoring logic or any metric's definition.
- **Merge gate:** nightly parity clean 3 nights in a row, counting from the night of 2026-10-13 (`clean nights in a row: 3` in the Backtest Accuracy log). Build and open the PR before that; merge only after.
- A switched reader always uses the Parquet value. No silent fallback to Postgres.
- `compare` never fails a run. A difference is a `::warning`.
- A table with no partition files raises `ops_store.MissingTable`, naming the table.
- Rows Postgres deleted (`pg_deleted IS TRUE`) are invisible through the store.
- The Postgres "already stored?" check in `_store_forecast_outcomes` stays on Postgres.
- Forecasts with `id >= ops_partitions.READ_ONLY_ID_BASE` (2**40) exist only in the partitions; their outcomes are never inserted into Postgres (its `forecast_id` is a 32-bit `Integer` with a foreign key).
- Frozen preregistered instruments are untouched: `measure_conformal_pid.py`, `ranking_transfer_report.py`, `centre_promotion_report.py`, `measure_direction_reopen.py`, `date_dependence_sensitivity.py`, and `backtest/candidate_resolution.py` (PR 2).
- Nothing runs against prod locally: `backend/.env` points at prod Supabase. Tests use in-memory SQLite and the hermetic `OPS_DIR` from conftest's `_hermetic_ops_dir`.
- Commands run from `backend/`. `$PY` is `venv/bin/python`; in a worktree without a venv use the main checkout's (`../../cs2-oracle/backend/venv/bin/python`).
- Before pushing, run the CI gate's own command: `$PY -m pytest tests -m "not slow" -q -n 4`. #118 passed serially and failed under xdist.
- No Claude/Codex attribution in commits or the PR.
- Work on branch `feat/ops-store-scoring` from `main`.

## Review Focus

1. **A partition write fails mid-run, then the scoring panel is read.** Expected: the panel reflects what the partitions hold (Postgres's new rows missing), the matured-count and headline read-twice warnings name the gap, and the next night's heal closes it. Pinned in Task 6 (`test_an_outcome_postgres_holds_but_the_partitions_lack_is_written_from_postgres`).
2. **A month file written by an older schema (column missing).** Expected: the column reads NULL for that month only; views still bind. Pinned in Task 1.
3. **Postgres unreachable while the backtest reads twice.** Expected: the comparison logs "Postgres read failed", rolls the session back, and scoring finishes on the partitions. Pinned in Task 2 (`test_a_failed_postgres_read_is_a_warning_and_rolls_back`).
4. **A read-only night followed by a writable one.** Expected: the read-only night's outcomes sit in the partitions with ids above 2**40; the next night does not re-resolve them (frozen in the partitions) and never inserts them into Postgres. Pinned in Task 6.
5. **The verdict refresh after the stores diverge.** Expected: the refresh diffs against the partitions, updates Postgres by id (a no-op for a partition-only id), and writes the partitions. Pinned in Task 4 (`test_the_verdict_refresh_reads_the_partitions`).

---

### Task 1: `db/ops_store.py` — views, rows, readers

**Files:**
- Create: `backend/db/ops_store.py`
- Test: `backend/tests/test_ops_store.py`

**Interfaces:**
- Consumes: `db.ops_partitions.SPECS`, `month_files(table, root)`, `ops_dir(root)`, `_sql_list(paths)`, `upsert(table, frame, *, allow_closed, root)`.
- Produces:
  - `TABLES: tuple[str, ...]` — the five table names.
  - `class MissingTable(RuntimeError)`.
  - `connect(root=None, tables=TABLES) -> duckdb.DuckDBPyConnection`.
  - `fetch(con, sql: str) -> list[NamedTuple]`.
  - `reader(tables=TABLES, root=None) -> Callable[[str], list]` — opens a fresh connection per call.
  - `check_columns(read, table: str, columns: Iterable[str]) -> None` — raises `MissingTable` when the table has rows and one of *columns* is NULL on every row.

- [ ] **Step 1: Write the failing tests**

```python
"""db/ops_store.py: the partitions read as if they were the Postgres tables (Phase 2)."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest
from db import ops_partitions, ops_store

DAY = date(2026, 9, 10)


def _forecast(fid, item_id=1, day=DAY, **extra):
    return {"id": fid, "item_id": item_id, "forecast_date": day, "horizon_days": 3, "price_mid": 10.0, **extra}


def _write(table, rows, root):
    ops_partitions.upsert(table, pd.DataFrame(rows), allow_closed=True, root=root)


def _all_five(root):
    _write("items", [{"id": 1, "item_id": "ak", "name": "ak", "type": "skin"}], root)
    _write("item_forecasts", [_forecast(10)], root)
    _write(
        "forecast_outcomes",
        [{"id": 20, "forecast_id": 10, "item_id": 1, "forecast_date": DAY, "horizon_days": 3, "actual_price": 11.0}],
        root,
    )
    _write(
        "forecast_candidates",
        [
            {
                "id": 30,
                "item_id": 1,
                "forecast_date": DAY,
                "horizon_days": 3,
                "component": "centre",
                "candidate_name": "naive",
                "candidate_version": "v1",
            }
        ],
        root,
    )
    _write("forecast_candidate_outcomes", [{"candidate_id": 30, "actual_price": 11.0, "_part": DAY}], root)


def test_views_are_named_and_typed_like_the_postgres_tables(tmp_path):
    _all_five(tmp_path)
    con = ops_store.connect(tmp_path)
    try:
        rows = ops_store.fetch(
            con,
            """
            SELECT i.item_id AS slug, f.price_mid, o.actual_price, co.actual_price AS cand_actual
            FROM forecast_outcomes o
            JOIN item_forecasts f ON f.id = o.forecast_id
            JOIN items i ON i.id = o.item_id
            JOIN forecast_candidates c ON c.item_id = f.item_id AND c.forecast_date = f.forecast_date
            JOIN forecast_candidate_outcomes co ON co.candidate_id = c.id
            """,
        )
    finally:
        con.close()
    assert len(rows) == 1
    assert (rows[0].slug, rows[0].price_mid, rows[0].actual_price, rows[0].cand_actual) == ("ak", 10.0, 11.0, 11.0)


def test_rows_have_attribute_access_and_python_types(tmp_path):
    _write("item_forecasts", [_forecast(10)], tmp_path)
    (row,) = ops_store.reader(("item_forecasts",), tmp_path)("SELECT * FROM item_forecasts")
    assert row.id == 10
    assert row.forecast_date == DAY
    assert isinstance(row.horizon_days, int)
    assert not hasattr(row, "pg_deleted")


def test_rows_postgres_deleted_are_hidden(tmp_path):
    _write("item_forecasts", [_forecast(10, pg_deleted=False), _forecast(11, item_id=2, pg_deleted=True)], tmp_path)
    rows = ops_store.reader(("item_forecasts",), tmp_path)("SELECT id FROM item_forecasts")
    assert [r.id for r in rows] == [10]


def test_a_column_an_older_month_lacks_reads_null(tmp_path):
    _write("item_forecasts", [_forecast(10, band_multiplier=1.5)], tmp_path)
    old = tmp_path / "item_forecasts" / "2026-08.parquet"
    duckdb.sql(
        "COPY (SELECT 11::BIGINT AS id, 2::BIGINT AS item_id, DATE '2026-08-10' AS forecast_date, "
        f"3::BIGINT AS horizon_days, 9.0::DOUBLE AS price_mid) TO '{old}' (FORMAT PARQUET)"
    )
    rows = ops_store.reader(("item_forecasts",), tmp_path)("SELECT id, band_multiplier FROM item_forecasts ORDER BY id")
    assert [(r.id, r.band_multiplier) for r in rows] == [(10, 1.5), (11, None)]


def test_a_column_no_file_has_reads_null(tmp_path):
    d = tmp_path / "item_forecasts"
    d.mkdir()
    duckdb.sql(
        "COPY (SELECT 11::BIGINT AS id, 2::BIGINT AS item_id, DATE '2026-08-10' AS forecast_date, "
        f"3::BIGINT AS horizon_days) TO '{d / '2026-08.parquet'}' (FORMAT PARQUET)"
    )
    (row,) = ops_store.reader(("item_forecasts",), tmp_path)("SELECT id, exceed_p FROM item_forecasts")
    assert row.exceed_p is None


def test_a_table_with_no_files_raises_naming_it(tmp_path):
    with pytest.raises(ops_store.MissingTable, match="forecast_outcomes"):
        ops_store.connect(tmp_path, ("forecast_outcomes",))


def test_only_the_named_tables_are_checked(tmp_path):
    _write("items", [{"id": 1, "item_id": "ak", "name": "ak", "type": "skin"}], tmp_path)
    assert ops_store.reader(("items",), tmp_path)("SELECT count(*) AS n FROM items")[0].n == 1


def test_a_reader_sees_files_written_after_it_was_made(tmp_path):
    _write("item_forecasts", [_forecast(10)], tmp_path)
    read = ops_store.reader(("item_forecasts",), tmp_path)
    assert len(read("SELECT id FROM item_forecasts")) == 1
    _write("item_forecasts", [_forecast(11, day=date(2026, 10, 1))], tmp_path)
    assert len(read("SELECT id FROM item_forecasts")) == 2


def test_check_columns_raises_when_a_required_column_is_all_null(tmp_path):
    _write("forecast_outcomes", [{"forecast_id": 10, "item_id": 1, "forecast_date": DAY, "horizon_days": 3}], tmp_path)
    read = ops_store.reader(("forecast_outcomes",), tmp_path)
    with pytest.raises(ops_store.MissingTable, match="forecast_outcomes.id"):
        ops_store.check_columns(read, "forecast_outcomes", ["id", "forecast_id"])
    ops_store.check_columns(read, "forecast_outcomes", ["forecast_id"])


def test_check_columns_accepts_an_empty_table(tmp_path):
    d = tmp_path / "forecast_outcomes"
    d.mkdir()
    duckdb.sql(f"COPY (SELECT 1::BIGINT AS id, 1::BIGINT AS forecast_id WHERE false) TO '{d / '2026-09.parquet'}' (FORMAT PARQUET)")
    ops_store.check_columns(ops_store.reader(("forecast_outcomes",), tmp_path), "forecast_outcomes", ["id"])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_ops_store.py -q`
Expected: FAIL, `ImportError: cannot import name 'ops_store' from 'db'`.

- [ ] **Step 3: Write `backend/db/ops_store.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_ops_store.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/db/ops_store.py backend/tests/test_ops_store.py
git commit -m "feat: ops_store — the monthly partitions as DuckDB views named like the Postgres tables"
```

---

### Task 2: `ops_store.compare` — read twice

**Files:**
- Modify: `backend/db/ops_store.py` (append)
- Test: `backend/tests/test_ops_store.py` (append)

**Interfaces:**
- Produces: `compare(name: str, pg: Callable[[], object], parquet_value, *, tol: float = 0.0, db=None) -> bool` — True when equal. Values are scalars or flat dicts of scalars.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ops_store.py`)

```python
def test_equal_values_compare_quietly(capsys):
    assert ops_store.compare("n", lambda: 3, 3) is True
    assert "::warning" not in capsys.readouterr().out


def test_a_difference_is_a_warning_naming_the_reader_and_both_values(capsys):
    assert ops_store.compare("matured forecasts", lambda: 3, 4) is False
    out = capsys.readouterr().out
    assert "::warning title=Read twice: matured forecasts::" in out
    assert "postgres=3" in out and "partitions=4" in out
    assert "Postgres was read-only" in out


def test_dicts_compare_key_by_key_and_name_each_differing_key(capsys):
    assert ops_store.compare("per date", lambda: {"2026-10-01": 5, "2026-10-02": 6}, {"2026-10-01": 5, "2026-10-03": 6}) is False
    out = capsys.readouterr().out
    assert "2026-10-02: postgres=6 partitions=None" in out
    assert "2026-10-03: postgres=None partitions=6" in out
    assert "2026-10-01" not in out


def test_tol_absorbs_float_noise_but_not_none():
    assert ops_store.compare("x", lambda: 1.0, 1.0 + 1e-12, tol=1e-9) is True
    assert ops_store.compare("x", lambda: 1.0, 1.1, tol=1e-9) is False
    assert ops_store.compare("x", lambda: None, 0.0, tol=1.0) is False


def test_a_failed_postgres_read_is_a_warning_and_rolls_back(capsys):
    class _DB:
        rolled_back = False

        def rollback(self):
            self.rolled_back = True

    db = _DB()

    def pg():
        raise RuntimeError("server closed the connection")

    assert ops_store.compare("n", pg, 3, db=db) is False
    assert db.rolled_back
    assert "::warning title=Read twice: n::Postgres read failed (RuntimeError)" in capsys.readouterr().out


def test_more_than_ten_differences_are_truncated(capsys):
    ops_store.compare("d", lambda: {i: 0 for i in range(15)}, {i: 1 for i in range(15)})
    assert "(+5 more)" in capsys.readouterr().out
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_ops_store.py -q -k "compare or warning or dicts or tol or failed or truncated"`
Expected: FAIL, `AttributeError: module 'db.ops_store' has no attribute 'compare'`.

- [ ] **Step 3: Append to `backend/db/ops_store.py`**

```python
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
```

- [ ] **Step 4: Run the whole file**

Run: `$PY -m pytest tests/test_ops_store.py -q`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/db/ops_store.py backend/tests/test_ops_store.py
git commit -m "feat: ops_store.compare — log the Parquet value beside Postgres, warn on a difference, never fail"
```

---

### Task 3: test helper — mirror a test's SQLite tables into the partitions

The backtest tests seed SQLite through the ORM. Once the readers switch, those rows must also be in the partitions. This helper copies them, and an opt-in fixture copies them before every `ops_store.connect`, so the existing tests keep their meaning without being rewritten.

**Files:**
- Create: `backend/tests/_ops_mirror.py`
- Modify: `backend/tests/conftest.py` (add fixture `ops_mirror`)
- Test: `backend/tests/test_ops_mirror_helper.py`

**Interfaces:**
- Consumes: `ops_store.TABLES`, `ops_store.connect`, `ops_partitions.SPECS / upsert / ops_dir`.
- Produces:
  - `tests._ops_mirror.mirror_session_to_ops(session, root=None, tables=TABLES) -> None` — replaces the partitions for *tables* with the session's rows (pending rows included); an empty table becomes one empty, typed file.
  - conftest fixture `ops_mirror` — opt-in (`pytestmark = pytest.mark.usefixtures("ops_mirror")`); needs the module to define a `session` fixture.

- [ ] **Step 1: Write the failing tests** (`tests/test_ops_mirror_helper.py`)

```python
"""tests/_ops_mirror.py: a test's SQLite rows, copied into the hermetic ops partitions."""

from __future__ import annotations

from datetime import date

import pytest
from database import Base, ForecastOutcome, Item, ItemForecast
from db import ops_store
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from tests._ops_mirror import mirror_session_to_ops


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)()
    engine.dispose()


def _seed(session):
    session.add(Item(id=1, item_id="ak", name="ak", type="skin"))
    session.add(ItemForecast(id=10, item_id=1, forecast_date=date(2026, 7, 1), horizon_days=3, price_mid=3.0))
    session.add(
        ForecastOutcome(
            id=20, forecast_id=10, item_id=1, forecast_date=date(2026, 7, 1), horizon_days=3,
            target_date=date(2026, 7, 4), predicted_price_mid=3.0, actual_price=3.1, abs_error=0.1,
        )
    )


def test_seeded_rows_read_back_through_the_store(session):
    _seed(session)  # not committed: pending rows count
    mirror_session_to_ops(session)
    rows = ops_store.reader()("SELECT o.id, f.price_mid, i.item_id FROM forecast_outcomes o JOIN item_forecasts f ON f.id = o.forecast_id JOIN items i ON i.id = o.item_id")
    assert [(r.id, r.price_mid, r.item_id) for r in rows] == [(20, 3.0, "ak")]


def test_an_empty_table_is_an_empty_view_not_a_missing_one(session):
    mirror_session_to_ops(session)
    assert ops_store.reader()("SELECT count(*) AS n FROM forecast_outcomes")[0].n == 0


def test_a_second_mirror_replaces_rather_than_merges(session):
    _seed(session)
    session.commit()
    mirror_session_to_ops(session)
    session.query(ForecastOutcome).delete()
    session.commit()
    mirror_session_to_ops(session)
    assert ops_store.reader()("SELECT count(*) AS n FROM forecast_outcomes")[0].n == 0


def test_it_refuses_a_directory_outside_pytest_tmp(session, tmp_path):
    from pathlib import Path

    with pytest.raises(RuntimeError, match="pytest"):
        mirror_session_to_ops(session, root=Path("/var/not-a-test-dir/ops"))


@pytest.mark.usefixtures("ops_mirror")
def test_the_fixture_mirrors_before_every_connect(session):
    _seed(session)
    session.commit()
    assert len(ops_store.reader(("items",))("SELECT id FROM items")) == 1
    session.add(Item(id=2, item_id="awp", name="awp", type="skin"))
    session.commit()
    assert len(ops_store.reader(("items",))("SELECT id FROM items")) == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_ops_mirror_helper.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tests._ops_mirror'`.

- [ ] **Step 3: Write `backend/tests/_ops_mirror.py`**

```python
"""Copy a test's SQLite tables into the hermetic ops partitions.

The backtest tests seed SQLite through the ORM. Since Phase 2 the backtest reads the forecast
record from the partitions, so those rows have to be there too. `mirror_session_to_ops` replaces
the partitions for each table with the session's rows; conftest's `ops_mirror` fixture calls it
before every `ops_store.connect`, so the store always equals the test's SQLite. Tests that need
the stores to DIFFER call this once and then change one side, without the fixture.

Closed months are written (allow_closed): the tests' dates are mostly July 2026.
"""

from __future__ import annotations

import shutil

import duckdb
import pandas as pd
from db import ops_partitions
from db.ops_store import TABLES
from sqlalchemy import text


def _empty(table: str, root) -> None:
    spec = ops_partitions.SPECS[table]
    d = ops_partitions.ops_dir(root)
    path = d / f"{table}.parquet" if spec.part_col is None else d / table / "1970-01.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ", ".join(f"CAST(NULL AS {t}) AS {c}" for c, t in spec.columns.items())
    con = duckdb.connect()
    try:
        con.execute(f"COPY (SELECT {cols} WHERE false) TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()


def mirror_session_to_ops(session, root=None, tables=TABLES) -> None:
    d = ops_partitions.ops_dir(root)
    if "pytest-" not in str(d):
        raise RuntimeError(f"mirror_session_to_ops writes only under a pytest tmp dir, not {d}")
    session.flush()
    conn = session.connection()
    for t in tables:
        spec = ops_partitions.SPECS[t]
        target = d / f"{t}.parquet" if spec.part_col is None else d / t
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        if t == "forecast_candidate_outcomes":
            sql = (
                "SELECT o.*, c.forecast_date AS _part FROM forecast_candidate_outcomes o "
                "JOIN forecast_candidates c ON c.id = o.candidate_id"
            )
        else:
            sql = f"SELECT * FROM {t}"
        frame = pd.read_sql(text(sql), conn)
        frame = frame[[c for c in frame.columns if c in spec.columns or c == "_part"]]
        if frame.empty:
            _empty(t, root)
        else:
            ops_partitions.upsert(t, frame, allow_closed=True, root=root)
```

- [ ] **Step 4: Add the fixture to `backend/tests/conftest.py`** (after `_hermetic_ops_dir`)

```python
@pytest.fixture()
def ops_mirror(request, monkeypatch):
    """Opt-in: before every ``ops_store.connect``, copy the test's ``session`` into ``OPS_DIR``.

    Since Phase 2 the backtest reads the forecast record from the partitions; this keeps tests
    that seed SQLite through the ORM meaning what they meant. Needs a ``session`` fixture in the
    module. Tests where the two stores must differ use ``tests._ops_mirror`` directly instead.
    """
    import db.ops_store as store
    from tests._ops_mirror import mirror_session_to_ops

    session = request.getfixturevalue("session")
    real = store.connect

    def connect(root=None, tables=store.TABLES):
        mirror_session_to_ops(session, root, tables)
        return real(root, tables)

    monkeypatch.setattr(store, "connect", connect)
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_ops_mirror_helper.py -q`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/tests/_ops_mirror.py backend/tests/conftest.py backend/tests/test_ops_mirror_helper.py
git commit -m "test: mirror a test's SQLite tables into the hermetic ops partitions; opt-in ops_mirror fixture"
```

---

### Task 4: switch the backtest's reads to the store

**Files:**
- Modify: `backend/scripts/backtest_accuracy.py` — `_iter_outcome_rows`, `_id_to_slug`, `_refresh_verdict_columns`, `_store_forecast_outcomes` (the `_id_to_slug` default only), `_records_from_frozen_outcomes`, `backtest_forecasts`; add `_pg_read`, `_store_read`, `_FORECASTS_SQL`, `_FROZEN_PANEL_SQL`, `_as_date`, `_mature`.
- Modify tests: `backend/tests/test_backtest_scoring.py`, `backend/tests/test_ops_item_slug.py`
- Test: `backend/tests/test_backtest_reads_partitions.py`

**Interfaces:**
- Consumes: `ops_store.reader`, `ops_store.check_columns`; `tests._ops_mirror.mirror_session_to_ops`; fixture `ops_mirror`.
- Produces (used by Tasks 5 and 6):
  - `_pg_read(db) -> Callable[[str], list]`
  - `_store_read(*tables, required=None) -> Callable[[str], list]`; `_panel_read()` (outcomes + forecasts, NULL-id check)
  - `_FORECASTS_SQL: str`, `_mature(rows, cutoff) -> list`, `_as_date(value) -> date`
  - `_id_to_slug(read) -> dict[int, str]` (was `_id_to_slug(db)`)
  - `_records_from_frozen_outcomes(read, min_price=0, forecast_ids=None)` (was `(db, ...)`)
  - `_iter_outcome_rows(read, forecast_ids=None, batch=None)` (was `(db, ...)`)

- [ ] **Step 1: Write the failing tests** (`tests/test_backtest_reads_partitions.py`)

```python
"""Phase 2 PR 1: the backtest reads the forecast record from the partitions, not Postgres."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from database import Base, ForecastOutcome, Item, ItemForecast
from db import ops_partitions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from tests._ops_mirror import mirror_session_to_ops

# An open month, so the backtest's own partition writes are not refused as closed.
DAY = date.today() - timedelta(days=10)


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)()
    engine.dispose()


def _seed(session, n=3, in_interval=1):
    for k in range(1, n + 1):
        session.add(Item(id=k, item_id=f"item_{k}", name=f"item {k}", type="skin"))
        session.add(
            ItemForecast(
                id=100 + k, item_id=k, forecast_date=DAY, horizon_days=3, price_low=2.5, price_mid=3.2,
                price_high=3.6, current_price=3.0, direction="up", confidence="high", model_version="lgbm-test",
            )
        )
        session.add(
            ForecastOutcome(
                id=200 + k, forecast_id=100 + k, item_id=k, forecast_date=DAY, horizon_days=3,
                target_date=DAY + timedelta(days=3), current_price=3.0, base_price=3.0,
                predicted_price_low=2.5, predicted_price_mid=3.2, predicted_price_high=3.6,
                actual_price=3.3, direction_predicted="up", direction_actual="up", direction_correct=1,
                in_interval=in_interval, abs_error=0.1, pct_error=3.0, model_version="lgbm-test",
            )
        )
    session.commit()


def _panel_n(groups):
    return sum(len(v) for v in groups.values())


def test_the_scoring_panel_reads_the_partitions_not_postgres(session):
    from scripts import backtest_accuracy as ba

    _seed(session)
    mirror_session_to_ops(session)
    session.query(ForecastOutcome).delete()
    session.commit()
    assert _panel_n(ba._records_from_frozen_outcomes(ba._panel_read())) == 3
    assert _panel_n(ba._records_from_frozen_outcomes(ba._pg_read(session))) == 0


def test_an_outcome_postgres_deleted_is_not_scored(session):
    from scripts import backtest_accuracy as ba

    _seed(session, n=1)
    mirror_session_to_ops(session)
    ops_partitions.upsert("forecast_outcomes", pd.DataFrame([{"forecast_id": 101, "item_id": 1, "forecast_date": DAY, "horizon_days": 3, "pg_deleted": True}]))
    assert _panel_n(ba._records_from_frozen_outcomes(ba._panel_read())) == 0


def test_both_stores_score_identically_on_the_same_rows(session):
    from scripts import backtest_accuracy as ba

    _seed(session)
    mirror_session_to_ops(session)
    pg = ba._score_groups(ba._records_from_frozen_outcomes(ba._pg_read(session)), DAY)
    pq = ba._score_groups(ba._records_from_frozen_outcomes(ba._panel_read()), DAY)
    key = lambda r: (r["horizon_days"], r["model_version"], r["price_tier"], r["sample_count"], r["metrics"].get("directional_accuracy"), r["metrics"].get("interval_coverage"))  # noqa: E731
    assert [key(r) for r in pq] == [key(r) for r in pg]
    assert pq


def test_id_to_slug_reads_items_from_the_store(session):
    from scripts import backtest_accuracy as ba

    _seed(session, n=2)
    mirror_session_to_ops(session)
    session.query(Item).delete()
    session.commit()
    assert ba._id_to_slug(ba._store_read("items")) == {1: "item_1", 2: "item_2"}


def test_the_verdict_refresh_reads_the_partitions(session):
    from scripts import backtest_accuracy as ba

    _seed(session, n=1, in_interval=0)  # 3.3 is inside [2.5, 3.6]: the stored 0 is stale
    mirror_session_to_ops(session)
    assert ba._refresh_verdict_columns(session) == 1
    assert session.query(ForecastOutcome).one().in_interval == 1
    (row,) = ba._store_read("forecast_outcomes")("SELECT in_interval FROM forecast_outcomes")
    assert row.in_interval == 1
    # The partitions now agree with current scoring, so a second run finds nothing to do.
    assert ba._refresh_verdict_columns(session) == 0


def test_rescore_scores_from_the_partitions_when_postgres_lacks_the_rows(session):
    from scripts import backtest_accuracy as ba

    _seed(session)
    mirror_session_to_ops(session)
    session.query(ForecastOutcome).delete()
    session.query(ItemForecast).delete()
    session.commit()
    results = ba.backtest_forecasts(session, today=DAY + timedelta(days=5), rescore=True)
    assert any(r["sample_count"] == 3 for r in results)


def test_an_all_null_outcome_id_fails_loudly(session):
    from db.ops_store import MissingTable
    from scripts import backtest_accuracy as ba

    ops_partitions.upsert("forecast_outcomes", pd.DataFrame([{"forecast_id": 101, "item_id": 1, "forecast_date": DAY, "horizon_days": 3}]))
    ops_partitions.upsert("item_forecasts", pd.DataFrame([{"id": 101, "item_id": 1, "forecast_date": DAY, "horizon_days": 3}]))
    with pytest.raises(MissingTable, match="forecast_outcomes.id"):
        ba._records_from_frozen_outcomes(ba._panel_read())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_backtest_reads_partitions.py -q`
Expected: FAIL, `AttributeError: module 'scripts.backtest_accuracy' has no attribute '_store_read'`.

- [ ] **Step 3: Add the readers to `backend/scripts/backtest_accuracy.py`** (just above `_iter_outcome_rows`)

```python
def _pg_read(db):
    """Postgres as a `sql -> rows` reader. Since Phase 2 it is only the read-twice side."""
    return lambda sql: db.execute(text(sql)).fetchall()


def _store_read(*tables, required=None):
    """The monthly ops partitions as a `sql -> rows` reader (db/ops_store.py).

    *required*: {table: columns} that must not be NULL on every row; checked on the first call.
    """
    from db import ops_store

    read = ops_store.reader(tables)
    if not required:
        return read
    checked = False

    def checked_read(sql):
        nonlocal checked
        if not checked:
            for table, cols in required.items():
                ops_store.check_columns(read, table, cols)
            checked = True
        return read(sql)

    return checked_read


def _panel_read():
    """The scoring panel's store reader; an outcome id that is NULL everywhere fails loudly."""
    return _store_read("forecast_outcomes", "item_forecasts", required={"forecast_outcomes": ["id", "forecast_id"]})


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
```

- [ ] **Step 4: Rewrite `_iter_outcome_rows`**

Replace the whole function (signature, docstring and body) with:

```python
def _iter_outcome_rows(read, forecast_ids=None, batch=None):
    """Yield pages of at most *batch* forecast_outcomes rows, in id order.

    *read* is a `sql -> rows` reader (`_store_read`). The table is read whole BEFORE the first
    page is yielded: the refresh rewrites partition files between pages, and a scan still open
    on those files could see the new ones mid-walk. Pages only bound how much the refresh
    accumulates before a flush.

    A *forecast_ids* restriction is applied in Python, not as SQL IN lists: the daily caller
    passes every mature forecast, which is nearly the whole table. A page whose rows all fall
    outside the restriction is skipped, never yielded empty.
    """
    batch = batch or OUTCOME_PAGE
    keep = None if forecast_ids is None else set(forecast_ids)
    rows = read("SELECT * FROM forecast_outcomes ORDER BY id")
    for i in range(0, len(rows), batch):
        page = rows[i : i + batch]
        if keep is not None:
            page = [r for r in page if r.forecast_id in keep]
        if page:
            yield page
```

Then remove `select` from the `from sqlalchemy import select, text` line if nothing else uses it (`grep -n "select(" backend/scripts/backtest_accuracy.py`).

- [ ] **Step 5: Rewrite `_id_to_slug`**

```python
def _id_to_slug(read) -> dict:
    """Surrogate `items.id` -> `items.item_id`, the market_hash_name slug.

    The price archive keys on `item_slug`; the ops tables key on `item_id`. *read* is a
    `sql -> rows` reader; the backtest passes `_store_read("items")`.
    """
    return {r.id: r.item_id for r in read("SELECT id, item_id FROM items")}
```

Update its callers:
- In `_refresh_verdict_columns`: `id_to_slug = _id_to_slug(db)` → `id_to_slug = _id_to_slug(_store_read("items"))`, and `for batch in _iter_outcome_rows(db, forecast_ids):` → `for batch in _iter_outcome_rows(_store_read("forecast_outcomes"), forecast_ids):`. Add one sentence to its docstring: "The stored verdicts are read from the partitions (Phase 2); the UPDATE still goes to Postgres by id, and is a no-op for a partition-only id."
- In `_store_forecast_outcomes`: `id_to_slug = _id_to_slug(db)` → `id_to_slug = _id_to_slug(_store_read("items"))`.
- In `backtest_forecasts`, replace

```python
    id_to_slug = {}
    if to_resolve:
        slug_rows = db.execute(text("SELECT id, item_id FROM items")).fetchall()
        id_to_slug = {r.id: r.item_id for r in slug_rows}
```

with

```python
    id_to_slug = _id_to_slug(_store_read("items")) if to_resolve else {}
```

- [ ] **Step 6: Rewrite `_records_from_frozen_outcomes`' read**

Change the signature to `def _records_from_frozen_outcomes(read, min_price=0, forecast_ids=None):`. In the docstring, replace the paragraph starting `forecast_ids: optional restriction` with:

```
    read: a `sql -> rows` reader — `_panel_read()` for the
        published figure, `_pg_read(db)` for the read-twice comparison. One query, no
        parameters, so the same SQL runs on both.

    forecast_ids: optional restriction to a set of forecast ids, applied in Python (an IN list
        over nearly the whole table is the slow path #113 removed). Rows come back in
        `forecast_outcomes.id` order either way.
```

Add above the function:

```python
_FROZEN_PANEL_SQL = """
    SELECT o.id, o.forecast_id, o.item_id, o.horizon_days, o.model_version,
           o.forecast_date, o.base_price, o.actual_price, o.current_price,
           o.predicted_price_low, o.predicted_price_mid,
           o.predicted_price_high, o.direction_predicted, f.confidence,
           f.model_version AS forecast_model_version
    FROM forecast_outcomes o
    LEFT JOIN item_forecasts f ON f.id = o.forecast_id
    ORDER BY o.id
"""
```

and replace the body from `page_sql = text("""` through the end of the `while True:` loop (the line `break` after `if len(page) < OUTCOME_PAGE:`) with:

```python
    keep = None if forecast_ids is None else set(forecast_ids)
    rows = read(_FROZEN_PANEL_SQL)
    if keep is not None:
        rows = [r for r in rows if r.forecast_id in keep]
```

The NULL-id check lives in `_panel_read` (Step 3), so the Postgres side of read twice runs the same function unchecked.

Update the two callers in `backtest_forecasts`:
- rescore: `groups = _records_from_frozen_outcomes(db, min_price=min_price)` → `groups = _records_from_frozen_outcomes(_panel_read(), min_price=min_price)`
- daily: `groups = _records_from_frozen_outcomes(db, min_price=min_price, forecast_ids=mature_ids)` → `groups = _records_from_frozen_outcomes(_panel_read(), min_price=min_price, forecast_ids=mature_ids)`

- [ ] **Step 7: Switch the matured-forecast and frozen-id reads in `backtest_forecasts`**

Add at module level, near `_FROZEN_PANEL_SQL`:

```python
_FORECASTS_SQL = """
    SELECT f.id, f.item_id, f.forecast_date, f.horizon_days,
           f.price_low, f.price_mid, f.price_high,
           f.current_price, f.direction, f.confidence, f.model_version
    FROM item_forecasts f
    WHERE f.price_mid IS NOT NULL
"""


def _mature(rows, cutoff):
    """The forecasts whose forecast_date + horizon is on or before *cutoff*."""
    return [r for r in rows if _as_date(r.forecast_date) + timedelta(days=r.horizon_days) <= cutoff]
```

In `backtest_forecasts`, replace from `# Fetch all forecasts with a midpoint price, filter for maturity in Python` through the end of the `for r in rows:` maturity loop with:

```python
    # Every forecast with a midpoint price, from the partitions; maturity is filtered in Python.
    mature = _mature(_store_read("item_forecasts", required={"item_forecasts": ["id"]})(_FORECASTS_SQL), cutoff)
```

and replace the frozen-id read

```python
        frozen_ids = {
            fid for (fid,) in db.execute(text("SELECT forecast_id FROM forecast_outcomes")) if fid in mature_set
        }
```

with

```python
        frozen_ids = {
            r.forecast_id
            for r in _store_read("forecast_outcomes")("SELECT forecast_id FROM forecast_outcomes")
            if r.forecast_id in mature_set
        }
```

Update the comment above it: "Frozen means frozen in the partitions. An outcome Postgres holds but the partitions lack is resolved again here and written to the partitions from Postgres by `_store_forecast_outcomes`, never re-inserted."

- [ ] **Step 8: Run the new tests**

Run: `$PY -m pytest tests/test_backtest_reads_partitions.py -q`
Expected: 7 passed.

- [ ] **Step 9: Move the existing tests onto the store**

In `backend/tests/test_backtest_scoring.py`, after the imports add:

```python
# Since Phase 2 the backtest reads the partitions; mirror each test's SQLite rows into them.
pytestmark = pytest.mark.usefixtures("ops_mirror")
```

and change the direct reader calls:
- `backtest_accuracy._records_from_frozen_outcomes(session)` → `backtest_accuracy._records_from_frozen_outcomes(backtest_accuracy._panel_read())` (two sites: the UNUSABLE-hint test and `test_frozen_outcome_records_carry_the_prediction_leg_and_the_horizon`).
- `backtest_accuracy._records_from_frozen_outcomes(session, forecast_ids=...)` → `backtest_accuracy._records_from_frozen_outcomes(backtest_accuracy._panel_read(), forecast_ids=...)` (three calls in `test_frozen_outcome_query_is_restricted_in_sql_not_in_python`). Rename that test to `test_frozen_outcome_query_honours_the_forecast_id_restriction` and its docstring to "The forecast_ids restriction returns exactly the requested forecasts, including past the old 999-parameter IN-list cap."

In `backend/tests/test_ops_item_slug.py`, add the same `pytestmark` line, import `_pg_read` from `scripts.backtest_accuracy`, and change `_id_to_slug(session)` → `_id_to_slug(_pg_read(session))` at its four sites (these tests pin the mapping, not the store).

Run: `$PY -m pytest tests/test_backtest_scoring.py tests/test_ops_item_slug.py tests/test_accuracy_mirror_write_order.py tests/test_interval_score.py tests/test_ops_dual_write_outcomes.py tests/test_gate_archive_gap.py -q`
Expected: all pass. If a module not listed here fails with `MissingTable`, it reaches a switched reader with SQLite-only rows: add the same `pytestmark` line (it needs a `session` fixture; if its fixture has another name, add `@pytest.fixture() def session(<its name>): return <its name>`).

- [ ] **Step 10: Commit**

```bash
git add backend/scripts/backtest_accuracy.py backend/tests/test_backtest_reads_partitions.py backend/tests/test_backtest_scoring.py backend/tests/test_ops_item_slug.py
git commit -m "feat: backtest scoring, maturity, frozen ids, slugs and the verdict refresh read the ops partitions"
```

---

### Task 5: read twice in the backtest

**Files:**
- Modify: `backend/scripts/backtest_accuracy.py` — add `_per_date`, `_headline_values`, `_pg_headlines`, `_quiet`; call `ops_store.compare` in `backtest_forecasts`.
- Test: `backend/tests/test_backtest_reads_partitions.py` (append)

**Interfaces:**
- Consumes: `ops_store.compare`, `_pg_read`, `_FORECASTS_SQL`, `_mature`, `_records_from_frozen_outcomes`, `score_by_tier`, `HEADLINE_TIER`.
- Produces: `_per_date(rows) -> dict[str, int]`; `_headline_values(results) -> dict[str, float | int | None]`; `_pg_headlines(db, min_price, forecast_ids) -> dict`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_matured_counts_match_when_the_stores_agree(session, capsys):
    from scripts import backtest_accuracy as ba

    _seed(session)
    mirror_session_to_ops(session)
    ba._read_twice_matured(session, DAY + timedelta(days=5))
    assert "::warning" not in capsys.readouterr().out


def test_a_forecast_only_postgres_holds_is_named_by_date(session, capsys):
    from scripts import backtest_accuracy as ba

    _seed(session, n=2)
    mirror_session_to_ops(session)
    session.add(Item(id=9, item_id="item_9", name="item 9", type="skin"))
    session.add(ItemForecast(id=109, item_id=9, forecast_date=DAY, horizon_days=3, price_mid=3.0))
    session.commit()
    ba._read_twice_matured(session, DAY + timedelta(days=5))
    out = capsys.readouterr().out
    assert "::warning title=Read twice: matured forecasts per forecast date::" in out
    assert f"{DAY}: postgres=3 partitions=2" in out


def test_headlines_match_when_the_stores_agree(session, capsys):
    from scripts import backtest_accuracy as ba

    _seed(session)
    mirror_session_to_ops(session)
    results = ba._score_groups(ba._records_from_frozen_outcomes(ba._panel_read()), DAY)
    ba._read_twice_headlines(session, results, 0, None)
    assert "::warning" not in capsys.readouterr().out


def test_a_headline_difference_is_named(session, capsys):
    from scripts import backtest_accuracy as ba

    _seed(session)
    mirror_session_to_ops(session)
    results = ba._score_groups(ba._records_from_frozen_outcomes(ba._panel_read()), DAY)
    session.query(ForecastOutcome).filter(ForecastOutcome.id == 201).delete()
    session.commit()
    ba._read_twice_headlines(session, results, 0, None)
    out = capsys.readouterr().out
    assert "::warning title=Read twice: headline metrics per horizon::" in out
    assert "3d/lgbm-test n: postgres=2 partitions=3" in out
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_backtest_reads_partitions.py -q -k "matured or headline"`
Expected: FAIL, `AttributeError: ... '_read_twice_matured'`.

- [ ] **Step 3: Add the helpers** (below `_mature`)

```python
def _per_date(rows) -> dict:
    """Row count per forecast date, keyed by ISO date: the matured-forecast read-twice value."""
    out = defaultdict(int)
    for r in rows:
        out[_as_date(r.forecast_date).isoformat()] += 1
    return dict(out)


def _read_twice_matured(db, cutoff, mature=None) -> None:
    """Matured forecasts per forecast date, partitions beside Postgres."""
    from db import ops_store

    if mature is None:
        mature = _mature(_store_read("item_forecasts")(_FORECASTS_SQL), cutoff)
    ops_store.compare(
        "matured forecasts per forecast date",
        lambda: _per_date(_mature(_pg_read(db)(_FORECASTS_SQL), cutoff)),
        _per_date(mature),
        db=db,
    )


_HEADLINE_METRICS = ("directional_accuracy", "interval_coverage")


def _headline_values(results) -> dict:
    """The headline tier's sample count and metrics per horizon and cohort, flattened for compare."""
    out = {}
    for r in results:
        if r["price_tier"] != HEADLINE_TIER:
            continue
        key = f"{r['horizon_days']}d/{r['model_version']}"
        out[f"{key} n"] = r["sample_count"]
        for m in _HEADLINE_METRICS:
            out[f"{key} {m}"] = r["metrics"].get(m)
    return out


@contextmanager
def _quiet():
    """Silence this module's logger: the Postgres side of read twice must not repeat the run's log."""
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


def _pg_headlines(db, min_price, forecast_ids) -> dict:
    with _quiet():
        groups = _records_from_frozen_outcomes(_pg_read(db), min_price=min_price, forecast_ids=forecast_ids)
        rows = [
            {"horizon_days": h, "model_version": mv, "price_tier": t, "sample_count": n, "metrics": m}
            for (h, mv), records in sorted(groups.items())
            for t, m, n in score_by_tier(records)
        ]
    return _headline_values(rows)


def _read_twice_headlines(db, results, min_price, forecast_ids) -> None:
    """The stored headline per horizon, partitions beside Postgres."""
    from db import ops_store

    ops_store.compare(
        "headline metrics per horizon",
        lambda: _pg_headlines(db, min_price, forecast_ids),
        _headline_values(results),
        tol=1e-9,
        db=db,
    )
```

Add `from contextlib import contextmanager` to the imports. Confirm `score_by_tier` and `HEADLINE_TIER` are already imported from `backtest.scoring` (`grep -n "score_by_tier\|HEADLINE_TIER" backend/scripts/backtest_accuracy.py | head -3`); add them to that import if not.

- [ ] **Step 4: Call them in `backtest_forecasts`**

- Right after the `mature = _mature(...)` line from Task 4 Step 7, before `if not mature:`:

```python
    _read_twice_matured(db, cutoff, mature)
```

- In the rescore branch, after `results = _score_groups(groups, today)`:

```python
        _read_twice_headlines(db, results, min_price, None)
```

- In the daily path, after `results = _score_groups(groups, today)`:

```python
    _read_twice_headlines(db, results, min_price, mature_ids)
```

- [ ] **Step 5: Run the module and the existing backtest tests**

Run: `$PY -m pytest tests/test_backtest_reads_partitions.py tests/test_backtest_scoring.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/backtest_accuracy.py backend/tests/test_backtest_reads_partitions.py
git commit -m "feat: read twice in the backtest — matured forecasts per date and the headline per horizon, Postgres beside the partitions"
```

---

### Task 6: the gap heal, partition-only forecasts, and read-only writes

**Files:**
- Modify: `backend/scripts/backtest_accuracy.py` — `_store_forecast_outcomes` (default path), `_flush_verdict_refresh`, `_upsert_accuracy`, `_resolve_candidates`; add `_partition_only_outcomes`.
- Test: `backend/tests/test_backtest_read_only_writes.py`

**Interfaces:**
- Consumes: `db.read_only.is_read_only_error`, `report_read_only`; `ops_partitions.assign_ids`, `upsert`, `dual_write`, `READ_ONLY_ID_BASE`; `_outcome_partitions(db, forecast_ids)`; `_refreshed_outcome_partitions(mirror)`.
- Produces: `_partition_only_outcomes(rows: list[dict], what: str) -> bool`.

- [ ] **Step 1: Write the failing tests** (`tests/test_backtest_read_only_writes.py`)

```python
"""Phase 2 PR 1: the backtest's writes on a read-only night, and the partition-gap heal."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from database import Base, ForecastOutcome, Item, ItemForecast, PredictionAccuracy
from db import ops_partitions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from tests._ops_mirror import mirror_session_to_ops

DAY = date.today() - timedelta(days=10)
READ_ONLY = RuntimeError("cannot execute INSERT in a read-only transaction")


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(Item(id=1, item_id="ak", name="ak", type="skin"))
    db.add(ItemForecast(id=10, item_id=1, forecast_date=DAY, horizon_days=3, price_mid=3.2))
    db.commit()
    yield db
    engine.dispose()


def _outcome(forecast_id=10):
    return {
        "forecast_id": forecast_id, "item_id": 1, "forecast_date": DAY, "horizon_days": 3,
        "target_date": DAY + timedelta(days=3), "current_price": 3.0, "base_price": 3.0,
        "predicted_price_low": 2.5, "predicted_price_mid": 3.2, "predicted_price_high": 3.6,
        "actual_price": 3.3, "direction_predicted": "up", "direction_actual": "up", "direction_correct": 1,
        "in_interval": 1, "abs_error": 0.1, "pct_error": 3.0, "model_version": "lgbm-test",
    }


def _partition(forecast_id):
    from scripts.backtest_accuracy import _store_read

    return [r for r in _store_read("forecast_outcomes")("SELECT * FROM forecast_outcomes") if r.forecast_id == forecast_id]


def test_a_read_only_insert_lands_in_the_partitions_with_high_ids(session, monkeypatch, capsys):
    from scripts import backtest_accuracy as ba

    monkeypatch.setattr(session, "bulk_insert_mappings", lambda *a, **k: (_ for _ in ()).throw(READ_ONLY))
    assert ba._store_forecast_outcomes(session, [_outcome()], id_to_slug={1: "ak"}) == 1
    (row,) = _partition(10)
    assert row.id > ops_partitions.READ_ONLY_ID_BASE
    assert row.item_slug == "ak"
    assert session.query(ForecastOutcome).count() == 0
    assert "::warning title=Postgres read-only::" in capsys.readouterr().out


def test_a_partition_only_forecast_is_never_inserted_into_postgres(session):
    from scripts import backtest_accuracy as ba

    big = ops_partitions.READ_ONLY_ID_BASE + 5
    assert ba._store_forecast_outcomes(session, [_outcome(), _outcome(big)], id_to_slug={1: "ak"}) == 2
    assert [o.forecast_id for o in session.query(ForecastOutcome).all()] == [10]
    assert len(_partition(big)) == 1 and _partition(big)[0].id > ops_partitions.READ_ONLY_ID_BASE
    assert len(_partition(10)) == 1


def test_an_outcome_postgres_holds_but_the_partitions_lack_is_written_from_postgres(session):
    from scripts import backtest_accuracy as ba

    held = ForecastOutcome(id=77, **{k: v for k, v in _outcome().items()})
    session.add(held)
    session.commit()
    mirror_session_to_ops(session, tables=("items",))  # partitions lack the outcome
    revised = dict(_outcome(), actual_price=9.99)  # tonight's re-resolution must NOT win
    assert ba._store_forecast_outcomes(session, [revised], id_to_slug={1: "ak"}) == 0
    assert session.query(ForecastOutcome).count() == 1
    (row,) = _partition(10)
    assert (row.id, row.actual_price) == (77, 3.3)


def test_a_read_only_verdict_refresh_still_writes_the_partitions(session, monkeypatch):
    from scripts import backtest_accuracy as ba

    calls = []
    monkeypatch.setattr(ba, "_refreshed_outcome_partitions", lambda rows: calls.append(len(rows)))
    monkeypatch.setattr("db.parquet.append_table", lambda *a, **k: None)

    def execute(*a, **k):
        raise READ_ONLY

    monkeypatch.setattr(session, "execute", execute)
    ba._flush_verdict_refresh(session, [{"id": 1}], [{"forecast_id": 10}])
    assert calls == [1]


def test_a_read_only_accuracy_upsert_is_skipped_after_the_parquet_copy(session, monkeypatch, capsys):
    from scripts import backtest_accuracy as ba

    written = []
    monkeypatch.setattr("db.parquet.append_table", lambda table, rows, keys: written.append(table))
    monkeypatch.setattr(session, "commit", lambda: (_ for _ in ()).throw(READ_ONLY))
    row = {
        "prediction_type": "forecast", "evaluation_date": DAY, "horizon_days": 3, "model_version": "lgbm-test",
        "price_tier": -1, "evaluation_window_days": None, "sample_count": 3, "metrics": {}, "created_at": None,
    }
    ba._upsert_accuracy(session, [row])
    assert written == ["prediction_accuracy"]
    assert "::warning title=Postgres read-only::" in capsys.readouterr().out
    # No monkeypatch.undo(): it would also undo conftest's hermetic OPS_DIR.
    assert session.query(PredictionAccuracy).count() == 0


def test_a_swallowed_candidate_failure_rolls_the_session_back(session, monkeypatch):
    from scripts import backtest_accuracy as ba

    rolled = []
    monkeypatch.setattr(session, "rollback", lambda: rolled.append(True))

    def boom(*a, **k):
        raise ValueError("candidate insert failed")

    monkeypatch.setattr("backtest.candidate_resolution.resolve_candidate_outcomes", boom)
    assert ba._resolve_candidates(session, DAY, None) == {}
    assert rolled == [True]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_backtest_read_only_writes.py -q`
Expected: 6 failures (the read-only error propagates; the partition-only forecast is inserted; the heal writes nothing; no rollback).

- [ ] **Step 3: Add `_partition_only_outcomes`** (below `_refreshed_outcome_partitions`)

```python
def _partition_only_outcomes(rows, what) -> bool:
    """Write outcomes Postgres cannot take straight to the partitions, ids from READ_ONLY_ID_BASE.

    Two cases: a read-only night (SQLSTATE 25006), and forecasts that exist only in the
    partitions (id >= READ_ONLY_ID_BASE, issued on an earlier read-only night) — Postgres's
    forecast_id is a 32-bit foreign key and cannot hold them. A stored outcome keeps its id.
    """
    from db import ops_partitions

    def _write():
        frame = ops_partitions.assign_ids(
            "forecast_outcomes", pd.DataFrame(rows), base=ops_partitions.READ_ONLY_ID_BASE
        )
        ops_partitions.upsert("forecast_outcomes", frame)

    return ops_partitions.dual_write(what, _write)
```

Add `from db.read_only import is_read_only_error, report_read_only` to the module imports.

- [ ] **Step 4: Rewrite the default path of `_store_forecast_outcomes`**

In the `else:` branch, after `to_write = [o for o in outcomes if o["forecast_id"] not in existing_ids]`, insert:

```python
        if existing_ids:
            # Frozen in Postgres but missing from the partitions (a failed partition write on an
            # earlier night): the caller resolved them again because it reads the partitions.
            # Postgres wins — its frozen row goes to the partitions, tonight's values are dropped.
            _outcome_partitions(db, sorted(existing_ids))
            logger.warning(
                f"  {len(existing_ids):,} outcome(s) Postgres holds were missing from the ops "
                f"partitions; written there from Postgres"
            )
```

Then replace the tail of the function, from `db.bulk_insert_mappings(ForecastOutcome, to_write)` (the one after the `if delete_ids:` block) through `_outcome_partitions(db, [o["forecast_id"] for o in to_write])`, with:

```python
    from db.ops_partitions import READ_ONLY_ID_BASE
    from db.parquet import append_table

    to_postgres = [o for o in to_write if o["forecast_id"] < READ_ONLY_ID_BASE]
    partition_only = [m for m in mirror if m["forecast_id"] >= READ_ONLY_ID_BASE]
    read_only = False
    if to_postgres:
        try:
            db.bulk_insert_mappings(ForecastOutcome, to_postgres)
            db.commit()
        except Exception as e:
            if not is_read_only_error(e):
                raise
            db.rollback()
            report_read_only(f"{len(to_postgres):,} forecast_outcomes inserts")
            read_only = True

    append_table("forecast_outcomes", mirror, ["forecast_id"])
    if read_only:
        _partition_only_outcomes(mirror, "forecast_outcomes (read-only night)")
    else:
        if to_postgres:
            _outcome_partitions(db, [o["forecast_id"] for o in to_postgres])
        if partition_only:
            _partition_only_outcomes(partition_only, "forecast_outcomes (partition-only forecasts)")
```

Leave the `--reresolve` (`if delete_ids:`) branch unchanged: it is an operator action, not the nightly path.

- [ ] **Step 5: `_flush_verdict_refresh` survives read-only**

Replace

```python
    for i in range(0, len(updates), CHUNK):
        db.execute(_REFRESH_VERDICTS_SQL, updates[i : i + CHUNK])
    db.commit()
    _refreshed_outcome_partitions(mirror)
```

with

```python
    try:
        for i in range(0, len(updates), CHUNK):
            db.execute(_REFRESH_VERDICTS_SQL, updates[i : i + CHUNK])
        db.commit()
    except Exception as e:
        if not is_read_only_error(e):
            raise
        db.rollback()
        report_read_only(f"{len(updates):,} verdict refresh updates")
    _refreshed_outcome_partitions(mirror)
```

- [ ] **Step 6: `_upsert_accuracy` survives read-only**

Wrap the DB half (the `for row in rows:` loop and `db.commit()`) as:

```python
    try:
        for row in rows:
            ...  # unchanged
        db.commit()
    except Exception as e:
        if not is_read_only_error(e):
            raise
        db.rollback()
        # prediction_accuracy has no partition yet (parent spec, "Run records"); the single-file
        # Parquet copy above already landed.
        report_read_only(f"{len(rows):,} prediction_accuracy upserts")
```

- [ ] **Step 7: `_resolve_candidates` rolls back what it swallows**

In the `except Exception as e:` branch, before the warning:

```python
        try:
            db.rollback()  # a failed insert leaves the transaction aborted for production scoring
        except Exception:
            logger.exception("  rollback after the candidate failure also failed")
```

- [ ] **Step 8: Run the tests**

Run: `$PY -m pytest tests/test_backtest_read_only_writes.py tests/test_backtest_reads_partitions.py tests/test_backtest_scoring.py tests/test_ops_dual_write_outcomes.py tests/test_ops_item_slug.py -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add backend/scripts/backtest_accuracy.py backend/tests/test_backtest_read_only_writes.py
git commit -m "feat: the backtest's writes survive a read-only Postgres; partition-only forecasts and partition gaps are written from the right store"
```

---

### Task 7: docs, parity log line, full gate, PR

**Files:**
- Create: `docs/changelog/<today>-backtest-reads-ops-partitions.md`
- Modify: `.claude/rules/archive-reads.md` (the dual-write bullet), `backend/scripts/ops_parity.py` (log line), `docs/specs/2026-10-13-ops-readers-switch-design.md` (PR 1 status line)

- [ ] **Step 1: The parity log line**

In `backend/scripts/ops_parity.py`, change `"(Phase 1 needs 7)"` to `"(Phase 2 starts at 3)"`. Check nothing asserts the old text: `grep -rn "Phase 1 needs 7" backend/` returns nothing after the edit.

- [ ] **Step 2: The rule**

In `.claude/rules/archive-reads.md`, in the bullet starting `**The forecast record is dual-written to monthly partitions since 2026-10-12**`, replace `Postgres is still the read source;` with:

```
Since Phase 2 PR 1 (`changelog/<today>-backtest-reads-ops-partitions.md`) the backtest reads
them through `db/ops_store.py` (views named like the Postgres tables, `pg_deleted` hidden, a
table with no files raises) and logs each switched value beside Postgres's (`ops_store.compare`).
Every other reader still reads Postgres until its PR (`specs/2026-10-13-ops-readers-switch-design.md`);
```

- [ ] **Step 3: The changelog note**

`docs/changelog/<today>-backtest-reads-ops-partitions.md`:

```markdown
# The backtest reads the forecast record from the ops partitions

**Date:** <today>
**PR:** #<n> (Phase 2, PR 1 of 6 — `docs/specs/2026-10-13-ops-readers-switch-design.md`)

Backtest Accuracy now reads matured forecasts, the frozen-id set, the scoring panel, the
verdict refresh and the id→slug map from the monthly partitions through `db/ops_store.py`.
Postgres is read only for comparison: the matured count per forecast date and the headline
per horizon are logged from both stores, and a difference is a `::warning`, never a failure.

What changed besides the read:

- **Partition gaps heal from Postgres.** "Already frozen?" now means frozen in the partitions.
  An outcome Postgres holds but the partitions lack is resolved again, not re-inserted, and its
  Postgres row is written to the partitions. Postgres wins.
- **The backtest survives a read-only Postgres.** #114 covered the forecast, run-record,
  migration, drift-alert and shadow writers, not the backtest. On SQLSTATE 25006 the outcome
  insert lands in the partitions with ids from `READ_ONLY_ID_BASE`, the verdict refresh still
  writes the partitions, and the `prediction_accuracy` upsert is skipped after its single-file
  Parquet copy.
- **Partition-only forecasts** (ids from a read-only night) never reach Postgres: its
  `forecast_id` is a 32-bit foreign key.
- **The verdict refresh diffs against the partitions.** If a Postgres UPDATE is lost, Postgres
  keeps the old verdict while the partitions carry the new one. Postgres is being retired, and
  `ops_parity.py` shows the delta.

Gate: merged after 3 clean parity nights from 2026-10-13. Not changed: forecasts, bands,
scoring, the frozen instruments, candidate resolution (PR 2).
```

Replace `<today>` with the merge date and `<n>` with the PR number.

- [ ] **Step 4: Spec status**

In `docs/specs/2026-10-13-ops-readers-switch-design.md`, under `### PR 1: ...`, add a first line: `**Shipped** in #<n> (<today>).`

- [ ] **Step 5: Lint, format and the CI gate**

Run from the repo root: `ruff check . && ruff format --check backend`
Expected: no errors. If `ruff format --check` lists files, run `ruff format backend/<file>` on those and recheck.

Run from `backend/`: `$PY -m pytest tests -m "not slow" -q -n 4`
Expected: all pass, 0 failures. Then run the three new modules serially too: `$PY -m pytest tests/test_ops_store.py tests/test_backtest_reads_partitions.py tests/test_backtest_read_only_writes.py -q`.

- [ ] **Step 6: Commit and open the PR**

```bash
git add docs/changelog .claude/rules/archive-reads.md backend/scripts/ops_parity.py docs/specs/2026-10-13-ops-readers-switch-design.md
git commit -m "docs: the backtest reads the ops partitions — changelog, rule, parity log line"
git push -u origin feat/ops-store-scoring
gh pr create --title "feat: the backtest reads the ops partitions (Phase 2, PR 1)" --body-file docs/changelog/<today>-backtest-reads-ops-partitions.md
```

Do not wait on CI after opening the PR. Report the PR link. Merge only once the Backtest Accuracy log shows `clean nights in a row: 3` (or higher), and after merging, check the next night's log for the two `read twice` lines.
