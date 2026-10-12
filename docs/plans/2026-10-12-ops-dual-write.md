# Ops Dual-Write (Phase 1.2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every writer of the forecast record also writes the monthly Parquet partitions in `cs2-oracle-data`, with the ids Postgres issued, and a nightly check proves the two stores agree.

**Architecture:** One partition module (`db/ops_partitions.py`) owns layout, schema, guards and the DuckDB merge. One read-back module (`db/ops_readback.py`) re-selects just-committed rows from Postgres. Each writer, after its unchanged Postgres commit, calls read-back then `upsert`/`replace` inside `dual_write`, which logs instead of raising. `scripts/ops_parity.py` compares per-date aggregates from both stores in Backtest Accuracy; `scripts/ops_sync_from_postgres.py` (dispatched via `ops-sync.yml`) fills and repairs.

**Tech Stack:** Python 3.13 (3.11 in CI), SQLAlchemy, DuckDB, pandas, pytest, GitHub Actions.

**Spec:** `docs/specs/2026-10-11-postgres-to-parquet-history-design.md`, section "Phase 1.2: dual-write (designed 2026-10-12)".

## Global Constraints

- Postgres stays the read source. No reader changes in this PR.
- Every existing Postgres write path stays byte-for-byte as it is; partitions are written **after** the Postgres commit.
- The single-file mirror (`ops/item_forecasts.parquet`, `ops/forecast_outcomes.parquet`) keeps its current writes and order.
- A partition write failure never fails a run: `dual_write` logs ERROR plus a `::error` annotation and returns False.
- Closed month = calendar end + 43 days; refused unless `allow_closed=True`.
- Partition schema is exactly what the 2026-10-12 backfill wrote (column lists in Task 1).
- A column the write does not carry keeps its stored value (`model_config`, `item_slug`, `pg_deleted`).
- Parity tolerance: counts and id sums exact; float sums `math.isclose(rel_tol=1e-9, abs_tol=1e-9)`; `pg_deleted` rows excluded.
- Only CI writes the data repo. Nothing here runs against prod locally: `backend/.env` points at prod Supabase, so tests use in-memory SQLite only.
- Run tests from `backend/` as `venv/bin/python -m pytest tests/<file> -q`.
- No Claude/Codex attribution in commits or the PR.

## Review Focus

1. **Verdict refresh touching a closed month** — `_refresh_verdict_columns` walks every row, so a scoring change can rewrite a verdict dated in a closed month. Expected: the partition write is refused and logged, Postgres still changes, and parity fails naming that month (a human decides). Pinned in Task 5.
2. **The same night written twice** (a forecast retry, or regime then global writes on one date) — expected: rows are replaced by natural key, ids unchanged, row count unchanged, and the other batch's rows (and their `model_config`) untouched. Pinned in Task 3.
3. **Prod schema behind the ORM** (a disclosure column the model declares but prod lacks) — expected: read-back selects only the columns the table has, and the partition keeps NULL / its stored value. Pinned in Task 2.
4. **Partition-issued ids on a read-only night** later meeting Postgres's own sequence — expected: parity reports an `id_sum` mismatch on that date rather than passing. Pinned in Task 9.
5. **Rows routed across a month boundary** — a frame spanning 10-31 and 11-01, and candidate outcomes routed by their candidate's `forecast_date` (`_part`), not the resolution date. Pinned in Task 1 and Task 6.

---

### Task 1: Partition module

**Files:**
- Create: `backend/db/ops_partitions.py`
- Modify: `backend/scripts/ops_releases.py:54-63` (import `CLOSE_AFTER_DAYS` and `closes_on` from the new module instead of defining them)
- Test: `backend/tests/test_ops_partitions.py`

**Interfaces:**
- Produces:
  - `CLOSE_AFTER_DAYS: int = 43`, `closes_on(month: str) -> date`
  - `FORECAST_KEY`, `CANDIDATE_KEY: tuple[str, ...]`
  - `Spec(key, order, columns, part_col)`; `SPECS: dict[str, Spec]` for `item_forecasts`, `forecast_outcomes`, `forecast_candidates`, `forecast_candidate_outcomes`, `items`
  - `class PartitionError(RuntimeError)`
  - `ops_dir(root: Path | str | None = None) -> Path` (default reads `db.parquet.OPS_DIR` at call time)
  - `month_files(table: str, root=None) -> list[Path]`
  - `upsert(table: str, frame: pd.DataFrame, *, allow_closed=False, root=None, today: date | None = None) -> dict[str, int]` (month → rows after write; `"all"` for `items`)
  - `replace(table: str, key_col: str, keys: Iterable, frame: pd.DataFrame, *, allow_closed=False, root=None, today=None) -> dict[str, int]`
  - `assign_ids(table: str, frame: pd.DataFrame, *, root=None) -> pd.DataFrame`
  - `dual_write(what: str, fn: Callable[[], object]) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
"""db/ops_partitions.py: the monthly partitions the Phase 1.2 dual-write lands in."""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pandas as pd
import pytest

TODAY = date(2026, 10, 12)


def _fc(item_id, day, h=7, mid=10.0, id_=None, **extra):
    row = {
        "id": id_,
        "item_id": item_id,
        "forecast_date": day,
        "horizon_days": h,
        "price_low": mid - 1,
        "price_mid": mid,
        "price_high": mid + 1,
        "current_price": mid,
        "direction": "up",
        "model_version": "v",
        "created_at": datetime(2026, 10, 12, 2, 0),
    }
    row.update(extra)
    return row


def _read(path):
    return duckdb.sql(f"SELECT * FROM read_parquet('{path}') ORDER BY item_id, horizon_days").df()


def test_rows_route_to_their_month_and_carry_the_full_schema(tmp_path):
    from db.ops_partitions import SPECS, upsert

    frame = pd.DataFrame([_fc(1, date(2026, 10, 31), id_=1), _fc(1, date(2026, 11, 1), id_=2)])
    out = upsert("item_forecasts", frame, root=tmp_path, today=TODAY)
    assert out == {"2026-10": 1, "2026-11": 1}
    oct_ = _read(tmp_path / "item_forecasts" / "2026-10.parquet")
    assert list(oct_.columns) == list(SPECS["item_forecasts"].columns)
    assert oct_["id"].tolist() == [1]
    assert oct_["pg_deleted"].isna().all()


def test_upsert_replaces_by_key_and_keeps_columns_the_write_lacks(tmp_path):
    from db.ops_partitions import upsert

    first = pd.DataFrame([_fc(1, date(2026, 10, 11), id_=5, model_config="global-only"), _fc(2, date(2026, 10, 11), id_=6)])
    upsert("item_forecasts", first, root=tmp_path, today=TODAY)
    again = pd.DataFrame([_fc(1, date(2026, 10, 11), id_=5, mid=12.0)])  # no model_config column
    assert upsert("item_forecasts", again, root=tmp_path, today=TODAY) == {"2026-10": 2}
    got = _read(tmp_path / "item_forecasts" / "2026-10.parquet")
    row = got[got.item_id == 1].iloc[0]
    assert row.price_mid == 12.0
    assert row.model_config == "global-only"
    assert got[got.item_id == 2].iloc[0].id == 6


def test_closed_months_are_refused_unless_allowed(tmp_path):
    from db.ops_partitions import PartitionError, closes_on, upsert

    assert closes_on("2026-08") == date(2026, 10, 13)
    frame = pd.DataFrame([_fc(1, date(2026, 7, 20), id_=1)])
    with pytest.raises(PartitionError, match="closed"):
        upsert("item_forecasts", frame, root=tmp_path, today=TODAY)
    assert not (tmp_path / "item_forecasts" / "2026-07.parquet").exists()
    assert upsert("item_forecasts", frame, root=tmp_path, today=TODAY, allow_closed=True) == {"2026-07": 1}


def test_unknown_columns_null_keys_and_repeated_keys_fail(tmp_path):
    from db.ops_partitions import PartitionError, upsert

    d = date(2026, 10, 11)
    with pytest.raises(PartitionError, match="schema"):
        upsert("item_forecasts", pd.DataFrame([_fc(1, d, surprise=1)]), root=tmp_path, today=TODAY)
    with pytest.raises(PartitionError, match="NULL"):
        upsert("item_forecasts", pd.DataFrame([_fc(None, d)]), root=tmp_path, today=TODAY)
    with pytest.raises(PartitionError, match="repeated"):
        upsert("item_forecasts", pd.DataFrame([_fc(1, d), _fc(1, d)]), root=tmp_path, today=TODAY)


def test_a_failed_write_leaves_the_old_file(tmp_path, monkeypatch):
    from db import ops_partitions
    from db.ops_partitions import upsert

    d = date(2026, 10, 11)
    upsert("item_forecasts", pd.DataFrame([_fc(1, d, id_=1)]), root=tmp_path, today=TODAY)
    path = tmp_path / "item_forecasts" / "2026-10.parquet"
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ops_partitions.os, "replace", boom)
    with pytest.raises(OSError):
        upsert("item_forecasts", pd.DataFrame([_fc(2, d, id_=2)]), root=tmp_path, today=TODAY)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


def test_replace_removes_named_keys_across_months(tmp_path):
    from db.ops_partitions import replace, upsert

    def oc(fid, day, id_):
        return {"id": id_, "forecast_id": fid, "item_id": 1, "forecast_date": day, "horizon_days": 7, "in_interval": 1}

    upsert("forecast_outcomes", pd.DataFrame([oc(10, date(2026, 9, 30), 1), oc(11, date(2026, 10, 1), 2)]), root=tmp_path, today=TODAY)
    out = replace("forecast_outcomes", "forecast_id", [10, 11], pd.DataFrame([oc(11, date(2026, 10, 1), 3)]), root=tmp_path, today=TODAY)
    assert out == {"2026-09": 0, "2026-10": 1}
    got = _read(tmp_path / "forecast_outcomes" / "2026-10.parquet")
    assert got["id"].tolist() == [3]


def test_candidate_outcomes_route_by_part_and_drop_it(tmp_path):
    from db.ops_partitions import upsert

    frame = pd.DataFrame([{"candidate_id": 7, "actual_price": 1.0, "in_interval": True, "_part": date(2026, 9, 30)}])
    assert upsert("forecast_candidate_outcomes", frame, root=tmp_path, today=TODAY) == {"2026-09": 1}
    got = duckdb.sql(f"SELECT * FROM read_parquet('{tmp_path}/forecast_candidate_outcomes/2026-09.parquet')").df()
    assert "_part" not in got.columns


def test_items_is_one_file(tmp_path):
    from db.ops_partitions import upsert

    frame = pd.DataFrame([{"id": 1, "item_id": "ak_1", "name": "AK 1", "type": "rifle"}])
    assert upsert("items", frame, root=tmp_path, today=TODAY) == {"all": 1}
    assert (tmp_path / "items.parquet").exists()


def test_assign_ids_keeps_stored_ids_and_issues_above_the_max(tmp_path):
    from db.ops_partitions import assign_ids, upsert

    d = date(2026, 10, 11)
    upsert("item_forecasts", pd.DataFrame([_fc(1, d, id_=40), _fc(2, d, id_=41)]), root=tmp_path, today=TODAY)
    got = assign_ids("item_forecasts", pd.DataFrame([_fc(2, d), _fc(3, d), _fc(4, d)]), root=tmp_path)
    assert got["id"].tolist() == [41, 42, 43]


def test_dual_write_logs_and_returns_false(capsys, caplog):
    from db.ops_partitions import dual_write

    def fail():
        raise RuntimeError("boom")

    assert dual_write("item_forecasts", fail) is False
    assert "::error title=Ops partition write failed::" in capsys.readouterr().out
    assert "boom" in caplog.text
    assert dual_write("item_forecasts", lambda: None) is True
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_partitions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'db.ops_partitions'`

- [ ] **Step 3: Write the module**

```python
"""Monthly ops partitions in the data repo: the dual-write side of Phase 1.2.

Layout (docs/specs/2026-10-11-postgres-to-parquet-history-design.md):

    ops/item_forecasts/YYYY-MM.parquet               (by forecast_date)
    ops/forecast_outcomes/YYYY-MM.parquet            (by forecast_date)
    ops/forecast_candidates/YYYY-MM.parquet          (by forecast_date)
    ops/forecast_candidate_outcomes/YYYY-MM.parquet  (by the candidate's forecast_date, passed as `_part`)
    ops/items.parquet

Postgres stays the read source. Writers call `upsert` after their Postgres commit with rows read
back from Postgres (db/ops_readback.py), inside `dual_write`, so a failure here never fails the
run; scripts/ops_parity.py catches it the same night.

Guards, all raising PartitionError: a closed month (calendar end + 43 days) is refused without
`allow_closed`; a column outside the schema the 2026-10-12 backfill wrote fails instead of being
dropped; NULL or repeated keys in one write fail; and each month's row count is checked against
the arithmetic of the merge before the file is swapped in.

A column the write does not carry keeps its stored value. Postgres has no `model_config`,
`item_slug` or `pg_deleted`, and a read-back must not blank them.
"""

from __future__ import annotations

import calendar
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

#: A month is closed once every forecast in it has matured: h=30 plus the 13-day embargo carry.
CLOSE_AFTER_DAYS = 43


def closes_on(month: str) -> date:
    """The first day month *month* (``YYYY-MM``) counts as closed."""
    y, m = map(int, month.split("-"))
    return date(y, m, calendar.monthrange(y, m)[1]) + timedelta(days=CLOSE_AFTER_DAYS)


FORECAST_KEY = ("item_id", "forecast_date", "horizon_days")
CANDIDATE_KEY = (*FORECAST_KEY, "component", "candidate_name", "candidate_version")

_TS = "TIMESTAMP_NS"


@dataclass(frozen=True)
class Spec:
    key: tuple[str, ...]
    order: str
    columns: dict[str, str]
    part_col: str | None  # None: one file, ops/<table>.parquet


# Column lists and types are what the 2026-10-12 backfill wrote (DESCRIBE on the data repo).
SPECS: dict[str, Spec] = {
    "item_forecasts": Spec(
        FORECAST_KEY,
        "item_id, forecast_date, horizon_days",
        {
            "id": "BIGINT", "item_id": "BIGINT", "forecast_date": "DATE", "horizon_days": "BIGINT",
            "price_low": "DOUBLE", "price_mid": "DOUBLE", "price_high": "DOUBLE", "current_price": "DOUBLE",
            "direction": "VARCHAR", "confidence": "VARCHAR", "model_version": "VARCHAR", "created_at": _TS,
            "anchor_clean": "BOOLEAN", "anchor_wedge_pct": "DOUBLE", "exceed_p": "DOUBLE", "anomaly_p": "DOUBLE",
            "band_multiplier": "DOUBLE", "model_config": "VARCHAR", "item_slug": "VARCHAR", "pg_deleted": "BOOLEAN",
        },
        "forecast_date",
    ),
    "forecast_outcomes": Spec(
        ("forecast_id",),
        "forecast_date, item_id, horizon_days, forecast_id",
        {
            "id": "BIGINT", "forecast_id": "BIGINT", "item_id": "BIGINT", "forecast_date": "DATE",
            "horizon_days": "BIGINT", "target_date": _TS, "current_price": "DOUBLE",
            "predicted_price_low": "DOUBLE", "predicted_price_mid": "DOUBLE", "predicted_price_high": "DOUBLE",
            "actual_price": "DOUBLE", "direction_predicted": "VARCHAR", "direction_actual": "VARCHAR",
            "direction_correct": "BIGINT", "in_interval": "BIGINT", "abs_error": "DOUBLE", "pct_error": "DOUBLE",
            "model_version": "VARCHAR", "evaluated_at": _TS, "base_price": "DOUBLE", "resolved_at": _TS,
            "base_stale_run_days": "DOUBLE", "item_slug": "VARCHAR", "pg_deleted": "BOOLEAN",
        },
        "forecast_date",
    ),
    "forecast_candidates": Spec(
        CANDIDATE_KEY,
        ", ".join(CANDIDATE_KEY),
        {
            "id": "BIGINT", "item_id": "BIGINT", "forecast_date": "DATE", "horizon_days": "BIGINT",
            "component": "VARCHAR", "candidate_name": "VARCHAR", "candidate_version": "VARCHAR",
            "centre_price": "DOUBLE", "predicted_price_low": "DOUBLE", "predicted_price_high": "DOUBLE",
            "score": "DOUBLE", "anchor_price": "DOUBLE", "feature_cutoff_at": _TS, "artifact_version": "VARCHAR",
            "config_fingerprint": "VARCHAR", "created_at": _TS,
        },
        "forecast_date",
    ),
    "forecast_candidate_outcomes": Spec(
        ("candidate_id",),
        "candidate_id",
        {
            "candidate_id": "BIGINT", "base_price": "DOUBLE", "actual_price": "DOUBLE", "resolved_at": _TS,
            "absolute_error": "DOUBLE", "percentage_error": "DOUBLE", "in_interval": "BOOLEAN",
            "resolution_version": "VARCHAR",
        },
        "_part",
    ),
    "items": Spec(
        ("id",),
        "id",
        {
            "id": "BIGINT", "item_id": "VARCHAR", "name": "VARCHAR", "type": "VARCHAR", "release_date": _TS,
            "created_at": _TS, "updated_at": _TS, "icon_url": "VARCHAR", "classid": "VARCHAR",
            "instanceid": "INTEGER", "is_backfilled": "BIGINT", "rarity": "INTEGER", "rarity_rank": "INTEGER",
            "weapon_type": "INTEGER", "is_trainable": "BIGINT",
        },
        None,
    ),
}


class PartitionError(RuntimeError):
    """A partition write that would lose, drop or misplace data."""


def ops_dir(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    from db import parquet  # read at call time, so tests can monkeypatch OPS_DIR

    return parquet.OPS_DIR


def _path(table: str, month: str, root) -> Path:
    d = ops_dir(root)
    return d / f"{table}.parquet" if SPECS[table].part_col is None else d / table / f"{month}.parquet"


def month_files(table: str, root: Path | str | None = None) -> list[Path]:
    if SPECS[table].part_col is None:
        p = _path(table, "", root)
        return [p] if p.exists() else []
    d = ops_dir(root) / table
    return sorted(d.glob("????-??.parquet")) if d.is_dir() else []


def _sql_list(paths: Iterable[Path]) -> str:
    return "[" + ", ".join(f"'{p}'" for p in paths) + "]"


def _coerce(spec: Spec, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    for c in frame.columns:
        t = spec.columns.get(c) or ("DATE" if c == "_part" else None)
        if t in ("DATE", _TS):
            frame[c] = pd.to_datetime(frame[c])
    return frame


def _check(table: str, spec: Spec, frame: pd.DataFrame) -> None:
    extra = set(frame.columns) - set(spec.columns) - {"_part"}
    if extra:
        raise PartitionError(f"{table}: columns outside the partition schema: {sorted(extra)}")
    need = list(spec.key) + ([spec.part_col] if spec.part_col else [])
    absent = [c for c in need if c not in frame.columns]
    if absent:
        raise PartitionError(f"{table}: write lacks {absent}")
    if frame[need].isna().any().any():
        raise PartitionError(f"{table}: NULL in key or partition column")
    dup = int(frame.duplicated(list(spec.key)).sum())
    if dup:
        raise PartitionError(f"{table}: {dup} repeated keys in one write")


def _months(spec: Spec, frame: pd.DataFrame) -> pd.Series:
    if spec.part_col is None:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[spec.part_col].dt.strftime("%Y-%m")


def _on(spec: Spec) -> str:
    return " AND ".join(f"CAST(e.{k} AS {spec.columns[k]}) = CAST(n.{k} AS {spec.columns[k]})" for k in spec.key)


_EMPTY_DTYPE = {"BIGINT": "Int64", "INTEGER": "Int64", "DOUBLE": "float64", "BOOLEAN": "boolean", "DATE": "datetime64[ns]", _TS: "datetime64[ns]", "VARCHAR": "string"}


def _merge(con, table: str, spec: Spec, path: Path, new: pd.DataFrame, drop) -> int:
    new = new.drop(columns=[c for c in new.columns if c not in spec.columns])
    if new.empty:
        # A replace with nothing to re-insert still needs typed key columns for the joins.
        new = pd.DataFrame({k: pd.Series(dtype=_EMPTY_DTYPE[spec.columns[k]]) for k in spec.key})
    con.register("_new", new)
    have = set(new.columns)
    if path.exists():
        src = f"read_parquet('{path}')"
        file_cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
        unknown = file_cols - set(spec.columns)
        if unknown:
            raise PartitionError(f"{path}: columns outside the partition schema: {sorted(unknown)}")

        def pick(c, t):
            if c in have:
                return f"CAST(n.{c} AS {t}) AS {c}"
            if c in file_cols:
                return f"CAST(e.{c} AS {t}) AS {c}"
            return f"CAST(NULL AS {t}) AS {c}"

        new_sql = f"SELECT {', '.join(pick(c, t) for c, t in spec.columns.items())} FROM _new n LEFT JOIN {src} e ON {_on(spec)}"
        gone = f"EXISTS (SELECT 1 FROM _new n WHERE {_on(spec)})"
        if drop is not None:
            col, keys = drop
            con.register("_drop", pd.DataFrame({col: list(keys)}))
            gone += f" OR e.{col} IN (SELECT {col} FROM _drop)"
        old_cols = ", ".join(
            f"CAST(e.{c} AS {t}) AS {c}" if c in file_cols else f"CAST(NULL AS {t}) AS {c}"
            for c, t in spec.columns.items()
        )
        body = f"{new_sql} UNION ALL SELECT {old_cols} FROM {src} e WHERE NOT ({gone})"
        before = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
        removed = con.execute(f"SELECT count(*) FROM {src} e WHERE {gone}").fetchone()[0]
    else:
        body = "SELECT " + ", ".join(
            f"CAST(n.{c} AS {t}) AS {c}" if c in have else f"CAST(NULL AS {t}) AS {c}" for c, t in spec.columns.items()
        ) + " FROM _new n"
        before = removed = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        con.execute(f"COPY (SELECT * FROM ({body}) ORDER BY {spec.order}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        after = con.execute(f"SELECT count(*) FROM read_parquet('{tmp}')").fetchone()[0]
        expected = before - removed + len(new)
        if after != expected:
            raise PartitionError(f"{path}: wrote {after} rows, expected {expected}")
        if drop is None and after < before:
            raise PartitionError(f"{path}: an upsert would lose {before - after} rows")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        con.unregister("_new")
    return after


def _months_holding(table: str, col: str, keys: list, root) -> set[str]:
    files = month_files(table, root)
    if not files or not keys:
        return set()
    con = duckdb.connect()
    try:
        con.register("_drop", pd.DataFrame({col: keys}))
        rows = con.execute(
            f"SELECT DISTINCT filename FROM read_parquet({_sql_list(files)}, filename = true, union_by_name = true) "
            f"WHERE {col} IN (SELECT {col} FROM _drop)"
        ).fetchall()
    finally:
        con.close()
    return {Path(r[0]).stem for r in rows}


def _apply(table, frame, drop, allow_closed, root, today) -> dict[str, int]:
    spec = SPECS[table]
    frame = _coerce(spec, frame)
    if frame.empty and drop is None:
        return {}
    if frame.empty:  # a read-back with no rows has no columns either
        months = pd.Series(dtype=object)
    else:
        _check(table, spec, frame)
        months = _months(spec, frame)
    targets = set(months)
    if drop is not None and spec.part_col is not None:
        targets |= _months_holding(table, drop[0], list(drop[1]), root)
    today = today or datetime.now(UTC).date()
    closed = sorted(m for m in targets if m and closes_on(m) <= today)
    if closed and not allow_closed:
        raise PartitionError(f"{table}: refusing to write closed month(s) {closed}; pass allow_closed")
    out: dict[str, int] = {}
    con = duckdb.connect()
    try:
        for m in sorted(targets):
            out[m or "all"] = _merge(con, table, spec, _path(table, m, root), frame[months == m], drop)
    finally:
        con.close()
    return out


def upsert(table: str, frame: pd.DataFrame, *, allow_closed: bool = False, root=None, today: date | None = None) -> dict[str, int]:
    """Write *frame* into its months, replacing rows with the same key. Returns rows per month after."""
    return _apply(table, frame, None, allow_closed, root, today)


def replace(
    table: str, key_col: str, keys: Iterable, frame: pd.DataFrame, *, allow_closed: bool = False, root=None, today: date | None = None
) -> dict[str, int]:
    """Remove every row whose *key_col* is in *keys*, wherever it is, then upsert *frame*."""
    return _apply(table, frame, (key_col, list(keys)), allow_closed, root, today)


def assign_ids(table: str, frame: pd.DataFrame, *, root=None) -> pd.DataFrame:
    """Ids for a read-only night: a key already stored keeps its id; the rest get max(id) + 1 upward."""
    spec = SPECS[table]
    out = _coerce(spec, frame.drop(columns=["id"], errors="ignore"))
    hit: dict[int, int] = {}
    top = 0
    files = month_files(table, root)
    if files:
        con = duckdb.connect()
        try:
            rel = f"read_parquet({_sql_list(files)}, union_by_name = true)"
            con.register("n", out[list(spec.key)].assign(_pos=range(len(out))))
            hit = dict(con.execute(f"SELECT n._pos, e.id FROM n JOIN (SELECT * FROM {rel} WHERE id IS NOT NULL) e ON {_on(spec)}").fetchall())
            top = con.execute(f"SELECT coalesce(max(id), 0) FROM {rel}").fetchone()[0]
        finally:
            con.close()
    ids = []
    for pos in range(len(out)):
        if pos in hit:
            ids.append(int(hit[pos]))
        else:
            top += 1
            ids.append(top)
    out["id"] = pd.array(ids, dtype="Int64")
    return out


def dual_write(what: str, fn: Callable[[], object]) -> bool:
    """Run one partition write; log a failure instead of raising.

    By the time this runs, Postgres and the single-file mirror hold the rows, so nothing is lost,
    and failing the run after its commit would only endanger the rest of the chain. The nightly
    parity check fails the same night instead.
    """
    try:
        fn()
    except Exception:
        logger.exception("ops partitions: %s write failed; Postgres holds the rows, parity will fail tonight", what)
        print(f"::error title=Ops partition write failed::{what}", flush=True)
        return False
    return True
```

Then in `backend/scripts/ops_releases.py`, delete the local `CLOSE_AFTER_DAYS = 43` and `def closes_on(...)` (lines 54 and 60-63) and add at the imports:

```python
from db.ops_partitions import CLOSE_AFTER_DAYS, closes_on  # noqa: F401  (re-exported for callers and tests)
```

Check how `ops_releases.py` imports backend modules (it runs as `python scripts/ops_releases.py` from `backend/`): if it has no `sys.path` setup, copy the one `scripts/backtest_accuracy.py` uses at its top.

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_partitions.py tests/test_ops_releases.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add backend/db/ops_partitions.py backend/scripts/ops_releases.py backend/tests/test_ops_partitions.py
git commit -m "feat: ops partition module — monthly upsert/replace with closed-month, schema and row-count guards"
```

---

### Task 2: Read-back from Postgres

**Files:**
- Create: `backend/db/ops_readback.py`
- Test: `backend/tests/test_ops_readback.py`

**Interfaces:**
- Consumes: ORM tables `Item`, `ItemForecast`, `ForecastOutcome`, `ForecastCandidate`, `ForecastCandidateOutcome` from `database`.
- Produces (each returns a `pd.DataFrame`, empty with no rows):
  - `forecasts(db, dates: Iterable[date], keys: set[tuple[int, date, int]] | None = None)` — with `item_slug`
  - `candidates(db, dates, keys: set[tuple] | None = None)` — keys in `CANDIDATE_KEY` order
  - `outcomes(db, *, forecast_ids: Iterable[int] | None = None, dates: Iterable[date] | None = None)` — with `item_slug`
  - `candidate_outcomes(db, *, candidate_ids=None, dates=None)` — with `_part` = the candidate's `forecast_date`
  - `items(db, slugs: Iterable[str] | None = None)`

- [ ] **Step 1: Write the failing tests**

```python
"""db/ops_readback.py: committed rows re-read from Postgres, with the ids Postgres issued."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

D = date(2026, 10, 11)


def _session():
    from database import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed(db):
    from database import ForecastCandidate, ForecastCandidateOutcome, ForecastOutcome, Item, ItemForecast

    db.add(Item(id=1, item_id="ak_1", name="AK 1", type="rifle"))
    db.add_all([
        ItemForecast(id=10, item_id=1, forecast_date=D, horizon_days=7, price_mid=10.0),
        ItemForecast(id=11, item_id=1, forecast_date=D, horizon_days=3, price_mid=9.0),
    ])
    db.add(ForecastOutcome(id=20, forecast_id=10, item_id=1, forecast_date=D, horizon_days=7, actual_price=11.0))
    db.add(ForecastCandidate(
        id=30, item_id=1, forecast_date=D, horizon_days=7, component="centre", candidate_name="last_price",
        candidate_version="v1", config_fingerprint="a" * 64, feature_cutoff_at=datetime(2026, 10, 11),
        artifact_version="x", anchor_price=10.0,
    ))
    db.add(ForecastCandidateOutcome(candidate_id=30, base_price=10.0, actual_price=11.0, resolved_at=datetime(2026, 10, 18), resolution_version="r1"))
    db.commit()


def test_forecasts_carry_ids_and_slug_and_filter_to_keys():
    from db import ops_readback

    db = _session()
    _seed(db)
    got = ops_readback.forecasts(db, {D}, {(1, D, 7)})
    assert got["id"].tolist() == [10]
    assert got["item_slug"].tolist() == ["ak_1"]


def test_outcomes_candidates_and_candidate_outcomes():
    from db import ops_readback

    db = _session()
    _seed(db)
    assert ops_readback.outcomes(db, forecast_ids=[10])["id"].tolist() == [20]
    assert ops_readback.outcomes(db, dates=[D])["item_slug"].tolist() == ["ak_1"]
    assert ops_readback.candidates(db, {D})["id"].tolist() == [30]
    co = ops_readback.candidate_outcomes(db, candidate_ids=[30])
    assert co["candidate_id"].tolist() == [30]
    assert str(co["_part"].iloc[0])[:10] == "2026-10-11"
    assert ops_readback.items(db, ["ak_1"])["id"].tolist() == [1]


def test_a_column_prod_lacks_is_not_selected():
    from db import ops_readback

    db = _session()
    _seed(db)
    db.execute(text("ALTER TABLE item_forecasts DROP COLUMN band_multiplier"))
    db.commit()
    got = ops_readback.forecasts(db, {D})
    assert "band_multiplier" not in got.columns
    assert len(got) == 2


def test_nothing_committed_is_an_empty_frame():
    from db import ops_readback

    db = _session()
    assert ops_readback.forecasts(db, {D}).empty
    assert ops_readback.outcomes(db, forecast_ids=[]).empty
```

If a model above needs a non-null column this seed omits, the test errors with `NOT NULL constraint failed: <table>.<column>`; add that column with any value.

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_readback.py -q`
Expected: FAIL with `ImportError: cannot import name 'ops_readback'`

- [ ] **Step 3: Write the module**

```python
"""Read just-committed rows back from Postgres for the ops partitions (Phase 1.2).

Postgres issues the surrogate ids and the partitions must carry the same ones, so each writer
re-reads what it committed instead of changing its insert. Each call covers one night's keys.

Only the columns the live table has are selected: prod runs behind its migrations, and the ORM
declaring a column is not evidence the table has one (see scripts/forecast_prices.py).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import inspect, select

CHUNK = 900


def _columns(db, table) -> list:
    present = {c["name"] for c in inspect(db.get_bind()).get_columns(table.name)}
    return [c for c in table.c if c.name in present]


def _frame(db, stmt) -> pd.DataFrame:
    res = db.execute(stmt)
    return pd.DataFrame(res.fetchall(), columns=list(res.keys()))


def _chunked(db, build, values) -> pd.DataFrame:
    values = sorted(set(values))
    frames = [_frame(db, build(values[i : i + CHUNK])) for i in range(0, len(values), CHUNK)]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _day(value):
    return pd.Timestamp(value).date()


def _keep(df: pd.DataFrame, cols: tuple[str, ...], keys) -> pd.DataFrame:
    if keys is None or df.empty:
        return df
    want = {tuple(_day(v) if c == "forecast_date" else v for c, v in zip(cols, k, strict=True)) for k in keys}
    mask = [tuple(_day(r[c]) if c == "forecast_date" else r[c] for c in cols) in want for _, r in df.iterrows()]
    return df[mask].reset_index(drop=True)


def forecasts(db, dates: Iterable, keys=None) -> pd.DataFrame:
    from database import Item, ItemForecast
    from db.ops_partitions import FORECAST_KEY

    t, items = ItemForecast.__table__, Item.__table__
    cols = [*_columns(db, t), items.c.item_id.label("item_slug")]
    j = t.outerjoin(items, items.c.id == t.c.item_id)
    df = _chunked(db, lambda vs: select(*cols).select_from(j).where(t.c.forecast_date.in_(vs)), dates)
    return _keep(df, FORECAST_KEY, keys)


def candidates(db, dates: Iterable, keys=None) -> pd.DataFrame:
    from database import ForecastCandidate
    from db.ops_partitions import CANDIDATE_KEY

    t = ForecastCandidate.__table__
    cols = _columns(db, t)
    df = _chunked(db, lambda vs: select(*cols).where(t.c.forecast_date.in_(vs)), dates)
    return _keep(df, CANDIDATE_KEY, keys)


def outcomes(db, *, forecast_ids: Iterable | None = None, dates: Iterable | None = None) -> pd.DataFrame:
    from database import ForecastOutcome, Item

    t, items = ForecastOutcome.__table__, Item.__table__
    cols = [*_columns(db, t), items.c.item_id.label("item_slug")]
    j = t.outerjoin(items, items.c.id == t.c.item_id)
    if forecast_ids is not None:
        return _chunked(db, lambda vs: select(*cols).select_from(j).where(t.c.forecast_id.in_(vs)), forecast_ids)
    return _chunked(db, lambda vs: select(*cols).select_from(j).where(t.c.forecast_date.in_(vs)), dates or [])


def candidate_outcomes(db, *, candidate_ids: Iterable | None = None, dates: Iterable | None = None) -> pd.DataFrame:
    from database import ForecastCandidate, ForecastCandidateOutcome

    t, c = ForecastCandidateOutcome.__table__, ForecastCandidate.__table__
    cols = [*_columns(db, t), c.c.forecast_date.label("_part")]
    j = t.join(c, c.c.id == t.c.candidate_id)
    if candidate_ids is not None:
        return _chunked(db, lambda vs: select(*cols).select_from(j).where(t.c.candidate_id.in_(vs)), candidate_ids)
    return _chunked(db, lambda vs: select(*cols).select_from(j).where(c.c.forecast_date.in_(vs)), dates or [])


def items(db, slugs: Iterable[str] | None = None) -> pd.DataFrame:
    from database import Item

    t = Item.__table__
    cols = _columns(db, t)
    if slugs is None:
        return _frame(db, select(*cols))
    return _chunked(db, lambda vs: select(*cols).where(t.c.item_id.in_(vs)), slugs)
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_readback.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/db/ops_readback.py backend/tests/test_ops_readback.py
git commit -m "feat: ops read-back — committed rows re-read from Postgres with their issued ids"
```

---

### Task 3: Forecast writer

**Files:**
- Modify: `backend/scripts/forecast_prices.py` — `_write_forecasts_to_db`, end of the `if forecast_rows:` block (just before `return len(forecast_rows)`, currently line 477)
- Test: `backend/tests/test_ops_dual_write_forecasts.py`

**Interfaces:**
- Consumes: `ops_partitions.upsert`, `ops_partitions.assign_ids`, `ops_partitions.dual_write` (Task 1); `ops_readback.forecasts` (Task 2).
- Produces: nothing new for later tasks.

- [ ] **Step 1: Write the failing tests**

```python
"""Price Forecast writes item_forecasts partitions with Postgres's ids (Phase 1.2)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import duckdb
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _session():
    from database import Base, Item

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add_all([Item(id=1, item_id="ak_1", name="AK 1", type="rifle"), Item(id=2, item_id="ak_2", name="AK 2", type="rifle")])
    db.commit()
    return db


def _results(day, slugs=("ak_1",), mid=10.0):
    return pd.DataFrame(
        [
            {
                "item_id": s,
                "current_price": mid,
                "anchor_clean": True,
                "anchor_wedge_pct": 0.0,
                "anchor_date": day,
                "forecasts": {3: {"low": mid - 1, "mid": mid, "high": mid + 1, "direction": "flat", "confidence": 0.5}},
            }
            for s in slugs
        ]
    )


def _partition(tmp_path, day):
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{tmp_path}/item_forecasts/{day:%Y-%m}.parquet') ORDER BY item_id"
    ).df()


def test_partitions_match_postgres_ids(tmp_path, monkeypatch):
    from database import ItemForecast
    from scripts.forecast_prices import _write_forecasts_to_db

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    _write_forecasts_to_db(db, _results(day), "lgbm-v3", {"ak_1": 1}, day, model_config="global-only")
    pg = db.query(ItemForecast).one()
    got = _partition(tmp_path, day)
    assert got["id"].tolist() == [pg.id]
    assert got["item_slug"].tolist() == ["ak_1"]
    assert got["model_config"].tolist() == ["global-only"]


def test_a_second_write_on_the_same_night_replaces_and_keeps_the_other_batch(tmp_path, monkeypatch):
    from scripts.forecast_prices import _write_forecasts_to_db

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    _write_forecasts_to_db(db, _results(day, ("ak_1",)), "v", {"ak_1": 1, "ak_2": 2}, day, model_config="regime")
    _write_forecasts_to_db(db, _results(day, ("ak_2",)), "v", {"ak_1": 1, "ak_2": 2}, day, model_config="global-only")
    first = _partition(tmp_path, day)
    _write_forecasts_to_db(db, _results(day, ("ak_2",), mid=12.0), "v", {"ak_1": 1, "ak_2": 2}, day, model_config="global-only")
    again = _partition(tmp_path, day)
    assert len(again) == 2
    assert again["id"].tolist() == first["id"].tolist()
    assert again["model_config"].tolist() == ["regime", "global-only"]
    assert again.loc[again.item_id == 2, "price_mid"].item() == 12.0


def test_a_read_only_night_issues_ids_from_the_partitions(tmp_path, monkeypatch):
    from scripts.forecast_prices import _write_forecasts_to_db
    from tests.test_read_only_postgres import _DB, _read_only_exc

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    _write_forecasts_to_db(db, _results(day, ("ak_1",)), "v", {"ak_1": 1}, day)  # writable: id from Postgres
    stored = _partition(tmp_path, day)["id"].max()

    class _Inspector:
        def get_columns(self, name):
            return [{"name": c} for c in ("anchor_clean", "anchor_wedge_pct", "exceed_p", "anomaly_p", "band_multiplier")]

    ro = _DB(_read_only_exc(), [])
    with (
        patch("sqlalchemy.inspect", return_value=_Inspector()),
        patch("sqlalchemy.dialects.postgresql.insert") as ins,
        patch("database.SessionLocal", return_value=ro),
    ):
        ins.return_value.values.return_value.on_conflict_do_update.return_value = "stmt"
        _write_forecasts_to_db(ro, _results(day, ("ak_2",)), "v", {"ak_2": 2}, day)
    got = _partition(tmp_path, day)
    assert got.loc[got.item_id == 2, "id"].item() == stored + 1


def test_a_partition_failure_does_not_fail_the_write(tmp_path, monkeypatch, capsys):
    from scripts.forecast_prices import _write_forecasts_to_db

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    monkeypatch.setattr("db.ops_partitions.upsert", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    day = date.today()
    assert _write_forecasts_to_db(_session(), _results(day), "v", {"ak_1": 1}, day) == 1
    assert "::error title=Ops partition write failed::item_forecasts" in capsys.readouterr().out
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_dual_write_forecasts.py -q`
Expected: FAIL — `item_forecasts/<month>.parquet` does not exist (IO Error from DuckDB), and the failure-annotation test fails on the missing `::error` line.

- [ ] **Step 3: Add the partition write**

In `_write_forecasts_to_db`, after the batch `for` loop and still inside `if forecast_rows:`, add:

```python
        # Phase 1.2 dual-write (docs/specs/2026-10-11-postgres-to-parquet-history-design.md):
        # AFTER the Postgres commit, so the partitions carry the ids Postgres issued. Read back
        # only this batch's keys: regime and global rows share a date and differ in model_config.
        # On a read-only night the ids come from the partitions instead. Never fails the run.
        from db import ops_partitions, ops_readback

        def _partitions():
            import pandas as pd

            if read_only:
                frame = ops_partitions.assign_ids("item_forecasts", pd.DataFrame(forecast_rows))
                frame["item_slug"] = frame["item_id"].map(id_to_slug)
            else:
                keys = {(r["item_id"], r["forecast_date"], r["horizon_days"]) for r in forecast_rows}
                frame = ops_readback.forecasts(db, {r["forecast_date"] for r in forecast_rows}, keys)
            frame["model_config"] = model_config
            ops_partitions.upsert("item_forecasts", frame)

        ops_partitions.dual_write("item_forecasts", _partitions)
```

`db` may have been replaced by a reconnect inside the loop; the closure reads the variable when called, so it uses the live session.

- [ ] **Step 4: Run the new tests and the forecast writer's existing tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_dual_write_forecasts.py tests/test_read_only_postgres.py -q`
Expected: PASS. In `test_read_only_postgres.py` the `events` assertions still hold: the partition write calls neither `append_table` nor `db.execute("stmt")`.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/forecast_prices.py backend/tests/test_ops_dual_write_forecasts.py
git commit -m "feat: forecasts dual-write to the monthly partitions with Postgres ids"
```

---

### Task 4: Shadow candidate writer

**Files:**
- Modify: `backend/db/candidate_store.py` — end of `write_candidate_batches` (after the final `db.commit()`)
- Modify: `backend/scripts/forecast_prices.py` — `_write_shadow_candidates`, the `except` branch that calls `report_read_only`
- Test: `backend/tests/test_ops_dual_write_candidates.py`

**Interfaces:**
- Consumes: Task 1 `upsert`, `assign_ids`, `dual_write`, `SPECS`; Task 2 `ops_readback.candidates`.
- Produces: `db.candidate_store.write_candidate_partitions_read_only(records: list) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
"""Shadow candidates dual-write to forecast_candidates partitions (Phase 1.2)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import duckdb

from tests.test_candidate_store import _record, _session


def _part(tmp_path, day):
    return duckdb.sql(f"SELECT * FROM read_parquet('{tmp_path}/forecast_candidates/{day:%Y-%m}.parquet') ORDER BY id").df()


def test_candidate_partitions_carry_postgres_ids(tmp_path, monkeypatch):
    from database import ForecastCandidate
    from db.candidate_store import write_candidate_batches

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    records = [replace(_record(item_id=1), forecast_date=day), replace(_record(item_id=2), forecast_date=day)]
    write_candidate_batches(db, records)
    ids = sorted(c.id for c in db.query(ForecastCandidate).all())
    assert _part(tmp_path, day)["id"].tolist() == ids


def test_a_retry_rewrites_nothing_new(tmp_path, monkeypatch):
    from db.candidate_store import write_candidate_batches

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    records = [replace(_record(item_id=1), forecast_date=day)]
    write_candidate_batches(db, records)
    write_candidate_batches(db, records)
    assert len(_part(tmp_path, day)) == 1


def test_read_only_candidates_land_with_partition_ids(tmp_path, monkeypatch):
    from db.candidate_store import write_candidate_partitions_read_only

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    write_candidate_partitions_read_only([replace(_record(item_id=1), forecast_date=day)])
    got = _part(tmp_path, day)
    assert got["id"].tolist() == [1]
    assert got["created_at"].notna().all()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_dual_write_candidates.py -q`
Expected: FAIL — partition file missing; `ImportError: cannot import name 'write_candidate_partitions_read_only'`.

- [ ] **Step 3: Implement**

At the end of `write_candidate_batches`, replace the closing `db.commit()` / `return result` with:

```python
    db.commit()
    # Phase 1.2 dual-write: after the commit, with the ids Postgres issued. Never fails the run.
    from db import ops_partitions, ops_readback

    def _partitions():
        frame = ops_readback.candidates(db, {r.forecast_date for r in records}, {_identity(r) for r in records})
        ops_partitions.upsert("forecast_candidates", frame)

    ops_partitions.dual_write("forecast_candidates", _partitions)
    return result
```

Add below it:

```python
def write_candidate_partitions_read_only(records: list) -> None:
    """A read-only night: land the shadow candidates in the partitions with partition-issued ids."""
    from dataclasses import asdict

    import pandas as pd

    from database import utcnow_naive
    from db import ops_partitions

    def _partitions():
        cols = ops_partitions.SPECS["forecast_candidates"].columns
        rows = [{k: v for k, v in asdict(r).items() if k in cols} for r in records]
        frame = ops_partitions.assign_ids("forecast_candidates", pd.DataFrame(rows))
        frame["created_at"] = utcnow_naive()
        ops_partitions.upsert("forecast_candidates", frame)

    ops_partitions.dual_write("forecast_candidates (read-only)", _partitions)
```

In `scripts/forecast_prices.py::_write_shadow_candidates`, change the read-only branch to:

```python
        db.rollback()
        report_read_only(f"the forecast_candidates write ({len(remapped)} shadow rows; partitions only)")
        from db.candidate_store import write_candidate_partitions_read_only

        write_candidate_partitions_read_only(remapped)
        return 0, 0, 0, 0
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_dual_write_candidates.py tests/test_candidate_store.py tests/test_read_only_postgres.py -q`
Expected: PASS. If a `test_candidate_store.py` test uses a fixed `forecast_date` in a now-closed month, its partition write is refused inside `dual_write` and logged; the test still passes because `dual_write` does not raise.

- [ ] **Step 5: Commit**

```bash
git add backend/db/candidate_store.py backend/scripts/forecast_prices.py backend/tests/test_ops_dual_write_candidates.py
git commit -m "feat: shadow candidates dual-write to the monthly partitions, read-only nights included"
```

---

### Task 5: Outcome writers (daily, `--reresolve`, verdict refresh)

**Files:**
- Modify: `backend/scripts/backtest_accuracy.py` — new helper `_outcome_partitions` near `_flush_verdict_refresh` (line 355); call sites in `_flush_verdict_refresh` (after `db.commit()`), and in `_store_forecast_outcomes` (after the `--reresolve` `db.commit()` and after the daily `append_table`)
- Test: `backend/tests/test_ops_dual_write_outcomes.py`

**Interfaces:**
- Consumes: Task 1 `upsert`, `replace`, `dual_write`; Task 2 `ops_readback.outcomes`.
- Produces: `scripts.backtest_accuracy._outcome_partitions(db, forecast_ids: list[int], drop: Iterable[int] | None = None) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
"""Outcome writers dual-write forecast_outcomes partitions (Phase 1.2)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _db(day):
    from database import Base, ForecastOutcome, Item, ItemForecast

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(Item(id=1, item_id="ak_1", name="AK 1", type="rifle"))
    db.add(ItemForecast(id=10, item_id=1, forecast_date=day, horizon_days=3, price_mid=10.0))
    db.add(ForecastOutcome(id=20, forecast_id=10, item_id=1, forecast_date=day, horizon_days=3, actual_price=11.0, in_interval=1))
    db.commit()
    return db


def _part(tmp_path, day):
    return duckdb.sql(f"SELECT * FROM read_parquet('{tmp_path}/forecast_outcomes/{day:%Y-%m}.parquet')").df()


def test_outcome_partitions_carry_postgres_ids(tmp_path, monkeypatch):
    from scripts.backtest_accuracy import _outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    assert _outcome_partitions(_db(day), [10]) is True
    got = _part(tmp_path, day)
    assert got["id"].tolist() == [20]
    assert got["item_slug"].tolist() == ["ak_1"]


def test_reresolve_replaces_dropped_forecasts(tmp_path, monkeypatch):
    from database import ForecastOutcome
    from scripts.backtest_accuracy import _outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today() - timedelta(days=5)
    db = _db(day)
    _outcome_partitions(db, [10])
    db.query(ForecastOutcome).delete()
    db.commit()
    assert _outcome_partitions(db, [], drop=[10]) is True
    assert _part(tmp_path, day).empty


def test_a_closed_month_refresh_is_refused_and_logged(tmp_path, monkeypatch, capsys):
    from scripts.backtest_accuracy import _outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    old = date(2026, 7, 20)
    assert _outcome_partitions(_db(old), [10]) is False
    assert "::error title=Ops partition write failed::forecast_outcomes" in capsys.readouterr().out
    assert not (tmp_path / "forecast_outcomes" / "2026-07.parquet").exists()


def test_the_verdict_refresh_writes_partitions_after_its_commit(monkeypatch):
    import scripts.backtest_accuracy as ba

    events = []

    class _DB:
        def execute(self, *a, **k):
            events.append("db")

        def commit(self):
            events.append("commit")

    monkeypatch.setattr("db.parquet.append_table", lambda *a, **k: events.append("mirror"))
    monkeypatch.setattr(ba, "_outcome_partitions", lambda db, ids, drop=None: events.append(("partitions", tuple(ids))))
    ba._flush_verdict_refresh(_DB(), [{"forecast_id": 10}], [{"forecast_id": 10}])
    assert events == ["mirror", "db", "commit", ("partitions", (10,))]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_dual_write_outcomes.py -q`
Expected: FAIL with `ImportError: cannot import name '_outcome_partitions'`

- [ ] **Step 3: Implement**

Add above `_flush_verdict_refresh`:

```python
def _outcome_partitions(db, forecast_ids, drop=None) -> bool:
    """Write committed outcome rows to the monthly partitions (Phase 1.2 dual-write).

    Runs AFTER the DB commit, so the rows carry Postgres's ids; the single-file mirror keeps its
    mirror-first order above. *drop*: forecast ids whose stored outcome a --reresolve removed.
    A refused closed month is logged, never raised; tonight's parity check names it.
    """
    from db import ops_partitions, ops_readback

    def _write():
        frame = ops_readback.outcomes(db, forecast_ids=list(forecast_ids))
        if drop is not None:
            ops_partitions.replace("forecast_outcomes", "forecast_id", drop, frame)
        else:
            ops_partitions.upsert("forecast_outcomes", frame)

    return ops_partitions.dual_write("forecast_outcomes", _write)
```

In `_flush_verdict_refresh`, after `db.commit()`:

```python
    _outcome_partitions(db, [m["forecast_id"] for m in mirror])
```

In `_store_forecast_outcomes`, in the `if delete_ids:` branch after `db.commit()`:

```python
        _outcome_partitions(db, [o["forecast_id"] for o in to_write], drop=delete_ids)
```

and in the daily path after `append_table("forecast_outcomes", mirror, ["forecast_id"])`:

```python
    _outcome_partitions(db, [o["forecast_id"] for o in to_write])
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_dual_write_outcomes.py tests/test_backtest_scoring.py tests/test_accuracy_mirror_write_order.py -q`
Expected: PASS. Existing tests that use fake sessions without `get_bind()` make `ops_readback` raise inside `dual_write`, which logs and returns False, so they still pass. If one asserts on exact log output, monkeypatch `scripts.backtest_accuracy._outcome_partitions` to a no-op in that test.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/backtest_accuracy.py backend/tests/test_ops_dual_write_outcomes.py
git commit -m "feat: outcome writers dual-write to the monthly partitions after the DB commit"
```

---

### Task 6: Candidate outcome writer

**Files:**
- Modify: `backend/backtest/candidate_resolution.py` — `_create_or_update` (line 245) gains a `touched: set[int]` parameter; its three callers in `resolve_candidate_outcomes` pass one; after the commit at line 235 and the commit at line 104 (`_rescore_candidate_outcomes`), write partitions
- Test: `backend/tests/test_ops_dual_write_candidate_outcomes.py`

**Interfaces:**
- Consumes: Task 1 `upsert`, `dual_write`; Task 2 `ops_readback.candidate_outcomes`.
- Produces: `backtest.candidate_resolution._candidate_outcome_partitions(db, candidate_ids: set[int]) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
"""Candidate outcomes dual-write, routed by their candidate's forecast_date (Phase 1.2)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _db(cand_day):
    from database import Base, ForecastCandidate, ForecastCandidateOutcome

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(ForecastCandidate(
        id=30, item_id=1, forecast_date=cand_day, horizon_days=7, component="centre", candidate_name="last_price",
        candidate_version="v1", config_fingerprint="a" * 64, feature_cutoff_at=datetime(2026, 10, 1),
        artifact_version="x", anchor_price=10.0,
    ))
    db.add(ForecastCandidateOutcome(candidate_id=30, base_price=10.0, actual_price=11.0, resolved_at=datetime.now(), resolution_version="r1"))
    db.commit()
    return db


def test_routed_by_the_candidate_s_date_not_the_resolution_date(tmp_path, monkeypatch):
    from backtest.candidate_resolution import _candidate_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    cand_day = (date.today().replace(day=1) - timedelta(days=1))  # last day of the previous month
    assert _candidate_outcome_partitions(_db(cand_day), {30}) is True
    path = tmp_path / "forecast_candidate_outcomes" / f"{cand_day:%Y-%m}.parquet"
    got = duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()
    assert got["candidate_id"].tolist() == [30]
    assert "_part" not in got.columns


def test_nothing_touched_writes_nothing(tmp_path, monkeypatch):
    from backtest.candidate_resolution import _candidate_outcome_partitions

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    assert _candidate_outcome_partitions(_db(date.today()), set()) is True
    assert not (tmp_path / "forecast_candidate_outcomes").exists()
```

Also add to `backend/tests/test_candidate_resolution.py`, beside its existing end-to-end `resolve_candidate_outcomes` test (copy that test's setup), an assertion that `_candidate_outcome_partitions` is called with the resolved candidate ids:

```python
def test_resolution_writes_partitions_for_touched_candidates(monkeypatch):
    import backtest.candidate_resolution as cr

    calls = []
    monkeypatch.setattr(cr, "_candidate_outcome_partitions", lambda db, ids: calls.append(set(ids)))
    # ...the existing end-to-end test's setup and its resolve_candidate_outcomes(...) call...
    assert calls and calls[-1]  # the resolved candidate ids
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_dual_write_candidate_outcomes.py -q`
Expected: FAIL with `ImportError: cannot import name '_candidate_outcome_partitions'`

- [ ] **Step 3: Implement**

Add to `backtest/candidate_resolution.py`:

```python
def _candidate_outcome_partitions(db, candidate_ids) -> bool:
    """Write committed candidate outcomes to the monthly partitions (Phase 1.2 dual-write).

    Routed by the candidate's forecast_date (`_part`), not the resolution date. Never raises.
    """
    from db import ops_partitions, ops_readback

    ids = sorted(candidate_ids)
    if not ids:
        return True

    def _write():
        ops_partitions.upsert("forecast_candidate_outcomes", ops_readback.candidate_outcomes(db, candidate_ids=ids))

    return ops_partitions.dual_write("forecast_candidate_outcomes", _write)
```

Change `_create_or_update(db, existing, candidate, base, actual, resolved_at, stats)` to `_create_or_update(db, existing, candidate, base, actual, resolved_at, stats, touched)` and add `touched.add(candidate.id)` as its first line after `derived = ...`. In `resolve_candidate_outcomes`, create `touched: set[int] = set()` before the loops, pass it at all three `_create_or_update` calls, and after the `db.commit()` at line 235 add:

```python
    _candidate_outcome_partitions(db, touched)
```

In `_rescore_candidate_outcomes`, collect `changed_ids.add(outcome.candidate_id)` where it sets `changed = True`, and after its `db.commit()` add `_candidate_outcome_partitions(db, changed_ids)`.

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_dual_write_candidate_outcomes.py tests/test_candidate_resolution.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/backtest/candidate_resolution.py backend/tests/test_ops_dual_write_candidate_outcomes.py backend/tests/test_candidate_resolution.py
git commit -m "feat: candidate outcomes dual-write, routed by the candidate's forecast date"
```

---

### Task 7: Catalog writer

**Files:**
- Modify: `backend/collectors/new_item_discovery.py:153` (after `session.commit()` in `discover_new_items`)
- Test: `backend/tests/test_new_item_discovery.py` (add one test)

**Interfaces:**
- Consumes: Task 1 `upsert`, `dual_write`; Task 2 `ops_readback.items`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_new_item_discovery.py`, reusing its existing session/dump fixtures (find the test that asserts an insert happened and copy its setup):

```python
def test_inserted_items_land_in_items_parquet(tmp_path, monkeypatch):
    import duckdb

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    # ...the existing insert test's setup, then:
    n = discover_new_items(session, raw_sources, snapshot_day)
    assert n > 0
    got = duckdb.sql(f"SELECT item_id FROM read_parquet('{tmp_path}/items.parquet')").df()
    assert len(got) == n
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && venv/bin/python -m pytest tests/test_new_item_discovery.py -q -k items_parquet`
Expected: FAIL — `items.parquet` does not exist.

- [ ] **Step 3: Implement**

After `session.commit()` in `discover_new_items`:

```python
    # Phase 1.2 dual-write: the new rows with the ids Postgres issued. Never fails the run.
    from db import ops_partitions, ops_readback

    ops_partitions.dual_write(
        "items", lambda: ops_partitions.upsert("items", ops_readback.items(session, [name for name, _ in new]))
    )
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_new_item_discovery.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/collectors/new_item_discovery.py backend/tests/test_new_item_discovery.py
git commit -m "feat: new-item discovery dual-writes items.parquet"
```

---

### Task 8: Fill and repair from Postgres

**Files:**
- Create: `backend/scripts/ops_sync_from_postgres.py`
- Create: `.github/workflows/ops-sync.yml`
- Test: `backend/tests/test_ops_sync_from_postgres.py`

**Interfaces:**
- Consumes: Task 1 `upsert`, `month_files`, `SPECS`; Task 2 read-back functions.
- Produces: `sync(db, dates: list[date], *, tables: list[str], apply: bool, allow_closed: bool = False, root=None, today=None) -> dict[str, int]` (rows read from Postgres per table); `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
"""scripts/ops_sync_from_postgres.py: fill ids and repair dates from Postgres."""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pandas as pd

from tests.test_ops_readback import D, _seed, _session

TODAY = date(2026, 10, 12)
TABLES = ["item_forecasts", "forecast_outcomes", "forecast_candidates", "forecast_candidate_outcomes"]


def _mirror_only_rows(tmp_path):
    """The 10-11 state: rows with no id, one extra column Postgres lacks."""
    from db.ops_partitions import upsert

    frame = pd.DataFrame([
        {"id": None, "item_id": 1, "forecast_date": D, "horizon_days": 7, "price_mid": 10.0, "model_config": "global-only"},
        {"id": None, "item_id": 1, "forecast_date": D, "horizon_days": 3, "price_mid": 9.0, "model_config": "global-only"},
    ])
    upsert("item_forecasts", frame, root=tmp_path, today=TODAY)


def test_dry_run_writes_nothing(tmp_path):
    from scripts.ops_sync_from_postgres import sync

    db = _session()
    _seed(db)
    _mirror_only_rows(tmp_path)
    before = (tmp_path / "item_forecasts" / "2026-10.parquet").read_bytes()
    plan = sync(db, [D], tables=TABLES, apply=False, root=tmp_path, today=TODAY)
    assert plan["item_forecasts"] == 2
    assert (tmp_path / "item_forecasts" / "2026-10.parquet").read_bytes() == before
    assert not (tmp_path / "forecast_outcomes").exists()


def test_apply_fills_ids_keeps_mirror_columns_and_adds_nothing_postgres_lacks(tmp_path):
    from scripts.ops_sync_from_postgres import sync

    db = _session()
    _seed(db)
    _mirror_only_rows(tmp_path)
    sync(db, [D], tables=TABLES, apply=True, root=tmp_path, today=TODAY)
    got = duckdb.sql(f"SELECT * FROM read_parquet('{tmp_path}/item_forecasts/2026-10.parquet') ORDER BY horizon_days").df()
    assert got["id"].tolist() == [11, 10]
    assert got["model_config"].tolist() == ["global-only", "global-only"]
    outcomes = duckdb.sql(f"SELECT id FROM read_parquet('{tmp_path}/forecast_outcomes/2026-10.parquet')").df()
    assert outcomes["id"].tolist() == [20]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_sync_from_postgres.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.ops_sync_from_postgres'`

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python3
"""
Copy named forecast dates from Postgres into the monthly ops partitions, ids included.

Phase 1.2 of docs/specs/2026-10-11-postgres-to-parquet-history-design.md. Two uses:

* FILL: the 22,507 forecasts dated 2026-10-11 reached the partitions from the single-file
  mirror, which has no id. Postgres has them; this writes Postgres's rows over them by
  (item_id, forecast_date, horizon_days). Columns Postgres lacks (model_config, item_slug for
  candidate rows, pg_deleted) keep their stored values.
* REPAIR: a night whose partition write failed (`dual_write` logged it, parity failed).

Writes only rows Postgres holds; never deletes. Dry run unless --apply. Closed months need
--allow-closed. Run by dispatching ops-sync.yml: only CI writes the data repo.

Usage:
    python scripts/ops_sync_from_postgres.py --archive-dir ../archive/price-archive --dates 2026-10-11
    python scripts/ops_sync_from_postgres.py --archive-dir ... --dates 2026-10-11 --apply
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import ops_partitions, ops_readback  # noqa: E402

logger = logging.getLogger("ops_sync_from_postgres")

TABLES = ("item_forecasts", "forecast_outcomes", "forecast_candidates", "forecast_candidate_outcomes")

_READ = {
    "item_forecasts": lambda db, ds: ops_readback.forecasts(db, ds),
    "forecast_outcomes": lambda db, ds: ops_readback.outcomes(db, dates=ds),
    "forecast_candidates": lambda db, ds: ops_readback.candidates(db, ds),
    "forecast_candidate_outcomes": lambda db, ds: ops_readback.candidate_outcomes(db, dates=ds),
}


def sync(db, dates, *, tables, apply, allow_closed=False, root=None, today=None) -> dict[str, int]:
    plan = {}
    for t in tables:
        frame = _READ[t](db, dates)
        plan[t] = len(frame)
        logger.info("%s: %d rows in Postgres for %s", t, len(frame), ", ".join(map(str, dates)))
        if apply and not frame.empty:
            out = ops_partitions.upsert(t, frame, allow_closed=allow_closed, root=root, today=today)
            logger.info("%s: wrote %s", t, out)
    if not apply:
        logger.info("dry run; pass --apply to write")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--archive-dir", type=Path, required=True, help="The data checkout's price-archive/ directory.")
    ap.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    ap.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--allow-closed", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from database import SessionLocal

    db = SessionLocal()
    try:
        sync(db, args.dates, tables=args.tables, apply=args.apply, allow_closed=args.allow_closed, root=args.archive_dir / "ops")
    except ops_partitions.PartitionError as e:
        logger.error("%s", e)
        return 1
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Copy `.github/workflows/ops-partition-backfill.yml` to `.github/workflows/ops-sync.yml` and change:
- `name: Ops Sync From Postgres`; the header comment to describe this script.
- `inputs`: keep `apply`; add `dates` (`type: string`, required, "space-separated YYYY-MM-DD") and `allow_closed` (`type: boolean`, default false).
- Delete the "Fetch and verify the backup" step.
- Replace the "Backfill" step with:

```yaml
      - name: Sync
        working-directory: ./backend
        env:
          DATABASE_URL: ${{ secrets.SUPABASE_DATABASE_URL }}
        run: |
          python scripts/ops_sync_from_postgres.py --archive-dir ../archive/price-archive \
            --dates ${{ inputs.dates }} \
            ${{ inputs.apply && '--apply' || '' }} ${{ inputs.allow_closed && '--allow-closed' || '' }}
```

- In the publish step, the commit message becomes `archive: ops partitions synced from Postgres (${{ inputs.dates }})`.

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_sync_from_postgres.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/ops_sync_from_postgres.py backend/tests/test_ops_sync_from_postgres.py .github/workflows/ops-sync.yml
git commit -m "feat: ops sync from Postgres — fill missing ids and repair dates, dispatch only"
```

---

### Task 9: Nightly parity check

**Files:**
- Create: `backend/scripts/ops_parity.py`
- Modify: `.github/workflows/backtest-accuracy.yml` — new step before "Publish updated archive (flat history)", new last step after "Release closed ops months"
- Test: `backend/tests/test_ops_parity.py`

**Interfaces:**
- Consumes: Task 1 `SPECS`, `month_files`, `ops_dir`; `db.parquet.append_table`.
- Produces: `Mismatch(table, forecast_date, field, postgres, partitions)`; `run_parity(db, root=None) -> list[Mismatch]`; `record(mismatches, run_date: date, root=None) -> None`; `clean_streak(root=None) -> int`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
"""scripts/ops_parity.py: Postgres and the ops partitions agree per table and forecast_date."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tests.test_ops_readback import D, _seed, _session

TODAY = date(2026, 10, 12)


def _mirror_postgres(db, tmp_path):
    from db import ops_partitions, ops_readback

    ops_partitions.upsert("item_forecasts", ops_readback.forecasts(db, {D}), root=tmp_path, today=TODAY)
    ops_partitions.upsert("forecast_outcomes", ops_readback.outcomes(db, dates=[D]), root=tmp_path, today=TODAY)
    ops_partitions.upsert("forecast_candidates", ops_readback.candidates(db, {D}), root=tmp_path, today=TODAY)
    ops_partitions.upsert("forecast_candidate_outcomes", ops_readback.candidate_outcomes(db, dates=[D]), root=tmp_path, today=TODAY)


def test_identical_stores_pass(tmp_path):
    from scripts.ops_parity import run_parity

    db = _session()
    _seed(db)
    _mirror_postgres(db, tmp_path)
    assert run_parity(db, root=tmp_path) == []


def test_pg_deleted_rows_do_not_count(tmp_path):
    from db.ops_partitions import upsert
    from scripts.ops_parity import run_parity

    db = _session()
    _seed(db)
    _mirror_postgres(db, tmp_path)
    upsert("item_forecasts", pd.DataFrame([{"id": None, "item_id": 9, "forecast_date": D, "horizon_days": 7, "pg_deleted": True}]), root=tmp_path, today=TODAY)
    assert run_parity(db, root=tmp_path) == []


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        ("DELETE FROM item_forecasts WHERE id = 11", "n"),
        ("UPDATE item_forecasts SET id = 99 WHERE id = 11", "id_sum"),
        ("UPDATE forecast_outcomes SET in_interval = 0", "n_in_interval"),
    ],
)
def test_a_missing_row_a_wrong_id_and_a_changed_verdict_fail(tmp_path, mutate, field):
    from sqlalchemy import text

    from scripts.ops_parity import run_parity

    db = _session()
    _seed(db)
    db.execute(text("UPDATE forecast_outcomes SET in_interval = 1"))
    db.commit()
    _mirror_postgres(db, tmp_path)
    db.execute(text(mutate))
    db.commit()
    got = run_parity(db, root=tmp_path)
    assert any(m.field == field and m.forecast_date == D for m in got), got


def test_streak_counts_consecutive_clean_nights(tmp_path, monkeypatch):
    from scripts.ops_parity import Mismatch, clean_streak, record

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    record([Mismatch("item_forecasts", D, "n", 1, 0)], date(2026, 10, 12))
    record([], date(2026, 10, 13))
    record([], date(2026, 10, 14))
    assert clean_streak() == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_parity.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.ops_parity'`

- [ ] **Step 3: Write the script**

```python
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

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from db.ops_partitions import SPECS, month_files, ops_dir  # noqa: E402

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
    forecast_date: date
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


def run_parity(db, root=None) -> list[Mismatch]:
    out = []
    for table in SUMMARY:
        out += compare(table, *_summaries(db, table, root))
    return out


def record(mismatches: list[Mismatch], run_date: date, root=None) -> None:
    from db.parquet import append_table

    now = datetime.now(UTC).replace(tzinfo=None)
    rows = [
        {
            "run_date": run_date,
            "table": t,
            "n_mismatches": sum(1 for m in mismatches if m.table == t),
            "ok": not any(m.table == t for m in mismatches),
            "run_at": now,
        }
        for t in SUMMARY
    ]
    append_table("parity_runs", rows, ["run_date", "table"])


def clean_streak(root=None) -> int:
    path = ops_dir(root) / "parity_runs.parquet"
    if not path.exists():
        return 0
    nights = duckdb.sql(f"SELECT run_date, bool_and(ok) AS ok FROM read_parquet('{path}') GROUP BY 1 ORDER BY 1 DESC").fetchall()
    streak = 0
    for _, ok in nights:
        if not ok:
            break
        streak += 1
    return streak


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from database import SessionLocal

    db = SessionLocal()
    try:
        mismatches = run_parity(db)
    finally:
        db.close()
    record(mismatches, datetime.now(UTC).date())
    for m in mismatches[:50]:
        logger.error("%s %s %s: postgres=%s partitions=%s", m.table, m.forecast_date, m.field, m.postgres, m.partitions)
    if len(mismatches) > 50:
        logger.error("... and %d more", len(mismatches) - 50)
    streak = clean_streak()
    logger.info("parity: %d mismatches; clean nights in a row: %d (Phase 1 needs 7)", len(mismatches), streak)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
```

In `.github/workflows/backtest-accuracy.yml`, insert before "Publish updated archive (flat history)":

```yaml
      # Phase 1.2 parity (docs/specs/2026-10-11-postgres-to-parquet-history-design.md): per table
      # and forecast_date, Postgres and the monthly partitions must agree. Before the publish so
      # ops/parity_runs.parquet is pushed; continue-on-error so a mismatch does not block the
      # closed-month release. "Fail on parity mismatch" at the end turns the run red.
      - name: Ops parity
        id: parity
        continue-on-error: true
        working-directory: ./backend
        env:
          DATABASE_URL: ${{ secrets.SUPABASE_DATABASE_URL }}
        run: python scripts/ops_parity.py
```

and after "Release closed ops months":

```yaml
      - name: Fail on parity mismatch
        if: steps.parity.outcome == 'failure'
        run: |
          echo "::error title=Ops parity::Postgres and the ops partitions disagree; see the Ops parity step"
          exit 1
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_ops_parity.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/ops_parity.py backend/tests/test_ops_parity.py .github/workflows/backtest-accuracy.yml
git commit -m "feat: nightly ops parity — per-date aggregates from Postgres and the partitions must agree"
```

---

### Task 10: Docs, full suite, hand-off

**Files:**
- Create: `docs/changelog/2026-10-12-ops-dual-write.md`
- Modify: `docs/specs/2026-10-11-postgres-to-parquet-history-design.md` (Phase 1 item 2: add "**Built 2026-10-12**" with the PR number once known)
- Modify: `.claude/rules/archive-reads.md` (one bullet: partitions are written after the Postgres commit with Postgres ids; `db/ops_partitions.py` is the only writer; parity in Backtest Accuracy)

- [ ] **Step 1: Write the changelog**

Follow the format of `docs/changelog/2026-10-11-ops-partition-backfill.md`. Cover: what is dual-written and where; ids from Postgres read back after the commit; read-only fallback; `dual_write` never fails a run; parity rules and `parity_runs.parquet`; the 10-11 fill (22,507 rows; the 22,043 `pg_deleted` rows of 08-04/08-05 stay id-less); next steps (dispatch `ops-sync.yml` for 10-11 dry run then apply; count 7 clean nights; Phase 2).

- [ ] **Step 2: Run the full suite**

Run: `cd backend && venv/bin/python -m pytest tests/ -q`
Expected: PASS, no new failures against `main`. If something unrelated already fails on `main`, confirm with `git stash && venv/bin/python -m pytest tests/<file> -q && git stash pop` and note it in the PR body.

- [ ] **Step 3: Lint**

Run: `cd backend && venv/bin/ruff check db/ops_partitions.py db/ops_readback.py scripts/ops_parity.py scripts/ops_sync_from_postgres.py && venv/bin/ruff format --check db scripts tests`
Expected: no errors (fix and re-run if any).

- [ ] **Step 4: Commit**

```bash
git add docs/changelog/2026-10-12-ops-dual-write.md docs/specs/2026-10-11-postgres-to-parquet-history-design.md .claude/rules/archive-reads.md
git commit -m "docs: ops dual-write changelog and rule"
```

- [ ] **Step 5: Hand back**

Do not push or open the PR without the user's go-ahead. Report: commits, test counts, and the post-merge sequence:
1. The first nightly chain after merge writes partitions; parity will fail on 10-11 (`n_id`, `id_sum`) until the fill.
2. Dispatch `ops-sync.yml` with `dates=2026-10-11`, dry run, then `apply`.
3. The next parity run should be clean; that is night one of seven.
