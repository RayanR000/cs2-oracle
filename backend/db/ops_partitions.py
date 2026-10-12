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


# Column lists and types are what the 2026-10-12 backfill wrote (DESCRIBE on the data repo), except
# items.instanceid / rarity / weapon_type: the backfill saw only NULLs and typed them INTEGER, but
# the ORM columns are strings (an instanceid of 3946324730 does not fit INTEGER).
SPECS: dict[str, Spec] = {
    "item_forecasts": Spec(
        FORECAST_KEY,
        "item_id, forecast_date, horizon_days",
        {
            "id": "BIGINT",
            "item_id": "BIGINT",
            "forecast_date": "DATE",
            "horizon_days": "BIGINT",
            "price_low": "DOUBLE",
            "price_mid": "DOUBLE",
            "price_high": "DOUBLE",
            "current_price": "DOUBLE",
            "direction": "VARCHAR",
            "confidence": "VARCHAR",
            "model_version": "VARCHAR",
            "created_at": _TS,
            "anchor_clean": "BOOLEAN",
            "anchor_wedge_pct": "DOUBLE",
            "exceed_p": "DOUBLE",
            "anomaly_p": "DOUBLE",
            "band_multiplier": "DOUBLE",
            "model_config": "VARCHAR",
            "item_slug": "VARCHAR",
            "pg_deleted": "BOOLEAN",
        },
        "forecast_date",
    ),
    "forecast_outcomes": Spec(
        ("forecast_id",),
        "forecast_date, item_id, horizon_days, forecast_id",
        {
            "id": "BIGINT",
            "forecast_id": "BIGINT",
            "item_id": "BIGINT",
            "forecast_date": "DATE",
            "horizon_days": "BIGINT",
            "target_date": _TS,
            "current_price": "DOUBLE",
            "predicted_price_low": "DOUBLE",
            "predicted_price_mid": "DOUBLE",
            "predicted_price_high": "DOUBLE",
            "actual_price": "DOUBLE",
            "direction_predicted": "VARCHAR",
            "direction_actual": "VARCHAR",
            "direction_correct": "BIGINT",
            "in_interval": "BIGINT",
            "abs_error": "DOUBLE",
            "pct_error": "DOUBLE",
            "model_version": "VARCHAR",
            "evaluated_at": _TS,
            "base_price": "DOUBLE",
            "resolved_at": _TS,
            "base_stale_run_days": "DOUBLE",
            "item_slug": "VARCHAR",
            "pg_deleted": "BOOLEAN",
        },
        "forecast_date",
    ),
    "forecast_candidates": Spec(
        CANDIDATE_KEY,
        ", ".join(CANDIDATE_KEY),
        {
            "id": "BIGINT",
            "item_id": "BIGINT",
            "forecast_date": "DATE",
            "horizon_days": "BIGINT",
            "component": "VARCHAR",
            "candidate_name": "VARCHAR",
            "candidate_version": "VARCHAR",
            "centre_price": "DOUBLE",
            "predicted_price_low": "DOUBLE",
            "predicted_price_high": "DOUBLE",
            "score": "DOUBLE",
            "anchor_price": "DOUBLE",
            "feature_cutoff_at": _TS,
            "artifact_version": "VARCHAR",
            "config_fingerprint": "VARCHAR",
            "created_at": _TS,
        },
        "forecast_date",
    ),
    "forecast_candidate_outcomes": Spec(
        ("candidate_id",),
        "candidate_id",
        {
            "candidate_id": "BIGINT",
            "base_price": "DOUBLE",
            "actual_price": "DOUBLE",
            "resolved_at": _TS,
            "absolute_error": "DOUBLE",
            "percentage_error": "DOUBLE",
            "in_interval": "BOOLEAN",
            "resolution_version": "VARCHAR",
        },
        "_part",
    ),
    "items": Spec(
        ("id",),
        "id",
        {
            "id": "BIGINT",
            "item_id": "VARCHAR",
            "name": "VARCHAR",
            "type": "VARCHAR",
            "release_date": _TS,
            "created_at": _TS,
            "updated_at": _TS,
            "icon_url": "VARCHAR",
            "classid": "VARCHAR",
            "instanceid": "VARCHAR",
            "is_backfilled": "BIGINT",
            "rarity": "VARCHAR",
            "rarity_rank": "INTEGER",
            "weapon_type": "VARCHAR",
            "is_trainable": "BIGINT",
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


_EMPTY_DTYPE = {
    "BIGINT": "Int64",
    "INTEGER": "Int64",
    "DOUBLE": "float64",
    "BOOLEAN": "boolean",
    "DATE": "datetime64[ns]",
    _TS: "datetime64[ns]",
    "VARCHAR": "string",
}


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
        body = (
            "SELECT "
            + ", ".join(
                f"CAST(n.{c} AS {t}) AS {c}" if c in have else f"CAST(NULL AS {t}) AS {c}"
                for c, t in spec.columns.items()
            )
            + " FROM _new n"
        )
        before = removed = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        con.execute(
            f"COPY (SELECT * FROM ({body}) ORDER BY {spec.order}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
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


def upsert(
    table: str, frame: pd.DataFrame, *, allow_closed: bool = False, root=None, today: date | None = None
) -> dict[str, int]:
    """Write *frame* into its months, replacing rows with the same key. Returns rows per month after."""
    return _apply(table, frame, None, allow_closed, root, today)


def replace(
    table: str,
    key_col: str,
    keys: Iterable,
    frame: pd.DataFrame,
    *,
    allow_closed: bool = False,
    root=None,
    today: date | None = None,
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
            hit = dict(
                con.execute(
                    f"SELECT n._pos, e.id FROM n JOIN (SELECT * FROM {rel} WHERE id IS NOT NULL) e ON {_on(spec)}"
                ).fetchall()
            )
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


def dual_write(what: str, fn: Callable[[], object], db=None) -> bool:
    """Run one partition write; log a failure instead of raising.

    By the time this runs, Postgres and the single-file mirror hold the rows, so nothing is lost,
    and failing the run after its commit would only endanger the rest of the chain. The nightly
    parity check fails the same night instead. Pass the writer's *db* so a failure rolls it back.
    """
    try:
        fn()
    except Exception:
        if db is not None:
            # A failed read-back leaves a Postgres transaction aborted, and the caller's next
            # statement on this session would raise InFailedSqlTransaction. Its commit is done.
            try:
                db.rollback()
            except Exception:
                logger.exception("ops partitions: rollback after the %s failure also failed", what)
        logger.exception("ops partitions: %s write failed; Postgres holds the rows, parity will fail tonight", what)
        print(f"::error title=Ops partition write failed::{what}", flush=True)
        return False
    return True
