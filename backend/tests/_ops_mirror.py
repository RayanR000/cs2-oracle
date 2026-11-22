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


def ops_to_session(session, table: str, root=None) -> None:
    """The reverse of `mirror_session_to_ops` for one table: replace the session's rows with the
    partitions'. Since Phase 4 the backtest writes only the partitions; conftest's `ops_mirror`
    calls this after each write so a test can assert on what landed through the ORM."""
    import datetime as dt
    import math

    from database import Base
    from sqlalchemy import Date, DateTime

    files = ops_partitions.month_files(table, root)
    con = duckdb.connect()
    try:
        frame = (
            con.execute(f"SELECT * FROM read_parquet({[str(f) for f in files]}, union_by_name = true)").df()
            if files
            else pd.DataFrame()
        )
    finally:
        con.close()
    t = Base.metadata.tables[table]

    def value(col, v):
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
            return None
        if isinstance(col.type, DateTime) and hasattr(v, "to_pydatetime"):
            return v.to_pydatetime()
        if isinstance(col.type, Date) and not isinstance(col.type, DateTime):
            return v.date() if isinstance(v, dt.datetime) or hasattr(v, "to_pydatetime") else v
        return v.item() if hasattr(v, "item") else v

    rows = [
        {c.name: value(c, r[c.name]) for c in t.columns if c.name in frame.columns}
        for r in frame.astype(object).to_dict("records")
    ]
    session.execute(t.delete())
    if rows:
        session.execute(t.insert(), rows)
    session.flush()


def write_back(monkeypatch, get_session) -> None:
    """Since Phase 4 the backtest writes only the partitions. After each `upsert`/`replace`, copy
    the written table back into the test's session (`get_session()`), so the next mirror keeps it
    and the test's ORM assertions read what landed. The tests' dates sit in closed months, so
    those are allowed."""
    import db.ops_partitions as parts

    for name in ("upsert", "replace"):
        real_write = getattr(parts, name)

        def write(*args, _real=real_write, **kwargs):
            kwargs["allow_closed"] = True
            out = _real(*args, **kwargs)
            session = get_session()
            if session is not None:
                ops_to_session(session, args[0], kwargs.get("root"))
            return out

        monkeypatch.setattr(parts, name, write)
