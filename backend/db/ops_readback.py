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
