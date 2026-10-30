"""The tier bias fit reads forecast_outcomes from the ops partitions (Phase 2 PR 4).

Off by default since #113; it moves anyway so it cannot fit on a stale Postgres panel if turned
on. No read twice — one equivalence test instead (docs/specs/2026-10-13-ops-readers-switch-design.md).
"""

import datetime as dt

import pandas as pd
import pytest
from database import Base, ForecastOutcome
from db import ops_store
from models.forecaster import ItemForecaster
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests._ops_mirror import mirror_session_to_ops


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    engine.dispose()


def _outcomes(n_dates=24, per_date=6):
    rows = []
    for d in range(n_dates):
        fd = dt.date(2026, 9, 1) + dt.timedelta(days=d)
        for k in range(per_date):
            fid = d * per_date + k + 1
            rows.append(
                dict(
                    id=fid,
                    forecast_id=fid,
                    item_id=fid,
                    forecast_date=fd,
                    horizon_days=7,
                    target_date=fd + dt.timedelta(days=7),
                    current_price=10.0,
                    actual_price=10.0,
                    abs_error=0.0,
                    predicted_price_mid=10.0 * (1 + (k - 2.5) / 100),
                    direction_actual=("up", "down", "flat")[(d + k) % 3],
                    model_version="lgbm-v3-clustered",
                )
            )
    return rows


def _fit(session):
    f = ItemForecaster.__new__(ItemForecaster)
    f.db = session
    f.bias_thresholds = {}
    f.bias_ewma_state = {}
    f._save_bias_corrections = lambda: None
    f.update_bias_corrections_from_outcomes()
    return f.bias_thresholds


def test_bias_fit_is_identical_from_both_stores(session, monkeypatch):
    for r in _outcomes():
        session.add(ForecastOutcome(**r))
    session.commit()
    mirror_session_to_ops(session)

    store = _fit(session)
    monkeypatch.setattr(ops_store, "reader", lambda tables=ops_store.TABLES, root=None: ops_store.pg_reader(session))
    pg = _fit(session)

    assert store
    assert store == pg


def test_bias_fit_reads_partition_only_outcomes(session):
    from db import ops_partitions

    ops_partitions.upsert("forecast_outcomes", pd.DataFrame(_outcomes()), allow_closed=True)

    assert 7 in _fit(session)
