"""The prior-day blend reads yesterday's forecast from the ops partitions (Phase 2 PR 4).

docs/specs/2026-10-13-ops-readers-switch-design.md. On a read-only night the forecast exists only
in the partitions (ids from READ_ONLY_ID_BASE), so a Postgres read would blend a stale day.
"""

import datetime as dt
import logging

import numpy as np
import pandas as pd
import pytest
from database import Base, Item, ItemForecast
from db import ops_partitions, ops_store
from models.forecaster import ItemForecaster
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests._ops_mirror import mirror_session_to_ops

ANCHOR = dt.datetime(2026, 9, 25)


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Item(id=1, item_id="a", name="a", type="skin", is_backfilled=1))
    s.add(Item(id=2, item_id="b", name="b", type="skin", is_backfilled=1))
    s.commit()
    yield s
    engine.dispose()


@pytest.fixture()
def forecaster(session):
    mirror_session_to_ops(session, tables=("items",))  # the slug map's catalog (Phase 2 PR 5)
    f = ItemForecaster.__new__(ItemForecaster)
    f.db = session
    f._now = lambda: ANCHOR
    return f


def _row(fid, item_id, fd, mid=12.0, **extra):
    return {
        "id": fid,
        "item_id": item_id,
        "forecast_date": fd,
        "horizon_days": 3,
        "price_low": 11.0,
        "price_mid": mid,
        "price_high": 13.0,
        "current_price": 10.0,
        "band_multiplier": 1.0,
        **extra,
    }


def _write(*rows):
    ops_partitions.upsert("item_forecasts", pd.DataFrame(list(rows)), allow_closed=True)


def test_prior_fetch_reads_the_partitions(forecaster):
    _write(_row(1, 1, dt.date(2026, 9, 24)))

    out = forecaster._fetch_prior_forecasts(np.array(["a", "b"]), 3)

    assert out["mask"].tolist() == [True, False]
    assert out["mid_ret"][0] == pytest.approx(20.0)


def test_prior_fetch_uses_partition_only_forecasts(forecaster):
    _write(
        _row(7, 1, dt.date(2026, 9, 23), mid=11.0),
        _row(ops_partitions.READ_ONLY_ID_BASE + 5, 1, dt.date(2026, 9, 24), mid=12.0),
    )

    out = forecaster._fetch_prior_forecasts(np.array(["a"]), 3)

    assert out["mid_ret"][0] == pytest.approx(20.0)


def test_prior_fetch_hides_pg_deleted(forecaster):
    _write(_row(7, 1, dt.date(2026, 9, 23), mid=11.0), _row(8, 1, dt.date(2026, 9, 24), mid=12.0, pg_deleted=True))

    out = forecaster._fetch_prior_forecasts(np.array(["a"]), 3)

    assert out["mid_ret"][0] == pytest.approx(10.0)


def test_prior_fetch_respects_the_replay_anchor(forecaster):
    _write(_row(7, 1, dt.date(2026, 9, 24), mid=11.0), _row(8, 1, dt.date(2026, 9, 25), mid=12.0))

    out = forecaster._fetch_prior_forecasts(np.array(["a"]), 3)

    assert out["mid_ret"][0] == pytest.approx(10.0)


def test_missing_store_skips_the_blend_loudly(forecaster, capsys):
    out = forecaster._fetch_prior_forecasts(np.array(["a", "b"]), 3)

    assert not out["mask"].any()
    assert "::error title=Prior-day blend::" in capsys.readouterr().out


def test_read_twice_counts_rows_and_sums_price_mid(forecaster, session, caplog):
    for r in (_row(1, 1, dt.date(2026, 9, 24)), _row(2, 2, dt.date(2026, 9, 23), mid=9.0)):
        session.add(ItemForecast(**r))
    session.commit()
    mirror_session_to_ops(session)

    with caplog.at_level(logging.INFO, logger=ops_store.logger.name):
        forecaster._fetch_prior_forecasts(np.array(["a", "b"]), 3)

    assert "read twice prior-day forecasts h=3: 2 keys, 0 differ" in caplog.text
