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
                id=100 + k,
                item_id=k,
                forecast_date=DAY,
                horizon_days=3,
                price_low=2.5,
                price_mid=3.2,
                price_high=3.6,
                current_price=3.0,
                direction="up",
                confidence="high",
                model_version="lgbm-test",
            )
        )
        session.add(
            ForecastOutcome(
                id=200 + k,
                forecast_id=100 + k,
                item_id=k,
                forecast_date=DAY,
                horizon_days=3,
                target_date=DAY + timedelta(days=3),
                current_price=3.0,
                base_price=3.0,
                predicted_price_low=2.5,
                predicted_price_mid=3.2,
                predicted_price_high=3.6,
                actual_price=3.3,
                direction_predicted="up",
                direction_actual="up",
                direction_correct=1,
                in_interval=in_interval,
                abs_error=0.1,
                pct_error=3.0,
                model_version="lgbm-test",
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
    ops_partitions.upsert(
        "forecast_outcomes",
        pd.DataFrame([{"forecast_id": 101, "item_id": 1, "forecast_date": DAY, "horizon_days": 3, "pg_deleted": True}]),
    )
    assert _panel_n(ba._records_from_frozen_outcomes(ba._panel_read())) == 0


def test_both_stores_score_identically_on_the_same_rows(session):
    from scripts import backtest_accuracy as ba

    _seed(session)
    mirror_session_to_ops(session)
    pg = ba._score_groups(ba._records_from_frozen_outcomes(ba._pg_read(session)), DAY)
    pq = ba._score_groups(ba._records_from_frozen_outcomes(ba._panel_read()), DAY)

    def key(r):
        m = r["metrics"]
        return (
            r["horizon_days"],
            r["model_version"],
            r["price_tier"],
            r["sample_count"],
            m.get("directional_accuracy"),
            m.get("interval_coverage"),
        )

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

    ops_partitions.upsert(
        "forecast_outcomes", pd.DataFrame([{"forecast_id": 101, "item_id": 1, "forecast_date": DAY, "horizon_days": 3}])
    )
    ops_partitions.upsert(
        "item_forecasts", pd.DataFrame([{"id": 101, "item_id": 1, "forecast_date": DAY, "horizon_days": 3}])
    )
    with pytest.raises(MissingTable, match=r"forecast_outcomes\.id"):
        ba._records_from_frozen_outcomes(ba._panel_read())
