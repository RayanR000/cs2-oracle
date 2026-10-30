"""Phase 2 PR 6: the accuracy routes read the ops store only, with no Postgres fallback.

`prediction_accuracy` is read from `ops/prediction_accuracy.parquet`, which the backtest writes
before Postgres. Outcomes are read from the monthly `forecast_outcomes` partitions, not the old
single-file mirror (60,737 rows short of the partitions on 2026-10-30).
"""

from __future__ import annotations

import inspect
from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from api.routes import accuracy as route
from db import ops_partitions, ops_store

DAY = date.today() - timedelta(days=10)


def _outcome(oid, horizon, correct, evaluated_at, **extra):
    return {
        "id": oid,
        "forecast_id": 100 + oid,
        "item_id": oid,
        "forecast_date": DAY,
        "horizon_days": horizon,
        "target_date": DAY + timedelta(days=horizon),
        "current_price": 10.0,
        "base_price": 10.0,
        "predicted_price_mid": 11.0,
        "actual_price": 12.0 if correct else 9.0,
        "direction_predicted": "up",
        "direction_actual": "up" if correct else "down",
        "direction_correct": 1 if correct else 0,
        "in_interval": 1,
        "abs_error": 1.0,
        "pct_error": 10.0,
        "model_version": "v1",
        "evaluated_at": evaluated_at,
        **extra,
    }


@pytest.fixture()
def outcomes():
    rows = [
        _outcome(1, 3, True, datetime(2026, 10, 1, 4)),
        _outcome(2, 3, False, datetime(2026, 10, 2, 4)),
        _outcome(3, 7, True, datetime(2026, 10, 3, 4)),
    ]
    ops_partitions.upsert("forecast_outcomes", pd.DataFrame(rows))
    return rows


@pytest.mark.parametrize(
    "fn",
    [
        route.list_accuracy,
        route.get_latest_accuracy,
        route.get_headline,
        route.get_accuracy_summary,
        route.list_outcomes,
        route.outcome_stats,
    ],
)
def test_no_accuracy_route_takes_a_database(fn):
    assert "db" not in inspect.signature(fn).parameters


def test_outcomes_come_from_the_partitions_newest_first(outcomes):
    rows = route.list_outcomes(item_id=None, horizon_days=3, correct=None, limit=50)
    assert [r["id"] for r in rows] == [2, 1]
    assert rows[0]["direction_correct"] is False


def test_a_postgres_deleted_outcome_is_not_served(outcomes):
    ops_partitions.upsert(
        "forecast_outcomes",
        pd.DataFrame([{**_outcome(1, 3, True, datetime(2026, 10, 1, 4)), "pg_deleted": True}]),
    )
    assert [r["id"] for r in route.list_outcomes(item_id=None, horizon_days=3, correct=None, limit=50)] == [2]


def test_outcome_stats_come_from_the_partitions(outcomes):
    stats = route.outcome_stats()
    assert stats["total_outcomes"] == 3
    assert stats["overall_accuracy"] == pytest.approx(66.67)
    assert [(h["horizon_days"], h["total"], h["correct"]) for h in stats["per_horizon"]] == [(3, 2, 1), (7, 1, 1)]


def test_missing_partitions_fail_loudly_instead_of_reading_postgres():
    with pytest.raises(ops_store.MissingTable):
        route.list_outcomes(item_id=None, horizon_days=None, correct=None, limit=50)
    with pytest.raises(ops_store.MissingTable):
        route.outcome_stats()


def test_accuracy_rows_come_from_the_parquet_store():
    from db.parquet import append_table

    append_table(
        "prediction_accuracy",
        [
            {
                "id": 1,
                "prediction_type": "forecast",
                "evaluation_date": date(2026, 10, 29),
                "horizon_days": 3,
                "model_version": "v1",
                "price_tier": -1,
                "evaluation_window_days": None,
                "sample_count": 100,
                "metrics": '{"pt_verdict": "no_skill"}',
                "created_at": datetime(2026, 10, 29, 5),
            }
        ],
        ["prediction_type", "evaluation_date", "horizon_days", "model_version", "price_tier"],
    )
    rows = route.list_accuracy(prediction_type="forecast", limit=50, price_tier=-1)
    assert [(r["horizon_days"], r["metrics"]["pt_verdict"]) for r in rows] == [(3, "no_skill")]


def test_a_missing_accuracy_store_fails_loudly():
    with pytest.raises(FileNotFoundError):
        route.list_accuracy(prediction_type=None, limit=50, price_tier=None)
