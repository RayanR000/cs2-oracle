"""Phase 4: the backtest writes the partitions only; Postgres is not opened."""

from __future__ import annotations

import inspect
from datetime import date, timedelta

from db import ops_partitions

DAY = date.today() - timedelta(days=10)


def _outcome(forecast_id=10):
    return {
        "forecast_id": forecast_id,
        "item_id": 1,
        "forecast_date": DAY,
        "horizon_days": 3,
        "target_date": DAY + timedelta(days=3),
        "current_price": 3.0,
        "base_price": 3.0,
        "predicted_price_low": 2.5,
        "predicted_price_mid": 3.2,
        "predicted_price_high": 3.6,
        "actual_price": 3.3,
        "direction_predicted": "up",
        "direction_actual": "up",
        "direction_correct": 1,
        "in_interval": 1,
        "abs_error": 0.1,
        "pct_error": 3.0,
        "model_version": "lgbm-test",
    }


def _partition(forecast_id):
    from scripts.backtest_accuracy import _store_read

    return [
        r for r in _store_read("forecast_outcomes")("SELECT * FROM forecast_outcomes") if r.forecast_id == forecast_id
    ]


def test_new_outcomes_land_in_the_partitions_with_high_ids_and_a_slug():
    from scripts import backtest_accuracy as ba

    big = ops_partitions.READ_ONLY_ID_BASE + 5  # a forecast that only ever existed in the partitions
    assert ba._store_forecast_outcomes([_outcome(), _outcome(big)], id_to_slug={1: "ak"}) == 2
    for fid in (10, big):
        (row,) = _partition(fid)
        assert row.id > ops_partitions.READ_ONLY_ID_BASE
        assert row.item_slug == "ak"


def test_a_second_night_issues_fresh_ids_above_the_first():
    from scripts import backtest_accuracy as ba

    ba._store_forecast_outcomes([_outcome(10)], id_to_slug={1: "ak"})
    ba._store_forecast_outcomes([_outcome(11)], id_to_slug={1: "ak"})
    (first,) = _partition(10)
    (second,) = _partition(11)
    assert second.id == first.id + 1


def test_the_verdict_refresh_writes_the_partitions(monkeypatch):
    from scripts import backtest_accuracy as ba

    calls = []
    monkeypatch.setattr(ba, "_refreshed_outcome_partitions", lambda rows: calls.append(len(rows)))
    monkeypatch.setattr("db.parquet.append_table", lambda *a, **k: None)
    ba._flush_verdict_refresh([{"forecast_id": 10}])
    assert calls == [1]


def test_a_swallowed_candidate_failure_still_returns_empty_stats(monkeypatch):
    from scripts import backtest_accuracy as ba

    def boom(*a, **k):
        raise ValueError("candidate insert failed")

    monkeypatch.setattr("backtest.candidate_resolution.resolve_candidate_outcomes", boom)
    assert ba._resolve_candidates(DAY, None) == {}


def test_neither_module_imports_the_database_engine():
    import backtest.candidate_resolution as cr
    from scripts import backtest_accuracy as ba

    for module in (ba, cr):
        source = inspect.getsource(module)
        assert "SessionLocal" not in source and "from database import" not in source, module.__name__
