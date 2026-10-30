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
        "prediction_type": "forecast",
        "evaluation_date": DAY,
        "horizon_days": 3,
        "model_version": "lgbm-test",
        "price_tier": -1,
        "evaluation_window_days": None,
        "sample_count": 3,
        "metrics": {},
        "created_at": None,
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
