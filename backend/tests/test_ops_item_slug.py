"""The ops Parquet mirror carries `item_slug`, so the archive self-joins.

`price-archive/*.parquet` keys on `item_slug` (the market_hash_name);
`price-archive/ops/*.parquet` keyed only on `item_id`, the Postgres surrogate.
With no bridge between them, "show me this forecast next to its price history"
could not be answered from the archive at all — it needed a round-trip to
production Supabase, which is the network hop the Parquet store exists to
avoid. These pin the denormalised column onto every writer of the mirror.
"""

import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Base, ForecastOutcome, Item
from scripts.backtest_accuracy import (
    _id_to_slug,
    _pg_read,
    _store_forecast_outcomes,
    _with_item_slug,
)

# Since Phase 2 the backtest reads the partitions; mirror each test's SQLite rows into them.
pytestmark = pytest.mark.usefixtures("ops_mirror")


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(Item(id=1, item_id="AK-47 | Redline (Field-Tested)", name="ak", type="skin"))
    db.add(Item(id=2, item_id="AWP | Asiimov (Field-Tested)", name="awp", type="skin"))
    db.commit()
    yield db
    engine.dispose()


@pytest.fixture
def ops(tmp_path, monkeypatch):
    """Redirect the ops store into tmp_path. The real price-archive/ops/ is
    production data and no test may write it."""
    import db.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod, "OPS_DIR", tmp_path / "ops")
    return tmp_path / "ops"


def _outcome(forecast_id, item_id):
    return {
        "forecast_id": forecast_id,
        "item_id": item_id,
        "forecast_date": date(2026, 7, 1),
        "horizon_days": 3,
        "target_date": date(2026, 7, 4),
        "current_price": 3.0,
        "base_price": 3.0,
        "predicted_price_low": 2.0,
        "predicted_price_mid": 3.0,
        "predicted_price_high": 4.0,
        "actual_price": 3.01,
        "direction_predicted": "flat",
        "direction_actual": "flat",
        "direction_correct": 1,
        "in_interval": 1,
        "abs_error": 0.01,
        "pct_error": 0.33,
        "model_version": "lgbm-test",
    }


# ── the mapping ──────────────────────────────────────────────────────────────


def test_id_to_slug_reads_the_items_table(session):
    assert _id_to_slug(_pg_read(session)) == {
        1: "AK-47 | Redline (Field-Tested)",
        2: "AWP | Asiimov (Field-Tested)",
    }


def test_with_item_slug_adds_the_column_without_touching_the_rest(session):
    row = _outcome(1, 1)
    (stamped,) = _with_item_slug([row], _id_to_slug(_pg_read(session)))
    assert stamped["item_slug"] == "AK-47 | Redline (Field-Tested)"
    assert {k: v for k, v in stamped.items() if k != "item_slug"} == row


def test_with_item_slug_leaves_the_input_rows_alone(session):
    """`to_write` goes to bulk_insert_mappings, which errors on an unmapped
    key. The projection must not mutate it in place."""
    row = _outcome(1, 1)
    _with_item_slug([row], _id_to_slug(_pg_read(session)))
    assert "item_slug" not in row


def test_an_unmapped_item_id_gets_a_null_slug_not_a_crash(session):
    (stamped,) = _with_item_slug([_outcome(1, 999)], _id_to_slug(_pg_read(session)))
    assert stamped["item_slug"] is None


# ── the write path ───────────────────────────────────────────────────────────


def test_stored_outcomes_land_in_parquet_with_a_slug(session, ops):
    assert _store_forecast_outcomes(session, [_outcome(1, 1), _outcome(2, 2)]) == 2

    import db.parquet as parquet_mod

    stored = parquet_mod.read_table("forecast_outcomes")
    assert dict(zip(stored["forecast_id"], stored["item_slug"])) == {
        1: "AK-47 | Redline (Field-Tested)",
        2: "AWP | Asiimov (Field-Tested)",
    }


def test_the_db_row_does_not_gain_the_column(session, ops):
    """Denormalised onto the mirror only — the DB has `items` to join against,
    and a second copy there would be another thing to keep true."""
    _store_forecast_outcomes(session, [_outcome(1, 1)])
    assert not hasattr(session.query(ForecastOutcome).one(), "item_slug")


def test_the_frozen_actuals_are_unaffected(session, ops):
    _store_forecast_outcomes(session, [_outcome(1, 1)])
    row = session.query(ForecastOutcome).one()
    assert (row.base_price, row.actual_price) == (3.0, 3.01)


# ── the point of the exercise ────────────────────────────────────────────────


def test_the_mirror_joins_to_the_price_archive_with_no_database(session, ops, tmp_path):
    """The whole reason for the column: an outcome and its price history in one
    DuckDB query, with `items` nowhere in it."""
    _store_forecast_outcomes(session, [_outcome(1, 1)])

    prices = tmp_path / "prices-2026-07.parquet"
    pd.DataFrame(
        {
            "item_slug": ["AK-47 | Redline (Field-Tested)"] * 2,
            "day": pd.to_datetime(["2026-07-01", "2026-07-04"]),
            "source": ["aggregator_sync"] * 2,
            "mean_price": [3.00, 3.01],
            "volume": [0, 0],
        }
    ).to_parquet(prices, index=False)

    con = duckdb.connect()
    try:
        row = con.sql(f"""
            SELECT o.forecast_id, o.item_slug, p.mean_price
            FROM read_parquet('{ops}/forecast_outcomes.parquet') o
            JOIN read_parquet('{prices}') p
              ON p.item_slug = o.item_slug
             AND CAST(p.day AS DATE) = CAST(o.target_date AS DATE)
        """).fetchall()
    finally:
        con.close()

    assert row == [(1, "AK-47 | Redline (Field-Tested)", 3.01)]


# ── the regression the mirror is exposed to ──────────────────────────────────


def test_a_second_write_does_not_blank_the_slug(session, ops):
    """`append_table` dedups on forecast_id and REPLACES the whole row, so a
    writer that omitted item_slug would null it out on everything it touched.
    Every mirror writer has to carry the column."""
    import db.parquet as parquet_mod

    _store_forecast_outcomes(session, [_outcome(1, 1)])
    _store_forecast_outcomes(session, [_outcome(2, 2)])

    stored = parquet_mod.read_table("forecast_outcomes")
    assert stored["item_slug"].notna().all()
    assert len(stored) == 2


def test_the_mirror_projection_carries_every_stored_column():
    """`_outcome_to_mapping` must project the WHOLE row, not a chosen subset.

    `append_table` dedups on `forecast_id` and REPLACES the row, so a column
    the projection forgets is blanked on every row a verdict refresh touches —
    silently, on data that only `--reresolve` could rebuild. That is the bug
    `item_slug` already cost one fix for, and `base_stale_run_days` (added
    2026-08-08) is the second column to land in this shape.

    Asserting against the table's own columns rather than a hardcoded list is
    the point: the next column added to `forecast_outcomes` fails here until it
    is projected, instead of failing in production as a column of NULLs.
    """
    from scripts.backtest_accuracy import _outcome_to_mapping

    stored = {c.name for c in ForecastOutcome.__table__.columns}
    # Three deliberate omissions, named rather than computed so a NEW column
    # cannot quietly join them:
    #   id            — the surrogate key; the mirror keys on forecast_id.
    #   evaluated_at  — stamped by _refresh_verdict_columns with the run clock.
    #   resolved_at   — stamped by the caller from the row in hand.
    # The last two are supplied immediately after the projection, so they are
    # caller-owned rather than dropped.
    caller_supplied = {"evaluated_at", "resolved_at"}
    expected = stored - {"id"} - caller_supplied

    row = ForecastOutcome(
        id=1,
        forecast_id=1,
        item_id=1,
        forecast_date=date(2026, 7, 1),
        horizon_days=7,
        target_date=date(2026, 7, 8),
        current_price=10.0,
        predicted_price_mid=11.0,
        actual_price=10.5,
        base_price=10.0,
        base_stale_run_days=3,
        direction_predicted="up",
    )
    projected = set(_outcome_to_mapping(row))

    assert not expected - projected, (
        f"_outcome_to_mapping drops stored column(s): {sorted(expected - projected)}. "
        f"append_table replaces the whole row, so these would be blanked on every "
        f"refreshed row."
    )
    assert _outcome_to_mapping(row)["base_stale_run_days"] == 3
