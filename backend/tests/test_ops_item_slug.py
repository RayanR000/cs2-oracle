"""The ops Parquet mirror carries `item_slug`, so the archive self-joins.

`price-archive/*.parquet` keys on `item_slug` (the market_hash_name);
`price-archive/ops/*.parquet` keyed only on `item_id`, the Postgres surrogate.
With no bridge between them, "show me this forecast next to its price history"
could not be answered from the archive at all — it needed a round-trip to
production Supabase, which is the network hop the Parquet store exists to
avoid. These pin the denormalised column onto every writer of the mirror.
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Base, ForecastOutcome, Item  # noqa: E402
from scripts.backtest_accuracy import (  # noqa: E402
    _id_to_slug,
    _store_forecast_outcomes,
    _with_item_slug,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
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
    assert _id_to_slug(session) == {
        1: "AK-47 | Redline (Field-Tested)",
        2: "AWP | Asiimov (Field-Tested)",
    }


def test_with_item_slug_adds_the_column_without_touching_the_rest(session):
    row = _outcome(1, 1)
    stamped, = _with_item_slug([row], _id_to_slug(session))
    assert stamped["item_slug"] == "AK-47 | Redline (Field-Tested)"
    assert {k: v for k, v in stamped.items() if k != "item_slug"} == row


def test_with_item_slug_leaves_the_input_rows_alone(session):
    """`to_write` goes to bulk_insert_mappings, which errors on an unmapped
    key. The projection must not mutate it in place."""
    row = _outcome(1, 1)
    _with_item_slug([row], _id_to_slug(session))
    assert "item_slug" not in row


def test_an_unmapped_item_id_gets_a_null_slug_not_a_crash(session):
    stamped, = _with_item_slug([_outcome(1, 999)], _id_to_slug(session))
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

def test_the_mirror_joins_to_the_price_archive_with_no_database(
    session, ops, tmp_path
):
    """The whole reason for the column: an outcome and its price history in one
    DuckDB query, with `items` nowhere in it."""
    _store_forecast_outcomes(session, [_outcome(1, 1)])

    prices = tmp_path / "prices-2026-07.parquet"
    pd.DataFrame({
        "item_slug": ["AK-47 | Redline (Field-Tested)"] * 2,
        "day": pd.to_datetime(["2026-07-01", "2026-07-04"]),
        "source": ["aggregator_sync"] * 2,
        "mean_price": [3.00, 3.01],
        "volume": [0, 0],
    }).to_parquet(prices, index=False)

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


# ── the one-off backfill of pre-existing rows ────────────────────────────────

@pytest.fixture
def ops_dir(tmp_path):
    d = tmp_path / "ops"
    d.mkdir()
    return d


MAP = {1: "AK-47 | Redline (Field-Tested)", 2: "AWP | Asiimov (Field-Tested)"}


def _legacy_forecasts(path, item_ids=(1, 2)):
    """An ops file as it was written before the slug column existed."""
    pd.DataFrame({
        "item_id": list(item_ids),
        "forecast_date": pd.to_datetime(["2026-07-01"] * len(item_ids)),
        "horizon_days": [3] * len(item_ids),
        "price_mid": [3.0 + i for i in range(len(item_ids))],
    }).to_parquet(path, index=False)


def _read(path):
    con = duckdb.connect()
    try:
        return con.sql(f"SELECT * FROM read_parquet('{path}')").fetchdf()
    finally:
        con.close()


def test_backfill_adds_the_column(ops_dir):
    from scripts.backfill_ops_item_slug import backfill

    _legacy_forecasts(ops_dir / "item_forecasts.parquet")
    assert backfill(ops_dir.parent, ["item_forecasts"], MAP, apply=True) == 2

    out = _read(ops_dir / "item_forecasts.parquet")
    assert list(out["item_slug"]) == [MAP[1], MAP[2]]


def test_backfill_preserves_the_other_columns_and_their_order(ops_dir):
    from scripts.backfill_ops_item_slug import backfill

    path = ops_dir / "item_forecasts.parquet"
    _legacy_forecasts(path)
    before = _read(path)
    backfill(ops_dir.parent, ["item_forecasts"], MAP, apply=True)
    after = _read(path)

    assert list(after.columns) == list(before.columns) + ["item_slug"]
    pd.testing.assert_frame_equal(after[before.columns], before)


def test_backfill_dry_run_writes_nothing(ops_dir):
    from scripts.backfill_ops_item_slug import backfill

    path = ops_dir / "item_forecasts.parquet"
    _legacy_forecasts(path)
    raw = path.read_bytes()
    assert backfill(ops_dir.parent, ["item_forecasts"], MAP, apply=False) == 2
    assert path.read_bytes() == raw


def test_backfill_is_idempotent(ops_dir):
    from scripts.backfill_ops_item_slug import backfill

    _legacy_forecasts(ops_dir / "item_forecasts.parquet")
    backfill(ops_dir.parent, ["item_forecasts"], MAP, apply=True)
    assert backfill(ops_dir.parent, ["item_forecasts"], MAP, apply=True) == 0


def test_backfill_fills_only_the_null_slugs_on_a_partial_file(ops_dir):
    from scripts.backfill_ops_item_slug import backfill

    path = ops_dir / "item_forecasts.parquet"
    pd.DataFrame({
        "item_id": [1, 2],
        "price_mid": [3.0, 4.0],
        "item_slug": ["already-set", None],
    }).to_parquet(path, index=False)

    assert backfill(ops_dir.parent, ["item_forecasts"], MAP, apply=True) == 1
    assert list(_read(path)["item_slug"]) == ["already-set", MAP[2]]


def test_backfill_leaves_an_unmapped_id_null_and_does_not_drop_the_row(ops_dir):
    from scripts.backfill_ops_item_slug import backfill

    path = ops_dir / "item_forecasts.parquet"
    _legacy_forecasts(path, item_ids=(1, 999))
    assert backfill(ops_dir.parent, ["item_forecasts"], MAP, apply=True) == 1

    out = _read(path)
    assert len(out) == 2
    assert out["item_slug"].isna().sum() == 1


def test_backfill_never_multiplies_rows(ops_dir):
    """The fill is a LEFT JOIN. It cannot drop rows, but a non-unique mapping
    would multiply them — doubling the served outcome table rather than
    failing. The post-write row count is checked before the file is replaced."""
    from scripts.backfill_ops_item_slug import backfill

    path = ops_dir / "item_forecasts.parquet"
    _legacy_forecasts(path, item_ids=(1, 2, 1, 2))
    backfill(ops_dir.parent, ["item_forecasts"], MAP, apply=True)
    assert len(_read(path)) == 4


def test_backfill_skips_a_table_with_no_item_id(ops_dir):
    from scripts.backfill_ops_item_slug import backfill

    pd.DataFrame({"event_id": [1], "note": ["x"]}).to_parquet(
        ops_dir / "events.parquet", index=False)
    assert backfill(ops_dir.parent, ["events"], MAP, apply=True) == 0
    assert "item_slug" not in _read(ops_dir / "events.parquet").columns


def test_backfill_skips_an_absent_table(ops_dir):
    from scripts.backfill_ops_item_slug import backfill
    assert backfill(ops_dir.parent, ["nope"], MAP, apply=True) == 0
