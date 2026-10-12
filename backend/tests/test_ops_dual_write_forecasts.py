"""Price Forecast writes item_forecasts partitions with Postgres's ids (Phase 1.2)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import duckdb
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _session():
    from database import Base, Item

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add_all(
        [Item(id=1, item_id="ak_1", name="AK 1", type="rifle"), Item(id=2, item_id="ak_2", name="AK 2", type="rifle")]
    )
    db.commit()
    return db


def _results(day, slugs=("ak_1",), mid=10.0):
    return pd.DataFrame(
        [
            {
                "item_id": s,
                "current_price": mid,
                "anchor_clean": True,
                "anchor_wedge_pct": 0.0,
                "anchor_date": day,
                "forecasts": {3: {"low": mid - 1, "mid": mid, "high": mid + 1, "direction": "flat", "confidence": 0.5}},
            }
            for s in slugs
        ]
    )


def _partition(tmp_path, day):
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{tmp_path}/item_forecasts/{day:%Y-%m}.parquet') ORDER BY item_id"
    ).df()


def test_partitions_match_postgres_ids(tmp_path, monkeypatch):
    from database import ItemForecast
    from scripts.forecast_prices import _write_forecasts_to_db

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    _write_forecasts_to_db(db, _results(day), "lgbm-v3", {"ak_1": 1}, day, model_config="global-only")
    pg = db.query(ItemForecast).one()
    got = _partition(tmp_path, day)
    assert got["id"].tolist() == [pg.id]
    assert got["item_slug"].tolist() == ["ak_1"]
    assert got["model_config"].tolist() == ["global-only"]


def test_a_second_write_on_the_same_night_replaces_and_keeps_the_other_batch(tmp_path, monkeypatch):
    from scripts.forecast_prices import _write_forecasts_to_db

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    _write_forecasts_to_db(db, _results(day, ("ak_1",)), "v", {"ak_1": 1, "ak_2": 2}, day, model_config="regime")
    _write_forecasts_to_db(db, _results(day, ("ak_2",)), "v", {"ak_1": 1, "ak_2": 2}, day, model_config="global-only")
    first = _partition(tmp_path, day)
    _write_forecasts_to_db(
        db, _results(day, ("ak_2",), mid=12.0), "v", {"ak_1": 1, "ak_2": 2}, day, model_config="global-only"
    )
    again = _partition(tmp_path, day)
    assert len(again) == 2
    assert again["id"].tolist() == first["id"].tolist()
    assert again["model_config"].tolist() == ["regime", "global-only"]
    assert again.loc[again.item_id == 2, "price_mid"].item() == 12.0


def test_a_read_only_night_issues_ids_from_the_partitions(tmp_path, monkeypatch):
    from scripts.forecast_prices import _write_forecasts_to_db

    from tests.test_read_only_postgres import _DB, _read_only_exc

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    day = date.today()
    db = _session()
    _write_forecasts_to_db(db, _results(day, ("ak_1",)), "v", {"ak_1": 1}, day)  # writable: id from Postgres
    stored = _partition(tmp_path, day)["id"].max()

    class _Inspector:
        def get_columns(self, name):
            return [
                {"name": c} for c in ("anchor_clean", "anchor_wedge_pct", "exceed_p", "anomaly_p", "band_multiplier")
            ]

    ro = _DB(_read_only_exc(), [])
    with (
        patch("sqlalchemy.inspect", return_value=_Inspector()),
        patch("sqlalchemy.dialects.postgresql.insert") as ins,
        patch("database.SessionLocal", return_value=ro),
    ):
        ins.return_value.values.return_value.on_conflict_do_update.return_value = "stmt"
        _write_forecasts_to_db(ro, _results(day, ("ak_2",)), "v", {"ak_2": 2}, day)
    got = _partition(tmp_path, day)
    assert got.loc[got.item_id == 2, "id"].item() == stored + 1


def test_a_partition_failure_does_not_fail_the_write(tmp_path, monkeypatch, capsys):
    from scripts.forecast_prices import _write_forecasts_to_db

    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    monkeypatch.setattr("db.ops_partitions.upsert", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    day = date.today()
    assert _write_forecasts_to_db(_session(), _results(day), "v", {"ak_1": 1}, day) == 1
    assert "::error title=Ops partition write failed::item_forecasts" in capsys.readouterr().out
