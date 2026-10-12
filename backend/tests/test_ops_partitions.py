"""db/ops_partitions.py: the monthly partitions the Phase 1.2 dual-write lands in."""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pandas as pd
import pytest

TODAY = date(2026, 10, 12)


def _fc(item_id, day, h=7, mid=10.0, id_=None, **extra):
    row = {
        "id": id_,
        "item_id": item_id,
        "forecast_date": day,
        "horizon_days": h,
        "price_low": mid - 1,
        "price_mid": mid,
        "price_high": mid + 1,
        "current_price": mid,
        "direction": "up",
        "model_version": "v",
        "created_at": datetime(2026, 10, 12, 2, 0),
    }
    row.update(extra)
    return row


def _read(path):
    return duckdb.sql(f"SELECT * FROM read_parquet('{path}') ORDER BY item_id, horizon_days").df()


def test_rows_route_to_their_month_and_carry_the_full_schema(tmp_path):
    from db.ops_partitions import SPECS, upsert

    frame = pd.DataFrame([_fc(1, date(2026, 10, 31), id_=1), _fc(1, date(2026, 11, 1), id_=2)])
    out = upsert("item_forecasts", frame, root=tmp_path, today=TODAY)
    assert out == {"2026-10": 1, "2026-11": 1}
    oct_ = _read(tmp_path / "item_forecasts" / "2026-10.parquet")
    assert list(oct_.columns) == list(SPECS["item_forecasts"].columns)
    assert oct_["id"].tolist() == [1]
    assert oct_["pg_deleted"].isna().all()


def test_upsert_replaces_by_key_and_keeps_columns_the_write_lacks(tmp_path):
    from db.ops_partitions import upsert

    first = pd.DataFrame(
        [_fc(1, date(2026, 10, 11), id_=5, model_config="global-only"), _fc(2, date(2026, 10, 11), id_=6)]
    )
    upsert("item_forecasts", first, root=tmp_path, today=TODAY)
    again = pd.DataFrame([_fc(1, date(2026, 10, 11), id_=5, mid=12.0)])  # no model_config column
    assert upsert("item_forecasts", again, root=tmp_path, today=TODAY) == {"2026-10": 2}
    got = _read(tmp_path / "item_forecasts" / "2026-10.parquet")
    row = got[got.item_id == 1].iloc[0]
    assert row.price_mid == 12.0
    assert row.model_config == "global-only"
    assert got[got.item_id == 2].iloc[0].id == 6


def test_closed_months_are_refused_unless_allowed(tmp_path):
    from db.ops_partitions import PartitionError, closes_on, upsert

    assert closes_on("2026-08") == date(2026, 10, 13)
    frame = pd.DataFrame([_fc(1, date(2026, 7, 20), id_=1)])
    with pytest.raises(PartitionError, match="closed"):
        upsert("item_forecasts", frame, root=tmp_path, today=TODAY)
    assert not (tmp_path / "item_forecasts" / "2026-07.parquet").exists()
    assert upsert("item_forecasts", frame, root=tmp_path, today=TODAY, allow_closed=True) == {"2026-07": 1}


def test_unknown_columns_null_keys_and_repeated_keys_fail(tmp_path):
    from db.ops_partitions import PartitionError, upsert

    d = date(2026, 10, 11)
    with pytest.raises(PartitionError, match="schema"):
        upsert("item_forecasts", pd.DataFrame([_fc(1, d, surprise=1)]), root=tmp_path, today=TODAY)
    with pytest.raises(PartitionError, match="NULL"):
        upsert("item_forecasts", pd.DataFrame([_fc(None, d)]), root=tmp_path, today=TODAY)
    with pytest.raises(PartitionError, match="repeated"):
        upsert("item_forecasts", pd.DataFrame([_fc(1, d), _fc(1, d)]), root=tmp_path, today=TODAY)


def test_a_failed_write_leaves_the_old_file(tmp_path, monkeypatch):
    from db import ops_partitions
    from db.ops_partitions import upsert

    d = date(2026, 10, 11)
    upsert("item_forecasts", pd.DataFrame([_fc(1, d, id_=1)]), root=tmp_path, today=TODAY)
    path = tmp_path / "item_forecasts" / "2026-10.parquet"
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ops_partitions.os, "replace", boom)
    with pytest.raises(OSError):
        upsert("item_forecasts", pd.DataFrame([_fc(2, d, id_=2)]), root=tmp_path, today=TODAY)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


def test_replace_removes_named_keys_across_months(tmp_path):
    from db.ops_partitions import replace, upsert

    def oc(fid, day, id_):
        return {"id": id_, "forecast_id": fid, "item_id": 1, "forecast_date": day, "horizon_days": 7, "in_interval": 1}

    upsert(
        "forecast_outcomes",
        pd.DataFrame([oc(10, date(2026, 9, 30), 1), oc(11, date(2026, 10, 1), 2)]),
        root=tmp_path,
        today=TODAY,
    )
    out = replace(
        "forecast_outcomes",
        "forecast_id",
        [10, 11],
        pd.DataFrame([oc(11, date(2026, 10, 1), 3)]),
        root=tmp_path,
        today=TODAY,
    )
    assert out == {"2026-09": 0, "2026-10": 1}
    got = _read(tmp_path / "forecast_outcomes" / "2026-10.parquet")
    assert got["id"].tolist() == [3]


def test_candidate_outcomes_route_by_part_and_drop_it(tmp_path):
    from db.ops_partitions import upsert

    frame = pd.DataFrame([{"candidate_id": 7, "actual_price": 1.0, "in_interval": True, "_part": date(2026, 9, 30)}])
    assert upsert("forecast_candidate_outcomes", frame, root=tmp_path, today=TODAY) == {"2026-09": 1}
    got = duckdb.sql(f"SELECT * FROM read_parquet('{tmp_path}/forecast_candidate_outcomes/2026-09.parquet')").df()
    assert "_part" not in got.columns


def test_items_is_one_file(tmp_path):
    from db.ops_partitions import upsert

    frame = pd.DataFrame([{"id": 1, "item_id": "ak_1", "name": "AK 1", "type": "rifle"}])
    assert upsert("items", frame, root=tmp_path, today=TODAY) == {"all": 1}
    assert (tmp_path / "items.parquet").exists()


def test_assign_ids_keeps_stored_ids_and_issues_above_the_max(tmp_path):
    from db.ops_partitions import assign_ids, upsert

    d = date(2026, 10, 11)
    upsert("item_forecasts", pd.DataFrame([_fc(1, d, id_=40), _fc(2, d, id_=41)]), root=tmp_path, today=TODAY)
    got = assign_ids("item_forecasts", pd.DataFrame([_fc(2, d), _fc(3, d), _fc(4, d)]), root=tmp_path)
    assert got["id"].tolist() == [41, 42, 43]


def test_dual_write_logs_and_returns_false(capsys, caplog):
    from db.ops_partitions import dual_write

    def fail():
        raise RuntimeError("boom")

    assert dual_write("item_forecasts", fail) is False
    assert "::error title=Ops partition write failed::" in capsys.readouterr().out
    assert "boom" in caplog.text
    assert dual_write("item_forecasts", lambda: None) is True


def test_tests_never_write_the_real_ops_dir():
    """conftest's autouse fixture: a writer a test forgot to redirect must not land in price-archive/ops."""
    from db import parquet
    from db.ops_partitions import ops_dir

    assert ops_dir() != parquet.ARCHIVE_ROOT / "ops"


def test_dual_write_rolls_back_the_session_it_was_given():
    """A failed read-back leaves a Postgres transaction aborted; the next writer must not inherit it."""
    from db.ops_partitions import dual_write

    class _DB:
        rolled_back = False

        def rollback(self):
            self.rolled_back = True

    db = _DB()

    def fail():
        raise RuntimeError("statement timeout")

    assert dual_write("item_forecasts", fail, db=db) is False
    assert db.rolled_back
