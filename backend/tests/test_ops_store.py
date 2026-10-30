"""db/ops_store.py: the partitions read as if they were the Postgres tables (Phase 2)."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest
from db import ops_partitions, ops_store

DAY = date(2026, 9, 10)


def _forecast(fid, item_id=1, day=DAY, **extra):
    return {"id": fid, "item_id": item_id, "forecast_date": day, "horizon_days": 3, "price_mid": 10.0, **extra}


def _write(table, rows, root):
    ops_partitions.upsert(table, pd.DataFrame(rows), allow_closed=True, root=root)


def _all_five(root):
    _write("items", [{"id": 1, "item_id": "ak", "name": "ak", "type": "skin"}], root)
    _write("item_forecasts", [_forecast(10)], root)
    _write(
        "forecast_outcomes",
        [{"id": 20, "forecast_id": 10, "item_id": 1, "forecast_date": DAY, "horizon_days": 3, "actual_price": 11.0}],
        root,
    )
    _write(
        "forecast_candidates",
        [
            {
                "id": 30,
                "item_id": 1,
                "forecast_date": DAY,
                "horizon_days": 3,
                "component": "centre",
                "candidate_name": "naive",
                "candidate_version": "v1",
            }
        ],
        root,
    )
    _write("forecast_candidate_outcomes", [{"candidate_id": 30, "actual_price": 11.0, "_part": DAY}], root)


def test_views_are_named_and_typed_like_the_postgres_tables(tmp_path):
    _all_five(tmp_path)
    con = ops_store.connect(tmp_path)
    try:
        rows = ops_store.fetch(
            con,
            """
            SELECT i.item_id AS slug, f.price_mid, o.actual_price, co.actual_price AS cand_actual
            FROM forecast_outcomes o
            JOIN item_forecasts f ON f.id = o.forecast_id
            JOIN items i ON i.id = o.item_id
            JOIN forecast_candidates c ON c.item_id = f.item_id AND c.forecast_date = f.forecast_date
            JOIN forecast_candidate_outcomes co ON co.candidate_id = c.id
            """,
        )
    finally:
        con.close()
    assert len(rows) == 1
    assert (rows[0].slug, rows[0].price_mid, rows[0].actual_price, rows[0].cand_actual) == ("ak", 10.0, 11.0, 11.0)


def test_rows_have_attribute_access_and_python_types(tmp_path):
    _write("item_forecasts", [_forecast(10)], tmp_path)
    (row,) = ops_store.reader(("item_forecasts",), tmp_path)("SELECT * FROM item_forecasts")
    assert row.id == 10
    assert row.forecast_date == DAY
    assert isinstance(row.horizon_days, int)
    assert not hasattr(row, "pg_deleted")


def test_rows_postgres_deleted_are_hidden(tmp_path):
    _write("item_forecasts", [_forecast(10, pg_deleted=False), _forecast(11, item_id=2, pg_deleted=True)], tmp_path)
    rows = ops_store.reader(("item_forecasts",), tmp_path)("SELECT id FROM item_forecasts")
    assert [r.id for r in rows] == [10]


def test_a_column_an_older_month_lacks_reads_null(tmp_path):
    _write("item_forecasts", [_forecast(10, band_multiplier=1.5)], tmp_path)
    old = tmp_path / "item_forecasts" / "2026-08.parquet"
    duckdb.sql(
        "COPY (SELECT 11::BIGINT AS id, 2::BIGINT AS item_id, DATE '2026-08-10' AS forecast_date, "
        f"3::BIGINT AS horizon_days, 9.0::DOUBLE AS price_mid) TO '{old}' (FORMAT PARQUET)"
    )
    rows = ops_store.reader(("item_forecasts",), tmp_path)("SELECT id, band_multiplier FROM item_forecasts ORDER BY id")
    assert [(r.id, r.band_multiplier) for r in rows] == [(10, 1.5), (11, None)]


def test_a_column_no_file_has_reads_null(tmp_path):
    d = tmp_path / "item_forecasts"
    d.mkdir()
    duckdb.sql(
        "COPY (SELECT 11::BIGINT AS id, 2::BIGINT AS item_id, DATE '2026-08-10' AS forecast_date, "
        f"3::BIGINT AS horizon_days) TO '{d / '2026-08.parquet'}' (FORMAT PARQUET)"
    )
    (row,) = ops_store.reader(("item_forecasts",), tmp_path)("SELECT id, exceed_p FROM item_forecasts")
    assert row.exceed_p is None


def test_a_table_with_no_files_raises_naming_it(tmp_path):
    with pytest.raises(ops_store.MissingTable, match="forecast_outcomes"):
        ops_store.connect(tmp_path, ("forecast_outcomes",))


def test_only_the_named_tables_are_checked(tmp_path):
    _write("items", [{"id": 1, "item_id": "ak", "name": "ak", "type": "skin"}], tmp_path)
    assert ops_store.reader(("items",), tmp_path)("SELECT count(*) AS n FROM items")[0].n == 1


def test_a_reader_sees_files_written_after_it_was_made(tmp_path):
    _write("item_forecasts", [_forecast(10)], tmp_path)
    read = ops_store.reader(("item_forecasts",), tmp_path)
    assert len(read("SELECT id FROM item_forecasts")) == 1
    _write("item_forecasts", [_forecast(11, day=date(2026, 10, 1))], tmp_path)
    assert len(read("SELECT id FROM item_forecasts")) == 2


def test_check_columns_raises_when_a_required_column_is_all_null(tmp_path):
    _write("forecast_outcomes", [{"forecast_id": 10, "item_id": 1, "forecast_date": DAY, "horizon_days": 3}], tmp_path)
    read = ops_store.reader(("forecast_outcomes",), tmp_path)
    with pytest.raises(ops_store.MissingTable, match="forecast_outcomes.id"):
        ops_store.check_columns(read, "forecast_outcomes", ["id", "forecast_id"])
    ops_store.check_columns(read, "forecast_outcomes", ["forecast_id"])


def test_check_columns_accepts_an_empty_table(tmp_path):
    d = tmp_path / "forecast_outcomes"
    d.mkdir()
    duckdb.sql(f"COPY (SELECT 1::BIGINT AS id, 1::BIGINT AS forecast_id WHERE false) TO '{d / '2026-09.parquet'}' (FORMAT PARQUET)")
    ops_store.check_columns(ops_store.reader(("forecast_outcomes",), tmp_path), "forecast_outcomes", ["id"])


def test_equal_values_compare_quietly(capsys):
    assert ops_store.compare("n", lambda: 3, 3) is True
    assert "::warning" not in capsys.readouterr().out


def test_a_difference_is_a_warning_naming_the_reader_and_both_values(capsys):
    assert ops_store.compare("matured forecasts", lambda: 3, 4) is False
    out = capsys.readouterr().out
    assert "::warning title=Read twice: matured forecasts::" in out
    assert "postgres=3" in out and "partitions=4" in out
    assert "Postgres was read-only" in out


def test_dicts_compare_key_by_key_and_name_each_differing_key(capsys):
    assert ops_store.compare("per date", lambda: {"2026-10-01": 5, "2026-10-02": 6}, {"2026-10-01": 5, "2026-10-03": 6}) is False
    out = capsys.readouterr().out
    assert "2026-10-02: postgres=6 partitions=None" in out
    assert "2026-10-03: postgres=None partitions=6" in out
    assert "2026-10-01" not in out


def test_tol_absorbs_float_noise_but_not_none():
    assert ops_store.compare("x", lambda: 1.0, 1.0 + 1e-12, tol=1e-9) is True
    assert ops_store.compare("x", lambda: 1.0, 1.1, tol=1e-9) is False
    assert ops_store.compare("x", lambda: None, 0.0, tol=1.0) is False


def test_a_failed_postgres_read_is_a_warning_and_rolls_back(capsys):
    class _DB:
        rolled_back = False

        def rollback(self):
            self.rolled_back = True

    db = _DB()

    def pg():
        raise RuntimeError("server closed the connection")

    assert ops_store.compare("n", pg, 3, db=db) is False
    assert db.rolled_back
    assert "::warning title=Read twice: n::Postgres read failed (RuntimeError)" in capsys.readouterr().out


def test_more_than_ten_differences_are_truncated(capsys):
    ops_store.compare("d", lambda: {i: 0 for i in range(15)}, {i: 1 for i in range(15)})
    assert "(+5 more)" in capsys.readouterr().out
