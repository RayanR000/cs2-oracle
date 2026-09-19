"""The served anomaly probability `anomaly_p` is persisted, and is guarded.

`predict()` emits `anomaly_p` on each forecast at the disclosed horizons
(`ItemForecaster.ANOMALY_SERVED_HORIZONS` = 3/7/14; 30d is None because its head
ranks without calibrating). `_write_forecasts_to_db` has always put the value in
its payload, but `anomaly_p` was absent from the missing-column probe that
narrows the SQL leg — so on a prod schema predating migration 0025 the INSERT
names a column the table lacks and the WHOLE daily batch raises after three
retries, taking the forecast run down for a disclosure field. That is precisely
the failure the guard exists to prevent for `exceed_p` and the anchor columns.

Reuses the harness in test_forecast_exceed_p_persistence.py rather than
restating it.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from tests.test_forecast_exceed_p_persistence import _BASE_DB_COLS, _capture_rows, _result

_FCAST = {
    "low": 9.0,
    "mid": 10.5,
    "high": 12.0,
    "direction": "up",
    "confidence": "low",
    "exceed_p": 0.42,
    "anomaly_p": 0.31,
}

_FULL_COLS = _BASE_DB_COLS | {"exceed_p", "anomaly_p"}


def _capture_db_payload(results, *, db_columns):
    """The rows actually handed to the INSERT, i.e. after the guard narrows."""
    from scripts.forecast_prices import _write_forecasts_to_db

    class _DB:
        def execute(self, stmt, *a, **k):
            return None

        def commit(self):
            return None

        def get_bind(self):
            return type("_Bind", (), {"dialect": type("_D", (), {"name": "postgresql"})()})()

    class _Inspector:
        def get_columns(self, name):
            return [{"name": c} for c in db_columns]

    with (
        patch("db.parquet.append_table", side_effect=lambda *a, **k: None),
        patch("sqlalchemy.inspect", return_value=_Inspector()),
        patch("sqlalchemy.dialects.postgresql.insert") as ins,
    ):
        (ins.return_value.values.return_value.on_conflict_do_update.return_value) = "stmt"
        _write_forecasts_to_db(_DB(), results, "lgbm-v3", {"ak_1": 1}, date(2026, 8, 15))
        return ins.return_value.values.call_args[0][0]


def test_anomaly_p_is_written_when_the_column_exists():
    rows = _capture_rows(_result(dict(_FCAST)), db_columns=_FULL_COLS)
    assert rows[0]["anomaly_p"] == 0.31


def test_db_payload_drops_anomaly_p_when_the_column_is_absent():
    """The guard must narrow the SQL leg — otherwise the batch raises."""
    batch = _capture_db_payload(_result(dict(_FCAST)), db_columns=_BASE_DB_COLS | {"exceed_p"})
    assert "anomaly_p" not in batch[0]
    assert batch[0]["exceed_p"] == 0.42  # only the absent column is dropped


def test_db_payload_keeps_anomaly_p_when_the_column_exists():
    batch = _capture_db_payload(_result(dict(_FCAST)), db_columns=_FULL_COLS)
    assert batch[0]["anomaly_p"] == 0.31


def test_parquet_mirror_keeps_anomaly_p_even_when_db_lacks_the_column():
    """The mirror has no schema to violate, so the disclosure lands either way."""
    rows = _capture_rows(_result(dict(_FCAST)), db_columns=_BASE_DB_COLS | {"exceed_p"})
    assert rows[0]["anomaly_p"] == 0.31


def test_absent_anomaly_p_is_written_as_none():
    """An artifact with no anomaly head (or the 30d horizon) -> NULL, not a crash."""
    fcast = dict(_FCAST)
    del fcast["anomaly_p"]
    rows = _capture_rows(_result(fcast), db_columns=_FULL_COLS)
    assert rows[0]["anomaly_p"] is None
