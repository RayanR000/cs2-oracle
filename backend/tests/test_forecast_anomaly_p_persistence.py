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


def test_anomaly_p_is_written():
    rows = _capture_rows(_result(dict(_FCAST)))
    assert rows[0]["anomaly_p"] == 0.31


def test_absent_anomaly_p_is_written_as_none():
    """An artifact with no anomaly head (or the 30d horizon) -> NULL, not a crash."""
    fcast = dict(_FCAST)
    del fcast["anomaly_p"]
    rows = _capture_rows(_result(fcast))
    assert rows[0]["anomaly_p"] is None
