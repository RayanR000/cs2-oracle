"""The stored `forecast_date` is the day the band was actually anchored on.

`predict` anchors the band on the newest day in the loaded price frame
(`df["date"].max()`), but the row was stamped `forecast_date = date.today()`.
Those differ whenever the archive lags the calendar (the dump lands ~22:00 UTC,
so the newest archived day is usually today-1), which decouples the label from
the price the band was quoted against. The scorer then resolves `base_price`
AND `target_date` at the stored `forecast_date` -- so a mislabelled anchor day
shifts both legs of the outcome by that gap.

This pins the label to the anchor day. An explicit `FORECAST_DATE_OVERRIDE`
still wins, because a historical replay deliberately stamps a chosen date.

See docs/research/2026-08-15-directional-accuracy-and-data-inventory.md and
docs/changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md (the ops/data-
freshness thread).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

# Reuse the real-booster fixture and the predict driver from the anchor-gate
# suite: the same two-feature forecaster is all this plumbing needs to travel
# through.
from tests.test_clean_anchor_gate import (
    _predict,
)
from tests.test_clean_anchor_gate import (
    forecaster_with_models as forecaster_with_models,
)

# ------------------------------------------------------- what predict computes


def test_predict_exposes_the_anchor_day(forecaster_with_models):
    """`predict` carries the anchor day -- the newest day in the frame it was
    built from -- so the writer can stamp the label on it instead of the wall
    clock. It is a property of the frame, so it is the same for every row."""
    f, price_df = forecaster_with_models
    expected = price_df["date"].max()

    result = _predict(f, price_df)

    assert "anchor_date" in result.columns, "predict does not expose the anchor day; the writer cannot recover it"
    assert set(result["anchor_date"]) == {expected}


# --------------------------------------------------------------- the write path


def _write_and_capture_forecast_date(results, *, today, override=None):
    """Drive `_write_forecasts_to_db` and return the `forecast_date` it stamped
    onto the first row handed to the Parquet mirror."""
    from scripts.forecast_prices import _write_forecasts_to_db

    captured = {}
    db_columns = {
        "item_id",
        "forecast_date",
        "horizon_days",
        "price_low",
        "price_mid",
        "price_high",
        "current_price",
        "direction",
        "confidence",
        "model_version",
        "created_at",
        "anchor_clean",
        "anchor_wedge_pct",
    }

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
        patch("db.parquet.append_table", side_effect=lambda name, rows, keys: captured.update(rows=rows)),
        patch("sqlalchemy.inspect", return_value=_Inspector()),
        patch("sqlalchemy.dialects.postgresql.insert") as ins,
    ):
        (ins.return_value.values.return_value.on_conflict_do_update.return_value) = "stmt"
        _write_forecasts_to_db(_DB(), results, "lgbm-v3", {"ak_1": 1}, today, forecast_date_override=override)
    return captured["rows"][0]["forecast_date"]


def _one_result(anchor_date):
    return pd.DataFrame(
        [
            {
                "item_id": "ak_1",
                "current_price": 10.0,
                "anchor_clean": True,
                "anchor_wedge_pct": 0.0,
                "anchor_date": anchor_date,
                "forecasts": {7: {"low": 9.0, "mid": 10.5, "high": 12.0, "direction": "up", "confidence": "low"}},
            }
        ]
    )


def test_writer_stamps_the_anchor_day_not_today():
    """The label is the day the band was anchored on, even though the run
    happens the next day."""
    anchor = date(2026, 8, 14)
    run_day = date(2026, 8, 15)

    stamped = _write_and_capture_forecast_date(_one_result(anchor), today=run_day)

    assert stamped == anchor


def test_an_explicit_override_still_wins():
    """A historical replay stamps a chosen date; the anchor day must not
    override the operator's explicit intent."""
    anchor = date(2026, 8, 14)
    override = date(2026, 1, 1)

    stamped = _write_and_capture_forecast_date(_one_result(anchor), today=override, override=override)

    assert stamped == override


def test_writer_falls_back_to_today_when_anchor_day_absent():
    """Legacy result frames carry no anchor day; the run day is the safe
    fallback rather than a crash or a NULL label."""
    results = _one_result(anchor_date=None).drop(columns=["anchor_date"])
    run_day = date(2026, 8, 15)

    stamped = _write_and_capture_forecast_date(results, today=run_day)

    assert stamped == run_day
