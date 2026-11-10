"""Phase C: the served exceedance probability `exceed_p` is persisted.

`predict()` emits `exceed_p` on each forecast — P(the h-day UPSIDE move clears
the round-trip cost), one-sided. `_write_forecasts` must carry it to
`item_forecasts` and the Parquet mirror, behind the SAME missing-column guard as
the anchor disclosure (0022/0024): a prod DB predating the column degrades to
writing without it rather than failing the whole daily batch, and the Parquet
mirror — which has no schema to violate — keeps it regardless.

Scope: docs/plans/2026-08-16-exceedance-band-scale-phase2-plan.md.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

_BASE_DB_COLS = {
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


def _capture_rows(results, *, today=date(2026, 8, 15)):
    """Drive `_write_forecasts` and return the rows handed to the single-file mirror."""
    from scripts.forecast_prices import _write_forecasts

    captured = {}
    with (
        patch("db.parquet.append_table", side_effect=lambda name, rows, keys: captured.update(rows=rows)),
        patch("scripts.forecast_prices._write_forecasts_partitions"),  # row construction only
    ):
        _write_forecasts(results, "lgbm-v3", {"ak_1": 1}, today)
    return captured["rows"]


def _result(forecast):
    return pd.DataFrame(
        [
            {
                "item_id": "ak_1",
                "current_price": 10.0,
                "anchor_clean": True,
                "anchor_wedge_pct": 0.0,
                "anchor_date": date(2026, 8, 14),
                "forecasts": {7: forecast},
            }
        ]
    )


_FCAST = {"low": 9.0, "mid": 10.5, "high": 12.0, "direction": "up", "confidence": "low", "exceed_p": 0.42}


def test_exceed_p_is_written():
    rows = _capture_rows(_result(dict(_FCAST)))
    assert rows[0]["exceed_p"] == 0.42


def test_absent_exceed_p_is_written_as_none():
    """A pre-Phase-A artifact emits no `exceed_p` key -> NULL, not a crash."""
    fcast = dict(_FCAST)
    del fcast["exceed_p"]
    rows = _capture_rows(_result(fcast))
    assert rows[0]["exceed_p"] is None
