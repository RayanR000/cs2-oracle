"""Each served row records the served-coverage multiplier its band was served at.

`predict()` scales q_hat by `served_qhat_multiplier(h)` and then blends 15% of the prior
day's band in, so the stored band is `(0.85 * m + 0.15 * m_prior) x` the base band. The
feedback refit needs that product to score a row against the base band
(tests/test_served_recalibration_band_multiplier.py). This file pins the serving half:
the blended value, the prior fetch that feeds it, and the write behind the same
missing-column guard as the other item_forecasts additions.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest
from models.forecaster import ItemForecaster

from tests.test_forecast_exceed_p_persistence import _BASE_DB_COLS, _capture_rows, _result

W = ItemForecaster.FORECAST_BLEND_WEIGHT


def _prior(mask, band_mult):
    return {"mask": np.asarray(mask, dtype=bool), "band_mult": np.asarray(band_mult, dtype=float)}


def test_blended_multiplier_mixes_in_the_prior_rows_multiplier():
    out = ItemForecaster._blended_band_multiplier(0.5385, _prior([True], [1.0]), W)
    assert out[0] == pytest.approx((1 - W) * 0.5385 + W * 1.0)


def test_rows_that_did_not_blend_record_the_bare_multiplier():
    out = ItemForecaster._blended_band_multiplier(0.5385, _prior([False, True], [1.0, 0.5385]), W)
    assert out[0] == pytest.approx(0.5385)
    assert out[1] == pytest.approx(0.5385)


def test_unknown_prior_multiplier_reads_as_the_current_one_so_nan_cannot_chain():
    out = ItemForecaster._blended_band_multiplier(0.6, _prior([True], [np.nan]), W)
    assert out[0] == pytest.approx(0.6)


def test_zero_weight_or_no_prior_is_the_bare_multiplier():
    assert ItemForecaster._blended_band_multiplier(0.7, _prior([True], [1.0]), 0.0)[0] == pytest.approx(0.7)
    assert ItemForecaster._blended_band_multiplier(0.7, {"mask": np.array([True])}, W)[0] == pytest.approx(
        (1 - W) * 0.7 + W * 0.7
    )


def test_blend_recursion_reproduces_the_observed_2026_09_17_transition():
    # fd 09-16 served at 1.0; fd 09-17 at the live factor 0.5251 -> stored band x0.5963.
    # Measured on prod: h=3/h=7 relative-width ratio fell x0.594 that day.
    out = ItemForecaster._blended_band_multiplier(0.5251, _prior([True], [1.0]), W)
    assert out[0] == pytest.approx(0.5963, abs=1e-4)


def _forecaster_with_prior_rows(rows, *, has_col):
    f = ItemForecaster.__new__(ItemForecaster)
    f.db = MagicMock()
    f._band_mult_col = has_col
    f.db.execute.side_effect = [
        MagicMock(fetchall=lambda: rows),
        MagicMock(fetchall=lambda: [MagicMock(id=1, item_id="a"), MagicMock(id=2, item_id="b")]),
    ]
    f._now = lambda: __import__("datetime").datetime(2026, 9, 25)
    return f


def _row(item_id, fd, mult):
    return MagicMock(
        item_id=item_id,
        price_low=9.0,
        price_mid=10.0,
        price_high=11.0,
        current_price=10.0,
        forecast_date=fd,
        band_multiplier=mult,
    )


def test_prior_fetch_resolves_each_items_latest_multiplier():
    rows = [
        _row(1, date(2026, 9, 24), 0.5385),
        _row(1, date(2026, 9, 23), 0.9),  # older, ignored
        _row(2, date(2026, 9, 10), None),  # pre-activation NULL -> 1.0
    ]
    out = _forecaster_with_prior_rows(rows, has_col=True)._fetch_prior_forecasts(np.array(["a", "b"]), 3)
    assert out["band_mult"][0] == pytest.approx(0.5385)
    assert out["band_mult"][1] == pytest.approx(1.0)


def test_prior_fetch_selects_a_null_placeholder_when_the_column_is_missing():
    f = _forecaster_with_prior_rows([_row(1, date(2026, 9, 24), None)], has_col=False)
    out = f._fetch_prior_forecasts(np.array(["a", "b"]), 3)
    sql = str(f.db.execute.call_args_list[0].args[0])
    assert "NULL AS band_multiplier" in sql
    assert out["mask"][0] and np.isnan(out["band_mult"][0])  # post-activation NULL: unknown


_FCAST = {"low": 9.0, "mid": 10.5, "high": 12.0, "direction": "up", "confidence": "low", "band_multiplier": 0.5385}


def test_band_multiplier_is_written_when_the_column_exists():
    rows = _capture_rows(_result(dict(_FCAST)), db_columns=_BASE_DB_COLS | {"band_multiplier"})
    assert rows[0]["band_multiplier"] == pytest.approx(0.5385)


def test_write_without_the_column_still_succeeds_and_mirrors_it():
    rows = _capture_rows(_result(dict(_FCAST)), db_columns=_BASE_DB_COLS)
    assert rows[0]["band_multiplier"] == pytest.approx(0.5385)


# --- backfill of the pre-column window (scripts/backfill_band_multiplier.py) ----------------


def test_backfill_reconstructs_the_blend_chain_from_the_run_history():
    from scripts.backfill_band_multiplier import served_multiplier

    m917 = (1 - W) * 0.5251 + W * 1.0
    m918 = (1 - W) * 0.5385 + W * ((1 - W) * 0.5251 + W * m917)
    m920 = (1 - W) * 0.5385 + W * m918
    assert served_multiplier(date(2026, 9, 17), 3) == pytest.approx(m917, abs=1e-4)
    assert served_multiplier(date(2026, 9, 18), 3) == pytest.approx(m918, abs=1e-4)
    assert served_multiplier(date(2026, 9, 20), 3) == pytest.approx(m920, abs=1e-4)
    assert served_multiplier(date(2026, 9, 25), 3) == pytest.approx(0.5385, abs=1e-4)


@pytest.mark.parametrize("h", [7, 14, 30])
def test_backfill_other_horizons_were_served_unscaled(h):
    from scripts.backfill_band_multiplier import served_multiplier

    assert served_multiplier(date(2026, 9, 20), h) == 1.0


@pytest.mark.parametrize("fd", [date(2026, 9, 16), date(2026, 9, 28)])
def test_backfill_refuses_dates_outside_the_reconstructed_window(fd):
    from scripts.backfill_band_multiplier import served_multiplier

    with pytest.raises(ValueError):
        served_multiplier(fd, 3)
