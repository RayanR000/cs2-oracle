"""The served-coverage factor must be refit in BASE-band units, not served-band units.

From 2026-09-17 the h=3 band was served at q_hat x ~0.53, so each stored band is already the
product of a multiplier. `factors_from_panel` measured r against the STORED half and the
retrain ASSIGNED the result as the new multiplier, so a refit over rows served at 1.0 and
rows served at 0.54 landed between the two (the band re-widened), and a panel served
entirely at the calibrated factor returned ~1.0 (the band snapped back to full width).

Scaling each row's r by the multiplier it was served at puts every row on the base band,
so the refit is a fixed point: a panel served at the correct factor returns that factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from backtest.scoring import HEADLINE_MIN_TIER
from models import served_recalibration as sr
from models.served_recalibration import factors_from_panel, resolve_band_multiplier


def _panel(r_base, mult, *, n_dates=12, start="2026-09-20", mid=100.0, base_half=10.0):
    """Rows whose BASE-band score is `r_base`, each served at multiplier `mult` (so the stored
    half is mult*base_half). Upper-half actuals, symmetric band about `mid`."""
    r_base = np.asarray(r_base, dtype=float)
    mult = np.broadcast_to(np.asarray(mult, dtype=float), r_base.shape)
    n = r_base.size
    dates = pd.to_datetime(start) + pd.to_timedelta(np.arange(n) % n_dates, unit="D")
    half = mult * base_half
    return pd.DataFrame(
        {
            "forecast_date": dates,
            "horizon_days": 3,
            "price_tier": HEADLINE_MIN_TIER,
            "predicted_price_low": mid - half,
            "predicted_price_mid": mid,
            "predicted_price_high": mid + half,
            "actual_price": mid + r_base * base_half,
            "band_multiplier": mult,
        }
    )


R_BASE = np.random.default_rng(0).uniform(0, 1.0, 400) ** 2  # over-covering base band


def _truth():
    return factors_from_panel(_panel(R_BASE, 1.0), [3])[3]


def test_panel_served_at_the_calibrated_factor_returns_that_factor():
    f = _truth()
    assert f < 0.9
    # Served at f, the stored band now covers ~80%. The old estimator returned ~1.0 here.
    assert factors_from_panel(_panel(R_BASE, f), [3])[3] == pytest.approx(f, rel=1e-9)


def test_mixed_multiplier_panel_matches_the_single_multiplier_panel():
    f = _truth()
    mult = np.where(np.arange(R_BASE.size) % 2 == 0, 1.0, 0.54)
    assert factors_from_panel(_panel(R_BASE, mult), [3])[3] == pytest.approx(f, rel=1e-9)


def test_unknown_multiplier_rows_are_dropped_not_read_as_one():
    # Rows served at 0.5 with no record: read as 1.0 they would each look like a 2x miss.
    mult = np.where(np.arange(R_BASE.size) < 40, 0.5, 1.0)
    panel = _panel(R_BASE, mult)
    panel.loc[panel.index < 40, "band_multiplier"] = np.nan
    got = factors_from_panel(panel, [3])[3]
    expected = factors_from_panel(_panel(R_BASE[40:], 1.0, start="2026-09-24"), [3])[3]
    assert got == pytest.approx(expected, rel=1e-9)


def test_panel_without_the_column_is_read_as_unscaled():
    panel = _panel(R_BASE, 1.0).drop(columns="band_multiplier")
    assert factors_from_panel(panel, [3])[3] == pytest.approx(_truth(), rel=1e-9)


@pytest.mark.parametrize(
    "fd, stored, expected",
    [
        ("2026-09-16", None, 1.0),  # before any multiplier served: provably 1.0
        ("2026-09-16", 0.7, 0.7),  # a recorded value always wins
        (sr.FEEDBACK_FIRST_SERVED_DATE, None, None),  # on/after first activation: unknown
        ("2026-10-01", None, None),
        ("2026-10-01", 0.5385, 0.5385),
        ("2026-10-01", float("nan"), None),
        ("2026-10-01", 0.0, None),  # a non-positive multiplier is corrupt, not a band
    ],
)
def test_resolve_band_multiplier(fd, stored, expected):
    got = resolve_band_multiplier(pd.Timestamp(fd), stored)
    if expected is None:
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_first_served_date_is_the_first_live_panel_read():
    # Price Forecast run 35306325544 (2026-09-18 04:17 UTC, a manual predict-only dispatch)
    # served forecast_date 2026-09-17 with the first non-empty live factor {3: 0.5251}.
    assert sr.FEEDBACK_FIRST_SERVED_DATE == "2026-09-17"
