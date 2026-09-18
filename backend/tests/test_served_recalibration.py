"""Served-outcome feedback calibration: the per-horizon q_hat multiplier.

Split conformal re-run on the served forecast_outcomes panel — the factor that would have
made the served (asymmetric, mid-anchored) band cover at the 80% nominal. Gated on
MIN_FORECAST_DATES distinct served dates per horizon, clamped to [0.5, 2.0], no-op (absent)
below the gate. See docs/superpowers/specs/2026-08-16-served-outcome-feedback-calibration-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtest.scoring import HEADLINE_MIN_TIER, MIN_FORECAST_DATES
from models.served_recalibration import (
    FACTOR_MAX,
    FACTOR_MIN,
    factors_from_panel,
)


def _panel(horizon, r_values, *, n_dates, tier=HEADLINE_MIN_TIER, mid=100.0, H=10.0):
    """Symmetric bands about `mid` (half-width H), all-upper-half actuals placed so that
    r_i = (actual-mid)/(high-mid) equals the requested value exactly; dates cycled to hit
    `n_dates` distinct forecast_date values."""
    r = np.asarray(r_values, dtype=float)
    n = len(r)
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) % n_dates, unit="D")
    return pd.DataFrame(
        {
            "forecast_date": dates,
            "horizon_days": horizon,
            "price_tier": tier,
            "predicted_price_low": mid - H,
            "predicted_price_mid": mid,
            "predicted_price_high": mid + H,
            "actual_price": mid + r * H,
        }
    )


def test_factor_recovers_the_multiplier_that_lands_80pct(_stub=None):
    rng = np.random.default_rng(0)
    # 92% of rows fall inside the band (r<1): the panel over-covers, so the factor must be < 1.
    r = np.concatenate([rng.uniform(0, 1, 92), rng.uniform(1, 2, 8)])
    panel = _panel(7, r, n_dates=25)
    f = factors_from_panel(panel, [7])
    assert 7 in f
    assert f[7] < 1.0  # narrow an over-covering band
    assert np.mean(r <= f[7]) >= 0.80  # applying it lands >= 80% on the panel


def test_under_covering_panel_widens(_stub=None):
    r = np.concatenate([np.full(60, 0.5), np.full(40, 1.5)])  # only 60% inside -> widen
    f = factors_from_panel(_panel(7, r, n_dates=25), [7])
    assert f[7] > 1.0
    assert np.mean(r <= f[7]) >= 0.80


def test_gate_needs_min_forecast_dates(_stub=None):
    r = np.random.default_rng(1).uniform(0, 1.2, 200)
    below = _panel(7, r, n_dates=MIN_FORECAST_DATES - 1)
    at = _panel(7, r, n_dates=MIN_FORECAST_DATES)
    assert 7 not in factors_from_panel(below, [7])
    assert 7 in factors_from_panel(at, [7])


def test_gate_is_per_horizon(_stub=None):
    r = np.random.default_rng(2).uniform(0, 1.2, 200)
    rich = _panel(3, r, n_dates=25)
    poor = _panel(30, r, n_dates=MIN_FORECAST_DATES - 1)
    f = factors_from_panel(pd.concat([rich, poor], ignore_index=True), [3, 30])
    assert 3 in f and 30 not in f


def test_factor_is_clamped(_stub=None):
    tiny = _panel(7, np.full(200, 0.01), n_dates=25)  # would be ~0.01
    huge = _panel(7, np.full(200, 8.0), n_dates=25)  # would be ~8.0
    assert factors_from_panel(tiny, [7])[7] == FACTOR_MIN
    assert factors_from_panel(huge, [7])[7] == FACTOR_MAX


def test_degenerate_rows_are_dropped(_stub=None):
    r = np.random.default_rng(3).uniform(0, 1.2, 200)
    panel = _panel(7, r, n_dates=25)
    good = factors_from_panel(panel, [7])[7]
    # Add rows with a zero upper half (high == mid) and a NaN actual — must not poison the factor.
    bad = panel.iloc[:5].copy()
    bad["predicted_price_high"] = bad["predicted_price_mid"]
    bad.loc[bad.index[:2], "actual_price"] = np.nan
    mixed = factors_from_panel(pd.concat([panel, bad], ignore_index=True), [7])[7]
    assert mixed == good


def test_sub_dollar_cohort_does_not_move_the_factor(_stub=None):
    r = np.random.default_rng(4).uniform(0, 1.2, 200)
    panel = _panel(7, r, n_dates=25)
    good = factors_from_panel(panel, [7])[7]
    junk = _panel(7, np.full(400, 6.0), n_dates=25, tier=HEADLINE_MIN_TIER - 1)
    withjunk = factors_from_panel(pd.concat([panel, junk], ignore_index=True), [7])[7]
    assert withjunk == good


def test_since_floor_drops_pre_cutover_geometry(_stub=None):
    """Rows before the signed-band cutover carry the old symmetric geometry; the `since` floor
    must drop them so the factor is fit only on current-geometry served bands. Dropping the panel
    below the gate is the visible consequence."""
    r = np.random.default_rng(5).uniform(0, 1.2, 200)
    panel = _panel(7, r, n_dates=25)  # dates run 2026-01-01 .. 2026-01-25
    assert 7 in factors_from_panel(panel, [7])  # no floor: 25 dates clear the gate
    assert 7 in factors_from_panel(panel, [7], since="2026-01-01")  # floor at the first date: all kept
    # A floor past every date leaves zero rows, so the horizon falls below MIN_FORECAST_DATES.
    assert 7 not in factors_from_panel(panel, [7], since="2027-01-01")


def test_since_floor_partitions_two_geometry_epochs(_stub=None):
    """A panel that mixes pre- and post-cutover dates must yield the post-cutover factor alone."""
    old = _panel(7, np.full(300, 6.0), n_dates=20)  # old geometry: would clamp to FACTOR_MAX
    new = _panel(7, np.random.default_rng(6).uniform(0, 1.2, 200), n_dates=25)
    new["forecast_date"] = pd.to_datetime("2026-06-01") + pd.to_timedelta(np.arange(len(new)) % 25, unit="D")
    mixed = pd.concat([old, new], ignore_index=True)
    floored = factors_from_panel(mixed, [7], since="2026-06-01")[7]
    clean = factors_from_panel(new, [7])[7]
    assert floored == clean  # the old epoch does not touch the factor
