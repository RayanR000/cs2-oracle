"""Outside baseline: naive band math, the as-of cutoff, and the scoring helpers.

The AutoETS arm needs the optional `baseline` extra and is not exercised here; the
pieces that decide whether the comparison is honest (no look-ahead, the same
smoothing as the scorer, the same panel rows) are.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.outside_baseline import (
    naive_band,
    paired_date_diff,
    rescale_to_coverage,
    smoothed_series,
    trailing_returns,
)

D0 = date(2026, 1, 1)


def _series(prices, start=D0):
    return pd.Series(prices, index=[start + timedelta(days=i) for i in range(len(prices))], dtype=float)


def test_smoothed_series_is_trailing_median_of_three_observations():
    s = smoothed_series(_series([10.0, 30.0, 20.0, 100.0, 40.0]))
    # 1 obs -> itself; 2 -> mean of the pair (even-count median); then median of 3.
    assert s.tolist() == [10.0, 20.0, 20.0, 30.0, 40.0]


def test_smoothed_series_skips_gaps_like_the_scorer():
    # resolve_anchors takes the last three *observations*, not the last three days.
    raw = pd.Series([10.0, 20.0, 30.0], index=[D0, D0 + timedelta(days=5), D0 + timedelta(days=9)])
    assert smoothed_series(raw).iloc[-1] == 20.0


def test_trailing_returns_never_read_past_the_as_of_date():
    prices = [10.0] * 60 + [1000.0] * 10  # a jump after the as-of date
    s = smoothed_series(_series(prices))
    asof = D0 + timedelta(days=59)
    r = trailing_returns(s, h=3, asof=asof, window_days=365)
    assert np.allclose(r, 0.0)
    assert len(r) == 60 - 3


def test_trailing_returns_respect_the_window():
    s = smoothed_series(_series(np.linspace(10, 20, 200)))
    asof = D0 + timedelta(days=199)
    r = trailing_returns(s, h=7, asof=asof, window_days=30)
    # Return end dates in (asof - 30, asof] -> 30 returns.
    assert len(r) == 30


def test_trailing_returns_measure_calendar_days_not_rows():
    # Observations every other day: an h=4 return spans 2 observations.
    idx = [D0 + timedelta(days=2 * i) for i in range(40)]
    s = pd.Series(np.exp(np.arange(40) * 0.01), index=idx)
    r = trailing_returns(s, h=4, asof=idx[-1], window_days=365)
    assert np.allclose(r, 0.02)


def test_naive_band_takes_the_central_quantiles():
    r = np.linspace(-0.5, 0.5, 101)
    lo, hi = naive_band(r, alpha=0.2)
    assert lo == pytest.approx(-0.4)
    assert hi == pytest.approx(0.4)


def test_naive_band_refuses_thin_history():
    assert naive_band(np.zeros(5), alpha=0.2, min_returns=30) is None


def test_rescale_to_coverage_hits_the_target():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    lo, hi = -np.ones_like(y), np.ones_like(y)  # ~68% coverage
    k = rescale_to_coverage(lo, hi, y, target=0.90)
    cov = np.mean((y >= k * lo) & (y <= k * hi))
    assert cov == pytest.approx(0.90, abs=0.005)
    assert k > 1


def test_paired_date_diff_is_served_minus_baseline_per_date():
    df = pd.DataFrame(
        {
            "forecast_date": [D0, D0, D0 + timedelta(days=1), D0 + timedelta(days=1)],
            "served": [1.0, 3.0, 2.0, 2.0],
            "base": [2.0, 2.0, 1.0, 1.0],
        }
    )
    est, lo, hi = paired_date_diff(df, "served", "base", n_boot=200, seed=0)
    # Per-date means: (2 - 2) = 0 and (2 - 1) = 1 -> mean of dates 0.5.
    assert est == pytest.approx(0.5)
    assert lo <= est <= hi
