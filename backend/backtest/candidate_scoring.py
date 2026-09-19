"""Paired daily metrics with equal weight per forecast date.

The independent statistical unit is `forecast_date`, not the item row:
items sharing a date share one market-wide move. Every interval here
resamples whole dates, so a 10,000-row date counts exactly as much as a
100-row one. Never the row bootstrap, since one date's items share the
market factor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 1000


@dataclass(frozen=True)
class PairedDailyInterval:
    point: float
    lower: float | None
    upper: float | None
    n_dates: int


def paired_daily_interval(
    a,
    b,
    dates,
    *,
    ci: int = 90,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> PairedDailyInterval:
    """Mean of (a - b) with equal weight per date, plus a date-block interval.

    `a` and `b` are row-aligned values (e.g. per-row absolute errors for two
    centres); rows are paired within each date by averaging their difference.
    Returns (None, None) bounds for fewer than 2 dates: a single date carries
    no information about between-date variation.
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    d_list = list(dates)
    if not (len(a_arr) == len(b_arr) == len(d_list)) or len(d_list) == 0:
        raise ValueError("a, b and dates must be non-empty and aligned")
    by_date: dict = defaultdict(lambda: [0.0, 0])
    for aval, bval, d in zip(a_arr, b_arr, d_list):
        if not (np.isfinite(aval) and np.isfinite(bval)):
            continue
        key = str(d)
        by_date[key][0] += aval - bval
        by_date[key][1] += 1
    daily = np.array([total / n for total, n in by_date.values() if n], dtype=float)
    n_dates = len(daily)
    if n_dates == 0:
        return PairedDailyInterval(point=0.0, lower=None, upper=None, n_dates=0)
    point = float(daily.mean())
    if n_dates < 2:
        return PairedDailyInterval(point=point, lower=None, upper=None, n_dates=n_dates)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_dates, size=(n_resamples, n_dates))
    stats = daily[draws].mean(axis=1)
    alpha = (100 - ci) / 2
    return PairedDailyInterval(
        point=point,
        lower=round(float(np.percentile(stats, alpha)), 4),
        upper=round(float(np.percentile(stats, 100 - alpha)), 4),
        n_dates=n_dates,
    )
