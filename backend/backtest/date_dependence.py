"""Date-overlap sensitivity for per-forecast-date statistics.

Pure: given one value per forecast date, this produces intervals and nothing
else. No I/O, no archive access, no clock.

WHY THIS EXISTS
---------------
Every date-level bootstrap in the repo (``outside_baseline.paired_date_diff``,
``measure_conformal_pid.paired_bootstrap``, ``scoring.block_bootstrap_ci``)
resamples forecast dates as if they were independent. They are not: two h-day
forecasts made k < h days apart score against h - k shared forward days, so a
per-date series is at least MA(h - 1) by construction, on top of any volatility
regime the dates share.

Measured on pre-window dates (``docs/research/2026-10-09-date-overlap-sensitivity.md``):
the h=7 served-minus-naive interval-score series has lag-1 autocorrelation
+0.86 to +0.94, and a dependence-robust 90% interval is 1.34-1.71x as wide as
the iid one. The frozen preregistrations keep their iid intervals; this module
supplies the sensitivity rows reported beside them.

WHAT IS COMPUTED
----------------
- ``moving_block_ci``: circular moving-block bootstrap of the mean (Politis &
  Romano 1992), block length = the horizon, so a block spans one overlap.
- ``hac_t_ci``: mean +- t_{n-1} * sqrt(LRV / n), LRV the Bartlett-kernel
  long-run variance (``directional_test.hac_long_run_variance``) at lag h - 1,
  the overlap's MA order.
- ``calendar_acf``: autocorrelation at a CALENDAR lag. Forecast dates have
  gaps (exclusions, missed days), and a positional lag would pair dates that
  share no forward days.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from scipy import stats

from backtest.directional_test import hac_long_run_variance


def calendar_acf(dates: list[date], values, lag: int) -> tuple[float | None, int]:
    """Autocorrelation between dates exactly ``lag`` calendar days apart.

    Uses the full-series mean and variance, as the standard ACF does. Returns
    ``(None, n_pairs)`` under 4 pairs: too few to mean anything.
    """
    x = np.asarray(values, dtype=float)
    by_date = dict(zip(dates, x))
    pairs = [(by_date[d], by_date[d + timedelta(days=lag)]) for d in dates if d + timedelta(days=lag) in by_date]
    if len(pairs) < 4:
        return None, len(pairs)
    m, var = x.mean(), x.var()
    if var == 0:
        return None, len(pairs)
    a = np.array(pairs)
    return float(np.mean((a[:, 0] - m) * (a[:, 1] - m)) / var), len(pairs)


def iid_ci(values, level: float, n_boot: int, seed: int) -> tuple[float, float]:
    """The repo's existing date bootstrap: resample dates with replacement."""
    x = np.asarray(values, dtype=float)
    boots = x[np.random.default_rng(seed).integers(0, x.size, (n_boot, x.size))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(boots, [tail, 100 - tail])
    return float(lo), float(hi)


def moving_block_ci(values, block: int, level: float, n_boot: int, seed: int) -> tuple[float, float]:
    """Circular moving-block bootstrap of the mean, in date order."""
    x = np.asarray(values, dtype=float)
    n = x.size
    block = max(1, min(block, n))
    n_blocks = -(-n // block)
    starts = np.random.default_rng(seed).integers(0, n, (n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(x[idx].mean(axis=1), [tail, 100 - tail])
    return float(lo), float(hi)


def hac_t_ci(values, lag: int, level: float) -> tuple[float, float]:
    """Mean +- t_{n-1} * Bartlett-HAC standard error."""
    x = np.asarray(values, dtype=float)
    se = float(np.sqrt(hac_long_run_variance(list(x), lag) / x.size))
    t = float(stats.t.ppf(1 - (1 - level) / 2, df=x.size - 1))
    return float(x.mean() - t * se), float(x.mean() + t * se)


def _sign(ci: tuple[float, float]) -> str:
    lo, hi = ci
    return "above_zero" if lo > 0 else "below_zero" if hi < 0 else "spans_zero"


def sensitivity(dates: list[date], values, horizon: int, level: float, n_boot: int, seed: int) -> dict:
    """The iid interval beside its two dependence-robust counterparts.

    ``fragile`` is True when either robust interval puts zero on a different
    side than the iid one -- i.e. when the iid verdict depends on treating
    overlapping dates as independent.
    """
    order = np.argsort(np.array(dates, dtype="datetime64[D]"))
    d = [dates[i] for i in order]
    x = np.asarray(values, dtype=float)[order]
    if x.size < 3:
        raise ValueError(f"need >= 3 dates for a dependence-robust interval, got {x.size}")
    iid = iid_ci(x, level, n_boot, seed)
    mbb = moving_block_ci(x, horizon, level, n_boot, seed)
    hac = hac_t_ci(x, max(horizon - 1, 0), level)
    width = iid[1] - iid[0]
    return {
        "n_dates": int(x.size),
        "mean": float(x.mean()),
        "level": level,
        "acf": {k: calendar_acf(d, x, k)[0] for k in range(1, horizon + 1)},
        "iid": iid,
        "moving_block": mbb,
        "hac": hac,
        "width_vs_iid": {
            "moving_block": (mbb[1] - mbb[0]) / width if width else None,
            "hac": (hac[1] - hac[0]) / width if width else None,
        },
        "sign": {"iid": _sign(iid), "moving_block": _sign(mbb), "hac": _sign(hac)},
        "fragile": _sign(mbb) != _sign(iid) or _sign(hac) != _sign(iid),
    }
