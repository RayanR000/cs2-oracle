"""Shared price estimator for the forecast backtest.

Both legs of ``actual_ret`` — the forecast-date base and the target-date
actual — go through :func:`smoothed_prices`. That is the entire determinism
guarantee: the same function, the same window, the same source on both sides.

Before 2026-08-01 the base leg was ``item_forecasts.current_price`` (a
3-observation median written at serving time) and the actual leg was a raw
single-day voted price read fresh from the archive on every run. Differencing
two different estimators against a 0.5% flat band is what let the same 5,512
forecasts score 61.76% one day and 33.74% the next.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from collectors.pipeline import FALLBACK_MAX_AGE_DAYS

# Mirrors ItemForecaster.predict()'s tail(3) median (forecaster.py:3513).
SMOOTH_WINDOW = 3

# The 3 observations must lie within this many calendar days of each other.
# Derived from collectors.pipeline.FALLBACK_MAX_AGE_DAYS to ensure a single
# staleness convention across the codebase. If an operator overrides
# FALLBACK_MAX_AGE_DAYS via environment variable, both sides (backtest and
# production) will use the same value. Measured cost: ~0.74% of item-days.
MAX_WINDOW_SPAN_DAYS = FALLBACK_MAX_AGE_DAYS


def smoothed_prices(
    voted: pd.DataFrame,
    anchors: set[tuple[str, date]],
    window: int = SMOOTH_WINDOW,
    max_span_days: int = MAX_WINDOW_SPAN_DAYS,
) -> dict[tuple[str, date], float]:
    """Median of the last ``window`` observed prices at or before each anchor.

    ``voted`` must already be voted to one row per item-day, with columns
    ``item_id`` (slug), ``date``, ``price``.

    Anchors that cannot be resolved — no observation at or before the anchor,
    or observations spanning more than ``max_span_days`` — are omitted from the
    result. Callers must treat a missing key as a dropped forecast rather than
    substituting a fallback, which would reintroduce the asymmetry this
    function exists to remove.
    """
    if voted.empty or not anchors:
        return {}

    by_item: dict[str, list[tuple[date, float]]] = {}
    for slug, group in voted.groupby("item_id", sort=False):
        ordered = group.sort_values("date")
        by_item[slug] = list(zip(ordered["date"], ordered["price"]))

    resolved: dict[tuple[str, date], float] = {}
    for slug, anchor in anchors:
        observations = by_item.get(slug)
        if not observations:
            continue

        # Last `window` observations at or before the anchor.
        selected = [(d, p) for d, p in observations if d <= anchor][-window:]
        if not selected:
            continue

        span = (selected[-1][0] - selected[0][0]).days
        if span > max_span_days:
            continue

        prices = sorted(p for _, p in selected)
        mid = len(prices) // 2
        if len(prices) % 2:
            resolved[(slug, anchor)] = float(prices[mid])
        else:
            resolved[(slug, anchor)] = float((prices[mid - 1] + prices[mid]) / 2)

    return resolved
