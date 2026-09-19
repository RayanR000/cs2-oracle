"""Standardized interval scoring for comparing ANY interval forecast method.

Pure numpy/scipy. No I/O, no DB. All functions take numpy arrays and return
floats, dicts, or DataFrames.

The interval score (Winkler, 1972) is the proper scoring rule for prediction
intervals: it rewards narrow intervals and penalizes miscoverage. Lower is
better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

# Reuse the project's price_tier function for tier-stratified coverage.
from backtest.scoring import price_tier


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def coverage(low: np.ndarray, high: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of actuals within [low, high]. Boundary-inclusive."""
    return float(np.mean((actual >= low) & (actual <= high)))


def average_width(low: np.ndarray, high: np.ndarray) -> float:
    """Mean interval width."""
    return float(np.mean(high - low))


def relative_width(low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> float:
    """Mean of (high - low) / mid. Use when comparing across price scales."""
    widths = (high - low) / mid
    return float(np.mean(widths))


def interval_score(
    low: np.ndarray,
    high: np.ndarray,
    actual: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """Mean Winkler interval score (proper scoring rule).

    For each observation:
      - actual inside [low, high]:  score = width
      - actual < low:               score = width + (2/alpha) * (low - actual)
      - actual > high:              score = width + (2/alpha) * (actual - high)

    Lower is better. Penalizes both excess width and miscoverage.
    """
    width = high - low
    penalty_low = (2.0 / alpha) * np.maximum(low - actual, 0.0)
    penalty_high = (2.0 / alpha) * np.maximum(actual - high, 0.0)
    scores = width + penalty_low + penalty_high
    return float(np.mean(scores))


def adaptivity(
    low: np.ndarray,
    mid: np.ndarray,
    high: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float]:
    """Spearman correlation between interval width and prediction error.

    Higher means the method widens when it should — a desirable property.
    Returns NaN for fewer than 3 observations (Spearman is undefined).
    """
    width = high - low
    abs_error = np.abs(actual - mid)
    sq_error = (actual - mid) ** 2

    if len(width) < 3:
        return {"width_vs_abs_error": float("nan"), "width_vs_sq_error": float("nan")}

    # Constant arrays produce NaN from spearmanr; that's the correct answer.
    rho_abs, _ = spearmanr(width, abs_error)
    rho_sq, _ = spearmanr(width, sq_error)
    return {
        "width_vs_abs_error": float(rho_abs),
        "width_vs_sq_error": float(rho_sq),
    }


def coverage_by_tier(
    low: np.ndarray,
    high: np.ndarray,
    actual: np.ndarray,
    prices: np.ndarray,
) -> dict[int, float]:
    """Coverage broken down by price tier (see backtest.scoring.price_tier)."""
    tiers = np.array([price_tier(p) for p in prices])
    inside = (actual >= low) & (actual <= high)
    result = {}
    for t in sorted(set(tiers)):
        mask = tiers == t
        result[int(t)] = float(np.mean(inside[mask]))
    return result


def calibration_curve(
    mid: np.ndarray,
    actual: np.ndarray,
    std_estimate: np.ndarray,
    nominal_levels: tuple[float, ...] | None = None,
) -> list[tuple[float, float]]:
    """Empirical coverage at a range of nominal levels.

    Constructs symmetric intervals from mid +/- z * std_estimate at each
    nominal level and measures the empirical coverage.

    Returns list of (nominal, empirical) pairs.
    """
    if nominal_levels is None:
        nominal_levels = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

    results = []
    for level in nominal_levels:
        z = norm.ppf(0.5 + level / 2.0)
        lo = mid - z * std_estimate
        hi = mid + z * std_estimate
        empirical = coverage(lo, hi, actual)
        results.append((level, empirical))
    return results


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------


def score_intervals(
    low: np.ndarray,
    mid: np.ndarray,
    high: np.ndarray,
    actual: np.ndarray,
    alpha: float = 0.20,
    prices: np.ndarray | None = None,
) -> dict:
    """All interval metrics in one dict.

    Parameters
    ----------
    low, mid, high : array-like
        Lower bound, centre, and upper bound of each interval.
    actual : array-like
        Realised values.
    alpha : float
        Nominal miscoverage rate (0.20 for 80% target coverage).
    prices : array-like, optional
        Prices for tier-stratified coverage. If None, coverage_by_tier is
        omitted from the result.
    """
    low = np.asarray(low, dtype=float)
    mid = np.asarray(mid, dtype=float)
    high = np.asarray(high, dtype=float)
    actual = np.asarray(actual, dtype=float)

    result = {
        "coverage": coverage(low, high, actual),
        "average_width": average_width(low, high),
        "relative_width": relative_width(low, mid, high),
        "interval_score": interval_score(low, high, actual, alpha=alpha),
        "adaptivity": adaptivity(low, mid, high, actual),
    }

    if prices is not None:
        prices = np.asarray(prices, dtype=float)
        result["coverage_by_tier"] = coverage_by_tier(low, high, actual, prices)

    return result


def compare_methods(
    results: dict[str, tuple],
    actual: np.ndarray,
    alpha: float = 0.20,
    prices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare multiple interval methods side by side.

    Parameters
    ----------
    results : dict
        {method_name: (low, mid, high)} for each method.
    actual : array-like
        Realised values (shared across methods).
    alpha : float
        Nominal miscoverage rate.
    prices : array-like, optional
        For tier-stratified coverage.

    Returns
    -------
    pd.DataFrame
        One row per method, columns are metric names.
    """
    actual = np.asarray(actual, dtype=float)
    rows = {}
    for name, (lo, mi, hi) in results.items():
        metrics = score_intervals(lo, mi, hi, actual, alpha=alpha, prices=prices)
        # Flatten adaptivity into top-level keys for the DataFrame
        adapt = metrics.pop("adaptivity", {})
        for k, v in adapt.items():
            metrics[f"adaptivity_{k}"] = v
        # Drop non-scalar values (coverage_by_tier dict, calibration_curve list)
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        rows[name] = metrics

    return pd.DataFrame.from_dict(rows, orient="index")
