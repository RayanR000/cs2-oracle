"""Locally-weighted split conformal prediction bands.

Replaces 24 p10/p90 quantile GBMs (303s of a 462s training budget) whose
empirical coverage was 39-48%. Their top feature was `price_std_60d` in 8 of
12 ensembles, i.e. they were learning "band width ~= recent volatility" — this
module states that relationship instead of fitting it.

The normalization is the point. A single global q_hat in return space would
give a $5,000 knife and a $1 case the same band width; dividing the
nonconformity score by a per-item sigma restores the item-level variation the
quantile models were supplying.

Pure: numpy only, no LightGBM, no I/O, no clock.
"""
from __future__ import annotations

import numpy as np

# Pinned. The old code stated two different targets: the comment at
# forecaster.py:3088 described (1-2*alpha) = 80% for a [p10, p90] base, while
# :3092 set alpha = 0.10 and logged "target coverage=90%". 80% is what the
# [p10, p90] band has always represented to the UI.
NOMINAL_COVERAGE = 0.80
ALPHA = 1.0 - NOMINAL_COVERAGE

# Percentiles of the cross-sectional sigma distribution used as clip bounds.
SIGMA_FLOOR_PCTL = 1.0
SIGMA_CAP_PCTL = 99.0


def sigma_bounds(sigma_raw) -> tuple[float, float]:
    """Clip bounds from the cross-sectional distribution of raw sigma.

    Computed on the training frame and persisted with the model: q_hat is
    calibrated against clipped sigmas, so serving must clip identically.
    """
    arr = np.asarray(sigma_raw, dtype=float)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        raise ValueError("cannot derive sigma bounds from an empty distribution")
    floor = float(np.percentile(finite, SIGMA_FLOOR_PCTL))
    cap = float(np.percentile(finite, SIGMA_CAP_PCTL))
    if floor <= 0:
        floor = float(finite.min())
    return floor, cap


def sigma_from_columns(price_std_60d, price, floor: float, cap: float,
                       fallback: float | None = None) -> np.ndarray:
    """Per-item volatility scale: the 60-day coefficient of variation.

    `price_std_60d` is in dollars, so it is divided by price to make the scale
    return-space and comparable across price tiers.

    PREDICT_MIN_HISTORY_DAYS = 14 means eligible items can carry a NaN or 0
    std (the rolling uses min_periods=1). Those rows take `fallback`, or the
    clip floor when no fallback is supplied. A NaN reaching forecast_low would
    surface in the UI.
    """
    std = np.asarray(price_std_60d, dtype=float)
    px = np.asarray(price, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = std / px

    default = float(fallback) if fallback is not None else float(floor)
    bad = ~np.isfinite(sigma) | (sigma <= 0) | ~np.isfinite(px) | (px <= 0)
    sigma = np.where(bad, default, sigma)
    return np.clip(sigma, floor, cap)


def calibrate(residuals_pct, sigma, alpha: float = ALPHA) -> float:
    """q_hat: the conformal quantile of normalized absolute residuals.

    `residuals_pct` are y - y_hat in percentage-return space, from
    out-of-fold predictions. Scores are |residual| / sigma, so q_hat is
    dimensionless and multiplies sigma at serve time.

    Uses the finite-sample corrected level ceil((n+1)(1-alpha))/n, which is
    what gives split conformal its distribution-free coverage guarantee.
    """
    res = np.asarray(residuals_pct, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    if res.size == 0:
        raise ValueError("empty calibration set: cannot compute q_hat")

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs(res) / sig
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("empty calibration set: no finite nonconformity scores")

    n = scores.size
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def band(mid_pct, sigma, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric band around the median, in percentage-return space.

    Cannot cross by construction, which is why predict() no longer needs
    _fix_quantile_crossing.
    """
    mid = np.asarray(mid_pct, dtype=float)
    half = float(q_hat) * np.asarray(sigma, dtype=float)
    return mid - half, mid + half
