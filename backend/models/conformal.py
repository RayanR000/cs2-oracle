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


def elasticity(residuals_pct, sigma) -> float:
    """`d log|residual| / d log sigma`, which this module's math assumes is 1.0.

    DIAGNOSTIC. Nothing here reads it; `calibrate` divides by `sigma ** 1`
    unconditionally.

    If the true value is below 1, `sigma` OVER-corrects: a high-sigma item's
    residual grows more slowly than its sigma does, so the score
    `|r| / sigma ∝ sigma ** (elasticity - 1)` FALLS as sigma rises. Low-sigma
    rows are then under-covered and high-sigma rows over-covered, while MARGINAL
    coverage stays exactly on target -- which is why the guarantee this module
    advertises cannot detect it. See `coverage_by_sigma_stratum`.

    OLS on the logs, on rows where both are strictly positive. Returns NaN when
    sigma has no spread to regress on.
    """
    r = np.abs(np.asarray(residuals_pct, dtype=float))
    s = np.asarray(sigma, dtype=float)
    ok = (r > 0) & (s > 0) & np.isfinite(r) & np.isfinite(s)
    if ok.sum() < 2:
        return float("nan")
    x = np.log(s[ok])
    y = np.log(r[ok])
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return float("nan")
    return float(np.dot(xc, y - y.mean()) / denom)


def coverage_by_sigma_stratum(residuals_pct, sigma, exponent: float = 1.0,
                              n_strata: int = 10, alpha: float = ALPHA
                              ) -> tuple[np.ndarray, float, float]:
    """LEVEL-MATCHED conditional coverage across strata of `sigma`.

    DIAGNOSTIC. Returns `(coverage_per_stratum, mean_abs_error_pp, threshold)`.

    The threshold is the empirical `1 - alpha` quantile of the scores, so
    **marginal coverage is exactly `1 - alpha` by construction** and any spread
    across strata is conditional miscalibration rather than the band being too
    wide overall.

    That separation is the entire point, and skipping it has already cost a read.
    `mean_d |coverage - target|` computed WITHOUT level-matching falls whenever
    marginal coverage moves toward target for any reason at all, so a uniformly
    narrower band scores as a conditional fix -- on 2026-08-12 a placebo with its
    state variable SHUFFLED across dates passed a pre-registered bar on it. See
    `docs/changelog/2026-08-12-the-band-is-tilted-in-sigma.md`.

    `exponent` is what sigma is raised to before dividing, so `1.0` reproduces
    what production serves and a fitted `elasticity` tests the remedy.
    """
    r = np.abs(np.asarray(residuals_pct, dtype=float))
    s = np.asarray(sigma, dtype=float)
    ok = np.isfinite(r) & np.isfinite(s) & (s > 0)
    r, s = r[ok], s[ok]
    if r.size < n_strata:
        return np.array([]), float("nan"), float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = r / (s ** float(exponent))
    good = np.isfinite(scores)
    scores, s = scores[good], s[good]
    if scores.size < n_strata:
        return np.array([]), float("nan"), float("nan")

    threshold = float(np.quantile(scores, 1.0 - alpha))
    covered = scores <= threshold

    # Stratify on sigma ITSELF, never on the exponentiated form: the strata must
    # be the same rows whatever `exponent` is, or two calls are not comparable.
    edges = np.quantile(s, np.linspace(0.0, 1.0, n_strata + 1)[1:-1])
    idx = np.searchsorted(edges, s, side="right")
    per = np.array([covered[idx == k].mean() if np.any(idx == k) else np.nan
                    for k in range(n_strata)])
    err = float(np.nanmean(np.abs(per - (1.0 - alpha)))) * 100.0
    return per, err, threshold


def band(mid_pct, sigma, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric band around the median, in percentage-return space.

    Cannot cross by construction, which is why predict() no longer needs
    _fix_quantile_crossing.
    """
    mid = np.asarray(mid_pct, dtype=float)
    half = float(q_hat) * np.asarray(sigma, dtype=float)
    return mid - half, mid + half
