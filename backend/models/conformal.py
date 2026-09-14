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

# The exponent `sigma` is raised to before it divides the residual. 1.0 is what
# locally-weighted split conformal assumes and what this module served until
# 2026-08-12; the measured value is 0.31-0.43, which is why the band is tilted
# (see `scale`). BETA_NEUTRAL is the no-op and the default everywhere.
BETA_NEUTRAL = 1.0
# Fitted betas are clamped here. 0.2 is below every value ever measured on this
# archive (the minimum across 24 time blocks was 0.209) and 1.0 means a fitted
# beta can never make the band WIDER than today by moving the exponent the wrong
# way. A clamp that binds is a bug, so callers log when it does.
BETA_MIN = 0.2
BETA_MAX = 1.0
# Below this standard deviation of `log sigma`, the elasticity regression has no
# x-axis and its slope is meaningless. `denom > 0` does NOT catch that case: for
# a constant sigma, `x - x.mean()` is floating-point noise around 1e-16 rather
# than exact zero, so the sum of squares is a tiny POSITIVE number and the slope
# comes back as the ratio of two noises -- measured at 0.5 on a constant sigma,
# a plausible-looking value that would have been persisted and served. Real data
# carries sd(log sigma) ~ 0.6, so this threshold cannot bind on it.
LOG_SIGMA_MIN_SD = 1e-8


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


def sigma_from_columns(price_std_60d, price, floor: float, cap: float, fallback: float | None = None) -> np.ndarray:
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


def scale(sigma, beta: float = BETA_NEUTRAL) -> np.ndarray:
    """The denominator the nonconformity score is divided by: `sigma ** beta`.

    THE ONE PLACE THE EXPONENT IS APPLIED, so `calibrate` and `band` cannot
    disagree about it. They must not: `sigma` is a fraction around 0.07, so
    `sigma ** 0.4` is roughly 0.35 -- five times larger -- and `q_hat` absorbs
    the whole difference. A `q_hat` fitted here at one beta and applied at
    another is not partially corrected, it is wrong by about 5x.

    `beta == 1.0` returns `sigma` itself rather than `sigma ** 1.0`, so the
    default path is provably byte-identical to what this module served before
    the exponent existed.
    """
    s = np.asarray(sigma, dtype=float)
    if float(beta) == BETA_NEUTRAL:
        return s
    with np.errstate(divide="ignore", invalid="ignore"):
        return s ** float(beta)


def resolve_scale(sigma, beta: float = BETA_NEUTRAL, learned=None) -> np.ndarray:
    """THE one place a nonconformity denominator is chosen.

    Two answers to the same question, and they are alternatives rather than
    layers:

    - `sigma ** beta` — the hand-picked variable, optionally damped.
    - `learned` — `models/scale_model.py`'s estimate of how large this item's
      error usually is, fitted directly on `|residual|`.

    **A learned scale is served as-is and `beta` does not apply to it.** The
    exponent exists to correct `sigma`'s over-reaction; a scale fitted against
    the residuals it normalises has no such distortion to correct, so raising it
    to 0.35 would be a second correction on top of an estimate that already fits
    — and would silently reintroduce the tilt in the opposite direction, which
    is exactly the failure `2026-08-12-served-sigma-profile.md` measured. Asking
    for both is therefore a mistake with no sensible reading, and it raises
    rather than picking one.
    """
    if learned is None:
        return scale(sigma, beta)
    if float(beta) != BETA_NEUTRAL:
        raise ValueError(
            f"both a learned scale and beta={beta} were supplied. They are "
            f"alternative denominators, not layers: the exponent corrects "
            f"sigma's over-reaction and a fitted scale has none to correct. "
            f"Set LEARNED_SCALE=1 or SIGMA_EXPONENT=1, never both."
        )
    return np.asarray(learned, dtype=float)


def fit_beta(residuals_pct, sigma, min_rows: int = 1_000) -> float:
    """`elasticity`, made safe to persist in an artifact and serve from.

    `elasticity` is the diagnostic and returns NaN freely. This is the
    production wrapper: it returns BETA_NEUTRAL -- never NaN -- on every
    degenerate input, because a NaN beta reaching `scale` produces a NaN band,
    and a NaN half-width surfaces in the API.

    Returns `(beta, clamped)` semantics via the log, not the signature: the
    clamp binding means the measured elasticity left the range this archive has
    ever produced, which is a data problem the caller should surface, so
    `beta_was_clamped` is exposed for that check rather than hidden.
    """
    r = np.abs(np.asarray(residuals_pct, dtype=float))
    s = np.asarray(sigma, dtype=float)
    ok = (r > 0) & (s > 0) & np.isfinite(r) & np.isfinite(s)
    if int(ok.sum()) < int(min_rows):
        return BETA_NEUTRAL
    b = elasticity(r[ok], s[ok])
    if not np.isfinite(b):
        return BETA_NEUTRAL
    return float(min(max(b, BETA_MIN), BETA_MAX))


def beta_was_clamped(residuals_pct, sigma, min_rows: int = 1_000) -> bool:
    """Whether `fit_beta` had to clamp -- i.e. whether the fit left [0.2, 1.0]."""
    r = np.abs(np.asarray(residuals_pct, dtype=float))
    s = np.asarray(sigma, dtype=float)
    ok = (r > 0) & (s > 0) & np.isfinite(r) & np.isfinite(s)
    if int(ok.sum()) < int(min_rows):
        return False
    b = elasticity(r[ok], s[ok])
    return bool(np.isfinite(b) and (b < BETA_MIN or b > BETA_MAX))


def calibrate(residuals_pct, sigma, alpha: float = ALPHA, beta: float = BETA_NEUTRAL, learned_scale=None) -> float:
    """q_hat: the conformal quantile of normalized absolute residuals.

    `residuals_pct` are y - y_hat in percentage-return space, from
    out-of-fold predictions. Scores are |residual| / sigma ** beta, so q_hat is
    dimensionless and multiplies `sigma ** beta` at serve time.

    ⚠️ q_hat IS ONLY MEANINGFUL BESIDE THE `beta` IT WAS FITTED AT, and beside
    the SCALE it was fitted against. Two q_hats from different betas -- or one
    from `sigma` and one from a learned scale -- must never be compared,
    differenced, or substituted for one another. See `resolve_scale`.

    Uses the finite-sample corrected level ceil((n+1)(1-alpha))/n, which is
    what gives split conformal its distribution-free coverage guarantee.
    """
    res = np.asarray(residuals_pct, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    if res.size == 0:
        raise ValueError("empty calibration set: cannot compute q_hat")

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs(res) / resolve_scale(sig, beta, learned_scale)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("empty calibration set: no finite nonconformity scores")

    n = scores.size
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def calibrate_signed(
    residuals_pct, sigma, alpha: float = ALPHA, beta: float = BETA_NEUTRAL, learned_scale=None
) -> tuple[float, float]:
    """Two SIGNED conformal quantiles of `residual / scale`: `(q_lo, q_hi)`.

    Where `calibrate` folds the residual with `np.abs` and returns one q_hat for
    a band symmetric about the mid, this splits `alpha` into two tails and reads
    the lower and upper quantiles of the *signed* score. On a residual whose
    median is not zero — what a q50 fitted above the median produces
    (DIRECTION_UPWEIGHT) — the absolute quantile is inflated by the fat tail;
    the signed pair centres on the residual's own median and is narrower at the
    same nominal coverage.

    ⚠️ `(q_lo, q_hi)` is a MATCHED SET with `beta` and the scale, exactly as
    `q_hat` is — see `calibrate` and `resolve_scale`. Never difference or
    substitute one across a beta or scale boundary. `band_signed(-q_hat, +q_hat)`
    reproduces the symmetric band bit-for-bit, which is the fallback for a
    pre-signed artifact.

    Two-sided finite-sample correction: the upper leg uses
    `ceil((n+1)(1-alpha/2))/n` and the lower `floor((n+1)(alpha/2))/n`, so the
    interval keeps split conformal's `1 - alpha` coverage.
    """
    res = np.asarray(residuals_pct, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    if res.size == 0:
        raise ValueError("empty calibration set: cannot compute signed q_hat")

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = res / resolve_scale(sig, beta, learned_scale)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("empty calibration set: no finite nonconformity scores")

    n = scores.size
    level_hi = min(np.ceil((n + 1) * (1.0 - alpha / 2.0)) / n, 1.0)
    level_lo = max(np.floor((n + 1) * (alpha / 2.0)) / n, 0.0)
    return (float(np.quantile(scores, level_lo)), float(np.quantile(scores, level_hi)))


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
    # Not `denom <= 0`: see LOG_SIGMA_MIN_SD. A constant sigma leaves noise, not
    # zeros, and returned a confident-looking 0.5.
    if not np.isfinite(denom) or np.sqrt(denom / xc.size) < LOG_SIGMA_MIN_SD:
        return float("nan")
    return float(np.dot(xc, y - y.mean()) / denom)


def coverage_by_sigma_stratum(
    residuals_pct, sigma, exponent: float = 1.0, n_strata: int = 10, alpha: float = ALPHA
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
    per = np.array([covered[idx == k].mean() if np.any(idx == k) else np.nan for k in range(n_strata)])
    err = float(np.nanmean(np.abs(per - (1.0 - alpha)))) * 100.0
    return per, err, threshold


def band(mid_pct, sigma, q_hat: float, beta: float = BETA_NEUTRAL, learned_scale=None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric band around the median, in percentage-return space.

    Cannot cross by construction, which is why predict() no longer needs
    _fix_quantile_crossing.

    ⚠️ `beta` MUST be the one `q_hat` was calibrated at. Serving a beta-era
    q_hat at `beta = 1.0` inflates the half-width by roughly 5x; serving a
    beta=1 q_hat at a fitted beta shrinks it by the same factor. There is no
    partially-correct pairing, which is why both are persisted together or not
    at all.

    ⚠️ The same holds for `learned_scale`, and harder: a `q_hat` calibrated
    against a learned scale is in units of that scale, so serving it against
    `sigma` is not a degraded band but an unrelated one. The artifact persists
    the scale model with `q_hat` for that reason.
    """
    mid = np.asarray(mid_pct, dtype=float)
    half = float(q_hat) * resolve_scale(sigma, beta, learned_scale)
    return mid - half, mid + half


WACI_N_BINS = 10
WACI_KERNEL_BW = 0.5


def calibrate_signed_waci(
    residuals_pct,
    sigma,
    alpha: float = ALPHA,
    beta: float = BETA_NEUTRAL,
    learned_scale=None,
    n_bins: int = WACI_N_BINS,
    kernel_bw: float = WACI_KERNEL_BW,
) -> dict:
    """Width-Adaptive Conformal Inference: per-scale-bin signed quantiles.

    Instead of one global (q_lo, q_hi), partitions the calibration set by
    `scale` (the band-width denominator) into `n_bins` quantile bins and
    computes a signed conformal quantile pair per bin. At serve time each
    item's scale selects its bin's quantiles via Gaussian-kernel interpolation,
    so the coverage guarantee is conditional on band width rather than only
    marginal.

    Returns a dict with keys:
        bin_edges: array of shape (n_bins - 1,) — scale quantile boundaries
        bin_q_lo:  array of shape (n_bins,) — lower quantile per bin
        bin_q_hi:  array of shape (n_bins,) — upper quantile per bin
        bin_centres: array of shape (n_bins,) — median scale in each bin
        fallback_q_lo: float — global q_lo (for items outside the range)
        fallback_q_hi: float — global q_hi
    """
    res = np.asarray(residuals_pct, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    if res.size == 0:
        raise ValueError("empty calibration set")

    sc = resolve_scale(sig, beta, learned_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = res / sc
    ok = np.isfinite(scores) & np.isfinite(sc) & (sc > 0)
    scores, sc = scores[ok], sc[ok]
    if scores.size < n_bins * 10:
        q_lo, q_hi = calibrate_signed(residuals_pct, sigma, alpha, beta, learned_scale)
        return {
            "bin_edges": np.array([]),
            "bin_q_lo": np.array([q_lo]),
            "bin_q_hi": np.array([q_hi]),
            "bin_centres": np.array([float(np.median(sc))]),
            "fallback_q_lo": q_lo,
            "fallback_q_hi": q_hi,
        }

    n = scores.size
    level_hi = min(np.ceil((n + 1) * (1.0 - alpha / 2.0)) / n, 1.0)
    level_lo = max(np.floor((n + 1) * (alpha / 2.0)) / n, 0.0)

    fallback_q_lo = float(np.quantile(scores, level_lo))
    fallback_q_hi = float(np.quantile(scores, level_hi))

    edges = np.quantile(sc, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    bin_idx = np.searchsorted(edges, sc, side="right")

    bin_q_lo = np.empty(n_bins)
    bin_q_hi = np.empty(n_bins)
    bin_centres = np.empty(n_bins)

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 20:
            bin_q_lo[b] = fallback_q_lo
            bin_q_hi[b] = fallback_q_hi
            bin_centres[b] = float(np.median(sc))
            continue
        bin_scores = scores[mask]
        nb = bin_scores.size
        lhi = min(np.ceil((nb + 1) * (1.0 - alpha / 2.0)) / nb, 1.0)
        llo = max(np.floor((nb + 1) * (alpha / 2.0)) / nb, 0.0)
        bin_q_lo[b] = float(np.quantile(bin_scores, llo))
        bin_q_hi[b] = float(np.quantile(bin_scores, lhi))
        bin_centres[b] = float(np.median(sc[mask]))

    return {
        "bin_edges": edges,
        "bin_q_lo": bin_q_lo,
        "bin_q_hi": bin_q_hi,
        "bin_centres": bin_centres,
        "fallback_q_lo": fallback_q_lo,
        "fallback_q_hi": fallback_q_hi,
    }


def waci_lookup(scale_values, waci_params: dict, kernel_bw: float = WACI_KERNEL_BW) -> tuple[np.ndarray, np.ndarray]:
    """Look up per-item (q_lo, q_hi) from WACI parameters via kernel smoothing.

    For each item's scale value, computes a Gaussian-kernel-weighted average
    of the bin quantiles, so the transition between bins is smooth.
    """
    sc = np.asarray(scale_values, dtype=float)
    centres = waci_params["bin_centres"]
    bq_lo = waci_params["bin_q_lo"]
    bq_hi = waci_params["bin_q_hi"]

    if centres.size <= 1:
        return (np.full(sc.shape, waci_params["fallback_q_lo"]), np.full(sc.shape, waci_params["fallback_q_hi"]))

    q_lo_out = np.empty(sc.shape)
    q_hi_out = np.empty(sc.shape)

    for i, s in enumerate(sc):
        dists = (s - centres) / (kernel_bw * np.std(centres) + 1e-12)
        weights = np.exp(-0.5 * dists**2)
        weights /= weights.sum() + 1e-12
        q_lo_out[i] = float(np.dot(weights, bq_lo))
        q_hi_out[i] = float(np.dot(weights, bq_hi))

    return q_lo_out, q_hi_out


def band_signed_waci(
    mid_pct, sigma, waci_params: dict, beta: float = BETA_NEUTRAL, learned_scale=None, kernel_bw: float = WACI_KERNEL_BW
) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric band using Width-Adaptive per-item quantiles.

    Like `band_signed` but each item gets its own (q_lo, q_hi) based on
    its scale value, producing tighter bands for low-vol items and wider
    bands for high-vol items — fixing the sigma-tilt coverage profile.
    """
    mid = np.asarray(mid_pct, dtype=float)
    sc = resolve_scale(sigma, beta, learned_scale)
    q_lo_arr, q_hi_arr = waci_lookup(sc, waci_params, kernel_bw)
    return mid + q_lo_arr * sc, mid + q_hi_arr * sc


def band_signed(
    mid_pct, sigma, q_lo: float, q_hi: float, beta: float = BETA_NEUTRAL, learned_scale=None
) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric band from a signed `(q_lo, q_hi)` pair: `mid + q·scale`.

    The signed counterpart of `band`. `q_lo` is typically negative, so the low
    edge sits below the mid and the high edge above, but the offsets need not be
    equal — that is the whole point. `band_signed(mid, sigma, -q_hat, q_hat)` is
    byte-identical to `band(mid, sigma, q_hat)`, which is what makes the pair a
    drop-in with a symmetric fallback.

    ⚠️ Same matched-pair rule as `band`: `beta` and `learned_scale` MUST be the
    ones the pair was calibrated at, or the band is wrong by the scale factor,
    not partially corrected. Cannot cross as long as `q_lo <= q_hi`, which
    `calibrate_signed` guarantees.
    """
    mid = np.asarray(mid_pct, dtype=float)
    sc = resolve_scale(sigma, beta, learned_scale)
    return mid + float(q_lo) * sc, mid + float(q_hi) * sc
