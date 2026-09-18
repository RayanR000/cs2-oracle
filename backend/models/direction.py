"""Pure direction / rank / band-centre math, extracted from models/forecaster.py.

These functions take arrays and return arrays: no self, no DB, no env (the two
module constants they read are defined here). `ItemForecaster` aliases them as
`_`-prefixed staticmethods so every existing call site keeps working; new code
should import this module directly. First cut of the forecaster decomposition
(see docs/changelog/2026-09-15-forecaster-decomposition.md).
"""

import numpy as np
import pandas as pd
from backtest.scoring import HEADLINE_MIN_TIER, MIN_HEADLINE_DATES
from scipy.stats import spearmanr

#: Flat band (percent) for the legacy fixed-threshold direction bucketing.
#: Moved here with the functions that read it; re-exported from
#: models.forecaster so `from models.forecaster import ...` keeps working.
DIRECTION_FLAT_TOLERANCE_PCT = 0.5


def directional_accuracy(pred_returns, actual_returns) -> float:
    """Percent of rows whose predicted return direction matches the actual.

    Uses the same 3-class up/flat/down bucketing (DIRECTION_FLAT_TOLERANCE_PCT)
    as model evaluation so baseline and model numbers are directly
    comparable. Rows with NaN actuals are skipped. Returns 0.0 for an
    empty comparison.
    """
    tol = DIRECTION_FLAT_TOLERANCE_PCT
    pred = np.asarray(pred_returns, dtype=float)
    actual = np.asarray(actual_returns, dtype=float)
    mask = ~np.isnan(actual)
    n = int(mask.sum())
    if not n:
        return 0.0
    hits = int((direction_classes(pred[mask], tol) == direction_classes(actual[mask], tol)).sum())
    return round(hits / n * 100, 1)


def direction_classes(returns, threshold=DIRECTION_FLAT_TOLERANCE_PCT) -> np.ndarray:
    """Bucket % returns into 0=down, 1=flat, 2=up using a flat band of
    ``threshold`` (scalar or per-row array, percent). Scalar reproduces the
    legacy fixed-±DIRECTION_FLAT_TOLERANCE_PCT behavior."""
    r = np.asarray(returns, dtype=float)
    thr = np.asarray(threshold, dtype=float)
    return np.where(r > thr, 2, np.where(r < -thr, 0, 1)).astype(int)


def demean_returns(returns, factor) -> np.ndarray:
    """Subtract the market factor from % returns, giving the idiosyncratic
    residual ``e = r - m``. See ``models/market_factor.py`` for the factor.

    A missing factor demeans by zero rather than producing NaN, so the row
    keeps the raw label instead of being dropped. That matters to any paired
    arm comparison: dropping rows in one arm only changes its row counts and
    breaks the pairing.

    Not used by training or serving. It survives the removal of the
    market-relative label experiment (refuted 2026-08-06, see
    docs/changelog/2026-08-06-market-relative-labels-refuted.md) because
    scripts/ab_test_item_metadata.py calls it to re-score its arms against a
    demeaned target -- the run that amended the item-metadata conclusions in
    docs/research/accuracy-opportunities.md.
    """
    r = np.asarray(returns, dtype=float)
    if factor is None:
        return r.copy()
    m = np.asarray(factor, dtype=float)
    return r - np.nan_to_num(m, nan=0.0)


def has_date_coverage(forecast_dates) -> bool:
    """True when distinct non-null forecast dates reach MIN_HEADLINE_DATES.

    Row count cannot substitute for this. Items sharing a forecast_date
    share one market-wide move, so a five-figure cohort on two dates is
    nearer two observations than 11,000 — see
    docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md.
    Reuses the reporting threshold so the value we fit on and the value we
    report cannot drift apart.
    """
    distinct = {d for d in forecast_dates if d is not None and not pd.isna(d)}
    return len(distinct) >= MIN_HEADLINE_DATES


def direction_threshold(sigma, horizon: int, k: float, floor: float, cap: float) -> np.ndarray:
    """Per-row flat-band threshold (percent) = clamp(k * sigma * sqrt(h),
    floor, cap). ``sigma`` is trailing daily-return std in percent; scalar
    or array. Always returns a float ndarray."""
    s = np.atleast_1d(np.asarray(sigma, dtype=float))
    raw = k * s * np.sqrt(float(horizon))
    return np.clip(raw, floor, cap)


def served_cohort_multiplier(base_weights, tiers, served_share: float) -> float:
    """Multiplier for served (>= $1) rows that makes them carry
    ``served_share`` of the total training weight.

    Solving ``m*W_s / (m*W_s + W_n) = share`` for m gives

        m = share * W_n / ((1 - share) * W_s)

    where W_s / W_n are the sums of *base_weights* over the served and
    non-served partitions. Taking the sums over the already-computed mover
    weights (rather than over row counts) is what makes the resulting share
    exact once the two weightings compose.

    The knob is a share and not a raw multiplier on purpose: a multiplier's
    meaning drifts with the frame's tier composition, which varies fold to
    fold, while the share is the quantity we actually mean.

    Returns 1.0 — leave the weights alone — when the target is unreachable
    or already met: no served rows (W_s == 0) and no non-served rows
    (W_n == 0) both have no multiplier that moves the share.
    """
    if not 0.0 < served_share < 1.0:
        raise ValueError(f"served_share must be in (0, 1), got {served_share!r}")
    w = np.asarray(base_weights, dtype=float)
    served = np.asarray(tiers) >= HEADLINE_MIN_TIER
    w_served = float(w[served].sum())
    w_other = float(w[~served].sum())
    if w_served <= 0.0 or w_other <= 0.0:
        return 1.0
    return served_share * w_other / ((1.0 - served_share) * w_served)


def recenter_on_momentum(low_ret, mid_ret, high_ret, momentum_ret):
    """Recenter quantile return forecasts on the trailing (momentum) return,
    preserving each item's calibrated interval half-widths.

    Where ``momentum_ret`` is NaN (insufficient history) the model's own
    forecast is kept unchanged. Returns (low, mid, high) in return space.
    Because the inputs are already monotone (low <= mid <= high), the
    preserved non-negative offsets keep the recentred triple monotone too.
    """
    mid_ret = np.asarray(mid_ret, dtype=float)
    low_ret = np.asarray(low_ret, dtype=float)
    high_ret = np.asarray(high_ret, dtype=float)
    momentum_ret = np.asarray(momentum_ret, dtype=float)
    low_off = mid_ret - low_ret
    high_off = high_ret - mid_ret
    new_mid = np.where(np.isnan(momentum_ret), mid_ret, momentum_ret)
    return new_mid - low_off, new_mid, new_mid + high_off


def recenter_on_direction(low_ret, mid_ret, high_ret, direction_class):
    """Recenter forecasts so the median's sign matches the classifier's call,
    preserving each item's interval half-widths.

    down(0) -> -|mid|, flat(1) -> 0, up(2) -> +|mid|. Keeping |mid| as the
    magnitude means the quantile model still sets *how much*; the classifier
    only sets *which way*. Returns (low, mid, high).
    """
    mid_ret = np.asarray(mid_ret, dtype=float)
    low_ret = np.asarray(low_ret, dtype=float)
    high_ret = np.asarray(high_ret, dtype=float)
    cls_arr = np.asarray(direction_class, dtype=int)
    low_off = mid_ret - low_ret
    high_off = high_ret - mid_ret
    mag = np.abs(mid_ret)
    new_mid = np.select(
        [cls_arr == 2, cls_arr == 0, cls_arr == 1],
        [mag, -mag, 0.0],
        default=mid_ret,
    )
    return new_mid - low_off, new_mid, new_mid + high_off


def fix_quantile_crossing(low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enforce low <= mid <= high via isotonic regression (PAV for 3 points).

    For each item where quantiles cross, projects [low, mid, high] onto
    the non-decreasing constraint using the Pool-Adjacent-Violators
    algorithm. This preserves item-level interval width as much as
    possible, unlike a global average-half-width imputation.

    Returns (low_fixed, high_fixed) arrays with low_fixed <= mid <= high_fixed.
    """
    v0, v1, v2 = low.copy().astype(np.float64), mid.copy().astype(np.float64), high.copy().astype(np.float64)

    # Pattern A: v0 > v1 (first two cross)
    cross_01 = v0 > v1
    if cross_01.any():
        pool_01 = (v0[cross_01] + v1[cross_01]) * 0.5
        v0[cross_01] = pool_01
        v1[cross_01] = pool_01

    # Pattern B: after fixing A, check v1 > v2 (last two cross)
    cross_12 = v1 > v2
    if cross_12.any():
        pool_12 = (v1[cross_12] + v2[cross_12]) * 0.5
        v1[cross_12] = pool_12
        v2[cross_12] = pool_12

    # Pattern C: after fixing B, check again if A was re-broken (pooled
    # v1,v2 < original v0). Only possible when all three crossed.
    recross_01 = v0 > v1
    if recross_01.any():
        pool_all = (v0[recross_01] + v1[recross_01] + v2[recross_01]) / 3.0
        v0[recross_01] = pool_all
        v1[recross_01] = pool_all
        v2[recross_01] = pool_all

    return v0, v2


def blend_returns_with_prior(
    low_ret_arr: np.ndarray,
    mid_ret_arr: np.ndarray,
    high_ret_arr: np.ndarray,
    prior: dict[str, np.ndarray],
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend current return-space predictions toward the prior day's.

    Returns the (low, mid, high) arrays after exponentially smoothing with
    the previous forecast. Items without a prior are left untouched.
    """
    mask = prior["mask"]
    if not mask.any() or weight <= 0:
        return low_ret_arr, mid_ret_arr, high_ret_arr
    mid_ret_arr = np.where(mask, (1 - weight) * mid_ret_arr + weight * prior["mid_ret"], mid_ret_arr)
    low_ret_arr = np.where(mask, (1 - weight) * low_ret_arr + weight * prior["low_ret"], low_ret_arr)
    high_ret_arr = np.where(mask, (1 - weight) * high_ret_arr + weight * prior["high_ret"], high_ret_arr)
    return low_ret_arr, mid_ret_arr, high_ret_arr


def direction_records(pred_returns, actual_returns, dates) -> list:
    """Rows in the shape `backtest/directional_test.py` expects.

    Built from the quantile-median sign, not the classifier, so the PT test
    and the constant-call baseline stay available when the diagnostic
    classifier is skipped. `forecast_date` is a string because
    `pesaran_timmermann` clusters on it and sorts it.
    """
    tol = DIRECTION_FLAT_TOLERANCE_PCT
    p = np.asarray(pred_returns, dtype=float)
    a = np.asarray(actual_returns, dtype=float)
    d = pd.to_datetime(pd.Series(dates).to_numpy()).strftime("%Y-%m-%d")
    pdir = np.where(p > tol, "up", np.where(p < -tol, "down", "flat"))
    adir = np.where(a > tol, "up", np.where(a < -tol, "down", "flat"))
    return [
        {
            "predicted_direction": str(pd_),
            "actual_direction": str(ad),
            "direction_correct": bool(pd_ == ad),
            "forecast_date": str(fd),
        }
        for pd_, ad, fd in zip(pdir, adir, d)
    ]


def direction_records_from_classes(pred_cls, actual_cls, dates) -> list:
    """The same record shape as `direction_records`, but for the SERVED
    classifier's predictions rather than the quantile-median sign.

    `direction_classes` buckets 0=down, 1=flat, 2=up, and it is what both
    the classifier is trained against and `actual_cls` is built from. The
    actual directions therefore agree row-for-row with `direction_records`,
    which is what makes `constant_call_accuracy` and `realised_down_rate` --
    properties of the outcomes alone -- shared between the two signals
    rather than needing to be recomputed per signal.

    Only available when CV_DIAGNOSTIC_CLASSIFIER=1. There is no fallback on
    purpose: a PT verdict that silently described a different signal
    depending on an environment variable is the failure this exists to fix.
    """
    names = ("down", "flat", "up")
    p = np.asarray(pred_cls, dtype=int)
    a = np.asarray(actual_cls, dtype=int)
    d = pd.to_datetime(pd.Series(dates).to_numpy()).strftime("%Y-%m-%d")
    return [
        {
            "predicted_direction": names[int(pc)],
            "actual_direction": names[int(ac)],
            "direction_correct": bool(pc == ac),
            "forecast_date": str(fd),
        }
        for pc, ac, fd in zip(p, a, d)
    ]


def summarise_rank_ic(fold_metrics: list) -> dict:
    """The rank IC block of `cv_results`: pooled, tied, and both bars.

    Two pairs, and the second is the one an arm is decided on.
    `mean_rank_ic` / `mean_naive_rank_ic` pool the whole served
    cross-section, so both legs are measured against a label whose
    denominator is the raw anchor quote the features are also built from.
    The `_tied` pair restricts to rows where that quote equals its own local
    median, which is the only cohort where neither the raw basis nor the
    smoothed one carries `p[d]/S[d]`.

    The pooled keys are NOT redefined. They are the series every historical
    `meta.json` holds and the trust warning reads, and silently changing
    what they mean would make the trend a comparison of two quantities.

    `tied_dates` rides along because the `min_rows` bar bites harder on a
    subset: a horizon whose tied number rests on two dates is not a
    measurement, and nothing else in the payload would say so.
    """

    def _mean(key):
        vals = [m[key] for m in fold_metrics if m.get(key) is not None]
        # 4 dp: a rank IC lives in [-1, 1] and the differences that matter
        # here are third-decimal. 2 dp rounds 0.1199 to 0.12 and makes the
        # naive comparison unreadable.
        return round(float(np.mean(vals)), 4) if vals else None

    def _edge(model, naive):
        return None if (model is None or naive is None) else round(model - naive, 4)

    mean_rank_ic = _mean("rank_ic")
    mean_naive = _mean("naive_rank_ic")
    mean_tied = _mean("rank_ic_tied")
    mean_naive_tied = _mean("naive_rank_ic_tied")
    # C2 lambdarank arm (None throughout when LAMBDARANK is off). The verdict
    # pair is the tied cohort: it must beat the naive baseline AND the q50's
    # own ordering, both where `p[d]/S[d] == 1`.
    mean_lr = _mean("lr_rank_ic")
    mean_lr_tied = _mean("lr_rank_ic_tied")
    return {
        "mean_rank_ic": mean_rank_ic,
        "mean_naive_rank_ic": mean_naive,
        "rank_ic_edge_vs_naive": _edge(mean_rank_ic, mean_naive),
        "mean_rank_ic_tied": mean_tied,
        "mean_naive_rank_ic_tied": mean_naive_tied,
        "rank_ic_edge_vs_naive_tied": _edge(mean_tied, mean_naive_tied),
        "mean_lr_rank_ic": mean_lr,
        "mean_lr_rank_ic_tied": mean_lr_tied,
        "lr_rank_ic_edge_vs_naive_tied": _edge(mean_lr_tied, mean_naive_tied),
        "lr_rank_ic_edge_vs_q50_tied": _edge(mean_lr_tied, mean_tied),
        "tied_rows": sum(int(m.get("n_tied") or 0) for m in fold_metrics),
        "tied_dates": sum(int(m.get("rank_ic_tied_dates") or 0) for m in fold_metrics),
        "rank_ic_dates": sum(int(m.get("rank_ic_dates") or 0) for m in fold_metrics),
    }


def within_date_rank_ic(pred, actual, dates, mask=None, min_rows: int = 20) -> float | None:
    """Mean within-date Spearman correlation of `pred` against `actual`.

    The cross-sectional metric: it measures whether the model orders items
    correctly on a given day, which is what the product serves, and it is
    immune to the realised direction mix that dominates DA. Dates with
    fewer than `min_rows` served rows, or with no variation in either leg,
    contribute nothing rather than a degenerate 0.
    """
    return within_date_rank_ic_detail(pred, actual, dates, mask, min_rows)[0]


def within_date_rank_ic_detail(pred, actual, dates, mask=None, min_rows: int = 20) -> "tuple[float | None, int]":
    """`within_date_rank_ic`, and the number of dates it actually read.

    The count is not decoration. The same `min_rows` bar applied to a
    SUBSET of the cross-section drops dates the pooled figure keeps -- the
    tied cohort is roughly a third of the panel, and one 2026-08-11 anchor
    had 26 tied items of 669 -- so the two columns can silently describe
    different calendars while looking like a paired comparison.
    """
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    d = pd.to_datetime(pd.Series(dates).to_numpy())
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        p, a, d = p[m], a[m], d[m]
    if len(p) < min_rows:
        return None, 0
    frame = pd.DataFrame({"d": d, "p": p, "a": a})
    ics = []
    for _, g in frame.groupby("d"):
        if len(g) < min_rows:
            continue
        if g["p"].nunique() < 2 or g["a"].nunique() < 2:
            continue
        ic = spearmanr(g["p"], g["a"]).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
    if not ics:
        return None, 0
    return round(float(np.mean(ics)), 4), len(ics)
