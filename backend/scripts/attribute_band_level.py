"""Why is the SERVED band 1.52-1.55x as wide as the same `q_hat`'s calibration band?

Measured 2026-08-13 (`changelog/2026-08-13-band-level-on-audited-anchors.md`): the
calibration line reports a median half-width of 6.41 / 9.53 / 13.66 / 20.61% of mid on
its own OOF records, and `replay_serving.py` reads 9.80 / 14.75 / 21.22 / 31.32% on the
four audited anchors. The ratio is 1.529 / 1.548 / 1.553 / 1.520 -- flat in horizon,
which is what a level defect looks like.

WHY THE RATIO IS A SIGMA RATIO. Both figures are half of `(high - low)` over the band's
own centre, and `conformal.band` sets `high - low = 2 * q_hat * sigma ** beta` in return
space, so with one `q_hat` and `beta = 1` the width ratio is the **sigma** ratio and
nothing else. (The `(1 + mid/100)` denominator in `range_pct` differs from 1 by the
predicted return, median 0.95%.) So this script decomposes exactly two things:

  WHICH items -- the served >=$1 cohort against the pooled calibration cross-section;
  WHEN       -- the same item's sigma on the anchor against its own pooled median.

AND THE RESIDUAL LEG, WHICH IS THE POINT. The conformal score is `|resid| / sigma`.
A sigma that is 1.5x higher at serving costs no coverage at all if `|resid|` is 1.5x
higher too -- the band is wider because the item is genuinely more volatile. Over-coverage
needs sigma to have risen MORE than the residual it normalises. So the reported quantity
is the RATIO OF RATIOS, and a value below 1 is the over-coverage.

READ THIS AS A STAND-IN, NOT AS THE ARTIFACT. Like `measure_conditional_qhat`, the panel
derives its own clip bounds and its own cohort, so no absolute number here is production's
-- only ratios are read, and the residual is the realised return rather than a booster's
error (justified at `attribute_marginal_coverage`'s head: predicted |return| is median
0.95% against half-widths of 10-31%).

Read-only: reads a voted price panel, writes nothing.

    venv/bin/python -m scripts.attribute_band_level --horizons 3,7,14,30
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import conformal
from models.forecaster import ItemForecaster, embargo_days
from scripts.measure_conditional_qhat import (
    default_voted_panel,
    load_panel,
    score_frame,
    sigma_bounds_for_panel,
)

# Both audits come from `replay_serving`, including the private profile query.
# Re-spelling that SQL here would put a second definition of "how was this day
# collected" in the repo -- which is the exact failure `cutovers_from_counts`
# was written to end -- and would have to re-apply AGENTS.md invariants 1 and 2
# by hand.
from scripts.replay_serving import (
    ANCHOR_NEIGHBOURHOOD_DAYS,
    DEFAULT_AUDIT_HORIZONS,
    _feed_profile,
    audit_anchor_feed,
    cutovers_from_counts,
    cutovers_in_outcome_window,
)

logger = logging.getLogger("attribute_band_level")

# Fixed by `docs/research/2026-08-13-date-level-sigma-rescaling-preregistration.md`
# BEFORE any number was seen. Do not tune.
DATE_LEVEL_SEED = 20260813
N_PERMUTATIONS = 200
N_BOOTSTRAP = 2_000
# The served control this instrument has to reproduce to referee anything, from
# run `31657639707` via `replay_serving.py`'s BAND COVERAGE. Leg (V).
SERVED_ANCHOR_COVERAGE = {
    3: {"2026-04-22": 89.85, "2026-05-16": 85.58, "2026-06-16": 85.99, "2026-07-06": 88.90},
    7: {"2026-04-22": 94.04, "2026-05-16": 91.52, "2026-06-16": 89.45, "2026-07-06": 80.86},
    14: {"2026-04-22": 95.90, "2026-05-16": 82.38, "2026-06-16": 85.99, "2026-07-06": 80.08},
    30: {"2026-04-22": 93.76, "2026-05-16": 88.12, "2026-06-16": 84.92, "2026-07-06": 77.94},
}
VALIDITY_MAE_PP = 3.0  # leg (V) bar
PLACEBO_MAX_PP = 1.0  # leg (P) bar

# The four anchors that pass `replay_serving.audit_anchor_feed`, i.e. the ones run
# `31657639707` served the 9.80 / 14.75 / 21.22 / 31.32% widths on.
AUDITED_ANCHORS = ["2026-04-22", "2026-05-16", "2026-06-16", "2026-07-06"]
# The served/calibration half-width ratios that run measured, per horizon. Printed
# beside this script's sigma ratio so the two can be compared without a second lookup.
SERVED_WIDTH_RATIO = {3: 1.529, 7: 1.548, 14: 1.553, 30: 1.520}
# The arm's anchor set, fixed by the pre-registration: six, not four, because the
# spread and dose-response bars are about variation ACROSS dates and four points
# cannot carry them. All six pass `replay_serving.audit_anchor_feed`.
ARM_ANCHORS = ["2026-02-14", "2026-03-10", "2026-04-06", "2026-04-22", "2026-06-16", "2026-07-06"]

# ---- the low-level read, fixed by
# `docs/research/2026-08-13-low-level-anchor-preregistration.md` BEFORE any
# coverage was computed. Do not tune.
LOW_LEVEL_QUANTILE = 0.25  # "low" = the bottom quartile of `L[t]`
LOW_ANCHOR_SPACING_DAYS = 30  # >= the longest horizon, so no two outcome windows overlap
LOW_ANCHOR_COUNT = 6  # the size of the high-vol set it is compared against
MIN_ANCHOR_ROWS = 300  # a rate on fewer rows than this is not a rate
# `sigma` is `price_std_60d / price`, so the panel's first 60 days hold a
# trailing window that is not yet 60 days long. Measured on this panel before
# any coverage was computed: 2024-07-09 has **100%** of its rows pinned at the
# sigma clip floor and the lowest `L[t]` in the panel by a factor of two, which
# would have made the warm-up the single largest dose in the low set. The rule
# is the feature's own window, not a fitted cutoff.
SIGMA_TRAILING_WINDOW_DAYS = 60
# Fixed by `--select-low` and written into the pre-registration BEFORE any
# coverage was computed. Leg B is the six lowest-`L[t]` eligible dates; every one
# is pre-2026, because the served regime holds only LEG_A_ANCHORS in the panel's
# bottom quartile at all four horizons. Leg A carries no bar: n = 1.
LOW_ANCHORS = ["2024-09-21", "2024-11-04", "2024-12-04", "2025-01-03", "2025-04-01", "2025-07-12"]
LEG_A_ANCHORS = ["2026-02-19"]
NOMINAL_COVERAGE_PP = 80.0  # the target every |cov - 80| below is measured against


def _med_ratio(num: pd.Series, den: pd.Series) -> float:
    a, b = float(np.nanmedian(num)), float(np.nanmedian(den))
    return a / b if b else float("nan")


def decompose(frame: pd.DataFrame, anchors) -> dict:
    """Split the served/pooled `sigma` and `|resid|` ratios into WHICH and WHEN.

    *frame* carries `date`, `item_id`, `sigma`, `absr`; *anchors* are the served
    dates. Returns both bases:

    `pooled_*`  the served rows against every row in the panel — cohort AND date
                effects together, which is the quantity a served-vs-calibration
                width ratio actually is.
    `same_*`    each served row against its OWN item's pooled median, so the
                cross-section cancels and only the date effect survives.

    `score_*` is `|resid|` ratio over `sigma` ratio. **That is the ratio the band's
    coverage responds to**, because the conformal score is `|resid| / sigma`: a
    served `sigma` twice the norm costs nothing if the residual doubled too, and
    below 1 means the band is wider than the residual justifies.
    """
    served = frame[frame["date"].isin(set(anchors))]
    if served.empty:
        return {}
    own_sigma = frame.groupby("item_id")["sigma"].median()
    own_absr = frame.groupby("item_id")["absr"].median()
    out = {
        "n_served": len(served),
        "n_pooled": len(frame),
        "pooled_sigma": _med_ratio(served["sigma"], frame["sigma"]),
        "pooled_absr": _med_ratio(served["absr"], frame["absr"]),
        "same_sigma": float(np.nanmedian(served["sigma"] / served["item_id"].map(own_sigma))),
        "same_absr": float(np.nanmedian(served["absr"] / served["item_id"].map(own_absr))),
    }
    for basis in ("pooled", "same"):
        s = out[f"{basis}_sigma"]
        out[f"score_{basis}"] = out[f"{basis}_absr"] / s if s else float("nan")
    return out


# --------------------------------------------------------------------------- #
# the date-level rescaling arm: sigma_tilde = sigma / L[t] ** gamma
# --------------------------------------------------------------------------- #


def date_levels(frame: pd.DataFrame) -> pd.Series:
    """`L[t]`: the cross-sectional median of `sigma` on each date.

    Derived ONCE, from the whole frame, and handed to every arm that needs it.
    That is not tidiness: a median taken over two different item sets in
    calibration and in serving is a basis difference, and the pre-registration
    makes it a void condition. One series, one universe, every arm.
    """
    return frame.groupby("date")["sigma"].median().sort_index()


def scaled_sigma(frame: pd.DataFrame, levels: pd.Series | None, gamma: float) -> np.ndarray:
    """`sigma / L[t] ** gamma`. `gamma = 0` returns `sigma` untouched."""
    sig = frame["sigma"].to_numpy(dtype=float)
    if not gamma:
        return sig
    if levels is None:
        levels = date_levels(frame)
    lv = frame["date"].map(levels).to_numpy(dtype=float)
    return sig / lv**gamma


def fit_level_elasticity(frame: pd.DataFrame, n_boot: int = N_BOOTSTRAP, seed: int = DATE_LEVEL_SEED) -> dict:
    """`b`, the elasticity of forward dispersion to the trailing level.

    OLS of `log median_i |resid[i,t]|` on `log L[t]` across dates, and
    `gamma = 1 - b`. The reading is the whole arm in one number: `b = 1` means a
    date whose trailing `sigma` is 50% high also realises 50% more forward move,
    the conformal score is already level-free, and production is right. `b = 0`
    means the level is pure noise in the denominator.

    THE BOOTSTRAP RESAMPLES DATES, not rows. The level is a per-date quantity, so
    a panel of 900 items on four dates carries four observations of it; resampling
    rows would report an interval ~30x too narrow and turn the void condition into
    a rubber stamp.
    """
    lv = date_levels(frame)
    med = frame.assign(_a=frame["resid"].abs()).groupby("date")["_a"].median()
    joined = pd.concat([lv.rename("L"), med.rename("r")], axis=1).dropna()
    joined = joined[(joined["L"] > 0) & (joined["r"] > 0)]
    x = np.log(joined["L"].to_numpy(dtype=float))
    y = np.log(joined["r"].to_numpy(dtype=float))
    b = _ols_slope(x, y)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        take = rng.integers(0, x.size, x.size)
        s = _ols_slope(x[take], y[take])
        if np.isfinite(s):
            draws.append(s)
    lo, hi = (
        (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (float("nan"), float("nan"))
    )
    return {"b": b, "gamma": 1.0 - b, "ci_lo": lo, "ci_hi": hi, "n_dates": int(x.size)}


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return float("nan")
    return float(np.dot(xc, y - y.mean()) / denom)


def anchor_coverage(
    frame: pd.DataFrame, anchors, horizon: int, gamma: float = 0.0, levels: pd.Series | None = None
) -> dict:
    """Per-anchor coverage of the pooled-`p80` band, on `sigma / L ** gamma`.

    `q_hat` comes from `conformal.calibrate` over every row anchored at or before
    `anchor - embargo_days(horizon)` -- production's expanding pooled window, with
    the H+13 embargo derived rather than re-spelled. An anchor with no calibration
    rows is absent from the result rather than reported at a `q_hat` it could not
    have had.

    `q_hat` absorbs `L ** gamma` and is therefore NOT comparable across gamma; only
    the coverage it produces is.
    """
    return {a: float(np.mean(c)) for a, c in _covered(frame, anchors, horizon, gamma, levels).items()}


def pooled_anchor_coverage(
    frame: pd.DataFrame, anchors, horizon: int, gamma: float = 0.0, levels: pd.Series | None = None
) -> float:
    """Marginal coverage over the anchor set, ROW-weighted.

    The quantity the placebo differences. Row-weighted, not a mean of the
    per-anchor rates: anchors carry unequal cohorts, and an equal-weight average
    would let the thinnest date move the bar.
    """
    cov = _covered(frame, anchors, horizon, gamma, levels)
    if not cov:
        return float("nan")
    return float(np.mean(np.concatenate(list(cov.values()))))


def _covered(frame: pd.DataFrame, anchors, horizon: int, gamma: float, levels: pd.Series | None) -> dict:
    """{anchor: boolean array over that anchor's rows}. One `q_hat` per anchor."""
    lv = levels if levels is not None else (date_levels(frame) if gamma else None)
    scale = scaled_sigma(frame, lv, gamma)
    resid = frame["resid"].abs().to_numpy(dtype=float)
    dates = frame["date"].to_numpy()
    out = {}
    for a in anchors:
        cutoff = a - datetime.timedelta(days=embargo_days(horizon))
        cal = dates <= cutoff
        on = dates == a
        if not cal.any() or not on.any():
            continue
        q = conformal.calibrate(resid[cal], scale[cal])
        out[a] = resid[on] <= q * scale[on]
    return out


def audit_eligibility(dates, horizons=DEFAULT_AUDIT_HORIZONS, archive_dir=None) -> dict:
    """`{date: {horizons that pass BOTH audits}}`, over one archive read.

    The feed audit asks how the anchor day itself was collected; the cutover
    check asks whether anything changed inside `(anchor, anchor + h]`, which the
    feed audit cannot see and which voids the label. A date failing the feed
    audit fails at every horizon — the anchor's own quote is the defect — while
    a cutover only removes the horizons whose outcome window spans it.

    One `_feed_profile` call covers the whole range rather than one per date:
    the per-day source set and item count are properties of the archive, not of
    the anchor, and 700 windowed queries would read the same days 7 times each.
    """
    ds = sorted(dates)
    if not ds:
        return {}
    span = max(int(h) for h in horizons)
    lo = ds[0] - datetime.timedelta(days=ANCHOR_NEIGHBOURHOOD_DAYS)
    hi = ds[-1] + datetime.timedelta(days=span + ANCHOR_NEIGHBOURHOOD_DAYS)
    mid = lo + (hi - lo) / 2
    profile = _feed_profile(mid, archive_dir, window=(hi - lo).days // 2 + 1)
    if profile.empty:
        return {d: set() for d in ds}
    profile = profile.assign(day=pd.to_datetime(profile["day"]).dt.date)
    counts = profile.set_index("day")["items"].sort_index()
    cutovers = cutovers_from_counts(counts)

    out = {}
    for d in ds:
        near = profile[
            (profile["day"] >= d - datetime.timedelta(days=ANCHOR_NEIGHBOURHOOD_DAYS))
            & (profile["day"] <= d + datetime.timedelta(days=ANCHOR_NEIGHBOURHOOD_DAYS))
        ]
        ok, _ = audit_anchor_feed(d, near)
        if not ok:
            out[d] = set()
            continue
        spanned = cutovers_in_outcome_window(d, horizons, cutovers)
        out[d] = {int(h) for h in horizons if int(h) not in spanned}
    return out


def select_low_level_anchors(
    levels: pd.Series,
    eligible: dict,
    rows: pd.Series,
    *,
    n: int = LOW_ANCHOR_COUNT,
    spacing_days: int = LOW_ANCHOR_SPACING_DAYS,
    quantile: float = LOW_LEVEL_QUANTILE,
    horizons=DEFAULT_AUDIT_HORIZONS,
    min_rows: int = MIN_ANCHOR_ROWS,
    candidates=None,
    not_before=None,
) -> list:
    """The lowest-`L[t]` dates the panel can referee, greedily spaced.

    **Selection sees `L[t]`, eligibility and row counts. It never sees coverage** —
    which is the whole point: the arm has only ever been observed on dates where it
    LOWERS the band, so the set that tests the other direction has to be picked on
    the level itself and fixed before a rate is computed.

    `eligible` is `{date: {horizons that pass BOTH audits}}`; a date is taken only
    if it passes at every horizon in `horizons`. That is the 2026-08-13 lesson
    applied at selection time rather than discovered afterwards — the collection
    audit and the label-voiding cutover detector disagreed about two anchors of six
    in the last set, and each was found only after the read.

    `candidates` restricts WHICH dates may be taken without moving what "low"
    MEANS: the cut is always the `quantile` of the whole `levels` series, so a
    sub-period leg is measured against the panel's own distribution and a period
    holding no calm dates returns an empty set instead of relabelling its
    quietest ordinary ones.

    Returns fewer than `n` if the frame cannot supply them, deliberately: relaxing
    the spacing or the quantile to reach a count is how a set stops being the set
    that was pre-registered.
    """
    lv = levels.dropna().sort_values()
    if lv.empty:
        return []
    cut = float(lv.quantile(quantile))
    allowed = None if candidates is None else set(candidates)
    taken: list = []
    for d, level in lv.items():
        if allowed is not None and d not in allowed:
            continue
        if not_before is not None and d < not_before:
            continue
        if len(taken) >= n:
            break
        if level > cut:
            break
        if not set(horizons) <= set(eligible.get(d, ())):
            continue
        if float(rows.get(d, 0)) < min_rows:
            continue
        if any(abs((d - t).days) < spacing_days for t in taken):
            continue
        taken.append(d)
    return taken


def shuffled_levels(levels: pd.Series, seed: int = DATE_LEVEL_SEED, n_perm: int = N_PERMUTATIONS):
    """`L[t]` with the date correspondence destroyed, `n_perm` times.

    The same marginal distribution of levels and the same rescaling arithmetic,
    so a shuffled arm that moves coverage as much as the real one has shown the
    effect to be `q_hat` re-absorbing a constant rather than a date-level channel.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        yield pd.Series(rng.permutation(levels.to_numpy()), index=levels.index)


def run_legs(frame: pd.DataFrame, horizon: int, arm_anchors) -> None:
    """The three offline legs, in the order the pre-registration fixes them.

    (0) fit `b` -> `gamma`, which can void the arm before anything is measured;
    (V) can this panel reproduce the SERVED control's per-anchor coverage;
    (P) does a date-shuffled level move coverage as much as the real one.

    Nothing here is a ship decision. A pass buys a paired `model-diagnostics.yml`
    dispatch on the real served band; the panel derives its own clip bounds and
    cohort, so its absolute coverage is a stand-in and only differences are read.
    """
    fit = fit_level_elasticity(frame)
    gamma = fit["gamma"]
    logger.info(
        "\n  (0) LEVEL ELASTICITY  b = %.3f  [%.3f, %.3f]  ->  gamma = %.3f   (%s dates)",
        fit["b"],
        fit["ci_lo"],
        fit["ci_hi"],
        gamma,
        f"{fit['n_dates']:,}",
    )
    void = []
    if not 0.0 <= fit["b"] <= 1.0:
        void.append(f"b = {fit['b']:.3f} outside [0, 1]")
    if fit["ci_lo"] <= 0.0 and fit["ci_hi"] >= 1.0:
        void.append("CI contains both 0 and 1")
    if void:
        logger.info("      ⚠️  VOID: %s — no defensible gamma, nothing dispatched.", "; ".join(void))
        return

    # (V) The panel is a stand-in. If it cannot reproduce the control it cannot
    # referee the arm, and no later leg on it means anything.
    audited = [pd.Timestamp(a).date() for a in AUDITED_ANCHORS]
    got = anchor_coverage(frame, audited, horizon)
    want = SERVED_ANCHOR_COVERAGE[horizon]
    errs = []
    logger.info("  (V) VALIDITY vs the SERVED control")
    for a in audited:
        key = a.isoformat()
        if a not in got:
            logger.info("      %s  panel has no rows (or no calibration set)", key)
            continue
        panel_pp, served_pp = got[a] * 100.0, want[key]
        errs.append(abs(panel_pp - served_pp))
        logger.info("      %s  panel %6.2f%%   served %6.2f%%   |err| %5.2fpp", key, panel_pp, served_pp, errs[-1])
    mae = float(np.mean(errs)) if errs else float("nan")
    ok_v = np.isfinite(mae) and mae <= VALIDITY_MAE_PP
    logger.info(
        "      MAE %.2fpp over %d of %d anchors — %s (bar %.1fpp)",
        mae,
        len(errs),
        len(audited),
        "PASS" if ok_v else "FAIL",
        VALIDITY_MAE_PP,
    )

    # (P) The leg that decides whether this is the refuted date-level class.
    present = [a for a in arm_anchors if a in set(frame["date"])]
    control = pooled_anchor_coverage(frame, present, horizon)
    arm = pooled_anchor_coverage(frame, present, horizon, gamma=gamma)
    real_delta = (arm - control) * 100.0
    levels = date_levels(frame)
    null = (
        np.array(
            [pooled_anchor_coverage(frame, present, horizon, gamma, perm) - control for perm in shuffled_levels(levels)]
        )
        * 100.0
    )
    logger.info(
        "  (P) PLACEBO on %d anchors  control %.2f%%  arm %.2f%%  real delta %+.2fpp",
        len(present),
        control * 100.0,
        arm * 100.0,
        real_delta,
    )
    logger.info(
        "      shuffled delta: mean %+.2fpp  |mean| %.2fpp  p50 %+.2fpp  p95 %+.2fpp  max|.| %.2fpp",
        float(np.mean(null)),
        abs(float(np.mean(null))),
        float(np.percentile(null, 50)),
        float(np.percentile(null, 95)),
        float(np.max(np.abs(null))),
    )
    ok_p = abs(float(np.mean(null))) <= PLACEBO_MAX_PP
    logger.info(
        "      placebo %s (bar |mean delta| <= %.1fpp); real effect is %.1fx the shuffled mean",
        "PASS" if ok_p else "FAIL",
        PLACEBO_MAX_PP,
        abs(real_delta) / abs(float(np.mean(null))) if np.mean(null) else float("inf"),
    )

    # The spread, which is the claim (S). Offline preview only: the bar is on the
    # served band, and these anchors are the panel's own cohort.
    per_c = anchor_coverage(frame, present, horizon)
    per_a = anchor_coverage(frame, present, horizon, gamma=gamma)
    if len(per_c) >= 2:
        sc = (max(per_c.values()) - min(per_c.values())) * 100.0
        sa = (max(per_a.values()) - min(per_a.values())) * 100.0
        logger.info(
            "  (S) across-anchor SPREAD  control %.2fpp  arm %.2fpp  (%+.0f%%, offline stand-in)",
            sc,
            sa,
            (sa - sc) / sc * 100.0 if sc else float("nan"),
        )
        for a in sorted(per_c):
            logger.info("      %s  control %6.2f%%  arm %6.2f%%", a.isoformat(), per_c[a] * 100.0, per_a[a] * 100.0)


def mean_abs_miss(per_anchor: dict) -> float:
    """Mean per-anchor `|coverage − 80|`, in pp. Bars (L2) and (J).

    Per-anchor and equal-weight, deliberately, where `pooled_anchor_coverage` is
    row-weighted: this statistic asks how far a TYPICAL DATE sits from target, so
    each date is one observation of it. The row-weighting that protects a marginal
    rate from a thin date would here let the fattest date own the answer.
    """
    if not per_anchor:
        return float("nan")
    return float(np.mean([abs(v * 100.0 - NOMINAL_COVERAGE_PP) for v in per_anchor.values()]))


def overshoot_breaches(control: dict, arm: dict, near_pp: float = 5.0, far_pp: float = 10.0) -> list:
    """Anchors the arm threw out of calibration. Bar (L3).

    A date the control already covers within `near_pp` of target that the arm
    moves beyond `far_pp` — in EITHER direction. The high-vol read overshot
    downward at this same `gamma` (77.68% pooled, 73–77% on two anchors); on a
    calm date the same arithmetic inflates a well-calibrated band instead, and a
    mean statistic would net the two out.
    """
    out = []
    for a, c in control.items():
        if a not in arm:
            continue
        c_pp, a_pp = c * 100.0, arm[a] * 100.0
        if abs(c_pp - NOMINAL_COVERAGE_PP) <= near_pp and abs(a_pp - NOMINAL_COVERAGE_PP) > far_pp:
            out.append((a, c_pp, a_pp))
    return out


def run_low_legs(frame: pd.DataFrame, horizon: int, low_anchors, high_anchors, leg_a_anchors) -> None:
    """The low-`L[t]` read, bars fixed by
    `docs/research/2026-08-13-low-level-anchor-preregistration.md`.

    Everything above measures the arm where it NARROWS the band. This measures the
    direction it has never been observed in. `gamma` is refit here rather than
    passed so the void condition can be checked against the published value.
    """
    fit = fit_level_elasticity(frame)
    gamma = fit["gamma"]
    levels = date_levels(frame)
    logger.info(
        "\nh=%-2s  gamma = %.3f  (b = %.3f [%.3f, %.3f], %s dates)%s",
        horizon,
        gamma,
        fit["b"],
        fit["ci_lo"],
        fit["ci_hi"],
        f"{fit['n_dates']:,}",
        "   ⚠️ VOIDED as a measurement (leg V, 4.53pp)" if horizon == 14 else "",
    )
    if not 0.0 <= fit["b"] <= 1.0 or (fit["ci_lo"] <= 0.0 <= 1.0 <= fit["ci_hi"]):
        logger.info("      ⚠️  VOID: b = %.3f has no defensible gamma.", fit["b"])
        return

    present = [a for a in low_anchors if a in set(frame["date"])]
    con = anchor_coverage(frame, present, horizon)
    arm = anchor_coverage(frame, present, horizon, gamma=gamma, levels=levels)
    if not con:
        logger.info("      no leg-B anchor has rows at this horizon")
        return
    for a in sorted(con):
        logger.info(
            "      %s  L %.4f (%.3fx)  control %6.2f%%  arm %6.2f%%  %+6.2fpp",
            a,
            levels.get(a, float("nan")),
            levels.get(a, float("nan")) / float(levels.median()),
            con[a] * 100.0,
            arm[a] * 100.0,
            (arm[a] - con[a]) * 100.0,
        )

    pooled_c = pooled_anchor_coverage(frame, present, horizon)
    pooled_a = pooled_anchor_coverage(frame, present, horizon, gamma, levels)
    logger.info(
        "  (L1) DIRECTION   pooled control %6.2f%%  arm %6.2f%%  %+.2fpp  — %s",
        pooled_c * 100.0,
        pooled_a * 100.0,
        (pooled_a - pooled_c) * 100.0,
        "UP" if pooled_a > pooled_c else "DOWN",
    )
    mc, ma = mean_abs_miss(con), mean_abs_miss(arm)
    logger.info("  (L2) CALIBRATION mean |cov-80| control %.2fpp  arm %.2fpp  (%+.2fpp)", mc, ma, ma - mc)
    breach = overshoot_breaches(con, arm)
    logger.info(
        "  (L3) OVERSHOOT   %s", "none" if not breach else "; ".join(f"{a} {c:.2f}% -> {v:.2f}%" for a, c, v in breach)
    )

    null = (
        np.array(
            [
                pooled_anchor_coverage(frame, present, horizon, gamma, perm) - pooled_c
                for perm in shuffled_levels(levels)
            ]
        )
        * 100.0
    )
    logger.info(
        "  (P)  PLACEBO     shuffled mean %+.2fpp  p95 %+.2fpp  max|.| %.2fpp  — %s (bar %.1fpp)",
        float(np.mean(null)),
        float(np.percentile(null, 95)),
        float(np.max(np.abs(null))),
        "PASS" if abs(float(np.mean(null))) <= PLACEBO_MAX_PP else "FAIL",
        PLACEBO_MAX_PP,
    )

    joint = sorted(set(present) | {a for a in high_anchors if a in set(frame["date"])})
    jc = anchor_coverage(frame, joint, horizon)
    ja = anchor_coverage(frame, joint, horizon, gamma=gamma, levels=levels)
    logger.info(
        "  (J)  JOINT       %d anchors  mean |cov-80| control %.2fpp  arm %.2fpp  (%+.2fpp)",
        len(jc),
        mean_abs_miss(jc),
        mean_abs_miss(ja),
        mean_abs_miss(ja) - mean_abs_miss(jc),
    )

    a_present = [a for a in leg_a_anchors if a in set(frame["date"])]
    ac = anchor_coverage(frame, a_present, horizon)
    aa = anchor_coverage(frame, a_present, horizon, gamma=gamma, levels=levels)
    for a in sorted(ac):
        logger.info(
            "  (A)  IN-REGIME   %s  control %6.2f%%  arm %6.2f%%  %+6.2fpp  — NO BAR, n=1 by construction",
            a,
            ac[a] * 100.0,
            aa[a] * 100.0,
            (aa[a] - ac[a]) * 100.0,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--voted", default=None)
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--anchors", default=",".join(AUDITED_ANCHORS))
    ap.add_argument("--arm-anchors", default=",".join(ARM_ANCHORS))
    ap.add_argument("--legs", action="store_true", help="run the date-level rescaling legs (0)/(V)/(P)/(S)")
    ap.add_argument("--low-legs", action="store_true", help="run the low-`L[t]` read (L1)/(L2)/(L3)/(P)/(J)/(A)")
    ap.add_argument("--low-anchors", default=",".join(LOW_ANCHORS))
    ap.add_argument("--leg-a-anchors", default=",".join(LEG_A_ANCHORS))
    ap.add_argument(
        "--select-low",
        action="store_true",
        help="print the low-`L[t]` anchor sets and exit, seeing no "
        "coverage — this is how the set is fixed before the read",
    )
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    anchors = [pd.Timestamp(a).date() for a in args.anchors.split(",") if a.strip()]
    arm_anchors = [pd.Timestamp(a).date() for a in args.arm_anchors.split(",") if a.strip()]
    low_anchors = [pd.Timestamp(a).date() for a in args.low_anchors.split(",") if a.strip()]
    leg_a_anchors = [pd.Timestamp(a).date() for a in args.leg_a_anchors.split(",") if a.strip()]

    panel = load_panel(args.voted or default_voted_panel())
    floor, cap = sigma_bounds_for_panel(panel)
    logger.info("sigma clip [%.4f, %.4f] from this panel's own cross-section", floor, cap)

    # `__init__`, not `__new__`: `prepare_targets` reports through
    # `self.label_voiding`, which only the constructor creates. `db_session=None`
    # is what the sibling instruments pass -- nothing here touches the DB.
    fc = ItemForecaster(db_session=None)

    # Is sigma trending? If the calibration pool spans a calmer era than the anchors,
    # "which rows q_hat saw" is a time defect and not a cohort one -- and unlike the
    # refuted trailing-window class, a secular shift is not a per-date state.
    trend = panel.copy()
    trend["sigma_raw"] = (trend["price_std_60d"] / trend["price"]).clip(floor, cap)
    trend["ym"] = pd.to_datetime(trend["date"]).dt.to_period("Q").astype(str)
    logger.info("\nmedian sigma by quarter (panel, >=$1):")
    for ym, g in trend.groupby("ym"):
        logger.info("  %s  n=%7s  median sigma %.4f", ym, f"{len(g):,}", float(np.nanmedian(g["sigma_raw"])))

    if args.select_low:
        # Scored on the SHORTEST horizon's frame, because a date must survive
        # every horizon to be selected and h=3 is the one whose row counts are
        # least reduced by label voiding -- the eligibility test is the audits,
        # not the frame.
        frame = score_frame(fc, panel, min(horizons), floor, cap)
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        lv, rows = date_levels(frame), frame.groupby("date").size()
        elig = audit_eligibility(list(lv.index), tuple(horizons))
        for label, keep in (
            ("B (full panel)", list(lv.index)),
            ("A (2026 only)", [d for d in lv.index if d.year == 2026]),
        ):
            got = select_low_level_anchors(
                lv,
                elig,
                rows,
                candidates=keep,
                horizons=tuple(horizons),
                not_before=min(lv.index) + datetime.timedelta(days=SIGMA_TRAILING_WINDOW_DAYS),
            )
            logger.info(
                "\nleg %s: %d of %d candidate dates eligible at all of %s; panel p%d of L = %.4f",
                label,
                sum(1 for d in keep if set(horizons) <= set(elig.get(d, ()))),
                len(keep),
                horizons,
                int(LOW_LEVEL_QUANTILE * 100),
                float(lv.quantile(LOW_LEVEL_QUANTILE)),
            )
            for d in got:
                logger.info(
                    "    %s  L = %.4f  (%.3fx the panel median)  n = %s",
                    d,
                    lv[d],
                    lv[d] / float(lv.median()),
                    f"{int(rows[d]):,}",
                )
            logger.info("    --arm-anchors %s", ",".join(str(d) for d in got))
        return 0

    for h in horizons:
        frame = score_frame(fc, panel, h, floor, cap)
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["absr"] = frame["resid"].abs()
        served = frame[frame["date"].isin(anchors)]
        if served.empty:
            logger.warning("h=%s: no panel rows on %s", h, anchors)
            continue

        d = decompose(frame, anchors)
        logger.info(
            "\nh=%-2s  served n=%s of %s pooled | measured served/cal WIDTH ratio %.3f",
            h,
            f"{d['n_served']:,}",
            f"{d['n_pooled']:,}",
            SERVED_WIDTH_RATIO.get(h, float("nan")),
        )
        logger.info(
            "  pooled basis (which items AND when):  sigma %.3fx   |resid| %.3fx   score |resid|/sigma %.3fx",
            d["pooled_sigma"],
            d["pooled_absr"],
            d["score_pooled"],
        )
        logger.info(
            "  same items (when only):               sigma %.3fx   |resid| %.3fx   score |resid|/sigma %.3fx",
            d["same_sigma"],
            d["same_absr"],
            d["score_same"],
        )
        for a in anchors:
            sa = served[served["date"] == a]
            if sa.empty:
                continue
            logger.info("    %s  n=%5s  sigma %.3fx pooled", a, f"{len(sa):,}", _med_ratio(sa["sigma"], frame["sigma"]))
        if args.legs:
            run_legs(frame, h, arm_anchors)
        if args.low_legs:
            run_low_legs(frame, h, low_anchors, arm_anchors, leg_a_anchors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
