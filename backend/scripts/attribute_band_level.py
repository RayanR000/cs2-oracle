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

from models import conformal  # noqa: E402
from models.forecaster import ItemForecaster, embargo_days  # noqa: E402
from scripts.measure_conditional_qhat import (  # noqa: E402
    default_voted_panel,
    load_panel,
    score_frame,
    sigma_bounds_for_panel,
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
VALIDITY_MAE_PP = 3.0     # leg (V) bar
PLACEBO_MAX_PP = 1.0      # leg (P) bar

# The four anchors that pass `replay_serving.audit_anchor_feed`, i.e. the ones run
# `31657639707` served the 9.80 / 14.75 / 21.22 / 31.32% widths on.
AUDITED_ANCHORS = ["2026-04-22", "2026-05-16", "2026-06-16", "2026-07-06"]
# The served/calibration half-width ratios that run measured, per horizon. Printed
# beside this script's sigma ratio so the two can be compared without a second lookup.
SERVED_WIDTH_RATIO = {3: 1.529, 7: 1.548, 14: 1.553, 30: 1.520}
# The arm's anchor set, fixed by the pre-registration: six, not four, because the
# spread and dose-response bars are about variation ACROSS dates and four points
# cannot carry them. All six pass `replay_serving.audit_anchor_feed`.
ARM_ANCHORS = ["2026-02-14", "2026-03-10", "2026-04-06", "2026-04-22",
               "2026-06-16", "2026-07-06"]


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
        "n_served": int(len(served)),
        "n_pooled": int(len(frame)),
        "pooled_sigma": _med_ratio(served["sigma"], frame["sigma"]),
        "pooled_absr": _med_ratio(served["absr"], frame["absr"]),
        "same_sigma": float(np.nanmedian(
            served["sigma"] / served["item_id"].map(own_sigma))),
        "same_absr": float(np.nanmedian(
            served["absr"] / served["item_id"].map(own_absr))),
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


def scaled_sigma(frame: pd.DataFrame, levels: pd.Series | None,
                 gamma: float) -> np.ndarray:
    """`sigma / L[t] ** gamma`. `gamma = 0` returns `sigma` untouched."""
    sig = frame["sigma"].to_numpy(dtype=float)
    if not gamma:
        return sig
    if levels is None:
        levels = date_levels(frame)
    lv = frame["date"].map(levels).to_numpy(dtype=float)
    return sig / lv ** gamma


def fit_level_elasticity(frame: pd.DataFrame, n_boot: int = N_BOOTSTRAP,
                         seed: int = DATE_LEVEL_SEED) -> dict:
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
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) \
        if draws else (float("nan"), float("nan"))
    return {"b": b, "gamma": 1.0 - b, "ci_lo": lo, "ci_hi": hi,
            "n_dates": int(x.size)}


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return float("nan")
    return float(np.dot(xc, y - y.mean()) / denom)


def anchor_coverage(frame: pd.DataFrame, anchors, horizon: int,
                    gamma: float = 0.0,
                    levels: pd.Series | None = None) -> dict:
    """Per-anchor coverage of the pooled-`p80` band, on `sigma / L ** gamma`.

    `q_hat` comes from `conformal.calibrate` over every row anchored at or before
    `anchor - embargo_days(horizon)` -- production's expanding pooled window, with
    the H+13 embargo derived rather than re-spelled. An anchor with no calibration
    rows is absent from the result rather than reported at a `q_hat` it could not
    have had.

    `q_hat` absorbs `L ** gamma` and is therefore NOT comparable across gamma; only
    the coverage it produces is.
    """
    return {a: float(np.mean(c))
            for a, c in _covered(frame, anchors, horizon, gamma, levels).items()}


def pooled_anchor_coverage(frame: pd.DataFrame, anchors, horizon: int,
                           gamma: float = 0.0,
                           levels: pd.Series | None = None) -> float:
    """Marginal coverage over the anchor set, ROW-weighted.

    The quantity the placebo differences. Row-weighted, not a mean of the
    per-anchor rates: anchors carry unequal cohorts, and an equal-weight average
    would let the thinnest date move the bar.
    """
    cov = _covered(frame, anchors, horizon, gamma, levels)
    if not cov:
        return float("nan")
    return float(np.mean(np.concatenate(list(cov.values()))))


def _covered(frame: pd.DataFrame, anchors, horizon: int, gamma: float,
             levels: pd.Series | None) -> dict:
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


def shuffled_levels(levels: pd.Series, seed: int = DATE_LEVEL_SEED,
                    n_perm: int = N_PERMUTATIONS):
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
    logger.info("\n  (0) LEVEL ELASTICITY  b = %.3f  [%.3f, %.3f]  ->  gamma = %.3f"
                "   (%s dates)", fit["b"], fit["ci_lo"], fit["ci_hi"], gamma,
                f"{fit['n_dates']:,}")
    void = []
    if not 0.0 <= fit["b"] <= 1.0:
        void.append(f"b = {fit['b']:.3f} outside [0, 1]")
    if fit["ci_lo"] <= 0.0 and fit["ci_hi"] >= 1.0:
        void.append("CI contains both 0 and 1")
    if void:
        logger.info("      ⚠️  VOID: %s — no defensible gamma, nothing dispatched.",
                    "; ".join(void))
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
        logger.info("      %s  panel %6.2f%%   served %6.2f%%   |err| %5.2fpp",
                    key, panel_pp, served_pp, errs[-1])
    mae = float(np.mean(errs)) if errs else float("nan")
    ok_v = np.isfinite(mae) and mae <= VALIDITY_MAE_PP
    logger.info("      MAE %.2fpp over %d of %d anchors — %s (bar %.1fpp)",
                mae, len(errs), len(audited), "PASS" if ok_v else "FAIL",
                VALIDITY_MAE_PP)

    # (P) The leg that decides whether this is the refuted date-level class.
    present = [a for a in arm_anchors if a in set(frame["date"])]
    control = pooled_anchor_coverage(frame, present, horizon)
    arm = pooled_anchor_coverage(frame, present, horizon, gamma=gamma)
    real_delta = (arm - control) * 100.0
    levels = date_levels(frame)
    null = np.array([
        pooled_anchor_coverage(frame, present, horizon, gamma, perm) - control
        for perm in shuffled_levels(levels)]) * 100.0
    logger.info("  (P) PLACEBO on %d anchors  control %.2f%%  arm %.2f%%  "
                "real delta %+.2fpp", len(present), control * 100.0,
                arm * 100.0, real_delta)
    logger.info("      shuffled delta: mean %+.2fpp  |mean| %.2fpp  "
                "p50 %+.2fpp  p95 %+.2fpp  max|.| %.2fpp",
                float(np.mean(null)), abs(float(np.mean(null))),
                float(np.percentile(null, 50)), float(np.percentile(null, 95)),
                float(np.max(np.abs(null))))
    ok_p = abs(float(np.mean(null))) <= PLACEBO_MAX_PP
    logger.info("      placebo %s (bar |mean delta| <= %.1fpp); real effect is "
                "%.1fx the shuffled mean", "PASS" if ok_p else "FAIL",
                PLACEBO_MAX_PP,
                abs(real_delta) / abs(float(np.mean(null)))
                if np.mean(null) else float("inf"))

    # The spread, which is the claim (S). Offline preview only: the bar is on the
    # served band, and these anchors are the panel's own cohort.
    per_c = anchor_coverage(frame, present, horizon)
    per_a = anchor_coverage(frame, present, horizon, gamma=gamma)
    if len(per_c) >= 2:
        sc = (max(per_c.values()) - min(per_c.values())) * 100.0
        sa = (max(per_a.values()) - min(per_a.values())) * 100.0
        logger.info("  (S) across-anchor SPREAD  control %.2fpp  arm %.2fpp  "
                    "(%+.0f%%, offline stand-in)", sc, sa,
                    (sa - sc) / sc * 100.0 if sc else float("nan"))
        for a in sorted(per_c):
            logger.info("      %s  control %6.2f%%  arm %6.2f%%",
                        a.isoformat(), per_c[a] * 100.0, per_a[a] * 100.0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--voted", default=None)
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--anchors", default=",".join(AUDITED_ANCHORS))
    ap.add_argument("--arm-anchors", default=",".join(ARM_ANCHORS))
    ap.add_argument("--legs", action="store_true",
                    help="run the date-level rescaling legs (0)/(V)/(P)/(S)")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    anchors = [pd.Timestamp(a).date() for a in args.anchors.split(",") if a.strip()]
    arm_anchors = [pd.Timestamp(a).date()
                   for a in args.arm_anchors.split(",") if a.strip()]

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
        logger.info("  %s  n=%7s  median sigma %.4f", ym, f"{len(g):,}",
                    float(np.nanmedian(g["sigma_raw"])))

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
            h, f"{d['n_served']:,}", f"{d['n_pooled']:,}",
            SERVED_WIDTH_RATIO.get(h, float("nan")))
        logger.info("  pooled basis (which items AND when):  sigma %.3fx   "
                    "|resid| %.3fx   score |resid|/sigma %.3fx",
                    d["pooled_sigma"], d["pooled_absr"], d["score_pooled"])
        logger.info("  same items (when only):               sigma %.3fx   "
                    "|resid| %.3fx   score |resid|/sigma %.3fx",
                    d["same_sigma"], d["same_absr"], d["score_same"])
        for a in anchors:
            sa = served[served["date"] == a]
            if sa.empty:
                continue
            logger.info("    %s  n=%5s  sigma %.3fx pooled", a, f"{len(sa):,}",
                        _med_ratio(sa["sigma"], frame["sigma"]))
        if args.legs:
            run_legs(frame, h, arm_anchors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
