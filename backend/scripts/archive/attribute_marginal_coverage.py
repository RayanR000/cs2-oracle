"""How much of the band's marginal over-coverage does the `sigma`-mix shift buy?

Pre-registered in `docs/research/2026-08-12-marginal-coverage-attribution-preregistration.md`.
Read that first: the bar, the placebo and the void conditions are fixed there and
this script only reports the numbers they apply to.

THE MECHANISM, IN ONE LINE. Conditional coverage is a monotone increasing
function of `sigma` (62->93 / 60->95 / 57->96 / 52->97% across deciles, on real
OOF residuals, `changelog/2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md`).
`q_hat` makes that curve integrate to 80% over the CALIBRATION `sigma`
distribution. Serving on a DIFFERENT `sigma` distribution integrates the same
curve to something else, with no second defect required. Production over-covers
at 87.2/91.8/90.6/89.0% and served `sigma` is on record at 1.28/1.34/1.33/0.93x
the calibration median, so the question is how many of those pp the shift buys.

WHY IT IS OFFLINE, AND WHY THAT IS NOW ESTABLISHED. `|actual| / sigma` is the
conformal score to first order (predicted |return| is median 0.95% against
half-widths of 10-31%), and the model-free instrument that argument justifies was
validated against real OOF residuals to 0.014-0.032 on the elasticity. Same
argument, same panel machinery, imported from `measure_conditional_qhat`.

WHAT THIS FIXES ABOUT THE INPUT. The published 1.28/1.34/1.33/0.93x implied
served `sigma` as `half_pct / q_hat` -- the same reconstruction whose elasticity
was refuted at 4 of 4. Here the served side is measured DIRECTLY, from
`price_std_60d / price` through `conformal.sigma_from_columns`, on a panel that
spans production's actual forecast anchors, with ONE clip applied to both sides.

Read-only: reads voted price panels and writes a CSV. No DB, no artifacts.

    venv/bin/python -m scripts.archive.attribute_marginal_coverage --horizons 3,7,14,30
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import conformal
from models.forecaster import ItemForecaster
from scripts.archive.measure_conditional_qhat import (
    default_voted_panel,
    load_panel,
    score_frame,
    sigma_bounds_for_panel,
)

logger = logging.getLogger("attribute_marginal_coverage")

TARGET = conformal.NOMINAL_COVERAGE

# Production, from `changelog/2026-08-12-conformal-basis-follows-serving.md`.
# The denominators of the attribution fraction; not measured here.
OBSERVED_MARGINAL = {3: 0.872, 7: 0.918, 14: 0.906, 30: 0.890}
# The published served/calibration `sigma` median ratio, implied as half_pct/q_hat
# there. Reported beside the directly measured one, never instead of it.
PUBLISHED_K = {3: 1.28, 7: 1.34, 14: 1.33, 30: 0.93}

N_DECILES = 10
MIN_DATES = 100  # void condition
MIN_ROWS_PER_DATE = 100
VALIDITY_BAR_PP = 2.0
RECENT_DATES_FOR_SERVED = 10  # the served-side proxy window


# --------------------------------------------------------------------------- #
# the coverage-vs-sigma curve
# --------------------------------------------------------------------------- #


class CoverageCurve:
    """`c(sigma)` = P(covered | sigma), as a location-scale model on the logs.

    `log|resid| = a + beta*log sigma + eps`, so a row is covered iff
    `|resid|/sigma <= q_hat`, i.e. iff `eps <= log q_hat - a + (1-beta)*log sigma`.
    With `beta < 1` the threshold RISES with sigma, which is the tilt.

    Parametric in the mean, EMPIRICAL in `eps` -- the residual CDF is the sorted
    sample, not a normal. That matters: these residuals are heavy-tailed, and a
    normal would misprice exactly the 80th percentile the whole question is about.

    A model rather than the empirical bins alone because the shift legs evaluate
    `c` at sigmas OUTSIDE the calibration support (a 1.3x shift pushes the top
    decile past anything observed). The bins are still computed, and the
    pre-registered validity bar is that the model reproduces them.
    """

    def __init__(self, resid: np.ndarray, sigma: np.ndarray, q_hat: float, beta: float | None = None):
        r = np.abs(resid)
        ok = (r > 0) & (sigma > 0) & np.isfinite(r) & np.isfinite(sigma)
        x = np.log(sigma[ok])
        y = np.log(r[ok])
        if beta is None:
            xc = x - x.mean()
            beta = float(np.dot(xc, y - y.mean()) / float(np.dot(xc, xc)))
        # `a` is refitted at the given beta, so the beta=1 placebo is the same
        # model under a different exponent and not a differently centred one.
        a = float(y.mean() - beta * x.mean())
        self.beta = float(beta)
        self.a = a
        self.log_q = float(np.log(q_hat))
        self.eps = np.sort(y - a - beta * x)

    def __call__(self, sigma: np.ndarray) -> np.ndarray:
        t = self.log_q - self.a + (1.0 - self.beta) * np.log(sigma)
        # Empirical CDF of eps at t, vectorised.
        return np.searchsorted(self.eps, t, side="right") / self.eps.size

    def marginal(self, sigma: np.ndarray) -> float:
        return float(np.mean(self(sigma)))


def decile_index(sigma: np.ndarray) -> np.ndarray:
    edges = np.quantile(sigma, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    return np.searchsorted(edges, sigma, side="right")


def empirical_decile_coverage(resid: np.ndarray, sigma: np.ndarray, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    """Realised coverage per sigma decile, and the decile index.

    The modelled counterpart must be averaged over the SAME rows, not evaluated
    at the decile's median sigma: `c` is steep inside the extreme deciles, so a
    point evaluation carries a Jensen gap against a decile mean and would charge
    the model for a defect it does not have. Measured at ~3pp in the bottom
    decile on synthetic data with a known elasticity, which is what
    `tests/test_marginal_coverage_attribution.py` pins.
    """
    idx = decile_index(sigma)
    covered = np.abs(resid) / sigma <= q_hat
    cov = np.array([covered[idx == k].mean() for k in range(N_DECILES)])
    return cov, idx


def shifted_sigma(sigma: np.ndarray, k: float, floor: float, cap: float) -> np.ndarray:
    """`k * sigma`, re-clipped exactly as serving clips it.

    The clip is not cosmetic here: it is the only thing that bounds the shift's
    effect, and at large `k` it is what the answer is made of. Its bite is
    reported beside every shift figure.
    """
    return np.clip(sigma * k, floor, cap)


def solve_k(curve: CoverageCurve, sigma: np.ndarray, target: float, floor: float, cap: float) -> float:
    """The multiplicative sigma shift whose predicted marginal coverage is `target`.

    Monotone in `k` while `beta < 1`, so bisection is exact to tolerance. Returns
    NaN when the target is unreachable on [1e-3, 1e6] -- which is itself the
    answer to the pre-registered inverse leg, not an error.
    """
    lo, hi = 1e-3, 1e6
    if curve.marginal(shifted_sigma(sigma, hi, floor, cap)) < target:
        return float("nan")
    if curve.marginal(shifted_sigma(sigma, lo, floor, cap)) > target:
        return float("nan")
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if curve.marginal(shifted_sigma(sigma, mid, floor, cap)) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# --------------------------------------------------------------------------- #
# the served side, measured directly
# --------------------------------------------------------------------------- #


def served_panel_path(explicit: str | None) -> str | None:
    """The voted cache whose LAST date is newest -- the served-side proxy.

    Deliberately the opposite selection rule to `default_voted_panel`, which
    picks the longest span for the calibration side. The two are different
    questions: calibration wants depth, the served `sigma` mix wants recency,
    and in this repo they are different files (731 dates to 2026-07-09 beside
    283 dates to 2026-08-08).
    """
    if explicit:
        return explicit
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = glob.glob(os.path.join(here, "data", "voted_*.parquet"))
    if not found:
        return None
    ends = {}
    for f in found:
        d = pd.read_parquet(f, columns=["date"])["date"]
        ends[f] = pd.Timestamp(d.max())
    return max(ends, key=ends.get)


def sigma_panel(path: str, floor: float, cap: float) -> pd.DataFrame:
    """`(item_id, date, sigma)` for one panel's >=$1 cohort."""
    df = load_panel(path)
    return df.assign(sigma=conformal.sigma_from_columns(df["price_std_60d"], df["price"], floor, cap))[
        ["item_id", "date", "sigma"]
    ]


def cache_agreement(cal: pd.DataFrame, srv: pd.DataFrame) -> None:
    """Do the two voted caches agree on the dates and items they SHARE?

    THE CONTROL THE DIRECT LEG NEEDS. The served side comes from a different
    cache than the calibration side, so an elevated served `sigma` could be a
    property of that file -- a different feed mix, a different span for the
    60-ROW rolling std -- rather than of the market. If the two agree on the
    overlap, the elevation at the served dates is real.

    Restricted to shared items as well as shared dates: the >=$1 median-price
    filter is applied within each panel's own span, so the shorter recent panel
    admits MORE items (946 against 1068), and a different item set moves the
    median `sigma` on its own.
    """
    items = set(cal["item_id"]) & set(srv["item_id"])
    dates = set(cal["date"]) & set(srv["date"])
    if not dates:
        logger.warning("  caches share no dates; the direct leg is uncontrolled")
        return
    a = cal[cal["item_id"].isin(items) & cal["date"].isin(dates)]
    b = srv[srv["item_id"].isin(items) & srv["date"].isin(dates)]
    ma = a.groupby("date")["sigma"].median()
    mb = b.groupby("date")["sigma"].median()
    ratio = (mb / ma).dropna()
    logger.info(
        "  cache agreement on the overlap: %d shared dates, %d shared items, %s -> %s",
        len(ratio),
        len(items),
        min(dates),
        max(dates),
    )
    logger.info(
        "  served/calibration median sigma on SHARED rows: p05 %.3f  p50 %.3f  p95 %.3f  (1.000 = the caches agree)",
        ratio.quantile(0.05),
        ratio.median(),
        ratio.quantile(0.95),
    )
    # WHERE they disagree decides whether the direct leg is usable. The rolling
    # std is over 60 ROWS, so the shorter cache's opening dates have a shallower
    # window and a mechanically smaller sigma. That contaminates its EARLY dates
    # and not the recent ones the served read uses -- but only if the divergence
    # is in fact confined to the opening. Printed so that is checked, not assumed.
    bad = ratio[ratio < 0.95]
    if len(bad):
        logger.info(
            "  disagreement (<0.95) on %d of %d shared dates, %s -> %s; "
            "first shared date %.3f rising to %.3f -- the 60-ROW window "
            "filling in the shorter cache",
            len(bad),
            len(ratio),
            bad.index.min(),
            bad.index.max(),
            ratio.iloc[0],
            ratio.iloc[-1],
        )


def served_sigma_by_date(df: pd.DataFrame, n_dates: int, cal_items: set) -> pd.DataFrame:
    """Per-anchor-date `sigma` distribution on the most recent dates.

    NOTE `sigma` CARRIES NO HORIZON TERM. It is `price_std_60d / price` at the
    anchor, so one anchor date has ONE sigma distribution for all four horizons.
    The four published ratios can therefore only differ because the served row
    SETS differ per horizon -- production has 7/6/3/1 forecast dates at
    3/7/14/30d, so the 30d ratio rests on a single date's mix.

    `sigma_common` is the same date restricted to the calibration panel's own
    items, so the headline ratio can be read free of cohort composition.
    """
    dates = sorted(df["date"].unique())[-n_dates:]
    rows = []
    for d in dates:
        sl = df[df["date"] == d]
        s = sl["sigma"].to_numpy(dtype=float)
        s = s[np.isfinite(s) & (s > 0)]
        if s.size < MIN_ROWS_PER_DATE:
            continue
        c = sl.loc[sl["item_id"].isin(cal_items), "sigma"].to_numpy(dtype=float)
        c = c[np.isfinite(c) & (c > 0)]
        rows.append(
            {
                "date": pd.Timestamp(d),
                "n": s.size,
                "median_sigma": float(np.median(s)),
                "n_common": c.size,
                "median_sigma_common": float(np.median(c)) if c.size else np.nan,
                "sigma": s,
                "sigma_common": c,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voted", default=None, help="calibration-side panel (default: longest span)")
    ap.add_argument("--served", default=None, help="served-side panel (default: newest last date)")
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--out", default="marginal_coverage_attribution.csv")
    args = ap.parse_args()

    cal_path = args.voted or default_voted_panel()
    logger.info("CALIBRATION panel")
    cal = load_panel(cal_path)
    floor, cap = sigma_bounds_for_panel(cal)
    logger.info(
        "sigma clip: floor=%.6f cap=%.6f (this panel's own, applied to BOTH sides -- read ratios, not levels)",
        floor,
        cap,
    )

    srv_path = served_panel_path(args.served)
    served = pd.DataFrame()
    if srv_path and os.path.abspath(srv_path) != os.path.abspath(cal_path):
        logger.info("")
        logger.info("SERVED panel (direct sigma, not half_pct/q_hat)")
        srv = sigma_panel(srv_path, floor, cap)
        cal_sig = cal.assign(sigma=conformal.sigma_from_columns(cal["price_std_60d"], cal["price"], floor, cap))
        cache_agreement(cal_sig[["item_id", "date", "sigma"]], srv)
        served = served_sigma_by_date(srv, RECENT_DATES_FOR_SERVED, set(cal["item_id"]))
    else:
        logger.warning("no distinct served-side panel; the direct leg is skipped")

    fc = ItemForecaster(db_session=None)
    rows: list[dict] = []
    horizons = [int(x) for x in args.horizons.split(",")]

    for h in horizons:
        sc = score_frame(fc, cal, h, floor, cap)
        resid = sc["resid"].to_numpy(dtype=float)
        sigma = sc["sigma"].to_numpy(dtype=float)
        dates = pd.to_datetime(sc["date"]).to_numpy()
        n_dates = len(np.unique(dates))
        logger.info("")
        logger.info("=== h=%dd: %s rows, %d anchor dates ===", h, f"{len(sc):,}", n_dates)
        if n_dates < MIN_DATES:
            logger.warning("  VOID: %d dates < %d required", n_dates, MIN_DATES)
            continue

        q_hat = conformal.calibrate(resid, sigma)
        curve = CoverageCurve(resid, sigma, q_hat)
        flat = CoverageCurve(resid, sigma, q_hat, beta=1.0)  # the placebo
        cal_med = float(np.median(sigma))
        logger.info("  q_hat=%.2f  beta=%.3f  median sigma=%.4f", q_hat, curve.beta, cal_med)

        # ---- V: does the model reproduce the bins it is fitted on? ----------
        emp_cov, dec = empirical_decile_coverage(resid, sigma, q_hat)
        c_all = curve(sigma)
        mod_cov = np.array([c_all[dec == k].mean() for k in range(N_DECILES)])
        v_err = float(np.mean(np.abs(mod_cov - emp_cov))) * 100.0
        logger.info("  V empirical decile cov  %s", " ".join(f"{v * 100:.0f}" for v in emp_cov))
        logger.info(
            "  V modelled  decile cov  %s   MAE=%.2fpp  %s",
            " ".join(f"{v * 100:.0f}" for v in mod_cov),
            v_err,
            "PASS" if v_err <= VALIDITY_BAR_PP else "FAIL",
        )
        base = curve.marginal(sigma)
        logger.info("  model marginal on calibration sigma: %.2f%% (construction: 80%%)", base * 100)

        # ---- A: the published shift, and the placebo -------------------------
        obs = OBSERVED_MARGINAL[h]
        k_pub = PUBLISHED_K[h]
        pred_pub = curve.marginal(shifted_sigma(sigma, k_pub, floor, cap))
        flat_pub = flat.marginal(shifted_sigma(sigma, k_pub, floor, cap))
        flat_move = abs(flat_pub - flat.marginal(sigma)) * 100.0
        a_pub = ((pred_pub - TARGET) / (obs - TARGET)) if obs != TARGET else float("nan")
        logger.info(
            "  A published k=%.2f -> predicted marginal %.2f%%  (+%.2fpp of the +%.2fpp excess, A=%.3f)",
            k_pub,
            pred_pub * 100,
            (pred_pub - TARGET) * 100,
            (obs - TARGET) * 100,
            a_pub,
        )
        logger.info(
            "  P placebo beta=1.0 at the same k moves marginal by %.4fpp  %s",
            flat_move,
            "PASS" if flat_move <= 0.05 else "VOID",
        )

        # ---- K: what shift would the whole excess need? ---------------------
        k_star = solve_k(curve, sigma, obs, floor, cap)
        clipped_at_star = float(np.mean(sigma * k_star >= cap)) * 100.0 if np.isfinite(k_star) else float("nan")
        logger.info(
            "  K k* for the FULL excess (%.1f%%): %s   (rows at the cap under k*: %s)",
            obs * 100,
            f"{k_star:.2f}x" if np.isfinite(k_star) else "UNREACHABLE",
            f"{clipped_at_star:.1f}%" if np.isfinite(k_star) else "n/a",
        )

        # The shift grid, so the curve's whole shape is on the record.
        grid = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0]
        logger.info(
            "  shift grid  %s",
            "  ".join(f"{k:g}x={curve.marginal(shifted_sigma(sigma, k, floor, cap)) * 100:.1f}%" for k in grid),
        )

        # ---- F: does the sigma mix predict realised per-date coverage? -------
        covered = np.abs(resid) / sigma <= q_hat
        by_date = pd.DataFrame({"date": dates, "covered": covered, "pred": c_all, "sigma": sigma}).groupby("date")
        agg = by_date.agg(
            n=("covered", "size"),
            realised=("covered", "mean"),
            predicted=("pred", "mean"),
            med_sigma=("sigma", "median"),
        )
        agg = agg[agg["n"] >= MIN_ROWS_PER_DATE]
        f_corr = float(agg["realised"].corr(agg["predicted"]))
        logger.info(
            "  F per-date: realised cov %.1f-%.1f%% (sd %.1fpp), "
            "sigma-mix predicted %.1f-%.1f%% (sd %.1fpp), corr=%+.3f  %s",
            agg["realised"].min() * 100,
            agg["realised"].max() * 100,
            agg["realised"].std() * 100,
            agg["predicted"].min() * 100,
            agg["predicted"].max() * 100,
            agg["predicted"].std() * 100,
            f_corr,
            "PASS" if f_corr > 0 else "fail",
        )

        # The natural range of k across the panel's own dates -- the ceiling the
        # inverse leg is compared against.
        k_dates = agg["med_sigma"] / cal_med
        logger.info(
            "  K natural per-date sigma-median ratio: min %.2f  p05 %.2f  p50 %.2f  p95 %.2f  MAX %.2f",
            k_dates.min(),
            k_dates.quantile(0.05),
            k_dates.median(),
            k_dates.quantile(0.95),
            k_dates.max(),
        )

        # THE ARTIFACT CONTROL for the direct leg. The served side comes from a
        # DIFFERENT voted cache, so an elevated served ratio could be a property
        # of that file rather than of the market. The calibration panel's OWN
        # most recent dates are measured on the same rows as everything above; if
        # they are already elevated, the ratio is continuous across the two
        # caches and the shift is real.
        tail_k = k_dates.tail(RECENT_DATES_FOR_SERVED)
        logger.info(
            "  K calibration panel's own last %d dates: ratio %.2f-%.2f "
            "(ends %s) -- artifact control for the direct leg",
            len(tail_k),
            tail_k.min(),
            tail_k.max(),
            pd.Timestamp(agg.index.max()).date(),
        )

        # ---- the served side, measured directly -----------------------------
        k_direct = float("nan")
        pred_direct = float("nan")
        if not served.empty:
            last = served.iloc[-1]
            k_direct = last["median_sigma"] / cal_med
            pred_direct = curve.marginal(np.clip(last["sigma"], floor, cap))
            ratios = served["median_sigma"] / cal_med
            logger.info(
                "  D direct served sigma on %s (n=%d): median %.4f "
                "= %.2fx calibration  -> predicted marginal %.2f%%  "
                "(A=%.3f)",
                last["date"].date(),
                int(last["n"]),
                last["median_sigma"],
                k_direct,
                pred_direct * 100,
                (pred_direct - TARGET) / (obs - TARGET),
            )
            logger.info(
                "  D last %d served dates: ratio %.2f-%.2f, predicted marginal %.1f-%.1f%%",
                len(served),
                ratios.min(),
                ratios.max(),
                min(curve.marginal(np.clip(s, floor, cap)) for s in served["sigma"]) * 100,
                max(curve.marginal(np.clip(s, floor, cap)) for s in served["sigma"]) * 100,
            )
            if last["n_common"]:
                k_common = last["median_sigma_common"] / cal_med
                pred_common = curve.marginal(np.clip(last["sigma_common"], floor, cap))
                logger.info(
                    "  D same date, calibration items only (n=%d): "
                    "%.2fx -> %.2f%% (A=%.3f) -- cohort composition "
                    "removed",
                    int(last["n_common"]),
                    k_common,
                    pred_common * 100,
                    (pred_common - TARGET) / (obs - TARGET),
                )

        rows.append(
            {
                "horizon": h,
                "n_rows": len(sc),
                "n_dates": n_dates,
                "q_hat": q_hat,
                "beta": curve.beta,
                "cal_median_sigma": cal_med,
                "V_decile_mae_pp": v_err,
                "model_marginal_on_cal": base,
                "observed_marginal": obs,
                "published_k": k_pub,
                "A_pred_marginal_published": pred_pub,
                "A_fraction_published": a_pub,
                "P_placebo_move_pp": flat_move,
                "K_k_star": k_star,
                "K_natural_k_max": float(k_dates.max()),
                "F_corr_date": f_corr,
                "D_k_direct": k_direct,
                "D_pred_marginal_direct": pred_direct,
                "D_A_fraction_direct": (
                    (pred_direct - TARGET) / (obs - TARGET) if np.isfinite(pred_direct) else float("nan")
                ),
            }
        )

    if not rows:
        logger.error("no results")
        return 1
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    logger.info("")
    logger.info("PRE-REGISTERED VERDICT (docs/research/2026-08-12-marginal-coverage-attribution-preregistration.md)")
    logger.info(
        "  V validity     PASS at %d/%d horizons (MAE <= %.0fpp)",
        int((out["V_decile_mae_pp"] <= VALIDITY_BAR_PP).sum()),
        len(out),
        VALIDITY_BAR_PP,
    )
    logger.info(
        "  P placebo      max move %.4fpp -> %s",
        out["P_placebo_move_pp"].max(),
        "PASS" if out["P_placebo_move_pp"].max() <= 0.05 else "VOID",
    )
    logger.info("  F footprint    positive at %d/%d horizons", int((out["F_corr_date"] > 0).sum()), len(out))
    for col, label in (("A_fraction_published", "published k"), ("D_A_fraction_direct", "direct k")):
        a = out[col]
        n_mat = int((a >= 0.50).sum())
        n_part = int(((a >= 0.20) & (a < 0.50)).sum())
        verdict = "MATERIAL" if n_mat >= 3 else "PARTIAL" if n_mat + n_part >= 3 else "REFUTED as a primary cause"
        logger.info(
            "  A on %-11s A = %s -> material %d/4, partial %d/4 -> %s",
            label,
            " ".join(f"{v:+.2f}" for v in a),
            n_mat,
            n_part,
            verdict,
        )
    logger.info("")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
