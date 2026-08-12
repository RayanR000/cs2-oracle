"""Which `sigma` scale should the band divide by? Walk-forward, four candidates.

The tilt is confirmed (`changelog/2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md`):
coverage ramps 62->93 / 60->95 / 57->96 / 52->97% across `sigma` deciles because
`d log|resid| / d log sigma` is 0.43/0.37/0.35/0.31, not the 1.0 that dividing by
`sigma ** 1` assumes. Held out, a single global exponent removed 76%/62% of the
conditional error at 3d/7d but only 26%/12% at 14d/30d.

THIS SCRIPT EXISTS TO SETTLE WHY, because the two reasons need different fixes:

  (a) `beta` is estimated noisily and drifts between windows  -> SHRINK it toward 1
  (b) the log-log line is the wrong SHAPE                     -> drop the parametric
                                                                 form entirely

Both are testable. (a) predicts that `beta` refitted on separate time blocks moves
a lot relative to its distance from 1, and that shrinking helps. (b) predicts that
`beta` is stable while a scheme free to bend beats it anyway. The per-block `beta`
table and the two non-parametric arms answer them directly.

THE SELECTION RULE IS FIXED HERE, BEFORE THE RUN, and applied by `main`:

    Fit and select on the FIRST 2/3 of test dates. Report on the LAST 1/3, which
    no selection touched. Among arms with |marginal - 80%| <= 2pp on the selection
    period, take the lowest sigma-decile error; then prefer the SIMPLER arm --
    ordered production < global beta < shrunk beta < binned scale < per-bin q_hat
    -- whenever it is within 1pp of the best. Simplicity is the tiebreak because
    every arm past the first adds a persisted artifact field and a serving path.

Nothing here reimplements production: targets are `prepare_targets`', `sigma` is
`conformal.sigma_from_columns`, every `q_hat` is `conformal.calibrate` including
its finite-sample level, and the panel machinery is imported from
`measure_conditional_qhat`, whose model-free approximation was validated against
real OOF residuals to 0.03 on the elasticity.

REFITS ARE EVERY 14 DAYS, not every date. That is production's retrain cadence
(age-based, 14d -- `.claude/rules/training-budget.md`), so a scheme is scored the
way it would actually be served, on a scale fitted from stale history rather than
one refreshed daily. It is also what keeps this a seconds-per-arm read.

Read-only: reads a voted panel and writes a CSV. No DB, no artifacts.

    venv/bin/python -m scripts.design_sigma_scale --horizons 3,7,14,30
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import conformal  # noqa: E402
from models.forecaster import ItemForecaster, embargo_days  # noqa: E402
from scripts.measure_conditional_qhat import (  # noqa: E402
    MIN_DATES_PER_HORIZON,
    MIN_HISTORY_DAYS,
    MIN_ROWS_PER_DATE,
    TRAILING_MIN_ROWS,
    default_voted_panel,
    load_panel,
    score_frame,
    sigma_bounds_for_panel,
)

logger = logging.getLogger("design_sigma_scale")

TARGET = conformal.NOMINAL_COVERAGE
REFIT_EVERY_DAYS = 14        # production's age-based retrain cadence
N_DECILES = 10
N_BLOCKS = 6                 # time blocks for the beta-stability read
N_SCALE_BINS = 20            # sigma bins for the binned-scale arm
SELECT_FRACTION = 2.0 / 3.0  # fit/select here, report on the remainder
MARGINAL_TOLERANCE = 0.02    # |marginal - 80%| gate on the selection period
SIMPLICITY_SLACK_PP = 1.0    # prefer a simpler arm within this of the best

# Simplest first. The tiebreak order, and also the order they are reported in.
ARM_ORDER = ["P0_production", "P1_global_beta", "P2_shrunk_beta",
             "P3_binned_scale", "P4_per_bin_qhat"]

WidthFn = Callable[[np.ndarray], np.ndarray]
Fitter = Callable[[np.ndarray, np.ndarray, np.ndarray], WidthFn]
"""(resid, sigma, dates) on embargoed history -> (sigma -> half-width)."""


# --------------------------------------------------------------------------- #
# the four candidates, plus production
# --------------------------------------------------------------------------- #

def _fit_beta(resid: np.ndarray, sigma: np.ndarray) -> float:
    b = conformal.elasticity(resid, sigma)
    return 1.0 if not np.isfinite(b) else float(b)


def fit_production(resid, sigma, dates) -> WidthFn:
    """`q_hat * sigma` -- exactly what is served today."""
    q = conformal.calibrate(resid, sigma)
    return lambda s: q * s


def fit_global_beta(resid, sigma, dates) -> WidthFn:
    """`q_hat * sigma ** beta`, one exponent per horizon.

    NOTE `q_hat` IS NOT COMPARABLE ACROSS ARMS. `sigma` is a fraction ~0.07, so
    `sigma ** 0.4` is ~0.35 -- five times larger -- and `q_hat` absorbs that.
    A `q_hat` persisted without its `beta` is meaningless, which is the whole
    reason the implementation has to write both or neither.
    """
    b = _fit_beta(resid, sigma)
    q = conformal.calibrate(resid, sigma ** b)
    return lambda s: q * s ** b


def beta_blocks(resid, sigma, dates, n_blocks: int = N_BLOCKS) -> np.ndarray:
    """`beta` refitted on `n_blocks` contiguous time blocks of the history.

    The read that separates "noisy estimate" from "wrong shape": if the spread
    here is small next to `beta - 1`, there is nothing for shrinkage to fix.
    """
    order = np.argsort(dates)
    cuts = np.array_split(order, n_blocks)
    out = []
    for c in cuts:
        if c.size < TRAILING_MIN_ROWS:
            continue
        out.append(_fit_beta(resid[c], sigma[c]))
    return np.asarray(out, dtype=float)


def shrink_factor(betas: np.ndarray, beta_hat: float) -> float:
    """How much of `beta_hat`'s departure from 1.0 to keep.

    `lambda = max(0, 1 - noise / signal)` with noise the squared standard error
    of the block mean and signal the squared departure -- the standard "shrink an
    estimate toward the null by its own noise" rule. `lambda -> 1` when the
    departure dwarfs its sampling spread, i.e. when shrinkage is pointless.
    """
    if betas.size < 2:
        return 1.0
    se2 = float(np.var(betas, ddof=1)) / betas.size
    signal2 = (beta_hat - 1.0) ** 2
    if signal2 <= 0:
        return 0.0
    return float(max(0.0, 1.0 - se2 / signal2))


def fit_shrunk_beta(resid, sigma, dates) -> WidthFn:
    b_hat = _fit_beta(resid, sigma)
    lam = shrink_factor(beta_blocks(resid, sigma, dates), b_hat)
    b = 1.0 + lam * (b_hat - 1.0)
    q = conformal.calibrate(resid, sigma ** b)
    return lambda s: q * s ** b


def fit_binned_scale(resid, sigma, dates) -> WidthFn:
    """Non-parametric scale: divide by the median |resid| observed AT that sigma.

    `m(sigma)` is the per-bin median of `|resid|` over `N_SCALE_BINS` quantile
    bins, interpolated in log-log space between bin centres and held flat past
    the ends. This is the arm that is free to BEND -- if the log-log line were
    the wrong shape, this beats the exponent; if the shape is right, it only adds
    variance and loses.
    """
    edges = np.quantile(sigma, np.linspace(0, 1, N_SCALE_BINS + 1)[1:-1])
    idx = np.searchsorted(edges, sigma, side="right")
    xs, ys = [], []
    for k in range(N_SCALE_BINS):
        sel = idx == k
        if sel.sum() < 50:
            continue
        xs.append(np.log(np.median(sigma[sel])))
        ys.append(np.log(max(np.median(np.abs(resid[sel])), 1e-9)))
    if len(xs) < 3:
        return fit_global_beta(resid, sigma, dates)
    xs, ys = np.asarray(xs), np.asarray(ys)

    def m(s: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.log(s), xs, ys))

    q = conformal.calibrate(resid, m(sigma))
    return lambda s: q * m(s)


def fit_per_bin_qhat(resid, sigma, dates) -> WidthFn:
    """Mondrian conformal: a separate `q_hat` inside each `sigma` decile.

    The most flexible arm and the one with the least to lose theoretically --
    split conformal's guarantee holds within each bin. What it costs is sample
    size per bin and a step-shaped width function, and it has to persist ten
    numbers and their edges instead of one.
    """
    edges = np.quantile(sigma, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    idx = np.searchsorted(edges, sigma, side="right")
    qs = np.empty(N_DECILES, dtype=float)
    pooled = conformal.calibrate(resid, sigma)
    for k in range(N_DECILES):
        sel = idx == k
        qs[k] = conformal.calibrate(resid[sel], sigma[sel]) if sel.sum() >= 200 \
            else pooled

    def width(s: np.ndarray) -> np.ndarray:
        return qs[np.searchsorted(edges, s, side="right")] * s

    return width


FITTERS: dict[str, Fitter] = {
    "P0_production": fit_production,
    "P1_global_beta": fit_global_beta,
    "P2_shrunk_beta": fit_shrunk_beta,
    "P3_binned_scale": fit_binned_scale,
    "P4_per_bin_qhat": fit_per_bin_qhat,
}


# --------------------------------------------------------------------------- #
# walk-forward evaluation
# --------------------------------------------------------------------------- #

def refit_dates(test_dates: list, every: int = REFIT_EVERY_DAYS) -> list:
    out = [test_dates[0]]
    for d in test_dates:
        if (d - out[-1]) / np.timedelta64(1, "D") >= every:
            out.append(d)
    return out


def walk_forward(resid, sigma, dates, test_dates, embargo, fitter) -> pd.DataFrame:
    """One row per scored test row: covered, width, sigma, date.

    The scale is refitted every `REFIT_EVERY_DAYS` on rows whose outcome had
    resolved by then (anchor <= date - embargo) and then held fixed, so nothing
    is scored against a scale that saw it.
    """
    order = np.argsort(dates, kind="stable")
    r, s, d = resid[order], sigma[order], dates[order]
    refits = refit_dates(test_dates, REFIT_EVERY_DAYS)
    fitted: list[tuple[np.datetime64, WidthFn]] = []
    for day in refits:
        hi = int(np.searchsorted(d, day - np.timedelta64(embargo, "D"),
                                 side="right"))
        if hi < TRAILING_MIN_ROWS:
            continue
        fitted.append((day, fitter(r[:hi], s[:hi], d[:hi])))
    if not fitted:
        return pd.DataFrame()

    keys = np.array([f[0] for f in fitted])
    rows = []
    for day in test_dates:
        j = int(np.searchsorted(keys, day, side="right")) - 1
        if j < 0:
            continue
        lo = int(np.searchsorted(d, day, side="left"))
        hi = int(np.searchsorted(d, day, side="right"))
        if hi - lo < MIN_ROWS_PER_DATE:
            continue
        st = s[lo:hi]
        w = fitted[j][1](st)
        rows.append(pd.DataFrame({
            "date": np.full(hi - lo, day), "sigma": st, "width": w,
            "covered": np.abs(r[lo:hi]) <= w}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score(res: pd.DataFrame, prod_width_median: float | None) -> dict:
    """Marginal coverage, the sigma-decile error, the date spread, and width."""
    sig = res["sigma"].to_numpy()
    edges = np.quantile(sig, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    idx = np.searchsorted(edges, sig, side="right")
    dec = res.groupby(idx)["covered"].mean()
    per_date = res.groupby("date")["covered"].mean()
    med_w = float(res["width"].median())
    return {
        "n_rows": len(res),
        "n_dates": int(res["date"].nunique()),
        "marginal": float(res["covered"].mean()),
        "decile_err_pp": float(np.mean(np.abs(dec - TARGET))) * 100.0,
        "decile_profile": " ".join(f"{v * 100:.0f}" for v in dec.sort_index()),
        "date_sd_pp": float(per_date.std()) * 100.0,
        "median_width_pct": med_w,
        "width_vs_prod": (med_w / prod_width_median
                          if prod_width_median else 1.0),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voted", default=None)
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--out", default="sigma_scale_design.csv")
    args = ap.parse_args()

    df = load_panel(args.voted or default_voted_panel())
    floor, cap = sigma_bounds_for_panel(df)
    logger.info("sigma clip: floor=%.6f cap=%.6f", floor, cap)
    fc = ItemForecaster(db_session=None)

    rows: list[dict] = []
    for h in [int(x) for x in args.horizons.split(",")]:
        sc = score_frame(fc, df, h, floor, cap)
        resid = sc["resid"].to_numpy(dtype=float)
        sigma = sc["sigma"].to_numpy(dtype=float)
        dates = pd.to_datetime(sc["date"]).to_numpy()
        embargo = embargo_days(h)
        uniq = np.unique(dates)
        first = uniq.min()
        td = [d for d in uniq
              if d - first >= np.timedelta64(MIN_HISTORY_DAYS + embargo, "D")]
        logger.info("")
        logger.info("=== h=%dd: %s rows, %d test dates, embargo %dd, "
                    "%d refits ===", h, f"{len(sc):,}", len(td), embargo,
                    len(refit_dates(td)))
        if len(td) < MIN_DATES_PER_HORIZON:
            logger.warning("  VOID: %d test dates", len(td))
            continue

        cut = td[int(len(td) * SELECT_FRACTION)]
        logger.info("  selection period ends %s; held-out period is %s -> %s",
                    pd.Timestamp(cut).date(), pd.Timestamp(cut).date(),
                    pd.Timestamp(td[-1]).date())

        # Is `beta` noisy, or is the shape wrong? Answered before any arm runs.
        hist_hi = int(np.searchsorted(np.sort(dates),
                                      cut - np.timedelta64(embargo, "D"),
                                      side="right"))
        o = np.argsort(dates, kind="stable")
        b_hat = _fit_beta(resid[o][:hist_hi], sigma[o][:hist_hi])
        blocks = beta_blocks(resid[o][:hist_hi], sigma[o][:hist_hi],
                             dates[o][:hist_hi])
        lam = shrink_factor(blocks, b_hat)
        logger.info("  beta=%.3f on the selection history; per-block %s "
                    "(sd %.3f, se %.3f); departure from 1 is %.3f -> shrink "
                    "keeps lambda=%.3f", b_hat,
                    " ".join(f"{b:.3f}" for b in blocks),
                    float(np.std(blocks, ddof=1)) if blocks.size > 1 else 0.0,
                    (float(np.std(blocks, ddof=1)) / np.sqrt(blocks.size)
                     if blocks.size > 1 else 0.0),
                    b_hat - 1.0, lam)

        sel, held, prod_w = {}, {}, {}
        for name in ARM_ORDER:
            res = walk_forward(resid, sigma, dates, td, embargo, FITTERS[name])
            if res.empty:
                logger.warning("  %-16s no fit", name)
                continue
            s_res = res[res["date"] < cut]
            h_res = res[res["date"] >= cut]
            if name == "P0_production":
                prod_w = {"sel": float(s_res["width"].median()),
                          "held": float(h_res["width"].median())}
            sel[name] = score(s_res, prod_w.get("sel"))
            held[name] = score(h_res, prod_w.get("held"))

        for label, book in (("SELECTION", sel), ("HELD-OUT", held)):
            logger.info("  --- %s ---", label)
            for name in ARM_ORDER:
                if name not in book:
                    continue
                m = book[name]
                logger.info("  %-16s marg=%5.1f%%  decile_err=%5.2fpp  "
                            "date_sd=%4.1fpp  width=%6.2f%% (%.2fx prod)  "
                            "deciles %s", name, m["marginal"] * 100,
                            m["decile_err_pp"], m["date_sd_pp"],
                            m["median_width_pct"], m["width_vs_prod"],
                            m["decile_profile"])
                rows.append({"horizon": h, "period": label, "arm": name,
                             "beta_hat": b_hat, "lambda": lam, **m})

        # The pinned selection rule, applied to the selection period only.
        ok = [n for n in ARM_ORDER if n in sel
              and abs(sel[n]["marginal"] - TARGET) <= MARGINAL_TOLERANCE]
        if not ok:
            logger.warning("  no arm inside the marginal gate; no pick")
            continue
        best = min(sel[n]["decile_err_pp"] for n in ok)
        pick = next(n for n in ok
                    if sel[n]["decile_err_pp"] <= best + SIMPLICITY_SLACK_PP)
        logger.info("  PICK (selection period, simplest within %.0fpp of "
                    "%.2fpp): %s  ->  held-out decile_err %.2fpp vs "
                    "production's %.2fpp, marginal %.1f%% vs %.1f%%, "
                    "width %.2fx", SIMPLICITY_SLACK_PP, best, pick,
                    held[pick]["decile_err_pp"],
                    held["P0_production"]["decile_err_pp"],
                    held[pick]["marginal"] * 100,
                    held["P0_production"]["marginal"] * 100,
                    held[pick]["width_vs_prod"])

    if not rows:
        logger.error("no results")
        return 1
    pd.DataFrame(rows).to_csv(args.out, index=False)
    logger.info("")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
