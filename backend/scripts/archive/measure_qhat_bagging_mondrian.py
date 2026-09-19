#!/usr/bin/env python3
"""Do bagged or Mondrian q_hats beat one pooled q_hat at equal coverage?

Pre-registration for the fold-composition probe. Read this before the CSV:
the bar, the placebos and the void conditions are fixed here, and the script
only reports the numbers they are applied to.

DEFECT (measured, not hypothesised): h=14 pooled q_hat swings
0.8288 -> 1.0440 -> 0.8527 (+-26%) across fold grids on identical data.
That is ~25% arbitrariness in every served half-width. Production's q_hat is
pooled OOF over the CV grids' validation windows
(`forecaster._cv_evaluate_horizon` + `_calibrate_conformal`), so shifting the
grid changes which dates count as OOF and the quantile follows the regime,
not the model.

ARMS (all walk-forward, all on the SAME fixed test dates, paired):

* S0_pooled    — production shape: one scalar over grid-0's OOF windows
                 (val_window=30 dates, step=150, end-anchored) resolved
                 before each test date (h+13 embargo).
* B_mean/B_med — bagged: q_hat over K shifted grids (offsets
                 g*step//K), served as their mean / median. Pure variance
                 reduction, no new signal, no new data.
* M_vol_dec    — Mondrian: per-sigma-decile symmetric q_hats, bin edges
                 fitted on calibration only, min rows per bin or fallback
                 to pooled. Plain empirical quantiles, no fitted model —
                 the difference from the refuted vol-rank GBM / SHRINK_K_GBM.
* P_shuffled   — placebo: M_vol_dec with sigma permuted across calibration
                 rows. Same n, same bins, same machinery, scale destroyed.

WHY OFFLINE AND MODEL-FREE. The score is |actual|/sigma with the REALISED
return as the residual (predicted |return| is median ~1% against half-widths
of 10-31%, so |actual| is the score to first order; validated against the
shipped q_hat at ratio 1.020/0.958/0.887/0.785 at h=3/7/14/30 in
`docs/research/2026-08-12-conditional-qhat-preregistration.md`). Nothing here
reimplements production: targets from `ItemForecaster.prepare_targets`,
sigma from `conformal.sigma_from_columns`, every quantile from
`conformal.calibrate` including the finite-sample level, grids mirror
`_compute_cv_splits` (end-anchored, 30-date val windows, 150-step).

THE VERDICT IS LEVEL-MATCHED. `c` scales each arm's widths to land marginal
coverage at exactly 80% on the test rows (fitted on the test rows, applied
identically to every arm), so only conditional information — or variance
reduction — can separate them. Primary metric is MEAN WIDTH at matched
coverage (narrower = better); sigma-strata err is secondary. Raw M2 has
already voided two reads in this repo by rewarding a uniformly narrower
band, so it is reported and never the bar.

PRE-REGISTERED BAR: an arm wins iff, at >= 3 of 4 horizons,
  (a) LM mean width is below S0, AND (b) raw M1 is inside 80+-3pp,
  AND (c) the placebo fails (a)-(b), AND (d) no LM sigma decile covers
  below 70% (kills the WACI trade that swapped top-decile over-coverage
  for under-coverage and scored it as a fix).
A pass buys ONE confirm on real OOF residuals; it does not buy a ship.

VOID: <100 walk-forward test dates at a horizon -> unreadable, not null;
2+ horizons void -> whole read void. If S0's own per-date coverage is flat
near 80% on this panel, the panel does not contain the defect.

Read-only: reads a voted price panel, writes a CSV. No DB, no artifacts.
`--test-stride N` scores every Nth test date (default 7; the grid stays
fixed and shared across arms, so the comparison stays paired).

    venv/bin/python -m scripts.archive.measure_qhat_bagging_mondrian --horizons 3,7,14,30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import conformal
from models.forecaster import ItemForecaster, embargo_days
from scripts.archive.measure_conditional_qhat import (
    MIN_DATES_PER_HORIZON,
    MIN_ROWS_PER_DATE,
    TARGET,
    TRAILING_MIN_ROWS,
    default_voted_panel,
    load_panel,
    score_frame,
    sigma_bounds_for_panel,
)

logger = logging.getLogger("qhat_bagging_mondrian")

# Production CV geometry, mirrored from forecaster._compute_cv_splits.
VAL_WINDOW = 30
CV_STEP = 150
CV_MIN_TRAIN_DATES = 200
N_BINS = 10
MIN_ROWS_PER_BIN = 200
PLACEBO_SEED = 20260913


# --------------------------------------------------------------------------- #
# calibration panel (lightweight: the fits below read only this surface,
# never the per-date state regressions of measure_conditional_qhat.Panel)
# --------------------------------------------------------------------------- #


class CalibPanel:
    """Date-sorted resid/sigma views with embargo arithmetic.

    score_frame guarantees date order. `horizon` sets the H+13 embargo via
    `embargo_days` (called, never re-derived).
    """

    def __init__(self, scores: pd.DataFrame, horizon: int):
        self.horizon = horizon
        self.embargo = embargo_days(horizon)
        self.resid = scores["resid"].to_numpy(dtype=float)
        self.sigma = scores["sigma"].to_numpy(dtype=float)
        self.dates = pd.to_datetime(scores["date"]).to_numpy()
        self.unique_dates = np.unique(self.dates)

    def upto(self, day) -> int:
        return int(np.searchsorted(self.dates, day, side="right"))

    def since(self, day) -> int:
        return int(np.searchsorted(self.dates, day, side="left"))

    def rows_on(self, day) -> slice:
        return slice(self.since(day), self.upto(day))

    def avail(self, day):
        return day - np.timedelta64(self.embargo, "D")


def test_dates_for_panel(p: CalibPanel, min_history_days: int = 120):
    """Fixed evaluation grid, same list for every arm (paired comparison)."""
    first = p.unique_dates.min()
    need = np.timedelta64(min_history_days + p.embargo, "D")
    return [d for d in p.unique_dates if d - first >= need]


# --------------------------------------------------------------------------- #
# calibration grids (the fold-composition instrument)
# --------------------------------------------------------------------------- #


def val_windows(
    unique_dates, shift: int = 0, val_window: int = VAL_WINDOW, step: int = CV_STEP, min_train: int = CV_MIN_TRAIN_DATES
):
    """End-anchored validation windows over date positions, offset by shift.

    Mirrors `_compute_cv_splits`: the newest window abuts the frame end
    (minus shift) and earlier windows step back by `step`. Returns a list of
    (start_pos, end_pos) index pairs into `unique_dates`. Position-based, not
    calendar-based, exactly like production.
    """
    n = len(unique_dates)
    last_end = n - val_window - int(shift)
    ends = []
    end = last_end
    while end >= min_train:
        ends.append(end)
        end -= step
    ends.reverse()
    return [(e, e + val_window) for e in ends if e + val_window <= n and val_window >= 7]


def grid_date_sets(unique_dates, k_grids: int):
    """The K shifted grids' full-history OOF date sets.

    Grid g offsets the end anchor by g*step//K positions, so K=2 reproduces
    the "shifted fold grid" contrast the +-26% was measured on. Each set is
    the union of that grid's validation windows (frozenset of dates).
    """
    step = CV_STEP
    grids = []
    for g in range(k_grids):
        shift = g * step // k_grids
        wins = val_windows(unique_dates, shift=shift)
        dates = set()
        for s, e in wins:
            dates.update(unique_dates[s:e].tolist())
        grids.append(frozenset(pd.to_datetime(list(dates))))
    return grids


def calib_mask_for(p, grid_dates: frozenset, day) -> np.ndarray:
    """Rows in this grid's OOF windows with anchor <= day - embargo."""
    avail = p.avail(day)
    dates = pd.to_datetime(p.dates)
    in_grid = np.array([d in grid_dates for d in dates], dtype=bool)
    return in_grid & (dates <= avail)


def pooled_qhat(p, mask: np.ndarray):
    """Production quantile on a masked calibration set, or None when thin."""
    if int(mask.sum()) < TRAILING_MIN_ROWS:
        return None
    try:
        return float(conformal.calibrate(p.resid[mask], p.sigma[mask]))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# arm fits: one test date -> per-row q predictions for that date's rows
# --------------------------------------------------------------------------- #


def fit_s0(p, day, grid_dates: frozenset):
    return pooled_qhat(p, calib_mask_for(p, grid_dates, day))


def fit_bagged(p, day, grids, how: str = "mean"):
    qs = []
    for gd in grids:
        q = pooled_qhat(p, calib_mask_for(p, gd, day))
        if q is not None:
            qs.append(q)
    if len(qs) < 2:
        return fit_s0(p, day, grids[0])
    return float(np.mean(qs)) if how == "mean" else float(np.median(qs))


def fit_mondrian(p, day, grid_dates: frozenset, shuffle: bool = False, seed: int = PLACEBO_SEED, n_bins: int = N_BINS):
    """Per-sigma-decile q_hats: (edges, per-bin q, pooled fallback).

    Edges are quantiles of the CALIBRATION sigma only; test rows map via
    searchsorted. Bins under MIN_ROWS_PER_BIN fall back to pooled. Placebo
    permutes sigma across calibration rows first (same n, same machinery).
    """
    mask = calib_mask_for(p, grid_dates, day)
    if int(mask.sum()) < TRAILING_MIN_ROWS:
        return None
    resid, sigma = p.resid[mask], p.sigma[mask]
    if shuffle:
        rng = np.random.default_rng(seed + int(pd.Timestamp(day).value % (2**31)))
        sigma = rng.permutation(sigma)
    try:
        fallback = float(conformal.calibrate(resid, sigma))
    except ValueError:
        return None
    edges = np.quantile(sigma, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    idx = np.searchsorted(edges, sigma, side="right")
    per_bin = np.empty(n_bins)
    for b in range(n_bins):
        m = idx == b
        if int(m.sum()) < MIN_ROWS_PER_BIN:
            per_bin[b] = fallback
            continue
        try:
            per_bin[b] = float(conformal.calibrate(resid[m], sigma[m]))
        except ValueError:
            per_bin[b] = fallback
    return edges, per_bin, fallback


def apply_fit(fit, sigma_test: np.ndarray) -> np.ndarray | None:
    """Per-row q for test rows: scalar broadcasts, Mondrian looks up bins."""
    if fit is None:
        return None
    if isinstance(fit, tuple):
        edges, per_bin, fallback = fit
        idx = np.searchsorted(np.asarray(edges), np.asarray(sigma_test), side="right")
        idx = np.clip(idx, 0, len(per_bin) - 1)
        return np.where(np.isfinite(sigma_test), per_bin[idx], fallback)
    return np.full(np.asarray(sigma_test).shape, float(fit))


# --------------------------------------------------------------------------- #
# evaluation (fixed test grid, level-matched verdict)
# --------------------------------------------------------------------------- #


def _cond_err(dates: np.ndarray, covered: np.ndarray) -> float:
    cov = pd.Series(covered).groupby(pd.Series(dates)).mean()
    return float(np.mean(np.abs(cov - TARGET))) * 100.0


def _sigma_stratum_err(sigma: np.ndarray, covered: np.ndarray, n_strata: int = N_BINS):
    edges = np.quantile(sigma, np.linspace(0, 1, n_strata + 1)[1:-1])
    idx = np.searchsorted(edges, sigma, side="right")
    cov = pd.Series(covered).groupby(pd.Series(idx)).mean().sort_index()
    err = float(np.mean(np.abs(cov - TARGET))) * 100.0
    return err, " ".join(f"{v * 100:.0f}" for v in cov), float(cov.min())


def _level_match_c(scores: np.ndarray, q: np.ndarray) -> float:
    lo, hi = 1e-6, 1e6
    for _ in range(40):
        mid = (lo + hi) / 2
        if float(np.mean(scores <= q * mid)) < TARGET:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def evaluate(p, fit_fn, test_dates) -> dict:
    days, sig, sc, q = [], [], [], []
    for day in test_dates:
        fit = fit_fn(p, day)
        sl = p.rows_on(day)
        n = sl.stop - sl.start
        if fit is None or n < MIN_ROWS_PER_DATE:
            continue
        s = p.sigma[sl]
        qq = apply_fit(fit, s)
        if qq is None:
            continue
        days.append(np.full(n, pd.Timestamp(day).to_datetime64()))
        sig.append(s)
        sc.append(np.abs(p.resid[sl]) / s)
        q.append(qq)
    if not days:
        return {}
    days = np.concatenate(days)
    sig = np.concatenate(sig)
    sc = np.concatenate(sc)
    q = np.concatenate(q)

    covered = sc <= q
    width = q * sig
    c = _level_match_c(sc, q)
    cov_lm = sc <= q * c
    width_lm = q * c * sig
    sig_err, _, _ = _sigma_stratum_err(sig, covered)
    sig_err_lm, profile_lm, min_dec_lm = _sigma_stratum_err(sig, cov_lm)
    return {
        "n_dates": int(pd.Series(days).nunique()),
        "n_rows": int(covered.size),
        "M1_marginal": float(covered.mean()),
        "M2_cond_err_pp": _cond_err(days, covered),
        "M2_sigma_err_pp": sig_err,
        "mean_width": float(width.mean()),
        "level_match_c": c,
        "M1lm_marginal": float(cov_lm.mean()),
        "M2lm_cond_err_pp": _cond_err(days, cov_lm),
        "M2lm_sigma_err_pp": sig_err_lm,
        "mean_width_lm": float(width_lm.mean()),
        "min_decile_cov_lm": min_dec_lm,
        "sigma_decile_cov_lm": profile_lm,
    }


def grid_stability(p, grids, label: str = "") -> list:
    """Each grid's pooled q_hat on its FULL-history OOF set (no walk-forward).

    This is the number the +-26% was measured on: same data, different grid,
    one scalar each. Reported, not scored — scoring happens walk-forward.
    """
    out = []
    for g, gd in enumerate(grids):
        mask = np.array([d in gd for d in pd.to_datetime(p.dates)])
        q = pooled_qhat(p, mask)
        out.append({"grid": g, "n_rows": int(mask.sum()), "q_hat": None if q is None else round(float(q), 4)})
    qs = [r["q_hat"] for r in out if r["q_hat"] is not None]
    if len(qs) >= 2:
        logger.info(
            "  grid stability%s: %s (max/min=%.3f)",
            label,
            " -> ".join(f"{v:.4f}" for v in qs),
            max(qs) / min(qs) if min(qs) > 0 else float("nan"),
        )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voted", default=None)
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--k-grids", type=int, default=4)
    ap.add_argument(
        "--test-stride",
        type=int,
        default=7,
        help="score every Nth test date. The grid stays fixed and "
        "shared across arms (paired); striding only thins it. "
        "1 reproduces the full walk-forward.",
    )
    ap.add_argument("--out", default="qhat_bagging_mondrian.csv")
    args = ap.parse_args()

    path = args.voted or default_voted_panel()
    df = load_panel(path)
    floor, cap = sigma_bounds_for_panel(df)
    logger.info("sigma clip: floor=%.6f cap=%.6f (panel's own; read ratios, not levels)", floor, cap)
    fc = ItemForecaster(db_session=None)

    rows = []
    for h in [int(x) for x in args.horizons.split(",")]:
        sc = score_frame(fc, df, h, floor, cap)
        p = CalibPanel(sc, h)
        td = test_dates_for_panel(p)[:: max(int(args.test_stride), 1)]
        logger.info("")
        logger.info(
            "h=%dd: %s scored rows, %d anchor dates, %d test dates", h, f"{len(sc):,}", len(p.unique_dates), len(td)
        )
        if len(td) < MIN_DATES_PER_HORIZON:
            logger.warning("  VOID: %d test dates < %d required", len(td), MIN_DATES_PER_HORIZON)
        grids = grid_date_sets(p.unique_dates, args.k_grids)
        grid_stability(p, grids, label=f" h={h}")

        # The K grid-qs are computed once per test date and shared by both
        # bagged arms; without this B_mean/B_median each pay K fits per date.
        _bag_cache: dict = {}

        def _bagged_q(pp, d, how: str, _bag_cache=_bag_cache, grids=grids):
            key = (pd.Timestamp(d).value, how)
            if key not in _bag_cache:
                qs = []
                for gd in grids:
                    q = pooled_qhat(pp, calib_mask_for(pp, gd, d))
                    if q is not None:
                        qs.append(q)
                if len(qs) < 2:
                    _bag_cache[key] = fit_s0(pp, d, grids[0])
                else:
                    _bag_cache[key] = float(np.mean(qs)) if how == "mean" else float(np.median(qs))
            return _bag_cache[key]

        arms = {
            "S0_pooled": lambda pp, d, _g=grids[0]: fit_s0(pp, d, _g),
            "B_mean": lambda pp, d: _bagged_q(pp, d, "mean"),
            "B_median": lambda pp, d: _bagged_q(pp, d, "median"),
            "M_vol_dec": lambda pp, d, _g=grids[0]: fit_mondrian(pp, d, _g),
            "P_shuffled": lambda pp, d, _g=grids[0]: fit_mondrian(pp, d, _g, shuffle=True),
        }
        for name, fn in arms.items():
            m = evaluate(p, fn, td)
            if not m:
                logger.warning("  %-12s no fit", name)
                continue
            m.update(horizon=h, scheme=name)
            rows.append(m)
            logger.info(
                "  %-12s M1=%.1f%% M2=%5.2fpp w=%.4f | LM c=%.3f M2*=%5.2fpp sig*=%5.2fpp w*=%.4f mindec*=%.0f%% [%s]",
                name,
                m["M1_marginal"] * 100,
                m["M2_cond_err_pp"],
                m["mean_width"],
                m["level_match_c"],
                m["M2lm_cond_err_pp"],
                m["M2lm_sigma_err_pp"],
                m["mean_width_lm"],
                m["min_decile_cov_lm"] * 100,
                m["sigma_decile_cov_lm"],
            )

    out = pd.DataFrame(rows)
    if out.empty:
        logger.error("no results")
        return 1
    out.to_csv(args.out, index=False)
    logger.info("")
    logger.info("wrote %s", args.out)

    base = out[out["scheme"] == "S0_pooled"].set_index("horizon")
    logger.info("")
    logger.info(
        "PRE-REGISTERED BAR — LM width below S0 AND M1 in 80+-3pp, "
        "at >=3/4 horizons; placebo must fail; no LM decile <70%%."
    )
    for name in out["scheme"].unique():
        if name == "S0_pooled":
            continue
        arm = out[out["scheme"] == name].set_index("horizon")
        common = arm.index.intersection(base.index)
        narrower = int((arm.loc[common, "mean_width_lm"] < base.loc[common, "mean_width_lm"]).sum())
        inband = int(((arm.loc[common, "M1_marginal"] - TARGET).abs() <= 0.03).sum())
        guard = bool((arm.loc[common, "min_decile_cov_lm"] >= 0.70).all())
        verdict = "PASS" if (narrower >= 3 and inband >= 3 and guard) else "fail"
        logger.info(
            "  %-12s narrower* %d/%d, M1 in band %d/%d, decile>=70%% %s -> %s",
            name,
            narrower,
            len(common),
            inband,
            len(common),
            guard,
            verdict,
        )
    logger.info("A P_shuffled PASS voids the bar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
