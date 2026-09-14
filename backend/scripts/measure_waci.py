#!/usr/bin/env python3
"""Does per-width-bin signed conformal (WACI) beat one pooled signed pair?

`models/conformal.py` carries a complete Width-Adaptive implementation
(`calibrate_signed_waci`, `waci_lookup`, `band_signed_waci`) with ZERO callers
— no wiring, no flag, no measurement. The 2026-09-09 stratum diagnostic says
width is the one axis with real conditional spread (narrow 84-86% vs wide
95-96% at h=3/7, CIs non-overlapping), so WACI is the Mondrian arm that might
actually earn its keep. This measures it before anyone wires it.

DESIGN (same honesty rules as `measure_conditional_qhat.py`):

* Offline and model-free: the conformal score is `resid / sigma` with the
  REALISED return as the residual (predicted |return| is median 0.95% against
  half-widths of 10-31%, so |actual| is the score to first order). Targets from
  `ItemForecaster.prepare_targets`, sigma from `conformal.sigma_from_columns`,
  every quantile from `conformal.calibrate_signed` / `calibrate_signed_waci` —
  including the finite-sample levels. Nothing reimplemented.
* Expanding history with the `horizon + 13` embargo (`embargo_days`, never
  re-derived): each test date's (q_lo, q_hi) — pooled or per-bin — is fitted
  on anchors whose outcomes resolved before it.
* LEVEL-MATCHED read is the verdict, not the raw one. A narrower band scores
  as a conditional fix on any unmatched statistic whenever marginal coverage
  moves toward target for any reason (the 2026-08-12 placebo lesson). `c`
  scales the served pair `(c*q_lo, c*q_hi)` to land marginal coverage at
  exactly 80% per arm; it is fitted on the test rows and applied identically
  to every arm, so only conditional information can separate them.
* Placebo: WACI fitted with sigma SHUFFLED across calibration rows (same
  sample size, same bin machinery, scale information destroyed). If the
  placebo passes the bar, the bar is void.

PRE-REGISTERED BAR: level-matched sigma-strata error below S0 at >= 3 of 4
horizons, AND marginal coverage within 80±3pp raw at >= 3 of 4, AND the
placebo must fail both. A pass buys ONE confirm on real OOF residuals; it
does not buy a ship.

Read-only: reads a voted price panel, writes a CSV. No DB, no artifacts.

    venv/bin/python -m scripts.measure_waci --horizons 3,7,14,30
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
from models.forecaster import ItemForecaster
from scripts.measure_conditional_qhat import (
    MIN_DATES_PER_HORIZON,
    MIN_ROWS_PER_DATE,
    TARGET,
    TRAILING_MIN_ROWS,
    default_voted_panel,
    load_panel,
    score_frame,
    sigma_bounds_for_panel,
    state_variables,
    test_dates_for,
)

logger = logging.getLogger("measure_waci")

N_STRATA = 10
PLACEBO_SEED = 20260909
WACI_BINS = 10


def fit_s0(p, day):
    """Production shape: one pooled signed pair over embargoed history."""
    hi = p.upto(p.avail(day))
    if hi < TRAILING_MIN_ROWS:
        return None
    return conformal.calibrate_signed(p.resid[:hi], p.sigma[:hi])


def fit_waci(p, day, shuffle: bool = False, seed: int = PLACEBO_SEED):
    """Per-width-bin signed pairs over embargoed history.

    Placebo shuffles sigma across calibration rows: same n, same bins, same
    kernel lookup — no scale information.
    """
    hi = p.upto(p.avail(day))
    if hi < TRAILING_MIN_ROWS:
        return None
    resid, sigma = p.resid[:hi], p.sigma[:hi]
    if shuffle:
        rng = np.random.default_rng(seed + int(pd.Timestamp(day).value % (2**31)))
        sigma = rng.permutation(sigma)
    try:
        return conformal.calibrate_signed_waci(resid, sigma, n_bins=WACI_BINS)
    except ValueError:
        return None


def apply_fit(fit, sigma_test: np.ndarray):
    """Per-row (q_lo, q_hi) for test rows. Pooled pair broadcasts; WACI looks
    up each row's scale through the served kernel path (`waci_lookup`)."""
    if fit is None:
        return None, None
    if isinstance(fit, dict):
        return conformal.waci_lookup(sigma_test, fit)
    q_lo, q_hi = fit
    return (np.full(sigma_test.shape, q_lo), np.full(sigma_test.shape, q_hi))


def _cond_err(dates: np.ndarray, covered: np.ndarray) -> float:
    cov = pd.Series(covered).groupby(pd.Series(dates)).mean()
    return float(np.mean(np.abs(cov - TARGET))) * 100.0


def _sigma_stratum_err(sigma: np.ndarray, covered: np.ndarray, n_strata: int = N_STRATA) -> tuple[float, str]:
    edges = np.quantile(sigma, np.linspace(0, 1, n_strata + 1)[1:-1])
    idx = np.searchsorted(edges, sigma, side="right")
    cov = pd.Series(covered).groupby(pd.Series(idx)).mean()
    err = float(np.mean(np.abs(cov - TARGET))) * 100.0
    return err, " ".join(f"{v * 100:.0f}" for v in cov.sort_index())


def _level_match(scores: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    """Scalar c with mean((scores >= c*q_lo) & (scores <= c*q_hi)) == 80%.

    Monotone in c > 0 (widening a centred-toward-mid interval only adds rows
    while q_lo <= 0 <= q_hi, which the signed fit guarantees up to sampling
    noise), so bisection is exact to tolerance.
    """
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(np.mean((scores >= mid * q_lo) & (scores <= mid * q_hi))) < TARGET:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def evaluate(p, fit_fn, test_dates) -> dict:
    days, sig, sc, qlo, qhi = [], [], [], [], []
    for day in test_dates:
        fit = fit_fn(p, day)
        sl = p.rows_on(day)
        n = sl.stop - sl.start
        if fit is None or n < MIN_ROWS_PER_DATE:
            continue
        s = p.sigma[sl]
        lo_arr, hi_arr = apply_fit(fit, s)
        if lo_arr is None:
            continue
        days.append(np.full(n, pd.Timestamp(day).to_datetime64()))
        sig.append(s)
        sc.append(p.resid[sl] / s)
        qlo.append(lo_arr)
        qhi.append(hi_arr)
    if not days:
        return {}
    days = np.concatenate(days)
    sig = np.concatenate(sig)
    sc = np.concatenate(sc)
    qlo = np.concatenate(qlo)
    qhi = np.concatenate(qhi)

    covered = (sc >= qlo) & (sc <= qhi)
    c = _level_match(sc, qlo, qhi)
    cov_lm = (sc >= c * qlo) & (sc <= c * qhi)
    sig_err, _ = _sigma_stratum_err(sig, covered)
    sig_err_lm, profile_lm = _sigma_stratum_err(sig, cov_lm)
    return {
        "n_dates": int(pd.Series(days).nunique()),
        "n_rows": int(covered.size),
        "M1_marginal": float(covered.mean()),
        "M2_cond_err_pp": _cond_err(days, covered),
        "M2_sigma_err_pp": sig_err,
        "level_match_c": c,
        "M1lm_marginal": float(cov_lm.mean()),
        "M2lm_cond_err_pp": _cond_err(days, cov_lm),
        "M2lm_sigma_err_pp": sig_err_lm,
        "sigma_decile_cov_lm": profile_lm,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voted", default=None)
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--out", default="waci_measure.csv")
    args = ap.parse_args()

    path = args.voted or default_voted_panel()
    df = load_panel(path)
    floor, cap = sigma_bounds_for_panel(df)
    state = state_variables(df, floor, cap)
    fc = ItemForecaster(db_session=None)

    arms = {
        "S0_pooled_signed": fit_s0,
        "WACI_per_width_bin": lambda p, d: fit_waci(p, d),
        "P_shuffled_scale": lambda p, d: fit_waci(p, d, shuffle=True),
    }

    rows = []
    for h in [int(x) for x in args.horizons.split(",")]:
        sc = score_frame(fc, df, h, floor, cap)
        p = __import__("scripts.measure_conditional_qhat", fromlist=["Panel"]).Panel(sc, h, state)
        td = test_dates_for(p)
        logger.info("")
        logger.info(
            "h=%dd: %s scored rows, %d anchor dates, %d test dates", h, f"{len(sc):,}", len(p.unique_dates), len(td)
        )
        if len(td) < MIN_DATES_PER_HORIZON:
            logger.warning("  VOID: %d test dates < %d required", len(td), MIN_DATES_PER_HORIZON)
        for name, fn in arms.items():
            m = evaluate(p, fn, td)
            if not m:
                logger.warning("  %-18s no fit", name)
                continue
            m.update(horizon=h, scheme=name)
            rows.append(m)
            logger.info(
                "  %-18s M1=%.1f%% M2=%5.2fpp sig=%5.2fpp | LM c=%.3f M2*=%5.2fpp sig*=%5.2fpp [%s]",
                name,
                m["M1_marginal"] * 100,
                m["M2_cond_err_pp"],
                m["M2_sigma_err_pp"],
                m["level_match_c"],
                m["M2lm_cond_err_pp"],
                m["M2lm_sigma_err_pp"],
                m["sigma_decile_cov_lm"],
            )

    out = pd.DataFrame(rows)
    if out.empty:
        logger.error("no results")
        return 1
    out.to_csv(args.out, index=False)
    logger.info("")
    logger.info("wrote %s", args.out)

    base = out[out["scheme"] == "S0_pooled_signed"].set_index("horizon")
    for name in ("WACI_per_width_bin", "P_shuffled_scale"):
        arm = out[out["scheme"] == name].set_index("horizon")
        common = arm.index.intersection(base.index)
        sig_better = int((arm.loc[common, "M2lm_sigma_err_pp"] < base.loc[common, "M2lm_sigma_err_pp"]).sum())
        inband = int(((arm.loc[common, "M1_marginal"] - TARGET).abs() <= 0.03).sum())
        verdict = "PASS" if (sig_better >= 3 and inband >= 3) else "fail"
        logger.info(
            "  %-18s LM-sigma better at %d/%d, M1 in band at %d/%d -> %s",
            name,
            sig_better,
            len(common),
            inband,
            len(common),
            verdict,
        )
    logger.info("A P_shuffled_scale PASS voids the bar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
