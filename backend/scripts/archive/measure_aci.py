#!/usr/bin/env python3
"""Does time-adaptive conformal (ACI) fix the per-date coverage error?

Instrument for `docs/research/2026-09-22-adaptive-conformal-preregistration.md`.
The bar, the arms and the void conditions are fixed there; this script only
computes the numbers they are applied to. Read the prereg before the CSV.

Harness: `measure_qhat_bagging_mondrian`'s CalibPanel, walk-forward test dates,
production-shaped S0 fit and `evaluate` (level-matched scoring), unchanged.
Model-free realised-return residuals, as validated there.

ACI is sequential -- the q served on day t depends on every miss resolved
before t -- so each arm's per-date q is precomputed in date order and handed to
`evaluate` as a lookup. Every arm is then scored by the same code on the same
dates.

ARMS
  S0_pooled  pooled q_hat over grid-0's OOF windows, H+13 embargo (production).
  R_roll     pooled q_hat over the last ROLL_DATES dates resolved by day t.
  A_aci      S0's calibration scores, quantile at 1 - alpha_t, where
             alpha <- alpha + gamma * (ALPHA_STAR - err) as each date's
             cross-sectional miss rate `err` resolves h days after it was served.
  P_aci      placebo, open-loop: A_aci's own update schedule and err values,
             with the values permuted across updates. Same gamma, same values,
             temporal order destroyed. (A closed-loop placebo is not defined:
             err depends on the band served, so "the same err values" can only
             be replayed.) The permutation spans all updates, so the placebo
             sees the sample's overall miss rate -- which is exactly the
             non-temporal information it exists to control for.

gamma is chosen on the TUNING block (first third of scored dates) by the lowest
level-matched date error, then frozen. Verdicts read the SCORING block only.

Read-only: reads a voted price panel, writes a CSV. No DB, no artifacts.

    venv/bin/python -m scripts.archive.measure_aci --horizons 3,7,14
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import conformal
from models.forecaster import ItemForecaster
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
from scripts.archive.measure_qhat_bagging_mondrian import (
    CalibPanel,
    evaluate,
    grid_date_sets,
    test_dates_for_panel,
)

logger = logging.getLogger("measure_aci")

ALPHA_STAR = 1.0 - TARGET
GAMMAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)  # prereg, as amended 2026-09-22
ALPHA_CLIP = (0.005, 0.60)
ROLL_DATES = 20
PLACEBO_SEED = 20260922
PRIMARY_HORIZONS = (3, 7)

# Pre-registered bar (prereg "Bar (fixed now)").
M1_BAND = 0.03
DATE_ERR_MARGIN_PP = 1.0
P10_MARGIN = 0.03
WIDTH_CAP = 1.05
MIN_DECILE = 0.70
S0_FLAT_PP = 3.0


# --------------------------------------------------------------------------- #
# calibration: S0's embargoed OOF scores, quantiles at arbitrary levels
# --------------------------------------------------------------------------- #


def quantile_levels(scores: np.ndarray, alphas) -> list[float]:
    """`conformal.calibrate`'s finite-sample quantile at several alphas at once.

    Same scores, same level ceil((n+1)(1-alpha))/n, same np.quantile; one
    partition instead of one per alpha. Tested equal to `calibrate`.
    """
    n = scores.size
    levels = [min(np.ceil((n + 1) * (1.0 - float(a))) / n, 1.0) for a in alphas]
    return [float(v) for v in np.quantile(scores, levels)]


class EmbargoedCalib:
    """S0's calibration set on day t: grid-0 OOF rows anchored <= t - embargo.

    Equivalent to `calib_mask_for` + `pooled_qhat`, with the grid membership and
    scores computed once rather than per test date (dates are sorted, so the
    embargo is a prefix).
    """

    def __init__(self, p: CalibPanel, grid_dates: frozenset):
        self.p = p
        grid = np.array(sorted(pd.to_datetime(list(grid_dates))), dtype="datetime64[ns]")
        self.in_grid = np.isin(p.dates.astype("datetime64[ns]"), grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.scores = np.abs(p.resid) / conformal.resolve_scale(p.sigma, conformal.BETA_NEUTRAL)

    def quantiles(self, day, alphas) -> list[float] | None:
        k = self.p.upto(self.p.avail(day))
        s = self.scores[:k][self.in_grid[:k]]
        s = s[np.isfinite(s)]
        if s.size < TRAILING_MIN_ROWS:
            return None
        return quantile_levels(s, alphas)


def _clip(alpha: float) -> float:
    return float(min(max(alpha, ALPHA_CLIP[0]), ALPHA_CLIP[1]))


def _day_scores(p: CalibPanel, day) -> np.ndarray:
    sl = p.rows_on(day)
    return np.abs(p.resid[sl]) / p.sigma[sl]


# --------------------------------------------------------------------------- #
# arms: each returns {day: q} in date order
# --------------------------------------------------------------------------- #


def run_aci(p: CalibPanel, calib: EmbargoedCalib, dates, gammas=GAMMAS):
    """S0 and one ACI path per gamma, in one pass over the dates.

    Returns (s0_qmap, {gamma: qmap}, {gamma: events}), where events are the
    (resolve_day, err) updates applied, in order -- the placebo replays them.
    A date's err is applied on the first date >= its anchor + h: the outcome of
    a forecast is not known before its target date.
    """
    lag = np.timedelta64(int(p.horizon), "D")
    alpha = dict.fromkeys(gammas, ALPHA_STAR)
    pending = {g: deque() for g in gammas}
    qmaps: dict = {g: {} for g in gammas}
    events: dict = {g: [] for g in gammas}
    s0: dict = {}
    for day in dates:
        for g in gammas:
            queue = pending[g]
            while queue and queue[0][0] <= day:
                resolve_day, err = queue.popleft()
                alpha[g] = _clip(alpha[g] + g * (ALPHA_STAR - err))
                events[g].append((resolve_day, err))
        qs = calib.quantiles(day, [ALPHA_STAR] + [alpha[g] for g in gammas])
        if qs is None:
            continue
        s0[day] = qs[0]
        sc = _day_scores(p, day)
        for g, q in zip(gammas, qs[1:], strict=True):
            qmaps[g][day] = q
            if sc.size:
                pending[g].append((day + lag, float(np.mean(sc > q))))
    return s0, qmaps, events


def run_placebo(p: CalibPanel, calib: EmbargoedCalib, dates, gamma: float, events, seed: int = PLACEBO_SEED) -> dict:
    """A_aci's schedule with its err values permuted across updates."""
    schedule = [d for d, _ in events]
    errs = np.random.default_rng(seed).permutation(np.array([e for _, e in events], dtype=float))
    alpha, k, qmap = ALPHA_STAR, 0, {}
    for day in dates:
        while k < len(schedule) and schedule[k] <= day:
            alpha = _clip(alpha + gamma * (ALPHA_STAR - errs[k]))
            k += 1
        qs = calib.quantiles(day, [alpha])
        if qs is not None:
            qmap[day] = qs[0]
    return qmap


def run_roll(p: CalibPanel, dates, n_dates: int = ROLL_DATES) -> dict:
    """Pooled q_hat over the newest `n_dates` anchor dates resolved by day t."""
    lag = np.timedelta64(int(p.horizon), "D")
    ud = p.unique_dates
    qmap = {}
    for day in dates:
        end = int(np.searchsorted(ud, day - lag, side="right"))
        if end < n_dates:
            continue
        sl = slice(p.since(ud[end - n_dates]), p.upto(ud[end - 1]))
        if sl.stop - sl.start < TRAILING_MIN_ROWS:
            continue
        qmap[day] = conformal.calibrate(p.resid[sl], p.sigma[sl])
    return qmap


# --------------------------------------------------------------------------- #
# scoring and the pre-registered bar
# --------------------------------------------------------------------------- #


def score(p: CalibPanel, qmap: dict, dates) -> dict:
    """`evaluate` on a precomputed qmap, plus the P10 per-date LM coverage."""
    m = evaluate(p, lambda _p, d: qmap.get(d), dates)
    if not m:
        return m
    c = m["level_match_c"]
    covs = [float(np.mean(_day_scores(p, d) <= qmap[d] * c)) for d in dates]
    m["p10_date_cov_lm"] = float(np.percentile(covs, 10))
    return m


def judge(arm: dict, s0: dict, others: list[dict]) -> dict:
    """Bar conditions (1)-(5). `others` are the arms (2) must also beat."""
    err = arm["M2lm_cond_err_pp"]
    return {
        "c1_marginal": abs(arm["M1_marginal"] - TARGET) <= M1_BAND,
        "c2_date_err": err <= s0["M2lm_cond_err_pp"] - DATE_ERR_MARGIN_PP
        and all(err < o["M2lm_cond_err_pp"] for o in others),
        "c3_p10": arm["p10_date_cov_lm"] >= s0["p10_date_cov_lm"] + P10_MARGIN,
        "c4_width": arm["mean_width_lm"] <= WIDTH_CAP * s0["mean_width_lm"],
        "c5_decile": arm["min_decile_cov_lm"] >= MIN_DECILE,
    }


def void_reasons(n_dates: int, s0: dict, placebo: dict) -> list[str]:
    out = []
    if n_dates < MIN_DATES_PER_HORIZON:
        out.append(f"{n_dates} scoring dates < {MIN_DATES_PER_HORIZON}")
    if s0["M2lm_cond_err_pp"] < S0_FLAT_PP:
        out.append(f"S0 date error {s0['M2lm_cond_err_pp']:.2f}pp < {S0_FLAT_PP}pp: panel lacks the defect")
    if placebo["M2lm_cond_err_pp"] <= s0["M2lm_cond_err_pp"] - DATE_ERR_MARGIN_PP:
        out.append("placebo clears the date-error bar against S0: the metric rewards machinery")
    return out


def run_horizon(p: CalibPanel, grid_dates: frozenset) -> dict:
    """Every arm, gamma selection, scores and verdict for one horizon."""
    dates = test_dates_for_panel(p)
    calib = EmbargoedCalib(p, grid_dates)
    s0_q, aci_q, events = run_aci(p, calib, dates)
    roll_q = run_roll(p, dates)

    # Paired: score only dates every arm served, with enough rows.
    scored = [
        d
        for d in dates
        if d in s0_q and d in roll_q and (p.rows_on(d).stop - p.rows_on(d).start) >= MIN_ROWS_PER_DATE
    ]
    cut = len(scored) // 3
    tune, test = scored[:cut], scored[cut:]

    tuning = {g: score(p, aci_q[g], tune) for g in GAMMAS}
    gamma = min(GAMMAS, key=lambda g: tuning[g]["M2lm_cond_err_pp"])
    placebo_q = run_placebo(p, calib, dates, gamma, events[gamma])

    arms = {
        "S0_pooled": score(p, s0_q, test),
        "R_roll": score(p, roll_q, test),
        "A_aci": score(p, aci_q[gamma], test),
        "P_aci": score(p, placebo_q, test),
    }
    s0 = arms["S0_pooled"]
    aci_bar = judge(arms["A_aci"], s0, [arms["R_roll"], arms["P_aci"]])
    roll_bar = judge(arms["R_roll"], s0, [])
    return {
        "gamma": gamma,
        "tuning_err_pp": {g: tuning[g]["M2lm_cond_err_pp"] for g in GAMMAS},
        "n_tune": len(tune),
        "n_test": len(test),
        "arms": arms,
        "aci_bar": aci_bar,
        "aci_pass": all(aci_bar.values()),
        "roll_bar": roll_bar,
        "roll_pass": all(roll_bar.values()),
        "void": void_reasons(len(test), s0, arms["P_aci"]),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voted", default=None)
    ap.add_argument("--horizons", default="3,7,14")
    ap.add_argument("--out", default="measure_aci.csv")
    args = ap.parse_args()

    path = args.voted or default_voted_panel()
    df = load_panel(path)
    floor, cap = sigma_bounds_for_panel(df)
    logger.info("sigma clip: floor=%.6f cap=%.6f (panel's own; read ratios, not levels)", floor, cap)
    fc = ItemForecaster(db_session=None)

    rows, results = [], {}
    for h in [int(x) for x in args.horizons.split(",")]:
        p = CalibPanel(score_frame(fc, df, h, floor, cap), h)
        r = run_horizon(p, grid_date_sets(p.unique_dates, 1)[0])
        results[h] = r
        role = "PRIMARY" if h in PRIMARY_HORIZONS else "report-only"
        logger.info("")
        logger.info(
            "h=%dd (%s): %d tuning + %d scoring dates; gamma=%s (tuning date err: %s)",
            h,
            role,
            r["n_tune"],
            r["n_test"],
            r["gamma"],
            " ".join(f"{g}:{e:.2f}pp" for g, e in r["tuning_err_pp"].items()),
        )
        for name, m in r["arms"].items():
            rows.append({"horizon": h, "scheme": name, "gamma": r["gamma"], **m})
            logger.info(
                "  %-10s M1=%.1f%% | LM date err=%5.2fpp P10 date=%.1f%% w*=%.4f mindec*=%.0f%%",
                name,
                m["M1_marginal"] * 100,
                m["M2lm_cond_err_pp"],
                m["p10_date_cov_lm"] * 100,
                m["mean_width_lm"],
                m["min_decile_cov_lm"] * 100,
            )
        logger.info("  A_aci bar: %s -> %s", r["aci_bar"], "PASS" if r["aci_pass"] else "fail")
        logger.info("  R_roll vs S0: %s -> %s", r["roll_bar"], "PASS" if r["roll_pass"] else "fail")
        for reason in r["void"]:
            logger.warning("  VOID: %s", reason)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    logger.info("")
    logger.info("wrote %s", args.out)

    primary = [results[h] for h in PRIMARY_HORIZONS if h in results]
    if len(primary) < len(PRIMARY_HORIZONS):
        logger.info("VERDICT: not all primary horizons run -- no verdict.")
    elif any(r["void"] for r in primary):
        logger.info("VERDICT: VOID at a primary horizon -- unreadable, not null.")
    elif all(r["aci_pass"] for r in primary):
        logger.info("VERDICT: ACI PASSES at h=3 and h=7 -> wire behind a flag, served confirm next.")
    elif all(r["roll_pass"] for r in primary):
        logger.info("VERDICT: R_roll passes, ACI does not -> shipped batch feedback suffices; close ACI.")
    else:
        logger.info("VERDICT: ACI closed for the band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
