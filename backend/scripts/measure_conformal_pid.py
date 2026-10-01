#!/usr/bin/env python3
"""Does conformal PID beat the batch served-feedback factor on per-date coverage?

Instrument for `docs/research/2026-09-23-conformal-pid-served-h3-preregistration.md`.
The arms, the bar and the void conditions are fixed there (including its Amendments);
this script only computes the numbers they are applied to. Read the prereg before the
output.

Scored by COUNTERFACTUAL REPLAY on served h=3 rows. `band_signed` scales q_lo/q_hi by the
multiplier, so with `r_base = r_stored * m_row` a row would have been covered under any
multiplier `m` iff `r_base <= m`. That is the dollar-band predicate the feedback factor is
fitted on (`served_recalibration.factors_from_panel`), so every arm is scored on the same
rows by the same predicate, and no prod change is needed.

Each arm is sequential -- the m served on date d depends on every date whose outcomes
had resolved by d (target + RESOLUTION_LAG_DAYS) -- so each arm's per-date m is computed
in date order, one update per resolved date (a date is a step, a row is not).

ARMS
  F1    fixed m = 1.0. Reference only.
  B     incumbent: the fixed batch refit. Activates on the first serve date with
        >= MIN_FEEDBACK_DATES resolved dates, then refits on Mondays (mode=full),
        via `served_recalibration.factors_from_panel` itself.
  Q     quantile tracker (P):  q <- q + ETA * (err - ALPHA).
  QI    candidate (P + I):     m = q + KI * tan(S * log(k+1) / (CSAT * (k+1))),
        S = sum(err - ALPHA), k = updates so far -- `saturation_fn_log` from
        aangelopoulos/conformal-time-series core/methods.py, ported (the reference uses
        np.infty, gone in NumPy 2, and one score per step).
  P_QI  placebo, open-loop: QI's own err values, permuted (PLACEBO_SEED), replayed on
        QI's update schedule. Closed-loop is undefined: err depends on the band served.

Every online arm starts at m = 1.0 and is served clipped to CLAMP; the unclipped state
evolves as in the reference.

Read-only. Refuses to read prod before READ_NOT_BEFORE (the prereg's single read).

    venv/bin/python -m scripts.measure_conformal_pid            # after 2026-10-23
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.scoring import HEADLINE_MIN_TIER, MIN_FEEDBACK_DATES
from models import served_recalibration

logger = logging.getLogger("measure_conformal_pid")

# Frozen by the prereg. Do not tune.
HORIZON = 3
ALPHA = 0.20
ETA = 0.5
CSAT = 1.0
KI = 0.3
CLAMP = (0.5, 2.0)
RESOLUTION_LAG_DAYS = 1
WINDOW = (date(2026, 9, 7), date(2026, 10, 18))
READ_NOT_BEFORE = date(2026, 10, 23)
BLEND_WEIGHT = 0.15  # ItemForecaster.FORECAST_BLEND_WEIGHT
PLACEBO_SEED = 0
N_BOOT = 10_000
BOOT_SEED = 0
N_DECILES = 10

# Bar (prereg "Bar (fixed now)").
M1_TOL_PP = 3.0
PRIMARY_MARGIN_PP = 1.0
P10_MARGIN_PP = 3.0
WIDTH_RATIO_MAX = 1.05
DECILE_DROP_MAX_PP = 5.0
MIN_SCORE_DATES = 25
B_MIN_PRIMARY_PP = 3.0


# --------------------------------------------------------------------------- panel


@dataclass
class Panel:
    """Served h=3 rows grouped by forecast date, in date order."""

    dates: list[date]
    r_base: list[np.ndarray]  # score against the un-multiplied band, per date
    width: list[np.ndarray]  # base relative half-width (served-scale proxy), per date
    blend_delta: list[np.ndarray]  # relative bound on r_base from the prior-row blend
    frames: list[pd.DataFrame] = field(repr=False, default_factory=list)  # for B's refit


def build_panel(df: pd.DataFrame, horizon: int = HORIZON, window=WINDOW) -> Panel:
    """Group a served-outcome frame into a Panel.

    Needs `forecast_date, horizon_days, price_tier, predicted_price_low/mid/high,
    actual_price, band_multiplier`; optional `prior_width_ratio` (B_prior / B_today) and
    `prior_multiplier` feed the blend bound (0 when absent). Keeps >= $1 rows with a
    usable band and a known multiplier, inside `window`.
    """
    d = df[(df["horizon_days"].to_numpy() == horizon) & (df["price_tier"].to_numpy() >= HEADLINE_MIN_TIER)].copy()
    d["forecast_date"] = pd.to_datetime(d["forecast_date"]).dt.date
    d = d[(d["forecast_date"] >= window[0]) & (d["forecast_date"] <= window[1])]

    mid = d["predicted_price_mid"].to_numpy(dtype=float)
    low = d["predicted_price_low"].to_numpy(dtype=float)
    high = d["predicted_price_high"].to_numpy(dtype=float)
    actual = d["actual_price"].to_numpy(dtype=float)
    mult = d["band_multiplier"].to_numpy(dtype=float)
    above = actual >= mid
    half = np.where(above, high - mid, mid - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_base = np.where(above, actual - mid, mid - actual) / half * mult
        width = (high - low) / 2.0 / mid / mult
    ratio = d["prior_width_ratio"].to_numpy(dtype=float) if "prior_width_ratio" in d else np.ones(len(d))
    m_prior = d["prior_multiplier"].to_numpy(dtype=float) if "prior_multiplier" in d else np.zeros(len(d))
    with np.errstate(invalid="ignore"):
        delta = np.nan_to_num(BLEND_WEIGHT * m_prior * np.abs(ratio - 1.0) / mult, nan=0.0)

    keep = np.isfinite(r_base) & (half > 0) & np.isfinite(width) & (mid > 0)
    d = d[keep].assign(_r=r_base[keep], _w=width[keep], _delta=delta[keep])
    dates, rs, ws, ds, frames = [], [], [], [], []
    for fd, g in d.groupby("forecast_date", sort=True):
        dates.append(fd)
        rs.append(g["_r"].to_numpy())
        ws.append(g["_w"].to_numpy())
        ds.append(g["_delta"].to_numpy())
        frames.append(g)
    return Panel(dates, rs, ws, ds, frames)


def resolved_by(forecast_date: date, horizon: int = HORIZON) -> date:
    """The first serve date on which `forecast_date`'s outcomes can inform the band."""
    return forecast_date + timedelta(days=horizon + RESOLUTION_LAG_DAYS)


def perturbed(panel: Panel, sign: int) -> Panel:
    """The panel with every r_base moved by its blend bound (sign +1 / -1)."""
    rs = [r * (1.0 + sign * dl) for r, dl in zip(panel.r_base, panel.blend_delta)]
    return Panel(panel.dates, rs, panel.width, panel.blend_delta, panel.frames)


# ---------------------------------------------------------------------- controllers


def _tan(x: float) -> float:
    """`mytan` from the reference: saturates to +-inf outside (-pi/2, pi/2)."""
    if x >= math.pi / 2:
        return math.inf
    if x <= -math.pi / 2:
        return -math.inf
    return math.tan(x)


def saturation_log(s: float, k: int, csat: float = CSAT, ki: float = KI) -> float:
    """`saturation_fn_log(x, t, Csat, KI)` from the reference."""
    if ki == 0:
        return 0.0
    return ki * _tan(s * math.log(k + 1) / (csat * (k + 1)))


class PID:
    """Quantile tracker (+ optional log-saturated integrator) in multiplier space."""

    def __init__(self, eta: float = ETA, ki: float = KI, csat: float = CSAT, integrate: bool = True):
        self.eta, self.ki, self.csat, self.integrate = eta, ki, csat, integrate
        self.q = 1.0
        self.s = 0.0
        self.k = 0

    def update(self, err: float) -> None:
        g = err - ALPHA
        self.q += self.eta * g
        self.s += g
        self.k += 1

    @property
    def raw(self) -> float:
        i = saturation_log(self.s, self.k, self.csat, self.ki) if (self.integrate and self.k > 0) else 0.0
        return self.q + i

    @property
    def served(self) -> float:
        return float(np.clip(self.raw, *CLAMP))


def miss_rate(r: np.ndarray, m: float) -> float:
    return float(np.mean(r > m))


def _pending(panel: Panel, i: int, applied: int) -> list[int]:
    """Indices of dates resolved by panel.dates[i] and not yet applied (date order)."""
    d = panel.dates[i]
    out = []
    j = applied
    while j < len(panel.dates) and resolved_by(panel.dates[j]) <= d:
        out.append(j)
        j += 1
    return out


def run_fixed(panel: Panel, m: float = 1.0) -> dict:
    return {"name": "F1", "m": np.full(len(panel.dates), m), "clamp_hits": 0}


def run_online(panel: Panel, integrate: bool, name: str) -> dict:
    """Closed-loop PID replay. Returns per-date served m, the err sequence it consumed and
    the schedule (how many updates had been applied before each serve date)."""
    ctl = PID(integrate=integrate)
    m = np.empty(len(panel.dates))
    errs: list[float] = []
    schedule: list[int] = []
    applied = 0
    clamp_hits = 0
    for i in range(len(panel.dates)):
        for j in _pending(panel, i, applied):
            e = miss_rate(panel.r_base[j], m[j])
            errs.append(e)
            ctl.update(e)
            applied = j + 1
        schedule.append(len(errs))
        m[i] = ctl.served
        clamp_hits += int(ctl.raw != m[i])
    return {"name": name, "m": m, "errs": errs, "schedule": schedule, "clamp_hits": clamp_hits}


def run_placebo(panel: Panel, qi: dict, seed: int = PLACEBO_SEED) -> dict:
    """Open-loop: QI's own err values, permuted, on QI's schedule."""
    errs = np.random.default_rng(seed).permutation(np.asarray(qi["errs"], dtype=float))
    ctl = PID(integrate=True)
    m = np.empty(len(panel.dates))
    fed = 0
    clamp_hits = 0
    for i, n in enumerate(qi["schedule"]):
        while fed < n:
            ctl.update(float(errs[fed]))
            fed += 1
        m[i] = ctl.served
        clamp_hits += int(ctl.raw != m[i])
    return {"name": "P_QI", "m": m, "errs": list(errs), "clamp_hits": clamp_hits}


def run_batch(panel: Panel, min_dates: int = MIN_FEEDBACK_DATES) -> dict:
    """Incumbent: `factors_from_panel` on the resolved rows (base-band scores), activating at
    the first serve date with >= min_dates resolved dates, refitting on Mondays."""
    m = np.empty(len(panel.dates))
    current = 1.0
    active = False
    applied = 0
    for i, d in enumerate(panel.dates):
        applied = max([applied] + [j + 1 for j in _pending(panel, i, applied)])
        if applied >= min_dates and (not active or d.weekday() == 0):
            resolved = pd.concat(panel.frames[:applied])
            # r_base is already on the base band; hand the refit multiplier-1 rows scored on it.
            fit = served_recalibration.factors_from_panel(
                _as_base_frame(resolved), [HORIZON], min_dates=min_dates, since=None
            )
            if HORIZON in fit:
                current = fit[HORIZON]
                active = True
        m[i] = current
    return {"name": "B", "m": m, "clamp_hits": 0}


def _as_base_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Rows re-expressed so factors_from_panel's r equals our r_base (band half = 1, mult 1)."""
    r = g["_r"].to_numpy()
    return pd.DataFrame(
        {
            "forecast_date": g["forecast_date"].to_numpy(),
            "horizon_days": HORIZON,
            "price_tier": HEADLINE_MIN_TIER,
            "predicted_price_low": -1.0,
            "predicted_price_mid": 0.0,
            "predicted_price_high": 1.0,
            "actual_price": r,  # actual >= mid, so r = actual / 1
            "band_multiplier": 1.0,
        }
    )


# -------------------------------------------------------------------------- scoring


def score(panel: Panel, arm: dict) -> dict:
    cov = np.array([1.0 - miss_rate(r, m) for r, m in zip(panel.r_base, arm["m"])])
    hits = sum(float(np.sum(r <= m)) for r, m in zip(panel.r_base, arm["m"]))
    n = sum(r.size for r in panel.r_base)
    return {
        "name": arm["name"],
        "n_dates": len(cov),
        "cov_by_date": cov,
        "primary_pp": float(np.mean(np.abs(cov - (1 - ALPHA)))) * 100,
        "p10_pp": float(np.percentile(cov, 10)) * 100,
        "m1_pp": hits / n * 100 if n else float("nan"),
        "mean_mult": float(np.mean(arm["m"])),
        "clamp_hits": arm["clamp_hits"],
        "decile_cov_pp": decile_coverage(panel, arm["m"]),
    }


def decile_edges(panel: Panel) -> np.ndarray:
    w = np.concatenate(panel.width)
    return np.quantile(w, np.linspace(0, 1, N_DECILES + 1)[1:-1])


def decile_coverage(panel: Panel, m: np.ndarray) -> np.ndarray:
    edges = decile_edges(panel)
    hits = np.zeros(N_DECILES)
    tot = np.zeros(N_DECILES)
    for r, w, mi in zip(panel.r_base, panel.width, m):
        b = np.searchsorted(edges, w, side="right")
        np.add.at(tot, b, 1)
        np.add.at(hits, b, (r <= mi).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return hits / tot * 100


def paired_bootstrap(a_cov: np.ndarray, b_cov: np.ndarray, n: int = N_BOOT, seed: int = BOOT_SEED):
    """Mean and 95% CI over dates of |cov_a - .8| - |cov_b - .8|, in pp."""
    diff = (np.abs(a_cov - (1 - ALPHA)) - np.abs(b_cov - (1 - ALPHA))) * 100
    idx = np.random.default_rng(seed).integers(0, diff.size, size=(n, diff.size))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def judge(qi: dict, b: dict, placebo: dict, ci: tuple) -> dict:
    """Bar (1)-(5). Returns each criterion and PASS / FAIL / UNRESOLVED."""
    _, _lo, hi = ci
    c = {
        "1_m1_in_band": abs(qi["m1_pp"] - 80.0) <= M1_TOL_PP,
        "2a_primary_margin": qi["primary_pp"] <= b["primary_pp"] - PRIMARY_MARGIN_PP,
        "2b_ci_excludes_0": hi < 0,
        "2c_below_placebo": qi["primary_pp"] < placebo["primary_pp"],
        "3_p10": qi["p10_pp"] >= b["p10_pp"] + P10_MARGIN_PP,
        "4_width": qi["mean_mult"] <= WIDTH_RATIO_MAX * b["mean_mult"],
        "5_deciles": bool(np.nanmin(qi["decile_cov_pp"] - b["decile_cov_pp"]) >= -DECILE_DROP_MAX_PP),
    }
    if all(c.values()):
        verdict = "PASS"
    elif not c["2b_ci_excludes_0"] and all(v for k, v in c.items() if k != "2b_ci_excludes_0"):
        verdict = "UNRESOLVED"
    else:
        verdict = "FAIL"
    return {"criteria": c, "verdict": verdict, "ci_pp": ci}


def void_reasons(n_dates: int, b: dict, placebo: dict, verdict: str, perturbed_verdicts: list[str]) -> list[str]:
    out = []
    if n_dates < MIN_SCORE_DATES:
        out.append(f"only {n_dates} scoreable dates (< {MIN_SCORE_DATES})")
    if b["primary_pp"] < B_MIN_PRIMARY_PP:
        out.append(f"B primary error {b['primary_pp']:.2f}pp < {B_MIN_PRIMARY_PP} (no time variation)")
    if placebo["primary_pp"] <= b["primary_pp"] - PRIMARY_MARGIN_PP:
        out.append("placebo also clears (2a) against B")
    if any(v != verdict for v in perturbed_verdicts):
        out.append(f"blend sensitivity: verdict {verdict} flips to {perturbed_verdicts} under +-blend bound")
    return out


def evaluate(panel: Panel) -> dict:
    """Every arm, the judgement and the void checks for one panel."""

    def once(p: Panel):
        arms = {"F1": run_fixed(p), "B": run_batch(p), "Q": run_online(p, False, "Q")}
        arms["QI"] = run_online(p, True, "QI")
        arms["P_QI"] = run_placebo(p, arms["QI"])
        s = {k: score(p, a) for k, a in arms.items()}
        ci = paired_bootstrap(s["QI"]["cov_by_date"], s["B"]["cov_by_date"])
        return s, judge(s["QI"], s["B"], s["P_QI"], ci)

    scores, j = once(panel)
    alt = [once(perturbed(panel, sign))[1]["verdict"] for sign in (+1, -1)]
    q_only = judge(
        scores["Q"],
        scores["B"],
        scores["P_QI"],
        paired_bootstrap(scores["Q"]["cov_by_date"], scores["B"]["cov_by_date"]),
    )
    return {
        "scores": scores,
        "judgement": j,
        "q_judgement": q_only,
        "void": void_reasons(len(panel.dates), scores["B"], scores["P_QI"], j["verdict"], alt),
    }


# --------------------------------------------------------------------------- loader


def load_prod_panel(session) -> pd.DataFrame:
    """Served h=3 outcomes plus each row's prior-row blend inputs. Read-only."""
    from sqlalchemy import text

    lo = WINDOW[0] - timedelta(days=14)  # priors for the first window dates
    rows = session.execute(
        text(
            "SELECT o.item_id, o.forecast_date, o.horizon_days, "
            "COALESCE(o.base_price, o.current_price) AS tier_price, "
            "o.predicted_price_low, o.predicted_price_mid, o.predicted_price_high, o.actual_price, "
            "f.band_multiplier "
            "FROM forecast_outcomes o JOIN item_forecasts f ON f.id = o.forecast_id "
            "WHERE o.horizon_days = :h AND o.forecast_date BETWEEN :a AND :b "
            "AND o.actual_price IS NOT NULL AND o.predicted_price_mid IS NOT NULL"
        ),
        {"h": HORIZON, "a": WINDOW[0], "b": WINDOW[1]},
    ).mappings()
    df = pd.DataFrame(rows)
    fc = pd.DataFrame(
        session.execute(
            text(
                # item_forecasts names these price_low/mid/high; only forecast_outcomes
                # carries the predicted_price_* spelling attach_priors reads.
                "SELECT item_id, forecast_date, price_low AS predicted_price_low, "
                "price_mid AS predicted_price_mid, price_high AS predicted_price_high, "
                "band_multiplier FROM item_forecasts "
                "WHERE horizon_days = :h AND forecast_date BETWEEN :a AND :b"
            ),
            {"h": HORIZON, "a": lo, "b": WINDOW[1]},
        ).mappings()
    )
    return attach_priors(df, fc)


def attach_priors(outcomes: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Resolve multipliers and tiers, and attach each row's latest prior forecast row
    (B_prior / B_today and m_prior) for the blend bound."""
    from backtest.scoring import price_tier

    df = outcomes.copy()
    df["price_tier"] = [price_tier(float(v or 0.0)) for v in df["tier_price"]]
    df["band_multiplier"] = [
        served_recalibration.resolve_band_multiplier(fd, None if pd.isna(m) else m)
        for fd, m in zip(df["forecast_date"], df["band_multiplier"])
    ]
    fc = forecasts.copy()
    fc["m"] = [
        served_recalibration.resolve_band_multiplier(fd, None if pd.isna(m) else m)
        for fd, m in zip(fc["forecast_date"], fc["band_multiplier"])
    ]
    fc["B"] = (fc["predicted_price_high"] - fc["predicted_price_low"]) / 2.0 / fc["predicted_price_mid"] / fc["m"]
    fc = fc.sort_values(["item_id", "forecast_date"])
    fc["prior_B"] = fc.groupby("item_id")["B"].shift(1)
    fc["prior_multiplier"] = fc.groupby("item_id")["m"].shift(1)
    keyed = fc.set_index(["item_id", "forecast_date"])[["B", "prior_B", "prior_multiplier"]]
    df = df.join(keyed, on=["item_id", "forecast_date"])
    df["prior_width_ratio"] = (df["prior_B"] / df["B"]).fillna(1.0)
    df["prior_multiplier"] = df["prior_multiplier"].fillna(0.0)
    return df


def report(result: dict) -> str:
    lines = []
    for k in ("F1", "B", "Q", "QI", "P_QI"):
        s = result["scores"][k]
        lines.append(
            f"  {k:<5} dates={s['n_dates']:>3}  primary={s['primary_pp']:6.2f}pp  P10={s['p10_pp']:6.2f}  "
            f"M1={s['m1_pp']:6.2f}  mean_m={s['mean_mult']:.4f}  clamps={s['clamp_hits']}"
        )
    j = result["judgement"]
    lines.append(f"  QI - B primary: {j['ci_pp'][0]:+.2f}pp  95% CI [{j['ci_pp'][1]:+.2f}, {j['ci_pp'][2]:+.2f}]")
    lines.append(f"  criteria: {j['criteria']}")
    lines.append(f"  QI verdict: {j['verdict']}   Q verdict: {result['q_judgement']['verdict']}")
    lines.append(f"  VOID: {result['void']}" if result["void"] else "  void checks: none")
    return "\n".join(lines)


def main(argv=None, today: date | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    today = today or date.today()
    if today < READ_NOT_BEFORE:
        logger.error(
            f"Refusing to read the served panel before {READ_NOT_BEFORE} (prereg: one read, after the "
            f"window {WINDOW[0]}..{WINDOW[1]} resolves). Use the synthetic tests until then."
        )
        return 2
    from database import SessionLocal

    db = SessionLocal()
    try:
        panel = build_panel(load_prod_panel(db))
    finally:
        db.close()
    logger.info(report(evaluate(panel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
