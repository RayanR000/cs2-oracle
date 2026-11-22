#!/usr/bin/env python3
"""Is served interval coverage uneven across price tiers, beyond the width tilt?

Instrument for `docs/research/2026-11-22-tier-conditional-coverage-preregistration.md`. The
window, the bars and the void conditions are fixed there; this script only computes the numbers
they are applied to. Read the prereg before the output.

Per horizon, on the FIRST WINDOW_DATES scoreable forecast dates >= WINDOW_START (served cohort
>= $1, documented date exclusions applied):

  G    coverage(tier 3, $20-100) - coverage(tier 1, $1-5), in percentage points.
  G_w  the same gap inside width terciles, averaged over terciles with enough rows.
  placebo  tier labels permuted within forecast date; must read null or the run is VOID.

Intervals come from a seeded moving-block bootstrap over forecast dates (block length = h dates).
h=3 and h=7 are the primary horizons. h=14 is scored the same way and reported. h=30 is
descriptive only: per-tier coverage, no interval, no verdict.

REFUSAL GUARD. A horizon with fewer than WINDOW_DATES scoreable window dates is refused: the
script prints its date count and computes nothing else. The window is the FIRST 20 dates, so a
re-run returns the same answer and later dates never enter it. One run per horizon is the
protocol; the script cannot enforce that, the prereg does.

The read never writes.

    venv/bin/python -m scripts.tier_coverage_read --horizon 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.scoring import excluded_forecast_date, price_tier

logger = logging.getLogger("tier_coverage_read")

WINDOW_START = date(2026, 11, 19)
WINDOW_DATES = 20  # the project's MIN_HEADLINE_DATES
PRIMARY_HORIZONS = (3, 7)
REPORTED_HORIZONS = (14,)
DESCRIPTIVE_HORIZONS = (30,)

BAR_G_PP = 8.0  # point estimate of G must reach this
BAR_G_LO_PP = 3.0  # and its interval's lower bound must exceed this
BAR_GW_PP = 4.0  # width-controlled gap point estimate
MIN_TIER_ROWS = 200  # per tier within a width tercile
N_BOOT = 2000
N_PLACEBO = 200
CI = 0.90
SEED = 20261122
PLACEBO_MEAN_MAX_PP = 1.0

TIERS = (1, 2, 3)
LOW_TIER, HIGH_TIER = 1, 3
N_TERCILES = 3


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


# --- panel -------------------------------------------------------------------------------------


def load_panel(read: Callable[[str], list], horizon: int, *, window_start: date = WINDOW_START) -> pd.DataFrame:
    """Scoreable served rows for *horizon*: columns date, tier, width, hit. Not yet windowed to 20 dates."""
    rows = read(
        "SELECT forecast_date, current_price, predicted_price_low AS lo, predicted_price_mid AS mid, "
        "predicted_price_high AS hi, in_interval "
        f"FROM forecast_outcomes WHERE horizon_days = {int(horizon)} AND in_interval IS NOT NULL "
        "AND current_price >= 1 AND predicted_price_mid > 0"
    )
    recs = []
    for r in rows:
        d = _as_date(r.forecast_date)
        if d is None or d < window_start or excluded_forecast_date(d) is not None:
            continue
        recs.append(
            (
                d,
                price_tier(float(r.current_price)),
                (float(r.hi) - float(r.lo)) / float(r.mid),
                float(bool(r.in_interval)),
            )
        )
    return pd.DataFrame(recs, columns=["date", "tier", "width", "hit"])


def first_dates(panel: pd.DataFrame, n: int = WINDOW_DATES) -> pd.DataFrame:
    """Rows on the first *n* distinct scoreable dates (the frozen window)."""
    keep = sorted(panel["date"].unique())[:n]
    return panel[panel["date"].isin(keep)].reset_index(drop=True)


def add_terciles(panel: pd.DataFrame) -> pd.DataFrame:
    """Width terciles fixed on the window's own rows."""
    out = panel.copy()
    out["terc"] = pd.qcut(out["width"].rank(method="first"), N_TERCILES, labels=False)
    return out


# --- statistics --------------------------------------------------------------------------------


def _aggregate(panel: pd.DataFrame) -> tuple[list[date], np.ndarray]:
    """hits[d, tier, terc] and counts[d, tier, terc] arrays, one slice per forecast date."""
    dates = sorted(panel["date"].unique())
    idx = {d: i for i, d in enumerate(dates)}
    hits = np.zeros((len(dates), len(TIERS), N_TERCILES))
    cnt = np.zeros_like(hits)
    sub = panel[panel["tier"].isin(TIERS)]
    di = sub["date"].map(idx).to_numpy()
    ti = sub["tier"].to_numpy() - 1
    ci = sub["terc"].to_numpy().astype(int)
    np.add.at(hits, (di, ti, ci), sub["hit"].to_numpy())
    np.add.at(cnt, (di, ti, ci), 1.0)
    return dates, np.stack([hits, cnt])


def _gradient(hits: np.ndarray, cnt: np.ndarray) -> float:
    lo_n, hi_n = cnt[:, LOW_TIER - 1].sum(), cnt[:, HIGH_TIER - 1].sum()
    if lo_n == 0 or hi_n == 0:
        return float("nan")
    return 100.0 * (hits[:, HIGH_TIER - 1].sum() / hi_n - hits[:, LOW_TIER - 1].sum() / lo_n)


def _gradient_width_controlled(hits: np.ndarray, cnt: np.ndarray) -> float:
    gaps = []
    for t in range(N_TERCILES):
        lo_n, hi_n = cnt[:, LOW_TIER - 1, t].sum(), cnt[:, HIGH_TIER - 1, t].sum()
        if lo_n >= MIN_TIER_ROWS and hi_n >= MIN_TIER_ROWS:
            gaps.append(hits[:, HIGH_TIER - 1, t].sum() / hi_n - hits[:, LOW_TIER - 1, t].sum() / lo_n)
    return 100.0 * float(np.mean(gaps)) if gaps else float("nan")


def _stats(agg_slice: np.ndarray) -> tuple[float, float]:
    hits, cnt = agg_slice[0], agg_slice[1]
    return _gradient(hits, cnt), _gradient_width_controlled(hits, cnt)


def block_bootstrap(agg: np.ndarray, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, 2) array of (G, G_w) under a moving-block resample of the date axis."""
    n = agg.shape[1]
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out[b] = _stats(agg[:, idx])
    return out


def interval(draws: np.ndarray, level: float = CI) -> tuple[float, float]:
    tail = (1 - level) / 2
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, tail)), float(np.quantile(finite, 1 - tail))


def placebo(panel: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """G with tier labels permuted within each forecast date."""
    base = panel[panel["tier"].isin(TIERS)]
    groups = [g.index.to_numpy() for _, g in base.groupby("date")]
    tiers0 = base["tier"].to_numpy()
    hit = base["hit"].to_numpy()
    pos = {ix: i for i, ix in enumerate(base.index.to_numpy())}
    group_pos = [np.array([pos[i] for i in g]) for g in groups]
    out = np.empty(n_perm)
    for k in range(n_perm):
        t = tiers0.copy()
        for gp in group_pos:
            t[gp] = rng.permutation(t[gp])
        lo, hi = hit[t == LOW_TIER], hit[t == HIGH_TIER]
        out[k] = 100.0 * (hi.mean() - lo.mean()) if len(lo) and len(hi) else np.nan
    return out


# --- verdict -----------------------------------------------------------------------------------


def verdict(g, g_lo, g_hi, gw, gw_lo, placebo_draws) -> str:
    """PASS | NOT_NEW | KILL | UNDERPOWERED | VOID for one primary horizon (the prereg's bars)."""
    p_lo, p_hi = interval(placebo_draws)
    if not (p_lo <= 0 <= p_hi) or abs(float(np.nanmean(placebo_draws))) > PLACEBO_MEAN_MAX_PP:
        return "VOID"
    if not np.isfinite(g) or (g_lo <= 0 <= g_hi) or g < BAR_G_PP:
        return "KILL"
    if g_lo <= BAR_G_LO_PP:
        return "UNDERPOWERED"
    if np.isfinite(gw) and gw >= BAR_GW_PP and gw_lo > 0:
        return "PASS"
    return "NOT_NEW"


def overall(verdicts: dict[int, str]) -> str:
    """Combine the two primary horizons into the prereg's overall outcome."""
    v = [verdicts.get(h, "MISSING") for h in PRIMARY_HORIZONS]
    if "VOID" in v:
        return "VOID"
    if "KILL" in v:
        return "KILL"
    if all(x == "PASS" for x in v):
        return "PASS"
    if "MISSING" in v:
        return "INCOMPLETE"
    if "NOT_NEW" in v:
        return "NOT_NEW"
    return "UNDERPOWERED"


# --- driver ------------------------------------------------------------------------------------


def coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("tier").agg(coverage=("hit", "mean"), rows=("hit", "size"))
    g["coverage"] = (100 * g["coverage"]).round(1)
    return g


def read_horizon(read: Callable[[str], list], horizon: int, *, min_dates: int = WINDOW_DATES, seed: int = SEED):
    """Score one horizon. Returns a dict; `refused` is set when the window is short. *min_dates* is
    for the synthetic tests only: the CLI never exposes it."""
    panel = load_panel(read, horizon)
    n_dates = panel["date"].nunique() if len(panel) else 0
    if n_dates < min_dates:
        return {"horizon": horizon, "refused": True, "dates": int(n_dates), "need": min_dates}
    panel = add_terciles(first_dates(panel, min_dates))
    out = {"horizon": horizon, "refused": False, "dates": int(panel["date"].nunique()), "table": coverage_table(panel)}
    if horizon in DESCRIPTIVE_HORIZONS:
        out["verdict"] = "REPORT_ONLY"
        return out
    rng = np.random.default_rng(seed)
    _, agg = _aggregate(panel)
    g, gw = _stats(agg)
    draws = block_bootstrap(agg, horizon, N_BOOT, rng)
    g_lo, g_hi = interval(draws[:, 0])
    gw_lo, gw_hi = interval(draws[:, 1])
    plc = placebo(panel, N_PLACEBO, rng)
    out.update(g=g, g_ci=(g_lo, g_hi), gw=gw, gw_ci=(gw_lo, gw_hi), placebo_mean=float(np.nanmean(plc)))
    out["placebo_ci"] = interval(plc)
    out["verdict"] = verdict(g, g_lo, g_hi, gw, gw_lo, plc) if horizon in PRIMARY_HORIZONS else "REPORT_ONLY"
    return out


def render(res: dict) -> str:
    h = res["horizon"]
    if res["refused"]:
        return f"h={h}: REFUSED. {res['dates']} scoreable window dates, need {res['need']}. Nothing computed."
    lines = [f"h={h}: {res['dates']} dates (first {WINDOW_DATES} on or after {WINDOW_START})", res["table"].to_string()]
    if "g" in res:
        lines += [
            f"G   = {res['g']:+.1f}pp  {int(CI * 100)}% CI [{res['g_ci'][0]:+.1f}, {res['g_ci'][1]:+.1f}]",
            f"G_w = {res['gw']:+.1f}pp  {int(CI * 100)}% CI [{res['gw_ci'][0]:+.1f}, {res['gw_ci'][1]:+.1f}]",
            f"placebo G mean {res['placebo_mean']:+.2f}pp, {int(CI * 100)}% interval "
            f"[{res['placebo_ci'][0]:+.1f}, {res['placebo_ci'][1]:+.1f}]",
        ]
    lines.append(f"verdict: {res['verdict']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--horizon", type=int, required=True, choices=PRIMARY_HORIZONS + REPORTED_HORIZONS + DESCRIPTIVE_HORIZONS
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from db import ops_store

    res = read_horizon(ops_store.reader(("forecast_outcomes",)), args.horizon)
    print(render(res))
    return 2 if res["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
