#!/usr/bin/env python3
"""Does served direction clear the preregistered reopening gate?

Instrument for `docs/research/2026-10-08-direction-reopening-preregistration.md`. The window,
the bar and the void conditions are fixed there; this script only computes the numbers they
are applied to. Read the prereg before the output.

Per primary horizon (h=3, h=7), on the first WINDOW_DATES usable forecast dates >= WINDOW_START:

  1. >= 20 usable dates.
  2. Pesaran-Timmermann excess > 0 with t >= PT_T_HURDLE.
  3. DA - realised_down_rate >= 0 (95% CI from a seeded date-level bootstrap, reported only).
  4. Drop the single date with the largest excess; PT stays positive with t >= DROP_ONE_T.

h=14 and h=30 are scored the same way and reported, never judged.

REFUSAL GUARD. A horizon with fewer than WINDOW_DATES usable window dates is refused: the
script prints its date count and computes no PT, DA or verdict for it. Reads are per horizon, so
h=3 can be read while h=7 is still outstanding. The window is the FIRST 20 dates, so a re-run
returns the same answer and later dates never enter it.

Records come from `scripts/backtest_accuracy.py::_records_from_frozen_outcomes` (>= $1, the
documented date exclusions applied), the backtest's own derivation, so they cannot drift from
the published scorer. The prod read runs in a READ ONLY transaction.

The scorer constants the prereg froze (FROZEN_CONSTANTS) are checked first. If one moved the
window is void and nothing is scored.

Verdicts: PASS | INFORMATION_WITHOUT_USABLE_CALL | UNRESOLVED | NULL | VOID | REPORT_ONLY.
UNRESOLVED also covers PT clearing the hurdle while the drop-one-date guard fails (one date
carries it). The prereg's outcome table must list that case; direction stays withheld in it.

    venv/bin/python -m scripts.measure_direction_reopen
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.directional_test import (
    PT_MIN_ROWS_PER_DATE,
    PT_T_HURDLE,
    _excess_hit_rate,
    pesaran_timmermann,
    realised_down_rate,
)
from backtest.scoring import FLAT_TOLERANCE, MIN_HEADLINE_DATES

logger = logging.getLogger(__name__)

WINDOW_START = date(2026, 10, 5)
WINDOW_DATES = 20
PRIMARY_HORIZONS = (3, 7)
REPORT_HORIZONS = (14, 30)
DROP_ONE_T = 2.0
UNRESOLVED_T = 2.0
N_BOOT = 10_000
BOOT_SEED = 0

# What the prereg froze. A mismatch with the live scorer voids the read (prereg, "Truncation").
FROZEN_CONSTANTS = {"PT_T_HURDLE": 3.0, "PT_MIN_ROWS_PER_DATE": 30, "FLAT_TOLERANCE": 0.005}


def constants_drift() -> list[str]:
    """Names of frozen scorer constants whose live value no longer matches the freeze."""
    live = {"PT_T_HURDLE": PT_T_HURDLE, "PT_MIN_ROWS_PER_DATE": PT_MIN_ROWS_PER_DATE, "FLAT_TOLERANCE": FLAT_TOLERANCE}
    return [name for name, frozen in FROZEN_CONSTANTS.items() if live[name] != frozen]


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def select_window(
    records: list[dict], start: date = WINDOW_START, n_dates: int = WINDOW_DATES, min_rows: int = PT_MIN_ROWS_PER_DATE
) -> tuple[list[dict], list[date]]:
    """The first `n_dates` usable forecast dates on or after `start`, and their records.

    A date is usable when it has at least `min_rows` records, as `pesaran_timmermann` requires.
    Fewer than `n_dates` come back when the window has not filled."""
    by_date: dict[date, list[dict]] = defaultdict(list)
    for r in records:
        d = _as_date(r["forecast_date"])
        if d >= start:
            by_date[d].append(r)
    usable = sorted(d for d, rows in by_date.items() if len(rows) >= min_rows)[:n_dates]
    return [r for d in usable for r in by_date[d]], usable


def da_vs_down(records: list[dict], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """DA, the realised down rate (the runnable always-down call), and their gap, in points.

    The 95% interval resamples dates with replacement, since outcomes on one date share a market move."""
    by_date: dict[date, list[dict]] = defaultdict(list)
    for r in records:
        by_date[_as_date(r["forecast_date"])].append(r)
    n = np.array([len(v) for v in by_date.values()], dtype=float)
    hits = np.array([sum(bool(r["direction_correct"]) for r in v) for v in by_date.values()], dtype=float)
    down = np.array([sum(r["actual_direction"] == "down" for r in v) for v in by_date.values()], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(n), size=(n_boot, len(n)))
    diffs = (hits[idx].sum(1) - down[idx].sum(1)) / n[idx].sum(1) * 100.0
    return {
        "da_pp": hits.sum() / n.sum() * 100.0,
        "down_pp": realised_down_rate(records),
        "diff_pp": (hits.sum() - down.sum()) / n.sum() * 100.0,
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
    }


def drop_one_date(records: list[dict]) -> dict:
    """Re-score PT after removing the single date with the largest excess hit rate."""
    by_date: dict[date, list[dict]] = defaultdict(list)
    for r in records:
        by_date[_as_date(r["forecast_date"])].append(r)
    dropped = max(by_date, key=lambda d: _excess_hit_rate(by_date[d]))
    rest = [r for d, rows in by_date.items() if d != dropped for r in rows]
    pt = pesaran_timmermann(rest, MIN_HEADLINE_DATES)
    return {
        "dropped_date": dropped,
        "excess_pp": pt["pt_excess_pp"],
        "t": pt["pt_t_stat"],
        "pt_n_dates": pt["pt_n_dates"],
    }


def judge(pt: dict, da: dict, drop: dict) -> str:
    """Apply the prereg's bar and outcome table to one horizon's numbers."""
    t, excess = pt["pt_t_stat"], pt["pt_excess_pp"]
    if t is None or excess is None:
        return "VOID"  # degenerate: no between-date variation to test
    clears_hurdle = excess > 0 and t >= PT_T_HURDLE
    beats_always_down = da["diff_pp"] >= 0
    survives_drop = drop["excess_pp"] is not None and drop["excess_pp"] > 0 and drop["t"] >= DROP_ONE_T
    if clears_hurdle and beats_always_down and survives_drop:
        return "PASS"
    if clears_hurdle and survives_drop:
        return "INFORMATION_WITHOUT_USABLE_CALL"
    if clears_hurdle or (excess > 0 and t >= UNRESOLVED_T):
        return "UNRESOLVED"
    return "NULL"


def achieved_mde_pp(pt: dict) -> float | None:
    """The excess (pp) that would have cleared the hurdle at this window's standard error."""
    excess, t = pt["pt_excess_pp"], pt["pt_t_stat"]
    if excess is None or t is None or t == 0:
        return None
    return PT_T_HURDLE * abs(excess) / abs(t)


def evaluate_horizon(records: list[dict], horizon: int) -> dict:
    """One horizon's read, or a refusal that carries only its date count."""
    window, dates = select_window(records)
    if len(dates) < WINDOW_DATES:
        return {"horizon": horizon, "status": "NOT_READY", "usable_dates": len(dates), "need": WINDOW_DATES}
    pt = pesaran_timmermann(window, MIN_HEADLINE_DATES)
    da = da_vs_down(window)
    drop = drop_one_date(window)
    if horizon not in PRIMARY_HORIZONS:
        verdict = "REPORT_ONLY"
    elif pt["pt_n_dates"] != WINDOW_DATES:
        verdict = "VOID"
    else:
        verdict = judge(pt, da, drop)
    return {
        "horizon": horizon,
        "status": "READ",
        "window": (dates[0], dates[-1]),
        "pt": pt,
        "da": da,
        "drop": drop,
        "mde_pp": achieved_mde_pp(pt),
        "verdict": verdict,
    }


def _fmt(x, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def report_line(res: dict) -> str:
    h = res["horizon"]
    if res["status"] == "NOT_READY":
        return f"h={h}: NOT READY, {res['usable_dates']}/{res['need']} usable window dates since {WINDOW_START}"
    pt, da, drop = res["pt"], res["da"], res["drop"]
    return (
        f"h={h}: {res['verdict']}  window {res['window'][0]}..{res['window'][1]}  "
        f"PT excess {_fmt(pt['pt_excess_pp'], '+.2f')}pp t={_fmt(pt['pt_t_stat'], '.2f')} "
        f"({pt['pt_n_dates']} dates, hurdle {PT_T_HURDLE})  "
        f"DA {da['da_pp']:.1f}% vs always-down {da['down_pp']:.1f}% "
        f"(gap {da['diff_pp']:+.1f}pp, 95% CI [{da['ci_lo']:+.1f}, {da['ci_hi']:+.1f}])  "
        f"drop-one-date: dropped {drop['dropped_date']} excess {_fmt(drop['excess_pp'], '+.2f')}pp "
        f"t={_fmt(drop['t'], '.2f')}  achieved MDE {_fmt(res['mde_pp'], '.2f')}pp"
    )


def load_records() -> dict[int, list[dict]]:
    """Scored >=$1 outcomes by horizon, all served identities pooled. Read-only."""
    from database import SessionLocal
    from sqlalchemy import text

    from scripts.backtest_accuracy import _records_from_frozen_outcomes

    db = SessionLocal()
    try:
        db.execute(text("SET TRANSACTION READ ONLY"))
        groups = _records_from_frozen_outcomes(db, min_price=1.0)
        db.rollback()
    finally:
        db.close()
    by_horizon: dict[int, list[dict]] = defaultdict(list)
    for (horizon, _identity), recs in groups.items():
        by_horizon[int(horizon)].extend(recs)
    return dict(by_horizon)


def main(argv=None, records_loader: Callable[[], dict[int, list[dict]]] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    drift = constants_drift()
    if drift:
        logger.error(f"VOID: scorer constants changed since the prereg froze them: {', '.join(drift)}. Nothing scored.")
        return 4
    by_horizon = (records_loader or load_records)()
    results = [evaluate_horizon(by_horizon.get(h, []), h) for h in (*PRIMARY_HORIZONS, *REPORT_HORIZONS)]
    for res in results:
        logger.info(report_line(res))
    outstanding = [r["horizon"] for r in results if r["horizon"] in PRIMARY_HORIZONS and r["status"] != "READ"]
    if outstanding:
        logger.info(
            f"Primary horizon(s) still outstanding: {outstanding}. Re-run once they reach {WINDOW_DATES} dates."
        )
        return 3
    logger.info("Both primary horizons read. The window is fixed, so a re-run returns the same answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
