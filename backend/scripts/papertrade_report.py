"""Report the paper-trading P&L of the served forecasts, net of real venue friction.

Reads the frozen ``forecast_outcomes`` panel read-only, replays each executable
forecast as a long round trip through ``backtest/papertrade.py``, and prints — per
horizon, per venue, per strategy — what the calls would have earned after fees and
the tier's bid-ask spread. Writes nothing.

    venv/bin/python -m scripts.papertrade_report              # >=$1 cohort, cash venues
    venv/bin/python -m scripts.papertrade_report --by-tier    # break out by price tier

**This reads whatever DATABASE_URL resolves to** — run from ``backend/`` it hits
production (read-only); the local DB is a synthetic fixture whose prices are
incoherent, so its numbers mean nothing. The environment is echoed in the header so
the source is never ambiguous.

The expected result is that no cash strategy clears friction — this is a
range forecaster (``AGENTS.md``), and the harness exists to price that, not to beat
it. Steam is reported on its own line and labelled WALLET-ONLY: its proceeds are not
cashable, so its return is never a cash P&L.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.papertrade import (
    PAPER_TRADE_HORIZONS,
    WALLED_VENUES,
    all_strategies,
)
from backtest.scoring import (
    MIN_FORECAST_DATES,
    excluded_forecast_date,
    price_tier,
)
from config import settings
from database import SessionLocal

# Cash venues to report, cheapest first. Steam is handled apart (walled).
CASH_VENUES = ("csfloat", "skinport")

_SELECT = """
    SELECT o.item_id, o.horizon_days, o.forecast_date,
           o.base_price, o.actual_price, o.current_price,
           o.predicted_price_low, o.predicted_price_mid, o.base_stale_run_days
    FROM forecast_outcomes o
    -- Pin the row order: the CI bootstrap is seeded, so an unordered scan can
    -- return a different `pt_profitable` on a strict `lower > 0` test run to run.
    -- The table is unique on (item_id, forecast_date, horizon_days).
    ORDER BY o.forecast_date, o.item_id, o.horizon_days
"""


def _load_records(db, min_price: float) -> list[dict]:
    """Frozen outcomes as papertrade records, on the resolver basis.

    Drops the same rows the scorer drops — non-positive base/actual, missing mid,
    below the price floor, and forecast dates served by a superseded direction rule
    — so the P&L describes the same cohort the accuracy headline does. Renames the
    DB's ``predicted_price_*`` columns to the sibling record shape
    (``predicted_mid`` / ``predicted_low``).
    """
    records = []
    for r in db.execute(text(_SELECT)).fetchall():
        base, actual, mid = r.base_price, r.actual_price, r.predicted_price_mid
        if base is None or base <= 0 or actual is None or actual <= 0 or mid is None:
            continue
        if base < min_price:
            continue
        if excluded_forecast_date(r.forecast_date):
            continue
        records.append(
            {
                "item_id": r.item_id,
                "horizon_days": r.horizon_days,
                "forecast_date": r.forecast_date,
                "base_price": base,
                "actual_price": actual,
                "current_price": r.current_price,
                "predicted_mid": mid,
                "predicted_low": r.predicted_price_low,
                "price_tier": price_tier(base),
                "base_stale_run_days": r.base_stale_run_days,
            }
        )
    return records


def _fmt(v, suffix="") -> str:
    return "   —  " if v is None else f"{v}{suffix}"


def _print_block(records: list[dict], horizon: int, venue: str, mode: str) -> None:
    walled = venue in WALLED_VENUES
    tag = "  [WALLET-ONLY — NOT CASH]" if walled else ""
    print(f"\n  venue={venue}{tag}")
    print(
        f"    {'strategy':<14}{'n/cand':>10}{'mean net%':>12}{'win%':>8}{'CI low%':>10}{'CI high%':>10}{'profit?':>9}"
    )
    out = all_strategies(records, horizon, MIN_FORECAST_DATES, venue=venue, mode=mode)
    for name in ("buy_and_hold", "exceedance", "q50_clears"):
        m = out[name]
        if m["pt_scope"] == "out_of_scope":
            continue
        nc = "  —  " if m["pt_n"] is None else f"{m['pt_n']}/{m['pt_n_candidates']}"
        print(
            f"    {name:<14}{nc:>10}{_fmt(m['pt_mean_net_pct']):>12}"
            f"{_fmt(m['pt_win_rate_pct']):>8}{_fmt(m['pt_mean_net_ci_lower']):>10}"
            f"{_fmt(m['pt_mean_net_ci_upper']):>10}"
            f"{_fmt(m['pt_profitable']):>9}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-price", type=float, default=1.0, help="price floor in USD (default 1.0, the >=$1 headline cohort)"
    )
    ap.add_argument("--by-tier", action="store_true", help="break results out by price tier as well as pooled")
    ap.add_argument(
        "--sell-only",
        action="store_true",
        help="charge sell-side friction only (fee + half spread), as "
        "when timing the sale of inventory you already hold",
    )
    args = ap.parse_args()
    mode = "sell_only" if args.sell_only else "round_trip"

    db = SessionLocal()
    try:
        records = _load_records(db, args.min_price)
    finally:
        db.close()

    cost_desc = (
        "fee + HALF spread (sell-only: inventory buy is sunk)"
        if mode == "sell_only"
        else "fee + full spread (round trip)"
    )
    print(
        f"paper-trade report  env={settings.environment}  mode={mode}  "
        f"min_price=${args.min_price:g}  n_records={len(records)}"
    )
    print(f"P&L on the resolver basis (buy base_price, sell actual_price) net of {cost_desc}. Read-only.")

    by_h = defaultdict(list)
    for r in records:
        by_h[r["horizon_days"]].append(r)

    for horizon in sorted(h for h in PAPER_TRADE_HORIZONS if h in by_h):
        rows = by_h[horizon]
        print(f"\n{'=' * 72}\nhorizon={horizon}d   n={len(rows)}")
        for venue in (*CASH_VENUES, *sorted(WALLED_VENUES)):
            _print_block(rows, horizon, venue, mode)
            if args.by_tier:
                by_tier = defaultdict(list)
                for r in rows:
                    by_tier[r["price_tier"]].append(r)
                for tier in sorted(by_tier):
                    print(f"      tier {tier} (n={len(by_tier[tier])}):")
                    _print_block(by_tier[tier], horizon, venue, mode)


if __name__ == "__main__":
    main()
