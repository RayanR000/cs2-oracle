#!/usr/bin/env python3
"""Archive-basis confirmation: does h=14 wedge-free ranking survive on served rows?

Implements `docs/research/2026-09-13-archive-basis-centre-rank-preregistration.md`.
Read-only throughout; no retrain, no serving change on any outcome.

The previous served leg VOIDed because the served quote cannot referee ranking: the
serve-time quote is a lagging median-of-3 whose gap to the contemporaneous voted price
is +0.60 rank-correlated with trailing 1d returns, which boosts momentum and penalises
reversal — and the 2025 edge is built on reversal features. So here ALL THREE legs are
anchored on the archive voted composite at the forecast date:

    r_hat    = predicted_price_mid / A(d) - 1      (charges the model for the wedge)
    realized = A(d+14*) / A(d) - 1
    naive    = -log(A(d) / A(d-1))                 (calendar-exact consecutive days)

`d+14*` is the nearest available NON-FROZEN archive day within +-2 of d+14 (ties toward
the earlier day). Whole frozen DAYS are barred as anchors; frozen ROWS are never dropped
— that is a cohort change that selects volatile items and it flipped the h=3 climatology
verdict spuriously.

Usage:
    venv/bin/python -m scripts.archive.archive_basis_centre_rank --gate
    venv/bin/python -m scripts.archive.archive_basis_centre_rank \\
        --archive-dir ../price-archive --out /tmp/abcr_h14.json
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scripts.archive.clean_era_centre_ab import bootstrap_ci, paired_delta_ci, per_date_ic
from scripts.archive.served_centre_rank import DURABLE_ARCHIVE, derive_panel, load_served

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("archive_basis_centre_rank")

HORIZON = 14
MIN_DATES = 20
MIN_ROWS_PER_DATE = 10
ANCHOR_TOL_DAYS = 2
FROZEN_THRESHOLD = 0.70
FROZEN_SENSITIVITY = 0.50


# --------------------------------------------------------------------------
# Pure functions (unit-tested; see tests/test_archive_basis_centre_rank.py)
# --------------------------------------------------------------------------


def resolve_forward_anchor(target, available, frozen, tol=ANCHOR_TOL_DAYS):
    """Nearest available non-frozen archive day to `target` within +-tol.

    Ties break toward the EARLIER day — the shorter, more conservative horizon.
    Returns None when nothing in the window qualifies.
    """
    avail = set(available) - set(frozen)
    best = None
    for off in range(-tol, tol + 1):
        day = target + dt.timedelta(days=off)
        if day not in avail:
            continue
        key = (abs(off), day)  # nearest first, then earliest
        if best is None or key < best[0]:
            best = (key, day)
    return None if best is None else best[1]


def frozen_anchor_dates(voted, threshold=FROZEN_THRESHOLD):
    """Archive days whose exactly-unchanged share is >= threshold.

    Only consecutive-day pairs carry evidence: after a gap, "unchanged" says
    nothing about a daily freeze. `voted` needs item_id / date / price.
    """
    df = voted[["item_id", "date", "price"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item_id", "date"])
    prev_price = df.groupby("item_id")["price"].shift(1)
    prev_date = df.groupby("item_id")["date"].shift(1)
    consec = (df["date"] - prev_date).dt.days == 1
    df = df[consec].copy()
    if df.empty:
        return set()
    df["frozen"] = np.isclose(df["price"], prev_price[consec], rtol=1e-9)
    share = df.groupby(df["date"].dt.date)["frozen"].mean()
    return {d for d, v in share.items() if v >= threshold}


def archive_legs(panel, price_map, horizon, available, frozen, tol=ANCHOR_TOL_DAYS):
    """Attach archive-anchored r_hat / realized / naive; drop rows missing any.

    `price_map` maps (slug, date) -> voted archive price. Pairing is on identical
    rows by construction: a row that cannot form all three legs leaves both arms.
    """
    available = set(available)
    frozen = set(frozen)
    fwd_cache = {}
    out = []
    for row in panel.itertuples(index=False):
        d = row.forecast_date
        if d in frozen:
            continue
        anchor = price_map.get((row.slug, d))
        prev = price_map.get((row.slug, d - dt.timedelta(days=1)))
        if anchor is None or prev is None or anchor <= 0 or prev <= 0:
            continue
        target = d + dt.timedelta(days=horizon)
        if target not in fwd_cache:
            fwd_cache[target] = resolve_forward_anchor(target, available, frozen, tol=tol)
        fwd_day = fwd_cache[target]
        if fwd_day is None:
            continue
        fwd = price_map.get((row.slug, fwd_day))
        if fwd is None or fwd <= 0:
            continue
        rec = row._asdict()
        rec["anchor"] = anchor
        rec["r_hat"] = row.predicted_price_mid / anchor - 1.0
        rec["realized"] = fwd / anchor - 1.0
        rec["naive"] = -float(np.log(anchor / prev))
        rec["fwd_offset"] = (fwd_day - target).days
        out.append(rec)
    return pd.DataFrame(out)


def _positive(ci):
    return ci is not None and ci["ci_low"] > 0


def evaluate_bars(model_ci, paired_ci, n_dates, min_dates=MIN_DATES):
    """Prereg bars. VOID is absolute: below min_dates no statistic is read.

    Returns a dict carrying the verdict, and the statistics ONLY when the power
    gate cleared — so a VOID result cannot be quietly read as underpowered.
    """
    if n_dates < min_dates:
        return {
            "verdict": "VOID",
            "n_dates": n_dates,
            "reason": f"{n_dates} qualifying dates < {min_dates}; no statistic read (prereg bar)",
        }
    if model_ci is None or paired_ci is None:
        return {"verdict": "VOID", "n_dates": n_dates, "reason": "a required interval could not be formed"}
    verdict = "CONFIRMED" if (_positive(model_ci) and _positive(paired_ci)) else "KILL"
    return {"verdict": verdict, "n_dates": n_dates, "model_ic": model_ci, "paired": paired_ci}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_voted(slugs, day_min, day_max, archive_dir):
    """Voted archive composite over the panel slugs. Read-only DuckDB."""
    import duckdb
    from db.archive import prices_relation
    from models.forecaster import ItemForecaster
    from models.item_parser import archive_universe_sql_filter

    con = duckdb.connect()
    try:
        rel = prices_relation(con, archive_dir=str(archive_dir))
        uni = archive_universe_sql_filter()
        lo = (pd.Timestamp(day_min) - pd.to_timedelta(5, unit="D")).date()
        hi = (pd.Timestamp(day_max) + pd.to_timedelta(int(HORIZON + ANCHOR_TOL_DAYS), unit="D")).date()
        placeholders = ", ".join("?" for _ in slugs)
        frame = con.sql(
            f"""
            SELECT item_slug AS item_id, CAST(day AS DATE) AS date, source,
                   mean_price AS price, volume
            FROM {rel}
            WHERE CAST(day AS DATE) BETWEEN DATE '{lo}' AND DATE '{hi}'
              AND item_slug IN ({placeholders})
              AND ({uni}) AND mean_price IS NOT NULL AND mean_price > 0
        """,
            params=list(slugs),
        ).fetchdf()
    finally:
        con.close()
    frame["date"] = pd.to_datetime(frame["date"])
    voted = ItemForecaster._apply_multi_source_voting(frame)
    voted["date"] = pd.to_datetime(voted["date"])
    return voted.sort_values(["item_id", "date"]).reset_index(drop=True)


def _cohort_dates(df, col):
    sub = df[df[col]] if col else df
    n = sub.groupby("forecast_date").size()
    return sorted(n[n >= MIN_ROWS_PER_DATE].index)


def run(archive_dir=None, frozen_threshold=FROZEN_THRESHOLD, exact_only=False):
    archive_dir = Path(archive_dir) if archive_dir else DURABLE_ARCHIVE
    if not archive_dir.exists():
        raise SystemExit(f"archive not found at {archive_dir}")

    panel = derive_panel(load_served())
    panel = panel[panel["coh_clean"] | panel["coh_wedge"]].copy()
    slugs = sorted(panel["slug"].unique())
    logger.info(f"panel: {len(panel):,} rows / {panel['forecast_date'].nunique()} dates / {len(slugs):,} slugs")

    voted = load_voted(slugs, panel["forecast_date"].min(), panel["forecast_date"].max(), archive_dir)
    frozen = frozen_anchor_dates(voted, threshold=frozen_threshold)
    logger.info(f"frozen anchor days (>= {frozen_threshold:.2f}): {sorted(str(d) for d in frozen)}")

    available = sorted(voted["date"].dt.date.unique())
    price_map = {(str(s), d.date()): float(p) for s, d, p in zip(voted["item_id"], voted["date"], voted["price"])}

    tol = 0 if exact_only else ANCHOR_TOL_DAYS
    legs = archive_legs(panel, price_map, HORIZON, available, frozen, tol=tol)
    logger.info(f"legs formed on {len(legs):,} rows")

    result = {
        "archive_dir": str(archive_dir),
        "frozen_threshold": frozen_threshold,
        "frozen_days": sorted(str(d) for d in frozen),
        "exact_only": exact_only,
        "cohorts": {},
    }

    for name, col in (("clean", "coh_clean"), ("wedge", "coh_wedge"), ("all", None)):
        sub = legs[legs[col]] if col else legs
        dates = _cohort_dates(sub, None)
        sub = sub[sub["forecast_date"].isin(dates)]
        if sub.empty:
            result["cohorts"][name] = {"verdict": "VOID", "n_dates": 0, "reason": "no qualifying dates"}
            continue
        model = per_date_ic(sub["forecast_date"], sub["r_hat"], sub["realized"])
        naive = per_date_ic(sub["forecast_date"], sub["naive"], sub["realized"])
        n_dates = len(model)
        cell = evaluate_bars(bootstrap_ci(model), paired_delta_ci(model, naive), n_dates)
        cell["naive_ic"] = bootstrap_ci(naive)
        cell["rows"] = len(sub)
        cell["dates"] = [str(d) for d in sorted(model)]
        result["cohorts"][name] = cell

    result["primary"] = "clean"
    result["verdict"] = result["cohorts"]["clean"]["verdict"]
    return result


def gate(archive_dir=None):
    """Power gate only — a date count, never a rank statistic."""
    panel = derive_panel(load_served())
    clean = panel[panel["coh_clean"]]
    n = clean.groupby("forecast_date").size()
    dates = sorted(n[n >= MIN_ROWS_PER_DATE].index)
    return {
        "qualifying_dates": len(dates),
        "min_dates": MIN_DATES,
        "runnable": len(dates) >= MIN_DATES,
        "dates": [str(d) for d in dates],
        "panel_dates": int(panel["forecast_date"].nunique()),
        "panel_rows": len(panel),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="print the power gate and exit; reads no statistic")
    ap.add_argument("--archive-dir", default=None)
    ap.add_argument("--frozen-threshold", type=float, default=FROZEN_THRESHOLD)
    ap.add_argument("--exact-only", action="store_true", help="sensitivity: require the exact d+14 anchor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.gate:
        g = gate(args.archive_dir)
        print(json.dumps(g, indent=2))
        print(
            f"\n{'RUNNABLE' if g['runnable'] else 'VOID'}: "
            f"{g['qualifying_dates']} qualifying anchor_clean dates "
            f"(need {g['min_dates']})"
        )
        return

    g = gate(args.archive_dir)
    if not g["runnable"]:
        print(json.dumps(g, indent=2))
        raise SystemExit(
            f"VOID: {g['qualifying_dates']} qualifying dates < {g['min_dates']}. "
            "The prereg bars this leg below the gate — no statistic is read."
        )

    result = run(args.archive_dir, args.frozen_threshold, args.exact_only)
    print(json.dumps(result, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
