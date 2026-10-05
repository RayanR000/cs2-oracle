#!/usr/bin/env python3
"""Warn when a CNY venue's price level jumps against csfloat: the FX detector.

BUFF163 and youpin price in CNY. CSGOTrader converts them to USD upstream at a
rate and date nobody records, so a wrong or stale rate is a market-wide
multiplicative error on 2 of the voted sources, and nothing local saw it
(docs/research/2026-08-19-deep-model-review.md §1d).

For each venue this takes the median, over items >= $1, of `venue / csfloat` on
the same day, and compares the newest day with the median of the prior
`--baseline-days` days. Over 2026-07-11..09-08 the CNY ratios held at
1.017 (buff163) and 1.027 (youpin), std 0.007, with a median day-to-day move
of 0.15% and a largest of 1.3%. `--threshold` (3%) is over twice that largest move.

- `cny_breach`: buff163 or youpin moved more than the threshold.
- `fx_signature`: both CNY venues moved past the threshold in the same direction while
  every USD control venue stayed within it. That is the shape of a conversion
  error, not of csfloat (the denominator) moving, which shifts every ratio.

Warn-only: a breach prints a `::warning::` annotation and the JSON summary, and
the exit code stays 0. Price Forecast chains off the Aggregator's success, and a real
market move must not stop the day's forecasts.

Usage (from backend/):
    venv/bin/python -m scripts.check_fx_ratios --archive-dir ../archive/price-archive
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
from db.archive import prices_relation
from models.item_parser import archive_universe_sql_filter

REFERENCE = "aggregator_csfloat"
CNY_VENUES = ("aggregator_buff163", "aggregator_youpin")
USD_CONTROLS = ("aggregator_skinport", "aggregator_csmoney")
MIN_PRICE_USD = 1.0
MIN_ITEMS = 200


def daily_ratios(archive_dir: Path, lookback_days: int) -> dict[str, dict[str, float]]:
    """{day: {venue: median(venue / csfloat)}} over the trailing window."""
    con = duckdb.connect()
    rel = prices_relation(con, archive_dir, columns=["item_slug", "day", "source", "mean_price"])
    venues = ", ".join(f"'{v}'" for v in (*CNY_VENUES, *USD_CONTROLS))
    rows = con.sql(f"""
        WITH p AS (
            SELECT item_slug, day, source, mean_price FROM {rel}
            WHERE day > (SELECT max(day) FROM {rel}) - INTERVAL {int(lookback_days)} DAY
              AND mean_price >= {MIN_PRICE_USD}
              AND {archive_universe_sql_filter("item_slug")}
        ),
        ref AS (SELECT item_slug, day, mean_price AS ref_price FROM p WHERE source = '{REFERENCE}')
        SELECT p.day, p.source, median(p.mean_price / ref.ref_price) AS ratio, count(*) AS n
        FROM p JOIN ref USING (item_slug, day)
        WHERE p.source IN ({venues})
        GROUP BY 1, 2
        HAVING count(*) >= {MIN_ITEMS}
        ORDER BY 1, 2
    """).fetchall()
    out: dict[str, dict[str, float]] = {}
    for day, source, ratio, _n in rows:
        out.setdefault(day.isoformat(), {})[source] = float(ratio)
    return out


def evaluate(ratios: dict[str, dict[str, float]], baseline_days: int, threshold: float) -> dict:
    days = sorted(ratios)
    if len(days) < 2:
        return {"status": "insufficient_history", "days": len(days)}
    latest, prior = days[-1], days[-1 - baseline_days : -1]
    moves = {}
    for venue in (*CNY_VENUES, *USD_CONTROLS):
        base = [ratios[d][venue] for d in prior if venue in ratios[d]]
        if venue not in ratios[latest] or not base:
            continue
        moves[venue] = round(ratios[latest][venue] / statistics.median(base) - 1, 5)
    cny = {v: moves[v] for v in CNY_VENUES if v in moves}
    usd = {v: moves[v] for v in USD_CONTROLS if v in moves}
    breached = sorted(v for v, m in cny.items() if abs(m) > threshold)
    signature = (
        len(breached) == len(CNY_VENUES)
        and len({m > 0 for m in cny.values()}) == 1
        and bool(usd)
        and all(abs(m) <= threshold for m in usd.values())
    )
    return {
        "status": "fx_signature" if signature else ("cny_breach" if breached else "ok"),
        "day": latest,
        "baseline_days": len(prior),
        "threshold": threshold,
        "moves": moves,
        "latest_ratios": {k: round(v, 5) for k, v in ratios[latest].items()},
        "breached": breached,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", type=Path, required=True)
    p.add_argument("--baseline-days", type=int, default=14)
    p.add_argument("--threshold", type=float, default=0.03)
    args = p.parse_args()

    report = evaluate(daily_ratios(args.archive_dir, args.baseline_days + 7), args.baseline_days, args.threshold)
    print(json.dumps(report))
    if report["status"] in ("cny_breach", "fx_signature"):
        moved = ", ".join(f"{v.removeprefix('aggregator_')} {report['moves'][v]:+.1%}" for v in report["breached"])
        what = (
            "likely a CNY conversion error upstream" if report["status"] == "fx_signature" else "check before trusting"
        )
        print(f"::warning::FX ratio monitor {report['day']}: {moved} vs csfloat ({what})")


if __name__ == "__main__":
    main()
