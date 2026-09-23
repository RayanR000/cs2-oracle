#!/usr/bin/env python3
"""Fill item_forecasts.band_multiplier for rows served before migration 0028 recorded it.

Rows before FEEDBACK_FIRST_SERVED_DATE need nothing (the loader reads NULL there as 1.0).
Rows from 2026-09-17 on were served under a non-unit h=3 factor, and without a value the
served-coverage refit drops them. This reconstructs the value each stored row was served
at from the Price Forecast run logs. The final stored band per (item, forecast_date,
horizon) is the LAST write (upsert), and each write blends 15% of the latest prior row
dated before the run's wall-clock day -- which, on a same-day redispatch, is that date's
own earlier write:

  run (UTC)          id           fd     h=3 factor  prior row (its m)        stored m
  09-18 04:17        35306325544  09-17  0.5251      09-17 earlier write (1)  0.85*0.5251+0.15*1      = 0.5963
  09-19 01:08        35411691064  09-18  0.5251      09-17 (0.5963)           0.5358, overwritten by:
  09-20 00:58        35480138859  09-18  0.5385      09-18 own write (0.5358) 0.85*0.5385+0.15*0.5358 = 0.5381
  09-21 01:00 (full) 35549467741  09-20  0.5385      09-18 (0.5381)           0.5384
  09-22 onward       predict-only 09-21+ 0.5385      ~0.5385                  0.5385

(09-17's two earlier writes, 01:13 and 04:03, served at 1.0 and were overwritten.) The
09-17 value is confirmed on prod: the h=3/h=7 relative-width ratio fell x0.594 that day.
Every other horizon had factor 1.0 throughout (the factor dict was {3: ...} only), so
their rows get 1.0. Items with no prior row would have stored the bare factor; on the
fixed 5,536-item universe that case does not arise.

Valid only through 2026-09-27: the next mode=full retrain (2026-09-28) refits the factor,
and rows it serves are written by code that records band_multiplier itself. Only NULL
rows are touched, so a re-run is a no-op.

    venv/bin/python -m scripts.backfill_band_multiplier            # dry run: counts only
    venv/bin/python -m scripts.backfill_band_multiplier --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("backfill_band_multiplier")

START = date(2026, 9, 17)
LAST_VALID = date(2026, 9, 27)
H3_SERVED = {date(2026, 9, 17): 0.5963, date(2026, 9, 18): 0.5381, date(2026, 9, 20): 0.5384}
H3_STEADY = 0.5385


def served_multiplier(forecast_date: date, horizon: int) -> float:
    """The multiplier a row stored for (forecast_date, horizon) was served at, START..LAST_VALID."""
    if not START <= forecast_date <= LAST_VALID:
        raise ValueError(f"{forecast_date} is outside the reconstructed window {START}..{LAST_VALID}")
    if horizon != 3:
        return 1.0
    return H3_SERVED.get(forecast_date, H3_STEADY)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the values (default: dry run)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from database import SessionLocal
    from sqlalchemy import inspect, text

    db = SessionLocal()
    try:
        cols = {c["name"] for c in inspect(db.get_bind()).get_columns("item_forecasts")}
        if "band_multiplier" not in cols:
            logger.error("item_forecasts has no band_multiplier column: apply migration 0028 first.")
            return 1
        groups = db.execute(
            text(
                "SELECT forecast_date, horizon_days, COUNT(*) AS n FROM item_forecasts "
                "WHERE band_multiplier IS NULL AND forecast_date BETWEEN :a AND :b "
                "GROUP BY 1, 2 ORDER BY 1, 2"
            ),
            {"a": START, "b": LAST_VALID},
        ).fetchall()
        total = 0
        for fd, h, n in groups:
            m = served_multiplier(fd, int(h))
            logger.info(f"  {fd}  h={h:<2}  {n:>6,} NULL rows -> {m:.4f}")
            total += n
            if args.apply:
                db.execute(
                    text(
                        "UPDATE item_forecasts SET band_multiplier = :m "
                        "WHERE band_multiplier IS NULL AND forecast_date = :fd AND horizon_days = :h"
                    ),
                    {"m": m, "fd": fd, "h": int(h)},
                )
        if args.apply:
            db.commit()
        logger.info(f"{'Updated' if args.apply else 'Would update'} {total:,} rows.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
