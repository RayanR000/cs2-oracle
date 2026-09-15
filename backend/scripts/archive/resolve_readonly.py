#!/usr/bin/env python3
"""Resolve pending forecasts READ-ONLY and emit a merged outcomes panel.

Why: the scored panel is 1-2 dates short of MIN_FORECAST_DATES=20 at h=3/7,
while forecasts whose horizon has ALREADY elapsed sit unresolved in prod. The
real fix is a backtest run, which writes prod. This reaches the same panel
depth with zero writes: read prod, resolve against the local Parquet archive,
and emit the merged panel to a scratch archive that centre_vs_lastprice.py
reads unmodified via --archive-dir.

NOT a second scoring path. Every derivation is imported from the production
modules -- resolve_anchors, load_voted_prices, _derive_verdict -- so this
cannot drift from what the real backtest would write. --validate proves it:
re-resolve rows prod has ALREADY scored and diff against the stored values.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env")

from backtest.price_resolution import load_voted_prices, resolve_anchors  # noqa: E402
from database import SessionLocal  # noqa: E402
from models.staleness import stale_run_lookup  # noqa: E402
from sqlalchemy import text  # noqa: E402

sys.path.insert(0, str(BACKEND / "scripts"))
from backtest_accuracy import _derive_verdict  # noqa: E402

# The CANONICAL archive clone by default, never the symlinked working copy: a
# predict run writes into whatever `price-archive` points at, so the working
# copy can be dirty. Override with --archive-dir.
DEFAULT_ARCHIVE = BACKEND.parent.parent / "cs2-oracle-data" / "price-archive"
MIN_PRICE = 1.0

# Pending = horizon elapsed, no outcome row. Same join the backtest uses.
_PENDING = """
    SELECT f.id, f.item_id, f.forecast_date, f.horizon_days,
           f.price_low, f.price_mid, f.price_high, f.current_price,
           f.direction, f.model_version
    FROM item_forecasts f
    LEFT JOIN forecast_outcomes o ON o.forecast_id = f.id
    WHERE o.id IS NULL
      AND f.forecast_date + (f.horizon_days || ' days')::interval <= current_date
      AND f.price_mid IS NOT NULL
"""
# --validate: rows prod already scored, to reproduce.
_SCORED = """
    SELECT f.id, f.item_id, f.forecast_date, f.horizon_days,
           f.price_low, f.price_mid, f.price_high, f.current_price,
           f.direction, f.model_version,
           o.base_price AS s_base, o.actual_price AS s_actual,
           o.in_interval AS s_in_interval, o.direction_correct AS s_dir
    FROM item_forecasts f
    JOIN forecast_outcomes o ON o.forecast_id = f.id
    WHERE o.actual_price IS NOT NULL AND f.forecast_date = :d
      AND f.horizon_days = :h
"""


def _resolve(rows, id_to_slug, archive: Path):
    """Resolve rows -> list of outcome dicts, applying the backtest's guards."""
    groups = defaultdict(list)
    for r in rows:
        groups[r.horizon_days].append(r)

    out, dropped = [], 0
    for horizon, forecasts in sorted(groups.items()):
        anchors, slugs = set(), set()
        for f in forecasts:
            slug = id_to_slug.get(f.item_id)
            if slug is None:
                continue
            d = f.forecast_date
            slugs.add(slug)
            anchors.add((slug, d))
            anchors.add((slug, d + timedelta(days=horizon)))
        if not anchors:
            dropped += len(forecasts)
            continue

        ad = [a[1] for a in anchors]
        voted = load_voted_prices(archive, sorted(slugs), min(ad), max(ad))
        prices = resolve_anchors(voted, anchors)
        stale = stale_run_lookup(voted)
        print(f"  h={horizon}: resolved {len(prices):,} of {len(anchors):,} anchors")

        for f in forecasts:
            slug = id_to_slug.get(f.item_id)
            d = f.forecast_date
            target = d + timedelta(days=horizon)
            base_res, actual_res = prices.get((slug, d)), prices.get((slug, target))
            if base_res is None or actual_res is None:
                dropped += 1
                continue
            base, actual = base_res.price, actual_res.price
            if base <= 0 or actual <= 0:
                dropped += 1
                continue
            # Disjoint-window guard: the actual leg must carry no pre-forecast
            # observation, or a manufactured 0.0% move scores as a flat market.
            if actual_res.oldest_observation <= d:
                dropped += 1
                continue
            if base < MIN_PRICE:
                continue  # out of the served cohort
            mid, low, high = f.price_mid, f.price_low, f.price_high
            v = _derive_verdict(base, actual, mid, low, high, f.direction, quote=f.current_price)
            out.append(
                {
                    "forecast_id": f.id,
                    "item_id": f.item_id,
                    "item_slug": slug,
                    "forecast_date": d,
                    "horizon_days": horizon,
                    "target_date": target,
                    "current_price": f.current_price,
                    "base_price": base,
                    "actual_price": actual,
                    "predicted_price_low": low,
                    "predicted_price_mid": mid,
                    "predicted_price_high": high,
                    "direction_predicted": f.direction or "flat",
                    "direction_actual": v["direction_actual"],
                    "direction_correct": v["direction_correct"],
                    "in_interval": v["in_interval"],
                    "abs_error": v["abs_error"],
                    "pct_error": v["pct_error"],
                    "model_version": f.model_version,
                    "base_stale_run_days": stale.get((slug, d)),
                    "evaluated_at": pd.Timestamp.utcnow(),
                    "resolved_at": pd.Timestamp.utcnow(),
                }
            )
    return out, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="panel", help="scratch archive dir to write")
    ap.add_argument(
        "--validate", nargs=2, metavar=("DATE", "HORIZON"), help="reproduce an already-scored date instead of resolving"
    )
    ap.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="price archive to resolve against (default: the canonical cs2-oracle-data clone)",
    )
    a = ap.parse_args()
    archive = a.archive_dir.resolve()
    if not archive.is_dir():
        print(f"archive not found: {archive}", file=sys.stderr)
        return 2
    print(f"archive: {archive}")

    db = SessionLocal()
    try:
        id_to_slug = {r.id: r.item_id for r in db.execute(text("SELECT id, item_id FROM items")).fetchall()}
        print(f"{len(id_to_slug):,} slug mappings")

        if a.validate:
            d, h = date.fromisoformat(a.validate[0]), int(a.validate[1])
            rows = db.execute(text(_SCORED), {"d": d, "h": h}).fetchall()
            print(f"validating {len(rows):,} already-scored rows @ {d} h={h}")
            stored = {r.id: r for r in rows}
            got, _ = _resolve(rows, id_to_slug, archive)
            nb = na = ni = nd = 0
            for o in got:
                s = stored[o["forecast_id"]]
                nb += abs(float(s.s_base) - o["base_price"]) > 1e-6
                na += abs(float(s.s_actual) - o["actual_price"]) > 1e-6
                ni += (s.s_in_interval or 0) != (o["in_interval"] or 0)
                nd += (s.s_dir or 0) != o["direction_correct"]
            print(
                f"reproduced {len(got):,} rows: base_price mismatches {nb}, "
                f"actual_price {na}, in_interval {ni}, direction_correct {nd}"
            )
            return 0 if nb == na == ni == nd == 0 else 1

        rows = db.execute(text(_PENDING)).fetchall()
        print(f"{len(rows):,} pending forecasts (horizon elapsed, no outcome)")

        # EVERY DB read happens here, before the resolve. The resolve leg runs
        # for minutes against the Parquet archive, and the Supabase pooler drops
        # an idle connection out from under it (SSL SYSCALL error: EOF).
        sql = """SELECT forecast_id, item_id, forecast_date,
                        horizon_days, target_date, current_price, base_price,
                        actual_price, predicted_price_low, predicted_price_mid,
                        predicted_price_high, direction_predicted,
                        direction_actual, direction_correct, in_interval,
                        abs_error, pct_error, model_version,
                        base_stale_run_days, evaluated_at, resolved_at
                 FROM forecast_outcomes WHERE actual_price IS NOT NULL"""
        existing = pd.DataFrame(db.execute(text(sql)).fetchall())
        existing["item_slug"] = None
        print(f"{len(existing):,} existing scored rows from prod")
    finally:
        db.close()

    new, dropped = _resolve(rows, id_to_slug, archive)
    print(f"resolved {len(new):,}, dropped {dropped:,}")
    nd = pd.DataFrame(new)
    print("NEW rows per horizon/date:")
    print(nd.groupby(["horizon_days", "forecast_date"]).size().to_string())

    merged = pd.concat([existing, pd.DataFrame(new)], ignore_index=True)
    for c in (
        "base_price",
        "actual_price",
        "current_price",
        "predicted_price_low",
        "predicted_price_mid",
        "predicted_price_high",
        "abs_error",
        "pct_error",
    ):
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged["forecast_date"] = pd.to_datetime(merged["forecast_date"])

    out = Path(a.out).resolve() / "ops"
    out.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out / "forecast_outcomes.parquet", index=False)
    print(f"wrote {len(merged):,} rows -> {out / 'forecast_outcomes.parquet'}")

    m = merged[merged.actual_price.notna() & (merged.base_price >= MIN_PRICE)]
    print(m.groupby("horizon_days").forecast_date.nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
