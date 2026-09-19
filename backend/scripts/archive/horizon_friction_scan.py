"""Does the market itself move enough to beat friction — at ANY horizon?

The paper-trade harness caps at h in {14, 30} because that is all the forecaster
serves, and every strategy loses there: friction (8-23% at the cheapest cash
venue) dwarfs a typical 14-30d move. This asks the prior, model-free question —
over 14/30/60/90/180-day windows, what fraction of items move more than the
round-trip cost? — straight off the price archive.

If even at 180 days almost nothing clears the bar, there is no tradable horizon
and the thread closes. If a fraction opens up at some N, that reframes the
modelling target from "better short-horizon accuracy" to "forecast longer".

Reads the Parquet archive read-only through the production voted-consensus path
(`prices_relation` + `_apply_multi_source_voting`), so the price series is the
same one the model trains on — universe rules, bid/trailing exclusion and all.

    venv/bin/python -m scripts.archive.horizon_friction_scan
    venv/bin/python -m scripts.archive.horizon_friction_scan --min-price 100 --days-back 1460
    venv/bin/python -m scripts.archive.horizon_friction_scan --selftest   # unit-check the matcher

Not a trade sim — no direction, no P&L. It measures the RAW move distribution
against the cost bar, which is the ceiling on what any long-only strategy at that
horizon could clear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.friction import ROUND_TRIP_COST, SPREAD_BY_TIER
from backtest.scoring import price_tier

# Anchor->target matching tolerance. The archive has day gaps, and a resolved
# price is knowable within LAG_TOLERANCE_DAYS of its nominal date, so a target
# is accepted on the first available day in [d+N, d+N+TOL]. Mirrors production's
# LAG_TOLERANCE_DAYS; a wider window would let a stale carry stand in for a move.
TARGET_TOL_DAYS = 3

HORIZONS = (14, 30, 60, 90, 180)
# Half the spread on the sell leg only; full spread on a round trip. Matches
# backtest.papertrade._SPREAD_CROSSINGS.
SPREAD_CROSSINGS = {"round_trip": 1.0, "sell_only": 0.5}


def forward_returns(
    days: np.ndarray, prices: np.ndarray, horizon: int, tol: int = TARGET_TOL_DAYS
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor prices and their forward returns over `horizon` days, for ONE item.

    `days` is the anchor day as an integer ordinal, sorted ascending; `prices`
    aligns with it. For each anchor `d`, the target is the first observed day in
    `[d + horizon, d + horizon + tol]`; anchors with no target in that window are
    dropped (no carry-forward — a fabricated flat is not a move). Returns
    `(anchor_price, ret)` arrays over the anchors that matched.
    """
    targets = days + horizon
    idx = np.searchsorted(days, targets, side="left")
    valid = (idx < len(days)) & (days[np.clip(idx, 0, len(days) - 1)] <= targets + tol)
    anchor_price = prices[valid]
    target_price = prices[idx[valid]]
    return anchor_price, target_price / anchor_price - 1.0


def _selftest() -> None:
    # Exact hit at d+horizon.
    ap, ret = forward_returns(np.array([0, 30]), np.array([100.0, 130.0]), 30)
    assert ap.tolist() == [100.0] and abs(ret[0] - 0.30) < 1e-9, (ap, ret)
    # Within tolerance (target lands at d+31, tol=3): matched.
    ap, ret = forward_returns(np.array([0, 31]), np.array([100.0, 110.0]), 30)
    assert ap.tolist() == [100.0] and abs(ret[0] - 0.10) < 1e-9, (ap, ret)
    # Outside tolerance (target at d+40): no match.
    ap, ret = forward_returns(np.array([0, 40]), np.array([100.0, 110.0]), 30)
    assert ap.size == 0, (ap, ret)
    # Earliest target in the window wins, not the closest to nominal.
    ap, ret = forward_returns(np.array([0, 32, 33]), np.array([100.0, 120.0, 150.0]), 30)
    assert abs(ret[0] - 0.20) < 1e-9, (ap, ret)
    print("selftest OK")


def _load_voted(days_back: int, min_price: float) -> pd.DataFrame:
    """Voted consensus price per item per day, over the last `days_back` days."""
    import duckdb
    from db.archive import prices_relation
    from models.forecaster import ItemForecaster
    from models.item_parser import (
        phantom_slug_sql_filter,
        phase_collapsed_sql_filter,
    )

    con = duckdb.connect()
    try:
        rel = prices_relation(con, columns=["item_slug", "day", "mean_price", "volume", "source"])
        cutoff = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
        df = con.sql(f"""
            SELECT item_slug AS item_id, day AS timestamp,
                   mean_price AS price, volume, source
            FROM {rel} sub
            WHERE day >= '{cutoff}'
              AND (source IS NULL OR source NOT LIKE 'historical_fallback:%')
              AND {phase_collapsed_sql_filter("sub.item_slug")}
              AND {phantom_slug_sql_filter("sub.item_slug")}
        """).fetchdf()
    finally:
        con.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["date"] = df["timestamp"].dt.date
    # Production voting: bid/trailing sources dropped, 2sigma outliers rejected with
    # >=3 sources, median otherwise. One consensus row per item-day.
    voted = ItemForecaster._apply_multi_source_voting(df)
    voted = voted[voted["price"] >= min_price].copy()
    voted["ord"] = pd.to_datetime(voted["date"]).astype("int64") // 86_400_000_000_000
    return voted


def _scan(voted: pd.DataFrame, venue: str) -> pd.DataFrame:
    """Per horizon and cost mode: fraction of anchors whose move beats friction."""
    rows = []
    grouped = [
        (np.asarray(g["ord"].to_numpy()), np.asarray(g["price"].to_numpy()))
        for _, g in voted.sort_values(["item_id", "ord"]).groupby("item_id")
    ]
    for horizon in HORIZONS:
        aps, rets = [], []
        for days, prices in grouped:
            if len(days) < 2:
                continue
            ap, r = forward_returns(days, prices, horizon)
            aps.append(ap)
            rets.append(r)
        anchor_price = np.concatenate(aps) if aps else np.array([])
        ret = np.concatenate(rets) if rets else np.array([])
        if ret.size == 0:
            continue
        tiers = np.array([price_tier(p) for p in anchor_price])
        spread = np.array([SPREAD_BY_TIER[t] for t in tiers])
        for mode, crossing in SPREAD_CROSSINGS.items():
            bar = ROUND_TRIP_COST[venue] + spread * crossing
            rows.append(
                {
                    "horizon": horizon,
                    "mode": mode,
                    "n": int(ret.size),
                    "median_abs_move_%": round(float(np.median(np.abs(ret))) * 100, 2),
                    "p90_abs_move_%": round(float(np.percentile(np.abs(ret), 90)) * 100, 2),
                    "median_bar_%": round(float(np.median(bar)) * 100, 2),
                    # Long clears: the UP move exceeds the round trip you'd pay.
                    "pct_long_clears": round(float(np.mean(ret > bar)) * 100, 2),
                    # Magnitude clears either side — the ceiling if you could pick
                    # direction perfectly. There is no per-item direction signal, so
                    # this is an unreachable upper bound, shown to bound the gap.
                    "pct_abs_clears": round(float(np.mean(np.abs(ret) > bar)) * 100, 2),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-price", type=float, default=1.0)
    ap.add_argument(
        "--days-back", type=int, default=1460, help="anchor window in days (default 1460 = 4yr, current regime)"
    )
    ap.add_argument("--venue", default="csfloat", choices=sorted(ROUND_TRIP_COST))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    voted = _load_voted(args.days_back, args.min_price)
    print(
        f"horizon-vs-friction scan  venue={args.venue}  "
        f"min_price=${args.min_price:g}  days_back={args.days_back}  "
        f"items={voted['item_id'].nunique():,}  item-days={len(voted):,}"
    )
    print(
        "pct_long_clears = share of windows whose UP move beats the round trip; "
        "pct_abs_clears = either-direction ceiling (no per-item direction signal "
        "exists to reach it).\n"
    )
    out = _scan(voted, args.venue)
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
