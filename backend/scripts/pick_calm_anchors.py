"""Rank serveable replay anchors by how CALM the market was around them.

The regime-reactive climatology arm narrows the band when recent vol is below an
item's baseline, so its win (if any) lands on CALM anchors. This picks them:
a market realized-volatility series = the cross-sectional MEDIAN of |1-day
return| among >=$1 items per day (voted-universe filter, faithful to serving),
smoothed over a trailing window, then the lowest-vol serveable dates are printed.
Each candidate is run through replay's own feed audit so we only propose anchors
the replay will accept.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.archive import prices_relation                          # noqa: E402
from models.item_parser import archive_universe_sql_filter      # noqa: E402
from scripts.replay_serving import audit_anchor_feed, _feed_profile  # noqa: E402

# Serveable window: 30d must resolve (archive runs into 2026-08), and stay clear
# of the early-2026 basis breaks the changelog flags.
LO, HI = date(2026, 3, 15), date(2026, 7, 1)
VOL_SMOOTH = 14   # trailing days
PRICE_FLOOR = 1.0


def market_vol_series() -> pd.DataFrame:
    con = duckdb.connect()
    try:
        rel = prices_relation(con, columns=["item_slug", "day", "mean_price", "source"])
        # One voted-ish price per item-day (median across sources), universe
        # filtered, then per-item daily return, then the cross-sectional median
        # |return| per day.
        df = con.sql(f"""
            WITH px AS (
                SELECT CAST(day AS DATE) AS d, item_slug AS iid,
                       median(mean_price) AS p
                FROM {rel} sub
                WHERE mean_price > 0
                  AND (source IS NULL OR source NOT LIKE 'historical_fallback:%')
                  AND {archive_universe_sql_filter("sub.item_slug", "sub.source")}
                  AND CAST(day AS DATE) >= DATE '2026-01-01'
                GROUP BY 1, 2
            )
            SELECT d, iid, p,
                   p / lag(p) OVER (PARTITION BY iid ORDER BY d) - 1 AS ret
            FROM px
        """).fetchdf()
    finally:
        con.close()
    df = df[df["p"] >= PRICE_FLOOR]
    daily = (df.dropna(subset=["ret"])
               .groupby("d")["ret"]
               .apply(lambda s: float(np.median(np.abs(s))) * 100.0)
               .rename("med_abs_ret_pct")
               .reset_index()
               .sort_values("d"))
    daily["vol_14d"] = daily["med_abs_ret_pct"].rolling(VOL_SMOOTH, min_periods=7).mean()
    daily["d"] = pd.to_datetime(daily["d"]).dt.date
    return daily


def main() -> int:
    daily = market_vol_series()
    cand = daily[(daily["d"] >= LO) & (daily["d"] <= HI) & daily["vol_14d"].notna()].copy()
    cand = cand.sort_values("vol_14d")
    print(f"market vol range over {LO}..{HI}: "
          f"{cand['vol_14d'].min():.3f}..{cand['vol_14d'].max():.3f} "
          f"(median |1d ret| %, 14d mean)\n")
    print("CALMEST candidates (ascending vol_14d), feed-audited:")
    print(f"{'anchor':>12} {'vol_14d':>8} {'day%':>7}  feed_audit")
    picked = []
    for _, r in cand.iterrows():
        anchor = r["d"] if isinstance(r["d"], date) else pd.Timestamp(r["d"]).date()
        ok, _ = audit_anchor_feed(anchor, _feed_profile(anchor))
        tag = "CLEAN" if ok else "dirty"
        print(f"{anchor.isoformat():>12} {r['vol_14d']:>8.3f} "
              f"{r['med_abs_ret_pct']:>7.3f}  {tag}")
        if ok:
            picked.append(anchor)
        if len(picked) >= 5:
            break
    print(f"\nPICKED (clean, calmest): {[a.isoformat() for a in picked]}")
    # Contrast: the most volatile serveable dates (where static clim should win).
    vol = cand.sort_values("vol_14d", ascending=False).head(3)
    print("busiest (for contrast):",
          [pd.Timestamp(d).date().isoformat() for d in vol["d"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
