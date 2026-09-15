"""SPIKE (throwaway): is there any per-item signal for the 180-day move?

The horizon scan showed 180d moves are large enough that ~28% of up-moves clear
friction — but that is a base rate, not an edge. This asks whether any cheap
price technical *predicts which* items make the move, using WITHIN-DATE rank-IC
(Spearman of feature-rank vs forward-return-rank across items on each anchor
date). Within-date is deliberate: a pooled IC just re-measures the market-wide
factor, and AGENTS.md's whole finding is that nothing idiosyncratic survives once
that factor is removed.

The number that decides it is not the IC — it is `n_indep`, the count of
NON-OVERLAPPING 180d episodes. Monthly anchors 180d apart share a forward window
and are not independent; with a few years of history there are only ~20 truly
independent episodes, so a large-looking IC on overlapping anchors can rest on a
handful of market moves. Read `mean_ic_indep` / `t_indep`, not `mean_ic_all`.

Model-free and rough by design — this is a go/no-go probe, not the production
harness. If every feature's `t_indep` is inside +-2 and near the placebo, there
is no 180d signal and the trading thread closes for good.

    venv/bin/python -m scripts.archive.signal_probe_180d --min-price 1 --days-back 3650
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.archive.horizon_friction_scan import _load_voted

HORIZON = 180
MIN_ITEMS_PER_DATE = 30  # a within-date IC on fewer names is noise
FEATURES = ("mom_30", "mom_90", "vol_30", "rev_z_90")


def _trailing_return(days: np.ndarray, prices: np.ndarray, back: int, tol: int = 15) -> np.ndarray:
    """Return over the last `back` calendar days: price now / price ~`back` ago.

    The reference is the last observation on or before `d - back`, accepted only
    if it sits in `[d-back-tol, d-back]` (no stale carry from months earlier).
    NaN where there is no such observation. Numpy as-of, gap-safe like the
    forward matcher.
    """
    target = days - back
    j = np.searchsorted(days, target, side="right") - 1
    valid = j >= 0
    jc = np.clip(j, 0, len(days) - 1)
    gap = days - days[jc]
    valid &= (gap >= back) & (gap <= back + tol)
    out = np.full(len(days), np.nan)
    out[valid] = prices[valid] / prices[jc][valid] - 1.0
    return out


def _engineer(voted: pd.DataFrame) -> pd.DataFrame:
    """Standard price technicals at each anchor, calendar-windowed per item.

    Momentum is a numpy as-of ratio; volatility and the mean-reversion z-score
    use time-based rolling ('30D'/'90D') so the archive's day gaps do not
    silently shorten a window. No forward-fill — a gap stays a gap.
    """
    frames = []
    for item_id, g in voted.sort_values(["item_id", "date"]).groupby("item_id"):
        g = g.set_index(pd.to_datetime(g["date"]))
        p = g["price"]
        days = g.index.astype("int64").to_numpy() // 86_400_000_000_000
        prices = p.to_numpy()
        logret = np.log(p).diff()
        frames.append(
            pd.DataFrame(
                {
                    "item_id": item_id,
                    "date": g.index.to_numpy(),
                    "price": prices,
                    "mom_30": _trailing_return(days, prices, 30),
                    "mom_90": _trailing_return(days, prices, 90),
                    "vol_30": logret.rolling("30D").std().to_numpy(),
                    # Distance from the 90d mean in 90d-vol units: a mean-reversion score.
                    "rev_z_90": ((p - p.rolling("90D").mean()) / p.rolling("90D").std()).to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _forward_return_col(voted: pd.DataFrame) -> pd.DataFrame:
    """The 180d forward return per anchor, matched with the scan's tolerance."""
    out = []
    for item_id, g in voted.sort_values(["item_id", "date"]).groupby("item_id"):
        days = pd.to_datetime(g["date"]).astype("int64").to_numpy() // 86_400_000_000_000
        prices = g["price"].to_numpy()
        targets = days + HORIZON
        idx = np.searchsorted(days, targets, side="left")
        ok = (idx < len(days)) & (days[np.clip(idx, 0, len(days) - 1)] <= targets + 3)
        fwd = np.full(len(days), np.nan)
        fwd[ok] = prices[idx[ok]] / prices[ok] - 1.0
        out.append(pd.DataFrame({"item_id": item_id, "date": g["date"].to_numpy(), "fwd": fwd}))
    return pd.concat(out, ignore_index=True)


def _within_date_ic(df: pd.DataFrame, feat: str, seed: int | None = None) -> pd.Series:
    """Spearman(feature, fwd) per anchor date; optional label shuffle (placebo)."""
    rng = np.random.default_rng(seed) if seed is not None else None
    ics = {}
    for day, g in df.groupby("date"):
        sub = g[[feat, "fwd"]].dropna()
        if len(sub) < MIN_ITEMS_PER_DATE:
            continue
        y = sub["fwd"].to_numpy()
        if rng is not None:
            y = rng.permutation(y)
        # Spearman = Pearson on ranks.
        fr = pd.Series(sub[feat].to_numpy()).rank().to_numpy()
        yr = pd.Series(y).rank().to_numpy()
        if fr.std() == 0 or yr.std() == 0:
            continue
        ics[pd.Timestamp(day)] = float(np.corrcoef(fr, yr)[0, 1])
    return pd.Series(ics).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-price", type=float, default=1.0)
    ap.add_argument(
        "--days-back", type=int, default=3650, help="history window; longer = more independent 180d episodes"
    )
    args = ap.parse_args()

    voted = _load_voted(args.days_back, args.min_price)
    feats = _engineer(voted)
    fwd = _forward_return_col(voted)
    feats["date"] = pd.to_datetime(feats["date"])
    fwd["date"] = pd.to_datetime(fwd["date"])
    df = feats.merge(fwd, on=["item_id", "date"])
    # Anchor on month starts: enough dates for a within-date read without
    # scoring every overlapping day.
    df = df[df["date"].dt.is_month_start | (df["date"].dt.day <= 3)]

    print(
        f"signal probe h={HORIZON}d  min_price=${args.min_price:g}  "
        f"days_back={args.days_back}  items={voted['item_id'].nunique():,}"
    )
    rows = []
    for feat in FEATURES:
        ic = _within_date_ic(df, feat)
        placebo = _within_date_ic(df, feat, seed=0)
        if ic.empty:
            continue
        # Independent episodes: anchor dates >=180d apart, greedily.
        indep_dates, last = [], None
        for d in ic.index:
            if last is None or (d - last).days >= HORIZON:
                indep_dates.append(d)
                last = d
        ic_indep = ic.loc[indep_dates]
        t = (
            ic_indep.mean() / (ic_indep.std(ddof=1) / np.sqrt(len(ic_indep)))
            if len(ic_indep) > 1 and ic_indep.std(ddof=1) > 0
            else np.nan
        )
        rows.append(
            {
                "feature": feat,
                "n_dates": len(ic),
                "mean_ic_all": round(ic.mean(), 4),
                "n_indep": len(ic_indep),
                "mean_ic_indep": round(ic_indep.mean(), 4),
                "t_indep": round(float(t), 2),
                "placebo_ic": round(placebo.mean(), 4),
            }
        )
    out = pd.DataFrame(rows)
    print("\nRead t_indep against +-2. mean_ic_all is inflated by overlap; placebo_ic is the ~0 floor.\n")
    with pd.option_context("display.width", 140):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
