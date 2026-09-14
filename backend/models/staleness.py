"""Consecutive bit-identical price runs, over the *unsmoothed* voted series.

A frozen run is a series that stopped reporting, not a market that stopped
moving. Getmansky, Lo & Makarov (2004, *JFE* 74(3), 529-609) is the mechanism:
an illiquid series reports a smoothed version of the true one, so its returns
are serially correlated and its zero returns are fabricated rather than
observed. A label measured to or from a day inside such a run is a confident
wrong number, in exactly the way `_snapshot_dates` describes for a whole
cross-section.

Two things a caller must know before quoting anything from here.

**This is a property of the raw voted daily series, not of resolved anchors.**
`backtest/price_resolution.py::resolve_anchors` takes a median over the last
`SMOOTH_WINDOW` observations, and a smoothed anchor is almost never
bit-identical to the previous one. Measured 2026-08-08: the >=$1 cohort reads
**0-1.8%** stale on resolved anchors and **12-27%** on the series here. Both are
real, they are different quantities, and neither may be used to size the other.

**The rate is dominated by the 2026 feeds.** Holding the item set fixed to what
was live in Q4 2025, the >=$1 `stale_run_days >= 1` rate steps from 0.54% in
2025-10 to 33.11% in 2026-06. The cause is that no Steam-derived series in this
archive is a point observation: `aggregator_steam_7d/30d/90d` are trailing-window
mean sale prices outright, and `aggregator_sync` and `aggregator_steam_17mafo`
are `last_24h` with a documented fallback to those same windows, which fires
precisely on the illiquid items. So a filter built on this catches a 2026 data
property far more than a 13-year one. See
`docs/superpowers/specs/2026-08-08-frozen-price-runs-design.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collectors.pipeline import FALLBACK_MAX_AGE_DAYS

# A gap wider than this breaks a run rather than continuing it. Two identical
# prices either side of a collection outage are not evidence that the feed
# froze — nothing was observed in between to be frozen.
#
# Derived from FALLBACK_MAX_AGE_DAYS for the same reason
# price_resolution.MAX_WINDOW_SPAN_DAYS is: one staleness convention across the
# codebase, so an operator overriding it by environment variable moves both the
# resolver's window and this run's break together. Like `embargo_days`, that
# makes the width env-dependent rather than constant.
#
# Not free. Measured 2026-08-08: 4.06% of consecutive observed pairs on the
# 2024+ >=$1 cohort are more than one calendar day apart, concentrated in 2026
# (missing days 2026-07-27, 07-30, 08-02, 08-03).
STALE_RUN_GAP_BREAK_DAYS = FALLBACK_MAX_AGE_DAYS


def stale_run_days(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    date_col: str = "date",
    price_col: str = "price",
    gap_break_days: int = STALE_RUN_GAP_BREAK_DAYS,
) -> pd.Series:
    """Consecutive preceding observed rows carrying a bit-identical price.

    ``0`` on a fresh price level, ``1`` on the first repeat, and so on. The
    result is indexed like *df* and is safe to assign straight back onto it;
    *df* itself is not reordered or mutated.

    *df* must already be voted to one row per (item, date) — this counts
    consecutive **observed rows**, so a frame still carrying one row per source
    would count sources as repeats. Equality is exact ``float`` equality, which
    is the intended test: the artifact being detected is a republished number,
    not a small move.

    Rows with a null price or a null date are never part of a run: they break
    it, and carry ``0`` themselves. A null cannot be shown identical to
    anything, and treating it as a continuation would let a collection hole
    manufacture an arbitrarily long run.
    """
    empty = pd.Series(np.zeros(len(df), dtype=np.int32), index=df.index)
    if df.empty:
        return empty
    missing = {item_col, date_col, price_col} - set(df.columns)
    if missing:
        raise KeyError(f"stale_run_days requires {sorted(missing)}; got {list(df.columns)}")

    work = pd.DataFrame(
        {
            "item": df[item_col].to_numpy(),
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "price": pd.to_numeric(df[price_col], errors="coerce"),
        },
        index=df.index,
    )

    # Sort is by (item, date) only. A stable kind keeps duplicate item-days in
    # their original order rather than an arbitrary one, so the result is
    # reproducible on a frame that was not fully voted — it is still wrong
    # there, but it is deterministically wrong and a test can see it.
    work = work.sort_values(["item", "date"], kind="stable")

    same_item = work["item"].eq(work["item"].shift(1))
    same_price = work["price"].eq(work["price"].shift(1))
    gap_days = (work["date"] - work["date"].shift(1)).dt.days
    within_gap = gap_days.le(gap_break_days)
    usable = work["price"].notna() & work["date"].notna()

    # A row CONTINUES the previous run only if every one of these holds. Any
    # NaN in the comparison chain makes `continues` False, which is the
    # conservative direction: an unknown is a fresh level, never a repeat.
    continues = same_item & same_price & within_gap & usable & usable.shift(1, fill_value=False)

    # Gaps-and-islands: each break opens a new group, and position within the
    # group is the run length.
    group = (~continues).cumsum()
    runs = work.groupby(group, sort=False).cumcount()
    runs = runs.where(usable, 0)

    return runs.astype(np.int32).reindex(df.index)


def stale_run_lookup(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    date_col: str = "date",
    price_col: str = "price",
    gap_break_days: int = STALE_RUN_GAP_BREAK_DAYS,
) -> dict:
    """``{(item, date): stale_run_days}`` for anchor-style point lookups.

    For callers holding anchors rather than a frame — `resolve_outcomes` knows
    a `(slug, date)` pair and needs one integer for it. Dates are normalised to
    `datetime.date` so a caller's plain `date` key matches a frame that stored
    timestamps.
    """
    if df.empty:
        return {}
    runs = stale_run_days(
        df,
        item_col=item_col,
        date_col=date_col,
        price_col=price_col,
        gap_break_days=gap_break_days,
    )
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return {(item, d.date()): int(r) for item, d, r in zip(df[item_col], dates, runs) if pd.notna(d)}
