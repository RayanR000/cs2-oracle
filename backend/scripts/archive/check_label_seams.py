#!/usr/bin/env python3
"""Does the training label fabricate market-wide moves where the feed changes?

The archive stores one row per (item, day, **source**); the consensus step
collapses whichever sources reported that day into a single "price". When the
reporting set changes between `t-1` and `t`, the measured return partly records
a change of *measurement basis* rather than a change of price. On 2026-07-09
only `aggregator_sync` quoted (5,502 items) and on 2026-07-10 only
`aggregator_steam_17mafo` (26,178) -- with **no same-day overlap** -- so the
voted series bounces between two differently-scaled feeds on alternating days
and reports the level difference as a market-wide move.

This script is the gate on that. It computes the cross-sectional **median**
one-day return per date -- median, so it describes the typical item rather than
a tail -- and flags any date whose move is too large to be a market. The median
is the right statistic precisely because a real market move of this size would
have to lift the whole cross-section, and a basis change does exactly that,
while an outlier or a thin cohort does not.

Two bases, selected by `--basis`:

``voted`` (default)
    Production's label: `ItemForecaster._apply_multi_source_voting`, the same
    static method the production loader calls. This is the frame the model
    trains on, so this is the basis whose failures are real.

``within-source``
    The proposed clean label: a return is computed only from sources quoting
    the item on **both** legs, then taken as the median over those common
    sources. Cross-source level differences cannot enter, because no return
    ever spans two sources. A NULL `source` is one named source like any other
    -- every archive row before 2026 carries `source IS NULL`, and treating
    NULL as never equal to itself would drop 13 years and silently turn the
    comparison into 2026-vs-history.

Running both is the before/after: the same measurement, one label apart.

A flagged date is not automatically an artifact. This measures "the whole
cross-section moved together", and a real market-wide move does that too --
which is what the `source_set_changed` column is for. Measured 2026-08-22:

* **2026** flags 2026-03-22, 07-09 and 07-10 on the `voted` basis, each with
  ~100% of the cohort changing source set that day, and **none** on
  `within-source`. Those are seams.
* **2025** flags 02-01, 04-01, 10-23, 10-24, 10-25, 10-31 and 11-01 at **0%**
  source change -- the legacy series is single-source, so a seam cannot be the
  cause. All seven **reproduce in an independently collected dataset**,
  `runtime/steam_listing_history.db` (262 items): -6.06 / -4.91 / -13.03 /
  -10.10 / +12.41 / +7.36 / +4.48% against the archive's -6.36 / -5.90 /
  -13.10 / -10.93 / +13.31 / +7.82 / +6.90%. Same venue, separate collection,
  so this rules out *our* pipeline rather than Steam's estimator -- but the
  2025 legacy series needs no repair, and a rerun of that check is wasted work.

Exit status is 1 when any date is flagged, so this can gate a retrain.

    venv/bin/python scripts/check_label_seams.py --from 2026-06-01 \
        --archive-dir ../../cs2-oracle-data/price-archive
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.item_parser import (
    BID_SOURCES,
    STEAM_SPOT_SOURCES,
    TRAILING_WINDOW_SOURCES,
)
from scripts.archive.measure_composition_stability import (
    ARCHIVE_ROOT,
    NULL_SOURCE_LABEL,
    load_source_rows,
    load_voted_series,
)

#: A cross-sectional median one-day return at or above this reads as a change
#: of measurement basis, not a market. The daily cross-sectional median return
#: has a standard deviation of 2.25% (`2026-08-19-deep-model-review.md` s1), so
#: 5% is over 2sd of the *median itself* -- an event that would move nearly
#: every item at once by more than skins have ever moved in a day. The observed
#: seams are +26.5% and -13.7%, so the threshold is not near them and does not
#: need tuning to catch one.
DEFAULT_THRESHOLD = 0.05

#: A date needs this many items before its median can flag. A large median over
#: a handful of items is a small-sample artifact, and flagging it would spend
#: the detector's credibility on the cells least able to support a claim.
DEFAULT_MIN_ITEMS = 100

#: Sources that must never enter a return, on either basis. The consensus drops
#: them before the group is read (`forecaster.py:2103`), so a within-source
#: return that kept them would be measuring a different label: a bid is the
#: wrong side of the book and a trailing mean is the wrong time basis.
EXCLUDED_SOURCES = BID_SOURCES | TRAILING_WINDOW_SOURCES | STEAM_SPOT_SOURCES

#: 1.4826 * MAD estimates a Gaussian sd but survives the seams it is measuring.
_MAD_TO_SD = 1.4826


def daily_median_returns(
    prices: pd.DataFrame,
    min_price: float,
) -> pd.DataFrame:
    """Cross-sectional median one-day return per date.

    *prices* is `item_id, date, price` -- one price per item-day, whatever
    basis produced it. Returns `date, n_items, median_return`.

    Both legs must be **consecutive calendar days** for the same item, matching
    `prepare_targets`'s exact-day lookup. An as-of lookup would substitute a
    stale price for a missing day and report a multi-day move as a daily one,
    which is the artifact under test. The *min_price* floor reads the anchor
    (`t-1`) leg, as the served cohort does; filtering on the later price would
    let an item that mooned into the cohort join it, which is selection on the
    outcome.
    """
    frame = prices[["item_id", "date", "price"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["item_id", "date"])

    prev_date = frame.groupby("item_id")["date"].shift(1)
    prev_price = frame.groupby("item_id")["price"].shift(1)
    consecutive = (frame["date"] - prev_date).dt.days == 1
    frame = frame[consecutive & (prev_price >= min_price) & (prev_price > 0)]
    returns = frame["price"] / prev_price[frame.index] - 1.0

    out = pd.DataFrame({"date": frame["date"], "return": returns})
    grouped = out.groupby("date")["return"]
    return pd.DataFrame(
        {
            "date": grouped.median().index,
            "n_items": grouped.size().to_numpy(),
            "median_return": grouped.median().to_numpy(),
        }
    ).reset_index(drop=True)


def within_source_returns(rows: pd.DataFrame) -> pd.DataFrame:
    """One-day returns computed inside a single source, never across two.

    *rows* is the raw per-source frame `item_id, date, source, price`. For each
    item and consecutive day pair, the return is the median over the sources
    quoting that item on **both** days. A pair with no common source yields no
    row: an absent source is an unobserved price, and substituting another
    feed's level for it fabricates exactly the move this basis exists to avoid.

    Returns `item_id, date, return, sources` -- `sources` naming the common
    sources used, so a flagged date stays attributable.
    """
    frame = rows[["item_id", "date", "source", "price"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["source"] = frame["source"].fillna(NULL_SOURCE_LABEL)
    frame = frame[~frame["source"].isin(EXCLUDED_SOURCES)]
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["price"])
    if frame.empty:
        return pd.DataFrame(columns=["item_id", "date", "return", "sources"])

    # One price per (item, date, source): a source can print an item twice in a
    # day, and the median over its own prints is the same rule the vote uses.
    frame = frame.groupby(["item_id", "date", "source"], as_index=False)["price"].median()

    frame = frame.sort_values(["item_id", "source", "date"])
    key = ["item_id", "source"]
    prev_date = frame.groupby(key)["date"].shift(1)
    prev_price = frame.groupby(key)["price"].shift(1)
    paired = ((frame["date"] - prev_date).dt.days == 1) & (prev_price > 0)
    frame = frame[paired]
    if frame.empty:
        return pd.DataFrame(columns=["item_id", "date", "return", "sources"])
    frame["return"] = frame["price"] / prev_price[frame.index] - 1.0

    out = (
        frame.groupby(["item_id", "date"])
        .agg(**{"return": ("return", "median"), "sources": ("source", lambda s: "+".join(sorted(s)))})
        .reset_index()
    )
    return out


def within_source_index(rows: pd.DataFrame, voted: pd.DataFrame) -> pd.DataFrame:
    """A per-item price *level* series with the cross-source seam steps removed.

    The gate and the trainer both need a single price per item-day to roll a
    dispersion and a forward return over. This starts from the **voted** level
    -- which already carries the consensus outlier vote, so a single source's
    bad print cannot move it -- and surgically neutralises only the steps that
    the vote cannot fix: the days where the reporting set turns over completely,
    so the measured return is a change of feed, not of price.

    A day keeps its voted return when at least one source quoted the item on
    **both** it and the day before (`within_source_returns` reports such a day);
    the return is then a genuine same-basis move and the vote has already
    cleaned it. A day with **no** common source -- the seam, where every quote
    is on a new basis -- contributes ``0`` instead: the level is carried forward
    unchanged rather than stepped across the cutover. That is the whole repair,
    and it is deliberately minimal. It does not rebuild the level from raw
    single-source returns, which would throw away the outlier vote and let a
    fat-finger print (observed up to 8.4e6% here) compound into the series.

    The level is anchored at each item's first voted price and returns
    ``item_id, date, price`` over exactly the voted item-days, so the cohort and
    row count match the voted basis and the two are paired.
    """
    base = voted[["item_id", "date", "price"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["item_id", "date"])

    # The robust step is the voted return; a cross-source-only day has no
    # within-source return and is zeroed. `has_within` marks the days that DO.
    steps = within_source_returns(rows)[["item_id", "date"]].copy()
    steps["date"] = pd.to_datetime(steps["date"])
    steps["has_within"] = True
    merged = base.merge(steps, on=["item_id", "date"], how="left")
    merged["has_within"] = merged["has_within"].fillna(False).astype(bool)

    prev_price = merged.groupby("item_id")["price"].shift(1)
    voted_ret = merged["price"] / prev_price - 1.0
    # First day of each item (prev_price is NaN) anchors: factor 1.0.
    factor = np.where(merged["has_within"].to_numpy() & prev_price.notna().to_numpy(), 1.0 + voted_ret.to_numpy(), 1.0)
    merged["cum"] = pd.Series(factor, index=merged.index).groupby(merged["item_id"]).cumprod()
    anchor = merged.groupby("item_id")["price"].transform("first")
    merged["price"] = anchor * merged["cum"]
    return merged[["item_id", "date", "price"]]


def within_source_prices(rows: pd.DataFrame) -> pd.DataFrame:
    """The anchor price each within-source return is measured from.

    `daily_median_returns` needs a price level to apply the cohort floor to,
    and the within-source basis has no single "price" -- it has one per source.
    The median over the item-day's voting sources is used, which is the
    consensus level; only the *return* is basis-restricted, so the floor
    selects the same cohort on both bases and the comparison stays paired.
    """
    frame = rows[["item_id", "date", "source", "price"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["source"] = frame["source"].fillna(NULL_SOURCE_LABEL)
    frame = frame[~frame["source"].isin(EXCLUDED_SOURCES)]
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    return frame.dropna(subset=["price"]).groupby(["item_id", "date"], as_index=False)["price"].median()


def flag_seams(
    daily: pd.DataFrame,
    threshold: float,
    min_items: int,
) -> pd.DataFrame:
    """Mark the dates whose median move is too large to be a market.

    Adds `flagged` and `robust_z`. The robust scale is a MAD over the
    **unflagged** days only: leaving the seams in the yardstick they are
    measured against collapses their own z-scores toward the bulk and makes the
    report understate how anomalous they are.
    """
    out = daily.copy()
    out["flagged"] = (out["median_return"].abs() >= threshold) & (out["n_items"] >= min_items)

    bulk = out.loc[~out["flagged"], "median_return"]
    scale = float(_MAD_TO_SD * (bulk - bulk.median()).abs().median()) if len(bulk) else 0.0
    out["robust_z"] = (out["median_return"] - (bulk.median() if len(bulk) else 0.0)) / scale if scale > 0 else np.nan
    return out


def attribute(daily: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Add the share of the cohort whose source set changed from the day before.

    A flagged date is only actionable if it can be tied to the cutover, and a
    seam shows up here as a share near 1.0.
    """
    frame = rows[["item_id", "date", "source"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["source"] = frame["source"].fillna(NULL_SOURCE_LABEL)
    frame = frame[~frame["source"].isin(EXCLUDED_SOURCES)]
    sets = frame.groupby(["item_id", "date"])["source"].agg(lambda s: "+".join(sorted(set(s)))).reset_index()

    sets = sets.sort_values(["item_id", "date"])
    prev_date = sets.groupby("item_id")["date"].shift(1)
    prev_set = sets.groupby("item_id")["source"].shift(1)
    sets = sets[(sets["date"] - prev_date).dt.days == 1]
    sets["changed"] = sets["source"].to_numpy() != prev_set[sets.index].to_numpy()

    share = sets.groupby("date")["changed"].mean().rename("source_set_changed")
    return daily.merge(share, on="date", how="left")


def report(flagged: pd.DataFrame, basis: str, min_price: float, threshold: float) -> None:
    print(f"\nLabel seam check - basis={basis}, >=${min_price:.2f} cohort, threshold={threshold:.1%}")
    print(f"{'date':12s} {'items':>8s} {'median ret':>11s} {'robust z':>9s} {'src changed':>12s}  flag")
    for _, row in flagged.iterrows():
        changed = row.get("source_set_changed")
        changed = "-" if pd.isna(changed) else f"{changed:11.1%}"
        z = "-" if pd.isna(row["robust_z"]) else f"{row['robust_z']:9.1f}"
        print(
            f"{row['date'].date()!s:12s} {int(row['n_items']):8,d} "
            f"{row['median_return']:10.2%} {z} {changed}  "
            f"{'FLAG' if row['flagged'] else ''}"
        )

    n = int(flagged["flagged"].sum())
    if n:
        worst = flagged.loc[flagged["flagged"], "median_return"].abs().max()
        print(
            f"\nFAIL: {n} date(s) move the cross-sectional median by more "
            f"than {threshold:.0%} (worst {worst:.1%}). On the `voted` basis "
            f"this is the label, and every feature and label computed across "
            f"such a date is fabricated."
        )
    else:
        print(f"\nPASS: no date moves the cross-sectional median by more than {threshold:.0%}.")


def measure(
    archive_dir: Path,
    start: date,
    end: date | None,
    basis: str,
    min_price: float,
    threshold: float,
    min_items: int,
) -> pd.DataFrame:
    rows = load_source_rows(archive_dir, start, end)
    if basis == "voted":
        prices = load_voted_series(archive_dir, start, end)[["item_id", "date", "price"]]
        daily = daily_median_returns(prices, min_price)
    else:
        returns = within_source_returns(rows)
        prices = within_source_prices(rows)
        # The cohort is selected exactly as on the voted basis -- consecutive
        # days, floor on the anchor leg -- so the two bases are paired and any
        # difference between them is the label, not the row set.
        eligible = _eligible_pairs(prices, min_price)
        returns = returns.merge(eligible, on=["item_id", "date"], how="inner")
        grouped = returns.groupby("date")["return"]
        daily = pd.DataFrame(
            {
                "date": grouped.median().index,
                "n_items": grouped.size().to_numpy(),
                "median_return": grouped.median().to_numpy(),
            }
        ).reset_index(drop=True)

    daily = flag_seams(daily, threshold, min_items)
    return attribute(daily, rows)


def _eligible_pairs(prices: pd.DataFrame, min_price: float) -> pd.DataFrame:
    """`item_id, date` pairs whose anchor day is consecutive and over the floor."""
    frame = prices[["item_id", "date", "price"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["item_id", "date"])
    prev_date = frame.groupby("item_id")["date"].shift(1)
    prev_price = frame.groupby("item_id")["price"].shift(1)
    keep = ((frame["date"] - prev_date).dt.days == 1) & (prev_price >= min_price)
    return frame.loc[keep, ["item_id", "date"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--from", dest="start", default="2026-01-01", help="first archive day to read (default: 2026-01-01)"
    )
    parser.add_argument("--to", dest="end", default=None, help="last archive day to read")
    parser.add_argument(
        "--basis",
        default="voted",
        choices=["voted", "within-source"],
        help="which label to measure (default: voted, i.e. production's)",
    )
    parser.add_argument("--min-price", type=float, default=1.0, help="anchor-price floor in USD (default: 1.0)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"median move that reads as a basis change (default: {DEFAULT_THRESHOLD:.0%})",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=DEFAULT_MIN_ITEMS,
        help=f"items a date needs before it can flag (default: {DEFAULT_MIN_ITEMS})",
    )
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_ROOT)
    args = parser.parse_args()

    flagged = measure(
        args.archive_dir,
        date.fromisoformat(args.start),
        date.fromisoformat(args.end) if args.end else None,
        args.basis,
        args.min_price,
        args.threshold,
        args.min_items,
    )
    report(flagged, args.basis, args.min_price, args.threshold)
    sys.exit(1 if bool(flagged["flagged"].any()) else 0)


if __name__ == "__main__":
    main()
