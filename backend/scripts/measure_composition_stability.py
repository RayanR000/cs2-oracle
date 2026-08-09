#!/usr/bin/env python3
"""Is the reversal signal a return, or a change of measurement basis?

The project's strongest "signal" is a reversal: ranking items by ``-return_1d``
predicts the forward return with a rank IC around 0.17 at 3d, better than the
model that consumes it. The archive's consensus price is a median over whichever
price sources reported that item that day, so when the reporting set changes
between ``t`` and ``t+h`` the measured return partly records a change of
*measurement basis* rather than a change of price. If the reversal survives
holding the source composition still, it is a return. If it vanishes, the label
is a quoting artifact and no feature or architecture work can reach it.

This script measures the rank IC of ``-r_t`` against the forward ``horizon``-day
return on the voted daily series, partitioned by whether the source composition
held still across the label window.

Definitions, each of which changes the answer:

* **Composition** has two bases, selected by ``--basis``:

  ``set`` (default) — the *set of source names* that voted on the item-day,
  carried as a bitmask over the archive's distinct sources. This is the basis
  that can tell one source being swapped for another apart from no change at
  all, which is exactly the case the count basis misses, and missing it biases
  the instrument toward "composition does not matter".

  ``count`` — ``n_ask_sources``, the number of distinct ask sources, straight
  off ``ItemForecaster._apply_multi_source_voting``.

  Both are built from the same rows the vote consumed, with bids already
  excluded, and a NULL ``source`` treated as one named source rather than as an
  unknown. That last point matters: every archive row before 2026 carries
  ``source IS NULL``, so under either basis the whole pre-2026 series reads as
  one constant source. Treating NULL as "never equal to itself" instead — which
  is what the refuted ``2026-08-08-model-review.md`` §5 measurement did — throws
  away 13 years and silently turns the comparison into 2026-vs-history.

* **Composition stable** for an item over ``t-1 … t+h`` means the composition
  takes the *same value* on every one of those days for that item **and every
  one of those days is present for that item**. A gap in the window is not
  stable: an absent day is an unobserved composition, not a matching one.
* **Rank IC** is computed *within date* and then averaged across dates, with
  ``t = mean / (sd / sqrt(n_dates))``. A pooled Spearman over all item-days is a
  different and much larger number because it absorbs the cross-sectional market
  factor, which is the thing this measurement is trying to look past.
* Returns use **exact calendar-day** lookups on both legs, matching
  ``prepare_targets``. An as-of lookup would silently substitute a stale price
  for a missing day, which is the artifact under test.

The script reads the archive directly, so ``prepare_targets`` never runs and
nothing else drops the dates whose labels that path voids. It applies the
exclusion itself, with the same two shapes ``prepare_targets`` uses — and
applied to the whole ``t-1 … t+h`` window, because ``-r_t`` is a return over
``(t-1, t]`` and is corruptible in exactly the way the label is:

* ``_snapshot_dates`` is an **endpoint** rule. A re-published day is a stale
  price, so a return measured to or from it is fabricated; a copy sitting
  mid-window shifts no level and is harmless.
* ``_collection_shift_dates`` is a **span** rule. A cutover anywhere inside
  ``(t-1, t+h]`` quotes the two ends of the window on different source bases.
  An endpoint-only exclusion would leave, for instance, an ``h=3`` window
  anchored two days before an isolated cutover entirely intact.

That makes the measurement conservative: the archive's largest known basis
changes are gone from *every* cell, including "all rows", before composition is
partitioned on at all.

Read-only. It opens no database session — ``config.py`` reads ``.env`` relative
to the working directory, so a script run from ``backend/`` that opened one
would be talking to production.

Usage::

    # primary: the source-labelled era, composition as the set of source names
    venv/bin/python scripts/measure_composition_stability.py --horizon 3 --from 2026-01-01

    # secondary: the whole archive on the count basis, which assumes the NULL
    # source of 2013-2025 is one constant source
    venv/bin/python scripts/measure_composition_stability.py \\
        --horizon 3 --from 2013-08-14 --basis count
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.archive import prices_relation  # noqa: E402
from models.item_parser import archive_universe_sql_filter  # noqa: E402

#: A cell with fewer contributing dates reads `underpowered` and must never be
#: quoted as a number. The multi-source era of the archive is a few dozen days
#: deep, so the cleanest cells are expected to land here; that is a legitimate
#: result and the floor is not to be lowered to get past it.
MIN_DATES_TO_REPORT = 30

#: A date needs at least this many items in a cell to contribute a rank IC.
#: Two items give a Spearman of exactly +/-1 whatever the data says; averaging
#: such a date in unweighted adds noise and nothing else. Kept deliberately
#: small — this is a degeneracy guard, not a power filter.
MIN_ITEMS_PER_DATE = 5

#: `n_ask_sources >= this` is the "several sources agree" cell.
MULTI_SOURCE_FLOOR = 3

#: What a NULL `source` is called when the source set is built. It is a name
#: like any other, so two NULL-source days read as the same composition.
NULL_SOURCE_LABEL = "<null>"

#: The column each `--basis` reads. `source_mask` is a bitmask over the
#: archive's distinct source names, so equality of the mask is equality of the
#: set; `n_ask_sources` is only its cardinality.
COMPOSITION_COLUMN = {"set": "source_mask", "count": "n_ask_sources"}

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent / "price-archive"

# Day zero for the integer day index the window arithmetic runs on. Any fixed
# date before the archive starts (2013-08-14) works; this one is round.
_EPOCH = pd.Timestamp("2013-01-01")


def load_voted_series(archive_dir: Path, start: date, end: date | None = None
                      ) -> pd.DataFrame:
    """The voted daily series over the archive, as production computes it.

    Reads through :func:`db.archive.prices_relation` with
    :func:`models.item_parser.archive_universe_sql_filter` applied, then passes
    the frame through ``ItemForecaster._apply_multi_source_voting`` — the same
    static method the production loader calls — rather than reimplementing the
    vote. A reimplementation that drifted from production would stop measuring
    production's label while still looking like it was.

    Returns columns ``item_id, date, price, volume, n_ask_sources,
    source_mask``.
    """
    import duckdb

    from models.forecaster import ItemForecaster

    con = duckdb.connect()
    try:
        relation = prices_relation(
            con, archive_dir,
            columns=["item_slug", "day", "mean_price", "volume", "source"])
        window = f"AND day <= DATE '{end}'" if end is not None else ""
        df = con.sql(f"""
            SELECT item_slug AS item_id, day AS timestamp, mean_price AS price,
                   volume, source
            FROM {relation} sub
            WHERE day >= DATE '{start}'
              {window}
              AND (source IS NULL OR source NOT LIKE 'historical_fallback:%')
              AND {archive_universe_sql_filter("sub.item_slug", "sub.source")}
        """).fetchdf()
    finally:
        con.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    # Some Parquet years store mean_price/volume as VARCHAR, and the glob union
    # then coerces the whole column to string.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["price"])

    voted = ItemForecaster._apply_multi_source_voting(df)
    voted = voted.merge(source_masks(df), on=["item_id", "date"], how="left",
                        validate="one_to_one")

    # The two bases must agree on cardinality or one of them is describing rows
    # the other did not see. This is the only cheap check that the set was built
    # from the same rows production voted on.
    mismatch = int((np.bitwise_count(voted["source_mask"].to_numpy())
                    != voted["n_ask_sources"].to_numpy()).sum())
    if mismatch:
        raise ValueError(
            f"{mismatch:,} item-days where the source set's size disagrees with "
            "n_ask_sources; the set and the vote are reading different rows")
    return voted


def source_masks(df: pd.DataFrame) -> pd.DataFrame:
    """One bitmask per item-day over the source names that voted on it.

    Built from the same frame the vote consumed — after the price dropna, with
    bids already excluded by the universe filter — so it describes exactly the
    rows ``n_ask_sources`` counted. Computing it from a separate query would let
    the two drift apart while both still looked right.

    A bitmask rather than a joined string: there are a dozen distinct sources in
    thirteen years, so the whole set fits in an integer, and set equality
    becomes integer equality on a column that costs 8 bytes a row instead of a
    Python object.

    ``_apply_multi_source_voting`` is not asked for this. It is production's
    loader and a later task on this branch owns its cache version; adding a
    column here keeps that surface untouched.
    """
    codes, names = pd.factorize(df["source"].fillna(NULL_SOURCE_LABEL))
    if len(names) > 62:
        raise ValueError(
            f"{len(names)} distinct sources will not fit in an int64 bitmask; "
            "the set basis needs a different encoding")

    # Group on one integer key rather than on the (item_id, date) object pair.
    # The pair is a string and a `datetime.date`, and grouping 20M rows of those
    # is minutes; the same grouping on int64 is seconds.
    item_codes, items = pd.factorize(df["item_id"])
    date_codes, dates = pd.factorize(df["date"])
    key = item_codes.astype(np.int64) * len(dates) + date_codes

    distinct = pd.DataFrame({
        "key": key,
        "bit": (np.int64(1) << codes.astype(np.int64)),
    }).drop_duplicates()
    # Summing DISTINCT bits within an item-day is a bitwise OR, which pandas has
    # no groupby aggregation for. The drop_duplicates above is what makes the
    # sum an OR rather than a count.
    masks = distinct.groupby("key", sort=False)["bit"].sum()

    grouped_key = masks.index.to_numpy()
    return pd.DataFrame({
        "item_id": items.to_numpy()[grouped_key // len(dates)],
        "date": dates.to_numpy()[grouped_key % len(dates)],
        "source_mask": masks.to_numpy(),
    })


def voided_dates(voted: pd.DataFrame) -> tuple[frozenset, frozenset]:
    """``(snapshot_dates, collection_shift_dates)`` for this frame.

    The two detectors ``prepare_targets`` runs, kept apart because they void
    different things — see :func:`voided_anchors`. Both are instance methods
    that touch nothing on the instance, so this constructs a forecaster with no
    database session at all rather than mocking one.
    """
    from models.forecaster import ItemForecaster

    forecaster = ItemForecaster(db_session=None, model_dir=tempfile.mkdtemp())
    return (forecaster._snapshot_dates(voted),
            forecaster._collection_shift_dates(voted))


def voided_anchors(snapshots: frozenset, shifts: frozenset, horizon: int
                   ) -> frozenset:
    """Anchor dates ``t`` whose ``t-1 … t+h`` window is unusable.

    A snapshot at ``d`` voids the anchors that would read it as an endpoint:
    ``d+1`` (its ``t-1`` leg), ``d`` itself, and ``d-h`` (its target leg).
    A collection shift at ``d`` voids every anchor whose window spans it, which
    is ``t-1 < d <= t+h`` — i.e. ``t`` from ``d-h`` through ``d``.
    """
    void: set[date] = set()
    for day in snapshots:
        void.update({day + timedelta(days=1), day, day - timedelta(days=horizon)})
    for day in shifts:
        void.update(day - timedelta(days=k) for k in range(horizon + 1))
    return frozenset(void)


def build_windows(voted: pd.DataFrame, horizon: int, min_price: float,
                  basis: str = "set") -> pd.DataFrame:
    """One row per usable ``(item, t)``, with its returns and stability flag.

    Columns: ``date``, ``x`` (``-r_t``), ``y`` (the forward ``horizon``-day
    return), ``stable``, ``n_ask_sources`` (at ``t``).

    *basis* selects which column composition is read from — see
    :data:`COMPOSITION_COLUMN`. It changes only ``stable``; the source-count
    cells always read ``n_ask_sources``.
    """
    composition_col = COMPOSITION_COLUMN[basis]
    wanted = ["item_id", "date", "price", "n_ask_sources"]
    if composition_col not in wanted:      # the count basis reads a column
        wanted.append(composition_col)     # that is already in the list
    frame = voted[wanted].copy()
    frame["date"] = pd.to_datetime(frame["date"])

    # A single int64 key per (item, day) turns every window lookup into a hash
    # join on a flat index: `key + k` is the same item k days later. Item codes
    # are dense from factorize and the day index is well under 100_000 (that is
    # 273 years), so the two never collide.
    codes, _ = pd.factorize(frame["item_id"])
    day_index = (frame["date"] - _EPOCH).dt.days.to_numpy()
    frame["_key"] = codes.astype(np.int64) * 100_000 + day_index

    # The vote yields one row per item-day; nothing downstream enforces it, and
    # a duplicated key would make the reindex lookups ambiguous.
    frame = frame.drop_duplicates("_key")

    keys = frame["_key"].to_numpy()
    price = pd.Series(frame["price"].to_numpy(), index=keys)

    p_t = frame["price"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_prev = price.reindex(keys - 1).to_numpy(dtype=float)
        p_fwd = price.reindex(keys + horizon).to_numpy(dtype=float)
        r_1d = (p_t - p_prev) / p_prev
        forward = (p_fwd - p_t) / p_t

    # Composition over t-1 ... t+h. Row POSITIONS are looked up rather than the
    # composition values themselves: a reindex that introduces a NaN casts the
    # column to float, and a 62-bit source mask does not survive that intact.
    positions = pd.Series(np.arange(len(frame), dtype=np.int64), index=keys)
    values = frame[composition_col].to_numpy(dtype=np.int64)
    window = [positions.reindex(keys + k).to_numpy(dtype=float)
              for k in range(-1, horizon + 1)]

    # An absent day comes back NaN — an unobserved composition, not a matching
    # one.
    present = np.ones(len(frame), dtype=bool)
    for offset in window:
        present &= ~np.isnan(offset)
    resolved = [np.nan_to_num(offset, nan=0.0).astype(np.int64)
                for offset in window]
    same = np.ones(len(frame), dtype=bool)
    for offset in resolved[1:]:
        same &= (values[offset] == values[resolved[0]])
    stable = present & same

    out = pd.DataFrame({
        "date": frame["date"].to_numpy(),
        "x": -r_1d,
        "y": forward,
        "stable": stable,
        "n_ask_sources": frame["n_ask_sources"].to_numpy(),
        "price": p_t,
    })
    usable = np.isfinite(out["x"]) & np.isfinite(out["y"])
    # Never pool across price tiers: the cheap tail is a different market with a
    # different staleness rate, and mixing it in is the penny-item score.
    return out[usable & (out["price"] >= min_price)].drop(columns=["price"])


def rank_ic(cell: pd.DataFrame) -> dict:
    """Mean within-date Spearman of ``x`` against ``y``, and its t statistic.

    Ranks are taken within each date and correlated there; the per-date series
    is then averaged, so the cross-sectional market factor common to a date
    cannot inflate the result the way a pooled Spearman does.
    """
    n_rows = len(cell)
    if n_rows < MIN_ITEMS_PER_DATE:
        return {"n_dates": 0, "n_rows": n_rows, "rank_ic": None, "t": None}

    grouped = cell.groupby("date", sort=True)
    ranks = pd.DataFrame({
        "date": cell["date"].to_numpy(),
        "rx": grouped["x"].rank().to_numpy(),
        "ry": grouped["y"].rank().to_numpy(),
    })
    ranks["xy"] = ranks["rx"] * ranks["ry"]
    ranks["xx"] = ranks["rx"] ** 2
    ranks["yy"] = ranks["ry"] ** 2

    by_date = ranks.groupby("date", sort=True)
    n = by_date.size()
    sx, sy = by_date["rx"].sum(), by_date["ry"].sum()
    sxy, sxx, syy = by_date["xy"].sum(), by_date["xx"].sum(), by_date["yy"].sum()

    numerator = sxy - sx * sy / n
    denominator = np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
    # A date where either side is constant has no defined correlation; it is
    # dropped rather than counted as zero, which would be an assertion the data
    # does not make.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_date = (numerator / denominator)[
            (n >= MIN_ITEMS_PER_DATE) & (denominator > 0)].dropna()

    n_dates = len(per_date)
    if n_dates == 0:
        return {"n_dates": 0, "n_rows": n_rows, "rank_ic": None, "t": None}

    mean = float(per_date.mean())
    sd = float(per_date.std(ddof=1)) if n_dates > 1 else float("nan")
    t_stat = (mean / (sd / np.sqrt(n_dates))
              if n_dates > 1 and sd > 0 else None)
    return {
        "n_dates": n_dates,
        "n_rows": n_rows,
        "rank_ic": mean,
        "t": None if t_stat is None else float(t_stat),
    }


def partitions(windows: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    stable = windows["stable"]
    n_src = windows["n_ask_sources"]
    return [
        ("all rows", windows),
        ("composition stable", windows[stable]),
        ("composition changed", windows[~stable]),
        ("stable & single source", windows[stable & (n_src == 1)]),
        (f"stable & >={MULTI_SOURCE_FLOOR} sources",
         windows[stable & (n_src >= MULTI_SOURCE_FLOOR)]),
    ]


def measure(archive_dir: Path, horizon: int, start: date, min_price: float,
            basis: str = "set", end: date | None = None) -> list[dict]:
    voted = load_voted_series(archive_dir, start, end)
    print(f"voted series: {len(voted):,} item-days, "
          f"{voted['item_id'].nunique():,} items")

    snapshots, shifts = voided_dates(voted)
    print(f"snapshot dates ({len(snapshots)}): "
          f"{', '.join(str(d) for d in sorted(snapshots)) or 'none'}")
    print(f"collection shift dates ({len(shifts)}): "
          f"{', '.join(str(d) for d in sorted(shifts)) or 'none'}")

    windows = build_windows(voted, horizon, min_price, basis)
    void = voided_anchors(snapshots, shifts, horizon)
    before = len(windows)
    windows = windows[~windows["date"].dt.date.isin(void)]
    print(f"voided {before - len(windows):,} of {before:,} windows on "
          f"{len(void)} anchor dates")
    print(f"usable (item, t) windows at >= ${min_price:g}: {len(windows):,}")

    results = []
    for name, cell in partitions(windows):
        stats = rank_ic(cell)
        stats["cell"] = name
        stats["verdict"] = ("measured" if stats["n_dates"] >= MIN_DATES_TO_REPORT
                            else "underpowered")
        results.append(stats)
    return results


def report(results: list[dict], horizon: int, start: date, min_price: float,
           basis: str) -> None:
    described = ("the SET of source names" if basis == "set"
                 else "the COUNT of ask sources")
    print()
    print(f"rank IC of -r_t vs the forward {horizon}d return, "
          f">= ${min_price:g}, from {start}")
    print(f"composition basis: {basis} ({described})")
    print(f"(within-date Spearman, averaged across dates; "
          f"cells under {MIN_DATES_TO_REPORT} dates are not quotable)")
    print()
    print(f"{'cell':28s} {'n_dates':>8s} {'n_rows':>12s} "
          f"{'rank_ic':>9s} {'t':>8s}  verdict")
    for row in results:
        if row["verdict"] == "measured":
            ic = f"{row['rank_ic']:+9.4f}"
            t = "     n/a" if row["t"] is None else f"{row['t']:8.1f}"
        else:
            # Never print a number for an underpowered cell. The whole point of
            # the floor is that the nearest quotable figure is not available.
            ic, t = f"{'--':>9s}", f"{'--':>8s}"
        print(f"{row['cell']:28s} {row['n_dates']:8d} {row['n_rows']:12,d} "
              f"{ic} {t}  {row['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--horizon", type=int, default=3, choices=[3, 7, 14, 30])
    parser.add_argument("--from", dest="start", default="2013-08-14",
                        help="first archive day to read (default: the archive's "
                             "first day)")
    parser.add_argument("--to", dest="end", default=None,
                        help="last archive day to read; for prototyping on a "
                             "narrow range")
    parser.add_argument("--min-price", type=float, default=1.0,
                        help="anchor-price floor in USD (default: 1.0)")
    parser.add_argument("--basis", default="set",
                        choices=sorted(COMPOSITION_COLUMN),
                        help="what composition means: the set of source names "
                             "(default) or their count")
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_ROOT)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else None
    results = measure(args.archive_dir, args.horizon, start, args.min_price,
                      args.basis, end)
    report(results, args.horizon, start, args.min_price, args.basis)


if __name__ == "__main__":
    main()
