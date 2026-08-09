"""Shared machinery for one-shot historical price-source imports.

Source-specific fetching and parsing live in ``collectors/price_history_sources/``.
Everything here is source-agnostic and is what a second backfill source reuses.
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS
from db.archive import CANONICAL_PRICE_COLUMNS, prices_relation
from db.parquet import append_monthly


class StalledSourceError(Exception):
    """Consecutive byte-identical day files: the upstream scraper stalled.

    This is fatal rather than a warning. A stalled scraper republishes the same
    prices under new dates, and a repeated price is not merely wrong — it
    becomes a frozen price RUN, which the label path treats as a real
    observation. `cs2-prices-tracker` is documented as stalling from roughly
    mid-July 2026, so a range that reaches into it must fail loudly.
    """

    def __init__(self, groups: list[list[date]]):
        self.groups = groups
        summary = "; ".join(
            f"{g[0].isoformat()}..{g[-1].isoformat()} ({len(g)} days)"
            for g in groups
        )
        super().__init__(
            f"upstream source stalled — byte-identical files across {summary}"
        )


def detect_stalled_days(day_digests: dict[date, str]) -> list[list[date]]:
    """Runs of >= 2 CONSECUTIVE days whose files hash identically.

    Adjacency is required. Two identical files a month apart are a coincidence
    of a quiet market; two in a row are a scraper that did not run.
    """
    groups: list[list[date]] = []
    run: list[date] = []
    previous_digest: str | None = None
    previous_day: date | None = None

    for day in sorted(day_digests):
        digest = day_digests[day]
        adjacent = previous_day is not None and (day - previous_day).days == 1
        if adjacent and digest == previous_digest:
            if not run:
                run = [previous_day]
            run.append(day)
        else:
            if len(run) >= 2:
                groups.append(run)
            run = []
        previous_digest, previous_day = digest, day

    if len(run) >= 2:
        groups.append(run)
    return groups


#: An item whose observations are further apart than this yields a punctured
#: series: lags compute over holes and the resolution window breaks rather than
#: continuing. Deliberately the archive's own constant, not a new number.
MAX_GAP_DAYS = MAX_WINDOW_SPAN_DAYS

#: Minimum distinct days an item must carry inside the imported range.
#:
#: NOT redundant with MAX_GAP_DAYS, which passes trivially for a near-empty
#: item: two consecutive observations have a max interior gap of 1. Measured on
#: the 90-day block, 904 items (611 at >= $1) clear the gap bar on fewer than 10
#: of 90 days, median 2 — each of which would flip ``is_backfilled`` on two
#: rows. Together the two conditions mean "densely observed over at least half a
#: year". A genuinely dense item carries 300+ of the range's 408 days, so this
#: never binds on real data.
MIN_DISTINCT_DAYS = 180


@dataclass(frozen=True)
class GapGateReport:
    kept_items: int
    rejected_gap_items: int
    rejected_sparse_items: int
    rejected_cheap_items: int
    kept_rows: int
    rejected_rows: int
    worst_gap_days: int


def apply_gap_gate(
    records: list[tuple[str, date, float]],
    max_gap_days: int = MAX_GAP_DAYS,
    min_distinct_days: int = MIN_DISTINCT_DAYS,
    min_median_price: float | None = None,
) -> tuple[list[tuple[str, date, float]], GapGateReport]:
    """Keep only densely-observed items: both conditions must hold.

    (a) no interior gap wider than *max_gap_days*, and
    (b) at least *min_distinct_days* distinct days.

    Filtering here rather than at training is deliberate. A sparse item still
    flips ``is_backfilled`` (derived as "has any row before 2026-01-01"), so
    admitting one buys a gate entry that carries no usable features — which is
    how that flag was rendered meaningless once before, when it read 8,691 of
    8,691. Price is NOT filtered here for cohort reasons: which cohort to
    train on belongs to ``TRAIN_MIN_MEDIAN_PRICE``, and the archive must not
    bake that in.

    *min_median_price*, when given, is a DATA-VALIDITY floor, not a cohort
    one: ``cs2_prices_tracker.STEAM_FEE_MULTIPLIER`` is measured accurate only
    at >= $1, so a caller importing that source passes the floor to drop rows
    it cannot fee-correct reliably. It defaults to None so the library itself
    bakes in no such decision. An item's PRICE is judged by the MEDIAN of its
    observations, not the last or mean, so a single spike cannot carry an
    otherwise-cheap item over the floor.

    Only gaps BETWEEN consecutive observations count. An item that starts late
    or stops early is judged on the span it covers. An item failing multiple
    conditions is counted once, against the first that applies (gap, then
    sparse, then cheap).
    """
    by_item: dict[str, dict[date, float]] = defaultdict(dict)
    for slug, day, price in records:
        by_item[slug].setdefault(day, price)

    kept: list[tuple[str, date, float]] = []
    kept_items = rejected_gap = rejected_sparse = rejected_cheap = rejected_rows = 0
    worst_gap = 0

    for slug, observations in by_item.items():
        days = sorted(observations)
        gaps = [(b - a).days for a, b in zip(days, days[1:])]
        item_worst = max(gaps) if gaps else 0

        if item_worst > max_gap_days:
            rejected_gap += 1
            rejected_rows += len(days)
            worst_gap = max(worst_gap, item_worst)
            continue
        if len(days) < min_distinct_days:
            rejected_sparse += 1
            rejected_rows += len(days)
            continue
        if min_median_price is not None:
            prices = sorted(observations[day] for day in days)
            mid = len(prices) // 2
            median_price = (prices[mid] if len(prices) % 2
                            else (prices[mid - 1] + prices[mid]) / 2)
            if median_price < min_median_price:
                rejected_cheap += 1
                rejected_rows += len(days)
                continue

        kept_items += 1
        kept.extend((slug, day, observations[day]) for day in days)

    return kept, GapGateReport(
        kept_items=kept_items,
        rejected_gap_items=rejected_gap,
        rejected_sparse_items=rejected_sparse,
        rejected_cheap_items=rejected_cheap,
        kept_rows=len(kept),
        rejected_rows=rejected_rows,
        worst_gap_days=worst_gap,
    )


#: The archive's natural key. A re-append REPLACES the row: `_append_parquet`
#: selects `_new` unconditionally and keeps an existing row only `WHERE NOT
#: EXISTS` a match (`db/parquet.py:250-262`), so the new row's `ingested_at`
#: wins too.
#:
#: That is wrong for an arrival timestamp, which is why `write_archive_frame`
#: restores first-arrival itself. The daily CI writer has the same problem and
#: solves it the same way (`scripts/append_to_parquet.py:267-270`); the
#: difference is that it does so before its own dedup, while `append_monthly`
#: offers no hook, so the correction has to happen on the frame going in.
DEDUP_KEYS = ["item_slug", "day", "source"]


def to_archive_frame(
    records: list[tuple[str, date, float]],
    source: str,
    ingested_at: datetime,
) -> pd.DataFrame:
    """Canonical price rows for *records*, labelled *source*.

    ``volume`` is a typed NULL, never 0. No aggregator feed carries volume and
    a real zero never occurs — a day with no sale produces an ABSENT row. The
    fabricated zeros are what kept ``has_volume`` reading True and
    ``volume_missing`` reporting "present", which shelved eleven features.

    ``ingested_at`` is the run's wall clock: when the row ARRIVED, not what it
    describes. A backfill writer violates "a row dated `d` was knowable on `d`"
    by definition, which is precisely why the column exists.
    """
    frame = pd.DataFrame(records, columns=["item_slug", "day", "mean_price"])
    frame["day"] = pd.to_datetime(frame["day"]).dt.date
    frame["source"] = source
    frame["volume"] = pd.Series([pd.NA] * len(frame), dtype="Int64")
    frame["ingested_at"] = pd.Timestamp(ingested_at)
    return frame[list(CANONICAL_PRICE_COLUMNS)]


def _preserve_first_arrival(
    frame: pd.DataFrame, out_dir: Path | str
) -> pd.DataFrame:
    """Roll each row's ``ingested_at`` back to its earliest recorded arrival.

    ``append_monthly`` replaces a colliding row wholesale, so without this a
    re-run re-stamps arrival forward on every row it touches — and the whole
    reason the column exists is the embargo. A value being corrected still
    became knowable on the day it first landed.

    ``min`` skips NaT, so a row predating the column takes the new timestamp
    rather than staying unknown.
    """
    files = sorted(Path(out_dir).glob("prices-*.parquet"))
    if not files:
        return frame

    con = duckdb.connect()
    try:
        rel = prices_relation(
            con,
            archive_dir=Path(out_dir),
            columns=["item_slug", "day", "source", "ingested_at"],
        )
        existing = con.sql(
            f"SELECT item_slug, day, source, ingested_at FROM {rel}"
        ).fetchdf()
    finally:
        con.close()

    if existing.empty:
        return frame

    existing["day"] = pd.to_datetime(existing["day"]).dt.date
    merged = frame.merge(
        existing.rename(columns={"ingested_at": "_prior"}),
        on=DEDUP_KEYS,
        how="left",
    )
    frame = frame.copy()
    frame["ingested_at"] = merged[["ingested_at", "_prior"]].min(axis=1)
    return frame


def write_archive_frame(frame: pd.DataFrame, out_dir: Path | str) -> int:
    """Append *frame* to the monthly price files under *out_dir*.

    Returns the row count written. *out_dir* is the ``price-archive`` directory
    itself, so a caller passing a staging root must append that component.
    """
    if frame.empty:
        return 0
    append_monthly(out_dir, "prices", _preserve_first_arrival(frame, out_dir),
                   DEDUP_KEYS)
    return len(frame)
