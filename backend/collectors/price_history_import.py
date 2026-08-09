"""Shared machinery for one-shot historical price-source imports.

Source-specific fetching and parsing live in ``collectors/price_history_sources/``.
Everything here is source-agnostic and is what a second backfill source reuses.
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS
from db.archive import CANONICAL_PRICE_COLUMNS
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
    kept_rows: int
    rejected_rows: int
    worst_gap_days: int


def apply_gap_gate(
    records: list[tuple[str, date, float]],
    max_gap_days: int = MAX_GAP_DAYS,
    min_distinct_days: int = MIN_DISTINCT_DAYS,
) -> tuple[list[tuple[str, date, float]], GapGateReport]:
    """Keep only densely-observed items: both conditions must hold.

    (a) no interior gap wider than *max_gap_days*, and
    (b) at least *min_distinct_days* distinct days.

    Filtering here rather than at training is deliberate. A sparse item still
    flips ``is_backfilled`` (derived as "has any row before 2026-01-01"), so
    admitting one buys a gate entry that carries no usable features — which is
    how that flag was rendered meaningless once before, when it read 8,691 of
    8,691. Price, by contrast, is NOT filtered here: which cohort to train on
    belongs to ``TRAIN_MIN_MEDIAN_PRICE``, and the archive must not bake it in.

    Only gaps BETWEEN consecutive observations count. An item that starts late
    or stops early is judged on the span it covers. An item failing both
    conditions is counted once, as a gap rejection.
    """
    by_item: dict[str, dict[date, float]] = defaultdict(dict)
    for slug, day, price in records:
        by_item[slug].setdefault(day, price)

    kept: list[tuple[str, date, float]] = []
    kept_items = rejected_gap = rejected_sparse = rejected_rows = 0
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

        kept_items += 1
        kept.extend((slug, day, observations[day]) for day in days)

    return kept, GapGateReport(
        kept_items=kept_items,
        rejected_gap_items=rejected_gap,
        rejected_sparse_items=rejected_sparse,
        kept_rows=len(kept),
        rejected_rows=rejected_rows,
        worst_gap_days=worst_gap,
    )


#: The archive's natural key. A re-append replaces the row rather than adding
#: one, and `_append_parquet` keeps the FIRST `ingested_at`.
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


def write_archive_frame(frame: pd.DataFrame, out_dir: Path | str) -> int:
    """Append *frame* to the monthly price files under *out_dir*.

    Returns the row count written. *out_dir* is the ``price-archive`` directory
    itself, so a caller passing a staging root must append that component.
    """
    if frame.empty:
        return 0
    append_monthly(out_dir, "prices", frame, DEDUP_KEYS)
    return len(frame)
