"""Shared machinery for one-shot historical price-source imports.

Source-specific fetching and parsing live in ``collectors/price_history_sources/``.
Everything here is source-agnostic and is what a second backfill source reuses.
"""
from datetime import date


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
