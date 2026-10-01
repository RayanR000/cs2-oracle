"""Which catalog items the forecast is SERVED for, and at which horizons.

Two cohorts, one definition, read by every serving-side lookup (the predict
universe, the prior-row blend, the forecast writer):

- **Established** — ``is_backfilled = 1``: the CSMarketAPI-backfilled items the
  model was built and calibrated on. All horizons.
- **Young** — a release the aggregator discovered after the backfill closed
  (``release_date`` set, never backfilled), once it has ``YOUNG_MIN_HISTORY_DAYS``
  of archive history. **h=3 only.**

Why h=3 only: a 2026-09-15-cutoff model replayed on the 2026-07-08 release
(~195 items >= $1) covered 83/81% at h=3 against 85/87% for established items,
but 75/74% at h=7 and 70% at h=14 against an 80% target
(`docs/changelog/2026-09-30-serve-new-releases-at-h3.md`). A per-cohort width
would be a Mondrian cut, which the action list rules out, so the longer
horizons are withheld instead until a re-read at ~120 days of history.

Young items are a SERVING cohort only. They never enter the train universe
(archive-derived, pre-2026), and the served-panel readers that calibrate or
judge the established band (the feedback refit, the PID prereg) exclude them
through ``ESTABLISHED_ITEM_SQL``.

Pure SQL over a session, no `database` import: that module builds the
production engine at import time.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy import text

#: History a discovered release needs before it is served, counted from its
#: ``release_date`` (the first day the archive holds a price for it). The
#: measurement that admitted the cohort was taken at exactly 60 days.
YOUNG_MIN_HISTORY_DAYS = 60

#: The only horizons a young item is served at.
YOUNG_SERVED_HORIZONS = frozenset({3})

#: Predicate for "an established item" on an ``items`` alias. Panel readers that
#: fit or judge the established band join on this so a young row cannot move them.
ESTABLISHED_ITEM_SQL = "{alias}.is_backfilled = 1"


def established_sql(alias: str = "i") -> str:
    return ESTABLISHED_ITEM_SQL.format(alias=alias)


def young_cutoff(as_of: date | datetime) -> datetime:
    """Served on ``as_of`` iff ``release_date < young_cutoff(as_of)``.

    An exclusive bound on the day AFTER the last eligible release day, so the
    gate is inclusive at exactly 60 days and a time-of-day on ``release_date``
    cannot move it. Exclusive rather than ``<=`` midnight because SQLite
    compares these as strings, where ``'... 00:00:00.000000' > '... 00:00:00'``."""
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    d = as_of - timedelta(days=YOUNG_MIN_HISTORY_DAYS - 1)
    return datetime(d.year, d.month, d.day)


def served_items(session, as_of: date | datetime) -> dict[str, tuple[int, bool]]:
    """``{slug: (items.id, is_young)}`` for every item served on ``as_of``."""
    rows = session.execute(
        text(
            "SELECT id, item_id, is_backfilled FROM items "
            "WHERE is_backfilled = 1 "
            "OR (release_date IS NOT NULL AND release_date < :cutoff)"
        ),
        {"cutoff": young_cutoff(as_of)},
    ).fetchall()
    return {r.item_id: (r.id, r.is_backfilled != 1) for r in rows}


def young_slugs(served: dict[str, tuple[int, bool]]) -> set[str]:
    return {slug for slug, (_, young) in served.items() if young}


def slug_to_id(served: dict[str, tuple[int, bool]], *, established_only: bool = False) -> dict[str, int]:
    return {slug: iid for slug, (iid, young) in served.items() if not (established_only and young)}
