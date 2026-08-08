#!/usr/bin/env python3
"""Upsert Valve announcements from `event-news.parquet` into the `events` table.

## Why this exists rather than "read the parquet directly"

`event_correlation_analysis.py` measures a price window around **one event** and
writes `event_impacts` rows whose `event_id` is a real foreign key to
`events.id`. So its events have to be DB rows; pointing the analyser at a
Parquet file would either break the FK or require synthesising ids that nothing
else agrees on.

`event-calendar.parquet` cannot feed it either -- that table is deliberately
*date-level*: one row per day, counts and `days_since_*`, no identity, no title.
It is a feature table for a date-level model. `event-news.parquet` is the
per-event companion this needs, emitted by the same ingest.

## The gap this closes

`events` is loaded from `data/cs2_events.json`, whose newest entry is
**2026-05-10**. From roughly 2026-08-08 the analyser's default 90-day window
contains nothing, and it returns `no_events_in_window` every Sunday -- a real
status, correctly not a failure, and also a permanently empty report.

## What is synced, and what is not

**Valve announcements only** (`feed_type == 1`). Syndicated press is kept out on
the same reasoning that keeps it in its own column upstream: a games-press
article is a reaction to the market as often as a cause of it, and mixing a
reactive series into a causal analysis is how you manufacture a correlation.

**No classification.** Every row lands as `type = "update"`. Both candidate
case-release classifiers were measured on 2026-08-06 and neither validated (33%
recall from announcement text; 14/14 disagreement against ByMykel
`first_sale_date`, median 38 days). A wrong `type` would silently partition
`event_patterns` by a label that means nothing, so the honest move is one type
until something validates.

**Idempotent on `gid`.** The gid is carried in the description suffix because
`events` has no column for it and adding one is a migration this does not need;
matching on `(timestamp, description)` alone would duplicate a row every time
Steam edits a title. Re-running is safe and reports 0 inserted.

## Running it

    venv/bin/python scripts/sync_events_from_news.py --dry-run
    venv/bin/python scripts/sync_events_from_news.py --since 2026-01-01

**`backend/.env` points at production Supabase**, so running this from
`backend/` writes to prod. That is what CI does deliberately; be sure that is
what you want locally.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sync_events_from_news")

DEFAULT_EVENTS_PARQUET = (
    Path(__file__).parent.parent.parent / "price-archive" / "event-news.parquet"
)

# One type, deliberately -- see the module docstring.
EVENT_TYPE = "update"

# `events.description` is String(500). The gid suffix is the idempotency key and
# must survive truncation, so the title is cut to leave room for it.
DESCRIPTION_MAX = 500
GID_MARKER = " [steam_news:"


def format_description(title: str, gid: str) -> str:
    """Title with a gid suffix, truncated so the suffix always survives.

    The suffix is what makes a re-run a no-op. Truncating the *title* rather
    than the whole string keeps it exact even for the longest announcements
    Steam publishes.
    """
    suffix = f"{GID_MARKER}{gid}]"
    room = DESCRIPTION_MAX - len(suffix)
    clean = " ".join((title or "").split())
    if room <= 0:  # pragma: no cover - only if a gid alone exceeds the column
        return suffix[:DESCRIPTION_MAX]
    if len(clean) > room:
        clean = clean[: room - 1] + "…"
    return f"{clean}{suffix}"


def load_valve_events(parquet_path: Path, since: date | None) -> pd.DataFrame:
    """Read the per-event table and keep Valve's own posts."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} does not exist. Run scripts/ingest_steam_news.py "
            "first -- it writes this table beside event-calendar.parquet."
        )
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df

    df = df[df["is_valve"] == 1].copy()
    df["day"] = pd.to_datetime(df["day"]).dt.date
    if since is not None:
        df = df[df["day"] >= since]
    return df.sort_values("published_at").reset_index(drop=True)


def sync(
    parquet_path: Path,
    db,
    since: date | None = None,
    dry_run: bool = False,
) -> dict:
    """Insert any Valve announcement not already present, matched on gid.

    Takes an open session rather than opening one: `.env` here points at
    PRODUCTION Supabase and the engine binds at import, so a test that let this
    open its own session would be writing to prod.
    """
    from database import Event

    rows = load_valve_events(parquet_path, since)
    if rows.empty:
        logger.warning(
            "No Valve announcements in %s%s -- nothing to sync.",
            parquet_path, f" on or after {since}" if since else "",
        )
        return {"status": "success", "candidates": 0, "events_inserted": 0,
                "already_present": 0, "dry_run": dry_run}

    existing = {
        desc.rsplit(GID_MARKER, 1)[1].rstrip("]")
        for (desc,) in db.query(Event.description).all()
        if desc and GID_MARKER in desc
    }

    to_add = []
    for row in rows.itertuples(index=False):
        if row.gid in existing:
            continue
        to_add.append(Event(
            type=EVENT_TYPE,
            timestamp=pd.Timestamp(row.published_at).tz_localize(None)
            if pd.Timestamp(row.published_at).tzinfo
            else pd.Timestamp(row.published_at),
            description=format_description(row.title, row.gid),
        ))

    if to_add and not dry_run:
        db.add_all(to_add)
        db.commit()

    logger.info(
        "Valve announcements: %d candidates, %d already present, %d %s",
        len(rows), len(rows) - len(to_add), len(to_add),
        "would be inserted (dry run)" if dry_run else "inserted",
    )
    return {
        "status": "success",
        "candidates": int(len(rows)),
        "events_inserted": 0 if dry_run else len(to_add),
        "would_insert": len(to_add) if dry_run else 0,
        "already_present": int(len(rows) - len(to_add)),
        "newest_day": str(rows["day"].max()),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-parquet", type=Path, default=DEFAULT_EVENTS_PARQUET)
    parser.add_argument(
        "--since", default=None,
        help="Only sync announcements on or after this YYYY-MM-DD. Defaults to "
             "the analyser's own lookback plus a margin, so a weekly run does "
             "not re-scan 13 years of feed on every pass.",
    )
    parser.add_argument("--all", action="store_true",
                        help="Ignore --since and sync the full history.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        since = None
    elif args.since:
        since = date.fromisoformat(args.since)
    else:
        since = date.today() - timedelta(days=400)

    from database import SessionLocal

    db = SessionLocal()
    try:
        summary = sync(args.events_parquet, db, since=since, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(json.dumps({"status": "failed", "events_inserted": 0,
                          "error": str(exc)}, indent=2))
        return 1
    finally:
        db.close()

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
