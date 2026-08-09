#!/usr/bin/env python3
"""One-shot import of a historical price source into a STAGING archive.

Usage (from ``backend/``)::

    venv/bin/python scripts/import_price_history_source.py \
        --source cs2_prices_tracker \
        --start 2025-02-17 --end 2026-03-31 \
        --cache-dir runtime/price_history_cache \
        --out-dir ../archive-staging

Writes ``{out-dir}/price-archive/prices-YYYY-MM.parquet``. It never touches
``price-archive/`` and never pushes to ``cs2-oracle-data`` — promotion is a
separate, deliberate step.

The fetch is resumable: a cached day that parses is not re-requested, so an
interrupted run resumes without re-downloading ~2.3 GB.

`backend/.env` points at production Supabase, but nothing here imports
``database``, so this script holds no DB connection.
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.price_history_import import (  # noqa: E402
    MIN_DISTINCT_DAYS,
    StalledSourceError,
    apply_gap_gate,
    detect_stalled_days,
    to_archive_frame,
    write_archive_frame,
)
from collectors.price_history_sources import ADAPTERS  # noqa: E402

logger = logging.getLogger(__name__)

REQUEST_DELAY = 0.2
REQUEST_TIMEOUT = 60


def daterange(start: date, end: date) -> list[date]:
    """Every day from *start* to *end*, both inclusive."""
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def cached_path(cache_dir: Path, source_name: str, day: date) -> Path:
    """Where one fetched day file lives, namespaced by source."""
    return Path(cache_dir) / source_name / f"{day.isoformat()}.json"


def load_cached_day(path: Path) -> dict | None:
    """The parsed payload, or None if absent, empty or unparseable.

    Returning None rather than raising is what makes the run resumable: a
    truncated file from an interrupted download is simply re-fetched.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def fetch_days(adapter, source_name, days, cache_dir, session):
    """Populate the cache for *days*. Returns {day: md5} for what is on disk."""
    digests: dict[date, str] = {}
    fetched = skipped = missing = 0

    for day in days:
        path = cached_path(cache_dir, source_name, day)
        path.parent.mkdir(parents=True, exist_ok=True)

        if load_cached_day(path) is None:
            response = session.get(adapter.day_url(day), timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                missing += 1
                logger.warning(f"  {day}: absent upstream (404)")
                continue
            response.raise_for_status()
            path.write_bytes(response.content)
            fetched += 1
            time.sleep(REQUEST_DELAY)
        else:
            skipped += 1

        digests[day] = hashlib.md5(path.read_bytes()).hexdigest()

    logger.info(
        f"  fetched {fetched:,}, cached {skipped:,}, absent upstream {missing:,}"
    )
    return digests


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--start", required=True, help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="inclusive, YYYY-MM-DD")
    ap.add_argument("--cache-dir", default="runtime/price_history_cache")
    ap.add_argument("--out-dir", default="../archive-staging")
    ap.add_argument("--fetch-only", action="store_true",
                    help="populate the cache and stop, writing nothing")
    args = ap.parse_args(argv)

    adapter = ADAPTERS[args.source]
    days = daterange(
        date.fromisoformat(args.start), date.fromisoformat(args.end)
    )
    logger.info(
        f"{args.source}: {len(days):,} days, "
        f"{days[0].isoformat()} -> {days[-1].isoformat()}"
    )

    session = requests.Session()
    digests = fetch_days(adapter, args.source, days, Path(args.cache_dir), session)

    stalled = detect_stalled_days(digests)
    if stalled:
        raise StalledSourceError(stalled)

    if args.fetch_only:
        logger.info("--fetch-only: nothing written")
        return 0

    records: list[tuple[str, date, float]] = []
    for day in sorted(digests):
        payload = load_cached_day(cached_path(Path(args.cache_dir), args.source, day))
        records.extend(adapter.parse_day(payload, day))
    logger.info(f"parsed {len(records):,} raw records")

    kept, report = apply_gap_gate(records)
    logger.info(
        f"quality gate: kept {report.kept_items:,} items / "
        f"{report.kept_rows:,} rows; rejected "
        f"{report.rejected_gap_items:,} on gaps (worst "
        f"{report.worst_gap_days or 0}d) and "
        f"{report.rejected_sparse_items:,} on the {MIN_DISTINCT_DAYS}-day floor, "
        f"{report.rejected_rows:,} rows total"
    )

    frame = to_archive_frame(kept, adapter.SOURCE, datetime.now())
    out_dir = Path(args.out_dir) / "price-archive"
    written = write_archive_frame(frame, out_dir)
    logger.info(f"wrote {written:,} rows as source={adapter.SOURCE} to {out_dir}")

    pre_2026 = sum(1 for r in kept if r[1] < date(2026, 1, 1))
    gate_items = len({r[0] for r in kept if r[1] < date(2026, 1, 1)})
    logger.info(
        f"pre-2026 rows {pre_2026:,} across {gate_items:,} items "
        f"— these are what is_backfilled tests for"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
