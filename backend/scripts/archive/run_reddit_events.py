#!/usr/bin/env python3
"""Reddit event-detector collection — entry point for CI/manual runs.

    venv/bin/python scripts/archive/run_reddit_events.py --dry-run
    venv/bin/python scripts/archive/run_reddit_events.py

Writes ONLY to price-archive/reddit-events-YYYY-MM.parquet. Without Reddit
credentials the run reports status=skipped (not a failure, not a success with
zero rows). Research panel only: nothing trains on it until a preregistered,
date-purged, placebo-controlled A/B clears. Never touches Postgres.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.reddit_events import collect

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_reddit_events")

DEFAULT_ARCHIVE = Path(__file__).parent.parent.parent / "price-archive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse, write nothing.")
    args = parser.parse_args()
    snapshot_day = date.fromisoformat(args.date) if args.date else None
    summary = collect(archive_dir=args.archive_dir, snapshot_day=snapshot_day, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
