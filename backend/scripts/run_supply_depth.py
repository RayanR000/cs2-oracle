#!/usr/bin/env python3
"""Daily supply-depth collection — entry point for CI and manual runs.

    # full daily run (scalar feeds + the lis-skins ladder)
    venv/bin/python scripts/run_supply_depth.py

    # scalar feeds only (~2 s); skips the 173 MB / ~7 min lis-skins export
    venv/bin/python scripts/run_supply_depth.py --no-ladder

    # verify connectivity and payload shapes without writing
    venv/bin/python scripts/run_supply_depth.py --dry-run

This writes **only** to `price-archive/supply-YYYY-MM.parquet`. It never touches
Postgres, which is deliberate: `backend/.env` points at production Supabase and
the engine binds at import, so a collector that imports `database` inherits a
prod connection it has no need for. Nothing here imports it.

Exit codes: 0 on success, 1 if every feed failed. A run where *some* feeds failed
still exits 0 but reports the failures in its JSON summary — see
`collectors/supply_depth.collect` for why that split is drawn there.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.supply_depth import SupplyFeedError, collect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_supply_depth")

DEFAULT_ARCHIVE = Path(__file__).parent.parent.parent / "price-archive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Where supply-YYYY-MM.parquet is written (default: ../price-archive)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Snapshot day as YYYY-MM-DD. Defaults to the aggregator's resolved "
        "snapshot date so supply rows carry the same day label as the price "
        "rows they will be joined to. These feeds are live snapshots with no "
        "as-of parameter, so this only labels the observation -- it cannot "
        "fetch a past day.",
    )
    parser.add_argument(
        "--no-ladder",
        action="store_true",
        help="Skip the lis-skins export. Drops ask-ladder shape and listing age, "
        "which are the only quantities here not already refuted.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse, write nothing.")
    args = parser.parse_args()

    snapshot_day = date.fromisoformat(args.date) if args.date else None

    try:
        summary = collect(
            archive_dir=args.archive_dir,
            snapshot_day=snapshot_day,
            include_ladder=not args.no_ladder,
            dry_run=args.dry_run,
        )
    except SupplyFeedError as exc:
        logger.error("Supply depth collection failed: %s", exc)
        print(json.dumps({"status": "failed", "supply_rows": 0, "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
