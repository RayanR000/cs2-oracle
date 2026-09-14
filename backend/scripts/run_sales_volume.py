#!/usr/bin/env python3
"""Daily completed-sale volume collection — entry point for CI and manual runs.

    # full daily run (one call, ~2 s)
    venv/bin/python scripts/run_sales_volume.py

    # verify connectivity and payload shape without writing
    venv/bin/python scripts/run_sales_volume.py --dry-run

This writes **only** to `price-archive/volume-YYYY-MM.parquet`. It never touches
Postgres, and nothing here imports `database` — `backend/.env` points at
production Supabase and the engine binds at import, so a collector that imports
it inherits a prod connection it has no need for.

It does **not** write `prices-*.parquet`'s `volume` column. That column holds
three different quantities already (a pre-2026 Steam sale count, an HF listing
count for 2026-03-22..04-15, fabricated zeros after) and is deliberately left
dead. See `collectors/sales_volume.py`.

Exit codes: 0 on success, 1 if the feed failed. A 403 from a Cloudflare-owned
egress IP is a WAF block on the *caller*, not an outage — the error message says
so, because this has been misdiagnosed as "Skinport is dead" twice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.sales_volume import SalesVolumeError, collect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_sales_volume")

DEFAULT_ARCHIVE = Path(__file__).parent.parent.parent / "price-archive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Where volume-YYYY-MM.parquet is written (default: ../price-archive)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Snapshot day as YYYY-MM-DD. Defaults to the aggregator's resolved "
        "snapshot date so these rows carry the same day label as the price "
        "rows they join to. Skinport's windows are trailing and have no "
        "as-of parameter, so this only labels the observation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse, write nothing.")
    args = parser.parse_args()

    snapshot_day = date.fromisoformat(args.date) if args.date else None

    try:
        summary = collect(
            archive_dir=args.archive_dir,
            snapshot_day=snapshot_day,
            dry_run=args.dry_run,
        )
    except SalesVolumeError as exc:
        logger.error("Sales volume collection failed: %s", exc)
        print(json.dumps({"status": "failed", "volume_rows": 0, "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
