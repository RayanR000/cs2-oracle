#!/usr/bin/env python3
"""
Copy named forecast dates from Postgres into the monthly ops partitions, ids included.

Phase 1.2 of docs/specs/2026-10-11-postgres-to-parquet-history-design.md. Two uses:

* FILL: the 22,507 forecasts dated 2026-10-11 reached the partitions from the single-file
  mirror, which has no id. Postgres has them; this writes Postgres's rows over them by
  (item_id, forecast_date, horizon_days). Columns Postgres lacks (model_config, item_slug for
  candidate rows, pg_deleted) keep their stored values.
* REPAIR: a night whose partition write failed (`dual_write` logged it, parity failed).
* CATALOG: `--tables items` copies the whole `items` table (no dates). The Aggregator runs it
  nightly after the data checkout, because new-item discovery runs before that checkout exists.

Writes only rows Postgres holds; never deletes. Dry run unless --apply. Closed months need
--allow-closed. Run by dispatching ops-sync.yml: only CI writes the data repo.

Usage:
    python scripts/ops_sync_from_postgres.py --archive-dir ../archive/price-archive --dates 2026-10-11
    python scripts/ops_sync_from_postgres.py --archive-dir ... --dates 2026-10-11 --apply
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import ops_partitions, ops_readback

logger = logging.getLogger("ops_sync_from_postgres")

DATED = ("item_forecasts", "forecast_outcomes", "forecast_candidates", "forecast_candidate_outcomes")
TABLES = (*DATED, "items")

_READ = {
    "item_forecasts": lambda db, ds: ops_readback.forecasts(db, ds),
    "forecast_outcomes": lambda db, ds: ops_readback.outcomes(db, dates=ds),
    "forecast_candidates": lambda db, ds: ops_readback.candidates(db, ds),
    "forecast_candidate_outcomes": lambda db, ds: ops_readback.candidate_outcomes(db, dates=ds),
    "items": lambda db, ds: ops_readback.items(db),  # the whole catalog; it has no dates
}


def sync(db, dates, *, tables, apply, allow_closed=False, released=None, root=None, today=None) -> dict[str, int]:
    """*released*: months (YYYY-MM) that already have an ops-YYYY-MM release. Never rewritten, even
    with *allow_closed*: a changed file fails ops_releases.publish every night after."""
    hit = sorted({f"{d:%Y-%m}" for d in dates} & set(released or ()))
    if allow_closed and hit:
        raise ops_partitions.PartitionError(f"refusing released month(s) {hit}: their files are frozen in a release")
    plan = {}
    for t in tables:
        frame = _READ[t](db, dates)
        plan[t] = len(frame)
        logger.info("%s: %d rows in Postgres for %s", t, len(frame), ", ".join(map(str, dates)))
        if apply and not frame.empty:
            out = ops_partitions.upsert(t, frame, allow_closed=allow_closed, root=root, today=today)
            logger.info("%s: wrote %s", t, out)
    if not apply:
        logger.info("dry run; pass --apply to write")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--archive-dir", type=Path, required=True, help="The data checkout's price-archive/ directory.")
    ap.add_argument("--dates", nargs="*", type=date.fromisoformat, default=[])
    ap.add_argument("--tables", nargs="+", choices=TABLES, default=list(DATED))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--allow-closed", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if any(t in DATED for t in args.tables) and not args.dates:
        ap.error("--dates is required for " + ", ".join(t for t in args.tables if t in DATED))

    from database import SessionLocal

    released = None
    if args.allow_closed:
        from scripts.ops_releases import TAG_PREFIX, release_tags

        released = {t.removeprefix(TAG_PREFIX) for t in release_tags()}

    db = SessionLocal()
    try:
        sync(
            db,
            args.dates,
            tables=args.tables,
            apply=args.apply,
            allow_closed=args.allow_closed,
            released=released,
            root=args.archive_dir / "ops",
        )
    except ops_partitions.PartitionError as e:
        logger.error("%s", e)
        return 1
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
