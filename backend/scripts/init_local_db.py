#!/usr/bin/env python3
"""
Initialize local SQLite database from parquet archive + events file.

Populates:
- items table (from parquet slugs)
- events table (from cs2_events.json)
- All other tables (created empty)

Run from backend/ directory:
    python scripts/init_local_db.py
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Event, Item, SessionLocal, init_db
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("init_local_db")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"
EVENTS_FILE = Path(__file__).parent.parent / "data" / "cs2_events.json"


def assert_local_db():
    """Abort unless the bound engine is a local SQLite file.

    This script inserts items and rewrites is_backfilled, so pointing it at a
    remote database is destructive. It used to "force SQLite" by assigning
    os.environ["DATABASE_URL"] inside main(), which never worked: database.py
    builds the engine at import time from settings.database_url, so the
    assignment always came too late. With backend/.env pointing DATABASE_URL at
    production Supabase, running this from backend/ wrote straight to prod.

    Check the engine that actually got built, not the environment.
    """
    from database import engine

    if engine.url.get_backend_name() != "sqlite":
        logger.error("=" * 60)
        logger.error("REFUSING TO RUN: bound database is not local SQLite")
        logger.error(f"  backend  : {engine.url.get_backend_name()}")
        logger.error(f"  host     : {engine.url.host or '(none)'}")
        logger.error("")
        logger.error("This script rewrites the items table and is local-only.")
        logger.error("To run it, point DATABASE_URL at a SQLite file, e.g.:")
        logger.error("  DATABASE_URL=sqlite:///cs2_market.db \\")
        logger.error("    python3 scripts/init_local_db.py")
        logger.error("=" * 60)
        sys.exit(1)

    logger.info(f"Target database: {engine.url.database} (sqlite)")


def populate_items(db):
    logger.info("Reading items from parquet archive...")
    import duckdb

    # Through prices_relation, never a raw glob (backend/AGENTS.md invariant 1).
    # day arrives as DATE and a missing source column reads as NULL, which is
    # exactly the "pre-2026 series" predicate below.
    from db.archive import prices_relation

    con = duckdb.connect()
    try:
        rel = prices_relation(con, ARCHIVE_DIR, columns=["item_slug", "day", "source"])
        rows = con.sql(f"""
            SELECT DISTINCT item_slug
            FROM {rel}
            ORDER BY item_slug
        """).fetchall()
        # is_backfilled marks items carrying the CSMarketAPI historical series
        # (see database.py), NOT merely "present in the archive". Flagging every
        # archive item made the flag meaningless — it read 8,691/8,691 = 100% —
        # and went stale as soon as the live aggregator added items, so the
        # backfilled_only filter in training/predict both admitted the
        # low-history live cohort and excluded most of the grown archive.
        #
        # The backfilled cohort is exactly the items with pre-2026 rows: that
        # series predates the source column, so those are also the only rows
        # with source IS NULL (both predicates select the same 5,542 items).
        # Cross-checked against the documented priority queue — 93% of them sit
        # in its sell_listings-DESC head, which is the order the backfill ran.
        backfilled = {
            r[0]
            for r in con.sql(f"""
                SELECT DISTINCT item_slug
                FROM {rel}
                WHERE day < '2026-01-01'
            """).fetchall()
        }
        # is_trainable narrows is_backfilled to exclude iflow-only history: a
        # pre-2026 row whose source is 'buff_iflow' is served but must not be
        # trained on (that backfill source is out of scope for the model).
        # `source IS DISTINCT FROM 'buff_iflow'` keeps the NULL-source legacy
        # rows (source predates this column) while excluding buff_iflow rows.
        trainable = {
            r[0]
            for r in con.sql(f"""
                SELECT DISTINCT item_slug
                FROM {rel}
                WHERE day < '2026-01-01' AND source IS DISTINCT FROM 'buff_iflow'
            """).fetchall()
        }
    finally:
        con.close()

    total = len(rows)
    logger.info(
        f"Found {total:,} unique items in parquet "
        f"({len(backfilled):,} carrying the historical backfill, "
        f"{len(trainable):,} trainable)"
    )

    existing = {r[0] for r in db.query(Item.item_id).all()}
    to_insert = []
    for (slug,) in rows:
        if slug not in existing:
            to_insert.append(
                Item(
                    item_id=slug,
                    name=slug,
                    type="skin",
                    is_backfilled=1 if slug in backfilled else 0,
                    is_trainable=1 if slug in trainable else 0,
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                    updated_at=datetime.now(UTC).replace(tzinfo=None),
                )
            )

    if to_insert:
        db.add_all(to_insert)
        db.commit()
        n_bf = sum(1 for i in to_insert if i.is_backfilled)
        logger.info(f"Inserted {len(to_insert)} new items ({n_bf} backfilled, {len(to_insert) - n_bf} not)")
    else:
        logger.info("No new items to insert")

    # Re-derive on every run so rows written by the older version of this
    # script (which flagged everything) get corrected, and so the flags keep
    # tracking the archive as the historical backfill is extended. Batched per
    # flag and direction — the per-row UPDATE above this replaced issued one
    # statement per item.
    now = datetime.now(UTC).replace(tzinfo=None)
    rows = db.query(Item.item_id, Item.is_backfilled, Item.is_trainable).all()
    bf_on = [iid for iid, bf, _tr in rows if not (bf or 0) and iid in backfilled]
    bf_off = [iid for iid, bf, _tr in rows if (bf or 0) and iid not in backfilled]
    tr_on = [iid for iid, _bf, tr in rows if not (tr or 0) and iid in trainable]
    tr_off = [iid for iid, _bf, tr in rows if (tr or 0) and iid not in trainable]
    changed = len(bf_on) + len(bf_off) + len(tr_on) + len(tr_off)
    if bf_on:
        db.query(Item).filter(Item.item_id.in_(bf_on)).update(
            {"is_backfilled": 1, "updated_at": now}, synchronize_session=False
        )
    if bf_off:
        db.query(Item).filter(Item.item_id.in_(bf_off)).update(
            {"is_backfilled": 0, "updated_at": now}, synchronize_session=False
        )
    if tr_on:
        db.query(Item).filter(Item.item_id.in_(tr_on)).update(
            {"is_trainable": 1, "updated_at": now}, synchronize_session=False
        )
    if tr_off:
        db.query(Item).filter(Item.item_id.in_(tr_off)).update(
            {"is_trainable": 0, "updated_at": now}, synchronize_session=False
        )
    if changed:
        db.commit()
        logger.info(f"Corrected is_backfilled/is_trainable on {changed:,} flag writes")

    total_in_db = db.query(Item).count()
    n_flagged = db.query(Item).filter(Item.is_backfilled == 1).count()
    n_trainable = db.query(Item).filter(Item.is_trainable == 1).count()
    logger.info(f"Total items in DB: {total_in_db:,} ({n_flagged:,} is_backfilled=1, {n_trainable:,} is_trainable=1)")


def populate_events(db):
    if not EVENTS_FILE.exists():
        logger.warning(f"Events file not found at {EVENTS_FILE}")
        return

    with open(EVENTS_FILE) as f:
        data = json.load(f)

    raw_events = data.get("events", [])
    logger.info(f"Found {len(raw_events)} events in {EVENTS_FILE}")

    existing = {(r.type, r.timestamp.date()) for r in db.query(Event).all()}
    to_insert = []
    for ev in raw_events:
        ev_type = ev.get("type", "update")
        ev_date = ev.get("date", "")
        try:
            parsed_date = datetime.strptime(ev_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.warning(f"Skipping event with bad date: {ev_date}")
            continue

        if (ev_type, parsed_date) not in existing:
            to_insert.append(
                Event(
                    type=ev_type,
                    timestamp=datetime.combine(parsed_date, datetime.min.time()),
                    description=ev.get("description", ""),
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                )
            )

    if to_insert:
        db.add_all(to_insert)
        db.commit()
        logger.info(f"Inserted {len(to_insert)} new events")

        from db.parquet import append_table

        parquet_rows = []
        for ev in to_insert:
            parquet_rows.append(
                {
                    "id": ev.id,
                    "type": ev.type,
                    "timestamp": ev.timestamp,
                    "description": ev.description,
                    "created_at": ev.created_at,
                }
            )
        append_table("events", parquet_rows, ["id"])
    else:
        logger.info("No new events to insert")

    total_in_db = db.query(Event).count()
    logger.info(f"Total events in DB: {total_in_db}")


def main():
    logger.info("=" * 60)
    logger.info("INITIALIZING LOCAL DATABASE")
    logger.info("=" * 60)

    assert_local_db()

    init_db()
    db = SessionLocal()
    try:
        populate_items(db)
        populate_events(db)

        logger.info("\nFinal table counts:")
        for table in [
            "items",
            "events",
            "item_forecasts",
            "prediction_accuracy",
            "forecast_outcomes",
            "accuracy_alerts",
            "price_history",
        ]:
            try:
                count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                logger.info(f"  {table}: {count:,}")
            except Exception:
                logger.info(f"  {table}: <empty or does not exist>")
    finally:
        db.close()

    logger.info("\n✅ Local database initialized successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
