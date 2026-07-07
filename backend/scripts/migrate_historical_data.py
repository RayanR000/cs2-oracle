#!/usr/bin/env python3
"""
Historical Data Migration: Clean Supabase + Import local data.
Implements DB_CLEANUP_AND_MIGRATION_PLAN.md

Phases:
  1a. Delete stale price_history sources (kaggle_csgo, historical_fallback:*, csgotrader)
  1b. Deduplicate items table (slug-format entries without classid)
  1c. Enrich items with classid and type from local DB
  2a. Import MARKETCSGO daily data (2022-2026, ~2.1M rows)
  2b. Import STEAMCOMMUNITY pre-2022 weekly downsampled (~557K rows)

Usage:
    python scripts/migrate_historical_data.py            # run all phases
    python scripts/migrate_historical_data.py --phase 1  # run only Phase 1
    python scripts/migrate_historical_data.py --phase 2  # run only Phase 2
    python scripts/migrate_historical_data.py --dry-run  # count/show without modifying
"""

import argparse
import csv
import io
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database import SessionLocal, engine

LOCAL_DB = Path(__file__).parent.parent / "runtime" / "csmarketapi.db"
CLASSID_MAP = Path(__file__).parent.parent / "data" / "classid_map.json"
BATCH_SIZE = 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("migration")

STALE_SOURCES = [
    "kaggle_csgo",
    "historical_fallback:kaggle_csgo",
    "historical_fallback:csgotrader",
    "historical_fallback:aggregator_sync",
    "csgotrader",
]


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def map_item_type(category: str | None, item_type: str | None, sticker_type: str | None) -> str:
    st = sticker_type or ""
    ct = category or ""
    it = item_type or ""

    if st in ("Sticker", "Player Autograph", "Tournament"):
        return "sticker"
    if st == "Graffiti":
        return "graffiti"
    if it == "Sticker":
        return "sticker"
    if it == "Graffiti":
        return "graffiti"
    if it == "Music Kit":
        return "musickit"
    if ct == "Sticker":
        return "sticker"
    if ct == "Graffiti":
        return "graffiti"
    if ct in ("Case", "Key"):
        return "case"
    if ct == "Gloves":
        return "gloves"
    if ct == "Knife":
        return "knife"
    if ct == "MusicKit":
        return "musickit"
    if ct == "Collectible":
        return "collectible"
    if ct == "Agent":
        return "agent"
    if ct == "Patch":
        return "patch"
    if ct == "Tool":
        return "tool"
    return "skin"


# ── Phase 1: Clean Supabase ──────────────────────────────────────────────


def phase_1a_delete_stale_sources(db, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("Phase 1a: Delete stale price_history sources")
    total = 0
    for src in STALE_SOURCES:
        count = db.execute(
            text("SELECT COUNT(*) FROM price_history WHERE source = :src"),
            {"src": src},
        ).scalar() or 0
        status = "WOULD DELETE" if dry_run else "DELETING"
        logger.info(f"  {src:<40} {status} {count:>10,} rows")
        total += count
        if count and not dry_run:
            db.execute(text("DELETE FROM price_history WHERE source = :src"), {"src": src})
    if not dry_run:
        db.commit()
    logger.info(f"  Total: {total:,} rows affected")
    logger.info("  ✅ Phase 1a complete")
    return total


def phase_1b_deduplicate_items(db, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("Phase 1b: Deduplicate items table")

    dupe_names = db.execute(
        text("""
            SELECT name FROM items
            WHERE item_id ~ '^[a-z0-9-]+$' AND classid IS NULL
            GROUP BY name HAVING COUNT(*) > 1
        """),
    ).fetchall()
    dupe_names = [r[0] for r in dupe_names]
    logger.info(f"  Found {len(dupe_names):,} duplicate names")

    total_deleted = 0
    total_repointed = 0

    for name in dupe_names:
        items = db.execute(
            text("""
                SELECT id, item_id, classid FROM items
                WHERE name = :name
                ORDER BY classid IS NULL, id
            """),
            {"name": name},
        ).fetchall()

        # Keeper = first row (has classid if available, or first by id)
        keeper_id = items[0][0]
        deleter_ids = [r[0] for r in items[1:]]

        if dry_run:
            total_deleted += len(deleter_ids)
            continue

        # Re-point price_history FK references from deleters to keeper
        for did in deleter_ids:
            ref_count = db.execute(
                text("SELECT COUNT(*) FROM price_history WHERE item_id = :id"),
                {"id": did},
            ).scalar() or 0

            if ref_count:
                # Remove rows that would cause unique violations after repointing
                db.execute(
                    text("""
                        DELETE FROM price_history ph
                        WHERE ph.item_id = :deleter
                        AND EXISTS (
                            SELECT 1 FROM price_history ph2
                            WHERE ph2.item_id = :keeper
                            AND ph2.timestamp = ph.timestamp
                            AND ph2.source = ph.source
                        )
                    """),
                    {"deleter": did, "keeper": keeper_id},
                )
                # Re-point remaining
                db.execute(
                    text("UPDATE price_history SET item_id = :keeper WHERE item_id = :deleter"),
                    {"keeper": keeper_id, "deleter": did},
                )
                total_repointed += ref_count

        # Delete the deleter items
        for did in deleter_ids:
            db.execute(text("DELETE FROM items WHERE id = :id"), {"id": did})
            total_deleted += 1

        db.commit()

    logger.info(f"  {'WOULD DELETE' if dry_run else 'Deleted'} {total_deleted:,} duplicate items")
    if total_repointed:
        logger.info(f"  Re-pointed {total_repointed:,} price_history FK references")
    logger.info("  ✅ Phase 1b complete")
    return total_deleted


def phase_1c_enrich_items(db, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("Phase 1c: Enrich items with classid and type from local DB")

    if not LOCAL_DB.exists():
        logger.warning(f"  Local DB not found at {LOCAL_DB}")
        logger.info("  ✅ Phase 1c complete (no-op)")
        return 0

    local_conn = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True)

    local_items: dict[str, dict] = {}
    for row in local_conn.execute(
        "SELECT market_hash_name, classid, category, type, sticker_type FROM items"
    ).fetchall():
        local_items[row[0]] = {
            "classid": row[1],
            "category": row[2],
            "type": row[3],
            "sticker_type": row[4],
        }
    local_conn.close()
    logger.info(f"  Loaded {len(local_items):,} items from local DB")

    classid_map: dict = {}
    if CLASSID_MAP.exists():
        classid_map = json.loads(CLASSID_MAP.read_text())
        logger.info(f"  Loaded {len(classid_map):,} entries from classid_map.json")

    all_items = db.execute(text("SELECT id, name, classid, type FROM items ORDER BY id")).fetchall()
    logger.info(f"  Processing {len(all_items):,} Supabase items...")

    updated = 0
    batch_updates: list[dict] = []

    for item_row in all_items:
        pk_id, name, current_classid, current_type = item_row
        new_classid = current_classid
        new_type = current_type

        local_data = local_items.get(name)
        if local_data:
            if not new_classid and local_data["classid"]:
                new_classid = str(local_data["classid"])
            if not current_type or current_type in ("", None) or current_type == "skin":
                mapped = map_item_type(
                    local_data["category"], local_data["type"], local_data["sticker_type"]
                )
                if mapped != current_type:
                    new_type = mapped
        elif name in classid_map:
            cm = classid_map[name]
            if not new_classid and cm.get("classid"):
                new_classid = cm["classid"]
            if not current_type and cm.get("type"):
                new_type = cm["type"]

        if new_classid != current_classid or new_type != current_type:
            batch_updates.append({"classid": new_classid, "type": new_type, "id": pk_id})

        if len(batch_updates) >= 500 and not dry_run:
            db.execute(
                text("UPDATE items SET classid = :classid, type = :type WHERE id = :id"),
                batch_updates,
            )
            db.commit()
            updated += len(batch_updates)
            batch_updates = []

    if batch_updates and not dry_run:
        db.execute(
            text("UPDATE items SET classid = :classid, type = :type WHERE id = :id"),
            batch_updates,
        )
        db.commit()
        updated += len(batch_updates)

    dry_total = len(batch_updates) + (updated if not dry_run else 0)
    logger.info(f"  {'WOULD UPDATE' if dry_run else 'Updated'} {dry_total:,} items")

    null_count = db.execute(
        text("SELECT COUNT(*) FROM items WHERE classid IS NULL")
    ).scalar() or 0
    logger.info(f"  Items still NULL classid: {null_count:,}")

    logger.info("  ✅ Phase 1c complete")
    return updated


# ── Helpers for Phase 2 ─────────────────────────────────────────────────


def _ensure_unique_constraint(db):
    """Add the (item_id, timestamp, source) unique constraint if missing."""
    exists = db.execute(
        text("""
            SELECT 1 FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            WHERE rel.relname = 'price_history'
            AND con.conname = 'uq_price_history_item_timestamp_source'
        """),
    ).scalar()
    if not exists:
        logger.info("  Adding unique constraint uq_price_history_item_timestamp_source...")
        db.execute(
            text("""
                ALTER TABLE price_history
                ADD CONSTRAINT uq_price_history_item_timestamp_source
                UNIQUE (item_id, timestamp, source)
            """),
        )
        db.commit()


def _ensure_items_exist(db, names: list[str], local_conn: sqlite3.Connection) -> dict[str, int]:
    """Return {name: item_pk} dict, creating missing items on the fly."""
    name_map: dict[str, int] = {}
    for row in db.execute(
        text("SELECT id, name FROM items WHERE name = ANY(:names)"),
        {"names": names},
    ).fetchall():
        name_map[row[1]] = row[0]

    missing = [n for n in names if n not in name_map]
    if not missing:
        return name_map

    logger.info(f"  Creating {len(missing):,} missing items...")

    local_data: dict[str, tuple] = {}
    placeholders = ",".join("?" for _ in missing)
    for row in local_conn.execute(
        f"SELECT market_hash_name, classid, category, type, sticker_type "
        f"FROM items WHERE market_hash_name IN ({placeholders})",
        missing,
    ).fetchall():
        local_data[row[0]] = row

    new_items: list[dict] = []
    for mn in missing:
        lid = local_data.get(mn)
        item_type = map_item_type(lid[1] if lid else None, lid[2] if lid else None, lid[3] if lid else None) if lid else "skin"
        new_items.append({
            "item_id": slugify(mn),
            "name": mn,
            "type": item_type,
            "classid": str(lid[1]) if lid and lid[1] else None,
        })

    for i in range(0, len(new_items), 500):
        batch = new_items[i : i + 500]
        db.execute(
            text("""
                INSERT INTO items (item_id, name, type, classid, created_at, updated_at)
                VALUES (:item_id, :name, :type, :classid, NOW(), NOW())
                ON CONFLICT (item_id) DO NOTHING
            """),
            batch,
        )
        db.commit()

    for row in db.execute(
        text("SELECT id, name FROM items WHERE name = ANY(:names)"),
        {"names": names},
    ).fetchall():
        name_map[row[1]] = row[0]

    return name_map


# ── Phase 2: Import Historical Data ─────────────────────────────────────


def _stream_csv_to_copy(
    cursor, name_map: dict, columns: list[str],
    table: str, source_label: str, total_expected: int, chunk_rows: int = 50000,
):
    """Stream SQLite rows → COPY batches into PostgreSQL temp table."""
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        col_defs = ", ".join(f"{c} TEXT" for c in columns)
        tmp = f"_staging_{table}_{source_label.replace('-','_').replace(':','_')}"
        cur.execute(f"DROP TABLE IF EXISTS {tmp}")
        cur.execute(f"CREATE TEMP TABLE {tmp} ({col_defs})")

        total_inserted = 0
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        chunk_count = 0

        for row in cursor:
            market_hash_name, day, price_val, median_val, vol = row
            item_pk = name_map.get(market_hash_name)
            if item_pk is None:
                continue

            writer.writerow((
                str(item_pk),
                f"{day} 00:00:00",
                f"{price_val:.6f}" if price_val is not None else "0",
                str(int(vol)) if vol is not None else "",
                f"{median_val:.6f}" if median_val is not None else "",
                source_label,
            ))
            chunk_count += 1

            if chunk_count >= chunk_rows:
                buf.seek(0)
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE source = '{source_label}'")
                before_total = cur.fetchone()[0]
                cur.copy_from(buf, tmp, null="", columns=columns)
                cur.execute(f"""
                    INSERT INTO {table}
                        (item_id, timestamp, price, volume, median_price, source)
                    SELECT NULLIF(item_id, '')::integer,
                           NULLIF(timestamp, '')::timestamp,
                           NULLIF(price, '')::numeric,
                           NULLIF(volume, '')::integer,
                           NULLIF(median_price, '')::numeric,
                           source
                    FROM {tmp}
                    ON CONFLICT (item_id, timestamp, source) DO NOTHING
                """)
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE source = '{source_label}'")
                after_total = cur.fetchone()[0]
                cur.execute(f"TRUNCATE TABLE {tmp}")
                raw.commit()
                inserted = after_total - before_total
                total_inserted += inserted
                buf = io.StringIO()
                writer = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                chunk_count = 0
                logger.info(f"    Imported {total_inserted:,}/{total_expected:,} rows")

        if chunk_count > 0:
            buf.seek(0)
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE source = '{source_label}'")
            before_total = cur.fetchone()[0]
            cur.copy_from(buf, tmp, null="", columns=columns)
            cur.execute(f"""
                INSERT INTO {table}
                    (item_id, timestamp, price, volume, median_price, source)
                SELECT NULLIF(item_id, '')::integer,
                       NULLIF(timestamp, '')::timestamp,
                       NULLIF(price, '')::numeric,
                       NULLIF(volume, '')::integer,
                       NULLIF(median_price, '')::numeric,
                       source
                FROM {tmp}
                ON CONFLICT (item_id, timestamp, source) DO NOTHING
            """)
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE source = '{source_label}'")
            after_total = cur.fetchone()[0]
            inserted = after_total - before_total
            total_inserted += inserted

        cur.execute(f"DROP TABLE IF EXISTS {tmp}")
        raw.commit()
        logger.info(f"  ✅ {source_label}: {total_inserted:,}/{total_expected:,} rows imported")
        return total_inserted
    finally:
        raw.close()


def phase_2a_import_marketcsgo(db, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("Phase 2a: Import MARKETCSGO daily data")

    if not LOCAL_DB.exists():
        logger.warning(f"  Local DB not found — skipping")
        return 0

    _ensure_unique_constraint(db)

    local_conn = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True)

    total_rows = local_conn.execute(
        "SELECT COUNT(*) FROM sales_history WHERE market = 'MARKETCSGO'"
    ).fetchone()[0]
    logger.info(f"  Rows in local DB: {total_rows:,}")
    if total_rows == 0:
        local_conn.close()
        return 0

    names = [r[0] for r in local_conn.execute(
        "SELECT DISTINCT market_hash_name FROM sales_history WHERE market = 'MARKETCSGO'"
    ).fetchall()]
    logger.info(f"  Unique items: {len(names):,}")

    if dry_run:
        logger.info(f"  Would import {total_rows:,} rows for {len(names):,} items")
        local_conn.close()
        return total_rows

    existing = db.execute(
        text("SELECT COUNT(*) FROM price_history WHERE source = 'market_csgo'")
    ).scalar() or 0
    if existing > 0:
        logger.info(f"  Source 'market_csgo' already has {existing:,} rows — skipping")
        local_conn.close()
        return existing

    name_map = _ensure_items_exist(db, names, local_conn)
    logger.info(f"  Name map has {len(name_map):,} entries")

    columns = ["item_id", "timestamp", "price", "volume", "median_price", "source"]
    cursor = local_conn.execute(
        """SELECT market_hash_name, day, mean_price, median_price, volume
           FROM sales_history WHERE market = 'MARKETCSGO'
           ORDER BY market_hash_name, day"""
    )

    imported = _stream_csv_to_copy(
        cursor, name_map, columns, "price_history", "market_csgo", total_rows
    )

    local_conn.close()
    return imported


def phase_2b_import_steam_weekly(db, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("Phase 2b: Import STEAMCOMMUNITY pre-2022 weekly downsampled data")

    if not LOCAL_DB.exists():
        logger.warning(f"  Local DB not found — skipping")
        return 0

    _ensure_unique_constraint(db)

    local_conn = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True)

    daily_count = local_conn.execute(
        "SELECT COUNT(*) FROM sales_history WHERE market = 'STEAMCOMMUNITY' AND day < '2022-01-01'"
    ).fetchone()[0]
    logger.info(f"  Daily rows to downsample: {daily_count:,}")
    if daily_count == 0:
        local_conn.close()
        return 0

    names = [r[0] for r in local_conn.execute(
        "SELECT DISTINCT market_hash_name FROM sales_history WHERE market = 'STEAMCOMMUNITY' AND day < '2022-01-01'"
    ).fetchall()]
    logger.info(f"  Unique items: {len(names):,}")

    weekly_est = local_conn.execute(
        """SELECT COUNT(*) FROM (
            SELECT 1 FROM sales_history
            WHERE market = 'STEAMCOMMUNITY' AND day < '2022-01-01'
            GROUP BY market_hash_name, strftime('%Y-%W', day)
        )"""
    ).fetchone()[0]
    logger.info(f"  Estimated weekly rows: {weekly_est:,}")

    if dry_run:
        logger.info(f"  Would import ~{weekly_est:,} weekly rows for {len(names):,} items")
        local_conn.close()
        return weekly_est

    existing = db.execute(
        text("SELECT COUNT(*) FROM price_history WHERE source = 'steam_historical'")
    ).scalar() or 0
    if existing > 0:
        logger.info(f"  Source 'steam_historical' already has {existing:,} rows — skipping")
        local_conn.close()
        return existing

    name_map = _ensure_items_exist(db, names, local_conn)
    logger.info(f"  Name map has {len(name_map):,} entries")

    columns = ["item_id", "timestamp", "price", "volume", "median_price", "source"]
    source_label = "steam_historical"
    col_defs = ", ".join(f"{c} TEXT" for c in columns)

    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        tmp = f"_staging_price_history_{source_label}"
        cur.execute(f"DROP TABLE IF EXISTS {tmp}")
        cur.execute(f"CREATE TEMP TABLE {tmp} ({col_defs})")

        total_inserted = 0
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        chunk_count = 0
        chunk_rows = 50000

        cursor = local_conn.execute(
            """SELECT market_hash_name, MIN(day) AS week_start,
                      AVG(median_price) AS avg_price, SUM(volume) AS total_volume
               FROM sales_history
               WHERE market = 'STEAMCOMMUNITY' AND day < '2022-01-01'
               GROUP BY market_hash_name, strftime('%Y-%W', day)
               ORDER BY market_hash_name, week_start"""
        )

        for row in cursor:
            market_hash_name, week_start, avg_price, total_volume = row
            item_pk = name_map.get(market_hash_name)
            if item_pk is None or avg_price is None:
                continue

            writer.writerow((
                str(item_pk),
                f"{week_start} 00:00:00",
                f"{avg_price:.6f}",
                str(int(total_volume)) if total_volume is not None else "",
                f"{avg_price:.6f}",
                source_label,
            ))
            chunk_count += 1

            if chunk_count >= chunk_rows:
                buf.seek(0)
                cur.execute(f"SELECT COUNT(*) FROM price_history WHERE source = '{source_label}'")
                before_total = cur.fetchone()[0]
                cur.copy_from(buf, tmp, null="", columns=columns)
                cur.execute(f"""
                    INSERT INTO price_history
                        (item_id, timestamp, price, volume, median_price, source)
                    SELECT NULLIF(item_id, '')::integer,
                           NULLIF(timestamp, '')::timestamp,
                           NULLIF(price, '')::numeric,
                           NULLIF(volume, '')::integer,
                           NULLIF(median_price, '')::numeric,
                           source
                    FROM {tmp}
                    ON CONFLICT (item_id, timestamp, source) DO NOTHING
                """)
                cur.execute(f"SELECT COUNT(*) FROM price_history WHERE source = '{source_label}'")
                after_total = cur.fetchone()[0]
                cur.execute(f"TRUNCATE TABLE {tmp}")
                raw.commit()
                inserted = after_total - before_total
                total_inserted += inserted
                buf = io.StringIO()
                writer = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                chunk_count = 0
                logger.info(f"    Imported {total_inserted:,}/{weekly_est:,} weekly rows")

        if chunk_count > 0:
            buf.seek(0)
            cur.execute(f"SELECT COUNT(*) FROM price_history WHERE source = '{source_label}'")
            before_total = cur.fetchone()[0]
            cur.copy_from(buf, tmp, null="", columns=columns)
            cur.execute(f"""
                INSERT INTO price_history
                    (item_id, timestamp, price, volume, median_price, source)
                SELECT NULLIF(item_id, '')::integer,
                       NULLIF(timestamp, '')::timestamp,
                       NULLIF(price, '')::numeric,
                       NULLIF(volume, '')::integer,
                       NULLIF(median_price, '')::numeric,
                       source
                FROM {tmp}
                ON CONFLICT (item_id, timestamp, source) DO NOTHING
            """)
            cur.execute("SELECT COUNT(*) FROM price_history WHERE source = %s", (source_label,))
            after_total = cur.fetchone()[0]
            inserted = after_total - before_total
            total_inserted += inserted

        cur.execute(f"DROP TABLE IF EXISTS {tmp}")
        raw.commit()
        logger.info(f"  ✅ steam_historical: {total_inserted:,} weekly rows imported")
        return total_inserted
    finally:
        raw.close()


# ── Verification ────────────────────────────────────────────────────────


def run_storage_check(db):
    logger.info("=" * 60)
    logger.info("Phase 3a: Storage verification")

    rows: dict[str, int] = {}
    all_sources = [
        "kaggle_csgo",
        "market_csgo",
        "steam_historical",
        "aggregator_sync",
        "csgotrader",
        "steam",
        "cs2sh_archive",
        "synthetic_demo",
        "ssr_history",
    ] + STALE_SOURCES

    for src in all_sources:
        c = db.execute(
            text("SELECT COUNT(*) FROM price_history WHERE source = :src"), {"src": src}
        ).scalar() or 0
        if c:
            rows[src] = c

    logger.info("  Price history by source:")
    for src, c in sorted(rows.items(), key=lambda x: -x[1]):
        logger.info(f"    {src:<40} {c:>10,}")

    total_ph = db.execute(text("SELECT COUNT(*) FROM price_history")).scalar() or 0
    total_items = db.execute(text("SELECT COUNT(*) FROM items")).scalar() or 0
    null_classid = db.execute(
        text("SELECT COUNT(*) FROM items WHERE classid IS NULL")
    ).scalar() or 0

    logger.info(f"  Total price_history: {total_ph:,}")
    logger.info(f"  Total items: {total_items:,}")
    logger.info(f"  Items with NULL classid: {null_classid:,}")
    
    # Type distribution
    type_dist = db.execute(
        text("SELECT type, COUNT(*) FROM items GROUP BY type ORDER BY COUNT(*) DESC")
    ).fetchall()
    if type_dist:
        logger.info("  Items by type:")
        for t, c in type_dist:
            logger.info(f"    {t:<15} {c:>8,}")
    
    logger.info("  ✅ Phase 3a complete")


def run_phase_1(db, dry_run: bool):
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║            PHASE 1: CLEAN SUPABASE                      ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    phase_1a_delete_stale_sources(db, dry_run)
    phase_1b_deduplicate_items(db, dry_run)
    phase_1c_enrich_items(db, dry_run)


def run_phase_2(db, dry_run: bool):
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║        PHASE 2: IMPORT HISTORICAL DATA                  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    phase_2a_import_marketcsgo(db, dry_run)
    phase_2b_import_steam_weekly(db, dry_run)


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Historical data migration: clean Supabase + import local data"
    )
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Run only a specific phase")
    parser.add_argument("--dry-run", action="store_true", help="Count/show without modifying")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("HISTORICAL DATA MIGRATION")
    logger.info(f"  Started at:  {datetime.now().isoformat()}")
    logger.info(f"  Dry run:     {args.dry_run}")
    logger.info(f"  Phase:       {'all' if not args.phase else args.phase}")
    logger.info("=" * 60)

    db = SessionLocal()

    try:
        if args.phase == 1 or not args.phase:
            run_phase_1(db, args.dry_run)
        if args.phase == 2 or not args.phase:
            run_phase_2(db, args.dry_run)

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║           VERIFICATION                                  ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")
        run_storage_check(db)

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ ALL PHASES COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
