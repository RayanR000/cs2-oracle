#!/usr/bin/env python3
"""
Repair gaps in market_catalog.db by re-fetching missing items.

Uses the same burst pattern (10 rapid + 30s cooldown) that worked for the
original build. Saves gaps incrementally for resumability.

Monitor:  tail -f backend/runtime/gap_repair.log
Resume:   python3 scripts/repair_catalog_gaps.py --resume
Fetch:    python3 scripts/repair_catalog_gaps.py --fetch-only
Scan:     python3 scripts/repair_catalog_gaps.py --scan-only
"""

import argparse
import logging
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

LOG_PATH = Path(__file__).parent.parent / "runtime" / "gap_repair.log"

file_handler = logging.FileHandler(str(LOG_PATH))
file_handler.stream = open(str(LOG_PATH), "a", buffering=1)  # line-buffered for tail -f

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        file_handler,
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "runtime" / "market_catalog.db"
GAPS_FILE = Path(__file__).parent.parent / "runtime" / "pending_gaps.txt"
ITEMS_PER_PAGE = 10
BURST_SIZE = 10
BURST_COOLDOWN = 30


def get_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
    return session


def fetch_page(session, offset):
    """Fetch one page. Returns (items_list, http_status) or (None, status) on failure."""
    for attempt in range(3):
        try:
            resp = session.get(
                "https://steamcommunity.com/market/search/render/",
                params={"appid": 730, "norender": 1, "start": offset, "count": ITEMS_PER_PAGE},
                timeout=30,
            )
            if resp.status_code == 429:
                logger.debug(f"  429 at offset {offset} (attempt {attempt + 1})")
                return None, 429
            if resp.status_code != 200:
                logger.warning(f"  HTTP {resp.status_code} at offset {offset} (attempt {attempt + 1})")
                time.sleep(5)
                continue
            data = resp.json()
            if data.get("results") is None:
                logger.debug(f"  Null results at offset {offset} (attempt {attempt + 1})")
                return None, 429
            return data.get("results", []), 200
        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout at offset {offset} (attempt {attempt + 1})")
            time.sleep(10)
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"  Connection error at offset {offset} (attempt {attempt + 1}): {e}")
            time.sleep(10)
        except Exception as e:
            logger.warning(f"  Error at offset {offset} (attempt {attempt + 1}): {e}")
            time.sleep(10)
    return None, 0


def item_in_db(conn, name):
    return conn.execute("SELECT COUNT(*) FROM market_items WHERE hash_name = ?", (name,)).fetchone()[0] > 0


def insert_item(conn, item):
    conn.execute(
        """INSERT OR IGNORE INTO market_items
           (hash_name, name, type, sell_price, sell_price_text, sale_price_text,
            sell_listings, tradable, commodity, classid, name_color, icon_url,
            bucket_group_id, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item.get("hash_name"),
            item.get("name"),
            item.get("type"),
            item.get("sell_price"),
            item.get("sell_price_text"),
            item.get("sale_price_text"),
            item.get("sell_listings"),
            item.get("tradable"),
            item.get("commodity"),
            item.get("classid"),
            item.get("name_color"),
            item.get("icon_url"),
            item.get("bucket_group_id"),
            datetime.now(UTC).isoformat(),
        ),
    )


def load_pending_gaps():
    if not GAPS_FILE.exists():
        return set()
    with open(GAPS_FILE) as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def get_highest_gap_offset():
    if not GAPS_FILE.exists():
        return 0
    with open(GAPS_FILE) as f:
        offsets = [int(line.strip()) for line in f if line.strip().isdigit()]
    return max(offsets) if offsets else 0


def save_gap(offset):
    with open(GAPS_FILE, "a") as f:
        f.write(f"{offset}\n")


def clear_gaps_file():
    if GAPS_FILE.exists():
        GAPS_FILE.unlink()


def scan_for_gaps(conn, session, max_offset, resume=False, start_offset=None):
    already_found = load_pending_gaps() if resume else set()
    if start_offset is not None:
        scan_start = start_offset
        gaps = list(already_found)
        logger.info(f"Starting scan from offset {scan_start} (user-specified)")
    elif already_found:
        scan_start = get_highest_gap_offset() + ITEMS_PER_PAGE
        gaps = list(already_found)
        logger.info(f"Resuming scan from offset {scan_start} — {len(already_found)} gaps already known")
    else:
        scan_start = 0
        gaps = []

    logger.info("=" * 60)
    logger.info(f"PHASE 1: SCAN — offset range {scan_start} to {max_offset}")
    logger.info(f"Burst pattern: {BURST_SIZE} rapid + {BURST_COOLDOWN}s cooldown")
    logger.info("=" * 60)

    scanned = 0
    request_count = 0
    rate_limit_hits = 0
    errors = 0
    start = time.time()

    for offset in range(scan_start, max_offset, ITEMS_PER_PAGE):
        t0 = time.time()
        items, status = fetch_page(session, offset)
        request_count += 1
        elapsed_req = time.time() - t0

        if items is None:
            rate_limit_hits += 1
            logger.warning(f"Rate limit #{rate_limit_hits} at offset {offset} — backing off 30s")
            time.sleep(30)
            request_count = 0
            t0 = time.time()
            items, status = fetch_page(session, offset)
            request_count += 1
            elapsed_req = time.time() - t0
            if items is None:
                errors += 1
                logger.error(f"FAILED offset {offset} after retry — adding to gaps")
                gaps.append(offset)
                save_gap(offset)
                continue

        if len(items) == 0:
            logger.debug(f"  Empty page at offset {offset} ({elapsed_req:.1f}s)")
            if request_count % BURST_SIZE == 0:
                logger.info(f"  Burst cooldown after {request_count} requests")
                time.sleep(BURST_COOLDOWN)
                request_count = 0
            continue

        # Check for missing items
        missing_names = []
        for item in items:
            name = item.get("name", "")
            if not item_in_db(conn, name):
                missing_names.append(name)

        if missing_names:
            gaps.append(offset)
            save_gap(offset)
            logger.info(f"  GAP offset {offset}: {len(missing_names)}/{len(items)} missing — {missing_names[:3]}")

        scanned += 1

        # Progress every 100 offsets
        if scanned % 100 == 0:
            elapsed = time.time() - start
            pct = (offset / max_offset) * 100
            rate = scanned / elapsed * 60
            logger.info(
                f"PROGRESS: {scanned} scanned, {len(gaps)} gaps, {pct:.0f}% of market, {rate:.0f} req/min ({elapsed / 60:.1f}m)"
            )

        # Burst cooldown
        if request_count % BURST_SIZE == 0:
            logger.debug(f"  Burst cooldown ({request_count} requests)")
            time.sleep(BURST_COOLDOWN)
            request_count = 0

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("SCAN COMPLETE")
    logger.info(f"  Scanned: {scanned} offsets")
    logger.info(f"  Gaps found: {len(gaps)}")
    logger.info(f"  Rate limits: {rate_limit_hits}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info(f"  Rate: {scanned / elapsed * 60:.0f} req/min")
    logger.info("=" * 60)
    return gaps


def fetch_gaps(conn, session, gaps):
    logger.info("=" * 60)
    logger.info(f"PHASE 2: FETCH — {len(gaps)} gap offsets")
    logger.info(f"Burst pattern: {BURST_SIZE} rapid + {BURST_COOLDOWN}s cooldown")
    logger.info("=" * 60)

    stats = {"inserted": 0, "already_exists": 0, "failed": 0, "rate_limits": 0}
    start_time = time.time()
    request_count = 0

    for i, offset in enumerate(gaps):
        t0 = time.time()
        items, status = fetch_page(session, offset)
        request_count += 1
        elapsed_req = time.time() - t0

        if items is None:
            stats["rate_limits"] += 1
            logger.warning(f"Rate limit #{stats['rate_limits']} at gap offset {offset} — backing off 30s")
            time.sleep(30)
            request_count = 0
            t0 = time.time()
            items, status = fetch_page(session, offset)
            request_count += 1
            elapsed_req = time.time() - t0
            if items is None:
                logger.error(f"FAILED gap offset {offset} after retry")
                stats["failed"] += 1
                continue

        page_inserted = 0
        page_existing = 0
        for item in items:
            name = item.get("name", "unknown")
            if item_in_db(conn, name):
                page_existing += 1
                stats["already_exists"] += 1
            else:
                insert_item(conn, item)
                page_inserted += 1
                stats["inserted"] += 1
                logger.info(f"  INSERTED [{offset}]: {name}")

        conn.commit()

        # Per-page summary
        logger.debug(
            f"  Page {offset}: {len(items)} items, {page_inserted} new, {page_existing} existing ({elapsed_req:.1f}s)"
        )

        # Progress every 25 gaps
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            pct = ((i + 1) / len(gaps)) * 100
            rate = (i + 1) / elapsed * 60
            db_count = conn.execute("SELECT COUNT(*) FROM market_items").fetchone()[0]
            logger.info(
                f"FETCH PROGRESS: {i + 1}/{len(gaps)} gaps ({pct:.0f}%) | "
                f"{stats['inserted']} inserted, {stats['already_exists']} existing, {stats['failed']} failed | "
                f"DB: {db_count} items | {rate:.0f} gaps/min ({elapsed / 60:.1f}m)"
            )

        # Burst cooldown
        if request_count % BURST_SIZE == 0 and i + 1 < len(gaps):
            logger.info(f"  Burst cooldown ({i + 1}/{len(gaps)} gaps done)")
            time.sleep(BURST_COOLDOWN)
            request_count = 0

    conn.commit()

    final_count = conn.execute("SELECT COUNT(*) FROM market_items").fetchone()[0]
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info(f"  Gaps attempted: {len(gaps)}")
    logger.info(f"  Items inserted: {stats['inserted']}")
    logger.info(f"  Already existed: {stats['already_exists']}")
    logger.info(f"  Rate limits: {stats['rate_limits']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Catalog: {final_count} items")
    logger.info("=" * 60)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Repair gaps in market catalog")
    parser.add_argument("--dry-run", action="store_true", help="Scan for gaps without fetching")
    parser.add_argument("--scan-only", action="store_true", help="Only scan, don't fetch")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch saved gaps, skip scan")
    parser.add_argument("--resume", action="store_true", help="Resume from saved gaps")
    parser.add_argument(
        "--start-offset", type=int, default=None, help="Start scan from this offset (overrides --resume)"
    )
    parser.add_argument("--max-offset", type=int, default=35000, help="Max offset to scan")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GAP REPAIR SCRIPT STARTED")
    mode = "fetch-only" if args.fetch_only else "scan-only" if args.scan_only else "scan+fetch"
    if args.start_offset is not None:
        mode += f" (start-offset={args.start_offset})"
    logger.info(f"Mode: {mode}")
    logger.info(f"DB: {DB_PATH}")
    logger.info(f"Gaps file: {GAPS_FILE}")
    logger.info(f"Time: {datetime.now(UTC).isoformat()}")
    logger.info("=" * 60)

    conn = sqlite3.connect(str(DB_PATH))
    session = get_session()
    current_count = conn.execute("SELECT COUNT(*) FROM market_items").fetchone()[0]
    logger.info(f"Current catalog: {current_count} items")

    if args.fetch_only:
        gaps = sorted(load_pending_gaps())
        if not gaps:
            logger.info("No saved gaps to fetch — run scan first")
            conn.close()
            return
        logger.info(f"Loaded {len(gaps)} saved gaps")
    else:
        gaps = scan_for_gaps(conn, session, args.max_offset, resume=args.resume, start_offset=args.start_offset)

    if not gaps:
        logger.info("No gaps found — catalog is complete!")
        clear_gaps_file()
        conn.close()
        return

    if args.dry_run or args.scan_only:
        logger.info(f"Gaps at offsets: {gaps}")
        conn.close()
        return

    # Fetch
    stats = fetch_gaps(conn, session, gaps)
    clear_gaps_file()

    final_count = conn.execute("SELECT COUNT(*) FROM market_items").fetchone()[0]
    logger.info(f"FINAL: {current_count} -> {final_count} items (+{final_count - current_count})")

    conn.close()


if __name__ == "__main__":
    main()
