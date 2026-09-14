#!/usr/bin/env python3
"""
Build a complete catalog of all CS2 items on the Steam Community Market.

Paginates through /market/search/render/ and saves all items to local SQLite.
Uses burst rate limiting (10 rapid requests + 30s pause) to avoid bans.

Features:
    - Burst rate limiting: 10 rapid requests + 30s pause
    - Auto-pause on sustained 429s (IP ban protection)
    - Failed pages tracked for retry
    - Pause/resume support
    - Health monitoring with periodic reports

Usage:
    python scripts/build_market_catalog.py                    # Full catalog
    python scripts/build_market_catalog.py --resume           # Resume from last offset
    python scripts/build_market_catalog.py --retry-failed     # Retry failed pages
    python scripts/build_market_catalog.py --status           # Show progress
    python scripts/build_market_catalog.py --burst-size 5     # Custom burst size
    python scripts/build_market_catalog.py --burst-pause 60   # Custom pause between bursts
"""

import argparse
import logging
import random
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# Ensure runtime directory exists before setting up file logging
(Path(__file__).parent.parent / "runtime").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(Path(__file__).parent.parent / "runtime" / "market_catalog.log")),
    ],
)
logger = logging.getLogger("market_catalog")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CATALOG_DB_PATH = Path(__file__).parent.parent / "runtime" / "market_catalog.db"
PROGRESS_FILE = Path(__file__).parent.parent / "runtime" / "market_catalog_progress.json"

# Rate limiting — burst pattern
DEFAULT_BURST_SIZE = 10  # requests per burst
DEFAULT_BURST_PAUSE = 30.0  # seconds between bursts
MAX_RETRIES = 3  # retries per request on 429
RETRY_BACKOFF = [30, 60, 120]  # seconds to wait after each 429

# Auto-pause thresholds
MAX_CONSECUTIVE_429 = 3
MAX_CONSECUTIVE_FAILURES = 10

HEALTH_REPORT_INTERVAL = 500  # log health report every N items

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


# ---------------------------------------------------------------------------
# Local SQLite schema
# ---------------------------------------------------------------------------


def init_catalog_db(db_path: Path) -> sqlite3.Connection:
    """Create/open the local market catalog database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS market_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash_name TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            sell_price INTEGER,
            sell_price_text TEXT,
            sale_price_text TEXT,
            sell_listings INTEGER,
            tradable INTEGER,
            commodity INTEGER,
            classid TEXT,
            name_color TEXT,
            icon_url TEXT,
            bucket_group_id TEXT,
            last_updated TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_mi_type ON market_items(type);
        CREATE INDEX IF NOT EXISTS idx_mi_hash ON market_items(hash_name);

        CREATE TABLE IF NOT EXISTS catalog_progress (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_offset INTEGER,
            total_items INTEGER,
            started_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS failed_pages (
            offset INTEGER PRIMARY KEY,
            error_reason TEXT,
            failed_at TEXT
        );
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def load_progress(local_conn: sqlite3.Connection) -> dict:
    """Load catalog progress from local DB."""
    row = local_conn.execute(
        "SELECT last_offset, total_items, started_at, updated_at FROM catalog_progress WHERE id = 1"
    ).fetchone()

    if row:
        return {
            "last_offset": row[0],
            "total_items": row[1],
            "started_at": row[2],
            "updated_at": row[3],
        }
    return {
        "last_offset": None,
        "total_items": None,
        "started_at": None,
        "updated_at": None,
    }


def save_progress(
    local_conn: sqlite3.Connection,
    last_offset: int,
    total_items: int,
):
    """Save or update catalog progress."""
    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    existing = load_progress(local_conn)
    started = existing.get("started_at") or now

    local_conn.execute(
        """INSERT OR REPLACE INTO catalog_progress
           (id, last_offset, total_items, started_at, updated_at)
           VALUES (1, ?, ?, ?, ?)""",
        (last_offset, total_items, started, now),
    )
    local_conn.commit()


def record_failed_page(
    local_conn: sqlite3.Connection,
    offset: int,
    reason: str = "all retries exhausted",
):
    """Record a failed page offset for later retry."""
    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    local_conn.execute(
        "INSERT OR REPLACE INTO failed_pages (offset, error_reason, failed_at) VALUES (?, ?, ?)",
        (offset, reason, now),
    )
    local_conn.commit()


def clear_failed_page(local_conn: sqlite3.Connection, offset: int):
    """Remove a failed page offset after successful retry."""
    local_conn.execute("DELETE FROM failed_pages WHERE offset = ?", (offset,))
    local_conn.commit()


def load_failed_pages(local_conn: sqlite3.Connection) -> list[int]:
    """Load all failed page offsets, sorted."""
    rows = local_conn.execute("SELECT offset FROM failed_pages ORDER BY offset").fetchall()
    return [r[0] for r in rows]


def get_failed_pages_count(local_conn: sqlite3.Connection) -> int:
    """Get count of failed pages."""
    return local_conn.execute("SELECT COUNT(*) FROM failed_pages").fetchone()[0]


# ---------------------------------------------------------------------------
# Health monitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Tracks API health during catalog build."""

    def __init__(self, max_consecutive_429: int = MAX_CONSECUTIVE_429):
        self.max_consecutive_429 = max_consecutive_429

        # Counters
        self.total_ok = 0
        self.total_failed = 0
        self.total_429 = 0
        self.total_items_fetched = 0

        # Consecutive tracking
        self.consecutive_ok = 0
        self.consecutive_failures = 0
        self.consecutive_429 = 0

        # State
        self.banned = False
        self.last_results = []  # last 20: ("OK"|"FAILED"|"429", offset)

    def record_ok(self, offset: int, item_count: int):
        self.total_ok += 1
        self.total_items_fetched += item_count
        self.consecutive_ok += 1
        self.consecutive_failures = 0
        self.consecutive_429 = 0
        self._append_result("OK", offset)

    def record_failed(self, offset: int):
        self.total_failed += 1
        self.consecutive_failures += 1
        self.consecutive_ok = 0
        self.consecutive_429 = 0
        self._append_result("FAILED", offset)

    def record_429(self, offset: int):
        self.total_429 += 1
        self.consecutive_429 += 1
        self.consecutive_failures = 0
        self.consecutive_ok = 0
        self._append_result("429", offset)

    def _append_result(self, status: str, offset: int):
        self.last_results.append((status, offset))
        if len(self.last_results) > 20:
            self.last_results.pop(0)

    def should_pause(self) -> str | None:
        """Check if we should auto-pause. Returns reason string or None."""
        if self.consecutive_429 >= self.max_consecutive_429:
            return (
                f"PAUSE: {self.consecutive_429} consecutive rate limits (429) "
                f"(threshold: {self.max_consecutive_429}). "
                f"Possible cause: IP banned. Wait before resuming."
            )
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return f"PAUSE: {self.consecutive_failures} consecutive failures (threshold: {MAX_CONSECUTIVE_FAILURES})."
        return None

    def log_health_report(self, offset: int, total: int, elapsed: float):
        """Log a periodic health report."""
        items_per_min = self.total_items_fetched / (elapsed / 60) if elapsed > 0 else 0
        pages_done = offset // 10
        total_pages = (total + 9) // 10
        eta_seconds = (total_pages - pages_done) / (pages_done / elapsed) if pages_done > 0 and elapsed > 0 else 0

        logger.info("=" * 70)
        logger.info(f"HEALTH REPORT — offset {offset}/{total} ({offset * 100 // total}%)")
        logger.info(f"  OK: {self.total_ok} | Failed: {self.total_failed} | 429s: {self.total_429}")
        logger.info(
            f"  Items fetched: {self.total_items_fetched} | "
            f"Rate: {items_per_min:.0f} items/min | "
            f"ETA: {eta_seconds / 3600:.1f} hrs"
        )
        logger.info(
            f"  Consecutive — OK: {self.consecutive_ok} | "
            f"Failures: {self.consecutive_failures} | "
            f"429: {self.consecutive_429}"
        )
        if self.banned:
            logger.warning("  WARNING: IP ban detected")
        logger.info("=" * 70)

    def log_final_summary(self, elapsed: float):
        """Log the final health summary."""
        logger.info("=" * 70)
        logger.info("FINAL HEALTH SUMMARY")
        logger.info(f"  Total pages fetched: {self.total_ok}")
        logger.info(f"  Total items: {self.total_items_fetched}")
        logger.info(f"  Failed pages: {self.total_failed}")
        logger.info(f"  Rate limited (429): {self.total_429}")
        logger.info(f"  Duration: {elapsed / 3600:.1f} hours")
        if self.banned:
            logger.warning("  IP ban detected — wait before resuming")
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Steam Market API client (burst pattern)
# ---------------------------------------------------------------------------


class SteamMarketCatalogClient:
    """Fetches item catalog from Steam's /market/search/render/ endpoint."""

    SEARCH_URL = "https://steamcommunity.com/market/search/render/"

    def __init__(self):
        self.session = requests.Session()
        self._rotate_ua()
        self.last_request_time = 0.0
        self.last_fetch_429_count = 0

    def _rotate_ua(self):
        ua = random.choice(USER_AGENTS)
        self.session.headers["User-Agent"] = ua

    def fetch_page(self, offset: int) -> list[dict] | None:
        """
        Fetch a single page of results (10 items).
        Returns list of item dicts or None on failure.
        Retries on 429 with exponential backoff.
        """
        params = {"appid": 730, "norender": 1, "start": offset, "count": 100}
        self.last_fetch_429_count = 0

        for attempt, backoff in enumerate(RETRY_BACKOFF):
            try:
                self._rotate_ua()
                resp = self.session.get(self.SEARCH_URL, params=params, timeout=30)

                if resp.status_code == 429:
                    self.last_fetch_429_count += 1
                    logger.warning(f"429 at offset={offset}, backing off {backoff}s")
                    time.sleep(backoff)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if data.get("results") is None:
                    # Null results = rate limited (not a 429 status but same effect)
                    self.last_fetch_429_count += 1
                    logger.warning(f"Null results at offset={offset}, backing off {backoff}s")
                    time.sleep(backoff)
                    continue

                return data.get("results", [])

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed at offset={offset} (attempt {attempt + 1}): {e}")
                if attempt < len(RETRY_BACKOFF) - 1:
                    time.sleep(backoff)

        return None

    def get_total_count(self) -> int | None:
        """Get total number of items on the market."""
        params = {"appid": 730, "norender": 1, "start": 0, "count": 10}
        try:
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("total_count")
        except Exception as e:
            logger.error(f"Failed to get total count: {e}")
            return None


# ---------------------------------------------------------------------------
# Item storage
# ---------------------------------------------------------------------------


def store_items(
    local_conn: sqlite3.Connection,
    items: list[dict],
    dry_run: bool = False,
) -> int:
    """Store items from a search/render page into the catalog. Returns rows inserted."""
    if not items:
        return 0

    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    rows = []
    for item in items:
        asset = item.get("asset_description", {})
        rows.append(
            (
                item.get("hash_name"),
                item.get("name"),
                asset.get("type"),
                item.get("sell_price"),
                item.get("sell_price_text"),
                item.get("sale_price_text"),
                item.get("sell_listings"),
                asset.get("tradable"),
                asset.get("commodity"),
                asset.get("classid"),
                asset.get("name_color"),
                asset.get("icon_url"),
                asset.get("market_bucket_group_id"),
                now,
            )
        )

    if dry_run:
        return len(rows)

    local_conn.executemany(
        """INSERT OR REPLACE INTO market_items
           (hash_name, name, type, sell_price, sell_price_text, sale_price_text,
            sell_listings, tradable, commodity, classid, name_color, icon_url,
            bucket_group_id, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    local_conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Main catalog build
# ---------------------------------------------------------------------------


def run_catalog_build(
    resume: bool = False,
    dry_run: bool = False,
    burst_size: int = DEFAULT_BURST_SIZE,
    burst_pause: float = DEFAULT_BURST_PAUSE,
):
    """Build the market catalog using burst rate limiting."""
    logger.info("=" * 70)
    logger.info("Market Catalog Build — Starting")
    logger.info(f"  Burst: {burst_size} requests, then {burst_pause}s pause")
    logger.info(f"  Auto-pause: {MAX_CONSECUTIVE_429} consecutive 429s")
    logger.info("=" * 70)

    # 1. Initialize
    client = SteamMarketCatalogClient()
    health = HealthMonitor()

    # 2. Get total count
    total = client.get_total_count()
    if total is None:
        logger.error("Could not get total item count from Steam. Aborting.")
        return
    logger.info(f"Total items on market: {total}")

    # 3. Initialize local DB
    local_conn = init_catalog_db(CATALOG_DB_PATH)

    # 4. Determine starting point
    progress = load_progress(local_conn)
    start_offset = 0

    if resume and progress["last_offset"] is not None:
        start_offset = progress["last_offset"] + 10  # next page
        logger.info(f"Resuming from offset={start_offset} — previously fetched {progress['total_items']} items")

    total_pages = (total + 9) // 10
    start_page = start_offset // 10
    logger.info(f"Pages to fetch: {total_pages - start_page} (of {total_pages} total)")

    if start_offset >= total:
        logger.info("Nothing to do — catalog already complete")
        print_catalog_summary(local_conn)
        return

    # 5. Build catalog in bursts
    total_rows = 0
    start_time = time.time()
    current_offset = start_offset
    paused = False

    while current_offset < total:
        # --- BURST: send burst_size requests as fast as possible ---
        burst_start = time.time()
        burst_items = 0

        for burst_idx in range(burst_size):
            if current_offset >= total:
                break

            items = client.fetch_page(current_offset)

            # Track 429s that occurred during this fetch (including retries)
            if client.last_fetch_429_count > 0:
                for _ in range(client.last_fetch_429_count):
                    health.record_429(current_offset)
                logger.info(
                    f"  [429 tracker] {client.last_fetch_429_count} rate limit(s) "
                    f"during fetch at offset={current_offset} "
                    f"(total 429s: {health.total_429})"
                )

            if items is None:
                # Failed after all retries
                health.record_failed(current_offset)
                record_failed_page(local_conn, current_offset)
                logger.warning(f"Failed page at offset={current_offset} (recorded for retry)")

                # Check auto-pause
                pause_reason = health.should_pause()
                if pause_reason:
                    logger.critical(pause_reason)
                    logger.critical(
                        f"Auto-paused at offset={current_offset}. "
                        f"Progress saved. Use --resume to continue after waiting."
                    )
                    paused = True
                    break

                current_offset += 10
                continue

            if len(items) == 0:
                # End of results
                logger.info(f"End of results at offset={current_offset}")
                break

            rows = store_items(local_conn, items, dry_run=dry_run)
            total_rows += rows
            burst_items += len(items)
            health.record_ok(current_offset, len(items))

            # Progress log every 100 pages
            pages_done = (current_offset // 10) + 1
            if pages_done % 100 == 0:
                elapsed = time.time() - start_time
                items_per_min = health.total_items_fetched / (elapsed / 60) if elapsed > 0 else 0
                eta_seconds = (
                    ((total_pages - pages_done) / (pages_done / elapsed)) if pages_done > 0 and elapsed > 0 else 0
                )
                logger.info(
                    f"Progress: {pages_done}/{total_pages} pages | "
                    f"Items: {health.total_items_fetched} | "
                    f"Rate: {items_per_min:.0f} items/min | "
                    f"ETA: {eta_seconds / 3600:.1f} hrs"
                )

            # Health report every HEALTH_REPORT_INTERVAL items
            if health.total_items_fetched % HEALTH_REPORT_INTERVAL < 10:
                elapsed = time.time() - start_time
                health.log_health_report(current_offset, total, elapsed)

            current_offset += 10

        if paused:
            break

        # Save progress after each burst
        save_progress(local_conn, current_offset - 10, health.total_items_fetched)

        # --- PAUSE: wait between bursts ---
        if current_offset < total:
            burst_elapsed = time.time() - burst_start
            remaining_pause = max(0, burst_pause - burst_elapsed)
            if remaining_pause > 0:
                time.sleep(remaining_pause)

    # 6. Summary
    elapsed = time.time() - start_time
    health.log_final_summary(elapsed)

    logger.info("=" * 70)
    if paused:
        logger.info("Catalog build PAUSED (auto-pause triggered)")
    else:
        logger.info(f"Catalog build {'(DRY RUN) ' if dry_run else ''}Complete")
    logger.info(f"  Items fetched: {health.total_items_fetched}")
    logger.info(f"  Rows stored: {total_rows}")
    logger.info(f"  Duration: {elapsed / 3600:.1f} hours")
    logger.info("=" * 70)

    print_catalog_summary(local_conn)
    local_conn.close()

    return paused


def print_catalog_summary(local_conn: sqlite3.Connection):
    """Print a summary of what's in the catalog database."""
    item_count = local_conn.execute("SELECT COUNT(*) FROM market_items").fetchone()[0]
    failed_count = get_failed_pages_count(local_conn)
    db_size = CATALOG_DB_PATH.stat().st_size / (1024 * 1024) if CATALOG_DB_PATH.exists() else 0

    # Type breakdown
    types = local_conn.execute(
        "SELECT type, COUNT(*) FROM market_items GROUP BY type ORDER BY COUNT(*) DESC LIMIT 10"
    ).fetchall()

    logger.info(f"Catalog DB ({CATALOG_DB_PATH.name}):")
    logger.info(f"  Items: {item_count}")
    logger.info(f"  Failed pages: {failed_count}")
    logger.info(f"  DB size: {db_size:.1f} MB")
    if types:
        logger.info("  Top types:")
        for t, c in types:
            logger.info(f"    {t}: {c}")


def run_retry_failed(
    dry_run: bool = False,
    burst_size: int = DEFAULT_BURST_SIZE,
    burst_pause: float = DEFAULT_BURST_PAUSE,
):
    """Retry all previously failed pages."""
    logger.info("=" * 70)
    logger.info("Market Catalog — Retry Failed Pages")
    logger.info("=" * 70)

    local_conn = init_catalog_db(CATALOG_DB_PATH)
    failed_offsets = load_failed_pages(local_conn)

    if not failed_offsets:
        logger.info("No failed pages to retry")
        print_catalog_summary(local_conn)
        local_conn.close()
        return

    logger.info(f"Found {len(failed_offsets)} failed pages to retry")
    logger.info(f"Offsets: {failed_offsets[:20]}{'...' if len(failed_offsets) > 20 else ''}")

    client = SteamMarketCatalogClient()
    health = HealthMonitor()

    start_time = time.time()
    total_rows = 0
    retried = 0
    still_failed = 0
    paused = False

    # Process failed offsets in bursts
    for burst_start_idx in range(0, len(failed_offsets), burst_size):
        burst_offsets = failed_offsets[burst_start_idx : burst_start_idx + burst_size]
        burst_time = time.time()

        for offset in burst_offsets:
            items = client.fetch_page(offset)

            # Track 429s during this fetch
            if client.last_fetch_429_count > 0:
                for _ in range(client.last_fetch_429_count):
                    health.record_429(offset)

            if items is None:
                still_failed += 1
                health.record_failed(offset)
                logger.warning(f"Still failed at offset={offset}")

                pause_reason = health.should_pause()
                if pause_reason:
                    logger.critical(pause_reason)
                    paused = True
                    break
                continue

            if len(items) == 0:
                # Page returned empty — might be end of results or deleted items
                clear_failed_page(local_conn, offset)
                retried += 1
                health.record_ok(offset, 0)
                logger.info(f"Cleared empty page at offset={offset}")
                continue

            rows = store_items(local_conn, items, dry_run=dry_run)
            total_rows += rows
            clear_failed_page(local_conn, offset)
            retried += 1
            health.record_ok(offset, len(items))
            logger.info(f"Retry OK: offset={offset} — {len(items)} items ({rows} rows)")

        if paused:
            break

        # Pause between bursts
        if burst_start_idx + burst_size < len(failed_offsets):
            burst_elapsed = time.time() - burst_time
            remaining_pause = max(0, burst_pause - burst_elapsed)
            if remaining_pause > 0:
                time.sleep(remaining_pause)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Retry Failed Pages — Complete")
    logger.info(f"  Retried successfully: {retried}")
    logger.info(f"  Still failed: {still_failed}")
    logger.info(f"  Items recovered: {health.total_items_fetched}")
    logger.info(f"  Rows stored: {total_rows}")
    logger.info(f"  Duration: {elapsed / 60:.1f} minutes")
    logger.info("=" * 70)

    remaining = get_failed_pages_count(local_conn)
    if remaining > 0:
        logger.warning(f"  {remaining} pages still failed — may need manual investigation")

    print_catalog_summary(local_conn)
    local_conn.close()


def print_status():
    """Print current catalog build status."""
    if not CATALOG_DB_PATH.exists():
        logger.info("No catalog database found — build not started")
        return

    local_conn = init_catalog_db(CATALOG_DB_PATH)
    progress = load_progress(local_conn)
    failed_count = get_failed_pages_count(local_conn)

    logger.info("Market Catalog Status:")
    if progress["last_offset"] is not None:
        logger.info(f"  Last offset: {progress['last_offset']}")
    logger.info(f"  Items fetched: {progress.get('total_items', 'N/A')}")
    logger.info(f"  Failed pages: {failed_count}")
    logger.info(f"  Started: {progress.get('started_at', 'N/A')}")
    logger.info(f"  Updated: {progress.get('updated_at', 'N/A')}")

    if failed_count > 0:
        failed_offsets = load_failed_pages(local_conn)
        logger.info(f"  Failed offsets: {failed_offsets[:20]}{'...' if len(failed_offsets) > 20 else ''}")
        logger.info("  To retry: python scripts/build_market_catalog.py --retry-failed")

    # Parse log for health
    log_path = Path(__file__).parent.parent / "runtime" / "market_catalog.log"
    if log_path.exists():
        try:
            lines = log_path.read_text().strip().split("\n")
            recent = lines[-500:] if len(lines) > 500 else lines
            ok_count = sum(1 for l in recent if "OK" in l and "HEALTH" not in l)
            failed_count_log = sum(1 for l in recent if "Failed" in l and "HEALTH" not in l)
            rate_429 = sum(1 for l in recent if "429" in l and "HEALTH" not in l)

            logger.info("  --- Recent Log Activity (last 500 lines) ---")
            logger.info(f"  OK: {ok_count} | Failed: {failed_count_log} | 429s: {rate_429}")

            pause_lines = [l for l in lines if "Auto-paused" in l or "PAUSE:" in l]
            if pause_lines:
                logger.warning(f"  AUTO-PAUSE DETECTED: {pause_lines[-1]}")
                logger.warning("  Wait, then use --resume to continue")
        except Exception as e:
            logger.info(f"  (Could not parse log: {e})")

    print_catalog_summary(local_conn)
    local_conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CS2 market catalog from Steam")
    parser.add_argument("--resume", action="store_true", help="Resume from last fetched offset")
    parser.add_argument("--retry-failed", action="store_true", help="Retry all previously failed pages")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument("--status", action="store_true", help="Show current progress")
    parser.add_argument(
        "--burst-size", type=int, default=DEFAULT_BURST_SIZE, help=f"Requests per burst (default: {DEFAULT_BURST_SIZE})"
    )
    parser.add_argument(
        "--burst-pause",
        type=float,
        default=DEFAULT_BURST_PAUSE,
        help=f"Seconds between bursts (default: {DEFAULT_BURST_PAUSE})",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.retry_failed:
        run_retry_failed(
            dry_run=args.dry_run,
            burst_size=args.burst_size,
            burst_pause=args.burst_pause,
        )
    else:
        run_catalog_build(
            resume=args.resume,
            dry_run=args.dry_run,
            burst_size=args.burst_size,
            burst_pause=args.burst_pause,
        )
