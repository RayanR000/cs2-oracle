#!/usr/bin/env python3
"""
Backfill SSR (Steam Supply Report) price history into local SQLite.

Fetches full historical price data from Steam's /market/pricehistory/ endpoint
for all items in the production database, stores downsampled results locally.

Features:
    - Tiered downsampling on ingest: daily (0-90d), weekly (91-730d), monthly (731d+)
    - Pause/resume: progress saved after every item
    - Auto-pause on sustained failures (rate limits, bans, session expiry)
    - Health monitoring with periodic reports every 500 items
    - Rate limiting with exponential backoff on 429s

Usage:
    python scripts/backfill_ssr_history.py                          # Full backfill (prod source)
    python scripts/backfill_ssr_history.py --source catalog         # Backfill from market_catalog.db (32k items)
    python scripts/backfill_ssr_history.py --resume                 # Resume from last position
    python scripts/backfill_ssr_history.py --limit 100              # Backfill only first 100 items
    python scripts/backfill_ssr_history.py --dry-run                # Preview without writing
    python scripts/backfill_ssr_history.py --source catalog --status  # Show catalog backfill progress
    python scripts/backfill_ssr_history.py --max-consecutive-failures 5   # Custom threshold
"""

import argparse
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from config import settings

# Ensure runtime directory exists before setting up file logging
(Path(__file__).parent.parent / "runtime").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(Path(__file__).parent.parent / "runtime" / "ssr_backfill.log")),
    ],
)
logger = logging.getLogger("ssr_backfill")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOCAL_DB_PATH = Path(__file__).parent.parent / "runtime" / "ssr_history.db"
CATALOG_LOCAL_DB_PATH = Path(__file__).parent.parent / "runtime" / "ssr_history_catalog.db"
CATALOG_DB_PATH = Path(__file__).parent.parent / "runtime" / "market_catalog.db"

# Price archive (repo root) — used by --modeled-only to find deep-history items
PRICE_ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"
# "Deep-history backbone": items with >= 1 year of distinct days in the archive.
# These are the ~4.8K items the forecaster most relies on (long-lookback features work).
MODELED_MIN_HISTORY_DAYS = 365
PROGRESS_FILE = Path(__file__).parent.parent / "runtime" / "ssr_backfill_progress.json"

REQUEST_DELAY = 4.0  # seconds between API calls (~15 req/min — ArchiSteamFarm-safe rate)
RETRY_ATTEMPTS = 5  # retries for genuine network errors (not 429s)
RETRY_DELAY = 10.0
BACKOFF_MULTIPLIER = 2.0
# 429s are NOT hammered: hitting Steam during an IP cooldown *refreshes* the ban.
# So we retry a 429 at most this many times, then surface it (RATE_LIMITED) and let the
# consecutive-429 auto-pause take over. Set to 0 for fail-fast (session validation).
MAX_429_RETRIES = 1

# Sentinel: get_price_history returns this (distinct from None/[] ) when rate-limited.
RATE_LIMITED = object()

# Auto-pause thresholds (configurable via CLI)
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10
DEFAULT_MAX_CONSECUTIVE_429 = 3  # pause fast on a real cooldown so we stop refreshing it
DEFAULT_MAX_CONSECUTIVE_EMPTY_AFTER_OK = 50
HEALTH_REPORT_INTERVAL = 500  # log health report every N items

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

# Downsampling tiers (applied on ingest)
DOWNSAMPLE_TIERS = [
    # (max_age_days, granularity): daily/weekly/monthly
    (90, "daily"),  # 0-90 days: daily candles
    (730, "weekly"),  # 91-730 days: weekly candles
    (float("inf"), "monthly"),  # 731+ days: monthly candles
]


# ---------------------------------------------------------------------------
# Health monitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Tracks API health, detects rate limits, bans, and session expiry."""

    def __init__(
        self,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        max_consecutive_429: int = DEFAULT_MAX_CONSECUTIVE_429,
        max_consecutive_empty_after_ok: int = DEFAULT_MAX_CONSECUTIVE_EMPTY_AFTER_OK,
    ):
        self.max_consecutive_failures = max_consecutive_failures
        self.max_consecutive_429 = max_consecutive_429
        self.max_consecutive_empty_after_ok = max_consecutive_empty_after_ok

        # Counters
        self.total_ok = 0
        self.total_empty = 0
        self.total_failed = 0
        self.total_429 = 0
        self.total_exceptions = 0

        # Consecutive tracking
        self.consecutive_ok = 0
        self.consecutive_failures = 0
        self.consecutive_429 = 0
        self.consecutive_empty_after_ok = 0  # EMPTY following a streak of OKs

        # State
        self.session_expired = False
        self.banned = False
        self.last_results = []  # last 20 results: ("OK"|"EMPTY"|"FAILED"|"429", item_name)
        self._had_ok_streak = False  # True if we've seen at least one OK in this run

    def record_ok(self, item_name: str):
        self.total_ok += 1
        self.consecutive_ok += 1
        self.consecutive_failures = 0
        self.consecutive_429 = 0
        self.consecutive_empty_after_ok = 0
        self._had_ok_streak = True
        self._append_result("OK", item_name)

    def record_empty(self, item_name: str):
        self.total_empty += 1
        self.consecutive_failures = 0
        self.consecutive_429 = 0
        if self._had_ok_streak and self.consecutive_ok == 0:
            # EMPTY after a streak of OKs (not immediately after another OK)
            self.consecutive_empty_after_ok += 1
        else:
            self.consecutive_empty_after_ok = 0
        self.consecutive_ok = 0
        self._append_result("EMPTY", item_name)

    def record_failed(self, item_name: str):
        self.total_failed += 1
        self.consecutive_failures += 1
        self.consecutive_ok = 0
        self.consecutive_429 = 0
        self.consecutive_empty_after_ok = 0
        self._append_result("FAILED", item_name)

    def record_429(self, item_name: str):
        self.total_429 += 1
        self.consecutive_429 += 1
        self.consecutive_failures = 0
        self.consecutive_ok = 0
        self.consecutive_empty_after_ok = 0
        self._append_result("429", item_name)

    def record_exception(self, item_name: str):
        self.total_exceptions += 1
        self.total_failed += 1
        self.consecutive_failures += 1
        self.consecutive_ok = 0
        self.consecutive_429 = 0
        self.consecutive_empty_after_ok = 0
        self._append_result("FAILED", item_name)

    def _append_result(self, status: str, item_name: str):
        self.last_results.append((status, item_name))
        if len(self.last_results) > 20:
            self.last_results.pop(0)

    def should_pause(self) -> str | None:
        """Check if we should auto-pause. Returns reason string or None."""
        if self.consecutive_failures >= self.max_consecutive_failures:
            return (
                f"PAUSE: {self.consecutive_failures} consecutive failures "
                f"(threshold: {self.max_consecutive_failures}). "
                f"Possible causes: cookie expiry, IP ban, network issue."
            )
        if self.consecutive_429 >= self.max_consecutive_429:
            return (
                f"PAUSE: {self.consecutive_429} consecutive rate limits (429) "
                f"(threshold: {self.max_consecutive_429}). "
                f"Possible causes: IP banned, too many requests."
            )
        if self.consecutive_empty_after_ok >= self.max_consecutive_empty_after_ok:
            self.session_expired = True
            return (
                f"PAUSE: {self.consecutive_empty_after_ok} consecutive EMPTY responses after OK streak "
                f"(threshold: {self.max_consecutive_empty_after_ok}). "
                f"Likely cause: session cookies expired."
            )
        return None

    def log_health_report(self, idx: int, total: int, elapsed: float):
        """Log a periodic health report."""
        rate = idx / elapsed * 3600 if elapsed > 0 else 0
        eta_hours = (total - idx) / rate if rate > 0 else 0
        total_items = self.total_ok + self.total_empty + self.total_failed
        success_rate = (self.total_ok / total_items * 100) if total_items > 0 else 0

        logger.info("=" * 70)
        logger.info(f"HEALTH REPORT — {idx}/{total} ({idx * 100 // total}%)")
        logger.info(
            f"  OK: {self.total_ok} | EMPTY: {self.total_empty} | "
            f"Failed: {self.total_failed} | 429s: {self.total_429} | "
            f"Exceptions: {self.total_exceptions}"
        )
        logger.info(f"  Success rate: {success_rate:.1f}% | Rate: {rate:.0f} items/hr | ETA: {eta_hours:.1f} hrs")
        logger.info(
            f"  Consecutive — OK: {self.consecutive_ok} | "
            f"Failures: {self.consecutive_failures} | "
            f"429: {self.consecutive_429} | "
            f"Empty(after OK): {self.consecutive_empty_after_ok}"
        )
        if self.session_expired:
            logger.warning("  WARNING: Session expiry detected")
        if self.banned:
            logger.warning("  WARNING: IP ban detected")
        logger.info("=" * 70)

    def log_final_summary(self, elapsed: float):
        """Log the final health summary at end of run."""
        total_items = self.total_ok + self.total_empty + self.total_failed
        success_rate = (self.total_ok / total_items * 100) if total_items > 0 else 0

        logger.info("=" * 70)
        logger.info("FINAL HEALTH SUMMARY")
        logger.info(f"  Total API calls: {total_items}")
        logger.info(f"  OK (with data): {self.total_ok}")
        logger.info(f"  EMPTY (no history): {self.total_empty}")
        logger.info(f"  FAILED (errors): {self.total_failed}")
        logger.info(f"  Rate limited (429): {self.total_429}")
        logger.info(f"  Exceptions: {self.total_exceptions}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Duration: {elapsed / 3600:.1f} hours")
        if self.session_expired:
            logger.warning("  Session expired during run — update cookies before resuming")
        if self.banned:
            logger.warning("  IP ban detected — wait before resuming")
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Local SQLite schema
# ---------------------------------------------------------------------------


def init_local_db(db_path: Path) -> sqlite3.Connection:
    """Create/open the local SSR history database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            item_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            type TEXT
        );

        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            price REAL NOT NULL,
            volume INTEGER,
            source TEXT NOT NULL DEFAULT 'ssr_history',
            FOREIGN KEY (item_id) REFERENCES items(id)
        );

        CREATE INDEX IF NOT EXISTS idx_ph_item_ts ON price_history(item_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_ph_source ON price_history(source);

        CREATE UNIQUE INDEX IF NOT EXISTS uq_price_history
            ON price_history(item_id, timestamp, source);

        CREATE TABLE IF NOT EXISTS backfill_progress (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_item_id INTEGER,
            last_item_name TEXT,
            items_completed INTEGER DEFAULT 0,
            items_failed INTEGER DEFAULT 0,
            started_at TEXT,
            updated_at TEXT
        );
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Item loading from production DB
# ---------------------------------------------------------------------------


def load_items_from_prod() -> list[dict]:
    """Load item list from the production Supabase database."""
    from database import Item, SessionLocal

    db = SessionLocal()
    try:
        items = db.query(Item.id, Item.item_id, Item.name, Item.type).all()
        return [{"id": item.id, "item_id": item.item_id, "name": item.name, "type": item.type} for item in items]
    finally:
        db.close()


def load_items_from_catalog() -> list[dict]:
    """Load item list from the local market catalog database."""
    import sqlite3

    if not CATALOG_DB_PATH.exists():
        logger.error(f"Market catalog not found at {CATALOG_DB_PATH}")
        return []

    conn = sqlite3.connect(str(CATALOG_DB_PATH))
    try:
        rows = conn.execute("SELECT id, hash_name, name, type FROM market_items ORDER BY id").fetchall()
        logger.info(f"Loaded {len(rows)} items from market catalog")
        return [{"id": row[0], "item_id": row[1], "name": row[2], "type": row[3] or ""} for row in rows]
    finally:
        conn.close()


def load_modeled_item_slugs() -> set:
    """Market_hash_names of the deep-history backbone.

    Returns items with >= MODELED_MIN_HISTORY_DAYS distinct days in the price
    archive (item_slug == market_hash_name). Empty set if the archive is missing,
    in which case --modeled-only is a no-op (caller keeps the full list).
    """
    try:
        import duckdb
    except ImportError:
        logger.warning("duckdb not installed — --modeled-only has no effect")
        return set()

    files = sorted(PRICE_ARCHIVE_DIR.glob("prices-*.parquet"))
    if not files:
        logger.warning("No price archive at %s — --modeled-only has no effect", PRICE_ARCHIVE_DIR)
        return set()

    flist = "['" + "','".join(str(f) for f in files) + "']"
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
            SELECT item_slug FROM read_parquet({flist}, union_by_name=true)
            GROUP BY item_slug
            HAVING COUNT(DISTINCT CAST(day AS DATE)) >= {MODELED_MIN_HISTORY_DAYS}
            """
        ).fetchall()
    finally:
        con.close()
    slugs = {r[0] for r in rows if r[0]}
    logger.info(
        "Modeled backbone: %s items with >= %s days of history",
        f"{len(slugs):,}",
        MODELED_MIN_HISTORY_DAYS,
    )
    return slugs


def sync_items_to_local(prod_items: list[dict], local_conn: sqlite3.Connection):
    """Copy item catalog from production to local SQLite."""
    local_conn.executemany(
        "INSERT OR IGNORE INTO items (id, item_id, name, type) VALUES (?, ?, ?, ?)",
        [(item["id"], item["item_id"], item["name"], item["type"]) for item in prod_items],
    )
    local_conn.commit()
    logger.info(f"Synced {len(prod_items)} items to local database")


# ---------------------------------------------------------------------------
# Steam API client
# ---------------------------------------------------------------------------


class SteamPriceHistoryClient:
    """Fetches full price history from Steam's authenticated endpoint."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENTS[0],
                "Referer": "https://steamcommunity.com/market/",
            }
        )
        # Attach login cookies for the authenticated pricehistory endpoint.
        if settings.steam_login_secure:
            self.session.cookies.set("steamLoginSecure", settings.steam_login_secure, domain="steamcommunity.com")
        if settings.steam_session_id:
            self.session.cookies.set("sessionid", settings.steam_session_id, domain="steamcommunity.com")
        if not settings.steam_login_secure:
            logger.warning(
                "STEAM_LOGIN_SECURE not set in .env — pricehistory requests will be unauthenticated "
                "and return empty. See config.py for how to grab the cookie."
            )
        self.last_request_time = 0.0
        self._rotate_ua()

    def _rotate_ua(self):
        import random

        ua = random.choice(USER_AGENTS)
        self.session.headers["User-Agent"] = ua

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def get_price_history(self, market_hash_name: str, max_429_retries: int = MAX_429_RETRIES):
        """
        Fetch full price history for an item.

        Returns:
            list  — [[date_str, price, volume_str], ...] (empty list = item has no history)
            None  — hard failure (non-429 error)
            RATE_LIMITED — got 429 after up to ``max_429_retries`` bounded retries. Surfaced
                (not swallowed) so the caller can count consecutive 429s and auto-pause
                instead of hammering Steam and refreshing the IP cooldown.
        """
        self._rate_limit()

        url = "https://steamcommunity.com/market/pricehistory/"
        params = {"appid": 730, "market_hash_name": market_hash_name}

        current_delay = RETRY_DELAY
        net_attempts = 0
        seen_429 = 0
        while True:
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    if seen_429 >= max_429_retries:
                        logger.warning(
                            f"Rate limited (429) on {market_hash_name} — surfacing after "
                            f"{seen_429} retr{'y' if seen_429 == 1 else 'ies'} (not hammering)"
                        )
                        return RATE_LIMITED
                    seen_429 += 1
                    logger.warning(
                        f"Rate limited (429) on {market_hash_name}, backing off {current_delay:.0f}s "
                        f"({seen_429}/{max_429_retries})"
                    )
                    time.sleep(current_delay)
                    current_delay *= BACKOFF_MULTIPLIER
                    continue

                if resp.status_code == 400:
                    # Empty array = no history or bad session
                    data = resp.json()
                    if data == []:
                        return []
                    logger.warning(f"Unexpected 400 for {market_hash_name}: {resp.text[:200]}")
                    return None

                resp.raise_for_status()
                data = resp.json()

                if data.get("success"):
                    return data.get("prices", [])
                else:
                    logger.warning(f"API returned success=false for {market_hash_name}")
                    return None

            except requests.exceptions.RequestException as e:
                net_attempts += 1
                logger.warning(f"Request failed (attempt {net_attempts}/{RETRY_ATTEMPTS}): {e}")
                if net_attempts < RETRY_ATTEMPTS:
                    time.sleep(current_delay)
                    current_delay *= BACKOFF_MULTIPLIER
                else:
                    return None

    def test_session(self) -> str:
        """Validate session cookies. Returns 'valid', 'invalid', or 'rate_limited'.

        Fails fast on 429 (max_429_retries=0) so we don't deepen an IP cooldown while
        merely checking — and so the caller can tell "cooled down" apart from "bad cookie".
        """
        try:
            result = self.get_price_history("AK-47 | Redline (Field-Tested)", max_429_retries=0)
        except Exception as e:
            logger.error(f"Session test failed: {e}")
            return "invalid"

        if result is RATE_LIMITED:
            logger.error(
                "Session test hit 429 (rate limited) — this is an IP cooldown, NOT a bad cookie. "
                "Wait for the cooldown to clear (no Steam requests) and retry."
            )
            return "rate_limited"
        if result is not None and len(result) > 0:
            logger.info(f"Session valid — got {len(result)} records for test item")
            return "valid"
        logger.error("Session invalid — empty/None response (check STEAM_LOGIN_SECURE in .env)")
        return "invalid"


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def downsample_prices(prices: list[list]) -> list[tuple[str, float, int]]:
    """
    Downsample raw price history into tiered candles.
    Input: [[date_str, price, volume_str], ...]
    Output: [(timestamp_str, avg_price, total_volume), ...]
    """
    now = datetime.now(UTC).replace(tzinfo=None)

    # Parse all records into (datetime, price, volume)
    parsed = []
    for date_str, price, volume_str in prices:
        try:
            # Steam format: "Jul 02 2014 01: +0" — strip the invalid timezone suffix
            cleaned = date_str.split(" +0")[0].strip()
            dt = datetime.strptime(cleaned, "%b %d %Y %H:")
            vol = int(volume_str) if volume_str else 0
            parsed.append((dt, float(price), vol))
        except (ValueError, TypeError):
            continue

    if not parsed:
        return []

    # Group by tier
    daily_records = []  # 0-90 days
    weekly_records = []  # 91-730 days
    monthly_records = []  # 731+ days

    for dt, price, volume in parsed:
        age_days = (now - dt).days
        if age_days <= 90:
            daily_records.append((dt, price, volume))
        elif age_days <= 730:
            weekly_records.append((dt, price, volume))
        else:
            monthly_records.append((dt, price, volume))

    result = []

    # Daily: group by date
    daily_groups = defaultdict(list)
    for dt, price, volume in daily_records:
        key = dt.strftime("%Y-%m-%d")
        daily_groups[key].append((price, volume))
    for date_str, group in sorted(daily_groups.items()):
        avg_price = sum(p for p, v in group) / len(group)
        total_vol = sum(v for p, v in group)
        result.append((date_str, round(avg_price, 3), total_vol))

    # Weekly: group by ISO week
    weekly_groups = defaultdict(list)
    for dt, price, volume in weekly_records:
        iso_year, iso_week, _ = dt.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        weekly_groups[key].append((dt, price, volume))
    for _week_key, group in sorted(weekly_groups.items()):
        # Use middle of the week as timestamp
        dates = [dt for dt, _, _ in group]
        mid_date = dates[len(dates) // 2]
        prices_vols = [(p, v) for _, p, v in group]
        avg_price = sum(p for p, v in prices_vols) / len(prices_vols)
        total_vol = sum(v for p, v in prices_vols)
        result.append((mid_date.strftime("%Y-%m-%d"), round(avg_price, 3), total_vol))

    # Monthly: group by year-month
    monthly_groups = defaultdict(list)
    for dt, price, volume in monthly_records:
        key = dt.strftime("%Y-%m")
        monthly_groups[key].append((price, volume))
    for month_key, group in sorted(monthly_groups.items()):
        avg_price = sum(p for p, v in group) / len(group)
        total_vol = sum(v for p, v in group)
        result.append((f"{month_key}-15", round(avg_price, 3), total_vol))

    return result


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def store_price_history(
    local_conn: sqlite3.Connection,
    item_local_id: int,
    candles: list[tuple[str, float, int]],
    dry_run: bool = False,
) -> int:
    """Store downsampled candles into local SQLite. Returns rows inserted."""
    if not candles:
        return 0

    rows = [(item_local_id, ts, price, vol, "ssr_history") for ts, price, vol in candles]

    if dry_run:
        return len(rows)

    local_conn.executemany(
        """INSERT OR IGNORE INTO price_history (item_id, timestamp, price, volume, source)
           VALUES (?, ?, ?, ?, ?)""",
        rows,
    )
    local_conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def load_progress(local_conn: sqlite3.Connection) -> dict:
    """Load backfill progress from local DB."""
    row = local_conn.execute(
        "SELECT last_item_id, last_item_name, items_completed, items_failed, started_at, updated_at "
        "FROM backfill_progress WHERE id = 1"
    ).fetchone()

    if row:
        return {
            "last_item_id": row[0],
            "last_item_name": row[1],
            "items_completed": row[2],
            "items_failed": row[3],
            "started_at": row[4],
            "updated_at": row[5],
        }
    return {
        "last_item_id": None,
        "last_item_name": None,
        "items_completed": 0,
        "items_failed": 0,
        "started_at": None,
        "updated_at": None,
    }


def save_progress(
    local_conn: sqlite3.Connection,
    item_id: int,
    item_name: str,
    completed: int,
    failed: int,
):
    """Save or update backfill progress."""
    now = datetime.now(UTC).replace(tzinfo=None).isoformat()
    existing = load_progress(local_conn)
    started = existing.get("started_at") or now

    local_conn.execute(
        """INSERT OR REPLACE INTO backfill_progress
           (id, last_item_id, last_item_name, items_completed, items_failed, started_at, updated_at)
           VALUES (1, ?, ?, ?, ?, ?, ?)""",
        (item_id, item_name, completed, failed, started, now),
    )
    local_conn.commit()


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------


def run_backfill(
    limit: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    max_consecutive_429: int = DEFAULT_MAX_CONSECUTIVE_429,
    max_consecutive_empty_after_ok: int = DEFAULT_MAX_CONSECUTIVE_EMPTY_AFTER_OK,
    source: str = "prod",
    modeled_only: bool = False,
):
    """Run the SSR history backfill.

    Args:
        source: "prod" for production Supabase, "catalog" for local market_catalog.db
        modeled_only: restrict to the deep-history backbone (>= MODELED_MIN_HISTORY_DAYS
            days in the archive, ~4.8K items) — the set the forecaster relies on.
    """
    logger.info("=" * 70)
    logger.info("SSR History Backfill — Starting")
    logger.info("=" * 70)

    # 0. Pick local DB path based on source
    local_db_path = CATALOG_LOCAL_DB_PATH if source == "catalog" else LOCAL_DB_PATH

    # 1. Validate session
    client = SteamPriceHistoryClient()
    status = client.test_session()
    if status == "rate_limited":
        logger.error(
            "Aborting: Steam IP is rate-limited (429). Wait for the cooldown to clear with NO "
            "Steam requests, then retry. Your cookie was not validated but is likely fine."
        )
        return
    if status != "valid":
        logger.error("Steam session is invalid. Update STEAM_SESSION_ID and STEAM_LOGIN_SECURE in .env")
        return

    # 2. Load items from source
    if source == "catalog":
        logger.info("Loading items from local market catalog...")
        prod_items = load_items_from_catalog()
    else:
        logger.info("Loading items from production database...")
        prod_items = load_items_from_prod()
    logger.info(f"Loaded {len(prod_items)} items")

    # 2b. Optional: restrict to the deep-history backbone
    if modeled_only:
        slugs = load_modeled_item_slugs()
        if slugs:
            before = len(prod_items)
            prod_items = [it for it in prod_items if it["name"] in slugs]
            logger.info(
                f"--modeled-only: {before} -> {len(prod_items)} deep-history items (>= {MODELED_MIN_HISTORY_DAYS} days)"
            )

    # 3. Initialize local DB
    local_conn = init_local_db(local_db_path)
    sync_items_to_local(prod_items, local_conn)

    # 4. Determine starting point
    progress = load_progress(local_conn)
    start_item_id = None

    if resume and progress["last_item_id"] is not None:
        start_item_id = progress["last_item_id"]
        logger.info(
            f"Resuming from item_id={start_item_id} ({progress['last_item_name']}) — "
            f"completed={progress['items_completed']}, failed={progress['items_failed']}"
        )

    # 5. Build work list
    work_items = []
    for item in prod_items:
        if start_item_id and item["id"] <= start_item_id:
            continue
        work_items.append(item)

    if limit:
        work_items = work_items[:limit]

    total = len(work_items)
    logger.info(f"Items to process: {total}")

    if total == 0:
        logger.info("Nothing to do — backfill already complete")
        print_progress_summary(local_conn)
        return

    # 6. Initialize health monitor
    health = HealthMonitor(
        max_consecutive_failures=max_consecutive_failures,
        max_consecutive_429=max_consecutive_429,
        max_consecutive_empty_after_ok=max_consecutive_empty_after_ok,
    )
    logger.info(
        f"Auto-pause thresholds: {max_consecutive_failures} consecutive failures, "
        f"{max_consecutive_429} consecutive 429s, "
        f"{max_consecutive_empty_after_ok} consecutive EMPTY after OK"
    )

    # 7. Process items
    completed = progress["items_completed"]
    failed = progress["items_failed"]
    total_rows = 0
    start_time = time.time()
    paused = False

    for idx, item in enumerate(work_items):
        item_name = item["name"]
        item_local_id = item["id"]

        # Progress logging
        if idx > 0 and idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed * 3600
            eta_hours = (total - idx) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {idx}/{total} ({idx * 100 // total}%) | "
                f"Completed: {completed} | Failed: {failed} | "
                f"Rate: {rate:.0f} items/hr | ETA: {eta_hours:.1f} hrs"
            )

        # Periodic health report
        if idx > 0 and idx % HEALTH_REPORT_INTERVAL == 0:
            elapsed = time.time() - start_time
            health.log_health_report(idx, total, elapsed)

        # Fetch price history
        try:
            prices = client.get_price_history(item_name)
        except Exception as e:
            logger.error(f"Exception fetching {item_name}: {e}")
            health.record_exception(item_name)
            failed += 1
            save_progress(local_conn, item["id"], item_name, completed, failed)
            # Check auto-pause
            pause_reason = health.should_pause()
            if pause_reason:
                logger.critical(pause_reason)
                logger.critical(
                    f"Auto-paused at item {idx + 1}/{total}. "
                    f"Progress saved. Use --resume to continue after fixing the issue."
                )
                paused = True
                break
            continue

        # Rate limited: surfaced by the client (not swallowed) so we can pause fast.
        # Deliberately do NOT save_progress here — leaving the checkpoint at the last
        # good item means --resume re-fetches these on the next run instead of skipping them.
        if prices is RATE_LIMITED:
            logger.warning(f"429: {item_name} (rate limited — not advancing progress)")
            health.record_429(item_name)
            pause_reason = health.should_pause()
            if pause_reason:
                logger.critical(pause_reason)
                logger.critical(
                    f"Auto-paused at item {idx + 1}/{total} on rate limits. Progress saved at last "
                    f"good item. Wait for the cooldown to clear, then --resume."
                )
                paused = True
                break
            continue

        if prices is None:
            # API error after retries exhausted
            # Could be 429 exhaustion or other error — we treat as failure
            # but the 429s were already logged by the client
            logger.warning(f"FAILED: {item_name}")
            health.record_failed(item_name)
            failed += 1
            save_progress(local_conn, item["id"], item_name, completed, failed)
            # Check auto-pause
            pause_reason = health.should_pause()
            if pause_reason:
                logger.critical(pause_reason)
                logger.critical(
                    f"Auto-paused at item {idx + 1}/{total}. "
                    f"Progress saved. Use --resume to continue after fixing the issue."
                )
                paused = True
                break
            continue

        if len(prices) == 0:
            # Item has no price history
            logger.info(f"EMPTY: {item_name} (no history)")
            health.record_empty(item_name)
            completed += 1
            save_progress(local_conn, item["id"], item_name, completed, failed)
            # Check auto-pause
            pause_reason = health.should_pause()
            if pause_reason:
                logger.critical(pause_reason)
                logger.critical(
                    f"Auto-paused at item {idx + 1}/{total}. "
                    f"Progress saved. Use --resume to continue after fixing the issue."
                )
                paused = True
                break
            continue

        # Downsample and store
        candles = downsample_prices(prices)
        rows_inserted = store_price_history(local_conn, item_local_id, candles, dry_run=dry_run)
        total_rows += rows_inserted
        completed += 1

        logger.info(f"OK: {item_name} — {len(prices)} raw -> {len(candles)} candles ({rows_inserted} rows)")
        health.record_ok(item_name)

        save_progress(local_conn, item["id"], item_name, completed, failed)

    # 8. Summary
    elapsed = time.time() - start_time
    health.log_final_summary(elapsed)

    logger.info("=" * 70)
    if paused:
        logger.info("Backfill PAUSED (auto-pause triggered)")
    else:
        logger.info(f"Backfill {'(DRY RUN) ' if dry_run else ''}Complete")
    logger.info(f"  Items processed: {completed + failed}")
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Duration: {elapsed / 3600:.1f} hours")
    logger.info("=" * 70)

    print_progress_summary(local_conn, db_path=local_db_path)
    local_conn.close()

    # Return pause state for programmatic use
    return paused


def print_progress_summary(local_conn: sqlite3.Connection, db_path: Path | None = None):
    """Print a summary of what's in the local database."""
    db_path = db_path or LOCAL_DB_PATH
    item_count = local_conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    price_count = local_conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
    unique_items = local_conn.execute("SELECT COUNT(DISTINCT item_id) FROM price_history").fetchone()[0]

    db_size = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0

    logger.info(f"Local DB ({db_path.name}):")
    logger.info(f"  Items: {item_count}")
    logger.info(f"  Price rows: {price_count}")
    logger.info(f"  Items with history: {unique_items}")
    logger.info(f"  DB size: {db_size:.1f} MB")


def print_status(source: str = "prod"):
    """Print current backfill status."""
    local_db_path = CATALOG_LOCAL_DB_PATH if source == "catalog" else LOCAL_DB_PATH

    if not local_db_path.exists():
        logger.info(f"No local database found at {local_db_path.name} — backfill not started")
        return

    local_conn = init_local_db(local_db_path)
    progress = load_progress(local_conn)

    logger.info("SSR Backfill Status:")
    if progress["last_item_name"]:
        logger.info(f"  Last item: {progress['last_item_name']} (id={progress['last_item_id']})")
    logger.info(f"  Completed: {progress['items_completed']}")
    logger.info(f"  Failed: {progress['items_failed']}")
    logger.info(f"  Started: {progress.get('started_at', 'N/A')}")
    logger.info(f"  Updated: {progress.get('updated_at', 'N/A')}")

    # Parse recent log entries for health metrics
    log_path = Path(__file__).parent.parent / "runtime" / "ssr_backfill.log"
    if log_path.exists():
        try:
            lines = log_path.read_text().strip().split("\n")
            # Count recent result types from last 500 log lines
            recent = lines[-500:] if len(lines) > 500 else lines
            ok_count = sum(1 for l in recent if " [INFO] OK:" in l)
            empty_count = sum(1 for l in recent if " [INFO] EMPTY:" in l)
            failed_count = sum(1 for l in recent if " [WARNING] FAILED:" in l)
            rate_429 = sum(1 for l in recent if "Rate limited (429)" in l)
            exceptions = sum(1 for l in recent if " [ERROR] Exception" in l)

            logger.info("  --- Recent Log Activity (last 500 lines) ---")
            logger.info(
                f"  OK: {ok_count} | EMPTY: {empty_count} | "
                f"Failed: {failed_count} | 429s: {rate_429} | Exceptions: {exceptions}"
            )

            # Check for auto-pause in log
            pause_lines = [l for l in lines if "Auto-paused" in l or "PAUSE:" in l]
            if pause_lines:
                logger.warning(f"  AUTO-PAUSE DETECTED: {pause_lines[-1]}")
                logger.warning("  Fix the issue, then use --resume to continue")
        except Exception as e:
            logger.info(f"  (Could not parse log for health metrics: {e})")

    print_progress_summary(local_conn, db_path=local_db_path)
    local_conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill SSR price history to local SQLite")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed item")
    parser.add_argument("--limit", type=int, help="Only process N items")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument("--status", action="store_true", help="Show current progress")
    parser.add_argument(
        "--source",
        choices=["prod", "catalog"],
        default="prod",
        help="Source of item list: prod (Supabase, 24k items) or catalog (local market_catalog.db, 32k items)",
    )
    parser.add_argument(
        "--modeled-only",
        action="store_true",
        help="Only backfill the deep-history backbone (>= 365 days in the archive, ~4.8K items) — "
        "the forecaster's core set. ~8-9h at 4s/req vs ~2 days for the full catalog.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help=f"Auto-pause after N consecutive failures (default: {DEFAULT_MAX_CONSECUTIVE_FAILURES})",
    )
    parser.add_argument(
        "--max-consecutive-429",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_429,
        help=f"Auto-pause after N consecutive 429 rate limits (default: {DEFAULT_MAX_CONSECUTIVE_429})",
    )
    parser.add_argument(
        "--max-consecutive-empty-after-ok",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_EMPTY_AFTER_OK,
        help=f"Auto-pause after N consecutive EMPTY responses following OKs (default: {DEFAULT_MAX_CONSECUTIVE_EMPTY_AFTER_OK})",
    )
    args = parser.parse_args()

    if args.status:
        print_status(source=args.source)
    else:
        run_backfill(
            limit=args.limit,
            resume=args.resume,
            dry_run=args.dry_run,
            max_consecutive_failures=args.max_consecutive_failures,
            max_consecutive_429=args.max_consecutive_429,
            max_consecutive_empty_after_ok=args.max_consecutive_empty_after_ok,
            source=args.source,
            modeled_only=args.modeled_only,
        )
