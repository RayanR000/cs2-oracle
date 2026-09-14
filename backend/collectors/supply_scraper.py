"""
Supply-side data collector.

Daily scraper that captures sell_listings from Steam Market.
Data is stored in the supply_snapshots table and consumed by
the forecaster for supply-depth features.

Source:
  - Steam Market /market/search/render/  (burst-limited, 10 items/req)
"""

import logging
import random
import time
from datetime import date

import requests
from database import SupplySnapshot, utcnow_naive
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ── Steam scraper config ────────────────────────────────────────────────
STEAM_SEARCH_URL = "https://steamcommunity.com/market/search/render/"
DEFAULT_BURST_SIZE = 20
DEFAULT_BURST_PAUSE = 30.0
MAX_RETRIES = 3
RETRY_BACKOFF = [30, 60, 120]
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


def _rotate_ua(session: requests.Session):
    session.headers["User-Agent"] = random.choice(USER_AGENTS)


def _fetch_steam_page(session: requests.Session, offset: int) -> tuple[list[dict], int] | None:
    """Fetch one page (10 items) from Steam Market.

    Returns (results_list, total_count) or None on failure.
    """
    params = {"appid": 730, "norender": 1, "start": offset, "count": 100}

    for attempt, backoff in enumerate(RETRY_BACKOFF):
        try:
            _rotate_ua(session)
            resp = session.get(STEAM_SEARCH_URL, params=params, timeout=30)

            if resp.status_code == 429:
                logger.warning(f"429 at offset={offset}, backing off {backoff}s")
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            data = resp.json()

            if data.get("results") is None:
                logger.warning(f"Null results at offset={offset}, backing off {backoff}s")
                time.sleep(backoff)
                continue

            return data["results"], data.get("total_count", 0)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed at offset={offset} (attempt {attempt + 1}): {e}")
            if attempt < len(RETRY_BACKOFF) - 1:
                time.sleep(backoff)

    return None


def get_total_item_count(session: requests.Session) -> int | None:
    """Get total number of items on the Steam Market."""
    result = _fetch_steam_page(session, 0)
    if result:
        _, total = result
        return total
    return None


class SupplyScraper:
    """Collects daily supply snapshots from Steam Market."""

    def __init__(self, db: Session, burst_size: int = DEFAULT_BURST_SIZE, burst_pause: float = DEFAULT_BURST_PAUSE):
        self.db = db
        self.burst_size = burst_size
        self.burst_pause = burst_pause
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ── Steam scrape ─────────────────────────────────────────────────

    def _load_tracked_hash_names(self) -> set[str]:
        """Load market_hash_names for items in the prediction DB.

        Uses the canonical item name column as the Steam market hash name.
        Returns a set for fast membership checks during catalog traversal.
        """
        rows = self.db.execute(
            text("""
            SELECT DISTINCT i.name
            FROM items i
            WHERE i.is_backfilled = 1
               OR EXISTS (SELECT 1 FROM price_history ph WHERE ph.item_id = i.id)
        """)
        ).fetchall()
        names = {r[0] for r in rows if r[0]}
        logger.info(f"Tracking {len(names):,} items for supply snapshots")
        return names

    def scrape_steam(self, tracked: set[str] | None = None) -> dict[str, int]:
        """Scrape sell_listings from Steam Market.

        Paginates the full catalog using burst rate limiting. Only stores
        results for items in the tracked set (or all items if None).

        Returns {market_hash_name: sell_listings} for matched items.
        """
        logger.info("─" * 50)
        logger.info("Steam Market — scraping sell_listings")
        logger.info(f"  Burst: {self.burst_size} req, then {self.burst_pause}s pause")

        if tracked is None:
            tracked = self._load_tracked_hash_names()

        # Get total count
        total = get_total_item_count(self.session)
        if total is None:
            logger.error("Could not get total item count from Steam. Aborting.")
            return {}

        logger.info(f"  Total items on market: {total:,}")

        results: dict[str, int] = {}
        current_offset = 0
        found = 0
        stats_ok = 0
        stats_429 = 0
        stats_failed = 0
        start_time = time.time()

        while current_offset < total:
            burst_start = time.time()
            burst_found = 0

            for _ in range(self.burst_size):
                if current_offset >= total:
                    break

                page = _fetch_steam_page(self.session, current_offset)
                if page is None:
                    stats_failed += 1
                    current_offset += 10
                    continue

                items, _ = page
                stats_ok += 1

                for item in items:
                    h = item.get("hash_name")
                    sl = item.get("sell_listings")
                    if h and sl is not None:
                        if h in tracked:
                            results[h] = int(sl)
                            found += 1
                            burst_found += 1

                current_offset += 10

            # Log progress after each burst
            elapsed = time.time() - start_time
            pct = min(100, current_offset * 100 // total)
            rate = current_offset / elapsed if elapsed > 0 else 0
            remaining = total - current_offset
            eta = remaining / rate if rate > 0 else 0

            logger.info(
                f"  [{current_offset:>6,}/{total:,}] ({pct:3d}%) "
                f"found:{found:,}  "
                f"rate:{rate:.0f}items/min  "
                f"ETA:{eta / 60:.0f}m"
            )

            if current_offset >= total:
                break

            # Pause between bursts
            burst_elapsed = time.time() - burst_start
            remaining_pause = max(0, self.burst_pause - burst_elapsed)
            if remaining_pause > 0:
                time.sleep(remaining_pause)

        elapsed = time.time() - start_time
        logger.info(f"  Done: {stats_ok} pages OK, {stats_429} rate-limited, {stats_failed} failed")
        logger.info(f"  Matched {found:,} tracked items in {elapsed / 60:.1f} min")
        return results

    # ── Storage ───────────────────────────────────────────────────────

    def store_snapshots(self, steam_data: dict[str, int]):
        """Write today's supply snapshots into DB + Parquet.

        Uses the item name → id mapping to resolve hash names.
        Upsert semantics: INSERT OR REPLACE on (item_id, snapshot_date).
        """
        today = date.today()

        # Build name → id lookup
        name_ids: dict[str, int] = {}
        rows = self.db.execute(text("SELECT id, name FROM items")).fetchall()
        for row in rows:
            name_ids[row[1]] = row[0]
        logger.info(f"  Name→id map: {len(name_ids):,} items")

        written = 0
        parquet_rows = []
        for h_name, listings in steam_data.items():
            item_id = name_ids.get(h_name)
            if item_id is None:
                continue
            existing = self.db.execute(
                text("SELECT 1 FROM supply_snapshots WHERE item_id = :iid AND snapshot_date = :sd"),
                {"iid": item_id, "sd": today},
            ).fetchone()
            if existing:
                self.db.execute(
                    text("""
                        UPDATE supply_snapshots
                        SET sell_listings = :sl, source = 'steam_burst', created_at = :now
                        WHERE item_id = :iid AND snapshot_date = :sd
                    """),
                    {"sl": listings, "iid": item_id, "sd": today, "now": utcnow_naive()},
                )
            else:
                snap = SupplySnapshot(
                    item_id=item_id,
                    snapshot_date=today,
                    sell_listings=listings,
                    source="steam_burst",
                )
                self.db.add(snap)
            parquet_rows.append(
                {
                    "item_id": item_id,
                    "snapshot_date": today,
                    "sell_listings": listings,
                    "skinport_quantity": None,
                    "source": "steam_burst",
                    "created_at": utcnow_naive(),
                }
            )
            written += 1

        self.db.commit()

        if parquet_rows:
            from db.parquet import append_table

            append_table("supply_snapshots", parquet_rows, ["item_id", "snapshot_date"])

        logger.info(f"  Stored {written:,} Steam snapshots")

    # ── Full run ──────────────────────────────────────────────────────

    def run(self, tracked: set[str] | None = None) -> dict:
        """Execute a full supply-scrape cycle: Steam burst → DB.

        Returns a status dict matching the pipeline convention.
        """
        start = time.time()

        try:
            tracked = tracked or self._load_tracked_hash_names()
            steam = self.scrape_steam(tracked)
            self.store_snapshots(steam)
        except Exception as e:
            logger.error(f"Supply scraper failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

        elapsed = time.time() - start
        logger.info(f"Supply scrape complete in {elapsed:.1f}s")

        return {
            "status": "success",
            "steam_items": len(steam),
            "errors": 0,
            "duration_seconds": round(elapsed, 1),
        }
