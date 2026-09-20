#!/usr/bin/env python3
"""
CSMarketAPI Multi-Market Price History Backfill
===============================================

Databases:
  csmarketapi.db                — items catalog + sales history (heavy, checkpointed)
  csmarketapi_reference.db      — markets, currency_rates, player_counts (refreshed independently)

Per-item data (fetched in popularity order):
  /v1/sales/history/aggregate   — daily OHLCV per market, 1 request/item

Reference data (fetched once, refreshable via --refresh-ref):
  /v1/markets                   — supported markets (12 rows)
  /v1/currency_rates            — exchange rates (5 rows)
  (player_counts/history NOT fetched — panel frozen, no feature reads it)

Key rotation: up to 4 API keys (1,000 requests/month each), auto-cycles.

Usage:
    python csmarketapi_backfill.py                         # start or resume
    python csmarketapi_backfill.py --limit 50              # test first 50 items
    python csmarketapi_backfill.py --dry-run               # show queue only
    python csmarketapi_backfill.py --stats                 # show progress + quota
    python csmarketapi_backfill.py --reset                 # reset backfill checkpoint
    python csmarketapi_backfill.py --refresh-ref           # re-fetch reference data
"""

import argparse
import logging
import signal
import sqlite3
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings

# ── Paths ────────────────────────────────────────────────────────────────────
LOCAL_DB = Path(__file__).parent.parent / "runtime" / "market_catalog.db"
OUTPUT_DB = Path(__file__).parent.parent / "runtime" / "csmarketapi.db"
REF_DB = Path(__file__).parent.parent / "runtime" / "csmarketapi_reference.db"
LOG_DIR = Path(__file__).parent.parent / "runtime" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"csmarketapi_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

BASE_URL = "https://api.csmarketapi.com/v1"
MAX_REQUESTS_PER_KEY = 1000
KEY_SWITCH_THRESHOLD = 950
REQUEST_DELAY = 0.1
MAX_RETRIES = 3

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Signal handling ──────────────────────────────────────────────────────────
_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        log.warning("Second interrupt — forcing exit")
        sys.exit(1)
    _shutdown_requested = True
    log.warning("Interrupt received — finishing current item then stopping")
    signal.signal(signum, _handle_signal)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ═══════════════════════════════════════════════════════════════════════════════
# DB
# ═══════════════════════════════════════════════════════════════════════════════


def connect_db(path: str, readonly: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro" if readonly else path, uri=readonly)
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_output_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            market_hash_name TEXT PRIMARY KEY,
            hash_name TEXT NOT NULL,
            nameid INTEGER,
            classid TEXT,
            exterior TEXT,
            category TEXT,
            weapon TEXT,
            quality TEXT,
            type TEXT,
            sticker_type TEXT,
            collection TEXT,
            sell_listings INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sales_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_hash_name TEXT NOT NULL,
            day TEXT NOT NULL,
            market TEXT NOT NULL,
            mean_price REAL,
            min_price REAL,
            max_price REAL,
            median_price REAL,
            volume INTEGER,
            UNIQUE(market_hash_name, day, market)
        );
        CREATE INDEX IF NOT EXISTS idx_sales_hash ON sales_history(market_hash_name);
        CREATE INDEX IF NOT EXISTS idx_sales_day  ON sales_history(day);

        CREATE TABLE IF NOT EXISTS backfill_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS failed_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash_name TEXT NOT NULL,
            reason TEXT NOT NULL,
            key_account TEXT,
            attempted_at TEXT NOT NULL,
            listings INTEGER DEFAULT 0
        );
    """)

    for k, v in {"last_hash_name": "", "total_attempted": "0", "total_completed": "0", "total_failed": "0"}.items():
        conn.execute("INSERT OR IGNORE INTO backfill_state (key, value) VALUES (?, ?)", (k, v))

    for i in range(len(settings.csmarketapi_keys)):
        conn.execute("INSERT OR IGNORE INTO backfill_state (key, value) VALUES (?, '0')", (f"req_idx_{i}",))
    conn.execute("INSERT OR IGNORE INTO backfill_state (key, value) VALUES ('active_key_idx', '0')")
    conn.commit()


def init_ref_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            market TEXT PRIMARY KEY,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS currency_rates (
            currency TEXT PRIMARY KEY,
            rate REAL,
            currency_name TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS player_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            players INTEGER,
            UNIQUE(timestamp)
        );
    """)
    conn.commit()


def gv(conn: sqlite3.Connection, key: str) -> str:
    r = conn.execute("SELECT value FROM backfill_state WHERE key = ?", (key,)).fetchone()
    return r[0] if r else ""


def sv(conn: sqlite3.Connection, key: str, value: str):
    conn.execute("REPLACE INTO backfill_state (key, value) VALUES (?, ?)", (key, value))


def inc(conn: sqlite3.Connection, key: str, n: int = 1):
    sv(conn, key, str(int(gv(conn, key) or "0") + n))


# ═══════════════════════════════════════════════════════════════════════════════
# API client
# ═══════════════════════════════════════════════════════════════════════════════


def api_get(endpoint: str, api_key: str, params: dict | None = None) -> dict | list | None:
    url = f"{BASE_URL}{endpoint}"
    query = dict(params or {})
    query["key"] = api_key
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = requests.get(url, params=query, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                log.warning(f"  429 on key …{api_key[-8:]} (attempt {attempt})")
                return None
            if resp.status_code == 404:
                return []
            if resp.status_code >= 500:
                log.warning(f"  {resp.status_code} server error (attempt {attempt}/{MAX_RETRIES})")
                if attempt <= MAX_RETRIES:
                    time.sleep(2**attempt)
                    continue
                return None
            log.warning(f"  Unexpected {resp.status_code}: {resp.text[:200]}")
            return None
        except requests.RequestException as e:
            log.warning(f"  Request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt <= MAX_RETRIES:
                time.sleep(2**attempt)
                continue
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference data (markets, currency_rates)
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_reference_data(apikey: str):
    log.info("─" * 50)
    log.info("Fetching reference data from CSMarketAPI …")
    ref_conn = connect_db(str(REF_DB))
    init_ref_db(ref_conn)

    # 1) Markets
    data = api_get("/markets", apikey)
    if isinstance(data, list):
        ref_conn.execute("DELETE FROM markets")
        ref_conn.executemany(
            "INSERT OR REPLACE INTO markets (market, description) VALUES (?, ?)",
            [(m.get("market", ""), m.get("description", "")) for m in data],
        )
        ref_conn.commit()
        log.info(f"  Markets:        1 request → {len(data)} markets stored")

    # 2) Currency rates
    data = api_get("/currency_rates", apikey)
    now = datetime.now(UTC).isoformat()
    if isinstance(data, list):
        ref_conn.execute("DELETE FROM currency_rates")
        ref_conn.executemany(
            "INSERT OR REPLACE INTO currency_rates (currency, rate, currency_name, updated_at) VALUES (?, ?, ?, ?)",
            [(r.get("currency_code", ""), r.get("rate"), r.get("currency_name", ""), now) for r in data],
        )
        ref_conn.commit()
        log.info(f"  Currency rates: 1 request → {len(data)} currencies stored")
    elif isinstance(data, dict):
        ref_conn.execute("DELETE FROM currency_rates")
        ref_conn.executemany(
            "INSERT OR REPLACE INTO currency_rates (currency, rate, updated_at) VALUES (?, ?, ?)",
            [(k, v, now) for k, v in data.items()],
        )
        ref_conn.commit()
        log.info(f"  Currency rates: 1 request → {len(data)} currencies stored")

    # Player counts are NOT refreshed: the panel is frozen and no feature
    # reads it (REFUTED — see the forecaster cleanup 2026-09-15). Skipping
    # saves one quota request per refresh.

    ref_conn.close()
    log.info("Reference data complete (2 requests)")


# ═══════════════════════════════════════════════════════════════════════════════
# Key rotation
# ═══════════════════════════════════════════════════════════════════════════════


def req_count(conn: sqlite3.Connection, idx: int) -> int:
    return int(gv(conn, f"req_idx_{idx}") or "0")


def find_key(conn: sqlite3.Connection) -> int | None:
    keys = settings.csmarketapi_keys
    if not keys:
        return None
    start = int(gv(conn, "active_key_idx") or "0")
    for off in range(len(keys)):
        idx = (start + off) % len(keys)
        if req_count(conn, idx) < KEY_SWITCH_THRESHOLD:
            return idx
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Queue builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_queue(
    local_conn: sqlite3.Connection, api_items: set[str], out_conn: sqlite3.Connection | None = None
) -> list[tuple[str, int]]:
    c = local_conn.execute("SELECT hash_name, sell_listings FROM market_items ORDER BY sell_listings DESC")
    queue = [(r[0], r[1] or 0) for r in c.fetchall() if r[0]]

    local_names = {h for h, _ in queue}
    missing = [h for h in api_items - local_names if h]

    if missing:
        if out_conn is None:
            queue.extend((h, -1) for h in missing)
            log.info(f"  {len(missing)} items without listing data (no out_conn provided)")
        else:
            lookup = {}
            for row in out_conn.execute(
                "SELECT market_hash_name, sell_listings FROM items WHERE market_hash_name IN ({})".format(
                    ",".join("?" for _ in missing)
                ),
                missing,
            ):
                lookup[row[0]] = row[1] or 0
            queue.extend((h, lookup.get(h, 0)) for h in missing)
            log.info(f"  {len(missing)} items resolved from CSMarketAPI catalog")

    queue.sort(key=lambda x: -x[1])

    tiers = Counter()
    for _, sl in queue:
        if sl < 0:
            tiers["unknown"] += 1
        elif sl == 0:
            tiers["0"] += 1
        elif sl < 10:
            tiers["1-9"] += 1
        elif sl < 100:
            tiers["10-99"] += 1
        elif sl < 1000:
            tiers["100-999"] += 1
        else:
            tiers["1000+"] += 1

    log.info(f"  Queue: {len(queue)} items total")
    for t in ["1000+", "100-999", "10-99", "1-9", "0", "unknown"]:
        if tiers[t]:
            log.info(f"    {t:>8} : {tiers[t]:,}")
    return queue


# ═══════════════════════════════════════════════════════════════════════════════
# Main backfill
# ═══════════════════════════════════════════════════════════════════════════════


def run_backfill(dry_run: bool = False, limit: int = 0, only_catalog: bool = False):
    global _shutdown_requested

    log.info("=" * 60)
    log.info(f"CSMarketAPI Backfill  —  started at {datetime.now().isoformat()}")
    log.info(f"  Local catalog:    {LOCAL_DB}")
    log.info(f"  Output DB:        {OUTPUT_DB}")
    log.info(f"  Reference DB:     {REF_DB}")
    log.info(f"  Log file:         {LOG_FILE}")
    log.info(f"  API keys:         {len(settings.csmarketapi_keys)}")
    for c in settings.csmarketapi_keys:
        log.info(f"    {c['account']:<20}  1,000 req/mo")
    if limit:
        log.info(f"  Limit:            {limit} items (test mode)")
    if only_catalog:
        log.info(f"  Mode:             only-catalog (ignoring API items table)")

    if not settings.csmarketapi_keys:
        log.error("No API keys in .env. Set CSMARKETAPI_KEY_N / CSMARKETAPI_ACCOUNT_N")
        sys.exit(1)

    local_conn = connect_db(str(LOCAL_DB), readonly=True)
    out_conn = connect_db(str(OUTPUT_DB))
    init_output_db(out_conn)

    # ── Dry-run ──────────────────────────────────────────────────────────────
    if dry_run:
        log.info("─" * 50)
        log.info("[DRY-RUN] Building queue from local DB only (no catalog fetch)…")
        queue = build_queue(local_conn, set(), out_conn)
        log.info("[DRY-RUN] Top 25 items:")
        for h, _ in queue[:25]:
            log.info(f"  {h}")
        log.info(f"[DRY-RUN] … and {max(0, len(queue) - 25)} more")
        return

    # ── Find first key ──────────────────────────────────────────────────────
    key_idx = find_key(out_conn)
    if key_idx is None:
        log.error("All keys exhausted at start. Wait for monthly reset or add keys.")
        sys.exit(1)
    sv(out_conn, "active_key_idx", str(key_idx))
    first_cred = settings.csmarketapi_keys[key_idx]

    # ── Reference data ──────────────────────────────────────────────────────
    if not REF_DB.exists():
        fetch_reference_data(first_cred["key"])
        inc(out_conn, f"req_idx_{key_idx}", 3)
        log.info(f"  (3 reference requests charged to key {key_idx})")
    else:
        log.info(f"Reference DB exists ({REF_DB.name}) — skipping reference fetch")

    if _shutdown_requested:
        log.info("Shutdown after reference data — items not started")

    # ── Catalog ──────────────────────────────────────────────────────────────
    c = out_conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    if c == 0:
        log.info("─" * 50)
        log.info("Fetching /v1/items/ catalog …")
        catalog = api_get("/items/", first_cred["key"])
        if not isinstance(catalog, list):
            log.error("Catalog fetch failed — aborting")
            sys.exit(1)

        inc(out_conn, f"req_idx_{key_idx}")
        log.info(f"  Catalog returned {len(catalog)} items (1 request)")

        inserts = []
        for i in catalog:
            inserts.append(
                (
                    i.get("market_hash_name", ""),
                    i.get("hash_name", ""),
                    i.get("nameid"),
                    i.get("classid"),
                    i.get("exterior"),
                    i.get("category"),
                    i.get("weapon"),
                    i.get("quality"),
                    i.get("type"),
                    i.get("sticker_type"),
                    i.get("sticker_collection") or i.get("collection"),
                )
            )
        out_conn.executemany(
            """INSERT OR REPLACE INTO items
               (market_hash_name, hash_name, nameid, classid, exterior,
                category, weapon, quality, type, sticker_type, collection)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            inserts,
        )

        listings = local_conn.execute("SELECT hash_name, sell_listings FROM market_items").fetchall()
        out_conn.executemany(
            "UPDATE items SET sell_listings = ? WHERE hash_name = ?",
            [(sl or 0, h) for h, sl in listings],
        )
        out_conn.commit()
        log.info(f"  Catalog stored — {len(catalog)} items in DB")
    else:
        log.info(f"  Items table already has {c:,} rows — skipping catalog fetch")

    if only_catalog:
        api_names = set()
    else:
        api_names = {r[0] for r in out_conn.execute("SELECT market_hash_name FROM items").fetchall() if r[0]}

    # ── Build queue ─────────────────────────────────────────────────────────
    log.info("─" * 50)
    log.info("Building priority queue …")
    queue = build_queue(local_conn, api_names, out_conn)
    if not queue:
        log.warning("Empty queue.")
        return

    # ── Resume point ────────────────────────────────────────────────────────
    last_hash = gv(out_conn, "last_hash_name")
    resume_idx = 0
    if last_hash:
        names_only = [h for h, _ in queue]
        try:
            resume_idx = names_only.index(last_hash) + 1
            log.info(f"Resume: skipping to index {resume_idx}/{len(queue)} (after '{last_hash}')")
        except ValueError:
            log.info(f"Resume: checkpoint '{last_hash}' not found — starting from beginning")

    remaining = queue[resume_idx:]

    if limit:
        remaining = remaining[:limit]
        queue = queue[: resume_idx + limit]

    log.info(f"  Items remaining this session: {len(remaining)}")
    completed_before = int(gv(out_conn, "total_completed") or "0")
    failed_before = int(gv(out_conn, "total_failed") or "0")
    log.info(f"  Stats from previous runs: {completed_before} completed, {failed_before} failed")

    # ── Main loop ───────────────────────────────────────────────────────────
    log.info("─" * 50)
    log.info("Starting per-item fetch …")

    start_time = time.time()
    t0 = start_time

    for i, (hash_name, listings) in enumerate(remaining):
        if _shutdown_requested:
            log.warning("Shutdown requested — stopping after current item")
            break

        item_num = resume_idx + i + 1
        total = len(queue)

        r = out_conn.execute("SELECT COUNT(*) FROM sales_history WHERE market_hash_name = ?", (hash_name,))
        if r.fetchone()[0] > 0:
            inc(out_conn, "total_completed")
            sv(out_conn, "last_hash_name", hash_name)
            log.info(
                f"[{item_num:,}/{total:,}] ({100 * item_num // total:3d}%) ⏭️  {hash_name}  — already in DB (skipping)"
            )
            continue

        key_idx = find_key(out_conn)
        if key_idx is None:
            log.warning("─" * 50)
            log.warning("ALL KEYS EXHAUSTED — pausing backfill")
            log.warning(f"  Completed: {gv(out_conn, 'total_completed')}  Failed: {gv(out_conn, 'total_failed')}")
            log.warning(f"  Next item: {hash_name}")
            break

        sv(out_conn, "active_key_idx", str(key_idx))
        cred = settings.csmarketapi_keys[key_idx]

        elapsed = time.time() - start_time
        done_so_far = i
        rate = done_so_far / elapsed if elapsed > 0 else 0
        remaining_items = len(remaining) - i
        eta_secs = remaining_items / rate if rate > 0 else 0
        quota_left = KEY_SWITCH_THRESHOLD - req_count(out_conn, key_idx)

        pct = 100 * item_num // total
        tier = f"[{listings:>5} listings]" if listings >= 0 else "[   ? listings]"

        log.info(f"[{item_num:,}/{total:,}] ({pct:3d}%) {hash_name}")
        log.info(
            f"       via {cred['account']:<15}  "
            f"quota:{quota_left:>3}/{KEY_SWITCH_THRESHOLD}  "
            f"rate:{rate:.2f}it/s  ETA:{eta_secs / 60:.0f}m  {tier}"
        )

        inc(out_conn, "total_attempted")
        sales_data = api_get(
            "/sales/history/aggregate", cred["key"], params={"market_hash_name": hash_name, "currency": "USD"}
        )
        inc(out_conn, f"req_idx_{key_idx}")

        if sales_data is None:
            log.warning("       FAILED — rotating key")
            out_conn.execute(
                "INSERT INTO failed_items (hash_name, reason, key_account, attempted_at, listings) VALUES (?, ?, ?, ?, ?)",
                (hash_name, "429 / exhausted", cred["account"], datetime.now().isoformat(), listings),
            )
            out_conn.commit()
            inc(out_conn, "total_failed")
            sv(out_conn, "active_key_idx", str((key_idx + 1) % len(settings.csmarketapi_keys)))
            continue

        out_conn.execute("DELETE FROM sales_history WHERE market_hash_name = ?", (hash_name,))

        market_rows = Counter()
        total_rows = 0
        rows = []
        for day_entry in sales_data:
            day = day_entry.get("day", "")
            for sale in day_entry.get("sales", []):
                mkt = sale.get("market", "UNKNOWN")
                market_rows[mkt] += 1
                rows.append(
                    (
                        hash_name,
                        day,
                        mkt,
                        sale.get("mean_price"),
                        sale.get("min_price"),
                        sale.get("max_price"),
                        sale.get("median_price"),
                        sale.get("volume"),
                    )
                )
                total_rows += 1
        if rows:
            out_conn.executemany(
                """INSERT OR IGNORE INTO sales_history
                   (market_hash_name, day, market, mean_price, min_price,
                    max_price, median_price, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        out_conn.commit()

        inc(out_conn, "total_completed")
        sv(out_conn, "last_hash_name", hash_name)

        total_done = int(gv(out_conn, "total_completed") or "0")
        total_fail = int(gv(out_conn, "total_failed") or "0")

        market_summary = ", ".join(f"{m}:{n}" for m, n in market_rows.most_common(5))
        extra_markets = len(market_rows) - 5
        if extra_markets > 0:
            market_summary += f" …+{extra_markets}"

        log.info(f"       ✓ {total_rows:>5} rows  [{market_summary}]  ({total_done} done, {total_fail} failed)")

        if _shutdown_requested:
            break

        time.sleep(REQUEST_DELAY)

    # ── Final summary ───────────────────────────────────────────────────────
    out_conn.commit()  # flush trailing sv()/inc() state (sv no longer commits per call)
    elapsed = time.time() - t0
    final_done = gv(out_conn, "total_completed")
    final_fail = gv(out_conn, "total_failed")
    final_last = gv(out_conn, "last_hash_name")

    log.info("=" * 60)
    log.info("Backfill session finished")
    log.info(f"  Session runtime:     {elapsed / 60:.1f}m")
    log.info(f"  Total completed:     {final_done}")
    log.info(f"  Total failed:        {final_fail}")
    log.info(f"  Checkpoint:          {final_last}")
    for i, cred in enumerate(settings.csmarketapi_keys):
        r = req_count(out_conn, i)
        pct = 100 * r / MAX_REQUESTS_PER_KEY
        bar = "█" * (r // 10) + "░" * ((MAX_REQUESTS_PER_KEY - r) // 10)
        log.info(f"  Key {i} ({cred['account']:<15}): {r:>4}/{MAX_REQUESTS_PER_KEY} ({pct:5.1f}%) {bar}")
    log.info("=" * 60)

    out_conn.close()
    local_conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════════


def show_stats():
    if not OUTPUT_DB.exists():
        log.info("No backfill database found — run the backfill first.")
        return

    conn = connect_db(str(OUTPUT_DB))
    items_total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    items_hist = conn.execute("SELECT COUNT(DISTINCT market_hash_name) FROM sales_history").fetchone()[0]
    price_rows = conn.execute("SELECT COUNT(*) FROM sales_history").fetchone()[0]
    days = conn.execute("SELECT COUNT(DISTINCT day) FROM sales_history").fetchone()[0]
    markets_list = [r[0] for r in conn.execute("SELECT DISTINCT market FROM sales_history ORDER BY market").fetchall()]
    completed = gv(conn, "total_completed")
    failed = gv(conn, "total_failed")
    last_item = gv(conn, "last_hash_name")
    conn.close()

    # Reference DB stats
    ref_exists = REF_DB.exists()
    ref_markets = ref_currencies = ref_pcount = 0
    if ref_exists:
        ref = connect_db(str(REF_DB))
        ref_markets = ref.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
        ref_currencies = ref.execute("SELECT COUNT(*) FROM currency_rates").fetchone()[0]
        ref_pcount = ref.execute("SELECT COUNT(*) FROM player_counts").fetchone()[0]
        ref.close()

    mkt_dist = {}
    if items_hist:
        conn2 = connect_db(str(OUTPUT_DB))
        for r in conn2.execute(
            "SELECT market, COUNT(*) as c FROM sales_history GROUP BY market ORDER BY c DESC"
        ).fetchall():
            mkt_dist[r[0]] = r[1]
        conn2.close()

    print(f"\n{' CSMarketAPI Backfill Stats ':=^66}\n")
    print(f"  ── Backfill DB ({OUTPUT_DB.name}) ──")
    print(f"  Items in catalog:      {items_total:>10,}")
    print(f"  Items with history:    {items_hist:>10,}")
    print(f"  Total price rows:      {price_rows:>10,}")
    print(f"  Days of data:          {days:>10,}")
    print(f"  Markets with data:     {len(markets_list)}")
    print(f"  Completed:             {completed:>10}")
    print(f"  Failed:                {failed:>10}")
    print(f"  Last item:             {last_item}")
    print()
    print(f"  ── Reference DB ({REF_DB.name}) ──")
    print(f"  Markets:        {ref_markets:>6,}" if ref_exists else "  (not yet fetched)")
    print(f"  Currency rates: {ref_currencies:>6,}" if ref_exists else "")
    print(f"  Player counts:  {ref_pcount:>6,}" if ref_exists else "")

    if mkt_dist:
        print("\n  ── Per-market price rows ──")
        print(f"  {'Market':<20} {'Rows':>10} {'%':>7}")
        print(f"  {'─' * 20} {'─' * 10} {'─' * 7}")
        for mkt, cnt in sorted(mkt_dist.items(), key=lambda x: -x[1]):
            print(f"  {mkt:<20} {cnt:>10,} {100 * cnt / price_rows:>6.1f}%")

    # Failed items
    conn3 = connect_db(str(OUTPUT_DB))
    conn3.execute("""
        CREATE TABLE IF NOT EXISTS failed_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash_name TEXT NOT NULL,
            reason TEXT NOT NULL,
            key_account TEXT,
            attempted_at TEXT NOT NULL,
            listings INTEGER DEFAULT 0
        )
    """)
    fail_rows = conn3.execute(
        "SELECT hash_name, reason, key_account, attempted_at FROM failed_items ORDER BY id DESC LIMIT 10"
    ).fetchall()
    fail_count = conn3.execute("SELECT COUNT(*) FROM failed_items").fetchone()[0]
    conn3.close()
    if fail_rows:
        print("\n  ── Recent failures (last 10) ──")
        print(f"  {'Item':<45} {'Reason':<20} {'Key':<18} {'Time':<19}")
        print(f"  {'─' * 45} {'─' * 20} {'─' * 18} {'─' * 19}")
        for h, r, k, t in fail_rows:
            print(f"  {h:<45} {r:<20} {k:<18} {t[:19]}")
        if fail_count > 10:
            print(f"  … and {fail_count - 10} more")

    print("\n  ── Key quota ──")
    for i, cred in enumerate(settings.csmarketapi_keys):
        conn3 = connect_db(str(OUTPUT_DB))
        r = req_count(conn3, i)
        conn3.close()
        pct = 100 * r / MAX_REQUESTS_PER_KEY
        bar = "█" * (r // 10) + "░" * ((MAX_REQUESTS_PER_KEY - r) // 10)
        print(f"  {cred['account']:<18} {bar} {r:>4}/{MAX_REQUESTS_PER_KEY} ({pct:5.1f}%)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Reset
# ═══════════════════════════════════════════════════════════════════════════════


def reset_state():
    if not OUTPUT_DB.exists():
        log.info("No database to reset.")
        return
    log.warning("Resetting backfill checkpoint (data preserved, progress reset)...")
    conn = connect_db(str(OUTPUT_DB))
    conn.execute("DELETE FROM backfill_state")
    init_output_db(conn)
    log.info("Done. Sales history data is intact. Next run starts from beginning.")
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="CSMarketAPI Multi-Market Price History Backfill")
    parser.add_argument("--dry-run", action="store_true", help="Show queue without API calls")
    parser.add_argument("--stats", action="store_true", help="Show progress, quota, reference data")
    parser.add_argument(
        "--reset", action="store_true", help="Reset backfill checkpoint (data kept, restart from beginning)"
    )
    parser.add_argument(
        "--refresh-ref", action="store_true", help="Re-fetch reference data (markets, currency_rates)"
    )
    parser.add_argument("--limit", type=int, default=0, help="Max items to fetch this session (for testing)")
    parser.add_argument("--only-catalog", action="store_true", help="Fetch only items in market_catalog.db, ignore API catalog")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.reset:
        reset_state()
    elif args.refresh_ref:
        if not settings.csmarketapi_keys:
            log.error("No API keys configured.")
            sys.exit(1)
        log.info("Using key 0 (skrup.chezz) for reference data…")
        fetch_reference_data(settings.csmarketapi_keys[0]["key"])
        log.info("Reference data refreshed. These 3 requests are NOT charged to backfill quota.")
    else:
        run_backfill(dry_run=args.dry_run, limit=args.limit, only_catalog=args.only_catalog)


if __name__ == "__main__":
    main()
