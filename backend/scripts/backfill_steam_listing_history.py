#!/usr/bin/env python3
"""
Backfill Steam daily price history from *listing pages* (no cookies required).

Unlike `backfill_ssr_history.py`, which calls the cookie-gated
`/market/pricehistory/` endpoint, this reads `/market/listings/730/<name>`.
That page embeds the same series inside its dehydrated React-Query cache, and
serves it to logged-out clients. Fields are identical: time, price_median,
purchases (daily traded volume). Volume matches `aggregator_sync` directly, but
price does NOT: the page quotes the buyer price, so it is divided by
STEAM_FEE_MULTIPLIER on ingest (see that constant).

One page often carries several variants (wear tiers, StatTrak, Souvenir) of the
same skin, so every response is harvested for *all* series it contains. That
means far fewer requests than items — the politeness win is the point.

Targets items outside the `is_backfilled` gate, i.e. those the forecaster
currently drops. Writes to a staging SQLite DB; it never touches prod or the
Parquet archive. Merging is a separate, deliberate step.

Usage:
    python scripts/backfill_steam_listing_history.py --limit 200
    python scripts/backfill_steam_listing_history.py --resume
    python scripts/backfill_steam_listing_history.py --status
    python scripts/backfill_steam_listing_history.py --min-price 1.0
"""

import argparse
import json
import logging
import random
import re
import sqlite3
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from scripts.backfill_ssr_history import HealthMonitor, USER_AGENTS

RUNTIME = Path(__file__).parent.parent / "runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)

DB_PATH = RUNTIME / "steam_listing_history.db"
PROGRESS_FILE = RUNTIME / "steam_listing_progress.json"
PRICE_ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# force=True: importing backfill_ssr_history already called basicConfig, which
# makes a second call a silent no-op — this run's log file would stay empty.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(RUNTIME / "steam_listing_backfill.log")),
    ],
    force=True,
)
logger = logging.getLogger("steam_listing")

BASE_URL = "https://steamcommunity.com/market/listings/730/"

# Deliberately slower than the measured-safe ceiling. A burst of 20 held at
# 1.75 req/s with zero 429s, but that says nothing about a multi-hour run, and
# the downside here is an IP ban that also kills the aggregator. ~0.4 req/s.
REQUEST_DELAY = 2.5
JITTER = 0.8

# A 429 means we are already in a cooldown; retrying *extends* it (same reason
# backfill_ssr_history sets MAX_429_RETRIES=1). Never retry one inline.
MAX_CONSECUTIVE_429 = 2
MAX_CONSECUTIVE_FAILURES = 6
# EMPTY is normal here (unlisted items), so the session-expiry heuristic that
# backfill_ssr_history uses does not apply. Disable it.
MAX_CONSECUTIVE_EMPTY = 10**9

COOLDOWN_ON_429 = 900  # 15 min, then re-check once before giving up

# The dehydrated cache stores each series just before the queryKey naming it.
RE_PRICES = re.compile(r'\\+"prices\\+":(\[\{.*?\}\]|\[\])')
RE_QUERYKEY = re.compile(r'\\+"queryKey\\+":\[\\+"market\\+",\\+"pricehistory\\+",730,\\+"(.*?)\\+"\]')


def _unescape(raw: str) -> str:
    """Collapse the page's variable-depth backslash escaping into plain JSON."""
    return re.sub(r'\\+"', '"', raw).replace("\\\\", "")


def extract_series(html: str) -> Dict[str, List[dict]]:
    """Pull every {market_hash_name: [{time, price_median, purchases}...]} on the page.

    Each prices array precedes the queryKey that names it, so pair each array
    with the first queryKey occurring after it.
    """
    prices = [(m.start(), m.group(1)) for m in RE_PRICES.finditer(html)]
    keys = [(m.start(), m.group(1)) for m in RE_QUERYKEY.finditer(html)]
    if not prices or not keys:
        return {}

    out: Dict[str, List[dict]] = {}
    for pos, raw in prices:
        nxt = next((name for kpos, name in keys if kpos > pos), None)
        if not nxt:
            continue
        try:
            series = json.loads(_unescape(raw))
        except json.JSONDecodeError:
            continue
        if not series:
            continue
        name = _unescape(nxt)
        # Keep the longest series if a name somehow appears twice.
        if len(series) > len(out.get(name, [])):
            out[name] = series
    return out


# The listing page serves the BUYER price (Steam's fee included); the archive's
# aggregator_sync rows store the net/seller price. Measured against 30,875
# overlapping pre-2024 rows the ratio is 1.1607 — p10 1.1565, p90 1.1656, and a
# within-item CV of 0.0021, i.e. a constant, not noise. Skipping this division
# would make every backfilled row ~16% high: the same basis mismatch that put a
# ~13-15% step in every timeline before the 2026-07-08 unification (that writeup,
# docs/historical/price-basis-swap.md, was deleted 2026-08-05 — see git history).
STEAM_FEE_MULTIPLIER = 1.1607


def to_daily_rows(name: str, series: List[dict]) -> List[Tuple[str, str, float, int]]:
    """Collapse Steam's mixed hourly/daily points into one row per UTC day.

    Recent points are hourly; older ones daily. Volume sums within a day and
    price is volume-weighted, matching how the archive stores a daily median.
    """
    buckets: Dict[str, Tuple[float, int]] = {}
    for p in series:
        try:
            day = datetime.fromtimestamp(p["time"], timezone.utc).strftime("%Y-%m-%d")
            price = float(p["price_median"])
            vol = int(p["purchases"])
        except (KeyError, TypeError, ValueError, OSError):
            continue
        if vol <= 0:
            vol = 1
        acc_p, acc_v = buckets.get(day, (0.0, 0))
        buckets[day] = (acc_p + price * vol, acc_v + vol)
    return [
        (name, d, round(pv / v / STEAM_FEE_MULTIPLIER, 4), v)
        for d, (pv, v) in sorted(buckets.items())
        if v > 0
    ]


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS price_history (
               item_name   TEXT NOT NULL,
               day         TEXT NOT NULL,
               median_price REAL NOT NULL,
               volume      INTEGER NOT NULL,
               PRIMARY KEY (item_name, day)
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_item ON price_history(item_name)")
    conn.commit()
    return conn


def store(conn: sqlite3.Connection, rows: List[Tuple[str, str, float, int]]) -> int:
    if not rows:
        return 0
    cur = conn.executemany(
        "INSERT OR REPLACE INTO price_history (item_name, day, median_price, volume) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return cur.rowcount


# Two mangled key formats exist in the archive, both duplicates of real items
# already inside the gate (see the slug-duplicates writeup). Neither is a
# market_hash_name, so both return a valid page with zero series — which looks
# exactly like a genuine miss and would poison the EMPTY rate.
#   slug   : 'sealed-graffiti-popdog-battle-green'  (migrate_historical_data.py)
#   steam_ : 'steam_sticker_|_sico_|_rio_2022'      (deleted real_data_collector)
# NOTE the slug regex does NOT match the steam_ form (it has '_' and '|'), so
# excluding slugs alone is not enough.
_SLUG_KEY = re.compile(r"^[a-z0-9][a-z0-9\-]*$")


def is_mangled_key(item_slug: str) -> bool:
    return bool(_SLUG_KEY.match(item_slug)) or item_slug.startswith("steam_")


def load_targets(
    min_price: Optional[float],
    shuffle_seed: Optional[int],
    active_days: int = 1,
) -> List[str]:
    """Items the daily aggregator is actively collecting, that training drops.

    Three conditions, all required:
      1. present on the latest `active_days` archive days — i.e. the daily
         aggregator is still writing them, so a backfill stays useful
      2. outside the `is_backfilled` gate (no pre-2026 history)
      3. a real market_hash_name, not one of the two mangled key formats
    """
    import duckdb

    con = duckdb.connect()
    glob = str(PRICE_ARCHIVE_DIR / "prices-*.parquet")
    con.execute(f"CREATE VIEW arch AS SELECT * FROM read_parquet('{glob}')")
    con.execute("CREATE TABLE gate AS SELECT DISTINCT item_slug FROM arch WHERE day < '2026-01-01'")

    recent = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT day FROM arch ORDER BY day DESC LIMIT {active_days}"
        ).fetchall()
    ]
    logger.info(f"active window = {len(recent)} most recent archive day(s): "
                f"{', '.join(str(d)[:10] for d in recent)}")
    con.execute("CREATE TABLE act(day DATE)")
    con.executemany("INSERT INTO act VALUES (?)", [(d,) for d in recent])

    price_clause = f"HAVING MAX(median_price) >= {min_price}" if min_price else ""
    rows = con.execute(
        f"""SELECT item_slug FROM arch
            WHERE day IN (SELECT day FROM act)
              AND item_slug NOT IN (SELECT item_slug FROM gate)
            GROUP BY 1 {price_clause}"""
    ).fetchall()

    names = sorted(r[0] for r in rows if not is_mangled_key(r[0]))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(names)
    return names


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"done": [], "started_at": None}


def save_progress(prog: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(prog))


# Steam does not always answer a throttle with 429. It also serves a stripped
# SSR shell: HTTP 200, no 302 to the nameid URL, ~250 KB, and zero occurrences
# of "pricehistory". A real hit redirects and embeds the cache. Treating the
# shell as "this item has no history" is how a blocked run looks healthy for
# 300 requests, so it is detected explicitly and counted as a rate limit.
SOFT_BLOCK_MAX_BYTES = 300_000


def classify(r: requests.Response) -> str:
    """OK | EMPTY | SOFT_BLOCK — distinguishes a stripped shell from a real miss."""
    html = r.text
    if "pricehistory" in html:
        return "OK"
    if not r.history and len(html) < SOFT_BLOCK_MAX_BYTES:
        return "SOFT_BLOCK"
    return "EMPTY"


def fetch(session: requests.Session, name: str) -> Tuple[str, Optional[str]]:
    """Return (status, html). status is OK | EMPTY | 429 | FAILED."""
    url = BASE_URL + urllib.parse.quote(name, safe="")
    try:
        r = session.get(url, timeout=30, headers={"User-Agent": random.choice(USER_AGENTS)})
    except requests.RequestException as exc:
        logger.warning(f"  request error for {name!r}: {type(exc).__name__}")
        return "FAILED", None
    if r.status_code == 429:
        return "429", None
    if r.status_code != 200:
        logger.warning(f"  HTTP {r.status_code} for {name!r}")
        return "FAILED", None
    verdict = classify(r)
    if verdict == "SOFT_BLOCK":
        return "429", None  # same handling path: back off, never retry inline
    return "OK", r.text


CANARY = "AK-47 | Redline (Field-Tested)"


def canary_ok(session: requests.Session) -> bool:
    """A liquid item that must always carry history. If it comes back stripped,
    we are throttled and every subsequent EMPTY would be a lie."""
    status, html = fetch(session, CANARY)
    return status == "OK" and bool(extract_series(html or ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="stop after N requests")
    ap.add_argument("--resume", action="store_true", help="skip already-fetched items")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--min-price", type=float, default=None, help="only items reaching this price")
    ap.add_argument("--delay", type=float, default=REQUEST_DELAY)
    ap.add_argument("--seed", type=int, default=None, help="shuffle targets for a representative sample")
    ap.add_argument("--active-days", type=int, default=1,
                    help="require presence on the N most recent archive days (default 1)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = init_db()

    if args.status:
        n_items, n_rows = conn.execute(
            "SELECT COUNT(DISTINCT item_name), COUNT(*) FROM price_history"
        ).fetchone()
        prog = load_progress()
        logger.info(f"stored: {n_items} items, {n_rows} rows | requested: {len(prog['done'])}")
        return 0

    targets = load_targets(args.min_price, args.seed, args.active_days)
    logger.info(f"{len(targets)} non-gated, name-keyed target items")

    prog = load_progress()
    if args.resume:
        # Skip anything already requested AND anything already harvested as a
        # variant of some other page. A page carries ~4.5 series, 3.7 of which
        # are themselves targets, so ignoring the second set would re-request
        # items we already hold and inflate the run ~3.7x.
        already = {
            r[0] for r in conn.execute("SELECT DISTINCT item_name FROM price_history")
        }
        done = set(prog["done"]) | already
        before = len(targets)
        targets = [t for t in targets if t not in done]
        logger.info(
            f"{len(targets)} remaining after resume "
            f"({before - len(targets)} skipped: {len(prog['done'])} requested, "
            f"{len(already)} already harvested)"
        )
    if args.limit:
        targets = targets[: args.limit]

    if args.dry_run:
        logger.info(f"dry run — would fetch {len(targets)} pages, first 5: {targets[:5]}")
        return 0

    health = HealthMonitor(
        max_consecutive_failures=MAX_CONSECUTIVE_FAILURES,
        max_consecutive_429=MAX_CONSECUTIVE_429,
        max_consecutive_empty_after_ok=MAX_CONSECUTIVE_EMPTY,
    )
    session = requests.Session()

    if not canary_ok(session):
        logger.error(
            f"canary {CANARY!r} returned no history before the run started — "
            "already throttled. Wait for the cooldown; do not grind it."
        )
        return 1
    logger.info("canary OK — Steam is serving hydrated pages")

    t0 = time.time()
    total_rows = 0
    # Seed from the staging DB so the in-run skip also covers earlier runs.
    harvested: set = {
        r[0] for r in conn.execute("SELECT DISTINCT item_name FROM price_history")
    }
    cooldown_used = False

    skipped_inrun = 0
    for i, name in enumerate(targets, 1):
        # A page returns the whole skin family, so a target is often already in
        # hand by the time its own turn comes up. Without this check the run
        # re-requests it — ~3.7 of every 4.5 harvested series are themselves
        # targets, so skipping is worth ~3.7x the total request count.
        if name in harvested:
            skipped_inrun += 1
            continue

        status, html = fetch(session, name)

        if status == "429":
            health.record_429(name)
            logger.warning(f"  429 on {name!r} — backing off, not retrying")
        elif status == "FAILED":
            health.record_failed(name)
        else:
            series = extract_series(html or "")
            if not series:
                health.record_empty(name)
            else:
                health.record_ok(name)
                for series_name, points in series.items():
                    rows = to_daily_rows(series_name, points)
                    total_rows += store(conn, rows)
                    harvested.add(series_name)

        prog["done"].append(name)
        save_progress(prog)

        reason = health.should_pause()
        if reason:
            logger.error(reason)
            # A single 15-min cooldown, once. If Steam is still refusing after
            # that, stop — grinding a cooldown is how an IP ban is earned.
            if health.consecutive_429 >= MAX_CONSECUTIVE_429 and not cooldown_used:
                cooldown_used = True
                logger.warning(f"  cooling down {COOLDOWN_ON_429}s before one retry")
                time.sleep(COOLDOWN_ON_429)
                health.consecutive_429 = 0
                continue
            logger.error("  aborting run — resume later with --resume")
            break

        if i % 50 == 0:
            health.log_health_report(i, len(targets), time.time() - t0)
            # Re-verify periodically: a throttle that starts mid-run turns every
            # later EMPTY into a false negative, and the item gets marked done.
            if not canary_ok(session):
                logger.error("canary lost mid-run — throttled. Aborting; resume later.")
                break
        if i < len(targets):
            time.sleep(args.delay + random.uniform(0, JITTER))

    health.log_final_summary(time.time() - t0)
    logger.info(
        f"harvested {len(harvested)} distinct items from {len(prog['done'])} requests "
        f"({skipped_inrun} targets skipped — already returned as page variants)"
    )
    logger.info(f"stored {total_rows} rows -> {DB_PATH}")

    # Zero-row guard: a green exit with nothing stored is the failure mode that
    # hid the supply scraper's death for 16 days.
    if total_rows == 0:
        logger.error("stored 0 rows — treating as failure")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
