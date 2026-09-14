#!/usr/bin/env python3
"""Ingest a date-level CS2 event calendar into price-archive/event-calendar.parquet.

Every feed this project has ingested and refuted -- ByMykel metadata, the CSFloat
basis, the six price primitives, float composition, Reddit sentiment -- is keyed on
the *item*. `docs/changelog/2026-08-06-*` records the consequence: demeaning labels
by the market factor drops accuracy below a constant call at every horizon, and a
constant always-down call beats the model on every stored date. The variance the
model has left to explain is a *date-level* common factor, and nothing in the
archive is keyed on the date alone. This table is.

**No lift is claimed.** `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` still
excludes everything here, the served model is untouched, and the MDE gate
(`scripts/compute_mde.py`, then a *permutation* A/B) has not been run for a
date-level feature. Note that a date-level column's effective N is the number of
distinct dates, not the number of rows, so the existing per-item MDE figures
(1.15pp at 3d .. 7.13pp at 30d) do **not** transfer -- recompute before believing
any result off this table.

## What is in here, and what is deliberately not

Two source families, both exact:

- **Valve announcements** -- `ISteamNews/GetNewsForApp` with `feed_type == 1`.
  Structural only: an official post happened on day D, and how many. No text
  classification is applied, for the reason below.
- **Crate issuance** -- `first_sale_date` from ByMykel `crates.json`, split by
  crate type, plus `release_date` from `collections.json`.

A **case-release-day flag is deliberately absent.** Both candidate sources were
measured on 2026-08-06 and neither survived:

- Regexing case names out of announcement text finds 14 of the 42 known cases
  (33% recall) and misfires on drop-pool *removals* ("Removed Gallery Case") and
  on incidental prose ("Fixed a case where...").
- ByMykel `first_sale_date` has 42/42 coverage but disagrees with the first news
  mention on **all 14** overlapping cases -- median 38 days apart, range -76 to
  +1878. The two fields are not the same quantity and neither is verifiably the
  day a case became purchasable.

So crate dates ship under their source's own name (`crate_first_sale_*`), not as
"release day", and the disagreement is left for the archive to arbitrate rather
than papered over with a classifier that did not validate.

## Causality

Every column is backward-looking as of its own `day`: counts are trailing windows
that include `day`, and `days_since_*` looks only at events on or before `day`.
There is no `days_until_next_*` column -- a case release is knowable in advance
only if it was announced in advance, and this table does not track announcements
separately from the events they describe.
`tests/test_event_calendar.py::test_truncating_the_future_changes_nothing` is the
invariant.

Usage:
    python scripts/ingest_steam_news.py                  # fetch + build
    python scripts/ingest_steam_news.py --offline        # use cached dumps
    python scripts/ingest_steam_news.py --coverage-only  # report, no write
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_steam_news")

NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
BYMYKEL_URL = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en"
APPID = 730

CACHE_DIR = Path(__file__).parent.parent / "runtime" / "steamnews"
PRICE_ARCHIVE = Path(__file__).parent.parent.parent / "price-archive"
OUTPUT_PARQUET = PRICE_ARCHIVE / "event-calendar.parquet"

# The per-*event* companion to the date panel above. `event-calendar.parquet` is
# deliberately structural -- one row per day, counts only, no identity -- which
# makes it a feature table and useless to anything that needs to point at a
# specific announcement. `event_correlation_analysis.py` needs exactly that:
# it measures a price window around one event, and `event_impacts.event_id` is a
# real FK to `events.id`.
#
# `fetch_news` already caches the full items and throws all of this away, so the
# rows cost nothing extra to emit. Structural fields only -- gid, title, day,
# feed_type -- and no classification, for the reason in the module docstring:
# both candidate case-release classifiers were measured on 2026-08-06 and
# neither validated (33% recall from text; 14/14 disagreement from ByMykel).
EVENTS_PARQUET = PRICE_ARCHIVE / "event-news.parquet"

EVENT_COLUMNS = ("gid", "day", "published_at", "feed_type", "is_valve", "title", "url")

# `count` is capped server-side well below anything useful in one shot, and the
# 500 it returns is what `data-sources.md` recorded as the feed's whole depth
# ("500 entries back to 2022-03-01"). It is not -- passing `enddate` walks
# backwards, and four pages reach 2012-03-16 for 1,752 unique items in ~2s.
PAGE_SIZE = 500
MAX_PAGES = 25
PAGE_DELAY_S = 0.3

# The calendar is dense over this span: one row per day, event or not, so a join
# from a price row never misses. Start bounds the CS:GO market itself.
CALENDAR_START = date(2013, 1, 1)

# Crate families worth separating. Weapon cases are the classic supply shock;
# capsules are tournament-linked and far more frequent. Everything else in
# crates.json (Souvenir packages, highlights, pins, graffiti) is either
# undated at source or too rare to carry a window.
CASE_TYPES = ("Case",)
CAPSULE_TYPES = ("Sticker Capsule", "Autograph Capsule", "Patch Capsule")

TRAILING_WINDOWS = (7, 30)
ISSUANCE_WINDOW = 365

OUTPUT_COLUMNS = (
    "day",
    "valve_announcements",
    "valve_announcements_7d",
    "valve_announcements_30d",
    "days_since_valve_announcement",
    "press_articles",
    "press_articles_7d",
    "press_articles_30d",
    "crate_case_first_sales",
    "crate_case_first_sales_365d",
    "days_since_crate_case_first_sale",
    "crate_capsule_first_sales",
    "crate_capsule_first_sales_30d",
    "days_since_crate_capsule_first_sale",
    "collection_releases",
    "days_since_collection_release",
)

_DATE_RE = re.compile(r"^\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s*$")


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def normalise_date(raw) -> date | None:
    """Parse ByMykel's date strings, which mix separators and zero-padding.

    Same shape problem `ingest_bymykel_metadata.py::normalise_date` documents --
    `crates.json` alone carries `2024-01-16`, `2013/12/17` and `2014/5/2`.
    Anything unparseable returns None rather than being guessed at.
    """
    if not isinstance(raw, str):
        return None
    m = _DATE_RE.match(raw)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def fetch_news(cache_dir: Path, offline: bool = False, session: requests.Session | None = None) -> list[dict]:
    """Page the news feed backwards via `enddate` until it stops yielding new gids.

    Dedupe is on `gid` first, then on `(day, title)`: Steam cross-posts the same
    announcement under several gids, and counting those separately would inflate
    every announcement-count column on exactly the days that matter most.
    """
    cache = cache_dir / "newsitems.json"
    if offline:
        if not cache.exists():
            raise FileNotFoundError(f"--offline but no cache at {cache}. Run once without it.")
        items = json.loads(cache.read_text())
        logger.info(f"Loaded {len(items):,} cached news items from {cache}")
        return items

    session = session or requests.Session()
    by_gid: dict[str, dict] = {}
    end: int | None = None

    for page in range(1, MAX_PAGES + 1):
        params = {"appid": APPID, "count": PAGE_SIZE}
        if end is not None:
            params["enddate"] = end
        resp = session.get(NEWS_URL, params=params, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("appnews", {}).get("newsitems", [])
        if not items:
            break
        new = sum(1 for i in items if i["gid"] not in by_gid)
        for i in items:
            by_gid.setdefault(i["gid"], i)
        oldest = min(i["date"] for i in items)
        logger.info(
            f"  page {page}: {len(items)} items, {new} new, oldest {datetime.fromtimestamp(oldest, UTC).date()}"
        )
        if new == 0:
            break
        end = oldest - 1
        time.sleep(PAGE_DELAY_S)
    else:
        logger.warning(f"Stopped at the {MAX_PAGES}-page ceiling; feed may go deeper")

    items = list(by_gid.values())
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items))
    logger.info(f"Fetched {len(items):,} unique news items -> {cache}")
    return items


def fetch_crates(cache_dir: Path, offline: bool = False, session: requests.Session | None = None) -> dict[str, list]:
    """Fetch crates.json and collections.json, the dated-issuance half."""
    out = {}
    session = session or requests.Session()
    for name in ("crates", "collections"):
        cache = cache_dir / f"{name}.json"
        if offline:
            if not cache.exists():
                raise FileNotFoundError(f"--offline but no cache at {cache}")
            data = json.loads(cache.read_text())
        else:
            resp = session.get(f"{BYMYKEL_URL}/{name}.json", timeout=60)
            resp.raise_for_status()
            data = resp.json()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        if isinstance(data, dict):
            data = list(data.values())
        out[name] = data
        logger.info(f"  {name}.json: {len(data):,} entries")
    return out


# --------------------------------------------------------------------------
# Event extraction
# --------------------------------------------------------------------------


def news_events(items: list[dict]) -> pd.DataFrame:
    """Collapse news items to per-day counts of Valve posts and press articles.

    `feed_type == 1` is `steam_community_announcements`, i.e. Valve itself. Type 0
    is syndicated press (PCGamesN, SteamDB, PC Gamer ...), kept separately as an
    attention proxy rather than mixed in -- a games-press article is a reaction to
    the market as often as a cause of it, and conflating the two would put a
    reactive series in the same column as a causal one.
    """
    seen: set[tuple[date, str]] = set()
    rows = []
    for i in items:
        ts = i.get("date")
        if not isinstance(ts, (int, float)):
            continue
        day = datetime.fromtimestamp(ts, UTC).date()
        key = (day, i.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"day": day, "is_valve": int(i.get("feed_type") == 1)})

    if not rows:
        return pd.DataFrame(columns=["day", "valve_announcements", "press_articles"])

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("day")
        .agg(
            valve_announcements=("is_valve", "sum"),
            press_articles=("is_valve", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    return grouped


def news_rows(items: list[dict]) -> pd.DataFrame:
    """Keep the news items themselves, one row per announcement.

    Same `(day, title)` dedupe as `news_events` and for the same reason: Steam
    cross-posts one announcement under several gids, and counting those
    separately would both inflate the panel and create duplicate `events` rows
    pointing at the same real-world event. The surviving `gid` is the first
    seen, which is stable across runs because `fetch_news` pages deterministically.

    Sorted oldest-first so a diff between two runs shows the new tail rather
    than a reshuffle.
    """
    seen: set[tuple[date, str]] = set()
    rows = []
    for i in items:
        ts = i.get("date")
        if not isinstance(ts, (int, float)):
            continue
        title = i.get("title", "") or ""
        day = datetime.fromtimestamp(ts, UTC).date()
        key = (day, title)
        if key in seen:
            continue
        seen.add(key)
        feed_type = i.get("feed_type")
        rows.append(
            {
                "gid": str(i.get("gid", "")),
                "day": day,
                "published_at": datetime.fromtimestamp(ts, UTC),
                "feed_type": int(feed_type) if isinstance(feed_type, (int, float)) else None,
                "is_valve": int(feed_type == 1),
                "title": title,
                "url": i.get("url", "") or "",
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(EVENT_COLUMNS))

    df = pd.DataFrame(rows).sort_values(["published_at", "gid"]).reset_index(drop=True)
    return df[list(EVENT_COLUMNS)]


def crate_events(dumps: dict[str, list]) -> pd.DataFrame:
    """Per-day counts of dated crate and collection first-sales.

    Column names carry the source field name (`first_sale_date`) on purpose. See
    the module docstring: this is *not* validated as the day a case became
    purchasable, and naming it `case_release` would assert exactly the thing the
    2026-08-06 measurement refused to support.
    """
    case_days: list[date] = []
    capsule_days: list[date] = []
    for crate in dumps.get("crates", []):
        day = normalise_date(crate.get("first_sale_date"))
        if day is None:
            continue
        ctype = crate.get("type")
        if ctype in CASE_TYPES:
            case_days.append(day)
        elif ctype in CAPSULE_TYPES:
            capsule_days.append(day)

    coll_days = [d for d in (normalise_date(c.get("release_date")) for c in dumps.get("collections", [])) if d]

    frames = []
    for days, col in (
        (case_days, "crate_case_first_sales"),
        (capsule_days, "crate_capsule_first_sales"),
        (coll_days, "collection_releases"),
    ):
        if days:
            s = pd.Series(days).value_counts().rename(col)
            s.index.name = "day"
            frames.append(s)

    if not frames:
        return pd.DataFrame(
            columns=["day", "crate_case_first_sales", "crate_capsule_first_sales", "collection_releases"]
        )
    return pd.concat(frames, axis=1).reset_index().rename(columns={"index": "day"})


# --------------------------------------------------------------------------
# Calendar assembly
# --------------------------------------------------------------------------


def _days_since(flags: pd.Series) -> pd.Series:
    """Days since the last day whose count was > 0, counting the event day as 0.

    Strictly backward-looking: the running index of the last event is taken with
    a forward-fill, which can only propagate a value to *later* rows. Days before
    the first event are NA, not 0 -- filling them would fabricate an event at the
    start of the calendar.
    """
    idx = pd.Series(range(len(flags)), index=flags.index, dtype="float64")
    last = idx.where(flags.to_numpy() > 0).ffill()
    return (idx - last).astype("Float64")


def build_calendar(
    news: pd.DataFrame, crates: pd.DataFrame, start: date = CALENDAR_START, end: date | None = None
) -> pd.DataFrame:
    """One dense row per day over [start, end], with trailing windows."""
    end = end or datetime.now(UTC).date()
    days = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"day": days})

    for src in (news, crates):
        if src is None or src.empty:
            continue
        s = src.copy()
        s["day"] = pd.to_datetime(s["day"])
        cal = cal.merge(s, on="day", how="left")

    count_cols = [
        "valve_announcements",
        "press_articles",
        "crate_case_first_sales",
        "crate_capsule_first_sales",
        "collection_releases",
    ]
    for col in count_cols:
        if col not in cal.columns:
            cal[col] = 0
        cal[col] = cal[col].fillna(0).astype("int64")

    # Trailing windows include the current day, so nothing reads forward.
    for w in TRAILING_WINDOWS:
        cal[f"valve_announcements_{w}d"] = cal["valve_announcements"].rolling(w, min_periods=1).sum().astype("int64")
        cal[f"press_articles_{w}d"] = cal["press_articles"].rolling(w, min_periods=1).sum().astype("int64")
    cal["crate_capsule_first_sales_30d"] = (
        cal["crate_capsule_first_sales"].rolling(30, min_periods=1).sum().astype("int64")
    )
    # A year of case issuance is the closest thing here to a supply-dilution rate.
    cal["crate_case_first_sales_365d"] = (
        cal["crate_case_first_sales"].rolling(ISSUANCE_WINDOW, min_periods=1).sum().astype("int64")
    )

    cal["days_since_valve_announcement"] = _days_since(cal["valve_announcements"])
    cal["days_since_crate_case_first_sale"] = _days_since(cal["crate_case_first_sales"])
    cal["days_since_crate_capsule_first_sale"] = _days_since(cal["crate_capsule_first_sales"])
    cal["days_since_collection_release"] = _days_since(cal["collection_releases"])

    cal["day"] = cal["day"].dt.date
    return cal[list(OUTPUT_COLUMNS)]


def report_coverage(cal: pd.DataFrame) -> None:
    """Print what fraction of the calendar actually carries each event class."""
    n = len(cal)
    logger.info(f"Calendar: {n:,} days, {cal['day'].min()} -> {cal['day'].max()}")
    for col in (
        "valve_announcements",
        "press_articles",
        "crate_case_first_sales",
        "crate_capsule_first_sales",
        "collection_releases",
    ):
        days = int((cal[col] > 0).sum())
        total = int(cal[col].sum())
        logger.info(f"    {col:32s} {days:>6,} days ({100.0 * days / n:5.2f}%), {total:,} events")
    for col in OUTPUT_COLUMNS:
        if col.startswith("days_since_"):
            na = int(cal[col].isna().sum())
            logger.info(f"    {col:32s} NA on first {na:,} days (before the first event)")


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a date-level CS2 event calendar")
    parser.add_argument("--offline", action="store_true", help="Use cached dumps instead of refetching")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--out", default=str(OUTPUT_PARQUET))
    parser.add_argument(
        "--events-out", default=str(EVENTS_PARQUET), help="Per-event companion table (event-news.parquet)"
    )
    parser.add_argument("--start", default=CALENDAR_START.isoformat())
    parser.add_argument("--coverage-only", action="store_true", help="Report coverage and exit without writing")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    session = requests.Session()

    logger.info("Fetching Steam news...")
    items = fetch_news(cache_dir, offline=args.offline, session=session)
    logger.info("Fetching ByMykel crate dumps...")
    dumps = fetch_crates(cache_dir, offline=args.offline, session=session)

    cal = build_calendar(news_events(items), crate_events(dumps), start=date.fromisoformat(args.start))
    report_coverage(cal)

    if args.coverage_only:
        logger.info("--coverage-only: nothing written")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(out, index=False)
    logger.info(f"Wrote {out} ({out.stat().st_size / 1e3:.1f} KB, {len(cal):,} rows x {len(cal.columns)} columns)")

    events = news_rows(items)
    events_out = Path(args.events_out)
    events_out.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_out, index=False)
    logger.info(
        f"Wrote {events_out} ({len(events):,} events, "
        f"{int(events['is_valve'].sum()) if len(events) else 0} from Valve, "
        f"{events['day'].min() if len(events) else '-'}.."
        f"{events['day'].max() if len(events) else '-'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
