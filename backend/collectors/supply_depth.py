#!/usr/bin/env python3
"""Daily supply-depth collector — live listing counts and ask-ladder shape.

Writes one row per ``(item_slug, snapshot_day, source)`` to
``price-archive/supply-YYYY-MM.parquet``. ``item_slug`` is ``market_hash_name``,
so the output joins to ``prices-*.parquet`` directly on name with no DB lookup.

## Why this exists, and what it is NOT claiming

`docs/changelog/2026-07-16-drop-supply-depth.md` dropped supply depth on three
grounds. Two of them died on measurement 2026-08-06: a free bulk listing-count
source *does* exist (four of them), and Wayback holds no retroactive history so
accumulation is the only path. The **accuracy** ground is untouched — trade
volume correlates with forward returns at |r| < 0.002 over 4.47M rows, and the
listing-count *level* is a ~0pp liquidity signal. Only the change/velocity
variant was ever argued predictive, at a calibrated +1-2pp.

This collector therefore exists to **accumulate the series that makes the test
possible**, not because supply depth is believed to work. Collection is the long
pole (velocity features need consecutive days); the A/B design and whether the
harness can even resolve +1-2pp are separate questions answered by
`scripts/archive/compute_mde.py`.

## The two grains, and why they are one table

The scalar feeds return a single integer per item — and they are *one
feature, not several*: Spearman 0.65-0.82 between them (2026-08-06). Collecting all
of them buys **coverage breadth** (~72% of the >=$1 cohort vs ~65% for the best
single feed), not independent signal.

lis-skins is qualitatively different: 2.3M individual listings with `price` and
`created_at`, which yields the **ask ladder shape** and **time-on-market**. That
is a distribution rather than a count, mechanistically distinct from the refuted
trade-volume level, and it is the one genuinely untested quantity here.

Both land in one long-format table keyed by `source`, with the ladder columns
NULL for the scalar feeds. Long format rather than one column per feed is
deliberate: adding or dropping a marketplace must not be a schema migration, and
`skinport_quantity` in the old `supply_snapshots` table is the fossil showing
what the wide shape costs — a column added for a second market that stayed 100%
NULL for its entire life.

## Traps this code is written against

- **Skinport has TWO distinct blocks, and they look alike.** (1) It answers 406
  when the request does not advertise brotli, and `requests` advertises it only
  when a codec is importable — so a missing `brotli` package presents as a
  server-side block. That misdiagnosis is on record in `data-sources.md` as
  "Cloudflare-dead"; `brotli` is now pinned in requirements.txt and
  `_probe_brotli()` fails loudly. (2) Separately, Skinport's WAF answers **403
  with an HTML challenge page** to traffic from Cloudflare-owned egress IPs
  (AS13335), which no header change fixes. Measured 2026-08-06: 403 on
  `/v1/items` for every User-Agent tried, with brotli decoding correctly. Before
  concluding anything about Skinport, check `server: cloudflare` on the response
  and the egress IP's ASN — a 403 here is about *where you are calling from*.
- **Silent zero-row success is this repo's recurring failure.** The Steam supply
  scraper reported green while storing nothing for 16 days. Every fetch here
  either returns rows or raises; a feed that returns an empty payload is an
  error, not an empty result. `collect()` returns per-feed counts so
  `scripts/run_task.py`'s zero-row guard can see them.
- **The min-ask anchor is contaminated.** 358 items show a lowest ask >10x the
  archive price (worst case 1,154x). Depth-within-x% measured from the raw
  minimum inherits that. Depth is therefore measured from `p05_ask`, a trimmed
  anchor, and `min_ask`/`p25_ask`/`median_ask` are all stored so a different
  anchor can be recomputed downstream without re-collecting.
- **`created_at` may be listing age or last-reprice time.** 99.3% of lis-skins
  inventory reads as created within ~37 days, which is either monthly book
  turnover or a field that resets on relist. These give opposite meanings to
  every age feature. A single snapshot cannot distinguish them; consecutive daily
  snapshots can, by tracking whether `created_at` moves for a listing `id` that
  persists. `listing_id_digest` exists for exactly that diagnostic.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from collectors.http import retrying_session
from collectors.snapshot_date import resolve_snapshot_date

logger = logging.getLogger(__name__)

# One row per (item_slug, snapshot_day, source). Ladder columns are NULL for the
# scalar feeds; listing_count is populated by every feed.
SUPPLY_COLUMNS = [
    "item_slug",
    "snapshot_day",
    "source",
    "listing_count",
    "min_ask",
    "p05_ask",
    "p25_ask",
    "median_ask",
    "depth_5pct",
    "depth_10pct",
    "age_median_days",
    "age_p90_days",
    "inflow_24h",
    "listing_id_digest",
    "collected_at",
]

DEDUP_KEYS = ["item_slug", "snapshot_day", "source"]

# Depth is measured from this quantile of the ask ladder rather than the raw
# minimum -- see the "min-ask anchor is contaminated" note in the module
# docstring. Stored quantiles let a downstream consumer pick a different anchor.
DEPTH_ANCHOR_Q = 0.05

USER_AGENT = "Mozilla/5.0 (compatible; CS2Oracle/1.0; +https://github.com/RayanR000)"

# Generous relative to the measured latencies (0.2-2.5 s for the scalar feeds,
# ~44 s for the 173 MB lis-skins export) so a slow day is a slow run, not a
# missing feed.
SCALAR_TIMEOUT_S = 60
LIS_SKINS_TIMEOUT_S = 300


class SupplyFeedError(RuntimeError):
    """A feed failed or returned an unusable payload.

    Raised rather than returning empty so a broken feed cannot present as a feed
    with nothing to report. `collect()` decides per-feed whether that is fatal.
    """


@dataclass
class FeedResult:
    """Outcome of one feed pull, successful or not."""

    source: str
    rows: pd.DataFrame | None = None
    error: str | None = None
    raw_items: int = 0
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def row_count(self) -> int:
        return 0 if self.rows is None else len(self.rows)


# ── HTTP ──────────────────────────────────────────────────────────────────────


def _probe_brotli() -> None:
    """Fail loudly if brotli is unavailable.

    Skinport answers 406 to a request that does not advertise brotli, and
    `requests` advertises it only when a codec is importable. Without this check
    the Skinport feed disappears behind an HTTP error that reads like a block.
    """
    try:
        import brotli  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SupplyFeedError(
            "brotli is not installed, so requests will not send "
            "`Accept-Encoding: br` and Skinport will answer 406. "
            "Install it (it is pinned in backend/requirements.txt) rather than "
            "treating the Skinport feed as blocked."
        ) from exc


def _get_json(url: str, timeout: int, session: requests.Session | None = None) -> Any:
    """GET and decode JSON, raising on anything that is not a usable payload."""
    owns_session = session is None
    sess = session or retrying_session()
    try:
        resp = sess.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "br, gzip, deflate"},
        )
    except requests.RequestException as exc:
        raise SupplyFeedError(f"{url}: request failed: {exc}") from exc
    finally:
        if owns_session:
            sess.close()

    if resp.status_code != 200:
        raise SupplyFeedError(f"{url}: HTTP {resp.status_code}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise SupplyFeedError(f"{url}: response was not JSON ({len(resp.content)} bytes)") from exc

    if not payload:
        raise SupplyFeedError(f"{url}: empty payload")
    return payload


# ── Parsers (pure; no network, so they are unit-testable) ─────────────────────


def _coerce_int(value: Any) -> int | None:
    """Parse a count that may arrive as int, float or string.

    market.csgo.com ships `volume` as a *string*; a silent failure here would
    zero out an entire feed while every other column looked healthy.
    """
    if value is None:
        return None
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out >= 0 else None


def _scalar_rows(
    pairs: Iterable[tuple[str, int | None, float | None]],
    source: str,
    snapshot_day: date,
    collected_at: datetime,
) -> pd.DataFrame:
    """Build the long-format frame for a feed that reports a count per item.

    Items whose count does not parse are dropped rather than zero-filled: a
    listing count of 0 is a real, meaningful observation (nothing for sale), and
    coercing a parse failure into it would fabricate exactly the signal this
    collector exists to measure.
    """
    records = []
    for slug, count, price in pairs:
        if not slug or count is None:
            continue
        records.append(
            {
                "item_slug": slug,
                "snapshot_day": snapshot_day,
                "source": source,
                "listing_count": count,
                "min_ask": price,
                "p05_ask": None,
                "p25_ask": None,
                "median_ask": None,
                "depth_5pct": None,
                "depth_10pct": None,
                "age_median_days": None,
                "age_p90_days": None,
                "inflow_24h": None,
                "listing_id_digest": None,
                "collected_at": collected_at,
            }
        )
    df = pd.DataFrame(records, columns=SUPPLY_COLUMNS)
    # Feeds ship duplicate names (Skinport's sales history carries 854). Keep the
    # deepest observation per item so a duplicate cannot silently halve a count.
    if not df.empty:
        df = (
            df.sort_values("listing_count", ascending=False)
            .drop_duplicates(subset=["item_slug"], keep="first")
            .reset_index(drop=True)
        )
    return df


def parse_skinport(payload: Any, snapshot_day: date, collected_at: datetime) -> pd.DataFrame:
    """api.skinport.com/v1/items -> `quantity` per item."""
    return _scalar_rows(
        (
            (it.get("market_hash_name"), _coerce_int(it.get("quantity")), _coerce_float(it.get("min_price")))
            for it in payload
            if isinstance(it, dict)
        ),
        "skinport",
        snapshot_day,
        collected_at,
    )


def parse_waxpeer(payload: Any, snapshot_day: date, collected_at: datetime) -> pd.DataFrame:
    """api.waxpeer.com/v1/prices -> `count` per item.

    Waxpeer quotes `min` in millicents (price/1000); it is normalised here so
    `min_ask` means USD in every row of the table regardless of source.
    """
    items = payload.get("items", payload) if isinstance(payload, dict) else payload

    def _price(it: dict) -> float | None:
        raw = _coerce_float(it.get("min"))
        return None if raw is None else raw / 1000.0

    return _scalar_rows(
        ((it.get("name"), _coerce_int(it.get("count")), _price(it)) for it in items if isinstance(it, dict)),
        "waxpeer",
        snapshot_day,
        collected_at,
    )


def parse_market_csgo(payload: Any, snapshot_day: date, collected_at: datetime) -> pd.DataFrame:
    """market.csgo.com/api/v2/prices/USD.json -> `volume` per item.

    NOTE the field name. In this repo `volume` normally means *trade* volume,
    which is refuted at |r| < 0.002. It was verified to be a **listing count**
    on 2026-08-06 (`scripts/probe_supply_feeds.py`) on four independent grounds:
    a heavy right tail (median 13, p99 813, max 17,028) matching the other depth
    feeds; Spearman 0.56 against Waxpeer's `count`; values exceeding CSFloat's
    genuine daily sale count for all 9 items tested, by a multiple that widens
    as liquidity falls (1.4x on Kilowatt Case to 15x on Glock Fade — the
    inventory ~= trade rate x dwell time signature); and the site exposing a
    per-physical-listing full export. It is also the single largest marginal
    contributor to coverage (+8.3pp on the >=$1 cohort).
    """
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    return _scalar_rows(
        (
            (it.get("market_hash_name"), _coerce_int(it.get("volume")), _coerce_float(it.get("price")))
            for it in items
            if isinstance(it, dict)
        ),
        "market_csgo",
        snapshot_day,
        collected_at,
    )


def _parse_created_at(raw: Any) -> pd.Timestamp | None:
    if not raw:
        return None
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _parse_created_at_series(raw: pd.Series) -> pd.Series:
    """Vectorised `created_at` parse, with a per-element fallback.

    The fast path pins `format="ISO8601"`, which the feed emits and which pandas
    can apply to the whole column at C speed. Rows that fail it are retried with
    `format="mixed"`, which is per-element and slow but only ever runs on the
    failures -- so a uniform export costs one vectorised call, and an off-format
    one still parses rather than silently dropping ages. Passing no format at all
    would be wrong here: a single unparseable value defeats pandas' inference and
    silently sends all 2.1M rows down the per-element dateutil path, which is the
    cost this function exists to avoid.
    """
    parsed = pd.to_datetime(raw, utc=True, errors="coerce", format="ISO8601")
    unparsed = parsed.isna() & raw.notna() & raw.ne("")
    if unparsed.any():
        parsed.loc[unparsed] = pd.to_datetime(raw[unparsed], utc=True, errors="coerce", format="mixed")
    return parsed


def aggregate_lis_skins(
    listings: Sequence[dict],
    snapshot_day: date,
    collected_at: datetime,
) -> pd.DataFrame:
    """Reduce 2.3M individual listings to per-item ladder + age aggregates.

    The raw ladder is deliberately NOT stored: 173 MB/day of individual listings
    to support a feature set that is untested would dominate the archive, which
    is 88 MB total for 20.7M price rows. Everything downstream needs is a
    per-item summary.

    Emits, per item:
      listing_count                       — depth
      min_ask/p05_ask/p25_ask/median_ask  — ladder quantiles, so the anchor is
                                            re-choosable without re-collecting
      depth_5pct/depth_10pct              — listings within 5%/10% of p05_ask
      age_median_days/age_p90_days        — time-on-market from `created_at`
      inflow_24h                          — listings created in the last 24 h
      listing_id_digest                   — order-independent digest of the
                                            listing-id set, so a later run can
                                            tell whether `created_at` moves for
                                            listings that persist

    Every reduction below is a grouped/vectorised op. This used to be a Python
    loop per listing, which cost 962s of an 18-minute workflow on 2026-09-07 --
    almost all of it a scalar `pd.to_datetime` paying pandas' inference path
    2.1M times, against 45s to download the export it was parsing. Group with
    `sort=False` throughout so rows stay in first-appearance order, as the old
    dict-insertion loop emitted them.
    """
    day_end = datetime.combine(snapshot_day, datetime.min.time(), tzinfo=UTC) + timedelta(days=1)

    frame = pd.DataFrame.from_records(
        [
            (entry.get("name"), entry.get("price"), entry.get("created_at"), entry.get("id"))
            for entry in listings
            if isinstance(entry, dict) and entry.get("name")
        ],
        columns=["item_slug", "price", "created_at", "listing_id"],
    )
    if frame.empty:
        return pd.DataFrame(columns=SUPPLY_COLUMNS)

    slug = frame["item_slug"]
    counts = frame.groupby(slug, sort=False).size().rename("listing_count")

    # ── Ask ladder. Non-numeric, non-finite and non-positive prices drop out,
    # and an item with no usable price is dropped entirely (not zero-filled).
    price = pd.to_numeric(frame["price"], errors="coerce")
    usable = price.gt(0) & np.isfinite(price)
    priced_slug, priced_price = slug[usable], price[usable]
    by_price = priced_price.groupby(priced_slug, sort=False)
    ladder = pd.DataFrame(
        {
            "min_ask": by_price.min(),
            "p05_ask": by_price.quantile(DEPTH_ANCHOR_Q),
            "p25_ask": by_price.quantile(0.25),
            "median_ask": by_price.median(),
        }
    )

    # Depth from the trimmed anchor, not the raw minimum -- one mispriced
    # listing must not redefine "near the bottom of the book".
    anchor = priced_slug.map(ladder["p05_ask"])
    depth = pd.DataFrame(
        {
            "depth_5pct": priced_price.le(anchor * 1.05).groupby(priced_slug, sort=False).sum(),
            "depth_10pct": priced_price.le(anchor * 1.10).groupby(priced_slug, sort=False).sum(),
        }
    )

    # ── Listing age. A negative age means the timestamp disagrees with the
    # snapshot boundary; clipping it to 0 would fabricate a real-looking value
    # at a meaningful edge, so those rows are excluded from the age stats (but
    # still counted as supply).
    age_days = (day_end - _parse_created_at_series(frame["created_at"])).dt.total_seconds() / 86400.0
    dated = age_days.notna() & age_days.ge(0)
    aged_slug, aged_age = slug[dated], age_days[dated]
    by_age = aged_age.groupby(aged_slug, sort=False)
    ages = pd.DataFrame(
        {
            "age_median_days": by_age.median(),
            "age_p90_days": by_age.quantile(0.90),
            "inflow_24h": aged_age.le(1.0).groupby(aged_slug, sort=False).sum(),
        }
    )

    # sorted() then join, so the digest depends on the id set and not on payload
    # ordering. Grouped-apply rather than a sort_values, to keep Python's exact
    # string collation.
    has_id = frame["listing_id"].notna()
    digest = (
        frame.loc[has_id, "listing_id"]
        .astype(str)
        .groupby(slug[has_id], sort=False)
        .apply(lambda ids: hashlib.sha1("|".join(sorted(ids)).encode()).hexdigest()[:16])
        .rename("listing_id_digest")
    )

    out = (
        pd.DataFrame(index=counts.index)
        .join([counts, ladder, depth, ages, digest])
        .rename_axis("item_slug")
        .reset_index()
    )
    out = out[out["min_ask"].notna()].copy()
    if out.empty:
        return pd.DataFrame(columns=SUPPLY_COLUMNS)

    out["snapshot_day"] = snapshot_day
    out["source"] = "lis_skins"
    # ns, not the us that pandas 2.x infers for a datetime scalar: every
    # existing supply-*.parquet is timestamp[ns], and a per-file precision
    # change is the schema drift a plain glob read swallows silently.
    out["collected_at"] = pd.Series([collected_at] * len(out), index=out.index, dtype="datetime64[ns, UTC]")
    for col in ("listing_count", "depth_5pct", "depth_10pct", "inflow_24h"):
        out[col] = out[col].fillna(0).astype(int)

    return out[SUPPLY_COLUMNS].reset_index(drop=True)


# ── Feed registry ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Feed:
    source: str
    url: str
    parser: Callable[[Any, date, datetime], pd.DataFrame]
    timeout: int = SCALAR_TIMEOUT_S
    # Feeds whose payload is the raw listing array rather than per-item rows.
    is_ladder: bool = False


SCALAR_FEEDS: tuple[Feed, ...] = (
    Feed("skinport", "https://api.skinport.com/v1/items?app_id=730&currency=USD", parse_skinport),
    Feed("waxpeer", "https://api.waxpeer.com/v1/prices?game=csgo", parse_waxpeer),
    Feed("market_csgo", "https://market.csgo.com/api/v2/prices/USD.json", parse_market_csgo),
)

LIS_SKINS_URL = "https://lis-skins.com/market_export_json/api_csgo_full.json"


def fetch_feed(
    feed: Feed, snapshot_day: date, collected_at: datetime, session: requests.Session | None = None
) -> FeedResult:
    """Pull and parse one scalar feed. Never raises; failure is carried in the result."""
    started = datetime.now(UTC)
    try:
        payload = _get_json(feed.url, feed.timeout, session)
        rows = feed.parser(payload, snapshot_day, collected_at)
        raw = len(payload.get("items", payload)) if isinstance(payload, dict) else len(payload)
        if rows.empty:
            raise SupplyFeedError(
                f"{feed.source}: {raw} raw items but 0 parsed -- the payload shape "
                "has probably changed; refusing to record an empty feed as success"
            )
        elapsed = (datetime.now(UTC) - started).total_seconds()
        logger.info("  %s: %s items in %.1fs", feed.source, f"{len(rows):,}", elapsed)
        return FeedResult(feed.source, rows=rows, raw_items=raw, elapsed_s=elapsed)
    except SupplyFeedError as exc:
        elapsed = (datetime.now(UTC) - started).total_seconds()
        logger.warning("  %s FAILED: %s", feed.source, exc)
        return FeedResult(feed.source, error=str(exc), elapsed_s=elapsed)


def fetch_lis_skins(snapshot_day: date, collected_at: datetime, session: requests.Session | None = None) -> FeedResult:
    """Pull the 173 MB lis-skins export and reduce it to per-item aggregates."""
    started = datetime.now(UTC)
    try:
        payload = _get_json(LIS_SKINS_URL, LIS_SKINS_TIMEOUT_S, session)
        listings = payload.get("items") if isinstance(payload, dict) else payload
        if not listings:
            raise SupplyFeedError("lis_skins: payload carried no `items` array")
        rows = aggregate_lis_skins(listings, snapshot_day, collected_at)
        if rows.empty:
            raise SupplyFeedError(
                f"lis_skins: {len(listings)} listings but 0 items aggregated -- payload shape has probably changed"
            )
        elapsed = (datetime.now(UTC) - started).total_seconds()
        logger.info("  lis_skins: %s listings -> %s items in %.1fs", f"{len(listings):,}", f"{len(rows):,}", elapsed)
        return FeedResult("lis_skins", rows=rows, raw_items=len(listings), elapsed_s=elapsed)
    except SupplyFeedError as exc:
        elapsed = (datetime.now(UTC) - started).total_seconds()
        logger.warning("  lis_skins FAILED: %s", exc)
        return FeedResult("lis_skins", error=str(exc), elapsed_s=elapsed)


# ── Persistence ───────────────────────────────────────────────────────────────


def supply_parquet_path(archive_dir: Path, snapshot_day: date) -> Path:
    """Monthly partition, matching `prices-YYYY-MM.parquet`.

    Supply lives beside prices rather than under `ops/` because it is market
    observation data, not an operational table. Monthly partitioning is the same
    100 MB-per-file constraint that drove the price archive's layout.
    """
    return archive_dir / f"supply-{snapshot_day:%Y-%m}.parquet"


def write_supply_rows(rows: pd.DataFrame, archive_dir: Path, snapshot_day: date) -> int:
    """Append to the month's partition, replacing any same-day/source rows.

    Re-running a day is idempotent: existing rows for the same
    (item_slug, snapshot_day, source) are dropped in favour of the new ones, so
    a partial run can simply be re-run rather than needing a repair step.
    """
    if rows.empty:
        return 0

    path = supply_parquet_path(archive_dir, snapshot_day)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = rows.copy()
    rows["snapshot_day"] = pd.to_datetime(rows["snapshot_day"]).dt.date

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, rows], ignore_index=True)
        # keep="last" -> the run just completed wins over whatever was there.
        combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="last")
    else:
        combined = rows

    combined = combined.sort_values(DEDUP_KEYS).reset_index(drop=True)
    combined.to_parquet(path, index=False, compression="zstd")
    return len(rows)


# ── Orchestration ─────────────────────────────────────────────────────────────


def collect(
    archive_dir: Path,
    snapshot_day: date | None = None,
    include_ladder: bool = True,
    feeds: Sequence[Feed] = SCALAR_FEEDS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Pull every feed, write one day of rows, and report per-feed counts.

    Returns a dict carrying `supply_rows` — a name registered in
    `scripts/run_task.py::ROW_COUNT_FIELDS`, so a run that stores nothing exits
    non-zero instead of going green. Per-feed counts are reported alongside it
    because a single dead feed among four would otherwise hide behind a healthy
    total, which is the same shape as the 2026-07-16 supply-scraper failure.

    A feed that fails is logged and skipped; the run still succeeds provided at
    least one feed delivered rows. Losing one marketplace should not cost the
    day's collection, but losing all of them must not read as success.
    """
    _probe_brotli()

    # Deliberately the aggregator's resolver, not the wall clock. These feeds are
    # live snapshots with no dump boundary of their own, so on their own terms
    # "today" would be right -- but a supply row is only useful joined to the
    # price row from the same run, and the price side labels its day from the
    # CSGOTrader dump boundary (~21:40 UTC). A run at 00:08 UTC would otherwise
    # stamp supply D+1 against prices D and the join would silently miss every
    # item. `AGGREGATOR_SNAPSHOT_DATE` pins one value across all workflow steps.
    snapshot_day = snapshot_day or resolve_snapshot_date()
    collected_at = datetime.now(UTC)

    # Parallel fetch — each thread gets its own session since a shared
    # requests.Session is not safe for concurrent use beyond simple GETs,
    # and per-thread sessions isolate connection-pool state.
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_with_own_session(feed: Feed) -> FeedResult:
        with retrying_session() as s:
            return fetch_feed(feed, snapshot_day, collected_at, s)

    if feeds:
        with ThreadPoolExecutor(max_workers=len(feeds)) as pool:
            results: list[FeedResult] = list(pool.map(_fetch_with_own_session, feeds))
    else:
        results = []
    if include_ladder:
        with retrying_session() as ladder_session:
            results.append(fetch_lis_skins(snapshot_day, collected_at, ladder_session))

    good = [r for r in results if r.ok and r.row_count]
    failed = [r for r in results if not r.ok]

    frames = [r.rows for r in good if r.rows is not None]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SUPPLY_COLUMNS)

    written = 0 if dry_run else write_supply_rows(combined, archive_dir, snapshot_day)

    summary: dict[str, Any] = {
        "status": "success" if good else "failed",
        "snapshot_day": snapshot_day.isoformat(),
        "supply_rows": len(combined) if dry_run else written,
        "distinct_items": int(combined["item_slug"].nunique()) if not combined.empty else 0,
        "feeds_ok": [r.source for r in good],
        "feeds_failed": {r.source: r.error for r in failed},
        "per_feed_rows": {r.source: r.row_count for r in good},
        "per_feed_seconds": {r.source: round(r.elapsed_s, 1) for r in results},
    }

    if not good:
        # Every feed down is indistinguishable from a network-level block, which
        # is precisely the condition that must never exit 0.
        raise SupplyFeedError(f"all {len(results)} supply feeds failed: {summary['feeds_failed']}")

    logger.info(
        "Supply depth %s: %s rows across %s items from %s/%s feeds",
        snapshot_day,
        f"{summary['supply_rows']:,}",
        f"{summary['distinct_items']:,}",
        len(good),
        len(results),
    )
    return summary
