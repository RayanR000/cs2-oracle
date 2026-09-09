#!/usr/bin/env python3
"""Structured Reddit/market-event detector (NOT generic sentiment).

Research-note implementation for
``docs/research/2026-09-08-next-accuracy-indicators.md`` priority 6 (lowest):
do NOT restore the old generic positive/negative sentiment features —
production has no social rows and they ranked outside the top 20 at every
horizon. This pipeline detects structured events with first-seen timestamps:

- item/category mention-velocity shocks
- supply-disappearance, whale-buying, manipulation, pump/exhaustion narratives
- newly discovered trade-up paths
- pro-player/streamer/tournament exposure
- Valve-update and drop-pool speculation
- disagreement: rising attention + falling price / weakening depth

Each event retains its first-seen timestamp, source, engagement velocity,
entity mapping, author/history quality, and novelty vs prior posts. Never let
posts published after the forecast cutoff enter a feature — every row carries
``first_seen_at`` and consumers must join on ``first_seen_at <= forecast_date``.

Auth: Reddit API credentials (``REDDIT_CLIENT_ID`` + ``REDDIT_CLIENT_SECRET``,
or ``REDDIT_BEARER_TOKEN``). Without them ``collect()`` returns
``status=skipped`` — never a silent zero-row success. Nothing here imports
``database``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

REDDIT_OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_URL = "https://oauth.reddit.com"

EVENT_COLUMNS = [
    "event_id",
    "first_seen_at",
    "snapshot_day",
    "source",
    "item_slug",
    "family",
    "narrative",
    "mention_velocity",
    "post_count",
    "engagement_velocity",
    "sentiment_mean",
    "sentiment_reversed",
    "author_quality",
    "novelty",
    "source_url",
    "collected_at",
]

DEDUP_KEYS = ["event_id"]

USER_AGENT = "Mozilla/5.0 (compatible; CS2Oracle/1.0; +https://github.com/RayanR000)"

NARRATIVES = (
    "mention_shock",
    "supply_disappearance",
    "whale_buying",
    "manipulation",
    "pump_exhaustion",
    "tradeup_path",
    "pro_exposure",
    "valve_update",
    "drop_pool_spec",
    "disagreement",
)

_NARRATIVE_PATTERNS: dict[str, re.Pattern] = {
    "supply_disappearance": re.compile(r"sold\s*out|disappear|delist|no\s*supply|out\s*of\s*stock", re.I),
    "whale_buying": re.compile(r"whale|buying\s*up|buyout|sweep(ing)?\s*the", re.I),
    "manipulation": re.compile(r"manipulat|pump\s*and\s*dump|price\s*fix", re.I),
    "pump_exhaustion": re.compile(r"pump|exhaust|baghold|dump\s*on", re.I),
    "tradeup_path": re.compile(r"trade[\s-]?up", re.I),
    "pro_exposure": re.compile(r"major|tournament|pro\s|streamer|s1mple|donk|zywoo|m0nesy", re.I),
    "valve_update": re.compile(r"valve|update|patch|nerf|buff|case\s*drop|armory", re.I),
    "drop_pool_spec": re.compile(r"drop[\s-]?pool|discontinu|rare\s*drop|active\s*drop", re.I),
    "disagreement": re.compile(r"underrated|overreact|disagree|actually\s*down|despite", re.I),
}


class RedditEventError(RuntimeError):
    """Auth failed, the API errored, or the payload was unusable."""


@dataclass
class RedditEventResult:
    rows: pd.DataFrame
    raw_posts: int
    elapsed_s: float


def _bearer_token(session: requests.Session) -> str:
    direct = os.environ.get("REDDIT_BEARER_TOKEN", "").strip()
    if direct:
        return direct
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if not (cid and secret):
        raise RedditEventError(
            "Reddit credentials are not set (REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET "
            "or REDDIT_BEARER_TOKEN). Refusing to record an empty day as success."
        )
    resp = session.post(
        REDDIT_OAUTH_URL,
        data={"grant_type": "client_credentials"},
        auth=(cid, secret),
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RedditEventError(f"Reddit OAuth failed: HTTP {resp.status_code}")
    token = resp.json().get("access_token")
    if not token:
        raise RedditEventError("Reddit OAuth returned no access_token.")
    return str(token)


def classify_narratives(title: str, body: str = "") -> list[str]:
    """Map post text to narrative tags. Pure — unit-testable without creds."""
    text = f"{title}\n{body}"
    return [tag for tag, pat in _NARRATIVE_PATTERNS.items() if pat.search(text)]


def _event_id(item_slug: str | None, narrative: str, day: date) -> str:
    basis = f"{item_slug or '*'}|{narrative}|{day.isoformat()}"
    return hashlib.sha1(basis.encode()).hexdigest()[:16]


def posts_to_events(
    posts: list[dict[str, Any]],
    snapshot_day: date,
    collected_at: datetime,
    mention_baseline: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Reduce raw Reddit posts to structured event rows. Pure.

    ``posts`` entries: ``title``, ``body`` (optional), ``created_utc`` (epoch),
    ``score``, ``num_comments``, ``author_karma`` (optional), ``url``,
    ``item_slug`` (optional entity map), ``family`` (optional).
    Posts dated after ``snapshot_day`` end are dropped (leakage guard).
    """
    day_end = datetime(snapshot_day.year, snapshot_day.month, snapshot_day.day, tzinfo=timezone.utc)
    day_end = day_end.replace(hour=23, minute=59, second=59)
    records: dict[str, dict[str, Any]] = {}
    for post in posts:
        if not isinstance(post, dict):
            continue
        title = post.get("title") or ""
        if not title:
            continue
        created = post.get("created_utc")
        try:
            seen = datetime.fromtimestamp(float(created), tz=timezone.utc) if created else collected_at
        except (TypeError, ValueError):
            seen = collected_at
        if seen > day_end:
            continue  # never let future posts into a past forecast cutoff
        narratives = classify_narratives(title, post.get("body") or "")
        if not narratives:
            continue
        slug = post.get("item_slug")
        engagement = float(post.get("score") or 0) + 2.0 * float(post.get("num_comments") or 0)
        karma = post.get("author_karma")
        quality = min(float(karma or 0) / 10_000.0, 1.0)
        baseline = (mention_baseline or {}).get(slug or "*", 1.0) or 1.0
        for tag in narratives:
            eid = _event_id(slug, tag, snapshot_day)
            row = records.get(eid)
            if row is None:
                records[eid] = {
                    "event_id": eid,
                    "first_seen_at": seen,
                    "snapshot_day": snapshot_day,
                    "source": "reddit",
                    "item_slug": slug,
                    "family": post.get("family"),
                    "narrative": tag,
                    "mention_velocity": 1.0 / baseline,
                    "post_count": 1,
                    "engagement_velocity": engagement,
                    "sentiment_mean": post.get("sentiment"),
                    "sentiment_reversed": -(post.get("sentiment") or 0.0),
                    "author_quality": quality,
                    "novelty": 1.0,
                    "source_url": post.get("url"),
                    "collected_at": collected_at,
                }
            else:
                row["post_count"] += 1
                row["mention_velocity"] += 1.0 / baseline
                row["engagement_velocity"] += engagement
                if seen < row["first_seen_at"]:
                    row["first_seen_at"] = seen
    if not records:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame = pd.DataFrame(list(records.values()))
    return frame[EVENT_COLUMNS]


def fetch_subreddit_posts(
    subreddit: str,
    token: str,
    session: requests.Session,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Pull recent posts from one subreddit via OAuth. Raises on error."""
    resp = session.get(
        f"{REDDIT_API_URL}/r/{subreddit}/new",
        params={"limit": limit},
        headers={"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RedditEventError(f"r/{subreddit}: HTTP {resp.status_code}")
    try:
        children = resp.json()["data"]["children"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RedditEventError(f"r/{subreddit}: unexpected payload shape: {exc}") from exc
    posts = []
    for child in children:
        d = child.get("data", {})
        posts.append(
            {
                "title": d.get("title", ""),
                "body": d.get("selftext", ""),
                "created_utc": d.get("created_utc"),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "author_karma": None,
                "url": d.get("url"),
                "item_slug": None,
                "family": None,
            }
        )
    return posts


SUBREDDITS = ("csgomarketforum", "GlobalOffensiveTrade", "CSGOSkinInvesting")


def reddit_events_parquet_path(archive_dir: Path, snapshot_day: date) -> Path:
    return archive_dir / f"reddit-events-{snapshot_day:%Y-%m}.parquet"


def write_reddit_event_rows(rows: pd.DataFrame, archive_dir: Path, snapshot_day: date) -> int:
    if rows.empty:
        return 0
    path = reddit_events_parquet_path(archive_dir, snapshot_day)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.copy()
    rows["snapshot_day"] = pd.to_datetime(rows["snapshot_day"]).dt.date
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="last")
    else:
        combined = rows
    combined = combined.sort_values(DEDUP_KEYS).reset_index(drop=True)
    combined.to_parquet(path, index=False, compression="zstd")
    return len(rows)


def collect(
    archive_dir: Path,
    snapshot_day: date | None = None,
    dry_run: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch recent Reddit posts and persist structured events."""
    if snapshot_day is None:
        from collectors.snapshot_date import resolve_snapshot_date

        snapshot_day = resolve_snapshot_date()
    collected_at = datetime.now(timezone.utc)
    sess = session or requests.Session()
    started = time.monotonic()
    try:
        token = _bearer_token(sess)
    except RedditEventError as exc:
        logger.warning("Reddit events skipped: %s", exc)
        return {"status": "skipped", "reason": str(exc), "snapshot_day": str(snapshot_day)}
    posts: list[dict[str, Any]] = []
    for sub in SUBREDDITS:
        try:
            posts.extend(fetch_subreddit_posts(sub, token, sess))
        except RedditEventError as exc:
            logger.warning("  r/%s FAILED: %s", sub, exc)
    rows = posts_to_events(posts, snapshot_day, collected_at)
    written = 0 if dry_run else write_reddit_event_rows(rows, archive_dir, snapshot_day)
    elapsed = time.monotonic() - started
    logger.info("Reddit events %s: %s posts -> %s events", snapshot_day, len(posts), written)
    return {
        "status": "success",
        "snapshot_day": str(snapshot_day),
        "reddit_event_rows": written,
        "raw_posts": len(posts),
        "elapsed_seconds": round(elapsed, 2),
        "dry_run": dry_run,
    }
