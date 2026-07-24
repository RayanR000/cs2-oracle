#!/usr/bin/env python3
"""
One-off signal test: does FinBERT sentiment from Reddit posts predict
forward price movement for CS2 items?

Scrapes Q1 2025 Reddit posts via pullpush API, matches item names,
scores sentiment with FinBERT, merges with price archive, and runs
return_h ~ sentiment_score OLS regression for 5 horizons.

Checkpointing: posts are saved to /tmp/social_signal_posts.json after
each subreddit. If re-run, completed subreddits are skipped.
"""

import json
import re
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from thefuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent.parent))
from collectors.social_sentiment import score_sentiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("social_signal_test")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_GLOB = str(PROJECT_ROOT / "price-archive" / "prices-*.parquet")
CHECKPOINT_PATH = Path("/tmp/social_signal_posts.json")
PULLPUSH_DELAY = 90
SUBREDDITS = ["GlobalOffensiveTrade", "csgomarketforum"]


def fetch_item_names_from_archive() -> set[str]:
    import duckdb
    con = duckdb.connect()
    try:
        rows = con.sql(f"""
            SELECT DISTINCT item_slug
            FROM read_parquet('{ARCHIVE_GLOB}')
        """).fetchall()
        names = {r[0].lower().strip() for r in rows}
        logger.info("Loaded %d item names from price archive", len(names))
        return names
    finally:
        con.close()


FUZZY_THRESHOLD = 60


class ItemMatcher:
    def __init__(self, names: set[str]):
        self.names = sorted(names, key=len, reverse=True)
        escaped = [re.escape(n) for n in self.names]
        self.exact_re = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
        self.lower_names = [n.lower() for n in self.names]
        logger.info("  ItemMatcher ready: %d names, fuzzy threshold %d%%",
                    len(self.names), FUZZY_THRESHOLD)

    def match(self, title: str) -> set[str]:
        exact = self._exact_match(title)
        if exact:
            return exact
        fuzzy = self._fuzzy_match(title)
        return fuzzy

    def _exact_match(self, title: str) -> set[str]:
        matches = self.exact_re.findall(title)
        return set(m.lower().strip() for m in matches)

    def _fuzzy_match(self, title: str) -> set[str]:
        title_lower = title.lower()
        result = process.extractOne(
            title_lower, self.lower_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if result is None:
            return set()
        return {self.names[self.lower_names.index(result[0])].lower().strip()}


def categorize_item(slug: str) -> tuple[str, str]:
    slug_lower = slug.lower()
    has_wear = any(w in slug_lower for w in [
        "(factory new)", "(minimal wear)", "(field-tested)",
        "(well-worn)", "(battle-scarred)",
    ])
    is_knife = slug_lower.startswith("\u2605") and "gloves" not in slug_lower
    is_gloves = slug_lower.startswith("\u2605") and "gloves" in slug_lower
    is_sticker = slug_lower.startswith("sticker |") or slug_lower.startswith("sticker slab")
    is_case = "case" in slug_lower and ("key" not in slug_lower or slug_lower.endswith("case key"))
    is_patch = slug_lower.startswith("patch")
    is_agent = slug_lower.startswith("agent")
    is_charm = slug_lower.startswith("charm")
    is_graffiti = slug_lower.startswith("sealed graffiti") or slug_lower.startswith("graffiti")
    is_music = slug_lower.startswith("music kit")
    is_pin = slug_lower.endswith("pin") and "|" not in slug_lower

    if is_sticker:
        weapon_part = "Sticker"
        return ("sticker", weapon_part)
    if is_case:
        return ("case", "Case")
    if is_patch:
        return ("patch", slug.split(" | ")[1] if " | " in slug else slug)
    if is_agent:
        return ("agent", slug.split(" | ")[1] if " | " in slug else slug)
    if is_charm:
        return ("charm", "Charm")
    if is_graffiti:
        return ("graffiti", "Graffiti")
    if is_music:
        return ("music_kit", "Music Kit")
    if is_gloves:
        return ("gloves", slug.split(" | ")[0].strip() if " | " in slug else slug)
    if is_knife:
        knife_name = slug.split(" | ")[0].strip() if " | " in slug else slug
        return ("knife", knife_name)
    if is_pin:
        return ("pin", "Pin")
    if has_wear and " | " in slug_lower:
        weapon = slug.split(" | ")[0].strip()
        return ("skin", weapon)

    return ("other", slug)


def fetch_pullpush_posts(subreddit: str, after_ts: int, before_ts: int) -> list[dict]:
    all_posts = []
    after = after_ts
    before = before_ts
    page = 0
    backoff = PULLPUSH_DELAY
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
    }

    while True:
        try:
            resp = requests.get(
                "https://api.pullpush.io/reddit/search/submission/",
                params={
                    "subreddit": subreddit,
                    "after": after,
                    "before": before,
                    "size": 100,
                    "sort_type": "created_utc",
                    "sort": "asc",
                },
                headers=headers,
                timeout=30,
            )
        except requests.RequestException as e:
            logger.warning("  %s page %d: connection error (%s), retrying in %ds",
                           subreddit, page, e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue

        if resp.status_code == 429:
            logger.warning("  %s page %d: rate limited, retrying in %ds",
                           subreddit, page, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue

        if resp.status_code != 200:
            logger.warning("  %s page %d: HTTP %d, stopping",
                           subreddit, page, resp.status_code)
            break

        backoff = PULLPUSH_DELAY
        data = resp.json().get("data", [])
        if not data:
            logger.info("  %s: done (%d posts, %d pages)",
                        subreddit, len(all_posts), page)
            break

        all_posts.extend(data)
        after = data[-1]["created_utc"]
        page += 1

        if len(data) < 100:
            logger.info("  %s: final page (%d posts, %d pages)",
                        subreddit, len(all_posts), page)
            break

        logger.info("  %s: page %d, oldest=%s — waiting %ds...",
                    subreddit, page,
                    datetime.fromtimestamp(after, tz=timezone.utc).date(),
                    PULLPUSH_DELAY)
        time.sleep(PULLPUSH_DELAY)

    return all_posts


def process_posts(posts: list[dict], matcher: ItemMatcher, name_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for p in posts:
        title = p.get("title", "")
        if not title:
            continue
        matches = matcher.match(title)
        if not matches:
            continue
        sentiment = score_sentiment(title)
        created = datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).date()
        for item_name in matches:
            raw_slug = name_map.get(item_name, item_name)
            category, weapon = categorize_item(raw_slug)
            rows.append({
                "item_slug": item_name,
                "category": category,
                "weapon": weapon,
                "date": created,
                "sentiment": sentiment,
                "post_id": p.get("id", ""),
                "subreddit": p.get("subreddit", ""),
                "title": title[:200],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["item_slug", "date", "post_id"])
        logger.info("  Processed %d mentions (%d unique item-date combos), %d matched posts",
                    len(rows), df.groupby(["item_slug", "date"]).ngroups,
                    df["post_id"].nunique())
    return df


def compute_forward_returns(
    mentions: pd.DataFrame, horizons: list[int]
) -> dict[int, pd.DataFrame]:
    import duckdb
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE mentions (item_slug VARCHAR, date DATE, sentiment DOUBLE, category VARCHAR, weapon VARCHAR)")
        for _, row in mentions.iterrows():
            con.execute(
                "INSERT INTO mentions VALUES (?, ?, ?, ?, ?)",
                [row["item_slug"], row["date"], row["sentiment"], row["category"], row["weapon"]],
            )

        for h in horizons:
            logger.info("  Computing %dd forward returns...", h)
            con.execute(f"""
                CREATE TABLE returns_{h}d AS
                SELECT
                    m.item_slug,
                    m.date,
                    m.sentiment,
                    m.category,
                    m.weapon,
                    (p_future.mean_price - p_now.mean_price)
                    / NULLIF(p_now.mean_price, 0) * 100 AS return_{h}d
                FROM mentions m
                JOIN read_parquet('{ARCHIVE_GLOB}') p_now
                  ON LOWER(p_now.item_slug) = m.item_slug
                 AND p_now.day::DATE = m.date
                LEFT JOIN read_parquet('{ARCHIVE_GLOB}') p_future
                  ON LOWER(p_future.item_slug) = m.item_slug
                 AND p_future.day::DATE = m.date + INTERVAL '{h} days'
                WHERE p_now.mean_price > 0
            """)
            n = con.execute(f"SELECT COUNT(*) FROM returns_{h}d").fetchone()[0]
            logger.info("    %d rows with non-null returns", n)

        dfs = {}
        for h in horizons:
            df_h = con.execute(f"SELECT * FROM returns_{h}d").fetchdf()
            df_h = df_h.dropna(subset=[f"return_{h}d"])
            df_h[f"return_{h}d"] = df_h[f"return_{h}d"].clip(-500.0, 500.0)
            dfs[h] = df_h
        return dfs
    finally:
        con.close()


def run_regression(df: pd.DataFrame, horizon: int) -> dict:
    import statsmodels.api as sm

    y = df[f"return_{horizon}d"].values
    X = sm.add_constant(df["sentiment"].values)
    model = sm.OLS(y, X).fit()
    return {
        "horizon": horizon,
        "n": len(df),
        "beta": model.params[1],
        "p_value": model.pvalues[1],
        "r_squared": model.rsquared,
        "conf_int_lower": model.conf_int()[1][0],
        "conf_int_upper": model.conf_int()[1][1],
        "mean_abs_return": float(np.abs(y).mean()),
    }


def load_checkpoint() -> list[dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return []


def save_checkpoint(posts: list[dict]):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(posts, f, default=str)
    logger.info("  Checkpoint saved: %d posts to %s", len(posts), CHECKPOINT_PATH)


def main():
    logger.info("=" * 70)
    logger.info("SOCIAL SENTIMENT SIGNAL TEST")
    logger.info("=" * 70)

    logger.info("\n[1/5] Loading item names from price archive...")
    item_names = fetch_item_names_from_archive()
    matcher = ItemMatcher(item_names)

    logger.info("\n[2/5] Fetching Q1 2025 posts from pullpush...")
    q1_start = int(datetime(2025, 1, 1).timestamp())
    q1_end = int(datetime(2025, 4, 1).timestamp())

    checkpoint = load_checkpoint()
    checkpoint_subs = {p.get("_subreddit") for p in checkpoint}
    all_posts = checkpoint

    for sub in SUBREDDITS:
        if sub in checkpoint_subs:
            logger.info("  r/%s: already in checkpoint (%d posts), skipping",
                        sub, sum(1 for p in checkpoint if p.get("_subreddit") == sub))
            continue

        logger.info("  r/%s...", sub)
        posts = fetch_pullpush_posts(sub, q1_start, q1_end)
        logger.info("  r/%s: %d posts", sub, len(posts))

        for p in posts:
            p["_subreddit"] = sub
            p["_timestamp"] = time.time()
        all_posts.extend(posts)
        save_checkpoint(all_posts)

    checkpoint_subs_final = {p.get("_subreddit") for p in all_posts}
    logger.info("  Total: %d posts across %d subreddits",
                len(all_posts), len(checkpoint_subs_final))

    if not all_posts:
        logger.error("No posts fetched — aborting.")
        sys.exit(1)

    logger.info("\n[3/5] Processing posts — matching items + scoring sentiment...")
    name_map = {n.lower().strip(): n for n in item_names}
    mentions = process_posts(all_posts, matcher, name_map)
    if mentions.empty:
        logger.error("No item mentions found — aborting.")
        sys.exit(1)
    n_unique_items = mentions["item_slug"].nunique()
    logger.info("  %d mentions, %d unique items across %d dates",
                len(mentions), n_unique_items, mentions["date"].nunique())

    logger.info("\n[4/5] Computing forward returns (3d, 7d, 14d, 30d)...")
    horizons = [3, 7, 14, 30]
    dfs = compute_forward_returns(mentions, horizons)

    logger.info("\n[5/5] Running regressions...")
    logger.info("=" * 70)
    logger.info(f"{'Horizon':>8} {'N':>6} {'Beta':>8} {'p-value':>9} {'R²':>7} "
                f"{'CI (95%)':>20} {'|Return|':>8}  {'Signal?':>8}")
    logger.info("-" * 70)

    results = []
    for h in horizons:
        df = dfs[h]
        if df.empty or len(df) < 30:
            logger.info(f"  {h:>4}d: insufficient data ({len(df)} rows)")
            continue
        r = run_regression(df, h)
        results.append(r)
        signal = "YES ★" if r["p_value"] < 0.05 else "no"
        logger.info(
            f"  {r['horizon']:>4}d {r['n']:>6} {r['beta']:>+8.4f} "
            f"{r['p_value']:>9.4f} {r['r_squared']:>7.4f} "
            f"[{r['conf_int_lower']:>7.3f}, {r['conf_int_upper']:>7.3f}] "
            f"{r['mean_abs_return']:>8.2f}%  {signal:>8}"
        )

    logger.info("-" * 70)
    sig_count = sum(1 for r in results if r["p_value"] < 0.05)
    if sig_count > 0:
        logger.info("CONCLUSION: Sentiment signal DETECTED at %d horizon(s). "
                    "Worth investing in social features.", sig_count)
    else:
        logger.info("CONCLUSION: No significant sentiment signal at any horizon.")
        if results:
            best = min(results, key=lambda r: r["p_value"])
            logger.info(
                "  Best p-value: %.4f (beta=%.4f) at %dd horizon",
                best["p_value"], best["beta"], best["horizon"],
            )

    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-CATEGORY ANALYSIS (30d horizon)")
    logger.info("=" * 70)
    logger.info(f"{'Category':>20} {'N':>6} {'Beta':>9} {'p-value':>9} {'R²':>7}  {'Signal?':>8}")
    logger.info("-" * 70)
    df_cat = dfs[30].copy()
    for cat in sorted(df_cat["category"].unique()):
        sub = df_cat[df_cat["category"] == cat]
        if len(sub) < 20:
            continue
        r = run_regression(sub, 30)
        signal = "YES ★" if r["p_value"] < 0.05 else "no"
        logger.info(
            f"  {cat:>20} {r['n']:>6} {r['beta']:>+9.4f} "
            f"{r['p_value']:>9.4f} {r['r_squared']:>7.4f}  {signal:>8}"
        )
    logger.info("-" * 70)

    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-WEAPON ANALYSIS (30d horizon, min 20 obs)")
    logger.info("=" * 70)
    logger.info(f"{'Weapon':>22} {'N':>6} {'Beta':>9} {'p-value':>9} {'R²':>7}  {'Signal?':>8}")
    logger.info("-" * 70)
    df_wpn = dfs[30].copy()
    for weapon in sorted(df_wpn["weapon"].unique()):
        sub = df_wpn[df_wpn["weapon"] == weapon]
        if len(sub) < 20:
            continue
        r = run_regression(sub, 30)
        signal = "YES ★" if r["p_value"] < 0.05 else "no"
        logger.info(
            f"  {weapon[:22]:>22} {r['n']:>6} {r['beta']:>+9.4f} "
            f"{r['p_value']:>9.4f} {r['r_squared']:>7.4f}  {signal:>8}"
        )
    logger.info("-" * 70)

    logger.info("")
    return results


if __name__ == "__main__":
    main()
