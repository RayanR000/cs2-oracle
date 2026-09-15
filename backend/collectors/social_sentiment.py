"""
Reddit social sentiment collector for CS2 skin mentions.

Monitors CS2-related subreddits for skin name mentions in post titles,
scores sentiment via FinBERT ONNX INT8, and stores results in the
social_mentions table.

Run directly:
    python -m collectors.social_sentiment

Or via the task runner:
    python scripts/run_task.py reddit_social
"""

import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from bs4 import BeautifulSoup
from sqlalchemy import text
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, utcnow_naive

logger = logging.getLogger("social_sentiment")

_FINBERT_CACHE = Path.home() / ".cache" / "finbert-int8"


class FinbertScorer:
    """FinBERT sentiment scorer using ONNX INT8 for CPU efficiency.

    Loads the pre-exported and quantized model from ~/.cache/finbert-int8/.
    Falls back to 0.0 if model is unavailable.
    """

    def __init__(self):
        self.tokenizer = None
        self.session = None
        self._load()

    def _load(self):
        onnx_path = _FINBERT_CACHE / "model_int8.onnx"
        if not onnx_path.exists():
            onnx_path = _FINBERT_CACHE / "model.onnx"
        if not onnx_path.exists():
            logger.warning("FinBERT ONNX model not found at %s", _FINBERT_CACHE)
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(_FINBERT_CACHE))
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("FinBERT ONNX INT8 scorer ready (%s, %.0f MB)", onnx_path.name, onnx_path.stat().st_size / 1e6)
        except Exception as e:
            logger.warning("Failed to load FinBERT ONNX model: %s", e)

    def score(self, text: str) -> float:
        """Return sentiment score in [-1, 1] using FinBERT."""
        if self.session is None or self.tokenizer is None:
            return 0.0
        try:
            inp = self.tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=128,
            )
            logits = self.session.run(
                None,
                {
                    "input_ids": inp["input_ids"],
                    "attention_mask": inp["attention_mask"],
                },
            )[0][0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            return float(probs[2] - probs[0])
        except Exception as e:
            logger.warning("FinBERT scoring error: %s", e)
            return 0.0


_finbert: FinbertScorer | None = None


def score_sentiment(text: str) -> float:
    """Return sentiment score in [-1, 1]."""
    global _finbert
    if _finbert is None:
        _finbert = FinbertScorer()
    return _finbert.score(text)


SUBREDDITS = {
    "GlobalOffensiveTrade": 150,
    "csgomarketforum": 50,
    "CSGOSkinInvesting": 25,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_item_names(db) -> dict[str, int]:
    """Load all item names from the database.

    Returns a dict mapping normalized name -> item_id.
    """
    rows = db.execute(text("SELECT id, name FROM items")).fetchall()
    mapping = {}
    for row in rows:
        mapping[row.name.lower().strip()] = row.id
    return mapping


def build_name_regex(name_map: dict[str, int]) -> re.Pattern:
    """Build a single compiled regex matching all item names.

    Sorts names by descending length so longer names match before
    substrings (e.g. 'Butterfly Knife Crimson Web' before 'Butterfly Knife').
    All special characters are escaped.
    """
    sorted_names = sorted(name_map.keys(), key=len, reverse=True)
    escaped = [re.escape(n) for n in sorted_names]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def fetch_subreddit_posts(subreddit: str, limit: int = 100) -> list[dict]:
    """Fetch recent posts from a subreddit via old.reddit.com HTML.

    Returns a list of dicts with keys: id, title, score, url, timestamp
    """
    url = f"https://old.reddit.com/r/{subreddit}/new/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("  Failed to fetch r/%s: %s", subreddit, e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    posts = []
    for thing in soup.select("div.thing")[:limit]:
        title_el = thing.select_one("a.title")
        if not title_el:
            continue

        post_id = thing.get("data-fullname", "")
        if not post_id:
            continue

        title = title_el.text.strip()
        if not title:
            continue

        score_el = thing.select_one("div.score.unvoted")
        score = 0
        if score_el:
            try:
                score = int(score_el.text.strip().replace(" points", "").replace(" point", ""))
            except (ValueError, AttributeError):
                score = 0

        time_el = thing.select_one("time")
        timestamp = None
        if time_el and time_el.has_attr("datetime"):
            try:
                ts = time_el["datetime"].replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                timestamp = None

        permalink = thing.select_one("a[data-event-action='permalink']")
        post_url = ""
        if permalink and permalink.has_attr("href"):
            href = permalink["href"]
            if href.startswith("/"):
                post_url = f"https://old.reddit.com{href}"
            elif href.startswith("https://"):
                post_url = href

        posts.append(
            {
                "id": post_id,
                "title": title,
                "score": score,
                "url": post_url,
                "timestamp": timestamp or utcnow_naive(),
            }
        )

    logger.info("  r/%-25s → %d posts", subreddit, len(posts))
    return posts


def collect_social_mentions(db) -> dict:
    """Main collection logic.

    Returns a stats dict with counts of mentions found.
    """
    start = datetime.now(UTC)
    logger.info("=" * 60)
    logger.info("Reddit Social Sentiment Collection")
    logger.info("=" * 60)

    name_map = fetch_item_names(db)
    logger.info("Loaded %d item names from database", len(name_map))

    if not name_map:
        return {"status": "error", "error": "no items in database"}

    name_regex = build_name_regex(name_map)
    logger.info("Built name regex (%d alternatives)", len(name_map))

    reverse_map = {name.lower().strip(): item_id for name, item_id in name_map.items()}

    total_mentions = 0
    inserted = 0
    now = utcnow_naive()
    parquet_rows = []

    for subreddit, limit in SUBREDDITS.items():
        posts = fetch_subreddit_posts(subreddit, limit=limit)
        mention_params = []
        pending_parquet = []
        for post in posts:
            matches = name_regex.findall(post["title"])
            if not matches:
                continue

            unique_matches = set(m.lower().strip() for m in matches)
            sentiment = score_sentiment(post["title"])

            for matched_name in unique_matches:
                item_id = reverse_map.get(matched_name)
                if item_id is None:
                    continue

                total_mentions += 1
                mention_params.append(
                    {
                        "item_id": item_id,
                        "post_id": post["id"],
                        "subreddit": subreddit,
                        "title": post["title"][:500] if post["title"] else "",
                        "score": post["score"],
                        "url": post["url"][:500] if post["url"] else "",
                        "sentiment": sentiment,
                        "mentioned_at": post["timestamp"],
                        "collected_at": now,
                    }
                )
                pending_parquet.append(
                    {
                        "item_id": item_id,
                        "source": "reddit",
                        "post_id": post["id"],
                        "subreddit": subreddit,
                        "post_title": post["title"][:500] if post["title"] else "",
                        "post_score": post["score"],
                        "post_url": post["url"][:500] if post["url"] else "",
                        "sentiment_score": sentiment,
                        "mentioned_at": post["timestamp"],
                        "collected_at": now,
                    }
                )

        if mention_params:
            try:
                db.execute(
                    text("""
                        INSERT INTO social_mentions
                            (item_id, source, post_id, subreddit, post_title,
                             post_score, post_url, sentiment_score,
                             mentioned_at, collected_at)
                        VALUES
                            (:item_id, 'reddit', :post_id, :subreddit, :title,
                             :score, :url, :sentiment,
                             :mentioned_at, :collected_at)
                        ON CONFLICT (item_id, source, post_id)
                        DO NOTHING
                    """),
                    mention_params,
                )
                parquet_rows.extend(pending_parquet)
                inserted += len(mention_params)
            except Exception as e:
                logger.warning(
                    "  Failed to insert %d mentions (subreddit=%s): %s",
                    len(mention_params),
                    subreddit,
                    e,
                )

        db.commit()

    if parquet_rows:
        from db.parquet import append_table

        append_table("social_mentions", parquet_rows, ["item_id", "source", "post_id"])

    elapsed = (datetime.now(UTC) - start).total_seconds()
    logger.info(
        "Done: %d total mentions, %d inserted in %.1fs",
        total_mentions,
        inserted,
        elapsed,
    )

    return {
        "status": "success",
        "total_mentions": total_mentions,
        "inserted": inserted,
        "elapsed_seconds": elapsed,
    }


def run():
    """Entry point for the task runner."""
    db = SessionLocal()
    try:
        result = collect_social_mentions(db)
        if isinstance(result, dict) and result.get("status") == "error":
            logger.error("Collection failed: %s", result.get("error"))
            sys.exit(1)
        return result
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run()
