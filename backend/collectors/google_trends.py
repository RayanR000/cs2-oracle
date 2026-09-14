"""
Google Trends collector for CS2 market-wide search interest.

Fetches daily interest scores for CS2-related keywords via Google's
internal Trends JSON endpoints (no API key needed). Stores results
to data/google_trends.parquet.

Run directly:
    python -m collectors.google_trends
    python -m collectors.google_trends --backfill 365
    python -m collectors.google_trends --backfill 30
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger("google_trends")

DATA_DIR = Path(__file__).parent.parent / "data"
PARQUET_PATH = DATA_DIR / "google_trends.parquet"

KEYWORDS = ["CS2 skins", "Counter-Strike 2 market"]

EXPLORE_URL = "https://trends.google.com/trends/api/explore"
MULTILINE_URL = "https://trends.google.com/trends/api/widgetdata/multiline"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

ANTI_XSSI_PREFIX = ")]}'\n"

MAX_RETRIES = 3
BASE_DELAY = 5.0


def _strip_xssi(text: str) -> str:
    if text.startswith(ANTI_XSSI_PREFIX):
        return text[len(ANTI_XSSI_PREFIX) :]
    if text.startswith(")]}'"):
        idx = text.index("\n")
        return text[idx + 1 :]
    return text


def _fetch_interest(keyword: str, time_range: str, session: requests.Session) -> pd.DataFrame | None:
    """Fetch daily interest for a single keyword over a time range.

    time_range: Google Trends format, e.g. "2025-09-01 2026-09-01"
    Returns DataFrame with columns [date, interest] or None on failure.
    """
    req_payload = {
        "comparisonItem": [{"keyword": keyword, "geo": "", "time": time_range}],
        "category": 0,
        "property": "",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(
                EXPLORE_URL,
                params={"hl": "en-US", "tz": "360", "req": json.dumps(req_payload)},
                timeout=30,
            )
            if resp.status_code == 429:
                wait = BASE_DELAY * (2**attempt)
                logger.warning("Rate limited on explore, waiting %.0fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()

            data = json.loads(_strip_xssi(resp.text))
            widgets = data.get("widgets", [])

            timeline_widget = None
            for w in widgets:
                if w.get("id") == "TIMESERIES":
                    timeline_widget = w
                    break
            if not timeline_widget:
                logger.warning("No TIMESERIES widget for %r", keyword)
                return None

            token = timeline_widget["token"]
            req_inner = timeline_widget["request"]

            time.sleep(1.5)

            resp2 = session.get(
                MULTILINE_URL,
                params={
                    "hl": "en-US",
                    "tz": "360",
                    "req": json.dumps(req_inner),
                    "token": token,
                },
                timeout=30,
            )
            if resp2.status_code == 429:
                wait = BASE_DELAY * (2**attempt)
                logger.warning("Rate limited on multiline, waiting %.0fs", wait)
                time.sleep(wait)
                continue
            resp2.raise_for_status()

            ts_data = json.loads(_strip_xssi(resp2.text))
            timeline = ts_data.get("default", {}).get("timelineData", [])
            if not timeline:
                logger.warning("Empty timeline for %r", keyword)
                return None

            rows = []
            for point in timeline:
                ts = int(point["time"])
                dt = datetime.fromtimestamp(ts, tz=None).date()
                value = point.get("value", [0])[0]
                rows.append({"date": dt, "interest": int(value)})

            return pd.DataFrame(rows)

        except requests.RequestException as e:
            wait = BASE_DELAY * (2**attempt)
            logger.warning("Request error for %r (attempt %d): %s", keyword, attempt + 1, e)
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("Parse error for %r: %s", keyword, e)
            return None

    logger.error("Failed to fetch %r after %d attempts", keyword, MAX_RETRIES)
    return None


def _load_existing() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return pd.DataFrame(columns=["date", "interest"])


def _save(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(PARQUET_PATH, index=False)
    logger.info("Saved %d rows to %s", len(df), PARQUET_PATH)


def collect(backfill_days: int = 180) -> dict:
    """Fetch Google Trends interest and append to the parquet store.

    Splits the request into <=6-month windows (Google returns daily
    granularity only for windows under ~6 months).
    """
    existing = _load_existing()
    existing_dates = set(existing["date"]) if not existing.empty else set()

    end = date.today()
    start = end - timedelta(days=backfill_days)

    if existing_dates:
        latest = max(existing_dates)
        if latest >= end - timedelta(days=1):
            logger.info("Already up to date (latest: %s)", latest)
            return {"status": "up_to_date", "rows": len(existing)}
        start = min(start, latest)

    windows = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=179), end)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)

    logger.info("Fetching %d window(s) from %s to %s", len(windows), start, end)

    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://trends.google.com/trends/explore", timeout=15)
    time.sleep(1)

    all_new = []
    for w_start, w_end in windows:
        time_range = f"{w_start} {w_end}"
        logger.info("Window: %s", time_range)

        keyword_frames = []
        for kw in KEYWORDS:
            df = _fetch_interest(kw, time_range, session)
            if df is not None and not df.empty:
                keyword_frames.append(df)
            time.sleep(2)

        if not keyword_frames:
            logger.warning("No data for window %s", time_range)
            continue

        combined = keyword_frames[0].rename(columns={"interest": "i0"})
        for i, kf in enumerate(keyword_frames[1:], 1):
            combined = combined.merge(
                kf.rename(columns={"interest": f"i{i}"}),
                on="date",
                how="outer",
            )
        interest_cols = [c for c in combined.columns if c.startswith("i")]
        combined["interest"] = combined[interest_cols].mean(axis=1).round().astype(int)
        combined = combined[["date", "interest"]]

        new_rows = combined[~combined["date"].isin(existing_dates)]
        all_new.append(new_rows)
        existing_dates.update(new_rows["date"])

        time.sleep(3)

    if not all_new:
        logger.info("No new data fetched")
        return {"status": "no_new_data", "rows": len(existing)}

    new_df = pd.concat(all_new, ignore_index=True)
    merged = pd.concat([existing, new_df], ignore_index=True)
    _save(merged)

    logger.info("Added %d new rows (%d total)", len(new_df), len(merged))
    return {"status": "success", "new_rows": len(new_df), "total_rows": len(merged)}


def run():
    import argparse

    parser = argparse.ArgumentParser(description="Google Trends CS2 collector")
    parser.add_argument("--backfill", type=int, default=365, help="Days to backfill (default: 365)")
    args = parser.parse_args()

    result = collect(backfill_days=args.backfill)
    logger.info("Result: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run()
