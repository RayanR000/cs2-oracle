#!/usr/bin/env python3
"""
Backfill the Apr 16 - Jul 8 2026 Steam price-history gap into the Parquet
archive from the free 17mafo/cs-price-tracker GitHub daily snapshots.

Each source file is static/prices/YYYY-MM-DD.json: a JSON object keyed by
Steam market_hash_name (== our item_slug) whose value is
{"steam": {"last_24h", "last_7d", "last_30d", "last_90d", "last_ever"}}.
We take last_24h as that day's Steam price.

Usage:
    python scripts/merge_17mafo_gap.py                 # full gap fill
    python scripts/merge_17mafo_gap.py --dry-run       # fetch + report, no write
    python scripts/merge_17mafo_gap.py --refresh       # re-download cached files
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

SOURCE = "aggregator_steam_17mafo"
PRICE_COLS = ["item_slug", "day", "source", "mean_price", "min_price",
              "max_price", "median_price", "volume"]
SNAP_COLS = ["item_slug", "day", "source", "price", "volume"]
RAW_URL_TEMPLATE = ("https://raw.githubusercontent.com/17mafo/cs-price-tracker/"
                    "main/static/prices/{date}.json")
_SESSION = requests.Session()


def transform_day(day_obj: dict, day: str) -> pd.DataFrame:
    """Convert one day's 17mafo JSON object into PRICE_COLS rows."""
    ts = pd.Timestamp(day)
    rows = []
    for slug, val in day_obj.items():
        if not isinstance(val, dict):
            continue
        steam = val.get("steam")
        if not isinstance(steam, dict):
            continue
        price = steam.get("last_24h")
        if price is None:
            continue
        rows.append({
            "item_slug": slug,
            "day": ts,
            "source": SOURCE,
            "mean_price": price,
            "min_price": price,
            "max_price": price,
            "median_price": price,
            "volume": 0,
        })
    return pd.DataFrame(rows, columns=PRICE_COLS)


def prices_to_snapshots(prices: pd.DataFrame) -> pd.DataFrame:
    """Derive snapshot rows (one price per item/day) from price rows."""
    snaps = prices[["item_slug", "day", "source", "mean_price", "volume"]].copy()
    snaps = snaps.rename(columns={"mean_price": "price"})
    return snaps[SNAP_COLS]


def gap_dates(start: str, end: str) -> list[str]:
    """Inclusive list of YYYY-MM-DD date strings from start to end."""
    rng = pd.date_range(start=start, end=end, freq="D")
    return [d.strftime("%Y-%m-%d") for d in rng]


def validate_coverage(prices: pd.DataFrame, expected_dates: list[str],
                      min_items: int = 20000) -> None:
    """Raise AssertionError if any expected day is missing or too sparse."""
    present = {pd.Timestamp(d) for d in prices["day"].unique()}
    for d in expected_dates:
        ts = pd.Timestamp(d)
        assert ts in present, f"missing day {d} in backfill"
        count = prices.loc[prices["day"] == ts, "item_slug"].nunique()
        assert count >= min_items, (
            f"low item count for {d}: {count} < {min_items}")


def load_day(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fetch_day(date: str, cache_dir: Path, refresh: bool = False,
              session=None) -> Path:
    """Return local path to <date>.json, downloading + caching if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{date}.json"
    if path.exists() and not refresh:
        return path
    sess = session or _SESSION
    resp = sess.get(RAW_URL_TEMPLATE.format(date=date), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"fetch {date} failed: HTTP {resp.status_code}")
    path.write_bytes(resp.content)
    return path
