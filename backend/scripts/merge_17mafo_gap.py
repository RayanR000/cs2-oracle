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

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.parquet import _append_parquet

SOURCE = "aggregator_steam_17mafo"
PRICE_COLS = ["item_slug", "day", "source", "mean_price", "min_price",
              "max_price", "median_price", "volume"]
SNAP_COLS = ["item_slug", "day", "source", "price", "volume"]
RAW_URL_TEMPLATE = ("https://raw.githubusercontent.com/17mafo/cs-price-tracker/"
                    "main/static/prices/{date}.json")
_SESSION = requests.Session()

DEFAULT_START = "2026-04-16"
DEFAULT_END = "2026-07-08"
DEDUP_KEYS = ["item_slug", "day", "source"]

OVERLAP_START = "2026-07-11"
OVERLAP_END = "2026-07-17"
STEAM_WINDOW_SOURCES = ("aggregator_steam_7d", "aggregator_steam_30d", "aggregator_steam_90d")
BUFF_BASIS_SOURCES = ("aggregator_buff163", "aggregator_csfloat", "aggregator_youpin")


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
        price = None
        for field in ("last_24h", "last_7d", "last_30d"):
            val = steam.get(field)
            if val is not None:
                price = val
                break
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
                      min_items: int = 24000) -> None:
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


def compute_buff_factors(prices_path, overlap_start: str = OVERLAP_START,
                         overlap_end: str = OVERLAP_END) -> tuple[dict, float]:
    """Per-item BUFF/Steam price ratio from the post-gap overlap window.

    Returns ({item_slug: factor}, global_factor) where
    factor = median BUFF-basis price / median Steam-window price over the window.
    global_factor is the median of per-item factors, or 1.0 if none available
    (e.g. the archive file does not exist yet or has no overlap rows).
    """
    prices_path = Path(prices_path)
    if not prices_path.exists():
        return {}, 1.0
    import duckdb
    import statistics
    steam_list = ",".join(f"'{s}'" for s in STEAM_WINDOW_SOURCES)
    buff_list = ",".join(f"'{s}'" for s in BUFF_BASIS_SOURCES)
    con = duckdb.connect()
    try:
        rows = con.sql(f"""
            WITH ov AS (
              SELECT item_slug,
                MEDIAN(CASE WHEN source IN ({steam_list}) THEN median_price END) AS steam_px,
                MEDIAN(CASE WHEN source IN ({buff_list}) THEN median_price END) AS buff_px
              FROM read_parquet('{prices_path}')
              WHERE day BETWEEN '{overlap_start}' AND '{overlap_end}'
              GROUP BY item_slug)
            SELECT item_slug, buff_px / steam_px AS factor
            FROM ov
            WHERE steam_px IS NOT NULL AND buff_px IS NOT NULL
              AND steam_px > 0 AND buff_px > 0
        """).fetchall()
    finally:
        con.close()
    factors = {slug: float(factor) for slug, factor in rows}
    if not factors:
        return {}, 1.0
    return factors, float(statistics.median(factors.values()))


def apply_rescale(prices, factors: dict, global_factor: float):
    """Multiply price columns by each item's BUFF-basis factor (global fallback)."""
    if prices.empty:
        return prices
    out = prices.copy()
    f = out["item_slug"].map(factors).fillna(global_factor)
    for col in ("mean_price", "median_price", "min_price", "max_price"):
        out[col] = out[col] * f
    return out


def run(start: str, end: str, out_dir: Path, cache_dir: Path,
        dry_run: bool = False, refresh: bool = False,
        min_items: int = 24000, fetch=fetch_day) -> pd.DataFrame:
    dates = gap_dates(start, end)
    frames = []
    for d in dates:
        path = fetch(d, cache_dir, refresh=refresh)
        frames.append(transform_day(load_day(path), d))
    prices = pd.concat(frames, ignore_index=True)
    print(f"Transformed {len(prices):,} price rows over {len(dates)} days")

    factors, global_factor = compute_buff_factors(out_dir / "prices-2026.parquet")
    prices = apply_rescale(prices, factors, global_factor)
    print(f"Rescaled to BUFF basis: {len(factors):,} per-item factors, "
          f"global fallback {global_factor:.3f}")

    validate_coverage(prices, dates, min_items=min_items)
    print("Coverage validation passed")

    if dry_run:
        print("Dry run — no files written")
        return prices

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = prices_to_snapshots(prices)
    _append_parquet(out_dir / "prices-2026.parquet", prices[PRICE_COLS], DEDUP_KEYS)
    _append_parquet(out_dir / "snapshots-2026.parquet", snapshots[SNAP_COLS], DEDUP_KEYS)
    print(f"Done. Appended {start}..{end} as source={SOURCE}")
    return prices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-date", default=DEFAULT_START)
    ap.add_argument("--end-date", default=DEFAULT_END)
    ap.add_argument("--out-dir", default="../price-archive")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "raw" / "17mafo"
    run(args.start_date, args.end_date, out_dir, cache_dir,
        dry_run=args.dry_run, refresh=args.refresh)


if __name__ == "__main__":
    main()
