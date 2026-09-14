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

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.parquet import append_monthly

SOURCE = "aggregator_steam_17mafo"
PRICE_COLS = ["item_slug", "day", "source", "mean_price", "min_price", "max_price", "median_price", "volume"]
SNAP_COLS = ["item_slug", "day", "source", "price", "volume"]
RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/17mafo/cs-price-tracker/main/static/prices/{date}.json"
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
        rows.append(
            {
                "item_slug": slug,
                "day": ts,
                "source": SOURCE,
                "mean_price": price,
                "min_price": price,
                "max_price": price,
                "median_price": price,
                "volume": 0,
            }
        )
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


def validate_coverage(prices: pd.DataFrame, expected_dates: list[str], min_items: int = 24000) -> None:
    """Raise AssertionError if any expected day is missing or too sparse."""
    present = {pd.Timestamp(d) for d in prices["day"].unique()}
    for d in expected_dates:
        ts = pd.Timestamp(d)
        assert ts in present, f"missing day {d} in backfill"
        count = prices.loc[prices["day"] == ts, "item_slug"].nunique()
        assert count >= min_items, f"low item count for {d}: {count} < {min_items}"


def load_day(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fetch_day(date: str, cache_dir: Path, refresh: bool = False, session=None) -> Path:
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


def compute_basis_factors(prices_path, overlap_start: str = OVERLAP_START, overlap_end: str = OVERLAP_END):
    """Per-item Steam->basis factors from the post-gap overlap window.

    Returns (start_factors, start_global, end_factors, end_global):
    - start_* : BUFF-basis factor (median BUFF / median Steam) — matches the
                pre-gap consensus.
    - end_*   : full-consensus factor (median of ALL non-fallback, non-backfill
                sources / median Steam) — matches the post-gap consensus, which
                includes the extra sources switched on at the gap's far edge.
    Each global is the median of its per-item factors, or 1.0 if none available.
    """
    # Accept a single file or a glob (e.g. prices-2026-*.parquet) so this works
    # with the monthly-partitioned archive as well as a single yearly file.
    prices_path = str(prices_path)
    import statistics

    import duckdb

    steam_list = ",".join(f"'{s}'" for s in STEAM_WINDOW_SOURCES)
    buff_list = ",".join(f"'{s}'" for s in BUFF_BASIS_SOURCES)
    con = duckdb.connect()
    try:
        try:
            rows = con.sql(f"""
            WITH ov AS (
              SELECT item_slug,
                MEDIAN(CASE WHEN source IN ({buff_list}) THEN median_price END) AS buff_px,
                MEDIAN(CASE WHEN source IN ({steam_list}) THEN median_price END) AS steam_px,
                MEDIAN(CASE WHEN source NOT LIKE 'historical_fallback:%'
                             AND source <> '{SOURCE}' THEN median_price END) AS cons_px
              FROM read_parquet('{prices_path}')
              WHERE day BETWEEN '{overlap_start}' AND '{overlap_end}'
              GROUP BY item_slug)
            SELECT item_slug, buff_px / steam_px AS f_start, cons_px / steam_px AS f_end
            FROM ov
            WHERE steam_px IS NOT NULL AND steam_px > 0
              AND buff_px IS NOT NULL AND buff_px > 0
              AND cons_px IS NOT NULL AND cons_px > 0
        """).fetchall()
        except duckdb.Error:
            rows = []  # no matching parquet files yet / unreadable
    finally:
        con.close()
    if not rows:
        return {}, 1.0, {}, 1.0
    start_factors = {r[0]: float(r[1]) for r in rows}
    end_factors = {r[0]: float(r[2]) for r in rows}
    start_global = float(statistics.median(start_factors.values()))
    end_global = float(statistics.median(end_factors.values()))
    return start_factors, start_global, end_factors, end_global


def apply_rescale(
    prices,
    start_factors: dict,
    start_global: float,
    end_factors: dict,
    end_global: float,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
):
    """Log-linearly ramp each item's Steam->basis factor across the gap.

    factor(day) = f_start^(1-t) * f_end^t, t = (day-start)/(end-start) in [0,1].
    On start_date the factor matches the pre-gap (BUFF) basis; on end_date the
    post-gap full-consensus basis. Multiplies mean/median/min/max_price.
    """
    if prices.empty:
        return prices
    out = prices.copy()
    span = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days

    def _t(day):
        if span <= 0:
            return 0.0
        return (day - pd.Timestamp(start_date)).days / span

    t = out["day"].map(_t).clip(0.0, 1.0)
    f_start = out["item_slug"].map(start_factors).fillna(start_global)
    f_end = out["item_slug"].map(end_factors).fillna(end_global)
    factor = np.exp((1.0 - t) * np.log(f_start) + t * np.log(f_end))
    for col in ("mean_price", "median_price", "min_price", "max_price"):
        out[col] = out[col] * factor
    return out


def run(
    start: str,
    end: str,
    out_dir: Path,
    cache_dir: Path,
    dry_run: bool = False,
    refresh: bool = False,
    min_items: int = 24000,
    fetch=fetch_day,
) -> pd.DataFrame:
    dates = gap_dates(start, end)
    frames = []
    for d in dates:
        path = fetch(d, cache_dir, refresh=refresh)
        frames.append(transform_day(load_day(path), d))
    prices = pd.concat(frames, ignore_index=True)
    print(f"Transformed {len(prices):,} price rows over {len(dates)} days")

    start_f, start_g, end_f, end_g = compute_basis_factors(out_dir / "prices-*.parquet")
    prices = apply_rescale(prices, start_f, start_g, end_f, end_g, start, end)
    print(f"Rescaled (ramp): {len(start_f):,} per-item factors, start global {start_g:.3f}, end global {end_g:.3f}")

    validate_coverage(prices, dates, min_items=min_items)
    print("Coverage validation passed")

    if dry_run:
        print("Dry run — no files written")
        return prices

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = prices_to_snapshots(prices)
    # Route rows into monthly partitions (prices-YYYY-MM.parquet); a gap can
    # span several months so each row goes to the file for its own day.
    append_monthly(out_dir, "prices", prices[PRICE_COLS], DEDUP_KEYS)
    append_monthly(out_dir, "snapshots", snapshots[SNAP_COLS], DEDUP_KEYS)
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
    run(args.start_date, args.end_date, out_dir, cache_dir, dry_run=args.dry_run, refresh=args.refresh)


if __name__ == "__main__":
    main()
