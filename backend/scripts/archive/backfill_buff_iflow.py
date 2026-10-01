#!/usr/bin/env python3
"""Backfill BUFF price history + Steam volume from the free iflow archive.

Source: api.iflow.work `priority_archive` (EricZhu-42/SteamTradingSiteTracker-Data),
12h dumps 2022-04-18 -> 2026-05-20. We take ONE dump/day, CS items only, and
write two staging outputs under --out-dir/price-archive:

  prices-YYYY-MM.parquet   canonical rows, source="buff_iflow", BUFF ask in USD
  volume-iflow-YYYY-MM.parquet   item_slug, day,
                                 count_in_24_pre_restructure,
                                 steam_volume_post_restructure,
                                 buff_buy_num, buff_sell_num

Two dump schemas are handled (the 2024-02-13 DB restructure):
  - <=2024-02-13  "DATA" db:     buff_reference_price (CNY), count_in_24
  - >=2024-02-13  "priority" db: buff_sell.price (CNY), steam_volume.volume

The two volume series are NEVER collapsed into one column: they come from
different upstreams on either side of an era boundary and are incomparable
(docs/specs/2026-09-14-wait-window-workplan.md Item 4). Consumers must pick an
era explicitly; there is no merged `steam_vol`/`steam_volume` output.

BUFF prices are raw CNY; converted to USD via exchange-rates-history.parquet
(rate = CNY per USD, so usd = cny / rate), forward-filled.

Usage (from backend/):
  venv/bin/python scripts/backfill_buff_iflow.py --dry-run --start 2026-04-01 --end 2026-04-07
  venv/bin/python scripts/backfill_buff_iflow.py --out-dir ../buff-iflow-staging
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import defaultdict

from collectors.price_history_import import to_archive_frame, write_archive_frame
from db.archive import prices_relation
from db.parquet import append_monthly

SOURCE = "buff_iflow"
LIST_URL = "https://api.iflow.work/export/list?dir_name=priority_archive"
DL_URL = "https://api.iflow.work/export/download?dir_name=priority_archive&file_name={}"
ARCHIVE = Path(__file__).resolve().parent.parent.parent.parent / "price-archive"
CACHE = Path(__file__).resolve().parents[2] / "runtime" / "iflow_cache"
RESTRUCTURE = date(2024, 2, 13)  # schema boundary
MIN_USD = 0.50  # drop junk below this; the >=$1 training gate is downstream
_SESSION = requests.Session()


def list_dumps() -> dict[date, str]:
    """One dump per calendar day (the earliest that day), day -> filename."""
    files = _SESSION.get(LIST_URL, timeout=60).json()["files"]
    by_day: dict[date, str] = {}
    for f in sorted(files):  # "YYYY-MM-DD-HH-MM.zip"
        d = date.fromisoformat(f[:10])
        by_day.setdefault(d, f)  # keep earliest HH-MM of the day
    return by_day


def load_fx() -> pd.DataFrame:
    fx = pd.read_parquet(ARCHIVE / "exchange-rates-history.parquet")
    fx = fx[fx["currency"] == "CNY"][["day", "rate"]].copy()
    fx["day"] = pd.to_datetime(fx["day"]).dt.date
    return fx.sort_values("day").reset_index(drop=True)


def load_known_slugs() -> set[str]:
    """The archive's real item universe (applies the universe filter)."""
    con = duckdb.connect()
    try:
        rel = prices_relation(con, archive_dir=ARCHIVE, columns=["item_slug"])
        df = con.sql(f"SELECT DISTINCT item_slug FROM {rel}").fetchdf()
    finally:
        con.close()
    return set(df["item_slug"])


# Era-explicit volume output columns. The pre/post series share no column by
# design (Item 4): a row may populate at most one of them, chosen by dump era.
VOL_PRE_RESTRUCTURE = "count_in_24_pre_restructure"  # DATA db `count_in_24`, day <= RESTRUCTURE
VOL_POST_RESTRUCTURE = "steam_volume_post_restructure"  # priority db `steam_volume.volume`, day > RESTRUCTURE


def parse_dump(raw: bytes, day: date) -> list[dict]:
    """CS records from one dump.

    Each record carries exactly one eligible volume column: pre-restructure
    days populate VOL_PRE_RESTRUCTURE (the other key is None) and
    post-restructure days populate VOL_POST_RESTRUCTURE. The two series are
    never merged -- a downstream consumer that wants volume must pick an era.
    """
    zf = zipfile.ZipFile(io.BytesIO(raw))
    old = day <= RESTRUCTURE
    out = []
    with zf.open(zf.namelist()[0]) as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("appid") != 730:  # CS only (570 = Dota)
                continue
            slug = r.get("hash_name")
            if not slug:
                continue
            if old:
                cny = r.get("buff_reference_price")
                vol_pre = r.get("count_in_24")
                vol_post = None
                buy_num = r.get("buff_buy_num")
                sell_num = r.get("buff_sell_num")
            else:
                bs = r.get("buff_sell") or {}
                cny = bs.get("price")
                sell_num = bs.get("count")
                buy_num = (r.get("buff_buy") or {}).get("count")
                vol_pre = None
                vol_post = (r.get("steam_volume") or {}).get("volume")
            if cny is None or cny <= 0:
                continue
            out.append(
                {
                    "slug": slug,
                    "cny": float(cny),
                    VOL_PRE_RESTRUCTURE: vol_pre,
                    VOL_POST_RESTRUCTURE: vol_post,
                    "buy_num": buy_num,
                    "sell_num": sell_num,
                }
            )
    return out


def fetch(day: date, fname: str, refresh: bool) -> bytes:
    cache_f = CACHE / fname
    if cache_f.exists() and not refresh:
        return cache_f.read_bytes()
    raw = _SESSION.get(DL_URL.format(fname), timeout=180).content
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_f.write_bytes(raw)
    return raw


def fx_lookup(fx: pd.DataFrame, day: date) -> float | None:
    """Latest FX fixing on or before *day* (forward fill), O(log n) via searchsorted."""
    if fx.empty:
        return None
    idx = int(np.searchsorted(fx["day"].values, day, side="right")) - 1
    if idx < 0:
        return None
    row = fx.iloc[idx]
    if (day - row["day"]).days > 7:  # MAX_FILL_DAYS
        return None
    return float(row["rate"])


def _flush_rows(price_rows: list, vol_rows: list, out_prices: Path) -> tuple[int, int]:
    """Write the accumulated rows through the existing per-month write path.

    `write_archive_frame`/`append_monthly` already fan out by month — this just
    bounds peak memory by calling them as each month completes instead of once
    at the end. Returns (price rows, volume rows) written. Clears the lists.
    """
    n_price = n_vol = 0
    if price_rows:
        frame = to_archive_frame(price_rows, SOURCE, datetime.utcnow())
        n_price = write_archive_frame(frame, out_prices)
        price_rows.clear()
    if vol_rows:
        vol = pd.DataFrame(vol_rows)
        vol["source"] = SOURCE
        append_monthly(out_prices, "volume-iflow", vol, ["item_slug", "day", "source"])
        n_vol = len(vol)
        vol_rows.clear()
    return n_price, n_vol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../buff-iflow-staging")
    ap.add_argument("--start", default="2022-04-18")
    ap.add_argument("--end", default="2026-05-20")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="parse + report, no write")
    ap.add_argument("--refresh", action="store_true", help="re-download cached dumps")
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    out_prices = Path(args.out_dir) / "price-archive"
    fx = load_fx()
    known = load_known_slugs()
    print(f"universe: {len(known)} known item_slugs; FX rows: {len(fx)}")

    dumps = {d: f for d, f in list_dumps().items() if start <= d <= end}
    days = sorted(dumps)
    print(f"days to pull: {len(days)}  ({days[0]} .. {days[-1]})\n")

    price_rows: list[tuple[str, date, float]] = []
    vol_rows: list[dict] = []
    stats = defaultdict(int)
    current_month: tuple[int, int] | None = None
    kept_total = 0
    wrote_price = 0
    wrote_vol = 0

    def work(day: date):
        return day, parse_dump(fetch(day, dumps[day], args.refresh), day)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for day, recs in ex.map(work, days):
            if not args.dry_run:
                month_key = (day.year, day.month)
                if current_month is not None and month_key != current_month and (price_rows or vol_rows):
                    n_p, n_v = _flush_rows(price_rows, vol_rows, out_prices)
                    wrote_price += n_p
                    wrote_vol += n_v
                current_month = month_key
            rate = fx_lookup(fx, day)
            if rate is None:
                stats["days_no_fx"] += 1
                continue
            for r in recs:
                if r["slug"] not in known:
                    stats["dropped_unknown"] += 1
                    continue
                usd = r["cny"] / rate
                if usd < MIN_USD:
                    stats["dropped_cheap"] += 1
                    continue
                price_rows.append((r["slug"], day, round(usd, 4)))
                kept_total += 1
                vol_rows.append(
                    {
                        "item_slug": r["slug"],
                        "day": day,
                        VOL_PRE_RESTRUCTURE: r[VOL_PRE_RESTRUCTURE],
                        VOL_POST_RESTRUCTURE: r[VOL_POST_RESTRUCTURE],
                        "buff_buy_num": r["buy_num"],
                        "buff_sell_num": r["sell_num"],
                    }
                )
            done += 1
            if done % 100 == 0:
                print(f"  parsed {done}/{len(days)} days, {kept_total} price rows")

    print(f"\nkept {kept_total} price rows over {done} days")
    print(f"dropped: {dict(stats)}")
    if kept_total == 0:
        print("nothing to write.")
        return

    if args.dry_run:
        pr = pd.DataFrame(price_rows, columns=["item_slug", "day", "usd"])
        print(
            f"\n[dry-run] distinct items={pr['item_slug'].nunique()}, "
            f"day span {pr['day'].min()}..{pr['day'].max()}, "
            f"median price ${pr['usd'].median():.2f}"
        )
        return

    n_p, n_v = _flush_rows(price_rows, vol_rows, out_prices)
    wrote_price += n_p
    wrote_vol += n_v
    print(f"\nwrote {wrote_price} price rows (source={SOURCE}) + {wrote_vol} volume rows -> {out_prices}")


if __name__ == "__main__":
    main()
