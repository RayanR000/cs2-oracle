#!/usr/bin/env python3
"""Backfill BUFF price history + Steam volume from the free iflow archive.

Source: api.iflow.work `priority_archive` (EricZhu-42/SteamTradingSiteTracker-Data),
12h dumps 2022-04-18 -> 2026-05-20. We take ONE dump/day, CS items only, and
write two staging outputs under --out-dir/price-archive:

  prices-YYYY-MM.parquet   canonical rows, source="buff_iflow", BUFF ask in USD
  volume-iflow-YYYY-MM.parquet   item_slug, day, steam_volume, buff_buy_num,
                                 buff_sell_num  (the liquidity re-test inputs)

Two dump schemas are handled (the 2024-02-13 DB restructure):
  - <=2024-02-13  "DATA" db:     buff_reference_price (CNY), count_in_24
  - >=2024-02-13  "priority" db: buff_sell.price (CNY), steam_volume.volume

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
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict

from collectors.price_history_import import to_archive_frame, write_archive_frame
from db.archive import prices_relation
from db.parquet import append_monthly

SOURCE = "buff_iflow"
LIST_URL = "https://api.iflow.work/export/list?dir_name=priority_archive"
DL_URL = "https://api.iflow.work/export/download?dir_name=priority_archive&file_name={}"
ARCHIVE = Path(__file__).parent.parent.parent / "price-archive"
CACHE = Path(__file__).parent.parent / "runtime" / "iflow_cache"
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


def load_fx() -> "pd.DataFrame":
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


def parse_dump(raw: bytes, day: date) -> list[dict]:
    """CS records from one dump -> [{slug, buff_cny, steam_vol, buy_num, sell_num}]."""
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
                steam_vol = r.get("count_in_24")
                buy_num = r.get("buff_buy_num")
                sell_num = r.get("buff_sell_num")
            else:
                bs = r.get("buff_sell") or {}
                cny = bs.get("price")
                sell_num = bs.get("count")
                buy_num = (r.get("buff_buy") or {}).get("count")
                steam_vol = (r.get("steam_volume") or {}).get("volume")
            if cny is None or cny <= 0:
                continue
            out.append({"slug": slug, "cny": float(cny), "steam_vol": steam_vol,
                        "buy_num": buy_num, "sell_num": sell_num})
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
    sub = fx[fx["day"] <= day]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    if (day - row["day"]).days > 7:  # MAX_FILL_DAYS
        return None
    return float(row["rate"])


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

    def work(day: date):
        return day, parse_dump(fetch(day, dumps[day], args.refresh), day)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for day, recs in ex.map(work, days):
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
                vol_rows.append({"item_slug": r["slug"], "day": day,
                                 "steam_volume": r["steam_vol"],
                                 "buff_buy_num": r["buy_num"],
                                 "buff_sell_num": r["sell_num"]})
            done += 1
            if done % 100 == 0:
                print(f"  parsed {done}/{len(days)} days, {len(price_rows)} price rows")

    print(f"\nkept {len(price_rows)} price rows over {done} days")
    print(f"dropped: {dict(stats)}")
    if not price_rows:
        print("nothing to write.")
        return

    if args.dry_run:
        pr = pd.DataFrame(price_rows, columns=["item_slug", "day", "usd"])
        print(f"\n[dry-run] distinct items={pr['item_slug'].nunique()}, "
              f"day span {pr['day'].min()}..{pr['day'].max()}, "
              f"median price ${pr['usd'].median():.2f}")
        return

    frame = to_archive_frame(price_rows, SOURCE, datetime.utcnow())
    n = write_archive_frame(frame, out_prices)
    vol = pd.DataFrame(vol_rows)
    vol["source"] = SOURCE
    append_monthly(out_prices, "volume-iflow", vol, ["item_slug", "day", "source"])
    print(f"\nwrote {n} price rows (source={SOURCE}) + {len(vol)} volume rows "
          f"-> {out_prices}")


if __name__ == "__main__":
    main()
