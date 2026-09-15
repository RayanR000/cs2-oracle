#!/usr/bin/env python3
"""Ingest daily USD-base FX history into price-archive/exchange-rates-history.parquet.

## Why this exists, and why it is a control column rather than a bet

`aggregator_buff163` and `aggregator_youpin` are the two largest live sources by
item count and both are CNY-denominated Chinese marketplaces whose prices reach
this archive already converted to USD. Arbitrage against Western venues is weak --
BUFF163 runs a persistent 10-30% discount to Steam that does not close, because
withdrawal and payment rails (Alipay/WeChat Pay, capital controls) are the binding
friction for non-Chinese users. Weak arbitrage means a currency move is *not*
absorbed by CNY prices adjusting to hold the USD price constant, so it passes
through into the USD series as a common-mode term that carries no information
about any skin.

That is the mechanism. **The magnitude is small, and it was measured here rather
than cited.** Over the 3,478 daily returns in this table's own window:

    CNY/USD daily sd        0.2307%   (annualised 3.66%)
    last 504 trading days   0.1711%   (annualised 2.72%)
    largest 1-day move      1.99%
    median |30-day move|    0.77%     (p95 3.30%)

USD/CNY is a PBoC-managed float with a counter-cyclical factor in the daily fixing,
and it is one of the lowest-volatility major pairs there is. Against skin moves this
is noise-level at 7d/14d/30d. The one place it could matter is 3d direction on
sticky, illiquid items, where a common 0.23% shift is a real fraction of a typical
daily move and can flip labels that sit near zero -- and 3d is also where the gate's
MDE is tightest. That is a narrow case, not a thesis.

So: ingested because it is one call and a few columns, wired nowhere, and expected
to carry nothing on its own. **No lift is claimed.**
`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` still excludes it.

## Why a separate file

`price-archive/exchange-rates-2026.parquet` is owned by the daily aggregator
(`collectors/pipeline.py::fetch_exchange_rates` -> `scripts/append_to_parquet.py`),
holds `currency, rate, day` sourced from the csgotrader dump, and starts 2026-07-11.
Writing history into it would put a second writer on a file the daily chain appends
to. This is a separate table with its own name and an extra `is_filled` column.

## Weekend and holiday fill, and the RUB trap

FX does not trade at weekends; skins do. Rates are forward-filled onto every
calendar day so a price row can always join, with `is_filled = 1` marking a carried
value. Forward-fill only ever propagates a rate to *later* days, so no future rate
reaches a past row.

The fill is capped at `MAX_FILL_DAYS`, and that cap is load-bearing rather than
tidiness. **The ECB stopped quoting RUB on 2022-03-01** (sanctions), so an uncapped
forward-fill carried one frozen number across the following four years -- 2,621 of
its 4,966 days. A constant is not a rate, and a model cannot tell the difference:
it would read as a clean, perfectly stable feature. With the cap, a currency simply
stops having rows once its source stops, which is the honest shape. Any currency
added here needs the same check -- print the per-currency quote range before
trusting a series.

Usage:
    python scripts/ingest_fx_history.py
    python scripts/ingest_fx_history.py --offline
    python scripts/ingest_fx_history.py --stats-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_fx_history")

# Frankfurter proxies the ECB reference rates. Free, no key, no quota published;
# the whole 2013-present range is one ~190 KB call in about a second.
FRANKFURTER_URL = "https://api.frankfurter.dev/v1"
CACHE_DIR = Path(__file__).parent.parent / "runtime" / "fx"
PRICE_ARCHIVE = Path(__file__).parent.parent.parent / "price-archive"
OUTPUT_PARQUET = PRICE_ARCHIVE / "exchange-rates-history.parquet"

# CNY is the one with a mechanism (above). EUR and GBP are the other two currencies
# the aggregator's own rate dump carries, kept so this table can stand in for it.
# RUB is here because market.csgo.com and several CIS-facing venues quote in it.
SYMBOLS = ("CNY", "EUR", "GBP", "RUB")
START = date(2013, 1, 1)

# Longest run of calendar days a quote may be carried across. Christmas/New Year
# closes the ECB for at most a few consecutive days, so 7 covers every real gap
# while refusing to manufacture a series out of a dead one -- see the RUB trap in
# the module docstring.
MAX_FILL_DAYS = 7

OUTPUT_COLUMNS = ("day", "currency", "rate", "is_filled")


def fetch_rates(
    start: date, end: date, cache_dir: Path, offline: bool = False, session: requests.Session | None = None
) -> dict[str, dict]:
    """Fetch the full daily series in one range call, USD as base."""
    cache = cache_dir / f"frankfurter-{start}-{end}.json"
    if offline:
        candidates = sorted(cache_dir.glob("frankfurter-*.json")) if cache_dir.exists() else []
        if not candidates:
            raise FileNotFoundError(f"--offline but no cache in {cache_dir}. Run once without it.")
        payload = json.loads(candidates[-1].read_text())
        logger.info(f"Loaded cached rates from {candidates[-1]}")
        return payload.get("rates", {})

    session = session or requests.Session()
    url = f"{FRANKFURTER_URL}/{start.isoformat()}..{end.isoformat()}"
    resp = session.get(url, params={"base": "USD", "symbols": ",".join(SYMBOLS)}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    rates = payload.get("rates", {})
    logger.info(f"Fetched {len(rates):,} quoted days -> {cache}")
    return rates


def build_frame(rates: dict[str, dict], start: date, end: date) -> pd.DataFrame:
    """Long-format `day, currency, rate, is_filled`, dense over every calendar day."""
    if not rates:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    quoted = (
        pd.DataFrame([{"day": pd.Timestamp(d), **vals} for d, vals in sorted(rates.items())])
        .set_index("day")
        .sort_index()
    )

    dense = quoted.reindex(pd.date_range(start, end, freq="D"))
    filled = dense.ffill(limit=MAX_FILL_DAYS)

    frames = []
    for cur in [c for c in SYMBOLS if c in filled.columns]:
        col = filled[cur]
        frames.append(
            pd.DataFrame(
                {
                    "day": col.index.date,
                    "currency": cur,
                    "rate": col.to_numpy(),
                    # A day is "filled" when the source had no quote for it (weekend,
                    # holiday) but a prior quote was carried forward.
                    "is_filled": dense[cur].isna().to_numpy().astype("int64"),
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    # Leading days before the first ECB quote have no rate to carry; drop rather
    # than back-fill, which would put a future rate on a past day.
    out = out[out["rate"].notna()].reset_index(drop=True)
    return out[list(OUTPUT_COLUMNS)]


def report_stats(df: pd.DataFrame) -> None:
    """Print per-currency depth and realised daily volatility.

    The CNY line is the number the module docstring's demotion argument rests on,
    so it is recomputed from the data on every run rather than quoted from a doc.
    """
    logger.info(f"Rows: {len(df):,}  days {df['day'].min()} -> {df['day'].max()}")
    for cur, sub in df.groupby("currency"):
        sub = sub.sort_values("day")
        quoted = sub[sub["is_filled"] == 0]
        vals = quoted["rate"].to_numpy()
        rets = [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals))]
        if len(rets) < 2:
            logger.info(f"    {cur}: {len(sub):,} days, too few quotes for vol")
            continue
        sd = statistics.pstdev(rets)
        logger.info(
            f"    {cur}: {len(sub):,} days ({len(quoted):,} quoted, "
            f"{len(sub) - len(quoted):,} filled)  "
            f"daily sd {sd * 100:.4f}%  annualised {sd * math.sqrt(252) * 100:.2f}%  "
            f"max 1d {max(abs(r) for r in rets) * 100:.2f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest daily USD-base FX history")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--out", default=str(OUTPUT_PARQUET))
    parser.add_argument("--start", default=START.isoformat())
    parser.add_argument("--stats-only", action="store_true", help="Report and exit without writing")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = datetime.now(UTC).date()

    rates = fetch_rates(start, end, Path(args.cache_dir), offline=args.offline)
    df = build_frame(rates, start, end)
    report_stats(df)

    if args.stats_only:
        logger.info("--stats-only: nothing written")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"Wrote {out} ({out.stat().st_size / 1e3:.1f} KB, {len(df):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
