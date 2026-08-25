#!/usr/bin/env python3
"""
DB-free daily snapshot export for the local archive workaround.

Produces the same two CSVs the DB-coupled aggregate step writes
(`collectors/pipeline.py`), but WITHOUT touching Supabase: it downloads the
full csgotrader.app dumps into `CSGOTraderAggregator._raw_sources` and emits
every item each source tracks, keyed by that source's own item key (the same
`item_slug` namespace the archive uses). `append_to_parquet.py --snapshot-csv`
dedups on (item_slug, day, source), so the raw rows drop straight in.

What it deliberately omits vs the DB path (each needs the `items` table):
  - DB-matched rows keyed by the DB slug — a subset of the raw rows, same
    namespace, so no loss for items present in today's dump.
  - Historical carry-forward for items absent from today's dump — they get no
    row today instead of a re-stamped stale price.
  - The separate backfilled/OHLCV CSV (needs `is_backfilled`).
  - `collection_runs` bookkeeping.

Usage:
    python scripts/export_snapshot_dbfree.py            # resolves snapshot day
    python scripts/export_snapshot_dbfree.py --date 2026-08-23
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.csgotrader_aggregator import CSGOTraderAggregator, _get_safe
from collectors.snapshot_date import resolve_snapshot_date

logger = logging.getLogger(__name__)

# `None` -> csv.writer emits an empty field -> pandas reads NaN -> the archive
# stores NULL. No aggregator feed carries a volume field, so every live row's
# volume is unknown (never 0). Mirrors collectors/pipeline.py.
VOLUME_NOT_OBSERVED = None

# Source label mapping for storage. Mirrors collectors/pipeline.py:146.
SOURCE_LABELS = {
    "steam": "aggregator_sync",
    "steam_7d": "aggregator_steam_7d",
    "steam_30d": "aggregator_steam_30d",
    "steam_90d": "aggregator_steam_90d",
    "skinport": "aggregator_skinport",
    "buff163": "aggregator_buff163",
    "buff163_buy": "aggregator_buff163_buy",
    "csfloat": "aggregator_csfloat",
    "csmoney": "aggregator_csmoney",
    "csgotrader": "aggregator_csgotrader",
    "youpin": "aggregator_youpin",
}


def write_snapshot_csv(aggregator: CSGOTraderAggregator, agg_date: str,
                       snapshot_csv_path: str) -> int:
    """Write every raw dump row to a snapshot CSV. Returns the row count.

    Reproduces the raw-source branch of
    `collectors/pipeline.py::run_full_aggregator_collection` (:298-375) with an
    empty `written` set — there is no DB-matched pass to dedup against here.
    """
    raw_count = 0
    with open(snapshot_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_slug", "day", "source", "price", "volume"])
        for src_name, src_data in aggregator._raw_sources.items():
            if not isinstance(src_data, dict):
                continue
            for item_key, info in src_data.items():
                if src_name == "steam":
                    p24 = _get_safe(info.get("last_24h")) if isinstance(info, dict) else _get_safe(info)
                    p7 = _get_safe(info.get("last_7d")) if isinstance(info, dict) else None
                    p30 = _get_safe(info.get("last_30d")) if isinstance(info, dict) else None
                    p90 = _get_safe(info.get("last_90d")) if isinstance(info, dict) else None
                    for label, p in [
                        ("aggregator_sync", p24 if p24 is not None else (p7 if p7 is not None else (p30 if p30 is not None else p90))),
                        # Clean Steam spot: last_24h with NO fallback, empty on
                        # illiquid items where aggregator_sync degrades to a
                        # trailing mean. The cross-venue basis feature's Steam leg.
                        ("aggregator_steam_spot", p24),
                        ("aggregator_steam_7d", p7),
                        ("aggregator_steam_30d", p30),
                        ("aggregator_steam_90d", p90),
                    ]:
                        if p is not None and p > 0:
                            w.writerow([item_key, agg_date, label, p, VOLUME_NOT_OBSERVED])
                            raw_count += 1
                    continue

                if not isinstance(info, dict):
                    p = _get_safe(info)
                    if p is not None and p > 0:
                        label = SOURCE_LABELS.get(src_name, f"aggregator_{src_name}")
                        w.writerow([item_key, agg_date, label, p, VOLUME_NOT_OBSERVED])
                        raw_count += 1
                    continue

                if src_name == "skinport":
                    p = _get_safe(info.get("starting_at"))
                elif src_name == "buff163":
                    starting = info.get("starting_at")
                    p = _get_safe(starting.get("price") if isinstance(starting, dict) else starting)
                    ho = info.get("highest_order")
                    if isinstance(ho, dict):
                        ho_p = _get_safe(ho.get("price"))
                        if ho_p is not None and ho_p > 0:
                            w.writerow([item_key, agg_date, "aggregator_buff163_buy", ho_p, VOLUME_NOT_OBSERVED])
                            raw_count += 1
                elif src_name == "csfloat":
                    p = _get_safe(info.get("price"))
                elif src_name == "csmoney":
                    p = _get_safe(info.get("price"))
                elif src_name == "csgotrader":
                    p = _get_safe(info.get("price"))
                elif src_name == "youpin":
                    p = _get_safe(info.get("price")) or _get_safe(info)
                else:
                    p = _get_safe(info.get("price"))

                if p is not None and p > 0:
                    label = SOURCE_LABELS.get(src_name, f"aggregator_{src_name}")
                    w.writerow([item_key, agg_date, label, p, VOLUME_NOT_OBSERVED])
                    raw_count += 1

    return raw_count


def write_exchange_rates_csv(rates: dict, agg_date: str, path: str) -> int:
    """Write exchange rates to CSV. Returns the row count."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["currency", "rate", "day"])
        for currency, rate in rates.items():
            writer.writerow([currency, rate, agg_date])
    return len(rates)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Snapshot day (YYYY-MM-DD, default: resolve_snapshot_date())")
    args = parser.parse_args()

    agg_date = args.date or resolve_snapshot_date().strftime("%Y-%m-%d")

    aggregator = CSGOTraderAggregator()
    aggregator.fetch_all_market_data()
    if not aggregator._raw_sources:
        logger.critical("No market data fetched — refusing to write an empty snapshot")
        sys.exit(1)

    snapshot_csv_path = f"/tmp/aggregator-snapshots-{agg_date}.csv"
    n = write_snapshot_csv(aggregator, agg_date, snapshot_csv_path)
    logger.info("Wrote %s snapshot rows to %s (all raw sources)", n, snapshot_csv_path)
    if n == 0:
        logger.critical("Snapshot CSV is empty — refusing (zero-row guard)")
        sys.exit(1)

    rates = aggregator.fetch_exchange_rates()
    if rates:
        rates_path = f"/tmp/exchange-rates-{agg_date}.csv"
        m = write_exchange_rates_csv(rates, agg_date, rates_path)
        logger.info("Wrote %s exchange rates to %s", m, rates_path)
    else:
        logger.warning("No exchange rates fetched — continuing without them")

    print(f"Snapshot day: {agg_date}")


if __name__ == "__main__":
    main()
