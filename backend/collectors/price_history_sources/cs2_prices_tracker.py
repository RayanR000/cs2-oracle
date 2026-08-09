"""`LukeX404/cs2-prices-tracker` — dated daily Steam price files on GitHub.

The repo commits one JSON per day under ``static/prices/date/``. Schema is
``{market_hash_name: {steam: {last_24h, last_7d, last_30d, last_90d}}}``.

**Only `last_24h` is read.** It is Steam's 24-hour average sale price — the same
quantity as ``aggregator_sync``'s primary field, so it needs no basis conversion
and is a legitimate ask-side vote. ``last_7d``/``30d``/``90d`` are trailing-window
means and are exactly the feeds documented as voting against point-in-time asks;
importing them would repeat that error.

Keys are the raw ``market_hash_name``, which is what the archive uses as
``item_slug`` (verified 2026-08-08: 0 key-format failures across 24,238 names).
"""
from datetime import date

#: Distinct label so the rows stay attributable and a read-time filter can drop
#: them without rewriting the archive. Not a bid, so NOT in ``BID_SOURCES``.
SOURCE = "tracker_steam_24h"

_BASE = (
    "https://raw.githubusercontent.com/LukeX404/cs2-prices-tracker/"
    "main/static/prices/date"
)


def day_url(day: date) -> str:
    """The raw-content URL for one dated price file."""
    return f"{_BASE}/{day.isoformat()}.json"


def parse_day(payload: dict, day: date) -> list[tuple[str, date, float]]:
    """``(item_slug, day, price)`` for every item priced on *day*.

    A null ``last_24h`` is an ABSENT row, not a zero — the item simply did not
    trade. Non-positive prices are dropped for the same reason: a fabricated
    zero is what defeated ``has_volume``/``volume_missing`` and shelved eleven
    features, and the same trap applies to price.
    """
    records: list[tuple[str, date, float]] = []
    for name, entry in (payload or {}).items():
        steam = (entry or {}).get("steam") or {}
        price = steam.get("last_24h")
        if price is None:
            continue
        price = float(price)
        if price <= 0:
            continue
        records.append((name, day, price))
    return records
