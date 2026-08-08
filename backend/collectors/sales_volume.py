"""Completed-sale volume from Skinport `/v1/sales/history`.

**This is the only real trade-volume feed in the pipeline.** The aggregator
feeds carry no volume field at all (probed 2026-08-08: `steam.json` returns
last_24h/7d/30d/90d *prices*, `buff163.json` returns starting_at/highest_order,
`csgotrader.json` returns price + doppler), which is why `prices-*.parquet`'s
`volume` column has been unobserved since the one-shot HF merge ran out on
2026-04-15.

## Why a separate file rather than `prices-*.parquet`'s `volume`

That column already holds three different quantities under one name -- a real
Steam sale count pre-2026, an `ask_volume` *listing count* spliced in by
`merge_hf_dataset.py` for 2026-03-22..04-15, and fabricated zeros after. Writing
a fourth meaning into it would make the series unmodellable, so this writes
`volume-YYYY-MM.parquet` and leaves the dead column dead. Nothing here repairs
stored rows.

## What the numbers mean, and what they are not

Skinport reports **trailing-window** counts, not a daily series: `sales_24h` is
the count over the 24 hours *ending at collection*, `sales_7d` over 7 days, and
so on. The windows overlap and are cumulative, so differencing them across days
is not a daily sale count and `sales_7d` is not `7 * sales_24h`. `day` labels
the observation, exactly as `snapshot_day` does for supply depth -- it is not an
as-of parameter, because the endpoint has none.

These are **Skinport's** sales, one cash venue, not the whole market. Treat the
level as a liquidity screen and a counting-noise denominator, which is the use
the research supports: trade volume as a *predictor* is refuted at |r| < 0.002
over 4.47M rows and nothing here reopens that.

## Operational constraints

- Needs `Accept-Encoding: br`; Skinport answers **406** without it. `requests`
  only advertises brotli when a codec is importable, so `_probe_brotli` fails
  loudly rather than letting the feed read as blocked.
- Skinport's WAF answers **403 with an HTML challenge** to Cloudflare-owned
  egress (AS13335), including consumer WARP. That is about *where* you call
  from, not the endpoint -- check the egress IP before concluding it is down.
- Never run from `backend/` if you care where writes land: `.env` there points
  at production Supabase. Nothing in this module imports `database`, so it has
  no DB connection to inherit.
- The entry point must report row counts. `scripts/run_task.py`'s zero-row guard
  is the only thing standing between a dead collector and a green badge.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SKINPORT_SALES_URL = "https://api.skinport.com/v1/sales/history"
SOURCE = "skinport_sales"

USER_AGENT = "Mozilla/5.0 (compatible; CS2Oracle/1.0; +https://github.com/RayanR000)"
REQUEST_TIMEOUT_S = 120

VOLUME_COLUMNS = [
    "item_slug",
    "day",
    "source",
    "sales_24h",
    "sales_7d",
    "sales_30d",
    "sales_90d",
    "median_30d",
    "collected_at",
]

DEDUP_KEYS = ["item_slug", "day", "source"]

# Response key -> our column stem. Skinport nests one {min,max,avg,median,
# volume} object per trailing window.
WINDOW_KEYS = {
    "last_24_hours": "24h",
    "last_7_days": "7d",
    "last_30_days": "30d",
    "last_90_days": "90d",
}

# A payload that parses to nothing is far more likely to be a changed schema
# than a genuinely empty market, and a silently-empty collector is the failure
# mode this repo keeps hitting. Below this share of parsed rows, fail the run.
MIN_PARSE_RATE = 0.5


class SalesVolumeError(RuntimeError):
    """The feed failed or returned an unusable payload.

    Raised rather than returning empty so a broken feed cannot present as a feed
    with nothing to report.
    """


@dataclass
class SalesVolumeResult:
    rows: pd.DataFrame
    raw_items: int
    elapsed_s: float


def _probe_brotli() -> None:
    """Fail loudly if brotli is unavailable -- see the module docstring."""
    try:
        import brotli  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SalesVolumeError(
            "brotli is not installed, so requests will not send "
            "`Accept-Encoding: br` and Skinport will answer 406. Install it "
            "(pinned in backend/requirements.txt) rather than treating the "
            "Skinport feed as blocked."
        ) from exc


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out < 0:  # NaN or negative
        return None
    return out


def parse_sales_history(
    payload: Any,
    snapshot_day: date,
    collected_at: datetime,
) -> pd.DataFrame:
    """Reduce the Skinport payload to one row per item.

    Pure -- no network -- so the schema contract is unit-testable without an
    egress that Skinport will answer.

    Raises `SalesVolumeError` if the payload is not a list, or if fewer than
    `MIN_PARSE_RATE` of its entries yield a row. Returning a short frame instead
    would let a renamed key drain the feed while the run still reported success.
    """
    if not isinstance(payload, list):
        raise SalesVolumeError(
            f"expected a JSON array of items, got {type(payload).__name__}. "
            "The endpoint shape changed -- read a sample before parsing."
        )

    records = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = entry.get("market_hash_name")
        if not name or not isinstance(name, str):
            continue

        row: dict[str, Any] = {
            "item_slug": name,
            "day": snapshot_day,
            "source": SOURCE,
            "collected_at": collected_at,
        }
        # An item Skinport has never sold reports the window as null rather
        # than zero. That is a genuine "no sales", not a missing observation,
        # so it maps to 0 -- unlike the aggregator's unobserved volume, which
        # is NULL. The distinction is the entire point of this collector.
        seen_window = False
        for key, stem in WINDOW_KEYS.items():
            window = entry.get(key)
            if isinstance(window, dict):
                seen_window = True
                row[f"sales_{stem}"] = _coerce_int(window.get("volume"))
                if stem == "30d":
                    row["median_30d"] = _coerce_float(window.get("median"))
            elif key in entry:
                seen_window = True
                row[f"sales_{stem}"] = 0
                if stem == "30d":
                    row["median_30d"] = None

        if not seen_window:
            continue

        for stem in WINDOW_KEYS.values():
            row.setdefault(f"sales_{stem}", None)
        row.setdefault("median_30d", None)
        records.append(row)

    if not records:
        raise SalesVolumeError(
            f"parsed 0 rows from {len(payload)} payload entries. Expected keys "
            f"`market_hash_name` and one of {sorted(WINDOW_KEYS)} -- the "
            "response schema has almost certainly changed."
        )

    rate = len(records) / max(len(payload), 1)
    if rate < MIN_PARSE_RATE:
        raise SalesVolumeError(
            f"parsed only {len(records)} of {len(payload)} entries "
            f"({rate:.1%} < {MIN_PARSE_RATE:.0%}). Treating this as a schema "
            "change rather than a thin market."
        )

    frame = pd.DataFrame.from_records(records)
    for stem in WINDOW_KEYS.values():
        frame[f"sales_{stem}"] = frame[f"sales_{stem}"].astype("Int64")
    return frame[VOLUME_COLUMNS]


def fetch_sales_history(
    snapshot_day: date,
    collected_at: datetime,
    session: requests.Session | None = None,
) -> SalesVolumeResult:
    """Pull the whole-catalogue sales history in one call."""
    _probe_brotli()
    sess = session or requests.Session()
    started = time.monotonic()
    try:
        resp = sess.get(
            SKINPORT_SALES_URL,
            params={"app_id": 730, "currency": "USD"},
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "br, gzip, deflate"},
        )
    except requests.RequestException as exc:
        raise SalesVolumeError(f"{SKINPORT_SALES_URL}: request failed: {exc}") from exc

    if resp.status_code != 200:
        hint = ""
        if resp.status_code == 403 and resp.headers.get("server") == "cloudflare":
            hint = (
                " -- Cloudflare WAF challenge. This is an egress-ASN block "
                "(AS13335, which includes consumer WARP), not an outage. Check "
                "the caller's egress IP before concluding the feed is down."
            )
        elif resp.status_code == 406:
            hint = " -- 406 means brotli was not advertised; see _probe_brotli."
        raise SalesVolumeError(f"{SKINPORT_SALES_URL}: HTTP {resp.status_code}{hint}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise SalesVolumeError(
            f"{SKINPORT_SALES_URL}: response was not JSON ({len(resp.content)} bytes)"
        ) from exc

    rows = parse_sales_history(payload, snapshot_day, collected_at)
    elapsed = time.monotonic() - started
    logger.info(
        "  skinport_sales: %s payload entries -> %s items in %.1fs",
        len(payload), len(rows), elapsed,
    )
    return SalesVolumeResult(rows=rows, raw_items=len(payload), elapsed_s=elapsed)


def volume_parquet_path(archive_dir: Path, snapshot_day: date) -> Path:
    return archive_dir / f"volume-{snapshot_day:%Y-%m}.parquet"


def write_volume_rows(rows: pd.DataFrame, archive_dir: Path, snapshot_day: date) -> int:
    """Append to the month's partition, replacing any same-day/source rows.

    Re-running a day is idempotent: existing rows for the same
    (item_slug, day, source) lose to the run that just completed, so a partial
    run can be re-run rather than needing a repair step.
    """
    if rows.empty:
        return 0

    path = volume_parquet_path(archive_dir, snapshot_day)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = rows.copy()
    rows["day"] = pd.to_datetime(rows["day"]).dt.date

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="last")
    else:
        combined = rows

    combined = combined.sort_values(DEDUP_KEYS).reset_index(drop=True)
    combined.to_parquet(path, index=False, compression="zstd")
    return len(rows)


def collect(
    archive_dir: Path,
    snapshot_day: date | None = None,
    dry_run: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch, parse and persist one day's sale counts.

    Returns a summary carrying `volume_rows`, which `scripts/run_task.py`'s
    zero-row guard reads. A task that reports no counts defeats that guard.
    """
    if snapshot_day is None:
        from collectors.snapshot_date import resolve_snapshot_date
        snapshot_day = resolve_snapshot_date()

    collected_at = datetime.now(timezone.utc)
    result = fetch_sales_history(snapshot_day, collected_at, session=session)
    written = 0 if dry_run else write_volume_rows(result.rows, archive_dir, snapshot_day)

    rows = result.rows
    with_sales = int((rows["sales_30d"].fillna(0) > 0).sum())
    logger.info(
        "Sales volume %s: %s rows, %s items, %s with a 30d sale",
        snapshot_day, written, len(rows), with_sales,
    )
    return {
        "status": "success",
        "snapshot_day": str(snapshot_day),
        "volume_rows": written,
        "distinct_items": int(rows["item_slug"].nunique()),
        "items_with_30d_sale": with_sales,
        "raw_payload_entries": result.raw_items,
        "elapsed_seconds": round(result.elapsed_s, 2),
        "dry_run": dry_run,
    }
