"""
Free-data backfill helpers.

This module ingests public CS2 price archives and Steam announcement data to
seed the database without paid data sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import text
from database import Event, Item, PriceHistory, SessionLocal

logger = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CS2ShClient:
    """Client for cs2.sh archive access."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def enabled(self) -> bool:
        return bool(self.api_key)

    def fetch_archive_history(
        self,
        items: Iterable[str],
        start: datetime,
        end: Optional[datetime] = None,
        interval: str = "1d",
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch archive history. Production code can override this client."""
        return {"items": {}}


class SteamAnnouncementsImporter:
    """Importer for official Steam announcements."""

    def fetch_announcements(self) -> List[Dict[str, Any]]:
        return []


class HistoricalDataGenerator:
    """Synthetic fallback generator for items missing archive history."""

    @staticmethod
    def generate_historical_prices(
        item_name: str,
        release_date: datetime,
        end_date: Optional[datetime] = None,
        days_back: int = 365,
    ) -> List[tuple[datetime, float, int]]:
        """Generate a simple placeholder series."""
        end_date = end_date or _utcnow_naive()
        start_date = max(release_date, end_date - timedelta(days=days_back))

        prices = []
        current = start_date
        price = 1.0
        while current <= end_date:
            prices.append((current, price, 0))
            current += timedelta(days=1)
            price += 0.01
        return prices


@dataclass
class FreeDataBackfillImporter:
    """Backfills price history from public archive and synthetic sources."""

    api_key: Optional[str] = None
    cs2sh_client: Optional[CS2ShClient] = None
    announcements_importer: Optional[SteamAnnouncementsImporter] = None

    def __post_init__(self) -> None:
        self.cs2sh_client = self.cs2sh_client or CS2ShClient(api_key=self.api_key)
        self.announcements_importer = self.announcements_importer or SteamAnnouncementsImporter()

    def backfill_price_history(
        self,
        db=None,
        history_start: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """Backfill price history from cs2.sh plus synthetic gaps."""
        should_close = db is None
        db = db or SessionLocal()
        archive_rows_added = 0
        synthetic_rows_added = 0

        try:
            items = db.query(Item).all()
            if not items:
                return {"archive_rows_added": 0, "synthetic_rows_added": 0}

            item_names = [item.name for item in items]
            history_start = history_start or min(
                (item.release_date for item in items if item.release_date is not None),
                default=_utcnow_naive() - timedelta(days=365),
            )
            archive_end = _utcnow_naive()

            archive_payload = {}
            if self.cs2sh_client and self.cs2sh_client.enabled():
                archive_payload = self.cs2sh_client.fetch_archive_history(
                    item_names,
                    history_start,
                    end=archive_end,
                    interval="1d",
                    sources=["aggregate"],
                )

            items_payload = archive_payload.get("items", {})
            now = _utcnow_naive()

            # Build batch inserts to avoid re-inserting duplicates on re-run
            archive_rows = []
            synthetic_rows = []

            for item in items:
                item_payload = (
                    items_payload.get(item.name)
                    or items_payload.get(item.item_id)
                    or _find_matching_archive_item(items_payload, item.name, item.item_id)
                    or {}
                )
                for point in item_payload.get("data", []):
                    bucket = point.get("bucket")
                    aggregate = point.get("aggregate", {})
                    if not bucket:
                        continue

                    price = aggregate.get("ask") or aggregate.get("bid")
                    if price is None:
                        continue

                    timestamp = _parse_iso8601(bucket)
                    archive_rows.append({
                        "item_id": item.id,
                        "timestamp": timestamp,
                        "price": float(price),
                        "volume": int(aggregate.get("hourly_volume") or aggregate.get("sample_count") or 0),
                        "source": "cs2sh_archive",
                    })
                    archive_rows_added += 1

                if item.release_date is None:
                    continue

                synthetic_end = min(history_start - timedelta(days=1), now)
                if synthetic_end >= item.release_date:
                    syn_rows = HistoricalDataGenerator.generate_historical_prices(
                        item.name,
                        item.release_date,
                        end_date=synthetic_end,
                    )
                    for timestamp, sp, vol in syn_rows:
                        synthetic_rows.append({
                            "item_id": item.id,
                            "timestamp": timestamp,
                            "price": float(sp),
                            "volume": int(vol or 0),
                            "source": "synthetic_demo",
                        })
                        synthetic_rows_added += 1

            if archive_rows:
                db.execute(
                    text("""
                        INSERT INTO price_history (item_id, timestamp, price, volume, source)
                        VALUES (:item_id, :timestamp, :price, :volume, :source)
                        ON CONFLICT (item_id, timestamp, source) DO NOTHING
                    """),
                    archive_rows,
                )

            if synthetic_rows:
                db.execute(
                    text("""
                        INSERT INTO price_history (item_id, timestamp, price, volume, source)
                        VALUES (:item_id, :timestamp, :price, :volume, :source)
                        ON CONFLICT (item_id, timestamp, source) DO NOTHING
                    """),
                    synthetic_rows,
                )

            db.commit()
            return {
                "archive_rows_added": archive_rows_added,
                "synthetic_rows_added": synthetic_rows_added,
            }
        except Exception:
            db.rollback()
            raise
        finally:
            if should_close:
                db.close()

    def import_official_events(self, db=None) -> Dict[str, int]:
        """Import announcement events into the events table."""
        should_close = db is None
        db = db or SessionLocal()
        events_added = 0

        try:
            announcements = self.announcements_importer.fetch_announcements() if self.announcements_importer else []
            for announcement in announcements:
                title = announcement.get("title", "Steam Announcement")
                timestamp = announcement.get("timestamp") or datetime.now(timezone.utc)
                summary = announcement.get("summary", "")
                link = announcement.get("link", "")
                event_type = announcement.get("type", "update")
                description = f"{title}\n{summary}\n{link}".strip()

                db.add(
                    Event(
                        type=event_type,
                        timestamp=timestamp,
                        description=description,
                    )
                )
                events_added += 1

            db.commit()
            return {"events_added": events_added}
        except Exception:
            db.rollback()
            raise
        finally:
            if should_close:
                db.close()


def load_free_cs2_data(db=None, api_key: Optional[str] = None) -> Dict[str, int]:
    """Convenience wrapper used by startup scripts."""
    importer = FreeDataBackfillImporter(api_key=api_key)
    history_stats = importer.backfill_price_history(db=db)
    event_stats = importer.import_official_events(db=db)
    return {**history_stats, **event_stats}


def _parse_iso8601(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _find_matching_archive_item(items_payload: Dict[str, Any], item_name: str, item_id: str) -> Optional[Dict[str, Any]]:
    """Find the best matching payload when the archive key differs from local naming."""
    normalized_candidates = {item_name.lower(), item_id.lower()}
    for key, payload in items_payload.items():
        if key.lower() in normalized_candidates:
            return payload
        market_hash_name = payload.get("market_hash_name")
        if isinstance(market_hash_name, str) and market_hash_name.lower() in normalized_candidates:
            return payload

    if len(items_payload) == 1:
        return next(iter(items_payload.values()))

    return None
