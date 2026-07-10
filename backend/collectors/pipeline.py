"""
Data pipeline orchestration
Manages scheduled data collection, validation, and storage
"""

import csv
import logging
import os
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Any, Optional, List, Dict
from sqlalchemy import bindparam, func, text
from sqlalchemy.orm import scoped_session

logger = logging.getLogger(__name__)

QUALITY_SUFFIXES = (
    "(Factory New)",
    "(Minimal Wear)",
    "(Field-Tested)",
    "(Well-Worn)",
    "(Battle-Scarred)",
)
SPECIAL_PREFIXES = ("StatTrak", "Souvenir")


def _historical_fallback_source(source: str) -> str:
    """Normalize fallback source labels so retries do not stack prefixes."""
    while source.startswith("historical_fallback:"):
        source = source[len("historical_fallback:"):]
    return f"historical_fallback:{source}"


class DataPipeline:
    """Orchestrates data collection and processing pipeline"""
    
    def __init__(self, db_session=None):
        """
        Initialize data pipeline
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session

    def _ensure_session(self):
        """Return a thread-safe session. Creates a thread-local one if none was injected."""
        if self.db_session is None:
            from database import SessionLocal
            self.db_session = scoped_session(SessionLocal)
        return self.db_session
    
    def run_priority_collection(self):
        """Execute priority collection for top 2000 items using fast aggregator"""
        return self.run_full_aggregator_collection(limit=2000)

    def run_full_aggregator_collection(self, limit: Optional[int] = None):
        """
        Execute collection for items using fast aggregator.
        If limit is None, updates ALL items in the database.
        Records execution to collection_runs table for monitoring.
        """
        from collectors.csgotrader_aggregator import CSGOTraderAggregator
        from database import Item, PriceHistory, CollectionRun

        start_time = datetime.utcnow()
        primary_items_collected = 0
        fallback_items_collected = 0
        errors_count = 0
        duplicate_name_count = 0
        duplicate_name_sample: List[str] = []
        missing_names: List[str] = []
        missing_name_report: Dict[str, Any] = {}

        try:
            logger.info(f"Starting aggregator collection (limit: {limit if limit else 'ALL'})")

            if not self.db_session:
                logger.error("Database session not available")
                return {"status": "failed", "error": "No database session"}

            # 1. Fetch items
            query = self.db_session.query(Item)
            if limit:
                # Use liquidity-based sorting for limited runs
                from repositories import ItemRepository
                items = ItemRepository.get_top_items(self.db_session, limit=limit)
            else:
                items = query.all()

            if not items:
                logger.warning("No items found to update.")
                return {"status": "skipped", "reason": "no_items"}

            logger.info(f"Found {len(items)} items to update")

            name_counts = Counter(item.name for item in items)
            duplicate_names = [name for name, count in name_counts.items() if count > 1]
            duplicate_name_count = len(duplicate_names)
            if duplicate_names:
                duplicate_name_sample = duplicate_names[:10]
                logger.warning(
                    "Found %s duplicate item names in the DB snapshot. "
                    "These rows will share the same external lookup key. Sample: %s",
                    duplicate_name_count,
                    duplicate_name_sample,
                )

            # 2. Fetch prices from aggregator
            aggregator = CSGOTraderAggregator()
            item_map = defaultdict(list)
            for item in items:
                item_map[item.name].append(item)

            results = aggregator.collect_batch_items(list(item_map.keys()))

            # Source label mapping for storage
            SOURCE_LABELS = {
                "steam": "aggregator_sync",
                "steam_7d": "aggregator_steam_7d",
                "steam_30d": "aggregator_steam_30d",
                "steam_90d": "aggregator_steam_90d",
                "skinport": "aggregator_skinport",
                "buff163": "aggregator_buff163",
                "buff163_buy": "aggregator_buff163_buy",
                "csfloat": "aggregator_csfloat",
            }

            # 3. Store results
            price_records = []
            now = datetime.utcnow()

            for name, matched_items in item_map.items():
                sources = results.get(name)
                if sources:
                    has_primary = False
                    for src_key, (price, volume, _ts) in sources.items():
                        if price is not None and price > 0:
                            db_source = SOURCE_LABELS.get(src_key, f"aggregator_{src_key}")
                            for item in matched_items:
                                price_records.append(PriceHistory(
                                    item_id=item.id,
                                    timestamp=now,
                                    price=price,
                                    volume=volume if volume is not None else 0,
                                    source=db_source,
                                ))
                            if src_key == "steam":
                                has_primary = True
                    if has_primary:
                        primary_items_collected += len(matched_items)
                    else:
                        # Item matched but has no primary Steam price
                        errors_count += len(matched_items)
                        missing_names.append(name)
                else:
                    errors_count += len(matched_items)
                    missing_names.append(name)

            if missing_names:
                non_aggregator_prices = self._load_latest_non_aggregator_prices(
                    {
                        item.id
                        for missing_name in missing_names
                        for item in item_map[missing_name]
                    }
                )
                latest_any_source_prices = self._load_latest_prices(
                    {
                        item.id
                        for missing_name in missing_names
                        for item in item_map[missing_name]
                    }
                )
                still_missing_names = []

                for name in missing_names:
                    matched_items = item_map[name]
                    recovered_row = next(
                        (
                            non_aggregator_prices.get(item.id)
                            for item in matched_items
                            if non_aggregator_prices.get(item.id) is not None
                        ),
                        None,
                    )
                    if recovered_row is None:
                        recovered_row = next(
                            (
                                latest_any_source_prices.get(item.id)
                                for item in matched_items
                                if latest_any_source_prices.get(item.id) is not None
                            ),
                            None,
                        )

                    if recovered_row is None:
                        still_missing_names.append(name)
                        continue

                    for item in matched_items:
                        price_records.append(PriceHistory(
                            item_id=item.id,
                            timestamp=now,
                            price=recovered_row.price,
                            volume=recovered_row.volume,
                            source=_historical_fallback_source(recovered_row.source),
                        ))
                        fallback_items_collected += 1

                missing_names = still_missing_names
                errors_count = sum(len(item_map[name]) for name in missing_names)

            # 4. Save prices
            backfilled_csv_path = None
            snapshot_csv_path = None
            if price_records:
                from database import Item

                # Get all items: slug map for CSV export
                all_items = self.db_session.query(Item.id, Item.item_id, Item.is_backfilled).all()
                id_to_slug = {row.id: row.item_id for row in all_items}
                hist_item_ids = {row.id for row in all_items if row.is_backfilled == 1}

                rows_as_dicts = [
                    {
                        "item_id": r.item_id,
                        "timestamp": r.timestamp,
                        "price": r.price,
                        "volume": r.volume,
                        "source": r.source,
                    }
                    for r in price_records
                ]

                agg_date = now.strftime("%Y-%m-%d")

                # ── Write ALL sources to snapshot CSV for Parquet archive ──
                snapshot_csv_path = f"/tmp/aggregator-snapshots-{agg_date}.csv"
                with open(snapshot_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["item_slug", "day", "source", "price", "volume"])
                    for d in rows_as_dicts:
                        slug = id_to_slug.get(d["item_id"])
                        if slug:
                            writer.writerow([slug, agg_date, d["source"], d["price"], d.get("volume", 0)])
                logger.info("Wrote %s snapshot rows to %s (all sources)", len(rows_as_dicts), snapshot_csv_path)

                # ── Write only aggregator_sync rows to backfilled CSV (for OHLCV Parquet) ──
                backfilled_dicts = [d for d in rows_as_dicts if d["item_id"] in hist_item_ids]
                parquet_rows = [d for d in backfilled_dicts if d["source"] == "aggregator_sync"]
                if parquet_rows:
                    csv_path = f"/tmp/aggregator-backfilled-{agg_date}.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["item_slug", "day", "price", "volume"])
                        for d in parquet_rows:
                            slug = id_to_slug.get(d["item_id"])
                            if slug:
                                writer.writerow([slug, agg_date, d["price"], d.get("volume", 0)])
                    backfilled_csv_path = csv_path
                    logger.info("Wrote %s backfilled rows to %s (Steam 24h for OHLCV)", len(parquet_rows), csv_path)

                # ── Write aggregator_sync + historical_fallback to Supabase ──
                supabase_rows = [d for d in rows_as_dicts
                                 if d["source"] == "aggregator_sync"
                                 or d["source"].startswith("historical_fallback:")]
                if supabase_rows:
                    snapshot_supabase = [d for d in supabase_rows if d["item_id"] not in hist_item_ids]
                    backfilled_supabase = [d for d in supabase_rows if d["item_id"] in hist_item_ids]

                    # Delete old snapshot-tier aggregator_sync rows
                    snapshot_ids = list({d["item_id"] for d in snapshot_supabase if d["source"] == "aggregator_sync"})
                    for i in range(0, len(snapshot_ids), 1000):
                        self.db_session.execute(
                            text(
                                "DELETE FROM price_history "
                                "WHERE item_id IN :ids "
                                "AND source = 'aggregator_sync'"
                            ).bindparams(bindparam("ids", expanding=True)),
                            {"ids": snapshot_ids[i:i + 1000]},
                        )

                    # Insert rows (aggregator_sync + historical_fallback)
                    all_supabase = []
                    if snapshot_supabase:
                        all_supabase.extend(snapshot_supabase)
                    if backfilled_supabase:
                        all_supabase.extend(backfilled_supabase)
                    if all_supabase:
                        self.db_session.execute(
                            text("""
                                INSERT INTO price_history (item_id, timestamp, price, volume, source)
                                VALUES (:item_id, :timestamp, :price, :volume, :source)
                                ON CONFLICT (item_id, timestamp, source) DO NOTHING
                            """),
                            all_supabase,
                        )
                        logger.info("Saved %s rows to price_history (Supabase): %s aggregator_sync, %s fallback",
                                    len(all_supabase),
                                    sum(1 for d in all_supabase if d["source"] == "aggregator_sync"),
                                    sum(1 for d in all_supabase if d["source"].startswith("historical_fallback:")))

            if missing_names:
                sample_missing = missing_names[:20]
                missing_name_report = self._build_missing_name_report(missing_names, item_map)
                logger.warning(
                    "Unmatched item names in this run: %s/%s items missing. Sample: %s",
                    errors_count,
                    len(items),
                    sample_missing,
                )
                logger.warning(
                    "Missing-name report: %s",
                    {
                        bucket["key"]: {
                            "count": bucket["count"],
                            "item_rows": bucket["item_rows"],
                            "sample": bucket["sample"],
                        }
                        for bucket in missing_name_report.get("buckets", [])
                    },
                )
                diagnostic_targets = sample_missing[:5]
                diagnostic_map = {
                    missing_name: aggregator.find_source_key_candidates(missing_name, limit=5)
                    for missing_name in diagnostic_targets
                }
                souvenir_charm_samples = []
                for bucket in missing_name_report.get("buckets", []):
                    if bucket.get("key") == "skin_variant_items":
                        for subgroup in bucket.get("subgroups", []):
                            if subgroup.get("key") == "stattrak_souvenir_items":
                                for nested in subgroup.get("subgroups", []):
                                    if nested.get("key") == "souvenir_charm_items":
                                        souvenir_charm_samples = nested.get("sample", [])[:5]
                                        break
                                break
                        break
                souvenir_charm_diagnostics = {
                    missing_name: aggregator.find_source_key_candidates(missing_name, limit=5)
                    for missing_name in souvenir_charm_samples
                } if souvenir_charm_samples else {}
                logger.warning(
                    "Aggregator source-key diagnostics for missing sample: %s",
                    diagnostic_map,
                )
                if souvenir_charm_diagnostics:
                    logger.warning(
                        "Souvenir Charm source-key diagnostics: %s",
                        souvenir_charm_diagnostics,
                    )
            elif fallback_items_collected:
                logger.info(
                    "Recovered %s items from historical fallback sources",
                    fallback_items_collected,
                )

            # 5. Record collection run for monitoring
            end_time = datetime.utcnow()
            duration_seconds = (end_time - start_time).total_seconds()
            total_collected = primary_items_collected + fallback_items_collected

            collection_run = CollectionRun(
                started_at=start_time,
                finished_at=end_time,
                status='completed',
                total_items=len(items),
                successful=total_collected,
                failed=errors_count,
                duration_seconds=duration_seconds,
                source_breakdown={
                    'aggregator': primary_items_collected,
                    'historical_fallback': fallback_items_collected,
                    'aggregator_steam_7d': sum(1 for r in price_records if r.source == 'aggregator_steam_7d'),
                    'aggregator_steam_30d': sum(1 for r in price_records if r.source == 'aggregator_steam_30d'),
                    'aggregator_steam_90d': sum(1 for r in price_records if r.source == 'aggregator_steam_90d'),
                    'aggregator_skinport': sum(1 for r in price_records if r.source == 'aggregator_skinport'),
                    'aggregator_buff163': sum(1 for r in price_records if r.source == 'aggregator_buff163'),
                    'aggregator_buff163_buy': sum(1 for r in price_records if r.source == 'aggregator_buff163_buy'),
                    'aggregator_csfloat': sum(1 for r in price_records if r.source == 'aggregator_csfloat'),
                }
            )
            self.db_session.add(collection_run)
            self.db_session.commit()

            if total_collected == 0:
                logger.error(
                    "ZERO items collected in this run — all endpoints may be down "
                    "or upstream data format has changed"
                )

            logger.info(
                f"✅ Collection complete: {total_collected}/{len(items)} items in {duration_seconds:.1f}s"
            )
            return {
                "status": "success",
                "items_collected": total_collected,
                "total_items": len(items),
                "errors": errors_count,
                "duplicate_names": duplicate_name_count,
                "duplicate_name_sample": duplicate_name_sample,
                "missing_name_sample": missing_names[:20],
                "missing_name_report": missing_name_report,
                "missing_name_diagnostics": {
                    name: aggregator.find_source_key_candidates(name, limit=5)
                    for name in missing_names[:5]
                } if missing_names else {},
                "duration_seconds": duration_seconds,
                "backfilled_csv_path": backfilled_csv_path,
                "snapshot_csv_path": snapshot_csv_path,
            }

        except Exception as e:
            if self.db_session:
                self.db_session.rollback()

            end_time = datetime.utcnow()
            duration_seconds = (end_time - start_time).total_seconds()

            # Try to record failed run
            try:
                from database import CollectionRun
                failed_run = CollectionRun(
                    started_at=start_time,
                    finished_at=end_time,
                    status='failed',
                    total_items=0,
                    successful=0,
                    failed=1,
                    duration_seconds=duration_seconds,
                    # error_message is VARCHAR(1000); oversized values abort
                    # the failure record itself
                    error_message=str(e)[:1000]
                )
                # discard the failed transaction so this insert can commit
                self.db_session.rollback()
                self.db_session.add(failed_run)
                self.db_session.commit()
            except Exception as record_error:
                logger.error(f"Could not record failed run: {record_error}")

            logger.error(f"❌ Aggregator collection failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e), "duration_seconds": duration_seconds}

    def _load_latest_non_aggregator_prices(self, item_ids):
        """Load the latest non-primary price snapshot for each exact item id."""
        from database import PriceHistory

        if not item_ids:
            return {}

        rows = (
            self.db_session.query(PriceHistory)
            .filter(
                PriceHistory.item_id.in_(item_ids),
                PriceHistory.source != 'aggregator_sync',
                ~PriceHistory.source.like('synthetic_demo'),
                ~PriceHistory.source.like('historical_fallback:%'),
            )
            .order_by(
                PriceHistory.item_id,
                PriceHistory.timestamp.desc(),
                PriceHistory.created_at.desc().nullslast(),
                PriceHistory.source.desc(),
            )
            .all()
        )

        latest_by_item_id = {}
        for row in rows:
            if row.item_id not in latest_by_item_id:
                latest_by_item_id[row.item_id] = row

        return latest_by_item_id

    def _load_latest_prices(self, item_ids):
        """Load the latest price snapshot for each exact item id from any source."""
        from database import PriceHistory

        if not item_ids:
            return {}

        rows = (
            self.db_session.query(PriceHistory)
            .filter(
                PriceHistory.item_id.in_(item_ids),
                ~PriceHistory.source.like('synthetic_demo'),
                ~PriceHistory.source.like('historical_fallback:%'),
            )
            .order_by(
                PriceHistory.item_id,
                PriceHistory.timestamp.desc(),
                PriceHistory.created_at.desc().nullslast(),
                PriceHistory.source.desc(),
            )
            .all()
        )

        latest_by_item_id = {}
        for row in rows:
            if row.item_id not in latest_by_item_id:
                latest_by_item_id[row.item_id] = row

        return latest_by_item_id

    @staticmethod
    def _classify_missing_name(name: str, matched_items: List[Any]) -> str:
        """Assign a conservative bucket to a missing item name."""
        name_lower = name.lower()
        item_types = {getattr(item, "type", None) for item in matched_items if getattr(item, "type", None)}

        if name_lower.startswith(("sticker | ", "sticker slab | ")):
            return "sticker_items"

        if name_lower.startswith("stattrak") or name_lower.startswith("souvenir ") or name.startswith("★"):
            return "skin_variant_items"

        if any(suffix.lower() in name_lower for suffix in QUALITY_SUFFIXES):
            return "skin_variant_items"

        if item_types & {"skin", "glove", "agent"}:
            if name.startswith(SPECIAL_PREFIXES):
                return "skin_variant_items"
            return "skin_items"

        if item_types & {"case", "capsule"}:
            return "container_items"

        return "other_items"

    @staticmethod
    def _classify_skin_variant_subgroup(name: str) -> str:
        """Split skin variants into narrow, ordered subgroups for investigation."""
        name_lower = name.lower()

        if name.startswith("★"):
            return "knife_items"
        if name_lower.startswith(("stattrak", "souvenir ")):
            return "stattrak_souvenir_items"
        if any(suffix.lower() in name_lower for suffix in QUALITY_SUFFIXES):
            return "wear_suffix_items"

        return "other_variant_items"

    @staticmethod
    def _classify_stattrak_souvenir_subgroup(name: str) -> str:
        """Split StatTrak/Souvenir rows into narrower, ordered investigation groups."""
        name_lower = name.lower()

        if name_lower.startswith("stattrak"):
            return "stattrak_items"
        if name_lower.startswith("souvenir charm"):
            return "souvenir_charm_items"
        if name_lower.startswith("souvenir "):
            return "other_souvenir_items"

        return "other_stattrak_souvenir_items"

    def _build_missing_name_report(
        self,
        missing_names: List[str],
        item_map: Dict[str, List[Any]],
        sample_size: int = 5,
    ) -> Dict[str, Any]:
        """Build an ordered, conservative breakdown of unresolved names."""
        if not missing_names:
            return {
                "total_missing_names": 0,
                "total_missing_rows": 0,
                "buckets": [],
            }

        bucket_order = [
            "sticker_items",
            "skin_variant_items",
            "skin_items",
            "container_items",
            "other_items",
        ]
        bucket_labels = {
            "sticker_items": "Sticker listings",
            "skin_variant_items": "Skins with wear or special-prefix variants",
            "skin_items": "Other catalog items",
            "container_items": "Cases and capsules",
            "other_items": "Other catalog items",
        }
        bucket_hints = {
            "sticker_items": "Sticker rows are grouped separately so finish/event drift stays visible.",
            "skin_variant_items": "Includes StatTrak, Souvenir, knife, and wear-suffixed skins.",
            "skin_items": "Catalog items without a stronger pattern signal.",
            "container_items": "Non-skin containers and capsules.",
            "other_items": "Items that do not fit a narrower pattern.",
        }

        buckets = {
            key: {
                "key": key,
                "label": bucket_labels[key],
                "count": 0,
                "item_rows": 0,
                "sample": [],
                "hint": bucket_hints[key],
            }
            for key in bucket_order
        }

        variant_bucket_order = [
            "knife_items",
            "stattrak_souvenir_items",
            "wear_suffix_items",
            "other_variant_items",
        ]
        variant_bucket_labels = {
            "knife_items": "Knife items",
            "stattrak_souvenir_items": "StatTrak and Souvenir items",
            "wear_suffix_items": "Wear-suffixed items",
            "other_variant_items": "Other variant items",
        }
        variant_bucket_hints = {
            "knife_items": "Star-prefixed knives are the highest-confidence alias candidates.",
            "stattrak_souvenir_items": "StatTrak and Souvenir rows share the same source naming drift risk.",
            "wear_suffix_items": "Factory New through Battle-Scarred rows may need quality aliasing.",
            "other_variant_items": "Variant rows that do not fit a narrower pattern.",
        }
        variant_buckets = {
            key: {
                "key": key,
                "label": variant_bucket_labels[key],
                "count": 0,
                "item_rows": 0,
                "sample": [],
                "hint": variant_bucket_hints[key],
            }
            for key in variant_bucket_order
        }

        stattrak_souvenir_bucket_order = [
            "stattrak_items",
            "souvenir_charm_items",
            "other_souvenir_items",
            "other_stattrak_souvenir_items",
        ]
        stattrak_souvenir_bucket_labels = {
            "stattrak_items": "StatTrak items",
            "souvenir_charm_items": "Souvenir Charm items",
            "other_souvenir_items": "Other Souvenir items",
            "other_stattrak_souvenir_items": "Other StatTrak/Souvenir items",
        }
        stattrak_souvenir_bucket_hints = {
            "stattrak_items": "StatTrak weapon names are the highest-confidence alias candidates.",
            "souvenir_charm_items": "Souvenir Charm rows often drift by event wording or missing descriptors.",
            "other_souvenir_items": "Other souvenir rows that do not use the Charm pattern.",
            "other_stattrak_souvenir_items": "StatTrak/Souvenir rows that do not fit a narrower pattern.",
        }
        stattrak_souvenir_buckets = {
            key: {
                "key": key,
                "label": stattrak_souvenir_bucket_labels[key],
                "count": 0,
                "item_rows": 0,
                "sample": [],
                "hint": stattrak_souvenir_bucket_hints[key],
            }
            for key in stattrak_souvenir_bucket_order
        }

        total_missing_rows = 0
        for name in missing_names:
            matched_items = item_map.get(name, [])
            bucket_key = self._classify_missing_name(name, matched_items)
            bucket = buckets[bucket_key]
            bucket["count"] += 1
            item_rows = len(matched_items) if matched_items else 1
            bucket["item_rows"] += item_rows
            total_missing_rows += item_rows
            if len(bucket["sample"]) < sample_size:
                bucket["sample"].append(name)

            if bucket_key == "skin_variant_items":
                variant_key = self._classify_skin_variant_subgroup(name)
                variant_bucket = variant_buckets[variant_key]
                variant_bucket["count"] += 1
                variant_bucket["item_rows"] += item_rows
                if len(variant_bucket["sample"]) < sample_size:
                    variant_bucket["sample"].append(name)

                if variant_key == "stattrak_souvenir_items":
                    stattrak_souvenir_key = self._classify_stattrak_souvenir_subgroup(name)
                    stattrak_bucket = stattrak_souvenir_buckets[stattrak_souvenir_key]
                    stattrak_bucket["count"] += 1
                    stattrak_bucket["item_rows"] += item_rows
                    if len(stattrak_bucket["sample"]) < sample_size:
                        stattrak_bucket["sample"].append(name)

        if buckets["skin_variant_items"]["count"] > 0:
            buckets["skin_variant_items"]["subgroups"] = [
                variant_buckets[key]
                for key in variant_bucket_order
                if variant_buckets[key]["count"] > 0
            ]
        if variant_buckets["stattrak_souvenir_items"]["count"] > 0:
            variant_buckets["stattrak_souvenir_items"]["subgroups"] = [
                stattrak_souvenir_buckets[key]
                for key in stattrak_souvenir_bucket_order
                if stattrak_souvenir_buckets[key]["count"] > 0
            ]

        return {
            "total_missing_names": len(missing_names),
            "total_missing_rows": total_missing_rows,
            "buckets": [buckets[key] for key in bucket_order if buckets[key]["count"] > 0],
        }

    def _load_recent_price_histories(self, item_ids, days: int = 90):
        """Load recent price history for many items in one query.

        Falls back to the Supabase price_history for small windows (<=14 days).
        For longer windows reads from the DuckDB Parquet archive for speed.
        """
        from database import PriceHistory

        if not item_ids:
            return {}

        use_parquet = days > 14
        if use_parquet:
            return self._load_parquet_histories(item_ids, days)

        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = self.db_session.query(
            PriceHistory.item_id,
            PriceHistory.timestamp,
            PriceHistory.price
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= cutoff,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        ).order_by(PriceHistory.item_id, PriceHistory.timestamp).all()

        histories = defaultdict(list)
        for item_id, timestamp, price in rows:
            histories[item_id].append((timestamp, price))

        return histories

    def _load_parquet_histories(self, item_ids, days: int = 90):
        """Load price history from Parquet archive via DuckDB.

        Maps internal item_ids to item_slugs, queries the Parquet files,
        and returns results in the same {item_id: [(timestamp, price)]} format.
        """
        import duckdb
        from database import Item

        if not item_ids or not self.db_session:
            return {}

        slug_rows = self.db_session.query(Item.id, Item.item_id).filter(
            Item.id.in_(item_ids)
        ).all()
        int_to_slug = {row.id: row.item_id for row in slug_rows}
        slug_to_int = {v: k for k, v in int_to_slug.items()}

        cutoff = (datetime.utcnow() - timedelta(days=days)).date()

        con = duckdb.connect()
        try:
            slug_list = list(slug_to_int.keys())
            placeholders = ','.join('?' for _ in slug_list)
            rows = con.sql(f"""
                SELECT item_slug, day, mean_price AS price
                FROM read_parquet('../price-archive/prices-*.parquet')
                WHERE item_slug IN ({placeholders})
                  AND day >= ?
                ORDER BY item_slug, day
            """, params=[*slug_list, cutoff.isoformat()]).fetchall()

            histories = defaultdict(list)
            for slug, day, price in rows:
                item_id = slug_to_int.get(slug)
                if item_id is None:
                    continue
                day_dt = datetime.combine(day, datetime.min.time())
                histories[item_id].append((day_dt, price))

            return dict(histories)
        finally:
            con.close()

    def _load_recent_price_counts(self, item_ids, days: int = 90):
        """Load recent price counts for many items in one grouped query."""
        from database import PriceHistory

        if not item_ids:
            return {}

        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = self.db_session.query(
            PriceHistory.item_id,
            func.count(PriceHistory.item_id)
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= cutoff,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        ).group_by(PriceHistory.item_id).all()

        return {item_id: count for item_id, count in rows}
    
    def run_trend_analysis(self):
        """Execute trend scoring and opportunity detection"""
        try:
            logger.info("Starting trend analysis")
            
            from analytics.trend_analyzer import TrendAnalyzer, OpportunityDetector
            from database import Item
            
            if not self.db_session:
                logger.error("Database session not available")
                return {"status": "failed", "error": "No database session"}
            
            items = self.db_session.query(Item.id, Item.name).all()
            item_ids = [item_id for item_id, _ in items]
            recent_counts = self._load_recent_price_counts(item_ids, days=90)
            recent_histories = self._load_recent_price_histories(
                [item_id for item_id, _ in items if recent_counts.get(item_id, 0) >= 7],
                days=90
            )
            opportunities_detected = 0
            
            for item_id, item_name in items:
                try:
                    price_history = recent_histories.get(item_id, [])
                    prices = [price for _, price in price_history]
                    
                    if len(prices) < 7:
                        continue
                    
                    # Compute trend
                    trend_score = TrendAnalyzer.compute_trend_score(prices)
                    if trend_score is not None:
                        direction, confidence = TrendAnalyzer.classify_trend(trend_score)
                        
                        # Detect opportunities
                        baseline = OpportunityDetector.compute_baseline_trend(prices)
                        current_price = prices[-1]
                        
                        if baseline:
                            is_undervalued, discount = OpportunityDetector.detect_undervalued(current_price, baseline)
                            is_overheated, premium = OpportunityDetector.detect_overheated(current_price, baseline)
                            has_momentum, change_pct, momentum_dir = OpportunityDetector.detect_momentum(prices)
                            
                            if is_undervalued or is_overheated or has_momentum:
                                opportunities_detected += 1
                                logger.info(f"{item_name}: {direction} ({confidence}) | "
                                          f"Undervalued: {is_undervalued} | "
                                          f"Overheated: {is_overheated} | "
                                          f"Momentum: {has_momentum}")
                
                except Exception as item_error:
                    logger.error(f"Error analyzing {item_name}: {item_error}")
            
            logger.info(f"Trend analysis completed: {opportunities_detected} opportunities detected")
            return {
                "status": "success",
                "timestamp": datetime.utcnow(),
                "opportunities_detected": opportunities_detected
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    

