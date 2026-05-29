"""
Data pipeline orchestration
Manages scheduled data collection, validation, and storage
"""

import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Callable
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func

logger = logging.getLogger(__name__)

class DataPipeline:
    """Orchestrates data collection and processing pipeline"""
    
    def __init__(self, db_session=None):
        """
        Initialize data pipeline
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.scheduler: Optional[BackgroundScheduler] = None
        self.is_running = False
    
    def start(self, scheduler: Optional[BackgroundScheduler] = None):
        """
        Start the data pipeline scheduler
        
        Args:
            scheduler: Optional BackgroundScheduler instance to use
        """
        if self.is_running:
            logger.warning("Data pipeline already running")
            return
        
        try:
            self.scheduler = scheduler or BackgroundScheduler()
            
            # Schedule priority data collection (Top 2000) every 6 hours
            self.scheduler.add_job(
                self.run_priority_collection,
                CronTrigger(hour='0,6,12,18', minute=30),
                id='priority_collection',
                name='Priority market data collection (Top 2000)',
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300
            )
            
            # Schedule full daily data collection at 1 AM UTC
            self.scheduler.add_job(
                self.run_daily_collection,
                CronTrigger(hour=1, minute=0),
                id='daily_collection',
                name='Full daily market data collection',
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300
            )
            
            # Schedule hourly feature computation at :00
            self.scheduler.add_job(
                self.run_feature_computation,
                CronTrigger(minute=0),
                id='hourly_features',
                name='Hourly feature computation',
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300
            )
            
            # Schedule daily trend analysis at 2 AM UTC
            self.scheduler.add_job(
                self.run_trend_analysis,
                CronTrigger(hour=2, minute=0),
                id='daily_trends',
                name='Daily trend analysis',
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300
            )
            
            # Schedule weekly database pruning at 3 AM UTC every Sunday
            self.scheduler.add_job(
                self.run_database_pruning,
                CronTrigger(day_of_week='sun', hour=3, minute=0),
                id='weekly_pruning',
                name='Weekly database pruning',
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300
            )
            
            self.scheduler.start()
            self.is_running = True
            logger.info("Data pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Error starting data pipeline: {e}")
            raise
    
    def stop(self):
        """Stop the data pipeline scheduler"""
        if self.scheduler and self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Data pipeline stopped")
    
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
        items_collected = 0
        errors_count = 0
        duplicate_name_count = 0
        duplicate_name_sample: List[str] = []
        missing_names: List[str] = []

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

            # 3. Store results
            price_records = []
            now = datetime.utcnow()

            for name, matched_items in item_map.items():
                res = results.get(name)
                if res and len(res) >= 2 and res[0] > 0:
                    # Aggregator returns (price, volume, timestamp)
                    price, volume = res[0], res[1]
                    for item in matched_items:
                        price_records.append(PriceHistory(
                            item_id=item.id,
                            timestamp=now,
                            price=price,
                            volume=volume,
                            source='aggregator_sync'
                        ))
                        items_collected += 1
                else:
                    errors_count += len(matched_items)
                    missing_names.append(name)

            # 4. Save prices
            if price_records:
                self.db_session.bulk_save_objects(price_records)
                logger.info(f"Saving {len(price_records)} price records...")

            if missing_names:
                sample_missing = missing_names[:20]
                logger.warning(
                    "Unmatched item names in this run: %s/%s items missing. Sample: %s",
                    errors_count,
                    len(items),
                    sample_missing,
                )
                diagnostic_targets = sample_missing[:5]
                diagnostic_map = {
                    missing_name: aggregator.find_source_key_candidates(missing_name, limit=5)
                    for missing_name in diagnostic_targets
                }
                logger.warning(
                    "Aggregator source-key diagnostics for missing sample: %s",
                    diagnostic_map,
                )

            # 5. Record collection run for monitoring
            end_time = datetime.utcnow()
            duration_seconds = (end_time - start_time).total_seconds()

            collection_run = CollectionRun(
                started_at=start_time,
                finished_at=end_time,
                status='completed',
                total_items=len(items),
                successful=items_collected,
                failed=errors_count,
                duration_seconds=duration_seconds,
                source_breakdown={'aggregator': items_collected}
            )
            self.db_session.add(collection_run)
            self.db_session.commit()

            logger.info(f"✅ Collection complete: {items_collected}/{len(items)} items in {duration_seconds:.1f}s")
            return {
                "status": "success",
                "items_collected": items_collected,
                "total_items": len(items),
                "errors": errors_count,
                "duplicate_names": duplicate_name_count,
                "duplicate_name_sample": duplicate_name_sample,
                "missing_name_sample": missing_names[:20],
                "missing_name_diagnostics": {
                    name: aggregator.find_source_key_candidates(name, limit=5)
                    for name in missing_names[:5]
                } if missing_names else {},
                "duration_seconds": duration_seconds
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
                    error_message=str(e)
                )
                self.db_session.add(failed_run)
                self.db_session.commit()
            except Exception as record_error:
                logger.error(f"Could not record failed run: {record_error}")

            logger.error(f"❌ Aggregator collection failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e), "duration_seconds": duration_seconds}

    def run_daily_collection(self):
        """Execute daily market data collection using high-efficiency batch method"""
        try:
            logger.info("Starting daily market data collection (Batch Method)")
            
            from collectors.steam_market import SteamMarketCollector
            from database import Item, PriceHistory
            
            # Using 20.0s delay as requested for safety
            collector = SteamMarketCollector(rate_limit_delay=20.0)
            
            if not self.db_session:
                logger.error("Database session not available")
                return {"status": "failed", "error": "No database session"}
            
            # Pre-load all items for mapping
            logger.info("Loading items from database...")
            all_items = self.db_session.query(Item).all()
            # Map item_id (hash_name) to database internal ID
            item_id_map = {item.item_id: item.id for item in all_items}
            logger.info(f"Loaded {len(item_id_map)} items for mapping")
            
            successful_collections = 0
            total_processed = 0
            start_index = 0
            batch_size = 100
            
            while True:
                logger.info(f"Fetching batch: items {start_index} to {start_index + batch_size}...")
                
                batch_data = collector.get_market_listings(start=start_index, count=batch_size)
                
                if not batch_data or not batch_data.get('results'):
                    logger.info("No more items found or request failed.")
                    break
                
                results = batch_data['results']
                total_on_steam = batch_data.get('total_count', 0)
                
                price_records = []
                now = datetime.utcnow()
                
                for res in results:
                    total_processed += 1
                    hash_name = res['hash_name']
                    
                    # Map to our DB
                    internal_id = item_id_map.get(hash_name)
                    if not internal_id:
                        # Log but don't stop; might be a new item
                        logger.debug(f"Steam item not in database: {hash_name}")
                        continue
                    
                    if res['price'] > 0:
                        price_records.append(PriceHistory(
                            item_id=internal_id,
                            timestamp=now,
                            price=res['price'],
                            volume=res['volume'],
                            source='steam_batch'
                        ))
                        successful_collections += 1
                
                # Bulk insert this batch
                if price_records:
                    try:
                        self.db_session.bulk_save_objects(price_records)
                        self.db_session.commit()
                        logger.info(f"  → Committed {len(price_records)} prices.")
                    except Exception as e:
                        logger.error(f"Failed to commit batch: {e}")
                        self.db_session.rollback()
                
                # Check if we've reached the end
                if start_index + batch_size >= total_on_steam or len(results) < batch_size:
                    logger.info("Reached end of Steam market catalog.")
                    break
                    
                start_index += batch_size
                
            logger.info(f"Daily collection completed: {successful_collections} successful, {total_processed} items seen")
            return {
                "status": "success",
                "timestamp": datetime.utcnow(),
                "successful": successful_collections,
                "total_seen": total_processed
            }
            
        except Exception as e:
            if self.db_session:
                self.db_session.rollback()
            logger.error(f"Error in daily collection: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _load_recent_price_histories(self, item_ids, days: int = 90):
        """Load recent price history for many items in one query."""
        from database import PriceHistory

        if not item_ids:
            return {}

        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = self.db_session.query(
            PriceHistory.item_id,
            PriceHistory.timestamp,
            PriceHistory.price
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= cutoff
        ).order_by(PriceHistory.item_id, PriceHistory.timestamp).all()

        histories = defaultdict(list)
        for item_id, timestamp, price in rows:
            histories[item_id].append((timestamp, price))

        return histories

    def _load_recent_price_counts(self, item_ids, days: int = 90):
        """Load recent price counts for many items in one grouped query."""
        from database import PriceHistory

        if not item_ids:
            return {}

        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = self.db_session.query(
            PriceHistory.item_id,
            func.count(PriceHistory.id)
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= cutoff
        ).group_by(PriceHistory.item_id).all()

        return {item_id: count for item_id, count in rows}
    
    def run_feature_computation(self):
        """Execute feature computation for trend analysis"""
        try:
            logger.info("Starting feature computation")
            
            from analytics.trend_analyzer import TrendAnalyzer
            from database import Item, TrendIndicator
            
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
            features_computed = 0
            
            for item_id, item_name in items:
                try:
                    prices_with_timestamps = recent_histories.get(item_id, [])
                    prices = [price for _, price in prices_with_timestamps]
                    
                    if len(prices) < 7:
                        continue
                    
                    # Compute indicators
                    sma_7 = TrendAnalyzer.compute_sma(prices, 7)
                    sma_30 = TrendAnalyzer.compute_sma(prices, 30)
                    volatility = TrendAnalyzer.compute_volatility(prices)
                    
                    # Store features
                    if sma_7 or sma_30:
                        trend_indicator = TrendIndicator(
                            item_id=item_id,
                            sma_7=sma_7,
                            sma_30=sma_30,
                            volatility=volatility,
                            timestamp=datetime.utcnow()
                        )
                        self.db_session.add(trend_indicator)
                        features_computed += 1
                
                except Exception as item_error:
                    logger.error(f"Error computing features for {item_name}: {item_error}")
            
            self.db_session.commit()
            
            logger.info(f"Feature computation completed: {features_computed} items processed")
            return {"status": "success", "timestamp": datetime.utcnow(), "items_processed": features_computed}
            
        except Exception as e:
            if self.db_session:
                self.db_session.rollback()
            logger.error(f"Error in feature computation: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    
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

    def run_database_pruning(self):
        """Execute weekly database pruning and downsampling with run tracking"""
        from scripts.prune_database import prune_price_history, prune_trend_indicators, downsample_price_history
        from database import CollectionRun

        start_time = datetime.utcnow()

        try:
            logger.info("Starting weekly database pruning")

            if not self.db_session:
                logger.error("Database session not available")
                return {"status": "failed", "error": "No database session"}

            # Downsample with tiered retention:
            # - 0-7 days: All data (6hr granularity)
            # - 7-30 days: Daily average
            # - 30-365 days: Weekly average
            # - 365+ days: Monthly average (kept indefinitely)
            downsampled = downsample_price_history(self.db_session, days_to_keep_granular=7, dry_run=False)

            # Delete very old trend indicators (keep 180 days)
            pruned_trends = prune_trend_indicators(self.db_session, days_to_keep=180, dry_run=False)
            pruned_history = 0  # Price history is preserved (just downsampled), not deleted

            # Record the run
            end_time = datetime.utcnow()
            duration_seconds = (end_time - start_time).total_seconds()
            total_pruned = pruned_history + pruned_trends + downsampled

            collection_run = CollectionRun(
                started_at=start_time,
                finished_at=end_time,
                status='completed',
                total_items=total_pruned,
                successful=total_pruned,
                failed=0,
                duration_seconds=duration_seconds,
                source_breakdown={'pruned_history': pruned_history, 'pruned_trends': pruned_trends, 'downsampled': downsampled}
            )
            self.db_session.add(collection_run)
            self.db_session.commit()

            logger.info(f"✅ Database pruning completed: "
                      f"Deleted {pruned_history} old history, "
                      f"{pruned_trends} old trends, "
                      f"{downsampled} redundant records in {duration_seconds:.1f}s")

            return {
                "status": "success",
                "timestamp": end_time,
                "pruned_history": pruned_history,
                "pruned_trends": pruned_trends,
                "downsampled": downsampled,
                "duration_seconds": duration_seconds
            }

        except Exception as e:
            if self.db_session:
                self.db_session.rollback()

            end_time = datetime.utcnow()
            duration_seconds = (end_time - start_time).total_seconds()

            # Try to record failed run
            try:
                failed_run = CollectionRun(
                    started_at=start_time,
                    finished_at=end_time,
                    status='failed',
                    total_items=0,
                    successful=0,
                    failed=1,
                    duration_seconds=duration_seconds,
                    error_message=str(e)
                )
                self.db_session.add(failed_run)
                self.db_session.commit()
            except Exception as record_error:
                logger.error(f"Could not record failed pruning run: {record_error}")

            logger.error(f"❌ Database pruning failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e), "duration_seconds": duration_seconds}
    
    def get_scheduled_jobs(self) -> List[Dict]:
        """Get list of scheduled jobs"""
        if not self.scheduler:
            return []
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'trigger': str(job.trigger),
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None
            })
        return jobs


class PipelineMonitor:
    """Monitors pipeline health and performance"""
    
    def __init__(self):
        """Initialize pipeline monitor"""
        self.collection_stats = {
            'last_run': None,
            'last_success': None,
            'last_error': None,
            'total_runs': 0,
            'total_failures': 0,
            'items_collected': 0
        }
        self.feature_stats = {
            'last_run': None,
            'last_success': None,
            'items_processed': 0,
            'total_features_computed': 0
        }
    
    def record_collection_run(self, success: bool, items_count: int = 0, error: Optional[str] = None):
        """Record a collection run"""
        self.collection_stats['last_run'] = datetime.utcnow()
        self.collection_stats['total_runs'] += 1
        
        if success:
            self.collection_stats['last_success'] = datetime.utcnow()
            self.collection_stats['items_collected'] += items_count
        else:
            self.collection_stats['total_failures'] += 1
            self.collection_stats['last_error'] = error
    
    def get_collection_health(self) -> Dict:
        """Get collection pipeline health status"""
        if self.collection_stats['total_runs'] == 0:
            return {'status': 'never_run'}
        
        success_rate = (self.collection_stats['total_runs'] - self.collection_stats['total_failures']) / self.collection_stats['total_runs']
        
        return {
            'status': 'healthy' if success_rate > 0.9 else 'degraded' if success_rate > 0.5 else 'unhealthy',
            'success_rate': success_rate,
            **self.collection_stats
        }
    
    def get_stats(self) -> Dict:
        """Get all pipeline statistics"""
        return {
            'collection': self.get_collection_health(),
            'features': self.feature_stats
        }
