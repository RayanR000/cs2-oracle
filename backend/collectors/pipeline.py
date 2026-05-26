"""
Data pipeline orchestration
Manages scheduled data collection, validation, and storage
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

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
            
            # Schedule daily data collection at 1 AM UTC
            self.scheduler.add_job(
                self.run_daily_collection,
                CronTrigger(hour=1, minute=0),
                id='daily_collection',
                name='Daily market data collection',
                replace_existing=True
            )
            
            # Schedule hourly feature computation at :00
            self.scheduler.add_job(
                self.run_feature_computation,
                CronTrigger(minute=0),
                id='hourly_features',
                name='Hourly feature computation',
                replace_existing=True
            )
            
            # Schedule daily trend analysis at 2 AM UTC
            self.scheduler.add_job(
                self.run_trend_analysis,
                CronTrigger(hour=2, minute=0),
                id='daily_trends',
                name='Daily trend analysis',
                replace_existing=True
            )
            
            # Schedule weekly database pruning at 3 AM UTC every Sunday
            self.scheduler.add_job(
                self.run_database_pruning,
                CronTrigger(day_of_week='sun', hour=3, minute=0),
                id='weekly_pruning',
                name='Weekly database pruning',
                replace_existing=True
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
    
    def run_feature_computation(self):
        """Execute feature computation for trend analysis"""
        try:
            logger.info("Starting feature computation")
            
            from analytics.trend_analyzer import TrendAnalyzer
            from database import Item, TrendIndicator
            
            if not self.db_session:
                logger.error("Database session not available")
                return {"status": "failed", "error": "No database session"}
            
            items = self.db_session.query(Item).all()
            features_computed = 0
            
            for item in items:
                try:
                    # Get recent price history
                    prices = sorted([h.price for h in item.price_histories[-90:]], 
                                   key=lambda p: p.timestamp if hasattr(p, 'timestamp') else 0)
                    
                    if len(prices) < 7:
                        continue
                    
                    # Compute indicators
                    sma_7 = TrendAnalyzer.compute_sma(prices, 7)
                    sma_30 = TrendAnalyzer.compute_sma(prices, 30)
                    volatility = TrendAnalyzer.compute_volatility(prices)
                    rsi = TrendAnalyzer.compute_rsi(prices)
                    bollinger = TrendAnalyzer.compute_bollinger_bands(prices)
                    macd = TrendAnalyzer.compute_macd(prices)
                    support_resist = TrendAnalyzer.compute_support_resistance(prices)
                    
                    # Store features
                    if sma_7 or sma_30:
                        trend_indicator = TrendIndicator(
                            item_id=item.id,
                            sma_7=sma_7,
                            sma_30=sma_30,
                            volatility=volatility,
                            rsi=rsi,
                            timestamp=datetime.utcnow()
                        )
                        self.db_session.add(trend_indicator)
                        features_computed += 1
                
                except Exception as item_error:
                    logger.error(f"Error computing features for {item.name}: {item_error}")
            
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
            
            items = self.db_session.query(Item).all()
            opportunities_detected = 0
            
            for item in items:
                try:
                    # Get price history
                    price_history = sorted(item.price_histories, key=lambda h: h.timestamp)
                    prices = [h.price for h in price_history[-90:]]
                    
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
                                logger.info(f"{item.name}: {direction} ({confidence}) | "
                                          f"Undervalued: {is_undervalued} | "
                                          f"Overheated: {is_overheated} | "
                                          f"Momentum: {has_momentum}")
                
                except Exception as item_error:
                    logger.error(f"Error analyzing {item.name}: {item_error}")
            
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
        """Execute weekly database pruning and downsampling"""
        try:
            logger.info("Starting weekly database pruning")
            
            from scripts.prune_database import prune_price_history, prune_trend_indicators, downsample_price_history
            
            if not self.db_session:
                logger.error("Database session not available")
                return {"status": "failed", "error": "No database session"}
            
            # Use defaults: 1 year for history, 180 days for trends, 30 days for granularity
            pruned_history = prune_price_history(self.db_session, days_to_keep=365, dry_run=False)
            pruned_trends = prune_trend_indicators(self.db_session, days_to_keep=180, dry_run=False)
            downsampled = downsample_price_history(self.db_session, days_to_keep_granular=30, dry_run=False)
            
            logger.info(f"Database pruning completed: "
                      f"Deleted {pruned_history} old history, "
                      f"{pruned_trends} old trends, "
                      f"{downsampled} redundant history records")
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow(),
                "pruned_history": pruned_history,
                "pruned_trends": pruned_trends,
                "downsampled": downsampled
            }
            
        except Exception as e:
            if self.db_session:
                self.db_session.rollback()
            logger.error(f"Error in database pruning: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    
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
