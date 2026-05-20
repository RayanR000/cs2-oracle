"""
Real-time data collection service
Fetches real CS2 market data from Steam API
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
import threading
import time
from copy import deepcopy
from collectors.steam_market import SteamMarketCollector
from collectors.data_validation import DataValidator, DataCleaner
from database import SessionLocal, Item, PriceHistory, CollectionRun
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class RealDataCollector:
    """Service for collecting real market data from Steam API"""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the data collector
        
        Args:
            enabled: Whether to enable real data collection
        """
        self.enabled = enabled
        self.collector = SteamMarketCollector(rate_limit_delay=2.0)
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.is_running = False
        self.collection_thread = None
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._collection_stats = {
            'last_run_started_at': None,
            'last_run_finished_at': None,
            'last_success_at': None,
            'last_error_at': None,
            'last_error': None,
            'last_run_duration_seconds': None,
            'last_run_total_items': 0,
            'last_run_successful': 0,
            'last_run_failed': 0,
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_items_collected': 0,
            'total_items_failed': 0,
        }

    def _record_run_start(self) -> None:
        with self._thread_lock:
            self._collection_stats['last_run_started_at'] = datetime.utcnow()
            self._collection_stats['last_run_finished_at'] = None
            self._collection_stats['last_run_duration_seconds'] = None
            self._collection_stats['last_run_total_items'] = 0
            self._collection_stats['last_run_successful'] = 0
            self._collection_stats['last_run_failed'] = 0
            self._collection_stats['last_error'] = None
            self._collection_stats['total_runs'] += 1

    def _record_run_result(self, stats: Dict[str, Any], duration_seconds: float, success: bool, error: Optional[str] = None) -> None:
        now = datetime.utcnow()
        with self._thread_lock:
            self._collection_stats['last_run_finished_at'] = now
            self._collection_stats['last_run_duration_seconds'] = round(duration_seconds, 3)
            self._collection_stats['last_run_total_items'] = stats.get('total_items', 0)
            self._collection_stats['last_run_successful'] = stats.get('successful', 0)
            self._collection_stats['last_run_failed'] = stats.get('failed', 0)
            self._collection_stats['total_items_collected'] += stats.get('successful', 0)
            self._collection_stats['total_items_failed'] += stats.get('failed', 0)

            if success:
                self._collection_stats['successful_runs'] += 1
                self._collection_stats['last_success_at'] = now
            else:
                self._collection_stats['failed_runs'] += 1
                self._collection_stats['last_error_at'] = now
                self._collection_stats['last_error'] = error

    def _persist_collection_run(
        self,
        started_at: datetime,
        finished_at: datetime,
        stats: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Persist a completed collection run for durability across restarts."""
        db = SessionLocal()
        try:
            run = CollectionRun(
                started_at=started_at,
                finished_at=finished_at,
                status="success" if success else "failed",
                total_items=stats.get('total_items', 0),
                successful=stats.get('successful', 0),
                failed=stats.get('failed', 0),
                duration_seconds=stats.get('duration_seconds'),
                error_message=error,
                source_breakdown={
                    "steam": stats.get('successful', 0)
                }
            )
            db.add(run)
            db.commit()
        except SQLAlchemyError as db_error:
            db.rollback()
            logger.error(f"Database error persisting collection run: {db_error}", exc_info=True)
        finally:
            db.close()

    def get_collection_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of the current collector health and counters."""
        with self._thread_lock:
            metrics = deepcopy(self._collection_stats)

            thread = self.collection_thread
            metrics['thread_alive'] = bool(thread and thread.is_alive())
            metrics['collection_enabled'] = self.enabled
            metrics['is_running'] = self.is_running
            metrics['status'] = (
                'active' if self.is_running and metrics['thread_alive']
                else 'idle' if self.enabled
                else 'disabled'
            )
            return metrics
    
    def collect_item_data(self, item: Item) -> Optional[PriceHistory]:
        """
        Collect real price data for a single item from Steam
        
        Args:
            item: Item object to collect data for
            
        Returns:
            PriceHistory record if successful, None otherwise
        """
        try:
            # Get current price from Steam API
            result = self.collector.get_item_price_history(item.name)
            
            if not result:
                logger.warning(f"No data returned for {item.name}")
                return None
            
            price, volume, timestamp = result
            
            # Validate the data
            price_record = {
                'price': price,
                'volume': volume,
                'timestamp': timestamp,
                'item_name': item.name
            }
            
            is_valid, error = self.validator.validate_price_record(price_record)
            if not is_valid:
                logger.warning(f"Validation failed for {item.name}: {error}")
                return None
            
            # Check for anomalies
            anomaly_score = self.validator.compute_anomaly_score(price, [price])
            if anomaly_score > 0.8:
                logger.warning(f"High anomaly score for {item.name}: {anomaly_score}")
            
            # Clean the data
            cleaned_price = self.cleaner.sanitize_price(price)
            cleaned_volume = self.cleaner.sanitize_volume(volume) if volume else None
            
            # Create database record
            db = SessionLocal()
            try:
                price_history = PriceHistory(
                    item_id=item.id,
                    timestamp=timestamp,
                    price=cleaned_price,
                    volume=cleaned_volume,
                    median_price=cleaned_price,
                    source="steam"
                )
                db.add(price_history)
                db.commit()
                
                logger.info(f"Collected real data for {item.name}: ${cleaned_price} (vol: {cleaned_volume})")
                return price_history
                
            except SQLAlchemyError as e:
                logger.error(f"Database error collecting {item.name}: {e}")
                db.rollback()
                return None
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f"Error collecting data for {item.name}: {e}")
            return None
    
    def collect_all_items(self) -> Dict[str, int]:
        """
        Collect data for all items in the database
        
        Returns:
            Dictionary with collection statistics
        """
        started_at = datetime.utcnow()
        self._record_run_start()

        stats = {
            'total_items': 0,
            'successful': 0,
            'failed': 0,
            'timestamp': started_at.isoformat(),
            'started_at': started_at.isoformat(),
            'finished_at': None,
            'duration_seconds': None
        }
        
        db = SessionLocal()
        try:
            items = db.query(Item).all()
            stats['total_items'] = len(items)
            
            logger.info(f"Starting collection for {len(items)} items")
            
            for item in items:
                result = self.collect_item_data(item)
                if result:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1

            finished_at = datetime.utcnow()
            stats['finished_at'] = finished_at.isoformat()
            stats['duration_seconds'] = round((finished_at - started_at).total_seconds(), 3)
            logger.info(f"Collection complete: {stats['successful']} successful, {stats['failed']} failed")
            self._record_run_result(stats, stats['duration_seconds'], success=True)
            self._persist_collection_run(started_at, finished_at, stats, success=True)
            return stats
        except Exception as e:
            finished_at = datetime.utcnow()
            stats['finished_at'] = finished_at.isoformat()
            stats['duration_seconds'] = round((finished_at - started_at).total_seconds(), 3)
            self._record_run_result(stats, stats['duration_seconds'], success=False, error=str(e))
            self._persist_collection_run(started_at, finished_at, stats, success=False, error=str(e))
            logger.error(f"Error collecting all items: {e}", exc_info=True)
            raise
        
        finally:
            db.close()
    
    def run_collection_loop(self, interval_seconds: int = 3600):
        """
        Run continuous data collection loop
        
        Args:
            interval_seconds: Seconds between collection cycles (default: 1 hour)
        """
        logger.info(f"Starting real data collection loop (interval: {interval_seconds}s)")
        self._stop_event.clear()
        self.is_running = True
        
        while not self._stop_event.is_set():
            try:
                logger.info("Running scheduled data collection")
                self.collect_all_items()
                
                logger.info(f"Sleeping for {interval_seconds}s until next collection")
                if self._stop_event.wait(interval_seconds):
                    break
            
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                if self._stop_event.wait(60):
                    break

        self.is_running = False
        logger.info("Collection loop exited")
    
    def start_background_collection(self, interval_seconds: int = 3600):
        """
        Start collection as a background daemon thread
        
        Args:
            interval_seconds: Seconds between collection cycles
        """
        if not self.enabled:
            logger.info("Real data collection is disabled")
            return

        with self._thread_lock:
            if self.collection_thread and self.collection_thread.is_alive():
                logger.warning("Collection loop already running")
                return

            self._stop_event.clear()
            self.is_running = True
            self.collection_thread = threading.Thread(
                target=self.run_collection_loop,
                args=(interval_seconds,),
                daemon=True
            )
            self.collection_thread.start()
            logger.info("Background data collection started")
    
    def stop_background_collection(self):
        """Stop the background collection thread"""
        with self._thread_lock:
            self._stop_event.set()
            self.is_running = False
            thread = self.collection_thread
            thread_alive = bool(thread and thread.is_alive())
            if not thread_alive:
                self.collection_thread = None

        if thread and thread_alive:
            thread.join(timeout=5)

        with self._thread_lock:
            if self.collection_thread and not self.collection_thread.is_alive():
                self.collection_thread = None

        logger.info("Background data collection stopped")


# Global collector instance
_collector = None

def get_collector() -> RealDataCollector:
    """Get or create the global collector instance"""
    global _collector
    if _collector is None:
        _collector = RealDataCollector(enabled=True)
    return _collector

def start_real_data_collection():
    """Start real-time data collection (call from app startup)"""
    collector = get_collector()
    # Start background thread that collects every 1 hour
    collector.start_background_collection(interval_seconds=3600)

def stop_real_data_collection():
    """Stop real-time data collection (call from app shutdown)"""
    collector = get_collector()
    collector.stop_background_collection()
