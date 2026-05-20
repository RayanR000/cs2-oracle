"""
Real-time data collection service
Fetches real CS2 market data from Steam API
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import threading
import time
from collectors.steam_market import SteamMarketCollector
from collectors.data_validation import DataValidator, DataCleaner
from database import SessionLocal, Item, PriceHistory, Event
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
                    median_price=cleaned_price
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
        stats = {
            'total_items': 0,
            'successful': 0,
            'failed': 0,
            'timestamp': datetime.utcnow().isoformat()
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
            
            logger.info(f"Collection complete: {stats['successful']} successful, {stats['failed']} failed")
            return stats
        
        finally:
            db.close()
    
    def run_collection_loop(self, interval_seconds: int = 3600):
        """
        Run continuous data collection loop
        
        Args:
            interval_seconds: Seconds between collection cycles (default: 1 hour)
        """
        logger.info(f"Starting real data collection loop (interval: {interval_seconds}s)")
        self.is_running = True
        
        while self.is_running:
            try:
                logger.info("Running scheduled data collection")
                self.collect_all_items()
                
                logger.info(f"Sleeping for {interval_seconds}s until next collection")
                time.sleep(interval_seconds)
            
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(60)  # Sleep 1 minute on error before retrying
    
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
            if self.is_running:
                logger.warning("Collection loop already running")
                return

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
            self.is_running = False
            thread = self.collection_thread
            self.collection_thread = None

        if thread:
            thread.join(timeout=5)
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
