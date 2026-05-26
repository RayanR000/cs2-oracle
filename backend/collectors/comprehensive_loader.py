"""
Comprehensive data loader for CS2 items and historical data.

Loads the CS2 catalog plus optional demo history. Production startup should
use the catalog without synthetic backfill; demo/local runs can opt into the
generated history path.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database import SessionLocal, Item, PriceHistory, Event
from collectors.cs2_data_sources import CS2ItemCatalog, CS2GameEvents, HistoricalDataGenerator

logger = logging.getLogger(__name__)


class ComprehensiveDataLoader:
    """Loads complete CS2 item catalog with historical data"""
    
    def __init__(self, db: Optional[Session] = None):
        """Initialize data loader"""
        self.db = db or SessionLocal()
        self.items_loaded = 0
        self.price_records_loaded = 0
        self.events_loaded = 0
    
    def load_complete_catalog(self, 
                             generate_history: bool = False,
                             history_days: int = 365,
                             force_refresh: bool = False) -> dict:
        """
        Load complete CS2 item catalog with optional historical data
        
        Args:
            generate_history: Whether to generate synthetic historical price data
            history_days: Number of days of history to generate
            force_refresh: Whether to force a refresh of items even if they exist
            
        Returns:
            Dictionary with statistics about loaded data
        """
        stats = {
            'items_added': 0,
            'items_skipped': 0,
            'price_records_added': 0,
            'events_added': 0,
            'errors': []
        }
        
        try:
            # Skip item load if items already exist and force_refresh is False
            item_count = self.db.query(Item).count()
            if item_count > 0 and not force_refresh:
                logger.info(f"Database already has {item_count} items. Skipping catalog load.")
                stats['items_skipped'] = item_count
            else:
                # Load items
                stats.update(self._load_items())
            
            # NEVER generate history if items exist, unless explicitly forced
            if generate_history and (item_count == 0 or force_refresh):
                stats.update(self._load_historical_data(history_days))
            
            # Load game events
            stats.update(self._load_game_events())
            
            self.db.commit()
            logger.info(f"Data load complete: {stats}")
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during load: {e}")
            stats['errors'].append(f"Database error: {str(e)}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during load: {e}")
            stats['errors'].append(f"Unexpected error: {str(e)}")
        
        return stats
    
    def _load_items(self) -> dict:
        """Load all items from catalog using efficient batch checks"""
        stats = {'items_added': 0, 'items_skipped': 0}
        
        try:
            catalog_items = CS2ItemCatalog.get_all_items()
            logger.info(f"Checking catalog of {len(catalog_items)} items against database...")
            
            # Fetch all existing item names in one query
            existing_names = set(
                name for (name,) in self.db.query(Item.name).all()
            )
            logger.info(f"Found {len(existing_names)} existing items in DB")
            
            new_items = []
            for catalog_item in catalog_items:
                if catalog_item['name'] in existing_names:
                    stats['items_skipped'] += 1
                    continue
                
                # Create new item
                item = Item(
                    item_id=self._generate_item_id(catalog_item['name']),
                    name=catalog_item['name'],
                    type=catalog_item['type'],
                    release_date=catalog_item['release_date'],
                    current_price=HistoricalDataGenerator._get_base_price(
                        catalog_item['name']
                    )
                )
                new_items.append(item)
                stats['items_added'] += 1
            
            # Batch insert new items
            if new_items:
                logger.info(f"Batch inserting {len(new_items)} new items...")
                self.db.bulk_save_objects(new_items)
                self.db.flush()
            
            logger.info(f"Items load complete: {stats}")
            
        except Exception as e:
            logger.error(f"Error loading items: {e}", exc_info=True)
        
        return stats
    
    def _load_historical_data(self, history_days: int = 365) -> dict:
        """Generate and load historical price data for all items using efficient batch checks"""
        stats = {'price_records_added': 0}
        
        try:
            # Get all items
            items = self.db.query(Item).all()
            logger.info(f"Generating historical data for {len(items)} items")
            
            # Fetch all items that ALREADY have price history in one query
            items_with_history = set(
                item_id for (item_id,) in self.db.query(PriceHistory.item_id).distinct().all()
            )
            logger.info(f"Found {len(items_with_history)} items with existing history")
            
            all_new_prices = []
            for idx, item in enumerate(items):
                try:
                    # Skip if item already has price history
                    if item.id in items_with_history:
                        continue
                    
                    # Generate historical prices
                    prices = HistoricalDataGenerator.generate_historical_prices(
                        item.name,
                        item.release_date,
                        datetime.now(),
                        history_days
                    )
                    
                    # Add to batch
                    for timestamp, price, volume in prices:
                        price_record = PriceHistory(
                            item_id=item.id,
                            timestamp=timestamp,
                            price=price,
                            volume=volume,
                            median_price=price,
                            source="synthetic_bootstrap"
                        )
                        all_new_prices.append(price_record)
                    
                    stats['price_records_added'] += len(prices)
                    
                    # Flush occasionally to keep memory usage in check
                    if len(all_new_prices) > 10000:
                        logger.info(f"Batch inserting {len(all_new_prices)} synthetic records...")
                        self.db.bulk_save_objects(all_new_prices)
                        self.db.flush()
                        all_new_prices = []
                    
                except Exception as e:
                    logger.error(f"Error generating history for item {item.name}: {e}")
                    continue
            
            # Final flush
            if all_new_prices:
                logger.info(f"Batch inserting {len(all_new_prices)} final synthetic records...")
                self.db.bulk_save_objects(all_new_prices)
                self.db.flush()
                
            logger.info(f"Historical data load complete: {stats['price_records_added']} records added")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}", exc_info=True)
        
        return stats
    
    def _load_game_events(self) -> dict:
        """Load game events using batch checks"""
        stats = {'events_added': 0}
        
        try:
            events = CS2GameEvents.get_all_events()
            logger.info(f"Loading {len(events)} game events")
            
            # Fetch existing events (tuple of type and name for uniqueness)
            existing_events = set(
                (e.event_type, e.event_name) 
                for e in self.db.query(Event.event_type, Event.event_name).all()
            )
            
            new_events = []
            for event_data in events:
                if (event_data['type'], event_data['name']) in existing_events:
                    continue
                
                # Create new event
                event = Event(
                    event_type=event_data['type'],
                    event_name=event_data['name'],
                    description=event_data['description'],
                    timestamp=event_data['date']
                )
                new_events.append(event)
                stats['events_added'] += 1
            
            if new_events:
                self.db.bulk_save_objects(new_events)
                self.db.flush()
                
            logger.info(f"Events load complete: {stats['events_added']} events added")
            
        except Exception as e:
            logger.error(f"Error loading events: {e}", exc_info=True)
        
        return stats
    
    def _generate_item_id(self, item_name: str) -> str:
        """Generate item ID from name"""
        # Convert to lowercase, replace spaces and special chars with hyphens
        item_id = item_name.lower().replace(' | ', '-').replace(' ', '-')
        item_id = ''.join(c if c.isalnum() or c == '-' else '' for c in item_id)
        return item_id
    
    def load_incremental(self, since_date: Optional[datetime] = None) -> dict:
        """
        Load only new/updated items since a specific date
        
        Args:
            since_date: Only load items released after this date
            
        Returns:
            Statistics about incremental load
        """
        if since_date is None:
            since_date = datetime.now() - timedelta(days=30)
        
        stats = {'items_added': 0, 'price_records_added': 0}
        
        try:
            catalog_items = CS2ItemCatalog.get_all_items()
            recent_items = [i for i in catalog_items if i['release_date'] >= since_date]
            
            logger.info(f"Loading {len(recent_items)} recent items since {since_date}")
            
            # Load items (existing _load_items logic for recent items only)
            for catalog_item in recent_items:
                existing = self.db.query(Item).filter(
                    Item.name == catalog_item['name']
                ).first()
                
                if not existing:
                    item = Item(
                        item_id=self._generate_item_id(catalog_item['name']),
                        name=catalog_item['name'],
                        type=catalog_item['type'],
                        release_date=catalog_item['release_date'],
                        current_price=HistoricalDataGenerator._get_base_price(
                            catalog_item['name']
                        )
                    )
                    self.db.add(item)
                    stats['items_added'] += 1
            
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in incremental load: {e}")
        
        return stats


def load_all_cs2_data(
    db: Optional[Session] = None,
    generate_history: bool = False,
    history_days: int = 365
) -> dict:
    """
    Convenience function to load the CS2 catalog, demo events, and optional
    synthetic history.
    
    Args:
        db: Database session (uses default if not provided)
        
    Returns:
        Statistics from load operation
    """
    loader = ComprehensiveDataLoader(db)
    return loader.load_complete_catalog(
        generate_history=generate_history,
        history_days=history_days
    )


def load_demo_cs2_data(db: Optional[Session] = None, history_days: int = 365) -> dict:
    """
    Convenience function to load the CS2 catalog with synthetic demo history.
    """
    return load_all_cs2_data(db=db, generate_history=True, history_days=history_days)


def load_catalog_only(db: Optional[Session] = None) -> dict:
    """
    Convenience function to load the CS2 catalog and events without backfill.
    """
    return load_all_cs2_data(db=db, generate_history=False)


if __name__ == "__main__":
    # Can be run standalone for one-time data load
    logging.basicConfig(level=logging.INFO)
    stats = load_demo_cs2_data()
    print(f"Data load statistics: {stats}")
