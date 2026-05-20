"""
Seed data and database initialization
Populates initial data for testing and demonstration only.

This module intentionally generates synthetic market history for local/demo
runs. Production bootstrap should avoid using these helpers.
"""

from datetime import datetime, timedelta
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Sample CS2 market items
SAMPLE_ITEMS = [
    # AK-47 Skins
    {'item_id': 'ak47-phantom-mw', 'name': 'AK-47 | Phantom Disruptor', 'type': 'skin', 'release_date': datetime(2020, 5, 1)},
    {'item_id': 'ak47-neon-ride', 'name': 'AK-47 | Neon Ride', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    {'item_id': 'ak47-front-side', 'name': 'AK-47 | Frontside Misty', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    {'item_id': 'ak47-phantom', 'name': 'AK-47 | Phantom Disruptor', 'type': 'skin', 'release_date': datetime(2020, 5, 1)},
    {'item_id': 'ak47-legion', 'name': 'AK-47 | Legion of Anubis', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    
    # M4A4/M4A1-S Skins
    {'item_id': 'm4a1-hyper', 'name': 'M4A1-S | Hyper Beast', 'type': 'skin', 'release_date': datetime(2015, 2, 1)},
    {'item_id': 'm4a4-asiimov', 'name': 'M4A4 | Asiimov', 'type': 'skin', 'release_date': datetime(2014, 1, 1)},
    {'item_id': 'm4a4-poseidon', 'name': 'M4A4 | Poseidon', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    {'item_id': 'm4a1-masterpiece', 'name': 'M4A1-S | Masterpiece', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    
    # AWP Dragon Lore Variants
    {'item_id': 'awp-dragon-lore', 'name': 'AWP Dragon Lore', 'type': 'skin', 'release_date': datetime(2013, 1, 1)},
    {'item_id': 'awp-asiimov', 'name': 'AWP Asiimov', 'type': 'skin', 'release_date': datetime(2014, 1, 1)},
    {'item_id': 'awp-medusa', 'name': 'AWP Medusa', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    {'item_id': 'awp-pink-ddpat', 'name': 'AWP Pink DDPAT', 'type': 'skin', 'release_date': datetime(2013, 8, 14)},
    
    # Knife Skins
    {'item_id': 'karambit-doppler', 'name': 'Karambit | Doppler', 'type': 'skin', 'release_date': datetime(2015, 1, 6)},
    {'item_id': 'karambit-marble', 'name': 'Karambit | Marble Fade', 'type': 'skin', 'release_date': datetime(2015, 1, 6)},
    {'item_id': 'butterfly-fade', 'name': 'Butterfly Knife | Fade', 'type': 'skin', 'release_date': datetime(2015, 1, 6)},
    {'item_id': 'bayonet-doppler', 'name': 'Bayonet | Doppler', 'type': 'skin', 'release_date': datetime(2015, 1, 6)},
    {'item_id': 'bowie-fade', 'name': 'Bowie Knife | Fade', 'type': 'skin', 'release_date': datetime(2015, 1, 6)},
    
    # Pistol Skins
    {'item_id': 'deagle-crimson-web', 'name': 'Desert Eagle | Crimson Web', 'type': 'skin', 'release_date': datetime(2014, 1, 21)},
    {'item_id': 'deagle-blaze', 'name': 'Desert Eagle | Blaze', 'type': 'skin', 'release_date': datetime(2014, 1, 21)},
    {'item_id': 'usp-neo-noir', 'name': 'USP-S | Neo-Noir', 'type': 'skin', 'release_date': datetime(2017, 9, 18)},
    {'item_id': 'glock-dragon-tattoo', 'name': 'Glock-18 | Dragon Tattoo', 'type': 'skin', 'release_date': datetime(2014, 1, 21)},
    
    # SMG & Rifle Skins
    {'item_id': 'ak47-neon', 'name': 'AK-47 | Neon Rider', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    {'item_id': 'famas-djinn', 'name': 'FAMAS | Djinn', 'type': 'skin', 'release_date': datetime(2017, 5, 23)},
    {'item_id': 'galil-chatterbox', 'name': 'Galil AR | Chatterbox', 'type': 'skin', 'release_date': datetime(2016, 8, 9)},
    {'item_id': 'mp9-briefcase', 'name': 'MP9 | Briefcase', 'type': 'skin', 'release_date': datetime(2017, 5, 23)},
    
    # Low Price Skins
    {'item_id': 'p250-sand-dune', 'name': 'P250 | Sand Dune', 'type': 'skin', 'release_date': datetime(2013, 8, 14)},
    {'item_id': 'famas-pulse', 'name': 'FAMAS | Pulse', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    {'item_id': 'ump-primal', 'name': 'UMP-45 | Primal Saber', 'type': 'skin', 'release_date': datetime(2017, 5, 23)},
    
    # Cases
    {'item_id': 'cs2-weapon-case', 'name': 'CS2 Weapon Case', 'type': 'case', 'release_date': datetime(2023, 9, 1)},
    {'item_id': 'operation-bravo-case', 'name': 'Operation Bravo Case', 'type': 'case', 'release_date': datetime(2014, 8, 28)},
    {'item_id': 'spectrum-2-case', 'name': 'Spectrum 2 Case', 'type': 'case', 'release_date': datetime(2017, 9, 18)},
    {'item_id': 'shadow-case', 'name': 'Shadow Case', 'type': 'case', 'release_date': datetime(2017, 5, 23)},
    {'item_id': 'clutch-case', 'name': 'Clutch Case', 'type': 'case', 'release_date': datetime(2018, 3, 22)},
    
    # Collections / Special Items
    {'item_id': 'dragon-lore-factory', 'name': 'Dragon Lore', 'type': 'skin', 'release_date': datetime(2013, 1, 1)},
    {'item_id': 'souvenir-packages', 'name': 'Souvenir Packages', 'type': 'case', 'release_date': datetime(2013, 8, 14)},
    
    # Stickers (Popular Teams/Events)
    {'item_id': 'sticker-navi', 'name': 'Navi Sticker', 'type': 'sticker', 'release_date': datetime(2022, 5, 15)},
    {'item_id': 'sticker-astralis', 'name': 'Astralis Sticker', 'type': 'sticker', 'release_date': datetime(2017, 1, 1)},
    {'item_id': 'sticker-faze', 'name': 'FaZe Clan Sticker', 'type': 'sticker', 'release_date': datetime(2017, 1, 1)},
    {'item_id': 'sticker-liquid', 'name': 'Team Liquid Sticker', 'type': 'sticker', 'release_date': datetime(2017, 1, 1)},
    {'item_id': 'sticker-sk', 'name': 'SK Gaming Sticker', 'type': 'sticker', 'release_date': datetime(2017, 1, 1)},
    
    # Major Skins (High Value)
    {'item_id': 'ak47-point-disarray', 'name': 'AK-47 | Point Disarray', 'type': 'skin', 'release_date': datetime(2017, 9, 18)},
    {'item_id': 'ak47-nightwish', 'name': 'AK-47 | Nightwish', 'type': 'skin', 'release_date': datetime(2016, 8, 9)},
    {'item_id': 'm4a1-nightmare', 'name': 'M4A1-S | Nightmare', 'type': 'skin', 'release_date': datetime(2016, 8, 9)},
    {'item_id': 'deagle-kumicho', 'name': 'Desert Eagle | Kumicho Dragon', 'type': 'skin', 'release_date': datetime(2016, 8, 9)},
    
    # Budget Friendly Items
    {'item_id': 'glock-wasteland', 'name': 'Glock-18 | Wasteland Rebel', 'type': 'skin', 'release_date': datetime(2015, 8, 18)},
    {'item_id': 'ak47-uncharted', 'name': 'AK-47 | Uncharted', 'type': 'skin', 'release_date': datetime(2018, 12, 6)},
    {'item_id': 'glock-catacombs', 'name': 'Glock-18 | Catacombs', 'type': 'skin', 'release_date': datetime(2017, 5, 23)},
]

# Sample market events
SAMPLE_EVENTS = [
    {
        'type': 'major',
        'timestamp': datetime.utcnow() - timedelta(days=60),
        'description': 'PGL Major Stockholm 2024'
    },
    {
        'type': 'case_drop',
        'timestamp': datetime.utcnow() - timedelta(days=45),
        'description': 'New weapon case added to drop pool'
    },
    {
        'type': 'operation',
        'timestamp': datetime.utcnow() - timedelta(days=30),
        'description': 'Operation Breakout started'
    },
    {
        'type': 'update',
        'timestamp': datetime.utcnow() - timedelta(days=15),
        'description': 'Major balance update affecting weapon prices'
    },
    {
        'type': 'major',
        'timestamp': datetime.utcnow() - timedelta(days=7),
        'description': 'Intel Extreme Masters World Championship'
    },
]


def generate_sample_price_history(item_id: str, num_days: int = 90) -> List[Dict]:
    """
    Generate realistic synthetic price history with market patterns.
    
    Args:
        item_id: Item identifier
        num_days: Number of days of history to generate
        
    Returns:
        List of price data points with realistic patterns
    """
    import random
    import math
    
    history = []
    
    # Create realistic base prices by item type
    base_prices = {
        'ak47-phantom-mw': 45.0,
        'dragon-lore-factory': 1200.0,
        'cs2-weapon-case': 2.50,
        'sticker-navi': 8.50,
        'deagle-crimson-web': 95.0,
        'karambit-doppler': 380.0,
        'm4a1-hyper': 65.0,
        'awp-dragon-lore': 950.0,
    }
    
    base_price = base_prices.get(item_id, random.uniform(20, 500))
    current_price = base_price
    
    # Add overall trend (slight upward or downward)
    trend = random.uniform(-0.001, 0.002)  # Daily trend
    volatility = random.uniform(0.01, 0.05)  # Daily volatility
    
    for day in range(num_days):
        timestamp = datetime.utcnow() - timedelta(days=num_days - day - 1)
        
        # Random walk with drift
        random_component = random.gauss(0, volatility)
        price_change = trend + random_component
        
        # Add occasional spikes (market events)
        if random.random() < 0.05:  # 5% chance of event
            price_change += random.uniform(-0.15, 0.15)
        
        current_price = current_price * (1 + price_change)
        current_price = max(0.01, current_price)  # Ensure positive
        
        # Volume patterns: higher on weekends, lower on weekdays
        day_of_week = timestamp.weekday()
        base_volume = 500 if day_of_week >= 4 else 200  # Higher on weekends
        volume = int(base_volume * random.uniform(0.5, 2.0))
        
        # Median price is typically close to current price
        median_price = current_price * random.uniform(0.95, 1.05)
        
        history.append({
            'item_id': item_id,
            'timestamp': timestamp,
            'price': round(current_price, 2),
            'volume': volume,
            'median_price': round(median_price, 2)
        })
    
    return history


class DatabaseSeeder:
    """Seeds database with initial data"""
    
    @staticmethod
    def seed_items(session) -> int:
        """
        Seed items table with sample data
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            Number of items seeded
        """
        from database import Item
        
        try:
            # Check if items already exist
            existing_count = session.query(Item).count()
            if existing_count > 0:
                logger.info(f"Items table already has {existing_count} items, skipping seed")
                return 0
            
            items = []
            for item_data in SAMPLE_ITEMS:
                item = Item(**item_data)
                items.append(item)
            
            session.add_all(items)
            session.commit()
            
            logger.info(f"Seeded {len(items)} items")
            return len(items)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding items: {e}")
            return 0
    
    @staticmethod
    def seed_events(session) -> int:
        """
        Seed events table with sample data
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            Number of events seeded
        """
        from database import Event
        
        try:
            # Check if events already exist
            existing_count = session.query(Event).count()
            if existing_count > 0:
                logger.info(f"Events table already has {existing_count} events, skipping seed")
                return 0
            
            events = []
            for event_data in SAMPLE_EVENTS:
                event = Event(**event_data)
                events.append(event)
            
            session.add_all(events)
            session.commit()
            
            logger.info(f"Seeded {len(events)} events")
            return len(events)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding events: {e}")
            return 0
    
    @staticmethod
    def seed_price_history(session) -> int:
        """
        Seed price history with sample data
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            Number of price history records seeded
        """
        from database import Item, PriceHistory
        
        try:
            # Check if price history exists
            existing_count = session.query(PriceHistory).count()
            if existing_count > 0:
                logger.info(f"PriceHistory table already has {existing_count} records, skipping seed")
                return 0
            
            items = session.query(Item).all()
            total_added = 0
            
            for item in items:
                history_data = generate_sample_price_history(item.item_id)
                
                for record in history_data:
                    price_history = PriceHistory(
                        item_id=item.id,
                        timestamp=record['timestamp'],
                        price=record['price'],
                        volume=record['volume'],
                        median_price=record.get('median_price')
                    )
                    session.add(price_history)
                    total_added += 1
            
            session.commit()
            logger.info(f"Seeded {total_added} price history records")
            return total_added
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding price history: {e}")
            return 0
    
    @staticmethod
    def seed_all(session) -> Dict[str, int]:
        """
        Seed all tables with synthetic demo data.
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            Dictionary with counts of seeded records
        """
        logger.info("Starting database seeding...")
        
        results = {
            'items': DatabaseSeeder.seed_items(session),
            'events': DatabaseSeeder.seed_events(session),
            'price_history': DatabaseSeeder.seed_price_history(session)
        }
        
        logger.info(f"Database seeding completed: {results}")
        return results
