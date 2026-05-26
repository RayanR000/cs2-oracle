"""
Database operations and queries
Handles all database interactions
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from database import Item, PriceHistory, Event, TrendIndicator

logger = logging.getLogger(__name__)

class ItemRepository:
    """Repository for item operations"""
    
    @staticmethod
    def get_all_items(db: Session, skip: int = 0, limit: int = 50) -> List[Item]:
        """Get all items with pagination"""
        return db.query(Item).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_item_by_id(db: Session, item_id: str) -> Optional[Item]:
        """Get item by item_id"""
        return db.query(Item).filter(Item.item_id == item_id).first()
    
    @staticmethod
    def search_items(db: Session, query: str, limit: int = 10) -> List[Item]:
        """Search items by name (case-insensitive)"""
        search_term = f"%{query}%"
        return db.query(Item).filter(
            Item.name.ilike(search_term)
        ).limit(limit).all()
    
    @staticmethod
    def get_items_by_type(db: Session, item_type: str, limit: int = 50) -> List[Item]:
        """Get items by type"""
        return db.query(Item).filter(Item.type == item_type).limit(limit).all()
    
    @staticmethod
    def create_item(db: Session, item_id: str, name: str, item_type: str, 
                   release_date: Optional[datetime] = None) -> Item:
        """Create a new item"""
        item = Item(
            item_id=item_id,
            name=name,
            type=item_type,
            release_date=release_date
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return item
    
    @staticmethod
    def get_trending_items(db: Session, days: int = 7, limit: int = 10) -> List[Dict]:
        """Get trending items by price change"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get latest price for each item
        latest_prices = db.query(
            Item.id,
            Item.name,
            Item.type,
            func.max(PriceHistory.price).label('latest_price')
        ).join(PriceHistory).filter(
            PriceHistory.timestamp >= cutoff_date
        ).group_by(Item.id, Item.name, Item.type).all()
        
        return [
            {
                'item_id': item[0],
                'name': item[1],
                'type': item[2],
                'latest_price': item[3]
            }
            for item in latest_prices
        ][:limit]

    @staticmethod
    def get_top_items(db: Session, limit: int = 2000) -> List[Item]:
        """
        Identify priority items based on volume and price.
        Returns the top items that should be scraped more frequently.
        """
        # Use a subquery to find the latest price history record for every item
        latest_history_ids = db.query(
            func.max(PriceHistory.id)
        ).group_by(PriceHistory.item_id).subquery()
        
        # Return items joined with their latest stats, ordered by volume then price
        return db.query(Item).join(
            PriceHistory, Item.id == PriceHistory.item_id
        ).filter(
            PriceHistory.id.in_(latest_history_ids)
        ).order_by(
            desc(PriceHistory.volume),
            desc(PriceHistory.price)
        ).limit(limit).all()


class PriceHistoryRepository:
    """Repository for price history operations"""
    
    @staticmethod
    def get_price_history(db: Session, item_id: int, days: int = 30, 
                         skip: int = 0, limit: int = 1000) -> List[PriceHistory]:
        """Get price history for an item"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return db.query(PriceHistory).filter(
            PriceHistory.item_id == item_id,
            PriceHistory.timestamp >= cutoff_date
        ).order_by(
            PriceHistory.timestamp
        ).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_latest_price(db: Session, item_id: int) -> Optional[PriceHistory]:
        """Get the most recent price for an item"""
        return db.query(PriceHistory).filter(
            PriceHistory.item_id == item_id
        ).order_by(desc(PriceHistory.timestamp)).first()
    
    @staticmethod
    def get_price_statistics(db: Session, item_id: int, days: int = 30) -> Dict:
        """Get price statistics for an item"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        prices = db.query(PriceHistory.price).filter(
            PriceHistory.item_id == item_id,
            PriceHistory.timestamp >= cutoff_date
        ).all()
        
        if not prices:
            return {}
        
        prices = [p[0] for p in prices]
        
        return {
            'min': min(prices),
            'max': max(prices),
            'avg': sum(prices) / len(prices),
            'current': prices[-1] if prices else None,
            'count': len(prices)
        }
    
    @staticmethod
    def add_price_record(db: Session, item_id: int, price: float, 
                        volume: Optional[int] = None) -> PriceHistory:
        """Add a new price record"""
        record = PriceHistory(
            item_id=item_id,
            timestamp=datetime.utcnow(),
            price=price,
            volume=volume
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record


class EventRepository:
    """Repository for event operations"""
    
    @staticmethod
    def get_all_events(db: Session, skip: int = 0, limit: int = 50) -> List[Event]:
        """Get all events"""
        return db.query(Event).order_by(
            desc(Event.timestamp)
        ).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_events_by_type(db: Session, event_type: str, limit: int = 20) -> List[Event]:
        """Get events by type"""
        return db.query(Event).filter(
            Event.type == event_type
        ).order_by(desc(Event.timestamp)).limit(limit).all()
    
    @staticmethod
    def get_recent_events(db: Session, days: int = 30, limit: int = 20) -> List[Event]:
        """Get recent events"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return db.query(Event).filter(
            Event.timestamp >= cutoff_date
        ).order_by(desc(Event.timestamp)).limit(limit).all()
    
    @staticmethod
    def create_event(db: Session, event_type: str, timestamp: datetime, 
                    description: str) -> Event:
        """Create a new event"""
        event = Event(
            type=event_type,
            timestamp=timestamp,
            description=description
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event


class TrendIndicatorRepository:
    """Repository for trend indicator operations"""
    
    @staticmethod
    def get_latest_trend(db: Session, item_id: int) -> Optional[TrendIndicator]:
        """Get latest trend indicator for an item"""
        return db.query(TrendIndicator).filter(
            TrendIndicator.item_id == item_id
        ).order_by(desc(TrendIndicator.timestamp)).first()
    
    @staticmethod
    def get_trend_history(db: Session, item_id: int, days: int = 30) -> List[TrendIndicator]:
        """Get trend history for an item"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return db.query(TrendIndicator).filter(
            TrendIndicator.item_id == item_id,
            TrendIndicator.timestamp >= cutoff_date
        ).order_by(TrendIndicator.timestamp).all()
    
    @staticmethod
    def create_trend_indicator(db: Session, item_id: int, sma_7: Optional[float] = None,
                              sma_30: Optional[float] = None, volatility: Optional[float] = None,
                              trend_score: Optional[float] = None, 
                              trend_direction: Optional[str] = None,
                              confidence: Optional[str] = None) -> TrendIndicator:
        """Create a new trend indicator"""
        indicator = TrendIndicator(
            item_id=item_id,
            timestamp=datetime.utcnow(),
            sma_7=sma_7,
            sma_30=sma_30,
            volatility=volatility,
            trend_score=trend_score,
            trend_direction=trend_direction,
            confidence=confidence
        )
        db.add(indicator)
        db.commit()
        db.refresh(indicator)
        return indicator


class UserRepository:
    """Repository for user operations"""
    
    @staticmethod
    def get_user_by_steam_id(db: Session, steam_id: str) -> Optional[User]:
        """Get user by steam_id"""
        from database import User
        return db.query(User).filter(User.steam_id == steam_id).first()
    
    @staticmethod
    def create_user(db: Session, steam_id: str, username: Optional[str] = None, 
                    avatar_url: Optional[str] = None) -> User:
        """Create a new user"""
        from database import User
        user = User(
            steam_id=steam_id,
            username=username,
            avatar_url=avatar_url
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def update_user(db: Session, steam_id: str, username: Optional[str] = None,
                    avatar_url: Optional[str] = None) -> Optional[User]:
        """Update user profile information"""
        from database import User
        user = db.query(User).filter(User.steam_id == steam_id).first()
        if user:
            if username:
                user.username = username
            if avatar_url:
                user.avatar_url = avatar_url
            user.last_login = datetime.utcnow()
            db.commit()
            db.refresh(user)
        return user
