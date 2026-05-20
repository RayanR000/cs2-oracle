"""
Database models for CS2 Market Intelligence Platform
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from config import settings

Base = declarative_base()

# Create engine
engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,  # Verify connections are alive before using
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)

class Item(Base):
    """Item model - skins, cases, stickers"""
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True)
    item_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # skin, case, sticker
    release_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    price_histories = relationship("PriceHistory", back_populates="item", cascade="all, delete-orphan")
    trend_indicators = relationship("TrendIndicator", back_populates="item", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_item_type', 'type'),
    )

class PriceHistory(Base):
    """Price history model - time-series price data"""
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)
    median_price = Column(Float, nullable=True)
    source = Column(String(50), nullable=False, default="steam")
    created_at = Column(DateTime, default=datetime.utcnow)

    item = relationship("Item", back_populates="price_histories")

    __table_args__ = (
        Index('idx_price_history_item_timestamp', 'item_id', 'timestamp'),
        Index('idx_price_history_source', 'source'),
    )

class CollectionRun(Base):
    """Collection run model - persisted collector health and run metadata"""
    __tablename__ = "collection_runs"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, nullable=False, index=True)
    finished_at = Column(DateTime, nullable=True, index=True)
    status = Column(String(50), nullable=False, index=True)
    total_items = Column(Integer, nullable=False, default=0)
    successful = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(String(1000), nullable=True)
    source_breakdown = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_collection_runs_started_at', 'started_at'),
        Index('idx_collection_runs_status', 'status'),
    )

class Event(Base):
    """Event model - market-moving events"""
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True)
    type = Column(String(50), nullable=False)  # major, update, case_drop, operation
    timestamp = Column(DateTime, nullable=False, index=True)
    description = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_event_type_timestamp', 'type', 'timestamp'),
    )

class TrendIndicator(Base):
    """Trend indicators model - computed analytics"""
    __tablename__ = "trend_indicators"
    
    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    sma_7 = Column(Float, nullable=True)  # 7-day simple moving average
    sma_30 = Column(Float, nullable=True)  # 30-day simple moving average
    volatility = Column(Float, nullable=True)  # Volatility measure
    trend_score = Column(Float, nullable=True)  # -1 (bearish) to 1 (bullish)
    trend_direction = Column(String(20), nullable=True)  # bullish, neutral, bearish
    confidence = Column(String(20), nullable=True)  # low, medium, high
    created_at = Column(DateTime, default=datetime.utcnow)
    
    item = relationship("Item", back_populates="trend_indicators")
    
    __table_args__ = (
        Index('idx_trend_item_timestamp', 'item_id', 'timestamp'),
    )

class User(Base):
    """User model - Steam authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    steam_id = Column(String(50), unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
