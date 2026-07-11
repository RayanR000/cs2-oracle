"""
Database models for CS2 Market Intelligence Platform
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, ForeignKey, Index, JSON, UniqueConstraint
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone
from config import settings

Base = declarative_base()


def utcnow_naive():
    """Return a naive UTC timestamp for compatibility with the current schema."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

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
    item_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False, index=True)
    type = Column(String(50), nullable=False)  # skin, case, sticker
    icon_url = Column(String(512), nullable=True)
    classid = Column(String(64), nullable=True)
    instanceid = Column(String(64), nullable=True)
    release_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)
    updated_at = Column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)
    is_backfilled = Column(Integer, default=0)  # boolean: has CSMarketAPI historical series
    
    price_histories = relationship("PriceHistory", back_populates="item", cascade="all, delete-orphan")
    daily_analyses = relationship("DailyAnalysis", back_populates="item", cascade="all, delete-orphan")
    forecasts = relationship("ItemForecast", back_populates="item", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_item_type', 'type'),
    )

class PriceHistory(Base):
    """Price history model - time-series price data"""
    __tablename__ = "price_history"

    # Composite natural primary key — no surrogate id. Saves the pkey index
    # (~80 MB at 2.8M rows) and lets the PK arbitrate the
    # ON CONFLICT (item_id, timestamp, source) upserts used by all writers.
    item_id = Column(Integer, ForeignKey("items.id"), primary_key=True)
    timestamp = Column(DateTime, primary_key=True, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)
    median_price = Column(Float, nullable=True)
    source = Column(String(255), primary_key=True, default="steam")
    created_at = Column(DateTime, default=utcnow_naive)

    item = relationship("Item", back_populates="price_histories")

    __table_args__ = (
        Index('idx_price_history_source', 'source'),
    )

# Sources whose presence marks an item as "backfilled": it has a real
# historical price series (from CSMarketAPI STEAMCOMMUNITY data), not just a
# live snapshot. Kept for backward compat; the canonical filter is now
# Item.is_backfilled == True.
BACKFILLED_SOURCES = ("steam_daily",)


def backfilled_item_clause():
    """SQLAlchemy filter expression: item has backfilled history.

    Listing endpoints filter on this so the site only surfaces items with
    enough data for charts and analysis; snapshot-tier items stay reachable
    by direct link but are not listed.
    """
    return Item.is_backfilled == 1


class CollectionRun(Base):
    """Collection run model - persisted collector health and run metadata"""
    __tablename__ = "collection_runs"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime, nullable=True, index=True)
    status = Column(String(50), nullable=False)
    total_items = Column(Integer, nullable=False, default=0)
    successful = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(String(1000), nullable=True)
    source_breakdown = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)

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
    created_at = Column(DateTime, default=utcnow_naive)
    
    __table_args__ = (
        Index('idx_event_type_timestamp', 'type', 'timestamp'),
    )

class DailyAnalysis(Base):
    """Daily analysis model - per-item daily computed signals."""
    __tablename__ = "daily_analysis"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    analysis_date = Column(Date, nullable=False)
    current_price = Column(Float, nullable=True)
    ma_7day = Column(Float, nullable=True)
    ma_30day = Column(Float, nullable=True)
    ma_90day = Column(Float, nullable=True)
    momentum_7day = Column(Float, nullable=True)
    momentum_30day = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    trend_direction = Column(String(20), nullable=True)
    momentum_score = Column(Float, nullable=True)
    opportunity_score = Column(Float, nullable=True)
    trading_volume_trend = Column(Float, nullable=True)
    price_stability = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)

    item = relationship("Item", back_populates="daily_analyses")

    __table_args__ = (
        UniqueConstraint('item_id', 'analysis_date', name='uq_daily_analysis_item_date'),
        Index('idx_daily_analysis_item_date', 'item_id', 'analysis_date'),
    )

class ItemForecast(Base):
    """ML model forecasts - LightGBM quantile regression predictions"""
    __tablename__ = "item_forecasts"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    forecast_date = Column(Date, nullable=False)
    horizon_days = Column(Integer, nullable=False)  # 7 or 30
    price_low = Column(Float, nullable=True)  # p10 quantile
    price_mid = Column(Float, nullable=True)  # p50 quantile (median)
    price_high = Column(Float, nullable=True)  # p90 quantile
    current_price = Column(Float, nullable=True)
    direction = Column(String(10), nullable=True)  # up, down, flat
    confidence = Column(String(10), nullable=True)  # low, medium, high
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)

    item = relationship("Item", back_populates="forecasts")

    __table_args__ = (
        Index('idx_forecast_item_date', 'item_id', 'forecast_date', 'horizon_days'),
        UniqueConstraint('item_id', 'forecast_date', 'horizon_days',
                         name='uq_item_forecast_date_horizon'),
    )


class User(Base):
    """User model - Steam authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    steam_id = Column(String(50), unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)
    last_login = Column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)


class EventImpact(Base):
    """Event impact model - historical price movements around events"""
    __tablename__ = "event_impacts"

    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=True)
    price_day_before = Column(Float, nullable=True)
    price_day_1 = Column(Float, nullable=True)
    price_day_3 = Column(Float, nullable=True)
    price_day_7 = Column(Float, nullable=True)
    impact_pct_1day = Column(Float, nullable=True)
    impact_pct_3day = Column(Float, nullable=True)
    impact_pct_7day = Column(Float, nullable=True)
    peak_impact_pct = Column(Float, nullable=True)
    peak_impact_day = Column(Integer, nullable=True)
    duration_days = Column(Integer, nullable=True)
    z_score = Column(Float, nullable=True)  # Statistical significance
    created_at = Column(DateTime, default=utcnow_naive)

    __table_args__ = (
        Index('idx_event_impact_event_item', 'event_id', 'item_id'),
        UniqueConstraint('event_id', 'item_id', name='uq_event_impact_event_item'),
    )


class EventPattern(Base):
    """Event pattern model - learned patterns from historical events"""
    __tablename__ = "event_patterns"

    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=True)
    sample_size = Column(Integer, nullable=False, default=0)
    avg_impact_1day = Column(Float, nullable=True)
    avg_impact_3day = Column(Float, nullable=True)
    avg_impact_7day = Column(Float, nullable=True)
    std_dev = Column(Float, nullable=True)
    consistency_score = Column(Float, nullable=True)  # 0-1: how consistent is the pattern
    holdout_accuracy = Column(Float, nullable=True)  # 0-1: validation accuracy
    created_at = Column(DateTime, default=utcnow_naive)
    updated_at = Column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)

    __table_args__ = (
        Index('idx_event_pattern_type_item', 'event_type', 'item_id'),
        UniqueConstraint('event_type', 'item_id', name='uq_event_pattern_type_item'),
    )


class PredictionAccuracy(Base):
    """Accuracy tracking for all prediction/analysis types.

    Stores aggregated accuracy metrics computed by the backtesting system.
    prediction_type: forecast | trend_direction | opportunity | event_impact
    metrics JSON schema varies by type:
      - forecast:      {mae, rmse, mape, directional_accuracy, interval_coverage,
                        confidence_accuracy_low, confidence_accuracy_medium,
                        confidence_accuracy_high, sample_count, horizon_days}
      - trend_direction: {overall_accuracy, confusion_matrix, sample_count,
                         avg_subsequent_return, avg_subsequent_return_days}
      - opportunity:     {undervalued_precision, undervalued_recall, overheated_precision,
                         overheated_recall, momentum_precision, avg_return, sample_count}
      - event_impact:    {mae, rmse, directional_accuracy, sample_count}
    """
    __tablename__ = "prediction_accuracy"

    id = Column(Integer, primary_key=True)
    prediction_type = Column(String(50), nullable=False, index=True)
    evaluation_date = Column(Date, nullable=False, index=True)
    horizon_days = Column(Integer, nullable=True)
    model_version = Column(String(50), nullable=True)
    evaluation_window_days = Column(Integer, nullable=True)
    sample_count = Column(Integer, nullable=False, default=0)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utcnow_naive)

    __table_args__ = (
        Index('idx_accuracy_type_date', 'prediction_type', 'evaluation_date'),
        UniqueConstraint('prediction_type', 'evaluation_date', 'horizon_days', 'model_version',
                         name='uq_accuracy_type_date_horizon_model'),
    )


class EventCorrelation(Base):
    """Event correlation model - causal analysis with statistical rigor"""
    __tablename__ = "event_correlations"

    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=True)

    # Raw measurements
    price_change_pct = Column(Float, nullable=True)
    control_group_change_pct = Column(Float, nullable=True)

    # 6-point statistical rigor checks
    significance_test_zscore = Column(Float, nullable=True)  # Is change > 2x baseline variance?
    significance_passed = Column(Integer, default=0)  # 0/1

    control_group_diff = Column(Float, nullable=True)  # Affected - Control
    control_group_passed = Column(Integer, default=0)  # 0/1: Is affected > control?

    pattern_consistency_score = Column(Float, nullable=True)  # 0-1: Does pattern repeat?
    pattern_passed = Column(Integer, default=0)  # 0/1: > 0.7 consistency?

    confounding_events_count = Column(Integer, default=0)  # Events same day?
    confounding_passed = Column(Integer, default=0)  # 0/1: Only 1 event on date?

    lag_analysis_peak_day = Column(Integer, nullable=True)  # When is impact strongest?
    lag_passed = Column(Integer, default=0)  # 0/1: Peak within expected window?

    holdout_validation_accuracy = Column(Float, nullable=True)  # How well does pattern work on new data?
    validation_passed = Column(Integer, default=0)  # 0/1: > 0.6 accuracy?

    # Final confidence score
    confidence_score = Column(Float, nullable=True)  # 0-1: Weighted average of 6 checks

    created_at = Column(DateTime, default=utcnow_naive)

    __table_args__ = (
        Index('idx_event_correlation_event_item', 'event_id', 'item_id'),
        UniqueConstraint('event_id', 'item_id', name='uq_event_correlation_event_item'),
    )
