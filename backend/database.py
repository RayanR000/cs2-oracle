"""
Database models for CS2 Market Intelligence Platform
"""

from datetime import UTC, datetime

from config import settings
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


def utcnow_naive():
    """Return a naive UTC timestamp for compatibility with the current schema."""
    return datetime.now(UTC).replace(tzinfo=None)


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
    is_trainable = Column(Integer, default=0)  # boolean: eligible for the TRAIN universe (non-iflow pre-2026 history)

    # Supply-side metadata (populated from Steam Market type field)
    rarity = Column(
        String(50), nullable=True
    )  # e.g. covert, milspec, restricted, classified, consumer, industrial, base, etc.
    rarity_rank = Column(Integer, nullable=True)  # ordinal 0-6 (higher = rarer)
    weapon_type = Column(String(50), nullable=True)  # e.g. rifle, pistol, smg, knife, glove, sticker, case, charm, etc.

    price_histories = relationship("PriceHistory", back_populates="item", cascade="all, delete-orphan")
    forecasts = relationship("ItemForecast", back_populates="item", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_item_type", "type"),
        Index("idx_item_rarity", "rarity"),
        Index("idx_item_weapon_type", "weapon_type"),
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

    __table_args__ = (Index("idx_price_history_source", "source"),)


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


def trainable_item_clause():
    """Items eligible for the TRAINING universe: pre-2026 history from a
    non-`buff_iflow` source. Distinct from `backfilled_item_clause()`, which is
    the wider SERVE universe that includes iflow-backfilled items."""
    return Item.is_trainable == 1


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
        Index("idx_collection_runs_started_at", "started_at"),
        Index("idx_collection_runs_status", "status"),
    )


class Event(Base):
    """Event model - market-moving events"""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    type = Column(String(50), nullable=False)  # major, update, case_drop, operation
    timestamp = Column(DateTime, nullable=False, index=True)
    description = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=utcnow_naive)

    __table_args__ = (Index("idx_event_type_timestamp", "type", "timestamp"),)


class ItemForecast(Base):
    """ML model forecasts - LightGBM quantile regression predictions"""

    __tablename__ = "item_forecasts"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    forecast_date = Column(Date, nullable=False)
    horizon_days = Column(Integer, nullable=False)  # 3, 7, 14, or 30
    price_low = Column(Float, nullable=True)  # p10 quantile
    price_mid = Column(Float, nullable=True)  # p50 quantile (median)
    price_high = Column(Float, nullable=True)  # p90 quantile
    current_price = Column(Float, nullable=True)
    direction = Column(String(10), nullable=True)  # up, down, flat
    confidence = Column(String(10), nullable=True)  # low, medium, high
    model_version = Column(String(50), nullable=True)
    # Did the quote this forecast was built from equal its own local median?
    # The cohort split every 2026-08-11 served-signal figure was measured on:
    # rank IC +0.13/+0.16/+0.17 at 3/7/14d where True, ~0 or negative where
    # False. NULL means "not recorded" -- every row predating the column -- and
    # api/serving_policy.py treats it as passing rather than as deviating.
    # `anchor_wedge_pct` is (raw quote / smoothed anchor - 1) * 100, published
    # because the split at exact equality is what was measured and whether the
    # effect is a cliff there or monotone in |p/S - 1| is not.
    anchor_clean = Column(Boolean, nullable=True)
    anchor_wedge_pct = Column(Float, nullable=True)
    # P(the h-day move clears the round-trip cost), ONE-SIDED: the upside
    # exceedance-head probability (`target_exceed_{h}d = ret > thr`, not
    # `|move| > thr`). A disclosed magnitude signal, never a directional call
    # (invariant 4) — the market-orthogonal quantity, served as its own field
    # rather than folded into the band width. NULL means "not recorded": every
    # row predating this column, and every row from an artifact with no
    # exceedance head (EXCEEDANCE_HEAD off, or a degenerate <2-class horizon).
    exceed_p = Column(Float, nullable=True)
    anomaly_p = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)

    item = relationship("Item", back_populates="forecasts")

    __table_args__ = (
        Index("idx_forecast_item_date", "item_id", "forecast_date", "horizon_days"),
        UniqueConstraint("item_id", "forecast_date", "horizon_days", name="uq_item_forecast_date_horizon"),
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
        Index("idx_event_impact_event_item", "event_id", "item_id"),
        UniqueConstraint("event_id", "item_id", name="uq_event_impact_event_item"),
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
        Index("idx_event_pattern_type_item", "event_type", "item_id"),
        UniqueConstraint("event_type", "item_id", name="uq_event_pattern_type_item"),
    )


class PredictionAccuracy(Base):
    """Accuracy tracking for ML forecasts.

    Stores aggregated accuracy metrics computed by the backtesting system.
    prediction_type: forecast
    metrics JSON schema:
      - forecast: {mae, rmse, mape, wmape, mape_by_tier, directional_accuracy,
                   interval_coverage, interval_coverage_dollar_basis,
                   interval_n_served_basis, interval_n_fallback_basis,
                   baseline_directional_accuracy,
                   improvement_over_baseline_pp, skill_vs_baseline,
                   conf_gap_pp, conf_high_interval_cov, conf_calibration_error,
                   directional_accuracy_ci_lower, directional_accuracy_ci_upper,
                   mae_ci_lower, mae_ci_upper, sample_count, horizon_days}
    """

    __tablename__ = "prediction_accuracy"

    id = Column(Integer, primary_key=True)
    prediction_type = Column(String(50), nullable=False, index=True)
    evaluation_date = Column(Date, nullable=False, index=True)
    horizon_days = Column(Integer, nullable=True)
    # Price tier of the cohort (0 = <$1 ... 4 = >=$100), or NULL for the
    # all-tiers aggregate row.
    price_tier = Column(Integer, nullable=True)
    model_version = Column(String(50), nullable=True)
    evaluation_window_days = Column(Integer, nullable=True)
    sample_count = Column(Integer, nullable=False, default=0)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utcnow_naive)

    __table_args__ = (
        Index("idx_accuracy_type_date", "prediction_type", "evaluation_date"),
        UniqueConstraint(
            "prediction_type",
            "evaluation_date",
            "horizon_days",
            "model_version",
            "price_tier",
            name="uq_accuracy_type_date_horizon_model_tier",
        ),
    )


class ForecastOutcome(Base):
    """Per-forecast correctness record.

    Stores whether each individual forecast was correct/wrong when the
    actual price becomes known. Written by backtest_forecasts() during
    the daily accuracy evaluation.
    """

    __tablename__ = "forecast_outcomes"

    id = Column(Integer, primary_key=True)
    forecast_id = Column(Integer, ForeignKey("item_forecasts.id"), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False)
    horizon_days = Column(Integer, nullable=False)
    target_date = Column(Date, nullable=False)
    # Nullable: written straight through from item_forecasts.current_price
    # (itself nullable), which may be unset. Never synthesized — a null here
    # means the serving-time snapshot is genuinely missing, distinct from
    # base_price which is always archive-resolved when the row exists.
    current_price = Column(Float, nullable=True)
    predicted_price_low = Column(Float, nullable=True)
    predicted_price_mid = Column(Float, nullable=False)
    predicted_price_high = Column(Float, nullable=True)
    actual_price = Column(Float, nullable=False)
    # The forecast-time leg of actual_ret, resolved by the backtest with the
    # same estimator as actual_price. Distinct from current_price, which is
    # whatever the serving run happened to write and is no longer scored on.
    base_price = Column(Float, nullable=True)
    # Length of the frozen (bit-identical) price run the forecast-date anchor
    # sat on, over the UNSMOOTHED voted series; 0 means a fresh price level.
    # A frozen anchor under-reports, so the return measured from it is the
    # Getmansky-Lo-Makarov MA(k) artifact rather than a market move.
    #
    # A frozen observation, like base_price: resolved from the archive at the
    # same moment, never in _REFRESH_VERDICTS_SQL's SET clause, moved only by
    # --reresolve. NULL means "unknown", NOT zero — every row resolved before
    # 2026-08-08 carries NULL and backtest.scoring buckets those as `unknown`
    # rather than pooling them with the fresh ones.
    #
    # Do not compare this against the raw-series staleness rates: this is a
    # property of a SMOOTHED anchor. The >=$1 cohort reads 0-1.8% stale here
    # and 12-27% on the unsmoothed series the label path sees.
    base_stale_run_days = Column(Integer, nullable=True)
    direction_predicted = Column(String(10), nullable=True)
    # --- DERIVED VERDICTS, REFRESHED TO MATCH CURRENT SCORING ---------------
    # direction_actual, direction_correct, in_interval, abs_error and pct_error
    # are NOT observations. They are metrics derived from base_price /
    # actual_price plus the prediction legs, and the number the backtest reports
    # is re-derived from those actuals on every run by
    # scripts/backtest_accuracy.py::_records_from_frozen_outcomes, which
    # deliberately does not read these columns.
    #
    # The freeze applies to the actuals, not the metrics: a later scoring change
    # (a different FLAT_TOLERANCE, a different interval rule) must land on
    # historical rows with no archive access. So that these stored copies cannot
    # drift away from the reported metric, every scoring run also calls
    # backtest_accuracy._refresh_verdict_columns, which UPDATEs exactly these
    # five columns (plus evaluated_at) wherever they disagree with the current
    # derivation. base_price, actual_price and resolved_at are never in that
    # UPDATE's SET clause — only --reresolve moves them.
    #
    # Consequence for readers: these columns are safe to read as the verdict as
    # of the last backtest run, and no longer go permanently stale awaiting a
    # --reresolve. Two places read them — models/forecaster.py::
    # update_bias_corrections_from_outcomes, which feeds production predict()
    # thresholds and which run_backtest() invokes in the same process right
    # after the refresh (so it sees no staleness at all), and the manual
    # scripts/tiered_breakdown.py, which sees whatever the last backtest run
    # left behind.
    direction_actual = Column(String(10), nullable=True)
    direction_correct = Column(Integer, nullable=False, default=0)
    in_interval = Column(Integer, nullable=True)
    abs_error = Column(Float, nullable=False)
    pct_error = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    evaluated_at = Column(DateTime, nullable=False, default=utcnow_naive)
    # Set once, when the outcome is first resolved. base_price and
    # actual_price are frozen from that moment; only --reresolve moves them.
    resolved_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_outcome_forecast_id", "forecast_id"),
        Index("idx_outcome_item_eval", "item_id", "evaluated_at"),
        Index("idx_outcome_correct", "direction_correct", "evaluated_at"),
    )


class AccuracyAlert(Base):
    """Concept drift monitoring — tracks accuracy degradation over time.

    Alerts are generated when sliding-window directional accuracy drops
    below a threshold, indicating the model has drifted and needs retraining.
    """

    __tablename__ = "accuracy_alerts"

    id = Column(Integer, primary_key=True)
    prediction_type = Column(String(50), nullable=False, index=True)
    horizon_days = Column(Integer, nullable=True)
    sliding_window_days = Column(Integer, nullable=False, default=7)
    current_accuracy = Column(Float, nullable=False)
    threshold_accuracy = Column(Float, nullable=False, default=60.0)
    sample_count = Column(Integer, nullable=False, default=0)
    triggered_at = Column(DateTime, nullable=False, default=utcnow_naive)
    resolved_at = Column(DateTime, nullable=True)
    details = Column(JSON, nullable=True)

    __table_args__ = (Index("idx_alert_type_triggered", "prediction_type", "triggered_at"),)


class SupplySnapshot(Base):
    """Daily snapshot of supply-side data (listing counts).

    Captures sell_listings from Steam Market and quantity from Skinport
    once per day. Used by the forecaster to compute supply-depth features
    like `sell_listings_log`, `supply_change_7d`, and `supply_to_volume_ratio`.

    Only the most recent snapshot per item is kept (upserted by date).
    """

    __tablename__ = "supply_snapshots"

    item_id = Column(Integer, ForeignKey("items.id"), primary_key=True)
    snapshot_date = Column(Date, primary_key=True)
    sell_listings = Column(Integer, nullable=True)
    skinport_quantity = Column(Integer, nullable=True)
    source = Column(String(50), default="steam_burst")  # steam_burst, skinport
    created_at = Column(DateTime, default=utcnow_naive)

    __table_args__ = (Index("idx_supply_item_date", "item_id", "snapshot_date"),)


class SocialMention(Base):
    """Reddit social mention — tracks skin name mentions and FinBERT sentiment."""

    __tablename__ = "social_mentions"

    item_id = Column(Integer, ForeignKey("items.id"), primary_key=True)
    source = Column(String(50), primary_key=True, default="reddit")
    post_id = Column(String(50), primary_key=True)
    subreddit = Column(String(50), nullable=True)
    post_title = Column(String(500), nullable=True)
    post_score = Column(Integer, nullable=True)
    post_url = Column(String(500), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    mentioned_at = Column(DateTime, nullable=False)
    collected_at = Column(DateTime, default=utcnow_naive)

    __table_args__ = (
        Index("idx_social_item_source", "item_id", "source"),
        Index("idx_social_mentioned_at", "mentioned_at"),
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
        Index("idx_event_correlation_event_item", "event_id", "item_id"),
        UniqueConstraint("event_id", "item_id", name="uq_event_correlation_event_item"),
    )
