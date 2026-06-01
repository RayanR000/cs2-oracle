"""Initial schema baseline.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def _has_table(inspector, name: str) -> bool:
    return name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "items"):
        op.create_table(
            "items",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("item_id", sa.String(length=255), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("type", sa.String(length=50), nullable=False),
            sa.Column("release_date", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("item_id", name="uq_items_item_id"),
        )
        op.create_index("idx_item_name", "items", ["name"])
        op.create_index("idx_item_type", "items", ["type"])

    if not _has_table(inspector, "users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("steam_id", sa.String(length=50), nullable=False),
            sa.Column("username", sa.String(length=255), nullable=True),
            sa.Column("avatar_url", sa.String(length=500), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("last_login", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("steam_id", name="uq_users_steam_id"),
        )
        op.create_index("ix_users_steam_id", "users", ["steam_id"])

    if not _has_table(inspector, "events"):
        op.create_table(
            "events",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("type", sa.String(length=50), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("description", sa.String(length=500), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_events_timestamp", "events", ["timestamp"])
        op.create_index("idx_event_type_timestamp", "events", ["type", "timestamp"])

    if not _has_table(inspector, "collection_runs"):
        op.create_table(
            "collection_runs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("started_at", sa.DateTime(), nullable=False),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.Column("status", sa.String(length=50), nullable=False),
            sa.Column("total_items", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("successful", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("duration_seconds", sa.Float(), nullable=True),
            sa.Column("error_message", sa.String(length=1000), nullable=True),
            sa.Column("source_breakdown", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_collection_runs_finished_at", "collection_runs", ["finished_at"])
        op.create_index("idx_collection_runs_started_at", "collection_runs", ["started_at"])
        op.create_index("idx_collection_runs_status", "collection_runs", ["status"])

    if not _has_table(inspector, "price_history"):
        op.create_table(
            "price_history",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("price", sa.Float(), nullable=False),
            sa.Column("volume", sa.Integer(), nullable=True),
            sa.Column("median_price", sa.Float(), nullable=True),
            sa.Column("source", sa.String(length=255), nullable=False, server_default="steam"),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_price_history_timestamp", "price_history", ["timestamp"])
        op.create_index("idx_price_history_item_timestamp", "price_history", ["item_id", "timestamp"])
        op.create_index("idx_price_history_source", "price_history", ["source"])

    if not _has_table(inspector, "trend_indicators"):
        op.create_table(
            "trend_indicators",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("sma_7", sa.Float(), nullable=True),
            sa.Column("sma_30", sa.Float(), nullable=True),
            sa.Column("volatility", sa.Float(), nullable=True),
            sa.Column("trend_score", sa.Float(), nullable=True),
            sa.Column("trend_direction", sa.String(length=20), nullable=True),
            sa.Column("confidence", sa.String(length=20), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("item_id", "timestamp", name="uq_trend_indicator_item_timestamp"),
        )
        op.create_index("idx_trend_item_timestamp", "trend_indicators", ["item_id", "timestamp"])

    if not _has_table(inspector, "daily_analysis"):
        op.create_table(
            "daily_analysis",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), nullable=False),
            sa.Column("analysis_date", sa.Date(), nullable=False),
            sa.Column("current_price", sa.Float(), nullable=True),
            sa.Column("ma_7day", sa.Float(), nullable=True),
            sa.Column("ma_30day", sa.Float(), nullable=True),
            sa.Column("ma_90day", sa.Float(), nullable=True),
            sa.Column("momentum_7day", sa.Float(), nullable=True),
            sa.Column("momentum_30day", sa.Float(), nullable=True),
            sa.Column("volatility", sa.Float(), nullable=True),
            sa.Column("trend_direction", sa.String(length=20), nullable=True),
            sa.Column("momentum_score", sa.Float(), nullable=True),
            sa.Column("opportunity_score", sa.Float(), nullable=True),
            sa.Column("trading_volume_trend", sa.Float(), nullable=True),
            sa.Column("price_stability", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("item_id", "analysis_date", name="uq_daily_analysis_item_date"),
        )
        op.create_index("idx_daily_analysis_item_date", "daily_analysis", ["item_id", "analysis_date"])

    if not _has_table(inspector, "event_impacts"):
        op.create_table(
            "event_impacts",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("event_id", sa.Integer(), sa.ForeignKey("events.id"), nullable=False),
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), nullable=True),
            sa.Column("price_day_before", sa.Float(), nullable=True),
            sa.Column("price_day_1", sa.Float(), nullable=True),
            sa.Column("price_day_3", sa.Float(), nullable=True),
            sa.Column("price_day_7", sa.Float(), nullable=True),
            sa.Column("impact_pct_1day", sa.Float(), nullable=True),
            sa.Column("impact_pct_3day", sa.Float(), nullable=True),
            sa.Column("impact_pct_7day", sa.Float(), nullable=True),
            sa.Column("peak_impact_pct", sa.Float(), nullable=True),
            sa.Column("peak_impact_day", sa.Integer(), nullable=True),
            sa.Column("duration_days", sa.Integer(), nullable=True),
            sa.Column("z_score", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("event_id", "item_id", name="uq_event_impact_event_item"),
        )
        op.create_index("idx_event_impact_event_item", "event_impacts", ["event_id", "item_id"])

    if not _has_table(inspector, "event_patterns"):
        op.create_table(
            "event_patterns",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("event_type", sa.String(length=50), nullable=False),
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), nullable=True),
            sa.Column("sample_size", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("avg_impact_1day", sa.Float(), nullable=True),
            sa.Column("avg_impact_3day", sa.Float(), nullable=True),
            sa.Column("avg_impact_7day", sa.Float(), nullable=True),
            sa.Column("std_dev", sa.Float(), nullable=True),
            sa.Column("consistency_score", sa.Float(), nullable=True),
            sa.Column("holdout_accuracy", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("event_type", "item_id", name="uq_event_pattern_type_item"),
        )
        op.create_index("idx_event_pattern_type_item", "event_patterns", ["event_type", "item_id"])

    if not _has_table(inspector, "event_correlations"):
        op.create_table(
            "event_correlations",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("event_id", sa.Integer(), sa.ForeignKey("events.id"), nullable=False),
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), nullable=True),
            sa.Column("price_change_pct", sa.Float(), nullable=True),
            sa.Column("control_group_change_pct", sa.Float(), nullable=True),
            sa.Column("significance_test_zscore", sa.Float(), nullable=True),
            sa.Column("significance_passed", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("control_group_diff", sa.Float(), nullable=True),
            sa.Column("control_group_passed", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("pattern_consistency_score", sa.Float(), nullable=True),
            sa.Column("pattern_passed", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("confounding_events_count", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("confounding_passed", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("lag_analysis_peak_day", sa.Integer(), nullable=True),
            sa.Column("lag_passed", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("holdout_validation_accuracy", sa.Float(), nullable=True),
            sa.Column("validation_passed", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("confidence_score", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("event_id", "item_id", name="uq_event_correlation_event_item"),
        )
        op.create_index("idx_event_correlation_event_item", "event_correlations", ["event_id", "item_id"])


def downgrade() -> None:
    for table_name in [
        "event_correlations",
        "event_patterns",
        "event_impacts",
        "daily_analysis",
        "trend_indicators",
        "price_history",
        "collection_runs",
        "events",
        "users",
        "items",
    ]:
        op.drop_table(table_name)
