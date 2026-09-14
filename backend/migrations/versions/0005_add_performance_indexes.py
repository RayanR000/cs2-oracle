"""Add performance indexes for pruning and query-heavy columns.

Revision ID: 0005_add_performance_indexes
Revises: 0004_add_item_metadata_images
Create Date: 2026-07-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_add_performance_indexes"
down_revision = "0004_add_item_metadata_images"
branch_labels = None
depends_on = None


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(bind)
    # The analysis tables were dropped from Supabase (Phase 0 cleanup) and are
    # recreated separately; skip index creation when the table is absent.
    if table_name not in inspector.get_table_names():
        return True
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()

    # daily_analysis.analysis_date — pruned weekly, currently sequential scan
    if not _index_exists(bind, "daily_analysis", "idx_daily_analysis_date"):
        op.create_index(
            "idx_daily_analysis_date",
            "daily_analysis",
            ["analysis_date"],
        )

    # event_impacts.created_at — pruned weekly
    if not _index_exists(bind, "event_impacts", "idx_event_impacts_created_at"):
        op.create_index(
            "idx_event_impacts_created_at",
            "event_impacts",
            ["created_at"],
        )

    # trend_indicators.timestamp — pruned weekly
    if not _index_exists(bind, "trend_indicators", "idx_trend_indicators_timestamp"):
        op.create_index(
            "idx_trend_indicators_timestamp",
            "trend_indicators",
            ["timestamp"],
        )

    # event_correlations.created_at — pruned weekly
    if not _index_exists(bind, "event_correlations", "idx_event_correlations_created_at"):
        op.create_index(
            "idx_event_correlations_created_at",
            "event_correlations",
            ["created_at"],
        )


def downgrade() -> None:
    op.drop_index("idx_event_correlations_created_at", table_name="event_correlations")
    op.drop_index("idx_trend_indicators_timestamp", table_name="trend_indicators")
    op.drop_index("idx_event_impacts_created_at", table_name="event_impacts")
    op.drop_index("idx_daily_analysis_date", table_name="daily_analysis")
