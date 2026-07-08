"""Drop trend_indicators table — redundant with daily_analysis.

Revision ID: 0010_drop_trend_indicators
Revises: 0009_prune_old_trend_indicators
Create Date: 2026-07-08
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0010_drop_trend_indicators"
down_revision = "0009_prune_old_trend_indicators"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TABLE IF EXISTS trend_indicators")
    else:
        inspector = sa.inspect(bind)
        if "trend_indicators" in inspector.get_table_names():
            op.drop_table("trend_indicators")


def downgrade() -> None:
    op.create_table(
        "trend_indicators",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("sma_7", sa.Float(), nullable=True),
        sa.Column("sma_30", sa.Float(), nullable=True),
        sa.Column("volatility", sa.Float(), nullable=True),
        sa.Column("trend_score", sa.Float(), nullable=True),
        sa.Column("trend_direction", sa.String(20), nullable=True),
        sa.Column("confidence", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_trend_item_timestamp", "trend_indicators", ["item_id", "timestamp"])
    op.create_unique_constraint(
        "uq_trend_indicator_item_timestamp",
        "trend_indicators",
        ["item_id", "timestamp"],
    )
