"""Add UNIQUE(item_id, timestamp, source) to price_history and create item_forecasts.

Revision ID: 0003_add_price_history_unique
Revises: 0002_expand_price_history_source
Create Date: 2026-07-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_add_price_history_unique"
down_revision = "0002_expand_price_history_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # 1. Add unique constraint to price_history
    #
    # Before adding the constraint, deduplicate rows that would violate it:
    # keep the row with the highest id (most recently inserted).
    op.execute("""
        DELETE FROM price_history ph
        WHERE ph.id NOT IN (
            SELECT MIN(id)
            FROM price_history
            GROUP BY item_id, timestamp, source
        )
    """)
    op.create_unique_constraint(
        "uq_price_history_item_timestamp_source",
        "price_history",
        ["item_id", "timestamp", "source"],
    )

    # 2. Create item_forecasts table if it doesn't exist
    if "item_forecasts" not in inspector.get_table_names():
        op.create_table(
            "item_forecasts",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), nullable=False),
            sa.Column("forecast_date", sa.Date(), nullable=False),
            sa.Column("horizon_days", sa.Integer(), nullable=False),
            sa.Column("price_low", sa.Float(), nullable=True),
            sa.Column("price_mid", sa.Float(), nullable=True),
            sa.Column("price_high", sa.Float(), nullable=True),
            sa.Column("current_price", sa.Float(), nullable=True),
            sa.Column("direction", sa.String(length=10), nullable=True),
            sa.Column("confidence", sa.String(length=10), nullable=True),
            sa.Column("model_version", sa.String(length=50), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint(
                "item_id", "forecast_date", "horizon_days",
                name="uq_item_forecast_date_horizon",
            ),
        )
        op.create_index(
            "idx_forecast_item_date",
            "item_forecasts",
            ["item_id", "forecast_date", "horizon_days"],
        )


def downgrade() -> None:
    op.drop_constraint(
        "uq_price_history_item_timestamp_source",
        "price_history",
        type_="unique",
    )
    op.drop_table("item_forecasts")
