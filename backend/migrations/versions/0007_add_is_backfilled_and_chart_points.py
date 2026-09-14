"""Add is_backfilled column to items and create chart_points table.

Revision ID: 0007_add_is_backfilled_and_chart_points
Revises: 0006_price_history_composite_pk
Create Date: 2026-07-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_add_is_backfilled_and_chart_points"
down_revision = "0006_price_history_composite_pk"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # Add is_backfilled column to items
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE items ADD COLUMN IF NOT EXISTS is_backfilled INTEGER DEFAULT 0")
    else:
        inspector = sa.inspect(bind)
        columns = {c["name"] for c in inspector.get_columns("items")}
        if "is_backfilled" not in columns:
            op.add_column("items", sa.Column("is_backfilled", sa.Integer(), default=0))

    # Create chart_points table
    if bind.dialect.name == "postgresql":
        op.execute("""
            CREATE TABLE IF NOT EXISTS chart_points (
                item_id INTEGER NOT NULL REFERENCES items(id),
                day DATE NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (item_id, day)
            )
        """)
        op.execute("CREATE INDEX IF NOT EXISTS idx_chart_point_item_day ON chart_points (item_id, day)")
    else:
        op.create_table(
            "chart_points",
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("items.id"), primary_key=True),
            sa.Column("day", sa.Date(), primary_key=True),
            sa.Column("close", sa.Float(), nullable=False),
        )
        op.create_index("idx_chart_point_item_day", "chart_points", ["item_id", "day"])


def downgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE items DROP COLUMN IF EXISTS is_backfilled")
        op.execute("DROP TABLE IF EXISTS chart_points")
    else:
        op.drop_table("chart_points")
        op.drop_column("items", "is_backfilled")
