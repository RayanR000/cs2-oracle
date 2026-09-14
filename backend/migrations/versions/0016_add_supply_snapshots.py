"""Add supply_snapshots table for daily listing-count snapshots.

Revision ID: 0016_add_supply_snapshots
Revises: 0015_drop_daily_analysis
Create Date: 2026-07-15
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0016_add_supply_snapshots"
down_revision = "0015_drop_daily_analysis"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "supply_snapshots",
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("sell_listings", sa.Integer(), nullable=True),
        sa.Column("skinport_quantity", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="steam_burst"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["items.id"],
        ),
        sa.PrimaryKeyConstraint("item_id", "snapshot_date"),
    )
    op.create_index("idx_supply_item_date", "supply_snapshots", ["item_id", "snapshot_date"])


def downgrade() -> None:
    op.drop_index("idx_supply_item_date", table_name="supply_snapshots")
    op.drop_table("supply_snapshots")
