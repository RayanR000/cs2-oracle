"""Drop redundant chart_points index and clean up price_history for backfilled items.

Revision ID: 0008_drop_redundant_chart_point_index_and_cleanup
Revises: 0007_add_is_backfilled_and_chart_points
Create Date: 2026-07-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_drop_redundant_chart_point_index_and_cleanup"
down_revision = "0007_add_is_backfilled_and_chart_points"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_chart_point_item_day")
    else:
        op.drop_index("idx_chart_point_item_day", table_name="chart_points")

    op.execute(
        "DELETE FROM price_history "
        "WHERE item_id IN (SELECT id FROM items WHERE is_backfilled = 1)"
    )


def downgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_chart_point_item_day "
            "ON chart_points (item_id, day)"
        )
    else:
        op.create_index("idx_chart_point_item_day", "chart_points", ["item_id", "day"])

    print(
        "WARNING: Downgrade cannot restore deleted price_history rows. "
        "Run the aggregator pipeline to repopulate."
    )
