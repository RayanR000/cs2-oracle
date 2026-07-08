"""Prune trend_indicators rows older than 90 days to cap unbounded growth.

Revision ID: 0009_prune_old_trend_indicators
Revises: 0008_drop_redundant_chart_point_index_and_cleanup
Create Date: 2026-07-08
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from datetime import datetime, timedelta

revision = "0009_prune_old_trend_indicators"
down_revision = "0008_drop_redundant_chart_point_index_and_cleanup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    cutoff = datetime.utcnow() - timedelta(days=90)
    result = bind.execute(
        sa.text("DELETE FROM trend_indicators WHERE timestamp < :cutoff"),
        {"cutoff": cutoff},
    )
    print(f"Pruned {result.rowcount} old trend_indicator rows (>90d)")


def downgrade() -> None:
    print("WARNING: Downgrade cannot restore deleted trend_indicator rows.")
    pass
