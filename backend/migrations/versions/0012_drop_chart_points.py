"""Drop chart_points table.

Data is safely archived in price-archive/prices-YYYY.parquet files
on the data-archive branch. chart_points was a Postgres cache of
daily OHLCV closes rebuilt from those parquet files.

Revision ID: 0012_drop_chart_points
Revises: 0011_add_prediction_accuracy
Create Date: 2026-07-11
"""

from __future__ import annotations

from alembic import op


revision = "0012_drop_chart_points"
down_revision = "0011_add_prediction_accuracy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chart_points")


def downgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS chart_points (
            item_id INTEGER NOT NULL REFERENCES items(id),
            day DATE NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (item_id, day)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chart_point_item_day "
        "ON chart_points (item_id, day)"
    )
