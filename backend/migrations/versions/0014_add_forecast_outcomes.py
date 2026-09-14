"""Add forecast_outcomes table for per-forecast correctness tracking.

Revision ID: 0014_add_forecast_outcomes
Revises: 0013_add_accuracy_alerts
Create Date: 2026-07-12
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0014_add_forecast_outcomes"
down_revision = "0013_add_accuracy_alerts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_outcomes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("forecast_id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("forecast_date", sa.Date(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("current_price", sa.Float(), nullable=False),
        sa.Column("predicted_price_low", sa.Float(), nullable=True),
        sa.Column("predicted_price_mid", sa.Float(), nullable=False),
        sa.Column("predicted_price_high", sa.Float(), nullable=True),
        sa.Column("actual_price", sa.Float(), nullable=False),
        sa.Column("direction_predicted", sa.String(10), nullable=True),
        sa.Column("direction_actual", sa.String(10), nullable=True),
        sa.Column("direction_correct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("in_interval", sa.Integer(), nullable=True),
        sa.Column("abs_error", sa.Float(), nullable=False),
        sa.Column("pct_error", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["forecast_id"],
            ["item_forecasts.id"],
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["items.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_outcome_forecast_id", "forecast_outcomes", ["forecast_id"])
    op.create_index("idx_outcome_item_eval", "forecast_outcomes", ["item_id", "evaluated_at"])
    op.create_index("idx_outcome_correct", "forecast_outcomes", ["direction_correct", "evaluated_at"])


def downgrade() -> None:
    op.drop_index("idx_outcome_correct", table_name="forecast_outcomes")
    op.drop_index("idx_outcome_item_eval", table_name="forecast_outcomes")
    op.drop_index("idx_outcome_forecast_id", table_name="forecast_outcomes")
    op.drop_table("forecast_outcomes")
