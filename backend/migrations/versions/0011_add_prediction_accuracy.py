"""Add prediction_accuracy table for tracking analysis quality over time.

Revision ID: 0011_add_prediction_accuracy
Revises: 0010_drop_trend_indicators
Create Date: 2026-07-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0011_add_prediction_accuracy"
down_revision = "0010_drop_trend_indicators"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_accuracy",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prediction_type", sa.String(50), nullable=False, index=True),
        sa.Column("evaluation_date", sa.Date(), nullable=False, index=True),
        sa.Column("horizon_days", sa.Integer(), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("evaluation_window_days", sa.Integer(), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=False, default=0),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "idx_accuracy_type_date",
        "prediction_accuracy",
        ["prediction_type", "evaluation_date"],
    )
    op.create_unique_constraint(
        "uq_accuracy_type_date_horizon_model",
        "prediction_accuracy",
        ["prediction_type", "evaluation_date", "horizon_days", "model_version"],
    )


def downgrade() -> None:
    op.drop_table("prediction_accuracy")
