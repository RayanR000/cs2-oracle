"""Add base_price/resolved_at to forecast_outcomes, price_tier to prediction_accuracy.

Also relaxes forecast_outcomes.current_price to nullable: the backtest now
writes item_forecasts.current_price straight through (itself nullable) rather
than synthesizing a value when it's missing, so the column has to allow NULL
too. See Task 5 fix round 1 in the deterministic-backtest plan.

Revision ID: 0019_freeze_forecast_outcome_actuals
Revises: 0018_add_social_mentions
Create Date: 2026-08-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019_freeze_forecast_outcome_actuals"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    outcome_cols = {c["name"] for c in inspector.get_columns("forecast_outcomes")}
    if "base_price" not in outcome_cols:
        op.add_column("forecast_outcomes", sa.Column("base_price", sa.Float(), nullable=True))
    if "resolved_at" not in outcome_cols:
        op.add_column("forecast_outcomes", sa.Column("resolved_at", sa.DateTime(), nullable=True))

    accuracy_cols = {c["name"] for c in inspector.get_columns("prediction_accuracy")}
    if "price_tier" not in accuracy_cols:
        op.add_column("prediction_accuracy", sa.Column("price_tier", sa.Integer(), nullable=True))

    current_price_col = next(
        (c for c in inspector.get_columns("forecast_outcomes") if c["name"] == "current_price"),
        None,
    )
    if current_price_col is not None and not current_price_col["nullable"]:
        if bind.dialect.name == "sqlite":
            with op.batch_alter_table("forecast_outcomes", recreate="always") as batch:
                batch.alter_column(
                    "current_price",
                    existing_type=sa.Float(),
                    nullable=True,
                )
        else:
            op.alter_column(
                "forecast_outcomes",
                "current_price",
                existing_type=sa.Float(),
                nullable=True,
            )


def downgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("forecast_outcomes", recreate="always") as batch:
            batch.alter_column(
                "current_price",
                existing_type=sa.Float(),
                nullable=False,
            )
    else:
        op.alter_column(
            "forecast_outcomes",
            "current_price",
            existing_type=sa.Float(),
            nullable=False,
        )

    op.drop_column("prediction_accuracy", "price_tier")
    op.drop_column("forecast_outcomes", "resolved_at")
    op.drop_column("forecast_outcomes", "base_price")
