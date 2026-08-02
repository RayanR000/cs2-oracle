"""Include price_tier in the prediction_accuracy unique constraint.

Task 7 of the deterministic-backtest plan starts writing one row per price
tier plus an all-tiers row for each (prediction_type, evaluation_date,
horizon_days, model_version) cohort. The constraint added in
0011_add_prediction_accuracy predates price_tier (added in
0019_freeze_forecast_outcome_actuals as a plain nullable column, not yet
wired into the constraint) and would otherwise reject every tier row after
the first as a duplicate of the same key.

Revision ID: 0020_tier_prediction_accuracy_unique
Revises: 0019_freeze_forecast_outcome_actuals
Create Date: 2026-08-01
"""

from __future__ import annotations

from alembic import op

revision = "0020_tier_prediction_accuracy_unique"
down_revision = "0019_freeze_forecast_outcome_actuals"
branch_labels = None
depends_on = None

OLD_NAME = "uq_accuracy_type_date_horizon_model"
NEW_NAME = "uq_accuracy_type_date_horizon_model_tier"
OLD_COLS = ["prediction_type", "evaluation_date", "horizon_days", "model_version"]
NEW_COLS = OLD_COLS + ["price_tier"]


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("prediction_accuracy", recreate="always") as batch:
            batch.drop_constraint(OLD_NAME, type_="unique")
            batch.create_unique_constraint(NEW_NAME, NEW_COLS)
    else:
        op.drop_constraint(OLD_NAME, "prediction_accuracy", type_="unique")
        op.create_unique_constraint(NEW_NAME, "prediction_accuracy", NEW_COLS)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("prediction_accuracy", recreate="always") as batch:
            batch.drop_constraint(NEW_NAME, type_="unique")
            batch.create_unique_constraint(OLD_NAME, OLD_COLS)
    else:
        op.drop_constraint(NEW_NAME, "prediction_accuracy", type_="unique")
        op.create_unique_constraint(OLD_NAME, "prediction_accuracy", OLD_COLS)
