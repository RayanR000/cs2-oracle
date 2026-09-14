"""Add anomaly_p to item_forecasts.

`ItemForecast.anomaly_p` is the served anomaly probability — P(|return_h| >
2σ_item) — emitted by `predict()` when the anomaly GBM head is trained
(ANOMALY_GBM=1). An alert/flag signal for regime detection, not a band input.

Nullable Float. NULL for rows predating the column and rows from artifacts
with no anomaly head.

Revision ID: 0025_add_forecast_anomaly_p
Revises: 0024_add_forecast_exceed_p
Create Date: 2026-09-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0025_add_forecast_anomaly_p"
down_revision = "0024_add_forecast_exceed_p"
branch_labels = None
depends_on = None


def _existing() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns("item_forecasts")}


def upgrade() -> None:
    if "anomaly_p" not in _existing():
        op.add_column(
            "item_forecasts",
            sa.Column("anomaly_p", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    if "anomaly_p" in _existing():
        op.drop_column("item_forecasts", "anomaly_p")
