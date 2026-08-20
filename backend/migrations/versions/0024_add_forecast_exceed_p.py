"""Add exceed_p to item_forecasts.

`ItemForecast.exceed_p` is the served one-sided exceedance probability — P(the
h-day upside move clears the round-trip cost) — emitted by `predict()` once the
exceedance head is trained (EXCEEDANCE_HEAD=1). It is a disclosed magnitude
signal, never a directional call (invariant 4). Added to the ORM alongside this
migration; the daily writer gates the INSERT on the column's presence and reads
NULL as "not recorded", so an unmigrated prod DB degrades to writing without it
rather than failing the batch — this migration realigns the schema so the field
actually lands.

Nullable Float, mirroring the anchor disclosure (0022). NULL for every row
predating the column and every row from an artifact with no head.

Revision ID: 0024_add_forecast_exceed_p
Revises: 0023_add_item_is_trainable
Create Date: 2026-08-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0024_add_forecast_exceed_p"
down_revision = "0023_add_item_is_trainable"
branch_labels = None
depends_on = None


def _existing() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(
        "item_forecasts")}


def upgrade() -> None:
    if "exceed_p" not in _existing():
        op.add_column(
            "item_forecasts",
            sa.Column("exceed_p", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    if "exceed_p" in _existing():
        op.drop_column("item_forecasts", "exceed_p")
