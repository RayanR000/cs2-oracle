"""Add band_multiplier to item_forecasts.

`ItemForecast.band_multiplier` is the served-coverage q_hat multiplier a band was
served at (after the prior-day blend). The feedback refit scales each served row's
nonconformity score by it so rows served under different multipliers pool on the
base band. Without it the refit re-widens the band it just narrowed.

Nullable Float. NULL for rows predating the column; the loader reads NULL as 1.0
before served_recalibration.FEEDBACK_FIRST_SERVED_DATE and as unknown (row dropped)
on or after it. Rows served 2026-09-17..27 are filled by
scripts/backfill_band_multiplier.py, not here: that is prod history, not schema.

Revision ID: 0028_add_forecast_band_multiplier
Revises: 0027_add_forecast_candidates
Create Date: 2026-09-23
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0028_add_forecast_band_multiplier"
down_revision = "0027_add_forecast_candidates"
branch_labels = None
depends_on = None


def _existing() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns("item_forecasts")}


def upgrade() -> None:
    if "band_multiplier" not in _existing():
        op.add_column(
            "item_forecasts",
            sa.Column("band_multiplier", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    if "band_multiplier" in _existing():
        op.drop_column("item_forecasts", "band_multiplier")
