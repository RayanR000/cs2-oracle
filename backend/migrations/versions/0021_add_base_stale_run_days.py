"""Add base_stale_run_days to forecast_outcomes.

Step 6 of docs/research/2026-08-07-next-steps.md. The length of the frozen
price run the forecast-date anchor sat on, in days, resolved from the archive
at the same moment as base_price.

It has to be STORED rather than derived at scoring time: backtest/scoring.py is
a pure function of the frozen row and `--rescore` is deliberately archive-free,
so the only layer that can know an outcome's run length is resolve_outcomes,
which already holds the voted frame. It therefore joins base_price /
actual_price / resolved_at as a frozen observation that only --reresolve may
move, and is deliberately absent from _REFRESH_VERDICTS_SQL's SET clause.

Nullable, and left NULL on every existing row on purpose. Backfilling would
mean a full --reresolve, which unfreezes every base_price and actual_price in
the table — far too much for what is a reporting axis. NULL means "run length
unknown", never zero, and backtest/scoring.py buckets those rows as `unknown`
rather than pooling them with the fresh ones.

Revision ID: 0021_add_base_stale_run_days
Revises: 0020_tier_prediction_accuracy_unique
Create Date: 2026-08-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021_add_base_stale_run_days"
down_revision = "0020_tier_prediction_accuracy_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("forecast_outcomes")}
    if "base_stale_run_days" not in cols:
        op.add_column(
            "forecast_outcomes",
            sa.Column("base_stale_run_days", sa.Integer(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("forecast_outcomes")}
    if "base_stale_run_days" in cols:
        op.drop_column("forecast_outcomes", "base_stale_run_days")
