"""Add is_trainable to items.

`Item.is_trainable` was added to the ORM (database.py:55, commits 701a4df / fde8314)
and merged with no migration, so prod `items` never grew the column. It is a mapped
column, so every full-entity `db.query(Item)` — including the aggregator's first task —
emits `SELECT items.is_trainable` and raises `UndefinedColumn`, which fails the
aggregator and silently skips the price-forecast and backtest-accuracy runs chained off
it. That stalls the served/scored panel (and with it the served-outcome q_hat feedback,
which self-activates at 20 served dates). This migration realigns the schema.

Integer boolean, mirroring is_backfilled (nullable, ORM client default 0). is_trainable
is the narrower TRAIN universe (non-iflow pre-2026 history) inside the wider is_backfilled
SERVE universe; the training universe is now derived from the archive rather than this
column (forecaster.py:1849), so the stored value is not consumed by training and
backfilling existing rows from is_backfilled is a safe, review-recommended default.

Revision ID: 0023_add_item_is_trainable
Revises: 0022_add_forecast_anchor_disclosure
Create Date: 2026-08-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0023_add_item_is_trainable"
down_revision = "0022_add_forecast_anchor_disclosure"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("items")}
    if "is_trainable" not in cols:
        op.add_column(
            "items",
            sa.Column("is_trainable", sa.Integer(), nullable=True),
        )
        # Backfill existing rows from is_backfilled: trainable is a subset of backfilled,
        # and the value is not read by training, so this is a safe non-NULL seed.
        op.execute("UPDATE items SET is_trainable = COALESCE(is_backfilled, 0) WHERE is_trainable IS NULL")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("items")}
    if "is_trainable" in cols:
        op.drop_column("items", "is_trainable")
