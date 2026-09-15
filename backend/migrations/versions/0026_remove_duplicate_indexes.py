"""Remove duplicate indexes on supply_snapshots and forecast_outcomes.

* ``idx_supply_item_date`` (item_id, snapshot_date) duplicates the composite
  primary key on ``supply_snapshots``, which is already indexed.
* ``idx_outcome_forecast_id`` (forecast_id) duplicates the auto-created index
  from ``ForecastOutcome.forecast_id``'s ``index=True``.

Revision ID: 0026_remove_duplicate_indexes
Revises: 0025_add_forecast_anomaly_p
Create Date: 2026-09-15
"""

from __future__ import annotations

from alembic import op

revision = "0026_remove_duplicate_indexes"
down_revision = "0025_add_forecast_anomaly_p"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_supply_item_date")
        op.execute("DROP INDEX IF EXISTS idx_outcome_forecast_id")
    else:
        op.drop_index("idx_supply_item_date", table_name="supply_snapshots")
        op.drop_index("idx_outcome_forecast_id", table_name="forecast_outcomes")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_supply_item_date "
            "ON supply_snapshots (item_id, snapshot_date)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcome_forecast_id "
            "ON forecast_outcomes (forecast_id)"
        )
    else:
        op.create_index(
            "idx_supply_item_date", "supply_snapshots", ["item_id", "snapshot_date"]
        )
        op.create_index(
            "idx_outcome_forecast_id", "forecast_outcomes", ["forecast_id"]
        )
