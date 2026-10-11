"""Replace the duplicate item_forecasts index; index forecast_outcomes on evaluated_at.

* ``idx_forecast_item_date`` (item_id, forecast_date, horizon_days) duplicates
  ``uq_item_forecast_date_horizon``, which Postgres already backs with an index.
* ``idx_forecast_horizon_date`` (horizon_days, forecast_date DESC) INCLUDE (item_id)
  serves the API's "latest forecast per item at horizon h" reads, which filter on
  horizon and a forecast_date floor. Nothing led on horizon_days before.
* ``idx_outcome_evaluated_at`` serves the accuracy routes' ``ORDER BY evaluated_at
  DESC LIMIT n`` and the backtest's recency windows; both existing outcome indexes
  lead on another column.

Performance review 2026-10-08, API section.

Revision ID: 0030_forecast_and_outcome_read_indexes
Revises: 0029_create_pg_trgm_extension
Create Date: 2026-10-10
"""

from __future__ import annotations

from alembic import op

revision = "0030_forecast_and_outcome_read_indexes"
down_revision = "0029_create_pg_trgm_extension"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_forecast_item_date")
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecast_horizon_date "
            "ON item_forecasts (horizon_days, forecast_date DESC) INCLUDE (item_id)"
        )
        op.execute("CREATE INDEX IF NOT EXISTS idx_outcome_evaluated_at ON forecast_outcomes (evaluated_at)")
    else:
        op.drop_index("idx_forecast_item_date", table_name="item_forecasts")
        op.create_index("idx_forecast_horizon_date", "item_forecasts", ["horizon_days", "forecast_date"])
        op.create_index("idx_outcome_evaluated_at", "forecast_outcomes", ["evaluated_at"])


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_outcome_evaluated_at")
        op.execute("DROP INDEX IF EXISTS idx_forecast_horizon_date")
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecast_item_date ON item_forecasts (item_id, forecast_date, horizon_days)"
        )
    else:
        op.drop_index("idx_outcome_evaluated_at", table_name="forecast_outcomes")
        op.drop_index("idx_forecast_horizon_date", table_name="item_forecasts")
        op.create_index(
            "idx_forecast_item_date",
            "item_forecasts",
            ["item_id", "forecast_date", "horizon_days"],
        )
