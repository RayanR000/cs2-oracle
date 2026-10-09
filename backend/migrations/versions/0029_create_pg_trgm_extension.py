"""Create the pg_trgm extension.

`api/routes/market.py` calls `similarity()` for item search (`/market?q=`), which
pg_trgm provides. No migration created it: prod has pg_trgm 1.6 in `public` because it
was installed by hand (verified 2026-10-09), so a database rebuilt from the migrations
would error on every search with a query. `IF NOT EXISTS` makes this a no-op on prod.

Downgrade does not drop it: it predates this revision on prod.

Revision ID: 0029_create_pg_trgm_extension
Revises: 0028_add_forecast_band_multiplier
Create Date: 2026-10-09
"""

from __future__ import annotations

from alembic import op

revision = "0029_create_pg_trgm_extension"
down_revision = "0028_add_forecast_band_multiplier"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


def downgrade() -> None:
    pass
