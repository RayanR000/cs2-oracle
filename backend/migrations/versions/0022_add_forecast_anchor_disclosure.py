"""Add anchor_clean / anchor_wedge_pct to item_forecasts.

The cohort split the 2026-08-11 clean-anchor result was measured on. Served rank
IC is +0.1321 / +0.1562 / +0.1747 at 3/7/14d on items whose anchor quote equals
its own local median (4 CI anchors of 4) and -0.2014 at h=3 (0 of 4) on the
rest. `api/serving_policy.py::meets_anchor_gate` uses the flag to keep the
deviating cohort off the ranked surfaces; per-item lookups still serve it.

Both columns are nullable and every existing row is left NULL, deliberately.
They are properties of the price frame at *serving* time -- the raw quote and
its local median on the forecast date -- and nothing stored on the row can
reconstruct them: `current_price` is the SERVED base, which under the shipped
arm is already the smoothed median, so a backfill from the table would find
every historical row "clean" and be exactly wrong. NULL means "not recorded",
and the gate treats it as passing so that adding the column cannot empty a
ranked surface. It self-heals after one forecast run.

`anchor_wedge_pct` is (raw quote / smoothed anchor - 1) * 100, signed. It is
carried because the split at exact equality is what was measured and whether
the effect is a cliff there or monotone in |p/S - 1| is not -- storing the size
is what lets that be read later without another serving change.

Revision ID: 0022_add_forecast_anchor_disclosure
Revises: 0021_add_base_stale_run_days
Create Date: 2026-08-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0022_add_forecast_anchor_disclosure"
down_revision = "0021_add_base_stale_run_days"
branch_labels = None
depends_on = None

# Built fresh per call rather than held as module-level Column objects: a
# Column instance binds to the first table it is added to, and `Column.copy()`
# is deprecated in SQLAlchemy 2.0.
_COLUMNS = (
    ("anchor_clean", lambda: sa.Column("anchor_clean", sa.Boolean(), nullable=True)),
    ("anchor_wedge_pct", lambda: sa.Column("anchor_wedge_pct", sa.Float(), nullable=True)),
)


def _existing() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns("item_forecasts")}


def upgrade() -> None:
    cols = _existing()
    for name, make in _COLUMNS:
        if name not in cols:
            op.add_column("item_forecasts", make())


def downgrade() -> None:
    cols = _existing()
    for name, _ in _COLUMNS:
        if name in cols:
            op.drop_column("item_forecasts", name)
