"""Add forecast candidate shadow tables.

`forecast_candidates` stores exact shadow predictions (centre + ranking
challengers) with stable identities and config fingerprints. Never exposed
through public schemas. `forecast_candidate_outcomes` stores frozen
candidate outcomes, immutable after first resolution.

Revision ID: 0027_add_forecast_candidates
Revises: 0026_remove_duplicate_indexes
Create Date: 2026-09-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0027_add_forecast_candidates"
down_revision = "0026_remove_duplicate_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_candidates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("forecast_date", sa.Date(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("component", sa.String(20), nullable=False),
        sa.Column("candidate_name", sa.String(50), nullable=False),
        sa.Column("candidate_version", sa.String(50), nullable=False),
        sa.Column("centre_price", sa.Float(), nullable=True),
        sa.Column("predicted_price_low", sa.Float(), nullable=True),
        sa.Column("predicted_price_high", sa.Float(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("anchor_price", sa.Float(), nullable=False),
        sa.Column("feature_cutoff_at", sa.DateTime(), nullable=False),
        sa.Column("artifact_version", sa.String(50), nullable=True),
        sa.Column("config_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "item_id",
            "forecast_date",
            "horizon_days",
            "component",
            "candidate_name",
            "candidate_version",
            name="uq_forecast_candidate_identity",
        ),
        sa.CheckConstraint("horizon_days IN (3, 7, 14, 30)", name="ck_forecast_candidate_horizon"),
        sa.CheckConstraint("component IN ('centre', 'ranking')", name="ck_forecast_candidate_component"),
        sa.CheckConstraint(
            "(component = 'centre' AND centre_price IS NOT NULL AND predicted_price_low IS NOT NULL "
            "AND predicted_price_high IS NOT NULL AND score IS NULL) OR "
            "(component = 'ranking' AND score IS NOT NULL AND centre_price IS NULL "
            "AND predicted_price_low IS NULL AND predicted_price_high IS NULL)",
            name="ck_forecast_candidate_payload",
        ),
        sa.CheckConstraint(
            "component <> 'centre' OR (predicted_price_low > 0 AND "
            "predicted_price_low <= centre_price AND centre_price <= predicted_price_high)",
            name="ck_forecast_candidate_ordering",
        ),
    )
    op.create_index(
        "idx_candidate_date_horizon_component",
        "forecast_candidates",
        ["forecast_date", "horizon_days", "component"],
    )
    op.create_index("idx_candidate_item_date", "forecast_candidates", ["item_id", "forecast_date"])

    op.create_table(
        "forecast_candidate_outcomes",
        sa.Column("candidate_id", sa.Integer(), nullable=False),
        sa.Column("base_price", sa.Float(), nullable=False),
        sa.Column("actual_price", sa.Float(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=False),
        sa.Column("absolute_error", sa.Float(), nullable=True),
        sa.Column("percentage_error", sa.Float(), nullable=True),
        sa.Column("in_interval", sa.Boolean(), nullable=True),
        sa.Column("resolution_version", sa.String(50), nullable=False),
        sa.ForeignKeyConstraint(["candidate_id"], ["forecast_candidates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("candidate_id"),
    )


def downgrade() -> None:
    op.drop_table("forecast_candidate_outcomes")
    op.drop_index("idx_candidate_item_date", table_name="forecast_candidates")
    op.drop_index("idx_candidate_date_horizon_component", table_name="forecast_candidates")
    op.drop_table("forecast_candidates")
