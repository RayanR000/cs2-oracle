"""Add social_mentions table for Reddit sentiment tracking

Revision ID: 0018
Revises: 0017_add_item_rarity_columns
Create Date: 2026-07-19

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0018"
down_revision: Union[str, None] = "0017_add_item_rarity_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "social_mentions",
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("post_id", sa.String(100), nullable=False),
        sa.Column("subreddit", sa.String(50), nullable=True),
        sa.Column("post_title", sa.Text(), nullable=True),
        sa.Column("post_score", sa.Integer(), nullable=True),
        sa.Column("post_url", sa.String(500), nullable=True),
        sa.Column("sentiment_score", sa.Float(), nullable=True),
        sa.Column("mentioned_at", sa.DateTime(), nullable=False),
        sa.Column("collected_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"]),
        sa.PrimaryKeyConstraint("item_id", "source", "post_id"),
    )
    op.create_index(
        "idx_social_item_source", "social_mentions", ["item_id", "source"]
    )
    op.create_index(
        "idx_social_mentioned_at", "social_mentions", ["mentioned_at"]
    )


def downgrade() -> None:
    op.drop_index("idx_social_mentioned_at", table_name="social_mentions")
    op.drop_index("idx_social_item_source", table_name="social_mentions")
    op.drop_table("social_mentions")
