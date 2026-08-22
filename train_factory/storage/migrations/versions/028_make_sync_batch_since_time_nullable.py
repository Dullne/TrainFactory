"""
Make external_sync_batches.since_time nullable.

First sync run uses since=None to indicate full fetch from beginning.

Revision ID: 028_make_sync_batch_since_time_nullable
Revises: 028_add_external_api_configs
Create Date: 2026-02-12
"""

from alembic import op
import sqlalchemy as sa

revision = "028_make_sync_batch_since_time_nullable"
down_revision = "028_add_external_api_configs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "external_sync_batches",
        "since_time",
        existing_type=sa.DateTime(),
        nullable=True,
    )


def downgrade() -> None:
    # Backfill to satisfy NOT NULL constraint before tightening column.
    op.execute(
        sa.text(
            "UPDATE external_sync_batches "
            "SET since_time = COALESCE(since_time, fetched_at) "
            "WHERE since_time IS NULL"
        )
    )
    op.alter_column(
        "external_sync_batches",
        "since_time",
        existing_type=sa.DateTime(),
        nullable=False,
    )
