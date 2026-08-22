"""Add boundary_rollback_seconds to external_sync_tasks.

Revision ID: 030_add_boundary_rollback_seconds
Revises: 029_rename_sync_config_to_task
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "030_add_boundary_rollback_seconds"
down_revision = "029_rename_sync_config_to_task"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    cols = [c["name"] for c in inspect(conn).get_columns("external_sync_tasks")]
    if "boundary_rollback_seconds" not in cols:
        op.add_column(
            "external_sync_tasks",
            sa.Column("boundary_rollback_seconds", sa.Integer(), nullable=False, server_default="5"),
        )


def downgrade():
    conn = op.get_bind()
    cols = [c["name"] for c in inspect(conn).get_columns("external_sync_tasks")]
    if "boundary_rollback_seconds" in cols:
        op.drop_column("external_sync_tasks", "boundary_rollback_seconds")
