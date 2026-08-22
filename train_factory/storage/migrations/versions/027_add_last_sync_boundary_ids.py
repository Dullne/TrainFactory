"""
Add last_sync_boundary_ids to external_sync_configs.

This column was introduced after some environments had already applied
025_add_external_sync. Add it in a forward migration for compatibility.

Revision ID: 027_add_last_sync_boundary_ids
Revises: 026_drop_rag_eval_fields
Create Date: 2026-02-12
"""

from alembic import op
import sqlalchemy as sa

revision = "027_add_last_sync_boundary_ids"
down_revision = "026_drop_rag_eval_fields"
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "external_sync_configs", "last_sync_boundary_ids"):
        op.add_column(
            "external_sync_configs",
            sa.Column("last_sync_boundary_ids", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "external_sync_configs", "last_sync_boundary_ids"):
        op.drop_column("external_sync_configs", "last_sync_boundary_ids")
