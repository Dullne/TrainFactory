"""
Drop legacy rag_eval_* columns from generation_tasks.

Keep only deep_eval_* fields as the canonical naming.

Revision ID: 026_drop_rag_eval_fields
Revises: 025_add_external_sync
Create Date: 2026-02-12
"""

from alembic import op
import sqlalchemy as sa

revision = "026_drop_rag_eval_fields"
down_revision = "025_add_external_sync"
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    conn = op.get_bind()

    if _column_exists(conn, "generation_tasks", "rag_eval_dataset_id"):
        op.drop_column("generation_tasks", "rag_eval_dataset_id")

    if _column_exists(conn, "generation_tasks", "rag_eval_path"):
        op.drop_column("generation_tasks", "rag_eval_path")


def downgrade() -> None:
    conn = op.get_bind()

    if not _column_exists(conn, "generation_tasks", "rag_eval_path"):
        op.add_column(
            "generation_tasks",
            sa.Column("rag_eval_path", sa.String(1024), nullable=True),
        )

    if not _column_exists(conn, "generation_tasks", "rag_eval_dataset_id"):
        op.add_column(
            "generation_tasks",
            sa.Column("rag_eval_dataset_id", sa.String(36), nullable=True),
        )
