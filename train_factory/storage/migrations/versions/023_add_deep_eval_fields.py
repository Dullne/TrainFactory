"""
Add deep evaluation dataset fields to generation_tasks.

Stores the path and registered dataset ID for deep evaluation data
generated during the pos/neg retrieval phase.

Revision ID: 023_add_deep_eval_fields
Revises: 021_add_source_dataset_id
Create Date: 2026-02-10
"""

from alembic import op
import sqlalchemy as sa

revision = '023_add_deep_eval_fields'
down_revision = '021_add_source_dataset_id'
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c['name'] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    """Add deep_eval_path and deep_eval_dataset_id columns."""
    conn = op.get_bind()

    if not _column_exists(conn, 'generation_tasks', 'deep_eval_path'):
        op.add_column(
            'generation_tasks',
            sa.Column('deep_eval_path', sa.String(1024), nullable=True)
        )

    if not _column_exists(conn, 'generation_tasks', 'deep_eval_dataset_id'):
        op.add_column(
            'generation_tasks',
            sa.Column('deep_eval_dataset_id', sa.String(36), nullable=True)
        )


def downgrade() -> None:
    """Remove deep_eval_path and deep_eval_dataset_id columns."""
    conn = op.get_bind()

    if _column_exists(conn, 'generation_tasks', 'deep_eval_dataset_id'):
        op.drop_column('generation_tasks', 'deep_eval_dataset_id')

    if _column_exists(conn, 'generation_tasks', 'deep_eval_path'):
        op.drop_column('generation_tasks', 'deep_eval_path')
