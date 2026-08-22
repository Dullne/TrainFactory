"""
Simplify generation modes and add pos_neg_method.

Replace training_gen with doc_to_training / qa_to_training,
add pos_neg_method column.

Revision ID: 020_simplify_generation_modes
Revises: 019_add_full_pipeline_fields
Create Date: 2026-02-06
"""

from alembic import op
import sqlalchemy as sa

revision = '020_simplify_generation_modes'
down_revision = '019_add_full_pipeline_fields'
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c['name'] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    """Add pos_neg_method column."""
    conn = op.get_bind()

    if not _column_exists(conn, 'generation_tasks', 'pos_neg_method'):
        op.add_column(
            'generation_tasks',
            sa.Column('pos_neg_method', sa.String(32), nullable=True, server_default='retrieval')
        )


def downgrade() -> None:
    """Remove pos_neg_method column."""
    conn = op.get_bind()

    if _column_exists(conn, 'generation_tasks', 'pos_neg_method'):
        op.drop_column('generation_tasks', 'pos_neg_method')
