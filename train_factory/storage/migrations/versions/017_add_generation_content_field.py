"""
Add content_field to generation_tasks.

Revision ID: 017_add_generation_content_field
Revises: 016_drop_training_task_stats
Create Date: 2026-02-05
"""

from alembic import op
import sqlalchemy as sa

revision = '017_add_generation_content_field'
down_revision = '016_drop_training_task_stats'
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c['name'] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    """Add content_field column to generation_tasks."""
    conn = op.get_bind()
    if not _column_exists(conn, 'generation_tasks', 'content_field'):
        op.add_column(
            'generation_tasks',
            sa.Column('content_field', sa.String(64), nullable=True)
        )


def downgrade() -> None:
    """Remove content_field column from generation_tasks."""
    conn = op.get_bind()
    if _column_exists(conn, 'generation_tasks', 'content_field'):
        op.drop_column('generation_tasks', 'content_field')
