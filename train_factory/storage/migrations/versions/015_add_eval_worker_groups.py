"""
Add worker_groups to evaluation_tasks.

Revision ID: 015_add_eval_worker_groups
Revises: 014_merge_evaluation_tasks
Create Date: 2026-02-05
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = '015_add_eval_worker_groups'
down_revision = '014_merge_evaluation_tasks'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add worker_groups column for deep evaluation concurrency config."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_columns = [col['name'] for col in inspector.get_columns('evaluation_tasks')]
    if 'worker_groups' not in existing_columns:
        op.execute(text('ALTER TABLE evaluation_tasks ADD COLUMN worker_groups JSON'))


def downgrade() -> None:
    """Remove worker_groups column."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_columns = [col['name'] for col in inspector.get_columns('evaluation_tasks')]
    if 'worker_groups' in existing_columns:
        op.execute(text('ALTER TABLE evaluation_tasks DROP COLUMN worker_groups'))
