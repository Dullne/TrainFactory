"""
Drop training_task_stats read model.

Revision ID: 016_drop_training_task_stats
Revises: 015_add_eval_worker_groups
Create Date: 2026-02-05
"""

from alembic import op
import sqlalchemy as sa

revision = '016_drop_training_task_stats'
down_revision = '015_add_eval_worker_groups'
branch_labels = None
depends_on = None


def _table_exists(conn, name: str) -> bool:
    inspector = sa.inspect(conn)
    return name in inspector.get_table_names()


def upgrade() -> None:
    """Drop training_task_stats table if present."""
    conn = op.get_bind()
    if _table_exists(conn, 'training_task_stats'):
        op.drop_table('training_task_stats')


def downgrade() -> None:
    """Recreate training_task_stats table."""
    conn = op.get_bind()
    if not _table_exists(conn, 'training_task_stats'):
        op.create_table(
            'training_task_stats',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('user_id', sa.String(length=64), nullable=True),
            sa.Column('status', sa.String(length=32), nullable=False),
            sa.Column('count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.UniqueConstraint('user_id', 'status', name='uq_task_stats_user_status'),
        )
        op.create_index('idx_task_stats_user_status', 'training_task_stats', ['user_id', 'status'])
        op.create_index('ix_training_task_stats_user_id', 'training_task_stats', ['user_id'])
        op.create_index('ix_training_task_stats_status', 'training_task_stats', ['status'])
