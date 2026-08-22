"""Add training task events and stats read models

Revision ID: 009_task_read_models
Revises: 008_remove_legacy_fields
Create Date: 2026-01-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '009_task_read_models'
down_revision: Union[str, None] = '008_remove_legacy_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not table_exists('training_task_events'):
        op.create_table(
            'training_task_events',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('event_id', sa.String(length=36), nullable=False),
            sa.Column('task_id', sa.String(length=36), nullable=False),
            sa.Column('user_id', sa.String(length=64), nullable=True),
            sa.Column('event_type', sa.String(length=64), nullable=False),
            sa.Column('payload', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
        )
        op.create_index('idx_task_event_task_time', 'training_task_events', ['task_id', 'created_at'])
        op.create_index('ix_training_task_events_event_id', 'training_task_events', ['event_id'], unique=True)
        op.create_index('ix_training_task_events_task_id', 'training_task_events', ['task_id'])
        op.create_index('ix_training_task_events_user_id', 'training_task_events', ['user_id'])
        op.create_index('ix_training_task_events_event_type', 'training_task_events', ['event_type'])
        op.create_index('ix_training_task_events_created_at', 'training_task_events', ['created_at'])

    if not table_exists('training_task_stats'):
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


def downgrade() -> None:
    if table_exists('training_task_events'):
        op.drop_index('idx_task_event_task_time', table_name='training_task_events')
        op.drop_index('ix_training_task_events_event_id', table_name='training_task_events')
        op.drop_index('ix_training_task_events_task_id', table_name='training_task_events')
        op.drop_index('ix_training_task_events_user_id', table_name='training_task_events')
        op.drop_index('ix_training_task_events_event_type', table_name='training_task_events')
        op.drop_index('ix_training_task_events_created_at', table_name='training_task_events')
        op.drop_table('training_task_events')

    if table_exists('training_task_stats'):
        op.drop_index('idx_task_stats_user_status', table_name='training_task_stats')
        op.drop_index('ix_training_task_stats_user_id', table_name='training_task_stats')
        op.drop_index('ix_training_task_stats_status', table_name='training_task_stats')
        op.drop_table('training_task_stats')
