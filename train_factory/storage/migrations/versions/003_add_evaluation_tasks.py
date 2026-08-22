"""Add evaluation tasks table

Creates evaluation_tasks table for independent benchmark evaluations.

Revision ID: 003_add_evaluation_tasks
Revises: 002_add_lora_hotload_support
Create Date: 2026-01-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '003_add_evaluation_tasks'
down_revision: Union[str, None] = '002_add_lora_hotload_support'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    """Check if a table exists."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade database schema - add evaluation tasks table."""

    if not table_exists('evaluation_tasks'):
        op.create_table(
            'evaluation_tasks',
            # Primary key
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('task_id', sa.String(36), nullable=False, unique=True),

            # Basic info
            sa.Column('task_name', sa.String(255), nullable=True),
            sa.Column('description', sa.Text(), nullable=True),

            # Evaluation type
            sa.Column('eval_type', sa.String(32), nullable=False, server_default='single'),

            # Configuration (JSON)
            sa.Column('model_configs', sa.JSON(), nullable=True),
            sa.Column('dataset_configs', sa.JSON(), nullable=True),

            # Evaluation parameters
            sa.Column('max_samples', sa.Integer(), nullable=True),
            sa.Column('batch_size', sa.Integer(), nullable=False, server_default='50'),
            sa.Column('workers', sa.Integer(), nullable=False, server_default='8'),
            sa.Column('model_workers', sa.Integer(), nullable=False, server_default='2'),

            # Status
            sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
            sa.Column('progress', sa.Float(), nullable=False, server_default='0.0'),
            sa.Column('current_model', sa.String(255), nullable=True),
            sa.Column('current_dataset', sa.String(255), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),

            # Per-model progress (JSON)
            sa.Column('model_progress', sa.JSON(), nullable=True),

            # Results (JSON)
            sa.Column('results', sa.JSON(), nullable=True),
            sa.Column('report_path', sa.String(1024), nullable=True),

            # User isolation
            sa.Column('user_id', sa.String(64), nullable=True),

            # Timestamps
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
        )

        # Create indexes
        op.create_index('idx_eval_task_id', 'evaluation_tasks', ['task_id'])
        op.create_index('idx_eval_status', 'evaluation_tasks', ['status'])
        op.create_index('idx_eval_user_created', 'evaluation_tasks', ['user_id', 'created_at'])


def downgrade() -> None:
    """Downgrade database schema - remove evaluation tasks table."""

    if table_exists('evaluation_tasks'):
        op.drop_index('idx_eval_user_created', table_name='evaluation_tasks')
        op.drop_index('idx_eval_status', table_name='evaluation_tasks')
        op.drop_index('idx_eval_task_id', table_name='evaluation_tasks')
        op.drop_table('evaluation_tasks')
