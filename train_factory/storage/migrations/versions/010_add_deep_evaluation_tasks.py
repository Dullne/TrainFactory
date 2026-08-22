"""Add deep evaluation tasks table

Creates deep_evaluation_tasks table for dataset-level deep evaluations.

Revision ID: 010_add_deep_evaluation_tasks
Revises: 009_task_read_models
Create Date: 2026-02-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '010_add_deep_evaluation_tasks'
down_revision: Union[str, None] = '009_task_read_models'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = inspector.get_columns(table_name)
    return any(col['name'] == column_name for col in columns)


def upgrade() -> None:
    if not table_exists('deep_evaluation_tasks'):
        op.create_table(
            'deep_evaluation_tasks',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('task_id', sa.String(36), nullable=False, unique=True),
            sa.Column('task_name', sa.String(255), nullable=True),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('eval_type', sa.String(32), nullable=False, server_default='rerank'),
            sa.Column('dataset_id', sa.String(36), nullable=True),
            sa.Column('sample_size', sa.Integer(), nullable=True),
            sa.Column('field_mapping', sa.JSON(), nullable=True),
            sa.Column('metrics', sa.JSON(), nullable=True),
            sa.Column('model_config_data', sa.JSON(), nullable=True),
            sa.Column('concurrency', sa.Integer(), nullable=False, server_default='5'),
            sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
            sa.Column('progress', sa.Float(), nullable=False, server_default='0.0'),
            sa.Column('total_samples', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('processed_samples', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('results_summary', sa.JSON(), nullable=True),
            sa.Column('results_path', sa.String(1024), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('user_id', sa.String(64), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
        )

        op.create_index('idx_deep_eval_task_id', 'deep_evaluation_tasks', ['task_id'])
        op.create_index('idx_deep_eval_status', 'deep_evaluation_tasks', ['status'])
        op.create_index('idx_deep_eval_dataset', 'deep_evaluation_tasks', ['dataset_id'])
        op.create_index('idx_deep_eval_user_created', 'deep_evaluation_tasks', ['user_id', 'created_at'])
    else:
        if not column_exists('deep_evaluation_tasks', 'eval_type'):
            op.add_column(
                'deep_evaluation_tasks',
                sa.Column('eval_type', sa.String(32), nullable=False, server_default='rerank'),
            )
        if not column_exists('deep_evaluation_tasks', 'field_mapping'):
            op.add_column(
                'deep_evaluation_tasks',
                sa.Column('field_mapping', sa.JSON(), nullable=True),
            )
        if not column_exists('deep_evaluation_tasks', 'model_config_data'):
            op.add_column(
                'deep_evaluation_tasks',
                sa.Column('model_config_data', sa.JSON(), nullable=True),
            )


def downgrade() -> None:
    if table_exists('deep_evaluation_tasks'):
        op.drop_index('idx_deep_eval_user_created', table_name='deep_evaluation_tasks')
        op.drop_index('idx_deep_eval_dataset', table_name='deep_evaluation_tasks')
        op.drop_index('idx_deep_eval_status', table_name='deep_evaluation_tasks')
        op.drop_index('idx_deep_eval_task_id', table_name='deep_evaluation_tasks')
        op.drop_table('deep_evaluation_tasks')
