"""Add generation tasks table

Creates generation_tasks table for data generation tasks.

Revision ID: 012_add_generation_tasks
Revises: 011_add_model_config_container_fields
Create Date: 2026-02-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '012_add_generation_tasks'
down_revision: Union[str, None] = '011_add_model_config_container_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not table_exists('generation_tasks'):
        op.create_table(
            'generation_tasks',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('task_id', sa.String(36), nullable=False, unique=True),
            sa.Column('task_name', sa.String(255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            # Input configuration
            sa.Column('input_path', sa.String(1024), nullable=False),
            sa.Column('input_format', sa.String(32), nullable=False, server_default='auto'),
            sa.Column('generation_mode', sa.String(32), nullable=False, server_default='passage_based'),
            # Output configuration
            sa.Column('output_path', sa.String(1024), nullable=True),
            sa.Column('output_format', sa.String(32), nullable=False, server_default='universal'),
            # Model configurations (JSON)
            sa.Column('llm_config', sa.JSON(), nullable=False),
            sa.Column('embedding_config', sa.JSON(), nullable=True),
            sa.Column('rerank_config', sa.JSON(), nullable=True),
            # Worker and steps configuration
            sa.Column('worker_config', sa.JSON(), nullable=False),
            sa.Column('steps_config', sa.JSON(), nullable=False),
            sa.Column('post_process_config', sa.JSON(), nullable=True),
            sa.Column('custom_prompts', sa.JSON(), nullable=True),
            # Progress tracking
            sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
            sa.Column('progress', sa.Float(), nullable=False, server_default='0.0'),
            sa.Column('total_docs', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('processed_docs', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('output_sample_count', sa.Integer(), nullable=False, server_default='0'),
            # Output dataset
            sa.Column('output_dataset_id', sa.String(36), nullable=True),
            sa.Column('auto_register_dataset', sa.Boolean(), nullable=False, server_default='1'),
            # Error info
            sa.Column('error_message', sa.Text(), nullable=True),
            # User isolation
            sa.Column('user_id', sa.String(64), nullable=True),
            # Timestamps
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
        )

        op.create_index('idx_gen_task_id', 'generation_tasks', ['task_id'])
        op.create_index('idx_gen_task_name', 'generation_tasks', ['task_name'])
        op.create_index('idx_gen_task_status', 'generation_tasks', ['status'])
        op.create_index('idx_gen_task_output_dataset', 'generation_tasks', ['output_dataset_id'])
        op.create_index('idx_gen_task_user_id', 'generation_tasks', ['user_id'])
        op.create_index('idx_gen_task_user_created', 'generation_tasks', ['user_id', 'created_at'])


def downgrade() -> None:
    if table_exists('generation_tasks'):
        op.drop_index('idx_gen_task_user_created', table_name='generation_tasks')
        op.drop_index('idx_gen_task_user_id', table_name='generation_tasks')
        op.drop_index('idx_gen_task_output_dataset', table_name='generation_tasks')
        op.drop_index('idx_gen_task_status', table_name='generation_tasks')
        op.drop_index('idx_gen_task_name', table_name='generation_tasks')
        op.drop_index('idx_gen_task_id', table_name='generation_tasks')
        op.drop_table('generation_tasks')
