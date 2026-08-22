"""Add LoRA hot-loading support

Extends deployments table with inference framework and LoRA configuration fields.
Creates new loaded_adapters table to track loaded adapters on deployments.

Revision ID: 002_add_lora_hotload_support
Revises: 001_add_missing_fields
Create Date: 2026-01-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '002_add_lora_hotload_support'
down_revision: Union[str, None] = '001_add_missing_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def table_exists(table_name: str) -> bool:
    """Check if a table exists."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade database schema - add LoRA hot-loading support."""

    # === Deployments table: add inference framework and LoRA fields ===
    if table_exists('deployments'):
        if not column_exists('deployments', 'inference_framework'):
            op.add_column('deployments', sa.Column(
                'inference_framework',
                sa.String(32),
                nullable=True,
                server_default='xinference'
            ))

        if not column_exists('deployments', 'enable_lora'):
            op.add_column('deployments', sa.Column(
                'enable_lora',
                sa.Boolean(),
                nullable=True,
                server_default='0'
            ))

        if not column_exists('deployments', 'max_loras'):
            op.add_column('deployments', sa.Column(
                'max_loras',
                sa.Integer(),
                nullable=True,
                server_default='4'
            ))

        if not column_exists('deployments', 'max_lora_rank'):
            op.add_column('deployments', sa.Column(
                'max_lora_rank',
                sa.Integer(),
                nullable=True,
                server_default='64'
            ))

    # === Create loaded_adapters table ===
    if not table_exists('loaded_adapters'):
        op.create_table(
            'loaded_adapters',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('adapter_id', sa.String(36), nullable=False, unique=True),
            sa.Column('deployment_id', sa.String(36), nullable=False),
            sa.Column('adapter_name', sa.String(255), nullable=False),
            sa.Column('adapter_path', sa.String(1024), nullable=False),
            sa.Column('source_task_id', sa.String(36), nullable=True),
            sa.Column('source_model_id', sa.String(36), nullable=True),
            sa.Column('status', sa.String(32), nullable=False, server_default='loading'),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('user_id', sa.String(64), nullable=True),
            sa.Column('loaded_at', sa.DateTime(), nullable=False),
            sa.Column('unloaded_at', sa.DateTime(), nullable=True),
        )
        # Create indexes
        op.create_index('idx_adapter_deployment', 'loaded_adapters', ['deployment_id'])
        op.create_index('idx_adapter_source_task', 'loaded_adapters', ['source_task_id'])
        op.create_index('idx_adapter_status', 'loaded_adapters', ['status'])
        op.create_index('idx_adapter_user', 'loaded_adapters', ['user_id'])


def downgrade() -> None:
    """Downgrade database schema - remove LoRA hot-loading support."""

    # === Drop loaded_adapters table ===
    if table_exists('loaded_adapters'):
        op.drop_index('idx_adapter_user', table_name='loaded_adapters')
        op.drop_index('idx_adapter_status', table_name='loaded_adapters')
        op.drop_index('idx_adapter_source_task', table_name='loaded_adapters')
        op.drop_index('idx_adapter_deployment', table_name='loaded_adapters')
        op.drop_table('loaded_adapters')

    # === Deployments table: remove LoRA fields ===
    if table_exists('deployments'):
        if column_exists('deployments', 'max_lora_rank'):
            op.drop_column('deployments', 'max_lora_rank')

        if column_exists('deployments', 'max_loras'):
            op.drop_column('deployments', 'max_loras')

        if column_exists('deployments', 'enable_lora'):
            op.drop_column('deployments', 'enable_lora')

        if column_exists('deployments', 'inference_framework'):
            op.drop_column('deployments', 'inference_framework')
