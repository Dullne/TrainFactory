"""Add missing fields to deployments and datasets tables

This migration adds fields that were added to the SQLModel entities
but were missing from the initial database schema (init.sql).

Revision ID: 001_add_missing_fields
Revises:
Create Date: 2026-01-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '001_add_missing_fields'
down_revision: Union[str, None] = None
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
    """Upgrade database schema - add missing fields."""

    # === Deployments table: add container deployment fields ===
    if table_exists('deployments'):
        if not column_exists('deployments', 'deploy_mode'):
            op.add_column('deployments', sa.Column(
                'deploy_mode',
                sa.String(32),
                nullable=True,
                server_default='shared'
            ))

        if not column_exists('deployments', 'container_name'):
            op.add_column('deployments', sa.Column(
                'container_name',
                sa.String(255),
                nullable=True
            ))

        if not column_exists('deployments', 'gpu_id'):
            op.add_column('deployments', sa.Column(
                'gpu_id',
                sa.Integer(),
                nullable=True
            ))

        if not column_exists('deployments', 'port'):
            op.add_column('deployments', sa.Column(
                'port',
                sa.Integer(),
                nullable=True
            ))

    # === Datasets table: add usage field ===
    if table_exists('datasets'):
        if not column_exists('datasets', 'usage'):
            op.add_column('datasets', sa.Column(
                'usage',
                sa.String(20),
                nullable=True,
                server_default='train'
            ))
            # Add index for usage field
            op.create_index('idx_datasets_usage', 'datasets', ['usage'])


def downgrade() -> None:
    """Downgrade database schema - remove added fields."""

    # === Datasets table: remove usage field ===
    if table_exists('datasets'):
        if column_exists('datasets', 'usage'):
            op.drop_index('idx_datasets_usage', table_name='datasets')
            op.drop_column('datasets', 'usage')

    # === Deployments table: remove container deployment fields ===
    if table_exists('deployments'):
        if column_exists('deployments', 'port'):
            op.drop_column('deployments', 'port')

        if column_exists('deployments', 'gpu_id'):
            op.drop_column('deployments', 'gpu_id')

        if column_exists('deployments', 'container_name'):
            op.drop_column('deployments', 'container_name')

        if column_exists('deployments', 'deploy_mode'):
            op.drop_column('deployments', 'deploy_mode')
