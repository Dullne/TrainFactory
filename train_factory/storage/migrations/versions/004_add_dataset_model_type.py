"""Add usage and model_type fields to datasets table

Adds usage field (train/eval) and model_type field (embedding/rerank/llm)
to support dataset classification for training and evaluation.

Revision ID: 004_add_dataset_model_type
Revises: 003_add_evaluation_tasks
Create Date: 2026-01-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '004_add_dataset_model_type'
down_revision: Union[str, None] = '003_add_evaluation_tasks'
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


def index_exists(index_name: str, table_name: str) -> bool:
    """Check if an index exists on a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = inspector.get_indexes(table_name)
    return any(idx['name'] == index_name for idx in indexes)


def upgrade() -> None:
    """Upgrade database schema - add usage and model_type to datasets."""

    if table_exists('datasets'):
        # Add usage column (train/eval)
        if not column_exists('datasets', 'usage'):
            op.add_column('datasets', sa.Column(
                'usage',
                sa.String(20),
                nullable=False,
                server_default='train'
            ))
            # Create index for usage
            if not index_exists('idx_usage', 'datasets'):
                op.create_index('idx_usage', 'datasets', ['usage'])

        # Add model_type column (embedding/rerank/llm)
        if not column_exists('datasets', 'model_type'):
            op.add_column('datasets', sa.Column(
                'model_type',
                sa.String(32),
                nullable=True
            ))
            # Create index for model_type
            if not index_exists('idx_model_type', 'datasets'):
                op.create_index('idx_model_type', 'datasets', ['model_type'])


def downgrade() -> None:
    """Downgrade database schema - remove usage and model_type from datasets."""

    if table_exists('datasets'):
        # Drop model_type column and index
        if column_exists('datasets', 'model_type'):
            if index_exists('idx_model_type', 'datasets'):
                op.drop_index('idx_model_type', table_name='datasets')
            op.drop_column('datasets', 'model_type')

        # Drop usage column and index
        if column_exists('datasets', 'usage'):
            if index_exists('idx_usage', 'datasets'):
                op.drop_index('idx_usage', table_name='datasets')
            op.drop_column('datasets', 'usage')
