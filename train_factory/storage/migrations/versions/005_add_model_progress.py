"""Add model_progress field to evaluation_tasks

Adds model_progress JSON field for per-model per-dataset progress tracking.

Revision ID: 005_add_model_progress
Revises: 004_add_dataset_model_type
Create Date: 2026-01-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '005_add_model_progress'
down_revision: Union[str, None] = '004_add_dataset_model_type'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """Add model_progress column to evaluation_tasks table."""

    # Add model_progress column if it doesn't exist
    if not column_exists('evaluation_tasks', 'model_progress'):
        op.add_column(
            'evaluation_tasks',
            sa.Column('model_progress', sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    """Remove model_progress column from evaluation_tasks table."""

    if column_exists('evaluation_tasks', 'model_progress'):
        op.drop_column('evaluation_tasks', 'model_progress')
