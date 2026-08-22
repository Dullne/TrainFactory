"""Remove legacy training fields and normalize dataset path

Revision ID: 008_remove_legacy_fields
Revises: 007_dataset_model_type_to_json
Create Date: 2026-01-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text


# revision identifiers, used by Alembic.
revision: str = '008_remove_legacy_fields'
down_revision: Union[str, None] = '007_dataset_model_type_to_json'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    if not table_exists('training_tasks'):
        return

    conn = op.get_bind()

    # Add train_dataset_path if missing
    if not column_exists('training_tasks', 'train_dataset_path'):
        op.add_column('training_tasks', sa.Column('train_dataset_path', sa.String(length=1024), nullable=True))

    # Backfill model_type from train_type if needed
    if column_exists('training_tasks', 'train_type') and column_exists('training_tasks', 'model_type'):
        conn.execute(text(
            "UPDATE training_tasks SET model_type = train_type "
            "WHERE (model_type IS NULL OR model_type = '') AND train_type IS NOT NULL"
        ))

    # Backfill train_dataset_path from dataset_name_or_path if needed
    if column_exists('training_tasks', 'dataset_name_or_path') and column_exists('training_tasks', 'train_dataset_path'):
        conn.execute(text(
            "UPDATE training_tasks SET train_dataset_path = dataset_name_or_path "
            "WHERE train_dataset_path IS NULL AND dataset_name_or_path IS NOT NULL"
        ))

    # Drop legacy columns
    if column_exists('training_tasks', 'train_type'):
        op.drop_column('training_tasks', 'train_type')
    if column_exists('training_tasks', 'dataset_name_or_path'):
        op.drop_column('training_tasks', 'dataset_name_or_path')


def downgrade() -> None:
    if not table_exists('training_tasks'):
        return

    conn = op.get_bind()

    # Re-add legacy columns
    if not column_exists('training_tasks', 'train_type'):
        op.add_column('training_tasks', sa.Column('train_type', sa.String(length=50), nullable=True))
    if not column_exists('training_tasks', 'dataset_name_or_path'):
        op.add_column('training_tasks', sa.Column('dataset_name_or_path', sa.String(length=1024), nullable=True))

    # Backfill legacy columns
    if column_exists('training_tasks', 'model_type') and column_exists('training_tasks', 'train_type'):
        conn.execute(text(
            "UPDATE training_tasks SET train_type = model_type "
            "WHERE train_type IS NULL AND model_type IS NOT NULL"
        ))
    if column_exists('training_tasks', 'train_dataset_path') and column_exists('training_tasks', 'dataset_name_or_path'):
        conn.execute(text(
            "UPDATE training_tasks SET dataset_name_or_path = train_dataset_path "
            "WHERE dataset_name_or_path IS NULL AND train_dataset_path IS NOT NULL"
        ))
