"""Convert datasets.model_type from VARCHAR to JSON for multi-select tags

The model_type field changes from a single string (e.g. "embedding") to a JSON
array (e.g. ["embedding", "rerank"]) to support tagging a dataset as applicable
to multiple model types.

Existing non-null values are migrated: "embedding" -> ["embedding"]

Revision ID: 007_dataset_model_type_to_json
Revises: 006_add_path_hashes
Create Date: 2026-01-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text


# revision identifiers, used by Alembic.
revision: str = '007_dataset_model_type_to_json'
down_revision: Union[str, None] = '006_add_path_hashes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def index_exists(index_name: str, table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = inspector.get_indexes(table_name)
    return any(idx['name'] == index_name for idx in indexes)


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not table_exists('datasets') or not column_exists('datasets', 'model_type'):
        return

    conn = op.get_bind()
    inspector = inspect(conn)

    # Check if column is already JSON type (idempotent)
    columns = {col['name']: col for col in inspector.get_columns('datasets')}
    col_type = str(columns.get('model_type', {}).get('type', ''))
    if 'JSON' in col_type.upper():
        # Already migrated (e.g. manually), skip
        return

    # Drop index on model_type if it exists (can't index JSON directly)
    if index_exists('idx_model_type', 'datasets'):
        op.drop_index('idx_model_type', table_name='datasets')

    # Step 1: Add a temporary JSON column
    op.add_column('datasets', sa.Column('model_type_new', sa.JSON(), nullable=True))

    # Step 2: Migrate existing data - convert string to JSON array
    conn.execute(text(
        "UPDATE datasets SET model_type_new = JSON_ARRAY(model_type) "
        "WHERE model_type IS NOT NULL AND model_type != ''"
    ))

    # Step 3: Drop old column and rename new one
    op.drop_column('datasets', 'model_type')
    op.alter_column(
        'datasets',
        'model_type_new',
        new_column_name='model_type',
        existing_type=sa.JSON(),
    )


def downgrade() -> None:
    if not table_exists('datasets') or not column_exists('datasets', 'model_type'):
        return

    conn = op.get_bind()

    # Step 1: Add a temporary VARCHAR column
    op.add_column('datasets', sa.Column('model_type_old', sa.String(32), nullable=True))

    # Step 2: Migrate data - take first element of JSON array
    conn.execute(text(
        "UPDATE datasets SET model_type_old = JSON_UNQUOTE(JSON_EXTRACT(model_type, '$[0]')) "
        "WHERE model_type IS NOT NULL"
    ))

    # Step 3: Drop JSON column and rename old one
    op.drop_column('datasets', 'model_type')
    op.alter_column(
        'datasets',
        'model_type_old',
        new_column_name='model_type',
        existing_type=sa.String(32),
    )

    # Re-create index
    if not index_exists('idx_model_type', 'datasets'):
        op.create_index('idx_model_type', 'datasets', ['model_type'])
