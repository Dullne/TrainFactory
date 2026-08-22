"""Add disabled flag to external_sync_generations.

Allows users to disable old generation results so they are excluded
from training aggregation (e.g. after re-generating with updated
models/parameters/strategies).

Revision ID: 033_add_generation_excluded
Revises: 032_fix_dataset_storage_schema
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "033_add_generation_excluded"
down_revision = "032_fix_dataset_storage_schema"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    cols = [c["name"] for c in inspect(conn).get_columns(table)]
    return column in cols


def upgrade():
    # Rename chain: originally "excluded" → "outdated" → "disabled"
    if _column_exists("external_sync_generations", "excluded"):
        op.alter_column(
            "external_sync_generations", "excluded",
            new_column_name="disabled",
            existing_type=sa.Boolean(),
            existing_server_default="0",
            existing_nullable=False,
        )
    elif _column_exists("external_sync_generations", "outdated"):
        op.alter_column(
            "external_sync_generations", "outdated",
            new_column_name="disabled",
            existing_type=sa.Boolean(),
            existing_server_default="0",
            existing_nullable=False,
        )
    elif not _column_exists("external_sync_generations", "disabled"):
        op.add_column(
            "external_sync_generations",
            sa.Column("disabled", sa.Boolean(), server_default="0", nullable=False),
        )


def downgrade():
    if _column_exists("external_sync_generations", "disabled"):
        op.drop_column("external_sync_generations", "disabled")
