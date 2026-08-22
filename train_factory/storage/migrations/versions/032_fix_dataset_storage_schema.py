"""Fix dataset storage schema: nullable columns + drop record_mode.

- datasets.storage_path: NOT NULL → nullable (S3 datasets have no local path)
- datasets.storage_path_hash: NOT NULL → nullable (S3 datasets have no local path)
- datasets.record_mode: drop column (not in current scope)
- dataset_lineage_edges.from_dataset_id: NOT NULL → nullable (sync_fetched has no upstream)

Revision ID: 032_fix_dataset_storage_schema
Revises: 031_dataset_storage_refactor
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "032_fix_dataset_storage_schema"
down_revision = "031_dataset_storage_refactor"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    cols = [c["name"] for c in inspect(conn).get_columns(table)]
    return column in cols


def upgrade():
    # 1. datasets.storage_path -> nullable
    op.alter_column(
        "datasets", "storage_path",
        existing_type=sa.String(1024),
        nullable=True,
    )

    # 2. datasets.storage_path_hash -> nullable
    op.alter_column(
        "datasets", "storage_path_hash",
        existing_type=sa.String(64),
        nullable=True,
    )

    # 3. Drop record_mode column (Phase 5, not current scope)
    if _column_exists("datasets", "record_mode"):
        op.drop_column("datasets", "record_mode")

    # 4. dataset_lineage_edges.from_dataset_id -> nullable
    op.alter_column(
        "dataset_lineage_edges", "from_dataset_id",
        existing_type=sa.String(36),
        nullable=True,
    )


def downgrade():
    # Reverse: make from_dataset_id NOT NULL again
    op.alter_column(
        "dataset_lineage_edges", "from_dataset_id",
        existing_type=sa.String(36),
        nullable=False,
    )

    # Re-add record_mode
    if not _column_exists("datasets", "record_mode"):
        op.add_column(
            "datasets",
            sa.Column("record_mode", sa.String(16), nullable=False, server_default="disabled"),
        )

    # Reverse: storage_path_hash NOT NULL
    op.alter_column(
        "datasets", "storage_path_hash",
        existing_type=sa.String(64),
        nullable=False,
    )

    # Reverse: storage_path NOT NULL
    op.alter_column(
        "datasets", "storage_path",
        existing_type=sa.String(1024),
        nullable=False,
    )
