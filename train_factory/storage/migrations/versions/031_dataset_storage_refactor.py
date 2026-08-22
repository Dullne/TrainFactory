"""Add dataset storage refactor: new fields on datasets + dataset_assets + dataset_lineage_edges.

Revision ID: 031_dataset_storage_refactor
Revises: 030_add_boundary_rollback_seconds
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "031_dataset_storage_refactor"
down_revision = "030_add_boundary_rollback_seconds"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    cols = [c["name"] for c in inspect(conn).get_columns(table)]
    return column in cols


def _table_exists(name: str) -> bool:
    conn = op.get_bind()
    return name in inspect(conn).get_table_names()


def upgrade():
    # ------------------------------------------------------------------
    # 1. datasets — add new columns (idempotent)
    # ------------------------------------------------------------------
    new_columns = [
        ("storage_backend", sa.String(16), False, "local"),
        ("storage_uri", sa.String(2048), True, None),
        ("version", sa.Integer(), False, "1"),
        ("content_schema", sa.JSON(), True, None),
        ("source_task_type", sa.String(32), True, None),
        ("source_task_id", sa.String(36), True, None),
    ]
    for col_name, col_type, nullable, default in new_columns:
        if not _column_exists("datasets", col_name):
            col_kwargs = {"nullable": nullable}
            if default is not None:
                col_kwargs["server_default"] = default
            op.add_column(
                "datasets",
                sa.Column(col_name, col_type, **col_kwargs),
            )

    # ------------------------------------------------------------------
    # 2. dataset_assets — new table
    # ------------------------------------------------------------------
    if not _table_exists("dataset_assets"):
        op.create_table(
            "dataset_assets",
            sa.Column("id", sa.BigInteger().with_variant(sa.BigInteger(), "mysql"), autoincrement=True, primary_key=True),
            sa.Column("asset_id", sa.String(36), nullable=False),
            sa.Column("dataset_id", sa.String(36), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("asset_type", sa.String(32), nullable=False, server_default="data"),
            sa.Column("storage_uri", sa.String(2048), nullable=False),
            sa.Column("file_format", sa.String(32), nullable=True),
            sa.Column("compression", sa.String(16), nullable=True),
            sa.Column("row_count", sa.Integer(), nullable=True),
            sa.Column("byte_size", sa.BigInteger(), nullable=True),
            sa.Column("checksum", sa.String(128), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("asset_id", name="uq_asset_id"),
        )
        op.create_index("idx_asset_dataset", "dataset_assets", ["dataset_id", "version"])
        op.create_index("idx_asset_type", "dataset_assets", ["dataset_id", "asset_type"])

    # ------------------------------------------------------------------
    # 3. dataset_lineage_edges — new table
    # ------------------------------------------------------------------
    if not _table_exists("dataset_lineage_edges"):
        op.create_table(
            "dataset_lineage_edges",
            sa.Column("id", sa.BigInteger().with_variant(sa.BigInteger(), "mysql"), autoincrement=True, primary_key=True),
            sa.Column("edge_id", sa.String(36), nullable=False),
            sa.Column("from_dataset_id", sa.String(36), nullable=True),
            sa.Column("to_dataset_id", sa.String(36), nullable=False),
            sa.Column("relation_type", sa.String(32), nullable=False),
            sa.Column("op_task_type", sa.String(32), nullable=True),
            sa.Column("op_task_id", sa.String(36), nullable=True),
            sa.Column("op_params", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("edge_id", name="uq_edge_id"),
        )
        op.create_index("idx_lineage_from", "dataset_lineage_edges", ["from_dataset_id"])
        op.create_index("idx_lineage_to", "dataset_lineage_edges", ["to_dataset_id"])
        op.create_index("idx_lineage_task", "dataset_lineage_edges", ["op_task_id"])


def downgrade():
    # Drop new tables
    if _table_exists("dataset_lineage_edges"):
        op.drop_table("dataset_lineage_edges")
    if _table_exists("dataset_assets"):
        op.drop_table("dataset_assets")

    # Remove new columns from datasets
    for col_name in ("source_task_id", "source_task_type",
                     "content_schema", "version", "storage_uri", "storage_backend"):
        if _column_exists("datasets", col_name):
            op.drop_column("datasets", col_name)
