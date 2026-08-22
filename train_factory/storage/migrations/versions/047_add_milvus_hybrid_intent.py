"""Persist immutable hybrid schema intent for manual Milvus creation.

Revision ID: 047_milvus_hybrid_intent
Revises: 046_milvus_sync_claims
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "047_milvus_hybrid_intent"
down_revision = "046_milvus_sync_claims"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if "milvus_collections" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("milvus_collections")
    }
    if "hybrid_enabled" not in columns:
        op.add_column(
            "milvus_collections",
            sa.Column("hybrid_enabled", sa.Boolean(), nullable=True),
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if "milvus_collections" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("milvus_collections")
    }
    if "hybrid_enabled" in columns:
        op.drop_column("milvus_collections", "hybrid_enabled")
