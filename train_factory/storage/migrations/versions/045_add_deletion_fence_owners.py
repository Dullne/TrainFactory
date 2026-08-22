"""Add durable owner tokens for dataset and Milvus deletion fences.

Revision ID: 045_deletion_fence_owners
Revises: 044_sync_training_claims
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "045_deletion_fence_owners"
down_revision = "044_sync_training_claims"
branch_labels = None
depends_on = None


def _add_owner_column(table_name: str) -> bool:
    inspector = inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return False
    columns = {
        column["name"] for column in inspector.get_columns(table_name)
    }
    if "deletion_owner" not in columns:
        op.add_column(
            table_name,
            sa.Column(
                "deletion_owner",
                sa.String(length=128),
                nullable=True,
            ),
        )
    return True


def _backfill_dataset_owners() -> None:
    bind = op.get_bind()
    datasets = sa.table(
        "datasets",
        sa.column("id", sa.Integer()),
        sa.column("dataset_id", sa.String(length=36)),
        sa.column("status", sa.String(length=50)),
        sa.column("deletion_owner", sa.String(length=128)),
    )
    rows = bind.execute(
        sa.select(datasets.c.id, datasets.c.dataset_id).where(
            datasets.c.status == "deleting",
            datasets.c.deletion_owner.is_(None),
        )
    ).all()
    for row in rows:
        bind.execute(
            datasets.update()
            .where(datasets.c.id == row.id)
            .values(deletion_owner=f"dataset:{row.dataset_id}"[:128])
        )


def _backfill_collection_owners() -> None:
    bind = op.get_bind()
    collections = sa.table(
        "milvus_collections",
        sa.column("id", sa.Integer()),
        sa.column("collection_id", sa.String(length=36)),
        sa.column("status", sa.String(length=32)),
        sa.column("deletion_owner", sa.String(length=128)),
    )
    rows = bind.execute(
        sa.select(collections.c.id, collections.c.collection_id).where(
            collections.c.status == "deleting",
            collections.c.deletion_owner.is_(None),
        )
    ).all()
    for row in rows:
        bind.execute(
            collections.update()
            .where(collections.c.id == row.id)
            .values(deletion_owner=f"manual:{row.collection_id}"[:128])
        )


def upgrade() -> None:
    if _add_owner_column("datasets"):
        _backfill_dataset_owners()
    if _add_owner_column("milvus_collections"):
        _backfill_collection_owners()


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    for table_name in ("milvus_collections", "datasets"):
        if table_name not in tables:
            continue
        columns = {
            column["name"]
            for column in inspect(op.get_bind()).get_columns(table_name)
        }
        if "deletion_owner" in columns:
            op.drop_column(table_name, "deletion_owner")
