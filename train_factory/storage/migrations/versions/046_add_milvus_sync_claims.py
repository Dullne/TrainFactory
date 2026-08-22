"""Add durable external-sync provenance to Milvus registry rows.

Revision ID: 046_milvus_sync_claims
Revises: 045_deletion_fence_owners
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "046_milvus_sync_claims"
down_revision = "045_deletion_fence_owners"
branch_labels = None
depends_on = None


def _backfill_unique_current_claims() -> None:
    """Backfill only exact, globally unique, same-tenant task pointers."""
    bind = op.get_bind()
    collections = sa.table(
        "milvus_collections",
        sa.column("id", sa.Integer()),
        sa.column("collection_name", sa.String(length=255)),
        sa.column("user_id", sa.String(length=64)),
        sa.column("sync_task_id", sa.String(length=36)),
    )
    tasks = sa.table(
        "external_sync_tasks",
        sa.column("task_id", sa.String(length=36)),
        sa.column("user_id", sa.String(length=64)),
        sa.column("milvus_collection_name", sa.String(length=255)),
    )

    collection_rows = bind.execute(
        sa.select(
            collections.c.id,
            collections.c.collection_name,
            collections.c.user_id,
        ).where(collections.c.sync_task_id.is_(None))
    ).all()
    task_rows = bind.execute(
        sa.select(
            tasks.c.task_id,
            tasks.c.user_id,
            tasks.c.milvus_collection_name,
        ).where(tasks.c.milvus_collection_name.is_not(None))
    ).all()

    tasks_by_current_name = {}
    for task in task_rows:
        current_name = task.milvus_collection_name
        if not current_name:
            continue
        tasks_by_current_name.setdefault(current_name, []).append(task)

    for collection in collection_rows:
        matches = tasks_by_current_name.get(collection.collection_name, [])
        if len(matches) != 1:
            continue
        task = matches[0]
        if task.user_id != collection.user_id:
            continue
        bind.execute(
            collections.update()
            .where(
                collections.c.id == collection.id,
                collections.c.sync_task_id.is_(None),
            )
            .values(sync_task_id=task.task_id)
        )


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "milvus_collections" not in tables:
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("milvus_collections")
    }
    if "sync_task_id" not in columns:
        op.add_column(
            "milvus_collections",
            sa.Column(
                "sync_task_id",
                sa.String(length=36),
                nullable=True,
            ),
        )
    if "external_sync_tasks" in tables:
        _backfill_unique_current_claims()


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if "milvus_collections" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("milvus_collections")
    }
    if "sync_task_id" in columns:
        op.drop_column("milvus_collections", "sync_task_id")
