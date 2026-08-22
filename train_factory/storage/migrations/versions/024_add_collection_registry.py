"""
Add milvus_collections and collection_dataset_links tables.

Provides independent management of Milvus vector collections with
many-to-many relationship to datasets. Each collection is bound to
exactly one embedding model.

Also back-fills existing collections from generation_tasks records.

Revision ID: 024_add_collection_registry
Revises: 023_add_deep_eval_fields
Create Date: 2026-02-10
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text

revision: str = "024_add_collection_registry"
down_revision: Union[str, None] = "023_add_deep_eval_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    # ── 1. Create milvus_collections table ──

    if not _table_exists("milvus_collections"):
        op.create_table(
            "milvus_collections",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("collection_id", sa.String(36), nullable=False, unique=True),
            sa.Column("collection_name", sa.String(255), nullable=False),
            sa.Column("display_name", sa.String(255), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("embedding_config_id", sa.String(36), nullable=True),
            sa.Column("embedding_model", sa.String(255), nullable=True),
            sa.Column("embedding_endpoint", sa.String(512), nullable=True),
            sa.Column("dim", sa.Integer(), nullable=False, server_default="1024"),
            sa.Column("metric_type", sa.String(32), nullable=False, server_default="COSINE"),
            sa.Column("status", sa.String(32), nullable=False, server_default="active"),
            sa.Column("user_id", sa.String(64), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_milvus_coll_id", "milvus_collections", ["collection_id"])
        op.create_index("idx_milvus_coll_name", "milvus_collections", ["collection_name"])
        op.create_index("idx_milvus_coll_status", "milvus_collections", ["status"])
        op.create_index("idx_milvus_coll_embedding", "milvus_collections", ["embedding_config_id"])
        op.create_index("idx_milvus_coll_user_created", "milvus_collections", ["user_id", "created_at"])
        op.create_unique_constraint("uq_milvus_collection_name", "milvus_collections", ["collection_name"])

    # ── 2. Create collection_dataset_links table ──

    if not _table_exists("collection_dataset_links"):
        op.create_table(
            "collection_dataset_links",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("collection_name", sa.String(255), nullable=False),
            sa.Column("dataset_id", sa.String(36), nullable=False),
            sa.Column("dataset_name", sa.String(255), nullable=True),
            sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("task_id", sa.String(36), nullable=True),
            sa.Column("linked_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_coll_link_collection", "collection_dataset_links", ["collection_name"])
        op.create_index("idx_coll_link_dataset", "collection_dataset_links", ["dataset_id"])
        op.create_unique_constraint(
            "uq_coll_dataset_link",
            "collection_dataset_links",
            ["collection_name", "dataset_id"],
        )

    # ── 3. Back-fill from existing generation_tasks ──

    conn = op.get_bind()

    # Find all generation tasks that have a milvus_collection set
    rows = conn.execute(
        text(
            "SELECT task_id, milvus_collection, embedding_config_id, "
            "embedding_config, source_dataset_id, user_id "
            "FROM generation_tasks "
            "WHERE milvus_collection IS NOT NULL AND milvus_collection != ''"
        )
    ).fetchall()

    seen_collections = set()

    def _ensure_dataset_link(collection_name: str, dataset_id: str, task_id: str) -> None:
        existing_link = conn.execute(
            text(
                "SELECT id FROM collection_dataset_links "
                "WHERE collection_name = :cn AND dataset_id = :did"
            ),
            {"cn": collection_name, "did": dataset_id},
        ).fetchone()
        if not existing_link:
            conn.execute(
                text(
                    "INSERT INTO collection_dataset_links "
                    "(collection_name, dataset_id, task_id) "
                    "VALUES (:cn, :did, :tid)"
                ),
                {"cn": collection_name, "did": dataset_id, "tid": task_id},
            )

    for row in rows:
        coll_name = row[1]  # milvus_collection
        if coll_name in seen_collections:
            # Already registered this collection, just add dataset link
            ds_id = row[4]  # source_dataset_id
            if ds_id:
                _ensure_dataset_link(coll_name, ds_id, row[0])
            continue

        seen_collections.add(coll_name)

        # Parse embedding_config JSON for model/endpoint
        import json
        emb_cfg = {}
        if row[3]:  # embedding_config
            try:
                emb_cfg = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            except Exception:
                pass

        emb_model = emb_cfg.get("model")
        emb_endpoint = emb_cfg.get("endpoint")
        emb_config_id = row[2]  # embedding_config_id

        import uuid
        coll_id = str(uuid.uuid4())

        existing_collection = conn.execute(
            text(
                "SELECT id FROM milvus_collections "
                "WHERE collection_name = :cn"
            ),
            {"cn": coll_name},
        ).fetchone()
        if not existing_collection:
            conn.execute(
                text(
                    "INSERT INTO milvus_collections "
                    "(collection_id, collection_name, display_name, "
                    "embedding_config_id, embedding_model, embedding_endpoint, "
                    "dim, metric_type, status, user_id, created_at, updated_at) "
                    "VALUES (:cid, :cn, :dn, :eci, :em, :ee, :dim, :mt, :st, :uid, NOW(), NOW())"
                ),
                {
                    "cid": coll_id,
                    "cn": coll_name,
                    "dn": coll_name,
                    "eci": emb_config_id,
                    "em": emb_model,
                    "ee": emb_endpoint,
                    "dim": 1024,  # All existing collections use 1024
                    "mt": "COSINE",
                    "st": "active",
                    "uid": row[5],  # user_id
                },
            )

        # Link source dataset if available
        ds_id = row[4]  # source_dataset_id
        if ds_id:
            _ensure_dataset_link(coll_name, ds_id, row[0])


def downgrade() -> None:
    if _table_exists("collection_dataset_links"):
        op.drop_table("collection_dataset_links")
    if _table_exists("milvus_collections"):
        op.drop_table("milvus_collections")
