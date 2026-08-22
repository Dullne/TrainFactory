"""Milvus collection registry and dataset link entities."""

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Index, UniqueConstraint

from train_factory.core.time_utils import now_naive


class MilvusCollectionDB(SQLModel, table=True):
    """Registry entry for a Milvus vector collection.

    Each collection is bound to exactly one embedding model.
    Multiple datasets can contribute vectors to the same collection.
    """

    __tablename__ = "milvus_collections"

    __table_args__ = (
        UniqueConstraint("collection_name", name="uq_milvus_collection_name"),
        Index("idx_milvus_coll_user_created", "user_id", "created_at"),
        Index("idx_milvus_coll_status", "status"),
        Index("idx_milvus_coll_embedding", "embedding_config_id"),
        Index(
            "idx_milvus_generation_attempt",
            "generation_task_id",
            "generation_run_token",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        index=True,
        unique=True,
    )
    collection_name: str = Field(max_length=255, index=True)
    display_name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None)

    # Embedding model binding (immutable after creation)
    embedding_config_id: Optional[str] = Field(default=None, max_length=36)
    embedding_model: Optional[str] = Field(default=None, max_length=255)
    embedding_endpoint: Optional[str] = Field(default=None, max_length=512)

    # Collection properties
    dim: int = Field(default=1024)
    metric_type: str = Field(default="COSINE", max_length=32)
    # Nullable keeps pre-migration creating rows fail-closed: their original
    # immutable hybrid schema choice cannot be inferred safely.
    hybrid_enabled: Optional[bool] = Field(default=None)

    # Status
    status: str = Field(default="active", max_length=32, index=True)
    deletion_owner: Optional[str] = Field(default=None, max_length=128)

    # Internal producer provenance.  Public DTOs intentionally omit this field.
    sync_task_id: Optional[str] = Field(default=None, max_length=36)
    generation_task_id: Optional[str] = Field(default=None, max_length=36)
    generation_run_token: Optional[str] = Field(default=None, max_length=36)

    # User isolation
    user_id: Optional[str] = Field(default=None, max_length=64)

    # Timestamps
    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)

    def to_dict(self):
        return {
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "display_name": self.display_name,
            "description": self.description,
            "embedding_config_id": self.embedding_config_id,
            "embedding_model": self.embedding_model,
            "embedding_endpoint": self.embedding_endpoint,
            "dim": self.dim,
            "metric_type": self.metric_type,
            "hybrid_enabled": self.hybrid_enabled,
            "status": self.status,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CollectionDatasetLinkDB(SQLModel, table=True):
    """Many-to-many link between Milvus collections and datasets."""

    __tablename__ = "collection_dataset_links"

    __table_args__ = (
        UniqueConstraint(
            "collection_name", "dataset_id",
            name="uq_coll_dataset_link",
        ),
        Index("idx_coll_link_collection", "collection_name"),
        Index("idx_coll_link_dataset", "dataset_id"),
        Index("idx_coll_link_generation_run_token", "generation_run_token"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    collection_name: str = Field(max_length=255)
    dataset_id: str = Field(max_length=36)
    dataset_name: Optional[str] = Field(default=None, max_length=255)
    chunk_count: int = Field(default=0)
    task_id: Optional[str] = Field(default=None, max_length=36)
    # Internal generation-attempt owner.  Public DTOs intentionally omit it.
    generation_run_token: Optional[str] = Field(default=None, max_length=36)
    linked_at: datetime = Field(default_factory=now_naive)

    def to_dict(self):
        return {
            "collection_name": self.collection_name,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "chunk_count": self.chunk_count,
            "task_id": self.task_id,
            "linked_at": self.linked_at.isoformat() if self.linked_at else None,
        }
