"""
Dataset database entity.

Manages dataset metadata for the training platform.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, ClassVar, Set
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import Index, UniqueConstraint
import uuid

from ...enums.dataset_status import DatasetStatus
from train_factory.core.time_utils import now_naive


class DatasetDB(SQLModel, table=True):
    """Dataset database model."""

    __tablename__ = "datasets"

    # Unique constraint: same user cannot have duplicate dataset names
    __table_args__ = (
        UniqueConstraint('user_id', 'dataset_name', name='uq_dataset_user_name'),
        UniqueConstraint('storage_path_hash', name='uq_dataset_storage_path_hash'),
        Index('idx_dataset_user_created', 'user_id', 'created_at'),
        Index('idx_dataset_type_status', 'dataset_type', 'status'),
        Index('idx_dataset_generation_run_token', 'generation_run_token'),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)

    # === Basic Info ===
    dataset_name: str = Field(max_length=255, index=True)
    display_name: Optional[str] = Field(default=None, max_length=256)
    description: Optional[str] = Field(default=None)

    # === Dataset Type ===
    # dataset_type: embedding_pair, embedding_triplet, rerank_pair, rerank_triplet,
    #               rerank_listwise, sft_instruct, dpo_preference, rl_reward, custom
    dataset_type: str = Field(default="custom", max_length=50, index=True)

    # === Usage ===
    # usage: raw (原始数据/源文档，用于生成训练数据), train (训练数据集), eval (验证数据集), test (测试数据集)
    usage: Optional[str] = Field(default=None, max_length=20, index=True)

    # === 适用模型标签 ===
    # 可选，用户自行标注，可多选: ["embedding"], ["embedding", "rerank"], ["llm"]
    model_type: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))

    # === Source Info ===
    # source_type: uploaded, huggingface, modelscope, local, generated
    source_type: str = Field(default="uploaded", max_length=32)
    source_path: Optional[str] = Field(default=None, max_length=1024)
    source_dataset_id: Optional[str] = Field(default=None, max_length=36, index=True)
    # For HuggingFace/ModelScope datasets
    remote_repo: Optional[str] = Field(default=None, max_length=512)
    hf_subset: Optional[str] = Field(default=None, max_length=128)

    # === Storage ===
    storage_path: Optional[str] = Field(default=None, max_length=1024)  # Local path (None for s3-only)
    storage_path_hash: Optional[str] = Field(default=None, max_length=64, index=True)
    file_format: str = Field(default="parquet", max_length=32)  # parquet, jsonl, json, csv, arrow

    # Object storage (MinIO/S3)
    storage_backend: str = Field(default="local", max_length=16)  # local | s3
    storage_uri: Optional[str] = Field(default=None, max_length=2048)  # s3://bucket/key
    version: int = Field(default=1)  # dataset version, incremented on content change

    # === Schema Info ===
    # columns: [{"name": "query", "type": "string"}, {"name": "positive", "type": "string"}]
    columns: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    # sample_data: First few rows for preview
    sample_data: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    # Auto-inferred content schema from dataset_type
    content_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Statistics ===
    num_rows: Optional[int] = Field(default=None)
    num_train: Optional[int] = Field(default=None)
    num_eval: Optional[int] = Field(default=None)
    num_test: Optional[int] = Field(default=None)
    file_size: Optional[int] = Field(default=None)  # in bytes

    # === Provenance ===
    source_task_type: Optional[str] = Field(default=None, max_length=32)  # sync/generation/training/manual/import
    source_task_id: Optional[str] = Field(default=None, max_length=36)
    # Internal attempt owner for non-consumable generation staging rows.
    generation_run_token: Optional[str] = Field(default=None, max_length=36)

    # === Metadata ===
    tags: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Status ===
    # status: registered, uploading, downloading, processing, staging, ready,
    #         error, archived, deleting
    status: str = Field(default="registered", max_length=50, index=True)
    deletion_owner: Optional[str] = Field(default=None, max_length=128)
    error_message: Optional[str] = Field(default=None)

    # === User Isolation ===
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)

    # === Timestamps ===
    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)

    # Valid status transitions
    VALID_TRANSITIONS: ClassVar[Dict[str, Set[str]]] = {
        "registered": {"uploading", "downloading", "processing", "ready", "error", "deleting"},
        "uploading": {"processing", "ready", "error", "deleting"},
        "downloading": {"processing", "ready", "error", "deleting"},
        "processing": {"ready", "error", "deleting"},
        "staging": {"ready", "error", "deleting"},
        "ready": {"archived", "deleting"},
        "error": {"registered", "uploading", "downloading", "processing", "deleting"},  # Allow retry
        "archived": {"ready", "deleting"},  # Allow unarchive
        "deleting": set(),
    }

    def update_status(self, status: str, error_message: Optional[str] = None):
        """Update dataset status with validation.

        Raises:
            ValueError: If the status is not a valid DatasetStatus
            ValueError: If the status transition is not allowed
        """
        # Validate status is a valid enum value
        valid_statuses = {s.value for s in DatasetStatus}
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid status: {status}. "
                f"Must be one of: {', '.join(valid_statuses)}"
            )

        # Validate transition (allow any transition from None/initial state)
        if self.status and self.status in self.VALID_TRANSITIONS:
            allowed = self.VALID_TRANSITIONS[self.status]
            if status not in allowed and status != self.status:
                raise ValueError(
                    f"Invalid status transition: {self.status} -> {status}. "
                    f"Allowed: {allowed or 'none'}"
                )

        self.status = status
        self.updated_at = now_naive()
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "display_name": self.display_name,
            "description": self.description,
            "dataset_type": self.dataset_type,
            "usage": self.usage,
            "model_type": self.model_type,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "source_dataset_id": self.source_dataset_id,
            "remote_repo": self.remote_repo,
            "hf_subset": self.hf_subset,
            "storage_path": self.storage_path,
            "storage_backend": self.storage_backend,
            "storage_uri": self.storage_uri,
            "version": self.version,
            "file_format": self.file_format,
            "columns": self.columns,
            "sample_data": self.sample_data,
            "content_schema": self.content_schema,
            "num_rows": self.num_rows,
            "num_train": self.num_train,
            "num_eval": self.num_eval,
            "num_test": self.num_test,
            "file_size": self.file_size,
            "source_task_type": self.source_task_type,
            "source_task_id": self.source_task_id,
            "tags": self.tags,
            "extra_metadata": self.extra_metadata,
            "status": self.status,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
