"""
Model registry database entity.

Stores information about registered models including metadata, versions, and metrics.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, ClassVar, Set
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import Index, UniqueConstraint
import uuid
from ...enums.model_status import ModelStatus
from train_factory.core.time_utils import now_naive


MODEL_DELETE_INTENT_METADATA_KEY = "_train_factory_model_delete_intent_v1"


def public_model_extra_metadata(
    extra_metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Strip operator-owned lifecycle metadata from public model payloads."""
    if not isinstance(extra_metadata, dict):
        return extra_metadata
    intent = extra_metadata.get(MODEL_DELETE_INTENT_METADATA_KEY)
    if isinstance(intent, dict) and type(intent.get("had_extra_metadata")) is bool:
        if not intent["had_extra_metadata"]:
            return None
    public_metadata = dict(extra_metadata)
    public_metadata.pop(MODEL_DELETE_INTENT_METADATA_KEY, None)
    return public_metadata

class ModelRegistryDB(SQLModel, table=True):
    """Model registry database model."""

    __tablename__ = "model_registry"

    # Unique constraint: same user cannot have duplicate model_name + version
    __table_args__ = (
        UniqueConstraint('user_id', 'model_name', 'version', name='uq_model_user_name_version'),
        UniqueConstraint('model_path_hash', name='uq_model_path_hash'),
        Index('idx_model_user_created', 'user_id', 'created_at'),
        Index('idx_model_type_status', 'model_type', 'status'),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)

    # === Basic Info ===
    model_name: str = Field(max_length=255, index=True)
    display_name: Optional[str] = Field(default=None, max_length=256)
    version: str = Field(default="v1.0.0", max_length=50)
    model_type: str = Field(max_length=50, index=True)  # embedding / reranker

    # === Source Info ===
    source_task_id: Optional[str] = Field(default=None, max_length=36, index=True)  # Related training task
    source_model_id: Optional[str] = Field(default=None, max_length=36, index=True)  # Parent model for lineage
    base_model_path: Optional[str] = Field(default=None, max_length=1024)
    model_path: str = Field(max_length=1024)  # Actual model storage path
    model_path_hash: Optional[str] = Field(default=None, max_length=64, index=True)
    best_checkpoint_path: Optional[str] = Field(default=None, max_length=1024)
    checkpoints: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Model Properties ===
    embedding_dim: Optional[int] = Field(default=None)
    source_type: str = Field(default="trained", max_length=32, index=True)  # trained / downloaded / uploaded / external_bind
    is_adapter: bool = Field(default=False)  # Is this an adapter/LoRA model

    # === Download Info (for downloaded models) ===
    download_source: Optional[str] = Field(default=None, max_length=32)  # modelscope / huggingface
    remote_repo: Optional[str] = Field(default=None, max_length=512)
    download_status: Optional[str] = Field(default=None, max_length=32, index=True)  # pending / downloading / completed / failed
    download_progress: int = Field(default=0)
    download_error: Optional[str] = Field(default=None)

    # === Metadata ===
    description: Optional[str] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    category: Optional[str] = Field(default=None, max_length=100)
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Performance Metrics ===
    metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    file_size: Optional[int] = Field(default=None)  # Model file size in bytes

    # === Status ===
    status: str = Field(default="registered", max_length=50, index=True)
    is_latest: bool = Field(default=True)  # Is this the latest version for this model name

    # === User Isolation ===
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)

    # === Timestamps ===
    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)

    # Valid status transitions
    VALID_TRANSITIONS: ClassVar[Dict[str, Set[str]]] = {
        ModelStatus.REGISTERED.value: {
            ModelStatus.AVAILABLE.value,
            ModelStatus.ARCHIVED.value,
        },
        ModelStatus.AVAILABLE.value: {
            ModelStatus.ARCHIVED.value,
        },
        ModelStatus.ARCHIVED.value: {
            ModelStatus.AVAILABLE.value,
        },
    }

    def update_status(self, status: str):
        """Update model status with validation."""
        valid_statuses = {s.value for s in ModelStatus}
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid status: {status}. Must be one of: {', '.join(sorted(valid_statuses))}"
            )

        if self.status and self.status in self.VALID_TRANSITIONS:
            allowed = self.VALID_TRANSITIONS[self.status]
            if status not in allowed and status != self.status:
                raise ValueError(
                    f"Invalid status transition: {self.status} -> {status}. "
                    f"Allowed: {allowed or 'none'}"
                )

        self.status = status
        self.updated_at = now_naive()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "version": self.version,
            "model_type": self.model_type,
            "source_task_id": self.source_task_id,
            "source_model_id": self.source_model_id,
            "base_model_path": self.base_model_path,
            "model_path": self.model_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "checkpoints": self.checkpoints,
            "embedding_dim": self.embedding_dim,
            "source_type": self.source_type,
            "is_adapter": self.is_adapter,
            "download_source": self.download_source,
            "remote_repo": self.remote_repo,
            "download_status": self.download_status,
            "download_progress": self.download_progress,
            "download_error": self.download_error,
            "description": self.description,
            "tags": self.tags,
            "category": self.category,
            "extra_metadata": public_model_extra_metadata(self.extra_metadata),
            "metrics": self.metrics,
            "file_size": self.file_size,
            "status": self.status,
            "is_latest": self.is_latest,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ModelVersionDB(SQLModel, table=True):
    """Model version history database model."""

    __tablename__ = "model_versions"

    # Unique constraint: same model cannot have duplicate versions
    __table_args__ = (
        UniqueConstraint('model_id', 'version', name='uq_version_model_version'),
        Index('idx_version_model_created', 'model_id', 'created_at'),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)

    # === Version Info ===
    model_id: str = Field(max_length=36, index=True)  # Related model registry ID
    version: str = Field(max_length=50)
    model_path: str = Field(max_length=1024)

    # === Changelog ===
    changelog: Optional[str] = Field(default=None)

    # === Metrics ===
    metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Timestamps ===
    created_at: datetime = Field(default_factory=now_naive)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "version": self.version,
            "model_path": self.model_path,
            "changelog": self.changelog,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
