"""
Loaded Adapter database entity.

Tracks LoRA adapters that are currently loaded on deployments.
"""

from datetime import datetime
from train_factory.core.time_utils import now_naive
from typing import Optional, Dict, Any, ClassVar, Set
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, ForeignKey, Index, String
import uuid


class LoadedAdapterDB(SQLModel, table=True):
    """Loaded Adapter database model.

    Tracks which LoRA adapters are loaded on which deployments.
    Only vLLM and SGLang support hot-loading of adapters.
    """

    __tablename__ = "loaded_adapters"

    __table_args__ = (
        Index('idx_adapter_deployment', 'deployment_id'),
        Index('idx_adapter_source_task', 'source_task_id'),
        Index('idx_adapter_status', 'status'),
        Index('idx_loaded_adapters_deployment_replica_id', 'deployment_replica_id'),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    adapter_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)

    # === Deployment Reference ===
    deployment_id: str = Field(max_length=36, index=True)  # Related deployment ID
    deployment_replica_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String(36),
            ForeignKey("deployment_replicas.replica_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # === Adapter Info ===
    adapter_name: str = Field(max_length=255)  # Name used in vLLM/SGLang API
    adapter_path: str = Field(max_length=1024)  # File system path to adapter weights

    # === Source Reference ===
    source_task_id: Optional[str] = Field(default=None, max_length=36, index=True)  # Training task that created this adapter
    source_model_id: Optional[str] = Field(default=None, max_length=36)  # If adapter is from model registry

    # === Status ===
    status: str = Field(default="loading", max_length=32, index=True)  # loading / loaded / unloading / unloaded / failed
    error_message: Optional[str] = Field(default=None)

    # === User Isolation ===
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)

    # === Timestamps ===
    loaded_at: datetime = Field(default_factory=now_naive)
    unloaded_at: Optional[datetime] = Field(default=None)

    # Valid status transitions
    VALID_TRANSITIONS: ClassVar[Dict[str, Set[str]]] = {
        "loading": {"loaded", "failed"},
        "loaded": {"unloading"},
        "unloading": {"unloaded", "failed"},
        "unloaded": set(),  # Terminal state
        "failed": {"loading"},  # Allow retry
    }

    def update_status(self, status: str, error_message: Optional[str] = None):
        """Update adapter status with transition validation."""
        valid_statuses = {"loading", "loaded", "unloading", "unloaded", "failed"}
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid status: {status}. "
                f"Must be one of: {', '.join(valid_statuses)}"
            )

        if self.status and self.status in self.VALID_TRANSITIONS:
            allowed = self.VALID_TRANSITIONS[self.status]
            if status not in allowed and status != self.status:
                raise ValueError(
                    f"Invalid status transition: {self.status} -> {status}. "
                    f"Allowed: {allowed or 'none'}"
                )

        self.status = status
        if error_message:
            self.error_message = error_message
        if status == "unloaded":
            self.unloaded_at = now_naive()

    def force_status(self, status: str, error_message: Optional[str] = None):
        """Force adapter status without transition validation.

        Used for administrative operations (auto-sync, cleanup, revert)
        where the normal state machine doesn't apply.
        """
        self.status = status
        if error_message is not None:
            self.error_message = error_message
        if status == "unloaded":
            self.unloaded_at = now_naive()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adapter_id": self.adapter_id,
            "deployment_id": self.deployment_id,
            "deployment_replica_id": self.deployment_replica_id,
            "adapter_name": self.adapter_name,
            "adapter_path": self.adapter_path,
            "source_task_id": self.source_task_id,
            "source_model_id": self.source_model_id,
            "status": self.status,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "unloaded_at": self.unloaded_at.isoformat() if self.unloaded_at else None,
        }
