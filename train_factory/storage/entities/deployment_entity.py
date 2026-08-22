"""
Deployment database entity.

Stores information about model deployments to Xinference or other serving platforms.
"""

from datetime import datetime
from train_factory.core.time_utils import now_naive
from typing import Optional, Dict, Any, ClassVar, Set
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import Index, UniqueConstraint
import uuid

from ...enums.deployment_status import DeploymentStatus


class DeploymentDB(SQLModel, table=True):
    """Deployment database model."""

    __tablename__ = "deployments"

    # Unique constraint: same model cannot have duplicate deployment names
    __table_args__ = (
        UniqueConstraint('user_id', 'deployment_name', name='uq_deployment_user_name'),
        Index('idx_deployment_model_status', 'model_id', 'status'),
        Index('idx_deployment_user_created', 'user_id', 'created_at'),
        Index('idx_deployment_external_api_config_id', 'external_api_config_id'),
        Index('idx_deployment_external_api_status', 'external_api_config_id', 'status'),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    deployment_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)

    # === Model Reference ===
    model_id: str = Field(max_length=36, index=True)  # Related model registry ID
    model_uid: Optional[str] = Field(default=None, max_length=255)  # Model UID in Xinference

    # === Deployment Name ===
    deployment_name: Optional[str] = Field(default=None, max_length=255)

    # === Deployment Configuration ===
    xinference_endpoint: str = Field(max_length=512)
    replica: int = Field(default=1)
    gpu_memory_utilization: float = Field(default=0.9)

    # === Container Mode ===
    deploy_mode: str = Field(default="shared", max_length=32)  # shared / container
    container_name: Optional[str] = Field(default=None, max_length=255)
    gpu_id: Optional[int] = Field(default=None)
    port: Optional[int] = Field(default=None)

    # === Inference Framework ===
    inference_framework: str = Field(default="xinference", max_length=32)  # vllm / sglang / xinference

    # === LoRA Hot-Loading ===
    enable_lora: bool = Field(default=False)
    max_loras: int = Field(default=4)
    max_lora_rank: int = Field(default=64)

    # === Additional Config ===
    external_api_config_id: Optional[str] = Field(default=None, max_length=36, index=True)
    config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Health Check ===
    health_check_url: Optional[str] = Field(default=None, max_length=512)
    last_health_check: Optional[datetime] = Field(default=None)
    health_status: str = Field(default="UNKNOWN", max_length=32)  # HEALTHY / UNHEALTHY / UNKNOWN

    # === Status ===
    status: str = Field(default="pending", max_length=50, index=True)
    error_message: Optional[str] = Field(default=None)

    # === User Isolation ===
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)

    # === Timestamps ===
    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)
    started_at: Optional[datetime] = Field(default=None)
    stopped_at: Optional[datetime] = Field(default=None)

    # Valid status transitions
    VALID_TRANSITIONS: ClassVar[Dict[str, Set[str]]] = {
        "pending": {"starting", "failed", "stopped", "restarting"},  # stopped: user cancel before start
        "starting": {"running", "failed", "stopping", "stopped", "restarting"},  # stopping: user abort during startup
        "running": {"stopping", "stopped", "failed", "restarting"},  # stopped: for shared stop detection (auto-sync)
        "stopping": {"stopped", "failed", "restarting"},
        "stopped": {"starting", "pending", "restarting"},  # Allow restart or reset
        "failed": {"pending", "starting", "restarting"},  # Allow retry from failed state
        "restarting": {"running", "failed", "stopped"},
    }

    def update_status(self, status: str, error_message: Optional[str] = None):
        """Update deployment status with transition validation.

        Raises:
            ValueError: If the status is not a valid DeploymentStatus
            ValueError: If the status transition is not allowed
        """
        # Validate status is a valid enum value
        valid_statuses = {s.value for s in DeploymentStatus}
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
        elif status == "running":
            # Clear error message when status changes to running
            self.error_message = None
        if status == "running" and self.started_at is None:
            self.started_at = now_naive()
        if status in ["stopped", "failed"]:
            self.stopped_at = now_naive()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        deploy_mode = "shared" if self.deploy_mode == "external" else self.deploy_mode
        return {
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "model_uid": self.model_uid,
            "deployment_name": self.deployment_name,
            "xinference_endpoint": self.xinference_endpoint,
            "replica": self.replica,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "deploy_mode": deploy_mode,
            "container_name": self.container_name,
            "gpu_id": self.gpu_id,
            "port": self.port,
            "inference_framework": self.inference_framework,
            "enable_lora": self.enable_lora,
            "max_loras": self.max_loras,
            "max_lora_rank": self.max_lora_rank,
            "external_api_config_id": self.external_api_config_id,
            "config": self.config,
            "health_check_url": self.health_check_url,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_status": self.health_status,
            "status": self.status,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
        }
