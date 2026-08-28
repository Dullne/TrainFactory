"""Canonical lifecycle record for one container deployment replica."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import Column, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

from train_factory.core.time_utils import now_naive


class DeploymentReplicaDB(SQLModel, table=True):
    """One independently addressable container inside a deployment group."""

    __tablename__ = "deployment_replicas"
    __table_args__ = (
        UniqueConstraint("replica_id", name="uq_deployment_replica_id"),
        UniqueConstraint(
            "deployment_id",
            "replica_index",
            name="uq_deployment_replica_index",
        ),
        UniqueConstraint(
            "container_name",
            name="uq_deployment_replica_container_name",
        ),
        UniqueConstraint("port", name="uq_deployment_replica_port"),
        Index("idx_deployment_replica_deployment_id", "deployment_id"),
        Index(
            "idx_deployment_replica_deployment_status",
            "deployment_id",
            "status",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    replica_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        max_length=36,
    )
    deployment_id: str = Field(
        sa_column=Column(
            String(36),
            ForeignKey("deployments.deployment_id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    replica_index: int = Field(ge=0)
    container_name: str = Field(max_length=255)
    endpoint: str = Field(max_length=512)
    port: int = Field(ge=1024, le=65535)
    gpu_ids: list[int] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )

    status: str = Field(default="pending", max_length=50)
    health_status: str = Field(default="UNKNOWN", max_length=32)
    error_message: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )

    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)
    started_at: datetime | None = Field(default=None)
    stopped_at: datetime | None = Field(default=None)

    VALID_TRANSITIONS: ClassVar[dict[str, set[str]]] = {
        "pending": {"starting", "stopped", "failed"},
        "starting": {"running", "stopping", "stopped", "failed"},
        "running": {"stopping", "stopped", "restarting", "failed"},
        "stopping": {"stopped", "failed"},
        "stopped": {"starting", "restarting"},
        "restarting": {"running", "stopped", "failed"},
        "failed": {"starting", "restarting", "stopped"},
    }

    def update_status(self, status: str, error_message: str | None = None) -> None:
        if status not in self.VALID_TRANSITIONS:
            raise ValueError("invalid deployment replica status")
        allowed = self.VALID_TRANSITIONS.get(self.status, set())
        if status != self.status and status not in allowed:
            raise ValueError(
                f"invalid deployment replica status transition: {self.status} -> {status}"
            )
        self.status = status
        self.updated_at = now_naive()
        if error_message is not None:
            self.error_message = error_message
        elif status == "running":
            self.error_message = None
        if status == "running" and self.started_at is None:
            self.started_at = now_naive()
        if status in {"stopped", "failed"}:
            self.stopped_at = now_naive()

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica_id": self.replica_id,
            "deployment_id": self.deployment_id,
            "replica_index": self.replica_index,
            "container_name": self.container_name,
            "endpoint": self.endpoint,
            "port": self.port,
            "gpu_ids": list(self.gpu_ids),
            "status": self.status,
            "health_status": self.health_status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
        }
