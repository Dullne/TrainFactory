"""
External data sync entities.

Manages external data synchronization configurations, batch records,
generation task tracking, and training task tracking for the
two-level threshold auto-trigger pipeline.
"""

import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import Index, Text
import uuid
from ...core.time_utils import sync_now_naive
from ...enums.sync_status import (
    SyncStatus,
    BatchStatus,
    SyncGenerationStatus,
    SyncTrainingStatus,
)


_SENSITIVE_CONFIG_KEYS = {
    "authorization",
    "bearer",
    "cookie",
    "token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "credential",
    "private_key",
    "privatekey",
    "access_key",
    "accesskey",
}
_SENSITIVE_CONFIG_SUFFIXES = (
    "_authorization",
    "_cookie",
    "_token",
    "_api_key",
    "_apikey",
    "_secret",
    "_password",
    "_credential",
    "_private_key",
    "_privatekey",
    "_access_key",
    "_accesskey",
)
_SENSITIVE_CONFIG_PREFIXES = (
    "authorization_",
    "cookie_",
    "api_key_",
    "apikey_",
    "secret_",
    "password_",
    "credential_",
    "private_key_",
    "privatekey_",
    "access_key_",
    "accesskey_",
)


def _is_sensitive_config_key(key: Any) -> bool:
    raw_key = re.sub(r"(?<!^)(?=[A-Z])", "_", str(key).strip())
    normalized_key = re.sub(
        r"_+",
        "_",
        raw_key.lower().replace("-", "_").replace(" ", "_"),
    )
    return (
        normalized_key in _SENSITIVE_CONFIG_KEYS
        or normalized_key.endswith(_SENSITIVE_CONFIG_SUFFIXES)
        or normalized_key.startswith(_SENSITIVE_CONFIG_PREFIXES)
    )


def _serialize_config_value(value: Any, *, mask_sensitive: bool) -> Any:
    """Copy JSON-like config data and recursively redact public secrets."""
    if isinstance(value, dict):
        serialized: Dict[Any, Any] = {}
        for key, child in value.items():
            if mask_sensitive and _is_sensitive_config_key(key):
                serialized[key] = "***"
            else:
                serialized[key] = _serialize_config_value(
                    child,
                    mask_sensitive=mask_sensitive,
                )
        return serialized
    if isinstance(value, list):
        return [
            _serialize_config_value(child, mask_sensitive=mask_sensitive)
            for child in value
        ]
    if isinstance(value, tuple):
        return [
            _serialize_config_value(child, mask_sensitive=mask_sensitive)
            for child in value
        ]
    return value


class ExternalSyncTaskDB(SQLModel, table=True):
    """外部数据同步任务"""

    __tablename__ = "external_sync_tasks"

    __table_args__ = (
        Index("idx_sync_user", "user_id"),
        Index("idx_sync_active", "is_active", "status"),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        index=True, unique=True, max_length=36,
    )
    task_name: str = Field(max_length=255)
    user_id: str = Field(max_length=64)

    # === 外部 API 配置 ===
    external_api_config_id: Optional[str] = Field(default=None, max_length=36)
    external_api_url: str = Field(default="", max_length=1024)
    external_auth_config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # === 同步参数 ===
    sync_interval_seconds: int = Field(default=300)
    boundary_rollback_seconds: int = Field(default=5)
    last_sync_at: Optional[datetime] = Field(default=None)
    last_sync_boundary_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    # === Level 1: 生成阈值 ===
    generation_threshold: int = Field(default=500)
    generation_mode: str = Field(default="doc_to_training", max_length=32)
    generation_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Level 2: 训练阈值 ===
    training_threshold: int = Field(default=1000)
    training_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === 部署配置 ===
    base_deployment_id: Optional[str] = Field(default=None, max_length=36)

    # === Milvus 向量库 ===
    milvus_collection_name: Optional[str] = Field(default=None, max_length=255)

    # === 计数器 ===
    pending_record_count: int = Field(default=0)
    pending_training_samples: int = Field(default=0)
    total_record_count: int = Field(default=0)
    total_training_samples: int = Field(default=0)
    total_trainings: int = Field(default=0)

    # === 当前 adapter ===
    current_adapter_name: Optional[str] = Field(default=None, max_length=255)
    current_adapter_id: Optional[str] = Field(default=None, max_length=36)
    current_training_id: Optional[str] = Field(default=None, max_length=36)

    # === 状态 ===
    is_active: bool = Field(default=True)
    status: str = Field(default=SyncStatus.IDLE, max_length=32)
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))

    # === Timestamps ===
    created_at: datetime = Field(default_factory=sync_now_naive)
    updated_at: datetime = Field(default_factory=sync_now_naive)

    def to_dict(self, mask_sensitive: bool = True) -> Dict[str, Any]:
        """Convert to a public DTO, or an explicit raw internal DTO."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "user_id": self.user_id,
            "external_api_config_id": self.external_api_config_id,
            "external_api_url": self.external_api_url,
            "external_auth_config": _serialize_config_value(
                self.external_auth_config or {},
                mask_sensitive=mask_sensitive,
            ),
            "sync_interval_seconds": self.sync_interval_seconds,
            "boundary_rollback_seconds": self.boundary_rollback_seconds,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "last_sync_boundary_ids": self.last_sync_boundary_ids or [],
            "generation_threshold": self.generation_threshold,
            "generation_mode": self.generation_mode,
            "generation_config": _serialize_config_value(
                self.generation_config,
                mask_sensitive=mask_sensitive,
            ),
            "training_threshold": self.training_threshold,
            "training_config": _serialize_config_value(
                self.training_config,
                mask_sensitive=mask_sensitive,
            ),
            "base_deployment_id": self.base_deployment_id,
            "milvus_collection_name": self.milvus_collection_name,
            "pending_record_count": self.pending_record_count,
            "pending_training_samples": self.pending_training_samples,
            "total_record_count": self.total_record_count,
            "total_training_samples": self.total_training_samples,
            "total_trainings": self.total_trainings,
            "current_adapter_name": self.current_adapter_name,
            "current_adapter_id": self.current_adapter_id,
            "current_training_id": self.current_training_id,
            "is_active": self.is_active,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ExternalSyncBatchDB(SQLModel, table=True):
    """外部数据同步批次"""

    __tablename__ = "external_sync_batches"

    __table_args__ = (
        Index("idx_batch_task", "task_id", "status"),
        Index("idx_batch_user", "user_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        index=True, unique=True, max_length=36,
    )
    task_id: str = Field(max_length=36)
    user_id: str = Field(max_length=64)

    record_count: int = Field(default=0)
    storage_path: str = Field(max_length=1024)
    dataset_id: Optional[str] = Field(default=None, max_length=36)

    # === 时间窗口 ===
    since_time: Optional[datetime] = Field(default=None)
    until_time: Optional[datetime] = Field(default=None)
    fetched_at: datetime = Field(default_factory=sync_now_naive)

    # === 状态追踪 ===
    status: str = Field(default=BatchStatus.FETCHED, max_length=32)
    generation_task_id: Optional[str] = Field(default=None, max_length=36)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "record_count": self.record_count,
            "storage_path": self.storage_path,
            "dataset_id": self.dataset_id,
            "since_time": self.since_time.isoformat() if self.since_time else None,
            "until_time": self.until_time.isoformat() if self.until_time else None,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "status": self.status,
            "generation_task_id": self.generation_task_id,
        }


class ExternalSyncTrainingTargetDB(SQLModel, table=True):
    """同步任务的训练目标配置"""

    __tablename__ = "external_sync_training_targets"

    __table_args__ = (
        Index("idx_training_target_task", "task_id", "is_active"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    target_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        index=True, unique=True, max_length=36,
    )
    task_id: str = Field(max_length=36, index=True)

    # 目标配置
    target_name: str = Field(max_length=255)
    model_type: str = Field(default="embedding", max_length=32)
    data_phase: str = Field(default="final", max_length=32)  # "qa" or "final"
    training_method: str = Field(default="sft", max_length=32)

    # 训练参数
    training_config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # 基座模型和部署
    base_model_path: str = Field(default="", max_length=1024)
    base_deployment_id: Optional[str] = Field(default=None, max_length=36)

    # 独立阈值和计数器
    training_threshold: int = Field(default=1000)
    pending_training_samples: int = Field(default=0)
    total_training_samples: int = Field(default=0)
    total_trainings: int = Field(default=0)

    # 独立 adapter 追踪
    current_adapter_name: Optional[str] = Field(default=None, max_length=255)
    current_adapter_id: Optional[str] = Field(default=None, max_length=36)
    current_training_id: Optional[str] = Field(default=None, max_length=36)

    # 调度
    priority: int = Field(default=0)
    status: str = Field(default="idle", max_length=32)

    is_active: bool = Field(default=True)
    sort_order: int = Field(default=0)
    created_at: datetime = Field(default_factory=sync_now_naive)
    updated_at: datetime = Field(default_factory=sync_now_naive)

    def to_dict(self, mask_sensitive: bool = True) -> Dict[str, Any]:
        data = self.model_dump()
        data["training_config"] = _serialize_config_value(
            self.training_config or {},
            mask_sensitive=mask_sensitive,
        )
        return data


class ExternalSyncGenerationDB(SQLModel, table=True):
    """外部同步关联的生成任务追踪"""

    __tablename__ = "external_sync_generations"

    __table_args__ = (
        Index("idx_syncgen_task_id", "task_id", "status"),
        Index("idx_syncgen_task", "generation_task_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(max_length=36)
    generation_task_id: str = Field(max_length=36, unique=True)
    user_id: str = Field(max_length=64)

    input_batch_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    input_record_count: int = Field(default=0)

    output_dataset_id: Optional[str] = Field(default=None, max_length=36)
    output_sample_count: int = Field(default=0)

    qa_dataset_id: Optional[str] = Field(default=None, max_length=36)
    qa_sample_count: int = Field(default=0)

    status: str = Field(default=SyncGenerationStatus.PENDING, max_length=32)
    created_at: datetime = Field(default_factory=sync_now_naive)
    completed_at: Optional[datetime] = Field(default=None)
    disabled: bool = Field(default=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "generation_task_id": self.generation_task_id,
            "user_id": self.user_id,
            "input_batch_ids": self.input_batch_ids,
            "input_record_count": self.input_record_count,
            "output_dataset_id": self.output_dataset_id,
            "output_sample_count": self.output_sample_count,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "disabled": self.disabled,
        }


class ExternalSyncTrainingDB(SQLModel, table=True):
    """外部同步关联的训练任务追踪"""

    __tablename__ = "external_sync_trainings"

    __table_args__ = (
        Index("idx_synctrain_task_id", "task_id", "status"),
        Index("idx_synctrain_task", "training_task_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(max_length=36)
    training_task_id: str = Field(max_length=36, unique=True)
    user_id: str = Field(max_length=64)

    target_id: Optional[str] = Field(default=None, max_length=36)

    input_dataset_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    total_samples: int = Field(default=0)
    training_round: int = Field(default=1)

    output_adapter_path: Optional[str] = Field(default=None, max_length=1024)
    output_model_registry_id: Optional[str] = Field(default=None, max_length=36)
    loaded_adapter_name: Optional[str] = Field(default=None, max_length=255)
    loaded_adapter_id: Optional[str] = Field(default=None, max_length=36)
    previous_training_task_id: Optional[str] = Field(default=None, max_length=36)
    claimed_sample_count: int = Field(default=0)
    previous_target_status: Optional[str] = Field(default=None, max_length=32)
    previous_parent_status: Optional[str] = Field(default=None, max_length=32)
    claim_reconciled: bool = Field(default=False)
    target_config_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
    )

    status: str = Field(default=SyncTrainingStatus.PENDING, max_length=32)
    created_at: datetime = Field(default_factory=sync_now_naive)
    completed_at: Optional[datetime] = Field(default=None)

    def to_dict(self, mask_sensitive: bool = True) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "training_task_id": self.training_task_id,
            "user_id": self.user_id,
            "target_id": self.target_id,
            "input_dataset_ids": self.input_dataset_ids,
            "total_samples": self.total_samples,
            "training_round": self.training_round,
            "output_adapter_path": self.output_adapter_path,
            "output_model_registry_id": self.output_model_registry_id,
            "loaded_adapter_name": self.loaded_adapter_name,
            "loaded_adapter_id": self.loaded_adapter_id,
            "previous_training_task_id": self.previous_training_task_id,
            "claimed_sample_count": self.claimed_sample_count,
            "previous_target_status": self.previous_target_status,
            "previous_parent_status": self.previous_parent_status,
            "claim_reconciled": self.claim_reconciled,
            "target_config_snapshot": _serialize_config_value(
                self.target_config_snapshot or {},
                mask_sensitive=mask_sensitive,
            ),
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
