"""
Training task database entity.

Simplified from the original business project, removing user-related fields.
"""

from datetime import datetime
from train_factory.core.time_utils import now_naive
from typing import Optional, Dict, Any, ClassVar, Set
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import Double, Index, UniqueConstraint
import uuid

from ...enums.training_status import TrainingStatus


class TrainingTaskDB(SQLModel, table=True):
    """Training task database model."""

    __tablename__ = "training_tasks"

    # Unique constraint: same user cannot have duplicate task names
    __table_args__ = (
        UniqueConstraint('user_id', 'task_name', name='uq_task_user_name'),
        Index('idx_task_user_created', 'user_id', 'created_at'),
        Index('idx_task_type_status', 'model_type', 'status'),
        # 与迁移 043_add_training_run_token 的索引名一致（此前 Field(index=True)
        # 会生成 ix_training_tasks_run_token，双轨命名造成运维困惑）
        Index('idx_training_run_token', 'run_token'),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)

    # === Basic Info ===
    task_name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None)
    # === Model & Training Type ===
    model_type: str = Field(default="embedding", max_length=32, index=True)
    # model_type: embedding / reranker / decoder_reranker / llm
    training_method: str = Field(default="sft", max_length=32, index=True)
    # training_method: sft / dpo / grpo / dapo / dr_grpo / two_stage
    model_architecture: str = Field(default="encoder", max_length=32)
    # model_architecture: encoder / decoder

    # === User Isolation ===
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)

    # === Model Info ===
    base_model_path: Optional[str] = Field(default=None, max_length=1024)
    final_model_path: Optional[str] = Field(default=None, max_length=1024)

    # === Dataset Info ===
    train_dataset_path: Optional[str] = Field(default=None, max_length=1024)

    # === Output & Device ===
    output_dir: Optional[str] = Field(default=None, max_length=1024)
    embedding_dim: Optional[int] = Field(default=None)
    device: Optional[str] = Field(default="cuda:0", max_length=64)

    # === LoRA & Checkpoints ===
    is_lora: bool = Field(default=False)
    checkpoints: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    best_checkpoint_path: Optional[str] = Field(default=None, max_length=1024)
    loss_data: Optional[str] = Field(default=None)

    # === Model Registry Link ===
    trained_model_registry_id: Optional[str] = Field(default=None, max_length=36, index=True)

    # === RL & Two-Stage Training (New) ===
    rl_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    # rl_config: {"rl_method": "dpo", "beta": 0.1, "kl_coef": 0.1, ...}
    loss_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    # loss_config: {"name": "lambda_loss", "metric": "ndcg", "sigma": 1.0}
    parent_task_id: Optional[str] = Field(default=None, max_length=36, index=True)
    # parent_task_id: For two-stage training, links to SFT task
    sft_checkpoint_path: Optional[str] = Field(default=None, max_length=1024)
    # sft_checkpoint_path: SFT checkpoint for RL stage

    # === Status ===
    status: str = Field(default="pending", max_length=50, index=True)
    progress: float = Field(default=0.0)
    error_message: Optional[str] = Field(default=None)

    # === Training Parameters (JSON) ===
    training_params: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Results (JSON) ===
    final_metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Process Info ===
    run_token: Optional[str] = Field(default=None, max_length=36)
    process_pid: Optional[int] = Field(default=None)
    process_status: Optional[str] = Field(default=None, max_length=50)
    process_create_time: Optional[float] = Field(
        default=None,
        sa_column=Column(Double, nullable=True),
    )

    # === Timestamps ===
    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Valid status transitions (matching TrainingStatus enum)
    # Note: Some states are reserved for future use:
    # - "preparing": For model download/data loading phase (not yet implemented)
    # - "evaluating": For post-training evaluation phase (not yet implemented)
    # - "cancelled": For tasks cancelled before starting (use "stopped" for now)
    VALID_TRANSITIONS: ClassVar[Dict[str, Set[str]]] = {
        "pending": {"preparing", "running", "failed", "stopped", "cancelled"},
        "preparing": {"running", "failed", "stopped"},
        "running": {"evaluating", "succeeded", "failed", "stopped"},
        "evaluating": {"succeeded", "failed", "stopped"},
        "succeeded": set(),  # Terminal state
        "failed": {"pending"},   # Allow resume from checkpoint
        "stopped": {"pending"},  # Allow resume from checkpoint
        "cancelled": set(),  # Terminal state
    }

    def update_status(self, status: str, error_message: Optional[str] = None):
        """Update task status with transition validation.

        Raises:
            ValueError: If the status is not a valid TrainingStatus
            ValueError: If the status transition is not allowed
        """
        # Validate status is a valid enum value
        valid_statuses = {s.value for s in TrainingStatus}
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
        if status == "running" and self.started_at is None:
            self.started_at = now_naive()
        if status in ["succeeded", "failed", "stopped", "cancelled"]:
            self.completed_at = now_naive()

    def update_progress(self, progress: float):
        """Update training progress."""
        self.progress = min(max(progress, 0.0), 100.0)
        self.updated_at = now_naive()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "description": self.description,
            "model_type": self.model_type,
            "training_method": self.training_method,
            "model_architecture": self.model_architecture,
            "user_id": self.user_id,
            "status": self.status,
            "progress": self.progress,
            "error_message": self.error_message,
            "base_model_path": self.base_model_path,
            "final_model_path": self.final_model_path,
            "train_dataset_path": self.train_dataset_path,
            "output_dir": self.output_dir,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "is_lora": self.is_lora,
            "checkpoints": self.checkpoints,
            "best_checkpoint_path": self.best_checkpoint_path,
            "loss_data": self.loss_data,
            "trained_model_registry_id": self.trained_model_registry_id,
            "rl_config": self.rl_config,
            "loss_config": self.loss_config,
            "parent_task_id": self.parent_task_id,
            "sft_checkpoint_path": self.sft_checkpoint_path,
            "training_params": self.training_params,
            "final_metrics": self.final_metrics,
            "run_token": self.run_token,
            "process_pid": self.process_pid,
            "process_status": self.process_status,
            "process_create_time": self.process_create_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
