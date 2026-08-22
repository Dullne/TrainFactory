"""
Evaluation task database entity.

Unified evaluation task entity supporting both MTEB and DeepEval frameworks.
"""

from datetime import datetime
from train_factory.core.time_utils import now_naive
from typing import Optional, Dict, Any, ClassVar, Set, List
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import Index
import uuid


class EvaluationFramework:
    """评估框架类型"""
    MTEB = "mteb"        # 有标签评估，不需要 LLM
    DEEPEVAL = "deepeval"  # 无标签评估，需要 LLM 评判


class EvaluationStatus:
    """评估任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"  # MTEB 用
    COMPLETED = "completed"  # DeepEval 用，兼容
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationTaskDB(SQLModel, table=True):
    """Unified evaluation task database model."""

    __tablename__ = "evaluation_tasks"

    __table_args__ = (
        Index('idx_eval_user_created', 'user_id', 'created_at'),
        Index('idx_eval_status', 'status'),
        Index('idx_eval_framework', 'eval_framework'),
        Index('idx_eval_run_token', 'run_token'),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)

    # === Basic Info ===
    task_name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None)

    # === Framework & Type ===
    eval_framework: str = Field(default=EvaluationFramework.MTEB, max_length=32)
    # eval_framework: "mteb" | "deepeval"
    eval_type: str = Field(default="single", max_length=32)
    # MTEB eval_type: single / multi_model / multi_dataset
    # DeepEval eval_type: embedding / rerank

    # === Model Configuration (JSON) ===
    # MTEB: [{"model_id": "...", "endpoint": "...", "model_name": "...", "name": "...", "gpu_id": 0}]
    # DeepEval: [{"endpoint": "...", "model_name": "...", "api_key": "..."}]
    model_configs: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))

    # === Dataset Configuration (JSON) ===
    # MTEB: [{"type": "mteb", "name": "T2Reranking"}, {"type": "local", "name": "...", "path": "..."}]
    # DeepEval: [{"dataset_id": "...", "name": "..."}]
    dataset_configs: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))

    # === Field Mapping (DeepEval) ===
    # {"query": "question", "positives": "positive_passages", "negatives": "negative_passages"}
    field_mapping: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Metrics (DeepEval) ===
    # ["mrr", "map", "ndcg@10", "recall@10", "precision@10"]
    metrics: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))

    # === LLM Configuration (DeepEval) ===
    # {"endpoint": "...", "model": "...", "api_key": "..."}
    llm_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Worker Group Configuration (DeepEval) ===
    # {"embedding": {"workers": 2, "concurrency": 8}, "rerank": {"workers": 1, "concurrency": 4}, "llm": {...}}
    worker_groups: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Evaluation Parameters ===
    max_samples: Optional[int] = Field(default=None)  # 采样数量
    batch_size: int = Field(default=50)
    workers: int = Field(default=8)  # MTEB: workers, DeepEval: concurrency
    model_workers: int = Field(default=2)  # MTEB only

    # === Status ===
    status: str = Field(default="pending", max_length=50, index=True)
    # Immutable execution-attempt fence for standard MTEB workers. DeepEval
    # tasks and pre-migration rows may keep this NULL.
    run_token: Optional[str] = Field(default=None, max_length=36)
    progress: float = Field(default=0.0)
    current_model: Optional[str] = Field(default=None, max_length=255)
    current_dataset: Optional[str] = Field(default=None, max_length=255)
    error_message: Optional[str] = Field(default=None)

    # === Progress Tracking ===
    total_samples: int = Field(default=0)  # DeepEval: 总样本数
    processed_samples: int = Field(default=0)  # DeepEval: 已处理样本数
    # MTEB: {model_name: {dataset_name: {"progress": 0-100, "status": "..."}}}
    model_progress: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))

    # === Results (JSON) ===
    # MTEB: {model_name: {dataset_name: {metric_name: value}}}
    # DeepEval: {metric_name: value, "details": [...]}
    results: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    results_path: Optional[str] = Field(default=None, max_length=1024)

    # === User Isolation ===
    user_id: Optional[str] = Field(default=None, max_length=64)

    # === Timestamps ===
    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Valid status transitions
    VALID_TRANSITIONS: ClassVar[Dict[str, Set[str]]] = {
        "pending": {"running", "failed", "cancelled"},
        "running": {"succeeded", "completed", "failed", "cancelled"},
        "succeeded": set(),  # Terminal state
        "completed": set(),  # Terminal state (DeepEval alias)
        "failed": {"pending", "running"},  # Allow retry/resume
        "cancelled": {"pending", "running"},  # Allow restart
    }

    def update_status(self, status: str, error_message: Optional[str] = None):
        """Update task status with transition validation."""
        valid_statuses = {"pending", "running", "succeeded", "completed", "failed", "cancelled"}
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid status: {status}. "
                f"Must be one of: {', '.join(valid_statuses)}"
            )

        # Validate transition
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
        if status in ["succeeded", "completed", "failed", "cancelled"]:
            self.completed_at = now_naive()

    def update_progress(
        self,
        progress: Optional[float] = None,
        current_model: Optional[str] = None,
        current_dataset: Optional[str] = None,
        processed_samples: Optional[int] = None,
        total_samples: Optional[int] = None,
    ):
        """Update evaluation progress."""
        if progress is not None:
            self.progress = min(max(progress, 0.0), 100.0)
        if current_model is not None:
            self.current_model = current_model
        if current_dataset is not None:
            self.current_dataset = current_dataset
        if processed_samples is not None:
            self.processed_samples = processed_samples
        if total_samples is not None:
            self.total_samples = total_samples
        # Auto-calculate progress for DeepEval
        if self.eval_framework == EvaluationFramework.DEEPEVAL and self.total_samples > 0:
            self.progress = (self.processed_samples / self.total_samples) * 100
        self.updated_at = now_naive()

    def update_model_progress(
        self,
        model_name: str,
        dataset_name: str,
        progress: float,
        status: str = "running",
        *,
        current_model: Optional[str] = None,
        current_dataset: Optional[str] = None,
    ):
        """Update progress for a specific model-dataset pair (MTEB)."""
        if self.model_progress is None:
            self.model_progress = {}
        if model_name not in self.model_progress:
            self.model_progress[model_name] = {}
        self.model_progress[model_name][dataset_name] = {
            "progress": min(max(progress, 0.0), 100.0),
            "status": status,
        }
        # Update current_model and current_dataset if running
        if status == "running":
            self.current_model = str(current_model or model_name)[:255]
            self.current_dataset = str(current_dataset or dataset_name)[:255]
        self.updated_at = now_naive()

    def to_dict(self, mask_api_key: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Mask API keys if requested
        model_configs = self.model_configs
        llm_config = self.llm_config
        if mask_api_key:
            if model_configs:
                model_configs = [
                    {**cfg, "api_key": "***"} if cfg.get("api_key") else cfg
                    for cfg in model_configs
                ]
            if llm_config and llm_config.get("api_key"):
                llm_config = {**llm_config, "api_key": "***"}

        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "description": self.description,
            "eval_framework": self.eval_framework,
            "eval_type": self.eval_type,
            "model_configs": model_configs,
            "dataset_configs": self.dataset_configs,
            "field_mapping": self.field_mapping,
            "metrics": self.metrics,
            "llm_config": llm_config,
            "worker_groups": self.worker_groups,
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "workers": self.workers,
            "model_workers": self.model_workers,
            "status": self.status,
            "run_token": self.run_token,
            "progress": self.progress,
            "current_model": self.current_model,
            "current_dataset": self.current_dataset,
            "total_samples": self.total_samples,
            "processed_samples": self.processed_samples,
            "model_progress": self.model_progress,
            "error_message": self.error_message,
            "results": self.results,
            "results_path": self.results_path,
            "report_path": self.results_path,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# Backward compatibility aliases
DeepEvaluationStatus = EvaluationStatus
DeepEvaluationTaskDB = EvaluationTaskDB
