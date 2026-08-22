"""
Data Generation Task database entity.

Manages data generation task metadata.
"""

from datetime import datetime
from train_factory.core.time_utils import now_naive
from typing import Optional, Dict, Any, ClassVar, Set
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import Index
import uuid


class GenerationStatus:
    """数据生成任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    PUBLISHING = "publishing"
    RECOVERING = "recovering"
    RESTARTING = "restarting"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    DELETING = "deleting"
    DELETING_CASCADE = "deleting_cascade"


class GenerationMode:
    """生成模式"""
    DOC_TO_TRAINING = "doc_to_training"  # 原始文档 → QA + 训练数据
    QA_TO_TRAINING = "qa_to_training"   # QA 数据集 → 训练数据
    QA_EXTRACTION = "qa_extraction"      # 原始文档 → QA 对（含 chunk 关联）
    DOC_TO_EVAL = "doc_to_eval"          # 原始文档 → 深度评估数据集
    QA_TO_EVAL = "qa_to_eval"            # QA 数据集 → 深度评估数据集


class GenerationTaskDB(SQLModel, table=True):
    """Data Generation Task database model."""

    __tablename__ = "generation_tasks"

    __table_args__ = (
        Index('idx_gen_task_user_created', 'user_id', 'created_at'),
        Index('idx_gen_task_status', 'status'),
        Index('idx_gen_task_run_token', 'run_token'),
    )

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)

    # === Basic Info ===
    task_name: str = Field(max_length=255, index=True)
    description: Optional[str] = Field(default=None)

    # === Input Configuration ===
    input_path: str = Field(max_length=1024)  # 输入文件/目录路径
    input_format: str = Field(default="auto", max_length=32)  # auto, jsonl, json, txt
    content_field: Optional[str] = Field(default=None, max_length=64)  # 内容字段名
    generation_mode: str = Field(default=GenerationMode.DOC_TO_TRAINING, max_length=32)

    # === 正负例生成方式 ===
    pos_neg_method: Optional[str] = Field(default="retrieval", max_length=32)  # retrieval 或 llm

    # === Output Configuration ===
    output_path: Optional[str] = Field(default=None, max_length=1024)
    output_format: str = Field(default="universal", max_length=32)

    # === LLM Configuration ===
    llm_config: Dict[str, Any] = Field(sa_column=Column(JSON))
    eval_llm_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Embedding Configuration (Optional) ===
    embedding_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Rerank Configuration (Optional) ===
    rerank_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Worker Configuration ===
    worker_config: Dict[str, Any] = Field(
        default={"concurrency": 10, "timeout_per_doc": 300},
        sa_column=Column(JSON)
    )

    # === Steps Configuration ===
    steps_config: Dict[str, Any] = Field(sa_column=Column(JSON))

    # === Post-process Configuration ===
    post_process_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Custom Prompts ===
    custom_prompts: Optional[Dict[str, str]] = Field(default=None, sa_column=Column(JSON))

    # === Progress ===
    status: str = Field(default=GenerationStatus.PENDING, max_length=50, index=True)
    progress: float = Field(default=0.0)  # 0-100
    total_docs: int = Field(default=0)
    processed_docs: int = Field(default=0)
    output_sample_count: int = Field(default=0)
    run_token: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        max_length=36,
    )

    # === Output Dataset ===
    output_dataset_id: Optional[str] = Field(default=None, max_length=36, index=True)
    auto_register_dataset: bool = Field(default=True)

    # === 源数据集（通用，记录输入数据集 ID） ===
    source_dataset_id: Optional[str] = Field(default=None, max_length=36, index=True)

    # === Training 模式配置 ===
    embedding_config_id: Optional[str] = Field(default=None, max_length=36)
    milvus_collection: Optional[str] = Field(default=None, max_length=255)
    similarity_threshold: float = Field(default=0.85)
    retrieval_top_k: int = Field(default=10)
    filter_stats: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # === Doc-to-Training 中间产物 ===
    qa_output_path: Optional[str] = Field(default=None, max_length=1024)
    qa_dataset_id: Optional[str] = Field(default=None, max_length=36)
    qa_filtered_path: Optional[str] = Field(default=None, max_length=1024)
    qa_filtered_dataset_id: Optional[str] = Field(default=None, max_length=36)

    # === 深度评估数据集 ===
    deep_eval_path: Optional[str] = Field(default=None, max_length=1024)
    deep_eval_dataset_id: Optional[str] = Field(default=None, max_length=36)

    # === Error Info ===
    error_message: Optional[str] = Field(default=None)

    # === User Isolation ===
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)

    # === Timestamps ===
    created_at: datetime = Field(default_factory=now_naive)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Valid status transitions
    VALID_TRANSITIONS: ClassVar[Dict[str, Set[str]]] = {
        GenerationStatus.PENDING: {
            GenerationStatus.RUNNING,
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
        },
        GenerationStatus.RUNNING: {
            GenerationStatus.STOPPING,
            GenerationStatus.PUBLISHING,
            GenerationStatus.FAILED,
        },
        GenerationStatus.STOPPING: {
            GenerationStatus.STOPPED,
            GenerationStatus.FAILED,
        },
        GenerationStatus.PUBLISHING: {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
        },
        GenerationStatus.RECOVERING: {
            GenerationStatus.STOPPED,
            GenerationStatus.FAILED,
        },
        GenerationStatus.RESTARTING: {
            GenerationStatus.PENDING,
            GenerationStatus.FAILED,
        },
        GenerationStatus.COMPLETED: {
            GenerationStatus.PENDING,
            GenerationStatus.RESTARTING,
            GenerationStatus.DELETING,
            GenerationStatus.DELETING_CASCADE,
        },
        GenerationStatus.FAILED: {
            GenerationStatus.PENDING,
            GenerationStatus.RESTARTING,
            GenerationStatus.DELETING,
            GenerationStatus.DELETING_CASCADE,
        },
        GenerationStatus.STOPPED: {
            GenerationStatus.PENDING,
            GenerationStatus.RESTARTING,
            GenerationStatus.DELETING,
            GenerationStatus.DELETING_CASCADE,
        },
        GenerationStatus.DELETING: set(),
        GenerationStatus.DELETING_CASCADE: set(),
    }

    def update_status(self, status: str, error_message: Optional[str] = None):
        """Update task status with validation."""
        valid_statuses = {
            GenerationStatus.PENDING,
            GenerationStatus.RUNNING,
            GenerationStatus.STOPPING,
            GenerationStatus.PUBLISHING,
            GenerationStatus.RECOVERING,
            GenerationStatus.RESTARTING,
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
            GenerationStatus.DELETING,
            GenerationStatus.DELETING_CASCADE,
        }
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of: {valid_statuses}")

        # Validate transition
        if self.status and self.status in self.VALID_TRANSITIONS:
            allowed = self.VALID_TRANSITIONS[self.status]
            if status not in allowed and status != self.status:
                raise ValueError(
                    f"Invalid status transition: {self.status} -> {status}. "
                    f"Allowed: {allowed or 'none'}"
                )

        self.status = status
        if status == GenerationStatus.RUNNING and not self.started_at:
            self.started_at = now_naive()
        elif status in {GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.STOPPED}:
            self.completed_at = now_naive()

        if error_message:
            self.error_message = error_message

    def update_progress(self, processed: int, total: Optional[int] = None, output_count: Optional[int] = None):
        """Update progress."""
        self.processed_docs = processed
        if total is not None:
            self.total_docs = total
        if output_count is not None:
            self.output_sample_count = output_count
        if self.total_docs > 0:
            self.progress = (processed / self.total_docs) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # 隐藏敏感信息
        def hide_api_key(config: Optional[Dict]) -> Optional[Dict]:
            if not config:
                return None
            result = config.copy()
            if "api_key" in result:
                result["api_key"] = "***" if result["api_key"] else None
            if "endpoints" in result:
                result["endpoints"] = [
                    {**ep, "api_key": "***" if ep.get("api_key") else None}
                    for ep in result["endpoints"]
                ]
            return result

        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "description": self.description,
            "input_path": self.input_path,
            "input_format": self.input_format,
            "content_field": self.content_field,
            "generation_mode": self.generation_mode,
            "pos_neg_method": self.pos_neg_method,
            "output_path": self.output_path,
            "output_format": self.output_format,
            "llm_config": hide_api_key(self.llm_config),
            "eval_llm_config": hide_api_key(self.eval_llm_config),
            "embedding_config": hide_api_key(self.embedding_config),
            "rerank_config": hide_api_key(self.rerank_config),
            "worker_config": self.worker_config,
            "steps_config": self.steps_config,
            "post_process_config": self.post_process_config,
            "status": self.status,
            "progress": self.progress,
            "total_docs": self.total_docs,
            "processed_docs": self.processed_docs,
            "output_sample_count": self.output_sample_count,
            "output_dataset_id": self.output_dataset_id,
            "auto_register_dataset": self.auto_register_dataset,
            "source_dataset_id": self.source_dataset_id,
            "embedding_config_id": self.embedding_config_id,
            "milvus_collection": self.milvus_collection,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_top_k": self.retrieval_top_k,
            "filter_stats": self.filter_stats,
            "qa_output_path": self.qa_output_path,
            "qa_dataset_id": self.qa_dataset_id,
            "qa_filtered_path": self.qa_filtered_path,
            "qa_filtered_dataset_id": self.qa_filtered_dataset_id,
            "deep_eval_path": self.deep_eval_path,
            "deep_eval_dataset_id": self.deep_eval_dataset_id,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
