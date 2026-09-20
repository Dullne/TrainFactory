"""
Training API routes.

Provides endpoints for creating, monitoring, and managing training tasks.
"""

import json
import logging
import math
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Header, Query
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from ..concurrency import threadpool_endpoint
from ...auth.dependencies import get_current_user, verify_resource_ownership, validate_storage_path
from ...auth.resource_provenance import (
    ResourceProvenanceError,
    require_managed_dataset_provenance,
    require_managed_model_provenance,
)
from ...core.gpu_resource_manager import gpu_resource_manager
from ...core.process_identity import (
    capture_process_create_time,
    terminate_process_if_matches,
)
from ...tuners.policy import (
    canonicalize_tuner_config,
    normalize_tuner_type,
)
from ...core.idempotency import check_idempotency, store_idempotency_response
from ...deployment.adapter_service import adapter_service
from ...storage.services.training_task_service import training_task_service
from ...storage.services.model_registry_service import model_registry_service
from ...storage.services.dataset_service import (
    DatasetConsumptionUnavailableError,
    dataset_service,
)
from ...storage.services.generation_task_service import generation_task_service
from ...storage.services.training_task_event_service import training_task_event_service
from ...storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    BackgroundTaskCapacityExceeded,
    background_task_admission_service,
)
from ...train import train_with_config
from ...enums import TrainingStatus
from ...config.settings import get_settings
from ...utils.path_utils import map_storage_path
from ...utils.public_diagnostics import (
    public_task_error_message,
    sanitize_public_diagnostics,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_PROCESS_START_ENV_LOCK = threading.Lock()
_TRAINING_PROCESS_MONITOR_INTERVAL_SECONDS = 1
_TRAINING_PROCESS_TERMINAL_DRAIN_SECONDS = 60
_TRAINING_FAILURE_PERSIST_ATTEMPTS = 3
DEFAULT_METRICS_HISTORY_RECORDS = 1_000
MAX_METRICS_HISTORY_RECORDS = 5_000
MAX_METRICS_HISTORY_BYTES = 8 * 1024 * 1024
MAX_METRICS_SUMMARY_BYTES = 1024 * 1024
_METRICS_TAIL_CHUNK_BYTES = 64 * 1024

_ACTIVE_TRAINING_STATUSES = {
    TrainingStatus.PREPARING.value,
    TrainingStatus.RUNNING.value,
    TrainingStatus.EVALUATING.value,
}
_STOPPED_TRAINING_STATUSES = {
    TrainingStatus.STOPPED.value,
    TrainingStatus.CANCELLED.value,
}
_COMPLETED_TRAINING_STATUSES = {
    TrainingStatus.SUCCEEDED.value,
    TrainingStatus.FAILED.value,
}
_TERMINAL_TRAINING_STATUSES = (
    _STOPPED_TRAINING_STATUSES | _COMPLETED_TRAINING_STATUSES
)


class _TrainingFailurePersistenceError(RuntimeError):
    """Raised when a current attempt cannot be made terminal in storage."""

MAX_TRAINING_DATASETS = 32
MAX_TRAINING_SAMPLES_PER_DATASET = 10_000_000
MAX_TRAINING_EPOCHS = 1_000
MAX_TRAINING_BATCH_SIZE = 4_096
MAX_TRAINING_GRADIENT_ACCUMULATION = 4_096
MAX_TRAINING_SEQUENCE_LENGTH = 131_072
MAX_TRAINING_GPU_COUNT = 16
MAX_TRAINING_GPU_ID = 1_024
MAX_TRAINING_JSON_CONFIG_BYTES = 65_536
MAX_LORA_R = 4_096
MAX_LORA_ALPHA = 65_536
MAX_RANKNET_PAIRS_PER_BATCH = 2_000_000
MAX_RL_ITERATIONS = 100


def _validate_bounded_number(
    config: Dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
    integer: bool = True,
    minimum_exclusive: bool = False,
) -> None:
    """Reject non-finite and out-of-range numeric resource settings."""
    if key not in config or config[key] is None:
        return

    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{key} must be finite")
    if integer and numeric_value != math.trunc(numeric_value):
        raise ValueError(f"{key} must be an integer")
    below_minimum = numeric_value <= minimum if minimum_exclusive else numeric_value < minimum
    if below_minimum or numeric_value > maximum:
        operator = ">" if minimum_exclusive else ">="
        raise ValueError(f"{key} must be {operator} {minimum} and <= {maximum}")


def _validate_bounded_json_config(name: str, value: Any) -> Dict[str, Any]:
    """Validate nested training configuration shape and serialized size."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{name} must contain valid JSON values") from exc
    if len(serialized) > MAX_TRAINING_JSON_CONFIG_BYTES:
        raise ValueError(
            f"{name} exceeds the {MAX_TRAINING_JSON_CONFIG_BYTES}-byte limit"
        )
    return value


def _validate_training_resource_limits(training_config: Dict[str, Any]) -> None:
    """Apply the same resource limits to API requests and persisted tasks."""
    numeric_limits = {
        "num_train_epochs": (1, MAX_TRAINING_EPOCHS),
        "per_device_train_batch_size": (1, MAX_TRAINING_BATCH_SIZE),
        "gradient_accumulation_steps": (1, MAX_TRAINING_GRADIENT_ACCUMULATION),
        "logging_steps": (1, 1_000_000),
        "eval_steps": (1, 10_000_000),
        "save_steps": (1, 10_000_000),
        "max_length": (1, MAX_TRAINING_SEQUENCE_LENGTH),
        "max_seq_length": (1, MAX_TRAINING_SEQUENCE_LENGTH),
        "lora_r": (1, MAX_LORA_R),
        "lora_alpha": (1, MAX_LORA_ALPHA),
    }
    for key, (minimum, maximum) in numeric_limits.items():
        _validate_bounded_number(
            training_config,
            key,
            minimum=minimum,
            maximum=maximum,
        )

    _validate_bounded_number(
        training_config,
        "learning_rate",
        minimum=0,
        maximum=1,
        integer=False,
        minimum_exclusive=True,
    )
    for key in ("warmup_ratio", "lora_dropout"):
        _validate_bounded_number(
            training_config,
            key,
            minimum=0,
            maximum=1,
            integer=False,
        )

    gpu_ids = training_config.get("gpu_ids")
    if gpu_ids is not None:
        if not isinstance(gpu_ids, list):
            raise ValueError("gpu_ids must be a list")
        if len(gpu_ids) > MAX_TRAINING_GPU_COUNT:
            raise ValueError(
                f"gpu_ids cannot contain more than {MAX_TRAINING_GPU_COUNT} entries"
            )
        seen_gpu_ids = set()
        for gpu_id in gpu_ids:
            if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
                raise ValueError("gpu_ids entries must be integers")
            if gpu_id < 0 or gpu_id > MAX_TRAINING_GPU_ID:
                raise ValueError(
                    f"gpu_ids entries must be between 0 and {MAX_TRAINING_GPU_ID}"
                )
            if gpu_id in seen_gpu_ids:
                raise ValueError("gpu_ids entries must be unique")
            seen_gpu_ids.add(gpu_id)

    datasets = training_config.get("datasets")
    if datasets is None:
        datasets = training_config.get("dataset_configs")
    if datasets is not None:
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("datasets must be a non-empty list")
        if len(datasets) > MAX_TRAINING_DATASETS:
            raise ValueError(
                f"datasets cannot contain more than {MAX_TRAINING_DATASETS} entries"
            )
        for index, dataset in enumerate(datasets):
            if hasattr(dataset, "model_dump"):
                dataset = dataset.model_dump()
            if not isinstance(dataset, dict):
                raise ValueError(f"datasets[{index}] must be an object")
            path = dataset.get("path")
            if not isinstance(path, str) or not path.strip() or len(path) > 2_048:
                raise ValueError(
                    f"datasets[{index}].path must contain 1 to 2048 characters"
                )
            split = dataset.get("split", "train")
            if split not in {"train", "eval", "test"}:
                raise ValueError(
                    f"datasets[{index}].split must be train, eval, or test"
                )
            if dataset.get("max_samples") is not None:
                _validate_bounded_number(
                    dataset,
                    "max_samples",
                    minimum=1,
                    maximum=MAX_TRAINING_SAMPLES_PER_DATASET,
                )

    loss_config = _validate_bounded_json_config(
        "loss_config", training_config.get("loss_config")
    )
    _validate_bounded_number(
        loss_config,
        "ranknet_max_pairs_per_batch",
        minimum=1,
        maximum=MAX_RANKNET_PAIRS_PER_BATCH,
    )
    for key in ("max_length", "n_docs", "n_pos"):
        _validate_bounded_number(
            loss_config,
            key,
            minimum=1,
            maximum=(
                MAX_TRAINING_SEQUENCE_LENGTH if key == "max_length" else 1_024
            ),
        )

    rl_config = _validate_bounded_json_config(
        "rl_config", training_config.get("rl_config")
    )
    _validate_bounded_number(
        rl_config,
        "num_iterations",
        minimum=1,
        maximum=MAX_RL_ITERATIONS,
    )
    for key, maximum in {
        "max_length": MAX_TRAINING_SEQUENCE_LENGTH,
        "reward_k": 1_000,
        "n_docs": 1_024,
        "num_generations": 128,
    }.items():
        _validate_bounded_number(
            rl_config,
            key,
            minimum=1,
            maximum=maximum,
        )
    _validate_bounded_number(
        rl_config,
        "chunk_size",
        minimum=0,
        maximum=1_000_000,
    )

    lora_config = _validate_bounded_json_config(
        "lora_config", training_config.get("lora_config")
    )
    _validate_bounded_number(
        lora_config,
        "r",
        minimum=1,
        maximum=MAX_LORA_R,
    )
    _validate_bounded_number(
        lora_config,
        "lora_alpha",
        minimum=1,
        maximum=MAX_LORA_ALPHA,
    )
    _validate_bounded_number(
        lora_config,
        "lora_dropout",
        minimum=0,
        maximum=1,
        integer=False,
    )


# === Request/Response Models ===

class DatasetConfig(BaseModel):
    """Single dataset configuration for training."""
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        min_length=1,
        max_length=2_048,
        description="Dataset path (local path or HuggingFace dataset name)",
    )
    max_samples: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_TRAINING_SAMPLES_PER_DATASET,
        description="Max samples to use from this dataset, None means no limit",
    )
    split: str = Field(default="train", description="Dataset split: 'train', 'eval', or 'test'")


class TrainingRequest(BaseModel):
    """Training request model."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    # Unified type system
    model_type: str = Field(
        default="embedding",
        description="Model type: 'embedding', 'reranker', 'decoder_reranker', 'llm'"
    )
    training_method: str = Field(
        default="sft",
        description=(
            "Training method: 'sft', 'cpt', 'dpo', 'grpo', 'dapo', 'dr_grpo', "
            "'kto', 'orpo', 'ppo', 'two_stage'"
        )
    )
    model_architecture: Optional[str] = Field(
        default=None,
        description="Model architecture: 'encoder' or 'decoder' (auto-detected if not specified)"
    )

    base_model_path: str = Field(..., min_length=1, max_length=2_048, description="Base model name or path")
    datasets: List[DatasetConfig] = Field(
        ...,
        min_length=1,
        max_length=MAX_TRAINING_DATASETS,
        description="Datasets with individual max_samples",
    )
    task_name: Optional[str] = Field(default=None, max_length=255, description="Task name")
    description: Optional[str] = Field(default=None, max_length=4_096, description="Task description")
    user_id: Optional[str] = Field(default=None, max_length=255, description="User ID for task isolation")

    # Training parameters
    num_train_epochs: int = Field(default=3, ge=1, le=MAX_TRAINING_EPOCHS, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=16, ge=1, le=MAX_TRAINING_BATCH_SIZE, description="Batch size per device")
    learning_rate: float = Field(default=2e-5, gt=0, le=1, description="Learning rate")
    warmup_ratio: float = Field(default=0.1, ge=0, le=1, description="Warmup ratio")
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=MAX_TRAINING_GRADIENT_ACCUMULATION, description="Gradient accumulation steps")
    logging_steps: int = Field(default=1, ge=1, le=1_000_000, description="Log metrics every N steps")
    max_length: Optional[int] = Field(default=None, ge=1, le=MAX_TRAINING_SEQUENCE_LENGTH, description="Max sequence length")

    # Optional parameters
    output_dir: Optional[str] = Field(default=None, description="Output directory")
    eval_strategy: str = Field(default="no", description="Evaluation strategy: no, epoch, steps")
    eval_steps: Optional[int] = Field(default=None, ge=1, le=10_000_000, description="Evaluate every N steps (when eval_strategy=steps)")
    save_strategy: str = Field(default="epoch", description="Save strategy")
    save_steps: Optional[int] = Field(default=None, ge=1, le=10_000_000, description="Save every N steps (when save_strategy=steps)")
    bf16: bool = Field(default=False, description="Use bf16 mixed precision")
    fp16: bool = Field(default=False, description="Use fp16 mixed precision")
    gradient_checkpointing: Optional[bool] = Field(
        default=None,
        description="Enable gradient checkpointing to reduce decoder model VRAM usage"
    )
    embedding_loss_name: Optional[str] = Field(
        default="auto",
        description="Embedding loss function name (encoder models)"
    )
    reranker_loss_name: Optional[str] = Field(
        default="auto",
        description=(
            "Reranker loss function name for encoder CrossEncoder training. "
            "Supported values are auto plus the sentence-transformers CrossEncoder losses "
            "enabled by this project: CrossEntropyLoss, BCEWithLogitsLoss "
            "(BinaryCrossEntropyLoss), MSELoss, MarginMSELoss, "
            "MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss, "
            "RankNetLoss, LambdaLoss, ListMLELoss, ListNetLoss, and PListMLELoss."
        )
    )

    # Tuner type (from frontend)
    tuner_type: Optional[Literal["lora", "qlora", "full", "freeze"]] = Field(
        default=None,
        description=(
            "Tuner type: 'lora' or 'full'; legacy qlora/freeze values are "
            "recognized and rejected with an explicit unsupported error"
        ),
    )

    # LoRA parameters
    use_lora: bool = Field(default=False, description="Enable LoRA fine-tuning")
    lora_r: int = Field(default=16, ge=1, le=MAX_LORA_R, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, le=MAX_LORA_ALPHA, description="LoRA alpha")
    lora_dropout: float = Field(default=0.0, ge=0, le=1, description="LoRA dropout")

    # Loss function configuration
    loss_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Loss function configuration (decoder_reranker/embedding/reranker). "
            "Example: {'name': 'lambda_loss', 'metric': 'ndcg', 'temperature': 0.05, "
            "'infonce_mode': 'single', 'ranknet_max_pairs_per_batch': 2000000}. "
            "metric options: ndcg | map | mrr"
        )
    )

    # RL training configuration (for dpo, grpo, dapo, dr_grpo)
    rl_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "RL training config: {'loss_type': 'dpo'|'grpo'|'dapo'|'dr_grpo', "
            "'beta': 0.1, 'reference_free': false, 'kl_coef': 0.1, 'clip_range': 0.2, "
            "'rankings_direction': 'auto'|'higher_is_better'|'lower_is_better', "
            "'reward_type': 'rank_based'|'score_based'|'ndcg_based'|'recall_based', "
            "'reward_k': 10, 'scale_rewards': false, 'num_iterations': 1, "
            "'chunk_size': 0, 'n_docs': 8, 'max_length': 4096}"
        )
    )

    # Two-stage training (RL after SFT)
    parent_task_id: Optional[str] = Field(
        default=None,
        description="Parent task ID for two-stage training (links to SFT task)"
    )
    sft_checkpoint_path: Optional[str] = Field(
        default=None,
        description="SFT checkpoint path for RL stage"
    )

    # DeepSpeed configuration
    deepspeed: Optional[str] = Field(
        default=None,
        description=(
            "DeepSpeed ZeRO stage preset: 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'. "
            "Only applicable to LLM and decoder_reranker training."
        )
    )

    # GPU allocation
    gpu_ids: Optional[List[int]] = Field(
        default=None,
        max_length=MAX_TRAINING_GPU_COUNT,
        description="GPU IDs to use for training (e.g., [0, 1]). Leave empty for auto-allocation."
    )

    @field_validator("tuner_type", mode="before")
    @classmethod
    def _normalize_tuner_type(cls, value):
        return normalize_tuner_type(value)

    @model_validator(mode="after")
    def validate_resource_limits(self):
        _validate_training_resource_limits(self.model_dump(exclude_none=True))
        return self


class TrainingResponse(BaseModel):
    """Training response model."""
    task_id: str
    task_name: Optional[str]
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Task status response model."""
    task_id: str
    task_name: Optional[str]
    # Unified type system
    model_type: str
    training_method: Optional[str] = None
    model_architecture: Optional[str] = None
    # Base model info
    base_model_path: str
    user_id: Optional[str]
    status: str
    progress: float
    error_message: Optional[str]
    final_model_path: Optional[str]
    # Dataset & Output
    train_dataset_path: str
    dataset_configs: Optional[List[Dict[str, Any]]] = None
    output_dir: Optional[str] = None
    # Training configuration
    training_params: Optional[Dict[str, Any]] = None
    is_lora: bool = False
    # Training progress details (extracted from training_params or runtime)
    learning_rate: Optional[float] = None
    num_train_epochs: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    # Real-time training metrics (from _runtime_metrics)
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    # GPU info
    gpu_ids: Optional[List[int]] = None
    # RL & Two-stage training
    rl_config: Optional[Dict[str, Any]] = None
    loss_config: Optional[Dict[str, Any]] = None
    parent_task_id: Optional[str] = None
    sft_checkpoint_path: Optional[str] = None
    # Results
    final_metrics: Optional[Dict[str, Any]] = None
    # Model registry link
    trained_model_registry_id: Optional[str] = None
    # Timestamps
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    updated_at: Optional[str] = None


class TaskStatsResponse(BaseModel):
    """Task statistics by status."""
    total: int = 0
    pending: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    stopped: int = 0


class TaskListResponse(BaseModel):
    """Task list response model."""
    tasks: List[TaskStatusResponse]
    total: int
    stats: Optional[TaskStatsResponse] = None


class TrainingMetricsResponse(BaseModel):
    """Training metrics response model for loss curves."""
    task_id: str
    # Loss history from JSONL file (for charting)
    loss_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Summary metrics from training_metrics.json
    summary: Optional[Dict[str, Any]] = None
    # Real-time metrics from database (_runtime_metrics)
    current_metrics: Optional[Dict[str, Any]] = None
    # Has data available
    has_data: bool = False


class TrainingTaskEventResponse(BaseModel):
    """Training task event response model."""
    event_id: str
    task_id: str
    user_id: Optional[str] = None
    event_type: str
    payload: Optional[Dict[str, Any]] = None
    created_at: Optional[str]


class TrainingTaskEventListResponse(BaseModel):
    """Training task event list response model."""
    events: List[TrainingTaskEventResponse]
    total: int


def _resolve_legacy_model_alias(
    path: str,
    user_id: Optional[str],
    model_type: Optional[str] = None,
) -> str:
    """Resolve historical /models/<name> inputs to a registered local model path."""
    if not path.startswith("/models/"):
        return path

    model_name = path.rsplit("/", 1)[-1].strip()
    if not model_name:
        return path

    try:
        candidates = model_registry_service.search_models(
            query=model_name,
            model_type=model_type,
            user_id=user_id,
            limit=20,
        )
    except Exception as exc:
        logger.warning("Failed to resolve legacy model alias %s: %s", path, exc)
        return path

    exact_matches = [
        model for model in candidates
        if model.get("model_name") == model_name and model.get("model_path")
    ]
    if exact_matches:
        resolved_path = exact_matches[0]["model_path"]
        logger.info("Resolved legacy model alias %s -> %s", path, resolved_path)
        return resolved_path

    return path


def _should_validate_storage_path(path: str) -> bool:
    """Return whether a path should be treated as a local filesystem path."""
    if not path:
        return False

    stripped = path.strip()
    if not stripped:
        return False

    if os.path.isabs(stripped):
        return True

    if stripped.startswith(("./", "../")):
        return True

    if stripped.startswith(("s3://", "http://", "https://")):
        return False

    if Path(stripped).exists() or Path(stripped).suffix.lower() in {
        ".arrow",
        ".csv",
        ".json",
        ".jsonl",
        ".parquet",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }:
        return True

    if "/" in stripped:
        first, second, *rest = stripped.split("/", 2)
        # Treat paths with a trailing file/path suffix as local filesystem paths.
        return bool(first and second and rest)

    return False


def _normalize_base_model_path(
    path: str,
    user_id: Optional[str],
    model_type: Optional[str] = None,
) -> str:
    """Normalize local base model paths to the API container view when possible."""
    if not path:
        return path

    stripped = path.strip()
    if not stripped:
        return stripped
    if not os.path.isabs(stripped) and not _should_validate_storage_path(stripped):
        return stripped

    resolved = _resolve_legacy_model_alias(stripped, user_id, model_type)
    if resolved == stripped and resolved.startswith("/models/"):
        resolved = resolved.replace("/models/", "/app/models/", 1)
    mapped_path, _ = map_storage_path(resolved)
    if _should_validate_storage_path(mapped_path):
        validate_storage_path(mapped_path, resource_type="base model")
        if Path(mapped_path).exists():
            mapped_path = str(Path(mapped_path).resolve())
    return mapped_path


def _normalize_storage_path(path: str, resource_type: str) -> str:
    """Normalize absolute storage paths to the API container view when possible."""
    if not path:
        return path

    stripped = path.strip()
    if not stripped:
        return stripped
    if not os.path.isabs(stripped) and not _should_validate_storage_path(stripped):
        return stripped

    mapped_path, _ = map_storage_path(stripped)
    if _should_validate_storage_path(mapped_path):
        validate_storage_path(mapped_path, resource_type=resource_type)
        if Path(mapped_path).exists():
            mapped_path = str(Path(mapped_path).resolve())
    return mapped_path


def _normalize_training_config_paths(training_config: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
    """Normalize training path fields before persisting or running a task."""
    normalized = dict(training_config)
    model_type = normalized.get("model_type")

    base_model_path = normalized.get("base_model_path")
    if isinstance(base_model_path, str) and base_model_path.strip():
        normalized_base_model_path = _normalize_base_model_path(
            base_model_path,
            user_id,
            model_type,
        )
        normalized["base_model_path"] = normalized_base_model_path

    loss_config = normalized.get("loss_config")
    if isinstance(loss_config, dict):
        guide_model = loss_config.get("guide_model")
        if isinstance(guide_model, str) and guide_model.strip():
            normalized_loss_config = dict(loss_config)
            normalized_loss_config["guide_model"] = _normalize_base_model_path(
                guide_model,
                user_id,
                "embedding",
            )
            normalized["loss_config"] = normalized_loss_config

    train_dataset_path = normalized.get("train_dataset_path")
    if isinstance(train_dataset_path, str) and train_dataset_path.strip():
        normalized["train_dataset_path"] = _normalize_storage_path(
            train_dataset_path,
            "train dataset",
        )

    sft_checkpoint_path = normalized.get("sft_checkpoint_path")
    if isinstance(sft_checkpoint_path, str) and sft_checkpoint_path.strip():
        normalized["sft_checkpoint_path"] = _normalize_storage_path(
            sft_checkpoint_path,
            "SFT checkpoint",
        )

    resume_from_checkpoint = normalized.get("resume_from_checkpoint")
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.strip():
        normalized["resume_from_checkpoint"] = _normalize_storage_path(
            resume_from_checkpoint,
            "resume checkpoint",
        )

    output_dir = normalized.get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        normalized["output_dir"] = _normalize_storage_path(
            output_dir,
            "output directory",
        )

    dataset_configs = normalized.get("dataset_configs")
    if isinstance(dataset_configs, list):
        normalized_dataset_configs = []
        for dataset_config in dataset_configs:
            if not isinstance(dataset_config, dict):
                normalized_dataset_configs.append(dataset_config)
                continue

            config_copy = dict(dataset_config)
            dataset_path = config_copy.get("path")
            if isinstance(dataset_path, str) and dataset_path.strip():
                config_copy["path"] = _normalize_storage_path(
                    dataset_path,
                    "dataset",
                )
            normalized_dataset_configs.append(config_copy)
        normalized["dataset_configs"] = normalized_dataset_configs

        train_datasets = [
            config for config in normalized_dataset_configs
            if isinstance(config, dict) and config.get("split") == "train"
        ]
        if train_datasets:
            normalized["train_dataset_path"] = train_datasets[0].get("path")
        elif normalized_dataset_configs and isinstance(normalized_dataset_configs[0], dict):
            normalized["train_dataset_path"] = normalized_dataset_configs[0].get("path")

    datasets = normalized.get("datasets")
    if isinstance(datasets, list):
        normalized_datasets = []
        for dataset in datasets:
            if not isinstance(dataset, dict):
                normalized_datasets.append(dataset)
                continue

            dataset_copy = dict(dataset)
            dataset_path = dataset_copy.get("path")
            if isinstance(dataset_path, str) and dataset_path.strip():
                dataset_copy["path"] = _normalize_storage_path(
                    dataset_path,
                    "dataset",
                )
            normalized_datasets.append(dataset_copy)
        normalized["datasets"] = normalized_datasets

    return normalized


def _resource_ownership_required(user_id: Optional[str]) -> bool:
    return bool(
        get_settings().auth_enabled
        and user_id
        and user_id != "anonymous"
    )


def _iter_sync_batches_for_provenance(
    task_id: str,
) -> Iterator[Dict[str, Any]]:
    """Yield tracked sync batches without loading an unbounded history at once."""
    from ...storage.services.external_sync_service import external_sync_service

    limit = 200
    offset = 0
    while True:
        batches, total = external_sync_service.list_batches(
            task_id=task_id,
            limit=limit,
            offset=offset,
        )
        if not batches:
            return
        yield from batches
        offset += len(batches)
        if not isinstance(total, int) or offset >= total:
            return


def _validate_owned_model_resource(
    model_path: str,
    user_id: str,
    *,
    display_name: str,
) -> None:
    """Require a tenant-owned registry record with managed model provenance."""
    if (
        model_path.startswith(("http://", "https://", "s3://"))
        or not _should_validate_storage_path(model_path)
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                f"{display_name}s must be downloaded and registered before training "
                "when authentication is enabled"
            ),
        )

    model = model_registry_service.get_model_by_path(
        model_path,
        user_id=user_id,
    )
    if not model or model.get("user_id") != user_id:
        raise HTTPException(
            status_code=403,
            detail=f"Not authorized to use this {display_name.lower()} path",
        )
    try:
        require_managed_model_provenance(
            model,
                model_path,
                user_id=user_id,
                models_dir=Path(get_settings().models_dir),
                training_output_dir=Path(get_settings().output_dir),
                training_task_lookup=training_task_service.get_task,
            )
    except ResourceProvenanceError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"{display_name} does not have verifiable API-managed provenance",
        ) from exc


def _validate_owned_training_resources(
    training_config: Dict[str, Any],
    user_id: Optional[str],
) -> None:
    """Require tenant-owned, API-managed provenance for local resource reads."""
    if not _resource_ownership_required(user_id):
        return

    from ...storage.services.external_sync_service import external_sync_service

    base_model_path = training_config.get("base_model_path")
    if isinstance(base_model_path, str) and base_model_path.strip():
        _validate_owned_model_resource(
            base_model_path,
            user_id,
            display_name="Base model",
        )

    loss_config = training_config.get("loss_config")
    if isinstance(loss_config, dict) and loss_config.get("guide_model") is not None:
        guide_model = loss_config["guide_model"]
        if not isinstance(guide_model, str) or not guide_model.strip():
            raise HTTPException(
                status_code=400,
                detail="Guide model path must be a non-empty string",
            )
        _validate_owned_model_resource(
            guide_model,
            user_id,
            display_name="Guide model",
        )

    dataset_paths = []
    dataset_configs = training_config.get("dataset_configs")
    if isinstance(dataset_configs, list):
        dataset_paths.extend(
            config.get("path")
            for config in dataset_configs
            if isinstance(config, dict) and config.get("path")
        )
    if not dataset_paths and training_config.get("train_dataset_path"):
        dataset_paths.append(training_config.get("train_dataset_path"))

    checked_paths = set()
    for dataset_path in dataset_paths:
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            continue
        if dataset_path in checked_paths:
            continue
        checked_paths.add(dataset_path)
        if (
            dataset_path.startswith(("http://", "https://"))
            or not (
                dataset_path.startswith("s3://")
                or _should_validate_storage_path(dataset_path)
            )
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Datasets must be uploaded, downloaded, or generated before "
                    "training when authentication is enabled"
                ),
            )
        dataset = dataset_service.get_dataset_by_storage_path(
            dataset_path,
            user_id=user_id,
        )
        if not dataset or dataset.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to use this dataset path",
            )
        if dataset.get("status") == "deleting":
            raise HTTPException(
                status_code=409,
                detail="Dataset deletion is in progress",
            )
        settings = get_settings()
        try:
            require_managed_dataset_provenance(
                dataset,
                dataset_path,
                user_id=user_id,
                datasets_dir=Path(settings.datasets_dir),
                s3_bucket=settings.minio_bucket,
                generation_output_dir=Path(
                    os.environ.get("GENERATION_OUTPUT_DIR", str(settings.datasets_dir))
                ),
                generation_task_lookup=generation_task_service.get_task,
                sync_data_dir=Path(os.environ.get("SYNC_DATA_DIR", "/app/data/sync")),
                sync_task_lookup=external_sync_service.get_task_raw,
                sync_training_lookup=external_sync_service.get_training_by_task_id,
                sync_batch_lookup=_iter_sync_batches_for_provenance,
                training_task_id=training_config.get("task_id"),
            )
        except ResourceProvenanceError as exc:
            raise HTTPException(
                status_code=403,
                detail="Dataset does not have verifiable API-managed provenance",
            ) from exc


def _validate_training_parent_checkpoint(
    parent_task_id: Optional[str],
    checkpoint_path: Optional[str],
    current_user: Dict[str, Any],
) -> None:
    """Bind a two-stage checkpoint to an owned parent task output directory."""
    if checkpoint_path and not parent_task_id:
        raise HTTPException(
            status_code=400,
            detail="sft_checkpoint_path requires parent_task_id",
        )
    if not parent_task_id:
        return

    parent_task = training_task_service.get_task(parent_task_id)
    verify_resource_ownership(parent_task, current_user, "Parent training task")
    if not checkpoint_path:
        return

    expected_root = Path(
        _resolve_server_managed_task_output(
            parent_task_id,
            parent_task,
            parent_task.get("output_dir"),
        )
    ).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    try:
        checkpoint.relative_to(expected_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="SFT checkpoint must belong to the parent training task",
        ) from exc


def _begin_training_dependency_guard(parent_task_id: Optional[str]):
    """Prevent parent artifact deletion while a dependent task is persisted."""
    if not parent_task_id:
        return None
    try:
        return background_task_admission_service.begin_deletion(
            "training",
            parent_task_id,
        )
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(
            status_code=409,
            detail="Parent training task is executing or being deleted",
        ) from exc


def _set_task_scoped_output(task_id: str, training_config: Dict[str, Any]) -> str:
    """Force API-created tasks to write only under output_dir/<task_id>."""
    output_dir = str(Path(get_settings().get_task_output_dir(task_id)).resolve())
    training_config["output_dir"] = output_dir
    return output_dir


def _resolve_server_managed_task_output(
    task_id: str,
    task: Dict[str, Any],
    configured_output: Optional[str] = None,
) -> str:
    """Accept only the standard task output or an authenticated sync output."""
    settings = get_settings()
    output_root = Path(settings.output_dir).resolve()
    candidate_value = configured_output or task.get("output_dir")
    if not candidate_value:
        candidate_value = str(settings.get_task_output_dir(task_id))
    candidate = Path(candidate_value).resolve()

    try:
        candidate.relative_to(output_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Training task output directory is outside the configured output root",
        ) from exc

    standard_output = Path(settings.get_task_output_dir(task_id)).resolve()
    if candidate == standard_output:
        return str(candidate)

    from ...storage.services.external_sync_service import external_sync_service

    sync_training = external_sync_service.get_training_by_task_id(task_id)
    if (
        not sync_training
        or sync_training.get("user_id") != task.get("user_id")
        or not sync_training.get("task_id")
        or not sync_training.get("target_id")
    ):
        raise HTTPException(
            status_code=400,
            detail="Training task output directory is not server-managed",
        )

    expected_sync_output = (
        output_root
        / "sync"
        / str(sync_training["task_id"])
        / str(sync_training["target_id"])
        / f"round_{int(sync_training.get('training_round') or 0)}"
    ).resolve()
    if candidate != expected_sync_output:
        raise HTTPException(
            status_code=400,
            detail="Sync training output directory does not match its tracking record",
        )
    return str(candidate)


def _resolve_resume_checkpoint(output_dir: str, checkpoint_path: str) -> str:
    """Resolve and bind a resume checkpoint to its task output directory."""
    resolved_output = Path(output_dir).resolve()
    resolved_checkpoint = Path(checkpoint_path).resolve()
    try:
        resolved_checkpoint.relative_to(resolved_output)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Checkpoint is outside the training task output directory",
        ) from exc
    if not resolved_checkpoint.is_dir():
        raise HTTPException(status_code=400, detail="Training checkpoint does not exist")
    return str(resolved_checkpoint)


def _device_to_cuda_visible(allocated_device: Optional[str]) -> Optional[str]:
    """Convert an allocated device string to CUDA_VISIBLE_DEVICES format."""
    if not allocated_device:
        return None
    if allocated_device.strip().lower() == "cpu":
        return ""

    cuda_ids = []
    for part in allocated_device.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            cuda_ids.append(part.split(":")[-1])
        else:
            cuda_ids.append(part)

    return ",".join(cuda_ids) if cuda_ids else None


def _training_gpu_lease_id(task_id: str, run_token: str) -> str:
    """Use an attempt-scoped GPU lease so an old worker cannot release a new run."""
    return f"training:{task_id}:{run_token}"


def _is_current_training_run(
    task: Optional[Dict[str, Any]],
    run_token: str,
    statuses: set[str],
) -> bool:
    return bool(
        task
        and task.get("run_token") == run_token
        and task.get("status") in statuses
    )


def _apply_llm_training_safety_defaults(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply conservative defaults for LLM training when legacy generic defaults are used."""
    config = dict(training_config)
    if config.get('model_type') != 'llm':
        return config

    if (
        config.get('per_device_train_batch_size') == 16
        and config.get('gradient_accumulation_steps') == 1
        and config.get('max_length') is None
        and not config.get('deepspeed')
    ):
        config['per_device_train_batch_size'] = 1
        config['gradient_accumulation_steps'] = 16
    if config.get('max_length') is None:
        config['max_length'] = 1024
    if config.get('gradient_checkpointing') is None:
        config['gradient_checkpointing'] = True

    return config


def _apply_max_length_to_nested_configs(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Propagate max_length into nested configs that consume it."""
    config = dict(training_config)
    configured_max_length = config.get('max_length')
    if configured_max_length is None:
        return config

    # SentenceTransformer uses max_seq_length rather than the generic API
    # field max_length. Keep an explicitly supplied canonical value intact for
    # callers that construct training configs outside this request model.
    if config.get('model_type') == 'embedding' and config.get('max_seq_length') is None:
        config['max_seq_length'] = configured_max_length

    # LLM trainers read top-level max_length directly. Avoid creating a
    # misleading loss_config for plain LLM SFT unless the caller provided one.
    if config.get('model_type') != 'llm' or config.get('loss_config') is not None:
        loss_config = dict(config.get('loss_config') or {})
        if loss_config.get('max_length') is None:
            loss_config['max_length'] = configured_max_length
        config['loss_config'] = loss_config

    if config.get('training_method') != 'sft':
        rl_config = dict(config.get('rl_config') or {})
        if rl_config.get('max_length') is None:
            rl_config['max_length'] = configured_max_length
        config['rl_config'] = rl_config

    return config


def _run_training_task_worker(
    task_id: str,
    training_config: Dict[str, Any],
    cuda_visible: Optional[str],
):
    """Execute the actual training work in an isolated child process."""
    run_token = training_config.get("_run_token")
    if not isinstance(run_token, str) or not run_token:
        logger.error("Training task %s has no run token; child start refused", task_id)
        return

    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        if cuda_visible:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        logger.info(f"Task {task_id} child process set CUDA_VISIBLE_DEVICES={cuda_visible}")

    try:
        task = training_task_service.get_task(task_id)
        if not _is_current_training_run(
            task,
            run_token,
            {
                TrainingStatus.PREPARING.value,
                TrainingStatus.RUNNING.value,
            },
        ):
            logger.info(
                "Training task %s was stopped before the child worker started",
                task_id,
            )
            return

        _validate_training_resource_limits(training_config)

        # The API can die immediately after spawn, before its PID commit.
        # The child must establish its own durable identity before touching a
        # model/GPU; startup recovery fences its no-process snapshot against
        # this same registration. A lost/failed registration never trains.
        child_pid = os.getpid()
        if not training_task_service.register_training_process(
            task_id,
            child_pid,
            capture_process_create_time(child_pid),
            run_token=run_token,
        ):
            logger.info("Training task %s child registration was rejected", task_id)
            return

        # Progress callback to update task progress
        def progress_callback(progress: float):
            try:
                training_task_service.update_task_progress(
                    task_id,
                    progress,
                    run_token=run_token,
                )
            except Exception as e:
                logger.warning(f"Failed to update progress for task {task_id}: {e}")

        # Initial progress
        progress_callback(0.0)

        # Run training with progress callback
        result = train_with_config(training_config, progress_callback=progress_callback)

        # Final progress
        progress_callback(100.0)

        # Atomically complete task with status and result
        completed = training_task_service.complete_task(
            task_id,
            status=TrainingStatus.SUCCEEDED.value,
            final_model_path=result.save_dir,
            final_metrics=result.final_metrics,
            run_token=run_token,
        )
        if not completed:
            logger.info(
                "Discarding completed training result for inactive task %s",
                task_id,
            )
            return

    except Exception as e:
        logger.error(f"Training task {task_id} failed: {e}")
        try:
            training_task_service.update_task_status(
                task_id,
                TrainingStatus.FAILED.value,
                str(e),
                run_token=run_token,
            )
        except ValueError:
            logger.info(
                "Training task %s became inactive while child failure was handled",
                task_id,
            )


def _start_training_process(process: multiprocessing.Process, task_id: str, cuda_visible: Optional[str]) -> None:
    """Start a training child process with CUDA_VISIBLE_DEVICES set before spawn imports."""
    with _PROCESS_START_ENV_LOCK:
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
            logger.info(f"Task {task_id} starting child process with CUDA_VISIBLE_DEVICES={cuda_visible}")
            process.start()
        finally:
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]


def _cleanup_training_process(
    process: Optional[multiprocessing.Process],
    task_id: str,
) -> bool:
    """Terminate a started subprocess and report whether it is confirmed exited."""
    if process is None:
        return True

    try:
        if getattr(process, "_popen", None) is None:
            return True
        if not process.is_alive():
            return True

        logger.warning(
            "Terminating training subprocess for task %s before GPU release (pid=%s)",
            task_id,
            process.pid,
        )
        process.terminate()
        process.join(timeout=30)
        if process.is_alive():
            logger.warning(
                "Training subprocess for task %s did not exit after terminate(); forcing kill",
                task_id,
            )
            if hasattr(process, "kill"):
                process.kill()
                process.join(timeout=10)
        if process.is_alive():
            logger.error(
                "Training subprocess for task %s is still alive after cleanup",
                task_id,
            )
            return False
        return True
    except Exception as cleanup_error:
        logger.warning(
            "Failed to clean up training subprocess for task %s: %s",
            task_id,
            cleanup_error,
        )
        return False


def _monitor_training_process(
    process: multiprocessing.Process,
    task_id: str,
    run_token: str,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Wait for one attempt while reacting to persisted task state changes."""
    while process.is_alive():
        task = training_task_service.get_task(task_id)
        if task is None:
            logger.info(
                "Training task %s disappeared while process %s was running",
                task_id,
                process.pid,
            )
            confirmed_exited = _cleanup_training_process(process, task_id)
            return None, confirmed_exited
        if task.get("run_token") != run_token:
            logger.info(
                "Training task %s moved to a new attempt while process %s was running",
                task_id,
                process.pid,
            )
            confirmed_exited = _cleanup_training_process(process, task_id)
            return task, confirmed_exited

        status = task.get("status")
        if status in _STOPPED_TRAINING_STATUSES:
            logger.info(
                "Training task %s reached %s; cleaning process %s",
                task_id,
                status,
                process.pid,
            )
            confirmed_exited = _cleanup_training_process(process, task_id)
            return task, confirmed_exited
        if status in _COMPLETED_TRAINING_STATUSES:
            process.join(timeout=_TRAINING_PROCESS_TERMINAL_DRAIN_SECONDS)
            confirmed_exited = not process.is_alive()
            if process.is_alive():
                logger.warning(
                    "Training process %s remained alive after task %s reached %s",
                    process.pid,
                    task_id,
                    status,
                )
                confirmed_exited = _cleanup_training_process(process, task_id)
            return task, confirmed_exited
        if status not in _ACTIVE_TRAINING_STATUSES:
            logger.warning(
                "Training task %s has unexpected status %s; cleaning process %s",
                task_id,
                status,
                process.pid,
            )
            confirmed_exited = _cleanup_training_process(process, task_id)
            return task, confirmed_exited

        process.join(timeout=_TRAINING_PROCESS_MONITOR_INTERVAL_SECONDS)

    return training_task_service.get_task(task_id), True


def _recover_sync_training_failure(task_id: str, reason: str) -> None:
    """Best-effort, idempotent compensation for sync-triggered training."""
    try:
        from ...sync.post_training_handler import on_training_failed

        on_training_failed(task_id, reason)
    except Exception:
        logger.exception(
            "Failed to reconcile sync training claim for task %s",
            task_id,
        )


def _training_attempt_is_settled(
    task: Optional[Dict[str, Any]],
    run_token: str,
) -> bool:
    """Return whether this attempt no longer owns an active database row."""
    return (
        task is None
        or task.get("run_token") != run_token
        or task.get("status") in _TERMINAL_TRAINING_STATUSES
    )


def _persist_training_failure(
    task_id: str,
    run_token: str,
    reason: str,
) -> bool:
    """Persist FAILED with bounded retries or prove another terminal winner."""
    last_error: Optional[Exception] = None
    for attempt in range(1, _TRAINING_FAILURE_PERSIST_ATTEMPTS + 1):
        try:
            if training_task_service.update_task_status(
                task_id,
                TrainingStatus.FAILED.value,
                reason,
                run_token=run_token,
            ):
                return True
        except Exception as status_error:
            last_error = status_error
            logger.warning(
                "Failed to persist FAILED status for training task %s "
                "(attempt %s/%s): %s",
                task_id,
                attempt,
                _TRAINING_FAILURE_PERSIST_ATTEMPTS,
                status_error,
            )

        try:
            task = training_task_service.get_task(task_id)
        except Exception as state_error:
            if last_error is None:
                last_error = state_error
            logger.warning(
                "Failed to verify terminal state for training task %s: %s",
                task_id,
                state_error,
            )
            continue

        if _training_attempt_is_settled(task, run_token):
            logger.info(
                "Training task %s reached another terminal or inactive winner",
                task_id,
            )
            return False

    if last_error is None:
        last_error = RuntimeError(
            "FAILED status update was rejected while the attempt remained active"
        )
    raise _TrainingFailurePersistenceError(
        f"Could not persist FAILED status for current training attempt {task_id}"
    ) from last_error


# === Background Training Function ===

def run_training_task(task_id: str, training_config: Dict[str, Any]):
    """Run training task in background."""
    process = None
    allocated_device = None
    invocation_claimed = False
    gpu_lease_owned = False
    process_info_owned = False
    process_create_time: Optional[float] = None
    process_exit_confirmed: Optional[bool] = None
    failure_reason: Optional[str] = None
    terminal_persistence_error: Optional[_TrainingFailurePersistenceError] = None
    training_config = dict(training_config)
    run_token = training_config.get("_run_token")
    if not isinstance(run_token, str) or not run_token:
        run_token = str(uuid4())
        training_config["_run_token"] = run_token
    gpu_lease_id = _training_gpu_lease_id(task_id, run_token)
    try:
        if not training_task_service.claim_preparing(task_id, run_token):
            logger.info(
                "Training task %s is no longer pending; background start skipped",
                task_id,
            )
            return
        invocation_claimed = True
        training_config = canonicalize_tuner_config(
            training_config,
            allow_registered_custom=True,
        )
        training_config = _apply_llm_training_safety_defaults(training_config)
        training_config = _apply_max_length_to_nested_configs(training_config)
        _validate_training_resource_limits(training_config)
        training_config = _normalize_training_config_paths(
            training_config,
            training_config.get("user_id"),
        )
        _validate_owned_training_resources(
            training_config,
            training_config.get("user_id"),
        )
        _validate_training_parent_checkpoint(
            training_config.get("parent_task_id"),
            training_config.get("sft_checkpoint_path"),
            {"user_id": training_config.get("user_id")},
        )
        task = training_task_service.get_task(task_id)
        if not _is_current_training_run(
            task,
            run_token,
            {TrainingStatus.PREPARING.value},
        ):
            logger.info("Training task %s stopped during validation", task_id)
            return
        resume_checkpoint = training_config.get("resume_from_checkpoint")
        if resume_checkpoint:
            task = training_task_service.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Training task not found")
            if task.get("user_id") != training_config.get("user_id"):
                raise HTTPException(
                    status_code=403,
                    detail="Training task owner changed before execution",
                )
            output_dir = _resolve_server_managed_task_output(
                task_id,
                task,
                training_config.get("output_dir"),
            )
            training_config["output_dir"] = output_dir
            training_config["resume_from_checkpoint"] = _resolve_resume_checkpoint(
                output_dir,
                resume_checkpoint,
            )

        # Allocate GPU resources before starting
        gpu_ids = training_config.get('gpu_ids')
        if gpu_ids:
            # User specified GPU IDs - convert to device request format
            device_request = ",".join([f"cuda:{gpu_id}" for gpu_id in gpu_ids])
        else:
            # Auto-allocate single GPU
            device_request = "auto"

        allocated_device = gpu_resource_manager.allocate_gpus_for_task(
            gpu_lease_id,
            device_request,
        )
        if (
            allocated_device is None
            and not gpu_ids
            and get_settings().training_allow_cpu_fallback
        ):
            logger.warning(
                "No GPU available for task %s; operator-enabled CPU fallback is active",
                task_id,
            )
            allocated_device = gpu_resource_manager.allocate_gpus_for_task(
                gpu_lease_id,
                "cpu",
            )

        if allocated_device is None:
            error_msg = f"GPU allocation failed: requested {device_request}, no available GPUs or invalid GPU IDs"
            logger.error(error_msg)
            failure_reason = error_msg
            _persist_training_failure(task_id, run_token, error_msg)
            return
        gpu_lease_owned = True

        task = training_task_service.get_task(task_id)
        if not _is_current_training_run(
            task,
            run_token,
            {TrainingStatus.PREPARING.value},
        ):
            logger.info("Training task %s stopped during resource allocation", task_id)
            return

        # Set device in training config
        training_config['device'] = allocated_device
        logger.info(f"Task {task_id} allocated device: {allocated_device}")

        if not training_task_service.update_task_execution_config(
            task_id,
            model_path=training_config.get("base_model_path"),
            train_dataset_path=training_config.get("train_dataset_path"),
            training_params=training_config,
            sft_checkpoint_path=training_config.get("sft_checkpoint_path"),
            output_dir=training_config.get("output_dir"),
            device=allocated_device,
            run_token=run_token,
        ):
            logger.info(
                "Training task %s became inactive before config persistence",
                task_id,
            )
            return

        cuda_visible = _device_to_cuda_visible(allocated_device)

        ctx = multiprocessing.get_context("spawn")
        process = ctx.Process(
            target=_run_training_task_worker,
            args=(task_id, training_config, cuda_visible),
            name=f"training-task-{task_id[:8]}",
        )
        task = training_task_service.get_task(task_id)
        if not _is_current_training_run(
            task,
            run_token,
            {TrainingStatus.PREPARING.value},
        ):
            logger.info("Training task %s stopped before process start", task_id)
            return
        _start_training_process(process, task_id, cuda_visible)
        process_create_time = capture_process_create_time(process.pid)
        # The child may have registered (or even finished) before this parent
        # reaches the commit. Cleanup is fenced to this exact child identity,
        # including when the parent's registration loses to a terminal state.
        process_info_owned = True
        process_info_persisted = training_task_service.register_training_process(
            task_id,
            process_pid=process.pid,
            process_create_time=process_create_time,
            run_token=run_token,
        )
        if not process_info_persisted:
            logger.info(
                "Training task %s became inactive before process info persistence",
                task_id,
            )
            process_exit_confirmed = _cleanup_training_process(process, task_id)
            return
        try:
            running_persisted = training_task_service.update_task_status(
                task_id,
                TrainingStatus.RUNNING.value,
                run_token=run_token,
            )
        except ValueError:
            running_persisted = False
        if not running_persisted:
            logger.info(
                "Training task %s changed state before its running transition",
                task_id,
            )
        task, process_exit_confirmed = _monitor_training_process(
            process,
            task_id,
            run_token,
        )
        if not process_exit_confirmed:
            logger.warning(
                "Training subprocess %s exit could not be confirmed; retaining its lease",
                process.pid,
            )

        if _is_current_training_run(
            task,
            run_token,
            _ACTIVE_TRAINING_STATUSES,
        ):
            error_msg = f"Training subprocess exited without finalizing task status (exit_code={process.exitcode})"
            logger.error(error_msg)
            failure_reason = error_msg
            _persist_training_failure(task_id, run_token, error_msg)

    except _TrainingFailurePersistenceError as persistence_error:
        terminal_persistence_error = persistence_error
    except Exception as e:
        logger.error(f"Training task {task_id} failed: {e}")
        failure_reason = str(e)
        if not invocation_claimed:
            logger.info(
                "Training task %s was not claimed by this invocation",
                task_id,
            )
        else:
            try:
                _persist_training_failure(task_id, run_token, failure_reason)
            except _TrainingFailurePersistenceError as persistence_error:
                terminal_persistence_error = persistence_error
    finally:
        if process_exit_confirmed is None:
            process_exit_confirmed = _cleanup_training_process(process, task_id)

        cleanup_attempt_allowed = bool(
            process_exit_confirmed and terminal_persistence_error is None
        )
        if (
            cleanup_attempt_allowed
            and not process_info_owned
            and process is not None
            and process.pid is not None
        ):
            # A self-registered child can exit before the parent's OS identity
            # lookup. Only after this owned Process is confirmed exited may we
            # adopt its durable identity for fenced cleanup of this attempt.
            try:
                registered = training_task_service.get_task(task_id)
                if (
                    registered
                    and registered.get("run_token") == run_token
                    and registered.get("process_pid") == process.pid
                    and registered.get("process_create_time") is not None
                ):
                    process_create_time = registered["process_create_time"]
                    process_info_owned = True
            except Exception as identity_error:
                logger.warning(
                    "Could not reconcile exited child identity for task %s: %s",
                    task_id,
                    identity_error,
                )
        if process_info_owned and cleanup_attempt_allowed:
            try:
                cleared = training_task_service.update_process_info(
                    task_id,
                    process_pid=None,
                    process_status=None,
                    process_create_time=None,
                    run_token=run_token,
                    expected_process_pid=process.pid,
                    expected_process_create_time=process_create_time,
                )
                if not cleared:
                    logger.warning(
                        "Training process info cleanup was not persisted for task %s",
                        task_id,
                    )
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to clear training process info for task %s: %s",
                    task_id,
                    cleanup_error,
                )

        if gpu_lease_owned and cleanup_attempt_allowed:
            try:
                released = gpu_resource_manager.release_gpus_for_task(gpu_lease_id)
                if released:
                    logger.info(f"Released GPU resources for task {task_id}")
                else:
                    logger.warning(
                        "GPU resources were not released for training task %s",
                        task_id,
                    )
            except Exception as cleanup_error:
                logger.error(f"Failed to release GPU resources for task {task_id}: {cleanup_error}")

        if (
            invocation_claimed
            and process_exit_confirmed
            and terminal_persistence_error is None
        ):
            try:
                post_cleanup_task = training_task_service.get_task(task_id)
            except Exception as state_error:
                logger.warning(
                    "Failed to read terminal training task %s after cleanup: %s",
                    task_id,
                    state_error,
                )
            else:
                if _is_current_training_run(
                    post_cleanup_task,
                    run_token,
                    {TrainingStatus.SUCCEEDED.value},
                ):
                    _finalize_successful_training(task_id, run_token)
                elif _is_current_training_run(
                    post_cleanup_task,
                    run_token,
                    {TrainingStatus.FAILED.value},
                ):
                    _recover_sync_training_failure(
                        task_id,
                        post_cleanup_task.get("error_message")
                        or failure_reason
                        or "Training task failed",
                    )

    if terminal_persistence_error is not None:
        raise terminal_persistence_error


def _finalize_successful_training(task_id: str, run_token: str) -> bool:
    """Register and reconcile a current successful attempt after resource release."""
    task = training_task_service.get_task(task_id)
    if not _is_current_training_run(
        task,
        run_token,
        {TrainingStatus.SUCCEEDED.value},
    ):
        logger.info(
            "Skipping post-training callback for inactive attempt of task %s",
            task_id,
        )
        return False

    final_model_path = task.get("final_model_path")
    if not final_model_path:
        logger.warning(
            "Skipping post-training callback for task %s without a final model path",
            task_id,
        )
        return False

    model_registry_id = None
    try:
        model_name = task.get('task_name') or f"trained-model-{task_id[:8]}"
        registered_model = model_registry_service.register_from_task(
            task_id=task_id,
            model_name=model_name,
            description=f"Auto-registered from training task: {task_id}",
            user_id=task.get("user_id"),
        )
        if registered_model:
            model_registry_id = registered_model.get("model_id")
            logger.info(
                "Model auto-registered: %s from task %s",
                registered_model["model_id"],
                task_id,
            )
        else:
            logger.warning("Failed to auto-register model from task %s", task_id)
    except Exception as reg_error:
        logger.warning("Auto-register model failed for task %s: %s", task_id, reg_error)

    try:
        from ...sync.post_training_handler import on_training_completed

        on_training_completed(
            training_task_id=task_id,
            final_model_path=final_model_path,
            model_registry_id=model_registry_id,
        )
    except Exception as sync_err:
        logger.warning("[sync] Post-training callback failed for task %s: %s", task_id, sync_err)
    return True


def _fail_unscheduled_training(task_id: str, run_token: str) -> None:
    """Persist and compensate a worker that could not be enqueued."""
    reason = "Task could not be scheduled for background execution"
    failure_persisted = _persist_training_failure(task_id, run_token, reason)
    if not failure_persisted:
        task = training_task_service.get_task(task_id)
        failure_persisted = _is_current_training_run(
            task,
            run_token,
            {TrainingStatus.FAILED.value},
        )
    if failure_persisted:
        _recover_sync_training_failure(task_id, reason)


# === API Endpoints ===

@router.post("/train", response_model=TrainingResponse)
async def create_training_task(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Create a new training task.

    The task will be executed in the background.

    Supports:
    - Encoder models: embedding, reranker (CrossEncoder) - SFT only
    - Decoder models: decoder_reranker (Qwen3-Reranker) - SFT/DPO/GRPO/DAPO/DR_GRPO

    Pass Idempotency-Key header to prevent duplicate task creation on retries.
    """
    # Get user_id from authenticated user
    user_id = current_user["user_id"]

    # Check idempotency
    is_duplicate, cached_response = check_idempotency(
        idempotency_key, user_id, "/api/train"
    )
    if is_duplicate and cached_response:
        return TrainingResponse(**cached_response)

    # Validate dataset configuration
    if not request.datasets:
        raise HTTPException(
            status_code=400,
            detail="'datasets' must be provided"
        )

    model_type = request.model_type
    training_method = request.training_method

    # Validate model_type
    valid_model_types = ['embedding', 'reranker', 'decoder_reranker', 'llm']
    if model_type not in valid_model_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {model_type}. Must be one of {valid_model_types}"
        )

    # Validate training_method
    valid_training_methods = ['sft', 'cpt', 'dpo', 'grpo', 'dapo', 'dr_grpo', 'kto', 'orpo', 'ppo', 'two_stage']
    if training_method not in valid_training_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid training_method: {training_method}. Must be one of {valid_training_methods}"
        )

    # Validate combination: encoder models only support SFT
    if model_type in ['embedding', 'reranker'] and training_method != 'sft':
        raise HTTPException(
            status_code=400,
            detail=f"Model type '{model_type}' only supports 'sft' training method"
        )
    if model_type == 'decoder_reranker':
        allowed_decoder_methods = ['sft', 'grpo', 'dapo', 'dr_grpo', 'dpo']
        if training_method not in allowed_decoder_methods:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model type 'decoder_reranker' only supports {allowed_decoder_methods} training methods"
                )
            )
    if model_type == 'llm':
        allowed_llm_methods = ['sft', 'dpo', 'orpo']
        if training_method not in allowed_llm_methods:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model type 'llm' only supports {allowed_llm_methods} training methods"
                )
            )

    # Auto-detect model architecture
    model_architecture = request.model_architecture
    if not model_architecture:
        model_architecture = 'encoder' if model_type in ['embedding', 'reranker'] else 'decoder'

    # Build training config
    training_config = request.model_dump(exclude_none=True)
    training_config['model_type'] = model_type
    training_config['training_method'] = training_method
    training_config['model_architecture'] = model_architecture
    training_config['user_id'] = user_id

    training_config = _apply_llm_training_safety_defaults(training_config)
    training_config = _apply_max_length_to_nested_configs(training_config)

    # Reject unsupported tuners and make tuner_type the single LoRA truth.
    try:
        training_config = canonicalize_tuner_config(training_config)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    use_lora = training_config['use_lora']

    # Handle LoRA config
    if use_lora:
        training_config['lora_config'].update({
            'use_lora': True,
            'r': request.lora_r,
            'lora_alpha': request.lora_alpha,
            'lora_dropout': request.lora_dropout,
        })

    # Handle dataset configuration - convert to dataset_configs format
    training_config['dataset_configs'] = [
        {'path': ds.path, 'max_samples': ds.max_samples, 'split': ds.split}
        for ds in request.datasets
    ]

    train_datasets = [ds for ds in request.datasets if ds.split == 'train']
    train_dataset_path = (train_datasets[0].path if train_datasets else request.datasets[0].path)
    training_config['train_dataset_path'] = train_dataset_path
    # Output locations are server-managed and become known only after task creation.
    training_config.pop('output_dir', None)
    training_config = _normalize_training_config_paths(training_config, user_id)
    _validate_owned_training_resources(training_config, user_id)

    # Bind the queued background invocation to this exact execution attempt.
    run_token = str(uuid4())
    parent_guard = _begin_training_dependency_guard(request.parent_task_id)

    try:
        _validate_training_parent_checkpoint(
            request.parent_task_id,
            training_config.get('sft_checkpoint_path'),
            current_user,
        )
        task_info, execution_lease = background_task_admission_service.admit_execution(
            "training",
            None,
            user_id,
            training_task_service.create_task,
            task_name=request.task_name,
            model_path=training_config['base_model_path'],
            train_dataset_path=training_config['train_dataset_path'],
            training_params=training_config,
            description=request.description,
            user_id=user_id,
            model_type=model_type,
            training_method=training_method,
            model_architecture=model_architecture,
            rl_config=training_config.get('rl_config'),
            loss_config=training_config.get('loss_config'),
            parent_task_id=request.parent_task_id,
            sft_checkpoint_path=training_config.get('sft_checkpoint_path'),
            output_dir=None,
            run_token=run_token,
            require_managed_datasets=bool(
                get_settings().auth_enabled
                and user_id
                and user_id != "anonymous"
            ),
        )
    except BackgroundTaskCapacityExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatasetConsumptionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        if parent_guard is not None:
            parent_guard.release()

    try:
        training_config['task_id'] = task_info['task_id']
        output_dir = _set_task_scoped_output(task_info['task_id'], training_config)
        training_task_service.update_task_output_dir(
            task_info['task_id'],
            output_dir,
            training_config,
        )
        training_config["_run_token"] = run_token

        logger.info(f"Created task {task_info['task_id']} with output_dir: {output_dir}")

        response = TrainingResponse(
            task_id=task_info['task_id'],
            task_name=task_info['task_name'],
            status="pending",
            message="Training task created successfully"
        )

        background_tasks.add_task(
            background_task_admission_service.run_sync,
            execution_lease,
            run_training_task,
            task_info['task_id'],
            training_config,
        )
        store_idempotency_response(
            idempotency_key, user_id, "/api/train",
            response.model_dump()
        )
    except Exception:
        execution_lease.release()
        _fail_unscheduled_training(task_info['task_id'], run_token)
        raise

    return response


def _normalize_gpu_ids(raw) -> Optional[List[int]]:
    """Coerce gpu_ids stored as string (e.g. '0' or '0,1') to List[int]."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, str):
        return [int(x) for x in raw.split(",") if x.strip().isdigit()]
    return None


def _enrich_dataset_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize dataset_configs for response without file-system scans."""
    result = []
    for cfg in configs:
        cfg = dict(cfg)
        num_rows = cfg.get('num_rows')
        if isinstance(num_rows, str) and num_rows.isdigit():
            cfg['num_rows'] = int(num_rows)
        result.append(cfg)
    return result


def _build_task_response(task: Dict[str, Any]) -> TaskStatusResponse:
    """Build TaskStatusResponse from task dict with extracted training params."""
    # Extract key params from training_params for convenience
    training_params = task.get('training_params') or {}
    learning_rate = training_params.get('learning_rate')
    num_train_epochs = training_params.get('num_train_epochs')
    per_device_train_batch_size = training_params.get('per_device_train_batch_size')

    # Extract runtime metrics for real-time progress display
    runtime_metrics = training_params.get('_runtime_metrics') or {}
    current_step = runtime_metrics.get('current_step')
    total_steps = runtime_metrics.get('total_steps')
    current_epoch = runtime_metrics.get('current_epoch')
    total_epochs = runtime_metrics.get('total_epochs') or num_train_epochs
    train_loss = runtime_metrics.get('train_loss')
    eval_loss = runtime_metrics.get('eval_loss')

    # Fallback to training_params if direct fields are None
    base_model_path = task.get('base_model_path') or training_params.get('base_model_path') or ""
    dataset_path = task.get('train_dataset_path') or training_params.get('train_dataset_path') or ""
    output_dir = task.get('output_dir') or training_params.get('output_dir')

    is_lora = bool(task.get('is_lora') or training_params.get('use_lora', False))
    loss_config = training_params.get('loss_config') or task.get('loss_config')
    rl_config = training_params.get('rl_config') or task.get('rl_config')

    # Enrich dataset_configs with num_rows if missing
    dataset_configs = training_params.get('dataset_configs')
    if dataset_configs:
        dataset_configs = _enrich_dataset_configs(dataset_configs)

    return TaskStatusResponse(
        task_id=task['task_id'],
        task_name=task['task_name'],
        model_type=task.get('model_type'),
        training_method=task.get('training_method'),
        model_architecture=task.get('model_architecture'),
        base_model_path=base_model_path,
        user_id=task['user_id'],
        status=task['status'],
        progress=task['progress'],
        error_message=public_task_error_message(task.get('error_message')),
        final_model_path=task.get('final_model_path'),
        train_dataset_path=dataset_path,
        dataset_configs=dataset_configs,
        output_dir=output_dir,
        training_params=training_params if training_params else None,
        is_lora=is_lora,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        current_step=current_step,
        total_steps=total_steps,
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        train_loss=train_loss,
        eval_loss=eval_loss,
        gpu_ids=_normalize_gpu_ids(training_params.get('gpu_ids')),
        rl_config=rl_config,
        loss_config=loss_config,
        parent_task_id=task.get('parent_task_id'),
        sft_checkpoint_path=task.get('sft_checkpoint_path'),
        final_metrics=task.get('final_metrics'),
        trained_model_registry_id=task.get('trained_model_registry_id'),
        created_at=task['created_at'].isoformat() if task.get('created_at') else None,
        started_at=task['started_at'].isoformat() if task.get('started_at') else None,
        completed_at=task['completed_at'].isoformat() if task.get('completed_at') else None,
        updated_at=task['updated_at'].isoformat() if task.get('updated_at') else None,
    )


@router.get("/train/{task_id}", response_model=TaskStatusResponse)
@threadpool_endpoint
def get_task_status(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get status of a training task."""
    task = training_task_service.get_task(task_id)
    # Verify ownership - returns task or raises 403/404
    task = verify_resource_ownership(task, current_user, "Training task")

    return _build_task_response(task)


@router.get("/train/{task_id}/events", response_model=TrainingTaskEventListResponse)
@threadpool_endpoint
def list_task_events(
    task_id: str,
    limit: int = Query(default=200, ge=1, le=500),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List recent events for a training task."""
    task = training_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Training task")

    events = training_task_event_service.list_events(task_id, limit=limit)
    return TrainingTaskEventListResponse(
        events=[
            TrainingTaskEventResponse(
                event_id=e["event_id"],
                task_id=e["task_id"],
                user_id=e.get("user_id"),
                event_type=e["event_type"],
                payload=sanitize_public_diagnostics(e.get("payload")),
                created_at=e["created_at"].isoformat() if e.get("created_at") else None,
            )
            for e in events
        ],
        total=len(events),
    )


@router.get("/train", response_model=TaskListResponse)
@threadpool_endpoint
def list_tasks(
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    training_method: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all training tasks for the current user."""
    # Use authenticated user_id for filtering
    user_id = current_user["user_id"]

    tasks, total = training_task_service.get_all_tasks(
        status=status,
        model_type=model_type,
        training_method=training_method,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )

    task_responses = [_build_task_response(task) for task in tasks]

    # Get global statistics (not affected by filters)
    stats_dict = training_task_service.get_task_stats(user_id=user_id)
    stats = TaskStatsResponse(**stats_dict)

    return TaskListResponse(tasks=task_responses, total=total, stats=stats)


@router.post("/train/{task_id}/stop")
async def stop_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Stop a running training task."""
    task = training_task_service.get_task(task_id)
    # Verify ownership
    task = verify_resource_ownership(task, current_user, "Training task")

    active_statuses = {
        TrainingStatus.PENDING.value,
        TrainingStatus.PREPARING.value,
        TrainingStatus.RUNNING.value,
        TrainingStatus.EVALUATING.value,
    }
    if task['status'] not in active_statuses:
        raise HTTPException(status_code=400, detail=f"Task cannot be stopped, current status: {task['status']}")

    if not training_task_service.stop_if_active(task_id):
        latest = training_task_service.get_task(task_id)
        current_status = latest.get("status") if latest else "missing"
        raise HTTPException(
            status_code=400,
            detail=f"Task cannot be stopped, current status: {current_status}",
        )

    _recover_sync_training_failure(task_id, "Training task stopped by user")

    # Re-read after the state transition so a PID persisted concurrently with
    # this request is not missed. The parent worker performs the matching
    # post-start check when the stop wins before PID persistence.
    latest = training_task_service.get_task(task_id) or task
    process_pid = latest.get("process_pid")
    if process_pid is not None and not terminate_process_if_matches(
        process_pid,
        latest.get("process_create_time"),
        wait_for_exit=False,
    ):
        logger.warning(
            "Refused to signal unverified training process %s for task %s",
            process_pid,
            task_id,
        )

    return {"message": f"Task {task_id} stopped"}


@router.post("/train/{task_id}/resume")
async def resume_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Resume a failed or stopped training task from the latest checkpoint."""
    task = training_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Training task")

    if task["status"] not in {TrainingStatus.FAILED.value, TrainingStatus.STOPPED.value}:
        raise HTTPException(
            status_code=400,
            detail=f"只能续传失败或已停止的任务，当前状态: {task['status']}",
        )

    if any(
        task.get(field) is not None
        for field in ("process_pid", "process_status", "process_create_time")
    ):
        raise HTTPException(
            status_code=409,
            detail="The previous training process is still being cleaned up",
        )

    training_config = dict(task.get("training_params") or {})
    training_config.setdefault("base_model_path", task.get("base_model_path"))
    training_config.setdefault("train_dataset_path", task.get("train_dataset_path"))
    training_config["task_id"] = task_id
    training_config["user_id"] = task.get("user_id")
    training_config = _apply_llm_training_safety_defaults(training_config)
    training_config = _apply_max_length_to_nested_configs(training_config)
    try:
        training_config = canonicalize_tuner_config(
            training_config,
            allow_registered_custom=True,
        )
        _validate_training_resource_limits(training_config)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Stored training configuration is invalid: {exc}",
        ) from exc
    training_config = _normalize_training_config_paths(
        training_config,
        task.get("user_id"),
    )
    _validate_owned_training_resources(training_config, task.get("user_id"))
    _validate_training_parent_checkpoint(
        task.get("parent_task_id"),
        training_config.get("sft_checkpoint_path"),
        current_user,
    )
    output_dir = _resolve_server_managed_task_output(
        task_id,
        task,
        training_config.get("output_dir"),
    )
    training_config["output_dir"] = output_dir

    # Only checkpoints inside this task's server-managed output may be resumed.
    checkpoint_path = training_task_service.find_latest_checkpoint(task_id)
    if not checkpoint_path:
        raise HTTPException(status_code=400, detail="未找到可用的 checkpoint，无法续传")
    checkpoint_path = _resolve_resume_checkpoint(output_dir, checkpoint_path)

    training_config["resume_from_checkpoint"] = checkpoint_path
    run_token = str(uuid4())
    try:
        claimed, execution_lease = background_task_admission_service.admit_execution(
            "training",
            task_id,
            task.get("user_id"),
            training_task_service.reset_for_resume,
            task_id,
            run_token,
            require_managed_datasets=bool(
                get_settings().auth_enabled
                and task.get("user_id")
                and task.get("user_id") != "anonymous"
            ),
        )
    except BackgroundTaskCapacityExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not claimed:
        raise HTTPException(
            status_code=409,
            detail=(
                "Training task was already resumed or its previous process is "
                "still being cleaned up"
            ),
        )
    try:
        training_task_service.update_task_output_dir(
            task_id,
            output_dir,
            training_config,
        )
        training_config["_run_token"] = run_token
        background_tasks.add_task(
            background_task_admission_service.run_sync,
            execution_lease,
            run_training_task,
            task_id,
            training_config,
        )
    except Exception:
        execution_lease.release()
        _fail_unscheduled_training(task_id, run_token)
        raise
    return {
        "message": "任务已从 checkpoint 续传",
        "task_id": task_id,
        "checkpoint": checkpoint_path,
    }


def _safe_delete_path(path: str, task: Dict[str, Any]):
    """Delete only an output directory proven to belong to the task."""
    import shutil

    if not path:
        return

    task_id = task["task_id"]
    try:
        target = Path(
            _resolve_server_managed_task_output(task_id, task, path)
        )
    except (HTTPException, OSError, ValueError) as exc:
        logger.warning(
            "Refused to delete unverified output for task %s (%s)",
            task_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=400,
            detail="Training output path is not a verified managed path",
        ) from exc

    try:
        if target.is_file():
            os.remove(target)
            logger.info(f"Deleted file: {target}")
        elif target.is_dir():
            shutil.rmtree(target)
            logger.info(f"Deleted directory: {target}")
    except OSError:
        logger.exception(
            "Failed to delete managed output for training task %s",
            task_id,
        )
        raise


def _find_registered_model_for_training_task(
    task: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Find current or legacy registry records that retain task artifacts."""
    owner_user_id = task.get("user_id")
    registry_id = task.get("trained_model_registry_id")
    if registry_id:
        model = model_registry_service.get_model(str(registry_id))
        if model and model.get("user_id") == owner_user_id:
            return model
    model = model_registry_service.get_model_by_source_task(
        str(task["task_id"]),
        user_id=owner_user_id,
    )
    if model and model.get("user_id") == owner_user_id:
        return model
    return None


def _get_training_artifact_dependencies(
    task: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Return persisted consumers that still require training output artifacts."""
    registered_model = _find_registered_model_for_training_task(task)
    registered_models = [registered_model] if registered_model else []
    if not registered_models:
        artifact_paths = [
            task.get("output_dir"),
            task.get("final_model_path"),
        ]
        registered_models = (
            model_registry_service.list_models_referencing_artifact_paths(
                artifact_paths,
                user_id=task.get("user_id"),
            )
            if any(artifact_paths)
            else []
        )
    return {
        "registered_models": registered_models,
        "active_adapters": adapter_service.list_active_training_output_consumers(
            str(task["task_id"]),
            task.get("final_model_path"),
            user_id=task.get("user_id"),
        ),
        "dependent_training_tasks": training_task_service.list_artifact_consumers(
            str(task["task_id"]),
            task.get("output_dir"),
        ),
    }


@router.delete("/train/{task_id}")
async def delete_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a training task and its output directory."""
    task = training_task_service.get_task(task_id)
    # Verify ownership
    task = verify_resource_ownership(task, current_user, "Training task")

    active_statuses = {
        TrainingStatus.PENDING.value,
        TrainingStatus.PREPARING.value,
        TrainingStatus.RUNNING.value,
        TrainingStatus.EVALUATING.value,
    }
    if (
        task['status'] in active_statuses
        or task.get("process_pid") is not None
        or task.get("process_status") is not None
        or task.get("process_create_time") is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete an active training task or one still being cleaned up",
        )

    try:
        deletion_guard = background_task_admission_service.begin_deletion(
            "training",
            task_id,
        )
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(
            status_code=409,
            detail="Training worker is still shutting down; retry deletion shortly",
        ) from exc

    try:
        # 清理训练输出目录
        dependencies = _get_training_artifact_dependencies(task)
        if dependencies["registered_models"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot delete a training task while a registered model "
                    "still references its artifacts"
                ),
            )

        if dependencies["active_adapters"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot delete a training task while an active adapter "
                    "still references its artifacts; unload the adapter first"
                ),
            )

        if dependencies["dependent_training_tasks"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot delete a training task while another training task "
                    "still references its checkpoints; delete the dependent task first"
                ),
            )

        output_dir = task.get("output_dir")
        if output_dir:
            _safe_delete_path(output_dir, task)

        if not training_task_service.delete_task(task_id):
            raise RuntimeError("Training task disappeared during deletion")

        return {"message": f"Task {task_id} deleted"}
    finally:
        deletion_guard.release()


@router.get("/train/{task_id}/metrics", response_model=TrainingMetricsResponse)
@threadpool_endpoint
def get_training_metrics(
    task_id: str,
    limit: Optional[int] = Query(
        default=DEFAULT_METRICS_HISTORY_RECORDS,
        ge=1,
        le=MAX_METRICS_HISTORY_RECORDS,
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get training metrics for a task including loss history for charts.

    Args:
        task_id: Training task ID
        limit: Bounded number of latest loss history records to return

    Returns:
        TrainingMetricsResponse with loss_history, summary, and current_metrics
    """
    from ...utils.strict_json import sanitize_json_value

    resolved_limit = (
        DEFAULT_METRICS_HISTORY_RECORDS if limit is None else limit
    )

    task = training_task_service.get_task(task_id)
    # Verify ownership
    task = verify_resource_ownership(task, current_user, "Training task")

    # Resolve persisted paths before any filesystem access. This preserves the
    # standard task output and authenticated sync outputs while rejecting
    # legacy or tampered paths.
    training_params = task.get('training_params') or {}
    configured_output = task.get('output_dir') or training_params.get('output_dir')
    output_dir = _resolve_server_managed_task_output(
        task_id,
        task,
        configured_output,
    )

    # Log output_dir sources for debugging
    logger.debug(f"Task {task_id} output_dir: task.output_dir={task.get('output_dir')}, params.output_dir={training_params.get('output_dir')}")

    loss_history = []
    summary = None
    has_data = False

    if output_dir:
        # Build metrics file paths
        logs_dir = Path(output_dir) / "logs" / "training" / task_id
        loss_history_file = logs_dir / "loss_history.jsonl"
        training_metrics_file = logs_dir / "training_metrics.json"

        # Log path info for debugging
        logger.debug(f"Looking for metrics at: {loss_history_file} (exists: {loss_history_file.exists()})")

        # Read loss history from JSONL
        if loss_history_file.exists():
            try:
                loss_history = _read_metrics_jsonl_tail(
                    loss_history_file,
                    resolved_limit,
                )
                has_data = True
            except Exception as e:
                logger.warning(f"Failed to read loss history for task {task_id}: {e}")

        # Read summary from training_metrics.json
        if training_metrics_file.exists():
            try:
                summary = _read_metrics_summary(training_metrics_file)
                has_data = True
            except Exception as e:
                logger.warning(f"Failed to read training metrics for task {task_id}: {e}")

    # Get real-time metrics from database
    current_metrics = sanitize_json_value(
        training_task_service.get_task_metrics(task_id)
    )
    if current_metrics:
        has_data = True

    return TrainingMetricsResponse(
        task_id=task_id,
        loss_history=loss_history,
        summary=summary,
        current_metrics=current_metrics,
        has_data=has_data,
    )


def _read_metrics_jsonl_tail(path: Path, limit: int) -> List[Dict[str, Any]]:
    """Read a bounded number of complete JSONL records from the file tail."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > MAX_METRICS_HISTORY_RECORDS
    ):
        raise ValueError(
            f"limit must be between 1 and {MAX_METRICS_HISTORY_RECORDS}"
        )

    from ...utils.strict_json import sanitize_json_value

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        if position == 0:
            return []
        handle.seek(position - 1)
        ends_with_newline = handle.read(1) in {b"\n", b"\r"}
        carry = b""
        newest_first: List[bytes] = []
        bytes_read = 0
        target_candidates = limit + (0 if ends_with_newline else 1)

        while (
            position > 0
            and bytes_read < MAX_METRICS_HISTORY_BYTES
            and len(newest_first) < target_candidates
        ):
            chunk_size = min(
                _METRICS_TAIL_CHUNK_BYTES,
                position,
                MAX_METRICS_HISTORY_BYTES - bytes_read,
            )
            position -= chunk_size
            handle.seek(position)
            chunk = handle.read(chunk_size)
            bytes_read += len(chunk)
            parts = (chunk + carry).split(b"\n")
            carry = parts[0]
            newest_first.extend(
                line.rstrip(b"\r")
                for line in reversed(parts[1:])
                if line.strip()
            )

        if position == 0 and carry.strip():
            newest_first.append(carry.rstrip(b"\r"))

    parsed_newest_first: List[Dict[str, Any]] = []
    for index, line in enumerate(newest_first):
        try:
            record = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == 0 and not ends_with_newline:
                continue
            raise
        sanitized_record = sanitize_json_value(record)
        if not isinstance(sanitized_record, dict):
            raise ValueError("Training metrics JSONL records must be JSON objects")
        parsed_newest_first.append(sanitized_record)
        if len(parsed_newest_first) == limit:
            break
    return list(reversed(parsed_newest_first))


def _read_metrics_summary(path: Path) -> Dict[str, Any]:
    """Read one metrics summary without allowing an unbounded JSON payload."""
    from ...utils.strict_json import sanitize_json_value

    with path.open("rb") as handle:
        payload = handle.read(MAX_METRICS_SUMMARY_BYTES + 1)
    if len(payload) > MAX_METRICS_SUMMARY_BYTES:
        raise ValueError("Training metrics summary exceeds the size limit")
    value = sanitize_json_value(json.loads(payload.decode("utf-8")))
    if not isinstance(value, dict):
        raise ValueError("Training metrics summary must be a JSON object")
    return value
