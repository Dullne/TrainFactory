"""
Deep Evaluation API routes.

Provides endpoints for deep evaluation of embedding/reranker models.
"""

import asyncio
import logging
from typing import Annotated, Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...auth.dependencies import (
    get_current_user,
    requires_tenant_provenance,
    verify_resource_ownership,
)
from ...auth.resource_provenance import ResourceProvenanceError
from ...core.ssrf import SSRFError
from ...storage.services.inference_authorization_service import authorize_inference_config
from ...deep_evaluation import (
    DeepEvaluator,
    EvaluationSample,
    MetricRegistry,
    create_llm_judge_from_dict,
    run_deep_evaluation_task,
    cancel_deep_evaluation,
    clear_deep_cancellation,
)
from ...deep_evaluation.deep_evaluation_runner import (
    MAX_DEEP_EVALUATION_COMBINATIONS,
    MAX_DEEP_EVALUATION_DATASETS,
    MAX_DEEP_EVALUATION_LLM_CONCURRENCY,
    MAX_DEEP_EVALUATION_LLM_TOKENS,
    MAX_DEEP_EVALUATION_METRICS,
    MAX_DEEP_EVALUATION_MODEL_CONCURRENCY,
    MAX_DEEP_EVALUATION_MODEL_GROUPS,
    MAX_DEEP_EVALUATION_MODEL_WORKERS,
    MAX_DEEP_EVALUATION_RETRIES,
    MAX_DEEP_EVALUATION_SAMPLES,
    MAX_DEEP_EVALUATION_TIMEOUT,
    MAX_DEEP_EVALUATION_TOP_K,
    validate_deep_evaluation_resource_config,
)
from ...storage.entities.evaluation_task_entity import EvaluationStatus
from ...storage.services.dataset_service import (
    DatasetConsumptionUnavailableError,
    dataset_service,
)
from ...storage.services.deep_evaluation_task_service import deep_evaluation_task_service
from ...storage.services.model_config_service import model_config_service
from ...storage.services.milvus_collection_service import (
    MilvusCollectionUnavailableError,
    milvus_collection_service,
)
from ...storage.services.runtime_dependency_service import (
    RuntimeDependencyUnavailableError,
)
from ...storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    BackgroundTaskCapacityExceeded,
    background_task_admission_service,
)
from ...storage.services.outbound_endpoint_policy import (
    validate_user_outbound_url,
)
from ...evaluation.dataset_access import (
    UnsupportedEvaluationDatasetStorageError,
    resolve_local_evaluation_dataset_record,
    resolve_managed_local_evaluation_dataset,
)
from ...utils.public_diagnostics import (
    public_task_error_message,
    sanitize_public_diagnostics,
)

DeepEvaluationStatus = EvaluationStatus

logger = logging.getLogger(__name__)

router = APIRouter()


def _fail_unscheduled_deep_evaluation(task_id: str) -> None:
    try:
        deep_evaluation_task_service.update_status(
            task_id,
            DeepEvaluationStatus.FAILED,
            "Task could not be scheduled for background execution",
        )
    except Exception:
        logger.exception(
            "Failed to finalize unscheduled deep evaluation task %s",
            task_id,
        )


def _run_claimed_deep_evaluation_task(
    task_id: str,
    *,
    existing_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Claim a pending task and recheck cancellation before entering the runner."""
    clear_cancellation = False
    claimed = False
    runner_started = False
    run_token = str(uuid4())
    try:
        if not deep_evaluation_task_service.claim_running(task_id, run_token=run_token):
            task = deep_evaluation_task_service.get_task(task_id)
            clear_cancellation = bool(
                task and task.get("status") == DeepEvaluationStatus.CANCELLED
            )
            logger.info(
                "Deep evaluation worker did not claim task %s; current status is %s",
                task_id,
                task.get("status") if task else "missing",
            )
            return

        claimed = True
        clear_cancellation = True
        task = deep_evaluation_task_service.get_task(task_id)
        if not task or task.get("status") != DeepEvaluationStatus.RUNNING:
            logger.info(
                "Deep evaluation task %s changed state before runner registration",
                task_id,
            )
            return
        runner_started = True
        run_deep_evaluation_task(
            task_id, existing_results=existing_results, expected_run_token=run_token,
        )
    except Exception:
        if claimed and not runner_started:
            try:
                deep_evaluation_task_service.fail_claimed_startup(task_id, run_token)
            except Exception:
                logger.exception("Failed to finalize deep evaluation startup for %s", task_id)
        raise
    finally:
        if clear_cancellation:
            clear_deep_cancellation(task_id)


# ===== Constants =====

# Traditional information retrieval metrics (no LLM required)
TRADITIONAL_METRICS = {
    "mrr": {
        "description": "Mean Reciprocal Rank - 平均倒数排名",
        "category": "retrieval",
        "requires_llm": False,
    },
    "map": {
        "description": "Mean Average Precision - 平均精度均值",
        "category": "retrieval",
        "requires_llm": False,
    },
    "ndcg@10": {
        "description": "Normalized DCG at 10 - 归一化折损累积增益",
        "category": "retrieval",
        "requires_llm": False,
    },
    "recall@10": {
        "description": "Recall at 10 - 前10召回率",
        "category": "retrieval",
        "requires_llm": False,
    },
    "precision@10": {
        "description": "Precision at 10 - 前10精确率",
        "category": "retrieval",
        "requires_llm": False,
    },
}

# Default metrics for deep evaluation (traditional IR metrics only)
DEFAULT_TRADITIONAL_METRICS = ["mrr", "ndcg@10"]


# ===== Helpers =====

def _resolve_model_config(config_id: str, current_user: Dict[str, Any]) -> Dict[str, Any]:
    config = model_config_service.get_config(config_id)
    config = verify_resource_ownership(config, current_user, "Model config")
    resolved = dict(config)
    if resolved.get("api_endpoint"):
        resolved["api_endpoint"] = validate_user_outbound_url(
            resolved["api_endpoint"],
            current_user.get("user_id"),
        )
    return resolved


def _detect_collection_model_mismatch(
    collection: Optional[Dict[str, Any]],
    retrieval_embedding: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Detect whether retrieval embedding model mismatches collection embedding model."""
    if not collection or not retrieval_embedding:
        return None

    collection_name = collection.get("collection_name", "unknown")
    collection_config_id = str(collection.get("embedding_config_id") or "").strip()
    retrieval_config_id = str(retrieval_embedding.get("config_id") or "").strip()

    if collection_config_id and retrieval_config_id and collection_config_id != retrieval_config_id:
        return (
            f"Collection '{collection_name}' was indexed with embedding config "
            f"'{collection_config_id}', but retrieval uses '{retrieval_config_id}'."
        )

    collection_model = str(collection.get("embedding_model") or "").strip().lower()
    retrieval_model = str(
        retrieval_embedding.get("model_name") or retrieval_embedding.get("model") or ""
    ).strip().lower()
    if collection_model and retrieval_model and collection_model != retrieval_model:
        return (
            f"Collection '{collection_name}' was indexed with model "
            f"'{collection_model}', but retrieval uses '{retrieval_model}'."
        )

    return None


def _resolve_deep_evaluation_dataset_configs(
    dataset_configs: List[Any],
    current_user: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Resolve deep-evaluation datasets from server-side records."""
    resolved_configs: List[Dict[str, Any]] = []
    require_provenance = requires_tenant_provenance(current_user)
    user_id = current_user.get("user_id")

    for dataset_config in dataset_configs:
        dataset = dataset_service.get_dataset(dataset_config.dataset_id)
        dataset = verify_resource_ownership(dataset, current_user, "Dataset")
        try:
            if require_provenance:
                dataset = resolve_managed_local_evaluation_dataset(
                    dataset_config.dataset_id,
                    user_id=user_id,
                )
            else:
                dataset = resolve_local_evaluation_dataset_record(
                    dataset_config.dataset_id
                )
        except UnsupportedEvaluationDatasetStorageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ResourceProvenanceError as exc:
            raise HTTPException(
                status_code=403,
                detail="Dataset does not have verifiable API-managed provenance",
            ) from exc

        resolved_configs.append(
            {
                "dataset_id": dataset["dataset_id"],
                "dataset_name": (
                    dataset.get("dataset_name")
                    or dataset.get("display_name")
                    or dataset["dataset_id"]
                ),
            }
        )
    return resolved_configs


def _revalidate_deep_evaluation_collection(
    task: Dict[str, Any],
    current_user: Dict[str, Any],
) -> None:
    """Recheck the current owner of an online task's persisted collection."""
    worker_groups = task.get("worker_groups") or {}
    if worker_groups.get("retrieval_mode") != "online":
        return
    collection_name = worker_groups.get("milvus_collection")
    collection = (
        milvus_collection_service.get_by_name(collection_name)
        if collection_name
        else None
    )
    collection = verify_resource_ownership(
        collection,
        current_user,
        "Milvus collection",
    )
    if collection.get("status") != "active":
        raise HTTPException(
            status_code=409,
            detail="Milvus collection is unavailable because it is being deleted",
        )




# ===== Request/Response Models =====

class ModelConfigModel(BaseModel):
    """模型配置（单个模型）"""
    config_id: Optional[str] = Field(default=None, max_length=36, description="模型配置 ID（从已注册配置加载）")
    endpoint: Optional[str] = Field(default=None, max_length=2048, description="模型 API 端点")
    model_name: Optional[str] = Field(default=None, max_length=256, description="模型名称")
    api_key: Optional[str] = Field(default=None, max_length=512, description="API 密钥")
    inference_framework: Optional[str] = Field(default=None, max_length=32, description="推理框架: vllm | sglang | xinference")
    concurrency: int = Field(
        default=8,
        ge=1,
        le=MAX_DEEP_EVALUATION_MODEL_CONCURRENCY,
        description="该模型的请求并发数",
    )


class LLMConfigModel(BaseModel):
    """LLM 配置"""
    config_id: Optional[str] = Field(default=None, max_length=36, description="模型配置 ID")
    endpoint: Optional[str] = Field(default=None, max_length=2048, description="LLM API 端点")
    model: Optional[str] = Field(default=None, max_length=256, description="模型名称")
    api_key: Optional[str] = Field(default=None, max_length=512, description="API 密钥")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p 采样")
    top_k: Optional[int] = Field(default=None, ge=0, le=MAX_DEEP_EVALUATION_TOP_K, description="Top-k 采样")
    max_tokens: int = Field(default=2048, ge=1, le=MAX_DEEP_EVALUATION_LLM_TOKENS, description="最大 token 数")
    timeout: int = Field(default=60, ge=1, le=MAX_DEEP_EVALUATION_TIMEOUT, description="超时时间（秒）")
    max_retries: int = Field(default=3, ge=0, le=MAX_DEEP_EVALUATION_RETRIES, description="最大重试次数")
    concurrency: int = Field(
        default=2,
        ge=1,
        le=MAX_DEEP_EVALUATION_LLM_CONCURRENCY,
        description="LLM 请求并发数",
    )


def _resolve_llm_config(config: LLMConfigModel, current_user: Dict[str, Any]) -> Dict[str, Any]:
    """解析 LLM 配置，支持从配置 ID 加载或直接使用参数"""
    if config.config_id:
        resolved = _resolve_model_config(config.config_id, current_user)
        if (resolved.get("model_type") or "").lower() not in {"llm", "chat", "gen"}:
            raise HTTPException(status_code=400, detail="请选择 LLM 类型的模型配置")
        return {
            "endpoint": resolved.get("api_endpoint"),
            "model": resolved.get("model_name"),
            "api_key": resolved.get("api_key"),
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "concurrency": config.concurrency,
        }

    if not config.endpoint or not config.model:
        raise HTTPException(status_code=400, detail="LLM 配置缺少 endpoint 或 model")

    return {
        "endpoint": validate_user_outbound_url(
            config.endpoint,
            current_user.get("user_id"),
        ),
        "model": config.model,
        "api_key": config.api_key,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "concurrency": config.concurrency,
    }


class ModelGroupConfigModel(BaseModel):
    """模型组配置 - 一个组包含 embedding/rerank/llm 三类模型（各可选）"""
    group_name: str = Field(..., min_length=1, max_length=255, description="组名称，用于标识和结果展示")
    embedding: Optional[ModelConfigModel] = Field(default=None, description="Embedding 模型配置")
    rerank: Optional[ModelConfigModel] = Field(default=None, description="Rerank 模型配置")
    llm: Optional[LLMConfigModel] = Field(default=None, description="LLM 模型配置")


class DatasetConfigModel(BaseModel):
    """数据集配置"""
    dataset_id: str = Field(..., min_length=1, max_length=36, description="数据集 ID")


class FieldMappingModel(BaseModel):
    """字段映射配置"""
    query: str = Field(default="query", description="查询字段名")
    positives: Optional[str] = Field(default="positives", description="正例字段名")
    negatives: Optional[str] = Field(default="negatives", description="负例字段名")
    input: Optional[str] = Field(default="input", description="输入字段名")
    expected_output: Optional[str] = Field(default="expected_output", description="期望输出字段名")
    actual_output: Optional[str] = Field(default="actual_output", description="实际输出字段名")
    retrieval_context: Optional[str] = Field(default="retrieval_context", description="检索上下文字段名")


class CreateDeepEvaluationTaskRequest(BaseModel):
    """创建深度评估任务请求"""

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    task_name: Optional[str] = Field(default=None, max_length=255, description="任务名称")
    description: Optional[str] = Field(default=None, max_length=4096, description="任务描述")
    model_configs: List[ModelGroupConfigModel] = Field(
        ...,
        min_length=1,
        max_length=MAX_DEEP_EVALUATION_MODEL_GROUPS,
        description="模型组配置列表",
    )
    dataset_configs: List[DatasetConfigModel] = Field(
        ...,
        min_length=1,
        max_length=MAX_DEEP_EVALUATION_DATASETS,
        description="数据集配置列表",
    )
    max_samples: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_DEEP_EVALUATION_SAMPLES,
        description="采样数量（空表示使用系统上限）",
    )
    field_mapping: Optional[FieldMappingModel] = Field(default=None, description="字段映射")
    metrics: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        max_length=MAX_DEEP_EVALUATION_METRICS,
        description="评估指标",
    )
    model_workers: int = Field(
        default=2,
        ge=1,
        le=MAX_DEEP_EVALUATION_MODEL_WORKERS,
        description="模型组并发数（同时评估几个组）",
    )
    chunk_eval_mode: Literal["batch", "individual"] = Field(
        default="batch",
        description="Chunk 评估模式: batch（一次调用）或 individual（逐个调用）",
    )

    # 在线检索模式
    retrieval_mode: Literal["offline", "online"] = Field(default="offline", description="评估模式: offline（使用数据集中的 positives/negatives）或 online（从 Milvus 在线检索）")
    milvus_collection: Optional[str] = Field(default=None, description="Milvus 集合名称（online 模式必填）")
    retrieval_embedding_config: Optional[ModelConfigModel] = Field(default=None, description="用于在线检索的 Embedding 模型配置（online 模式必填）")
    retrieval_top_k: int = Field(default=20, ge=1, le=100, description="在线检索返回的候选数量")
    allow_collection_model_mismatch: bool = Field(
        default=False,
        description="允许检索模型与集合入库模型不一致（仅用于参考，不作为质量结论）",
    )

    @model_validator(mode="after")
    def validate_combination_count(self):
        if (
            len(self.model_configs) * len(self.dataset_configs)
            > MAX_DEEP_EVALUATION_COMBINATIONS
        ):
            raise ValueError(
                "Deep evaluation model/dataset combinations exceed the configured limit"
            )
        return self


class CreateDeepEvaluationTaskResponse(BaseModel):
    """创建深度评估任务响应"""
    task_id: str
    message: str = "Deep evaluation task created"


class DeepEvaluationTaskResponse(BaseModel):
    """深度评估任务响应"""

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    task_id: str
    task_name: Optional[str] = None
    description: Optional[str] = None
    eval_type: str
    model_configs: List[Dict[str, Any]] = Field(default_factory=list)
    dataset_configs: List[Dict[str, Any]] = Field(default_factory=list)
    max_samples: Optional[int] = None
    field_mapping: Optional[Dict[str, Any]] = None
    metrics: List[str]
    worker_groups: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None
    status: str
    progress: float
    total_samples: int
    processed_samples: int
    model_progress: Optional[Dict[str, Any]] = None
    results_summary: Optional[Dict[str, Any]] = None
    results_path: Optional[str] = None
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class DeepEvaluationTaskListResponse(BaseModel):
    """深度评估任务列表响应"""
    items: List[DeepEvaluationTaskResponse]
    total: int


def _public_deep_evaluation_task(task: Dict[str, Any]) -> Dict[str, Any]:
    public_task = dict(task)
    public_task["model_progress"] = sanitize_public_diagnostics(
        public_task.get("model_progress")
    )
    public_task["results_summary"] = sanitize_public_diagnostics(
        public_task.get("results_summary")
    )
    public_task["results_path"] = None
    public_task["error_message"] = public_task_error_message(
        public_task.get("error_message")
    )
    return public_task


# ===== API Endpoints =====

class MetricInfoResponse(BaseModel):
    """指标信息响应"""
    name: str
    description: str
    category: str
    requires_llm: bool
    requires_expected_output: bool = False
    requires_actual_output: bool = False
    requires_retrieval_context: bool = False


@router.get("/eval-types")
async def list_eval_types():
    """获取支持的深度评估类型"""
    metrics = list(TRADITIONAL_METRICS.keys())
    return {
        "rerank": {
            "description": "Reranker 模型评估",
            "metrics": metrics,
            "default_metrics": DEFAULT_TRADITIONAL_METRICS,
        },
        "embedding": {
            "description": "Embedding 模型评估",
            "metrics": metrics,
            "default_metrics": DEFAULT_TRADITIONAL_METRICS,
        },
    }


@router.get("/metrics", response_model=List[MetricInfoResponse])
async def list_metrics(
    category: Optional[str] = None,
):
    """获取可用的评估指标"""
    if category in {"traditional", "retrieval"}:
        return [
            MetricInfoResponse(
                name=name,
                description=info["description"],
                category=info["category"],
                requires_llm=info["requires_llm"],
            )
            for name, info in TRADITIONAL_METRICS.items()
        ]

    metrics = MetricRegistry.list_metrics()
    return [
        MetricInfoResponse(
            name=metric["name"],
            description=metric["description"],
            category=metric.get("category", "llm"),
            requires_llm=metric.get("requires_llm", True),
            requires_expected_output=metric.get("requires_expected_output", False),
            requires_actual_output=metric.get("requires_actual_output", False),
            requires_retrieval_context=metric.get("requires_retrieval_context", False),
        )
        for metric in metrics
    ]


def _build_worker_groups(
    request: CreateDeepEvaluationTaskRequest,
    resolved_retrieval_emb: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """构建 worker_groups 配置，包含在线检索配置"""
    wg: Dict[str, Any] = {
        "model_workers": request.model_workers,
        "chunk_eval_mode": request.chunk_eval_mode,
        "retrieval_mode": request.retrieval_mode,
    }
    if request.retrieval_mode == "online" and resolved_retrieval_emb:
        wg["milvus_collection"] = request.milvus_collection
        wg["retrieval_embedding_config"] = resolved_retrieval_emb
        wg["retrieval_top_k"] = request.retrieval_top_k
    return wg


# ===== Deep Evaluation Task Endpoints =====

@router.get("/datasets/eval-ready")
async def list_eval_ready_datasets(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取可用于在线评估的集合列表（从集合注册表查询）"""
    user_id = current_user.get("user_id")
    if user_id == "anonymous":
        user_id = None

    collections, _ = milvus_collection_service.list_collections(
        user_id=user_id,
        status="active",
        limit=100,
    )

    eval_ready = []
    for coll in collections:
        linked = milvus_collection_service.get_linked_datasets(coll["collection_name"])
        eval_ready.append({
            "collection_name": coll["collection_name"],
            "display_name": coll.get("display_name"),
            "collection_id": coll["collection_id"],
            "embedding_config_id": coll.get("embedding_config_id"),
            "embedding_model": coll.get("embedding_model"),
            "embedding_endpoint": coll.get("embedding_endpoint"),
            "dim": coll.get("dim"),
            "linked_datasets": [
                {
                    "dataset_id": linked_dataset["dataset_id"],
                    "dataset_name": linked_dataset.get("dataset_name"),
                    "chunk_count": linked_dataset.get("chunk_count", 0),
                }
                for linked_dataset in linked
            ],
            # Backward compat fields
            "milvus_collection": coll["collection_name"],
            "embedding_config": {
                "config_id": coll.get("embedding_config_id"),
                "endpoint": coll.get("embedding_endpoint"),
                "model": coll.get("embedding_model"),
            } if coll.get("embedding_config_id") else None,
        })
    return eval_ready


@router.post("/tasks", response_model=CreateDeepEvaluationTaskResponse)
async def create_deep_evaluation_task(
    request: CreateDeepEvaluationTaskRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """创建深度评估任务（支持按组配置模型）"""
    if not request.dataset_configs:
        raise HTTPException(status_code=400, detail="数据集配置不能为空")
    if not request.model_configs:
        raise HTTPException(status_code=400, detail="模型组配置不能为空")

    # 组名唯一，否则评估结果按 group_name 聚合时会静默互相覆盖
    group_names = [g.group_name for g in request.model_configs]
    if len(group_names) != len(set(group_names)):
        raise HTTPException(
            status_code=400,
            detail="模型组名称必须唯一（评估结果按组名聚合）",
        )

    llm_metric_info = {m["name"]: m for m in MetricRegistry.list_metrics()}

    # Validate or use default metrics
    metrics = request.metrics or DEFAULT_TRADITIONAL_METRICS
    for metric in metrics:
        if metric not in TRADITIONAL_METRICS and metric not in llm_metric_info:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的指标: '{metric}'"
            )

    has_retrieval_metrics = any(metric in TRADITIONAL_METRICS for metric in metrics)
    requires_llm = any(
        llm_metric_info[metric].get("requires_llm")
        for metric in metrics
        if metric in llm_metric_info
    )

    # 验证数据集
    dataset_configs = _resolve_deep_evaluation_dataset_configs(
        request.dataset_configs,
        current_user,
    )

    # 解析按组的模型配置
    resolved_model_groups: List[Dict[str, Any]] = []
    has_embedding = False
    has_rerank = False
    has_llm_model = False

    for group_cfg in request.model_configs:
        resolved_group: Dict[str, Any] = {
            "group_name": group_cfg.group_name,
        }

        # 解析 Embedding 模型
        if group_cfg.embedding:
            embedding_cfg = group_cfg.embedding
            if embedding_cfg.config_id:
                config = _resolve_model_config(embedding_cfg.config_id, current_user)
                resolved_group["embedding"] = {
                    "config_id": config.get("config_id"),
                    "config_name": config.get("config_name"),
                    "endpoint": config.get("api_endpoint"),
                    "model_name": config.get("model_name"),
                    "api_key": config.get("api_key"),
                    "inference_framework": config.get("inference_framework"),
                    "concurrency": embedding_cfg.concurrency,
                }
            elif embedding_cfg.endpoint and embedding_cfg.model_name:
                resolved_group["embedding"] = {
                    "endpoint": validate_user_outbound_url(
                        embedding_cfg.endpoint,
                        current_user.get("user_id"),
                    ),
                    "model_name": embedding_cfg.model_name,
                    "api_key": embedding_cfg.api_key,
                    "inference_framework": embedding_cfg.inference_framework,
                    "concurrency": embedding_cfg.concurrency,
                }
            has_embedding = True

        # 解析 Rerank 模型
        if group_cfg.rerank:
            rerank_cfg = group_cfg.rerank
            if rerank_cfg.config_id:
                config = _resolve_model_config(rerank_cfg.config_id, current_user)
                resolved_group["rerank"] = {
                    "config_id": config.get("config_id"),
                    "config_name": config.get("config_name"),
                    "endpoint": config.get("api_endpoint"),
                    "model_name": config.get("model_name"),
                    "api_key": config.get("api_key"),
                    "inference_framework": config.get("inference_framework"),
                    "concurrency": rerank_cfg.concurrency,
                }
            elif rerank_cfg.endpoint and rerank_cfg.model_name:
                resolved_group["rerank"] = {
                    "endpoint": validate_user_outbound_url(
                        rerank_cfg.endpoint,
                        current_user.get("user_id"),
                    ),
                    "model_name": rerank_cfg.model_name,
                    "api_key": rerank_cfg.api_key,
                    "inference_framework": rerank_cfg.inference_framework,
                    "concurrency": rerank_cfg.concurrency,
                }
            has_rerank = True

        # 解析 LLM 模型
        if group_cfg.llm:
            llm_cfg = group_cfg.llm
            if llm_cfg.config_id:
                config = _resolve_model_config(llm_cfg.config_id, current_user)
                resolved_group["llm"] = {
                    "config_id": config.get("config_id"),
                    "config_name": config.get("config_name"),
                    "endpoint": config.get("api_endpoint"),
                    "model": config.get("model_name"),
                    "api_key": config.get("api_key"),
                    "temperature": llm_cfg.temperature,
                    "top_p": llm_cfg.top_p,
                    "top_k": llm_cfg.top_k,
                    "max_tokens": llm_cfg.max_tokens,
                    "timeout": llm_cfg.timeout,
                    "max_retries": llm_cfg.max_retries,
                    "concurrency": llm_cfg.concurrency,
                }
            elif llm_cfg.endpoint and llm_cfg.model:
                resolved_group["llm"] = {
                    "endpoint": validate_user_outbound_url(
                        llm_cfg.endpoint,
                        current_user.get("user_id"),
                    ),
                    "model": llm_cfg.model,
                    "api_key": llm_cfg.api_key,
                    "temperature": llm_cfg.temperature,
                    "top_p": llm_cfg.top_p,
                    "top_k": llm_cfg.top_k,
                    "max_tokens": llm_cfg.max_tokens,
                    "timeout": llm_cfg.timeout,
                    "max_retries": llm_cfg.max_retries,
                    "concurrency": llm_cfg.concurrency,
                }
            has_llm_model = True

        # 验证组至少有一个模型
        if not any([resolved_group.get("embedding"), resolved_group.get("rerank"), resolved_group.get("llm")]):
            raise HTTPException(
                status_code=400,
                detail=f"模型组 '{group_cfg.group_name}' 至少需要配置一个模型"
            )

        for model_config in (
            resolved_group.get("embedding"), resolved_group.get("rerank"),
            resolved_group.get("llm"),
        ):
            if model_config:
                await asyncio.to_thread(
                    authorize_inference_config, model_config, current_user.get("user_id"),
                    current_user=current_user,
                )
        resolved_model_groups.append(resolved_group)

    # 验证指标和模型配置的一致性
    if has_retrieval_metrics:
        if not (has_embedding or has_rerank):
            raise HTTPException(status_code=400, detail="检索指标评估需要至少一个 Embedding 或 Rerank 模型")
        if any(not (g.get("embedding") or g.get("rerank")) for g in resolved_model_groups):
            raise HTTPException(status_code=400, detail="检索指标评估要求每个模型组至少配置 Embedding 或 Rerank")
    if requires_llm:
        if not has_llm_model:
            raise HTTPException(status_code=400, detail="所选 LLM 指标需要配置 LLM 模型")
        if any(not g.get("llm") for g in resolved_model_groups):
            raise HTTPException(status_code=400, detail="所选 LLM 指标要求每个模型组配置 LLM")

    # 验证在线检索模式
    resolved_retrieval_emb = None
    collection_registry: Optional[Dict[str, Any]] = None
    mismatch_reason: Optional[str] = None
    if request.retrieval_mode == "online":
        if not request.milvus_collection:
            raise HTTPException(status_code=400, detail="在线检索模式需要指定 Milvus 集合")

        collection_registry = milvus_collection_service.get_by_name(request.milvus_collection)
        collection_registry = verify_resource_ownership(
            collection_registry,
            current_user,
            "Milvus collection",
        )
        if collection_registry.get("status") != "active":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Milvus collection is unavailable because it is being deleted"
                ),
            )

        # Try to auto-resolve embedding config from collection registry
        if not request.retrieval_embedding_config or (
            not request.retrieval_embedding_config.config_id
            and not request.retrieval_embedding_config.endpoint
        ):
            if collection_registry.get("embedding_config_id"):
                config = _resolve_model_config(
                    collection_registry["embedding_config_id"],
                    current_user,
                )
                resolved_retrieval_emb = {
                    "config_id": config.get("config_id"),
                    "config_name": config.get("config_name"),
                    "endpoint": config.get("api_endpoint"),
                    "model_name": config.get("model_name"),
                    "api_key": config.get("api_key"),
                }

        if not resolved_retrieval_emb:
            if not request.retrieval_embedding_config:
                raise HTTPException(status_code=400, detail="在线检索模式需要指定检索用 Embedding 模型")
            ret_emb = request.retrieval_embedding_config
            if ret_emb.config_id:
                config = _resolve_model_config(ret_emb.config_id, current_user)
                resolved_retrieval_emb = {
                    "config_id": config.get("config_id"),
                    "config_name": config.get("config_name"),
                    "endpoint": config.get("api_endpoint"),
                    "model_name": config.get("model_name"),
                    "api_key": config.get("api_key"),
                }
            elif ret_emb.endpoint and ret_emb.model_name:
                resolved_retrieval_emb = {
                    "endpoint": validate_user_outbound_url(
                        ret_emb.endpoint,
                        current_user.get("user_id"),
                    ),
                    "model_name": ret_emb.model_name,
                    "api_key": ret_emb.api_key,
                }
            else:
                raise HTTPException(status_code=400, detail="检索 Embedding 模型需要提供 config_id 或 endpoint+model_name")

        mismatch_reason = _detect_collection_model_mismatch(
            collection_registry, resolved_retrieval_emb
        )
        if mismatch_reason and not request.allow_collection_model_mismatch:
            raise HTTPException(
                status_code=400,
                detail=(
                    "检索模型与集合入库模型不一致。请改为匹配模型，"
                    f"或显式允许 mismatch（结果仅供参考）。{mismatch_reason}"
                ),
            )

    # 确定评估类型
    eval_type = "multi"
    if has_embedding and not has_rerank and not has_llm_model:
        eval_type = "embedding"
    elif has_rerank and not has_embedding and not has_llm_model:
        eval_type = "rerank"
    elif has_llm_model and not has_embedding and not has_rerank:
        eval_type = "llm"

    if resolved_retrieval_emb:
        await asyncio.to_thread(
            authorize_inference_config, resolved_retrieval_emb, current_user.get("user_id"),
            current_user=current_user,
        )
    worker_groups = _build_worker_groups(request, resolved_retrieval_emb)
    if request.retrieval_mode == "online":
        worker_groups["evaluation_mode"] = "fallback" if mismatch_reason else "strict"
        worker_groups["model_mismatch"] = bool(mismatch_reason)
        if mismatch_reason:
            worker_groups["model_mismatch_reason"] = mismatch_reason

    validate_deep_evaluation_resource_config(
        {
            "model_configs": resolved_model_groups,
            "dataset_configs": dataset_configs,
            "max_samples": request.max_samples,
            "metrics": metrics,
            "worker_groups": worker_groups,
        }
    )

    try:
        task, execution_lease = background_task_admission_service.admit_execution(
            "evaluation",
            None,
            current_user.get("user_id"),
            deep_evaluation_task_service.create_task,
            task_name=request.task_name,
            description=request.description,
            eval_type=eval_type,
            dataset_configs=dataset_configs,
            max_samples=request.max_samples,
            field_mapping=(
                request.field_mapping.model_dump() if request.field_mapping else None
            ),
            model_configs=resolved_model_groups,
            metrics=metrics,
            worker_groups=worker_groups,
            user_id=current_user.get("user_id"),
            require_managed_datasets=requires_tenant_provenance(current_user),
        )
    except BackgroundTaskCapacityExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MilvusCollectionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatasetConsumptionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeDependencyUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        response = CreateDeepEvaluationTaskResponse(task_id=task["task_id"])
        background_tasks.add_task(
            background_task_admission_service.run_sync,
            execution_lease,
            _run_claimed_deep_evaluation_task,
            task["task_id"],
        )
    except Exception:
        execution_lease.release()
        _fail_unscheduled_deep_evaluation(task["task_id"])
        raise
    return response


@router.get("/tasks", response_model=DeepEvaluationTaskListResponse)
async def list_deep_evaluation_tasks(
    status: Optional[str] = None,
    eval_type: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取深度评估任务列表"""
    tasks, total = deep_evaluation_task_service.get_all_tasks(
        status=status,
        eval_type=eval_type,
        user_id=current_user["user_id"],
        limit=limit,
        offset=offset,
    )
    return DeepEvaluationTaskListResponse(
        items=[
            DeepEvaluationTaskResponse(**_public_deep_evaluation_task(task))
            for task in tasks
        ],
        total=total,
    )


@router.get("/tasks/{task_id}", response_model=DeepEvaluationTaskResponse)
async def get_deep_evaluation_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    task = deep_evaluation_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Deep evaluation task")
    return DeepEvaluationTaskResponse(**_public_deep_evaluation_task(task))


@router.post("/tasks/{task_id}/cancel")
async def cancel_deep_evaluation_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    task = deep_evaluation_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Deep evaluation task")
    if task.get("status") not in {DeepEvaluationStatus.PENDING, DeepEvaluationStatus.RUNNING}:
        raise HTTPException(status_code=400, detail="任务不在运行状态")

    cancel_deep_evaluation(task_id)
    if not deep_evaluation_task_service.cancel_task(task_id):
        raise HTTPException(
            status_code=409,
            detail="Deep evaluation task state changed before cancellation completed",
        )
    return {"message": "任务已取消", "task_id": task_id}


@router.post("/tasks/{task_id}/resume")
async def resume_deep_evaluation_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    task = deep_evaluation_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Deep evaluation task")
    if task.get("status") not in {DeepEvaluationStatus.FAILED, DeepEvaluationStatus.CANCELLED}:
        raise HTTPException(status_code=400, detail="只能重试失败或已取消的任务")

    # 保存已有结果，用于 resume 时跳过已完成的模型组
    # results_summary 结构: {"overall": ..., "metrics": ..., "by_group": {group_name: result}}
    raw_results = task.get("results_summary") or {}
    existing_results = raw_results.get("by_group") or {}

    _resolve_deep_evaluation_dataset_configs(
        [DatasetConfigModel(**config) for config in task.get("dataset_configs") or []],
        current_user,
    )
    _revalidate_deep_evaluation_collection(task, current_user)

    try:
        validate_deep_evaluation_resource_config(task)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Stored deep evaluation configuration exceeds current resource limits",
        ) from exc

    # 计算已完成的组数
    model_progress = task.get("model_progress") or {}
    skipped_groups = sum(
        1 for gp in model_progress.values()
        if gp and all(ds.get("status") == "completed" for ds in gp.values())
    )

    try:
        claimed, execution_lease = background_task_admission_service.admit_execution(
            "evaluation",
            task_id,
            current_user.get("user_id"),
            deep_evaluation_task_service.reset_for_resume,
            task_id,
            require_managed_datasets=requires_tenant_provenance(current_user),
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
            detail="Deep evaluation task was already resumed by another request",
        )
    try:
        clear_deep_cancellation(task_id)
        background_tasks.add_task(
            background_task_admission_service.run_sync,
            execution_lease,
            _run_claimed_deep_evaluation_task,
            task_id,
            existing_results=existing_results,
        )
    except Exception:
        execution_lease.release()
        _fail_unscheduled_deep_evaluation(task_id)
        raise
    return {
        "message": "任务已重新启动",
        "task_id": task_id,
        "skipped_groups": skipped_groups,
    }


@router.delete("/tasks/{task_id}")
async def delete_deep_evaluation_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    task = deep_evaluation_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Deep evaluation task")
    if task.get("status") in {DeepEvaluationStatus.RUNNING, DeepEvaluationStatus.PENDING}:
        raise HTTPException(status_code=400, detail="运行中的任务无法删除，请先取消")
    try:
        deletion_guard = background_task_admission_service.begin_deletion(
            "evaluation",
            task_id,
        )
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(
            status_code=409,
            detail="Deep evaluation worker is still shutting down; retry deletion shortly",
        ) from exc

    try:
        deep_evaluation_task_service.delete_task(task_id)
    finally:
        deletion_guard.release()
    return {"message": "任务已删除", "task_id": task_id}


# ===== Online Evaluation Endpoints (for quick testing) =====


MAX_DEEP_EVALUATION_QUICK_INPUT_CHARS = 16_384
MAX_DEEP_EVALUATION_QUICK_OUTPUT_CHARS = 65_536
MAX_DEEP_EVALUATION_QUICK_CONTEXT_ITEM_CHARS = 32_768
MAX_DEEP_EVALUATION_QUICK_CONTEXT_TOTAL_CHARS = 262_144
MAX_DEEP_EVALUATION_QUICK_METRIC_NAME_CHARS = 128

QuickEvaluationContext = Annotated[
    str,
    Field(max_length=MAX_DEEP_EVALUATION_QUICK_CONTEXT_ITEM_CHARS),
]
QuickEvaluationMetricName = Annotated[
    str,
    Field(min_length=1, max_length=MAX_DEEP_EVALUATION_QUICK_METRIC_NAME_CHARS),
]


class EvaluateSampleRequest(BaseModel):
    """单样本在线评估请求"""

    input: str = Field(
        ...,
        max_length=MAX_DEEP_EVALUATION_QUICK_INPUT_CHARS,
        description="用户问题/Query",
    )
    expected_output: Optional[str] = Field(
        default=None,
        max_length=MAX_DEEP_EVALUATION_QUICK_OUTPUT_CHARS,
        description="期望的答案",
    )
    actual_output: Optional[str] = Field(
        default=None,
        max_length=MAX_DEEP_EVALUATION_QUICK_OUTPUT_CHARS,
        description="实际生成的答案",
    )
    retrieval_context: List[QuickEvaluationContext] = Field(
        default_factory=list,
        max_length=MAX_DEEP_EVALUATION_TOP_K,
        description="检索到的上下文",
    )
    metrics: List[QuickEvaluationMetricName] = Field(
        default_factory=lambda: ["answer_relevancy"],
        min_length=1,
        max_length=MAX_DEEP_EVALUATION_METRICS,
        description="要使用的评估指标",
    )
    llm_config: LLMConfigModel = Field(..., description="LLM 配置")
    chunk_eval_mode: Literal["batch", "individual"] = Field(
        default="batch",
        description="Chunk 评估模式: batch 或 individual",
    )

    @field_validator("retrieval_context")
    @classmethod
    def validate_retrieval_context_total_chars(
        cls,
        contexts: List[str],
    ) -> List[str]:
        if sum(len(context) for context in contexts) > (
            MAX_DEEP_EVALUATION_QUICK_CONTEXT_TOTAL_CHARS
        ):
            raise ValueError("retrieval_context exceeds the total character limit")
        return contexts


class EvaluateSampleResponse(BaseModel):
    """单样本评估响应"""
    results: Dict[str, Dict[str, Any]] = Field(..., description="各指标评估结果")
    overall_score: float = Field(..., description="综合得分")


@router.post("/evaluate", response_model=EvaluateSampleResponse)
async def evaluate_sample(
    request: EvaluateSampleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """在线评估单个样本（深度评估）"""
    try:
        _, execution_lease = background_task_admission_service.admit_execution(
            "evaluation",
            None,
            current_user.get("user_id"),
            lambda: {"task_id": f"quick-evaluation-{uuid4()}"},
        )
    except BackgroundTaskCapacityExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        llm_config_dict = request.llm_config.model_dump(exclude_none=True)
        if request.llm_config.config_id:
            config = _resolve_model_config(request.llm_config.config_id, current_user)
            llm_config_dict = {
                "endpoint": config.get("api_endpoint"),
                "model": config.get("model_name"),
                "api_key": config.get("api_key"),
                "temperature": request.llm_config.temperature,
                "max_tokens": request.llm_config.max_tokens,
                "timeout": request.llm_config.timeout,
                "max_retries": request.llm_config.max_retries,
            }
        else:
            if not request.llm_config.endpoint or not request.llm_config.model:
                raise HTTPException(status_code=400, detail="LLM 配置缺少 endpoint 或 model")
        if not request.llm_config.config_id:
            llm_config_dict["endpoint"] = validate_user_outbound_url(
                request.llm_config.endpoint or "",
                current_user.get("user_id"),
            )
        llm_config_dict["user_id"] = current_user.get("user_id")
        await asyncio.to_thread(
            authorize_inference_config, llm_config_dict, current_user.get("user_id"),
            current_user=current_user,
        )
        llm_judge = create_llm_judge_from_dict(llm_config_dict)

        evaluator = DeepEvaluator(
            metrics=request.metrics,
            llm_client=llm_judge,
            concurrency=request.llm_config.concurrency,
            chunk_eval_mode=request.chunk_eval_mode,
        )

        sample = EvaluationSample(
            input=request.input,
            expected_output=request.expected_output,
            actual_output=request.actual_output,
            retrieval_context=request.retrieval_context,
        )

        async with llm_judge:
            result = await evaluator.evaluate(sample, request.metrics)

        return EvaluateSampleResponse(
            results=sanitize_public_diagnostics(
                {
                    name: res.to_dict()
                    for name, res in result.metric_results.items()
                }
            ),
            overall_score=result.overall_score or 0.0,
        )

    except SSRFError:
        raise
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="Invalid evaluation request",
        ) from e
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(
            status_code=500,
            detail=public_task_error_message(e),
        ) from e
    finally:
        if execution_lease is not None:
            execution_lease.release()
