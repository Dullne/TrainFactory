"""评估 API 路由

独立评估任务管理，用于评估已部署的模型。
训练任务的评估结果（test 集指标）在 TrainingDetail 页面展示。
"""
import hashlib
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, model_validator

from ...evaluation import (
    cancel_evaluation,
    clear_evaluation_cancellation,
    evaluation_task_service,
    is_evaluation_cancelled,
    run_evaluation_task,
)
from ...evaluation.evaluation_runner import (
    EVALUATION_IDENTITY_SCHEMA_VERSION,
    MAX_EVALUATION_BATCH_SIZE,
    MAX_EVALUATION_COMBINATIONS,
    MAX_EVALUATION_DATASETS,
    MAX_EVALUATION_MODELS,
    MAX_EVALUATION_MODEL_WORKERS,
    MAX_EVALUATION_SAMPLES,
    MAX_EVALUATION_WORKERS,
    EVALUATION_IDENTITY_FIELD,
    canonical_evaluation_dataset_key,
    canonical_evaluation_model_name,
    canonicalize_evaluation_dataset_configs,
    canonicalize_evaluation_model_configs,
    canonicalize_evaluation_resume_state,
    resolve_evaluation_dataset_path,
    validate_evaluation_resource_config,
)
from ...auth.dependencies import (
    get_current_user,
    requires_tenant_provenance,
    verify_resource_ownership,
)
from ...auth.resource_provenance import ResourceProvenanceError
from ...evaluation.dataset_access import (
    UnsupportedEvaluationDatasetStorageError,
    evaluation_dataset_allowed_roots,
    resolve_local_evaluation_dataset_record,
    resolve_managed_local_evaluation_dataset,
)
from ...storage.services.outbound_endpoint_policy import validate_user_outbound_url
from ...storage.services.inference_authorization_service import authorize_inference_config
from starlette.concurrency import run_in_threadpool
from ...deployment.deployment_service import deployment_service
from ...storage.services.dataset_service import (
    DatasetConsumptionUnavailableError,
)
from ...storage.services.runtime_dependency_service import (
    RuntimeDependencyUnavailableError,
)
from ...storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    BackgroundTaskCapacityExceeded,
    background_task_admission_service,
)
from ...utils.public_diagnostics import (
    public_safe_identifier,
    public_task_error_message,
    sanitize_public_diagnostics,
)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])
logger = logging.getLogger(__name__)


def _persisted_dataset_configs(
    configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    persisted = []
    for raw_config in configs:
        config = dict(raw_config)
        config.pop("result_key", None)
        persisted.append(config)
    return persisted


def _public_evaluation_task(task: Dict[str, Any]) -> Dict[str, Any]:
    public_task = dict(task)
    raw_model_configs = task.get("model_configs") or []
    raw_dataset_configs = task.get("dataset_configs") or []
    identity_map: Dict[str, Dict[str, Dict[str, str]]] = {
        "models": {},
        "datasets": {},
    }

    public_models = []
    for raw_config in raw_model_configs:
        if not isinstance(raw_config, dict):
            continue
        config = dict(raw_config)
        config.pop(EVALUATION_IDENTITY_FIELD, None)
        config.pop("result_key", None)
        public_models.append(config)
        try:
            result_key = canonical_evaluation_model_name(raw_config)
        except ValueError:
            continue
        identity_map["models"][result_key] = {"name": result_key}

    public_datasets = []
    dataset_aliases: Dict[str, set[str]] = {}
    dataset_names: Dict[str, str] = {}
    private_paths = []
    for raw_config in raw_dataset_configs:
        if not isinstance(raw_config, dict):
            continue
        config = dict(raw_config)
        raw_path = config.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            raw_path = raw_path.strip()
            private_paths.append(raw_path)
        else:
            raw_path = None
        try:
            result_key = canonical_evaluation_dataset_key(config)
        except ValueError:
            dataset_type = str(config.get("type") or "").strip().lower()
            name = str(config.get("name") or "").strip()
            if dataset_type == "mteb" and name:
                result_key = f"mteb:{name}"
            elif raw_path:
                digest = hashlib.sha256(raw_path.encode("utf-8")).hexdigest()
                result_key = f"local:sha256:{digest}"
            else:
                result_key = None

        config.pop(EVALUATION_IDENTITY_FIELD, None)
        config.pop("path", None)
        config.pop("result_key", None)
        if str(config.get("type") or "").strip().lower() == "mteb":
            config.pop("dataset_id", None)
        if result_key:
            config["result_key"] = result_key
            dataset_type = str(config.get("type") or "").strip().lower()
            dataset_name = str(config.get("name") or result_key)
            identity_map["datasets"][result_key] = {
                "name": dataset_name,
                "type": dataset_type,
            }
            dataset_names[result_key] = dataset_name
            aliases = {result_key, dataset_name}
            if raw_path:
                aliases.update({raw_path, f"local:{raw_path}"})
            for alias in aliases:
                dataset_aliases.setdefault(alias, set()).add(result_key)
        public_datasets.append(config)

    def public_dataset_key(stored_key: Any) -> Any:
        if not isinstance(stored_key, str) or stored_key == "_error":
            return stored_key
        candidates = dataset_aliases.get(stored_key, set())
        if len(candidates) == 1:
            return next(iter(candidates))
        if stored_key.startswith("local:"):
            digest = hashlib.sha256(stored_key.encode("utf-8")).hexdigest()
            return f"local:legacy-sha256:{digest}"
        return public_safe_identifier(stored_key)

    def rewrite_matrix(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        rewritten = {}
        for model_key, datasets in value.items():
            if not isinstance(datasets, dict):
                rewritten[model_key] = datasets
                continue
            rewritten_datasets = {}
            for stored_key, stored_value in datasets.items():
                key = public_dataset_key(stored_key)
                if key in rewritten_datasets:
                    digest = hashlib.sha256(str(stored_key).encode("utf-8")).hexdigest()
                    key = f"legacy:sha256:{digest}"
                rewritten_datasets[key] = stored_value
            rewritten[model_key] = rewritten_datasets
        return rewritten

    def redact_paths(value: Any) -> Any:
        if isinstance(value, str):
            for path in private_paths:
                value = value.replace(path, "[local dataset]")
            return value
        if isinstance(value, list):
            return [redact_paths(item) for item in value]
        if isinstance(value, dict):
            return {
                redact_paths(key): redact_paths(item)
                for key, item in value.items()
            }
        return value

    public_task["model_configs"] = public_models
    public_task["dataset_configs"] = public_datasets
    public_task["results"] = sanitize_public_diagnostics(
        rewrite_matrix(public_task.get("results"))
    )
    public_task["model_progress"] = sanitize_public_diagnostics(
        rewrite_matrix(public_task.get("model_progress"))
    )
    public_task["error_message"] = public_task_error_message(
        public_task.get("error_message")
    )
    public_task["report_path"] = None
    public_task["results_path"] = None
    current_dataset_key = public_dataset_key(public_task.get("current_dataset"))
    public_task["current_dataset"] = dataset_names.get(
        current_dataset_key,
        public_safe_identifier(current_dataset_key),
    )
    public_task["identity_schema_version"] = EVALUATION_IDENTITY_SCHEMA_VERSION
    public_task["identity_map"] = identity_map
    public_task.pop("run_token", None)
    public_task = redact_paths(public_task)
    return public_task


def _fail_unscheduled_evaluation(task_id: str) -> None:
    try:
        evaluation_task_service.update_status(
            task_id,
            "failed",
            "Task could not be scheduled for background execution",
        )
    except Exception:
        logger.exception("Failed to finalize unscheduled evaluation task %s", task_id)


def _run_claimed_evaluation_task(
    task_id: str,
    config: Dict[str, Any],
) -> None:
    """Claim a pending task and recheck cancellation before entering the runner."""
    clear_cancellation = False
    claimed = False
    runner_started = False
    run_token = str(uuid4())
    try:
        if not evaluation_task_service.claim_running(task_id, run_token=run_token):
            task = evaluation_task_service.get_task(task_id)
            clear_cancellation = bool(task and task.get("status") == "cancelled")
            logger.info(
                "Evaluation worker did not claim task %s; current status is %s",
                task_id,
                task.get("status") if task else "missing",
            )
            return

        claimed = True
        clear_cancellation = True
        task = evaluation_task_service.get_task(task_id)
        if not task or task.get("status") != "running":
            logger.info(
                "Evaluation task %s changed state before runner registration",
                task_id,
            )
            return
        if is_evaluation_cancelled(task_id):
            logger.info(
                "Evaluation task %s was cancelled before runner registration",
                task_id,
            )
            return
        runner_started = True
        run_evaluation_task(task_id, config, expected_run_token=run_token)
    except Exception:
        if claimed and not runner_started:
            try:
                evaluation_task_service.fail_claimed_startup(task_id, run_token)
            except Exception:
                logger.exception("Failed to finalize evaluation startup for %s", task_id)
        raise
    finally:
        if clear_cancellation:
            clear_evaluation_cancellation(task_id)


# ============================================================================
# Request/Response Models for Evaluation Tasks
# ============================================================================

class ModelConfig(BaseModel):
    """模型配置"""
    model_id: Optional[str] = Field(default=None, max_length=36, description="已注册模型 ID")
    endpoint: str = Field(..., min_length=1, max_length=2048, description="API 端点")
    model_name: Optional[str] = Field(default=None, max_length=256, description="模型名称")
    name: Optional[str] = Field(default=None, max_length=256, description="显示名称")
    inference_framework: Optional[str] = Field(default=None, max_length=32, description="推理框架: vllm | sglang | xinference")
    deployment_id: Optional[str] = Field(default=None, max_length=36)
    deployment_replica_id: Optional[str] = Field(default=None, max_length=36)

    @model_validator(mode="after")
    def normalize_identity(self):
        normalized = canonicalize_evaluation_model_configs([self.model_dump()])[0]
        self.name = normalized["name"]
        self.model_name = normalized.get("model_name")
        return self


class DatasetConfig(BaseModel):
    """数据集配置"""
    type: str = Field(..., min_length=1, max_length=32, description="数据集类型: mteb | local | registered")
    name: str = Field(..., min_length=1, max_length=256, description="数据集名称")
    path: Optional[str] = Field(default=None, max_length=1024, description="本地数据集路径")
    dataset_id: Optional[str] = Field(default=None, max_length=36, description="已注册数据集 ID")


    @model_validator(mode="after")
    def normalize_and_validate_source_fields(self):
        self.type = self.type.strip().lower()
        self.name = self.name.strip()
        if self.type == "mteb" and (
            self.dataset_id is not None or self.path is not None
        ):
            raise ValueError("MTEB datasets cannot include dataset_id or path")
        return self


class CreateEvaluationRequest(BaseModel):
    """创建评估任务请求"""
    task_name: Optional[str] = Field(default=None, max_length=255, description="任务名称")
    model_configs: List[ModelConfig] = Field(..., min_length=1, max_length=MAX_EVALUATION_MODELS, description="模型配置列表")
    dataset_configs: List[DatasetConfig] = Field(..., min_length=1, max_length=MAX_EVALUATION_DATASETS, description="数据集配置列表")
    max_samples: Optional[int] = Field(default=None, ge=1, le=MAX_EVALUATION_SAMPLES, description="最大样本数")
    batch_size: int = Field(default=50, ge=1, le=MAX_EVALUATION_BATCH_SIZE, description="批处理大小")
    workers: int = Field(default=8, ge=1, le=MAX_EVALUATION_WORKERS, description="API 并发数")
    model_workers: int = Field(default=2, ge=1, le=MAX_EVALUATION_MODEL_WORKERS, description="模型并发数")

    @model_validator(mode="after")
    def validate_combination_count(self):
        if len(self.model_configs) * len(self.dataset_configs) > MAX_EVALUATION_COMBINATIONS:
            raise ValueError(
                "Evaluation model/dataset combinations exceed the configured limit"
            )
        return self


class CreateEvaluationResponse(BaseModel):
    """创建评估任务响应"""
    task_id: str
    message: str = "Evaluation task created"


class EvaluationTaskResponse(BaseModel):
    """评估任务响应"""
    task_id: str
    task_name: Optional[str] = None
    description: Optional[str] = None
    eval_type: str
    model_configs: Optional[List[Dict[str, Any]]] = None
    dataset_configs: Optional[List[Dict[str, Any]]] = None
    identity_schema_version: Optional[int] = None
    identity_map: Optional[Dict[str, Any]] = None
    max_samples: Optional[int] = None
    batch_size: int = 50
    workers: int = 8
    model_workers: int = 2
    status: str
    progress: float
    current_model: Optional[str] = None
    current_dataset: Optional[str] = None
    # Per-model per-dataset progress matrix
    model_progress: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    report_path: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class EvaluationTaskListResponse(BaseModel):
    """评估任务列表响应"""
    items: List[EvaluationTaskResponse]
    total: int


# ============================================================================
# MTEB Dataset Information
# ============================================================================

# MTEB Reranking datasets
RERANKING_DATASETS = {
    # 中文数据集
    "T2Reranking": {"split": "dev", "lang": "zh", "description": "中文通用重排序"},
    "MMarcoReranking": {"split": "dev", "lang": "zh", "description": "中文 MS MARCO"},
    "CMedQAv1-reranking": {"split": "test", "lang": "zh", "description": "中文医疗问答 v1"},
    "CMedQAv2-reranking": {"split": "test", "lang": "zh", "description": "中文医疗问答 v2"},
    # 英文数据集
    "AskUbuntuDupQuestions": {"split": "test", "lang": "en", "description": "Ubuntu 问题重复检测"},
    "MindSmallReranking": {"split": "test", "lang": "en", "description": "新闻推荐"},
    "SciDocsRR": {"split": "test", "lang": "en", "description": "科学文档"},
    "StackOverflowDupQuestions": {"split": "test", "lang": "en", "description": "StackOverflow 重复问题"},
    "WebLINXCandidatesReranking": {"split": "test", "lang": "en", "description": "Web 导航"},
    "BuiltBenchReranking": {"split": "test", "lang": "en", "description": "建筑领域"},
}

# 数据集组
DATASET_GROUPS = {
    "chinese": ["T2Reranking", "MMarcoReranking", "CMedQAv1-reranking", "CMedQAv2-reranking"],
    "english": ["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR",
                "StackOverflowDupQuestions", "WebLINXCandidatesReranking", "BuiltBenchReranking"],
    "all": list(RERANKING_DATASETS.keys()),
}


def _validate_model_configs(
    model_configs: List[Dict[str, Any]],
    user_id: Optional[str],
) -> List[Dict[str, Any]]:
    validated = []
    for raw_config in model_configs:
        config = dict(raw_config)
        deployment_id = config.get("deployment_id")
        replica_id = config.get("deployment_replica_id")
        if deployment_id:
            deployment, replica = deployment_service.resolve_replica_selection(
                deployment_id,
                replica_id,
                user_id=user_id,
                require_healthy=True,
            )
            if replica is None:
                config["endpoint"] = deployment["xinference_endpoint"]
                config.pop("deployment_replica_id", None)
            else:
                config["endpoint"] = replica["endpoint"]
                config["deployment_replica_id"] = replica["replica_id"]
            config["model_name"] = deployment.get("model_uid") or config.get(
                "model_name"
            )
            config["inference_framework"] = deployment.get(
                "inference_framework"
            )
        else:
            if replica_id is not None:
                raise ValueError("deployment_replica_id requires deployment_id")
            config.pop("deployment_id", None)
            config.pop("deployment_replica_id", None)
            config["endpoint"] = validate_user_outbound_url(
                config.get("endpoint", ""),
                user_id,
            )
        authorize_inference_config(config, user_id)
        validated.append(config)
    return canonicalize_evaluation_model_configs(validated)


def _validate_dataset_configs(
    dataset_configs: List[Dict[str, Any]],
    current_user: Dict[str, Any],
) -> List[Dict[str, Any]]:
    validated = []
    require_provenance = requires_tenant_provenance(current_user)
    user_id = current_user.get("user_id")
    for raw_config in dataset_configs:
        config = dict(raw_config)
        dataset_type = config.get("type")
        if not isinstance(dataset_type, str) or not dataset_type.strip():
            raise HTTPException(
                status_code=400,
                detail="Evaluation dataset type must be non-empty",
            )
        dataset_type = dataset_type.strip().lower()
        config["type"] = dataset_type
        name = config.get("name")
        if isinstance(name, str):
            config["name"] = name.strip()
        if dataset_type == "mteb":
            local_only_fields = [
                field_name
                for field_name in ("dataset_id", "path")
                if config.get(field_name) is not None
            ]
            if local_only_fields:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "MTEB datasets cannot include local-only fields: "
                        + ", ".join(local_only_fields)
                    ),
                )
            # MTEB 数据集名必须来自白名单：外部库按名从 HF hub 下载，自由
            # 字符串可指向任意仓库（供应链/数据投毒/下载无配额限制）。
            if config.get("name") not in RERANKING_DATASETS:
                raise HTTPException(
                    status_code=400,
                    detail=f"MTEB 数据集 '{config.get('name')}' 不在白名单中",
                )
        elif dataset_type in {"local", "registered"}:
            try:
                dataset_id = config.get("dataset_id")
                dataset = None
                if require_provenance:
                    if not dataset_id:
                        raise HTTPException(
                            status_code=403,
                            detail=(
                                "Local evaluation datasets must be selected by dataset_id "
                                "when authentication is enabled"
                            ),
                        )
                    dataset = resolve_managed_local_evaluation_dataset(
                        dataset_id,
                        user_id=user_id,
                    )
                elif dataset_id:
                    dataset = resolve_local_evaluation_dataset_record(dataset_id)

                if dataset:
                    config["dataset_id"] = dataset["dataset_id"]
                    config["name"] = (
                        dataset.get("dataset_name")
                        or dataset.get("display_name")
                        or dataset["dataset_id"]
                    )
                    config["path"] = dataset["storage_path"]

                config["path"] = resolve_evaluation_dataset_path(
                    config.get("path", ""),
                    allowed_roots=evaluation_dataset_allowed_roots(),
                )
            except HTTPException:
                raise
            except UnsupportedEvaluationDatasetStorageError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ResourceProvenanceError as exc:
                raise HTTPException(
                    status_code=403,
                    detail="Dataset does not have verifiable API-managed provenance",
                ) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported evaluation dataset type: {dataset_type}",
            )
        validated.append(config)
    try:
        canonicalized = canonicalize_evaluation_dataset_configs(validated)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # The nested marker is persisted; the top-level result_key is runtime-only.
    # Public route responses remove both internal identity fields.
    return _persisted_dataset_configs(canonicalized)


# ============================================================================
# Evaluation Task Management Endpoints
# ============================================================================

@router.post("", response_model=CreateEvaluationResponse)
async def create_evaluation_task(
    request: CreateEvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    创建评估任务

    创建一个独立的评估任务，支持多模型、多数据集并行评估。
    """
    # Validate request
    if not request.model_configs:
        raise HTTPException(status_code=400, detail="至少需要一个模型配置")
    if not request.dataset_configs:
        raise HTTPException(status_code=400, detail="至少需要一个数据集配置")

    # 多模型对比要求显示名唯一，否则同名模型的结果会静默互相覆盖
    model_names = [
        mc.name or mc.model_name
        for mc in request.model_configs
        if (mc.name or mc.model_name)
    ]
    if len(model_names) != len(set(model_names)):
        raise HTTPException(
            status_code=400,
            detail="模型显示名称必须唯一（多模型对比结果按名称聚合）",
        )

    # Serialize configs once
    try:
        model_configs_data = await run_in_threadpool(
            _validate_model_configs, [mc.model_dump() for mc in request.model_configs],
            current_user.get("user_id"),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset_configs_data = _validate_dataset_configs(
        [dc.model_dump(exclude_none=True) for dc in request.dataset_configs],
        current_user,
    )

    # Determine eval_type only after dataset types are normalized and checked.
    dataset_types = {config["type"] for config in dataset_configs_data}
    if dataset_types == {"mteb"}:
        eval_type = "reranker-mteb"
    elif dataset_types == {"local"}:
        eval_type = "reranker-local"
    elif dataset_types == {"registered"}:
        eval_type = "reranker-registered"
    elif len(dataset_types) > 1:
        eval_type = "reranker-mixed"
    else:
        eval_type = "reranker"

    # Create task in database
    try:
        task_info, execution_lease = background_task_admission_service.admit_execution(
            "evaluation",
            None,
            current_user.get("user_id"),
            evaluation_task_service.create_task,
            task_name=request.task_name,
            eval_type=eval_type,
            model_configs=model_configs_data,
            dataset_configs=dataset_configs_data,
            max_samples=request.max_samples,
            batch_size=request.batch_size,
            workers=request.workers,
            model_workers=request.model_workers,
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
    except DatasetConsumptionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeDependencyUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Prepare config for runner
    config = {
        "model_configs": model_configs_data,
        "dataset_configs": dataset_configs_data,
        "max_samples": request.max_samples,
        "batch_size": request.batch_size,
        "workers": request.workers,
        "model_workers": request.model_workers,
    }

    try:
        response = CreateEvaluationResponse(
            task_id=task_info["task_id"],
            message="评估任务已创建",
        )
        background_tasks.add_task(
            background_task_admission_service.run_sync,
            execution_lease,
            _run_claimed_evaluation_task,
            task_info["task_id"],
            config,
        )
    except Exception:
        execution_lease.release()
        _fail_unscheduled_evaluation(task_info["task_id"])
        raise

    return response


@router.get("/tasks", response_model=EvaluationTaskListResponse)
async def list_evaluation_tasks(
    status: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    获取评估任务列表
    """
    tasks, total = evaluation_task_service.get_all_tasks(
        status=status,
        user_id=current_user.get("user_id"),
        limit=limit,
        offset=offset,
    )

    return EvaluationTaskListResponse(
        items=[EvaluationTaskResponse(**_public_evaluation_task(t)) for t in tasks],
        total=total,
    )


@router.get("/tasks/{task_id}", response_model=EvaluationTaskResponse)
async def get_evaluation_task_detail(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    获取评估任务详情
    """
    task = evaluation_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Evaluation task")
    return EvaluationTaskResponse(**_public_evaluation_task(task))


@router.post("/tasks/{task_id}/cancel")
async def cancel_evaluation_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    取消评估任务
    """
    task = evaluation_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Evaluation task")

    if task["status"] not in ["pending", "running"]:
        raise HTTPException(status_code=400, detail=f"无法取消状态为 {task['status']} 的任务")

    # Signal runner to stop (for running tasks)
    cancel_evaluation(task_id)

    # Update status in database
    if not evaluation_task_service.cancel_task(task_id):
        raise HTTPException(
            status_code=409,
            detail="Evaluation task state changed before cancellation completed",
        )

    return {"message": "任务已取消", "task_id": task_id}


@router.delete("/tasks/{task_id}")
async def delete_evaluation_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    删除评估任务
    """
    task = evaluation_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Evaluation task")

    if task["status"] in {"pending", "running"}:
        raise HTTPException(status_code=400, detail="无法删除正在运行的任务，请先取消")
    try:
        deletion_guard = background_task_admission_service.begin_deletion(
            "evaluation",
            task_id,
        )
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(
            status_code=409,
            detail="Evaluation worker is still shutting down; retry deletion shortly",
        ) from exc

    try:
        success = evaluation_task_service.delete_task(task_id)
        if not success:
            raise HTTPException(status_code=500, detail="删除任务失败")
    finally:
        deletion_guard.release()

    return {"message": "任务已删除", "task_id": task_id}


@router.post("/tasks/{task_id}/resume")
async def resume_evaluation_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    续跑评估任务

    重新执行失败或取消的任务，跳过已完成的模型+数据集组合。
    """
    task = evaluation_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Evaluation task")

    if task["status"] not in ["failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"只能续跑失败或已取消的任务，当前状态: {task['status']}"
        )

    try:
        model_configs_data = await run_in_threadpool(
            _validate_model_configs, task["model_configs"],
            current_user.get("user_id"),
        )
        dataset_configs_data = _validate_dataset_configs(
            task["dataset_configs"],
            current_user,
        )
        resume_state = canonicalize_evaluation_resume_state(
            task["model_configs"],
            task["dataset_configs"],
            task.get("results") or {},
            task.get("model_progress") or {},
            canonical_model_configs=model_configs_data,
            canonical_dataset_configs=dataset_configs_data,
        )
        model_configs_data = resume_state["model_configs"]
        dataset_configs_data = _persisted_dataset_configs(
            resume_state["dataset_configs"]
        )
        existing_results = resume_state["results"]
        existing_model_progress = resume_state["model_progress"]
        validate_evaluation_resource_config(
            {
                "model_configs": model_configs_data,
                "dataset_configs": dataset_configs_data,
                "max_samples": task.get("max_samples"),
                "batch_size": task.get("batch_size", 50),
                "workers": task.get("workers", 8),
                "model_workers": task.get("model_workers", 2),
            }
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Stored evaluation configuration is invalid or exceeds current limits",
        ) from exc

    # Count only successfully completed evaluations (exclude errors) before
    # claiming the task, so malformed legacy result data cannot strand a lease.
    skipped_count = sum(
        1
        for datasets in existing_results.values()
        if isinstance(datasets, dict)
        for result in datasets.values()
        if isinstance(result, dict) and "error" not in result
    )

    # Claim the resumable state atomically with the cross-task capacity check.
    try:
        claimed, execution_lease = background_task_admission_service.admit_execution(
            "evaluation",
            task_id,
            current_user.get("user_id"),
            evaluation_task_service.reset_for_resume,
            task_id,
            require_managed_datasets=requires_tenant_provenance(current_user),
            model_configs=model_configs_data,
            dataset_configs=dataset_configs_data,
            results=existing_results,
            model_progress=existing_model_progress,
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
            detail="Evaluation task is no longer resumable",
        )
    # Prepare config for runner
    config = {
        "model_configs": model_configs_data,
        "dataset_configs": dataset_configs_data,
        "max_samples": task.get("max_samples"),
        "batch_size": task.get("batch_size", 50),
        "workers": task.get("workers", 8),
        "model_workers": task.get("model_workers", 2),
        "existing_results": existing_results,  # Pass existing results to skip
        "existing_model_progress": existing_model_progress,
    }

    response = {
        "message": "任务已重新启动",
        "task_id": task_id,
        "skipped_evaluations": skipped_count,
    }
    try:
        clear_evaluation_cancellation(task_id)
        background_tasks.add_task(
            background_task_admission_service.run_sync,
            execution_lease,
            _run_claimed_evaluation_task,
            task_id,
            config,
        )
    except Exception:
        execution_lease.release()
        _fail_unscheduled_evaluation(task_id)
        raise

    return response


@router.get("/datasets")
async def list_available_datasets():
    """
    获取可用的评估数据集列表
    """
    return {
        "datasets": RERANKING_DATASETS,
        "groups": DATASET_GROUPS,
    }
