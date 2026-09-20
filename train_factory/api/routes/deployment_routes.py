"""
Deployment API routes.

Provides endpoints for managing model deployments to Xinference.
Supports two deployment modes:
- shared: Connect to a shared Xinference service
- container: Auto-create Docker container with Xinference
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends, Header, BackgroundTasks
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)

from ...auth.dependencies import (
    get_current_user,
    requires_tenant_provenance,
    verify_resource_ownership,
)
from ...auth.resource_provenance import (
    ResourceProvenanceError,
    require_managed_model_provenance,
)
from ...config.settings import get_settings
from ...core.idempotency import check_idempotency, store_idempotency_response
from ...core.ssrf import SSRFError
from ...deployment.deployment_service import (
    DEFERRED_AUTO_START_CONFIG_KEY,
    REPLICA_OPERATION_LEASE_SECONDS,
    DeploymentReplicaNotFoundError,
    DeploymentReplicaStateConflictError,
    ReplicaOperationBusyError,
    ReplicaOperationLostError,
    deployment_service,
    get_default_xinference_endpoint,
)
from ...deployment.docker_deployer import docker_deployer
from ...deployment.launch_config import LaunchConfig, normalize_launch_config
from ...storage.services.model_registry_service import model_registry_service
from ...storage.services.training_task_service import training_task_service
from ...storage.services.outbound_endpoint_policy import (
    request_user_outbound,
    validate_user_outbound_url,
)

logger = logging.getLogger(__name__)

router = APIRouter()
_deferred_start_tasks: Dict[str, asyncio.Task[None]] = {}
_DEFERRED_START_RETRY_SECONDS = 1.0
_DEFERRED_START_MAX_BUSY_ATTEMPTS = 3
_DEFERRED_START_MAX_TRANSIENT_ATTEMPTS = 3
_DEFERRED_START_LEASE_RETRY_SECONDS = float(REPLICA_OPERATION_LEASE_SECONDS)
_DEFERRED_START_SHUTDOWN_POLL_SECONDS = 1.0
_deferred_start_shutdown = False


async def _sleep_for_deferred_retry(delay: float) -> bool:
    """Sleep in bounded slices so shutdown can interrupt a lease wait."""
    remaining = max(0.0, delay)
    while remaining > 0:
        if _deferred_start_shutdown:
            return False
        interval = min(remaining, _DEFERRED_START_SHUTDOWN_POLL_SECONDS)
        await asyncio.sleep(interval)
        remaining -= interval
    return not _deferred_start_shutdown


async def _start_deployment_in_background(deployment_id: str) -> None:
    """Start a deployment after the create response has returned."""
    try:
        deployment = deployment_service.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment not found: {deployment_id}")
        _validate_deployment_endpoint(deployment)
        await asyncio.to_thread(
            deployment_service.start_deployment,
            deployment_id,
            user_id=deployment.get("user_id"),
        )
    except Exception as e:
        logger.exception(f"Background start failed for deployment {deployment_id}: {e}")


async def _start_deferred_deployment_in_background(deployment_id: str) -> None:
    """Consume durable auto-start intent, waiting out an older process claim."""
    busy_attempts = 0
    transient_attempts = 0
    while True:
        try:
            deployment = deployment_service.get_deployment(deployment_id)
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")
            _validate_deployment_endpoint(deployment)
            await asyncio.to_thread(
                deployment_service.start_deferred_deployment,
                deployment_id,
                user_id=deployment.get("user_id"),
            )
            return
        except ReplicaOperationBusyError as exc:
            busy_attempts += 1
            if _deferred_start_shutdown:
                return
            if busy_attempts >= _DEFERRED_START_MAX_BUSY_ATTEMPTS:
                logger.error(
                    "Deferred background start exhausted %s busy attempts for "
                    "deployment %s; durable auto-start intent retained: %s",
                    busy_attempts,
                    deployment_id,
                    exc,
                )
                if not await _sleep_for_deferred_retry(
                    _DEFERRED_START_LEASE_RETRY_SECONDS
                ):
                    return
                busy_attempts = 0
                continue
            if not await _sleep_for_deferred_retry(
                _DEFERRED_START_RETRY_SECONDS * (2 ** (busy_attempts - 1))
            ):
                return
        except Exception as e:
            transient_attempts += 1
            logger.exception(
                "Deferred background start attempt %s failed for deployment %s: %s",
                transient_attempts,
                deployment_id,
                e,
            )
            if _deferred_start_shutdown:
                return
            if transient_attempts >= _DEFERRED_START_MAX_TRANSIENT_ATTEMPTS:
                logger.error(
                    "Deferred background start exhausted %s transient attempts for "
                    "deployment %s; durable auto-start intent retained",
                    transient_attempts,
                    deployment_id,
                )
                if not await _sleep_for_deferred_retry(
                    _DEFERRED_START_LEASE_RETRY_SECONDS
                ):
                    return
                transient_attempts = 0
                continue
            if not await _sleep_for_deferred_retry(_DEFERRED_START_RETRY_SECONDS):
                return


def _ensure_deferred_deployment_start(
    deployment_id: str,
) -> asyncio.Task[None]:
    current = _deferred_start_tasks.get(deployment_id)
    if current is not None and not current.done():
        return current
    task = asyncio.create_task(
        _start_deferred_deployment_in_background(deployment_id),
        name=f"deployment-auto-start-{deployment_id}",
    )
    _deferred_start_tasks[deployment_id] = task

    def discard(completed: asyncio.Task[None]) -> None:
        if _deferred_start_tasks.get(deployment_id) is completed:
            _deferred_start_tasks.pop(deployment_id, None)

    task.add_done_callback(discard)
    return task


def _reschedule_cached_deferred_start(cached_response: Dict[str, Any]) -> None:
    config = cached_response.get("config")
    deployment_id = cached_response.get("deployment_id")
    if (
        isinstance(config, dict)
        and config.get(DEFERRED_AUTO_START_CONFIG_KEY) is True
        and isinstance(deployment_id, str)
        and deployment_id
    ):
        _ensure_deferred_deployment_start(deployment_id)


async def _run_registered_deferred_deployment_start(deployment_id: str) -> None:
    """Attach response background work to the process task registry."""
    task = _ensure_deferred_deployment_start(deployment_id)
    await asyncio.shield(task)


async def _resume_deferred_deployment_starts() -> None:
    """Register durable auto-start plans left by an earlier process."""
    global _deferred_start_shutdown
    _deferred_start_shutdown = False
    deployment_ids = await asyncio.to_thread(
        deployment_service.list_deferred_start_deployment_ids
    )
    for deployment_id in deployment_ids:
        _ensure_deferred_deployment_start(deployment_id)


async def _drain_deferred_deployment_starts() -> None:
    """Wait for all registered starts so shutdown never abandons a live worker."""
    global _deferred_start_shutdown
    _deferred_start_shutdown = True
    while _deferred_start_tasks:
        tasks = list(_deferred_start_tasks.values())
        await asyncio.gather(*tasks, return_exceptions=True)


def _build_deployment_config_with_external_api(
    config: Optional[Dict[str, Any]],
    external_api_config_id: Optional[str],
    current_user: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Normalize and validate external_api_config binding in deployment config."""
    merged_config: Dict[str, Any] = dict(config or {})

    request_api_config_id = (external_api_config_id or "").strip()
    raw_config_api_config_id = merged_config.get("external_api_config_id")
    if raw_config_api_config_id is not None and not isinstance(
        raw_config_api_config_id, str
    ):
        raise HTTPException(
            status_code=400,
            detail="config.external_api_config_id must be a string",
        )
    config_api_config_id = (raw_config_api_config_id or "").strip()

    if (
        request_api_config_id
        and config_api_config_id
        and request_api_config_id != config_api_config_id
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "external_api_config_id mismatch between field and config payload: "
                f"{request_api_config_id} != {config_api_config_id}"
            ),
        )

    target_api_config_id = request_api_config_id or config_api_config_id
    if target_api_config_id:
        from ...storage.services.external_api_config_service import (
            external_api_config_service,
        )

        api_config = external_api_config_service.get_config(target_api_config_id)
        verify_resource_ownership(api_config, current_user, "External API config")
        merged_config["external_api_config_id"] = target_api_config_id

    return merged_config or None


# === Request/Response Models ===


def _normalized_request_launch_config(
    request: "CreateDeploymentRequest | CreateContainerDeploymentRequest",
) -> LaunchConfig | None:
    if request.config and "launch_config" in request.config:
        raise ValueError(
            "config.launch_config is reserved; use the typed launch_config field"
        )
    return normalize_launch_config(
        framework=request.inference_framework,
        replica_count=request.replica,
        gpu_id=request.gpu_id,
        legacy_config=request.config,
        launch_config=request.launch_config,
    )


def _attach_launch_config(
    config: Optional[Dict[str, Any]],
    launch_config: LaunchConfig | None,
) -> Optional[Dict[str, Any]]:
    if launch_config is None:
        return config
    result = dict(config or {})
    result["launch_config"] = launch_config.model_dump(mode="json")
    return result


class CreateDeploymentRequest(BaseModel):
    """Create deployment request (shared mode)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Model ID to deploy")
    xinference_endpoint: Optional[str] = Field(
        default=None,
        description="Inference server endpoint (uses default if not specified)",
    )
    deployment_name: Optional[str] = Field(default=None, description="Deployment name")
    gpu_id: Optional[StrictInt] = Field(
        default=None,
        ge=0,
        description="Preferred GPU device ID for Xinference shared mode",
    )
    replica: StrictInt = Field(
        default=1, ge=1, le=8, description="Number of replicas (1-8)"
    )
    gpu_memory_utilization: Optional[StrictFloat] = Field(
        default=None,
        ge=0.05,
        le=1.0,
        description="GPU memory utilization (auto-calculated if not specified)",
    )
    inference_framework: str = Field(
        default="xinference",
        pattern="^(vllm|sglang|xinference)$",
        description="Inference framework",
    )
    enable_lora: bool = Field(default=False, description="Enable LoRA hot-loading")
    max_loras: StrictInt = Field(
        default=4, ge=1, le=16, description="Maximum number of LoRA adapters"
    )
    max_lora_rank: StrictInt = Field(
        default=64, ge=8, le=256, description="Maximum LoRA rank"
    )
    auto_start: bool = Field(default=True, description="Auto start after creation")
    external_api_config_id: Optional[str] = Field(
        default=None,
        description="Optional external API config id used as tenant binding",
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional configuration"
    )
    launch_config: Optional[LaunchConfig] = Field(
        default=None,
        description="Typed vLLM or SGLang container launch configuration",
    )
    user_id: Optional[str] = Field(default=None, description="User ID for isolation")

    @model_validator(mode="after")
    def _validate_launch_config(self) -> "CreateDeploymentRequest":
        self.normalized_launch_config()
        return self

    def normalized_launch_config(self) -> LaunchConfig | None:
        return _normalized_request_launch_config(self)


class CreateContainerDeploymentRequest(BaseModel):
    """Create container deployment request (auto-create Docker container)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Model ID to deploy")
    deployment_name: Optional[str] = Field(default=None, description="Deployment name")
    gpu_id: Optional[StrictInt] = Field(
        default=None, ge=0, description="GPU device ID (auto-select if not specified)"
    )
    port: Optional[StrictInt] = Field(
        default=None,
        ge=1024,
        le=65535,
        description="Port to expose (auto-assign if not specified)",
    )
    replica: StrictInt = Field(
        default=1, ge=1, le=8, description="Number of replicas (1-8)"
    )
    gpu_memory_utilization: Optional[StrictFloat] = Field(
        default=None,
        ge=0.05,
        le=1.0,
        description="GPU memory utilization (auto-calculated if not specified)",
    )
    inference_framework: str = Field(
        default="xinference",
        pattern="^(vllm|sglang|xinference)$",
        description="Inference framework",
    )
    enable_lora: bool = Field(default=False, description="Enable LoRA hot-loading")
    max_loras: StrictInt = Field(
        default=4, ge=1, le=16, description="Maximum number of LoRA adapters"
    )
    max_lora_rank: StrictInt = Field(
        default=64, ge=8, le=256, description="Maximum LoRA rank"
    )
    external_api_config_id: Optional[str] = Field(
        default=None,
        description="Optional external API config id used as tenant binding",
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional configuration"
    )
    launch_config: Optional[LaunchConfig] = Field(
        default=None,
        description="Typed vLLM or SGLang container launch configuration",
    )
    user_id: Optional[str] = Field(default=None, description="User ID for isolation")
    auto_start: bool = Field(default=True, description="Auto start after creation")

    @model_validator(mode="after")
    def _validate_launch_config(self) -> "CreateContainerDeploymentRequest":
        self.normalized_launch_config()
        return self

    def normalized_launch_config(self) -> LaunchConfig | None:
        return _normalized_request_launch_config(self)


class QuickDeployRequest(BaseModel):
    """Quick deploy request."""

    xinference_endpoint: str = Field(..., description="Xinference server endpoint")
    deployment_name: Optional[str] = Field(default=None, description="Deployment name")
    gpu_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="Preferred GPU device ID for Xinference shared mode",
    )
    replica: int = Field(default=1, ge=1, le=8, description="Number of replicas (1-8)")
    gpu_memory_utilization: Optional[float] = Field(
        default=None,
        ge=0.05,
        le=1.0,
        description="GPU memory utilization (auto-calculated if not specified)",
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional configuration"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for isolation")
    auto_start: bool = Field(default=True, description="Auto start after creation")


class RestartDeploymentRequest(BaseModel):
    """Restart deployment request."""

    mode: str = Field(
        default="auto",
        pattern="^(auto|model|container)$",
        description="Restart mode: auto (recommended), model (Xinference only), container",
    )
    reset_gpu: bool = Field(
        default=False, description="Reset GPU before restart (container mode only)"
    )


class UpdateDeploymentConfigRequest(BaseModel):
    """Update deployment config request."""

    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Deployment config (JSON)"
    )


class DeploymentReplicaResponse(BaseModel):
    """One independently addressable container replica."""

    replica_id: str
    deployment_id: str
    replica_index: int
    endpoint: str
    port: int
    gpu_ids: List[int]
    status: str
    health_status: str
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None


class DeploymentResponse(BaseModel):
    """Deployment response."""

    deployment_id: str
    model_id: str
    model_uid: Optional[str]
    deployment_name: Optional[str]
    xinference_endpoint: str
    replica: int
    gpu_memory_utilization: float  # Configured limit (static)
    gpu_memory_used_mb: Optional[int] = None  # Real-time usage in MB
    gpu_memory_used_percent: Optional[float] = None  # Real-time usage percentage
    deploy_mode: str = "shared"
    container_name: Optional[str] = None
    gpu_id: Optional[int] = None
    port: Optional[int] = None
    inference_framework: str = "xinference"
    enable_lora: bool = False
    max_loras: int = 4
    max_lora_rank: int = 64
    external_api_config_id: Optional[str] = None
    config: Optional[Dict[str, Any]]
    launch_config: Optional[LaunchConfig] = None
    replica_instances: List[DeploymentReplicaResponse] = Field(default_factory=list)
    runtime_info: Optional[Dict[str, Any]] = None
    status: str
    error_message: Optional[str]
    user_id: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    started_at: Optional[str]
    stopped_at: Optional[str]


class DeploymentStats(BaseModel):
    """Status counts for all deployments matching the authorized list query."""

    total: int
    by_status: Dict[str, int]


class DeploymentListResponse(BaseModel):
    """Deployment list response."""

    deployments: List[DeploymentResponse]
    total: int
    stats: DeploymentStats


def _get_xinference_runtime_map(
    deployments: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    runtime_map_by_endpoint: Dict[str, Dict[str, Dict[str, Any]]] = {}
    endpoints: Dict[str, Optional[str]] = {}

    for deployment in deployments:
        deploy_mode = deployment.get("deploy_mode", "shared")
        if deploy_mode == "external":
            deploy_mode = "shared"
        if deploy_mode != "shared":
            continue
        if deployment.get("status") != "running":
            continue
        if deployment.get("inference_framework", "xinference") != "xinference":
            continue
        endpoint = deployment.get("xinference_endpoint")
        if endpoint:
            try:
                endpoint = validate_user_outbound_url(
                    endpoint,
                    deployment.get("user_id"),
                )
            except SSRFError as exc:
                logger.warning(
                    "Blocked unsafe deployment runtime endpoint %s: %s",
                    deployment.get("deployment_id"),
                    exc,
                )
                continue
            endpoints[endpoint] = deployment.get("user_id")

    for endpoint, user_id in endpoints.items():
        try:
            client = deployment_service._get_xinference_client(
                endpoint,
                timeout=8,
                user_id=user_id,
            )
            models = client.list_models()
            model_map: Dict[str, Dict[str, Any]] = {}
            for model in models:
                model_id = (
                    model.get("id") or model.get("model_uid") or model.get("model_id")
                )
                if model_id:
                    model_map[model_id] = model
            if model_map:
                runtime_map_by_endpoint[endpoint] = model_map
        except Exception as e:
            logger.debug(
                f"Failed to fetch Xinference runtime info from {endpoint}: {e}"
            )

    return runtime_map_by_endpoint


class GpuInfoResponse(BaseModel):
    """GPU information response."""

    gpu_id: int
    used_mb: int
    total_mb: int
    free_mb: int
    utilization: float


class GpuListResponse(BaseModel):
    """GPU list response."""

    gpus: List[GpuInfoResponse]


# === Helper Functions ===


def _verify_registered_model_ownership(
    model_id: str,
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    """Load a registered model and enforce tenant ownership before deployment."""
    model = model_registry_service.get_model(model_id)
    model = verify_resource_ownership(model, current_user, "Model")
    if requires_tenant_provenance(current_user):
        try:
            require_managed_model_provenance(
                model,
                model.get("model_path", ""),
                user_id=current_user["user_id"],
                models_dir=Path(get_settings().models_dir),
                training_output_dir=Path(get_settings().output_dir),
                training_task_lookup=training_task_service.get_task,
            )
        except ResourceProvenanceError as exc:
            raise HTTPException(
                status_code=403,
                detail="Model does not have verifiable API-managed provenance",
            ) from exc
    return model


def _validate_deployment_endpoint(
    deployment: Dict[str, Any],
) -> Dict[str, Any]:
    """Revalidate a persisted deployment endpoint immediately before use."""
    validated = dict(deployment)
    validated["xinference_endpoint"] = validate_user_outbound_url(
        deployment.get("xinference_endpoint", ""),
        deployment.get("user_id"),
    )
    return validated


def _deployment_to_response(
    deployment: Dict[str, Any],
    gpu_usage: Optional[Dict[str, Any]] = None,
    model_usage_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    runtime_map_by_endpoint: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> DeploymentResponse:
    """Convert deployment dict to response."""
    # Get real-time GPU usage for running deployments
    gpu_memory_used_mb = None
    gpu_memory_used_percent = None

    deploy_mode = deployment.get("deploy_mode", "shared")
    if deploy_mode == "external":
        deploy_mode = "shared"

    config = deployment.get("config") or {}
    unmanaged_binding = (
        config.get("runtime_managed") is False and config.get("read_only") is True
    )
    container_name = None if unmanaged_binding else deployment.get("container_name")
    effective_gpu_id = deployment.get("gpu_id")
    model_uid = deployment.get("model_uid")
    runtime_info = None

    # For shared Xinference deployments without explicit container_name,
    # use the default shared container name
    if (
        not unmanaged_binding
        and not container_name
        and deployment.get("inference_framework") == "xinference"
        and deploy_mode == "shared"
    ):
        import os

        container_name = os.environ.get("XINFERENCE_CONTAINER_NAME", "xinference")

    if container_name and deployment["status"] == "running":
        usage = None

        # For shared containers, get model-specific GPU usage
        if (
            deploy_mode == "shared"
            and deployment.get("inference_framework") == "xinference"
            and model_uid
            and model_usage_map
        ):
            container_usage = model_usage_map.get(container_name) or {}
            usage = container_usage.get(model_uid)
            if not usage:
                import re

                uid_match = re.search(
                    r"-([0-9a-f]{8})(?:-\d+)?$", model_uid, re.IGNORECASE
                )
                if uid_match:
                    short_uid = uid_match.group(1)
                    usage = container_usage.get(short_uid)
                if not usage:
                    for key, value in container_usage.items():
                        if key.lower().endswith(model_uid.lower()) or (
                            uid_match and key.lower().endswith(short_uid.lower())
                        ):
                            usage = value
                            break
            if usage:
                # Update GPU ID from model-specific usage
                if usage.get("gpu_id") is not None:
                    effective_gpu_id = usage.get("gpu_id")

        # Fallback to container-level usage (non-shared only to avoid overstating per-model)
        if not usage and deploy_mode != "shared":
            if gpu_usage:
                usage = gpu_usage.get(container_name)
            if not usage:
                usage = docker_deployer.get_container_gpu_usage(
                    container_name, gpu_id=effective_gpu_id
                )

        if usage:
            gpu_memory_used_mb = usage.get("gpu_memory_used_mb")
            gpu_memory_used_percent = usage.get("gpu_memory_used_percent")

    if runtime_map_by_endpoint and model_uid:
        endpoint = deployment.get("xinference_endpoint")
        endpoint_map = runtime_map_by_endpoint.get(endpoint) or {}
        runtime_info = endpoint_map.get(model_uid)
        if not runtime_info:
            import re

            uid_match = re.search(r"-([0-9a-f]{8})(?:-\d+)?$", model_uid, re.IGNORECASE)
            if uid_match:
                short_uid = uid_match.group(1)
                runtime_info = endpoint_map.get(short_uid)
            if not runtime_info:
                for key, value in endpoint_map.items():
                    if key.lower().endswith(model_uid.lower()) or (
                        uid_match and key.lower().endswith(short_uid.lower())
                    ):
                        runtime_info = value
                        break
        if runtime_info:
            accelerators = runtime_info.get("accelerators")
            if (
                isinstance(accelerators, list)
                and accelerators
                and effective_gpu_id is None
            ):
                if len(accelerators) == 1:
                    try:
                        effective_gpu_id = int(accelerators[0])
                    except (ValueError, TypeError):
                        pass

    return DeploymentResponse(
        deployment_id=deployment["deployment_id"],
        model_id=deployment["model_id"],
        model_uid=deployment.get("model_uid"),
        deployment_name=deployment.get("deployment_name"),
        xinference_endpoint=deployment["xinference_endpoint"],
        replica=deployment["replica"],
        gpu_memory_utilization=deployment["gpu_memory_utilization"],
        gpu_memory_used_mb=gpu_memory_used_mb,
        gpu_memory_used_percent=gpu_memory_used_percent,
        deploy_mode=deploy_mode,
        container_name=container_name,
        gpu_id=effective_gpu_id,
        port=deployment.get("port"),
        inference_framework=deployment.get("inference_framework", "xinference"),
        enable_lora=deployment.get("enable_lora", False),
        max_loras=deployment.get("max_loras", 4),
        max_lora_rank=deployment.get("max_lora_rank", 64),
        external_api_config_id=deployment.get("external_api_config_id"),
        config=deployment.get("config"),
        launch_config=(deployment.get("config") or {}).get("launch_config"),
        replica_instances=deployment.get("replica_instances") or [],
        runtime_info=runtime_info,
        status=deployment["status"],
        error_message=deployment.get("error_message"),
        user_id=deployment.get("user_id"),
        created_at=deployment["created_at"].isoformat()
        if deployment.get("created_at")
        else None,
        updated_at=deployment["updated_at"].isoformat()
        if deployment.get("updated_at")
        else None,
        started_at=deployment["started_at"].isoformat()
        if deployment.get("started_at")
        else None,
        stopped_at=deployment["stopped_at"].isoformat()
        if deployment.get("stopped_at")
        else None,
    )


# === API Endpoints ===


@router.post("/deployments", response_model=DeploymentResponse)
async def create_deployment(
    request: CreateDeploymentRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Create a new deployment.

    Pass Idempotency-Key header to prevent duplicate creation on retries.
    """
    _verify_registered_model_ownership(request.model_id, current_user)

    try:
        # Use authenticated user_id
        user_id = current_user["user_id"]
        deployment_config = _build_deployment_config_with_external_api(
            request.config,
            request.external_api_config_id,
            current_user,
        )
        deployment_config = _attach_launch_config(
            deployment_config,
            request.normalized_launch_config(),
        )
        resolved_external_api_config_id = (
            request.external_api_config_id or ""
        ).strip() or (deployment_config or {}).get("external_api_config_id")

        # Check idempotency
        is_duplicate, cached_response = check_idempotency(
            idempotency_key, user_id, "/api/deployments"
        )
        if is_duplicate and cached_response:
            _reschedule_cached_deferred_start(cached_response)
            return DeploymentResponse(**cached_response)

        # vLLM and SGLang require container mode (独立容器)
        # Xinference can use shared mode (共享容器)
        if request.inference_framework in ("vllm", "sglang"):
            deployment = await asyncio.to_thread(
                deployment_service.create_container_deployment,
                model_id=request.model_id,
                deployment_name=request.deployment_name,
                gpu_id=request.gpu_id,
                port=None,  # Auto-assign port
                replica=request.replica,
                gpu_memory_utilization=request.gpu_memory_utilization,
                inference_framework=request.inference_framework,
                enable_lora=request.enable_lora,
                max_loras=request.max_loras,
                max_lora_rank=request.max_lora_rank,
                external_api_config_id=resolved_external_api_config_id,
                config=deployment_config,
                user_id=user_id,
                auto_start=request.auto_start,
                defer_start=request.auto_start,
            )
            if request.auto_start:
                background_tasks.add_task(
                    _run_registered_deferred_deployment_start,
                    deployment["deployment_id"],
                )
        else:
            # Use default endpoint if not specified
            endpoint = request.xinference_endpoint or get_default_xinference_endpoint()
            endpoint = validate_user_outbound_url(endpoint, user_id)
            deployment = await asyncio.to_thread(
                deployment_service.create_deployment,
                model_id=request.model_id,
                xinference_endpoint=endpoint,
                deployment_name=request.deployment_name,
                gpu_id=request.gpu_id,
                replica=request.replica,
                gpu_memory_utilization=request.gpu_memory_utilization,
                inference_framework=request.inference_framework,
                enable_lora=request.enable_lora,
                max_loras=request.max_loras,
                max_lora_rank=request.max_lora_rank,
                external_api_config_id=resolved_external_api_config_id,
                config=deployment_config,
                user_id=user_id,
            )
            if request.auto_start:
                background_tasks.add_task(
                    _start_deployment_in_background,
                    deployment["deployment_id"],
                )
        response = _deployment_to_response(deployment)

        # Store for idempotency
        store_idempotency_response(
            idempotency_key, user_id, "/api/deployments", response.model_dump()
        )

        return response
    except ReplicaOperationBusyError:
        raise HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/deployments", response_model=DeploymentListResponse)
def list_deployments(
    model_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    sync: bool = Query(
        default=False, description="Sync status with actual service before returning"
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all deployments for the current user with optional filters."""
    # Use authenticated user_id
    user_id = current_user["user_id"]

    deployments, total, stats = deployment_service.list_deployments_with_stats(
        model_id=model_id,
        status=status,
        user_id=user_id,
        limit=limit,
        offset=offset,
        sync=False,
    )

    if sync:
        for deployment in deployments:
            _validate_deployment_endpoint(deployment)
            deployment_service.sync_status(deployment["deployment_id"])
        deployments, total, stats = deployment_service.list_deployments_with_stats(
            model_id=model_id,
            status=status,
            user_id=user_id,
            limit=limit,
            offset=offset,
            sync=False,
        )

    # Batch fetch GPU usage for all containers
    gpu_usage = docker_deployer.get_all_container_gpu_usage()
    model_usage_map: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # For shared Xinference deployments, map models to GPU IDs and collect per-GPU usage
    shared_xinference = [
        d
        for d in deployments
        if d.get("deploy_mode") in ("shared", "external")
        and d.get("inference_framework", "xinference") == "xinference"
    ]
    if shared_xinference:
        import os

        shared_name = os.environ.get("XINFERENCE_CONTAINER_NAME", "xinference")
        containers = set()
        for d in shared_xinference:
            containers.add(d.get("container_name") or shared_name)

        for container_name in containers:
            model_usage_map[container_name] = (
                docker_deployer.get_container_model_gpu_usage(container_name)
            )

    runtime_map_by_endpoint = (
        _get_xinference_runtime_map(shared_xinference) if shared_xinference else {}
    )

    return DeploymentListResponse(
        deployments=[
            _deployment_to_response(
                d, gpu_usage, model_usage_map, runtime_map_by_endpoint
            )
            for d in deployments
        ],
        total=total,
        stats=stats,
    )


# === Discover Models ===
# NOTE: These routes MUST be defined before /deployments/{deployment_id}
# to prevent the path parameter from matching "discover-models" as a deployment_id.


class DiscoveredModel(BaseModel):
    """Discovered model from inference endpoint."""

    model_uid: str = Field(..., description="Model UID/ID")
    model_name: Optional[str] = Field(default=None, description="Model name")
    model_type: Optional[str] = Field(
        default=None, description="Model type (embedding/rerank/llm)"
    )
    model_path: Optional[str] = Field(default=None, description="Model path")
    status: Optional[str] = Field(default=None, description="Model status")


class DiscoverModelsResponse(BaseModel):
    """Discover models response."""

    endpoint: str
    framework: str
    models: List[DiscoveredModel]


def _validate_discovery_endpoint(endpoint: str, user_id: str) -> str:
    """Validate discovery endpoint and block sensitive SSRF targets.

    Thin wrapper over the shared :func:`train_factory.core.ssrf.validate_outbound_url`
    that injects this user's per-deployment private-host allowlist and maps
    :class:`SSRFError` to the appropriate HTTP status.
    """
    try:
        return validate_user_outbound_url(endpoint, user_id)
    except SSRFError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e


@router.get("/discover-models", response_model=DiscoverModelsResponse)
@router.get("/deployments/discover-models", response_model=DiscoverModelsResponse)
def discover_models(
    endpoint: str = Query(..., description="Inference server endpoint"),
    framework: str = Query(
        default="xinference",
        pattern="^(xinference|vllm|sglang)$",
        description="Inference framework",
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Discover models running on an inference endpoint.

    Supports Xinference, vLLM, and SGLang endpoints.
    Returns a list of models currently loaded on the server.
    """
    import requests
    from urllib.parse import quote

    endpoint = _validate_discovery_endpoint(endpoint, current_user["user_id"])
    user_id = current_user["user_id"]
    restrict_to_owned = bool(
        get_settings().auth_enabled and not current_user.get("is_admin")
    )
    owned_model_uids: set[str] = set()
    if restrict_to_owned:
        deployments, _ = deployment_service.list_deployments(
            user_id=user_id,
            limit=1000,
        )
        normalized_endpoint = endpoint.rstrip("/")
        owned_model_uids = {
            deployment.get("model_uid")
            for deployment in deployments
            if deployment.get("model_uid")
            and (deployment.get("xinference_endpoint") or "").rstrip("/")
            == normalized_endpoint
            and (deployment.get("inference_framework") or "xinference") == framework
            and deployment.get("status") == "running"
        }
        if not owned_model_uids:
            return DiscoverModelsResponse(
                endpoint=endpoint,
                framework=framework,
                models=[],
            )
    models = []

    try:
        if framework == "xinference":
            # Xinference: GET /v1/models returns list of models
            response = request_user_outbound(
                "GET", f"{endpoint}/v1/models", user_id, timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                # Xinference returns {"data": [{"id": "...", "object": "model", ...}]}
                for model in data.get("data", []):
                    model_uid = model.get("id", "")
                    if restrict_to_owned:
                        if model_uid not in owned_model_uids:
                            continue
                        models.append(
                            DiscoveredModel(
                                model_uid=model_uid,
                                model_name=model.get("model_name", model_uid),
                                model_type=model.get("model_type"),
                                status="running",
                            )
                        )
                        continue
                    # Get detailed model info
                    try:
                        detail_resp = request_user_outbound(
                            "GET",
                            f"{endpoint}/v1/models/{quote(model_uid, safe='')}",
                            user_id,
                            timeout=5,
                        )
                        if detail_resp.status_code == 200:
                            detail = detail_resp.json()
                            models.append(
                                DiscoveredModel(
                                    model_uid=model_uid,
                                    model_name=detail.get("model_name", model_uid),
                                    model_type=detail.get("model_type", "unknown"),
                                    model_path=detail.get("model_path"),
                                    status="running",
                                )
                            )
                        else:
                            models.append(
                                DiscoveredModel(
                                    model_uid=model_uid,
                                    model_name=model_uid,
                                    status="running",
                                )
                            )
                    except Exception:
                        models.append(
                            DiscoveredModel(
                                model_uid=model_uid,
                                model_name=model_uid,
                                status="running",
                            )
                        )

        elif framework == "vllm":
            # vLLM: GET /v1/models returns loaded model(s)
            response = request_user_outbound(
                "GET", f"{endpoint}/v1/models", user_id, timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if restrict_to_owned and model_id not in owned_model_uids:
                        continue
                    models.append(
                        DiscoveredModel(
                            model_uid=model_id,
                            model_name=model_id,
                            model_type="llm",  # vLLM primarily for LLM
                            status="running",
                        )
                    )

        elif framework == "sglang":
            # SGLang: GET /get_model_info or /v1/models
            if restrict_to_owned:
                response = request_user_outbound(
                    "GET", f"{endpoint}/v1/models", user_id, timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("data", []):
                        model_id = model.get("id", "")
                        if model_id not in owned_model_uids:
                            continue
                        models.append(
                            DiscoveredModel(
                                model_uid=model_id,
                                model_name=model_id,
                                status="running",
                            )
                        )
                return DiscoverModelsResponse(
                    endpoint=endpoint,
                    framework=framework,
                    models=models,
                )
            try:
                response = request_user_outbound(
                    "GET", f"{endpoint}/get_model_info", user_id, timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    model_path = data.get("model_path", "")
                    model_name = (
                        model_path.split("/")[-1] if model_path else "sglang-model"
                    )
                    models.append(
                        DiscoveredModel(
                            model_uid=model_name,
                            model_name=model_name,
                            model_path=model_path,
                            status="running",
                        )
                    )
            except Exception:
                # Fallback to /v1/models
                response = request_user_outbound(
                    "GET", f"{endpoint}/v1/models", user_id, timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("data", []):
                        model_id = model.get("id", "")
                        models.append(
                            DiscoveredModel(
                                model_uid=model_id,
                                model_name=model_id,
                                status="running",
                            )
                        )

        return DiscoverModelsResponse(
            endpoint=endpoint,
            framework=framework,
            models=models,
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail=f"无法连接到 {endpoint}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail=f"连接 {endpoint} 超时")
    except Exception as e:
        logger.error(f"Failed to discover models from {endpoint}: {e}")
        raise HTTPException(status_code=500, detail=f"发现模型失败: {str(e)}")


# === Bind Existing Model ===
# NOTE: Must be defined before /deployments/{deployment_id} routes.


class BindExistingModelRequest(BaseModel):
    """Bind existing model request."""

    endpoint: str = Field(..., description="Inference server endpoint")
    model_uid: str = Field(..., description="Model UID on the server")
    model_name: Optional[str] = Field(default=None, description="Display name")
    model_type: str = Field(
        default="embedding", description="Model type: embedding/rerank/llm"
    )
    deployment_name: Optional[str] = Field(default=None, description="Deployment name")
    inference_framework: str = Field(
        default="xinference", pattern="^(vllm|sglang|xinference)$"
    )
    container_name: Optional[str] = Field(
        default=None, description="Docker container name (for local deployments)"
    )
    gpu_id: Optional[int] = Field(
        default=None, description="GPU device ID (for local deployments)"
    )
    external_api_config_id: Optional[str] = Field(
        default=None,
        description="Optional external API config id used as tenant binding",
    )


@router.post("/bind-existing", response_model=DeploymentResponse)
@router.post("/deployments/bind-existing", response_model=DeploymentResponse)
async def bind_existing_model(
    request: BindExistingModelRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Bind an existing model on an inference endpoint.

    Creates a deployment record for a model that's already running on the server,
    without launching a new model. Useful for connecting to shared/managed
    inference services.
    """
    if get_settings().auth_enabled and not current_user.get("is_admin"):
        raise HTTPException(
            status_code=403,
            detail="Administrator access required to bind external models",
        )

    try:
        user_id = current_user["user_id"]
        endpoint = validate_user_outbound_url(request.endpoint, user_id)
        deployment_config = _build_deployment_config_with_external_api(
            None,
            request.external_api_config_id,
            current_user,
        )

        deployment = await asyncio.to_thread(
            deployment_service.bind_existing_model,
            endpoint=endpoint,
            model_uid=request.model_uid,
            model_name=request.model_name,
            model_type=request.model_type,
            deployment_name=request.deployment_name,
            inference_framework=request.inference_framework,
            container_name=None,
            gpu_id=None,
            user_id=user_id,
            config=deployment_config,
        )
        return _deployment_to_response(deployment)
    except SSRFError:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to bind existing model: {e}")
        raise HTTPException(status_code=500, detail=f"绑定模型失败: {str(e)}")


def _replica_route_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ReplicaOperationBusyError):
        return HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        )
    if isinstance(exc, DeploymentReplicaNotFoundError):
        return HTTPException(
            status_code=404,
            detail="Deployment replica not found",
        )
    if isinstance(
        exc,
        (DeploymentReplicaStateConflictError, ReplicaOperationLostError),
    ):
        return HTTPException(
            status_code=409,
            detail="Deployment replica state conflicts with requested operation",
        )
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=400,
            detail="Invalid deployment replica request",
        )
    return HTTPException(
        status_code=500,
        detail="Deployment replica operation failed",
    )


@router.get(
    "/deployments/{deployment_id}/replicas",
    response_model=List[DeploymentReplicaResponse],
)
async def list_deployment_replicas(
    deployment_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List independently addressable replicas owned by the current user."""
    try:
        replicas = await asyncio.to_thread(
            deployment_service.list_replicas,
            str(deployment_id),
            user_id=current_user["user_id"],
        )
        return [DeploymentReplicaResponse(**replica) for replica in replicas]
    except Exception as exc:
        raise _replica_route_error(exc) from None


async def _run_replica_action(
    action: str,
    deployment_id: UUID,
    replica_id: UUID,
    current_user: Dict[str, Any],
) -> DeploymentReplicaResponse:
    try:
        method = {
            "start": deployment_service.start_replica,
            "stop": deployment_service.stop_replica,
            "restart": deployment_service.restart_replica,
            "recreate": deployment_service.recreate_replica,
        }[action]
        replica = await asyncio.to_thread(
            method,
            str(deployment_id),
            str(replica_id),
            user_id=current_user["user_id"],
        )
        return DeploymentReplicaResponse(**replica)
    except Exception as exc:
        raise _replica_route_error(exc) from None


@router.post(
    "/deployments/{deployment_id}/replicas/{replica_id}/start",
    response_model=DeploymentReplicaResponse,
)
async def start_deployment_replica(
    deployment_id: UUID,
    replica_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return await _run_replica_action(
        "start",
        deployment_id,
        replica_id,
        current_user,
    )


@router.post(
    "/deployments/{deployment_id}/replicas/{replica_id}/stop",
    response_model=DeploymentReplicaResponse,
)
async def stop_deployment_replica(
    deployment_id: UUID,
    replica_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return await _run_replica_action(
        "stop",
        deployment_id,
        replica_id,
        current_user,
    )


@router.post(
    "/deployments/{deployment_id}/replicas/{replica_id}/restart",
    response_model=DeploymentReplicaResponse,
)
async def restart_deployment_replica(
    deployment_id: UUID,
    replica_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return await _run_replica_action(
        "restart",
        deployment_id,
        replica_id,
        current_user,
    )


@router.post(
    "/deployments/{deployment_id}/replicas/{replica_id}/recreate",
    response_model=DeploymentReplicaResponse,
)
async def recreate_deployment_replica(
    deployment_id: UUID,
    replica_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return await _run_replica_action(
        "recreate",
        deployment_id,
        replica_id,
        current_user,
    )


# === Individual Deployment Routes ===
# NOTE: {deployment_id} routes must come AFTER all fixed-path /deployments/* routes above.


@router.get("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get deployment by ID."""
    deployment = deployment_service.get_deployment(deployment_id)
    deployment = verify_resource_ownership(deployment, current_user, "Deployment")
    runtime_map_by_endpoint = _get_xinference_runtime_map([deployment])
    return _deployment_to_response(
        deployment, runtime_map_by_endpoint=runtime_map_by_endpoint
    )


@router.post("/deployments/{deployment_id}/start", response_model=DeploymentResponse)
async def start_deployment(
    deployment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Start a deployment."""
    try:
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")
        _validate_deployment_endpoint(deployment)
        deployment = await asyncio.to_thread(
            deployment_service.start_deployment,
            deployment_id,
            user_id=current_user["user_id"],
        )
        return _deployment_to_response(deployment)
    except SSRFError:
        raise
    except ReplicaOperationBusyError:
        raise HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start deployment: {e}")


@router.post("/deployments/{deployment_id}/stop", response_model=DeploymentResponse)
async def stop_deployment(
    deployment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Stop a running deployment."""
    try:
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")
        _validate_deployment_endpoint(deployment)
        deployment = await asyncio.to_thread(
            deployment_service.stop_deployment,
            deployment_id,
            user_id=current_user["user_id"],
        )
        return _deployment_to_response(deployment)
    except ReplicaOperationBusyError:
        raise HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/deployments/{deployment_id}/restart", response_model=DeploymentResponse)
async def restart_deployment(
    deployment_id: str,
    request: Optional[RestartDeploymentRequest] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Restart a deployment to clear GPU memory cache.

    Different frameworks have different restart behaviors:
    - vLLM/SGLang (single-model container): Always restarts container
    - Xinference (multi-model shared): Can reload model only or restart container

    Modes:
    - auto (default): Auto-select best strategy based on framework
        - vLLM/SGLang → restart container
        - Xinference → reload model (preserves other models)
    - model: Reload model only (Xinference only)
    - container: Force restart entire container
    """
    try:
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")
        _validate_deployment_endpoint(deployment)
        mode = request.mode if request else "auto"
        reset_gpu = request.reset_gpu if request else False
        deployment = await asyncio.to_thread(
            deployment_service.restart_deployment,
            deployment_id,
            mode=mode,
            reset_gpu=reset_gpu,
            user_id=current_user["user_id"],
        )
        return _deployment_to_response(deployment)
    except SSRFError:
        raise
    except ReplicaOperationBusyError:
        raise HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to restart deployment: {e}"
        )


@router.patch("/deployments/{deployment_id}/config", response_model=DeploymentResponse)
async def update_deployment_config(
    deployment_id: str,
    request: UpdateDeploymentConfigRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Update deployment config (takes effect after restart)."""
    try:
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")
        request_config = request.config
        has_external_api_config_key = (
            isinstance(request_config, dict)
            and "external_api_config_id" in request_config
        )
        normalized_config = _build_deployment_config_with_external_api(
            request_config,
            None
            if has_external_api_config_key
            else deployment.get("external_api_config_id"),
            current_user,
        )
        deployment = deployment_service.update_deployment_config(
            deployment_id,
            normalized_config,
            user_id=current_user["user_id"],
        )
        return _deployment_to_response(deployment)
    except ReplicaOperationBusyError:
        raise HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update deployment config: {e}"
        )


@router.post("/deployments/{deployment_id}/sync", response_model=DeploymentResponse)
async def sync_deployment_status(
    deployment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Sync deployment status with Xinference."""
    try:
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")
        _validate_deployment_endpoint(deployment)
        deployment = await asyncio.to_thread(
            deployment_service.sync_status,
            deployment_id,
        )
        return _deployment_to_response(deployment)
    except DeploymentReplicaNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReplicaOperationBusyError:
        raise HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/deployments/{deployment_id}")
async def delete_deployment(
    deployment_id: str,
    force: bool = False,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a deployment.

    Args:
        deployment_id: The deployment ID to delete
        force: If True, also delete related model configs
    """
    try:
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")
        _validate_deployment_endpoint(deployment)
        success = await asyncio.to_thread(
            deployment_service.delete_deployment,
            deployment_id,
            force=force,
            user_id=current_user["user_id"],
        )
        if not success:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete deployment: {deployment_id}"
            )
        return {"message": f"Deployment {deployment_id} deleted"}
    except ReplicaOperationBusyError:
        raise HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# === Quick Deploy ===


@router.post("/deployments/from-model/{model_id}", response_model=DeploymentResponse)
async def quick_deploy(
    model_id: str,
    request: QuickDeployRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Quick deploy a registered model to shared Xinference."""
    _verify_registered_model_ownership(model_id, current_user)

    try:
        # Use authenticated user_id
        user_id = current_user["user_id"]
        endpoint = validate_user_outbound_url(
            request.xinference_endpoint,
            user_id,
        )

        deployment = await asyncio.to_thread(
            deployment_service.deploy_from_model,
            model_id=model_id,
            xinference_endpoint=endpoint,
            deployment_name=request.deployment_name,
            gpu_id=request.gpu_id,
            replica=request.replica,
            gpu_memory_utilization=request.gpu_memory_utilization,
            config=request.config,
            user_id=user_id,
            auto_start=request.auto_start,
        )
        return _deployment_to_response(deployment)
    except SSRFError:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {e}")


# === Container Deploy ===


@router.post("/deployments/container", response_model=DeploymentResponse)
async def create_container_deployment(
    request: CreateContainerDeploymentRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Create a container-based deployment.

    This will auto-create a Docker container running Xinference and deploy the model.
    If gpu_id is not specified, the GPU with most free memory will be selected.
    If port is not specified, an available port will be assigned.

    Pass Idempotency-Key header to prevent duplicate creation on retries.
    """
    _verify_registered_model_ownership(request.model_id, current_user)

    try:
        # Use authenticated user_id
        user_id = current_user["user_id"]
        deployment_config = _build_deployment_config_with_external_api(
            request.config,
            request.external_api_config_id,
            current_user,
        )
        deployment_config = _attach_launch_config(
            deployment_config,
            request.normalized_launch_config(),
        )
        resolved_external_api_config_id = (
            request.external_api_config_id or ""
        ).strip() or (deployment_config or {}).get("external_api_config_id")

        # Check idempotency
        is_duplicate, cached_response = check_idempotency(
            idempotency_key, user_id, "/api/deployments/container"
        )
        if is_duplicate and cached_response:
            _reschedule_cached_deferred_start(cached_response)
            return DeploymentResponse(**cached_response)

        defer_start = request.auto_start and request.inference_framework in (
            "vllm",
            "sglang",
        )
        deployment = await asyncio.to_thread(
            deployment_service.create_container_deployment,
            model_id=request.model_id,
            deployment_name=request.deployment_name,
            gpu_id=request.gpu_id,
            port=request.port,
            replica=request.replica,
            gpu_memory_utilization=request.gpu_memory_utilization,
            inference_framework=request.inference_framework,
            enable_lora=request.enable_lora,
            max_loras=request.max_loras,
            max_lora_rank=request.max_lora_rank,
            external_api_config_id=resolved_external_api_config_id,
            config=deployment_config,
            user_id=user_id,
            auto_start=request.auto_start,
            defer_start=defer_start,
        )
        if defer_start:
            background_tasks.add_task(
                _run_registered_deferred_deployment_start,
                deployment["deployment_id"],
            )
        response = _deployment_to_response(deployment)

        # Store for idempotency
        store_idempotency_response(
            idempotency_key,
            user_id,
            "/api/deployments/container",
            response.model_dump(),
        )

        return response
    except ReplicaOperationBusyError:
        raise HTTPException(
            status_code=409,
            detail="Deployment replica operation in progress",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Container deployment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Container deployment failed: {e}")


# === Create Config for Deployment ===


@router.post("/deployments/{deployment_id}/create-config")
async def create_config_for_deployment(
    deployment_id: str,
    replica_id: Optional[UUID] = Query(default=None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Create a model config for an existing deployment.

    Use this endpoint when the automatic config creation failed during deployment
    (e.g., due to health check timeout) but the deployment is actually running.
    """
    try:
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")
        _validate_deployment_endpoint(deployment)
        config = deployment_service.create_config_for_deployment(
            deployment_id,
            deployment_replica_id=str(replica_id) if replica_id else None,
            user_id=current_user["user_id"],
        )
        return {
            "message": f"Config created for deployment {deployment_id}",
            "config": config,
        }
    except SSRFError:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create config for deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create config: {e}")


# === GPU Info ===


@router.get("/gpus", response_model=GpuListResponse)
async def get_gpu_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get GPU memory usage information."""
    try:
        gpu_info = docker_deployer.get_gpu_memory_usage()
        gpus = [
            GpuInfoResponse(
                gpu_id=gpu_id,
                used_mb=info["used_mb"],
                total_mb=info["total_mb"],
                free_mb=info["free_mb"],
                utilization=info["utilization"],
            )
            for gpu_id, info in sorted(gpu_info.items())
        ]
        return GpuListResponse(gpus=gpus)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get GPU info: {e}")


# === Default Endpoint ===


@router.get("/default-endpoint")
async def get_default_endpoint(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get the default Xinference endpoint from configuration."""
    return {"endpoint": get_default_xinference_endpoint()}
