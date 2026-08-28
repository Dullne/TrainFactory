"""
Adapter API routes.

Provides endpoints for managing LoRA adapters on deployments.
"""

import logging
import posixpath
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...auth.dependencies import (
    get_current_user,
    requires_tenant_provenance,
    verify_resource_ownership,
)
from ...deployment.adapter_service import adapter_service
from ...deployment.deployment_service import deployment_service
from ...deployment.deployment_service import (
    DeploymentReplicaNotFoundError,
    ReplicaOperationBusyError,
    ReplicaOperationLostError,
)
from ...storage.services.model_registry_service import model_registry_service
from ...storage.services.training_task_service import training_task_service
from ...storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    background_task_admission_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


_ADAPTER_ROUTE_ERRORS = (
    DeploymentReplicaNotFoundError,
    ReplicaOperationBusyError,
    ReplicaOperationLostError,
    ValueError,
)


def _adapter_route_http_exception(exc: Exception) -> HTTPException:
    """Map adapter lifecycle errors without collapsing not-found into 400."""
    if isinstance(exc, DeploymentReplicaNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ReplicaOperationBusyError, ReplicaOperationLostError)):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


# === Request/Response Models ===

class LoadAdapterRequest(BaseModel):
    """Load adapter request."""
    adapter_name: str = Field(..., description="Unique name for the adapter (used in inference)")
    adapter_path: str = Field(..., description="Path to adapter weights")
    source_task_id: Optional[str] = Field(default=None, description="Training task that created this adapter")
    source_model_id: Optional[str] = Field(default=None, description="Model registry ID if adapter is registered")
    replica_id: Optional[UUID] = Field(
        default=None,
        description="Target deployment replica; required for multi-replica deployments",
    )


class LoadAdapterFromTaskRequest(BaseModel):
    """Load adapter from training task request."""
    task_id: str = Field(..., description="Training task ID")
    adapter_name: Optional[str] = Field(default=None, description="Custom name for the adapter (auto-generated if not specified)")
    replica_id: Optional[UUID] = Field(
        default=None,
        description="Target deployment replica; required for multi-replica deployments",
    )


class AdapterResponse(BaseModel):
    """Adapter response."""
    adapter_id: str
    deployment_id: str
    deployment_replica_id: Optional[str]
    adapter_name: str
    adapter_path: str
    source_task_id: Optional[str]
    source_model_id: Optional[str]
    status: str
    error_message: Optional[str]
    user_id: Optional[str]
    loaded_at: Optional[str]
    unloaded_at: Optional[str]


class AdapterListResponse(BaseModel):
    """Adapter list response."""
    adapters: List[AdapterResponse]


class AvailableAdapterResponse(BaseModel):
    """Available adapter response."""
    source: str  # training_task or model_registry
    source_id: str
    name: str
    path: str
    base_model_id: Optional[str]
    created_at: Optional[str]
    description: Optional[str]


class AvailableAdapterListResponse(BaseModel):
    """Available adapter list response."""
    adapters: List[AvailableAdapterResponse]


# === Helper Functions ===

def _adapter_to_response(adapter: Dict[str, Any]) -> AdapterResponse:
    """Convert adapter dict to response."""
    return AdapterResponse(
        adapter_id=adapter['adapter_id'],
        deployment_id=adapter['deployment_id'],
        deployment_replica_id=adapter.get('deployment_replica_id'),
        adapter_name=adapter['adapter_name'],
        adapter_path=adapter['adapter_path'],
        source_task_id=adapter.get('source_task_id'),
        source_model_id=adapter.get('source_model_id'),
        status=adapter['status'],
        error_message=adapter.get('error_message'),
        user_id=adapter.get('user_id'),
        loaded_at=adapter.get('loaded_at'),
        unloaded_at=adapter.get('unloaded_at'),
    )


def _normalize_adapter_path(path: str) -> str:
    """Normalize container paths before comparing a request with its resource."""
    return posixpath.normpath(path.strip().replace("\\", "/"))


def _require_matching_adapter_path(requested_path: str, owned_path: str) -> str:
    if _normalize_adapter_path(requested_path) != _normalize_adapter_path(owned_path):
        raise ValueError("Adapter path does not match the referenced resource")
    return owned_path


def _resolve_owned_task_adapter_path(
    task_id: str,
    current_user: Dict[str, Any],
) -> str:
    task = training_task_service.get_task(task_id)
    task = verify_resource_ownership(task, current_user, "Training task")

    if not task.get("is_lora"):
        raise ValueError(f"Training task {task_id} is not a LoRA task")
    if task.get("status") != "succeeded":
        raise ValueError(f"Training task {task_id} has not completed successfully")

    adapter_path = task.get("final_model_path")
    if not adapter_path:
        raise ValueError(f"Training task {task_id} has no output model path")
    return adapter_path


def _resolve_owned_model_adapter_path(
    model_id: str,
    current_user: Dict[str, Any],
) -> str:
    model = model_registry_service.get_model(model_id)
    model = verify_resource_ownership(model, current_user, "Model")

    if not model.get("is_adapter"):
        raise ValueError(f"Model {model_id} is not an adapter")

    adapter_path = model.get("model_path")
    if not adapter_path:
        raise ValueError(f"Model {model_id} has no model path")
    if requires_tenant_provenance(current_user):
        source_task_id = model.get("source_task_id")
        if not source_task_id:
            raise ValueError(
                "Adapter model does not have verifiable training-task provenance"
            )
        task_path = _resolve_owned_task_adapter_path(
            source_task_id,
            current_user,
        )
        _require_matching_adapter_path(adapter_path, task_path)
    return adapter_path


def _resolve_direct_adapter_path(
    request: LoadAdapterRequest,
    current_user: Dict[str, Any],
) -> str:
    source_count = sum(
        source_id is not None
        for source_id in (request.source_task_id, request.source_model_id)
    )
    if source_count != 1:
        raise ValueError(
            "Exactly one of source_task_id or source_model_id is required"
        )

    if request.source_task_id:
        owned_path = _resolve_owned_task_adapter_path(
            request.source_task_id,
            current_user,
        )
    else:
        owned_path = _resolve_owned_model_adapter_path(
            request.source_model_id or "",
            current_user,
        )

    return _require_matching_adapter_path(request.adapter_path, owned_path)


def _begin_training_artifact_guard(task_id: str):
    try:
        return background_task_admission_service.begin_deletion(
            "training",
            task_id,
        )
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(
            status_code=409,
            detail="Training task is executing or being deleted",
        ) from exc


# === API Endpoints ===

@router.post("/deployments/{deployment_id}/adapters", response_model=AdapterResponse)
async def load_adapter(
    deployment_id: str,
    request: LoadAdapterRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Load a LoRA adapter onto a deployment.

    The deployment must:
    - Be running
    - Have LoRA enabled (enable_lora=True)
    - Use vLLM or SGLang framework (Xinference doesn't support hot-loading)
    """
    task_guard = None
    try:
        # Verify deployment ownership
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")

        if request.source_task_id:
            task_guard = _begin_training_artifact_guard(request.source_task_id)
        adapter_path = _resolve_direct_adapter_path(request, current_user)
        user_id = current_user["user_id"]
        adapter = adapter_service.load_adapter(
            deployment_id=deployment_id,
            adapter_name=request.adapter_name,
            adapter_path=adapter_path,
            source_task_id=request.source_task_id,
            source_model_id=request.source_model_id,
            deployment_replica_id=(
                str(request.replica_id) if request.replica_id is not None else None
            ),
            user_id=user_id,
        )
        return _adapter_to_response(adapter)
    except _ADAPTER_ROUTE_ERRORS as exc:
        raise _adapter_route_http_exception(exc) from exc
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if task_guard is not None:
            task_guard.release()


@router.post("/deployments/{deployment_id}/adapters/from-task", response_model=AdapterResponse)
async def load_adapter_from_task(
    deployment_id: str,
    request: LoadAdapterFromTaskRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Load a LoRA adapter from a training task.

    Convenience endpoint that looks up the adapter path from the training task.
    """
    task_guard = None
    try:
        # Verify deployment ownership
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")

        task_guard = _begin_training_artifact_guard(request.task_id)
        adapter_path = _resolve_owned_task_adapter_path(request.task_id, current_user)

        # Generate adapter name if not specified
        adapter_name = request.adapter_name or f"task-{request.task_id[:8]}"

        user_id = current_user["user_id"]
        adapter = adapter_service.load_adapter(
            deployment_id=deployment_id,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
            source_task_id=request.task_id,
            deployment_replica_id=(
                str(request.replica_id) if request.replica_id is not None else None
            ),
            user_id=user_id,
        )
        return _adapter_to_response(adapter)
    except _ADAPTER_ROUTE_ERRORS as exc:
        raise _adapter_route_http_exception(exc) from exc
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if task_guard is not None:
            task_guard.release()


@router.delete("/deployments/{deployment_id}/adapters/{adapter_name}")
async def unload_adapter(
    deployment_id: str,
    adapter_name: str,
    replica_id: Optional[UUID] = Query(
        default=None,
        description="Target deployment replica; required for multi-replica deployments",
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Unload a LoRA adapter from a deployment."""
    try:
        # Verify deployment ownership
        deployment = deployment_service.get_deployment(deployment_id)
        deployment = verify_resource_ownership(deployment, current_user, "Deployment")

        success = adapter_service.unload_adapter(
            deployment_id,
            adapter_name,
            deployment_replica_id=(
                str(replica_id) if replica_id is not None else None
            ),
            user_id=current_user["user_id"],
        )
        if success:
            return {"message": f"Adapter '{adapter_name}' unloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to unload adapter")
    except _ADAPTER_ROUTE_ERRORS as exc:
        raise _adapter_route_http_exception(exc) from exc
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}/adapters", response_model=AdapterListResponse)
async def list_loaded_adapters(
    deployment_id: str,
    include_unloaded: bool = Query(default=False, description="Include unloaded/failed adapters"),
    replica_id: Optional[UUID] = Query(
        default=None,
        description="Target deployment replica; required for multi-replica deployments",
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List adapters loaded on a deployment."""
    # Verify deployment ownership
    deployment = deployment_service.get_deployment(deployment_id)
    deployment = verify_resource_ownership(deployment, current_user, "Deployment")

    try:
        adapters = adapter_service.list_loaded_adapters(
            deployment_id,
            include_unloaded,
            deployment_replica_id=(
                str(replica_id) if replica_id is not None else None
            ),
            user_id=current_user["user_id"],
        )
    except _ADAPTER_ROUTE_ERRORS as exc:
        raise _adapter_route_http_exception(exc) from exc
    return AdapterListResponse(
        adapters=[_adapter_to_response(a) for a in adapters]
    )


@router.post("/deployments/{deployment_id}/adapters/sync")
async def sync_loaded_adapters(
    deployment_id: str,
    replica_id: Optional[UUID] = Query(
        default=None,
        description="Target deployment replica; required for multi-replica deployments",
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Sync loaded adapters with actual state from inference service."""
    # Verify deployment ownership
    deployment = deployment_service.get_deployment(deployment_id)
    deployment = verify_resource_ownership(deployment, current_user, "Deployment")

    try:
        updated = adapter_service.sync_loaded_adapters(
            deployment_id,
            deployment_replica_id=(
                str(replica_id) if replica_id is not None else None
            ),
            user_id=current_user["user_id"],
        )
    except _ADAPTER_ROUTE_ERRORS as exc:
        raise _adapter_route_http_exception(exc) from exc
    return {"message": f"Synced adapters, {updated} updated"}


@router.get("/adapters/available", response_model=AvailableAdapterListResponse)
async def list_available_adapters(
    base_model_id: Optional[str] = Query(default=None, description="Filter by base model ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    List available adapters that can be loaded.

    Searches:
    - Completed LoRA training tasks
    - Model registry entries marked as adapters
    """
    user_id = current_user["user_id"]
    adapters = adapter_service.get_available_adapters(base_model_id, user_id)
    return AvailableAdapterListResponse(
        adapters=[AvailableAdapterResponse(**a) for a in adapters]
    )


@router.get("/adapters/{adapter_id}", response_model=AdapterResponse)
async def get_adapter(
    adapter_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get adapter by ID."""
    adapter = adapter_service.get_adapter(adapter_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_id}")

    # Verify ownership through deployment
    deployment = deployment_service.get_deployment(adapter['deployment_id'])
    deployment = verify_resource_ownership(deployment, current_user, "Deployment")

    return _adapter_to_response(adapter)
