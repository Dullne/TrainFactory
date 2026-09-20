"""
Model Registry API routes.

Provides endpoints for registering, managing, and querying models.
"""

import asyncio
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Depends, Header
from pydantic import BaseModel, Field, AliasChoices

from ..concurrency import threadpool_endpoint
from ...auth.dependencies import (
    get_current_user,
    requires_tenant_provenance,
    validate_storage_path,
    verify_resource_ownership,
)
from ...core.idempotency import check_idempotency, store_idempotency_response
from ...core.remote_download_security import (
    DownloadConcurrencyExceeded,
    DownloadMetadataUnavailable,
    DownloadSizeExceeded,
    DownloadStorageQuotaExceeded,
    validate_remote_repo_id,
)
from ...deployment.deployment_service import ReplicaOperationBusyError
from ...storage.services.model_registry_service import model_registry_service
from ...storage.services.training_task_service import training_task_service
from ...storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    background_task_admission_service,
)
from ...utils.path_utils import map_storage_path, unmap_storage_path

logger = logging.getLogger(__name__)

router = APIRouter()


# === Request/Response Models ===

class RegisterModelRequest(BaseModel):
    """Register model request."""
    model_name: str = Field(..., description="Model name")
    model_path: str = Field(..., description="Path to model files")
    model_type: str = Field(..., description="Model type: 'embedding', 'reranker', 'decoder_reranker', or 'llm'")
    version: str = Field(default="v1.0.0", description="Version string")
    source_task_id: Optional[str] = Field(default=None, description="Source training task ID")
    base_model_path: Optional[str] = Field(default=None, description="Base model path")
    description: Optional[str] = Field(default=None, description="Model description")
    tags: Optional[List[str]] = Field(default=None, description="Tags for categorization")
    category: Optional[str] = Field(default=None, description="Category name")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")
    is_adapter: bool = Field(default=False, description="Whether the model is a PEFT/LoRA adapter")
    user_id: Optional[str] = Field(default=None, description="User ID for isolation")


class UpdateModelRequest(BaseModel):
    """Update model request."""
    description: Optional[str] = Field(default=None, description="Model description")
    tags: Optional[List[str]] = Field(default=None, description="Tags for categorization")
    category: Optional[str] = Field(default=None, description="Category name")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    status: Optional[str] = Field(default=None, description="Model status")


class AddVersionRequest(BaseModel):
    """Add version request."""
    version: str = Field(..., description="Version string")
    model_path: str = Field(..., description="Path to model files")
    changelog: Optional[str] = Field(default=None, description="Version changelog")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")
    set_as_latest: bool = Field(default=True, description="Set as latest version")


class RegisterFromTaskRequest(BaseModel):
    """Register from task request."""
    model_name: str = Field(
        ...,
        validation_alias=AliasChoices("model_name", "name"),
        description="Model name"
    )
    version: str = Field(default="v1.0.0", description="Version string")
    description: Optional[str] = Field(default=None, description="Model description")
    tags: Optional[List[str]] = Field(default=None, description="Tags for categorization")
    category: Optional[str] = Field(default=None, description="Category name")


class CompareModelsRequest(BaseModel):
    """Compare models request."""
    model_ids: List[str] = Field(..., description="List of model IDs to compare")


class ModelResponse(BaseModel):
    """Model response."""
    model_id: str
    model_name: str
    version: str
    model_type: str
    source_type: Optional[str]
    is_adapter: bool = False
    source_task_id: Optional[str]
    base_model_path: Optional[str]
    model_path: str
    description: Optional[str]
    tags: Optional[List[str]]
    category: Optional[str]
    extra_metadata: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    file_size: Optional[int]
    status: str
    is_latest: bool
    user_id: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class VersionResponse(BaseModel):
    """Version response."""
    version_id: str
    model_id: str
    version: str
    model_path: str
    changelog: Optional[str]
    metrics: Optional[Dict[str, Any]]
    created_at: Optional[str]


class ModelListResponse(BaseModel):
    """Model list response."""
    models: List[ModelResponse]
    total: int
    # Global (user-scoped, filter-independent) stats for summary cards.
    stats: Optional[Dict[str, Any]] = None


class VersionListResponse(BaseModel):
    """Version list response."""
    versions: List[VersionResponse]
    total: int


# === Helper Functions ===

def _model_to_response(model: Dict[str, Any]) -> ModelResponse:
    """Convert model dict to response with unmapped paths."""
    # Convert container paths back to host paths for display
    model_path = unmap_storage_path(model['model_path']) if model.get('model_path') else model['model_path']
    base_model_path = unmap_storage_path(model['base_model_path']) if model.get('base_model_path') else None
    return ModelResponse(
        model_id=model['model_id'],
        model_name=model['model_name'],
        version=model['version'],
        model_type=model['model_type'],
        source_type=model.get('source_type'),
        is_adapter=bool(model.get('is_adapter', False)),
        source_task_id=model.get('source_task_id'),
        base_model_path=base_model_path,
        model_path=model_path,
        description=model.get('description'),
        tags=model.get('tags'),
        category=model.get('category'),
        extra_metadata=model.get('extra_metadata'),
        metrics=model.get('metrics'),
        file_size=model.get('file_size'),
        status=model['status'],
        is_latest=model['is_latest'],
        user_id=model.get('user_id'),
        created_at=model['created_at'].isoformat() if model.get('created_at') else None,
        updated_at=model['updated_at'].isoformat() if model.get('updated_at') else None,
    )


def _version_to_response(version: Dict[str, Any]) -> VersionResponse:
    """Convert version dict to response."""
    return VersionResponse(
        version_id=version['version_id'],
        model_id=version['model_id'],
        version=version['version'],
        model_path=version['model_path'],
        changelog=version.get('changelog'),
        metrics=version.get('metrics'),
        created_at=version['created_at'].isoformat() if version.get('created_at') else None,
    )


def _normalize_registered_path(path: str) -> str:
    return path.strip().replace("\\", "/").rstrip("/")


def _resolve_owned_task_model_path(
    source_task_id: str,
    requested_path: str,
    current_user: Dict[str, Any],
    *,
    require_adapter: bool,
) -> str:
    task = training_task_service.get_task(source_task_id)
    task = verify_resource_ownership(task, current_user, "Training task")
    if task.get("status") != "succeeded":
        raise HTTPException(
            status_code=400,
            detail="Source training task has not completed successfully",
        )
    if require_adapter and not task.get("is_lora"):
        raise HTTPException(
            status_code=400,
            detail="Adapter models must come from a LoRA training task",
        )

    final_model_path = task.get("final_model_path")
    if not final_model_path:
        raise HTTPException(
            status_code=400,
            detail="Source training task has no final model path",
        )
    if _normalize_registered_path(requested_path) != _normalize_registered_path(
        final_model_path
    ):
        raise HTTPException(
            status_code=400,
            detail="Model path must match the source task final model path",
        )
    return final_model_path


@contextmanager
def _guard_training_artifacts(source_task_id: Optional[str]):
    """Serialize source-task registration with training artifact deletion."""
    if not source_task_id:
        yield
        return
    try:
        task_guard = background_task_admission_service.begin_deletion(
            "training",
            source_task_id,
        )
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(
            status_code=409,
            detail="Training task is executing or being deleted",
        ) from exc
    try:
        yield
    finally:
        task_guard.release()


# === API Endpoints ===

@router.post("/models", response_model=ModelResponse)
async def register_model(
    request: RegisterModelRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Register a new model.

    Pass Idempotency-Key header to prevent duplicate registration on retries.
    """
    valid_model_types = ['embedding', 'reranker', 'decoder_reranker', 'llm']
    if request.model_type not in valid_model_types:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {request.model_type}. Valid types: {valid_model_types}")

    # Map host paths to container paths if configured
    mapped_model_path, host_model_path = map_storage_path(request.model_path)
    mapped_base_model_path = None
    host_base_model_path = None
    if request.base_model_path:
        mapped_base_model_path, host_base_model_path = map_storage_path(request.base_model_path)

    # Validate model path to prevent path traversal
    validate_storage_path(mapped_model_path, resource_type="model")
    if mapped_base_model_path:
        validate_storage_path(mapped_base_model_path, resource_type="base model")

    # Use authenticated user_id
    user_id = current_user["user_id"]
    with _guard_training_artifacts(request.source_task_id):
        if requires_tenant_provenance(current_user):
            if not request.source_task_id:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Direct model paths are disabled while authentication is enabled; "
                        "use model download or register from an owned training task"
                    ),
                )
            mapped_model_path = _resolve_owned_task_model_path(
                request.source_task_id,
                mapped_model_path,
                current_user,
                require_adapter=request.is_adapter,
            )

        # Check idempotency
        is_duplicate, cached_response = check_idempotency(
            idempotency_key, user_id, "/api/models"
        )
        if is_duplicate and cached_response:
            return ModelResponse(**cached_response)

        # Enforce unique model path per user
        existing = model_registry_service.get_model_by_path(mapped_model_path)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Model path already registered: {mapped_model_path}"
            )

        model = model_registry_service.register_model(
            model_name=request.model_name,
            model_path=mapped_model_path,
            model_type=request.model_type,
            version=request.version,
            source_task_id=request.source_task_id,
            base_model_path=mapped_base_model_path,
            description=request.description,
            tags=request.tags,
            category=request.category,
            extra_metadata={
                **(request.extra_metadata or {}),
                **({"host_model_path": host_model_path} if host_model_path and host_model_path != mapped_model_path else {}),
                **({"host_base_model_path": host_base_model_path} if host_base_model_path and mapped_base_model_path and host_base_model_path != mapped_base_model_path else {}),
            } or None,
            metrics=request.metrics,
            user_id=user_id,
            is_adapter=request.is_adapter,
        )
        response = _model_to_response(model)

        # Store for idempotency
        store_idempotency_response(
            idempotency_key, user_id, "/api/models",
            response.model_dump()
        )

        return response


@router.get("/models", response_model=ModelListResponse)
@threadpool_endpoint
def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    latest_only: bool = False,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all models for the current user with optional filters."""
    # Use authenticated user_id
    user_id = current_user["user_id"]

    models, total = model_registry_service.list_models(
        model_type=model_type,
        status=status,
        category=category,
        user_id=user_id,
        latest_only=latest_only,
        limit=limit,
        offset=offset,
    )
    return ModelListResponse(
        models=[_model_to_response(m) for m in models],
        total=total,
        stats=model_registry_service.get_stats(user_id=user_id),
    )


@router.get("/models/search", response_model=ModelListResponse)
@threadpool_endpoint
def search_models(
    query: str = Query(..., description="Search query"),
    model_type: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Search models by name or description."""
    # Use authenticated user_id
    user_id = current_user["user_id"]

    models = model_registry_service.search_models(
        query=query,
        model_type=model_type,
        user_id=user_id,
        limit=limit,
    )
    return ModelListResponse(
        models=[_model_to_response(m) for m in models],
        total=len(models),
    )


# === Model Download (must be before /models/{model_id} to avoid route conflict) ===

class DownloadModelRequest(BaseModel):
    """Download model request."""
    download_source: Literal["modelscope", "huggingface"] = Field(
        ...,
        description="Download source: 'modelscope' or 'huggingface'",
    )
    remote_repo: str = Field(
        ...,
        min_length=3,
        max_length=96,
        description="Remote repository (e.g., 'BAAI/bge-base-zh-v1.5')",
    )
    model_type: Literal["embedding", "reranker", "decoder_reranker", "llm"] = Field(
        default="embedding",
        description="Model type: 'embedding', 'reranker', 'decoder_reranker', or 'llm'",
    )
    display_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Display name for the model",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=4000,
        description="Model description",
    )
    user_id: Optional[str] = Field(default=None, max_length=36, description="User ID")


class DownloadResponse(BaseModel):
    """Download response."""
    registry_id: str
    model_name: str
    display_name: str
    status: str
    model_path: str


class DownloadProgressResponse(BaseModel):
    """Download progress response."""
    registry_id: str
    model_name: Optional[str]
    display_name: Optional[str]
    download_source: Optional[str]
    remote_repo: Optional[str]
    model_path: Optional[str]
    status: str
    progress: int
    error: Optional[str]
    created_at: Optional[str]


@router.post("/models/download", response_model=DownloadResponse)
async def download_model(
    request: DownloadModelRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Download a model from remote repository (ModelScope/HuggingFace).

    The model will be downloaded in the background and registered automatically.
    Use the progress endpoint to check download status.
    """
    from ...storage.services.model_download_service import model_download_service

    # Use authenticated user_id
    user_id = current_user["user_id"]

    # Validate download source
    if request.download_source not in ["modelscope", "huggingface"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported download source: {request.download_source}. Use 'modelscope' or 'huggingface'."
        )

    if requires_tenant_provenance(current_user):
        try:
            validate_remote_repo_id(request.remote_repo)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Validate model type
    valid_model_types = ["embedding", "reranker", "decoder_reranker", "llm"]
    if request.model_type not in valid_model_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: {request.model_type}. Use one of: {valid_model_types}"
        )

    try:
        result = await asyncio.to_thread(
            model_download_service.start_download,
            download_source=request.download_source,
            remote_repo=request.remote_repo,
            model_type=request.model_type,
            display_name=request.display_name,
            description=request.description,
            user_id=user_id,
        )

        logger.info(f"Model download started: registry_id={result['registry_id']}, repo={request.remote_repo}")

        return DownloadResponse(
            registry_id=result["registry_id"],
            model_name=result["model_name"],
            display_name=result["display_name"],
            status=result["status"],
            model_path=result["model_path"],
        )

    except DownloadConcurrencyExceeded as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": "5"},
        ) from e
    except DownloadSizeExceeded as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except DownloadMetadataUnavailable as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except DownloadStorageQuotaExceeded as e:
        raise HTTPException(status_code=507, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Failed to start model download (error_type=%s)",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to start remote model download",
        ) from e


@router.get("/models/download/{registry_id}/progress", response_model=DownloadProgressResponse)
@threadpool_endpoint
def get_download_progress(
    registry_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get download progress for a model."""
    from ...storage.services.model_download_service import model_download_service

    # Verify ownership of the model being downloaded
    model = model_registry_service.get_model(registry_id)
    model = verify_resource_ownership(model, current_user, "Model")

    progress = model_download_service.get_download_progress(registry_id)
    if not progress:
        raise HTTPException(status_code=404, detail=f"Download task not found: {registry_id}")

    return DownloadProgressResponse(
        registry_id=progress.get("registry_id", registry_id),
        model_name=progress.get("model_name"),
        display_name=progress.get("display_name"),
        download_source=progress.get("download_source"),
        remote_repo=progress.get("remote_repo"),
        model_path=progress.get("model_path"),
        status=progress.get("status", "unknown"),
        progress=progress.get("progress", 0),
        error=progress.get("error"),
        created_at=progress.get("created_at"),
    )


@router.get("/models/downloads")
@threadpool_endpoint
def list_downloads(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all download tasks."""
    from ...storage.services.model_download_service import model_download_service

    downloads = model_download_service.list_downloads(user_id=current_user.get("user_id"))
    return {"downloads": downloads, "total": len(downloads)}


# === Model CRUD (dynamic routes must be after static routes) ===

@router.get("/models/{model_id}", response_model=ModelResponse)
@threadpool_endpoint
def get_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get model by ID."""
    model = model_registry_service.get_model(model_id)
    model = verify_resource_ownership(model, current_user, "Model")
    return _model_to_response(model)


@router.put("/models/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    request: UpdateModelRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Update model information."""
    # Verify ownership before update
    model = model_registry_service.get_model(model_id)
    model = verify_resource_ownership(model, current_user, "Model")

    try:
        success = model_registry_service.update_model(
            model_id=model_id,
            description=request.description,
            tags=request.tags,
            category=request.category,
            extra_metadata=request.extra_metadata,
            status=request.status,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not success:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    model = model_registry_service.get_model(model_id)
    return _model_to_response(model)


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    force: bool = False,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a model and its versions.

    Args:
        model_id: The model ID to delete
        force: If True, also delete related deployments and configs
    """
    # Verify ownership before delete
    model = model_registry_service.get_model(model_id)
    model = verify_resource_ownership(model, current_user, "Model")

    try:
        success = model_registry_service.delete_model(model_id, force=force)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        return {"message": f"Model {model_id} deleted"}
    except (ValueError, ReplicaOperationBusyError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    except RuntimeError as e:
        logger.error(
            "Failed to delete model %s (error_type=%s)",
            model_id,
            type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Model cleanup failed; record retained",
        ) from e


# === Version Management ===

@router.post("/models/{model_id}/versions", response_model=VersionResponse)
async def add_version(
    model_id: str,
    request: AddVersionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Add a new version to an existing model."""
    # Verify ownership before adding version
    model = model_registry_service.get_model(model_id)
    model = verify_resource_ownership(model, current_user, "Model")
    if requires_tenant_provenance(current_user):
        raise HTTPException(
            status_code=403,
            detail=(
                "Direct model version paths are disabled while authentication is enabled; "
                "register a new model from an owned training task instead"
            ),
        )

    mapped_model_path, _ = map_storage_path(request.model_path)
    validate_storage_path(mapped_model_path, resource_type="model")

    version = model_registry_service.add_version(
        model_id=model_id,
        version=request.version,
        model_path=mapped_model_path,
        changelog=request.changelog,
        metrics=request.metrics,
        set_as_latest=request.set_as_latest,
    )
    if not version:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return _version_to_response(version)


@router.get("/models/{model_id}/versions", response_model=VersionListResponse)
@threadpool_endpoint
def get_versions(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get all versions of a model."""
    # Verify model exists and ownership
    model = model_registry_service.get_model(model_id)
    model = verify_resource_ownership(model, current_user, "Model")

    versions = model_registry_service.get_versions(model_id)
    return VersionListResponse(
        versions=[_version_to_response(v) for v in versions],
        total=len(versions),
    )


# === Model Comparison ===

@router.post("/models/compare")
async def compare_models(
    request: CompareModelsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Compare metrics across multiple models."""
    if len(request.model_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 models required for comparison")

    for model_id in request.model_ids:
        model = model_registry_service.get_model(model_id)
        verify_resource_ownership(model, current_user, "Model")

    return model_registry_service.compare_models(request.model_ids)


# === Register from Task ===

@router.post("/models/from-task/{task_id}", response_model=ModelResponse)
async def register_from_task(
    task_id: str,
    request: RegisterFromTaskRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Register a model from a completed training task."""
    try:
        task_guard = background_task_admission_service.begin_deletion(
            "training",
            task_id,
        )
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(
            status_code=409,
            detail="Training task is executing or being deleted",
        ) from exc

    try:
        task = training_task_service.get_task(task_id)
        verify_resource_ownership(task, current_user, "Training task")

        model = model_registry_service.register_from_task(
            task_id=task_id,
            model_name=request.model_name,
            version=request.version,
            description=request.description,
            tags=request.tags,
            category=request.category,
            user_id=current_user.get("user_id"),
        )
        if not model:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot register from task {task_id}. Task must be "
                    "succeeded with final_model_path."
                ),
            )
        return _model_to_response(model)
    finally:
        task_guard.release()
