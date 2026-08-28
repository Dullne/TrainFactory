"""
REST API routes for external data sync management.

Provides endpoints to manage sync tasks, trigger sync/generation/training,
and view sync history (batches, generations, trainings).
"""

import asyncio
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...auth.dependencies import get_current_user, verify_resource_ownership
from ...config.settings import get_settings
from ...enums.sync_status import (
    SyncStatus,
    SyncTrainingStatus,
    SyncGenerationStatus,
    TrainingTargetStatus,
)
from ...storage.services.outbound_endpoint_policy import validate_user_outbound_url
from ...storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    BackgroundTaskCapacityExceeded,
    background_task_admission_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_CASCADE_SNAPSHOT_PAGE_SIZE = 500


def _snapshot_all_pages(
    fetch_page: Callable[..., tuple[List[Dict[str, Any]], int]],
    *,
    identity_key: str,
    resource_name: str,
    snapshot_keys: tuple[str, ...],
) -> List[Dict[str, Any]]:
    """Read a stable full snapshot before a cascade mutates its source rows."""
    snapshot: List[Dict[str, Any]] = []
    seen_ids = set()
    offset = 0
    expected_total: Optional[int] = None

    while expected_total is None or offset < expected_total:
        page, total = fetch_page(
            limit=_CASCADE_SNAPSHOT_PAGE_SIZE,
            offset=offset,
        )
        total = int(total or 0)
        if total < 0:
            raise HTTPException(
                status_code=409,
                detail=f"Invalid {resource_name} snapshot size",
            )
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise HTTPException(
                status_code=409,
                detail=f"{resource_name} changed during cascade snapshot",
            )

        if not page:
            if offset < expected_total:
                raise HTTPException(
                    status_code=409,
                    detail=f"Incomplete {resource_name} cascade snapshot",
                )
            break

        for item in page:
            identity = item.get(identity_key)
            if identity in (None, "") or identity in seen_ids:
                raise HTTPException(
                    status_code=409,
                    detail=f"Unstable {resource_name} cascade snapshot",
                )
            seen_ids.add(identity)
            snapshot.append({key: item.get(key) for key in snapshot_keys})

        offset += len(page)
        if offset > expected_total:
            raise HTTPException(
                status_code=409,
                detail=f"Invalid {resource_name} cascade snapshot",
            )

    if len(snapshot) != (expected_total or 0):
        raise HTTPException(
            status_code=409,
            detail=f"Incomplete {resource_name} cascade snapshot",
        )
    return snapshot


def _build_sync_batch_file_manifest(
    task_id: str,
    owner_user_id: Any,
) -> tuple[tuple[Dict[str, Any], ...], tuple[Path, ...]]:
    """Snapshot and validate all managed batch files before any unlink."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_worker import _resolve_managed_batch_path

    batch_snapshot = _snapshot_all_pages(
        lambda **page: external_sync_service.list_batches(
            task_id=task_id,
            **page,
        ),
        identity_key="batch_id",
        resource_name="sync batch history",
        snapshot_keys=(
            "batch_id",
            "task_id",
            "user_id",
            "storage_path",
            "status",
            "dataset_id",
            "generation_task_id",
        ),
    )
    resolved_batch_paths: List[Path] = []
    for batch in batch_snapshot:
        try:
            resolved_batch_paths.append(
                _resolve_managed_batch_path(
                    task_id,
                    owner_user_id,
                    batch,
                )
            )
        except FileNotFoundError:
            # A prior cleanup attempt may already have removed the file.
            continue
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Sync batch ownership or path is invalid",
            ) from exc
    return tuple(batch_snapshot), tuple(resolved_batch_paths)


def _resolve_safe_delete_path(
    path: str,
    trusted_roots: Optional[List[str]] = None,
    expected_leaf_name: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a managed deletion target without mutating it."""
    if not path:
        return None

    if trusted_roots is None:
        settings = get_settings()
        trusted_roots = [
            str(settings.datasets_dir),
            str(settings.local_cache_dir),
            os.environ.get("SYNC_DATA_DIR", "/app/data/sync"),
            os.environ.get("GENERATION_OUTPUT_DIR", str(settings.datasets_dir)),
        ]

    try:
        target = Path(path).resolve()
    except Exception as exc:
        raise ValueError(f"Invalid managed deletion path: {path!r}") from exc

    rel = None
    for root in trusted_roots:
        try:
            rel = target.relative_to(Path(root).resolve())
            break
        except Exception:
            continue

    if rel is None:
        raise ValueError(f"Deletion path is outside trusted roots: {target}")
    if not rel.parts:
        raise ValueError(f"Refused to delete trusted root: {target}")
    if target.is_dir() and len(rel.parts) < 2:
        # Allow first-level managed artifacts only when the leaf name is
        # explicitly bound to the expected resource id.
        if not expected_leaf_name or target.name != expected_leaf_name:
            raise ValueError(f"Refused to recursively delete broad directory: {target}")
    return target


def _safe_delete_path(
    path: str,
    trusted_roots: Optional[List[str]] = None,
    expected_leaf_name: Optional[str] = None,
):
    """Delete files/dirs only under trusted managed roots."""
    target = _resolve_safe_delete_path(
        path,
        trusted_roots=trusted_roots,
        expected_leaf_name=expected_leaf_name,
    )
    if target is None:
        return
    _delete_resolved_path(target)


def _delete_resolved_path(target: Path) -> None:
    """Delete a path that already passed managed-root preflight."""
    if target.is_file():
        os.remove(target)
        logger.info("Deleted file: %s", target)
    elif target.is_dir():
        shutil.rmtree(target)
        logger.info("Deleted directory: %s", target)


def _resolve_s3_object(storage_uri: str):
    """Resolve an S3 URI to its managed store and object key."""
    if not storage_uri or not storage_uri.startswith("s3://"):
        return None
    from ...storage.object_store import get_object_store, uri_to_key

    store = get_object_store()
    object_key = uri_to_key(storage_uri, expected_bucket=store.bucket)
    return store, object_key


def _safe_delete_s3_object(storage_uri: str):
    """Delete an S3 object by URI."""
    resolved = _resolve_s3_object(storage_uri)
    if resolved is None:
        return
    store, object_key = resolved
    store.delete_object(object_key)


def _resolve_sync_user_id(current_user: Dict[str, Any], for_query: bool = False) -> Optional[str]:
    """Resolve user scope for sync APIs.

    In development mode (AUTH_ENABLED=false), do not apply user isolation.
    """
    if not get_settings().auth_enabled:
        return None if for_query else ""
    return current_user.get("user_id")


def _require_sync_task_mutable(config: Dict[str, Any]) -> None:
    """Reject every mutation once durable deletion has begun."""
    if config.get("status") in {
        SyncStatus.DELETING,
        SyncStatus.DELETING_CASCADE,
    }:
        raise HTTPException(
            status_code=409,
            detail="Sync task deletion is in progress",
        )


def _is_matching_delete_intent(
    config: Dict[str, Any],
    *,
    cascade: bool,
) -> bool:
    """Return whether an existing durable delete intent matches this request."""
    status = config.get("status")
    expected = SyncStatus.DELETING_CASCADE if cascade else SyncStatus.DELETING
    if status not in {SyncStatus.DELETING, SyncStatus.DELETING_CASCADE}:
        return False
    if status != expected:
        raise HTTPException(
            status_code=409,
            detail="Sync task deletion mode cannot change after cleanup starts",
        )
    return True


def _validate_external_api_config_reference(
    external_api_config_id: Optional[str],
    current_user: Dict[str, Any],
) -> None:
    """Validate referenced external API config exists and belongs to current user."""
    if not external_api_config_id:
        return

    from ...storage.services.external_api_config_service import external_api_config_service

    api_config = external_api_config_service.get_config(external_api_config_id)
    verify_resource_ownership(api_config, current_user, "External API config")


_GENERATION_MODEL_CONFIG_KEYS = (
    "llm_config",
    "eval_llm_config",
    "embedding_config",
    "rerank_config",
)


def _validate_generation_config(
    generation_config: Optional[Dict[str, Any]],
    current_user: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Validate model references and direct endpoints before persistence."""
    if generation_config is None:
        return None
    if not isinstance(generation_config, dict):
        raise HTTPException(status_code=400, detail="generation_config must be an object")

    from ...storage.services.model_config_service import model_config_service

    user_id = current_user.get("user_id")
    auth_enabled = get_settings().auth_enabled
    validated = dict(generation_config)
    raw_milvus_config = validated.get("milvus_config")
    if raw_milvus_config is not None and not isinstance(raw_milvus_config, dict):
        raise HTTPException(
            status_code=400,
            detail="generation_config.milvus_config must be an object",
        )
    if auth_enabled:
        # Milvus uses gRPC, so HTTP SSRF pinning cannot protect arbitrary
        # tenant-provided targets. Authenticated tasks use only operator env.
        validated.pop("milvus_config", None)

    for config_key in _GENERATION_MODEL_CONFIG_KEYS:
        raw_sub_config = validated.get(config_key)
        if raw_sub_config is None:
            continue
        if not isinstance(raw_sub_config, dict):
            raise HTTPException(
                status_code=400,
                detail=f"generation_config.{config_key} must be an object",
            )

        sub_config = dict(raw_sub_config)
        config_id = str(sub_config.get("config_id") or "").strip()
        if config_id:
            model_config = model_config_service.get_config(config_id)
            if not model_config:
                raise HTTPException(status_code=404, detail="Model config not found")
            if auth_enabled and (
                not user_id or model_config.get("user_id") != user_id
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to use this model config",
                )
            persisted_endpoint = model_config.get("api_endpoint")
            if persisted_endpoint:
                validate_user_outbound_url(persisted_endpoint, user_id)

        direct_endpoint = sub_config.get("endpoint")
        if direct_endpoint:
            sub_config["endpoint"] = validate_user_outbound_url(
                direct_endpoint,
                user_id,
            )
        validated[config_key] = sub_config

    return validated


def _validate_base_deployment_reference(
    base_deployment_id: Optional[str],
    current_user: Dict[str, Any],
    external_api_config_id: Optional[str] = None,
    base_deployment_replica_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate referenced deployment ownership and api config consistency."""
    if base_deployment_replica_id and not base_deployment_id:
        raise HTTPException(
            status_code=400,
            detail="base_deployment_replica_id requires base_deployment_id",
        )
    if not base_deployment_id:
        return None
    from ...deployment.deployment_service import deployment_service

    try:
        deployment, _replica = deployment_service.resolve_replica_selection(
            base_deployment_id,
            base_deployment_replica_id,
            user_id=_resolve_sync_user_id(current_user, for_query=True),
            require_healthy=True,
        )
    except ValueError as exc:
        message = str(exc)
        if "access denied" in message:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to use this deployment",
            ) from exc
        if "not found" in message:
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    deployment = verify_resource_ownership(deployment, current_user, "Deployment")

    if not external_api_config_id:
        return deployment

    dep_api_config = (deployment.get("external_api_config_id") or "").strip()
    if not dep_api_config:
        dep_config = deployment.get("config") or {}
        dep_api_config = (dep_config.get("external_api_config_id") or "").strip()
    if dep_api_config and dep_api_config != external_api_config_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Deployment api_config mismatch: deployment={dep_api_config}, "
                f"sync task={external_api_config_id}"
            ),
        )
    return deployment


def _legacy_training_target_id(task_id: str) -> str:
    """Use a deterministic target_id so legacy migration is idempotent."""
    from ...storage.services.external_sync_service import legacy_training_target_id

    return legacy_training_target_id(task_id)


def _get_training_target_for_task(
    task_id: str,
    target_id: str,
) -> Dict[str, Any]:
    """Return a target only when it belongs to the requested parent task."""
    from ...storage.services.external_sync_service import external_sync_service

    target = external_sync_service.get_training_target(target_id)
    if not target or target.get("task_id") != task_id:
        raise HTTPException(status_code=404, detail="Training target not found")
    return target


def _require_training_target_mutable(target: Dict[str, Any]) -> None:
    if target.get("status") in {
        TrainingTargetStatus.TRAINING,
        TrainingTargetStatus.LOADING_ADAPTER,
    }:
        raise HTTPException(
            status_code=409,
            detail="Training target is active and cannot be modified",
        )


def _legacy_training_target_view(
    task_id: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build the stable read-only target view for legacy task-level training."""
    train_cfg = config.get("training_config")
    training_threshold = int(config.get("training_threshold", 0) or 0)
    if not isinstance(train_cfg, dict) or training_threshold <= 0:
        return None
    task_status = config.get("status")
    target_status = (
        task_status
        if task_status
        in {
            TrainingTargetStatus.TRAINING,
            TrainingTargetStatus.LOADING_ADAPTER,
            TrainingTargetStatus.ERROR,
        }
        else TrainingTargetStatus.IDLE
    )
    model_type = train_cfg.get("model_type", "embedding")
    return {
        "target_id": _legacy_training_target_id(task_id),
        "task_id": task_id,
        "target_name": f"{model_type.upper()} (legacy)",
        "model_type": model_type,
        "data_phase": "final",
        "training_method": train_cfg.get("training_method", "sft"),
        "training_config": train_cfg,
        "base_model_path": train_cfg.get("base_model_path", ""),
        "base_deployment_id": config.get("base_deployment_id"),
        "base_deployment_replica_id": config.get("base_deployment_replica_id"),
        "training_threshold": training_threshold,
        "pending_training_samples": int(
            config.get("pending_training_samples", 0) or 0
        ),
        "total_training_samples": int(
            config.get("total_training_samples", 0) or 0
        ),
        "total_trainings": int(config.get("total_trainings", 0) or 0),
        "current_adapter_name": config.get("current_adapter_name"),
        "current_adapter_id": config.get("current_adapter_id"),
        "current_training_id": config.get("current_training_id"),
        "priority": 0,
        "status": target_status,
        "is_active": bool(config.get("is_active", True)),
        "sort_order": 0,
        "created_at": config.get("created_at"),
        "updated_at": config.get("updated_at"),
    }


def _get_task_training_targets(task_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return active targets without mutating legacy task state."""
    from ...storage.services.external_sync_service import external_sync_service

    all_targets = external_sync_service.list_training_targets(task_id, is_active=None)
    if all_targets:
        return external_sync_service.list_training_targets(task_id)
    legacy_target = _legacy_training_target_view(task_id, config)
    return [legacy_target] if legacy_target is not None else []


def _migrate_legacy_training_target_for_write(
    task_id: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Persist the synthetic legacy target only from an explicit write path."""
    from ...storage.services.external_sync_service import external_sync_service

    target = _legacy_training_target_view(task_id, config)
    if target is None:
        return None
    return external_sync_service.migrate_legacy_training_target(
        task_id=task_id,
        target_id=target["target_id"],
        target_name=target["target_name"],
        model_type=target["model_type"],
        data_phase=target["data_phase"],
        training_method=target["training_method"],
        training_config=target["training_config"],
        base_model_path=target["base_model_path"],
        base_deployment_id=target["base_deployment_id"],
        base_deployment_replica_id=target["base_deployment_replica_id"],
        training_threshold=target["training_threshold"],
        expected_user_id=config.get("user_id"),
    )


def _get_or_materialize_training_target_for_write(
    task_id: str,
    target_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve a real target, materializing only the exact legacy UUID5 ID."""
    from ...storage.services.external_sync_service import (
        external_sync_service,
        legacy_training_target_id,
    )

    target = external_sync_service.get_training_target(target_id)
    if target is not None:
        if target.get("task_id") != task_id:
            raise HTTPException(status_code=404, detail="Training target not found")
        return target

    if target_id != legacy_training_target_id(task_id):
        raise HTTPException(status_code=404, detail="Training target not found")

    migrated = _migrate_legacy_training_target_for_write(task_id, config)
    if (
        migrated is None
        or migrated.get("target_id") != target_id
        or migrated.get("task_id") != task_id
    ):
        raise HTTPException(status_code=404, detail="Training target not found")

    target = external_sync_service.get_training_target(target_id)
    if target is None or target.get("task_id") != task_id:
        raise HTTPException(status_code=404, detail="Training target not found")
    return target


# ── Request / Response Models ──────────────────────────────────────


class TrainingTargetCreateRequest(BaseModel):
    target_name: str = Field(..., min_length=1, max_length=255)
    model_type: str = Field("embedding")
    data_phase: str = Field("final")
    training_method: str = Field("sft")
    training_config: Optional[Dict[str, Any]] = None
    base_model_path: str = Field("")
    base_deployment_id: Optional[str] = None
    base_deployment_replica_id: Optional[str] = None
    training_threshold: int = Field(1000, ge=0)
    priority: int = Field(0, ge=0)
    sort_order: int = Field(0, ge=0)


class TrainingTargetReplaceRequest(TrainingTargetCreateRequest):
    target_id: str = Field(..., min_length=1, max_length=36)


class SyncTaskCreateRequest(BaseModel):
    task_name: str = Field(..., min_length=1, max_length=255)
    external_api_config_id: Optional[str] = Field(None, description="Reference to external API config (also serves as tenant identifier)")
    external_api_url: Optional[str] = Field(None, min_length=1)
    external_auth_config: Optional[Dict[str, Any]] = Field(None, description="Auth credentials, e.g. {token: '...'}")
    sync_interval_seconds: int = Field(300, ge=10)
    generation_threshold: int = Field(500, ge=0)
    generation_mode: str = Field("doc_to_training")
    generation_config: Optional[Dict[str, Any]] = None
    training_threshold: int = Field(1000, ge=0)
    training_config: Optional[Dict[str, Any]] = None
    base_deployment_id: Optional[str] = None
    base_deployment_replica_id: Optional[str] = None
    training_targets: List[TrainingTargetCreateRequest] = Field(default_factory=list)
    is_active: bool = True


class SyncTaskUpdateRequest(BaseModel):
    task_name: Optional[str] = None
    external_api_config_id: Optional[str] = Field(None, min_length=1, max_length=36)
    external_api_url: Optional[str] = Field(None, min_length=1)
    external_auth_config: Optional[Dict[str, Any]] = None
    sync_interval_seconds: Optional[int] = Field(None, ge=10)
    generation_threshold: Optional[int] = Field(None, ge=0)
    generation_mode: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    training_threshold: Optional[int] = Field(None, ge=0)
    training_config: Optional[Dict[str, Any]] = None
    base_deployment_id: Optional[str] = None
    base_deployment_replica_id: Optional[str] = None
    training_targets: Optional[List[TrainingTargetReplaceRequest]] = None
    is_active: Optional[bool] = None


class TrainingTargetUpdateRequest(BaseModel):
    target_name: Optional[str] = None
    model_type: Optional[str] = None
    data_phase: Optional[str] = None
    training_method: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    base_model_path: Optional[str] = None
    base_deployment_id: Optional[str] = None
    base_deployment_replica_id: Optional[str] = None
    training_threshold: Optional[int] = Field(None, ge=0)
    priority: Optional[int] = Field(None, ge=0)
    sort_order: Optional[int] = Field(None, ge=0)
    is_active: Optional[bool] = None


# ── Task CRUD ─────────────────────────────────────────────────────


@router.post("/tasks", status_code=201)
async def create_sync_task(
    request: SyncTaskCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Create a new sync task."""
    from ...storage.services.external_sync_service import external_sync_service

    # Validate: must provide either external_api_config_id or (url + auth)
    if not request.external_api_config_id and not request.external_api_url:
        raise HTTPException(
            status_code=400,
            detail="Either external_api_config_id or external_api_url is required",
        )

    user_id = _resolve_sync_user_id(current_user)
    external_api_url = request.external_api_url or ""
    if external_api_url:
        external_api_url = validate_user_outbound_url(
            external_api_url,
            user_id,
        )

    _validate_external_api_config_reference(request.external_api_config_id, current_user)
    generation_config = _validate_generation_config(
        request.generation_config,
        current_user,
    )
    _validate_base_deployment_reference(
        request.base_deployment_id,
        current_user,
        request.external_api_config_id,
        request.base_deployment_replica_id,
    )
    for target in request.training_targets:
        _validate_base_deployment_reference(
            target.base_deployment_id,
            current_user,
            request.external_api_config_id,
            target.base_deployment_replica_id,
        )

    try:
        config = external_sync_service.create_task(
            task_name=request.task_name,
            user_id=user_id,
            external_api_config_id=request.external_api_config_id,
            external_api_url=external_api_url,
            external_auth_config=request.external_auth_config,
            sync_interval_seconds=request.sync_interval_seconds,
            generation_threshold=request.generation_threshold,
            generation_mode=request.generation_mode,
            generation_config=generation_config,
            training_threshold=request.training_threshold,
            training_config=request.training_config,
            base_deployment_id=request.base_deployment_id,
            base_deployment_replica_id=request.base_deployment_replica_id,
            training_targets=[
                target.model_dump() for target in request.training_targets
            ],
            is_active=request.is_active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if config.get("is_active"):
        from ...sync.sync_manager import sync_manager

        sync_manager.start_worker(config["task_id"])
    return {"message": "Sync task created", "task": config}


@router.get("/tasks")
async def list_sync_tasks(
    external_api_config_id: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all sync tasks for the current user."""
    from ...storage.services.external_sync_service import external_sync_service

    user_id = _resolve_sync_user_id(current_user, for_query=True)
    configs, total = external_sync_service.list_tasks(
        user_id=user_id,
        external_api_config_id=external_api_config_id,
        limit=limit,
        offset=offset,
    )
    return {"tasks": configs, "total": total}


@router.get("/tasks/{task_id}")
async def get_sync_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get sync task details."""
    from ...storage.services.external_sync_service import external_sync_service

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    targets = _get_task_training_targets(task_id, config)
    config["training_targets"] = targets
    return {"task": config}


@router.patch("/tasks/{task_id}")
async def update_sync_task(
    task_id: str,
    request: SyncTaskUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Update a sync task."""
    from ...storage.services.external_sync_service import external_sync_service

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    raw_update_fields = request.model_dump(exclude_unset=True)
    explicitly_set_fields = request.model_fields_set
    nullable_update_fields = {
        "generation_config",
        "training_config",
        "base_deployment_id",
        "base_deployment_replica_id",
    }
    update_fields = {
        key: value
        for key, value in raw_update_fields.items()
        if key in explicitly_set_fields
        and (value is not None or key in nullable_update_fields)
    }
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    if update_fields.get("external_api_url"):
        update_fields["external_api_url"] = validate_user_outbound_url(
            update_fields["external_api_url"],
            _resolve_sync_user_id(current_user),
        )
    if "generation_config" in update_fields:
        update_fields["generation_config"] = _validate_generation_config(
            update_fields["generation_config"],
            current_user,
        )

    from ...sync.sync_manager import sync_manager

    source_identity_keys = {"external_api_config_id", "external_api_url"}
    source_identity_changed = False
    worker_stopped = False
    updated = None
    resume_worker = bool(config.get("is_active"))
    try:
        async with sync_manager.task_operation_lock(task_id):
            latest = external_sync_service.get_task(task_id)
            latest = verify_resource_ownership(latest, current_user, "Sync task")
            _require_sync_task_mutable(latest)
            resume_worker = bool(latest.get("is_active"))
            source_identity_changed = any(
                key in update_fields
                and (update_fields.get(key) or "") != (latest.get(key) or "")
                for key in source_identity_keys
            )
            if source_identity_changed:
                await sync_manager.stop_worker(task_id)
                worker_stopped = True

            _validate_external_api_config_reference(
                update_fields.get("external_api_config_id"),
                current_user,
            )
            target_api_config = update_fields.get(
                "external_api_config_id",
                latest.get("external_api_config_id"),
            )
            _validate_base_deployment_reference(
                update_fields.get(
                    "base_deployment_id",
                    latest.get("base_deployment_id"),
                ),
                current_user,
                target_api_config,
                update_fields.get(
                    "base_deployment_replica_id",
                    latest.get("base_deployment_replica_id"),
                ),
            )
            updated = external_sync_service.update_task(
                task_id,
                expected_user_id=latest.get("user_id"),
                **update_fields,
            )
            if updated is None:
                raise HTTPException(status_code=404, detail="Sync task not found")
    except ValueError as exc:
        if worker_stopped and resume_worker and sync_manager.running:
            sync_manager.start_worker(task_id)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        if worker_stopped and resume_worker and sync_manager.running:
            sync_manager.start_worker(task_id)
        raise

    if source_identity_changed and updated.get("is_active") and sync_manager.running:
        sync_manager.start_worker(task_id)

    # If sync_interval changed or is_active toggled, restart worker
    if sync_manager.running and not source_identity_changed:
        if "is_active" in update_fields:
            if update_fields["is_active"]:
                sync_manager.start_worker(task_id)
            else:
                await sync_manager.stop_worker(task_id)
        elif any(
            key in update_fields
            for key in (
                "sync_interval_seconds",
                "external_api_config_id",
                "external_api_url",
                "external_auth_config",
            )
        ):
            await sync_manager.restart_worker(task_id)

    return {"message": "Sync task updated", "task": updated}


@router.delete("/tasks/{task_id}")
async def delete_sync_task(
    task_id: str,
    cascade: bool = False,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a sync task and stop its worker.

    cascade=true: also deletes all datasets, lineage edges, asset records,
    and batch files created by this sync task.
    """
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_manager import sync_manager

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _is_matching_delete_intent(config, cascade=cascade)

    # Cancelling the periodic worker does not stop manual operations. The
    # per-task operation lock closes that race and waits for any in-flight
    # manual cycle before deletion snapshots are taken.
    await sync_manager.stop_worker(task_id)
    async with sync_manager.task_operation_lock(task_id):
        latest = external_sync_service.get_task(task_id)
        latest = verify_resource_ownership(latest, current_user, "Sync task")
        return _delete_sync_task_locked(
            task_id,
            cascade=cascade,
            config=latest,
            # The pre-lock snapshot can become stale while another delete
            # request holds the task operation lock.  Rollback eligibility
            # must be derived from the state observed inside this lock so an
            # existing durable intent is never mistaken for one created by
            # this request.
            original_config=latest,
            sync_manager=sync_manager,
        )


def _snapshot_sync_deletion_children(
    task_id: str,
    config: Dict[str, Any],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[str],
    List[str],
]:
    """Snapshot all tracked children and identify any active generic tasks."""
    from ...enums import TrainingStatus
    from ...storage.entities.generation_task_entity import GenerationStatus
    from ...storage.services.external_sync_service import external_sync_service
    from ...storage.services.generation_task_service import generation_task_service
    from ...storage.services.training_task_service import training_task_service

    generation_snapshot = _snapshot_all_pages(
        lambda **page: external_sync_service.list_generations(
            task_id=task_id,
            **page,
        ),
        identity_key="id",
        resource_name="sync generation history",
        snapshot_keys=(
            "id",
            "task_id",
            "user_id",
            "generation_task_id",
            "output_dataset_id",
            "status",
        ),
    )
    training_snapshot = _snapshot_all_pages(
        lambda **page: external_sync_service.list_trainings(
            task_id=task_id,
            **page,
        ),
        identity_key="id",
        resource_name="sync training history",
        snapshot_keys=(
            "id",
            "task_id",
            "user_id",
            "training_task_id",
            "status",
            "loaded_adapter_name",
            "loaded_adapter_id",
        ),
    )

    active_generation_statuses = {
        GenerationStatus.PENDING,
        GenerationStatus.RUNNING,
        GenerationStatus.STOPPING,
        GenerationStatus.PUBLISHING,
        GenerationStatus.RECOVERING,
        GenerationStatus.RESTARTING,
    }
    active_training_statuses = {
        TrainingStatus.PENDING.value,
        TrainingStatus.PREPARING.value,
        TrainingStatus.RUNNING.value,
        TrainingStatus.EVALUATING.value,
    }
    owner_user_id = config.get("user_id")
    active_generation_ids: List[str] = []
    active_training_ids: List[str] = []
    for generation in generation_snapshot:
        if (
            generation.get("task_id") != task_id
            or generation.get("user_id") != owner_user_id
        ):
            raise PermissionError(
                "Sync generation tracking owner changed during deletion"
            )
        child_id = generation.get("generation_task_id")
        if not child_id:
            continue
        child = generation_task_service.get_task(child_id)
        if child is not None and child.get("user_id") != owner_user_id:
            raise PermissionError(
                "Generation child owner changed during sync deletion"
            )
        status = child.get("status") if child else generation.get("status")
        if status in active_generation_statuses:
            active_generation_ids.append(str(child_id))

    for training in training_snapshot:
        if (
            training.get("task_id") != task_id
            or training.get("user_id") != owner_user_id
        ):
            raise PermissionError(
                "Sync training tracking owner changed during deletion"
            )
        child_id = training.get("training_task_id")
        if not child_id:
            continue
        child = training_task_service.get_task(child_id)
        if child is not None and child.get("user_id") != owner_user_id:
            raise PermissionError(
                "Training child owner changed during sync deletion"
            )
        status = child.get("status") if child else training.get("status")
        has_process_lease = bool(
            child
            and (
                child.get("process_pid") is not None
                or child.get("process_status") is not None
            )
        )
        if status in active_training_statuses or has_process_lease:
            active_training_ids.append(str(child_id))

    return (
        generation_snapshot,
        training_snapshot,
        active_generation_ids,
        active_training_ids,
    )


def _delete_sync_task_locked(
    task_id: str,
    *,
    cascade: bool,
    config: Dict[str, Any],
    original_config: Dict[str, Any],
    sync_manager: Any,
) -> Dict[str, Any]:
    """Delete one quiesced sync task while its operation lock is held."""
    from ...storage.services.external_sync_service import external_sync_service

    def _resume_worker_after_aborted_cascade() -> None:
        if original_config.get("is_active") and sync_manager.running:
            try:
                sync_manager.start_worker(task_id)
            except Exception:
                logger.exception(
                    "Could not resume sync worker after cascade delete aborted: %s",
                    task_id,
                )

    try:
        (
            generation_snapshot,
            training_snapshot,
            active_generation_ids,
            active_training_ids,
        ) = _snapshot_sync_deletion_children(task_id, config)
    except Exception:
        _resume_worker_after_aborted_cascade()
        raise

    if active_generation_ids or active_training_ids:
        _resume_worker_after_aborted_cascade()
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot delete while downstream tasks are active. "
                f"generation={active_generation_ids[:5]}, "
                f"training={active_training_ids[:5]}"
            ),
        )

    deletion_guards = []
    child_keys = {
        ("generation", str(item["generation_task_id"]))
        for item in generation_snapshot
        if item.get("generation_task_id")
    }
    child_keys.update(
        {
            ("training", str(item["training_task_id"]))
            for item in training_snapshot
            if item.get("training_task_id")
        }
    )
    try:
        for task_kind, child_task_id in sorted(child_keys):
            deletion_guards.append(
                background_task_admission_service.begin_deletion(
                    task_kind,
                    child_task_id,
                )
            )
    except BackgroundTaskAlreadyExecuting as exc:
        for guard in reversed(deletion_guards):
            guard.release()
        _resume_worker_after_aborted_cascade()
        raise HTTPException(
            status_code=409,
            detail="A downstream task is executing or being deleted",
        ) from exc

    if cascade:
        try:
            training_artifact_conflicts = (
                _list_sync_training_artifact_conflicts(training_snapshot)
            )
        except Exception:
            for guard in reversed(deletion_guards):
                guard.release()
            _resume_worker_after_aborted_cascade()
            raise
        if training_artifact_conflicts:
            for guard in reversed(deletion_guards):
                guard.release()
            _resume_worker_after_aborted_cascade()
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot cascade delete training artifacts with external "
                    f"consumers: {training_artifact_conflicts[:5]}"
                ),
            )

    try:
        config = external_sync_service.begin_task_deletion(
            task_id,
            cascade=cascade,
            expected_user_id=config.get("user_id"),
        )
    except ValueError as exc:
        for guard in reversed(deletion_guards):
            guard.release()
        _resume_worker_after_aborted_cascade()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        for guard in reversed(deletion_guards):
            guard.release()
        _resume_worker_after_aborted_cascade()
        raise
    if not config:
        for guard in reversed(deletion_guards):
            guard.release()
        raise HTTPException(status_code=404, detail="Sync task not found")

    try:
        return _perform_sync_task_deletion(
            task_id,
            cascade=cascade,
            config=config,
            original_config=original_config,
            generation_snapshot=generation_snapshot,
            training_snapshot=training_snapshot,
            resume_worker=_resume_worker_after_aborted_cascade,
        )
    finally:
        for guard in reversed(deletion_guards):
            guard.release()


def _sync_milvus_collection_prefixes(task_id: str) -> tuple[str, str, str]:
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return (
        f"tf_sync_{task_id[:8]}_",
        f"tf_sync_v2_{task_id[:8]}_",
        f"tf_sync_v3_{task_hash}_",
    )


class SyncMilvusCollectionSharedError(RuntimeError):
    """Raised before drop when a sync collection still serves other datasets."""


class SyncMilvusCollectionConsumerError(RuntimeError):
    """Raised before drop when an active generation still uses a collection."""


def _sync_milvus_exact_collection_names(
    config: Dict[str, Any],
    training_snapshot: List[Dict[str, Any]],
) -> set[str]:
    """Return legacy names whose ownership is backed by persisted sync state."""
    task_id = str(config["task_id"])
    legacy_prefix, v2_prefix, v3_prefix = _sync_milvus_collection_prefixes(
        task_id
    )
    exact_names: set[str] = set()

    current_collection = config.get("milvus_collection_name")
    if isinstance(current_collection, str) and current_collection.strip():
        # The persisted exact name is durable task provenance even for
        # deployments that predate the tf_sync naming convention.
        exact_names.add(current_collection.strip())

    for training in training_snapshot:
        adapter_id = training.get("loaded_adapter_id")
        adapter_name = training.get("loaded_adapter_name")
        if not adapter_id or not adapter_name:
            continue
        legacy_hash = hashlib.md5(  # noqa: S324 - legacy name compatibility
            str(adapter_name).encode("utf-8")
        ).hexdigest()[:8]
        adapter_suffix = f"a{str(adapter_id)[:8]}_{legacy_hash}"
        exact_names.add(f"{legacy_prefix}{adapter_suffix}")
        exact_names.add(f"{v2_prefix}{adapter_suffix}")

    return exact_names


def _is_owned_sync_milvus_collection(
    config: Dict[str, Any],
    training_snapshot: List[Dict[str, Any]],
    collection_name: Any,
) -> bool:
    """Require exact legacy provenance; v3 uses a full-task hash namespace."""
    if not isinstance(collection_name, str):
        return False
    task_id = str(config["task_id"])
    v3_prefix = _sync_milvus_collection_prefixes(task_id)[2]
    return (
        collection_name in _sync_milvus_exact_collection_names(
            config,
            training_snapshot,
        )
        or collection_name.startswith(v3_prefix)
    )


def _external_sync_milvus_links(
    config: Dict[str, Any],
    training_snapshot: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter dataset links that are not owned by this sync task."""
    return [
        link
        for link in links
        if not _is_owned_sync_milvus_collection(
            config,
            training_snapshot,
            link.get("collection_name"),
        )
    ]


def _delete_sync_milvus_collections(
    config: Dict[str, Any],
    training_snapshot: List[Dict[str, Any]],
    linked_collection_names: tuple[str, ...] = (),
    owned_dataset_ids: tuple[str, ...] = (),
    owned_generation_task_ids: tuple[str, ...] = (),
    on_preflight_failure: Optional[Callable[[], None]] = None,
    on_durable_intent_detected: Optional[Callable[[], None]] = None,
) -> List[str]:
    """Fence and drop every identifiable base/adapter collection."""
    from ...generation.clients.milvus_client import MilvusClient, MilvusConfig
    from ...storage.services.milvus_collection_service import (
        MilvusCollectionDeletionOwnerConflictError,
        MilvusCollectionUnavailableError,
        milvus_collection_service,
    )
    from ...storage.services.generation_task_service import (
        generation_task_service,
    )
    from ...storage.services.deep_evaluation_task_service import (
        deep_evaluation_task_service,
    )
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_worker import _resolve_sync_milvus_connection_config

    task_id = str(config["task_id"])
    deletion_owner = f"sync:{task_id}"
    legacy_prefix, v2_prefix, v3_prefix = _sync_milvus_collection_prefixes(
        task_id
    )
    acquisition = None
    client = None

    def _restore_new_fences_and_parent() -> None:
        if acquisition is not None:
            restored = milvus_collection_service.restore_deletion_fences(
                acquisition.newly_fenced,
                deletion_owner=deletion_owner,
                created_placeholders=acquisition.created_placeholders,
            )
            if not restored:
                raise RuntimeError(
                    "Could not restore sync Milvus deletion fences"
                )
        if on_preflight_failure is not None:
            on_preflight_failure()

    try:
        try:
            candidates = _sync_milvus_exact_collection_names(
                config,
                training_snapshot,
            )
            pending_fences = milvus_collection_service.list_deletion_fences(
                deletion_owner=deletion_owner,
            )
            if pending_fences and on_durable_intent_detected is not None:
                on_durable_intent_detected()
            for collection_name in pending_fences:
                if not _is_owned_sync_milvus_collection(
                    config,
                    training_snapshot,
                    collection_name,
                ):
                    raise RuntimeError(
                        "Refusing an unowned persisted Milvus deletion fence"
                    )
                candidates.add(collection_name)

            # Registry rows are durable ownership evidence even when the remote
            # collection was never created successfully or disappeared externally.
            # Include them so cascade finalization cannot leave orphaned v3 rows.
            registered_collections = _snapshot_all_pages(
                lambda **page: (
                    milvus_collection_service.list_deletion_registry_snapshot(
                        **page
                    )
                ),
                identity_key="collection_id",
                resource_name="sync Milvus registry",
                snapshot_keys=(
                    "collection_id",
                    "collection_name",
                    "user_id",
                    "status",
                    "sync_task_id",
                    "deletion_owner",
                ),
            )
            registry_collection_ids: Dict[str, str] = {}
            for registry in registered_collections:
                collection_name = registry.get("collection_name")
                collection_id = registry.get("collection_id")
                if isinstance(collection_name, str) and isinstance(
                    collection_id, str
                ):
                    registry_collection_ids[collection_name] = collection_id
                if not _is_owned_sync_milvus_collection(
                    config,
                    training_snapshot,
                    collection_name,
                ):
                    continue

                same_user = registry.get("user_id") == config.get("user_id")
                sync_claim = registry.get("sync_task_id")
                row_deletion_owner = registry.get("deletion_owner")
                durable_retry_fence = (
                    registry.get("status") == "deleting"
                    and row_deletion_owner == deletion_owner
                    and sync_claim in (None, task_id)
                )
                active_current_claim = (
                    registry.get("status") != "deleting"
                    and row_deletion_owner in (None, "")
                    and sync_claim == task_id
                )
                if not same_user or not (
                    durable_retry_fence or active_current_claim
                ):
                    raise SyncMilvusCollectionConsumerError(
                        "A Milvus registry namespace has foreign provenance"
                    )
                candidates.add(collection_name)

            for collection_name in linked_collection_names:
                if not _is_owned_sync_milvus_collection(
                    config,
                    training_snapshot,
                    collection_name,
                ):
                    raise RuntimeError(
                        "Refusing to delete a Milvus collection not owned by "
                        "the sync task"
                    )
                candidates.add(collection_name)

            current_collection = config.get("milvus_collection_name")
            if isinstance(current_collection, str) and current_collection.strip():
                current_collection = current_collection.strip()
                if not _is_owned_sync_milvus_collection(
                    config,
                    training_snapshot,
                    current_collection,
                ):
                    raise RuntimeError(
                        "Refusing to delete a Milvus collection not owned by the "
                        f"sync task: {current_collection}"
                    )

            generation_config = config.get("generation_config") or {}
            has_embedding_config = bool(
                isinstance(generation_config, dict)
                and generation_config.get("embedding_config")
            )
            if not candidates and not has_embedding_config:
                return []

            client = MilvusClient(
                MilvusConfig(
                    **_resolve_sync_milvus_connection_config(
                        generation_config
                    )
                )
            )
            client.connect()
            candidates.update(
                name
                for name in client.list_collections()
                if isinstance(name, str) and name.startswith(v3_prefix)
            )
            expected_collection_ids = {
                collection_name: registry_collection_ids.get(collection_name)
                for collection_name in candidates
            }
            try:
                acquisition = milvus_collection_service.acquire_deletion_fences(
                    candidates,
                    deletion_owner=deletion_owner,
                    user_id=config.get("user_id"),
                    expected_collection_ids=expected_collection_ids,
                )
            except (
                MilvusCollectionDeletionOwnerConflictError,
                MilvusCollectionUnavailableError,
            ) as exc:
                raise SyncMilvusCollectionConsumerError(
                    "A Milvus collection is unavailable for this sync deletion"
                ) from exc

            active_sync_consumers = [
                consumer_task_id
                for consumer_task_id in (
                    external_sync_service.list_active_collection_consumers(
                        list(candidates)
                    )
                )
                if str(consumer_task_id) != task_id
            ]
            active_generation_consumers = (
                generation_task_service.list_active_collection_consumers(
                    list(candidates),
                    exclude_task_ids=owned_generation_task_ids,
                )
            )
            active_deep_evaluation_consumers = []
            if not active_generation_consumers:
                active_deep_evaluation_consumers = (
                    deep_evaluation_task_service.list_active_collection_consumers(
                        list(candidates)
                    )
                )
            owned_dataset_id_set = set(owned_dataset_ids)
            shared_links = []
            for collection_name in sorted(candidates):
                shared_links.extend(
                    link
                    for link in milvus_collection_service.get_linked_datasets(
                        collection_name
                    )
                    if link.get("dataset_id") not in owned_dataset_id_set
                )
            if (
                active_sync_consumers
                or active_generation_consumers
                or active_deep_evaluation_consumers
                or shared_links
            ):
                if (
                    active_sync_consumers
                    or active_generation_consumers
                    or active_deep_evaluation_consumers
                ):
                    raise SyncMilvusCollectionConsumerError(
                        "Refusing to delete a Milvus collection used by active tasks"
                    )
                raise SyncMilvusCollectionSharedError(
                    "Refusing to delete a Milvus collection linked to datasets "
                    "outside this sync cascade"
                )
        except Exception:
            _restore_new_fences_and_parent()
            raise

        deleted = []
        for collection_name in sorted(candidates):
            client.drop_collection(collection_name)
            deleted.append(collection_name)
        return deleted
    finally:
        if client is not None:
            client.close()


def _build_sync_child_cleanup_manifest(
    generation_snapshot: List[Dict[str, Any]],
    training_snapshot: List[Dict[str, Any]],
    parent_task_id: str,
    parent_user_id: Any,
) -> Dict[str, tuple[Dict[str, Any], ...]]:
    """Resolve all child cleanup targets without mutating storage."""
    from ...storage.services.generation_task_service import generation_task_service
    from ...storage.services.training_task_service import training_task_service
    from .generation_routes import _resolve_generation_delete_path
    from .training_routes import _resolve_server_managed_task_output

    generation_children = []
    training_children = []
    for generation in generation_snapshot:
        if (
            generation.get("task_id") != parent_task_id
            or generation.get("user_id") != parent_user_id
        ):
            raise PermissionError(
                "Sync generation tracking owner changed during cleanup"
            )
        child_id = generation.get("generation_task_id")
        if not child_id:
            continue
        child_task = generation_task_service.get_task(child_id)
        if child_task is None:
            continue
        if child_task.get("user_id") != parent_user_id:
            raise PermissionError(
                "Generation child owner changed during sync cleanup"
            )
        artifact_values = tuple(
            child_task.get(key)
            for key in (
                "output_path",
                "qa_output_path",
                "qa_filtered_path",
                "deep_eval_path",
            )
        )
        artifact_paths = tuple(
            _resolve_generation_delete_path(path)
            for path in artifact_values
            if isinstance(path, str) and path
        )
        generation_children.append(
            {
                "child_id": child_id,
                "artifact_values": artifact_values,
                "artifact_paths": artifact_paths,
            }
        )

    for training in training_snapshot:
        if (
            training.get("task_id") != parent_task_id
            or training.get("user_id") != parent_user_id
        ):
            raise PermissionError(
                "Sync training tracking owner changed during cleanup"
            )
        child_id = training.get("training_task_id")
        if not child_id:
            continue
        child_task = training_task_service.get_task(child_id)
        if child_task is None:
            continue
        if child_task.get("user_id") != parent_user_id:
            raise PermissionError(
                "Training child owner changed during sync cleanup"
            )
        output_value = child_task.get("output_dir")
        output_path = None
        if isinstance(output_value, str) and output_value:
            output_path = Path(
                _resolve_server_managed_task_output(
                    child_id,
                    child_task,
                    output_value,
                )
            )
        training_children.append(
            {
                "child_id": child_id,
                "output_value": output_value,
                "output_path": output_path,
            }
        )

    return {
        "generation": tuple(generation_children),
        "training": tuple(training_children),
    }


def _delete_sync_child_tasks(
    generation_snapshot: List[Dict[str, Any]],
    training_snapshot: List[Dict[str, Any]],
    parent_task_id: str,
    parent_user_id: Any,
    cleanup_manifest: Optional[Dict[str, tuple[Dict[str, Any], ...]]] = None,
) -> None:
    """Revalidate and delete terminal generic children as one safe manifest."""
    from ...storage.services.generation_task_service import generation_task_service
    from ...storage.services.training_task_service import training_task_service

    manifest = cleanup_manifest or _build_sync_child_cleanup_manifest(
        generation_snapshot,
        training_snapshot,
        parent_task_id,
        parent_user_id,
    )
    generation_entries = {
        str(entry["child_id"]): entry for entry in manifest["generation"]
    }
    training_entries = {
        str(entry["child_id"]): entry for entry in manifest["training"]
    }
    generation_children = []
    training_children = []

    # Re-fetch and validate every child before deleting the first artifact.
    for generation in generation_snapshot:
        if (
            generation.get("task_id") != parent_task_id
            or generation.get("user_id") != parent_user_id
        ):
            raise PermissionError(
                "Sync generation tracking owner changed during cleanup"
            )
        child_id = generation.get("generation_task_id")
        if not child_id:
            continue
        child_task = generation_task_service.get_task(child_id)
        if child_task is None:
            continue
        if child_task.get("user_id") != parent_user_id:
            raise PermissionError(
                "Generation child owner changed during sync cleanup"
            )
        entry = generation_entries.get(str(child_id))
        if entry is None:
            raise RuntimeError("Generation child appeared after cleanup preflight")
        current_values = tuple(
            child_task.get(key)
            for key in (
                "output_path",
                "qa_output_path",
                "qa_filtered_path",
                "deep_eval_path",
            )
        )
        if current_values != entry["artifact_values"]:
            raise RuntimeError("Generation child artifacts changed during cleanup")
        generation_children.append((child_id, entry))

    for training in training_snapshot:
        if (
            training.get("task_id") != parent_task_id
            or training.get("user_id") != parent_user_id
        ):
            raise PermissionError(
                "Sync training tracking owner changed during cleanup"
            )
        child_id = training.get("training_task_id")
        if not child_id:
            continue
        child_task = training_task_service.get_task(child_id)
        if child_task is None:
            continue
        if child_task.get("user_id") != parent_user_id:
            raise PermissionError(
                "Training child owner changed during sync cleanup"
            )
        entry = training_entries.get(str(child_id))
        if entry is None:
            raise RuntimeError("Training child appeared after cleanup preflight")
        if child_task.get("output_dir") != entry["output_value"]:
            raise RuntimeError("Training child output changed during cleanup")
        training_children.append((child_id, entry))

    for child_id, entry in generation_children:
        for artifact_path in entry["artifact_paths"]:
            _delete_resolved_path(artifact_path)
        if not generation_task_service.delete_task(child_id):
            raise RuntimeError(f"Failed to delete generation child task: {child_id}")

    for child_id, entry in training_children:
        if entry["output_path"] is not None:
            _delete_resolved_path(entry["output_path"])
        if not training_task_service.delete_task(child_id):
            raise RuntimeError(f"Failed to delete training child task: {child_id}")


def _list_sync_training_artifact_conflicts(
    training_snapshot: List[Dict[str, Any]],
) -> List[str]:
    """Return sync training children whose artifacts have external consumers."""
    from .training_routes import _get_training_artifact_dependencies
    from ...storage.services.training_task_service import training_task_service

    snapshot_ids = {
        str(item["training_task_id"])
        for item in training_snapshot
        if item.get("training_task_id")
    }
    conflicts = []
    for child_id in sorted(snapshot_ids):
        task = training_task_service.get_task(child_id)
        if not task:
            continue
        dependencies = _get_training_artifact_dependencies(task)
        external_children = [
            child
            for child in dependencies["dependent_training_tasks"]
            if str(child.get("task_id")) not in snapshot_ids
        ]
        if (
            dependencies["registered_models"]
            or dependencies["active_adapters"]
            or external_children
        ):
            conflicts.append(child_id)
    return conflicts


def _perform_sync_task_deletion(
    task_id: str,
    *,
    cascade: bool,
    config: Dict[str, Any],
    original_config: Optional[Dict[str, Any]] = None,
    generation_snapshot: Optional[List[Dict[str, Any]]] = None,
    training_snapshot: Optional[List[Dict[str, Any]]] = None,
    resume_worker: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Idempotently remove resources for a task with durable delete intent."""
    from ...storage.services.external_sync_service import external_sync_service

    generation_snapshot = generation_snapshot or []
    training_snapshot = training_snapshot or []

    sync_data_root = os.environ.get("SYNC_DATA_DIR", "/app/data/sync")
    generation_output_root = os.environ.get(
        "GENERATION_OUTPUT_DIR", str(get_settings().datasets_dir)
    )

    deleted_datasets: List[str] = []
    fenced_milvus_collections: tuple[str, ...] = ()
    batch_snapshot: tuple[Dict[str, Any], ...] = ()
    batch_file_manifest: tuple[Path, ...] = ()
    if not cascade:
        from ...storage.services.dataset_service import dataset_service
        from ...storage.services.evaluation_task_service import (
            evaluation_task_service,
        )
        from ...storage.services.generation_task_service import (
            generation_task_service,
        )
        from ...storage.services.training_task_service import training_task_service

        try:
            batch_snapshot, batch_file_manifest = _build_sync_batch_file_manifest(
                task_id,
                config.get("user_id"),
            )
            raw_batch_paths = sorted(
                {
                    path
                    for batch in batch_snapshot
                    if isinstance((path := batch.get("storage_path")), str)
                    and path
                }
            )
            dataset_consumers = (
                dataset_service.list_external_storage_reference_consumers(
                    raw_batch_paths,
                )
            )
            generation_consumers = (
                generation_task_service.list_artifact_reference_consumers(
                    raw_batch_paths,
                )
            )
            training_consumers = (
                training_task_service.list_active_dataset_consumers(raw_batch_paths)
            )
            evaluation_consumers = (
                evaluation_task_service.list_active_dataset_consumers(
                    [],
                    raw_batch_paths,
                )
            )
            if (
                dataset_consumers
                or generation_consumers
                or training_consumers
                or evaluation_consumers
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Cannot delete shared sync batch storage. "
                        f"dataset_reference_count={len(dataset_consumers)}, "
                        "generation_reference_count="
                        f"{len(generation_consumers)}, "
                        f"training={training_consumers[:5]}, "
                        f"evaluation={evaluation_consumers[:5]}"
                    ),
                )
        except Exception:
            previous = original_config
            previous_intent_is_durable = bool(
                previous
                and previous.get("status")
                in {SyncStatus.DELETING, SyncStatus.DELETING_CASCADE}
            )
            if previous and not previous_intent_is_durable:
                restored = external_sync_service.cancel_task_deletion(
                    task_id,
                    status=previous.get("status") or SyncStatus.IDLE,
                    is_active=bool(previous.get("is_active")),
                    expected_user_id=config.get("user_id"),
                    expected_deleting_status=SyncStatus.DELETING,
                )
                if not restored:
                    raise RuntimeError(
                        "Could not restore sync task after batch preflight failure"
                    )
                if resume_worker:
                    resume_worker()
            raise
    if cascade:
        from ...storage.services.dataset_service import (
            DatasetDeletionOwnerConflictError,
            dataset_service,
        )
        from ...storage.services.dataset_lineage_service import dataset_lineage_service
        from ...storage.services.dataset_asset_service import dataset_asset_service
        from ...storage.services.generation_task_service import generation_task_service
        from ...storage.services.training_task_service import training_task_service
        from ...storage.services.evaluation_task_service import evaluation_task_service
        from ...storage.services.milvus_collection_service import (
            milvus_collection_service,
        )

        owner_user_id = config.get("user_id")
        dataset_deletion_owner = f"sync:{task_id}"
        candidate_dataset_ids: List[str] = []
        seen_dataset_ids = set()
        expected_dataset_sources: Dict[str, set[tuple[str, str]]] = {}
        dataset_records: Dict[str, Dict[str, Any]] = {}
        has_existing_child_intent = False
        dataset_storage_manifest: Dict[
            str,
            tuple[tuple[str, Any], ...],
        ] = {}

        def _mark_existing_child_intent() -> None:
            nonlocal has_existing_child_intent
            has_existing_child_intent = True

        def _restore_pre_destructive_state() -> None:
            previous = original_config or {}
            delete_was_already_pending = previous.get("status") in {
                SyncStatus.DELETING,
                SyncStatus.DELETING_CASCADE,
            }
            if delete_was_already_pending:
                return
            if has_existing_child_intent:
                return

            restored = True
            for dataset_id, dataset in reversed(list(dataset_records.items())):
                if dataset.get("status") == "deleting":
                    continue
                try:
                    restored = (
                        dataset_service.restore_from_deleting(
                            dataset_id,
                            deletion_owner=dataset_deletion_owner,
                            status=dataset.get("status") or "ready",
                            user_id=owner_user_id,
                        )
                        and restored
                    )
                except DatasetDeletionOwnerConflictError as exc:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A dataset deletion owner changed during rollback"
                        ),
                    ) from exc
            if not restored:
                raise RuntimeError(
                    "Could not restore sync pre-destructive deletion state"
                )
            parent_restored = external_sync_service.cancel_task_deletion(
                task_id,
                status=previous.get("status") or SyncStatus.IDLE,
                is_active=bool(previous.get("is_active")),
                expected_user_id=owner_user_id,
                expected_deleting_status=SyncStatus.DELETING_CASCADE,
            )
            if not parent_restored:
                raise RuntimeError(
                    "Could not restore sync pre-destructive deletion state"
                )
            if resume_worker:
                resume_worker()

        def _add_dataset_id(
            dataset_id: Optional[str],
            source_task_type: str,
            source_task_id: str,
        ) -> None:
            if not dataset_id:
                return
            expected_dataset_sources.setdefault(dataset_id, set()).add(
                (source_task_type, source_task_id)
            )
            if dataset_id not in seen_dataset_ids:
                seen_dataset_ids.add(dataset_id)
                candidate_dataset_ids.append(dataset_id)

        def _collect_datasets(source_task_type: str, source_task_id: str) -> None:
            datasets = _snapshot_all_pages(
                lambda **page: dataset_service.list_datasets(
                    source_task_type=source_task_type,
                    source_task_id=source_task_id,
                    # This is an internal provenance snapshot, not a tenant
                    # listing. Filtering here would hide source rows whose
                    # owner drifted and let the parent be deleted as an orphan.
                    user_id=None,
                    **page,
                ),
                identity_key="dataset_id",
                resource_name="sync dataset history",
                snapshot_keys=("dataset_id",),
            )
            for dataset in datasets:
                _add_dataset_id(
                    dataset.get("dataset_id"),
                    source_task_type,
                    source_task_id,
                )

        try:
            _collect_datasets("sync", task_id)
            for generation in generation_snapshot:
                generation_task_id = generation.get("generation_task_id")
                output_dataset_id = generation.get("output_dataset_id")
                if output_dataset_id and not generation_task_id:
                    raise HTTPException(
                        status_code=409,
                        detail="Sync generation output is missing task provenance",
                    )
                if not generation_task_id:
                    continue
                _add_dataset_id(
                    output_dataset_id,
                    "generation",
                    generation_task_id,
                )
                _collect_datasets("generation", generation_task_id)
                for edge in dataset_lineage_service.get_edges_by_task(
                    generation_task_id
                ):
                    _add_dataset_id(
                        edge.get("to_dataset_id"),
                        "generation",
                        generation_task_id,
                    )

            # Validate the full candidate set before fencing any dataset. A
            # late provenance mismatch must not strand earlier valid rows.
            validated_dataset_records: Dict[str, Dict[str, Any]] = {}
            for dataset_id in candidate_dataset_ids:
                dataset = dataset_service.get_dataset(dataset_id)
                if not dataset:
                    continue
                if owner_user_id and dataset.get("user_id") != owner_user_id:
                    raise PermissionError(
                        f"Dataset owner changed during cascade: {dataset_id}"
                    )
                actual_source = (
                    dataset.get("source_task_type"),
                    dataset.get("source_task_id"),
                )
                if actual_source not in expected_dataset_sources.get(
                    dataset_id,
                    set(),
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Dataset is not owned by this sync cascade: "
                            f"{dataset_id}"
                        ),
                    )
                validated_dataset_records[dataset_id] = dataset

            for dataset_id, dataset in validated_dataset_records.items():
                if dataset.get("status") == "deleting":
                    _mark_existing_child_intent()
                marked = dataset_service.mark_deleting(
                    dataset_id,
                    deletion_owner=dataset_deletion_owner,
                    user_id=owner_user_id,
                )
                if not marked:
                    raise RuntimeError(
                        "Dataset disappeared while marking deletion: "
                        f"{dataset_id}"
                    )
                dataset_records[dataset_id] = dataset
        except DatasetDeletionOwnerConflictError as exc:
            _restore_pre_destructive_state()
            raise HTTPException(
                status_code=409,
                detail="A dataset is already owned by another deletion operation",
            ) from exc
        except Exception:
            _restore_pre_destructive_state()
            raise

        generation_child_ids = tuple(
            sorted(
                {
                    str(item["generation_task_id"])
                    for item in generation_snapshot
                    if item.get("generation_task_id")
                }
            )
        )
        training_child_ids = tuple(
            sorted(
                {
                    str(item["training_task_id"])
                    for item in training_snapshot
                    if item.get("training_task_id")
                }
            )
        )
        try:
            child_cleanup_manifest = _build_sync_child_cleanup_manifest(
                generation_snapshot,
                training_snapshot,
                task_id,
                owner_user_id,
            )
            batch_snapshot, batch_file_manifest = _build_sync_batch_file_manifest(
                task_id,
                owner_user_id,
            )
            raw_batch_paths = {
                path
                for batch in batch_snapshot
                if isinstance((path := batch.get("storage_path")), str) and path
            }
            child_storage_references = {
                value
                for entry in child_cleanup_manifest["generation"]
                for value in entry["artifact_values"]
                if isinstance(value, str) and value
            }
            child_storage_references.update(
                entry["output_value"]
                for entry in child_cleanup_manifest["training"]
                if isinstance(entry["output_value"], str)
                and entry["output_value"]
            )
            active_generation_consumers = (
                generation_task_service.list_active_dataset_consumers(
                    list(dataset_records),
                    exclude_task_ids=generation_child_ids,
                )
            )
            dataset_paths = sorted(
                {
                    path
                    for dataset in dataset_records.values()
                    for path in (
                        dataset.get("storage_path"),
                        dataset.get("storage_uri"),
                    )
                    if isinstance(path, str) and path
                }
            )
            consumer_storage_paths = sorted(
                set(dataset_paths) | child_storage_references | raw_batch_paths
            )
            active_training_consumers = (
                training_task_service.list_active_dataset_consumers(
                    consumer_storage_paths,
                    exclude_task_ids=training_child_ids,
                )
            )
            active_evaluation_consumers = (
                evaluation_task_service.list_active_dataset_consumers(
                    list(dataset_records),
                    consumer_storage_paths,
                )
            )
            milvus_links = milvus_collection_service.list_dataset_links(
                dataset_records
            )
            external_milvus_links = _external_sync_milvus_links(
                config,
                training_snapshot,
                milvus_links,
            )

            if (
                active_generation_consumers
                or active_training_consumers
                or active_evaluation_consumers
                or external_milvus_links
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Cannot cascade delete datasets with active consumers. "
                        f"generation={active_generation_consumers[:5]}, "
                        f"training={active_training_consumers[:5]}, "
                        f"evaluation={active_evaluation_consumers[:5]}, "
                        "external_milvus_link_count="
                        f"{len(external_milvus_links)}"
                    ),
                )

            owned_linked_collections = tuple(
                sorted(
                    {
                        link["collection_name"]
                        for link in milvus_links
                        if isinstance(link.get("collection_name"), str)
                    }
                )
            )
            owned_milvus_links_by_dataset: Dict[str, set[str]] = {}
            for link in milvus_links:
                dataset_id = link.get("dataset_id")
                collection_name = link.get("collection_name")
                if (
                    dataset_id in dataset_records
                    and isinstance(collection_name, str)
                    and _is_owned_sync_milvus_collection(
                        config,
                        training_snapshot,
                        collection_name,
                    )
                ):
                    owned_milvus_links_by_dataset.setdefault(
                        str(dataset_id),
                        set(),
                    ).add(collection_name)

            # Snapshot every dataset storage reference before validating or
            # mutating any target. This guarantees late asset-read failures
            # still precede all physical cleanup.
            trusted_dataset_roots = [
                sync_data_root,
                generation_output_root,
                str(get_settings().datasets_dir),
            ]
            raw_dataset_storage_references: Dict[str, set[str]] = {}
            for dataset_id, dataset in dataset_records.items():
                storage_references: set[str] = set()
                for reference in (
                    dataset.get("storage_path"),
                    dataset.get("storage_uri"),
                ):
                    if reference in (None, ""):
                        continue
                    if not isinstance(reference, str):
                        raise HTTPException(
                            status_code=409,
                            detail="Dataset storage manifest is invalid",
                        )
                    storage_references.add(reference)

                for asset in dataset_asset_service.list_assets(
                    dataset_id=dataset_id
                ):
                    reference = asset.get("storage_uri")
                    if reference in (None, ""):
                        continue
                    if not isinstance(reference, str):
                        raise HTTPException(
                            status_code=409,
                            detail="Dataset asset manifest is invalid",
                        )
                    storage_references.add(reference)

                raw_dataset_storage_references[dataset_id] = storage_references

            all_storage_references = {
                reference
                for references in raw_dataset_storage_references.values()
                for reference in references
            }
            all_storage_references.update(child_storage_references)
            all_storage_references.update(raw_batch_paths)
            external_storage_consumers = (
                dataset_service.list_external_storage_reference_consumers(
                    all_storage_references,
                    exclude_dataset_ids=tuple(dataset_records),
                )
            )
            if external_storage_consumers:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Cannot cascade delete shared dataset storage; "
                        f"external_reference_count={len(external_storage_consumers)}"
                    ),
                )
            external_generation_storage_consumers = (
                generation_task_service.list_artifact_reference_consumers(
                    all_storage_references,
                    exclude_task_ids=generation_child_ids,
                )
            )
            if external_generation_storage_consumers:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Cannot cascade delete storage referenced by another "
                        "generation task; external_reference_count="
                        f"{len(external_generation_storage_consumers)}"
                    ),
                )

            for dataset_id, storage_references in (
                raw_dataset_storage_references.items()
            ):
                resolved_references: List[tuple[str, Any]] = []
                for reference in sorted(storage_references):
                    if reference.startswith("s3://"):
                        resolved = _resolve_s3_object(reference)
                        if resolved is None:
                            raise HTTPException(
                                status_code=409,
                                detail="Dataset object manifest is invalid",
                            )
                        resolved_references.append(("s3", resolved))
                    else:
                        resolved_path = _resolve_safe_delete_path(
                            reference,
                            trusted_roots=trusted_dataset_roots,
                            expected_leaf_name=dataset_id,
                        )
                        if resolved_path is not None:
                            resolved_references.append(("local", resolved_path))
                dataset_storage_manifest[dataset_id] = tuple(
                    resolved_references
                )

            raw_config = external_sync_service.get_task_raw(task_id) or config
        except Exception:
            _restore_pre_destructive_state()
            raise

        try:
            fenced_milvus_collections = tuple(
                _delete_sync_milvus_collections(
                    raw_config,
                    training_snapshot,
                    owned_linked_collections,
                    tuple(candidate_dataset_ids),
                    generation_child_ids,
                    on_preflight_failure=_restore_pre_destructive_state,
                    on_durable_intent_detected=_mark_existing_child_intent,
                )
            )
        except (
            SyncMilvusCollectionConsumerError,
            SyncMilvusCollectionSharedError,
        ) as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot cascade delete Milvus collections with active "
                    "or external consumers"
                ),
            ) from exc

        def _delete_dataset(ds: Dict[str, Any]):
            ds_id = ds["dataset_id"]
            for storage_kind, resolved_reference in dataset_storage_manifest.get(
                ds_id,
                (),
            ):
                if storage_kind == "s3":
                    store, object_key = resolved_reference
                    store.delete_object(object_key)
                else:
                    _delete_resolved_path(resolved_reference)
            dataset_lineage_service.delete_edges_for_dataset(ds_id)
            dataset_asset_service.delete_assets_for_dataset(ds_id)
            for collection_name in sorted(
                owned_milvus_links_by_dataset.get(ds_id, set())
            ):
                milvus_collection_service.unlink_dataset(
                    collection_name,
                    ds_id,
                )
            try:
                deleted = dataset_service.delete_dataset(
                    ds_id,
                    deletion_owner=dataset_deletion_owner,
                    user_id=owner_user_id,
                )
            except DatasetDeletionOwnerConflictError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="A dataset deletion owner changed during cleanup",
                ) from exc
            if not deleted:
                raise RuntimeError(f"Failed to delete dataset record: {ds_id}")
            deleted_datasets.append(ds_id)

        for dataset_id in candidate_dataset_ids:
            dataset = dataset_records.get(dataset_id)
            if dataset:
                _delete_dataset(dataset)

        _delete_sync_child_tasks(
            generation_snapshot,
            training_snapshot,
            task_id,
            owner_user_id,
            child_cleanup_manifest,
        )

    # Batch rows remain durable until every managed file has been removed.
    for batch_path in batch_file_manifest:
        try:
            _delete_resolved_path(batch_path)
        except FileNotFoundError:
            pass

    if fenced_milvus_collections:
        from ...storage.services.milvus_collection_service import (
            MilvusCollectionDeletionOwnerConflictError,
            milvus_collection_service,
        )

        try:
            milvus_collection_service.delete_collections(
                fenced_milvus_collections,
                deletion_owner=f"sync:{task_id}",
            )
        except MilvusCollectionDeletionOwnerConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="A Milvus collection deletion owner changed during cleanup",
            ) from exc

    try:
        finalized = external_sync_service.finalize_task_deletion(
            task_id,
            cascade=cascade,
            expected_user_id=config.get("user_id"),
            expected_task_identity=config.get("_deletion_identity"),
            batch_snapshot=batch_snapshot,
            generation_snapshot=tuple(generation_snapshot),
            training_snapshot=tuple(training_snapshot),
            expected_training_target_ids=tuple(
                config.get("_training_target_ids") or ()
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not finalized:
        raise RuntimeError("Sync task disappeared during deletion finalization")

    result: Dict[str, Any] = {"message": "Sync task deleted"}
    if cascade:
        result["deleted_datasets"] = deleted_datasets
    return result


async def resume_pending_sync_deletions() -> tuple[int, int]:
    """Retry durable sync deletions before periodic workers start."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_manager import sync_manager

    tasks = _snapshot_all_pages(
        lambda **page: external_sync_service.list_tasks(**page),
        identity_key="task_id",
        resource_name="sync task deletion recovery",
        snapshot_keys=("task_id", "status", "user_id", "is_active"),
    )
    resumed = 0
    failed = 0
    for config in tasks:
        status = config.get("status")
        if status not in {SyncStatus.DELETING, SyncStatus.DELETING_CASCADE}:
            continue
        task_id = str(config["task_id"])
        cascade = status == SyncStatus.DELETING_CASCADE
        try:
            async with sync_manager.task_operation_lock(task_id):
                latest = external_sync_service.get_task(task_id)
                if not latest:
                    continue
                if not _is_matching_delete_intent(latest, cascade=cascade):
                    continue
                _delete_sync_task_locked(
                    task_id,
                    cascade=cascade,
                    config=latest,
                    original_config=latest,
                    sync_manager=sync_manager,
                )
            resumed += 1
        except Exception:
            failed += 1
            logger.exception(
                "Could not resume pending sync deletion for task %s",
                task_id,
            )
    return resumed, failed


# ── Worker Control ─────────────────────────────────────────────────


@router.post("/tasks/{task_id}/start")
async def start_sync(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Activate sync task and start background worker."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_manager import sync_manager

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    async with sync_manager.task_operation_lock(task_id):
        latest = external_sync_service.get_task(task_id)
        latest = verify_resource_ownership(latest, current_user, "Sync task")
        _require_sync_task_mutable(latest)
        updated = external_sync_service.update_task(
            task_id,
            expected_user_id=latest.get("user_id"),
            is_active=True,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Sync task not found")
    sync_manager.start_worker(task_id)

    return {"message": "Sync started", "worker_status": sync_manager.get_worker_status(task_id)}


@router.post("/tasks/{task_id}/stop")
async def stop_sync(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Deactivate sync task and stop background worker."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_manager import sync_manager

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    async with sync_manager.task_operation_lock(task_id):
        latest = external_sync_service.get_task(task_id)
        latest = verify_resource_ownership(latest, current_user, "Sync task")
        _require_sync_task_mutable(latest)
        updated = external_sync_service.update_task(
            task_id,
            expected_user_id=latest.get("user_id"),
            is_active=False,
            status=SyncStatus.IDLE,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Sync task not found")
        await sync_manager.stop_worker(task_id)

    return {"message": "Sync stopped", "worker_status": sync_manager.get_worker_status(task_id)}


@router.post("/tasks/{task_id}/sync-now")
async def sync_now(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Manually trigger one sync cycle (bypass interval wait)."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_manager import (
        SyncGenerationReconciliationError,
        SyncTaskBusyError,
        sync_manager,
    )

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    try:
        await sync_manager.run_once(task_id)
        return {"message": "Sync cycle completed"}
    except BackgroundTaskCapacityExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SyncTaskBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SyncGenerationReconciliationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Sync generation state is temporarily unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Manual sync cycle failed for task %s", task_id)
        raise HTTPException(status_code=500, detail="Sync failed") from exc


@router.post("/tasks/{task_id}/trigger-generation")
async def trigger_generation(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Manually trigger generation (ignore threshold)."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_manager import (
        SyncGenerationReconciliationError,
        SyncTaskBusyError,
        sync_manager,
    )

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    try:
        await sync_manager.trigger_generation(task_id)
        return {"message": "Generation triggered"}
    except BackgroundTaskCapacityExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SyncTaskBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SyncGenerationReconciliationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Sync generation state is temporarily unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Manual generation trigger failed for task %s", task_id)
        raise HTTPException(
            status_code=500,
            detail="Generation trigger failed",
        ) from exc


@router.post("/tasks/{task_id}/trigger-training")
async def trigger_training(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Manually trigger training (ignore threshold)."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_manager import SyncTaskBusyError, sync_manager

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    try:
        await sync_manager.trigger_training(task_id)
        return {"message": "Training triggered"}
    except SyncTaskBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training trigger failed: {e}")


@router.post("/tasks/{task_id}/trainings/{training_task_id}/retry-adapter-load")
async def retry_adapter_load(
    task_id: str,
    training_task_id: str,
    replace: bool = Query(default=True, description="Unload all existing adapters before loading"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Retry loading adapter for a training that failed at the adapter loading step."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...storage.services.training_task_service import training_task_service
    from ...sync.post_training_handler import load_adapter_for_training
    from ...sync.sync_manager import sync_manager

    async with sync_manager.task_operation_lock(task_id):
        config = external_sync_service.get_task(task_id)
        config = verify_resource_ownership(config, current_user, "Sync task")
        _require_sync_task_mutable(config)

        sync_training = external_sync_service.get_training_by_task_id(
            training_task_id
        )
        if not sync_training:
            raise HTTPException(
                status_code=404,
                detail="Sync training record not found",
            )
        if sync_training.get("task_id") != task_id:
            raise HTTPException(
                status_code=404,
                detail="Sync training record not found for this task",
            )
        if sync_training.get("user_id") != config.get("user_id"):
            raise HTTPException(status_code=403, detail="Forbidden")

        allowed_training_statuses = {
            SyncTrainingStatus.ADAPTER_LOAD_FAILED,
            SyncTrainingStatus.ADAPTER_FAILED,
            SyncTrainingStatus.COMPLETED,
            SyncTrainingStatus.ADAPTER_UNLOADED,
            SyncTrainingStatus.ADAPTER_LOADED,
        }
        if sync_training.get("status") not in allowed_training_statuses:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Training status is '{sync_training.get('status')}', "
                    "cannot load adapter"
                ),
            )

        try:
            training_guard = background_task_admission_service.begin_deletion(
                "training",
                training_task_id,
            )
        except BackgroundTaskAlreadyExecuting as exc:
            raise HTTPException(
                status_code=409,
                detail="Training task is executing or being deleted",
            ) from exc

        guard_handed_off = False
        try:
            # Resolve the artifact only after deletion has been excluded.
            task_info = training_task_service.get_task(training_task_id)
            if not task_info:
                raise HTTPException(
                    status_code=404,
                    detail="Training task not found",
                )
            task_info = verify_resource_ownership(
                task_info,
                current_user,
                "Training task",
            )

            final_model_path = task_info.get("final_model_path")
            if not final_model_path:
                raise HTTPException(
                    status_code=400,
                    detail="Training task has no final_model_path",
                )

            model_registry_id = task_info.get("model_registry_id")

            import threading

            thread = threading.Thread(
                target=(
                    background_task_admission_service.run_with_deletion_guard
                ),
                args=(
                    training_guard,
                    load_adapter_for_training,
                    task_id,
                    training_task_id,
                    final_model_path,
                    model_registry_id,
                    replace,
                ),
                daemon=True,
            )
            thread.start()
            guard_handed_off = True
        finally:
            if not guard_handed_off:
                training_guard.release()

    return {"message": "Adapter loading retry started"}


@router.post("/tasks/{task_id}/unload-adapter")
async def unload_current_adapter(
    task_id: str,
    target_id: Optional[str] = Query(default=None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Unload the current adapter from one target runtime binding."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.post_training_handler import unload_current_adapter as _unload

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)
    if target_id:
        _get_training_target_for_task(task_id, target_id)

    try:
        await asyncio.to_thread(_unload, task_id, target_id)
        return {"message": "Adapter unloaded"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload adapter: {e}")


# ── History Queries ────────────────────────────────────────────────


@router.get("/tasks/{task_id}/batches")
async def list_batches(
    task_id: str,
    status: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List sync batches for a task."""
    from ...storage.services.external_sync_service import external_sync_service

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")

    batches, total = external_sync_service.list_batches(
        task_id=task_id, status=status, limit=limit, offset=offset,
    )
    return {"batches": batches, "total": total}


@router.get("/tasks/{task_id}/generations")
async def list_generations(
    task_id: str,
    status: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List generation tasks for a task."""
    from ...storage.services.external_sync_service import external_sync_service

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")

    generations, total = external_sync_service.list_generations(
        task_id=task_id, status=status, limit=limit, offset=offset,
    )
    return {"generations": generations, "total": total}


@router.patch("/tasks/{task_id}/generations/{generation_task_id}/disabled")
async def toggle_generation_disabled(
    task_id: str,
    generation_task_id: str,
    body: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Toggle the disabled flag on a generation record."""
    from ...storage.services.external_sync_service import external_sync_service

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    disabled = body.get("disabled")
    if disabled is None or not isinstance(disabled, bool):
        raise HTTPException(status_code=400, detail="'disabled' (bool) is required")

    generation = external_sync_service.get_generation_by_task_id(generation_task_id)
    if not generation or generation.get("task_id") != task_id:
        raise HTTPException(status_code=404, detail="Generation record not found")
    if generation.get("user_id") != config.get("user_id"):
        raise HTTPException(status_code=403, detail="Forbidden")
    if generation.get("status") != SyncGenerationStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Generation status is '{generation.get('status')}', only completed records can be toggled",
        )

    result = external_sync_service.toggle_generation_disabled(generation_task_id, disabled)
    if not result:
        raise HTTPException(status_code=404, detail="Generation record not found")
    return result


@router.post("/tasks/{task_id}/recalculate-counters")
async def recalculate_counters(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Recalculate training sample counters from actual generation records.

    Fixes counter drift caused by direct DB edits or failed toggle operations.
    Returns old and new values for both total_training_samples and pending_training_samples.
    """
    from ...storage.services.external_sync_service import external_sync_service

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    result = external_sync_service.recalculate_sample_counters(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Sync task not found")
    return result


@router.get("/tasks/{task_id}/trainings")
async def list_trainings(
    task_id: str,
    status: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List training tasks for a task."""
    from ...storage.services.external_sync_service import external_sync_service

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")

    trainings, total = external_sync_service.list_trainings(
        task_id=task_id, status=status, limit=limit, offset=offset,
    )
    return {"trainings": trainings, "total": total}


@router.get("/tasks/{task_id}/status")
async def get_sync_status(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get comprehensive sync status including worker state and counters."""
    from ...storage.services.external_sync_service import external_sync_service
    from ...sync.sync_manager import sync_manager

    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")

    targets = external_sync_service.list_training_targets(task_id)

    return {
        "task_id": task_id,
        "status": config.get("status", "unknown"),
        "is_active": config.get("is_active", False),
        "worker_status": sync_manager.get_worker_status(task_id),
        "pending_record_count": config.get("pending_record_count", 0),
        "generation_threshold": config.get("generation_threshold", 0),
        "pending_training_samples": config.get("pending_training_samples", 0),
        "training_threshold": config.get("training_threshold", 0),
        "total_record_count": config.get("total_record_count", 0),
        "total_training_samples": config.get("total_training_samples", 0),
        "total_trainings": config.get("total_trainings", 0),
        "current_adapter_name": config.get("current_adapter_name"),
        "last_sync_at": config.get("last_sync_at"),
        "training_targets": targets,
    }


# ── Training Targets ──────────────────────────────────

@router.get("/tasks/{task_id}/targets")
async def list_training_targets(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List training targets for a sync task."""
    from ...storage.services.external_sync_service import external_sync_service
    config = external_sync_service.get_task(task_id)
    verify_resource_ownership(config, current_user, "Sync task")
    targets = _get_task_training_targets(task_id, config)
    return {"targets": targets, "total": len(targets)}


@router.post("/tasks/{task_id}/targets", status_code=201)
async def create_training_target(
    task_id: str,
    request: TrainingTargetCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Create a training target for a sync task."""
    from ...storage.services.external_sync_service import external_sync_service
    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    from ...sync.sync_manager import sync_manager

    try:
        async with sync_manager.task_operation_lock(task_id):
            config = external_sync_service.get_task(task_id)
            config = verify_resource_ownership(config, current_user, "Sync task")
            _require_sync_task_mutable(config)
            _validate_base_deployment_reference(
                request.base_deployment_id,
                current_user,
                config.get("external_api_config_id"),
                request.base_deployment_replica_id,
            )
            target = external_sync_service.create_training_target(
                task_id=task_id,
                expected_user_id=config.get("user_id"),
                **request.model_dump(),
            )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"message": "Training target created", "target": target}


@router.patch("/tasks/{task_id}/targets/{target_id}")
async def update_training_target(
    task_id: str,
    target_id: str,
    request: TrainingTargetUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Update a training target."""
    from ...storage.services.external_sync_service import external_sync_service
    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    raw_updates = request.model_dump(exclude_unset=True)
    nullable_binding_fields = {
        "base_deployment_id",
        "base_deployment_replica_id",
    }
    updates = {
        key: value
        for key, value in raw_updates.items()
        if value is not None or key in nullable_binding_fields
    }
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    from ...sync.sync_manager import sync_manager

    try:
        async with sync_manager.task_operation_lock(task_id):
            config = external_sync_service.get_task(task_id)
            config = verify_resource_ownership(config, current_user, "Sync task")
            _require_sync_task_mutable(config)
            target = _get_or_materialize_training_target_for_write(
                task_id,
                target_id,
                config,
            )
            if "base_deployment_id" in updates or "base_deployment_replica_id" in updates:
                _validate_base_deployment_reference(
                    updates.get("base_deployment_id", target.get("base_deployment_id")),
                    current_user,
                    config.get("external_api_config_id"),
                    updates.get(
                        "base_deployment_replica_id",
                        target.get("base_deployment_replica_id"),
                    ),
                )
            updated = external_sync_service.update_training_target(
                target["target_id"],
                task_id=task_id,
                expected_user_id=config.get("user_id"),
                **updates,
            )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="Training target not found")
    return {"message": "Training target updated", "target": updated}


@router.delete("/tasks/{task_id}/targets/{target_id}")
async def delete_training_target(
    task_id: str,
    target_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a training target."""
    from ...storage.services.external_sync_service import external_sync_service
    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    from ...sync.sync_manager import sync_manager

    try:
        async with sync_manager.task_operation_lock(task_id):
            config = external_sync_service.get_task(task_id)
            config = verify_resource_ownership(config, current_user, "Sync task")
            _require_sync_task_mutable(config)
            target = _get_or_materialize_training_target_for_write(
                task_id,
                target_id,
                config,
            )
            _require_training_target_mutable(target)
            deleted = external_sync_service.delete_training_target(
                target["target_id"],
                task_id=task_id,
                expected_user_id=config.get("user_id"),
            )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Training target not found")
    return {"message": "Training target deleted"}


@router.post("/tasks/{task_id}/targets/{target_id}/trigger-training")
async def trigger_target_training(
    task_id: str,
    target_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Manually trigger training for a specific target (ignore threshold)."""
    from ...storage.services.external_sync_service import external_sync_service
    config = external_sync_service.get_task(task_id)
    config = verify_resource_ownership(config, current_user, "Sync task")
    _require_sync_task_mutable(config)

    try:
        target = _get_or_materialize_training_target_for_write(
            task_id,
            target_id,
            config,
        )
        _validate_base_deployment_reference(
            target.get("base_deployment_id"),
            current_user,
            config.get("external_api_config_id"),
            target.get("base_deployment_replica_id"),
        )
        from ...sync.level2_handler import _trigger_training_for_target
        raw_target = external_sync_service.get_training_target_raw(target_id)
        if not raw_target or raw_target.get("task_id") != task_id:
            raise HTTPException(status_code=404, detail="Training target not found")
        launched = _trigger_training_for_target(task_id, raw_target)
        if not launched:
            raise HTTPException(
                status_code=409,
                detail="Training target is already running or has no trainable data",
            )
        return {"message": f"Training triggered for target '{target['target_name']}'"}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
