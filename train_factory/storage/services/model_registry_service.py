"""
Model registry service for database operations.
"""

import logging
import math
import numbers
import shutil
import threading
from copy import deepcopy
from datetime import datetime
from ...utils.path_utils import (
    artifact_path_uses_root,
    canonicalize_artifact_path,
    hash_path,
)
import os
from typing import Optional, List, Dict, Any, Iterable, Tuple
from uuid import uuid4
from train_factory.core.time_utils import now_naive

from sqlalchemy import delete, update
from sqlmodel import select, or_, func

from ..database import get_session
from ..entities.model_registry_entity import (
    MODEL_DELETE_INTENT_METADATA_KEY,
    ModelRegistryDB,
    ModelVersionDB,
    public_model_extra_metadata,
)
from ..entities.deployment_entity import DeploymentDB
from ..entities.deployment_replica_entity import DeploymentReplicaDB
from ..entities.loaded_adapter_entity import LoadedAdapterDB
from ..entities.training_task_entity import TrainingTaskDB
from .background_task_admission_service import background_task_admission_service
from .model_artifact_membership_service import lock_model_artifact_membership
from .runtime_dependency_service import (
    RuntimeDependencyClaimConflictError,
    RuntimeDependencyUnavailableError,
    _guard_model_delete_runtime_dependencies,
    _lock_model_delete_dependency_scope,
    snapshot_runtime_executions,
)
from ...config import settings
from ...core.remote_download_security import resolve_managed_artifact_directory

logger = logging.getLogger(__name__)

_UNSET_USER_ID = object()
_ACTIVE_ADAPTER_STATUSES = ("loading", "loaded", "unloading")
_ACTIVE_TRAINING_STATUSES = ("pending", "preparing", "running", "evaluating")
MODEL_DELETE_INTENT_LEASE_SECONDS = 120
MODEL_DELETE_INTENT_HEARTBEAT_SECONDS = 15


class ModelDeletionInProgressError(ValueError):
    """Raised when a dependency writer encounters a model delete intent."""


class ModelDeletionOwnershipLostError(RuntimeError):
    """Raised only after a heartbeat proves its durable token is no longer owned."""


def _normalize_artifact_path(path: Optional[str]) -> Optional[str]:
    """Compatibility alias for the shared artifact path canonicalizer."""
    return canonicalize_artifact_path(path)


def _is_same_or_descendant(path: str, root: str) -> bool:
    return artifact_path_uses_root(path, root)


def _artifact_paths_overlap(left: Optional[str], right: Optional[str]) -> bool:
    return artifact_path_uses_root(left, right) or artifact_path_uses_root(
        right,
        left,
    )


def _lock_registry_membership_write_scope(
    session,
    *,
    model_name: str,
    user_id: Optional[str],
    artifact_paths: Iterable[Optional[str]],
    required_model_id: Optional[str] = None,
) -> List[ModelRegistryDB]:
    """Lock one family/path-owner union after the caller holds the gate."""
    paths = [path for path in artifact_paths if _normalize_artifact_path(path)]
    candidates = list(
        session.exec(
            select(ModelRegistryDB).order_by(ModelRegistryDB.model_id)
        ).all()
    )
    if required_model_id is not None and all(
        model.model_id != required_model_id for model in candidates
    ):
        return []

    def belongs_to_family(model: ModelRegistryDB) -> bool:
        return bool(
            model.model_name == model_name
            and model.user_id == user_id
        )

    def owns_prospective_path(model: ModelRegistryDB) -> bool:
        return any(
            _artifact_paths_overlap(path, model.model_path) for path in paths
        )

    expected_ids = tuple(
        sorted(
            model.model_id
            for model in candidates
            if model.model_id == required_model_id
            or belongs_to_family(model)
            or owns_prospective_path(model)
        )
    )
    if not expected_ids:
        return []

    locked = list(
        session.exec(
            select(ModelRegistryDB)
            .where(ModelRegistryDB.model_id.in_(expected_ids))
            .order_by(ModelRegistryDB.model_id)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).all()
    )
    if tuple(model.model_id for model in locked) != expected_ids:
        raise ModelDeletionInProgressError(
            "Registry model membership changed while acquiring locks; retry"
        )
    for model in locked:
        if not (
            model.model_id == required_model_id
            or belongs_to_family(model)
            or owns_prospective_path(model)
        ):
            raise ModelDeletionInProgressError(
                "Registry model paths changed while acquiring locks; retry"
            )
        if model.status == "deleting" or _model_delete_intent_token(model):
            raise ModelDeletionInProgressError(
                f"Model is being deleted: {model.model_id}"
            )
    return locked


def _require_no_other_registry_artifact_overlap(
    session,
    *,
    model_id: str,
    model_path: str,
    lock: bool,
) -> None:
    model_candidates = list(session.exec(select(ModelRegistryDB)).all())
    matching_model_ids = sorted(
        model.model_id
        for model in model_candidates
        if model.model_id != model_id
        and _artifact_paths_overlap(model.model_path, model_path)
    )
    if matching_model_ids:
        statement = select(ModelRegistryDB).where(
            ModelRegistryDB.model_id.in_(matching_model_ids)
        ).order_by(ModelRegistryDB.model_id)
        if lock:
            statement = statement.with_for_update()
        locked_models = list(session.exec(statement).all())
        if {model.model_id for model in locked_models} != set(matching_model_ids):
            raise ModelDeletionInProgressError(
                "Registry model membership changed while checking artifact overlap"
            )
        if any(
            not _artifact_paths_overlap(model.model_path, model_path)
            for model in locked_models
        ):
            raise ModelDeletionInProgressError(
                "Registry model paths changed while checking artifact overlap"
            )
        raise ValueError(
            f"Cannot delete model {model_id}: registry artifact paths overlap."
        )

    version_candidates = list(
        session.exec(
            select(ModelVersionDB).where(ModelVersionDB.model_id != model_id)
        ).all()
    )
    matching_version_ids = sorted(
        version.version_id
        for version in version_candidates
        if _artifact_paths_overlap(version.model_path, model_path)
    )
    if matching_version_ids:
        statement = select(ModelVersionDB).where(
            ModelVersionDB.version_id.in_(matching_version_ids)
        ).order_by(ModelVersionDB.version_id)
        if lock:
            statement = statement.with_for_update()
        locked_versions = list(session.exec(statement).all())
        if {version.version_id for version in locked_versions} != set(
            matching_version_ids
        ):
            raise ModelDeletionInProgressError(
                "Registry version membership changed while checking artifact overlap"
            )
        if any(
            not _artifact_paths_overlap(version.model_path, model_path)
            for version in locked_versions
        ):
            raise ModelDeletionInProgressError(
                "Registry version paths changed while checking artifact overlap"
            )
        raise ValueError(
            f"Cannot delete model {model_id}: registry version paths overlap."
        )


def _list_active_loaded_adapter_references(
    session,
    *,
    model_id: str,
    managed_model_path: Optional[str],
    lock: bool = False,
) -> List[LoadedAdapterDB]:
    """Return active adapter rows that keep a registry artifact in use."""
    normalized_model_path = _normalize_artifact_path(managed_model_path)
    statement = select(LoadedAdapterDB).where(
        LoadedAdapterDB.status.in_(_ACTIVE_ADAPTER_STATUSES)
    ).order_by(LoadedAdapterDB.adapter_id)
    if lock:
        statement = statement.with_for_update()
    active_adapters = list(session.exec(statement).all())
    references = []
    for adapter in active_adapters:
        if adapter.source_model_id == model_id:
            references.append(adapter)
            continue
        # A runtime-only adapter discovered by auto-sync may not expose its
        # source path (vLLM commonly reports only its name).  Until an operator
        # unloads or otherwise reconciles that active row, no managed artifact
        # can be proven unreferenced safely.
        if (
            normalized_model_path is not None
            and adapter.adapter_path == "<unknown:auto-sync>"
        ):
            references.append(adapter)
            continue
        normalized_adapter_path = _normalize_artifact_path(adapter.adapter_path)
        if (
            normalized_model_path is not None
            and normalized_adapter_path is not None
            and _is_same_or_descendant(
                normalized_adapter_path,
                normalized_model_path,
            )
        ):
            references.append(adapter)
    return references


def _list_deployment_owned_adapters(
    session,
    deployments: Iterable[DeploymentDB],
    *,
    lock: bool,
) -> List[LoadedAdapterDB]:
    deployment_ids = sorted(
        deployment.deployment_id for deployment in deployments
    )
    if not deployment_ids:
        return []
    statement = select(LoadedAdapterDB).where(
        LoadedAdapterDB.deployment_id.in_(deployment_ids)
    ).order_by(LoadedAdapterDB.adapter_id)
    if lock:
        statement = statement.with_for_update()
    return list(session.exec(statement).all())


def _adapter_snapshot_signature(adapter: LoadedAdapterDB) -> tuple:
    return (
        adapter.adapter_id,
        adapter.deployment_id,
        adapter.deployment_replica_id,
        adapter.adapter_name,
        adapter.adapter_path,
        adapter.source_model_id,
        adapter.status,
    )


def _model_delete_intent_token(model: ModelRegistryDB) -> Optional[str]:
    if model.status != "deleting" or not isinstance(model.extra_metadata, dict):
        return None
    intent = model.extra_metadata.get(MODEL_DELETE_INTENT_METADATA_KEY)
    if not isinstance(intent, dict):
        return None
    token = intent.get("token")
    return token if isinstance(token, str) and token else None


def _training_model_artifact_paths(task: TrainingTaskDB) -> List[str]:
    paths: List[str] = []

    def add_path(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())

    add_path(task.base_model_path)
    if isinstance(task.loss_config, dict):
        add_path(task.loss_config.get("guide_model"))
    params = task.training_params or {}
    if isinstance(params, dict):
        add_path(params.get("base_model_path"))
        nested_loss = params.get("loss_config")
        if isinstance(nested_loss, dict):
            add_path(nested_loss.get("guide_model"))
    return paths


def _list_active_training_model_references(
    session,
    *,
    managed_model_path: str,
    executing_task_ids: Iterable[str] = (),
    lock: bool = False,
) -> List[TrainingTaskDB]:
    normalized_model_path = _normalize_artifact_path(managed_model_path)
    if normalized_model_path is None:
        return []
    executing_task_ids = set(executing_task_ids)
    live_conditions = [
        TrainingTaskDB.status.in_(_ACTIVE_TRAINING_STATUSES),
        TrainingTaskDB.process_pid.is_not(None),
        TrainingTaskDB.process_status.is_not(None),
        TrainingTaskDB.process_create_time.is_not(None),
    ]
    if executing_task_ids:
        live_conditions.append(TrainingTaskDB.task_id.in_(executing_task_ids))
    statement = select(TrainingTaskDB).where(
        or_(*live_conditions)
    ).order_by(TrainingTaskDB.task_id)
    if lock:
        statement = statement.with_for_update()
    tasks = list(session.exec(statement).all())
    return [
        task
        for task in tasks
        if any(
            (
                normalized_candidate := _normalize_artifact_path(candidate)
            )
            is not None
            and _is_same_or_descendant(
                normalized_candidate,
                normalized_model_path,
            )
            for candidate in _training_model_artifact_paths(task)
        )
    ]


def _remove_deployment_container(
    deployment: DeploymentDB,
    model_id: str,
    replicas: Iterable[DeploymentReplicaDB] = (),
    replica_operation_claim: Any = None,
) -> None:
    """Remove a managed deployment runtime before deleting its database row."""
    from ...deployment.deployment_service import deployment_service
    from ...deployment.docker_deployer import docker_deployer

    def require_cleanup_ownership() -> None:
        if replica_operation_claim is None:
            raise RuntimeError(
                "Cannot delete model because deployment cleanup ownership "
                "is missing"
            )
        deployment_service._require_replica_operation_ownership(
            replica_operation_claim
        )

    if deployment_service._is_unmanaged_binding(deployment):
        return

    deploy_mode = deployment_service._normalize_deploy_mode(deployment.deploy_mode)
    if deploy_mode == "container":
        replicas = list(replicas)
        if replicas:
            cleanup_failed = False
            for replica in sorted(replicas, key=lambda item: item.replica_index):
                deployment_service._expected_replica_container_name(
                    deployment,
                    replica,
                )
                try:
                    if not docker_deployer.container_exists_authoritative(
                        replica.container_name
                    ):
                        continue
                    container_id = docker_deployer.get_managed_container_id(
                        replica.container_name,
                        deployment_id=deployment.deployment_id,
                        replica_id=replica.replica_id,
                    )
                except Exception:
                    cleanup_failed = True
                    continue
                require_cleanup_ownership()
                try:
                    if not docker_deployer.remove_container_identity(container_id):
                        cleanup_failed = True
                except Exception:
                    cleanup_failed = True
            if cleanup_failed:
                raise RuntimeError(
                    "Cannot delete model because replica cleanup failed "
                    f"for deployment {deployment.deployment_id}"
                ) from None
            return
        if not deployment.container_name:
            return
        if deployment_service._uses_replica_lifecycle(deployment):
            try:
                canonical_container_exists = (
                    docker_deployer.container_exists_authoritative(
                        deployment.container_name
                    )
                )
            except Exception as inspect_err:
                raise RuntimeError(
                    "Cannot delete model because canonical replica identity "
                    f"is unknown for deployment {deployment.deployment_id}"
                ) from inspect_err
            if not canonical_container_exists:
                return
            # A canonical group without its child row has no durable replica
            # ID with which to validate TrainFactory labels.  Never mutate a
            # same-name replacement or foreign container through the mutable
            # parent mirror.
            raise RuntimeError(
                "Cannot delete model because canonical replica identity is "
                f"missing for deployment {deployment.deployment_id}"
            )
        deployment_service._require_managed_container(deployment)
        require_cleanup_ownership()
        try:
            removed = docker_deployer.remove_container(deployment.container_name)
        except Exception as stop_err:
            raise RuntimeError(
                "Cannot delete model because container cleanup failed "
                f"for deployment {deployment.deployment_id}"
            ) from stop_err
        if not removed:
            raise RuntimeError(
                "Cannot delete model because container cleanup failed "
                f"for deployment {deployment.deployment_id}"
            )
        logger.warning(
            f"Stopped container {deployment.container_name} for model {model_id}"
        )
        return

    if deployment.status not in {"running", "starting", "restarting", "stopping"}:
        return
    if not deployment.model_uid:
        return
    require_cleanup_ownership()
    try:
        client = deployment_service._get_xinference_client(
            deployment.xinference_endpoint,
            user_id=deployment.user_id,
        )
    except Exception as stop_err:
        raise RuntimeError(
            "Cannot delete model because shared runtime cleanup failed "
            f"for deployment {deployment.deployment_id}"
        ) from stop_err
    try:
        client.terminate_model(deployment.model_uid)
    except Exception as stop_err:
        # A previous owner may have completed termination before a later
        # database validation failed.  Renew exact ownership, then accept only
        # an authoritative absent result; query failures and a still-present
        # runtime remain fail-closed.
        require_cleanup_ownership()
        try:
            remaining = client.get_model(deployment.model_uid)
        except Exception as verify_err:
            raise RuntimeError(
                "Cannot delete model because shared runtime cleanup status "
                f"is unknown for deployment {deployment.deployment_id}"
            ) from verify_err
        if remaining is not None:
            raise RuntimeError(
                "Cannot delete model because shared runtime cleanup failed "
                f"for deployment {deployment.deployment_id}"
            ) from stop_err
    logger.warning(
        f"Terminated shared model {deployment.model_uid} for model {model_id}"
    )


class ModelRegistryService:
    """Service for model registry database operations."""

    def __init__(self) -> None:
        self._model_delete_heartbeat_lock = threading.Lock()
        self._model_delete_heartbeats: Dict[
            str,
            Tuple[threading.Event, threading.Thread],
        ] = {}

    @staticmethod
    def lock_model_reference(
        session,
        model_id: str,
        *,
        membership_gate_locked: bool = False,
        delete_owner_token: Optional[str] = None,
    ) -> ModelRegistryDB:
        """Lock a registry row before inserting a dependent record."""
        if not membership_gate_locked:
            lock_model_artifact_membership(session)
        model = session.exec(
            select(ModelRegistryDB)
            .where(ModelRegistryDB.model_id == model_id)
            .with_for_update()
        ).first()
        if model is None:
            raise ValueError(f"Model not found: {model_id}")
        current_delete_token = _model_delete_intent_token(model)
        if (model.status == "deleting" or current_delete_token) and (
            not delete_owner_token or current_delete_token != delete_owner_token
        ):
            raise ModelDeletionInProgressError(
                f"Model is being deleted: {model_id}"
            )
        return model

    @staticmethod
    def lock_model_artifact_references(
        session,
        artifact_paths: Iterable[Optional[str]],
        *,
        required_model_ids: Iterable[str] = (),
        bidirectional: bool = False,
        membership_gate_locked: bool = False,
    ) -> List[ModelRegistryDB]:
        """Gate and lock every registry row owning the supplied paths."""
        if not membership_gate_locked:
            lock_model_artifact_membership(session)
        paths = [path for path in artifact_paths if _normalize_artifact_path(path)]
        required_ids = {model_id for model_id in required_model_ids if model_id}
        candidates = list(session.exec(select(ModelRegistryDB)).all())
        matching_ids = sorted(
            model.model_id
            for model in candidates
            if model.model_id in required_ids
            or any(
                artifact_path_uses_root(path, model.model_path)
                or (
                    bidirectional
                    and artifact_path_uses_root(model.model_path, path)
                )
                for path in paths
            )
        )
        if required_ids - set(matching_ids):
            missing = sorted(required_ids - set(matching_ids))[0]
            raise ValueError(f"Model not found: {missing}")
        if not matching_ids:
            return []
        locked = list(
            session.exec(
                select(ModelRegistryDB)
                .where(ModelRegistryDB.model_id.in_(matching_ids))
                .order_by(ModelRegistryDB.model_id)
                .with_for_update()
            ).all()
        )
        if {model.model_id for model in locked} != set(matching_ids):
            raise ModelDeletionInProgressError(
                "Registry model membership changed while acquiring locks; retry"
            )
        for model in locked:
            if model.model_id not in required_ids and not any(
                artifact_path_uses_root(path, model.model_path)
                or (
                    bidirectional
                    and artifact_path_uses_root(model.model_path, path)
                )
                for path in paths
            ):
                raise ModelDeletionInProgressError(
                    "Registry model paths changed while acquiring locks; retry"
                )
            if model.status == "deleting" or _model_delete_intent_token(model):
                raise ModelDeletionInProgressError(
                    f"Model is being deleted: {model.model_id}"
                )
        return locked

    def _renew_model_delete_intent(self, model_id: str, token: str) -> None:
        with get_session() as session:
            model = session.exec(
                select(ModelRegistryDB)
                .where(ModelRegistryDB.model_id == model_id)
                .with_for_update()
            ).first()
            if model is None or _model_delete_intent_token(model) != token:
                raise ModelDeletionOwnershipLostError(
                    "model deletion ownership was lost"
                )
            metadata = deepcopy(model.extra_metadata) or {}
            intent = deepcopy(metadata[MODEL_DELETE_INTENT_METADATA_KEY])
            intent["heartbeat_at"] = now_naive().isoformat()
            metadata[MODEL_DELETE_INTENT_METADATA_KEY] = intent
            model.extra_metadata = metadata
            model.updated_at = now_naive()
            session.add(model)
            session.commit()

    def _register_model_delete_heartbeat(
        self,
        model_id: str,
        token: str,
    ) -> None:
        stop_event = threading.Event()

        def heartbeat() -> None:
            try:
                while not stop_event.wait(
                    MODEL_DELETE_INTENT_HEARTBEAT_SECONDS
                ):
                    try:
                        self._renew_model_delete_intent(model_id, token)
                    except ModelDeletionOwnershipLostError:
                        logger.warning(
                            "Model delete heartbeat lost ownership for %s",
                            model_id,
                        )
                        return
                    except Exception:
                        logger.exception(
                            "Transient model delete heartbeat failure for %s; retrying",
                            model_id,
                        )
            finally:
                with self._model_delete_heartbeat_lock:
                    current = self._model_delete_heartbeats.get(token)
                    if current is not None and current[1] is threading.current_thread():
                        self._model_delete_heartbeats.pop(token, None)

        thread = threading.Thread(
            target=heartbeat,
            name=f"model-delete-heartbeat-{model_id[:8]}",
            daemon=True,
        )
        with self._model_delete_heartbeat_lock:
            self._model_delete_heartbeats[token] = (stop_event, thread)
        thread.start()

    def _stop_model_delete_heartbeat(self, token: str) -> None:
        with self._model_delete_heartbeat_lock:
            heartbeat = self._model_delete_heartbeats.pop(token, None)
        if heartbeat is None:
            return
        stop_event, thread = heartbeat
        stop_event.set()
        if thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _split_metrics(self, metrics: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        """Split metrics into numeric and raw buckets."""
        if not metrics or not isinstance(metrics, dict):
            return None, None
        numeric: Dict[str, float] = {}
        raw: Dict[str, Any] = {}
        for key, value in metrics.items():
            num = self._normalize_metric_value(value)
            if num is not None:
                numeric[key] = num
            else:
                raw[key] = value
        return (numeric or None), (raw or None)

    def _normalize_metric_value(self, value: Any) -> Optional[float]:
        """Coerce a metric value to float when possible; return None otherwise."""
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, numbers.Number):
            num = float(value)
            return num if math.isfinite(num) else None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("%"):
                text = text[:-1].strip()
            try:
                num = float(text)
            except ValueError:
                return None
            return num if math.isfinite(num) else None
        return None

    def _normalize_metrics(self, metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Normalize metrics to numeric values; drop invalid entries."""
        if not metrics or not isinstance(metrics, dict):
            return None
        normalized: Dict[str, float] = {}
        for key, value in metrics.items():
            num = self._normalize_metric_value(value)
            if num is not None:
                normalized[key] = num
        return normalized or None

    def _get_path_unique_key_for_model(self, model: ModelRegistryDB) -> Optional[str]:
        """Determine a stable salt for model_path hashing when needed."""
        if model.source_type == "external_bind":
            extra = model.extra_metadata or {}
            if isinstance(extra, dict):
                bound_uid = extra.get("bound_model_uid")
                if bound_uid:
                    return str(bound_uid)
            if model.model_name:
                return str(model.model_name)
            return str(model.model_id)
        return None

    def _model_to_dict(self, model: ModelRegistryDB) -> Dict[str, Any]:
        """Convert model ORM object to dictionary."""
        return {
            "id": model.id,
            "model_id": model.model_id,
            "model_name": model.model_name,
            "version": model.version,
            "model_type": model.model_type,
            "source_task_id": model.source_task_id,
            "base_model_path": model.base_model_path,
            "model_path": model.model_path,
            "description": model.description,
            "tags": model.tags,
            "category": model.category,
            "extra_metadata": public_model_extra_metadata(model.extra_metadata),
            "source_type": model.source_type,
            "is_adapter": bool(model.is_adapter),
            "metrics": self._normalize_metrics(model.metrics),
            "file_size": model.file_size,
            "status": model.status,
            "is_latest": model.is_latest,
            "user_id": model.user_id,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }

    def _version_to_dict(self, version: ModelVersionDB) -> Dict[str, Any]:
        """Convert version ORM object to dictionary."""
        return {
            "id": version.id,
            "version_id": version.version_id,
            "model_id": version.model_id,
            "version": version.version,
            "model_path": version.model_path,
            "changelog": version.changelog,
            "metrics": self._normalize_metrics(version.metrics),
            "created_at": version.created_at,
        }

    # ==================== CRUD Operations ====================

    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        version: str = "v1.0.0",
        source_task_id: Optional[str] = None,
        base_model_path: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        path_unique_key: Optional[str] = None,
        is_adapter: bool = False,
    ) -> Dict[str, Any]:
        """Register a new model. Returns dict with model info."""
        if (
            isinstance(extra_metadata, dict)
            and MODEL_DELETE_INTENT_METADATA_KEY in extra_metadata
        ):
            raise ValueError("model delete intent metadata is reserved")
        normalized_metrics, raw_metrics = self._split_metrics(metrics)
        extra_metadata_payload = extra_metadata
        if raw_metrics:
            if isinstance(extra_metadata, dict):
                extra_metadata_payload = dict(extra_metadata)
            else:
                extra_metadata_payload = {}
            extra_metadata_payload["_metrics_raw"] = raw_metrics
        # Filesystem inspection must complete before the short membership
        # transaction takes its singleton gate.
        file_size = None
        if os.path.exists(model_path):
            if os.path.isfile(model_path):
                file_size = os.path.getsize(model_path)
            elif os.path.isdir(model_path):
                file_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(model_path)
                    for filename in filenames
                )
        with get_session() as session:
            lock_model_artifact_membership(session)
            _lock_registry_membership_write_scope(
                session,
                model_name=model_name,
                user_id=user_id,
                artifact_paths=[model_path],
            )

            # Mark previous versions as not latest（原子 UPDATE，防并发注册
            # 产生两个 is_latest=True）
            statement = update(ModelRegistryDB).where(
                ModelRegistryDB.model_name == model_name,
                ModelRegistryDB.is_latest.is_(True),
            )
            if user_id is None:
                statement = statement.where(ModelRegistryDB.user_id.is_(None))
            else:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            session.exec(statement.values(is_latest=False))

            # Determine status: if model_path exists locally, set to "available"
            if status is None:
                if os.path.exists(model_path):
                    status = "available"
                else:
                    status = "registered"

            # Create new model registry entry
            model = ModelRegistryDB(
                model_name=model_name,
                model_path=model_path,
                model_path_hash=hash_path(model_path, path_unique_key),
                model_type=model_type,
                version=version,
                source_task_id=source_task_id,
                base_model_path=base_model_path,
                description=description,
                tags=tags,
                category=category,
                extra_metadata=extra_metadata_payload,
                metrics=normalized_metrics,
                file_size=file_size,
                user_id=user_id,
                is_latest=True,
                status=status,
                source_type=source_type or "trained",
                is_adapter=is_adapter,
            )
            session.add(model)
            # Flush to get model_id without committing
            session.flush()
            session.refresh(model)

            # Also create a version entry
            version_entry = ModelVersionDB(
                model_id=model.model_id,
                version=version,
                model_path=model_path,
                changelog="Initial registration",
                metrics=normalized_metrics,
            )
            session.add(version_entry)

            # Single commit for both model and version
            session.commit()

            logger.info(f"Registered model: {model.model_id} ({model_name} {version})")
            return self._model_to_dict(model)

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by model_id. Returns dict or None."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
            model = session.exec(statement).first()
            if model:
                return self._model_to_dict(model)
            return None

    def get_model_by_path(
        self, model_path: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get model by model_path."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(
                ModelRegistryDB.model_path_hash == hash_path(model_path)
            )
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            model = session.exec(statement).first()
            return self._model_to_dict(model) if model else None

    def list_models_referencing_artifact_paths(
        self,
        artifact_paths: Iterable[Optional[str]],
        *,
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """List same-tenant models stored at or below artifact paths."""
        normalized_roots = {
            normalized
            for path in artifact_paths
            if (normalized := _normalize_artifact_path(path)) is not None
        }
        if not normalized_roots:
            return []

        with get_session() as session:
            models = session.exec(
                select(ModelRegistryDB).where(
                    ModelRegistryDB.user_id == user_id
                )
            ).all()
            references = []
            for model in models:
                normalized_model_path = _normalize_artifact_path(model.model_path)
                if normalized_model_path and any(
                    _is_same_or_descendant(normalized_model_path, root)
                    for root in normalized_roots
                ):
                    references.append(self._model_to_dict(model))
            return references

    def get_model_by_source_task(
        self,
        task_id: str,
        user_id: Any = _UNSET_USER_ID,
    ) -> Optional[Dict[str, Any]]:
        """Get model by source training task ID."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(
                ModelRegistryDB.source_task_id == task_id
            )
            if user_id is not _UNSET_USER_ID:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            model = session.exec(statement).first()
            return self._model_to_dict(model) if model else None

    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
        latest_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List models with optional filters.

        Returns:
            Tuple of (models, total_count)
        """
        with get_session() as session:
            # Build filter conditions
            conditions = []
            if model_type:
                conditions.append(ModelRegistryDB.model_type == model_type)
            if status:
                conditions.append(ModelRegistryDB.status == status)
            if category:
                conditions.append(ModelRegistryDB.category == category)
            if user_id:
                conditions.append(ModelRegistryDB.user_id == user_id)
            if latest_only:
                conditions.append(ModelRegistryDB.is_latest.is_(True))

            # Get total count
            count_stmt = select(func.count()).select_from(ModelRegistryDB)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Get paginated data
            statement = select(ModelRegistryDB)
            for cond in conditions:
                statement = statement.where(cond)
            statement = statement.order_by(ModelRegistryDB.created_at.desc())
            statement = statement.offset(offset).limit(limit)
            models = session.exec(statement).all()
            return [self._model_to_dict(model) for model in models], total

    def update_model(
        self,
        model_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Update model information."""
        if (
            isinstance(extra_metadata, dict)
            and MODEL_DELETE_INTENT_METADATA_KEY in extra_metadata
        ):
            raise ValueError("model delete intent metadata is reserved")
        with get_session() as session:
            statement = (
                select(ModelRegistryDB)
                .where(ModelRegistryDB.model_id == model_id)
                .with_for_update()
            )
            model = session.exec(statement).first()
            if model:
                if model.status == "deleting" or _model_delete_intent_token(model):
                    raise ValueError(f"Model is being deleted: {model_id}")
                if description is not None:
                    model.description = description
                if tags is not None:
                    model.tags = tags
                if category is not None:
                    model.category = category
                if extra_metadata is not None:
                    model.extra_metadata = extra_metadata
                if status is not None:
                    model.update_status(status)
                else:
                    model.updated_at = now_naive()
                session.add(model)
                session.commit()
                logger.info(f"Updated model {model_id}")
                return True
            return False

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate model-registry statistics (user-scoped, filter-independent).

        Used by the list endpoint so dashboard cards reflect real totals rather
        than only the current page.
        """
        with get_session() as session:
            def _grouped(column):
                stmt = select(column, func.count()).group_by(column)
                if user_id:
                    stmt = stmt.where(ModelRegistryDB.user_id == user_id)
                # row[0] = group value, row[1] = count; skip null/empty keys
                return {row[0]: row[1] for row in session.exec(stmt).all() if row[0]}

            count_stmt = select(func.count()).select_from(ModelRegistryDB)
            if user_id:
                count_stmt = count_stmt.where(ModelRegistryDB.user_id == user_id)
            total = session.exec(count_stmt).one()

            return {
                "total": total,
                "by_status": _grouped(ModelRegistryDB.status),
                "by_type": _grouped(ModelRegistryDB.model_type),
            }

    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """Delete a model and its versions.

        Args:
            model_id: The model ID to delete
            force: If True, also delete related deployments and configs

        Returns:
            True if deleted, False if model not found

        Raises:
            ValueError: If model has active deployments or configs and force=False
        """
        from ...deployment.deployment_service import (
            ReplicaOperationBusyError,
            ReplicaOperationClaim,
            ReplicaOperationLostError,
            deployment_service,
        )
        claims: List[ReplicaOperationClaim] = []
        delete_token = str(uuid4())
        intent_owned = False
        heartbeat_registered = False
        external_cleanup_started = False
        original_status: Optional[str] = None
        original_extra_metadata: Optional[Dict[str, Any]] = None
        original_extra_metadata_present = False

        try:
            # Phase A1 publishes only the durable target-model intent.  It
            # commits before Phase A2 discovers or locks any other parent, so
            # this transaction can never form target-model -> union-model
            # lock inversion with a dependency writer.
            with get_session() as session:
                lock_model_artifact_membership(session)
                model = session.exec(
                    select(ModelRegistryDB)
                    .where(ModelRegistryDB.model_id == model_id)
                    .with_for_update()
                ).first()
                if model is None:
                    return False

                if model.download_status in {"pending", "downloading"}:
                    raise ValueError(
                        f"Cannot delete model {model_id}: download is active"
                    )

                now = now_naive()
                existing_token = _model_delete_intent_token(model)
                if model.status == "deleting":
                    intent = (
                        (model.extra_metadata or {}).get(
                            MODEL_DELETE_INTENT_METADATA_KEY
                        )
                        if isinstance(model.extra_metadata, dict)
                        else None
                    )
                    if not existing_token or not isinstance(intent, dict):
                        raise ReplicaOperationBusyError(
                            "model deletion is already in progress"
                        )
                    try:
                        heartbeat_at = datetime.fromisoformat(
                            str(intent["heartbeat_at"])
                        )
                    except (KeyError, TypeError, ValueError):
                        raise ReplicaOperationBusyError(
                            "model deletion is already in progress"
                        ) from None
                    if heartbeat_at.tzinfo is not None:
                        raise ReplicaOperationBusyError(
                            "model deletion is already in progress"
                        )
                    if (
                        now - heartbeat_at
                    ).total_seconds() <= MODEL_DELETE_INTENT_LEASE_SECONDS:
                        raise ReplicaOperationBusyError(
                            "model deletion is already in progress"
                        )
                    original_status_value = intent.get("original_status")
                    original_metadata_value = intent.get(
                        "original_extra_metadata"
                    )
                    original_metadata_present_value = intent.get(
                        "had_extra_metadata"
                    )
                    public_metadata = public_model_extra_metadata(
                        model.extra_metadata
                    )
                    canonical_public_metadata = (
                        public_metadata
                        if original_metadata_present_value
                        else None
                        if public_metadata == {}
                        else public_metadata
                    )
                    if (
                        not isinstance(original_status_value, str)
                        or original_status_value == "deleting"
                        or type(original_metadata_present_value) is not bool
                        or (
                            original_metadata_present_value
                            and not isinstance(original_metadata_value, dict)
                        )
                        or (
                            not original_metadata_present_value
                            and original_metadata_value is not None
                        )
                        or canonical_public_metadata != original_metadata_value
                    ):
                        raise ReplicaOperationBusyError(
                            "model deletion is already in progress"
                        )
                    original_status = original_status_value
                    original_extra_metadata = deepcopy(original_metadata_value)
                    original_extra_metadata_present = (
                        original_metadata_present_value
                    )
                else:
                    if (
                        isinstance(model.extra_metadata, dict)
                        and MODEL_DELETE_INTENT_METADATA_KEY
                        in model.extra_metadata
                    ):
                        raise ValueError("model delete intent metadata is reserved")
                    original_status = model.status
                    original_extra_metadata = deepcopy(model.extra_metadata)
                    original_extra_metadata_present = (
                        model.extra_metadata is not None
                    )

                timestamp = now.isoformat()
                intent_metadata = deepcopy(original_extra_metadata) or {}
                intent_metadata[MODEL_DELETE_INTENT_METADATA_KEY] = {
                    "token": delete_token,
                    "started_at": timestamp,
                    "heartbeat_at": timestamp,
                    "original_status": original_status,
                    "original_extra_metadata": deepcopy(
                        original_extra_metadata
                    ),
                    "had_extra_metadata": original_extra_metadata_present,
                }
                model.status = "deleting"
                model.extra_metadata = intent_metadata
                model.updated_at = now
                session.add(model)
                session.commit()

            intent_owned = True
            self._register_model_delete_heartbeat(model_id, delete_token)
            heartbeat_registered = True

            # Phase A2 starts a fresh transaction and session. Capture the
            # in-process execution state before opening either one.
            phase_a_executing_task_ids = (
                background_task_admission_service.get_executing_task_ids(
                    "training"
                )
            )
            phase_a_runtime_execution_snapshot = snapshot_runtime_executions()
            with get_session() as session:
                lock_model_artifact_membership(session)

                try:
                    dependency_scope = _lock_model_delete_dependency_scope(
                        session,
                        model_id=model_id,
                        delete_token=delete_token,
                        deployment_claims={},
                        execution_snapshot=phase_a_runtime_execution_snapshot,
                        training_execution_task_ids=tuple(
                            phase_a_executing_task_ids
                        ),
                    )
                except RuntimeDependencyClaimConflictError:
                    raise ReplicaOperationBusyError(
                        "deployment replica operation already in progress"
                    ) from None
                model = dependency_scope.model
                deployments = list(dependency_scope.deployments)
                configs = list(dependency_scope.configs)
                phase_a_deployment_adapters = _list_deployment_owned_adapters(
                    session,
                    deployments,
                    lock=True,
                )
                phase_a_deployment_adapter_signatures = {
                    _adapter_snapshot_signature(adapter)
                    for adapter in phase_a_deployment_adapters
                }
                if (deployments or configs) and not force:
                    deps = []
                    if deployments:
                        deps.append(f"{len(deployments)} deployment(s)")
                    if configs:
                        deps.append(f"{len(configs)} config(s)")
                    raise ValueError(
                        f"Cannot delete model {model_id}: has {', '.join(deps)}. "
                        "Use force=True to delete anyway."
                    )

                expected_path = resolve_managed_artifact_directory(
                    settings.models_dir,
                    model_id,
                )
                try:
                    stored_path = resolve_managed_artifact_directory(
                        settings.models_dir,
                        model_id,
                        model.model_path,
                    )
                except ValueError:
                    if model.source_type == "downloaded":
                        raise
                    stored_path = None

                active_adapters = _list_active_loaded_adapter_references(
                    session,
                    model_id=model_id,
                    managed_model_path=(
                        str(stored_path)
                        if stored_path == expected_path
                        else None
                    ),
                    lock=True,
                )
                if active_adapters:
                    raise ValueError(
                        f"Cannot delete model {model_id}: has "
                        f"{len(active_adapters)} active loaded adapter reference(s)."
                    )

                if stored_path == expected_path:
                    _require_no_other_registry_artifact_overlap(
                        session,
                        model_id=model_id,
                        model_path=model.model_path,
                        lock=False,
                    )

                _guard_model_delete_runtime_dependencies(
                    session,
                    dependency_scope,
                    execution_snapshot=phase_a_runtime_execution_snapshot,
                    training_execution_task_ids=tuple(
                        phase_a_executing_task_ids
                    ),
                )

                legacy_deployments = [
                    deployment
                    for deployment in deployments
                    if not deployment_service._uses_replica_lifecycle(deployment)
                ]
                if legacy_deployments:
                    raise ValueError(
                        f"Cannot delete model {model_id}: remove legacy "
                        "deployments through the deployment lifecycle first."
                    )

                replica_deployments = [
                    (deployment.deployment_id, deployment.user_id)
                    for deployment in deployments
                    if deployment_service._uses_replica_lifecycle(deployment)
                ]
                session.commit()

            if force:
                for deployment_id, deployment_user_id in replica_deployments:
                    self._renew_model_delete_intent(model_id, delete_token)
                    claims.append(
                        deployment_service._claim_replica_operation(
                            deployment_id,
                            operation="delete",
                            replica_id=None,
                            user_id=deployment_user_id,
                            _model_delete_token=delete_token,
                        )
                    )

            claims_by_deployment = {
                claim.deployment_id: claim for claim in claims
            }
            delete_claim_owners = {
                claim.deployment_id: (claim.token, claim.generation)
                for claim in claims
            }
            self._renew_model_delete_intent(model_id, delete_token)
            phase_b_executing_task_ids = (
                background_task_admission_service.get_executing_task_ids(
                    "training"
                )
            )
            phase_b_runtime_execution_snapshot = snapshot_runtime_executions()
            with get_session() as session:
                lock_model_artifact_membership(session)
                dependency_scope = _lock_model_delete_dependency_scope(
                    session,
                    model_id=model_id,
                    delete_token=delete_token,
                    deployment_claims=delete_claim_owners,
                    execution_snapshot=phase_b_runtime_execution_snapshot,
                    training_execution_task_ids=tuple(
                        phase_b_executing_task_ids
                    ),
                )
                model = dependency_scope.model
                deployments = list(dependency_scope.deployments)
                configs = list(dependency_scope.configs)
                deployment_adapters = _list_deployment_owned_adapters(
                    session,
                    deployments,
                    lock=True,
                )
                if {
                    _adapter_snapshot_signature(adapter)
                    for adapter in deployment_adapters
                } != phase_a_deployment_adapter_signatures:
                    raise ReplicaOperationBusyError(
                        "deployment adapter set changed during model deletion"
                    )
                if (deployments or configs) and not force:
                    raise ValueError(
                        f"Cannot delete model {model_id}: dependencies changed."
                    )

                expected_path = resolve_managed_artifact_directory(
                    settings.models_dir,
                    model_id,
                )
                try:
                    stored_path = resolve_managed_artifact_directory(
                        settings.models_dir,
                        model_id,
                        model.model_path,
                    )
                except ValueError:
                    if model.source_type == "downloaded":
                        raise
                    stored_path = None

                active_adapters = _list_active_loaded_adapter_references(
                    session,
                    model_id=model_id,
                    managed_model_path=(
                        str(stored_path)
                        if stored_path == expected_path
                        else None
                    ),
                    lock=True,
                )
                if active_adapters:
                    raise ValueError(
                        f"Cannot delete model {model_id}: has "
                        f"{len(active_adapters)} active loaded adapter reference(s)."
                    )

                if stored_path == expected_path:
                    _require_no_other_registry_artifact_overlap(
                        session,
                        model_id=model_id,
                        model_path=model.model_path,
                        lock=False,
                    )

                versions = list(
                    session.exec(
                        select(ModelVersionDB)
                        .where(ModelVersionDB.model_id == model_id)
                        .order_by(ModelVersionDB.version_id)
                        .with_for_update()
                    ).all()
                )

                locked_replicas = _guard_model_delete_runtime_dependencies(
                    session,
                    dependency_scope,
                    execution_snapshot=phase_b_runtime_execution_snapshot,
                    training_execution_task_ids=tuple(
                        phase_b_executing_task_ids
                    ),
                )

                current_replica_ids = {
                    deployment.deployment_id
                    for deployment in deployments
                    if deployment_service._uses_replica_lifecycle(deployment)
                }
                if current_replica_ids != set(claims_by_deployment):
                    raise ReplicaOperationBusyError(
                        "deployment replica set changed during model deletion"
                    )

                replicas_by_deployment: Dict[
                    str, List[DeploymentReplicaDB]
                ] = {
                    deployment.deployment_id: []
                    for deployment in deployments
                }
                for replica in locked_replicas:
                    replicas_by_deployment.setdefault(
                        replica.deployment_id,
                        [],
                    ).append(replica)
                for deployment in deployments:
                    claim = claims_by_deployment.get(deployment.deployment_id)
                    if claim is not None and not (
                        deployment.replica_operation_token == claim.token
                        and deployment.replica_operation_generation
                        == claim.generation
                    ):
                        raise ReplicaOperationLostError(
                            "deployment replica operation ownership was lost"
                        )

                deployment_snapshots = [
                    deployment.model_copy(deep=True) for deployment in deployments
                ]
                replica_snapshots = {
                    deployment_id: [
                        replica.model_copy(deep=True) for replica in replicas
                    ]
                    for deployment_id, replicas in replicas_by_deployment.items()
                }
                snapshot_model_signature = (
                    model.model_path,
                    model.source_type,
                    bool(model.is_adapter),
                    model.download_status,
                )
                snapshot_deployment_signatures = {
                    deployment.deployment_id: (
                        deployment.model_id,
                        deployment.deploy_mode,
                        deployment.container_name,
                        deployment.model_uid,
                        deployment.xinference_endpoint,
                        deployment.inference_framework,
                        deployment.user_id,
                        deployment.replica_operation_token,
                        deployment.replica_operation_generation,
                    )
                    for deployment in deployments
                }
                snapshot_config_ids = {config.config_id for config in configs}
                snapshot_replica_signatures = {
                    deployment_id: {
                        (
                            replica.replica_id,
                            replica.replica_index,
                            replica.container_name,
                        )
                        for replica in replicas
                    }
                    for deployment_id, replicas in replicas_by_deployment.items()
                }
                snapshot_version_ids = {version.version_id for version in versions}
                snapshot_adapter_signatures: set[tuple] = set()
                snapshot_deployment_adapter_signatures = {
                    _adapter_snapshot_signature(adapter)
                    for adapter in deployment_adapters
                }

            def validate_snapshot(
                session,
                *,
                stage: str,
                executing_task_ids: Iterable[str],
                runtime_execution_snapshot,
            ):
                drift_message = (
                    f"model deletion state changed after {stage}; "
                    "database records retained"
                )
                lock_model_artifact_membership(session)
                try:
                    dependency_scope = _lock_model_delete_dependency_scope(
                        session,
                        model_id=model_id,
                        delete_token=delete_token,
                        deployment_claims=delete_claim_owners,
                        execution_snapshot=runtime_execution_snapshot,
                        training_execution_task_ids=tuple(executing_task_ids),
                    )
                except RuntimeDependencyUnavailableError:
                    raise ReplicaOperationLostError(drift_message) from None
                current_model = dependency_scope.model
                if (
                    (
                        current_model.model_path,
                        current_model.source_type,
                        bool(current_model.is_adapter),
                        current_model.download_status,
                    )
                    != snapshot_model_signature
                ):
                    raise ReplicaOperationLostError(drift_message)
                current_deployments = list(dependency_scope.deployments)
                current_configs = list(dependency_scope.configs)
                current_deployment_adapters = _list_deployment_owned_adapters(
                    session,
                    current_deployments,
                    lock=True,
                )
                current_versions = list(
                    session.exec(
                        select(ModelVersionDB)
                        .where(ModelVersionDB.model_id == model_id)
                        .order_by(ModelVersionDB.version_id)
                        .with_for_update()
                    ).all()
                )
                current_adapters = _list_active_loaded_adapter_references(
                    session,
                    model_id=model_id,
                    managed_model_path=(
                        str(stored_path)
                        if stored_path == expected_path
                        else None
                    ),
                    lock=True,
                )
                if stored_path == expected_path:
                    _require_no_other_registry_artifact_overlap(
                        session,
                        model_id=model_id,
                        model_path=current_model.model_path,
                        lock=False,
                    )
                locked_replicas = _guard_model_delete_runtime_dependencies(
                    session,
                    dependency_scope,
                    execution_snapshot=runtime_execution_snapshot,
                    training_execution_task_ids=tuple(executing_task_ids),
                )
                current_deployment_signatures = {
                    deployment.deployment_id: (
                        deployment.model_id,
                        deployment.deploy_mode,
                        deployment.container_name,
                        deployment.model_uid,
                        deployment.xinference_endpoint,
                        deployment.inference_framework,
                        deployment.user_id,
                        deployment.replica_operation_token,
                        deployment.replica_operation_generation,
                    )
                    for deployment in current_deployments
                }
                if (
                    current_deployment_signatures
                    != snapshot_deployment_signatures
                    or {config.config_id for config in current_configs}
                    != snapshot_config_ids
                    or {version.version_id for version in current_versions}
                    != snapshot_version_ids
                    or {
                        _adapter_snapshot_signature(adapter)
                        for adapter in current_deployment_adapters
                    }
                    != snapshot_deployment_adapter_signatures
                    or {
                        (
                            adapter.adapter_id,
                            adapter.source_model_id,
                            adapter.adapter_path,
                            adapter.status,
                        )
                        for adapter in current_adapters
                    }
                    != snapshot_adapter_signatures
                ):
                    raise ReplicaOperationLostError(drift_message)
                current_replicas_by_deployment = {
                    deployment.deployment_id: []
                    for deployment in current_deployments
                }
                for replica in locked_replicas:
                    current_replicas_by_deployment.setdefault(
                        replica.deployment_id,
                        [],
                    ).append(replica)
                for deployment in current_deployments:
                    current_replicas = current_replicas_by_deployment[
                        deployment.deployment_id
                    ]
                    signatures = {
                        (
                            replica.replica_id,
                            replica.replica_index,
                            replica.container_name,
                        )
                        for replica in current_replicas
                    }
                    if signatures != snapshot_replica_signatures.get(
                        deployment.deployment_id,
                        set(),
                    ):
                        raise ReplicaOperationLostError(drift_message)
                return (
                    current_model,
                    current_deployments,
                    current_configs,
                    current_versions,
                    current_replicas_by_deployment,
                    current_deployment_adapters,
                )

            # Revalidate the complete runtime/Milvus dependency fence before
            # the first irreversible runtime mutation.
            self._renew_model_delete_intent(model_id, delete_token)
            pre_runtime_executing_task_ids = (
                background_task_admission_service.get_executing_task_ids(
                    "training"
                )
            )
            pre_runtime_execution_snapshot = snapshot_runtime_executions()
            with get_session() as session:
                validate_snapshot(
                    session,
                    stage="pre-runtime cleanup",
                    executing_task_ids=pre_runtime_executing_task_ids,
                    runtime_execution_snapshot=(
                        pre_runtime_execution_snapshot
                    ),
                )

            # Runtime and filesystem cleanup stay outside every database
            # session. The background heartbeat covers long single calls, and
            # each mutation is fenced again immediately before it begins.
            for deployment in deployment_snapshots:
                self._renew_model_delete_intent(model_id, delete_token)
                external_cleanup_started = True
                _remove_deployment_container(
                    deployment,
                    model_id,
                    replica_snapshots[deployment.deployment_id],
                    claims_by_deployment.get(deployment.deployment_id),
                )

            self._renew_model_delete_intent(model_id, delete_token)
            runtime_validation_executing_task_ids = (
                background_task_admission_service.get_executing_task_ids(
                    "training"
                )
            )
            runtime_validation_execution_snapshot = (
                snapshot_runtime_executions()
            )
            with get_session() as session:
                validate_snapshot(
                    session,
                    stage="runtime cleanup",
                    executing_task_ids=runtime_validation_executing_task_ids,
                    runtime_execution_snapshot=(
                        runtime_validation_execution_snapshot
                    ),
                )

            if stored_path == expected_path and stored_path.exists():
                self._renew_model_delete_intent(model_id, delete_token)
                external_cleanup_started = True
                try:
                    shutil.rmtree(stored_path)
                except OSError as exc:
                    raise RuntimeError(
                        f"Model {model_id} storage cleanup failed; record retained"
                    ) from exc

            self._renew_model_delete_intent(model_id, delete_token)
            final_executing_task_ids = (
                background_task_admission_service.get_executing_task_ids(
                    "training"
                )
            )
            final_runtime_execution_snapshot = snapshot_runtime_executions()
            with get_session() as session:
                (
                    model,
                    deployments,
                    configs,
                    versions,
                    _replicas_by_deployment,
                    deployment_adapters,
                ) = validate_snapshot(
                    session,
                    stage="external cleanup",
                    executing_task_ids=final_executing_task_ids,
                    runtime_execution_snapshot=(
                        final_runtime_execution_snapshot
                    ),
                )

                for config in configs:
                    session.delete(config)
                    logger.warning(
                        "Force deleted config %s for model %s",
                        config.config_id,
                        model_id,
                    )
                for adapter in deployment_adapters:
                    session.delete(adapter)
                for deployment in deployments:
                    claim = claims_by_deployment.get(deployment.deployment_id)
                    if claim is None:
                        session.delete(deployment)
                    else:
                        parent_owned = (
                            select(DeploymentDB.id)
                            .where(
                                DeploymentDB.deployment_id
                                == claim.deployment_id,
                                DeploymentDB.replica_operation_token
                                == claim.token,
                                DeploymentDB.replica_operation_generation
                                == claim.generation,
                            )
                            .exists()
                        )
                        session.exec(
                            delete(DeploymentReplicaDB).where(
                                DeploymentReplicaDB.deployment_id
                                == claim.deployment_id,
                                parent_owned,
                            )
                        )
                        result = session.exec(
                            delete(DeploymentDB).where(
                                DeploymentDB.deployment_id
                                == claim.deployment_id,
                                DeploymentDB.replica_operation_token
                                == claim.token,
                                DeploymentDB.replica_operation_generation
                                == claim.generation,
                            )
                        )
                        if result.rowcount != 1:
                            session.rollback()
                            raise ReplicaOperationLostError(
                                "model deletion state changed after external cleanup; "
                                "database records retained"
                            )
                    logger.warning(
                        "Force deleted deployment %s for model %s",
                        deployment.deployment_id,
                        model_id,
                    )

                for version in versions:
                    session.delete(version)
                session.delete(model)
                session.commit()

            for claim in reversed(claims):
                deployment_service._stop_claim_heartbeat(claim)
            logger.info("Deleted model %s", model_id)
            return True
        except Exception:
            for claim in reversed(claims):
                deployment_service._release_replica_operation(claim)
            if intent_owned and not external_cleanup_started:
                try:
                    with get_session() as session:
                        model = session.exec(
                            select(ModelRegistryDB)
                            .where(ModelRegistryDB.model_id == model_id)
                            .with_for_update()
                        ).first()
                        if (
                            model is not None
                            and _model_delete_intent_token(model) == delete_token
                        ):
                            model.status = original_status or "registered"
                            model.extra_metadata = deepcopy(
                                original_extra_metadata
                            )
                            model.updated_at = now_naive()
                            session.add(model)
                            session.commit()
                except Exception:
                    logger.exception(
                        "Failed to conditionally restore model delete intent for %s",
                        model_id,
                    )
            raise
        finally:
            if heartbeat_registered:
                self._stop_model_delete_heartbeat(delete_token)

    # ==================== Version Management ====================

    def add_version(
        self,
        model_id: str,
        version: str,
        model_path: str,
        changelog: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        set_as_latest: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Add a new version to an existing model."""
        with get_session() as session:
            lock_model_artifact_membership(session)
            normalized_metrics, raw_metrics = self._split_metrics(metrics)
            candidate = session.exec(
                select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
            ).first()
            if candidate is None:
                return None
            family = _lock_registry_membership_write_scope(
                session,
                model_name=candidate.model_name,
                user_id=candidate.user_id,
                artifact_paths=[model_path],
                required_model_id=model_id,
            )
            model = next(
                (item for item in family if item.model_id == model_id),
                None,
            )
            if model is None:
                return None

            # Create version entry
            version_entry = ModelVersionDB(
                model_id=model_id,
                version=version,
                model_path=model_path,
                changelog=changelog,
                metrics=normalized_metrics,
            )
            session.add(version_entry)

            # Update model if setting as latest
            if set_as_latest:
                # Mark all other versions of this model as not latest（原子 UPDATE）
                session.exec(
                    update(ModelRegistryDB)
                    .where(
                        ModelRegistryDB.model_name == model.model_name,
                        ModelRegistryDB.user_id == model.user_id,
                        ModelRegistryDB.is_latest.is_(True),
                    )
                    .values(is_latest=False)
                )

                path_unique_key = self._get_path_unique_key_for_model(model)
                if path_unique_key is None and model.model_path_hash:
                    # Preserve salted hashes for legacy duplicate paths.
                    if model.model_path_hash != hash_path(model.model_path):
                        path_unique_key = str(model.model_id or model.id)

                if raw_metrics:
                    extra_metadata_payload = dict(model.extra_metadata or {})
                    extra_metadata_payload["_metrics_raw"] = raw_metrics
                    model.extra_metadata = extra_metadata_payload

                model.version = version
                model.model_path = model_path
                model.model_path_hash = hash_path(model_path, path_unique_key)
                model.metrics = normalized_metrics
                model.is_latest = True
                model.updated_at = now_naive()
                session.add(model)

            session.commit()
            session.refresh(version_entry)
            logger.info(f"Added version {version} to model {model_id}")
            return self._version_to_dict(version_entry)

    def get_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a model."""
        with get_session() as session:
            statement = select(ModelVersionDB).where(ModelVersionDB.model_id == model_id)
            statement = statement.order_by(ModelVersionDB.created_at.desc())
            versions = session.exec(statement).all()
            return [self._version_to_dict(v) for v in versions]

    # ==================== Search & Filter ====================

    def search_models(
        self,
        query: str,
        model_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search models by name or description."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(
                or_(
                    ModelRegistryDB.model_name.contains(query),
                    ModelRegistryDB.description.contains(query),
                )
            )
            if model_type:
                statement = statement.where(ModelRegistryDB.model_type == model_type)
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            statement = statement.limit(limit)
            models = session.exec(statement).all()
            return [self._model_to_dict(model) for model in models]

    def get_models_by_tag(
        self,
        tag: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get models with a specific tag."""
        # Note: JSON array containment query varies by database
        # For MySQL, we use JSON_CONTAINS
        with get_session() as session:
            statement = select(ModelRegistryDB)
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            # Filter in Python for simplicity (works across databases)
            models = session.exec(statement).all()
            result = []
            for model in models:
                if model.tags and tag in model.tags:
                    result.append(self._model_to_dict(model))
            return result

    def get_models_by_category(
        self,
        category: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get models in a specific category."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(ModelRegistryDB.category == category)
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            models = session.exec(statement).all()
            return [self._model_to_dict(model) for model in models]

    # ==================== Model Comparison ====================

    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple models."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(ModelRegistryDB.model_id.in_(model_ids))
            models = session.exec(statement).all()

            comparison = {
                "models": [],
                "metrics_keys": set(),
            }

            for model in models:
                normalized_metrics = self._normalize_metrics(model.metrics) or {}
                model_info = {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "version": model.version,
                    "model_type": model.model_type,
                    "metrics": normalized_metrics,
                }
                comparison["models"].append(model_info)
                if normalized_metrics:
                    comparison["metrics_keys"].update(normalized_metrics.keys())

            comparison["metrics_keys"] = list(comparison["metrics_keys"])
            return comparison

    # ==================== Register from Task ====================

    def register_from_task(
        self,
        task_id: str,
        model_name: str,
        version: str = "v1.0.0",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Register a model from a completed training task."""
        from .training_task_service import training_task_service

        existing = self.get_model_by_source_task(task_id, user_id=user_id)
        if existing:
            logger.info(f"Model already registered for task {task_id}")
            return existing

        task = training_task_service.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return None

        if task['status'] != 'succeeded':
            logger.error(f"Task {task_id} has not succeeded, status: {task['status']}")
            return None

        if not task['final_model_path']:
            logger.error(f"Task {task_id} has no final model path")
            return None

        if user_id and task.get('user_id') and task['user_id'] != user_id:
            logger.warning(f"User {user_id} not authorized to register task {task_id}")
            return None

        training_params = task.get('training_params') or {}
        is_adapter = bool(task.get('is_lora') or training_params.get('use_lora'))

        model = self.register_model(
            model_name=model_name,
            model_path=task['final_model_path'],
            model_type=task['model_type'],
            version=version,
            source_task_id=task_id,
            base_model_path=task['base_model_path'],
            description=description or task['description'],
            tags=tags,
            category=category,
            metrics=task['final_metrics'],
            user_id=user_id or task['user_id'],
            is_adapter=is_adapter,
        )
        if model:
            training_task_service.update_task_trained_model_registry_id(task_id, model['model_id'])
        return model


# Global service instance
model_registry_service = ModelRegistryService()
