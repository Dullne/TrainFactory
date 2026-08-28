"""Transactional guards for persisted runtime model dependencies.

The parsers in this module deliberately inspect only schema-owned reference
fields.  Arbitrary ``config_id`` keys in user JSON must not become lifecycle
dependencies by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sqlalchemy import or_
from sqlmodel import select

from ..entities.deployment_entity import DeploymentDB
from ..entities.deployment_replica_entity import DeploymentReplicaDB
from ..entities.evaluation_task_entity import (
    EvaluationStatus,
    EvaluationTaskDB,
)
from ..entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from ..entities.generation_task_entity import GenerationStatus, GenerationTaskDB
from ..entities.milvus_collection_entity import MilvusCollectionDB
from ..entities.model_config_entity import ModelConfigDB
from ..entities.model_registry_entity import (
    MODEL_DELETE_INTENT_METADATA_KEY,
    ModelRegistryDB,
    ModelVersionDB,
)
from ..entities.training_task_entity import TrainingTaskDB
from ...enums.sync_status import SyncStatus, SyncTrainingStatus
from ...utils.path_utils import (
    artifact_path_uses_root,
    canonicalize_artifact_path,
)


class RuntimeDependencyUnavailableError(ValueError):
    """A referenced runtime dependency cannot accept new work."""


class RuntimeDependencyChangedError(RuntimeDependencyUnavailableError):
    """A dependency graph changed while its ordered locks were acquired."""


class RuntimeDependencyClaimConflictError(RuntimeDependencyUnavailableError):
    """Another lifecycle owner already holds a required deployment claim."""


@dataclass(frozen=True)
class RuntimeDependencyReferences:
    """Stable, deduplicated IDs consumed by one runtime task."""

    config_ids: tuple[str, ...] = ()
    deployment_ids: tuple[str, ...] = ()
    artifact_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class LockedRuntimeDependencies:
    """Rows held by the caller's current transaction."""

    models: tuple[ModelRegistryDB, ...] = ()
    deployments: tuple[DeploymentDB, ...] = ()
    configs: tuple[ModelConfigDB, ...] = ()


@dataclass(frozen=True)
class RuntimeExecutionSnapshot:
    """Live-worker IDs captured before any database row lock is taken."""

    generation_task_ids: frozenset[str] = frozenset()
    evaluation_task_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RuntimeDependencyConsumer:
    """One active task that prevents dependency deletion."""

    kind: str
    task_id: str


@dataclass(frozen=True)
class _LockedModelDeleteDependencyScope:
    """Rows protected by the registry model-delete dependency fence."""

    model: ModelRegistryDB
    deployments: tuple[DeploymentDB, ...]
    configs: tuple[ModelConfigDB, ...]


_GENERATION_ACTIVE_STATUSES = (
    GenerationStatus.PENDING,
    GenerationStatus.RUNNING,
    GenerationStatus.STOPPING,
    GenerationStatus.PUBLISHING,
    GenerationStatus.RECOVERING,
    GenerationStatus.RESTARTING,
)
_EVALUATION_ACTIVE_STATUSES = (
    EvaluationStatus.PENDING,
    EvaluationStatus.RUNNING,
)
_SYNC_DELETING_STATUSES = (
    SyncStatus.DELETING,
    SyncStatus.DELETING_CASCADE,
)
_TRAINING_ACTIVE_STATUSES = ("pending", "preparing", "running", "evaluating")
_SYNC_BINDING_ACTIVE_TRAINING_STATUSES = (
    SyncTrainingStatus.PENDING,
    SyncTrainingStatus.COMPLETED,
    SyncTrainingStatus.ADAPTER_LOADED,
)


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _add_config_id(target: set[str], value: Any) -> None:
    if isinstance(value, Mapping):
        config_id = _identifier(value.get("config_id"))
        if config_id:
            target.add(config_id)


def _references(
    *,
    config_ids: set[str] | None = None,
    deployment_ids: set[str] | None = None,
    artifact_paths: set[str] | None = None,
) -> RuntimeDependencyReferences:
    return RuntimeDependencyReferences(
        config_ids=tuple(sorted(config_ids or ())),
        deployment_ids=tuple(sorted(deployment_ids or ())),
        artifact_paths=tuple(
            sorted(
                {
                    canonical
                    for path in artifact_paths or ()
                    if (
                        canonical := canonicalize_artifact_path(path)
                    ) is not None
                }
            )
        ),
    )


def generation_runtime_dependency_references(
    task: Any,
) -> RuntimeDependencyReferences:
    """Extract model-config references persisted by generation tasks."""
    config_ids: set[str] = set()
    for field_name in (
        "llm_config",
        "eval_llm_config",
        "embedding_config",
        "rerank_config",
    ):
        _add_config_id(config_ids, _value(task, field_name))
    embedding_config_id = _identifier(_value(task, "embedding_config_id"))
    if embedding_config_id:
        config_ids.add(embedding_config_id)
    return _references(config_ids=config_ids)


def evaluation_runtime_dependency_references(
    task: Any,
) -> RuntimeDependencyReferences:
    """Extract MTEB and DeepEval deployment/model-config references."""
    config_ids: set[str] = set()
    deployment_ids: set[str] = set()
    model_configs = _value(task, "model_configs")
    if isinstance(model_configs, list):
        for raw_config in model_configs:
            if not isinstance(raw_config, Mapping):
                continue
            deployment_id = _identifier(raw_config.get("deployment_id"))
            if deployment_id:
                deployment_ids.add(deployment_id)
            _add_config_id(config_ids, raw_config)
            for group_key in ("embedding", "rerank", "llm"):
                group_config = raw_config.get(group_key)
                _add_config_id(config_ids, group_config)
                if isinstance(group_config, Mapping):
                    deployment_id = _identifier(
                        group_config.get("deployment_id")
                    )
                    if deployment_id:
                        deployment_ids.add(deployment_id)

    _add_config_id(config_ids, _value(task, "llm_config"))
    worker_groups = _value(task, "worker_groups")
    if isinstance(worker_groups, Mapping):
        _add_config_id(
            config_ids,
            worker_groups.get("retrieval_embedding_config"),
        )
    return _references(
        config_ids=config_ids,
        deployment_ids=deployment_ids,
    )


def external_sync_runtime_dependency_references(
    task: Any,
) -> RuntimeDependencyReferences:
    """Extract model-config references used by external-sync generation."""
    config_ids: set[str] = set()
    generation_config = _value(task, "generation_config")
    if isinstance(generation_config, Mapping):
        for field_name in (
            "llm_config",
            "eval_llm_config",
            "embedding_config",
            "rerank_config",
        ):
            _add_config_id(config_ids, generation_config.get(field_name))
    return _references(config_ids=config_ids)


def external_sync_writer_dependency_references(
    task: Any,
    training_targets: Any,
) -> RuntimeDependencyReferences:
    """Build the complete dependency union for one sync-task writer.

    Deployment bindings remain lifecycle dependencies even while the sync task
    is inactive.  Generation configs are execution dependencies and therefore
    join the writer lock set only while the prospective task is active.
    """
    config_ids: set[str] = set()
    if bool(_value(task, "is_active", False)):
        config_ids.update(
            external_sync_runtime_dependency_references(task).config_ids
        )

    deployment_ids: set[str] = set()
    base_deployment_id = _identifier(_value(task, "base_deployment_id"))
    if base_deployment_id:
        deployment_ids.add(base_deployment_id)
    if isinstance(training_targets, (list, tuple)):
        for target in training_targets:
            target_deployment_id = _identifier(
                _value(target, "base_deployment_id")
            )
            if target_deployment_id:
                deployment_ids.add(target_deployment_id)

    artifact_paths = set(
        external_sync_training_artifact_paths(task, training_targets)
    )
    return _references(
        config_ids=config_ids,
        deployment_ids=deployment_ids,
        artifact_paths=artifact_paths,
    )


def external_sync_training_artifact_paths(
    task: Any,
    training_targets: Any,
) -> tuple[str, ...]:
    """Return active sync training paths as canonical physical references."""
    if not bool(_value(task, "is_active", False)):
        return ()

    paths: set[str] = set()

    def add_path(value: Any) -> None:
        canonical = canonicalize_artifact_path(
            value if isinstance(value, str) else None
        )
        if canonical is not None:
            paths.add(canonical)

    training_config = _value(task, "training_config")
    if isinstance(training_config, Mapping):
        add_path(training_config.get("base_model_path"))
    if isinstance(training_targets, (list, tuple)):
        for target in training_targets:
            if bool(_value(target, "is_active", True)):
                add_path(_value(target, "base_model_path"))
    return tuple(sorted(paths))


def runtime_dependency_signature(
    task: Any,
    reference_parser: Callable[[Any], RuntimeDependencyReferences],
) -> RuntimeDependencyReferences:
    """Capture the exact runtime IDs that must survive task-row locking."""
    return reference_parser(task)


def require_runtime_dependency_signature(
    task: Any,
    reference_parser: Callable[[Any], RuntimeDependencyReferences],
    expected: RuntimeDependencyReferences,
) -> None:
    """Fail closed when task references drift before its row lock is held."""
    if runtime_dependency_signature(task, reference_parser) != expected:
        raise RuntimeDependencyChangedError(
            "Task runtime references changed while acquiring locks; retry"
        )


def _missing_identifier(
    requested: tuple[str, ...],
    actual: set[str],
) -> str | None:
    return next((item for item in requested if item not in actual), None)


def _require_runtime_dependency_ownership(
    records: list[Any],
    expected_user_id: str | None,
) -> None:
    if expected_user_id is None:
        return
    if any(_value(record, "user_id") != expected_user_id for record in records):
        raise RuntimeDependencyUnavailableError(
            "Runtime dependency is unavailable"
        )


def _delete_intent_token(model: ModelRegistryDB) -> str | None:
    if model.status != "deleting" or not isinstance(model.extra_metadata, dict):
        return None
    intent = model.extra_metadata.get(MODEL_DELETE_INTENT_METADATA_KEY)
    if not isinstance(intent, Mapping):
        return None
    return _identifier(intent.get("token"))


def _training_artifact_paths(task: TrainingTaskDB) -> tuple[str, ...]:
    paths: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())

    add(task.base_model_path)
    if isinstance(task.loss_config, Mapping):
        add(task.loss_config.get("guide_model"))
    params = task.training_params or {}
    if isinstance(params, Mapping):
        add(params.get("base_model_path"))
        nested_loss = params.get("loss_config")
        if isinstance(nested_loss, Mapping):
            add(nested_loss.get("guide_model"))
    return tuple(paths)


def _lock_model_delete_dependency_scope(
    session: Any,
    *,
    model_id: str,
    delete_token: str,
    deployment_claims: Mapping[str, tuple[str, int]],
    execution_snapshot: RuntimeExecutionSnapshot,
    training_execution_task_ids: tuple[str, ...] = (),
) -> _LockedModelDeleteDependencyScope:
    """Fence one registry force-delete without weakening writer admission.

    This private delete-only path accepts exact durable ownership evidence.  It
    never changes the public writer guard: unrelated deleting parents and any
    stale/extra deployment claim remain unavailable.
    """
    normalized_model_id = _identifier(model_id)
    normalized_delete_token = _identifier(delete_token)
    if normalized_model_id is None or normalized_delete_token is None:
        raise RuntimeDependencyUnavailableError(
            "Model deletion ownership was lost"
        )

    preliminary_target_model = session.exec(
        select(ModelRegistryDB).where(
            ModelRegistryDB.model_id == normalized_model_id
        )
    ).first()
    if preliminary_target_model is None:
        raise RuntimeDependencyUnavailableError(
            "Model deletion ownership was lost"
        )
    target_model_path = preliminary_target_model.model_path

    def overlaps_target(candidate_path: Any) -> bool:
        return bool(
            artifact_path_uses_root(candidate_path, target_model_path)
            or artifact_path_uses_root(target_model_path, candidate_path)
        )

    overlap_owner_ids = {
        model.model_id
        for model in session.exec(
            select(ModelRegistryDB).order_by(ModelRegistryDB.model_id)
        ).all()
        if model.model_id != normalized_model_id
        and overlaps_target(model.model_path)
    }
    overlap_owner_ids.update(
        version.model_id
        for version in session.exec(
            select(ModelVersionDB)
            .where(ModelVersionDB.model_id != normalized_model_id)
            .order_by(ModelVersionDB.version_id)
        ).all()
        if overlaps_target(version.model_path)
    )

    preliminary_target_deployments = list(
        session.exec(
            select(DeploymentDB)
            .where(DeploymentDB.model_id == normalized_model_id)
            .order_by(DeploymentDB.deployment_id)
        ).all()
    )
    preliminary_target_ids = tuple(
        deployment.deployment_id
        for deployment in preliminary_target_deployments
    )
    config_condition = ModelConfigDB.registry_id == normalized_model_id
    if preliminary_target_ids:
        config_condition = or_(
            config_condition,
            ModelConfigDB.deployment_id.in_(preliminary_target_ids),
        )
    preliminary_configs = list(
        session.exec(
            select(ModelConfigDB)
            .where(config_condition)
            .order_by(ModelConfigDB.config_id)
        ).all()
    )
    preliminary_config_signatures = {
        config.config_id: (config.registry_id, config.deployment_id)
        for config in preliminary_configs
    }

    referenced_deployment_ids = tuple(
        sorted(
            set(preliminary_target_ids)
            | {
                deployment_id
                for _registry_id, deployment_id
                in preliminary_config_signatures.values()
                if deployment_id
            }
        )
    )
    preliminary_deployments = []
    if referenced_deployment_ids:
        preliminary_deployments = list(
            session.exec(
                select(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id.in_(referenced_deployment_ids)
                )
                .order_by(DeploymentDB.deployment_id)
            ).all()
        )
    preliminary_deployment_signatures = {
        deployment.deployment_id: deployment.model_id
        for deployment in preliminary_deployments
    }
    model_ids = tuple(
        sorted(
            {normalized_model_id}
            | overlap_owner_ids
            | {
                registry_id
                for registry_id, _deployment_id
                in preliminary_config_signatures.values()
                if registry_id
            }
            | set(preliminary_deployment_signatures.values())
        )
    )

    locked_models = list(
        session.exec(
            select(ModelRegistryDB)
            .where(ModelRegistryDB.model_id.in_(model_ids))
            .order_by(ModelRegistryDB.model_id)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).all()
    )
    locked_models_by_id = {
        model.model_id: model for model in locked_models
    }
    target_model = locked_models_by_id.get(normalized_model_id)
    if (
        target_model is None
        or _delete_intent_token(target_model) != normalized_delete_token
    ):
        raise RuntimeDependencyUnavailableError(
            "Model deletion ownership was lost"
        )
    if target_model.model_path != target_model_path:
        raise RuntimeDependencyChangedError(
            "Model artifact path changed during model deletion; retry"
        )
    missing_model_id = _missing_identifier(
        model_ids,
        set(locked_models_by_id),
    )
    if missing_model_id is not None:
        raise RuntimeDependencyUnavailableError(
            "A model-config parent is unavailable during model deletion"
        )
    unrelated_deleting_model = next(
        (
            model
            for model in locked_models
            if model.model_id != normalized_model_id
            and model.status == "deleting"
        ),
        None,
    )
    if unrelated_deleting_model is not None:
        raise RuntimeDependencyUnavailableError(
            "An unrelated model deletion is in progress"
        )

    deployment_condition = DeploymentDB.model_id == normalized_model_id
    if referenced_deployment_ids:
        deployment_condition = or_(
            deployment_condition,
            DeploymentDB.deployment_id.in_(referenced_deployment_ids),
        )
    locked_deployments = list(
        session.exec(
            select(DeploymentDB)
            .where(deployment_condition)
            .order_by(DeploymentDB.deployment_id)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).all()
    )
    locked_deployments_by_id = {
        deployment.deployment_id: deployment
        for deployment in locked_deployments
    }
    for deployment_id, expected_model_id in (
        preliminary_deployment_signatures.items()
    ):
        locked = locked_deployments_by_id.get(deployment_id)
        if locked is None or locked.model_id != expected_model_id:
            raise RuntimeDependencyChangedError(
                "Deployment references changed during model deletion; retry"
            )
    target_deployments = tuple(
        deployment
        for deployment in locked_deployments
        if deployment.model_id == normalized_model_id
    )
    target_deployment_ids = {
        deployment.deployment_id for deployment in target_deployments
    }
    if target_deployment_ids != set(preliminary_target_ids):
        raise RuntimeDependencyChangedError(
            "Deployment membership changed during model deletion; retry"
        )

    normalized_claims: dict[str, tuple[str, int]] = {}
    for deployment_id, owner in deployment_claims.items():
        if (
            _identifier(deployment_id) is None
            or not isinstance(owner, tuple)
            or len(owner) != 2
            or _identifier(owner[0]) is None
            or type(owner[1]) is not int
        ):
            raise RuntimeDependencyUnavailableError(
                "Deployment deletion ownership was lost"
            )
        normalized_claims[deployment_id] = (owner[0], owner[1])

    if normalized_claims:
        if set(normalized_claims) != target_deployment_ids:
            raise RuntimeDependencyUnavailableError(
                "Deployment deletion ownership was lost"
            )
        for deployment in target_deployments:
            expected_token, expected_generation = normalized_claims[
                deployment.deployment_id
            ]
            if not (
                deployment.replica_operation_kind == "delete"
                and deployment.replica_operation_token == expected_token
                and deployment.replica_operation_generation
                == expected_generation
            ):
                raise RuntimeDependencyUnavailableError(
                    "Deployment deletion ownership was lost"
                )
    else:
        for deployment in target_deployments:
            if (
                deployment.replica_operation_token is not None
                or deployment.replica_operation_kind is not None
            ):
                raise RuntimeDependencyClaimConflictError(
                    "Deployment deletion ownership was lost"
                )

    for deployment in locked_deployments:
        if deployment.deployment_id in target_deployment_ids:
            continue
        if (
            deployment.replica_operation_token
            and deployment.replica_operation_kind == "delete"
        ):
            raise RuntimeDependencyUnavailableError(
                "An unrelated deployment deletion is in progress"
            )

    locked_config_condition = (
        ModelConfigDB.registry_id == normalized_model_id
    )
    if target_deployment_ids:
        locked_config_condition = or_(
            locked_config_condition,
            ModelConfigDB.deployment_id.in_(target_deployment_ids),
        )
    locked_configs = list(
        session.exec(
            select(ModelConfigDB)
            .where(locked_config_condition)
            .order_by(ModelConfigDB.config_id)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).all()
    )
    locked_config_signatures = {
        config.config_id: (config.registry_id, config.deployment_id)
        for config in locked_configs
    }
    if locked_config_signatures != preliminary_config_signatures:
        raise RuntimeDependencyChangedError(
            "Model config membership changed during model deletion; retry"
        )

    return _LockedModelDeleteDependencyScope(
        model=target_model,
        deployments=target_deployments,
        configs=tuple(locked_configs),
    )


def _guard_model_delete_runtime_dependencies(
    session: Any,
    scope: _LockedModelDeleteDependencyScope,
    *,
    execution_snapshot: RuntimeExecutionSnapshot,
    training_execution_task_ids: tuple[str, ...] = (),
) -> tuple[DeploymentReplicaDB, ...]:
    """Fence every task/binding consumer after parent dependency locks.

    External-sync task IDs are discovered as one union: active generation
    config consumers, direct deployment bindings, and owners of bound training
    targets.  The union is locked once, in ID order, before its targets and
    active training rows.  This mirrors the sync writer order and prevents a
    bound task lock from being followed by a second, lower-ID sync-task lock.
    """
    references = RuntimeDependencyReferences(
        config_ids=tuple(
            sorted(config.config_id for config in scope.configs)
        ),
        deployment_ids=tuple(
            sorted(
                deployment.deployment_id
                for deployment in scope.deployments
            )
        ),
    )

    target_deployment_ids = tuple(references.deployment_ids)
    preliminary_replicas: list[DeploymentReplicaDB] = []
    if target_deployment_ids:
        preliminary_replicas = list(
            session.exec(
                select(DeploymentReplicaDB)
                .where(
                    DeploymentReplicaDB.deployment_id.in_(
                        target_deployment_ids
                    )
                )
                .order_by(DeploymentReplicaDB.replica_id)
            ).all()
        )
    preliminary_replica_signatures = {
        replica.replica_id: replica.deployment_id
        for replica in preliminary_replicas
    }
    target_replica_ids = tuple(sorted(preliminary_replica_signatures))

    task_binding_conditions = []
    target_binding_conditions = []
    if target_deployment_ids:
        task_binding_conditions.append(
            ExternalSyncTaskDB.base_deployment_id.in_(
                target_deployment_ids
            )
        )
        target_binding_conditions.append(
            ExternalSyncTrainingTargetDB.base_deployment_id.in_(
                target_deployment_ids
            )
        )
    if target_replica_ids:
        task_binding_conditions.append(
            ExternalSyncTaskDB.base_deployment_replica_id.in_(
                target_replica_ids
            )
        )
        target_binding_conditions.append(
            ExternalSyncTrainingTargetDB.base_deployment_replica_id.in_(
                target_replica_ids
            )
        )

    bound_sync_tasks: list[ExternalSyncTaskDB] = []
    bound_sync_targets: list[ExternalSyncTrainingTargetDB] = []
    if task_binding_conditions:
        bound_sync_tasks = list(
            session.exec(
                select(ExternalSyncTaskDB)
                .where(or_(*task_binding_conditions))
                .order_by(ExternalSyncTaskDB.task_id)
            ).all()
        )
    if target_binding_conditions:
        bound_sync_targets = list(
            session.exec(
                select(ExternalSyncTrainingTargetDB)
                .where(or_(*target_binding_conditions))
                .order_by(ExternalSyncTrainingTargetDB.target_id)
            ).all()
        )

    sync_runtime_candidates = list(
        session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.is_active.is_(True),
                ExternalSyncTaskDB.status.notin_(_SYNC_DELETING_STATUSES),
            )
        ).all()
    )
    active_sync_ids = tuple(
        sorted(task.task_id for task in sync_runtime_candidates)
    )
    active_sync_targets: list[ExternalSyncTrainingTargetDB] = []
    if active_sync_ids:
        active_sync_targets = list(
            session.exec(
                select(ExternalSyncTrainingTargetDB)
                .where(
                    ExternalSyncTrainingTargetDB.task_id.in_(
                        active_sync_ids
                    )
                )
                .order_by(ExternalSyncTrainingTargetDB.target_id)
            ).all()
        )
    active_targets_by_task: dict[
        str,
        list[ExternalSyncTrainingTargetDB],
    ] = {task_id: [] for task_id in active_sync_ids}
    for target in active_sync_targets:
        active_targets_by_task.setdefault(target.task_id, []).append(target)
    path_runtime_sync_ids = {
        task.task_id
        for task in sync_runtime_candidates
        if any(
            artifact_path_uses_root(path, scope.model.model_path)
            for path in external_sync_training_artifact_paths(
                task,
                active_targets_by_task.get(task.task_id, ()),
            )
        )
    }
    runtime_sync_ids = {
        task.task_id
        for task in sync_runtime_candidates
        if _references_overlap(
            external_sync_runtime_dependency_references(task),
            references,
        )
    } | path_runtime_sync_ids
    binding_sync_ids = {
        task.task_id for task in bound_sync_tasks
    } | {target.task_id for target in bound_sync_targets}
    sync_ids = tuple(sorted(runtime_sync_ids | binding_sync_ids))
    preliminary_sync_tasks: list[ExternalSyncTaskDB] = []
    preliminary_sync_targets: list[ExternalSyncTrainingTargetDB] = []
    if sync_ids:
        preliminary_sync_tasks = list(
            session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id.in_(sync_ids))
                .order_by(ExternalSyncTaskDB.task_id)
            ).all()
        )
        preliminary_sync_targets = list(
            session.exec(
                select(ExternalSyncTrainingTargetDB)
                .where(ExternalSyncTrainingTargetDB.task_id.in_(sync_ids))
                .order_by(ExternalSyncTrainingTargetDB.target_id)
            ).all()
        )
    preliminary_targets_by_task: dict[
        str,
        list[ExternalSyncTrainingTargetDB],
    ] = {task_id: [] for task_id in sync_ids}
    for target in preliminary_sync_targets:
        preliminary_targets_by_task.setdefault(target.task_id, []).append(
            target
        )
    sync_signatures = {
        task.task_id: (
            task.base_deployment_id,
            task.base_deployment_replica_id,
            bool(task.is_active),
            task.status,
            external_sync_writer_dependency_references(
                task,
                preliminary_targets_by_task.get(task.task_id, ()),
            ),
        )
        for task in preliminary_sync_tasks
    }
    if set(sync_signatures) != set(sync_ids):
        raise RuntimeDependencyChangedError(
            "External sync task membership changed during model deletion; retry"
        )
    sync_target_signatures = {
        target.target_id: (
            target.task_id,
            target.base_deployment_id,
            target.base_deployment_replica_id,
            target.base_model_path,
            bool(target.is_active),
        )
        for target in preliminary_sync_targets
    }

    training_execution_ids = tuple(
        sorted(set(training_execution_task_ids))
    )
    training_conditions = [
        TrainingTaskDB.status.in_(_TRAINING_ACTIVE_STATUSES),
        TrainingTaskDB.process_pid.is_not(None),
        TrainingTaskDB.process_status.is_not(None),
        TrainingTaskDB.process_create_time.is_not(None),
    ]
    if training_execution_ids:
        training_conditions.append(
            TrainingTaskDB.task_id.in_(training_execution_ids)
        )
    preliminary_training_tasks = list(
        session.exec(
            select(TrainingTaskDB)
            .where(or_(*training_conditions))
            .order_by(TrainingTaskDB.task_id)
        ).all()
    )
    training_signatures = {
        task.task_id: _training_artifact_paths(task)
        for task in preliminary_training_tasks
        if any(
            artifact_path_uses_root(path, scope.model.model_path)
            for path in _training_artifact_paths(task)
        )
    }
    training_ids = tuple(sorted(training_signatures))

    generation_condition = GenerationTaskDB.status.in_(
        _GENERATION_ACTIVE_STATUSES
    )
    if execution_snapshot.generation_task_ids:
        generation_condition = or_(
            generation_condition,
            GenerationTaskDB.task_id.in_(
                execution_snapshot.generation_task_ids
            ),
        )
    generation_candidates = list(
        session.exec(
            select(GenerationTaskDB).where(generation_condition)
        ).all()
    )
    generation_ids = tuple(
        sorted(
            task.task_id
            for task in generation_candidates
            if _references_overlap(
                generation_runtime_dependency_references(task),
                references,
            )
        )
    )

    evaluation_condition = EvaluationTaskDB.status.in_(
        _EVALUATION_ACTIVE_STATUSES
    )
    if execution_snapshot.evaluation_task_ids:
        evaluation_condition = or_(
            evaluation_condition,
            EvaluationTaskDB.task_id.in_(
                execution_snapshot.evaluation_task_ids
            ),
        )
    evaluation_candidates = list(
        session.exec(
            select(EvaluationTaskDB).where(evaluation_condition)
        ).all()
    )
    evaluation_ids = tuple(
        sorted(
            task.task_id
            for task in evaluation_candidates
            if _references_overlap(
                evaluation_runtime_dependency_references(task),
                references,
            )
        )
    )

    locked_sync_tasks: list[ExternalSyncTaskDB] = []
    locked_sync_targets: list[ExternalSyncTrainingTargetDB] = []
    if sync_ids:
        locked_sync_tasks = list(
            session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id.in_(sync_ids))
                .order_by(ExternalSyncTaskDB.task_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        locked_sync_targets = list(
            session.exec(
                select(ExternalSyncTrainingTargetDB)
                .where(ExternalSyncTrainingTargetDB.task_id.in_(sync_ids))
                .order_by(ExternalSyncTrainingTargetDB.target_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        current_target_signatures = {
            target.target_id: (
                target.task_id,
                target.base_deployment_id,
                target.base_deployment_replica_id,
                target.base_model_path,
                bool(target.is_active),
            )
            for target in locked_sync_targets
        }
        if current_target_signatures != sync_target_signatures:
            raise RuntimeDependencyChangedError(
                "External sync target bindings changed during model deletion; retry"
            )

        locked_targets_by_task: dict[
            str,
            list[ExternalSyncTrainingTargetDB],
        ] = {task_id: [] for task_id in sync_ids}
        for target in locked_sync_targets:
            locked_targets_by_task.setdefault(target.task_id, []).append(
                target
            )
        current_sync_signatures = {
            task.task_id: (
                task.base_deployment_id,
                task.base_deployment_replica_id,
                bool(task.is_active),
                task.status,
                external_sync_writer_dependency_references(
                    task,
                    locked_targets_by_task.get(task.task_id, ()),
                ),
            )
            for task in locked_sync_tasks
        }
        if current_sync_signatures != sync_signatures:
            raise RuntimeDependencyChangedError(
                "External sync task bindings changed during model deletion; retry"
            )

        # Binding writers use task -> targets -> active trainings.  Lock only
        # the matching training rows instead of the table-wide active set.
        list(
            session.exec(
                select(ExternalSyncTrainingDB)
                .where(
                    ExternalSyncTrainingDB.task_id.in_(sync_ids),
                    ExternalSyncTrainingDB.status.in_(
                        _SYNC_BINDING_ACTIVE_TRAINING_STATUSES
                    ),
                )
                .order_by(ExternalSyncTrainingDB.training_task_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )

    if binding_sync_ids:
        raise RuntimeDependencyUnavailableError(
            "Model deployment is referenced by external sync"
        )
    if path_runtime_sync_ids:
        raise RuntimeDependencyUnavailableError(
            "Model artifact is referenced by active external sync"
        )

    locked_replicas: list[DeploymentReplicaDB] = []
    if target_deployment_ids:
        locked_replicas = list(
            session.exec(
                select(DeploymentReplicaDB)
                .where(
                    DeploymentReplicaDB.deployment_id.in_(
                        target_deployment_ids
                    )
                )
                .order_by(DeploymentReplicaDB.replica_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
    if {
        replica.replica_id: replica.deployment_id
        for replica in locked_replicas
    } != preliminary_replica_signatures:
        raise RuntimeDependencyChangedError(
            "Deployment replica membership changed during model deletion; retry"
        )

    locked_training_tasks: list[TrainingTaskDB] = []
    if training_ids:
        locked_training_tasks = list(
            session.exec(
                select(TrainingTaskDB)
                .where(TrainingTaskDB.task_id.in_(training_ids))
                .order_by(TrainingTaskDB.task_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        if {
            task.task_id: _training_artifact_paths(task)
            for task in locked_training_tasks
        } != training_signatures:
            raise RuntimeDependencyChangedError(
                "Training model references changed during model deletion; retry"
            )

    locked_generations: list[GenerationTaskDB] = []
    if generation_ids:
        locked_generations = list(
            session.exec(
                select(GenerationTaskDB)
                .where(GenerationTaskDB.task_id.in_(generation_ids))
                .order_by(GenerationTaskDB.task_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
    locked_evaluations: list[EvaluationTaskDB] = []
    if evaluation_ids:
        locked_evaluations = list(
            session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id.in_(evaluation_ids))
                .order_by(EvaluationTaskDB.task_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )

    consumers = [
        RuntimeDependencyConsumer("external_sync", task.task_id)
        for task in locked_sync_tasks
        if _external_sync_is_active(task)
        and _references_overlap(
            external_sync_runtime_dependency_references(task),
            references,
        )
    ]
    consumers.extend(
        RuntimeDependencyConsumer("training", task.task_id)
        for task in locked_training_tasks
        if (
            task.status in _TRAINING_ACTIVE_STATUSES
            or task.task_id in training_execution_ids
            or task.process_pid is not None
            or task.process_status is not None
            or task.process_create_time is not None
        )
    )
    consumers.extend(
        RuntimeDependencyConsumer("generation", task.task_id)
        for task in locked_generations
        if _generation_is_active(task, execution_snapshot)
        and _references_overlap(
            generation_runtime_dependency_references(task),
            references,
        )
    )
    consumers.extend(
        RuntimeDependencyConsumer(task.eval_framework, task.task_id)
        for task in locked_evaluations
        if _evaluation_is_active(task, execution_snapshot)
        and _references_overlap(
            evaluation_runtime_dependency_references(task),
            references,
        )
    )
    if any(consumer.kind == "training" for consumer in consumers):
        raise RuntimeDependencyUnavailableError(
            "Model artifact is referenced by an active training task"
        )
    if consumers:
        raise RuntimeDependencyUnavailableError(
            "Model dependency is referenced by an active runtime task"
        )

    if references.config_ids:
        milvus_references = list(
            session.exec(
                select(MilvusCollectionDB)
                .where(
                    MilvusCollectionDB.embedding_config_id.in_(
                        references.config_ids
                    )
                )
                .order_by(MilvusCollectionDB.collection_name)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        if milvus_references:
            raise RuntimeDependencyUnavailableError(
                "Model config is referenced by a Milvus collection"
            )
    return tuple(locked_replicas)


def lock_runtime_dependencies(
    session: Any,
    references: RuntimeDependencyReferences,
    *,
    expected_user_id: str | None = None,
) -> LockedRuntimeDependencies:
    """Lock one dependency graph in model -> deployment -> config order.

    Config and deployment rows are first read without locks only to discover
    their parent IDs.  Every discovered relationship is revalidated after the
    ordered locks are held.  A drift therefore fails closed instead of taking
    a late parent lock in the opposite order.

    When ``expected_user_id`` is provided, every locked dependency must belong
    to that owner.
    """
    config_ids = tuple(sorted(set(references.config_ids)))
    direct_deployment_ids = tuple(sorted(set(references.deployment_ids)))
    artifact_paths = tuple(sorted(set(references.artifact_paths)))

    preliminary_configs: list[ModelConfigDB] = []
    if config_ids:
        preliminary_configs = list(
            session.exec(
                select(ModelConfigDB).where(ModelConfigDB.config_id.in_(config_ids))
            ).all()
        )
        missing = _missing_identifier(
            config_ids,
            {config.config_id for config in preliminary_configs},
        )
        if missing:
            raise RuntimeDependencyUnavailableError(
                f"Model config not found: {missing}"
            )
        _require_runtime_dependency_ownership(
            preliminary_configs,
            expected_user_id,
        )

    config_signatures = {
        config.config_id: (config.registry_id, config.deployment_id)
        for config in preliminary_configs
    }
    deployment_ids = tuple(
        sorted(
            set(direct_deployment_ids)
            | {
                deployment_id
                for _registry_id, deployment_id in config_signatures.values()
                if deployment_id
            }
        )
    )

    preliminary_deployments: list[DeploymentDB] = []
    if deployment_ids:
        preliminary_deployments = list(
            session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id.in_(deployment_ids)
                )
            ).all()
        )
        missing = _missing_identifier(
            deployment_ids,
            {
                deployment.deployment_id
                for deployment in preliminary_deployments
            },
        )
        if missing:
            raise RuntimeDependencyUnavailableError(
                f"Deployment not found: {missing}"
            )
        _require_runtime_dependency_ownership(
            preliminary_deployments,
            expected_user_id,
        )

    deployment_signatures = {
        deployment.deployment_id: deployment.model_id
        for deployment in preliminary_deployments
    }
    artifact_model_signatures: dict[str, str] = {}
    if artifact_paths:
        artifact_model_signatures = {
            model.model_id: model.model_path
            for model in session.exec(
                select(ModelRegistryDB).order_by(ModelRegistryDB.model_id)
            ).all()
            if any(
                artifact_path_uses_root(path, model.model_path)
                for path in artifact_paths
            )
        }
    required_model_ids = tuple(
        sorted(
            {
                registry_id
                for registry_id, _deployment_id in config_signatures.values()
                if registry_id
            }
        )
    )
    model_ids = tuple(
        sorted(
            set(required_model_ids)
            | set(deployment_signatures.values())
            | set(artifact_model_signatures)
        )
    )

    locked_models: list[ModelRegistryDB] = []
    if model_ids:
        locked_models = list(
            session.exec(
                select(ModelRegistryDB)
                .where(ModelRegistryDB.model_id.in_(model_ids))
                .order_by(ModelRegistryDB.model_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        missing = _missing_identifier(
            required_model_ids,
            {model.model_id for model in locked_models},
        )
        if missing:
            raise RuntimeDependencyUnavailableError(f"Model not found: {missing}")
        _require_runtime_dependency_ownership(
            locked_models,
            expected_user_id,
        )
        if any(
            artifact_model_signatures.get(model.model_id) != model.model_path
            or not any(
                artifact_path_uses_root(path, model.model_path)
                for path in artifact_paths
            )
            for model in locked_models
            if model.model_id in artifact_model_signatures
        ) or set(artifact_model_signatures) - {
            model.model_id for model in locked_models
        }:
            raise RuntimeDependencyChangedError(
                "Model artifact membership changed while acquiring locks; retry"
            )
        deleting_model = next(
            (model for model in locked_models if model.status == "deleting"),
            None,
        )
        if deleting_model is not None:
            raise RuntimeDependencyUnavailableError(
                f"Model is being deleted: {deleting_model.model_id}"
            )

    locked_deployments: list[DeploymentDB] = []
    if deployment_ids:
        locked_deployments = list(
            session.exec(
                select(DeploymentDB)
                .where(DeploymentDB.deployment_id.in_(deployment_ids))
                .order_by(DeploymentDB.deployment_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        missing = _missing_identifier(
            deployment_ids,
            {deployment.deployment_id for deployment in locked_deployments},
        )
        if missing:
            raise RuntimeDependencyUnavailableError(
                f"Deployment not found: {missing}"
            )
        _require_runtime_dependency_ownership(
            locked_deployments,
            expected_user_id,
        )
        for deployment in locked_deployments:
            if (
                deployment_signatures.get(deployment.deployment_id)
                != deployment.model_id
            ):
                raise RuntimeDependencyChangedError(
                    "Deployment model reference changed while acquiring locks; retry"
                )
            if (
                deployment.replica_operation_token
                and deployment.replica_operation_kind == "delete"
            ):
                raise RuntimeDependencyUnavailableError(
                    "Deployment deletion is in progress: "
                    f"{deployment.deployment_id}"
                )

    locked_configs: list[ModelConfigDB] = []
    if config_ids:
        locked_configs = list(
            session.exec(
                select(ModelConfigDB)
                .where(ModelConfigDB.config_id.in_(config_ids))
                .order_by(ModelConfigDB.config_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        missing = _missing_identifier(
            config_ids,
            {config.config_id for config in locked_configs},
        )
        if missing:
            raise RuntimeDependencyUnavailableError(
                f"Model config not found: {missing}"
            )
        _require_runtime_dependency_ownership(
            locked_configs,
            expected_user_id,
        )
        for config in locked_configs:
            if config_signatures.get(config.config_id) != (
                config.registry_id,
                config.deployment_id,
            ):
                raise RuntimeDependencyChangedError(
                    "Model config references changed while acquiring locks; retry"
                )

    return LockedRuntimeDependencies(
        models=tuple(locked_models),
        deployments=tuple(locked_deployments),
        configs=tuple(locked_configs),
    )


def lock_embedding_model_config_for_binding(
    session: Any,
    config_id: str,
    *,
    expected_user_id: str | None,
) -> ModelConfigDB:
    """Lock and validate one config before persisting a Milvus binding."""
    normalized_id = _identifier(config_id)
    if normalized_id is None:
        raise RuntimeDependencyUnavailableError(
            "Embedding config is unavailable"
        )
    locked = lock_runtime_dependencies(
        session,
        RuntimeDependencyReferences(config_ids=(normalized_id,)),
        expected_user_id=expected_user_id,
    )
    config = next(
        (item for item in locked.configs if item.config_id == normalized_id),
        None,
    )
    if (
        config is None
        or str(config.status).lower() == "deleting"
        or config.model_type != "embedding"
        or (
            expected_user_id is not None
            and config.user_id != expected_user_id
        )
    ):
        raise RuntimeDependencyUnavailableError(
            "Embedding config is unavailable"
        )
    return config


def lock_runtime_task_for_transition(
    session: Any,
    *,
    entity: Any,
    conditions: tuple[Any, ...],
    reference_parser: Callable[[Any], RuntimeDependencyReferences],
    dependency_reference_builder: (
        Callable[[Any], RuntimeDependencyReferences] | None
    ) = None,
) -> Any | None:
    """Lock dependencies before the task row and recheck its reference IDs."""
    candidate = session.exec(
        select(entity).where(*conditions)
    ).first()
    if candidate is None:
        return None
    expected_signature = runtime_dependency_signature(
        candidate,
        reference_parser,
    )
    references = (
        dependency_reference_builder(candidate)
        if dependency_reference_builder is not None
        else expected_signature
    )
    lock_runtime_dependencies(session, references)
    return lock_runtime_task_after_dependencies(
        session,
        entity=entity,
        conditions=conditions,
        reference_parser=reference_parser,
        expected_signature=expected_signature,
    )


def lock_runtime_task_after_dependencies(
    session: Any,
    *,
    entity: Any,
    conditions: tuple[Any, ...],
    reference_parser: Callable[[Any], RuntimeDependencyReferences],
    expected_signature: RuntimeDependencyReferences,
) -> Any | None:
    """Lock a task after its complete dependency union is already locked."""
    task = session.exec(
        select(entity)
        .where(*conditions)
        .execution_options(populate_existing=True)
        .with_for_update()
    ).first()
    if task is None:
        return None
    require_runtime_dependency_signature(
        task,
        reference_parser,
        expected_signature,
    )
    return task


def snapshot_runtime_executions() -> RuntimeExecutionSnapshot:
    """Read the in-process admission registry without holding database locks."""
    from .background_task_admission_service import (
        background_task_admission_service,
    )

    return RuntimeExecutionSnapshot(
        generation_task_ids=frozenset(
            background_task_admission_service.get_executing_task_ids(
                "generation"
            )
        ),
        evaluation_task_ids=frozenset(
            background_task_admission_service.get_executing_task_ids(
                "evaluation"
            )
        ),
    )


def _references_overlap(
    left: RuntimeDependencyReferences,
    right: RuntimeDependencyReferences,
) -> bool:
    return bool(
        set(left.config_ids) & set(right.config_ids)
        or set(left.deployment_ids) & set(right.deployment_ids)
    )


def _generation_is_active(
    task: GenerationTaskDB,
    execution_snapshot: RuntimeExecutionSnapshot,
) -> bool:
    return bool(
        task.status in _GENERATION_ACTIVE_STATUSES
        or task.task_id in execution_snapshot.generation_task_ids
    )


def _evaluation_is_active(
    task: EvaluationTaskDB,
    execution_snapshot: RuntimeExecutionSnapshot,
) -> bool:
    return bool(
        task.status in _EVALUATION_ACTIVE_STATUSES
        or task.task_id in execution_snapshot.evaluation_task_ids
    )


def _external_sync_is_active(task: ExternalSyncTaskDB) -> bool:
    return bool(
        task.is_active and task.status not in _SYNC_DELETING_STATUSES
    )


def lock_active_runtime_dependency_consumers(
    session: Any,
    references: RuntimeDependencyReferences,
    *,
    execution_snapshot: RuntimeExecutionSnapshot,
) -> tuple[RuntimeDependencyConsumer, ...]:
    """Lock active task rows that consume already-locked dependencies.

    The caller must capture ``execution_snapshot`` before taking any database
    lock and must lock model/deployment/config rows before calling this helper.
    Compliant task writers take those same dependency locks before changing a
    task reference or moving an inactive task back into an active state.
    """
    if not references.config_ids and not references.deployment_ids:
        return ()

    generation_condition = GenerationTaskDB.status.in_(
        _GENERATION_ACTIVE_STATUSES
    )
    if execution_snapshot.generation_task_ids:
        generation_condition = or_(
            generation_condition,
            GenerationTaskDB.task_id.in_(
                execution_snapshot.generation_task_ids
            ),
        )
    generation_candidates = list(
        session.exec(
            select(GenerationTaskDB).where(generation_condition)
        ).all()
    )
    generation_ids = sorted(
        task.task_id
        for task in generation_candidates
        if _references_overlap(
            generation_runtime_dependency_references(task),
            references,
        )
    )

    evaluation_condition = EvaluationTaskDB.status.in_(
        _EVALUATION_ACTIVE_STATUSES
    )
    if execution_snapshot.evaluation_task_ids:
        evaluation_condition = or_(
            evaluation_condition,
            EvaluationTaskDB.task_id.in_(
                execution_snapshot.evaluation_task_ids
            ),
        )
    evaluation_candidates = list(
        session.exec(
            select(EvaluationTaskDB).where(evaluation_condition)
        ).all()
    )
    evaluation_ids = sorted(
        task.task_id
        for task in evaluation_candidates
        if _references_overlap(
            evaluation_runtime_dependency_references(task),
            references,
        )
    )

    sync_candidates = list(
        session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.is_active.is_(True),
                ExternalSyncTaskDB.status.notin_(_SYNC_DELETING_STATUSES),
            )
        ).all()
    )
    sync_ids = sorted(
        task.task_id
        for task in sync_candidates
        if _references_overlap(
            external_sync_runtime_dependency_references(task),
            references,
        )
    )

    locked_generations: list[GenerationTaskDB] = []
    if generation_ids:
        locked_generations = list(
            session.exec(
                select(GenerationTaskDB)
                .where(GenerationTaskDB.task_id.in_(generation_ids))
                .order_by(GenerationTaskDB.task_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
    locked_evaluations: list[EvaluationTaskDB] = []
    if evaluation_ids:
        locked_evaluations = list(
            session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id.in_(evaluation_ids))
                .order_by(EvaluationTaskDB.task_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
    locked_sync_tasks: list[ExternalSyncTaskDB] = []
    if sync_ids:
        locked_sync_tasks = list(
            session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id.in_(sync_ids))
                .order_by(ExternalSyncTaskDB.task_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )

    consumers = [
        RuntimeDependencyConsumer("generation", task.task_id)
        for task in locked_generations
        if _generation_is_active(task, execution_snapshot)
        and _references_overlap(
            generation_runtime_dependency_references(task),
            references,
        )
    ]
    consumers.extend(
        RuntimeDependencyConsumer(task.eval_framework, task.task_id)
        for task in locked_evaluations
        if _evaluation_is_active(task, execution_snapshot)
        and _references_overlap(
            evaluation_runtime_dependency_references(task),
            references,
        )
    )
    consumers.extend(
        RuntimeDependencyConsumer("external_sync", task.task_id)
        for task in locked_sync_tasks
        if _external_sync_is_active(task)
        and _references_overlap(
            external_sync_runtime_dependency_references(task),
            references,
        )
    )
    return tuple(sorted(consumers, key=lambda item: (item.kind, item.task_id)))
