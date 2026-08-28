import asyncio
import importlib
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.evaluation_task_entity import (
    EvaluationFramework,
    EvaluationStatus,
    EvaluationTaskDB,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.entities.external_api_config_entity import (
    ExternalApiConfigDB,
)
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.enums.sync_status import SyncStatus


def _runtime_module():
    return importlib.import_module(
        "train_factory.storage.services.runtime_dependency_service"
    )


def test_generation_reference_parser_collects_only_persisted_model_config_ids():
    runtime = _runtime_module()

    references = runtime.generation_runtime_dependency_references(
        {
            "llm_config": {"config_id": "cfg-llm"},
            "eval_llm_config": {"config_id": "cfg-eval"},
            "embedding_config": {"config_id": "cfg-embedding"},
            "rerank_config": {"config_id": "cfg-rerank"},
            "embedding_config_id": "cfg-embedding",
            "steps_config": {"config_id": "not-a-runtime-reference"},
        }
    )

    assert references.config_ids == (
        "cfg-embedding",
        "cfg-eval",
        "cfg-llm",
        "cfg-rerank",
    )
    assert references.deployment_ids == ()


def test_mteb_reference_parser_collects_direct_deployment_ids():
    runtime = _runtime_module()

    references = runtime.evaluation_runtime_dependency_references(
        {
            "eval_framework": "mteb",
            "model_configs": [
                {"deployment_id": "dep-b", "deployment_replica_id": "replica-2"},
                {"deployment_id": "dep-a"},
                {"endpoint": "https://example.invalid/v1"},
            ],
        }
    )

    assert references.config_ids == ()
    assert references.deployment_ids == ("dep-a", "dep-b")


def test_deep_evaluation_reference_parser_collects_group_and_retrieval_configs():
    runtime = _runtime_module()

    references = runtime.evaluation_runtime_dependency_references(
        {
            "eval_framework": "deepeval",
            "model_configs": [
                {
                    "group_name": "group-a",
                    "embedding": {"config_id": "cfg-embedding"},
                    "rerank": {"config_id": "cfg-rerank"},
                    "llm": {"config_id": "cfg-llm"},
                }
            ],
            "worker_groups": {
                "retrieval_embedding_config": {"config_id": "cfg-retrieval"},
                "unrelated": {"config_id": "not-a-runtime-reference"},
            },
            "llm_config": {"config_id": "cfg-legacy-llm"},
        }
    )

    assert references.config_ids == (
        "cfg-embedding",
        "cfg-legacy-llm",
        "cfg-llm",
        "cfg-rerank",
        "cfg-retrieval",
    )
    assert references.deployment_ids == ()


def test_external_sync_reference_parser_collects_generation_config_ids_only():
    runtime = _runtime_module()

    references = runtime.external_sync_runtime_dependency_references(
        {
            "generation_config": {
                "llm_config": {"config_id": "cfg-llm"},
                "embedding_config": {"config_id": "cfg-embedding"},
                "rerank_config": {"config_id": "cfg-rerank"},
                "worker_config": {"config_id": "not-a-runtime-reference"},
            },
            "base_deployment_id": "existing-binding-is-guarded-elsewhere",
        }
    )

    assert references.config_ids == (
        "cfg-embedding",
        "cfg-llm",
        "cfg-rerank",
    )
    assert references.deployment_ids == ()


def test_external_sync_writer_references_union_active_configs_and_bindings():
    runtime = _runtime_module()

    references = runtime.external_sync_writer_dependency_references(
        {
            "is_active": True,
            "generation_config": {
                "embedding_config": {"config_id": "config-1"}
            },
            "base_deployment_id": "deployment-task",
        },
        [
            {"base_deployment_id": "deployment-target-b"},
            {"base_deployment_id": "deployment-target-a"},
        ],
    )

    assert references.config_ids == ("config-1",)
    assert references.deployment_ids == (
        "deployment-target-a",
        "deployment-target-b",
        "deployment-task",
    )


def test_external_sync_inactive_writer_references_still_lock_bindings():
    runtime = _runtime_module()

    references = runtime.external_sync_writer_dependency_references(
        {
            "is_active": False,
            "generation_config": {
                "embedding_config": {"config_id": "stale-config"}
            },
            "base_deployment_id": "deployment-task",
        },
        [{"base_deployment_id": "deployment-target"}],
    )

    assert references.config_ids == ()
    assert references.deployment_ids == (
        "deployment-target",
        "deployment-task",
    )


@pytest.fixture
def runtime_dependency_database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'runtime-dependencies.db'}")
    SQLModel.metadata.create_all(
        engine,
        tables=[
            ModelRegistryDB.__table__,
            DeploymentDB.__table__,
            ModelConfigDB.__table__,
        ],
    )
    return engine


def _add_runtime_dependency_graph(
    engine,
    *,
    model_status="available",
    deployment_delete_claim=False,
    user_id=None,
):
    with Session(engine) as session:
        session.add(
            ModelRegistryDB(
                model_id="model-1",
                model_name="model-1",
                model_type="embedding",
                model_path="/models/model-1",
                status=model_status,
                user_id=user_id,
            )
        )
        session.add(
            DeploymentDB(
                deployment_id="deployment-1",
                model_id="model-1",
                xinference_endpoint="http://inference:8000",
                replica_operation_token=(
                    "delete-token" if deployment_delete_claim else None
                ),
                replica_operation_kind=(
                    "delete" if deployment_delete_claim else None
                ),
                user_id=user_id,
            )
        )
        session.add(
            ModelConfigDB(
                config_id="config-1",
                config_name="config-1",
                source_type="local_deployed",
                registry_id="model-1",
                deployment_id="deployment-1",
                model_type="embedding",
                provider="openai-compatible",
                api_endpoint="http://inference:8000/v1",
                model_name="model-1",
                user_id=user_id,
            )
        )
        session.commit()


def _add_owned_runtime_dependency_graph(
    engine,
    *,
    model_user_id="user-1",
    deployment_user_id="user-1",
    config_user_id="user-1",
):
    with Session(engine) as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id="owned-model",
                    model_name="owned-model",
                    model_type="embedding",
                    model_path="/models/owned-model",
                    status="available",
                    user_id=model_user_id,
                ),
                DeploymentDB(
                    deployment_id="owned-deployment",
                    model_id="owned-model",
                    xinference_endpoint="http://inference:8000",
                    user_id=deployment_user_id,
                ),
                ModelConfigDB(
                    config_id="owned-config",
                    config_name="owned-config",
                    source_type="local_deployed",
                    registry_id="owned-model",
                    deployment_id="owned-deployment",
                    model_type="embedding",
                    provider="openai-compatible",
                    api_endpoint="http://inference:8000/v1",
                    model_name="owned-model",
                    user_id=config_user_id,
                ),
            ]
        )
        session.commit()


def _add_external_cross_binding_graph(engine):
    with Session(engine) as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id="model-1",
                    model_name="model-1",
                    model_type="embedding",
                    model_path="/models/model-1",
                    status="available",
                    user_id="user-1",
                ),
                ModelRegistryDB(
                    model_id="model-2",
                    model_name="model-2",
                    model_type="embedding",
                    model_path="/models/model-2",
                    status="available",
                    user_id="user-1",
                ),
                DeploymentDB(
                    deployment_id="deployment-1",
                    model_id="model-1",
                    xinference_endpoint="http://inference:8001",
                    user_id="user-1",
                ),
                DeploymentDB(
                    deployment_id="deployment-2",
                    model_id="model-2",
                    xinference_endpoint="http://inference:8002",
                    user_id="user-1",
                ),
                ModelConfigDB(
                    config_id="config-1",
                    config_name="config-1",
                    source_type="local_deployed",
                    registry_id="model-1",
                    deployment_id="deployment-1",
                    model_type="embedding",
                    provider="openai-compatible",
                    api_endpoint="http://inference:8001/v1",
                    model_name="model-1",
                    user_id="user-1",
                ),
            ]
        )
        session.commit()


def _add_external_api_config(engine, config_id, *, user_id="user-1"):
    with Session(engine) as session:
        session.add(
            ExternalApiConfigDB(
                config_id=config_id,
                config_name=config_id,
                user_id=user_id,
                api_url="https://example.invalid/api",
                auth_config={},
            )
        )
        session.commit()


def _record_runtime_dependency_locks(monkeypatch):
    runtime = _runtime_module()
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    original = runtime.lock_runtime_dependencies
    recorded = []

    def recording_lock(session, references, *, expected_user_id=None):
        recorded.append(references)
        return original(
            session,
            references,
            expected_user_id=expected_user_id,
        )

    monkeypatch.setattr(runtime, "lock_runtime_dependencies", recording_lock)
    monkeypatch.setattr(sync_module, "lock_runtime_dependencies", recording_lock)
    return recorded


def test_writer_guard_rejects_missing_model_config(runtime_dependency_database):
    runtime = _runtime_module()

    with Session(runtime_dependency_database) as session:
        with pytest.raises(
            runtime.RuntimeDependencyUnavailableError,
            match="Model config not found: missing-config",
        ):
            runtime.lock_runtime_dependencies(
                session,
                runtime.RuntimeDependencyReferences(
                    config_ids=("missing-config",),
                ),
            )


@pytest.mark.parametrize(
    ("foreign_resource", "references"),
    [
        (
            "model_config",
            {"config_ids": ("owned-config",)},
        ),
        (
            "deployment",
            {"deployment_ids": ("owned-deployment",)},
        ),
        (
            "model_artifact",
            {"artifact_paths": ("/models/owned-model/child",)},
        ),
    ],
)
def test_writer_guard_rejects_foreign_owned_dependency_without_disclosing_record(
    runtime_dependency_database,
    foreign_resource,
    references,
):
    runtime = _runtime_module()
    owner_overrides = {
        "model_config": {"config_user_id": "user-2"},
        "deployment": {"deployment_user_id": "user-2"},
        "model_artifact": {"model_user_id": "user-2"},
    }
    _add_owned_runtime_dependency_graph(
        runtime_dependency_database,
        **owner_overrides[foreign_resource],
    )

    with Session(runtime_dependency_database) as session:
        with pytest.raises(
            runtime.RuntimeDependencyUnavailableError,
            match="^Runtime dependency is unavailable$",
        ):
            runtime.lock_runtime_dependencies(
                session,
                runtime.RuntimeDependencyReferences(**references),
                expected_user_id="user-1",
            )


def test_writer_guard_accepts_same_tenant_config_deployment_and_artifact(
    runtime_dependency_database,
):
    runtime = _runtime_module()
    _add_owned_runtime_dependency_graph(runtime_dependency_database)

    with Session(runtime_dependency_database) as session:
        locked = runtime.lock_runtime_dependencies(
            session,
            runtime.RuntimeDependencyReferences(
                config_ids=("owned-config",),
                deployment_ids=("owned-deployment",),
                artifact_paths=("/models/owned-model/child",),
            ),
            expected_user_id="user-1",
        )

    assert [item.model_id for item in locked.models] == ["owned-model"]
    assert [item.deployment_id for item in locked.deployments] == [
        "owned-deployment"
    ]
    assert [item.config_id for item in locked.configs] == ["owned-config"]


@pytest.mark.parametrize("owner_id", [None, "user-2"])
@pytest.mark.parametrize("resource_kind", ["model_config", "deployment"])
def test_writer_guard_rejects_unowned_preliminary_row_before_parent_lookup(
    runtime_dependency_database,
    owner_id,
    resource_kind,
):
    runtime = _runtime_module()
    with Session(runtime_dependency_database) as session:
        if resource_kind == "model_config":
            session.add(
                ModelConfigDB(
                    config_id="foreign-config",
                    config_name="foreign-config",
                    source_type="local_deployed",
                    registry_id="private-parent-model",
                    deployment_id="private-parent-deployment",
                    model_type="embedding",
                    provider="openai-compatible",
                    api_endpoint="http://inference:8000/v1",
                    model_name="foreign-model",
                    user_id=owner_id,
                )
            )
            references = runtime.RuntimeDependencyReferences(
                config_ids=("foreign-config",),
            )
        else:
            session.add(
                DeploymentDB(
                    deployment_id="foreign-deployment",
                    model_id="private-parent-model",
                    xinference_endpoint="http://inference:8000",
                    user_id=owner_id,
                )
            )
            references = runtime.RuntimeDependencyReferences(
                deployment_ids=("foreign-deployment",),
            )
        session.commit()

    selected_tables = []

    def record_select(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        normalized = statement.lower().lstrip()
        if not normalized.startswith("select"):
            return
        for table_name in ("model_configs", "deployments", "model_registry"):
            if f"from {table_name}" in normalized:
                selected_tables.append(table_name)
                return

    event.listen(
        runtime_dependency_database,
        "before_cursor_execute",
        record_select,
    )
    try:
        with Session(runtime_dependency_database) as session:
            with pytest.raises(
                runtime.RuntimeDependencyUnavailableError,
                match="^Runtime dependency is unavailable$",
            ):
                runtime.lock_runtime_dependencies(
                    session,
                    references,
                    expected_user_id="user-1",
                )
    finally:
        event.remove(
            runtime_dependency_database,
            "before_cursor_execute",
            record_select,
        )

    assert selected_tables == [
        "model_configs" if resource_kind == "model_config" else "deployments"
    ]


@pytest.mark.parametrize("owner_id", [None, "user-2"])
def test_writer_guard_keeps_auth_disabled_owner_compatibility(
    runtime_dependency_database,
    owner_id,
):
    runtime = _runtime_module()
    _add_owned_runtime_dependency_graph(
        runtime_dependency_database,
        model_user_id=owner_id,
        deployment_user_id=owner_id,
        config_user_id=owner_id,
    )

    with Session(runtime_dependency_database) as session:
        locked = runtime.lock_runtime_dependencies(
            session,
            runtime.RuntimeDependencyReferences(config_ids=("owned-config",)),
            expected_user_id=None,
        )

    assert [item.config_id for item in locked.configs] == ["owned-config"]


def test_writer_guard_rejects_model_delete_intent(runtime_dependency_database):
    runtime = _runtime_module()
    _add_runtime_dependency_graph(
        runtime_dependency_database,
        model_status="deleting",
    )

    with Session(runtime_dependency_database) as session:
        with pytest.raises(
            runtime.RuntimeDependencyUnavailableError,
            match="Model is being deleted: model-1",
        ):
            runtime.lock_runtime_dependencies(
                session,
                runtime.RuntimeDependencyReferences(config_ids=("config-1",)),
            )


def test_writer_guard_rejects_deployment_delete_claim(
    runtime_dependency_database,
):
    runtime = _runtime_module()
    _add_runtime_dependency_graph(
        runtime_dependency_database,
        deployment_delete_claim=True,
    )

    with Session(runtime_dependency_database) as session:
        with pytest.raises(
            runtime.RuntimeDependencyUnavailableError,
            match="Deployment deletion is in progress: deployment-1",
        ):
            runtime.lock_runtime_dependencies(
                session,
                runtime.RuntimeDependencyReferences(
                    deployment_ids=("deployment-1",),
                ),
            )


def test_writer_guard_locks_model_then_deployment_then_config():
    runtime = _runtime_module()
    model = SimpleNamespace(model_id="model-1", status="available")
    deployment = SimpleNamespace(
        deployment_id="deployment-1",
        model_id="model-1",
        replica_operation_token=None,
        replica_operation_kind=None,
    )
    config = SimpleNamespace(
        config_id="config-1",
        registry_id="model-1",
        deployment_id="deployment-1",
    )

    class _Result:
        def __init__(self, values):
            self._values = values

        def all(self):
            return list(self._values)

    class _RecordingSession:
        def __init__(self):
            self.locked_tables = []
            self.lock_options = []

        def exec(self, statement):
            table_name = statement.get_final_froms()[0].name
            is_lock = statement._for_update_arg is not None
            if is_lock:
                self.locked_tables.append(table_name)
                self.lock_options.append(
                    statement.get_execution_options().get("populate_existing")
                )
            values = {
                "model_registry": [model],
                "deployments": [deployment],
                "model_configs": [config],
            }[table_name]
            return _Result(values)

    session = _RecordingSession()
    runtime.lock_runtime_dependencies(
        session,
        runtime.RuntimeDependencyReferences(config_ids=("config-1",)),
    )

    assert session.locked_tables == [
        "model_registry",
        "deployments",
        "model_configs",
    ]
    assert session.lock_options == [True, True, True]


def test_runtime_reference_signature_detects_task_reference_drift():
    runtime = _runtime_module()
    task = GenerationTaskDB(
        task_id="generation-signature",
        task_name="generation-signature",
        input_path="/managed/input.jsonl",
        llm_config={"config_id": "config-before"},
        steps_config={},
    )
    expected = runtime.runtime_dependency_signature(
        task,
        runtime.generation_runtime_dependency_references,
    )
    task.llm_config = {"config_id": "config-after"}

    with pytest.raises(
        runtime.RuntimeDependencyChangedError,
        match="Task runtime references changed while acquiring locks; retry",
    ):
        runtime.require_runtime_dependency_signature(
            task,
            runtime.generation_runtime_dependency_references,
            expected,
        )


def test_task_transition_rechecks_signature_after_task_row_lock():
    runtime = _runtime_module()
    before = GenerationTaskDB(
        task_id="generation-signature-lock",
        task_name="generation-signature-lock",
        input_path="/managed/input.jsonl",
        llm_config={"config_id": "config-before"},
        steps_config={},
    )
    after = GenerationTaskDB(
        task_id="generation-signature-lock",
        task_name="generation-signature-lock",
        input_path="/managed/input.jsonl",
        llm_config={"config_id": "config-after"},
        steps_config={},
    )

    class _Result:
        def __init__(self, value):
            self._value = value

        def first(self):
            return self._value

    class _ChangingTaskSession:
        def exec(self, statement):
            if statement._for_update_arg is None:
                return _Result(before)
            assert (
                statement.get_execution_options().get("populate_existing")
                is True
            )
            return _Result(after)

    with pytest.raises(
        runtime.RuntimeDependencyChangedError,
        match="Task runtime references changed while acquiring locks; retry",
    ):
        runtime.lock_runtime_task_for_transition(
            _ChangingTaskSession(),
            entity=GenerationTaskDB,
            conditions=(
                GenerationTaskDB.task_id == "generation-signature-lock",
            ),
            reference_parser=runtime.generation_runtime_dependency_references,
            dependency_reference_builder=(
                lambda _task: runtime.RuntimeDependencyReferences()
            ),
        )


@pytest.mark.parametrize(
    ("status", "expected_active"),
    (
        (GenerationStatus.PENDING, True),
        (GenerationStatus.RUNNING, True),
        (GenerationStatus.STOPPING, True),
        (GenerationStatus.PUBLISHING, True),
        (GenerationStatus.RECOVERING, True),
        (GenerationStatus.RESTARTING, True),
        (GenerationStatus.COMPLETED, False),
        (GenerationStatus.FAILED, False),
        (GenerationStatus.STOPPED, False),
        (GenerationStatus.DELETING, False),
        (GenerationStatus.DELETING_CASCADE, False),
    ),
)
def test_generation_active_status_matrix_uses_entity_statuses(
    status,
    expected_active,
):
    runtime = _runtime_module()

    assert runtime._generation_is_active(
        SimpleNamespace(task_id="generation-status", status=status),
        runtime.RuntimeExecutionSnapshot(),
    ) is expected_active


@pytest.mark.parametrize(
    ("status", "expected_active"),
    (
        (EvaluationStatus.PENDING, True),
        (EvaluationStatus.RUNNING, True),
        (EvaluationStatus.SUCCEEDED, False),
        (EvaluationStatus.COMPLETED, False),
        (EvaluationStatus.FAILED, False),
        (EvaluationStatus.CANCELLED, False),
    ),
)
def test_evaluation_active_status_matrix_uses_entity_statuses(
    status,
    expected_active,
):
    runtime = _runtime_module()

    assert runtime._evaluation_is_active(
        SimpleNamespace(task_id="evaluation-status", status=status),
        runtime.RuntimeExecutionSnapshot(),
    ) is expected_active


@pytest.mark.parametrize(
    ("status", "is_active", "expected_active"),
    tuple(
        (status, True, status not in {SyncStatus.DELETING, SyncStatus.DELETING_CASCADE})
        for status in (
            SyncStatus.IDLE,
            SyncStatus.SYNCING,
            SyncStatus.GENERATING,
            SyncStatus.TRAINING,
            SyncStatus.LOADING_ADAPTER,
            SyncStatus.ERROR,
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        )
    )
    + ((SyncStatus.SYNCING, False, False),),
)
def test_external_sync_active_status_matrix_requires_active_non_deleting_task(
    status,
    is_active,
    expected_active,
):
    runtime = _runtime_module()

    assert runtime._external_sync_is_active(
        SimpleNamespace(status=status, is_active=is_active)
    ) is expected_active


@pytest.fixture
def runtime_consumer_database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'runtime-consumers.db'}")
    SQLModel.metadata.create_all(
        engine,
        tables=[
            GenerationTaskDB.__table__,
            EvaluationTaskDB.__table__,
            ExternalSyncTaskDB.__table__,
        ],
    )
    return engine


def test_active_consumer_matrix_includes_live_workers_and_excludes_terminal_rows(
    runtime_consumer_database,
):
    runtime = _runtime_module()
    with Session(runtime_consumer_database) as session:
        session.add_all(
            [
                GenerationTaskDB(
                    task_id="generation-pending",
                    task_name="generation-pending",
                    input_path="/managed/input.jsonl",
                    llm_config={"config_id": "config-1"},
                    steps_config={},
                    status=GenerationStatus.PENDING,
                ),
                GenerationTaskDB(
                    task_id="generation-failed-worker",
                    task_name="generation-failed-worker",
                    input_path="/managed/input.jsonl",
                    llm_config={"config_id": "config-1"},
                    steps_config={},
                    status=GenerationStatus.FAILED,
                ),
                GenerationTaskDB(
                    task_id="generation-failed",
                    task_name="generation-failed",
                    input_path="/managed/input.jsonl",
                    llm_config={"config_id": "config-1"},
                    steps_config={},
                    status=GenerationStatus.FAILED,
                ),
                EvaluationTaskDB(
                    task_id="mteb-running",
                    eval_framework=EvaluationFramework.MTEB,
                    model_configs=[{"deployment_id": "deployment-1"}],
                    status=EvaluationStatus.RUNNING,
                ),
                EvaluationTaskDB(
                    task_id="deepeval-completed-worker",
                    eval_framework=EvaluationFramework.DEEPEVAL,
                    model_configs=[
                        {
                            "group_name": "group",
                            "embedding": {"config_id": "config-1"},
                        }
                    ],
                    status=EvaluationStatus.COMPLETED,
                ),
                EvaluationTaskDB(
                    task_id="deepeval-completed",
                    eval_framework=EvaluationFramework.DEEPEVAL,
                    model_configs=[
                        {
                            "group_name": "group",
                            "embedding": {"config_id": "config-1"},
                        }
                    ],
                    status=EvaluationStatus.COMPLETED,
                ),
                ExternalSyncTaskDB(
                    task_id="sync-active",
                    task_name="sync-active",
                    user_id="user-1",
                    generation_config={
                        "embedding_config": {"config_id": "config-1"}
                    },
                    is_active=True,
                    status=SyncStatus.IDLE,
                ),
                ExternalSyncTaskDB(
                    task_id="sync-inactive",
                    task_name="sync-inactive",
                    user_id="user-1",
                    generation_config={
                        "embedding_config": {"config_id": "config-1"}
                    },
                    is_active=False,
                    status=SyncStatus.SYNCING,
                ),
                ExternalSyncTaskDB(
                    task_id="sync-deleting",
                    task_name="sync-deleting",
                    user_id="user-1",
                    generation_config={
                        "embedding_config": {"config_id": "config-1"}
                    },
                    is_active=True,
                    status=SyncStatus.DELETING,
                ),
            ]
        )
        session.commit()

    snapshot = runtime.RuntimeExecutionSnapshot(
        generation_task_ids=frozenset({"generation-failed-worker"}),
        evaluation_task_ids=frozenset({"deepeval-completed-worker"}),
    )
    with Session(runtime_consumer_database) as session:
        consumers = runtime.lock_active_runtime_dependency_consumers(
            session,
            runtime.RuntimeDependencyReferences(
                config_ids=("config-1",),
                deployment_ids=("deployment-1",),
            ),
            execution_snapshot=snapshot,
        )

    assert [(item.kind, item.task_id) for item in consumers] == [
        ("deepeval", "deepeval-completed-worker"),
        ("external_sync", "sync-active"),
        ("generation", "generation-failed-worker"),
        ("generation", "generation-pending"),
        ("mteb", "mteb-running"),
    ]


@pytest.fixture
def guarded_task_services(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'guarded-tasks.db'}")
    SQLModel.metadata.create_all(engine)

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    evaluation_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    deep_module = importlib.import_module(
        "train_factory.storage.services.deep_evaluation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(generation_module, "get_session", test_session)
    monkeypatch.setattr(evaluation_module, "get_session", test_session)
    monkeypatch.setattr(deep_module, "get_session", test_session)
    sync_service = sync_module.ExternalSyncService()
    sync_service.engine = engine
    return SimpleNamespace(
        engine=engine,
        generation=generation_module.GenerationTaskService(),
        evaluation=evaluation_module.EvaluationTaskService(),
        deep=deep_module.DeepEvaluationTaskService(),
        sync=sync_service,
    )


def test_external_sync_create_locks_config_and_all_bindings_in_one_union(
    guarded_task_services,
    monkeypatch,
):
    runtime = _runtime_module()
    _add_external_cross_binding_graph(guarded_task_services.engine)
    recorded = _record_runtime_dependency_locks(monkeypatch)

    guarded_task_services.sync.create_task(
        task_name="sync-cross-create",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "config-1"}
        },
        base_deployment_id="deployment-2",
        training_targets=[
            {
                "target_name": "target",
                "base_deployment_id": "deployment-2",
            }
        ],
        is_active=True,
    )

    assert recorded == [
        runtime.RuntimeDependencyReferences(
            config_ids=("config-1",),
            deployment_ids=("deployment-2",),
        )
    ]


@pytest.mark.parametrize("operation", ["create", "update"])
def test_external_sync_writer_passes_task_owner_to_dependency_guard(
    guarded_task_services,
    monkeypatch,
    operation,
):
    runtime = _runtime_module()
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    recorded_owners = []

    if operation == "update":
        existing = guarded_task_services.sync.create_task(
            task_name="sync-owner-update",
            user_id="user-1",
            external_api_url="https://example.invalid/api",
            is_active=False,
        )

    def recording_lock(session, references, *, expected_user_id=None):
        recorded_owners.append(expected_user_id)
        return runtime.LockedRuntimeDependencies()

    monkeypatch.setattr(
        sync_module,
        "lock_runtime_dependencies",
        recording_lock,
    )

    if operation == "create":
        guarded_task_services.sync.create_task(
            task_name="sync-owner-create",
            user_id="user-1",
            external_api_url="https://example.invalid/api",
            generation_config={
                "eval_llm_config": {"config_id": "config-owned"}
            },
            is_active=True,
        )
    else:
        guarded_task_services.sync.update_task(
            existing["task_id"],
            expected_user_id="user-1",
            generation_config={
                "eval_llm_config": {"config_id": "config-owned"}
            },
            is_active=True,
        )

    assert recorded_owners == ["user-1"]


@pytest.mark.parametrize("operation", ["create", "activate"])
def test_auth_disabled_sync_route_normalizes_empty_owner_for_runtime_dependencies(
    guarded_task_services,
    monkeypatch,
    operation,
):
    sync_routes = importlib.import_module("train_factory.api.routes.sync_routes")
    sync_service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    model_config_module = importlib.import_module(
        "train_factory.storage.services.model_config_service"
    )
    sync_manager_module = importlib.import_module(
        "train_factory.sync.sync_manager"
    )
    _add_owned_runtime_dependency_graph(
        guarded_task_services.engine,
        model_user_id=None,
        deployment_user_id=None,
        config_user_id=None,
    )
    monkeypatch.setattr(
        sync_service_module.external_sync_service,
        "engine",
        guarded_task_services.engine,
    )
    monkeypatch.setattr(
        sync_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=False),
    )
    monkeypatch.setattr(
        sync_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )
    monkeypatch.setattr(
        model_config_module.model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "owned-config",
            "user_id": None,
            "api_endpoint": "https://models.example.invalid/v1",
        },
    )
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "start_worker",
        lambda _task_id: None,
    )
    monkeypatch.setattr(sync_manager_module.sync_manager, "_running", False)

    if operation == "create":
        response = asyncio.run(
            sync_routes.create_sync_task(
                sync_routes.SyncTaskCreateRequest(
                    task_name="auth-disabled-create",
                    external_api_url="https://sync.example.invalid/data",
                    generation_config={
                        "embedding_config": {"config_id": "owned-config"}
                    },
                    is_active=True,
                ),
                {"user_id": None, "username": "anonymous"},
            )
        )
        task_id = response["task"]["task_id"]
    else:
        with Session(guarded_task_services.engine) as session:
            task = ExternalSyncTaskDB(
                task_id="auth-disabled-activate",
                task_name="auth-disabled-activate",
                user_id="",
                generation_config={
                    "embedding_config": {"config_id": "owned-config"}
                },
                status=SyncStatus.IDLE,
                is_active=False,
            )
            session.add(task)
            session.commit()

        @asynccontextmanager
        async def operation_lock(_task_id):
            yield

        monkeypatch.setattr(
            sync_manager_module.sync_manager,
            "task_operation_lock",
            operation_lock,
        )
        response = asyncio.run(
            sync_routes.update_sync_task(
                "auth-disabled-activate",
                sync_routes.SyncTaskUpdateRequest(is_active=True),
                {"user_id": None, "username": "anonymous"},
            )
        )
        task_id = response["task"]["task_id"]

    persisted = sync_service_module.external_sync_service.get_task(task_id)
    assert persisted is not None
    assert persisted["user_id"] == ""
    assert persisted["is_active"] is True


@pytest.mark.parametrize("candidate_owner", ["", "user-2"])
def test_external_sync_writer_rejects_authenticated_owner_drift_before_dependencies(
    guarded_task_services,
    candidate_owner,
):
    runtime = _runtime_module()
    task_id = f"early-owner-drift-{candidate_owner or 'empty'}"
    original_generation_config = {
        "embedding_config": {"config_id": "private-config-before"}
    }
    with Session(guarded_task_services.engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="before",
                user_id=candidate_owner,
                generation_config=original_generation_config,
                status=SyncStatus.IDLE,
                is_active=True,
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id=f"{task_id}-target",
                task_id=task_id,
                target_name="unchanged",
                is_active=True,
            )
        )
        session.commit()

    selected_tables = []

    def record_select(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        normalized = statement.lower().lstrip()
        if not normalized.startswith("select"):
            return
        for table_name in (
            "external_sync_tasks",
            "external_sync_training_targets",
            "model_configs",
            "deployments",
            "model_registry",
        ):
            if f"from {table_name}" in normalized:
                selected_tables.append(table_name)
                return

    event.listen(
        guarded_task_services.engine,
        "before_cursor_execute",
        record_select,
    )
    try:
        with pytest.raises(runtime.RuntimeDependencyUnavailableError) as exc_info:
            guarded_task_services.sync.update_task(
                task_id,
                expected_user_id="user-1",
                task_name="after",
                generation_config={
                    "embedding_config": {
                        "config_id": "private-config-after"
                    }
                },
            )
    finally:
        event.remove(
            guarded_task_services.engine,
            "before_cursor_execute",
            record_select,
        )

    assert (
        selected_tables,
        str(exc_info.value),
    ) == (
        ["external_sync_tasks"],
        "Runtime dependency is unavailable",
    )
    assert "private-config" not in str(exc_info.value)

    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == task_id
            )
        ).one()
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.task_id == task_id
            )
        ).one()
        assert task.task_name == "before"
        assert task.generation_config == original_generation_config
        assert target.target_name == "unchanged"


@pytest.mark.parametrize("task_owner", ["", "user-2"])
@pytest.mark.parametrize("operation", ["create", "update", "legacy"])
def test_external_sync_target_writer_rejects_unowned_task_without_mutation(
    guarded_task_services,
    task_owner,
    operation,
):
    task_id = f"unowned-{operation}-{task_owner or 'ownerless'}"
    original_training_config = (
        {"model_type": "embedding"} if operation == "legacy" else None
    )
    with Session(guarded_task_services.engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name=task_id,
                user_id=task_owner,
                status=SyncStatus.IDLE,
                is_active=True,
                training_config=original_training_config,
                pending_training_samples=7,
            )
        )
        if operation == "update":
            session.add(
                ExternalSyncTrainingTargetDB(
                    target_id=f"{task_id}-target",
                    task_id=task_id,
                    target_name="before",
                    is_active=True,
                )
            )
        session.commit()

    with pytest.raises(
        _runtime_module().RuntimeDependencyUnavailableError,
        match="^Runtime dependency is unavailable$",
    ) as exc_info:
        if operation == "create":
            guarded_task_services.sync.create_training_target(
                task_id,
                "blocked",
                target_id=f"{task_id}-target",
                expected_user_id="user-1",
            )
        elif operation == "update":
            guarded_task_services.sync.update_training_target(
                f"{task_id}-target",
                task_id=task_id,
                target_name="after",
                expected_user_id="user-1",
            )
        else:
            guarded_task_services.sync.migrate_legacy_training_target(
                task_id=task_id,
                target_name="legacy",
                target_id=f"{task_id}-target",
                training_config=original_training_config,
                expected_user_id="user-1",
            )

    assert task_id not in str(exc_info.value)
    if task_owner:
        assert task_owner not in str(exc_info.value)
    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == task_id
            )
        ).one()
        targets = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.task_id == task_id
            )
        ).all()
        assert task.status == SyncStatus.IDLE
        assert task.pending_training_samples == 7
        assert task.training_config == original_training_config
        if operation == "update":
            assert len(targets) == 1
            assert targets[0].target_name == "before"
        else:
            assert targets == []


@pytest.mark.parametrize("operation", ["create", "update", "legacy"])
def test_external_sync_target_writer_uses_persisted_task_owner_for_dependencies(
    guarded_task_services,
    monkeypatch,
    operation,
):
    runtime = _runtime_module()
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    task_id = f"persisted-owner-{operation}"
    with Session(guarded_task_services.engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name=task_id,
                user_id="user-1",
                status=SyncStatus.IDLE,
                is_active=True,
                training_config=(
                    {"model_type": "embedding"}
                    if operation == "legacy"
                    else None
                ),
            )
        )
        if operation == "update":
            session.add(
                ExternalSyncTrainingTargetDB(
                    target_id=f"{task_id}-target",
                    task_id=task_id,
                    target_name="before",
                    is_active=False,
                )
            )
        session.commit()

    recorded_owners = []

    def recording_lock(session, references, *, expected_user_id=None):
        recorded_owners.append(expected_user_id)
        return runtime.LockedRuntimeDependencies()

    monkeypatch.setattr(
        sync_module,
        "lock_runtime_dependencies",
        recording_lock,
    )

    if operation == "create":
        guarded_task_services.sync.create_training_target(
            task_id,
            "created",
            target_id=f"{task_id}-target",
        )
    elif operation == "update":
        guarded_task_services.sync.update_training_target(
            f"{task_id}-target",
            task_id=task_id,
            is_active=True,
        )
    else:
        guarded_task_services.sync.migrate_legacy_training_target(
            task_id=task_id,
            target_name="legacy",
            target_id=f"{task_id}-target",
            training_config={"model_type": "embedding"},
        )

    assert recorded_owners == ["user-1"]


def test_external_sync_update_locks_prospective_config_and_bindings_once(
    guarded_task_services,
    monkeypatch,
):
    runtime = _runtime_module()
    _add_external_cross_binding_graph(guarded_task_services.engine)
    created = guarded_task_services.sync.create_task(
        task_name="sync-cross-update",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        base_deployment_id="deployment-1",
        is_active=False,
    )
    recorded = _record_runtime_dependency_locks(monkeypatch)

    guarded_task_services.sync.update_task(
        created["task_id"],
        expected_user_id="user-1",
        is_active=True,
        generation_config={
            "embedding_config": {"config_id": "config-1"}
        },
        base_deployment_id="deployment-2",
    )

    assert recorded == [
        runtime.RuntimeDependencyReferences(
            config_ids=("config-1",),
            deployment_ids=("deployment-2",),
        )
    ]


def test_external_sync_target_create_locks_parent_graph_and_new_binding_once(
    guarded_task_services,
    monkeypatch,
):
    runtime = _runtime_module()
    _add_external_cross_binding_graph(guarded_task_services.engine)
    created = guarded_task_services.sync.create_task(
        task_name="sync-target-create",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "config-1"}
        },
        base_deployment_id="deployment-1",
        is_active=True,
    )
    recorded = _record_runtime_dependency_locks(monkeypatch)

    guarded_task_services.sync.create_training_target(
        created["task_id"],
        target_id="target-created",
        target_name="target-created",
        base_deployment_id="deployment-2",
        expected_user_id="user-1",
    )

    assert recorded == [
        runtime.RuntimeDependencyReferences(
            config_ids=("config-1",),
            deployment_ids=("deployment-1", "deployment-2"),
        )
    ]


def test_external_sync_target_update_locks_parent_graph_and_new_binding_once(
    guarded_task_services,
    monkeypatch,
):
    runtime = _runtime_module()
    _add_external_cross_binding_graph(guarded_task_services.engine)
    created = guarded_task_services.sync.create_task(
        task_name="sync-target-update",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "config-1"}
        },
        base_deployment_id="deployment-1",
        training_targets=[
            {
                "target_id": "target-updated",
                "target_name": "target-updated",
                "base_deployment_id": "deployment-1",
            }
        ],
        is_active=True,
    )
    recorded = _record_runtime_dependency_locks(monkeypatch)

    guarded_task_services.sync.update_training_target(
        "target-updated",
        task_id=created["task_id"],
        expected_user_id="user-1",
        base_deployment_id="deployment-2",
    )

    assert recorded == [
        runtime.RuntimeDependencyReferences(
            config_ids=("config-1",),
            deployment_ids=("deployment-1", "deployment-2"),
        )
    ]


def test_external_sync_create_locks_external_api_before_runtime_dependencies(
    guarded_task_services,
):
    _add_external_cross_binding_graph(guarded_task_services.engine)
    _add_external_api_config(guarded_task_services.engine, "api-config-1")
    selected_tables = []

    def record_select(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        normalized = statement.lower().lstrip()
        if not normalized.startswith("select"):
            return
        for table in (
            "external_api_configs",
            "model_configs",
            "deployments",
            "model_registry",
        ):
            if f"from {table}" in normalized:
                selected_tables.append(table)
                return

    event.listen(
        guarded_task_services.engine,
        "before_cursor_execute",
        record_select,
    )
    try:
        guarded_task_services.sync.create_task(
            task_name="sync-api-order-create",
            user_id="user-1",
            external_api_config_id="api-config-1",
            generation_config={
                "embedding_config": {"config_id": "config-1"}
            },
            base_deployment_id="deployment-2",
            is_active=True,
        )
    finally:
        event.remove(
            guarded_task_services.engine,
            "before_cursor_execute",
            record_select,
        )

    assert selected_tables[0] == "external_api_configs"


def test_external_sync_update_locks_sorted_old_new_api_union_before_runtime(
    guarded_task_services,
):
    _add_external_cross_binding_graph(guarded_task_services.engine)
    _add_external_api_config(guarded_task_services.engine, "api-config-a")
    _add_external_api_config(guarded_task_services.engine, "api-config-b")
    with Session(guarded_task_services.engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-api-order-update",
                task_name="sync-api-order-update",
                user_id="user-1",
                external_api_config_id="api-config-b",
                is_active=False,
            )
        )
        session.commit()
    selected = []

    def record_select(
        _connection,
        _cursor,
        statement,
        parameters,
        _context,
        _executemany,
    ):
        normalized = statement.lower().lstrip()
        if not normalized.startswith("select"):
            return
        for table in (
            "external_api_configs",
            "model_configs",
            "deployments",
            "model_registry",
        ):
            if f"from {table}" in normalized:
                selected.append((table, tuple(parameters)))
                return

    event.listen(
        guarded_task_services.engine,
        "before_cursor_execute",
        record_select,
    )
    try:
        guarded_task_services.sync.update_task(
            "sync-api-order-update",
            expected_user_id="user-1",
            external_api_config_id="api-config-a",
            is_active=True,
            generation_config={
                "embedding_config": {"config_id": "config-1"}
            },
            base_deployment_id="deployment-2",
        )
    finally:
        event.remove(
            guarded_task_services.engine,
            "before_cursor_execute",
            record_select,
        )

    assert selected[0] == (
        "external_api_configs",
        ("api-config-a", "api-config-b"),
    )


def test_external_api_delete_fails_closed_when_sync_task_references_config(
    guarded_task_services,
):
    service_module = importlib.import_module(
        "train_factory.storage.services.external_api_config_service"
    )
    service = service_module.ExternalApiConfigService()
    service.engine = guarded_task_services.engine
    _add_external_api_config(guarded_task_services.engine, "api-config-delete")
    with Session(guarded_task_services.engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-api-delete-reference",
                task_name="sync-api-delete-reference",
                user_id="user-1",
                external_api_config_id="api-config-delete",
                is_active=False,
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="referenced"):
        service.delete_config("api-config-delete")

    assert service.get_config("api-config-delete") is not None


def test_external_api_delete_fails_closed_for_legacy_deployment_reference(
    guarded_task_services,
):
    service_module = importlib.import_module(
        "train_factory.storage.services.external_api_config_service"
    )
    service = service_module.ExternalApiConfigService()
    service.engine = guarded_task_services.engine
    _add_external_api_config(guarded_task_services.engine, "api-config-legacy")
    with Session(guarded_task_services.engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="deployment-api-legacy",
                model_id="model-api-legacy",
                xinference_endpoint="http://inference:8010",
                user_id="user-1",
                config={"external_api_config_id": "api-config-legacy"},
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="referenced"):
        service.delete_config("api-config-legacy")

    assert service.get_config("api-config-legacy") is not None


def test_external_sync_metadata_update_does_not_lock_runtime_dependencies(
    guarded_task_services,
    monkeypatch,
):
    created = guarded_task_services.sync.create_task(
        task_name="sync-metadata-before",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "stale-config"}
        },
        is_active=False,
    )
    recorded = _record_runtime_dependency_locks(monkeypatch)

    updated = guarded_task_services.sync.update_task(
        created["task_id"],
        expected_user_id="user-1",
        task_name="sync-metadata-after",
    )

    assert updated["task_name"] == "sync-metadata-after"
    assert recorded == []


def test_external_sync_target_metadata_update_does_not_lock_parent_graph(
    guarded_task_services,
    monkeypatch,
):
    _add_external_cross_binding_graph(guarded_task_services.engine)
    created = guarded_task_services.sync.create_task(
        task_name="sync-target-metadata",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "config-1"}
        },
        base_deployment_id="deployment-1",
        training_targets=[
            {
                "target_id": "target-metadata",
                "target_name": "before",
                "base_deployment_id": "deployment-1",
            }
        ],
        is_active=True,
    )
    recorded = _record_runtime_dependency_locks(monkeypatch)

    updated = guarded_task_services.sync.update_training_target(
        "target-metadata",
        task_id=created["task_id"],
        expected_user_id="user-1",
        target_name="after",
    )

    assert updated["target_name"] == "after"
    assert recorded == []


def test_generation_create_rejects_config_whose_model_is_deleting(
    guarded_task_services,
):
    runtime = _runtime_module()
    _add_runtime_dependency_graph(
        guarded_task_services.engine,
        model_status="deleting",
    )

    with pytest.raises(
        runtime.RuntimeDependencyUnavailableError,
        match="Model is being deleted: model-1",
    ):
        guarded_task_services.generation.create_task(
            task_name="generation",
            input_path="/managed/input.jsonl",
            llm_config={"config_id": "config-1"},
            steps_config={},
        )


def test_mteb_create_rejects_deployment_delete_claim(guarded_task_services):
    runtime = _runtime_module()
    _add_runtime_dependency_graph(
        guarded_task_services.engine,
        deployment_delete_claim=True,
    )

    with pytest.raises(
        runtime.RuntimeDependencyUnavailableError,
        match="Deployment deletion is in progress: deployment-1",
    ):
        guarded_task_services.evaluation.create_task(
            model_configs=[{"deployment_id": "deployment-1"}],
            dataset_configs=[],
        )


def test_deepeval_create_rejects_config_whose_model_is_deleting(
    guarded_task_services,
):
    runtime = _runtime_module()
    _add_runtime_dependency_graph(
        guarded_task_services.engine,
        model_status="deleting",
    )

    with pytest.raises(
        runtime.RuntimeDependencyUnavailableError,
        match="Model is being deleted: model-1",
    ):
        guarded_task_services.deep.create_task(
            model_configs=[
                {
                    "group_name": "group",
                    "embedding": {"config_id": "config-1"},
                }
            ],
            dataset_configs=[],
        )


def test_external_sync_active_create_rejects_missing_generation_config(
    guarded_task_services,
):
    runtime = _runtime_module()

    with pytest.raises(
        runtime.RuntimeDependencyUnavailableError,
        match="Model config not found: missing-config",
    ):
        guarded_task_services.sync.create_task(
            task_name="sync-active",
            user_id="user-1",
            external_api_url="https://example.invalid/api",
            generation_config={
                "embedding_config": {"config_id": "missing-config"}
            },
            is_active=True,
        )


def test_external_sync_inactive_create_may_retain_stale_generation_config(
    guarded_task_services,
):
    created = guarded_task_services.sync.create_task(
        task_name="sync-inactive",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "missing-config"}
        },
        is_active=False,
    )

    assert created["is_active"] is False


def _mark_model_deleting(engine):
    with Session(engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == "model-1")
        ).one()
        model.status = "deleting"
        session.add(model)
        session.commit()


def _claim_deployment_deletion(engine):
    with Session(engine) as session:
        deployment = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-1"
            )
        ).one()
        deployment.replica_operation_token = "delete-token"
        deployment.replica_operation_kind = "delete"
        session.add(deployment)
        session.commit()


def _generation_with_runtime_config(services, task_id):
    return services.generation.create_task(
        task_id=task_id,
        task_name=task_id,
        input_path="/managed/input.jsonl",
        llm_config={"config_id": "config-1"},
        steps_config={},
    )


def test_generation_claim_rechecks_runtime_dependencies(guarded_task_services):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    _generation_with_runtime_config(guarded_task_services, "generation-claim")
    _mark_model_deleting(guarded_task_services.engine)

    assert (
        guarded_task_services.generation.claim_running("generation-claim")
        is None
    )
    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == "generation-claim"
            )
        ).one()
        assert task.status == GenerationStatus.PENDING


def test_generation_direct_running_transition_rechecks_runtime_dependencies(
    guarded_task_services,
):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    _generation_with_runtime_config(
        guarded_task_services,
        "generation-direct-running",
    )
    _mark_model_deleting(guarded_task_services.engine)

    assert not guarded_task_services.generation.update_status(
        "generation-direct-running",
        GenerationStatus.RUNNING,
    )


def test_generation_terminal_to_pending_rechecks_runtime_dependencies(
    guarded_task_services,
):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    _generation_with_runtime_config(guarded_task_services, "generation-retry")
    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == "generation-retry"
            )
        ).one()
        task.status = GenerationStatus.FAILED
        session.add(task)
        session.commit()
    _mark_model_deleting(guarded_task_services.engine)

    assert not guarded_task_services.generation.update_status(
        "generation-retry",
        GenerationStatus.PENDING,
    )


def test_generation_begin_restart_rechecks_runtime_dependencies(
    guarded_task_services,
):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    _generation_with_runtime_config(guarded_task_services, "generation-restart")
    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == "generation-restart"
            )
        ).one()
        task.status = GenerationStatus.FAILED
        session.add(task)
        session.commit()
    _mark_model_deleting(guarded_task_services.engine)

    assert (
        guarded_task_services.generation.begin_restart(
            "generation-restart",
            new_run_token=str(uuid4()),
        )
        is None
    )


def test_mteb_claim_rechecks_runtime_dependencies(guarded_task_services):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    created = guarded_task_services.evaluation.create_task(
        model_configs=[{"deployment_id": "deployment-1"}],
        dataset_configs=[],
    )
    _claim_deployment_deletion(guarded_task_services.engine)

    assert not guarded_task_services.evaluation.claim_running(created["task_id"])


def test_mteb_direct_running_transition_rechecks_runtime_dependencies(
    guarded_task_services,
):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    created = guarded_task_services.evaluation.create_task(
        model_configs=[{"deployment_id": "deployment-1"}],
        dataset_configs=[],
    )
    _claim_deployment_deletion(guarded_task_services.engine)

    assert not guarded_task_services.evaluation.update_status(
        created["task_id"],
        EvaluationStatus.RUNNING,
    )


def test_mteb_resume_rechecks_runtime_dependencies(guarded_task_services):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    created = guarded_task_services.evaluation.create_task(
        model_configs=[{"deployment_id": "deployment-1"}],
        dataset_configs=[],
    )
    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(EvaluationTaskDB).where(
                EvaluationTaskDB.task_id == created["task_id"]
            )
        ).one()
        task.status = EvaluationStatus.FAILED
        session.add(task)
        session.commit()
    _claim_deployment_deletion(guarded_task_services.engine)

    assert not guarded_task_services.evaluation.reset_for_resume(
        created["task_id"]
    )


def test_mteb_resume_guards_prospective_model_configs(guarded_task_services):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    created = guarded_task_services.evaluation.create_task(
        model_configs=[{"deployment_id": "deployment-1"}],
        dataset_configs=[],
    )
    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(EvaluationTaskDB).where(
                EvaluationTaskDB.task_id == created["task_id"]
            )
        ).one()
        task.status = EvaluationStatus.FAILED
        session.add(task)
        session.commit()

    assert not guarded_task_services.evaluation.reset_for_resume(
        created["task_id"],
        model_configs=[{"deployment_id": "missing-deployment"}],
    )
    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(EvaluationTaskDB).where(
                EvaluationTaskDB.task_id == created["task_id"]
            )
        ).one()
        assert task.status == EvaluationStatus.FAILED
        assert task.model_configs == [{"deployment_id": "deployment-1"}]


def test_deepeval_claim_rechecks_runtime_dependencies(guarded_task_services):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    created = guarded_task_services.deep.create_task(
        model_configs=[
            {
                "group_name": "group",
                "embedding": {"config_id": "config-1"},
            }
        ],
        dataset_configs=[],
    )
    _mark_model_deleting(guarded_task_services.engine)

    assert not guarded_task_services.deep.claim_running(created["task_id"])


def test_deepeval_direct_running_transition_rechecks_runtime_dependencies(
    guarded_task_services,
):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    created = guarded_task_services.deep.create_task(
        model_configs=[
            {
                "group_name": "group",
                "embedding": {"config_id": "config-1"},
            }
        ],
        dataset_configs=[],
    )
    _mark_model_deleting(guarded_task_services.engine)

    assert not guarded_task_services.deep.update_status(
        created["task_id"],
        EvaluationStatus.RUNNING,
    )


def test_deepeval_resume_rechecks_runtime_dependencies(guarded_task_services):
    _add_runtime_dependency_graph(guarded_task_services.engine)
    created = guarded_task_services.deep.create_task(
        model_configs=[
            {
                "group_name": "group",
                "embedding": {"config_id": "config-1"},
            }
        ],
        dataset_configs=[],
    )
    with Session(guarded_task_services.engine) as session:
        task = session.exec(
            select(EvaluationTaskDB).where(
                EvaluationTaskDB.task_id == created["task_id"]
            )
        ).one()
        task.status = EvaluationStatus.FAILED
        session.add(task)
        session.commit()
    _mark_model_deleting(guarded_task_services.engine)

    assert not guarded_task_services.deep.reset_for_resume(created["task_id"])


def test_external_sync_inactive_to_active_rechecks_generation_configs(
    guarded_task_services,
):
    runtime = _runtime_module()
    created = guarded_task_services.sync.create_task(
        task_name="sync-inactive",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "missing-config"}
        },
        is_active=False,
    )

    with pytest.raises(
        runtime.RuntimeDependencyUnavailableError,
        match="Model config not found: missing-config",
    ):
        guarded_task_services.sync.update_task(
            created["task_id"],
            expected_user_id="user-1",
            is_active=True,
        )
    assert guarded_task_services.sync.get_task(created["task_id"])["is_active"] is False


def test_external_sync_active_config_replacement_rechecks_generation_configs(
    guarded_task_services,
):
    runtime = _runtime_module()
    _add_runtime_dependency_graph(
        guarded_task_services.engine,
        user_id="user-1",
    )
    created = guarded_task_services.sync.create_task(
        task_name="sync-active",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "config-1"}
        },
        is_active=True,
    )

    with pytest.raises(
        runtime.RuntimeDependencyUnavailableError,
        match="Model config not found: missing-config",
    ):
        guarded_task_services.sync.update_task(
            created["task_id"],
            expected_user_id="user-1",
            generation_config={
                "embedding_config": {"config_id": "missing-config"}
            },
        )
    persisted = guarded_task_services.sync.get_task(created["task_id"])
    assert persisted["generation_config"] == {
        "embedding_config": {"config_id": "config-1"}
    }


def test_external_sync_deletion_cancel_rechecks_dependencies_before_reactivation(
    guarded_task_services,
):
    runtime = _runtime_module()
    _add_external_cross_binding_graph(guarded_task_services.engine)
    created = guarded_task_services.sync.create_task(
        task_name="sync-cancel-delete",
        user_id="user-1",
        external_api_url="https://example.invalid/api",
        generation_config={
            "embedding_config": {"config_id": "config-1"}
        },
        base_deployment_id="deployment-2",
        is_active=True,
    )
    guarded_task_services.sync.begin_task_deletion(
        created["task_id"],
        cascade=False,
        expected_user_id="user-1",
    )
    with Session(guarded_task_services.engine) as session:
        config = session.exec(
            select(ModelConfigDB).where(ModelConfigDB.config_id == "config-1")
        ).one()
        session.delete(config)
        session.commit()

    with pytest.raises(
        runtime.RuntimeDependencyUnavailableError,
        match="Model config not found: config-1",
    ):
        guarded_task_services.sync.cancel_task_deletion(
            created["task_id"],
            status=SyncStatus.IDLE,
            is_active=True,
            expected_user_id="user-1",
            expected_deleting_status=SyncStatus.DELETING,
        )

    persisted = guarded_task_services.sync.get_task(created["task_id"])
    assert persisted["status"] == SyncStatus.DELETING
    assert persisted["is_active"] is False


def test_standard_evaluation_create_maps_runtime_dependency_race_to_409(
    monkeypatch,
):
    from train_factory.api.routes import evaluation_routes

    runtime = _runtime_module()
    request = evaluation_routes.CreateEvaluationRequest(
        model_configs=[
            evaluation_routes.ModelConfig(
                endpoint="https://example.com/v1/rerank",
                name="model",
            )
        ],
        dataset_configs=[
            evaluation_routes.DatasetConfig(type="mteb", name="T2Reranking")
        ],
    )
    monkeypatch.setattr(
        evaluation_routes,
        "_validate_model_configs",
        lambda configs, _user_id: configs,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "_validate_dataset_configs",
        lambda configs, _user: configs,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime.RuntimeDependencyUnavailableError(
                "runtime dependency is being deleted"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.create_evaluation_task(
                request,
                BackgroundTasks(),
                {"user_id": "user-1", "username": "user", "role": "user"},
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "runtime dependency is being deleted"


def test_deep_evaluation_create_maps_runtime_dependency_race_to_409(
    monkeypatch,
):
    from train_factory.api.routes import deep_evaluation_routes

    runtime = _runtime_module()
    request = deep_evaluation_routes.CreateDeepEvaluationTaskRequest(
        model_configs=[
            {
                "group_name": "group-1",
                "embedding": {
                    "endpoint": "https://example.com/v1",
                    "model_name": "embedding",
                },
            }
        ],
        dataset_configs=[{"dataset_id": "dataset-1"}],
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "_resolve_deep_evaluation_dataset_configs",
        lambda *_args, **_kwargs: [
            {"dataset_id": "dataset-1", "dataset_name": "dataset"}
        ],
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, *_args, **_kwargs: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime.RuntimeDependencyUnavailableError(
                "runtime dependency is being deleted"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                request,
                BackgroundTasks(),
                {"user_id": "user-1", "username": "user", "role": "user"},
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "runtime dependency is being deleted"


def test_generation_create_maps_runtime_dependency_race_to_409(
    monkeypatch,
    tmp_path,
):
    from train_factory.api.routes import generation_routes

    runtime = _runtime_module()
    input_path = tmp_path / "input.txt"
    input_path.write_text("document", encoding="utf-8")
    request = generation_routes.CreateTaskRequest(
        task_name="runtime-dependency-race",
        dataset_id="dataset-1",
        input_format="txt",
        generation_mode="qa_extraction",
        llm_config={
            "endpoint": "https://example.com/v1",
            "model": "model",
        },
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_owned_generation_dataset",
        lambda *_args, **_kwargs: {
            "dataset_id": "dataset-1",
            "storage_path": str(input_path),
            "extra_metadata": {},
        },
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "validate_user_outbound_url",
        lambda url, *_args, **_kwargs: url,
    )
    monkeypatch.setattr(
        generation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime.RuntimeDependencyUnavailableError(
                "runtime dependency is being deleted"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.create_task(
                request,
                BackgroundTasks(),
                {"user_id": "user-1", "username": "user", "role": "user"},
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "runtime dependency is being deleted"


def test_external_api_delete_maps_authoritative_reference_race_to_409(
    monkeypatch,
):
    from train_factory.api.routes import external_api_config_routes

    service = importlib.import_module(
        "train_factory.storage.services.external_api_config_service"
    ).external_api_config_service
    monkeypatch.setattr(
        service,
        "get_config",
        lambda _config_id: {
            "config_id": "api-config-delete",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        service,
        "is_referenced_by_sync",
        lambda _config_id: False,
    )
    monkeypatch.setattr(
        service,
        "delete_config",
        lambda _config_id: (_ for _ in ()).throw(
            ValueError("External API config is referenced")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            external_api_config_routes.delete_api_config(
                "api-config-delete",
                {"user_id": "user-1", "username": "user", "role": "user"},
            )
        )

    assert exc_info.value.status_code == 409
