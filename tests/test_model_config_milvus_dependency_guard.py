"""Model-config deletion and durable Milvus binding regression tests."""

from __future__ import annotations

from contextlib import contextmanager
import asyncio
import importlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import model_config_routes
from train_factory.enums.dataset_status import DatasetStatus
from train_factory.enums.sync_status import SyncStatus
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.evaluation_task_entity import (
    EvaluationFramework,
    EvaluationStatus,
    EvaluationTaskDB,
)
from train_factory.storage.entities.external_sync_entity import ExternalSyncTaskDB
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.milvus_collection_entity import (
    MilvusCollectionDB,
)
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.services.model_config_service import ModelConfigService


milvus_module = importlib.import_module(
    "train_factory.storage.services.milvus_collection_service"
)
publication_module = importlib.import_module(
    "train_factory.storage.services.generation_publication_service"
)


@pytest.fixture
def config_dependency_services(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'model-config-milvus-dependencies.db'}"
    )
    SQLModel.metadata.create_all(engine)
    service = ModelConfigService()
    service.engine = engine

    @contextmanager
    def test_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(milvus_module, "get_session", test_session)
    return SimpleNamespace(
        engine=engine,
        configs=service,
        collections=milvus_module.MilvusCollectionService(),
        publication=publication_module.GenerationPublicationService(
            session_factory=test_session
        ),
    )


def _add_embedding_config(engine, config_id: str = "embedding-config") -> None:
    with Session(engine) as session:
        session.add(
            ModelConfigDB(
                config_id=config_id,
                config_name=config_id,
                model_type="embedding",
                provider="openai-compatible",
                api_endpoint="https://embedding.example.invalid/v1",
                model_name="embedding-model",
                user_id="user-1",
            )
        )
        session.commit()


def test_model_config_delete_rejects_active_runtime_consumer(
    config_dependency_services,
):
    _add_embedding_config(config_dependency_services.engine)
    with Session(config_dependency_services.engine) as session:
        session.add(
            GenerationTaskDB(
                task_id="active-generation",
                task_name="active-generation",
                input_path="/managed/input.jsonl",
                llm_config={},
                steps_config={},
                embedding_config_id="embedding-config",
                status=GenerationStatus.RUNNING,
                user_id="user-1",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="active runtime task"):
        config_dependency_services.configs.delete_config("embedding-config")

    with Session(config_dependency_services.engine) as session:
        assert session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "embedding-config"
            )
        ).one()


@pytest.mark.parametrize("consumer_kind", ["mteb", "deepeval", "external_sync"])
def test_model_config_delete_rejects_each_active_runtime_consumer_kind(
    config_dependency_services,
    consumer_kind,
):
    _add_embedding_config(config_dependency_services.engine)
    with Session(config_dependency_services.engine) as session:
        if consumer_kind == "mteb":
            consumer = EvaluationTaskDB(
                task_id="active-mteb",
                eval_framework=EvaluationFramework.MTEB,
                model_configs=[{"config_id": "embedding-config"}],
                status=EvaluationStatus.RUNNING,
                user_id="user-1",
            )
        elif consumer_kind == "deepeval":
            consumer = EvaluationTaskDB(
                task_id="active-deepeval",
                eval_framework=EvaluationFramework.DEEPEVAL,
                model_configs=[
                    {
                        "group_name": "embedding",
                        "embedding": {"config_id": "embedding-config"},
                    }
                ],
                status=EvaluationStatus.RUNNING,
                user_id="user-1",
            )
        else:
            consumer = ExternalSyncTaskDB(
                task_id="active-sync",
                task_name="active-sync",
                generation_config={
                    "embedding_config": {"config_id": "embedding-config"}
                },
                status=SyncStatus.IDLE,
                is_active=True,
                user_id="user-1",
            )
        session.add(consumer)
        session.commit()

    with pytest.raises(ValueError, match="active runtime task"):
        config_dependency_services.configs.delete_config("embedding-config")


def test_model_config_delete_succeeds_without_runtime_or_milvus_reference(
    config_dependency_services,
):
    _add_embedding_config(config_dependency_services.engine)

    assert config_dependency_services.configs.delete_config(
        "embedding-config"
    ) is True

    with Session(config_dependency_services.engine) as session:
        assert session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "embedding-config"
            )
        ).first() is None


def test_model_config_delete_rejects_parent_delete_without_exact_owner_claim(
    config_dependency_services,
):
    with Session(config_dependency_services.engine) as session:
        session.add(
            ModelRegistryDB(
                model_id="deleting-model",
                model_name="deleting-model",
                model_type="embedding",
                model_path="/models/deleting-model",
                status="deleting",
            )
        )
        session.add(
            DeploymentDB(
                deployment_id="deleting-deployment",
                model_id="deleting-model",
                xinference_endpoint="http://inference:8000",
                replica_operation_token="delete-token",
                replica_operation_kind="delete",
                user_id="user-1",
            )
        )
        session.add(
            ModelConfigDB(
                config_id="deleting-parent-config",
                config_name="deleting-parent-config",
                source_type="local_deployed",
                registry_id="deleting-model",
                deployment_id="deleting-deployment",
                model_type="embedding",
                provider="openai-compatible",
                api_endpoint="http://inference:8000/v1",
                model_name="deleting-model",
                user_id="user-1",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="Model is being deleted"):
        config_dependency_services.configs.delete_config(
            "deleting-parent-config"
        )

    with Session(config_dependency_services.engine) as session:
        assert session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "deleting-parent-config"
            )
        ).one()


@pytest.mark.parametrize("collection_status", ["creating", "active", "deleting"])
def test_model_config_delete_rejects_any_durable_milvus_binding(
    config_dependency_services,
    collection_status,
):
    _add_embedding_config(config_dependency_services.engine)
    with Session(config_dependency_services.engine) as session:
        session.add(
            MilvusCollectionDB(
                collection_name=f"bound-{collection_status}",
                embedding_config_id="embedding-config",
                status=collection_status,
                user_id="user-1",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="Milvus collection"):
        config_dependency_services.configs.delete_config("embedding-config")

    with Session(config_dependency_services.engine) as session:
        assert session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "embedding-config"
            )
        ).one()


@pytest.mark.parametrize("writer", ["register", "reserve"])
def test_milvus_writer_rejects_missing_embedding_config(
    config_dependency_services,
    writer,
):
    def operation():
        if writer == "register":
            return config_dependency_services.collections.register_collection(
                "missing-config-collection",
                embedding_config_id="missing-config",
                user_id="user-1",
            )
        return config_dependency_services.collections.reserve_manual_collection(
            "missing-config-collection",
            embedding_config_id="missing-config",
            user_id="user-1",
        )

    with pytest.raises(
        milvus_module.MilvusCollectionUnavailableError,
        match="Embedding config is unavailable",
    ):
        operation()

    with Session(config_dependency_services.engine) as session:
        assert session.exec(select(MilvusCollectionDB)).all() == []


@pytest.mark.parametrize("writer", ["register", "reserve"])
def test_milvus_writer_rejects_deleting_embedding_config(
    config_dependency_services,
    writer,
):
    _add_embedding_config(config_dependency_services.engine)
    with Session(config_dependency_services.engine) as session:
        config = session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "embedding-config"
            )
        ).one()
        config.status = "deleting"
        session.add(config)
        session.commit()

    def operation():
        if writer == "register":
            return config_dependency_services.collections.register_collection(
                "deleting-config-collection",
                embedding_config_id="embedding-config",
                user_id="user-1",
            )
        return config_dependency_services.collections.reserve_manual_collection(
            "deleting-config-collection",
            embedding_config_id="embedding-config",
            user_id="user-1",
        )

    with pytest.raises(
        milvus_module.MilvusCollectionUnavailableError,
        match="Embedding config is unavailable",
    ):
        operation()

    with Session(config_dependency_services.engine) as session:
        assert session.exec(select(MilvusCollectionDB)).all() == []


def test_milvus_writer_locks_config_before_sync_namespace_and_collection(
    config_dependency_services,
    monkeypatch,
):
    events = []

    def lock_config(_session, config_id, *, expected_user_id):
        events.append(("config", config_id, expected_user_id))

    def lock_sync(_session, collection_name):
        events.append(("sync", collection_name))
        return [], []

    monkeypatch.setattr(
        milvus_module,
        "lock_embedding_model_config_for_binding",
        lock_config,
        raising=False,
    )
    monkeypatch.setattr(milvus_module, "_lock_sync_namespace_tasks", lock_sync)

    config_dependency_services.collections.register_collection(
        "ordered-collection",
        embedding_config_id="embedding-config",
        user_id="user-1",
    )

    assert events == [
        ("config", "embedding-config", "user-1"),
        ("sync", "ordered-collection"),
    ]


def test_milvus_namespace_lock_orders_sync_tasks_in_sql():
    statements = []

    class Result:
        def all(self):
            return []

    class RecordingSession:
        def exec(self, statement):
            statements.append(statement)
            return Result()

    milvus_module._lock_sync_namespace_tasks(
        RecordingSession(),
        "ordered-collection",
    )

    compiled = str(statements[0])
    assert "ORDER BY external_sync_tasks.task_id" in compiled


def test_milvus_writer_fails_closed_on_config_signature_drift(
    config_dependency_services,
    monkeypatch,
):
    runtime_module = importlib.import_module(
        "train_factory.storage.services.runtime_dependency_service"
    )

    def drift(*_args, **_kwargs):
        raise runtime_module.RuntimeDependencyChangedError(
            "Model config references changed while acquiring locks; retry"
        )

    monkeypatch.setattr(
        milvus_module,
        "lock_embedding_model_config_for_binding",
        drift,
        raising=False,
    )

    with pytest.raises(
        milvus_module.MilvusCollectionUnavailableError,
        match="Embedding config is unavailable",
    ):
        config_dependency_services.collections.register_collection(
            "drifted-config-collection",
            embedding_config_id="embedding-config",
            user_id="user-1",
        )

    with Session(config_dependency_services.engine) as session:
        assert session.exec(select(MilvusCollectionDB)).all() == []


def test_embedding_binding_guard_detects_locked_config_parent_drift():
    runtime_module = importlib.import_module(
        "train_factory.storage.services.runtime_dependency_service"
    )
    preliminary = SimpleNamespace(
        config_id="embedding-config",
        registry_id="model-before",
        deployment_id=None,
        status="active",
        model_type="embedding",
        user_id="user-1",
    )
    locked = SimpleNamespace(
        config_id="embedding-config",
        registry_id="model-after",
        deployment_id=None,
        status="active",
        model_type="embedding",
        user_id="user-1",
    )
    model = SimpleNamespace(
        model_id="model-before",
        status="available",
        user_id="user-1",
    )

    class Result:
        def __init__(self, values):
            self.values = values

        def all(self):
            return list(self.values)

    class DriftSession:
        def exec(self, statement):
            table_name = statement.get_final_froms()[0].name
            if table_name == "model_registry":
                return Result([model])
            if table_name == "model_configs":
                if statement._for_update_arg is not None:
                    return Result([locked])
                return Result([preliminary])
            raise AssertionError(f"Unexpected table: {table_name}")

    with pytest.raises(
        runtime_module.RuntimeDependencyChangedError,
        match="Model config references changed while acquiring locks",
    ):
        runtime_module.lock_embedding_model_config_for_binding(
            DriftSession(),
            "embedding-config",
            expected_user_id="user-1",
        )


@pytest.mark.parametrize("owner_id", [None, "user-2"])
def test_embedding_binding_guard_rejects_unowned_config_before_parent_lookup(
    config_dependency_services,
    owner_id,
):
    runtime_module = importlib.import_module(
        "train_factory.storage.services.runtime_dependency_service"
    )
    with Session(config_dependency_services.engine) as session:
        session.add(
            ModelConfigDB(
                config_id="private-embedding-config",
                config_name="private-embedding-config",
                source_type="local_deployed",
                registry_id="private-parent-model",
                deployment_id="private-parent-deployment",
                model_type="embedding",
                provider="openai-compatible",
                api_endpoint="https://embedding.example.invalid/v1",
                model_name="private-embedding-model",
                user_id=owner_id,
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
        for table_name in ("model_configs", "deployments", "model_registry"):
            if f"from {table_name}" in normalized:
                selected_tables.append(table_name)
                return

    event.listen(
        config_dependency_services.engine,
        "before_cursor_execute",
        record_select,
    )
    try:
        with Session(config_dependency_services.engine) as session:
            with pytest.raises(
                runtime_module.RuntimeDependencyUnavailableError,
                match="^Runtime dependency is unavailable$",
            ):
                runtime_module.lock_embedding_model_config_for_binding(
                    session,
                    "private-embedding-config",
                    expected_user_id="user-1",
                )
    finally:
        event.remove(
            config_dependency_services.engine,
            "before_cursor_execute",
            record_select,
        )

    assert selected_tables == ["model_configs"]


def _add_publication_inputs(engine) -> None:
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id="publishing-generation",
                task_name="publishing-generation",
                input_path="/managed/input.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.PUBLISHING,
                run_token="11111111-1111-4111-8111-111111111111",
                user_id="user-1",
            )
        )
        session.add(
            DatasetDB(
                dataset_id="publication-source",
                dataset_name="publication-source",
                storage_path="/managed/publication-source.jsonl",
                status=DatasetStatus.READY.value,
                user_id="user-1",
            )
        )
        session.commit()


def _complete_collection_publication(services, *, config_id: str) -> bool:
    return services.publication.complete_publication(
        "publishing-generation",
        expected_run_token="11111111-1111-4111-8111-111111111111",
        dataset_bindings={},
        milvus_registration={
            "collection_name": "publication-collection",
            "dataset_id": "publication-source",
            "user_id": "user-1",
            "embedding_config_id": config_id,
            "embedding_model": "embedding-model",
            "embedding_endpoint": "https://embedding.example.invalid/v1",
        },
    )


def test_generation_publication_rejects_missing_embedding_config_binding(
    config_dependency_services,
):
    _add_publication_inputs(config_dependency_services.engine)

    assert _complete_collection_publication(
        config_dependency_services,
        config_id="missing-config",
    ) is False

    with Session(config_dependency_services.engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == "publishing-generation"
            )
        ).one()
        assert task.status == GenerationStatus.PUBLISHING
        assert session.exec(select(MilvusCollectionDB)).all() == []


def test_generation_publication_locks_config_before_generation_task(
    config_dependency_services,
    monkeypatch,
):
    _add_publication_inputs(config_dependency_services.engine)
    events = []

    def lock_config(_session, config_id, *, expected_user_id):
        events.append(("config", config_id, expected_user_id))

    monkeypatch.setattr(
        publication_module,
        "lock_embedding_model_config_for_binding",
        lock_config,
        raising=False,
    )

    @event.listens_for(config_dependency_services.engine, "before_cursor_execute")
    def record_task_select(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        if statement.lstrip().upper().startswith("SELECT") and (
            "FROM generation_tasks" in statement
        ):
            events.append(("task", "publishing-generation", "user-1"))

    assert _complete_collection_publication(
        config_dependency_services,
        config_id="embedding-config",
    ) is True
    assert events[0] == ("config", "embedding-config", "user-1")
    assert events[1] == ("task", "publishing-generation", "user-1")


def test_model_config_delete_route_maps_dependency_conflict_to_409(monkeypatch):
    runtime_module = importlib.import_module(
        "train_factory.storage.services.runtime_dependency_service"
    )
    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "embedding-config",
            "config_name": "embedding-config",
            "user_id": "user-1",
        },
    )

    def reject_delete(_config_id):
        raise runtime_module.RuntimeDependencyUnavailableError(
            "Model config is referenced by an active runtime task"
        )

    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "delete_config",
        reject_delete,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            model_config_routes.delete_config(
                "embedding-config",
                {"user_id": "user-1", "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 409
