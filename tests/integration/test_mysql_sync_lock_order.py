"""Opt-in real-MySQL concurrency coverage for sync claim lock ordering."""

from __future__ import annotations

import importlib
import os
import re
import secrets
import threading
from contextlib import contextmanager
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlmodel import Session, select

from train_factory.enums.sync_status import (
    SyncStatus,
    SyncTrainingStatus,
    TrainingTargetStatus,
)
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import (
    DeploymentReplicaDB,
)
from train_factory.storage.entities.external_api_config_entity import (
    ExternalApiConfigDB,
)
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.generation_task_entity import GenerationTaskDB
from train_factory.storage.entities.milvus_collection_entity import (
    MilvusCollectionDB,
)
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_registry_entity import (
    ModelRegistryDB,
    ModelVersionDB,
)
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.services.external_sync_service import ExternalSyncService
from train_factory.storage.services.external_api_config_service import (
    ExternalApiConfigService,
)
from train_factory.storage.services.model_registry_service import ModelRegistryService
from train_factory.storage.services.model_config_service import ModelConfigService
from train_factory.storage.services.generation_task_service import (
    GenerationTaskService,
)
from train_factory.storage.services.model_artifact_membership_service import (
    lock_model_artifact_membership,
)

deployment_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)
milvus_module = importlib.import_module(
    "train_factory.storage.services.milvus_collection_service"
)
registry_module = importlib.import_module(
    "train_factory.storage.services.model_registry_service"
)
generation_module = importlib.import_module(
    "train_factory.storage.services.generation_task_service"
)


for service_module in (
    deployment_module,
    milvus_module,
    registry_module,
    generation_module,
):
    assert isinstance(service_module, ModuleType)
    assert hasattr(service_module, "get_session")


pytestmark = pytest.mark.skipif(
    "TRAINFACTORY_TEST_MYSQL_URL" not in os.environ,
    reason="requires the isolated MySQL runner",
)


def _runner_base_url():
    run_id = os.environ.get("TRAINFACTORY_MYSQL_RUN_ID", "")
    expected_uuid = os.environ.get("TRAINFACTORY_MYSQL_SERVER_UUID", "")
    try:
        parsed = make_url(os.environ.get("TRAINFACTORY_TEST_MYSQL_URL", ""))
        valid = (
            re.fullmatch(r"[0-9a-f]{32}", run_id) is not None
            and parsed.drivername == "mysql+pymysql"
            and parsed.username == "root"
            and bool(parsed.password)
            and parsed.host == "127.0.0.1"
            and parsed.port is not None
            and 1024 <= parsed.port <= 65535
            and parsed.port != 3306
            and parsed.database == f"tf_runner_{run_id}"
            and not parsed.query
            and re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                expected_uuid,
            )
            is not None
        )
    except Exception:
        valid = False
    if not valid:
        raise RuntimeError("Isolated MySQL runner environment is invalid") from None
    return parsed, run_id, expected_uuid


def _quote_database(name: str) -> str:
    assert re.fullmatch(r"tf_sync_lock_[0-9a-f]{32}_[0-9a-f]{16}", name)
    return f"`{name}`"


@pytest.fixture
def mysql_sync_engine() -> Iterator[sa.Engine]:
    base_url, run_id, expected_uuid = _runner_base_url()
    database = f"tf_sync_lock_{run_id}_{secrets.token_hex(8)}"
    admin = sa.create_engine(base_url, pool_pre_ping=True)
    created = False
    engine = None
    try:
        with admin.begin() as connection:
            actual_uuid, actual_database = connection.execute(
                sa.text("SELECT @@server_uuid, DATABASE()")
            ).one()
            if actual_uuid != expected_uuid or actual_database != f"tf_runner_{run_id}":
                raise RuntimeError("Isolated MySQL server identity mismatch")
            connection.execute(
                sa.text(
                    f"CREATE DATABASE {_quote_database(database)} "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            created = True

        case_url = base_url.set(database=database).render_as_string(
            hide_password=False
        )
        engine = sa.create_engine(case_url, pool_pre_ping=True)

        @sa.event.listens_for(engine, "connect")
        def set_bounded_lock_wait(dbapi_connection, _connection_record):
            with dbapi_connection.cursor() as cursor:
                cursor.execute("SET SESSION innodb_lock_wait_timeout = 5")

        with engine.begin() as connection:
            connection.exec_driver_sql("SET FOREIGN_KEY_CHECKS = 0")
            for table in (
                ExternalApiConfigDB.__table__,
                ModelArtifactMembershipGateDB.__table__,
                ModelRegistryDB.__table__,
                ModelVersionDB.__table__,
                DeploymentDB.__table__,
                DeploymentReplicaDB.__table__,
                LoadedAdapterDB.__table__,
                ModelConfigDB.__table__,
                GenerationTaskDB.__table__,
                EvaluationTaskDB.__table__,
                TrainingTaskDB.__table__,
                ExternalSyncTaskDB.__table__,
                ExternalSyncTrainingTargetDB.__table__,
                ExternalSyncTrainingDB.__table__,
                MilvusCollectionDB.__table__,
            ):
                table.create(connection)
            connection.exec_driver_sql("SET FOREIGN_KEY_CHECKS = 1")
        with Session(engine) as session:
            session.add(ModelArtifactMembershipGateDB(gate_id=1))
            session.commit()
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        if created:
            with admin.begin() as connection:
                connection.execute(
                    sa.text(f"DROP DATABASE {_quote_database(database)}")
                )
        admin.dispose()


@pytest.mark.parametrize("operation", ["complete", "fail"])
def test_binding_update_and_claim_reconciliation_do_not_deadlock_on_mysql(
    mysql_sync_engine: sa.Engine,
    operation: str,
):
    service = ExternalSyncService()
    service.engine = mysql_sync_engine

    for index in range(5):
        suffix = secrets.token_hex(8)
        task_id = f"s{index}{suffix}"
        target_id = f"g{index}{suffix}"
        training_id = f"r{index}{suffix}"
        with Session(mysql_sync_engine) as session:
            session.add(
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="mysql lock order",
                    user_id="user-1",
                    status=SyncStatus.TRAINING,
                    current_training_id=training_id,
                )
            )
            session.add(
                ExternalSyncTrainingTargetDB(
                    target_id=target_id,
                    task_id=task_id,
                    target_name="mysql lock order",
                    status=TrainingTargetStatus.TRAINING,
                    current_training_id=training_id,
                )
            )
            session.add(
                ExternalSyncTrainingDB(
                    task_id=task_id,
                    training_task_id=training_id,
                    user_id="user-1",
                    target_id=target_id,
                    status=SyncTrainingStatus.PENDING,
                )
            )
            session.add(
                ExternalSyncTrainingDB(
                    task_id=task_id,
                    training_task_id=f"a{index}{suffix}",
                    user_id="user-1",
                    status=SyncTrainingStatus.ADAPTER_LOADED,
                )
            )
            session.commit()

        start = threading.Barrier(2)

        def update_binding():
            start.wait(timeout=5)
            with pytest.raises(ValueError):
                service.update_training_target(
                    target_id,
                    base_deployment_id="missing-deployment",
                )

        def reconcile_claim():
            start.wait(timeout=5)
            if operation == "complete":
                return service.complete_training_claim(training_id)
            return service.fail_training_and_restore_claim(
                training_id,
                "mysql concurrency probe",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            update_future = executor.submit(update_binding)
            reconcile_future = executor.submit(reconcile_claim)
            update_future.result(timeout=15)
            result = reconcile_future.result(timeout=15)

        assert result["completed" if operation == "complete" else "recovered"]


def test_cross_config_and_binding_updates_lock_one_sorted_dependency_union(
    mysql_sync_engine: sa.Engine,
):
    service = ExternalSyncService()
    service.engine = mysql_sync_engine

    with Session(mysql_sync_engine) as session:
        for suffix in ("a", "b"):
            port = 11000 if suffix == "a" else 12000
            session.add(
                ModelRegistryDB(
                    model_id=f"model-{suffix}",
                    model_name=f"model-{suffix}",
                    model_type="embedding",
                    model_path=f"/models/model-{suffix}",
                    status="available",
                    user_id="user-1",
                )
            )
            session.add(
                DeploymentDB(
                    deployment_id=f"deployment-{suffix}",
                    model_id=f"model-{suffix}",
                    deployment_name=f"deployment-{suffix}",
                    xinference_endpoint=f"http://127.0.0.1:{port}",
                    user_id="user-1",
                )
            )
            session.add(
                ModelConfigDB(
                    config_id=f"config-{suffix}",
                    config_name=f"config-{suffix}",
                    source_type="local_deployed",
                    registry_id=f"model-{suffix}",
                    deployment_id=f"deployment-{suffix}",
                    model_type="embedding",
                    provider="openai-compatible",
                    api_endpoint=f"http://127.0.0.1:{port}/v1",
                    model_name=f"model-{suffix}",
                    user_id="user-1",
                )
            )
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="cross-task-a",
                    task_name="cross-task-a",
                    user_id="user-1",
                    is_active=False,
                    status=SyncStatus.IDLE,
                ),
                ExternalSyncTaskDB(
                    task_id="cross-task-b",
                    task_name="cross-task-b",
                    user_id="user-1",
                    is_active=False,
                    status=SyncStatus.IDLE,
                ),
            ]
        )
        session.commit()

    start = threading.Barrier(2)

    def activate(task_suffix: str, config_suffix: str, binding_suffix: str):
        start.wait(timeout=5)
        return service.update_task(
            f"cross-task-{task_suffix}",
            expected_user_id="user-1",
            is_active=True,
            generation_config={
                "embedding_config": {
                    "config_id": f"config-{config_suffix}"
                }
            },
            base_deployment_id=f"deployment-{binding_suffix}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(activate, "a", "a", "b")
        second = executor.submit(activate, "b", "b", "a")
        results = [first.result(timeout=15), second.result(timeout=15)]

    assert all(result is not None and result["is_active"] for result in results)
    assert {
        result["task_id"]: result["base_deployment_id"] for result in results
    } == {
        "cross-task-a": "deployment-b",
        "cross-task-b": "deployment-a",
    }


def test_sync_create_and_external_api_delete_have_one_atomic_winner(
    mysql_sync_engine: sa.Engine,
):
    sync_service = ExternalSyncService()
    sync_service.engine = mysql_sync_engine
    api_service = ExternalApiConfigService()
    api_service.engine = mysql_sync_engine
    created_config = api_service.create_config(
        config_name="api-delete-race",
        user_id="user-1",
        api_url="https://example.invalid/api",
        auth_config={},
    )
    config_id = created_config["config_id"]

    start = threading.Barrier(2)

    def create_sync():
        start.wait(timeout=5)
        try:
            task = sync_service.create_task(
                task_name="api-delete-race",
                user_id="user-1",
                external_api_config_id=config_id,
                is_active=False,
            )
            return "created", task["task_id"]
        except ValueError as exc:
            return "rejected", str(exc)

    def delete_api():
        start.wait(timeout=5)
        try:
            return "deleted", api_service.delete_config(config_id)
        except ValueError as exc:
            return "rejected", str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        create_future = executor.submit(create_sync)
        delete_future = executor.submit(delete_api)
        create_result = create_future.result(timeout=15)
        delete_result = delete_future.result(timeout=15)

    assert (create_result[0], delete_result[0]) in {
        ("created", "rejected"),
        ("rejected", "deleted"),
    }
    with Session(mysql_sync_engine) as session:
        config_exists = session.exec(
            sa.select(ExternalApiConfigDB.id).where(
                ExternalApiConfigDB.config_id == config_id
            )
        ).first()
        tasks = list(
            session.exec(
                sa.select(ExternalSyncTaskDB).where(
                    ExternalSyncTaskDB.external_api_config_id == config_id
                )
            ).all()
        )
    assert not tasks or config_exists is not None


def test_sync_create_and_external_api_url_update_do_not_deadlock(
    mysql_sync_engine: sa.Engine,
):
    sync_service = ExternalSyncService()
    sync_service.engine = mysql_sync_engine
    api_service = ExternalApiConfigService()
    api_service.engine = mysql_sync_engine
    created_config = api_service.create_config(
        config_name="api-update-race",
        user_id="user-1",
        api_url="https://before.example.invalid/api",
        auth_config={},
    )
    config_id = created_config["config_id"]
    start = threading.Barrier(2)

    def create_sync():
        start.wait(timeout=5)
        return sync_service.create_task(
            task_name="api-update-race",
            user_id="user-1",
            external_api_config_id=config_id,
            is_active=False,
        )

    def update_api():
        start.wait(timeout=5)
        try:
            updated = api_service.update_config(
                config_id,
                api_url="https://after.example.invalid/api",
            )
            return "updated", updated["api_url"]
        except ValueError as exc:
            return "rejected", str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        create_future = executor.submit(create_sync)
        update_future = executor.submit(update_api)
        task = create_future.result(timeout=15)
        update_result = update_future.result(timeout=15)

    assert task["external_api_config_id"] == config_id
    assert update_result[0] in {"updated", "rejected"}
    persisted = api_service.get_config(config_id)
    assert persisted is not None
    if update_result[0] == "updated":
        assert persisted["api_url"] == "https://after.example.invalid/api"


def test_model_delete_and_external_api_delete_lock_deployments_in_one_order(
    monkeypatch: pytest.MonkeyPatch,
    mysql_sync_engine: sa.Engine,
):
    api_service = ExternalApiConfigService()
    api_service.engine = mysql_sync_engine
    api_config = api_service.create_config(
        config_name="model-delete-api-race",
        user_id="user-1",
        api_url="https://model-delete.invalid/api",
        auth_config={},
    )
    config_id = api_config["config_id"]
    model_id = "model-delete-api-race"

    with Session(mysql_sync_engine) as session:
        session.add(
            ModelRegistryDB(
                model_id=model_id,
                model_name=model_id,
                model_type="llm",
                model_path="/external/model-delete-api-race",
                source_type="trained",
                status="available",
                user_id="user-1",
            )
        )
        session.commit()
    for suffix in ("2", "1"):
        with Session(mysql_sync_engine) as session:
            session.add(
                DeploymentDB(
                    deployment_id=f"deployment-{suffix}",
                    model_id=model_id,
                    deployment_name=f"deployment-{suffix}",
                    xinference_endpoint=f"http://deployment-{suffix}:11000",
                    deploy_mode="container",
                    container_name=f"trainfactory-vllm-race-{suffix}",
                    inference_framework="vllm",
                    external_api_config_id=config_id,
                    config={
                        "replica_schema_version": 1,
                        "external_api_config_id": config_id,
                    },
                    status="stopped",
                    user_id="user-1",
                )
            )
            session.commit()

    @contextmanager
    def get_session():
        with Session(mysql_sync_engine) as session:
            yield session

    monkeypatch.setattr(registry_module, "get_session", get_session)
    monkeypatch.setattr(deployment_module, "get_session", get_session)
    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)
    registry = ModelRegistryService()
    start = threading.Barrier(2)

    def delete_model():
        start.wait(timeout=5)
        try:
            return "deleted", registry.delete_model(model_id, force=True)
        except ValueError as exc:
            return "rejected", str(exc)

    def delete_api():
        start.wait(timeout=5)
        try:
            return "deleted", api_service.delete_config(config_id)
        except ValueError as exc:
            return "rejected", str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        model_future = executor.submit(delete_model)
        api_future = executor.submit(delete_api)
        model_result = model_future.result(timeout=20)
        api_result = api_future.result(timeout=20)

    assert model_result == ("deleted", True)
    assert api_result[0] in {"deleted", "rejected"}
    with Session(mysql_sync_engine) as session:
        model_exists = session.exec(
            sa.select(ModelRegistryDB.id).where(
                ModelRegistryDB.model_id == model_id
            )
        ).first()
        deployments = list(
            session.exec(
                sa.select(DeploymentDB).where(
                    DeploymentDB.model_id == model_id
                )
            ).all()
        )
        config_exists = session.exec(
            sa.select(ExternalApiConfigDB.id).where(
                ExternalApiConfigDB.config_id == config_id
            )
        ).first()
    assert model_exists is None
    assert deployments == []
    assert (config_exists is None) is (api_result[0] == "deleted")


def test_model_config_delete_and_milvus_register_have_one_atomic_winner(
    monkeypatch: pytest.MonkeyPatch,
    mysql_sync_engine: sa.Engine,
):
    config_service = ModelConfigService()
    config_service.engine = mysql_sync_engine

    @contextmanager
    def mysql_session():
        with Session(mysql_sync_engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(milvus_module, "get_session", mysql_session)
    collection_service = milvus_module.MilvusCollectionService()
    config_id = f"config-{secrets.token_hex(8)}"
    collection_name = f"collection-{secrets.token_hex(8)}"
    with Session(mysql_sync_engine) as session:
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

    start = threading.Barrier(2)

    def register_collection():
        start.wait(timeout=5)
        try:
            result = collection_service.register_collection(
                collection_name,
                embedding_config_id=config_id,
                embedding_model="embedding-model",
                embedding_endpoint="https://embedding.example.invalid/v1",
                user_id="user-1",
            )
            return "registered", result["collection_id"]
        except milvus_module.MilvusCollectionUnavailableError as exc:
            return "rejected", str(exc)

    def delete_config():
        start.wait(timeout=5)
        try:
            return "deleted", config_service.delete_config(config_id)
        except ValueError as exc:
            return "rejected", str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        register_future = executor.submit(register_collection)
        delete_future = executor.submit(delete_config)
        register_result = register_future.result(timeout=15)
        delete_result = delete_future.result(timeout=15)

    assert (register_result[0], delete_result[0]) in {
        ("registered", "rejected"),
        ("rejected", "deleted"),
    }
    with Session(mysql_sync_engine) as session:
        config_exists = session.exec(
            sa.select(ModelConfigDB.id).where(
                ModelConfigDB.config_id == config_id
            )
        ).first()
        collections = list(
            session.exec(
                sa.select(MilvusCollectionDB).where(
                    MilvusCollectionDB.embedding_config_id == config_id
                )
            ).all()
        )
    assert not collections or config_exists is not None


def test_model_delete_and_generation_create_have_one_atomic_winner(
    monkeypatch: pytest.MonkeyPatch,
    mysql_sync_engine: sa.Engine,
):
    model_id = f"runtime-race-{secrets.token_hex(8)}"
    config_id = f"runtime-config-{secrets.token_hex(8)}"
    generation_task_id = f"generation-{secrets.token_hex(8)}"
    with Session(mysql_sync_engine) as session:
        session.add(
            ModelRegistryDB(
                model_id=model_id,
                model_name=model_id,
                model_type="embedding",
                model_path=f"/external/{model_id}",
                source_type="trained",
                status="available",
                user_id="user-1",
            )
        )
        session.add(
            ModelConfigDB(
                config_id=config_id,
                config_name=config_id,
                source_type="local_deployed",
                registry_id=model_id,
                model_type="embedding",
                provider="openai-compatible",
                api_endpoint="https://embedding.example.invalid/v1",
                model_name=model_id,
                user_id="user-1",
            )
        )
        session.commit()

    @contextmanager
    def mysql_session():
        with Session(mysql_sync_engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    cleanup_calls: list[str] = []
    monkeypatch.setattr(registry_module, "get_session", mysql_session)
    monkeypatch.setattr(generation_module, "get_session", mysql_session)
    monkeypatch.setattr(deployment_module, "get_session", mysql_session)
    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: cleanup_calls.append("runtime"),
    )
    monkeypatch.setattr(
        registry_module.shutil,
        "rmtree",
        lambda _path: cleanup_calls.append("file"),
    )
    registry = ModelRegistryService()
    generation = GenerationTaskService()
    start = threading.Barrier(2)

    def delete_model():
        start.wait(timeout=5)
        try:
            return "deleted", registry.delete_model(model_id, force=True)
        except ValueError as exc:
            return "rejected", str(exc)

    def create_generation():
        start.wait(timeout=5)
        try:
            result = generation.create_task(
                task_id=generation_task_id,
                task_name=generation_task_id,
                input_path="/external/input.jsonl",
                llm_config={},
                steps_config={},
                embedding_config_id=config_id,
                user_id="user-1",
            )
            return "created", result["task_id"]
        except generation_module.RuntimeDependencyUnavailableError as exc:
            return "rejected", str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        delete_future = executor.submit(delete_model)
        generation_future = executor.submit(create_generation)
        delete_result = delete_future.result(timeout=20)
        generation_result = generation_future.result(timeout=20)

    assert (delete_result[0], generation_result[0]) in {
        ("deleted", "rejected"),
        ("rejected", "created"),
    }
    assert cleanup_calls == []

    with Session(mysql_sync_engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == model_id
            )
        ).first()
        config = session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == config_id
            )
        ).first()
        tasks = list(
            session.exec(
                select(GenerationTaskDB).where(
                    GenerationTaskDB.task_id == generation_task_id
                )
            ).all()
        )
    if delete_result[0] == "deleted":
        assert model is None
        assert config is None
        assert tasks == []
    else:
        assert model is not None
        assert model.status == "available"
        assert config is not None
        assert len(tasks) == 1


def test_model_artifact_membership_gate_serializes_mysql_transactions(
    mysql_sync_engine: sa.Engine,
):
    """Prove the 058 singleton is a real MySQL row fence, not only SQL text."""
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_sql_started = threading.Event()
    second_acquired = threading.Event()
    second_thread_id: dict[str, int] = {}

    def record_gate_update(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        if (
            threading.get_ident() == second_thread_id.get("value")
            and " ".join(statement.lower().split()).startswith(
                "update model_artifact_membership_gate"
            )
        ):
            second_sql_started.set()

    def hold_first_gate():
        with Session(mysql_sync_engine) as session:
            lock_model_artifact_membership(session)
            first_acquired.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("membership gate release was not signalled")
            session.commit()

    def acquire_second_gate():
        if not first_acquired.wait(timeout=5):
            raise TimeoutError("first membership gate was not acquired")
        second_thread_id["value"] = threading.get_ident()
        with Session(mysql_sync_engine) as session:
            lock_model_artifact_membership(session)
            second_acquired.set()
            session.commit()

    sa.event.listen(
        mysql_sync_engine,
        "before_cursor_execute",
        record_gate_update,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(hold_first_gate)
            assert first_acquired.wait(timeout=5)
            second_future = executor.submit(acquire_second_gate)
            assert second_sql_started.wait(timeout=5)
            assert not second_acquired.wait(timeout=0.25)
            release_first.set()
            first_future.result(timeout=10)
            second_future.result(timeout=10)
        assert second_acquired.is_set()
    finally:
        release_first.set()
        sa.event.remove(
            mysql_sync_engine,
            "before_cursor_execute",
            record_gate_update,
        )
