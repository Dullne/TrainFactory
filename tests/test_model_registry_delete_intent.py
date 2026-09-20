from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlmodel import Session, select

from train_factory.deployment.replica_planner import DeploymentPlan, ReplicaPlan
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_registry_entity import (
    ModelRegistryDB,
    ModelVersionDB,
)
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.entities.generation_task_entity import GenerationTaskDB
from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.milvus_collection_entity import (
    MilvusCollectionDB,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)


adapter_module = importlib.import_module("train_factory.deployment.adapter_service")
admission_module = importlib.import_module(
    "train_factory.storage.services.background_task_admission_service"
)
deployment_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)
config_module = importlib.import_module(
    "train_factory.storage.services.model_config_service"
)
download_module = importlib.import_module(
    "train_factory.storage.services.model_download_service"
)
registry_module = importlib.import_module(
    "train_factory.storage.services.model_registry_service"
)
training_module = importlib.import_module(
    "train_factory.storage.services.training_task_service"
)
training_event_module = importlib.import_module(
    "train_factory.storage.services.training_task_event_service"
)


DELETE_INTENT_KEY = "_train_factory_model_delete_intent_v1"


@pytest.fixture
def registry_dependency_db(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'registry-intent.db'}")
    for table in (
        ModelRegistryDB.__table__,
        ModelArtifactMembershipGateDB.__table__,
        ModelVersionDB.__table__,
        DeploymentDB.__table__,
        DeploymentReplicaDB.__table__,
        LoadedAdapterDB.__table__,
        ModelConfigDB.__table__,
        TrainingTaskDB.__table__,
        GenerationTaskDB.__table__,
        EvaluationTaskDB.__table__,
        ExternalSyncTaskDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
        ExternalSyncTrainingDB.__table__,
        MilvusCollectionDB.__table__,
    ):
        table.create(engine)
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(registry_module, "get_session", get_session)
    monkeypatch.setattr(deployment_module, "get_session", get_session)
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(download_module, "get_session", get_session)
    monkeypatch.setattr(training_module, "get_session", get_session)
    monkeypatch.setattr(
        training_event_module.training_task_event_service,
        "log_event",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(registry_module.settings, "models_dir", tmp_path / "models")

    config_service = config_module.ModelConfigService()
    config_service.engine = engine

    def add_model(
        model_id: str,
        *,
        status: str = "available",
        metadata: dict | None = None,
        model_name: str | None = None,
        source_type: str = "trained",
    ) -> ModelRegistryDB:
        model_path = tmp_path / "models" / model_id
        model_path.mkdir(parents=True, exist_ok=True)
        model = ModelRegistryDB(
            model_id=model_id,
            model_name=model_name or model_id,
            model_type="llm",
            model_path=str(model_path),
            source_type=source_type,
            status=status,
            extra_metadata=metadata,
        )
        with Session(engine) as session:
            session.add(model)
            session.commit()
            session.refresh(model)
        return model

    return engine, config_service, add_model


def _deleting_metadata(
    token: str = "private-delete-owner",
    *,
    heartbeat_at: str = "2999-01-01T00:00:00",
) -> dict:
    original_metadata = {"user_metadata": {"preserve": True}}
    return {
        **original_metadata,
        DELETE_INTENT_KEY: {
            "token": token,
            "started_at": heartbeat_at,
            "heartbeat_at": heartbeat_at,
            "original_status": "available",
            "original_extra_metadata": original_metadata,
            "had_extra_metadata": True,
        },
    }


def test_model_delete_multrow_lock_queries_use_sql_stable_order(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model("ordered-delete")
    with Session(engine) as session:
        for suffix in ("z", "a"):
            deployment_id = f"deployment-{suffix}"
            container_name = f"trainfactory-vllm-ordered-{suffix}"
            port = 11000 if suffix == "z" else 11001
            session.add(
                DeploymentDB(
                    deployment_id=deployment_id,
                    model_id=model.model_id,
                    deployment_name=deployment_id,
                    xinference_endpoint=f"http://{container_name}:{port}",
                    deploy_mode="container",
                    container_name=container_name,
                    inference_framework="vllm",
                    config={"replica_schema_version": 1},
                    status="stopped",
                    user_id="user-1",
                )
            )
            session.add(
                DeploymentReplicaDB(
                    replica_id=f"replica-{suffix}",
                    deployment_id=deployment_id,
                    replica_index=0,
                    container_name=container_name,
                    endpoint=f"http://{container_name}:{port}",
                    port=port,
                    gpu_ids=[0],
                    status="stopped",
                )
            )
            session.add(
                ModelConfigDB(
                    config_id=f"config-{suffix}",
                    config_name=f"config-{suffix}",
                    source_type="local_deployed",
                    registry_id=model.model_id,
                    deployment_id=deployment_id,
                    model_type="llm",
                    provider="openai-compatible",
                    api_endpoint=f"http://{container_name}:{port}/v1",
                    model_name="ordered-delete",
                )
            )
            session.add(
                LoadedAdapterDB(
                    adapter_id=f"adapter-{suffix}",
                    deployment_id=deployment_id,
                    adapter_name=f"adapter-{suffix}",
                    adapter_path=f"/external/adapter-{suffix}",
                    status="unloaded",
                    user_id="user-1",
                )
            )
            session.add(
                ModelVersionDB(
                    version_id=f"version-{suffix}",
                    model_id=model.model_id,
                    version=f"v-{suffix}",
                    model_path=f"{model.model_path}/version-{suffix}",
                )
            )
            session.add(
                TrainingTaskDB(
                    task_id=f"training-{suffix}",
                    task_name=f"training-{suffix}",
                    base_model_path=f"/external/training-{suffix}",
                    status="pending",
                    user_id="user-1",
                )
            )
        session.commit()

    statements: list[str] = []

    def record(_connection, _cursor, statement, _params, _context, _many):
        statements.append(" ".join(statement.lower().split()))

    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)
    sa.event.listen(engine, "before_cursor_execute", record)
    try:
        assert registry_module.ModelRegistryService().delete_model(
            model.model_id,
            force=True,
        ) is True
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)

    expected_orders = (
        (
            "deployments",
            "where deployments.model_id =",
            "order by deployments.deployment_id",
        ),
        (
            "model_configs",
            "where model_configs.registry_id =",
            "order by model_configs.config_id",
        ),
        (
            "model_versions",
            "where model_versions.model_id =",
            "order by model_versions.version_id",
        ),
        (
            "loaded_adapters",
            "where loaded_adapters.deployment_id in",
            "order by loaded_adapters.adapter_id",
        ),
        (
            "loaded_adapters",
            "where loaded_adapters.status in",
            "order by loaded_adapters.adapter_id",
        ),
            (
                "deployment_replicas",
                "where deployment_replicas.deployment_id in",
                "order by deployment_replicas.replica_id",
            ),
        (
            "training_tasks",
            "where training_tasks.status in",
            "order by training_tasks.task_id",
        ),
    )
    for table_name, where_fragment, order_fragment in expected_orders:
        dependency_queries = [
            statement
            for statement in statements
            if statement.startswith("select")
            and " from " in statement
            and statement.split(" from ", 1)[1].startswith(f"{table_name} ")
            and where_fragment in statement
        ]
        assert dependency_queries, (table_name, where_fragment)
        assert all(
            order_fragment in statement for statement in dependency_queries
        ), dependency_queries


def test_delete_intent_is_hidden_and_cannot_be_overwritten_by_model_update(
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model(
        "model-deleting",
        status="deleting",
        metadata=_deleting_metadata(),
    )
    registry = registry_module.ModelRegistryService()

    assert registry._model_to_dict(model)["extra_metadata"] == {
        "user_metadata": {"preserve": True}
    }
    assert model.to_dict()["extra_metadata"] == {
        "user_metadata": {"preserve": True}
    }

    with pytest.raises(ValueError, match="being deleted"):
        registry.update_model(
            "model-deleting",
            status="archived",
            extra_metadata={"user_metadata": {"changed": True}},
        )

    with Session(engine) as session:
        current = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == "model-deleting"
            )
        ).one()
        assert current.status == "deleting"
        assert current.extra_metadata == _deleting_metadata()


def test_reserved_delete_intent_metadata_cannot_be_registered(
    registry_dependency_db,
) -> None:
    _engine, _config_service, _add_model = registry_dependency_db

    with pytest.raises(ValueError, match="reserved"):
        registry_module.ModelRegistryService().register_model(
            model_name="forged-intent",
            model_path="/not/materialized/forged-intent",
            model_type="llm",
            extra_metadata=_deleting_metadata("caller-forged"),
        )


def test_add_version_rejects_deleting_registry_model_in_insert_transaction(
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("model-deleting", status="deleting", metadata=_deleting_metadata())

    with pytest.raises(ValueError, match="being deleted"):
        registry_module.ModelRegistryService().add_version(
            "model-deleting",
            "v2",
            "/models/model-deleting-v2",
        )

    with Session(engine) as session:
        assert session.exec(select(ModelVersionDB)).all() == []


@pytest.mark.parametrize("operation", ["register", "add_version"])
def test_registry_membership_writes_lock_singleton_gate_first(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    operation: str,
) -> None:
    _engine, _config_service, add_model = registry_dependency_db
    calls: list[str] = []
    original_lock = registry_module.lock_model_artifact_membership

    def observe_lock(session) -> None:
        calls.append("gate")
        original_lock(session)

    monkeypatch.setattr(
        registry_module,
        "lock_model_artifact_membership",
        observe_lock,
    )
    registry = registry_module.ModelRegistryService()
    if operation == "register":
        registry.register_model(
            model_name="gate-register",
            model_path="/external/gate-register",
            model_type="llm",
        )
    else:
        model = add_model("gate-add-version")
        registry.add_version(
            model.model_id,
            version="v2",
            model_path=model.model_path + "/v2",
        )

    assert calls == ["gate"]


@pytest.mark.parametrize("operation", ["register", "add_version"])
def test_registry_write_locks_family_and_path_owners_in_one_sorted_union(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    operation: str,
) -> None:
    engine, _config_service, _add_model = registry_dependency_db
    with Session(engine) as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id="aaa-path-owner",
                    model_name="path-owner",
                    model_type="llm",
                    model_path="/managed/shared-root",
                    user_id="tenant-a",
                    is_latest=True,
                ),
                ModelRegistryDB(
                    model_id="zzz-family-target",
                    model_name="shared-family",
                    model_type="llm",
                    version="v0",
                    model_path="/external/family-target",
                    user_id="tenant-a",
                    is_latest=True,
                ),
            ]
        )
        session.commit()

    locked_model_id_sets: list[tuple[str, ...]] = []

    class RecordingResult:
        def __init__(self, result) -> None:
            self._result = result

        def all(self):
            rows = self._result.all()
            locked_model_id_sets.append(
                tuple(row.model_id for row in rows)
            )
            return rows

        def __getattr__(self, name):
            return getattr(self._result, name)

    @contextmanager
    def recording_get_session():
        with Session(engine) as session:
            original_exec = session.exec

            def recording_exec(statement, *args, **kwargs):
                result = original_exec(statement, *args, **kwargs)
                get_final_froms = getattr(statement, "get_final_froms", None)
                from_tables = get_final_froms() if get_final_froms else ()
                if (
                    getattr(statement, "_for_update_arg", None) is not None
                    and from_tables
                    and from_tables[0].name == "model_registry"
                ):
                    return RecordingResult(result)
                return result

            session.exec = recording_exec
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(registry_module, "get_session", recording_get_session)
    registry = registry_module.ModelRegistryService()
    prospective_path = "/managed/shared-root/new-version"
    if operation == "register":
        registry.register_model(
            model_name="shared-family",
            model_path=prospective_path,
            model_type="llm",
            user_id="tenant-a",
        )
    else:
        registry.add_version(
            "zzz-family-target",
            version="v2",
            model_path=prospective_path,
        )

    assert locked_model_id_sets == [
        ("aaa-path-owner", "zzz-family-target")
    ]


def test_anonymous_register_does_not_clear_another_tenants_latest_model(
    registry_dependency_db,
) -> None:
    engine, _config_service, _add_model = registry_dependency_db
    with Session(engine) as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id="anonymous-existing",
                    model_name="shared-name",
                    model_type="llm",
                    model_path="/external/anonymous-existing",
                    user_id=None,
                    is_latest=True,
                ),
                ModelRegistryDB(
                    model_id="tenant-existing",
                    model_name="shared-name",
                    model_type="llm",
                    model_path="/external/tenant-existing",
                    user_id="user-a",
                    is_latest=True,
                ),
            ]
        )
        session.commit()

    registry_module.ModelRegistryService().register_model(
        model_name="shared-name",
        model_path="/external/anonymous-new",
        model_type="llm",
        user_id=None,
    )

    with Session(engine) as session:
        tenant = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == "tenant-existing"
            )
        ).one()
        anonymous = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == "anonymous-existing"
            )
        ).one()
        assert tenant.is_latest is True
        assert anonymous.is_latest is False


@pytest.mark.parametrize("operation", ["register", "add_version"])
def test_registry_write_rejects_prospective_path_under_deleting_root(
    registry_dependency_db,
    operation: str,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    deleting = add_model(
        "overlap-deleting",
        status="deleting",
        metadata=_deleting_metadata(),
    )
    registry = registry_module.ModelRegistryService()
    prospective_path = deleting.model_path + "/new-child"
    if operation == "register":
        with pytest.raises(ValueError, match="being deleted"):
            registry.register_model(
                model_name="overlap-new",
                model_path=prospective_path,
                model_type="llm",
            )
    else:
        target = add_model("overlap-version-target")
        with pytest.raises(ValueError, match="being deleted"):
            registry.add_version(
                target.model_id,
                version="v2",
                model_path=prospective_path,
            )

    with Session(engine) as session:
        assert session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_name == "overlap-new"
            )
        ).all() == []
        assert session.exec(
            select(ModelVersionDB).where(ModelVersionDB.version == "v2")
        ).all() == []


@pytest.mark.parametrize("reference_kind", ["registry", "version"])
@pytest.mark.parametrize("relationship", ["descendant", "ancestor"])
def test_delete_rejects_other_registry_artifact_overlap_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    reference_kind: str,
    relationship: str,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    target = add_model("overlap-delete-target", source_type="downloaded")
    target_path = Path(target.model_path)
    overlapping_path = (
        target_path / "nested"
        if relationship == "descendant"
        else target_path.parent
    )
    other = add_model("overlap-other")
    with Session(engine) as session:
        current_other = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == other.model_id
            )
        ).one()
        if reference_kind == "registry":
            current_other.model_path = str(overlapping_path)
            session.add(current_other)
        else:
            session.add(
                ModelVersionDB(
                    model_id=other.model_id,
                    version="overlap-history",
                    model_path=str(overlapping_path),
                )
            )
        session.commit()
    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        registry_module.shutil,
        "rmtree",
        lambda _path: cleanup_calls.append("file"),
    )

    with pytest.raises(ValueError, match="overlap"):
        registry_module.ModelRegistryService().delete_model(
            target.model_id,
            force=True,
        )

    assert cleanup_calls == []
    with Session(engine) as session:
        current = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == target.model_id
            )
        ).one()
        assert current.status == "available"
        assert DELETE_INTENT_KEY not in (current.extra_metadata or {})


def test_external_bound_models_sharing_endpoint_delete_independently(
    registry_dependency_db,
) -> None:
    engine, _config_service, _add_model = registry_dependency_db
    with Session(engine) as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id="external-a",
                    model_name="external-a",
                    model_type="llm",
                    model_path="https://inference.example.test/v1",
                    source_type="external_bind",
                    status="available",
                ),
                ModelRegistryDB(
                    model_id="external-b",
                    model_name="external-b",
                    model_type="llm",
                    model_path="https://inference.example.test/v1",
                    source_type="external_bind",
                    status="available",
                ),
            ]
        )
        session.commit()

    registry = registry_module.ModelRegistryService()
    assert registry.delete_model("external-a", force=True) is True
    assert registry.delete_model("external-b", force=True) is True
    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).all() == []


def test_download_record_mutations_cannot_clear_or_delete_model_delete_intent(
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model(
        "download-deleting",
        status="deleting",
        metadata=_deleting_metadata(),
        source_type="downloaded",
    )
    with Session(engine) as session:
        model = session.exec(select(ModelRegistryDB)).one()
        model.download_status = "pending"
        model.download_progress = 10
        session.add(model)
        session.commit()

    download_module.ModelDownloadService._delete_pending_record(
        "download-deleting"
    )
    with pytest.raises(ValueError, match="being deleted"):
        download_module.ModelDownloadService()._update_model_record(
            "download-deleting",
            status="available",
            download_status="completed",
            download_progress=100,
            file_size=123,
        )

    with Session(engine) as session:
        current = session.exec(select(ModelRegistryDB)).one()
        assert current.status == "deleting"
        assert current.download_status == "pending"
        assert current.download_progress == 10
        assert current.file_size is None
        assert current.extra_metadata == _deleting_metadata()


def test_model_download_registry_insert_locks_membership_gate(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    _engine, _config_service, _add_model = registry_dependency_db
    calls: list[str] = []
    original_lock = registry_module.lock_model_artifact_membership

    class Lease:
        def release(self) -> None:
            pass

    class Reservation:
        def release(self) -> None:
            pass

    class Thread:
        def __init__(self, **_kwargs):
            pass

        def start(self) -> None:
            pass

    def observe_lock(session) -> None:
        calls.append("gate")
        original_lock(session)

    monkeypatch.setattr(
        download_module,
        "lock_model_artifact_membership",
        observe_lock,
        raising=False,
    )
    monkeypatch.setattr(
        download_module.download_concurrency_limiter,
        "acquire",
        lambda *_args, **_kwargs: Lease(),
    )
    monkeypatch.setattr(
        download_module.download_storage_quota_service,
        "reserve",
        lambda *_args, **_kwargs: Reservation(),
    )
    monkeypatch.setattr(
        download_module,
        "preflight_remote_repo_size",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(download_module.threading, "Thread", Thread)

    result = download_module.ModelDownloadService().start_download(
        "huggingface",
        "owner/repository",
        user_id="user-1",
    )

    assert result["registry_id"]
    assert calls == ["gate"]


def test_force_delete_rejects_legacy_deployment_before_runtime_or_file_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model("legacy-model", source_type="downloaded")
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="legacy-deployment",
                model_id="legacy-model",
                deployment_name="legacy-deployment",
                xinference_endpoint="http://127.0.0.1:12009",
                inference_framework="vllm",
                deploy_mode="container",
                status="running",
                user_id="user-1",
            )
        )
        session.commit()

    cleanup_calls: list[str] = []
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

    with pytest.raises(ValueError, match="deployment lifecycle"):
        registry_module.ModelRegistryService().delete_model(
            "legacy-model",
            force=True,
        )

    assert cleanup_calls == []
    with Session(engine) as session:
        current = session.exec(select(ModelRegistryDB)).one()
        assert current.status == "available"
        assert session.exec(select(DeploymentDB)).one().deployment_id == (
            "legacy-deployment"
        )
    assert model.model_path


def test_force_delete_rejects_sync_referenced_deployment_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("sync-referenced-model", source_type="downloaded")
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="sync-referenced-deployment",
                model_id="sync-referenced-model",
                deployment_name="sync-referenced-deployment",
                xinference_endpoint="http://127.0.0.1:12101",
                inference_framework="vllm",
                deploy_mode="container",
                config={"replica_schema_version": 1},
                status="stopped",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="sync-referenced-r0",
                deployment_id="sync-referenced-deployment",
                replica_index=0,
                container_name="trainfactory-vllm-sync-referenced",
                endpoint="http://127.0.0.1:12101",
                port=12101,
                gpu_ids=[0],
                status="stopped",
            )
        )
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-reference",
                task_name="sync-reference",
                user_id="user-1",
                base_deployment_id="sync-referenced-deployment",
            )
        )
        session.commit()
    cleanup_calls: list[str] = []
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

    with pytest.raises(ValueError, match="sync"):
        registry_module.ModelRegistryService().delete_model(
            "sync-referenced-model",
            force=True,
        )

    assert cleanup_calls == []
    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).one().status == "available"
        assert session.exec(select(DeploymentDB)).one()


def test_force_delete_removes_all_adapter_rows_owned_by_deleted_deployment(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("base-model-delete", source_type="downloaded")
    adapter_source = add_model("other-adapter-model")
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="adapter-owning-deployment",
                model_id="base-model-delete",
                deployment_name="adapter-owning-deployment",
                xinference_endpoint="http://127.0.0.1:12102",
                inference_framework="vllm",
                deploy_mode="container",
                config={"replica_schema_version": 1},
                status="stopped",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="adapter-owning-r0",
                deployment_id="adapter-owning-deployment",
                replica_index=0,
                container_name="trainfactory-vllm-adapter-owning",
                endpoint="http://127.0.0.1:12102",
                port=12102,
                gpu_ids=[0],
                status="stopped",
            )
        )
        session.add(
            LoadedAdapterDB(
                deployment_id="adapter-owning-deployment",
                deployment_replica_id="adapter-owning-r0",
                adapter_name="other-source-adapter",
                adapter_path=adapter_source.model_path,
                source_model_id=adapter_source.model_id,
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()
    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)

    assert registry_module.ModelRegistryService().delete_model(
        "base-model-delete",
        force=True,
    ) is True

    with Session(engine) as session:
        assert session.exec(select(LoadedAdapterDB)).all() == []
        assert session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == adapter_source.model_id
            )
        ).one()


def test_delete_rejects_path_only_adapter_under_managed_base_model_root(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    base = add_model("managed-base-with-path-adapter", source_type="downloaded")
    with Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                deployment_id="other-deployment",
                adapter_name="path-only-under-base",
                adapter_path=base.model_path + "/lora-child",
                source_model_id=None,
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()
    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        registry_module.shutil,
        "rmtree",
        lambda _path: cleanup_calls.append("file"),
    )

    with pytest.raises(ValueError, match="active loaded adapter"):
        registry_module.ModelRegistryService().delete_model(
            base.model_id,
            force=True,
        )

    assert cleanup_calls == []
    with Session(engine) as session:
        assert session.exec(select(LoadedAdapterDB)).one().status == "loaded"


@pytest.mark.parametrize("force", [False, True])
def test_delete_rejects_unknown_runtime_adapter_dependency_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    force: bool,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model("managed-model-with-unknown-runtime-adapter", source_type="downloaded")
    with Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                deployment_id="other-deployment",
                adapter_name="runtime-only-adapter",
                adapter_path="<unknown:auto-sync>",
                source_model_id=None,
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()
    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        registry_module.shutil,
        "rmtree",
        lambda _path: cleanup_calls.append("file"),
    )

    with pytest.raises(ValueError, match="active loaded adapter"):
        registry_module.ModelRegistryService().delete_model(
            model.model_id,
            force=force,
        )

    assert cleanup_calls == []
    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).one().status == "available"
        assert session.exec(select(LoadedAdapterDB)).one().status == "loaded"


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize(
    "task_status",
    ["pending", "preparing", "running", "evaluating"],
)
@pytest.mark.parametrize(
    "reference_kind",
    ["base_model_path", "task_guide_model", "params_guide_model"],
)
@pytest.mark.parametrize("path_shape", ["exact", "descendant"])
def test_delete_rejects_active_training_artifact_consumer_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    force: bool,
    task_status: str,
    reference_kind: str,
    path_shape: str,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model("training-model", source_type="downloaded")
    normalized_child = model.model_path.replace("\\", "/")
    if path_shape == "descendant":
        normalized_child += "/weights/."
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="active-training",
                task_name="active-training",
                base_model_path=(
                    normalized_child
                    if reference_kind == "base_model_path"
                    else "/external/base-model"
                ),
                loss_config=(
                    {"guide_model": normalized_child}
                    if reference_kind == "task_guide_model"
                    else None
                ),
                training_params=(
                    {"loss_config": {"guide_model": normalized_child}}
                    if reference_kind == "params_guide_model"
                    else {}
                ),
                status=task_status,
                user_id="user-1",
            )
        )
        session.commit()

    cleanup_calls: list[str] = []
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

    with pytest.raises(ValueError, match="active training"):
        registry_module.ModelRegistryService().delete_model(
            "training-model",
            force=force,
        )

    assert cleanup_calls == []
    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).one().status == "available"
        assert session.exec(select(TrainingTaskDB)).one().status == task_status


@pytest.mark.parametrize("task_status", ["stopped", "failed"])
@pytest.mark.parametrize(
    ("lease_kind", "process_fields"),
    [
        ("pid", {"process_pid": 4321}),
        ("status", {"process_status": "terminating"}),
        ("create_time", {"process_create_time": 123.5}),
        ("admission", {}),
    ],
)
def test_delete_rejects_terminal_training_with_live_execution_evidence(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    task_status: str,
    lease_kind: str,
    process_fields: dict,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model("terminal-training-model", source_type="downloaded")
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="terminal-training",
                task_name="terminal-training",
                base_model_path=model.model_path,
                status=task_status,
                user_id="user-1",
                **process_fields,
            )
        )
        session.commit()
    monkeypatch.setattr(
        admission_module.background_task_admission_service,
        "get_executing_task_ids",
        lambda task_kind: (
            {"terminal-training"}
            if task_kind == "training" and lease_kind == "admission"
            else set()
        ),
    )
    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        registry_module.shutil,
        "rmtree",
        lambda _path: cleanup_calls.append("file"),
    )

    with pytest.raises(ValueError, match="active training"):
        registry_module.ModelRegistryService().delete_model(
            "terminal-training-model",
            force=True,
        )

    assert cleanup_calls == []
    with Session(engine) as session:
        current = session.exec(select(ModelRegistryDB)).one()
        assert current.status == "available"


@pytest.mark.parametrize("task_status", ["stopped", "failed", "succeeded"])
def test_delete_allows_terminal_training_history_without_execution_evidence(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    task_status: str,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model("terminal-history-model", source_type="downloaded")
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="terminal-history",
                task_name="terminal-history",
                base_model_path=model.model_path,
                status=task_status,
                user_id="user-1",
            )
        )
        session.commit()
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)

    assert registry_module.ModelRegistryService().delete_model(
        "terminal-history-model",
        force=True,
    ) is True

    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).all() == []


def test_delete_snapshots_training_admission_before_each_database_transaction(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    _engine, _config_service, add_model = registry_dependency_db
    add_model("admission-order-model", source_type="downloaded")
    original_get_session = registry_module.get_session
    session_active = threading.local()
    snapshot_calls: list[bool] = []

    @contextmanager
    def observed_get_session():
        assert not getattr(session_active, "value", False)
        with original_get_session() as session:
            session_active.value = True
            try:
                yield session
            finally:
                session_active.value = False

    def get_executing_task_ids(task_kind: str) -> set[str]:
        assert task_kind in {"training", "generation", "evaluation"}
        snapshot_calls.append(getattr(session_active, "value", False))
        assert not getattr(session_active, "value", False), (
            "training admission mutex must be snapshotted before model row locks"
        )
        return set()

    monkeypatch.setattr(registry_module, "get_session", observed_get_session)
    monkeypatch.setattr(
        admission_module.background_task_admission_service,
        "get_executing_task_ids",
        get_executing_task_ids,
    )
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)

    assert registry_module.ModelRegistryService().delete_model(
        "admission-order-model",
        force=True,
    ) is True
    assert snapshot_calls
    assert snapshot_calls == [False] * len(snapshot_calls)


def test_model_delete_phase_a_locks_membership_gate_before_registry_row(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("gate-order-delete", source_type="downloaded")
    statements: list[str] = []

    def observe(_connection, _cursor, statement, _parameters, _context, _many):
        normalized = " ".join(statement.lower().split())
        if (
            "model_artifact_membership_gate" in normalized
            or " from model_registry " in f" {normalized} "
        ):
            statements.append(normalized)

    sa.event.listen(engine, "before_cursor_execute", observe)
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)
    try:
        assert registry_module.ModelRegistryService().delete_model(
            "gate-order-delete",
            force=True,
        ) is True
    finally:
        sa.event.remove(engine, "before_cursor_execute", observe)

    assert statements
    assert statements[0].startswith(
        "update model_artifact_membership_gate set gate_id="
    )


def test_model_delete_final_transaction_reacquires_membership_gate(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("gate-order-final", source_type="downloaded")
    gate_updates: list[str] = []

    def observe(_connection, _cursor, statement, _parameters, _context, _many):
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("update model_artifact_membership_gate"):
            gate_updates.append(normalized)

    sa.event.listen(engine, "before_cursor_execute", observe)
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)
    try:
        assert registry_module.ModelRegistryService().delete_model(
            "gate-order-final",
            force=True,
        ) is True
    finally:
        sa.event.remove(engine, "before_cursor_execute", observe)

    # A1 intent, A2 dependency admission, Phase B claim validation, and the
    # pre-runtime/post-runtime/final snapshot validations each use a fresh
    # transaction and reacquire the membership gate.
    assert len(gate_updates) == 6


@pytest.mark.parametrize(
    "reference_kind",
    ["base_model_path", "task_guide_model", "params_guide_model"],
)
@pytest.mark.parametrize(
    "path_alias",
    [
        pytest.param(
            "case_alias",
            marks=pytest.mark.skipif(
                os.name != "nt",
                reason="case aliases are equivalent only on Windows",
            ),
        ),
        "relative_alias",
    ],
)
def test_training_create_locks_matching_registry_path_and_rejects_delete_intent(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    reference_kind: str,
    path_alias: str,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model(
        "training-model-deleting",
        status="deleting",
        metadata=_deleting_metadata(),
    )
    child_path = model.model_path.replace("\\", "/") + "/snapshot"
    if path_alias == "case_alias":
        child_path = child_path.swapcase()
    else:
        monkeypatch.chdir(os.path.dirname(model.model_path))
        child_path = os.path.relpath(child_path)
    kwargs = {
        "task_name": "must-not-persist",
        "model_path": (
            child_path
            if reference_kind == "base_model_path"
            else "/external/base-model"
        ),
        "loss_config": (
            {"guide_model": child_path}
            if reference_kind == "task_guide_model"
            else None
        ),
        "training_params": (
            {"loss_config": {"guide_model": child_path}}
            if reference_kind == "params_guide_model"
            else {}
        ),
        "user_id": "user-1",
    }

    with pytest.raises(ValueError, match="being deleted"):
        training_module.TrainingTaskService().create_task(**kwargs)

    with Session(engine) as session:
        assert session.exec(select(TrainingTaskDB)).all() == []


def test_training_create_rejects_symlink_alias_of_deleting_model(
    registry_dependency_db,
    tmp_path,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model(
        "training-model-symlink",
        status="deleting",
        metadata=_deleting_metadata(),
    )
    alias = Path(tmp_path) / "model-alias"
    try:
        alias.symlink_to(Path(model.model_path), target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink is unavailable: {exc}")

    with pytest.raises(ValueError, match="being deleted"):
        training_module.TrainingTaskService().create_task(
            task_name="symlink-alias-must-not-persist",
            model_path=str(alias / "snapshot"),
            training_params={},
            user_id="user-1",
        )

    with Session(engine) as session:
        assert session.exec(select(TrainingTaskDB)).all() == []


def test_training_registry_lock_query_targets_only_matching_model_ids() -> None:
    task = TrainingTaskDB(
        task_id="targeted-lock",
        task_name="targeted-lock",
        base_model_path="C:/models/target/snapshot",
        status="pending",
    )
    candidates = [
        ModelRegistryDB(
            model_id="unrelated",
            model_name="unrelated",
            model_type="llm",
            model_path="C:/models/unrelated",
        ),
        ModelRegistryDB(
            model_id="target",
            model_name="target",
            model_type="llm",
            model_path="C:/models/target",
        ),
    ]
    statements: list[str] = []

    class Result:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

        def first(self):
            return self._rows[0] if self._rows else None

    class FakeSession:
        def exec(self, statement):
            sql = str(statement)
            statements.append(sql)
            if "model_artifact_membership_gate" in sql:
                return Result([object()])
            return Result(candidates if "FOR UPDATE" not in sql else [candidates[1]])

    locked = training_module._lock_registry_models_for_training(
        FakeSession(),
        task,
    )

    locking_statements = [sql for sql in statements if "FOR UPDATE" in sql]
    assert [model.model_id for model in locked] == ["target"]
    assert len(locking_statements) == 2
    assert "model_artifact_membership_gate" in locking_statements[0]
    assert "WHERE model_registry.model_id IN" in locking_statements[1]
    assert "ORDER BY model_registry.model_id" in locking_statements[1]


def test_training_reactivation_locks_gate_before_reading_candidate_task(
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model("training-gate-order")
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="training-gate-order-task",
                task_name="training-gate-order-task",
                base_model_path=model.model_path,
                status="pending",
                user_id="user-1",
            )
        )
        session.commit()
    statements: list[str] = []

    def observe(_connection, _cursor, statement, _parameters, _context, _many):
        normalized = " ".join(statement.lower().split())
        if (
            "model_artifact_membership_gate" in normalized
            or " from training_tasks " in f" {normalized} "
        ):
            statements.append(normalized)

    sa.event.listen(engine, "before_cursor_execute", observe)
    try:
        assert training_module.TrainingTaskService().claim_preparing(
            "training-gate-order-task",
            "run-token",
        ) is True
    finally:
        sa.event.remove(engine, "before_cursor_execute", observe)

    assert statements[0].startswith(
        "update model_artifact_membership_gate set gate_id="
    )


@pytest.mark.parametrize(
    ("operation", "initial_status"),
    [
        ("claim", "pending"),
        ("resume", "stopped"),
        ("execution_config", "failed"),
        ("execution_config_guide", "failed"),
        ("status_running", "pending"),
    ],
)
def test_training_reactivation_and_path_updates_recheck_model_delete_intent(
    registry_dependency_db,
    operation: str,
    initial_status: str,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model(
        "training-model-deleting",
        status="deleting",
        metadata=_deleting_metadata(),
    )
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="training-retry",
                task_name="training-retry",
                base_model_path=model.model_path,
                training_params={"base_model_path": model.model_path},
                status=initial_status,
                user_id="user-1",
            )
        )
        session.commit()

    service = training_module.TrainingTaskService()
    if operation == "claim":
        assert service.claim_preparing("training-retry", "run-token") is False
    elif operation == "resume":
        assert service.reset_for_resume("training-retry", "run-token") is False
    elif operation == "execution_config":
        with pytest.raises(ValueError, match="being deleted"):
            service.update_task_execution_config(
                "training-retry",
                model_path=model.model_path + "/new-snapshot",
            )
    elif operation == "execution_config_guide":
        with pytest.raises(ValueError, match="being deleted"):
            service.update_task_execution_config(
                "training-retry",
                model_path="/external/base-model",
                training_params={
                    "loss_config": {
                        "guide_model": model.model_path + "/guide-snapshot"
                    }
                },
            )
    else:
        with pytest.raises(ValueError, match="being deleted"):
            service.update_task_status("training-retry", "running")

    with Session(engine) as session:
        task = session.exec(select(TrainingTaskDB)).one()
        assert task.status == initial_status
        assert task.base_model_path == model.model_path


@pytest.mark.parametrize(
    ("original_metadata", "had_extra_metadata"),
    [(None, False), ({}, True)],
)
def test_stale_intent_preserves_none_versus_empty_metadata_on_takeover(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    original_metadata,
    had_extra_metadata: bool,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    intent = {
        "token": "stale-owner",
        "started_at": "2000-01-01T00:00:00",
        "heartbeat_at": "2000-01-01T00:00:00",
        "original_status": "available",
        "original_extra_metadata": original_metadata,
        "had_extra_metadata": had_extra_metadata,
    }
    add_model(
        "model-stale-metadata",
        status="deleting",
        metadata={DELETE_INTENT_KEY: intent},
        source_type="downloaded",
    )
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)

    assert registry_module.ModelRegistryService().delete_model(
        "model-stale-metadata",
        force=True,
    ) is True
    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).all() == []


@pytest.mark.parametrize("original_metadata", [None, {}])
def test_pre_cleanup_failure_restores_exact_original_metadata_shape(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    original_metadata,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("model-restore-metadata", metadata=original_metadata)
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="canonical-restore",
                model_id="model-restore-metadata",
                deployment_name="canonical-restore",
                xinference_endpoint="http://127.0.0.1:12008",
                inference_framework="vllm",
                deploy_mode="container",
                config={"replica_schema_version": 1},
                status="stopped",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="canonical-restore-r0",
                deployment_id="canonical-restore",
                replica_index=0,
                container_name="trainfactory-vllm-canonical-restore",
                endpoint="http://127.0.0.1:12008",
                port=12008,
                gpu_ids=[0],
                status="stopped",
            )
        )
        session.commit()

    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_claim_replica_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_module.ReplicaOperationBusyError("claim busy")
        ),
    )

    with pytest.raises(deployment_module.ReplicaOperationBusyError, match="claim busy"):
        registry_module.ModelRegistryService().delete_model(
            "model-restore-metadata",
            force=True,
        )

    with Session(engine) as session:
        current = session.exec(select(ModelRegistryDB)).one()
        assert current.status == "available"
        assert current.extra_metadata == original_metadata


def test_fresh_delete_intent_retry_is_busy_without_cleanup_or_database_mutation(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    original_metadata = _deleting_metadata("fresh-owner")
    model = add_model(
        "model-fresh-intent",
        status="deleting",
        metadata=original_metadata,
        source_type="downloaded",
    )
    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        registry_module.shutil,
        "rmtree",
        lambda _path: cleanup_calls.append("file"),
    )

    with pytest.raises(
        deployment_module.ReplicaOperationBusyError,
        match="already in progress",
    ):
        registry_module.ModelRegistryService().delete_model(
            "model-fresh-intent",
            force=True,
        )

    assert cleanup_calls == []
    with Session(engine) as session:
        current = session.exec(select(ModelRegistryDB)).one()
        assert current.status == "deleting"
        assert current.extra_metadata == original_metadata
    assert model.model_path


def test_stale_intent_only_crash_is_taken_over_and_delete_completes(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model(
        "model-stale-intent",
        status="deleting",
        metadata=_deleting_metadata(
            "stale-owner",
            heartbeat_at="2000-01-01T00:00:00",
        ),
        source_type="downloaded",
    )
    observed_tokens: list[str] = []

    def observe_new_owner(_path) -> None:
        with Session(engine) as session:
            current = session.exec(select(ModelRegistryDB)).one()
            observed_tokens.append(
                current.extra_metadata[DELETE_INTENT_KEY]["token"]
            )

    monkeypatch.setattr(registry_module.shutil, "rmtree", observe_new_owner)

    assert registry_module.ModelRegistryService().delete_model(
        "model-stale-intent",
        force=True,
    ) is True
    assert len(observed_tokens) == 1
    assert observed_tokens[0] != "stale-owner"
    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).all() == []
    assert model.model_path


def test_background_heartbeat_prevents_takeover_during_long_external_step(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("model-long-cleanup", source_type="downloaded")
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="long-cleanup-deployment",
                model_id="model-long-cleanup",
                deployment_name="long-cleanup-deployment",
                xinference_endpoint="http://127.0.0.1:12007",
                inference_framework="vllm",
                deploy_mode="container",
                config={"replica_schema_version": 1},
                status="stopped",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="long-cleanup-r0",
                deployment_id="long-cleanup-deployment",
                replica_index=0,
                container_name="trainfactory-xinference-long-cleanup",
                endpoint="http://127.0.0.1:12007",
                port=12007,
                gpu_ids=[0],
                status="stopped",
            )
        )
        session.commit()

    monkeypatch.setattr(
        registry_module,
        "MODEL_DELETE_INTENT_HEARTBEAT_SECONDS",
        0.01,
    )
    clock_lock = threading.Lock()
    initial_time = datetime(2026, 1, 1)
    expired_time = initial_time + timedelta(
        seconds=registry_module.MODEL_DELETE_INTENT_LEASE_SECONDS + 1,
    )
    lease_time = initial_time

    def lease_now() -> datetime:
        with clock_lock:
            return lease_time

    monkeypatch.setattr(registry_module, "now_naive", lease_now)
    owner_in_cleanup = threading.Event()
    release_owner = threading.Event()
    heartbeat_after_lease = threading.Event()
    retry_cleanup_calls: list[str] = []

    def long_cleanup(*_args, **_kwargs) -> None:
        nonlocal lease_time
        if threading.current_thread().name.startswith("model-delete-owner"):
            # The original lease is now expired. Keep the clock fixed so a
            # real committed heartbeat, not host scheduling speed, determines
            # whether the contender may take ownership.
            with clock_lock:
                lease_time = expired_time
            owner_in_cleanup.set()
            assert release_owner.wait(timeout=5)
        else:
            retry_cleanup_calls.append(threading.current_thread().name)

    monkeypatch.setattr(registry_module, "_remove_deployment_container", long_cleanup)
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda _path: None)
    service = registry_module.ModelRegistryService()
    original_renew = service._renew_model_delete_intent

    def observe_renewal(model_id: str, token: str) -> None:
        renewal_time = lease_now()
        original_renew(model_id, token)
        if (
            threading.current_thread().name.startswith("model-delete-heartbeat")
            and renewal_time == expired_time
        ):
            heartbeat_after_lease.set()

    monkeypatch.setattr(service, "_renew_model_delete_intent", observe_renewal)

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="model-delete-owner",
    ) as executor:
        owner = executor.submit(service.delete_model, "model-long-cleanup", True)
        if not owner_in_cleanup.wait(timeout=5):
            owner.result(timeout=1)
            pytest.fail("model delete did not reach runtime cleanup")
        try:
            assert heartbeat_after_lease.wait(timeout=5)
            with Session(engine) as session:
                current = session.exec(select(ModelRegistryDB)).one()
                assert current.extra_metadata[DELETE_INTENT_KEY]["heartbeat_at"] == (
                    expired_time.isoformat()
                )
            with pytest.raises(
                deployment_module.ReplicaOperationBusyError,
                match="already in progress",
            ):
                service.delete_model("model-long-cleanup", force=True)
        finally:
            release_owner.set()
        assert owner.result(timeout=5) is True

    assert retry_cleanup_calls == []


def test_model_delete_heartbeat_recovers_after_one_transient_database_failure(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    _engine, _config_service, add_model = registry_dependency_db
    add_model("model-flaky-heartbeat", source_type="downloaded")
    monkeypatch.setattr(registry_module, "MODEL_DELETE_INTENT_LEASE_SECONDS", 0.05)
    monkeypatch.setattr(
        registry_module,
        "MODEL_DELETE_INTENT_HEARTBEAT_SECONDS",
        0.01,
    )
    owner_in_cleanup = threading.Event()
    release_owner = threading.Event()
    heartbeat_failed = threading.Event()
    heartbeat_recovered_after_lease = threading.Event()
    cleanup_started_at: list[float] = []

    def long_cleanup(_path) -> None:
        if threading.current_thread().name.startswith("model-delete-owner"):
            cleanup_started_at.append(time.monotonic())
            owner_in_cleanup.set()
            assert release_owner.wait(timeout=5)

    monkeypatch.setattr(registry_module.shutil, "rmtree", long_cleanup)
    service = registry_module.ModelRegistryService()
    original_renew = service._renew_model_delete_intent

    def flaky_renew(model_id: str, token: str) -> None:
        if (
            threading.current_thread().name.startswith("model-delete-heartbeat")
            and not heartbeat_failed.is_set()
        ):
            heartbeat_failed.set()
            raise RuntimeError("transient database failure")
        original_renew(model_id, token)
        if (
            heartbeat_failed.is_set()
            and cleanup_started_at
            and time.monotonic() - cleanup_started_at[0]
            > registry_module.MODEL_DELETE_INTENT_LEASE_SECONDS
        ):
            heartbeat_recovered_after_lease.set()

    monkeypatch.setattr(service, "_renew_model_delete_intent", flaky_renew)
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="model-delete-owner",
    ) as executor:
        owner = executor.submit(
            service.delete_model,
            "model-flaky-heartbeat",
            True,
        )
        assert owner_in_cleanup.wait(timeout=5)
        assert heartbeat_failed.wait(timeout=5)
        assert heartbeat_recovered_after_lease.wait(timeout=5)
        try:
            with pytest.raises(
                deployment_module.ReplicaOperationBusyError,
                match="already in progress",
            ):
                service.delete_model("model-flaky-heartbeat", force=True)
        finally:
            release_owner.set()
        assert owner.result(timeout=5) is True

    assert service._model_delete_heartbeats == {}


def test_create_deployment_rejects_deleting_registry_model_in_insert_transaction(
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("model-deleting", status="deleting", metadata=_deleting_metadata())

    with pytest.raises(ValueError, match="being deleted"):
        deployment_module.DeploymentService().create_deployment(
            model_id="model-deleting",
            xinference_endpoint="http://127.0.0.1:12000",
            deployment_name="must-not-persist",
            user_id="user-1",
        )

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).all() == []


def test_persist_replica_plan_rejects_deleting_registry_model_before_rows(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("model-deleting", status="deleting", metadata=_deleting_metadata())
    service = deployment_module.DeploymentService()
    monkeypatch.setattr(service, "_register_claim_heartbeat", lambda _claim: None)
    plan = DeploymentPlan(
        replicas=(
            ReplicaPlan(
                replica_index=0,
                container_name="trainfactory-vllm-delete-intent",
                gpu_ids=(0,),
                port=12001,
                endpoint="http://trainfactory-vllm-delete-intent:12001",
            ),
        ),
        required_gpus_per_replica=1,
        reservation_owner="reservation-owner",
    )

    with pytest.raises(ValueError, match="being deleted"):
        service._persist_replica_plan(
            deployment_id="planned-deployment",
            model={"model_id": "model-deleting", "model_name": "deleting"},
            deployment_name="must-not-persist",
            replica_count=1,
            gpu_memory_utilization=0.8,
            config={},
            user_id="user-1",
            inference_framework="vllm",
            enable_lora=False,
            max_loras=4,
            max_lora_rank=64,
            external_api_config_id=None,
            plan=plan,
            auto_start=False,
            defer_start=False,
        )

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).all() == []
        assert session.exec(select(DeploymentReplicaDB)).all() == []


@pytest.mark.parametrize("use_source_model_id", [True, False])
def test_adapter_load_rejects_deleting_source_model_before_insert_or_runtime(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    use_source_model_id: bool,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    source = add_model(
        "adapter-deleting",
        status="deleting",
        metadata=_deleting_metadata(),
    )
    add_model("base-model")
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="adapter-target",
                model_id="base-model",
                deployment_name="adapter-target",
                xinference_endpoint="http://127.0.0.1:12002",
                inference_framework="vllm",
                deploy_mode="container",
                enable_lora=True,
                status="running",
                user_id="user-1",
            )
        )
        session.commit()

    runtime_calls: list[str] = []
    fake_client = type(
        "FakeClient",
        (),
        {
            "load_lora_adapter": lambda _self, name, _path: runtime_calls.append(name),
            "unload_lora_adapter": lambda _self, _name: None,
        },
    )()
    service = adapter_module.AdapterService()
    monkeypatch.setattr(service, "_get_inference_client", lambda *_a, **_k: fake_client)

    with pytest.raises(ValueError, match="being deleted"):
        service._load_adapter_claimed(
            "adapter-target",
            "adapter-under-delete",
            source.model_path,
            source_model_id=(
                "adapter-deleting" if use_source_model_id else None
            ),
            user_id="user-1",
            _claim=None,
        )

    assert runtime_calls == []
    with Session(engine) as session:
        assert session.exec(select(LoadedAdapterDB)).all() == []


@pytest.mark.parametrize("use_source_model_id", [True, False])
def test_adapter_sync_cannot_reactivate_deleting_source_model_dependency(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    use_source_model_id: bool,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    source = add_model(
        "adapter-deleting",
        status="deleting",
        metadata=_deleting_metadata(),
    )
    add_model("base-model")
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="adapter-target",
                model_id="base-model",
                deployment_name="adapter-target",
                xinference_endpoint="http://127.0.0.1:12003",
                inference_framework="vllm",
                deploy_mode="container",
                enable_lora=True,
                status="running",
                user_id="user-1",
            )
        )
        session.add(
            LoadedAdapterDB(
                deployment_id="adapter-target",
                adapter_name="stale-adapter",
                adapter_path=source.model_path,
                source_model_id=(
                    "adapter-deleting" if use_source_model_id else None
                ),
                status="unloaded",
                user_id="user-1",
            )
        )
        session.commit()
        adapter_before = session.exec(select(LoadedAdapterDB)).one().model_dump()

    runtime_calls: list[str] = []
    fake_client = type(
        "FakeClient",
        (),
        {
            "list_lora_adapters": lambda _self: runtime_calls.append("list")
            or [{"name": "stale-adapter"}]
        },
    )()
    service = adapter_module.AdapterService()
    monkeypatch.setattr(service, "_get_inference_client", lambda *_a, **_k: fake_client)

    with pytest.raises(
        registry_module.ModelDeletionInProgressError,
        match="being deleted",
    ):
        service.sync_loaded_adapters(
            "adapter-target",
            user_id="user-1",
        )

    assert runtime_calls == ["list"]
    with Session(engine) as session:
        adapter = session.exec(select(LoadedAdapterDB)).one()
        assert adapter.model_dump() == adapter_before
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token is None
        assert parent.replica_operation_kind is None
        assert parent.replica_operation_generation == 1


@pytest.mark.parametrize(
    "takeover",
    ["parent_deleted", "token_replaced", "operation_changed_to_delete"],
)
def test_adapter_sync_rechecks_exact_parent_claim_after_runtime_list(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
    takeover: str,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    add_model("base-model")
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="adapter-sync-target",
                model_id="base-model",
                deployment_name="adapter-sync-target",
                xinference_endpoint="http://127.0.0.1:12013",
                inference_framework="vllm",
                deploy_mode="container",
                container_name="trainfactory-vllm-adapter-sync-target",
                enable_lora=True,
                config={"replica_schema_version": 1},
                status="running",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="adapter-sync-target-r0",
                deployment_id="adapter-sync-target",
                replica_index=0,
                container_name="trainfactory-vllm-adapter-sync-target",
                endpoint="http://127.0.0.1:12013",
                port=12013,
                gpu_ids=[0],
                status="running",
                health_status="HEALTHY",
            )
        )
        session.commit()

    takeover_owner: dict[str, object] = {}

    class Client:
        def __init__(self, deployment: DeploymentDB) -> None:
            self.deployment = deployment

        def list_lora_adapters(self):
            owner_session = sa.inspect(self.deployment).session
            assert owner_session is None
            with Session(engine) as takeover_session:
                parent = takeover_session.exec(
                    select(DeploymentDB).where(
                        DeploymentDB.deployment_id == "adapter-sync-target"
                    )
                ).one()
                if takeover == "parent_deleted":
                    takeover_session.delete(parent)
                elif takeover == "token_replaced":
                    parent.replica_operation_token = "replacement-delete-owner"
                    parent.replica_operation_generation += 1
                    parent.replica_operation_kind = "delete"
                    takeover_owner.update(
                        token=parent.replica_operation_token,
                        generation=parent.replica_operation_generation,
                    )
                else:
                    parent.replica_operation_kind = "delete"
                    takeover_owner.update(
                        token=parent.replica_operation_token,
                        generation=parent.replica_operation_generation,
                    )
                takeover_session.commit()
            return [{"name": "runtime-only-adapter"}]

    lifecycle_service = deployment_module.DeploymentService()
    service = adapter_module.AdapterService(lifecycle_service)
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda deployment, **_kwargs: Client(deployment),
    )

    with pytest.raises(
        deployment_module.ReplicaOperationLostError,
        match="operation ownership was lost",
    ):
        service.sync_loaded_adapters(
            "adapter-sync-target",
            deployment_replica_id="adapter-sync-target-r0",
            user_id="user-1",
        )
    with Session(engine) as session:
        assert session.exec(select(LoadedAdapterDB)).all() == []
        parent = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == "adapter-sync-target"
            )
        ).first()
        if takeover == "parent_deleted":
            assert parent is None
        else:
            assert parent is not None
            assert parent.replica_operation_token == takeover_owner["token"]
            assert parent.replica_operation_generation == takeover_owner["generation"]
            assert parent.replica_operation_kind == "delete"
    assert lifecycle_service._claim_heartbeats == {}


def _config_kwargs() -> dict:
    return {
        "config_name": "must-not-persist",
        "model_type": "llm",
        "provider": "xinference",
        "api_endpoint": "http://127.0.0.1:12004",
        "model_name": "model-deleting",
        "source_type": "local_deployed",
        "registry_id": "model-deleting",
        "deployment_id": "config-target",
        "user_id": "user-1",
        "validate": False,
    }


def _seed_deleting_config_target(engine, add_model) -> None:
    add_model("model-deleting", status="deleting", metadata=_deleting_metadata())
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="config-target",
                model_id="model-deleting",
                deployment_name="config-target",
                xinference_endpoint="http://127.0.0.1:12004",
                inference_framework="xinference",
                deploy_mode="shared",
                config={"replica_schema_version": 1},
                status="running",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="config-target-r0",
                deployment_id="config-target",
                replica_index=0,
                container_name="trainfactory-xinference-config-target",
                endpoint="http://127.0.0.1:12004",
                port=12004,
                gpu_ids=[0],
                status="running",
            )
        )
        session.commit()


def test_sync_model_config_create_rejects_deleting_registry_model(
    registry_dependency_db,
) -> None:
    engine, config_service, add_model = registry_dependency_db
    _seed_deleting_config_target(engine, add_model)

    with pytest.raises(registry_module.ModelDeletionInProgressError, match="being deleted"):
        config_service.create_config(**_config_kwargs())

    with Session(engine) as session:
        assert session.exec(select(ModelConfigDB)).all() == []


def test_async_model_config_create_rejects_deleting_registry_model(
    registry_dependency_db,
) -> None:
    engine, config_service, add_model = registry_dependency_db
    _seed_deleting_config_target(engine, add_model)

    with pytest.raises(registry_module.ModelDeletionInProgressError, match="being deleted"):
        asyncio.run(config_service.create_config_async(**_config_kwargs()))

    with Session(engine) as session:
        assert session.exec(select(ModelConfigDB)).all() == []


def test_delete_publishes_hidden_intent_before_runtime_cleanup_and_blocks_create(
    monkeypatch: pytest.MonkeyPatch,
    registry_dependency_db,
) -> None:
    engine, _config_service, add_model = registry_dependency_db
    model = add_model(
        "model-delete",
        metadata={"user_metadata": {"preserve": True}},
        source_type="downloaded",
    )
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="existing-deployment",
                model_id="model-delete",
                deployment_name="existing-deployment",
                xinference_endpoint="http://127.0.0.1:12005",
                inference_framework="vllm",
                deploy_mode="container",
                config={"replica_schema_version": 1},
                status="stopped",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="existing-deployment-r0",
                deployment_id="existing-deployment",
                replica_index=0,
                container_name="trainfactory-xinference-existing-deployment",
                endpoint="http://127.0.0.1:12005",
                port=12005,
                gpu_ids=[0],
                status="stopped",
            )
        )
        session.commit()

    observations: dict[str, object] = {}

    def inspect_intent_and_try_create(*_args, **_kwargs) -> None:
        with Session(engine) as session:
            deleting = session.exec(
                select(ModelRegistryDB).where(
                    ModelRegistryDB.model_id == "model-delete"
                )
            ).one()
            observations["status"] = deleting.status
            observations["stored_token"] = bool(
                (deleting.extra_metadata or {}).get(DELETE_INTENT_KEY, {}).get(
                    "token"
                )
            )
        public = registry_module.ModelRegistryService().get_model("model-delete")
        observations["public_metadata"] = public["extra_metadata"]
        with pytest.raises(ValueError, match="being deleted"):
            deployment_module.DeploymentService().create_deployment(
                model_id="model-delete",
                xinference_endpoint="http://127.0.0.1:12006",
                deployment_name="racing-deployment",
                user_id="user-1",
            )

    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        inspect_intent_and_try_create,
    )

    assert registry_module.ModelRegistryService().delete_model(
        "model-delete",
        force=True,
    ) is True
    assert observations == {
        "status": "deleting",
        "stored_token": True,
        "public_metadata": {"user_metadata": {"preserve": True}},
    }
    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).all() == []
        assert session.exec(select(DeploymentDB)).all() == []
    assert model.model_path
