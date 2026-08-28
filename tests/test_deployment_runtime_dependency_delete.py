"""Runtime-consumer fences for force-deleting deployments."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta
import importlib

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from train_factory.enums.sync_status import SyncStatus
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import (
    DeploymentReplicaDB,
)
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
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB


deployment_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)
routes_module = importlib.import_module(
    "train_factory.api.routes.deployment_routes"
)
runtime_module = importlib.import_module(
    "train_factory.storage.services.runtime_dependency_service"
)


class _DeletionDocker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.exists = False
        self.on_remove = None

    def container_exists(self, container_name: str) -> bool:
        self.calls.append(("exists", container_name))
        return self.exists

    def container_exists_authoritative(self, container_name: str) -> bool:
        self.calls.append(("exists-authoritative", container_name))
        return self.exists

    def get_managed_container_id(
        self,
        container_name: str,
        *,
        deployment_id: str,
        replica_id: str,
    ) -> str:
        self.calls.append(("inspect", container_name))
        assert deployment_id == "deployment-runtime-guard"
        assert replica_id == "replica-0"
        return "sha256:replica-0"

    def get_legacy_managed_container_id(
        self,
        container_name: str,
        *,
        deployment_id: str,
        replica_id: str,
    ) -> str:
        self.calls.append(("inspect", container_name))
        assert deployment_id == "deployment-runtime-guard"
        assert replica_id == "legacy"
        return "sha256:legacy"

    def remove_container_identity(self, container_id: str) -> bool:
        self.calls.append(("remove", container_id))
        if self.on_remove is not None:
            callback, self.on_remove = self.on_remove, None
            callback()
        return True


@pytest.fixture(params=["canonical", "legacy"])
def deployment_runtime_guard_db(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    engine = create_engine(
        f"sqlite:///{tmp_path / f'deployment-runtime-{request.param}.db'}"
    )
    SQLModel.metadata.create_all(engine)

    @contextmanager
    def test_session():
        with Session(engine) as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise

    docker = _DeletionDocker()
    monkeypatch.setattr(deployment_module, "get_session", test_session)
    monkeypatch.setattr(deployment_module, "docker_deployer", docker)
    monkeypatch.setattr(
        deployment_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": model_id,
            "model_path": f"/app/models/{model_id}",
            "model_type": "embedding",
        },
    )

    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        for model_id in ("deployment-model", "mismatched-config-model"):
            session.add(
                ModelRegistryDB(
                    model_id=model_id,
                    model_name=model_id,
                    model_type="embedding",
                    model_path=f"/app/models/{model_id}",
                    source_type="trained",
                    status="available",
                    user_id="user-1",
                )
            )
        canonical = request.param == "canonical"
        session.add(
            DeploymentDB(
                deployment_id="deployment-runtime-guard",
                deployment_name="runtime-guard",
                model_id="deployment-model",
                model_uid="runtime-guard-model",
                xinference_endpoint="http://127.0.0.1:12121",
                deploy_mode="container" if canonical else "shared",
                container_name=(
                    "trainfactory-vllm-model-deployme" if canonical else None
                ),
                inference_framework="vllm" if canonical else "xinference",
                config=(
                    {
                        "launch_config": {"framework": "vllm"},
                        "replica_schema_version": 1,
                    }
                    if canonical
                    else {"legacy": True}
                ),
                status="stopped",
                user_id="user-1",
            )
        )
        if canonical:
            session.add(
                DeploymentReplicaDB(
                    replica_id="replica-0",
                    deployment_id="deployment-runtime-guard",
                    replica_index=0,
                    container_name="trainfactory-vllm-model-deployme",
                    endpoint="http://127.0.0.1:12121",
                    port=12121,
                    gpu_ids=[0],
                    status="stopped",
                )
            )
        session.add(
            ModelConfigDB(
                config_id="deployment-runtime-config",
                config_name="deployment-runtime-config",
                source_type="local_deployed",
                registry_id="mismatched-config-model",
                deployment_id="deployment-runtime-guard",
                model_type="embedding",
                provider="openai-compatible",
                api_endpoint="http://runtime.invalid/v1",
                model_name="runtime-guard",
                user_id="user-1",
            )
        )
        session.commit()

    return request.param, engine, docker


def _add_runtime_dependency(engine, dependency_kind: str) -> None:
    with Session(engine) as session:
        if dependency_kind == "generation":
            session.add(
                GenerationTaskDB(
                    task_id="active-generation",
                    task_name="active-generation",
                    input_path="/managed/input.jsonl",
                    llm_config={},
                    steps_config={},
                    embedding_config_id="deployment-runtime-config",
                    status=GenerationStatus.RUNNING,
                    user_id="user-1",
                )
            )
        elif dependency_kind == "evaluation":
            session.add(
                EvaluationTaskDB(
                    task_id="active-evaluation",
                    eval_framework=EvaluationFramework.MTEB,
                    model_configs=[
                        {"deployment_id": "deployment-runtime-guard"}
                    ],
                    status=EvaluationStatus.RUNNING,
                    user_id="user-1",
                )
            )
        elif dependency_kind == "deepeval":
            session.add(
                EvaluationTaskDB(
                    task_id="active-deepeval",
                    eval_framework=EvaluationFramework.DEEPEVAL,
                    model_configs=[
                        {
                            "group_name": "group",
                            "embedding": {
                                "config_id": "deployment-runtime-config"
                            },
                        }
                    ],
                    status=EvaluationStatus.RUNNING,
                    user_id="user-1",
                )
            )
        elif dependency_kind == "external_sync":
            session.add(
                ExternalSyncTaskDB(
                    task_id="active-external-sync",
                    task_name="active-external-sync",
                    generation_config={
                        "embedding_config": {
                            "config_id": "deployment-runtime-config"
                        }
                    },
                    is_active=True,
                    status=SyncStatus.IDLE,
                    user_id="user-1",
                )
            )
        else:
            session.add(
                MilvusCollectionDB(
                    collection_name="deployment-runtime-collection",
                    embedding_config_id="deployment-runtime-config",
                    status="active",
                    user_id="user-1",
                )
            )
        session.commit()


@pytest.mark.parametrize(
    "dependency_kind",
    ["generation", "evaluation", "deepeval", "external_sync", "milvus"],
)
def test_force_delete_rejects_runtime_dependency_before_first_side_effect(
    deployment_runtime_guard_db,
    dependency_kind: str,
) -> None:
    lifecycle, engine, docker = deployment_runtime_guard_db
    _add_runtime_dependency(engine, dependency_kind)

    expected = "Milvus collection" if dependency_kind == "milvus" else "active runtime"
    with pytest.raises(ValueError, match=expected):
        deployment_module.DeploymentService().delete_deployment(
            "deployment-runtime-guard",
            force=True,
            user_id="user-1",
        )

    assert docker.calls == []
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one()
        assert session.exec(select(ModelConfigDB)).one()
        replicas = list(session.exec(select(DeploymentReplicaDB)).all())
        assert len(replicas) == (1 if lifecycle == "canonical" else 0)


def test_delete_route_maps_runtime_dependency_conflict_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes_module.deployment_service,
        "get_deployment",
        lambda _deployment_id: {"deployment_id": "deployment-runtime-guard"},
    )
    monkeypatch.setattr(
        routes_module,
        "verify_resource_ownership",
        lambda deployment, *_args: deployment,
    )
    monkeypatch.setattr(
        routes_module,
        "_validate_deployment_endpoint",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        routes_module.deployment_service,
        "delete_deployment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime_module.RuntimeDependencyUnavailableError(
                "deployment is referenced by an active runtime task"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes_module.delete_deployment(
                "deployment-runtime-guard",
                force=True,
                current_user={"user_id": "user-1", "role": "user"},
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "deployment is referenced by an active runtime task"
    )


def test_dependency_appearing_after_runtime_cleanup_blocks_final_delete(
    deployment_runtime_guard_db,
) -> None:
    lifecycle, engine, docker = deployment_runtime_guard_db
    service = deployment_module.DeploymentService()
    if lifecycle == "legacy":
        with Session(engine) as session:
            deployment = session.exec(select(DeploymentDB)).one()
            deployment.deploy_mode = "container"
            deployment.inference_framework = "vllm"
            deployment.container_name = service._build_managed_container_name(
                deployment.deployment_id,
                "deployment-model",
                "vllm",
            )
            session.add(deployment)
            session.commit()

    def add_late_consumer() -> None:
        _add_runtime_dependency(engine, "generation")

    docker.exists = True
    docker.on_remove = add_late_consumer

    with pytest.raises(ValueError, match="active runtime"):
        service.delete_deployment(
            "deployment-runtime-guard",
            force=True,
            user_id="user-1",
        )

    assert any(method == "remove" for method, _value in docker.calls)
    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        assert deployment.replica_operation_kind == "delete"
        assert deployment.replica_operation_token
        assert session.exec(select(ModelConfigDB)).one()
    assert service._claim_heartbeats == {}
    with pytest.raises(deployment_module.ReplicaOperationBusyError):
        service.delete_deployment(
            "deployment-runtime-guard",
            force=True,
            user_id="user-1",
        )


def test_deployment_delete_cleanup_is_allowed_during_model_delete_intent(
    deployment_runtime_guard_db,
) -> None:
    lifecycle, engine, docker = deployment_runtime_guard_db
    service = deployment_module.DeploymentService()
    with Session(engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == "deployment-model"
            )
        ).one()
        model.status = "deleting"
        model.extra_metadata = {
            "_train_factory_model_delete_intent_v1": {
                "token": "model-delete-owner",
                "had_extra_metadata": False,
            }
        }
        deployment = session.exec(select(DeploymentDB)).one()
        deployment.status = "running"
        if lifecycle == "legacy":
            deployment.deploy_mode = "container"
            deployment.inference_framework = "vllm"
            deployment.container_name = service._build_managed_container_name(
                deployment.deployment_id,
                deployment.model_id,
                "vllm",
            )
        session.add(model)
        session.add(deployment)
        session.commit()
    docker.exists = True

    assert service.delete_deployment(
        "deployment-runtime-guard",
        force=True,
        user_id="user-1",
    )

    assert any(call[0] == "remove" for call in docker.calls)
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).all() == []
        assert session.exec(select(ModelConfigDB)).all() == []
        deleting_model = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == "deployment-model"
            )
        ).one()
        assert deleting_model.status == "deleting"
    assert service._claim_heartbeats == {}


def test_legacy_shared_delete_recovers_stale_claim_and_reconciles_absence(
    monkeypatch: pytest.MonkeyPatch,
    deployment_runtime_guard_db,
) -> None:
    lifecycle, engine, _docker = deployment_runtime_guard_db
    if lifecycle != "legacy":
        pytest.skip("shared Xinference recovery is a legacy lifecycle contract")

    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        deployment.status = "running"
        session.add(deployment)
        session.commit()

    runtime_present = True
    runtime_calls: list[str] = []

    class Client:
        def terminate_model(self, model_uid: str) -> None:
            nonlocal runtime_present
            assert model_uid == "runtime-guard-model"
            runtime_calls.append("terminate")
            if runtime_present:
                runtime_present = False
                _add_runtime_dependency(engine, "generation")
                return
            raise RuntimeError("Xinference returned 404")

        def get_model(self, model_uid: str):
            assert model_uid == "runtime-guard-model"
            runtime_calls.append("get-model")
            return None if not runtime_present else {"model_uid": model_uid}

    service = deployment_module.DeploymentService()
    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(ValueError, match="active runtime"):
        service.delete_deployment(
            "deployment-runtime-guard",
            force=True,
            user_id="user-1",
        )

    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        assert deployment.replica_operation_kind == "delete"
        assert deployment.replica_operation_generation == 1
        assert deployment.replica_operation_token
        session.delete(session.exec(select(GenerationTaskDB)).one())
        deployment.replica_operation_heartbeat_at = datetime.now() - timedelta(
            hours=1
        )
        deployment.replica_operation_started_at = datetime.now() - timedelta(
            hours=1
        )
        session.add(deployment)
        session.commit()

    assert service._claim_heartbeats == {}
    assert service.delete_deployment(
        "deployment-runtime-guard",
        force=True,
        user_id="user-1",
    )

    assert runtime_calls == ["terminate", "get-model"]
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).all() == []
        assert session.exec(select(ModelConfigDB)).all() == []
    assert service._claim_heartbeats == {}


@pytest.mark.parametrize(
    "observation",
    ["absent", "present", "unavailable"],
)
def test_legacy_shared_delete_reconciles_uncertain_terminate_result(
    monkeypatch: pytest.MonkeyPatch,
    deployment_runtime_guard_db,
    observation: str,
) -> None:
    lifecycle, engine, _docker = deployment_runtime_guard_db
    if lifecycle != "legacy":
        pytest.skip("shared Xinference recovery is a legacy lifecycle contract")

    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        deployment.status = "running"
        session.add(deployment)
        session.commit()

    runtime_calls: list[str] = []

    class Client:
        @staticmethod
        def terminate_model(model_uid: str) -> None:
            assert model_uid == "runtime-guard-model"
            runtime_calls.append("terminate")
            raise RuntimeError("terminate response unavailable")

        @staticmethod
        def get_model(model_uid: str):
            assert model_uid == "runtime-guard-model"
            runtime_calls.append("get-model")
            if observation == "unavailable":
                raise RuntimeError("runtime registry unavailable")
            if observation == "present":
                return {"model_uid": model_uid}
            return None

    service = deployment_module.DeploymentService()
    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: Client(),
    )

    if observation == "absent":
        assert service.delete_deployment(
            "deployment-runtime-guard",
            force=True,
            user_id="user-1",
        )
        with Session(engine) as session:
            assert session.exec(select(DeploymentDB)).all() == []
            assert session.exec(select(ModelConfigDB)).all() == []
    else:
        with pytest.raises(RuntimeError, match="shared runtime"):
            service.delete_deployment(
                "deployment-runtime-guard",
                force=True,
                user_id="user-1",
            )
        with Session(engine) as session:
            deployment = session.exec(select(DeploymentDB)).one()
            assert deployment.replica_operation_kind == "delete"
            assert deployment.replica_operation_token
            assert session.exec(select(ModelConfigDB)).one()

    assert runtime_calls == ["terminate", "get-model"]
    assert service._claim_heartbeats == {}


@pytest.mark.parametrize("operation", ["stop", "restart"])
@pytest.mark.parametrize("observation", ["absent", "present", "unavailable"])
def test_legacy_shared_lifecycle_reconciles_uncertain_terminate_result(
    monkeypatch: pytest.MonkeyPatch,
    deployment_runtime_guard_db,
    operation: str,
    observation: str,
) -> None:
    lifecycle, engine, docker = deployment_runtime_guard_db
    if lifecycle != "legacy":
        pytest.skip("shared Xinference recovery is a legacy lifecycle contract")

    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        deployment.status = "running"
        session.add(deployment)
        session.commit()

    runtime_calls: list[str] = []

    class Client:
        @staticmethod
        def terminate_model(model_uid: str) -> None:
            assert model_uid == "runtime-guard-model"
            runtime_calls.append("terminate")
            raise RuntimeError("terminate response unavailable")

        @staticmethod
        def get_model(model_uid: str):
            assert model_uid == "runtime-guard-model"
            runtime_calls.append("get-model")
            if observation == "unavailable":
                raise RuntimeError("runtime registry unavailable")
            if observation == "present":
                return {"model_uid": model_uid}
            return None

        @staticmethod
        def launch_model(**_kwargs) -> None:
            runtime_calls.append("launch")

    service = deployment_module.DeploymentService()
    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: Client(),
    )
    monkeypatch.setattr(
        docker,
        "wait_for_model",
        lambda *_args, **_kwargs: True,
        raising=False,
    )

    def call():
        if operation == "stop":
            return service.stop_deployment(
                "deployment-runtime-guard",
                user_id="user-1",
            )
        return service.restart_deployment(
            "deployment-runtime-guard",
            mode="model",
            user_id="user-1",
        )
    if observation == "absent":
        result = call()
        assert result["status"] == ("stopped" if operation == "stop" else "running")
        expected_calls = ["terminate", "get-model"]
        if operation == "restart":
            expected_calls.append("launch")
        assert runtime_calls == expected_calls
    else:
        with pytest.raises(RuntimeError, match="shared runtime"):
            call()
        assert runtime_calls == ["terminate", "get-model"]
        with Session(engine) as session:
            deployment = session.exec(select(DeploymentDB)).one()
            assert deployment.status in {"stopping", "failed"}
    assert service._claim_heartbeats == {}


def test_legacy_shared_start_failure_cleanup_requires_verified_absence(
    monkeypatch: pytest.MonkeyPatch,
    deployment_runtime_guard_db,
) -> None:
    lifecycle, engine, _docker = deployment_runtime_guard_db
    if lifecycle != "legacy":
        pytest.skip("shared Xinference recovery is a legacy lifecycle contract")

    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        deployment.status = "stopped"
        deployment.model_uid = None
        session.add(deployment)
        session.commit()

    runtime_calls: list[str] = []

    class Client:
        @staticmethod
        def terminate_model(model_uid: str) -> None:
            assert model_uid.startswith("deployment-model-")
            runtime_calls.append("terminate")
            raise RuntimeError("terminate response unavailable")

        @staticmethod
        def get_model(model_uid: str):
            assert model_uid.startswith("deployment-model-")
            runtime_calls.append("get-model")
            return None

    service = deployment_module.DeploymentService()
    monkeypatch.setattr(
        service,
        "_start_external_deployment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("launch response unavailable")
        ),
    )
    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(RuntimeError, match="launch response unavailable"):
        service.start_deployment(
            "deployment-runtime-guard",
            user_id="user-1",
        )

    assert runtime_calls == ["terminate", "get-model"]
    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        assert deployment.status == "failed"
    assert service._claim_heartbeats == {}
