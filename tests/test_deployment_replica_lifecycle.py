from __future__ import annotations

import importlib
import inspect
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import requests
import sqlalchemy as sa
from sqlmodel import Session, select

from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.generation_task_entity import GenerationTaskDB
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.entities.milvus_collection_entity import MilvusCollectionDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.services.model_artifact_membership_service import (
    lock_model_artifact_membership,
)


service_module = importlib.import_module("train_factory.deployment.deployment_service")
registry_module = importlib.import_module(
    "train_factory.storage.services.model_registry_service"
)
docker_module = importlib.import_module("train_factory.deployment.docker_deployer")
database_module = importlib.import_module("train_factory.storage.database")
server_module = importlib.import_module("train_factory.api.server")
adapter_module = importlib.import_module("train_factory.deployment.adapter_service")
sglang_client_module = importlib.import_module(
    "train_factory.deployment.sglang_client"
)
vllm_client_module = importlib.import_module("train_factory.deployment.vllm_client")


class _ProbeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FakeLifecycleDocker:
    def __init__(self) -> None:
        self.existing: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.fail_names: set[str] = set()
        self.fail_authoritative_presence_names: set[str] = set()
        self.authoritative_presence_error = (
            "docker container presence could not be verified"
        )
        self.fail_wait_endpoints: set[str] = set()
        self.running: set[str] = set()
        self.running_names: set[str] = set()
        self.fail_running_identity_ids: set[str] = set()
        self.unready_probe_endpoints: set[str] = set()
        self.on_running_identity = None
        self.on_restart = None
        self.on_stop = None
        self.on_remove = None
        self.fail_restart_ids: set[str] = set()
        self.fail_stop_ids: set[str] = set()
        self.fail_remove_ids: set[str] = set()

    def get_managed_container_id(
        self,
        container_name: str,
        *,
        deployment_id: str,
        replica_id: str,
    ) -> str:
        self.calls.append(("inspect", container_name))
        if container_name in self.fail_names:
            raise RuntimeError("foreign container")
        return f"sha256:{replica_id}"

    def get_legacy_managed_container_id(
        self,
        container_name: str,
        *,
        deployment_id: str,
        replica_id: str,
    ) -> str:
        self.calls.append(("inspect", container_name))
        if container_name in self.fail_names:
            raise RuntimeError("foreign container")
        assert deployment_id == "deployment-1"
        assert replica_id == "legacy"
        return "sha256:legacy"

    def container_exists(self, container_name: str) -> bool:
        self.calls.append(("exists", container_name))
        return container_name in self.existing

    def container_exists_authoritative(self, container_name: str) -> bool:
        self.calls.append(("exists-authoritative", container_name))
        if container_name in self.fail_authoritative_presence_names:
            raise RuntimeError(self.authoritative_presence_error)
        return container_name in self.existing

    def container_running_identity(self, container_id: str) -> bool:
        self.calls.append(("running", container_id))
        if container_id in self.fail_running_identity_ids:
            raise RuntimeError("container running inspection timed out")
        if self.on_running_identity is not None:
            callback, self.on_running_identity = self.on_running_identity, None
            callback()
        return container_id.removeprefix("sha256:") in self.running

    def container_running(self, container_name: str) -> bool:
        self.calls.append(("running-name", container_name))
        return container_name in self.running_names

    def restart_container_identity(self, container_id: str):
        self.calls.append(("restart", container_id))
        if self.on_restart is not None:
            callback, self.on_restart = self.on_restart, None
            callback()
        return container_id not in self.fail_restart_ids, "ok"

    def stop_container_identity(self, container_id: str) -> bool:
        self.calls.append(("stop", container_id))
        if self.on_stop is not None:
            callback, self.on_stop = self.on_stop, None
            callback()
        return container_id not in self.fail_stop_ids

    def remove_container_identity(self, container_id: str) -> bool:
        self.calls.append(("remove", container_id))
        if self.on_remove is not None:
            callback, self.on_remove = self.on_remove, None
            callback()
        return container_id not in self.fail_remove_ids

    def wait_for_service(self, endpoint: str, **_kwargs) -> bool:
        self.calls.append(("wait", endpoint))
        return endpoint not in self.fail_wait_endpoints

    def probe_service_ready(
        self,
        endpoint: str,
        *,
        user_id: str | None,
        framework: str,
        timeout: float = 2.0,
    ) -> bool:
        self.calls.append(("probe", endpoint, user_id, framework, timeout))
        return endpoint not in self.unready_probe_endpoints

    def create_vllm_container(self, *, container_name: str, **_kwargs):
        self.calls.append(("create", container_name))
        self.existing.add(container_name)
        return True, "ok", "redacted"

    def create_sglang_container(self, *, container_name: str, **kwargs):
        return self.create_vllm_container(container_name=container_name, **kwargs)


@pytest.fixture
def lifecycle_service(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'lifecycle.db'}")
    ModelArtifactMembershipGateDB.__table__.create(engine)
    ModelRegistryDB.__table__.create(engine)
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)
    ModelConfigDB.__table__.create(engine)
    LoadedAdapterDB.__table__.create(engine)
    ExternalSyncTaskDB.__table__.create(engine)
    ExternalSyncTrainingDB.__table__.create(engine)
    ExternalSyncTrainingTargetDB.__table__.create(engine)
    TrainingTaskDB.__table__.create(engine)
    GenerationTaskDB.__table__.create(engine)
    EvaluationTaskDB.__table__.create(engine)
    MilvusCollectionDB.__table__.create(engine)

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    docker = FakeLifecycleDocker()
    monkeypatch.setattr(service_module, "get_session", get_session)
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    monkeypatch.setattr(
        service_module.model_registry_service,
        "get_model",
        lambda _model_id: {
            "model_id": "model-1",
            "model_name": "Model",
            "model_path": "/app/models/model-1",
            "model_type": "llm",
        },
    )
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.add(
            ModelRegistryDB(
                model_id="model-1",
                model_name="Model",
                model_type="llm",
                model_path="/app/models/model-1",
                source_type="trained",
                status="available",
            )
        )
        parent = DeploymentDB(
            deployment_id="deployment-1",
            model_id="model-1",
            model_uid="served-model",
            deployment_name="group",
            xinference_endpoint="http://127.0.0.1:11000",
            replica=2,
            gpu_memory_utilization=0.8,
            deploy_mode="container",
            container_name="trainfactory-vllm-model-deployme",
            gpu_id=0,
            port=11000,
            inference_framework="vllm",
            config={
                "launch_config": {"framework": "vllm"},
                "replica_schema_version": 1,
            },
            status="stopped",
            user_id="user-1",
        )
        session.add(parent)
        session.commit()
        children = [
            DeploymentReplicaDB(
                replica_id=f"replica-{index}",
                deployment_id=parent.deployment_id,
                replica_index=index,
                container_name=(
                    parent.container_name if index == 0 else f"{parent.container_name}-r1"
                ),
                endpoint=f"http://127.0.0.1:{11000 + index}",
                port=11000 + index,
                gpu_ids=[index],
                status="stopped",
            )
            for index in range(2)
        ]
        session.add_all(children)
        session.commit()
    return service_module.DeploymentService(), docker, engine


def _make_legacy_deployment(
    engine,
    *,
    status: str,
    enable_lora: bool = False,
    framework: str = "vllm",
) -> None:
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {"legacy": True}
        parent.replica = 1
        parent.status = status
        parent.enable_lora = enable_lora
        parent.inference_framework = framework
        parent.container_name = (
            f"trainfactory-{'xf' if framework == 'xinference' else framework}-"
            "model-deployme"
        )
        session.add(parent)
        session.exec(sa.delete(DeploymentReplicaDB))
        session.commit()


def _canonical_vllm_launch_config(**overrides):
    return service_module.parse_launch_config(
        {"framework": "vllm", **overrides}
    ).model_dump(mode="json")


@pytest.mark.parametrize("initial_status", ["failed", "pending"])
@pytest.mark.parametrize("deploy_mode", ["container", "shared"])
@pytest.mark.parametrize("runtime_absent", [False, True])
@pytest.mark.parametrize("operation", ["stop", "delete"])
def test_legacy_failed_or_pending_stop_requires_verified_runtime_cleanup(
    lifecycle_service, monkeypatch, initial_status, deploy_mode, runtime_absent, operation,
):
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status=initial_status, framework="xinference")
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.deploy_mode = deploy_mode
        session.add(parent)
        session.commit()
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(service_module, "gpu_resource_manager", manager)
    runtime_calls = []

    class Client:
        def terminate_model(self, model_uid):
            runtime_calls.append(("terminate", model_uid))
            raise RuntimeError("termination uncertain")

        def get_model(self, model_uid):
            runtime_calls.append(("get", model_uid))
            return None if runtime_absent else {"model_uid": model_uid}

    monkeypatch.setattr(service, "_get_xinference_client", lambda *_args, **_kwargs: Client())
    if not runtime_absent:
        docker.existing.add("trainfactory-xf-model-deployme")
        docker.fail_stop_ids.add("sha256:legacy")
        docker.fail_remove_ids.add("sha256:legacy")

    operate = service.stop_deployment if operation == "stop" else service.delete_deployment

    if runtime_absent:
        result = operate("deployment-1", user_id="user-1")
        assert result["status"] == "stopped" if operation == "stop" else result is True
        assert manager.allocate_gpus_for_task("after-stop", "cuda:0") == "cuda:0"
    else:
        with pytest.raises(RuntimeError):
            operate("deployment-1", user_id="user-1")
        with Session(engine) as session:
            persisted = session.exec(select(DeploymentDB)).one()
            assert persisted.status == ("stopping" if operation == "stop" else initial_status)
        assert manager.allocate_gpus_for_task("after-uncertain-stop", "cuda:0") is None

    if deploy_mode == "container":
        assert ("exists-authoritative", "trainfactory-xf-model-deployme") in docker.calls
    else:
        assert runtime_calls == [("terminate", "served-model"), ("get", "served-model")]


@pytest.mark.parametrize("operation", ["start", "restart"])
def test_local_shared_restart_launches_on_its_reserved_gpu(lifecycle_service, monkeypatch, operation):
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, _docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running" if operation == "restart" else "stopped", framework="xinference")
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.deploy_mode = "shared"
        parent.gpu_id = 1
        parent.config = {**parent.config, "gpu_idx": 0}
        session.add(parent)
        session.commit()
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(service_module, "gpu_resource_manager", manager)
    assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") == "cuda:0"
    launches = []
    client = SimpleNamespace(
        terminate_model=lambda _uid: None,
        launch_model=lambda **kwargs: launches.append(kwargs),
    )
    monkeypatch.setattr(service, "_get_xinference_client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(service, "_get_xinference_model_name", lambda _model: "model")
    monkeypatch.setattr(service, "_create_config_for_deployment", lambda *_args: None)
    monkeypatch.setattr(service, "_sync_configs_from_deployments", lambda *_args: None)
    monkeypatch.setattr(service_module.docker_deployer, "wait_for_model", lambda *_args, **_kwargs: True, raising=False)

    operate = service.restart_deployment if operation == "restart" else service.start_deployment
    operate("deployment-1", user_id="user-1")

    assert len(launches) == 1
    assert launches[0]["gpu_idx"] == 1


def test_shared_launch_persists_runtime_identity_before_external_io(lifecycle_service, monkeypatch):
    service, _docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="stopped", framework="xinference")
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.deploy_mode = "shared"
        parent.model_uid = None
        session.add(parent)
        session.commit()

    class SimulatedProcessExit(BaseException):
        pass

    def launch(_deployment, _model, model_uid):
        with Session(engine) as session:
            persisted = session.exec(select(DeploymentDB)).one()
            assert persisted.model_uid == model_uid
            assert persisted.status == "starting"
        raise SimulatedProcessExit()

    monkeypatch.setattr(service, "_start_external_deployment", launch)
    with pytest.raises(SimulatedProcessExit):
        service.start_deployment("deployment-1", user_id="user-1")


@pytest.mark.parametrize("operation", ["stop", "delete"])
def test_unresolved_shared_runtime_without_identity_cannot_release_gpu(lifecycle_service, monkeypatch, operation):
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, _docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="failed", framework="xinference")
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.deploy_mode = "shared"
        parent.model_uid = None
        session.add(parent)
        session.commit()
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    operate = service.stop_deployment if operation == "stop" else service.delete_deployment
    with pytest.raises(ValueError, match="runtime identity"):
        operate("deployment-1", user_id="user-1")
    assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") is None


def test_start_cannot_use_gpu_reserved_by_training(lifecycle_service, monkeypatch):
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, docker, engine = lifecycle_service
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(service_module, "gpu_resource_manager", manager, raising=False)
    assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") == "cuda:0"

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service.start_replica("deployment-1", "replica-0", user_id="user-1")

    assert not docker.calls
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().replica_operation_token is None


def test_start_claim_excludes_training_before_container_launch(lifecycle_service, monkeypatch):
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, _docker, _engine = lifecycle_service
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(service_module, "gpu_resource_manager", manager, raising=False)
    claim = service._claim_replica_operation(
        "deployment-1", operation="start", replica_id="replica-0", user_id="user-1",
    )
    try:
        assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") is None
        assert manager.allocate_gpus_for_task("other-attempt", "cuda:1") == "cuda:1"
    finally:
        service._release_replica_operation(claim)
    assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") == "cuda:0"


@pytest.mark.parametrize("stop_succeeds", [True, False])
def test_gpu_remains_reserved_until_container_stop_confirmed(lifecycle_service, monkeypatch, stop_succeeds):
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, docker, _engine = lifecycle_service
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(service_module, "gpu_resource_manager", manager)
    service.start_replica("deployment-1", "replica-0", user_id="user-1")
    assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") is None
    if not stop_succeeds:
        docker.fail_stop_ids.add("sha256:replica-0")
        with pytest.raises(RuntimeError):
            service.stop_replica("deployment-1", "replica-0", user_id="user-1")
        assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") is None
    else:
        service.stop_replica("deployment-1", "replica-0", user_id="user-1")
        assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") == "cuda:0"


def test_concurrent_training_and_deployment_have_only_one_gpu_winner(lifecycle_service, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, _docker, _engine = lifecycle_service
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(service_module, "gpu_resource_manager", manager)
    barrier = Barrier(2)

    def train():
        barrier.wait(timeout=5)
        return manager.allocate_gpus_for_task("training-attempt", "cuda:0")

    def deploy():
        barrier.wait(timeout=5)
        try:
            return service._claim_replica_operation(
                "deployment-1", operation="start", replica_id="replica-0", user_id="user-1",
            )
        except service_module.ReplicaOperationBusyError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        training, deployment = executor.submit(train), executor.submit(deploy)
        training_result, claim = training.result(timeout=10), deployment.result(timeout=10)
    try:
        assert sum(result is not None for result in (training_result, claim)) == 1
    finally:
        if claim:
            service._release_replica_operation(claim)


def test_legacy_start_cannot_take_training_gpu(lifecycle_service, monkeypatch):
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="stopped")
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(service_module, "gpu_resource_manager", manager)
    assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") == "cuda:0"
    with pytest.raises(service_module.ReplicaOperationBusyError):
        service.start_deployment("deployment-1", user_id="user-1")
    assert not docker.calls


@pytest.mark.parametrize("auto", [True, False])
def test_local_shared_start_uses_the_same_admission_as_training(lifecycle_service, monkeypatch, auto):
    from train_factory.core.gpu_resource_manager import GPUResourceManager

    service, _docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="stopped", framework="xinference")
    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        deployment.deploy_mode = "shared"
        deployment.xinference_endpoint = "http://xinference:9997"
        deployment.gpu_id = None if auto else 0
        session.add(deployment)
        session.commit()
    monkeypatch.setattr(database_module, "get_session", service_module.get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(service_module, "gpu_resource_manager", manager)
    assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") == "cuda:0"
    if not auto:
        with pytest.raises(service_module.ReplicaOperationBusyError):
            service._claim_replica_operation(
                "deployment-1", operation="start", replica_id=None, user_id="user-1",
            )
        return
    claim = service._claim_replica_operation(
        "deployment-1", operation="start", replica_id=None, user_id="user-1",
    )
    try:
        with Session(engine) as session:
            assert session.exec(select(DeploymentDB)).one().gpu_id == 1
        assert manager.allocate_gpus_for_task("other-training", "cuda:1") is None
    finally:
        service._release_replica_operation(claim)


def test_two_deployments_cannot_claim_the_same_gpu(lifecycle_service):
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        session.add(DeploymentDB(
            deployment_id="22222222-2222-4222-8222-222222222222", model_id="model-1",
            deployment_name="second-group", deploy_mode="container", status="stopped",
            user_id="user-1", gpu_id=0, inference_framework="vllm",
            container_name="trainfactory-vllm-model-22222222",
            xinference_endpoint="http://127.0.0.1:12000",
            config={"replica_schema_version": 1, "launch_config": {"framework": "vllm"}},
        ))
        session.flush()
        session.add(DeploymentReplicaDB(
            deployment_id="22222222-2222-4222-8222-222222222222", replica_id="second-replica",
            replica_index=0, container_name="trainfactory-vllm-model-22222222",
            endpoint="http://127.0.0.1:12000", port=12000, gpu_ids=[0], status="stopped",
        ))
        session.commit()
    first = service._claim_replica_operation(
        "deployment-1", operation="start", replica_id="replica-0", user_id="user-1",
    )
    try:
        with pytest.raises(service_module.ReplicaOperationBusyError):
            service._claim_replica_operation(
                "22222222-2222-4222-8222-222222222222", operation="start",
                replica_id="second-replica", user_id="user-1",
            )
    finally:
        service._release_replica_operation(first)
    second = service._claim_replica_operation(
        "22222222-2222-4222-8222-222222222222", operation="start",
        replica_id="second-replica", user_id="user-1",
    )
    service._release_replica_operation(second)


def test_helpers_bind_replica_to_parent_and_user_before_docker(
    lifecycle_service,
) -> None:
    service, docker, _engine = lifecycle_service

    with pytest.raises(ValueError, match="access denied"):
        service.get_replica("deployment-1", "replica-0", user_id="user-2")
    with pytest.raises(ValueError, match="not found"):
        service.get_replica("deployment-1", "replica-foreign", user_id="user-1")

    assert docker.calls == []


@pytest.mark.parametrize(
    "caller_marker",
    [None, 2, True],
    ids=["null", "unsupported-version", "boolean"],
)
def test_config_update_preserves_canonical_launch_config_and_operator_marker(
    lifecycle_service,
    caller_marker,
) -> None:
    service, _docker, engine = lifecycle_service

    updated = service.update_deployment_config(
        "deployment-1",
        {"caller_value": "kept", "replica_schema_version": caller_marker},
    )

    assert updated["config"] == {
        "caller_value": "kept",
        "launch_config": _canonical_vllm_launch_config(),
        "replica_schema_version": 1,
    }
    assert type(updated["config"]["replica_schema_version"]) is int
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert service._uses_replica_lifecycle(parent) is True


def test_config_update_with_absent_launch_config_preserves_existing(
    lifecycle_service,
) -> None:
    service, _docker, _engine = lifecycle_service

    updated = service.update_deployment_config(
        "deployment-1",
        {"caller_value": "updated"},
    )

    assert updated["config"] == {
        "caller_value": "updated",
        "launch_config": _canonical_vllm_launch_config(),
        "replica_schema_version": 1,
    }


def test_config_update_rejects_explicit_null_launch_config_without_database_write(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        original_config = dict(parent.config)
        original_updated_at = parent.updated_at

    with pytest.raises(ValueError, match="launch_config is invalid"):
        service.update_deployment_config(
            "deployment-1",
            {"caller_value": "must-not-persist", "launch_config": None},
        )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.config == original_config
        assert parent.updated_at == original_updated_at


def test_config_update_preserves_trusted_artifact_and_deferred_markers(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    trusted_signature = {
        "model_id": "model-1",
        "model_path_sha256": "trusted-path",
        "version": "v1",
    }
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {
            "launch_config": {"framework": "vllm"},
            "replica_schema_version": 1,
            "_deferred_auto_start": True,
            "_registry_artifact_signature": trusted_signature,
        }
        session.add(parent)
        session.commit()

    updated = service.update_deployment_config(
        "deployment-1",
        {
            "caller_value": "kept",
            "_deferred_auto_start": False,
            "_registry_artifact_signature": {"model_id": "forged"},
        },
    )

    assert updated["config"] == {
        "caller_value": "kept",
        "launch_config": _canonical_vllm_launch_config(),
        "replica_schema_version": 1,
        "_deferred_auto_start": True,
        "_registry_artifact_signature": trusted_signature,
    }


@pytest.mark.parametrize(
    "broken_config",
    [
        {"replica_schema_version": 1, "keep": "missing"},
        {
            "replica_schema_version": 1,
            "keep": "invalid",
            "launch_config": {"framework": "vllm", "tensor_parallel_size": 0},
        },
    ],
    ids=["missing", "invalid"],
)
def test_config_update_without_launch_config_fails_closed_when_existing_is_broken(
    lifecycle_service,
    broken_config,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = broken_config
        session.add(parent)
        session.commit()

    with pytest.raises(ValueError, match="existing launch_config"):
        service.update_deployment_config("deployment-1", {"caller_value": "new"})

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().config == broken_config


def test_config_update_can_repair_missing_launch_config_when_child_plan_matches(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {"replica_schema_version": 1, "keep": "old"}
        session.add(parent)
        session.commit()

    updated = service.update_deployment_config(
        "deployment-1",
        {
            "caller_value": "repaired",
            "launch_config": {"framework": "vllm", "gpu_pool": [0, 1]},
        },
    )

    assert updated["config"] == {
        "caller_value": "repaired",
        "launch_config": _canonical_vllm_launch_config(gpu_pool=[0, 1]),
        "replica_schema_version": 1,
    }


def test_config_update_uses_child_rows_to_repair_a_missing_lifecycle_marker(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {"keep": "old"}
        session.add(parent)
        session.commit()

    updated = service.update_deployment_config(
        "deployment-1",
        {
            "caller_value": "repaired",
            "launch_config": {"framework": "vllm", "gpu_pool": [0, 1]},
        },
    )

    assert updated["config"] == {
        "caller_value": "repaired",
        "launch_config": _canonical_vllm_launch_config(gpu_pool=[0, 1]),
        "replica_schema_version": 1,
    }


@pytest.mark.parametrize(
    "parallel_sizes",
    [
        {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
        {"tensor_parallel_size": 1, "pipeline_parallel_size": 2},
    ],
    ids=["tensor-parallel", "pipeline-parallel"],
)
def test_config_repair_rejects_ambiguous_multi_gpu_child_topology(
    lifecycle_service,
    parallel_sizes,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {"replica_schema_version": 1, "keep": "old"}
        parent.replica = 1
        replica = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        replica.gpu_ids = [0, 1]
        session.add(parent)
        session.add(replica)
        session.exec(
            sa.delete(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-1"
            )
        )
        session.commit()
        original = dict(parent.config)

    with pytest.raises(ValueError, match="ambiguous|recreate"):
        service.update_deployment_config(
            "deployment-1",
            {
                "caller_value": "must-not-persist",
                "launch_config": {
                    "framework": "vllm",
                    "gpu_pool": [0, 1],
                    **parallel_sizes,
                },
            },
        )

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().config == original


@pytest.mark.parametrize(
    "runtime_only_override",
    [
        {"enable_expert_parallel": True},
        {"enforce_eager": True},
    ],
    ids=["expert-parallel", "enforce-eager"],
)
def test_config_repair_rejects_nonderivable_runtime_only_fields(
    lifecycle_service,
    runtime_only_override,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {"replica_schema_version": 1, "keep": "old"}
        session.add(parent)
        session.commit()
        original = dict(parent.config)

    with pytest.raises(ValueError, match="ambiguous|recreate"):
        service.update_deployment_config(
            "deployment-1",
            {
                "caller_value": "must-not-persist",
                "launch_config": {
                    "framework": "vllm",
                    "gpu_pool": [0, 1],
                    **runtime_only_override,
                },
            },
        )

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().config == original


@pytest.mark.parametrize(
    "launch_config",
    [
        {"framework": "sglang"},
        {"framework": "vllm", "tensor_parallel_size": 2},
        {"framework": "vllm", "pipeline_parallel_size": 2},
        {"framework": "vllm", "data_parallel_size": 2},
        {"framework": "vllm", "gpu_pool": [1, 0]},
        {
            "framework": "vllm",
            "gpu_pool": [0, 1],
            "replica_gpu_overrides": [
                {"replica_index": 0, "gpu_ids": [1]},
                {"replica_index": 1, "gpu_ids": [0]},
            ],
        },
        {"framework": "vllm", "allow_gpu_reuse": True},
    ],
    ids=[
        "framework",
        "tensor-parallel",
        "pipeline-parallel",
        "data-parallel",
        "gpu-pool",
        "replica-overrides",
        "gpu-reuse",
    ],
)
def test_config_update_rejects_child_topology_changes_without_mutating_database(
    lifecycle_service,
    launch_config,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        original = dict(parent.config)

    with pytest.raises(ValueError, match="framework|topology"):
        service.update_deployment_config(
            "deployment-1",
            {"caller_value": "must-not-persist", "launch_config": launch_config},
        )

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().config == original


def test_config_update_rejects_malformed_launch_config_without_mutating_database(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        original = dict(session.exec(select(DeploymentDB)).one().config)

    with pytest.raises(ValueError):
        service.update_deployment_config(
            "deployment-1",
            {
                "caller_value": "must-not-persist",
                "launch_config": {"framework": "vllm", "unexpected": True},
            },
        )

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().config == original


@pytest.mark.parametrize("caller_marker", [1, True], ids=["integer", "boolean"])
def test_legacy_config_update_cannot_enable_replica_lifecycle(
    lifecycle_service,
    caller_marker,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {"legacy": True}
        session.add(parent)
        session.exec(sa.delete(DeploymentReplicaDB))
        session.commit()

    updated = service.update_deployment_config(
        "deployment-1",
        {"caller_value": "kept", "replica_schema_version": caller_marker},
    )

    assert updated["config"] == {"caller_value": "kept"}
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert service._uses_replica_lifecycle(parent) is False


LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY = (
    "_operator_054_legacy_replica_schema_marker_backup"
)
LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_PAYLOAD = {
    "migration_revision": "054_add_deployment_replicas",
    "replica_schema_version": 1,
}


def test_legacy_config_update_cannot_spoof_migration_marker_backup(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="stopped")

    updated = service.update_deployment_config(
        "deployment-1",
        {
            "caller_value": "kept",
            LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY: "forged",
        },
    )

    assert updated["config"] == {"caller_value": "kept"}


@pytest.mark.parametrize(
    ("replacement", "expected_caller_config"),
    [
        (None, {}),
        (
            {
                "caller_value": "kept",
                LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY: "forged",
            },
            {"caller_value": "kept"},
        ),
    ],
    ids=["null-replacement", "forged-replacement"],
)
def test_legacy_config_update_preserves_real_migration_marker_backup(
    lifecycle_service,
    replacement,
    expected_caller_config,
) -> None:
    service, _docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="stopped")
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {
            "legacy": True,
            LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY: (
                LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_PAYLOAD
            ),
        }
        session.add(parent)
        session.commit()

    updated = service.update_deployment_config("deployment-1", replacement)

    assert updated["config"] == {
        **expected_caller_config,
        LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY: (
            LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_PAYLOAD
        ),
    }


def test_legacy_xinference_ignores_stale_replica_schema_marker() -> None:
    deployment = SimpleNamespace(
        inference_framework="xinference",
        config={"replica_schema_version": 1},
    )

    assert service_module.DeploymentService._uses_replica_lifecycle(deployment) is False


def test_xinference_config_patch_ignores_stale_marker_and_child_rows(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.inference_framework = "xinference"
        parent.config = {
            "replica_schema_version": 1,
            "launch_config": {"framework": "vllm"},
        }
        session.add(parent)
        session.commit()

    updated = service.update_deployment_config(
        "deployment-1",
        {"caller_value": "kept"},
    )

    assert updated["config"] == {"caller_value": "kept"}


def test_legacy_start_claimed_path_has_no_unreachable_runtime_implementation() -> None:
    source = inspect.getsource(
        service_module.DeploymentService._start_legacy_deployment_claimed
    )

    assert "intentionally unreachable" not in source


def test_legacy_start_claim_blocks_delete_after_prepare_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="stopped")
    delete_outcomes: list[object] = []
    runtime_events: list[tuple[str, str, bool | None]] = []
    live_containers: set[str] = set()

    def remove_container(name: str) -> bool:
        existed = name in live_containers
        runtime_events.append(("remove", name, existed))
        live_containers.discard(name)
        return True

    def start_runtime(deployment, _model, _model_uid) -> None:
        try:
            delete_outcomes.append(
                service.delete_deployment(
                    deployment.deployment_id,
                    force=True,
                    user_id="user-1",
                )
            )
        except service_module.ReplicaOperationBusyError:
            delete_outcomes.append("busy")
        runtime_events.append(("create", deployment.container_name, None))
        live_containers.add(deployment.container_name)

    monkeypatch.setattr(docker, "remove_container", remove_container, raising=False)
    monkeypatch.setattr(service, "_start_container_deployment", start_runtime)

    start_result = None
    start_error = None
    try:
        start_result = service.start_deployment(
            "deployment-1",
            user_id="user-1",
        )
    except Exception as exc:  # Capture the unsafe interleaving as RED evidence.
        start_error = type(exc).__name__

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).first()
        parent_state = (
            None
            if parent is None
            else (
                parent.status,
                parent.replica_operation_generation,
                parent.replica_operation_token,
            )
        )

    assert {
        "delete_outcomes": delete_outcomes,
        "start_error": start_error,
        "start_status": None if start_result is None else start_result["status"],
        "parent_state": parent_state,
        "runtime_events": runtime_events,
        "heartbeat_registry": service._claim_heartbeats,
    } == {
        "delete_outcomes": ["busy"],
        "start_error": None,
        "start_status": "running",
        "parent_state": ("running", 1, None),
        "runtime_events": [
            ("create", "trainfactory-vllm-model-deployme", None),
        ],
        "heartbeat_registry": {},
    }


def test_legacy_container_start_passes_synthetic_identity_labels(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="stopped")
    create_kwargs: list[dict] = []

    def create_vllm_container(**kwargs):
        create_kwargs.append(kwargs)
        return True, "ok", "redacted"

    monkeypatch.setattr(docker, "create_vllm_container", create_vllm_container)

    assert service.start_deployment(
        "deployment-1",
        user_id="user-1",
    )["status"] == "running"
    assert create_kwargs[0]["deployment_id"] == "deployment-1"
    assert create_kwargs[0]["replica_id"] == "legacy"


def test_legacy_adapter_load_claim_blocks_concurrent_delete_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running", enable_lora=True)
    delete_outcomes: list[object] = []
    runtime_events: list[tuple[str, str]] = []
    active_adapter_sessions = 0

    @contextmanager
    def get_session():
        nonlocal active_adapter_sessions
        active_adapter_sessions += 1
        try:
            with Session(engine) as session:
                yield session
        finally:
            active_adapter_sessions -= 1

    def remove_container(name: str) -> bool:
        runtime_events.append(("remove-container", name))
        return True

    class Client:
        def load_lora_adapter(self, name: str, _path: str) -> None:
            assert active_adapter_sessions == 0
            try:
                delete_outcomes.append(
                    service.delete_deployment(
                        "deployment-1",
                        force=True,
                        user_id="user-1",
                    )
                )
            except service_module.ReplicaOperationBusyError:
                delete_outcomes.append("busy")
            runtime_events.append(("load-adapter", name))

    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(docker, "remove_container", remove_container, raising=False)
    adapter_service = adapter_module.AdapterService(service)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    load_result = adapter_service.load_adapter(
        "deployment-1",
        "legacy-racing-adapter",
        "/external/legacy-racing-adapter",
        user_id="user-1",
    )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).first()
        adapter = session.exec(select(LoadedAdapterDB)).one()
        parent_state = (
            None
            if parent is None
            else (
                parent.status,
                parent.replica_operation_generation,
                parent.replica_operation_token,
            )
        )

    assert {
        "delete_outcomes": delete_outcomes,
        "load_status": load_result["status"],
        "parent_state": parent_state,
        "adapter_state": (adapter.status, adapter.deployment_id),
        "runtime_events": runtime_events,
        "heartbeat_registry": service._claim_heartbeats,
    } == {
        "delete_outcomes": ["busy"],
        "load_status": "loaded",
        "parent_state": ("running", 1, None),
        "adapter_state": ("loaded", "deployment-1"),
        "runtime_events": [("load-adapter", "legacy-racing-adapter")],
        "heartbeat_registry": {},
    }


def test_legacy_stop_claim_blocks_concurrent_start_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running")
    docker.existing.add("trainfactory-vllm-model-deployme")
    start_outcomes: list[object] = []
    runtime_events: list[tuple[str, str]] = []

    def start_runtime(deployment, _model, _model_uid) -> None:
        runtime_events.append(("cross-owner-start", deployment.container_name))

    def stop_container(container_id: str) -> bool:
        runtime_events.append(("stop", container_id))
        try:
            start_outcomes.append(
                service.start_deployment("deployment-1", user_id="user-1")
            )
        except service_module.ReplicaOperationBusyError:
            start_outcomes.append("busy")
        return True

    monkeypatch.setattr(service, "_start_container_deployment", start_runtime)
    monkeypatch.setattr(docker, "stop_container_identity", stop_container)

    result = service.stop_deployment("deployment-1", user_id="user-1")

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent_state = (
            parent.status,
            parent.replica_operation_generation,
            parent.replica_operation_token,
        )
    assert {
        "start_outcomes": start_outcomes,
        "stop_status": result["status"],
        "parent_state": parent_state,
        "runtime_events": runtime_events,
        "heartbeat_registry": service._claim_heartbeats,
    } == {
        "start_outcomes": ["busy"],
        "stop_status": "stopped",
        "parent_state": ("stopped", 1, None),
        "runtime_events": [("stop", "sha256:legacy")],
        "heartbeat_registry": {},
    }


def test_legacy_restart_claim_blocks_delete_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running")
    docker.existing.add("trainfactory-vllm-model-deployme")
    delete_outcomes: list[object] = []
    runtime_events: list[tuple[str, str]] = []

    def remove_container(name: str) -> bool:
        runtime_events.append(("remove", name))
        return True

    def restart_runtime(container_id: str):
        try:
            delete_outcomes.append(
                service.delete_deployment(
                    "deployment-1",
                    force=True,
                    user_id="user-1",
                )
            )
        except service_module.ReplicaOperationBusyError:
            delete_outcomes.append("busy")
        runtime_events.append(("restart", container_id))
        return True, "ok"

    monkeypatch.setattr(docker, "remove_container", remove_container, raising=False)
    monkeypatch.setattr(docker, "restart_container_identity", restart_runtime)

    result = service.restart_deployment(
        "deployment-1",
        mode="auto",
        user_id="user-1",
    )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent_state = (
            parent.status,
            parent.replica_operation_generation,
            parent.replica_operation_token,
        )
    assert {
        "delete_outcomes": delete_outcomes,
        "restart_status": result["status"],
        "parent_state": parent_state,
        "runtime_events": runtime_events,
        "heartbeat_registry": service._claim_heartbeats,
    } == {
        "delete_outcomes": ["busy"],
        "restart_status": "running",
        "parent_state": ("running", 1, None),
        "runtime_events": [("restart", "sha256:legacy")],
        "heartbeat_registry": {},
    }


def test_legacy_adapter_unload_claim_blocks_delete_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running", enable_lora=True)
    with Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                adapter_name="legacy-loaded-adapter",
                adapter_path="/external/legacy-loaded-adapter",
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()
    delete_outcomes: list[object] = []
    runtime_events: list[tuple[str, str]] = []
    active_adapter_sessions = 0

    @contextmanager
    def get_session():
        nonlocal active_adapter_sessions
        active_adapter_sessions += 1
        try:
            with Session(engine) as session:
                yield session
        finally:
            active_adapter_sessions -= 1

    def remove_container(name: str) -> bool:
        runtime_events.append(("remove-container", name))
        return True

    class Client:
        def unload_lora_adapter(self, name: str) -> None:
            assert active_adapter_sessions == 0
            try:
                delete_outcomes.append(
                    service.delete_deployment(
                        "deployment-1",
                        force=True,
                        user_id="user-1",
                    )
                )
            except service_module.ReplicaOperationBusyError:
                delete_outcomes.append("busy")
            runtime_events.append(("unload-adapter", name))

    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(docker, "remove_container", remove_container, raising=False)
    adapter_service = adapter_module.AdapterService(service)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    assert adapter_service.unload_adapter(
        "deployment-1",
        "legacy-loaded-adapter",
        user_id="user-1",
    ) is True

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).first()
        adapter = session.exec(select(LoadedAdapterDB)).one()
        parent_state = (
            None
            if parent is None
            else (
                parent.status,
                parent.replica_operation_generation,
                parent.replica_operation_token,
            )
        )
    assert {
        "delete_outcomes": delete_outcomes,
        "parent_state": parent_state,
        "adapter_state": (adapter.status, adapter.deployment_id),
        "runtime_events": runtime_events,
        "heartbeat_registry": service._claim_heartbeats,
    } == {
        "delete_outcomes": ["busy"],
        "parent_state": ("running", 1, None),
        "adapter_state": ("unloaded", "deployment-1"),
        "runtime_events": [("unload-adapter", "legacy-loaded-adapter")],
        "heartbeat_registry": {},
    }


def test_group_start_visits_every_child_once_and_derives_running(
    lifecycle_service,
) -> None:
    service, docker, _engine = lifecycle_service

    result = service.start_deployment("deployment-1", user_id="user-1")

    assert sorted(call[1] for call in docker.calls if call[0] == "create") == [
        "trainfactory-vllm-model-deployme",
        "trainfactory-vllm-model-deployme-r1",
    ]
    assert result["status"] == "running"
    assert [child["health_status"] for child in result["replica_instances"]] == [
        "HEALTHY",
        "HEALTHY",
    ]


def test_start_replica_is_noop_for_exact_running_healthy_identity(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    name = "trainfactory-vllm-model-deployme"
    docker.existing.add(name)
    docker.running.add("replica-0")
    with Session(engine) as session:
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(child)
        session.commit()

    result = service.start_replica(
        "deployment-1",
        "replica-0",
        user_id="user-1",
    )

    assert result["status"] == "running"
    assert ("inspect", name) in docker.calls
    assert ("running", "sha256:replica-0") in docker.calls
    assert not any(call[0] in {"restart", "create", "wait"} for call in docker.calls)
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().status == "degraded"


@pytest.mark.parametrize("presence_error", ["daemon unavailable", "query timed out"])
def test_running_replica_noop_fails_closed_when_presence_probe_is_unavailable(
    lifecycle_service,
    presence_error: str,
) -> None:
    service, docker, engine = lifecycle_service
    container_name = "trainfactory-vllm-model-deployme"
    with Session(engine) as session:
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(child)
        session.commit()
    docker.existing.add(container_name)
    docker.running.add("replica-0")
    docker.fail_authoritative_presence_names.add(container_name)
    docker.authoritative_presence_error = presence_error

    with pytest.raises(RuntimeError, match="replica lifecycle operation failed"):
        service.start_replica("deployment-1", "replica-0", user_id="user-1")

    assert ("exists-authoritative", container_name) in docker.calls
    assert not any(
        call[0] in {"inspect", "create", "restart", "stop", "remove"}
        for call in docker.calls
    )
    with Session(engine) as session:
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        assert child.status == "failed"
        assert session.exec(select(DeploymentDB)).one().replica_operation_token is None
    assert service._claim_heartbeats == {}


def test_group_start_is_noop_for_all_exact_running_healthy_children(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    names = {
        "trainfactory-vllm-model-deployme",
        "trainfactory-vllm-model-deployme-r1",
    }
    docker.existing.update(names)
    docker.running.update({"replica-0", "replica-1"})
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.health_status = "HEALTHY"
        session.add(parent)
        children = session.exec(select(DeploymentReplicaDB)).all()
        for child in children:
            child.status = "running"
            child.health_status = "HEALTHY"
            session.add(child)
        session.commit()

    result = service.start_deployment("deployment-1", user_id="user-1")

    assert result["status"] == "running"
    assert sorted(call[1] for call in docker.calls if call[0] == "inspect") == sorted(
        names
    )
    assert not any(call[0] in {"restart", "create", "wait"} for call in docker.calls)


def test_per_replica_stop_touches_only_requested_owned_identity(
    lifecycle_service,
) -> None:
    service, docker, _engine = lifecycle_service
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )

    result = service.stop_replica(
        "deployment-1",
        "replica-1",
        user_id="user-1",
    )

    assert ("stop", "sha256:replica-1") in docker.calls
    assert not any(
        call == ("stop", "sha256:replica-0") for call in docker.calls
    )
    assert result["replica_id"] == "replica-1"
    assert result["status"] == "stopped"


def test_recreate_preserves_child_identity_and_replaces_only_owned_container(
    lifecycle_service,
) -> None:
    service, docker, _engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme-r1")

    result = service.recreate_replica(
        "deployment-1",
        "replica-1",
        user_id="user-1",
    )

    assert result["replica_id"] == "replica-1"
    assert result["replica_index"] == 1
    assert ("remove", "sha256:replica-1") in docker.calls
    assert ("create", "trainfactory-vllm-model-deployme-r1") in docker.calls


def _seed_runtime_reset_adapters(engine) -> None:
    preserved_unloaded_at = datetime(2026, 1, 2, 3, 4, 5)
    with Session(engine) as session:
        rows = [
            LoadedAdapterDB(
                adapter_id=f"target-{status}",
                deployment_id="deployment-1",
                deployment_replica_id="replica-0",
                adapter_name=f"target-{status}",
                adapter_path=f"/adapters/target-{status}",
                user_id="user-1",
                status=status,
                error_message="stale runtime error",
                unloaded_at=(preserved_unloaded_at if status == "unloaded" else None),
            )
            for status in ("loading", "loaded", "unloading", "failed", "unloaded")
        ]
        rows.append(
            LoadedAdapterDB(
                adapter_id="sibling-loaded",
                deployment_id="deployment-1",
                deployment_replica_id="replica-1",
                adapter_name="sibling-loaded",
                adapter_path="/adapters/sibling-loaded",
                user_id="user-1",
                status="loaded",
                error_message="sibling error",
            )
        )
        session.add_all(rows)
        session.commit()


def _seed_legacy_runtime_reset_adapters(engine) -> None:
    preserved_unloaded_at = datetime(2026, 1, 2, 3, 4, 5)
    with Session(engine) as session:
        rows = [
            LoadedAdapterDB(
                adapter_id=f"legacy-{status}",
                deployment_id="deployment-1",
                deployment_replica_id=None,
                adapter_name=f"legacy-{status}",
                adapter_path=f"/adapters/legacy-{status}",
                user_id="user-1",
                status=status,
                error_message="stale runtime error",
                unloaded_at=(preserved_unloaded_at if status == "unloaded" else None),
            )
            for status in ("loading", "loaded", "unloading", "failed", "unloaded")
        ]
        rows.append(
            LoadedAdapterDB(
                adapter_id="bound-sibling-loaded",
                deployment_id="deployment-1",
                deployment_replica_id="replica-0",
                adapter_name="bound-sibling-loaded",
                adapter_path="/adapters/bound-sibling-loaded",
                user_id="user-1",
                status="loaded",
                error_message="sibling error",
            )
        )
        session.add_all(rows)
        session.commit()


def _runtime_reset_adapter_states(engine):
    with Session(engine) as session:
        return {
            row.adapter_id: (row.status, row.error_message, row.unloaded_at)
            for row in session.exec(
                select(LoadedAdapterDB).order_by(LoadedAdapterDB.adapter_id)
            ).all()
        }


@pytest.mark.parametrize("operation", ["start", "stop"])
@pytest.mark.parametrize("runtime_exists", [True, False], ids=["existing", "absent"])
def test_start_and_stop_reset_target_adapters_after_runtime_is_reset_or_absent(
    lifecycle_service,
    operation: str,
    runtime_exists: bool,
) -> None:
    service, docker, engine = lifecycle_service
    if runtime_exists:
        docker.existing.add("trainfactory-vllm-model-deployme")
    _seed_runtime_reset_adapters(engine)

    getattr(service, f"{operation}_replica")(
        "deployment-1",
        "replica-0",
        user_id="user-1",
    )

    states = _runtime_reset_adapter_states(engine)
    for status in ("loading", "loaded", "unloading"):
        assert states[f"target-{status}"][0] == "unloaded"
        assert states[f"target-{status}"][1] is None
        assert states[f"target-{status}"][2] is not None
    assert states["target-failed"][:2] == ("failed", "stale runtime error")
    assert states["target-unloaded"] == (
        "unloaded",
        "stale runtime error",
        datetime(2026, 1, 2, 3, 4, 5),
    )
    assert states["sibling-loaded"][:2] == ("loaded", "sibling error")


def test_absent_runtime_start_resets_adapters_before_container_create(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _seed_runtime_reset_adapters(engine)
    state_seen_by_create: list[str] = []
    original_create = docker.create_vllm_container

    def create_vllm_container(**kwargs):
        state_seen_by_create.append(
            _runtime_reset_adapter_states(engine)["target-loaded"][0]
        )
        return original_create(**kwargs)

    monkeypatch.setattr(docker, "create_vllm_container", create_vllm_container)

    service.start_replica("deployment-1", "replica-0", user_id="user-1")

    assert state_seen_by_create == ["unloaded"]


def test_start_runtime_reset_survives_later_readiness_failure(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.fail_wait_endpoints.add("http://127.0.0.1:11000")
    _seed_runtime_reset_adapters(engine)

    with pytest.raises(RuntimeError, match="replica lifecycle operation failed"):
        service.start_replica("deployment-1", "replica-0", user_id="user-1")

    assert _runtime_reset_adapter_states(engine)["target-loaded"][0] == "unloaded"


@pytest.mark.parametrize("operation", ["restart", "recreate"])
def test_runtime_reset_unloads_only_active_adapters_bound_to_target_replica(
    lifecycle_service,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    _seed_runtime_reset_adapters(engine)

    getattr(service, f"{operation}_replica")(
        "deployment-1",
        "replica-0",
        user_id="user-1",
    )

    states = _runtime_reset_adapter_states(engine)
    for status in ("loading", "loaded", "unloading"):
        reset_status, error, unloaded_at = states[f"target-{status}"]
        assert reset_status == "unloaded"
        assert error is None
        assert unloaded_at is not None
    assert states["target-failed"][:2] == ("failed", "stale runtime error")
    assert states["target-unloaded"] == (
        "unloaded",
        "stale runtime error",
        datetime(2026, 1, 2, 3, 4, 5),
    )
    assert states["sibling-loaded"][:2] == ("loaded", "sibling error")


@pytest.mark.parametrize("operation", ["restart", "recreate"])
def test_runtime_reset_state_survives_later_readiness_failure(
    lifecycle_service,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.fail_wait_endpoints.add("http://127.0.0.1:11000")
    _seed_runtime_reset_adapters(engine)

    with pytest.raises(RuntimeError, match="replica lifecycle operation failed"):
        getattr(service, f"{operation}_replica")(
            "deployment-1",
            "replica-0",
            user_id="user-1",
        )

    assert _runtime_reset_adapter_states(engine)["target-loaded"][0] == "unloaded"


@pytest.mark.parametrize("operation", ["start", "stop", "restart", "recreate"])
def test_failed_runtime_mutation_does_not_change_adapter_state(
    lifecycle_service,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    if operation in ("start", "restart"):
        docker.fail_restart_ids.add("sha256:replica-0")
    elif operation == "stop":
        docker.fail_stop_ids.add("sha256:replica-0")
    else:
        docker.fail_remove_ids.add("sha256:replica-0")
    _seed_runtime_reset_adapters(engine)

    with pytest.raises(RuntimeError, match="replica lifecycle operation failed"):
        getattr(service, f"{operation}_replica")(
            "deployment-1",
            "replica-0",
            user_id="user-1",
        )

    assert _runtime_reset_adapter_states(engine)["target-loaded"][:2] == (
        "loaded",
        "stale runtime error",
    )


def test_recreate_marks_adapters_unloaded_when_old_container_is_already_absent(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    _seed_runtime_reset_adapters(engine)

    service.recreate_replica("deployment-1", "replica-0", user_id="user-1")

    assert _runtime_reset_adapter_states(engine)["target-loaded"][0] == "unloaded"


@pytest.mark.parametrize("operation", ["start", "stop", "restart", "recreate"])
def test_worker_that_loses_claim_after_runtime_reset_cannot_update_adapters(
    lifecycle_service,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    _seed_runtime_reset_adapters(engine)

    def replace_claim() -> None:
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "replacement-token"
            parent.replica_operation_generation += 1
            session.add(parent)
            session.commit()

    if operation in ("start", "restart"):
        docker.on_restart = replace_claim
    elif operation == "stop":
        docker.on_stop = replace_claim
    else:
        docker.on_remove = replace_claim

    with pytest.raises(
        service_module.ReplicaOperationLostError,
        match="ownership was lost",
    ):
        getattr(service, f"{operation}_replica")(
            "deployment-1",
            "replica-0",
            user_id="user-1",
        )

    assert _runtime_reset_adapter_states(engine)["target-loaded"][:2] == (
        "loaded",
        "stale runtime error",
    )


@pytest.mark.parametrize("operation", ["start", "stop", "restart", "recreate"])
def test_stale_recovery_resets_only_target_adapters_after_runtime_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    container_name = "trainfactory-vllm-model-deployme"
    docker.existing.add(container_name)
    if operation in {"start", "restart"}:
        docker.running.add("replica-0")
    _seed_runtime_reset_adapters(engine)
    observed_now = datetime(2026, 8, 25, 12, 0, 0)
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)
    monkeypatch.setattr(
        service,
        "_list_runtime_adapters_for_endpoint",
        lambda **_kwargs: [],
        raising=False,
    )

    with Session(engine) as session:
        if operation != "start":
            replica = session.exec(
                select(DeploymentReplicaDB).where(
                    DeploymentReplicaDB.replica_id == "replica-0"
                )
            ).one()
            replica.status = "running"
            replica.health_status = "HEALTHY"
            session.add(replica)
            session.commit()

    def lose_claim_after_runtime_side_effect() -> None:
        if operation == "recreate":
            docker.existing.discard(container_name)
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "replacement-stale-token"
            parent.replica_operation_generation += 1
            parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
            parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
            session.add(parent)
            session.commit()

    if operation in {"start", "restart"}:
        docker.on_restart = lose_claim_after_runtime_side_effect
    elif operation == "stop":
        docker.on_stop = lose_claim_after_runtime_side_effect
    else:
        docker.on_remove = lose_claim_after_runtime_side_effect

    with pytest.raises(service_module.ReplicaOperationLostError):
        getattr(service, f"{operation}_replica")(
            "deployment-1",
            "replica-0",
            user_id="user-1",
        )
    assert _runtime_reset_adapter_states(engine)["target-loaded"][0] == "loaded"

    replacement = service._claim_replica_operation(
        "deployment-1",
        operation="stop",
        replica_id="replica-1",
        user_id="user-1",
    )

    states = _runtime_reset_adapter_states(engine)
    assert states["target-loaded"][0] == "unloaded"
    assert states["target-loaded"][1] is None
    assert states["target-loaded"][2] is not None
    assert states["sibling-loaded"][:2] == ("loaded", "sibling error")
    assert service._release_replica_operation(replacement) is True
    assert service._claim_heartbeats == {}


@pytest.mark.parametrize(
    ("operation", "transitional_status"),
    [("start", "starting"), ("restart", "restarting")],
)
def test_stale_running_recovery_preserves_exact_runtime_adapter_identity(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    operation: str,
    transitional_status: str,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.running.add("replica-0")
    _seed_runtime_reset_adapters(engine)
    observed_now = datetime(2026, 8, 25, 12, 0, 0)
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)
    monkeypatch.setattr(
        service,
        "_list_runtime_adapters_for_endpoint",
        lambda **_kwargs: [
            {
                "name": "target-loaded",
                "path": "/adapters/target-loaded",
            }
        ],
        raising=False,
    )
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation=operation,
        replica_id="replica-0",
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        replica = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        replica.status = transitional_status
        session.add(parent)
        session.add(replica)
        session.commit()

    assert service._recover_stale_replica_operation(
        "deployment-1",
        user_id="user-1",
        observed_at=observed_now,
    )

    states = _runtime_reset_adapter_states(engine)
    assert states["target-loaded"][:2] == ("loaded", "stale runtime error")
    assert states["target-loading"][0] == "unloaded"
    assert states["target-unloading"][0] == "unloaded"
    assert states["sibling-loaded"][:2] == ("loaded", "sibling error")
    assert service._claim_heartbeats == {}


def test_stale_running_recovery_fails_closed_when_adapter_listing_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.running.add("replica-0")
    _seed_runtime_reset_adapters(engine)
    observed_now = datetime(2026, 8, 25, 12, 0, 0)
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    def unavailable(**_kwargs):
        raise RuntimeError("adapter listing unavailable")

    monkeypatch.setattr(
        service,
        "_list_runtime_adapters_for_endpoint",
        unavailable,
        raising=False,
    )
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation="restart",
        replica_id="replica-0",
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        replica = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        replica.status = "restarting"
        session.add(parent)
        session.add(replica)
        session.commit()

    assert (
        service._recover_stale_replica_operation(
            "deployment-1",
            user_id="user-1",
            observed_at=observed_now,
        )
        is False
    )
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token == old_claim.token
        assert parent.replica_operation_generation == old_claim.generation
    assert _runtime_reset_adapter_states(engine)["target-loaded"][:2] == (
        "loaded",
        "stale runtime error",
    )
    assert service._claim_heartbeats == {}


def test_stale_recovery_adapter_reset_is_fenced_by_recovery_claim_cas(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.running.add("replica-0")
    _seed_runtime_reset_adapters(engine)
    observed_now = datetime(2026, 8, 25, 12, 0, 0)
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation="restart",
        replica_id="replica-0",
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        replica = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        replica.status = "restarting"
        session.add(parent)
        session.add(replica)
        session.commit()

    def replace_claim_during_runtime_observation() -> None:
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "newer-owner-token"
            parent.replica_operation_generation += 1
            session.add(parent)
            session.commit()

    docker.on_running_identity = replace_claim_during_runtime_observation

    assert (
        service._recover_stale_replica_operation(
            "deployment-1",
            user_id="user-1",
            observed_at=observed_now,
        )
        is False
    )
    states = _runtime_reset_adapter_states(engine)
    assert states["target-loaded"][:2] == ("loaded", "stale runtime error")
    assert states["sibling-loaded"][:2] == ("loaded", "sibling error")


def test_legacy_stop_retry_after_stale_recovery_resets_unbound_adapters(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(
        engine,
        status="running",
        enable_lora=True,
        framework="vllm",
    )
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.running.add("legacy")
    _seed_legacy_runtime_reset_adapters(engine)
    observed_now = datetime(2026, 8, 25, 12, 0, 0)
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    def lose_claim_after_stop() -> None:
        docker.running.discard("legacy")
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "replacement-stale-token"
            parent.replica_operation_generation += 1
            parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
            parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
            session.add(parent)
            session.commit()

    docker.on_stop = lose_claim_after_stop

    with pytest.raises(service_module.ReplicaOperationLostError):
        service.stop_deployment("deployment-1", user_id="user-1")
    assert _runtime_reset_adapter_states(engine)["legacy-loaded"][0] == "loaded"

    result = service.stop_deployment("deployment-1", user_id="user-1")

    states = _runtime_reset_adapter_states(engine)
    assert result["status"] == "stopped"
    assert states["legacy-loaded"][0] == "unloaded"
    assert states["legacy-loaded"][1] is None
    assert states["legacy-loaded"][2] is not None
    assert states["bound-sibling-loaded"][:2] == ("loaded", "sibling error")
    assert service._claim_heartbeats == {}


def test_group_restart_resets_active_adapters_for_each_restarted_child(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )
    _seed_runtime_reset_adapters(engine)

    service.restart_deployment("deployment-1", user_id="user-1")

    states = _runtime_reset_adapter_states(engine)
    assert states["target-loaded"][0] == "unloaded"
    assert states["sibling-loaded"][0] == "unloaded"


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_legacy_container_lifecycle_resets_only_unbound_active_adapters(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    framework: str,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(
        engine,
        status="stopped" if operation == "start" else "running",
        enable_lora=True,
        framework=framework,
    )
    container_name = f"trainfactory-{framework}-model-deployme"
    docker.existing.add(container_name)
    _seed_legacy_runtime_reset_adapters(engine)
    if operation == "start":
        monkeypatch.setattr(
            service,
            "_start_container_deployment",
            lambda *_args, **_kwargs: None,
        )

    getattr(service, f"{operation}_deployment")(
        "deployment-1",
        user_id="user-1",
    )

    states = _runtime_reset_adapter_states(engine)
    for status in ("loading", "loaded", "unloading"):
        reset_status, error, unloaded_at = states[f"legacy-{status}"]
        assert reset_status == "unloaded"
        assert error is None
        assert unloaded_at is not None
    assert states["legacy-failed"][:2] == ("failed", "stale runtime error")
    assert states["legacy-unloaded"] == (
        "unloaded",
        "stale runtime error",
        datetime(2026, 1, 2, 3, 4, 5),
    )
    assert states["bound-sibling-loaded"][:2] == ("loaded", "sibling error")


@pytest.mark.parametrize("operation", ["start", "stop"])
def test_legacy_absent_container_resets_adapters_before_create_or_stop_complete(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    operation: str,
) -> None:
    service, _docker, engine = lifecycle_service
    _make_legacy_deployment(
        engine,
        status="stopped" if operation == "start" else "running",
        enable_lora=True,
    )
    _seed_legacy_runtime_reset_adapters(engine)
    state_seen_by_create: list[str] = []
    if operation == "start":
        monkeypatch.setattr(
            service,
            "_start_container_deployment",
            lambda *_args, **_kwargs: state_seen_by_create.append(
                _runtime_reset_adapter_states(engine)["legacy-loaded"][0]
            ),
        )

    getattr(service, f"{operation}_deployment")(
        "deployment-1",
        user_id="user-1",
    )

    assert _runtime_reset_adapter_states(engine)["legacy-loaded"][0] == "unloaded"
    if operation == "start":
        assert state_seen_by_create == ["unloaded"]


def test_legacy_restart_reset_survives_later_readiness_failure(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running", enable_lora=True)
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.fail_wait_endpoints.add("http://127.0.0.1:11000")
    _seed_legacy_runtime_reset_adapters(engine)

    with pytest.raises(RuntimeError, match="service did not restart in time"):
        service.restart_deployment("deployment-1", user_id="user-1")

    assert _runtime_reset_adapter_states(engine)["legacy-loaded"][0] == "unloaded"


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_failed_legacy_runtime_mutation_does_not_change_adapter_state(
    lifecycle_service,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(
        engine,
        status="stopped" if operation == "start" else "running",
        enable_lora=True,
    )
    docker.existing.add("trainfactory-vllm-model-deployme")
    _seed_legacy_runtime_reset_adapters(engine)
    if operation == "start":
        docker.fail_remove_ids.add("sha256:legacy")
    elif operation == "stop":
        docker.fail_stop_ids.add("sha256:legacy")
    else:
        docker.fail_restart_ids.add("sha256:legacy")

    with pytest.raises(RuntimeError):
        getattr(service, f"{operation}_deployment")(
            "deployment-1",
            user_id="user-1",
        )

    assert _runtime_reset_adapter_states(engine)["legacy-loaded"][:2] == (
        "loaded",
        "stale runtime error",
    )


def test_legacy_worker_that_loses_claim_after_restart_cannot_reset_adapters(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running", enable_lora=True)
    docker.existing.add("trainfactory-vllm-model-deployme")
    _seed_legacy_runtime_reset_adapters(engine)

    def replace_claim() -> None:
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "replacement-token"
            parent.replica_operation_generation += 1
            session.add(parent)
            session.commit()

    docker.on_restart = replace_claim

    with pytest.raises(
        service_module.ReplicaOperationLostError,
        match="ownership was lost",
    ):
        service.restart_deployment("deployment-1", user_id="user-1")

    assert _runtime_reset_adapter_states(engine)["legacy-loaded"][:2] == (
        "loaded",
        "stale runtime error",
    )


def test_xinference_container_stop_does_not_apply_legacy_lora_reset(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(
        engine,
        status="running",
        enable_lora=True,
        framework="xinference",
    )
    docker.existing.add("trainfactory-xf-model-deployme")
    _seed_legacy_runtime_reset_adapters(engine)

    service.stop_deployment("deployment-1", user_id="user-1")

    assert _runtime_reset_adapter_states(engine)["legacy-loaded"][:2] == (
        "loaded",
        "stale runtime error",
    )


@pytest.mark.parametrize("presence_error", ["daemon unavailable", "query timed out"])
@pytest.mark.parametrize("operation", ["start", "stop", "restart", "recreate"])
def test_replica_lifecycle_fails_closed_when_presence_probe_is_unavailable(
    lifecycle_service,
    operation: str,
    presence_error: str,
) -> None:
    service, docker, engine = lifecycle_service
    container_name = "trainfactory-vllm-model-deployme"
    docker.existing.add(container_name)
    docker.fail_authoritative_presence_names.add(container_name)
    docker.authoritative_presence_error = presence_error

    method = getattr(service, f"{operation}_replica")
    with pytest.raises(RuntimeError, match="replica lifecycle operation failed"):
        method("deployment-1", "replica-0", user_id="user-1")

    assert ("exists-authoritative", container_name) in docker.calls
    assert not any(
        call[0] in {"inspect", "create", "restart", "stop", "remove"}
        for call in docker.calls
    )
    with Session(engine) as session:
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        parent = session.exec(select(DeploymentDB)).one()
        assert child.status == "failed"
        assert parent.replica_operation_token is None
    assert service._claim_heartbeats == {}


def test_failed_new_replica_readiness_removes_only_the_new_owned_container(
    lifecycle_service,
) -> None:
    service, docker, _engine = lifecycle_service
    docker.fail_wait_endpoints.add("http://127.0.0.1:11001")

    with pytest.raises(RuntimeError, match="replica lifecycle operation failed"):
        service.start_replica("deployment-1", "replica-1", user_id="user-1")

    assert ("create", "trainfactory-vllm-model-deployme-r1") in docker.calls
    assert ("remove", "sha256:replica-1") in docker.calls
    assert not any(call == ("remove", "sha256:replica-0") for call in docker.calls)


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ([('running', 'HEALTHY'), ('running', 'HEALTHY')], "running"),
        ([('stopped', 'UNKNOWN'), ('stopped', 'UNKNOWN')], "stopped"),
        ([('failed', 'UNHEALTHY'), ('failed', 'UNHEALTHY')], "failed"),
        ([('running', 'HEALTHY'), ('stopped', 'UNKNOWN')], "degraded"),
    ],
)
def test_group_status_derivation_is_exact(states, expected) -> None:
    replicas = [
        {"status": status, "health_status": health}
        for status, health in states
    ]

    assert service_module.DeploymentService.derive_group_status(replicas) == expected


@pytest.mark.parametrize("presence_error", ["daemon unavailable", "query timed out"])
def test_canonical_status_sync_preserves_state_when_presence_probe_is_unavailable(
    lifecycle_service,
    presence_error: str,
) -> None:
    service, docker, engine = lifecycle_service
    names = {
        "trainfactory-vllm-model-deployme",
        "trainfactory-vllm-model-deployme-r1",
    }
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        session.add(parent)
        for child in session.exec(select(DeploymentReplicaDB)).all():
            child.status = "running"
            child.health_status = "HEALTHY"
            session.add(child)
        session.commit()
    docker.fail_authoritative_presence_names.update(names)
    docker.authoritative_presence_error = presence_error

    result = service._sync_replica_group_status("deployment-1")

    assert result["status"] == "running"
    assert {
        (child["status"], child["health_status"])
        for child in result["replica_instances"]
    } == {("running", "HEALTHY")}
    assert not any(call[0] == "inspect" for call in docker.calls)


@pytest.mark.parametrize("presence_error", ["daemon unavailable", "query timed out"])
def test_legacy_auto_sync_preserves_state_when_presence_probe_is_unavailable(
    lifecycle_service,
    presence_error: str,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running")
    container_name = "trainfactory-vllm-model-deployme"
    docker.fail_authoritative_presence_names.add(container_name)
    docker.authoritative_presence_error = presence_error

    assert service.sync_all_running() == 0

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == "running"
        assert parent.error_message is None
    assert ("exists-authoritative", container_name) in docker.calls
    assert not any(call[0] == "inspect" for call in docker.calls)


@pytest.mark.parametrize("status", ["running", "starting", "restarting", "stopping"])
@pytest.mark.parametrize("runtime_failure", ["running-timeout", "foreign-identity"])
def test_legacy_auto_sync_requires_immutable_runtime_observation(
    lifecycle_service,
    status: str,
    runtime_failure: str,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status=status)
    container_name = "trainfactory-vllm-model-deployme"
    docker.existing.add(container_name)
    if runtime_failure == "running-timeout":
        docker.fail_running_identity_ids.add("sha256:legacy")
    else:
        docker.fail_names.add(container_name)

    assert service.sync_all_running() == 0

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == status
        assert parent.error_message is None
    assert ("exists-authoritative", container_name) in docker.calls
    assert ("inspect", container_name) in docker.calls
    assert not any(call[0] == "running-name" for call in docker.calls)


def test_failed_child_is_not_overwritten_by_healthy_sibling(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.fail_names.add("trainfactory-vllm-model-deployme-r1")
    docker.existing.add("trainfactory-vllm-model-deployme-r1")

    with pytest.raises(RuntimeError, match="replica group operation failed"):
        service.start_deployment("deployment-1", user_id="user-1")

    with Session(engine) as session:
        children = session.exec(
            select(DeploymentReplicaDB).order_by(DeploymentReplicaDB.replica_index)
        ).all()
        assert children[0].status == "running"
        assert children[1].status == "failed"
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == "degraded"


def test_group_stop_and_restart_visit_every_child_in_order(
    lifecycle_service,
) -> None:
    service, docker, _engine = lifecycle_service
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )

    stopped = service.stop_deployment("deployment-1", user_id="user-1")
    restarted = service.restart_deployment("deployment-1", user_id="user-1")

    assert [call for call in docker.calls if call[0] == "stop"] == [
        ("stop", "sha256:replica-0"),
        ("stop", "sha256:replica-1"),
    ]
    assert [call for call in docker.calls if call[0] == "restart"] == [
        ("restart", "sha256:replica-0"),
        ("restart", "sha256:replica-1"),
    ]
    assert stopped["status"] == "stopped"
    assert restarted["status"] == "running"


def test_group_delete_removes_each_owned_child_before_database_rows(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )

    assert service.delete_deployment(
        "deployment-1",
        force=True,
        user_id="user-1",
    )

    assert [call for call in docker.calls if call[0] == "remove"] == [
        ("remove", "sha256:replica-0"),
        ("remove", "sha256:replica-1"),
    ]
    with Session(engine) as session:
        assert session.exec(select(DeploymentReplicaDB)).all() == []
        assert session.exec(select(DeploymentDB)).all() == []


@pytest.mark.parametrize("presence_error", ["daemon unavailable", "query timed out"])
def test_group_delete_stops_at_unavailable_presence_probe_without_deleting_rows(
    lifecycle_service,
    presence_error: str,
) -> None:
    service, docker, engine = lifecycle_service
    first_name = "trainfactory-vllm-model-deployme"
    second_name = "trainfactory-vllm-model-deployme-r1"
    docker.existing.update({first_name, second_name})
    docker.fail_authoritative_presence_names.add(first_name)
    docker.authoritative_presence_error = presence_error

    with pytest.raises(RuntimeError, match="replica group cleanup failed"):
        service.delete_deployment(
            "deployment-1",
            force=True,
            user_id="user-1",
        )

    assert [call for call in docker.calls if call[0].startswith("exists")] == [
        ("exists-authoritative", first_name)
    ]
    assert not any(
        call[0] in {"inspect", "remove", "stop", "restart", "create"}
        for call in docker.calls
    )
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token is None
        assert len(session.exec(select(DeploymentReplicaDB)).all()) == 2
    assert service._claim_heartbeats == {}


def test_successful_repeated_group_delete_stops_and_removes_heartbeat_entry(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, _engine = lifecycle_service
    stopped: list[str] = []
    joined: list[float | None] = []
    tokens: list[str] = []

    def register_without_thread(claim) -> None:
        tokens.append(claim.token)
        service._claim_heartbeats[claim.token] = (
            SimpleNamespace(set=lambda: stopped.append(claim.token)),
            SimpleNamespace(join=lambda timeout=None: joined.append(timeout)),
        )

    monkeypatch.setattr(service, "_register_claim_heartbeat", register_without_thread)

    assert service.delete_deployment(
        "deployment-1",
        force=True,
        user_id="user-1",
    ) is True
    assert service.delete_deployment(
        "deployment-1",
        force=True,
        user_id="user-1",
    ) is False
    assert stopped == tokens
    assert joined == [1.0]
    assert service._claim_heartbeats == {}


@pytest.mark.parametrize("consumer_kind", ["task", "target"])
@pytest.mark.parametrize("binding_kind", ["parent", "child"])
def test_group_delete_rejects_external_sync_bindings_before_docker_side_effects(
    lifecycle_service,
    consumer_kind: str,
    binding_kind: str,
) -> None:
    service, docker, engine = lifecycle_service
    binding = (
        {"base_deployment_id": "deployment-1"}
        if binding_kind == "parent"
        else {"base_deployment_replica_id": "replica-1"}
    )
    with Session(engine) as session:
        task = ExternalSyncTaskDB(
            task_id="sync-consumer",
            task_name="consumer",
            user_id="user-1",
            external_api_url="https://sync.example.com/data",
            **(binding if consumer_kind == "task" else {}),
        )
        session.add(task)
        session.commit()
        if consumer_kind == "target":
            session.add(
                ExternalSyncTrainingTargetDB(
                    target_id="target-consumer",
                    task_id=task.task_id,
                    target_name="consumer",
                    **binding,
                )
            )
            session.commit()

    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )

    with pytest.raises(ValueError, match="external sync"):
        service.delete_deployment(
            "deployment-1",
            force=True,
            user_id="user-1",
        )

    assert docker.calls == []
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().deployment_id == "deployment-1"
        assert len(session.exec(select(DeploymentReplicaDB)).all()) == 2


def test_model_registry_cleanup_walks_children_not_parent_mirror(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    _service, docker, engine = lifecycle_service
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )
    monkeypatch.setattr(docker_module, "docker_deployer", docker)
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    with Session(engine) as session:
        deployment = session.exec(select(DeploymentDB)).one()
        deployment.replica_operation_token = "registry-delete-token"
        deployment.replica_operation_kind = "delete"
        deployment.replica_operation_generation = 3
        session.add(deployment)
        session.commit()
        replicas = session.exec(
            select(DeploymentReplicaDB).order_by(DeploymentReplicaDB.replica_index)
        ).all()
        registry_module._remove_deployment_container(
            deployment,
            "model-1",
            replicas,
            service_module.ReplicaOperationClaim(
                deployment_id=deployment.deployment_id,
                token="registry-delete-token",
                generation=3,
                operation="delete",
                replica_id=None,
            ),
        )

    assert [call for call in docker.calls if call[0] == "remove"] == [
        ("remove", "sha256:replica-0"),
        ("remove", "sha256:replica-1"),
    ]


def test_startup_orphan_cleanup_keeps_every_legitimate_child(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    _service, docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        session.add(parent)
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    observed: list[set[str]] = []
    monkeypatch.setattr(database_module, "get_session", get_session)
    monkeypatch.setattr(docker_module, "docker_deployer", docker)
    monkeypatch.setattr(
        docker,
        "cleanup_orphan_containers",
        lambda valid: observed.append(set(valid)) or 0,
        raising=False,
    )

    server_module.cleanup_orphan_containers()

    assert observed == [
        {
            "trainfactory-vllm-model-deployme",
            "trainfactory-vllm-model-deployme-r1",
        }
    ]


def test_lifecycle_claim_is_atomic_across_independent_sessions(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service

    first = service._claim_replica_operation(
        "deployment-1",
        operation="start",
        replica_id="replica-0",
        user_id="user-1",
    )
    with pytest.raises(service_module.ReplicaOperationBusyError):
        service._claim_replica_operation(
            "deployment-1",
            operation="stop",
            replica_id="replica-1",
            user_id="user-1",
        )

    assert service._release_replica_operation(first) is True
    second = service._claim_replica_operation(
        "deployment-1",
        operation="stop",
        replica_id="replica-1",
        user_id="user-1",
    )
    assert second.generation == first.generation + 1
    assert service._release_replica_operation(first) is False
    assert service._release_replica_operation(second) is True

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token is None
        assert parent.replica_operation_generation == second.generation


def test_group_operation_claims_once_for_all_children(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, _engine = lifecycle_service
    original = service._claim_replica_operation
    claims: list[tuple[str, str | None]] = []

    def recording_claim(*args, **kwargs):
        claims.append((kwargs["operation"], kwargs.get("replica_id")))
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_claim_replica_operation", recording_claim)

    service.start_deployment("deployment-1", user_id="user-1")

    assert claims == [("start", None)]


def test_group_operation_stops_after_first_child_loses_fence(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, _engine = lifecycle_service
    called: list[str] = []

    def lose_on_first(
        _deployment_id: str,
        replica_id: str,
        **_kwargs,
    ) -> None:
        called.append(replica_id)
        if replica_id == "replica-0":
            raise service_module.ReplicaOperationLostError("lost")

    monkeypatch.setattr(service, "stop_replica", lose_on_first)

    with pytest.raises(service_module.ReplicaOperationLostError, match="lost"):
        service._operate_replica_group(
            "deployment-1",
            "stop",
            "user-1",
        )

    assert called == ["replica-0"]


def test_stop_replica_rechecks_fence_immediately_before_docker_mutation(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    original_get_id = docker.get_managed_container_id

    def get_id_then_replace_claim(*args, **kwargs):
        container_id = original_get_id(*args, **kwargs)
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "replacement-token"
            parent.replica_operation_generation += 1
            session.add(parent)
            session.commit()
        return container_id

    monkeypatch.setattr(docker, "get_managed_container_id", get_id_then_replace_claim)

    with pytest.raises(service_module.ReplicaOperationLostError):
        service.stop_replica(
            "deployment-1",
            "replica-0",
            user_id="user-1",
        )

    assert not any(call[0] == "stop" for call in docker.calls)


def test_stop_replica_renews_claim_before_docker_mutation(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.add("trainfactory-vllm-model-deployme")
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)
    original_get_id = docker.get_managed_container_id
    takeover_results: list[str] = []

    def get_id_then_expire_claim(*args, **kwargs):
        container_id = original_get_id(*args, **kwargs)
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
            session.add(parent)
            session.commit()
        return container_id

    def stop_after_takeover_attempt(container_id: str) -> bool:
        try:
            replacement = service._claim_replica_operation(
                "deployment-1",
                operation="start",
                replica_id=None,
                user_id="user-1",
            )
        except service_module.ReplicaOperationBusyError:
            takeover_results.append("busy")
        else:
            takeover_results.append("claimed")
            service._release_replica_operation(replacement)
        docker.calls.append(("stop", container_id))
        return True

    monkeypatch.setattr(docker, "get_managed_container_id", get_id_then_expire_claim)
    monkeypatch.setattr(docker, "stop_container_identity", stop_after_takeover_attempt)

    result = service.stop_replica(
        "deployment-1",
        "replica-0",
        user_id="user-1",
    )

    assert result["status"] == "stopped"
    assert takeover_results == ["busy"]
    assert [call for call in docker.calls if call[0] == "stop"] == [
        ("stop", "sha256:replica-0")
    ]


def test_residual_operation_claim_is_never_stolen(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    claim = service._claim_replica_operation(
        "deployment-1",
        operation="restart",
        replica_id=None,
        user_id="user-1",
    )
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_started_at = datetime(2000, 1, 1)
        session.add(parent)
        session.commit()

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service._claim_replica_operation(
            "deployment-1",
            operation="start",
            replica_id=None,
            user_id="user-1",
        )

    assert service._release_replica_operation(claim) is True


def test_stale_operation_claim_is_reconciled_before_single_worker_reclaims(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    claim = service._claim_replica_operation(
        "deployment-1",
        operation="restart",
        replica_id=None,
        user_id="user-1",
    )
    observed_now = datetime(2026, 8, 23, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )
    docker.running.add("replica-0")
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    replacement = service._claim_replica_operation(
        "deployment-1",
        operation="start",
        replica_id=None,
        user_id="user-1",
    )

    assert replacement.generation == claim.generation + 2
    assert service._release_replica_operation(claim) is False
    with pytest.raises(service_module.ReplicaOperationLostError):
        service._write_replica_state(
            claim,
            "replica-0",
            status="failed",
            health_status="UNHEALTHY",
        )
    with pytest.raises(service_module.ReplicaOperationBusyError):
        service._claim_replica_operation(
            "deployment-1",
            operation="stop",
            replica_id=None,
            user_id="user-1",
        )
    assert service._release_replica_operation(replacement) is True
    with Session(engine) as session:
        children = session.exec(
            select(DeploymentReplicaDB).order_by(DeploymentReplicaDB.replica_index)
        ).all()
        assert [(item.status, item.health_status) for item in children] == [
            ("running", "HEALTHY"),
            ("stopped", "UNKNOWN"),
        ]


def test_recent_heartbeat_prevents_reclaim_even_when_started_at_is_old(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    claim = service._claim_replica_operation(
        "deployment-1",
        operation="restart",
        replica_id=None,
        user_id="user-1",
    )
    observed_now = datetime(2026, 8, 23, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_started_at = datetime(2000, 1, 1)
        parent.replica_operation_heartbeat_at = observed_now
        session.add(parent)
        session.commit()
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service._claim_replica_operation(
            "deployment-1",
            operation="start",
            replica_id=None,
            user_id="user-1",
        )

    assert service._release_replica_operation(claim) is True


def test_stale_claim_recovery_fails_closed_on_container_identity_error(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    claim = service._claim_replica_operation(
        "deployment-1",
        operation="restart",
        replica_id=None,
        user_id="user-1",
    )
    observed_now = datetime(2026, 8, 23, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.fail_names.add("trainfactory-vllm-model-deployme")
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service._claim_replica_operation(
            "deployment-1",
            operation="start",
            replica_id=None,
            user_id="user-1",
        )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token == claim.token
        assert parent.replica_operation_generation == claim.generation
    assert service._release_replica_operation(claim) is True


@pytest.mark.parametrize("lifecycle", ["canonical", "legacy"])
def test_stale_claim_recovery_preserves_owner_when_presence_probe_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    lifecycle: str,
) -> None:
    service, docker, engine = lifecycle_service
    if lifecycle == "legacy":
        _make_legacy_deployment(engine, status="starting")
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation="start",
        replica_id=None,
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    container_name = "trainfactory-vllm-model-deployme"
    docker.existing.add(container_name)
    docker.fail_authoritative_presence_names.add(container_name)
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    recovered = service._recover_stale_replica_operation(
        "deployment-1",
        user_id="user-1",
        observed_at=observed_now,
    )

    assert recovered is False
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token == old_claim.token
        assert parent.replica_operation_generation == old_claim.generation
    assert any(call[0] == "exists-authoritative" for call in docker.calls)
    assert not any(
        call[0] in {"create", "restart", "stop", "remove"} for call in docker.calls
    )
    assert service._claim_heartbeats == {}


@pytest.mark.parametrize(
    ("old_operation", "initial_status", "runtime_running", "expected_status"),
    [
        ("start", "starting", True, "running"),
        ("stop", "stopping", False, "stopped"),
        ("restart", "restarting", True, "running"),
        ("delete", "running", False, "stopped"),
    ],
)
def test_legacy_stale_claim_reconciles_runtime_before_generation_takeover(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    old_operation: str,
    initial_status: str,
    runtime_running: bool,
    expected_status: str,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status=initial_status)
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation=old_operation,
        replica_id=None,
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    if old_operation != "delete":
        docker.existing.add("trainfactory-vllm-model-deployme")
    if runtime_running:
        docker.running.add("legacy")
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    replacement = service._claim_replica_operation(
        "deployment-1",
        operation="stop",
        replica_id=None,
        user_id="user-1",
    )

    assert replacement.generation == old_claim.generation + 2
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == expected_status
        assert parent.replica_operation_token == replacement.token
    calls_before_old_owner = list(docker.calls)
    with pytest.raises(service_module.ReplicaOperationLostError):
        service._require_replica_operation_ownership(old_claim)
    assert docker.calls == calls_before_old_owner
    assert service._release_replica_operation(replacement)
    assert service._claim_heartbeats == {}


@pytest.mark.parametrize(
    (
        "old_operation",
        "runtime_observation",
        "expected_status",
        "recovery_allowed",
    ),
    [
        ("load_adapter", "exact", "loaded", True),
        ("load_adapter", "absent", "failed", True),
        ("unload_adapter", "exact", "loaded", True),
        ("unload_adapter", "absent", "unloaded", True),
        ("load_adapter", "wrong-path", "loading", False),
        ("load_adapter", "no-path", "loading", False),
        ("unload_adapter", "unavailable", "unloading", False),
    ],
)
def test_legacy_stale_adapter_claim_requires_exact_runtime_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    old_operation: str,
    runtime_observation: str,
    expected_status: str,
    recovery_allowed: bool,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running", enable_lora=True)
    initial_adapter_status = (
        "loading" if old_operation == "load_adapter" else "unloading"
    )
    with Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                adapter_id="legacy-stale-adapter",
                deployment_id="deployment-1",
                deployment_replica_id=None,
                adapter_name="adapter-a",
                adapter_path="/models/AdapterA",
                status=initial_adapter_status,
                user_id="user-1",
            )
        )
        session.commit()
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation=old_operation,
        replica_id=None,
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.running.add("legacy")
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    def list_runtime_adapters(_deployment):
        if runtime_observation == "unavailable":
            raise RuntimeError("adapter registry unavailable")
        if runtime_observation == "absent":
            return []
        if runtime_observation == "no-path":
            return [{"name": "adapter-a"}]
        return [
            {
                "name": "adapter-a",
                "path": (
                    "/models/other"
                    if runtime_observation == "wrong-path"
                    else "/models/AdapterA"
                ),
            }
        ]

    monkeypatch.setattr(
        service,
        "_list_legacy_runtime_adapters",
        list_runtime_adapters,
        raising=False,
    )

    if recovery_allowed:
        replacement = service._claim_replica_operation(
            "deployment-1",
            operation="stop",
            replica_id=None,
            user_id="user-1",
        )
        assert replacement.generation == old_claim.generation + 2
        assert service._release_replica_operation(replacement)
    else:
        with pytest.raises(service_module.ReplicaOperationBusyError):
            service._claim_replica_operation(
                "deployment-1",
                operation="stop",
                replica_id=None,
                user_id="user-1",
            )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        adapter = session.exec(select(LoadedAdapterDB)).one()
        assert adapter.status == expected_status
        if recovery_allowed:
            assert parent.replica_operation_token is None
        else:
            assert parent.replica_operation_token == old_claim.token
            assert parent.replica_operation_generation == old_claim.generation
    assert service._claim_heartbeats == {}


def test_legacy_stale_adapter_recovery_rejects_row_drift_after_runtime_observation(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="running", enable_lora=True)
    with Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                adapter_id="legacy-stale-adapter",
                deployment_id="deployment-1",
                deployment_replica_id=None,
                adapter_name="adapter-a",
                adapter_path="/models/AdapterA",
                status="loading",
                user_id="user-1",
            )
        )
        session.commit()
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation="load_adapter",
        replica_id=None,
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.running.add("legacy")
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    def observe_then_replace_adapter(_deployment):
        with Session(engine) as session:
            adapter = session.exec(select(LoadedAdapterDB)).one()
            adapter.adapter_path = "/models/replaced"
            adapter.force_status("failed", "new owner state")
            session.add(adapter)
            session.commit()
        return [{"name": "adapter-a", "path": "/models/AdapterA"}]

    monkeypatch.setattr(
        service,
        "_list_legacy_runtime_adapters",
        observe_then_replace_adapter,
    )

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service._claim_replica_operation(
            "deployment-1",
            operation="stop",
            replica_id=None,
            user_id="user-1",
        )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        adapter = session.exec(select(LoadedAdapterDB)).one()
        assert parent.replica_operation_token == old_claim.token
        assert parent.replica_operation_generation == old_claim.generation
        assert adapter.adapter_path == "/models/replaced"
        assert adapter.status == "failed"
        assert adapter.error_message == "new owner state"
    assert service._claim_heartbeats == {}


def test_fresh_legacy_heartbeat_cannot_be_taken_over(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="starting")
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation="start",
        replica_id=None,
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_started_at = datetime(2000, 1, 1)
        parent.replica_operation_heartbeat_at = observed_now
        session.add(parent)
        session.commit()
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service._claim_replica_operation(
            "deployment-1",
            operation="stop",
            replica_id=None,
            user_id="user-1",
        )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token == old_claim.token
        assert parent.replica_operation_generation == old_claim.generation
    assert service._release_replica_operation(old_claim)


def test_legacy_stale_recovery_rejects_foreign_container_identity(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="starting")
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation="start",
        replica_id=None,
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.fail_names.add("trainfactory-vllm-model-deployme")
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service._claim_replica_operation(
            "deployment-1",
            operation="stop",
            replica_id=None,
            user_id="user-1",
        )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token == old_claim.token
        assert parent.replica_operation_generation == old_claim.generation
    assert not any(call[0] in {"stop", "remove", "restart"} for call in docker.calls)
    assert service._release_replica_operation(old_claim)


def test_legacy_stale_recovery_fences_old_id_and_new_owner_uses_replacement_id(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="restarting")
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation="restart",
        replica_id=None,
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.running.add("old-container-id")
    current_id = {"value": "sha256:old-container-id"}

    def get_current_id(*_args, **_kwargs) -> str:
        docker.calls.append(("inspect", "trainfactory-vllm-model-deployme"))
        return current_id["value"]

    def replace_name_target() -> None:
        current_id["value"] = "sha256:new-container-id"

    monkeypatch.setattr(docker, "get_legacy_managed_container_id", get_current_id)
    docker.on_running_identity = replace_name_target
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    replacement = service._claim_replica_operation(
        "deployment-1",
        operation="stop",
        replica_id=None,
        user_id="user-1",
    )
    with pytest.raises(service_module.ReplicaOperationLostError):
        service._require_replica_operation_ownership(old_claim)
    result = service._stop_legacy_deployment_claimed(
        replacement,
        user_id="user-1",
    )

    assert result["status"] == "stopped"
    assert ("stop", "sha256:new-container-id") in docker.calls
    assert ("stop", "sha256:old-container-id") not in docker.calls
    assert service._release_replica_operation(replacement)
    assert service._claim_heartbeats == {}


@pytest.mark.parametrize("operation", ["start", "restart", "load_adapter"])
def test_legacy_runtime_creation_rejects_model_delete_intent_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(
        engine,
        status="stopped" if operation == "start" else "running",
        enable_lora=operation == "load_adapter",
    )
    with Session(engine) as session:
        model = session.exec(select(ModelRegistryDB)).one()
        model.status = "deleting"
        model.extra_metadata = {
            "_train_factory_model_delete_intent_v1": {
                "token": "model-delete-owner",
                "had_extra_metadata": False,
            }
        }
        session.add(model)
        session.commit()
    docker.existing.add("trainfactory-vllm-model-deployme")

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    runtime_calls: list[str] = []

    class Client:
        @staticmethod
        def load_lora_adapter(*_args) -> None:
            runtime_calls.append("load")

    adapter_service = adapter_module.AdapterService(service)
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(ValueError, match="being deleted"):
        if operation == "start":
            service.start_deployment("deployment-1", user_id="user-1")
        elif operation == "restart":
            service.restart_deployment("deployment-1", user_id="user-1")
        else:
            adapter_service.load_adapter(
                "deployment-1",
                "adapter-a",
                "/external/adapter-a",
                user_id="user-1",
            )

    assert runtime_calls == []
    assert not any(
        call[0] in {"create", "restart", "remove", "stop"}
        for call in docker.calls
    )
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token is None
        assert session.exec(select(LoadedAdapterDB)).all() == []
    assert service._claim_heartbeats == {}


@pytest.mark.parametrize("operation", ["stop", "unload_adapter"])
def test_legacy_runtime_cleanup_is_allowed_during_model_delete_intent(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    operation: str,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(
        engine,
        status="running",
        enable_lora=operation == "unload_adapter",
    )
    with Session(engine) as session:
        model = session.exec(select(ModelRegistryDB)).one()
        model.status = "deleting"
        model.extra_metadata = {
            "_train_factory_model_delete_intent_v1": {
                "token": "model-delete-owner",
                "had_extra_metadata": False,
            }
        }
        session.add(model)
        if operation == "unload_adapter":
            session.add(
                LoadedAdapterDB(
                    adapter_id="legacy-cleanup-adapter",
                    deployment_id="deployment-1",
                    deployment_replica_id=None,
                    adapter_name="adapter-a",
                    adapter_path="/external/adapter-a",
                    status="loaded",
                    user_id="user-1",
                )
            )
        session.commit()
    docker.existing.add("trainfactory-vllm-model-deployme")

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    runtime_calls: list[str] = []

    class Client:
        @staticmethod
        def unload_lora_adapter(name: str) -> None:
            runtime_calls.append(f"unload:{name}")

    adapter_service = adapter_module.AdapterService(service)
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    if operation == "stop":
        result = service.stop_deployment("deployment-1", user_id="user-1")
        assert result["status"] == "stopped"
        assert ("stop", "sha256:legacy") in docker.calls
    else:
        assert adapter_service.unload_adapter(
            "deployment-1",
            "adapter-a",
            user_id="user-1",
        )
        assert runtime_calls == ["unload:adapter-a"]
        with Session(engine) as session:
            assert session.exec(select(LoadedAdapterDB)).one().status == "unloaded"
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().replica_operation_token is None
    assert service._claim_heartbeats == {}


def test_startup_scanner_recovers_stale_legacy_claim_without_new_request(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    _make_legacy_deployment(engine, status="starting")
    old_claim = service._claim_replica_operation(
        "deployment-1",
        operation="start",
        replica_id=None,
        user_id="user-1",
    )
    service._stop_claim_heartbeat(old_claim)
    observed_now = datetime(2026, 8, 24, 12, 0, 0)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
        parent.replica_operation_started_at = observed_now - timedelta(minutes=10)
        session.add(parent)
        session.commit()
    docker.existing.add("trainfactory-vllm-model-deployme")
    docker.running.add("legacy")
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)

    assert service.recover_stale_replica_operations(observed_at=observed_now) == 1

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == "running"
        assert parent.replica_operation_generation == old_claim.generation + 1
        assert parent.replica_operation_token is None
    assert service._claim_heartbeats == {}


def test_adapter_load_uses_lifecycle_claim_before_network_or_database_write(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: pytest.fail("network must not be reached"),
    )
    active = service._claim_replica_operation(
        "deployment-1",
        operation="stop",
        replica_id="replica-0",
        user_id="user-1",
    )

    with pytest.raises(service_module.ReplicaOperationBusyError):
        adapter_service.load_adapter(
            "deployment-1",
            "adapter-a",
            "/app/output/a/final_model",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )

    with Session(engine) as session:
        assert session.exec(select(LoadedAdapterDB)).all() == []
    assert service._release_replica_operation(active) is True


@pytest.mark.parametrize(
    ("operation", "initial_status", "expected_stale_status"),
    [("load", None, "loading"), ("unload", "loaded", "unloading")],
)
def test_adapter_generation_replacement_rejects_stale_writeback(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    operation: str,
    initial_status: str | None,
    expected_stale_status: str,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        if initial_status is not None:
            session.add(
                LoadedAdapterDB(
                    deployment_id="deployment-1",
                    deployment_replica_id="replica-0",
                    adapter_name="adapter-a",
                    adapter_path="/app/output/a/final_model",
                    user_id="user-1",
                    status=initial_status,
                )
            )
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)

    class Client:
        def load_lora_adapter(self, *_args):
            _replace_generation()

        def unload_lora_adapter(self, *_args):
            _replace_generation()

    def _replace_generation() -> None:
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "replacement-token"
            parent.replica_operation_generation += 1
            parent.replica_operation_heartbeat_at = datetime(2026, 8, 23)
            session.add(parent)
            session.commit()

    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(service_module.ReplicaOperationLostError):
        if operation == "load":
            adapter_service.load_adapter(
                "deployment-1",
                "adapter-a",
                "/app/output/a/final_model",
                deployment_replica_id="replica-0",
                user_id="user-1",
            )
        else:
            adapter_service.unload_adapter(
                "deployment-1",
                "adapter-a",
                deployment_replica_id="replica-0",
                user_id="user-1",
            )

    with Session(engine) as session:
        adapter = session.exec(select(LoadedAdapterDB)).one()
        assert adapter.status == expected_stale_status


def test_new_claim_reconciles_adapter_stuck_unloading_after_lease_loss(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                deployment_replica_id="replica-0",
                adapter_name="adapter-a",
                adapter_path="/app/output/a/final_model",
                source_task_id="task-a",
                user_id="user-1",
                status="loaded",
            )
        )
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    runtime_calls: list[str] = []

    class Client:
        def unload_lora_adapter(self, name):
            runtime_calls.append(name)
            if len(runtime_calls) == 1:
                with Session(engine) as session:
                    parent = session.exec(select(DeploymentDB)).one()
                    parent.replica_operation_token = "replacement-token"
                    parent.replica_operation_generation += 1
                    parent.replica_operation_heartbeat_at = datetime(2026, 8, 23)
                    session.add(parent)
                    session.commit()

    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(service_module.ReplicaOperationLostError):
        adapter_service.unload_adapter(
            "deployment-1",
            "adapter-a",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_token = None
        parent.replica_operation_kind = None
        parent.replica_operation_replica_id = None
        parent.replica_operation_started_at = None
        parent.replica_operation_heartbeat_at = None
        session.add(parent)
        session.commit()

    assert adapter_service.unload_adapter(
        "deployment-1",
        "adapter-a",
        deployment_replica_id="replica-0",
        user_id="user-1",
    )

    with Session(engine) as session:
        adapter = session.exec(select(LoadedAdapterDB)).one()
        assert adapter.status == "unloaded"
    assert runtime_calls == ["adapter-a", "adapter-a"]
    assert adapter_service.list_active_training_output_consumers(
        "task-a",
        "/app/output/a/final_model",
        user_id="user-1",
    ) == []


@pytest.mark.parametrize(
    ("listed_adapters", "expected_status", "expect_error"),
    [([], "unloaded", False), ([{"name": "adapter-a"}], "unloading", True)],
)
def test_unloading_recovery_requires_verified_runtime_absence(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    listed_adapters: list[dict[str, str]],
    expected_status: str,
    expect_error: bool,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                deployment_replica_id="replica-0",
                adapter_name="adapter-a",
                adapter_path="/app/output/a/final_model",
                user_id="user-1",
                status="unloading",
            )
        )
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)

    class Client:
        def unload_lora_adapter(self, _name):
            raise RuntimeError("unload result unknown")

        def list_lora_adapters(self):
            return listed_adapters

    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    if expect_error:
        with pytest.raises(RuntimeError, match="verify adapter runtime absence"):
            adapter_service.unload_adapter(
                "deployment-1",
                "adapter-a",
                deployment_replica_id="replica-0",
                user_id="user-1",
            )
    else:
        assert adapter_service.unload_adapter(
            "deployment-1",
            "adapter-a",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )

    with Session(engine) as session:
        adapter = session.exec(select(LoadedAdapterDB)).one()
        assert adapter.status == expected_status


@pytest.mark.parametrize("initial_status", [None, "loading", "failed"])
@pytest.mark.parametrize("list_mode", ["present", "unavailable"])
def test_orphan_unload_never_succeeds_without_verified_runtime_absence(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    initial_status: str | None,
    list_mode: str,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        if initial_status is not None:
            session.add(
                LoadedAdapterDB(
                    deployment_id="deployment-1",
                    deployment_replica_id="replica-0",
                    adapter_name="adapter-a",
                    adapter_path="/app/output/a/final_model",
                    user_id="user-1",
                    status=initial_status,
                )
            )
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)

    class Client:
        def unload_lora_adapter(self, _name):
            raise RuntimeError("unload failed")

        def list_lora_adapters(self):
            if list_mode == "unavailable":
                raise RuntimeError("list unavailable")
            return [{"name": "adapter-a"}]

    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(RuntimeError, match="verify adapter runtime absence"):
        adapter_service.unload_adapter(
            "deployment-1",
            "adapter-a",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )

    with Session(engine) as session:
        rows = session.exec(select(LoadedAdapterDB)).all()
        if initial_status is None:
            assert rows == []
        else:
            assert [row.status for row in rows] == [initial_status]


@pytest.mark.parametrize("operation", ["load", "unload"])
def test_adapter_renews_claim_immediately_before_runtime_mutation(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    operation: str,
) -> None:
    service, _docker, engine = lifecycle_service
    observed_now = datetime(2026, 8, 24, 13, 0, 0)
    monkeypatch.setattr(service_module, "now_naive", lambda: observed_now)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        if operation == "unload":
            session.add(
                LoadedAdapterDB(
                    deployment_id="deployment-1",
                    deployment_replica_id="replica-0",
                    adapter_name="adapter-a",
                    adapter_path="/app/output/a/final_model",
                    user_id="user-1",
                    status="loaded",
                )
            )
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    takeover_results: list[str] = []

    def expire_active_claim() -> None:
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_heartbeat_at = observed_now - timedelta(minutes=10)
            session.add(parent)
            session.commit()

    def attempt_takeover() -> None:
        try:
            replacement = service._claim_replica_operation(
                "deployment-1",
                operation="start",
                replica_id=None,
                user_id="user-1",
            )
        except service_module.ReplicaOperationBusyError:
            takeover_results.append("busy")
        else:
            takeover_results.append("claimed")
            service._release_replica_operation(replacement)

    class Client:
        def load_lora_adapter(self, *_args):
            attempt_takeover()

        def unload_lora_adapter(self, *_args):
            attempt_takeover()

    def client_after_claim_expiry(*_args, **_kwargs):
        expire_active_claim()
        return Client()

    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        client_after_claim_expiry,
    )

    if operation == "load":
        result = adapter_service.load_adapter(
            "deployment-1",
            "adapter-a",
            "/app/output/a/final_model",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )
        assert result["status"] == "loaded"
    else:
        assert adapter_service.unload_adapter(
            "deployment-1",
            "adapter-a",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )
    assert takeover_results == ["busy"]


@pytest.mark.parametrize(
    "runtime_phase",
    ["load", "stale-load", "unload", "orphan-unload", "sync"],
)
def test_canonical_adapter_runtime_io_has_no_active_database_session(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    runtime_phase: str,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        if runtime_phase in {"stale-load", "unload"}:
            session.add(
                LoadedAdapterDB(
                    deployment_id="deployment-1",
                    deployment_replica_id="replica-0",
                    adapter_name="adapter-a",
                    adapter_path="/app/output/a/final_model",
                    user_id="user-1",
                    status="failed" if runtime_phase == "stale-load" else "loaded",
                )
            )
        session.commit()

    active_sessions = 0
    observations: list[tuple[str, int]] = []

    @contextmanager
    def tracked_session():
        nonlocal active_sessions
        active_sessions += 1
        try:
            with Session(engine) as session:
                yield session
        finally:
            active_sessions -= 1

    class Client:
        def load_lora_adapter(self, *_args):
            observations.append(("load", active_sessions))

        def unload_lora_adapter(self, *_args):
            observations.append(("unload", active_sessions))
            if runtime_phase == "orphan-unload":
                raise RuntimeError("unload result unknown")

        def list_lora_adapters(self):
            observations.append(("list", active_sessions))
            return []

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", tracked_session)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    if runtime_phase in {"load", "stale-load"}:
        adapter_service.load_adapter(
            "deployment-1",
            "adapter-a",
            "/app/output/a/final_model",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )
    elif runtime_phase in {"unload", "orphan-unload"}:
        assert adapter_service.unload_adapter(
            "deployment-1",
            "adapter-a",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )
    else:
        assert (
            adapter_service.sync_loaded_adapters(
                "deployment-1",
                deployment_replica_id="replica-0",
                user_id="user-1",
            )
            == 0
        )

    assert observations
    assert all(active == 0 for _method, active in observations)


def test_unknown_runtime_adapter_sync_rejects_managed_model_delete_intent(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        model = ModelRegistryDB(
            model_id="model-delete-target",
            model_name="Model pending delete",
            model_type="llm",
            model_path=str(
                adapter_module.settings.models_dir / "model-delete-target"
            ),
            source_type="trained",
            status="deleting",
            extra_metadata={
                "_train_factory_model_delete_intent_v1": {
                    "token": "delete-owner",
                    "had_extra_metadata": False,
                }
            },
        )
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(model)
        session.add(parent)
        session.add(child)
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    class Client:
        @staticmethod
        def list_lora_adapters():
            return [{"name": "runtime-only-adapter"}]

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    assert (
        adapter_service.sync_loaded_adapters(
            "deployment-1",
            deployment_replica_id="replica-0",
            user_id="user-1",
            best_effort=True,
        )
        == 0
    )
    with Session(engine) as session:
        assert session.exec(select(LoadedAdapterDB)).all() == []


def test_unknown_runtime_adapter_sync_serializes_before_model_delete_scan(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    class Client:
        @staticmethod
        def list_lora_adapters():
            return [{"name": "runtime-only-adapter"}]

    sync_holds_gate = Event()
    allow_sync_commit = Event()
    sync_finished = Event()
    delete_started = Event()
    delete_finished = Event()
    sync_results: list[int] = []
    delete_results: list[str] = []

    def pause_after_gate_lock(session) -> None:
        lock_model_artifact_membership(session)
        sync_holds_gate.set()
        assert allow_sync_commit.wait(5), "timed out waiting to commit unknown adapter"

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(
        adapter_module,
        "lock_model_artifact_membership",
        pause_after_gate_lock,
        raising=False,
    )
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    def sync_worker() -> None:
        try:
            sync_results.append(
                adapter_service.sync_loaded_adapters(
                    "deployment-1",
                    deployment_replica_id="replica-0",
                    user_id="user-1",
                )
            )
        finally:
            sync_finished.set()

    def delete_phase_worker() -> None:
        delete_started.set()
        try:
            with Session(engine) as session:
                lock_model_artifact_membership(session)
                active_unknown = session.exec(
                    select(LoadedAdapterDB).where(
                        LoadedAdapterDB.status.in_(
                            ("loading", "loaded", "unloading")
                        ),
                        LoadedAdapterDB.adapter_path == "<unknown:auto-sync>",
                    )
                ).first()
                delete_results.append(
                    "active-adapter" if active_unknown is not None else "rmtree"
                )
                session.commit()
        finally:
            delete_finished.set()

    sync_thread = Thread(target=sync_worker, daemon=True)
    sync_thread.start()
    assert sync_holds_gate.wait(5), "sync never acquired the membership gate"

    delete_thread = Thread(target=delete_phase_worker, daemon=True)
    delete_thread.start()
    assert delete_started.wait(5)
    assert not delete_finished.wait(0.2), "delete phase crossed the held gate"

    allow_sync_commit.set()
    sync_thread.join(5)
    delete_thread.join(5)

    assert sync_finished.is_set()
    assert delete_finished.is_set()
    assert sync_results == [1]
    assert delete_results == ["active-adapter"]


@pytest.mark.parametrize(
    ("framework", "client_type"),
    [
        ("vllm", vllm_client_module.VLLMClient),
        ("sglang", sglang_client_module.SGLangClient),
    ],
)
def test_real_client_list_failure_does_not_downgrade_loaded_adapter(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    framework: str,
    client_type,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        parent.inference_framework = framework
        child = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        child.status = "running"
        child.health_status = "HEALTHY"
        session.add(parent)
        session.add(child)
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                deployment_replica_id="replica-0",
                adapter_name="adapter-a",
                adapter_path="/app/output/a/final_model",
                user_id="user-1",
                status="loaded",
            )
        )
        session.commit()

    client = client_type("http://runtime.invalid")

    def unavailable(*_args, **_kwargs):
        raise TimeoutError("adapter registry unavailable")

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(client, "_request", unavailable)
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: client,
    )

    assert (
        adapter_service.sync_loaded_adapters(
            "deployment-1",
            deployment_replica_id="replica-0",
            user_id="user-1",
            best_effort=True,
        )
        == 0
    )
    with Session(engine) as session:
        assert session.exec(select(LoadedAdapterDB)).one().status == "loaded"


@pytest.mark.parametrize("legacy", [False, True], ids=["canonical", "legacy"])
@pytest.mark.parametrize("list_mode", ["present", "unavailable"])
def test_stale_adapter_cleanup_requires_verified_runtime_absence(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    legacy: bool,
    list_mode: str,
) -> None:
    service, _docker, engine = lifecycle_service
    if legacy:
        _make_legacy_deployment(engine, status="running", enable_lora=True)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        session.add(parent)
        if not legacy:
            child = session.exec(
                select(DeploymentReplicaDB).where(
                    DeploymentReplicaDB.replica_id == "replica-0"
                )
            ).one()
            child.status = "running"
            child.health_status = "HEALTHY"
            session.add(child)
        session.add(
            LoadedAdapterDB(
                adapter_id="stale-adapter",
                deployment_id="deployment-1",
                deployment_replica_id=None if legacy else "replica-0",
                adapter_name="adapter-a",
                adapter_path="/app/output/a/final_model",
                user_id="user-1",
                status="failed",
            )
        )
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    load_calls: list[str] = []

    class Client:
        def unload_lora_adapter(self, _name):
            raise RuntimeError("unload result unknown")

        def list_lora_adapters(self):
            if list_mode == "unavailable":
                raise RuntimeError("list unavailable")
            return [{"name": "adapter-a", "path": "/app/output/a/final_model"}]

        def load_lora_adapter(self, name, _path):
            load_calls.append(name)

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(RuntimeError, match="verify adapter runtime absence"):
        adapter_service.load_adapter(
            "deployment-1",
            "adapter-a",
            "/app/output/a/final_model",
            deployment_replica_id=None if legacy else "replica-0",
            user_id="user-1",
        )

    assert load_calls == []
    with Session(engine) as session:
        stale = session.exec(
            select(LoadedAdapterDB).where(
                LoadedAdapterDB.adapter_id == "stale-adapter"
            )
        ).one()
        assert stale.status == "failed"


@pytest.mark.parametrize("legacy", [False, True], ids=["canonical", "legacy"])
@pytest.mark.parametrize(
    "runtime_state",
    ["loaded", "absent", "unavailable", "case-different"],
)
def test_adapter_load_unknown_result_is_reconciled_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    legacy: bool,
    runtime_state: str,
) -> None:
    service, _docker, engine = lifecycle_service
    if legacy:
        _make_legacy_deployment(engine, status="running", enable_lora=True)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "running"
        parent.enable_lora = True
        session.add(parent)
        if not legacy:
            child = session.exec(
                select(DeploymentReplicaDB).where(
                    DeploymentReplicaDB.replica_id == "replica-0"
                )
            ).one()
            child.status = "running"
            child.health_status = "HEALTHY"
            session.add(child)
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    list_calls: list[str] = []

    class Client:
        def load_lora_adapter(self, _name, _path):
            raise RuntimeError("load response lost")

        def list_lora_adapters(self):
            list_calls.append("list")
            if runtime_state == "unavailable":
                raise RuntimeError("list unavailable")
            if runtime_state == "loaded":
                return [
                    {
                        "name": "adapter-a",
                        "path": "/app/output/a/final_model",
                    }
                ]
            if runtime_state == "case-different":
                return [
                    {
                        "name": "adapter-a",
                        "path": "/APP/OUTPUT/A/FINAL_MODEL",
                    }
                ]
            return []

    adapter_service = adapter_module.AdapterService(
        deployment_lifecycle_service=service
    )
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(
        adapter_service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    if runtime_state == "loaded":
        result = adapter_service.load_adapter(
            "deployment-1",
            "adapter-a",
            "/app/output/a/final_model",
            deployment_replica_id=None if legacy else "replica-0",
            user_id="user-1",
        )
        assert result["status"] == "loaded"
    else:
        with pytest.raises(RuntimeError, match="Failed to load adapter"):
            adapter_service.load_adapter(
                "deployment-1",
                "adapter-a",
                "/app/output/a/final_model",
                deployment_replica_id=None if legacy else "replica-0",
                user_id="user-1",
            )

    assert list_calls == ["list"]
    with Session(engine) as session:
        adapter = session.exec(select(LoadedAdapterDB)).one()
        expected_status = {
            "loaded": "loaded",
            "absent": "failed",
            "unavailable": "loading",
            "case-different": "loading",
        }[runtime_state]
        assert adapter.status == expected_status


def test_replica_status_sync_checks_each_owned_identity_and_derives_parent(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )
    docker.running.add("replica-0")
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "degraded"
        children = session.exec(select(DeploymentReplicaDB)).all()
        for child in children:
            child.status = "running"
            child.health_status = "HEALTHY"
            session.add(child)
        session.add(parent)
        session.commit()

    result = service.sync_status("deployment-1")

    assert ("inspect", "trainfactory-vllm-model-deployme") in docker.calls
    assert ("inspect", "trainfactory-vllm-model-deployme-r1") in docker.calls
    assert result["status"] == "degraded"
    assert [child["status"] for child in result["replica_instances"]] == [
        "running",
        "stopped",
    ]


def _configure_legacy_status_sync_parent(
    engine,
    *,
    status: str,
    deploy_mode: str = "shared",
) -> str:
    container_name = "trainfactory-vllm-model-deployme"
    with Session(engine) as session:
        session.exec(sa.delete(DeploymentReplicaDB))
        parent = session.exec(select(DeploymentDB)).one()
        parent.config = {}
        parent.replica = 1
        parent.status = status
        parent.deploy_mode = deploy_mode
        parent.inference_framework = (
            "vllm" if deploy_mode == "container" else "xinference"
        )
        parent.container_name = container_name if deploy_mode == "container" else None
        parent.model_uid = "served-model"
        parent.replica_operation_token = None
        parent.replica_operation_kind = None
        parent.replica_operation_replica_id = None
        session.add(parent)
        session.commit()
    return container_name


def test_legacy_sync_status_probes_outside_session_and_updates_normally(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    _configure_legacy_status_sync_parent(engine, status="starting")
    active_sessions = 0
    observed_active_sessions: list[int] = []

    @contextmanager
    def tracked_session():
        nonlocal active_sessions
        with Session(engine) as session:
            active_sessions += 1
            try:
                yield session
            finally:
                active_sessions -= 1

    class Client:
        def get_model_status(self, _model_uid):
            observed_active_sessions.append(active_sessions)
            return "running"

    monkeypatch.setattr(service_module, "get_session", tracked_session)
    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: Client(),
    )

    result = service.sync_status("deployment-1")

    assert observed_active_sessions == [0]
    assert result["status"] == "running"
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().status == "running"


def test_legacy_sync_status_does_not_overwrite_concurrent_stop_after_probe(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    _configure_legacy_status_sync_parent(engine, status="running")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
    probe_entered = Event()
    release_probe = Event()
    sync_results: list[dict] = []
    sync_errors: list[BaseException] = []

    class Client:
        def get_model_status(self, _model_uid):
            probe_entered.set()
            assert release_probe.wait(timeout=5)
            return "running"

        def terminate_model(self, _model_uid):
            return None

    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: Client(),
    )

    def run_sync() -> None:
        try:
            sync_results.append(service.sync_status("deployment-1"))
        except BaseException as exc:  # pragma: no cover - asserted below
            sync_errors.append(exc)

    thread = Thread(target=run_sync, daemon=True)
    thread.start()
    assert probe_entered.wait(timeout=5)
    try:
        stopped = service.stop_deployment("deployment-1", user_id="user-1")
        assert stopped["status"] == "stopped"
    finally:
        release_probe.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert sync_results == []
    assert len(sync_errors) == 1
    assert isinstance(sync_errors[0], service_module.ReplicaOperationBusyError)
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == "stopped"
        assert parent.replica_operation_token is None
        assert parent.replica_operation_generation == 1


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_legacy_sync_status_rejects_active_claim_before_runtime_probe(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
    operation: str,
) -> None:
    service, _docker, engine = lifecycle_service
    _configure_legacy_status_sync_parent(engine, status="running")
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_token = "active-owner"
        parent.replica_operation_kind = operation
        parent.replica_operation_generation = 7
        session.add(parent)
        session.commit()
    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: pytest.fail(
            "active lifecycle claim reached runtime status probe"
        ),
    )

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service.sync_status("deployment-1")

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == "running"
        assert parent.replica_operation_token == "active-owner"
        assert parent.replica_operation_generation == 7


def test_legacy_sync_status_generation_cas_rejects_status_aba(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    _configure_legacy_status_sync_parent(engine, status="starting")

    class Client:
        def get_model_status(self, _model_uid):
            with Session(engine) as session:
                parent = session.exec(select(DeploymentDB)).one()
                parent.replica_operation_generation += 1
                session.add(parent)
                session.commit()
            return "running"

    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(service_module.ReplicaOperationBusyError):
        service.sync_status("deployment-1")
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == "starting"
        assert parent.replica_operation_generation == 1
        assert parent.replica_operation_token is None


def test_legacy_sync_status_does_not_recreate_deleted_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    _configure_legacy_status_sync_parent(engine, status="starting")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")

    class Client:
        def get_model_status(self, _model_uid):
            with Session(engine) as session:
                parent = session.exec(select(DeploymentDB)).one()
                session.delete(parent)
                session.commit()
            return "running"

    monkeypatch.setattr(
        service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(
        service_module.DeploymentReplicaNotFoundError,
        match="not found",
    ):
        service.sync_status("deployment-1")

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).first() is None


def test_legacy_batch_sync_probes_outside_session_and_loses_to_concurrent_stop(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    container_name = _configure_legacy_status_sync_parent(
        engine,
        status="starting",
        deploy_mode="container",
    )
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
    docker.existing.add(container_name)
    docker.running.add("legacy")
    active_sessions = 0
    observed_active_sessions: list[int] = []
    probe_entered = Event()
    release_probe = Event()
    sync_counts: list[int] = []
    sync_errors: list[BaseException] = []

    @contextmanager
    def tracked_session():
        nonlocal active_sessions
        with Session(engine) as session:
            active_sessions += 1
            try:
                yield session
            finally:
                active_sessions -= 1

    def pause_running_probe() -> None:
        observed_active_sessions.append(active_sessions)
        probe_entered.set()
        assert release_probe.wait(timeout=5)

    docker.on_running_identity = pause_running_probe
    monkeypatch.setattr(service_module, "get_session", tracked_session)
    monkeypatch.setattr(
        service,
        "_sync_configs_from_deployments",
        lambda _deployments: None,
    )

    def run_sync() -> None:
        try:
            sync_counts.append(service.sync_all_running())
        except BaseException as exc:  # pragma: no cover - asserted below
            sync_errors.append(exc)

    thread = Thread(target=run_sync, daemon=True)
    thread.start()
    assert probe_entered.wait(timeout=5)
    try:
        stopped = service.stop_deployment("deployment-1", user_id="user-1")
        assert stopped["status"] == "stopped"
    finally:
        release_probe.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert sync_errors == []
    assert observed_active_sessions == [0]
    assert sync_counts == [0]
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.status == "stopped"
        assert parent.replica_operation_token is None
        assert parent.replica_operation_generation == 1


def test_status_sync_keeps_running_but_unready_child_out_of_healthy_selection(
    lifecycle_service,
) -> None:
    service, docker, _engine = lifecycle_service
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )
    docker.running.update({"replica-0", "replica-1"})
    docker.unready_probe_endpoints.add("http://127.0.0.1:11001")

    result = service.sync_status("deployment-1")

    assert result["status"] == "degraded"
    assert [
        (child["status"], child["health_status"], child["error_message"])
        for child in result["replica_instances"]
    ] == [
        ("running", "HEALTHY", None),
        ("running", "UNHEALTHY", "replica readiness probe failed"),
    ]
    assert [call for call in docker.calls if call[0] == "probe"] == [
        ("probe", "http://127.0.0.1:11000", "user-1", "vllm", 2.0),
        ("probe", "http://127.0.0.1:11001", "user-1", "vllm", 2.0),
    ]

    with pytest.raises(ValueError, match="not running and healthy"):
        service.resolve_replica_selection(
            "deployment-1",
            "replica-1",
            user_id="user-1",
            require_healthy=True,
        )
    _deployment, replica = service.resolve_replica_selection(
        "deployment-1",
        "replica-0",
        user_id="user-1",
        require_healthy=True,
    )
    assert replica is not None and replica["replica_id"] == "replica-0"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_ProbeResponse(503, {}), False),
        (_ProbeResponse(200, {"data": []}), False),
        (_ProbeResponse(200, {"data": [{"id": "served-model"}]}), True),
        (requests.Timeout("probe timeout"), False),
    ],
    ids=["http-503", "empty-model-list", "ready", "timeout"],
)
def test_model_readiness_probe_is_single_shot_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    outcome,
    expected: bool,
) -> None:
    calls = []

    def request(method, url, user_id, **kwargs):
        calls.append((method, url, user_id, kwargs))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(docker_module, "request_user_outbound", request)
    monkeypatch.setattr(
        docker_module.time,
        "sleep",
        lambda *_args, **_kwargs: pytest.fail("readiness probe must not poll"),
    )

    result = docker_module.DockerDeployer().probe_service_ready(
        "http://127.0.0.1:11001/",
        user_id="user-1",
        framework="vllm",
        timeout=2.0,
    )

    assert result is expected
    assert calls == [
        (
            "GET",
            "http://127.0.0.1:11001/v1/models",
            "user-1",
            {"timeout": 2.0},
        )
    ]


def test_healthy_replica_selection_reconciles_runtime_before_returning(
    lifecycle_service,
) -> None:
    service, _docker, engine = lifecycle_service
    with Session(engine) as session:
        replica = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        replica.status = "running"
        replica.health_status = "HEALTHY"
        session.add(replica)
        session.commit()

    with pytest.raises(ValueError, match="not running and healthy"):
        service.resolve_replica_selection(
            "deployment-1",
            "replica-0",
            user_id="user-1",
            require_healthy=True,
        )


def test_status_sync_rejects_stale_write_after_generation_aba(
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    docker.existing.update(
        {"trainfactory-vllm-model-deployme", "trainfactory-vllm-model-deployme-r1"}
    )
    docker.running.update({"replica-0", "replica-1"})

    def advance_generation() -> None:
        claim = service._claim_replica_operation(
            "deployment-1",
            operation="stop",
            replica_id="replica-0",
            user_id="user-1",
        )
        assert service._release_replica_operation(claim)

    docker.on_running_identity = advance_generation
    with pytest.raises(service_module.ReplicaOperationBusyError):
        service.sync_status("deployment-1")

    with Session(engine) as session:
        children = session.exec(
            select(DeploymentReplicaDB).order_by(DeploymentReplicaDB.replica_index)
        ).all()
        assert [child.status for child in children] == ["stopped", "stopped"]


def test_startup_orphan_cleanup_keeps_degraded_children_and_residual_claim(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_service,
) -> None:
    service, docker, engine = lifecycle_service
    claim = service._claim_replica_operation(
        "deployment-1",
        operation="restart",
        replica_id=None,
        user_id="user-1",
    )
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "degraded"
        session.add(parent)
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    observed: list[set[str]] = []
    recovery_calls: list[str] = []
    monkeypatch.setattr(database_module, "get_session", get_session)
    monkeypatch.setattr(docker_module, "docker_deployer", docker)
    monkeypatch.setattr(
        service_module.deployment_service,
        "recover_stale_replica_operations",
        lambda: recovery_calls.append("recover") or 0,
        raising=False,
    )
    monkeypatch.setattr(
        docker,
        "cleanup_orphan_containers",
        lambda valid: observed.append(set(valid)) or 0,
        raising=False,
    )

    server_module.cleanup_orphan_containers()

    assert recovery_calls == ["recover"]
    assert observed == [
        {
            "trainfactory-vllm-model-deployme",
            "trainfactory-vllm-model-deployme-r1",
        }
    ]
    assert service._release_replica_operation(claim)


def test_orphan_cleanup_removes_only_verified_immutable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = docker_module.DockerDeployer()
    removed: list[str] = []
    monkeypatch.setattr(
        deployer,
        "list_xinference_containers",
        lambda: ["trainfactory-vllm-model-deadbeef-r2"],
    )
    monkeypatch.setattr(
        deployer,
        "get_managed_replica_container_id",
        lambda _name: "a" * 64,
        raising=False,
    )
    monkeypatch.setattr(
        deployer,
        "remove_container_identity",
        lambda container_id: removed.append(container_id) or True,
    )
    monkeypatch.setattr(
        deployer,
        "remove_container",
        lambda _name: pytest.fail("name-based removal is not allowed"),
    )

    assert deployer.cleanup_orphan_containers(set()) == 1
    assert removed == ["a" * 64]
