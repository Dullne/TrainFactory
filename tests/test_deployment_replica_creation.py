from __future__ import annotations

import importlib
import threading
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlmodel import Session, select

from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB


service_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)
model_config_service_module = importlib.import_module(
    "train_factory.storage.services.model_config_service"
)


def test_deferred_auto_start_marker_is_operator_owned() -> None:
    sanitized = service_module._sanitize_deployment_config(
        {
            "_deferred_auto_start": True,
            "_registry_artifact_signature": {"model_id": "forged"},
            "dtype": "auto",
        }
    )

    assert sanitized == {"dtype": "auto"}


class FakeDocker:
    def __init__(
        self,
        ports: list[int],
        *,
        fail_create_index: int | None = None,
        fail_wait_index: int | None = None,
        fail_remove_indexes: set[int] | None = None,
    ) -> None:
        self.ports = iter(ports)
        self.fail_create_index = fail_create_index
        self.fail_wait_index = fail_wait_index
        self.fail_remove_indexes = fail_remove_indexes or set()
        self.reserved: list[tuple[int, str]] = []
        self.released: list[tuple[int, str]] = []
        self.confirmed: list[tuple[int, str]] = []
        self.created: list[str] = []
        self.create_kwargs: list[dict[str, object]] = []
        self.container_ids: dict[str, str] = {}
        self.removed: list[str] = []
        self.foreign_names: set[str] = set()
        self.authoritative_presence_calls = 0
        self.fail_authoritative_presence_call: int | None = None
        self.authoritative_presence_error = "docker presence query failed"
        self.wait_calls = 0
        self.on_create = None
        self.on_wait = None

    def get_gpu_memory_usage(self):
        return {
            0: {"free_mb": 20_000, "total_mb": 24_000},
            1: {"free_mb": 20_000, "total_mb": 24_000},
            2: {"free_mb": 20_000, "total_mb": 24_000},
        }

    def find_available_port(self, *, owner_token: str) -> int:
        port = next(self.ports)
        self.reserved.append((port, owner_token))
        return port

    def release_port(self, port: int, *, owner_token: str) -> bool:
        self.released.append((port, owner_token))
        return True

    def confirm_port(self, port: int, *, owner_token: str) -> bool:
        self.confirmed.append((port, owner_token))
        return True

    def container_exists(self, container_name: str) -> bool:
        return container_name in self.foreign_names

    def container_exists_authoritative(self, container_name: str) -> bool:
        self.authoritative_presence_calls += 1
        if (
            self.fail_authoritative_presence_call
            == self.authoritative_presence_calls
        ):
            raise RuntimeError(self.authoritative_presence_error)
        return container_name in self.foreign_names

    def create_vllm_container(self, *, container_name: str, **kwargs):
        create_index = len(self.created)
        if self.on_create is not None:
            callback, self.on_create = self.on_create, None
            callback()
        if self.fail_create_index == create_index:
            return False, "fixed create failure", "docker-redacted"
        self.created.append(container_name)
        self.create_kwargs.append(dict(kwargs))
        self.container_ids[container_name] = f"{create_index + 1:064x}"
        return True, "ok", "docker-redacted"

    def create_sglang_container(self, *, container_name: str, **_kwargs):
        return self.create_vllm_container(container_name=container_name)

    def wait_for_service(self, *_args, **_kwargs) -> bool:
        current = self.wait_calls
        self.wait_calls += 1
        if self.on_wait is not None:
            callback, self.on_wait = self.on_wait, None
            callback()
        if self.fail_wait_index == current:
            return False
        return True

    def get_managed_container_id(self, container_name, deployment_id, replica_id):
        assert deployment_id
        assert replica_id
        return self.container_ids[container_name]

    def remove_container_identity(self, container_id: str) -> bool:
        current = len(self.removed)
        self.removed.append(container_id)
        if current in self.fail_remove_indexes:
            return False
        return True


@pytest.fixture
def replica_service(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'replicas.db'}")
    ModelArtifactMembershipGateDB.__table__.create(engine)
    ModelRegistryDB.__table__.create(engine)
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)
    LoadedAdapterDB.__table__.create(engine)

    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.add(
            ModelRegistryDB(
                model_id="model-1",
                model_name="Test Model",
                model_type="llm",
                model_path="/app/models/model-1",
                source_type="trained",
                status="available",
            )
        )
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", get_session)
    monkeypatch.setattr(
        service_module.model_registry_service,
        "get_model",
        lambda _model_id: {
            "model_id": "model-1",
            "model_name": "Test Model",
            "version": "v1.0.0",
            "model_path": "/app/models/model-1",
            "model_type": "llm",
            "base_model_path": None,
            "source_type": "trained",
            "is_adapter": False,
            "file_size": None,
        },
    )
    return service_module.DeploymentService(), engine


def _create(
    service,
    *,
    replica: int,
    auto_start: bool = True,
    defer_start: bool = False,
):
    return service.create_container_deployment(
        model_id="model-1",
        deployment_name="replica-group",
        replica=replica,
        gpu_memory_utilization=0.8,
        config={
            "launch_config": {
                "framework": "vllm",
                "gpu_pool": list(range(replica)),
            }
        },
        user_id="user-1",
        auto_start=auto_start,
        defer_start=defer_start,
        inference_framework="vllm",
    )


def test_happy_path_persists_parent_and_children_in_replica_order(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    result = _create(service, replica=2)

    assert result["replica"] == 2
    assert [item["replica_index"] for item in result["replica_instances"]] == [0, 1]
    assert result["container_name"] == result["replica_instances"][0]["container_name"]
    assert result["gpu_id"] == 0
    assert result["port"] == 11000
    assert result["xinference_endpoint"] == result["replica_instances"][0]["endpoint"]
    assert docker.created == [
        result["replica_instances"][0]["container_name"],
        result["replica_instances"][1]["container_name"],
    ]
    assert [port for port, _owner in docker.confirmed] == [11000, 11001]
    with Session(engine) as session:
        children = session.exec(
            select(DeploymentReplicaDB).order_by(DeploymentReplicaDB.replica_index)
        ).all()
        assert [child.status for child in children] == ["running", "running"]


def test_single_replica_service_fallback_persists_canonical_launch_config(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    created = service.create_container_deployment(
        model_id="model-1",
        deployment_name="fallback",
        gpu_id=0,
        replica=1,
        gpu_memory_utilization=0.8,
        config={"caller_value": "kept"},
        user_id="user-1",
        auto_start=False,
        inference_framework="vllm",
    )

    expected = service_module.parse_launch_config(
        {"framework": "vllm", "gpu_pool": [0], "dtype": "auto"}
    ).model_dump(mode="json")
    assert created["config"]["launch_config"] == expected
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).one().config["launch_config"] == expected


@pytest.mark.parametrize("presence_error", ["daemon unavailable", "query timed out"])
@pytest.mark.parametrize(
    ("failure_phase", "failure_call"),
    [("before-plan", 1), ("after-plan", 2)],
)
def test_container_create_fails_closed_when_collision_probe_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
    failure_phase: str,
    failure_call: int,
    presence_error: str,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    docker.fail_authoritative_presence_call = failure_call
    docker.authoritative_presence_error = presence_error
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    with pytest.raises(RuntimeError, match="replica deployment failed"):
        _create(service, replica=1)

    assert docker.created == [], failure_phase
    assert docker.authoritative_presence_calls == failure_call
    assert docker.released == docker.reserved
    with Session(engine) as session:
        assert session.exec(select(DeploymentReplicaDB)).all() == []
        assert session.exec(select(DeploymentDB)).all() == []
    assert service._claim_heartbeats == {}


def test_container_create_uses_model_snapshot_locked_during_plan_persist(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    with Session(engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == "model-1")
        ).one()
        model.model_path = "/app/models/model-1-v2"
        model.version = "v2"
        session.add(model)
        session.commit()

    created = _create(service, replica=1)

    server_argv = docker.create_kwargs[0]["server_argv"]
    assert "/app/models/model-1-v2" in server_argv
    assert "/app/models/model-1" not in [
        argument for argument in server_argv if argument != "/app/models/model-1-v2"
    ]
    assert created["config"]["_registry_artifact_signature"]["version"] == "v2"


def test_container_create_aborts_if_locked_model_invalidates_preplanned_fields(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    with Session(engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == "model-1")
        ).one()
        model.model_name = "Trusted Model"
        model.file_size = 24 * 1024**3
        session.add(model)
        session.commit()

    with pytest.raises(
        service_module.ReplicaOperationBusyError,
        match="model changed while planning",
    ):
        service.create_container_deployment(
            model_id="model-1",
            deployment_name="replica-group",
            replica=1,
            gpu_memory_utilization=None,
            config={
                "launch_config": {
                    "framework": "vllm",
                    "gpu_pool": [0],
                }
            },
            user_id="user-1",
            auto_start=True,
            inference_framework="vllm",
        )

    assert docker.created == []
    assert docker.released == [docker.reserved[0]]
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).all() == []


def test_deferred_create_fails_closed_if_persisted_model_artifact_drifts(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=1, defer_start=True)
    with Session(engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == "model-1")
        ).one()
        model.model_path = "/app/models/model-1-v2"
        model.version = "v2"
        session.add(model)
        session.commit()
    monkeypatch.setattr(
        service_module.model_registry_service,
        "get_model",
        lambda _model_id: {
            "model_id": "model-1",
            "model_name": "Test Model",
            "version": "v2",
            "model_path": "/app/models/model-1-v2",
            "model_type": "llm",
            "base_model_path": None,
            "source_type": "trained",
            "is_adapter": False,
            "file_size": None,
        },
    )

    with pytest.raises(
        service_module.DeploymentReplicaStateConflictError,
        match="artifact changed",
    ):
        service.start_deferred_deployment(
            created["deployment_id"],
            user_id="user-1",
        )

    assert docker.created == []


def test_auto_start_false_persists_stopped_plan_without_docker_create(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    result = _create(service, replica=2, auto_start=False)

    assert result["status"] == "stopped"
    assert [item["status"] for item in result["replica_instances"]] == [
        "stopped",
        "stopped",
    ]
    assert docker.created == []
    with Session(engine) as session:
        assert len(session.exec(select(DeploymentReplicaDB)).all()) == 2


def test_deferred_auto_start_returns_persisted_plan_before_docker_create(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    result = _create(service, replica=2, defer_start=True)

    assert result["status"] == "starting"
    assert [item["status"] for item in result["replica_instances"]] == [
        "pending",
        "pending",
    ]
    assert docker.created == []
    assert [port for port, _owner in docker.confirmed] == [11000, 11001]
    assert result["config"]["_deferred_auto_start"] is True
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token is None
        assert parent.replica_operation_kind is None


def test_deferred_auto_start_survives_process_exit_before_worker_runs(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, _engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=2, defer_start=True)

    restarted_service = service_module.DeploymentService()
    assert restarted_service.list_deferred_start_deployment_ids() == [
        created["deployment_id"]
    ]

    result = restarted_service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert result["status"] == "running"
    assert len(docker.created) == 2
    assert restarted_service.list_deferred_start_deployment_ids() == []


def test_deferred_auto_start_recovers_failed_parent_while_intent_remains(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=2, defer_start=True)

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.status = "failed"
        parent.error_message = "worker exited before clearing deferred intent"
        session.add(parent)
        session.commit()

    restarted_service = service_module.DeploymentService()
    assert restarted_service.list_deferred_start_deployment_ids() == [
        created["deployment_id"]
    ]

    result = restarted_service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert result["status"] == "running"
    assert restarted_service.list_deferred_start_deployment_ids() == []


def test_deferred_auto_start_creates_config_for_single_replica(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, _engine = replica_service
    docker = FakeDocker([11000])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=1, defer_start=True)
    calls: list[tuple[str, str | None, str | None]] = []

    def record_config(
        deployment_id: str,
        deployment_replica_id: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        calls.append((deployment_id, deployment_replica_id, user_id))
        return {}

    monkeypatch.setattr(service, "create_config_for_deployment", record_config)

    service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert calls == [
        (
            created["deployment_id"],
            created["replica_instances"][0]["replica_id"],
            "user-1",
        )
    ]


@pytest.mark.parametrize("defer_start", [False, True], ids=["direct", "deferred"])
def test_single_replica_auto_config_runs_after_lifecycle_claim_release(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
    defer_start: bool,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    docker.container_exists_authoritative = lambda name: (
        name in docker.container_ids or name in docker.foreign_names
    )
    docker.container_running_identity = lambda _container_id: True
    docker.probe_service_ready = lambda *_args, **_kwargs: True
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created_configs: list[dict] = []
    monkeypatch.setattr(
        model_config_service_module.model_config_service,
        "get_config_by_deployment_replica_id",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_config_service_module.model_config_service,
        "create_config",
        lambda **kwargs: created_configs.append(kwargs)
        or {**kwargs, "config_id": "config-1"},
    )

    created = _create(service, replica=1, defer_start=defer_start)
    if defer_start:
        service.start_deferred_deployment(
            created["deployment_id"],
            user_id="user-1",
        )

    assert len(created_configs) == 1
    assert created_configs[0]["deployment_id"] == created["deployment_id"]
    assert created_configs[0]["deployment_replica_id"] == (
        created["replica_instances"][0]["replica_id"]
    )
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        assert parent.replica_operation_token is None


def test_deferred_auto_start_does_not_auto_bind_multi_replica_config(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, _engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=2, defer_start=True)
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        service,
        "create_config_for_deployment",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert calls == []


def test_deferred_auto_start_recovers_only_missing_replica_after_worker_crash(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=2, defer_start=True)
    first, second = created["replica_instances"]
    docker.foreign_names.add(first["container_name"])
    docker.container_ids[first["container_name"]] = "a" * 64
    docker.container_running_identity = lambda _container_id: True
    docker.probe_service_ready = lambda *_args, **_kwargs: True

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        parent.replica_operation_token = "dead-worker-token"
        parent.replica_operation_kind = "start"
        parent.replica_operation_generation += 1
        parent.replica_operation_started_at = service_module.now_naive() - timedelta(
            minutes=10
        )
        parent.replica_operation_heartbeat_at = parent.replica_operation_started_at
        first_row = session.exec(
            select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == first["replica_id"]
            )
        ).one()
        first_row.status = "starting"
        session.add(parent)
        session.add(first_row)
        session.commit()

    restarted_service = service_module.DeploymentService()
    result = restarted_service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert result["status"] == "running"
    assert [item["status"] for item in result["replica_instances"]] == [
        "running",
        "running",
    ]
    assert docker.created == [second["container_name"]]


@pytest.mark.parametrize(
    ("runtime_state", "expected_targeted"),
    [
        ("missing", True),
        ("stopped", True),
        ("foreign", True),
        ("exact-running", False),
    ],
)
def test_deferred_auto_start_revalidates_running_healthy_child_runtime(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
    runtime_state: str,
    expected_targeted: bool,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=1, defer_start=True)
    child = created["replica_instances"][0]
    with Session(engine) as session:
        parent_row = session.exec(select(DeploymentDB)).one()
        parent_row.status = "running"
        child_row = session.exec(select(DeploymentReplicaDB)).one()
        child_row.status = "running"
        child_row.health_status = "HEALTHY"
        session.add(parent_row)
        session.add(child_row)
        session.commit()

    inspected: list[str] = []
    running_checks: list[str] = []
    original_get_managed_container_id = docker.get_managed_container_id
    if runtime_state != "missing":
        docker.foreign_names.add(child["container_name"])
        docker.container_ids[child["container_name"]] = "a" * 64

    def inspect_identity(
        container_name: str,
        deployment_id: str,
        replica_id: str,
    ) -> str:
        inspected.append(container_name)
        if runtime_state == "foreign":
            raise RuntimeError("managed replica container identity is invalid")
        return original_get_managed_container_id(
            container_name,
            deployment_id,
            replica_id,
        )

    def running_identity(container_id: str) -> bool:
        running_checks.append(container_id)
        return runtime_state == "exact-running"

    monkeypatch.setattr(docker, "get_managed_container_id", inspect_identity)
    monkeypatch.setattr(
        docker,
        "container_running_identity",
        running_identity,
        raising=False,
    )
    targeted: list[set[str]] = []

    def record_targets(
        deployment_id: str,
        _operation: str,
        _user_id: str | None,
        **kwargs,
    ) -> dict:
        targeted.append(set(kwargs["_replica_ids"]))
        return service._load_replica_group(deployment_id)

    monkeypatch.setattr(service, "_operate_replica_group", record_targets)
    monkeypatch.setattr(service, "create_config_for_deployment", lambda *_a, **_k: {})

    service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert targeted == ([{child["replica_id"]}] if expected_targeted else [set()])
    assert inspected == (
        [] if runtime_state == "missing" else [child["container_name"]]
    )
    assert running_checks == (
        ["a" * 64] if runtime_state in {"stopped", "exact-running"} else []
    )


def test_deferred_auto_start_failure_is_persisted_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000], fail_create_index=0)
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=1, defer_start=True)

    with pytest.raises(RuntimeError, match="replica group operation failed"):
        service.start_deferred_deployment(
            created["deployment_id"],
            user_id="user-1",
        )

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        child = session.exec(select(DeploymentReplicaDB)).one()
        assert parent.status == "failed"
        assert parent.error_message == "replica group degraded"
        assert child.status == "failed"
        assert child.error_message == "replica lifecycle operation failed"
        assert parent.replica_operation_token is None
        assert parent.config.get("_deferred_auto_start") is not True


def test_deferred_auto_start_is_consumed_only_once(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, _engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=2, defer_start=True)

    first = service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )
    second = service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert first["status"] == "running"
    assert second["status"] == "running"
    assert len(docker.created) == 2


def test_deferred_auto_start_does_not_resurrect_user_stopped_plan(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, _engine = replica_service
    docker = FakeDocker([11000])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=1, defer_start=True)
    service.stop_deployment(created["deployment_id"], user_id="user-1")

    result = service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert result["status"] == "stopped"
    assert result["replica_instances"][0]["status"] == "stopped"
    assert docker.created == []
    assert result["config"].get("_deferred_auto_start") is not True


def test_manual_replica_stop_cancels_whole_deferred_auto_start(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, _engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=2, defer_start=True)
    first_replica_id = created["replica_instances"][0]["replica_id"]

    service.stop_replica(
        created["deployment_id"],
        first_replica_id,
        user_id="user-1",
    )
    result = service.start_deferred_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert result["status"] == "degraded"
    assert result["config"].get("_deferred_auto_start") is not True
    assert docker.created == []


def test_group_start_runs_independent_replicas_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, _engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=2, auto_start=False)
    active = 0
    max_active = 0
    lock = threading.Lock()
    both_started = threading.Event()

    def start_planned(**_kwargs) -> None:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                both_started.set()
        both_started.wait(timeout=0.25)
        with lock:
            active -= 1

    monkeypatch.setattr(service, "_start_planned_replica", start_planned)

    result = service.start_deployment(
        created["deployment_id"],
        user_id="user-1",
    )

    assert result["status"] == "running"
    assert max_active == 2


def test_auto_start_plan_holds_durable_claim_before_rows_are_visible(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    original_load = service._load_replica_group
    observed_claims: list[tuple[str | None, str | None, int]] = []

    def load_with_concurrent_stop(deployment_id: str):
        if not observed_claims:
            with Session(engine) as session:
                parent = session.exec(select(DeploymentDB)).one()
                observed_claims.append(
                    (
                        parent.replica_operation_token,
                        parent.replica_operation_kind,
                        parent.replica_operation_generation,
                    )
                )
            with pytest.raises(
                service_module.ReplicaOperationBusyError,
                match="already in progress",
            ):
                service._claim_replica_operation(
                    deployment_id,
                    operation="stop",
                    replica_id=None,
                    user_id="user-1",
                )
        return original_load(deployment_id)

    monkeypatch.setattr(service, "_load_replica_group", load_with_concurrent_stop)

    result = _create(service, replica=2)

    token, operation, generation = observed_claims[0]
    assert token
    assert operation == "create"
    assert generation == 1
    with Session(engine) as session:
        parent = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == result["deployment_id"]
            )
        ).one()
        assert parent.replica_operation_token is None
        assert parent.replica_operation_generation == 1


def test_stale_creator_cannot_publish_success_after_generation_changes(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    def replace_claim() -> None:
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "recovery-owner"
            parent.replica_operation_kind = "restart"
            parent.replica_operation_generation += 1
            session.add(parent)
            session.commit()

    docker.on_wait = replace_claim

    with pytest.raises(
        service_module.ReplicaOperationLostError,
        match="ownership was lost",
    ):
        _create(service, replica=2)

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        children = session.exec(
            select(DeploymentReplicaDB).order_by(DeploymentReplicaDB.replica_index)
        ).all()
        assert parent.replica_operation_token == "recovery-owner"
        assert parent.replica_operation_generation == 2
        assert [child.status for child in children] == ["starting", "pending"]
    assert docker.removed == []
    assert [port for port, _owner in docker.confirmed] == [11000, 11001]


def test_stale_creator_failure_cannot_delete_reclaimed_plan(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000], fail_create_index=0)
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    def replace_claim() -> None:
        with Session(engine) as session:
            parent = session.exec(select(DeploymentDB)).one()
            parent.replica_operation_token = "recovery-owner"
            parent.replica_operation_kind = "restart"
            parent.replica_operation_generation += 1
            session.add(parent)
            session.commit()

    docker.on_create = replace_claim

    with pytest.raises(
        service_module.ReplicaOperationLostError,
        match="ownership was lost",
    ):
        _create(service, replica=1)

    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        child = session.exec(select(DeploymentReplicaDB)).one()
        assert parent.replica_operation_token == "recovery-owner"
        assert parent.replica_operation_generation == 2
        assert child.status == "starting"
    assert [port for port, _owner in docker.confirmed] == [11000]


def test_later_replica_failure_cleans_owned_containers_and_rolls_back_plan(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001, 11002], fail_create_index=2)
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    with pytest.raises(RuntimeError, match="replica deployment failed"):
        _create(service, replica=3)

    assert docker.removed == [
        docker.container_ids[name] for name in reversed(docker.created)
    ]
    assert [port for port, _owner in docker.released] == [11002, 11001, 11000]
    with Session(engine) as session:
        assert session.exec(select(DeploymentReplicaDB)).all() == []
        assert session.exec(select(DeploymentDB)).all() == []


def test_readiness_failure_after_create_still_cleans_the_created_container(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000, 11001], fail_wait_index=0)
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    with pytest.raises(RuntimeError, match="replica deployment failed"):
        _create(service, replica=2)

    assert docker.created
    assert docker.removed == [
        docker.container_ids[name] for name in reversed(docker.created)
    ]
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).all() == []


def test_transient_identity_inspection_failure_after_create_cleans_exact_container(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    original_get_managed_container_id = docker.get_managed_container_id
    inspect_attempts = 0

    def fail_first_inspection(
        container_name: str,
        deployment_id: str,
        replica_id: str,
    ) -> str:
        nonlocal inspect_attempts
        inspect_attempts += 1
        if inspect_attempts == 1:
            raise RuntimeError("managed replica container inspection failed")
        return original_get_managed_container_id(
            container_name,
            deployment_id,
            replica_id,
        )

    monkeypatch.setattr(
        docker,
        "get_managed_container_id",
        fail_first_inspection,
    )
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    with pytest.raises(RuntimeError, match="replica deployment failed"):
        _create(service, replica=1)

    assert inspect_attempts == 2
    assert docker.removed == [docker.container_ids[docker.created[0]]]
    assert [port for port, _owner in docker.released] == [11000]
    with Session(engine) as session:
        assert session.exec(select(DeploymentReplicaDB)).all() == []
        assert session.exec(select(DeploymentDB)).all() == []


def test_unverifiable_created_container_retains_recoverable_plan_and_port(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    inspect_attempts = 0

    def reject_unverifiable_identity(
        container_name: str,
        deployment_id: str,
        replica_id: str,
    ) -> str:
        nonlocal inspect_attempts
        assert container_name
        assert deployment_id
        assert replica_id
        inspect_attempts += 1
        raise RuntimeError("managed replica container identity is invalid")

    monkeypatch.setattr(
        docker,
        "get_managed_container_id",
        reject_unverifiable_identity,
    )
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    with pytest.raises(
        RuntimeError,
        match="identity could not be verified; deployment plan retained",
    ):
        _create(service, replica=1)

    assert inspect_attempts == 2
    assert docker.removed == []
    assert docker.released == []
    assert [port for port, _owner in docker.confirmed] == [11000]
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        child = session.exec(select(DeploymentReplicaDB)).one()
        assert parent.status == "failed"
        assert parent.replica_operation_token is None
        assert child.status == "failed"
        assert child.port == 11000


def test_cleanup_failure_is_fixed_and_other_owned_cleanup_still_runs(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker(
        [11000, 11001, 11002],
        fail_create_index=2,
        fail_remove_indexes={0},
    )
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    heartbeat_events: dict[str, threading.Event] = {}

    def register_observable_heartbeat(claim) -> None:
        event = threading.Event()
        heartbeat_events[claim.token] = event
        service._claim_heartbeats[claim.token] = (
            event,
            SimpleNamespace(join=lambda timeout=None: None),
        )

    monkeypatch.setattr(
        service,
        "_register_claim_heartbeat",
        register_observable_heartbeat,
    )

    with pytest.raises(
        RuntimeError,
        match="replica deployment failed; owned cleanup incomplete",
    ) as exc_info:
        _create(service, replica=3)

    assert "/app/models" not in str(exc_info.value)
    assert docker.removed == [
        docker.container_ids[name] for name in reversed(docker.created)
    ]
    assert [port for port, _owner in docker.confirmed] == [11000, 11001, 11002]
    with Session(engine) as session:
        parent = session.exec(select(DeploymentDB)).one()
        children = session.exec(select(DeploymentReplicaDB)).all()
        stale_token = parent.replica_operation_token
        stale_generation = parent.replica_operation_generation
        assert parent.status == "failed"
        assert "recovery required" in (parent.error_message or "")
        assert len(children) == 3
        assert all(child.status == "failed" for child in children)

    assert stale_token
    assert heartbeat_events[stale_token].is_set()
    assert service._claim_heartbeats == {}

    recovery_claim = service._claim_replica_operation(
        parent.deployment_id,
        operation="delete",
        replica_id=None,
        user_id="user-1",
    )
    assert recovery_claim.generation > stale_generation
    assert service._release_replica_operation(recovery_claim) is True
    assert service._claim_heartbeats == {}


def test_foreign_same_name_container_blocks_before_create_or_removal(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    original_builder = service._build_managed_container_name
    docker.foreign_names.add(
        original_builder("00000000-0000-0000-0000-000000000000", "Test Model", "vllm")
    )
    monkeypatch.setattr(
        service_module,
        "uuid4",
        lambda: type(
            "FixedUuid",
            (),
            {
                "hex": "owned-reservation",
                "__str__": lambda self: "00000000-0000-0000-0000-000000000000",
            },
        )(),
    )

    with pytest.raises(RuntimeError, match="replica deployment failed"):
        _create(service, replica=1)

    assert docker.created == []
    assert docker.removed == []
    with Session(engine) as session:
        assert session.exec(select(DeploymentReplicaDB)).all() == []


def test_insufficient_gpu_plan_has_no_port_or_docker_mutation(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, engine = replica_service
    docker = FakeDocker([11000])
    docker.get_gpu_memory_usage = lambda: {0: {"free_mb": 20_000}}
    monkeypatch.setattr(service_module, "docker_deployer", docker)

    with pytest.raises(ValueError, match="GPU inventory"):
        _create(service, replica=2)

    assert docker.reserved == []
    assert docker.created == []
    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).all() == []


def test_detail_and_list_reads_include_children_in_replica_order(
    monkeypatch: pytest.MonkeyPatch,
    replica_service,
) -> None:
    service, _engine = replica_service
    docker = FakeDocker([11000, 11001])
    monkeypatch.setattr(service_module, "docker_deployer", docker)
    created = _create(service, replica=2, auto_start=False)

    detail = service.get_deployment(created["deployment_id"])
    listed, total = service.list_deployments(user_id="user-1")

    assert detail is not None
    assert [item["replica_index"] for item in detail["replica_instances"]] == [
        0,
        1,
    ]
    assert total == 1
    assert listed[0]["replica_instances"] == detail["replica_instances"]
