import asyncio
import importlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from train_factory.api.routes import deployment_routes, model_config_routes
from train_factory.deployment.docker_deployer import DockerDeployer, docker_deployer
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.storage.services.model_registry_service import (
    _remove_deployment_container,
)


deployment_service_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)

ADMIN = {
    "user_id": "admin-1",
    "username": "admin",
    "is_admin": True,
}
REGULAR_USER = {
    "user_id": "user-1",
    "username": "alice",
    "is_admin": False,
}
SHARED_ENDPOINT = "http://xinference:9997"


@pytest.fixture(autouse=True)
def _isolate_managed_container_runtime(monkeypatch):
    """Keep lifecycle unit tests on the immutable-ID Docker contract."""
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "container_exists_authoritative",
        lambda _name: True,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "get_legacy_managed_container_id",
        lambda _name, **_identity: "owned-container-id",
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "container_running_identity",
        lambda _container_id: True,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "stop_container_identity",
        lambda _container_id: True,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "remove_container_identity",
        lambda _container_id: True,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "restart_container_identity",
        lambda _container_id: (True, "restarted"),
    )


class _FakeResult:
    def __init__(self, *, first=None, rows=None, one=None, rowcount=1):
        self._first = first
        self._rows = list(rows or [])
        self._one = one
        self.rowcount = rowcount

    def first(self):
        return self._first

    def all(self):
        return self._rows

    def one(self):
        if self._one is not None:
            return self._one
        if self._first is not None:
            return self._first
        if len(self._rows) == 1:
            return self._rows[0]
        return 0


class _FakeSession:
    def __init__(self, results):
        results = list(results)
        self._deployments = []
        for result in results:
            candidates = [result._first, *result._rows]
            for candidate in candidates:
                if (
                    isinstance(candidate, DeploymentDB)
                    and candidate not in self._deployments
                ):
                    self._deployments.append(candidate)
        self._results = iter(
            result
            for result in results
            if not any(
                isinstance(candidate, DeploymentDB)
                for candidate in [result._first, *result._rows]
            )
        )
        model_id = self._deployments[0].model_id if self._deployments else "model-1"
        self._model = ModelRegistryDB(
            model_id=model_id,
            model_name="owned-model",
            model_type="embedding",
            model_path="/models/owned-model",
            source_type="trained",
            status="available",
        )
        self._gate = ModelArtifactMembershipGateDB(gate_id=1)
        self.added = []
        self.deleted = []
        self.commits = 0
        self.rollbacks = 0
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def exec(self, statement):
        self.statements.append(statement)
        sql = str(statement).lower()
        is_update = sql.lstrip().startswith("update ")

        if "model_artifact_membership_gate" in sql:
            if is_update:
                return _FakeResult(rowcount=1)
            return _FakeResult(first=self._gate, rows=[self._gate])

        if "from model_registry" in sql or sql.lstrip().startswith("update model_registry "):
            if is_update:
                return _FakeResult(rowcount=1)
            return _FakeResult(first=self._model, rows=[self._model])

        if "deployments" in sql:
            if is_update:
                return _FakeResult(rowcount=1)
            if sql.lstrip().startswith(
                "select deployments.replica_operation_generation "
            ):
                generation = max(
                    (
                        deployment.replica_operation_generation or 0
                        for deployment in self._deployments
                    ),
                    default=0,
                ) + 1
                return _FakeResult(one=generation)
            if (
                "deployments.container_name =" in sql
                and "deployments.deployment_id !=" in sql
            ):
                return _FakeResult(first=None, rows=[])
            if self._deployments:
                return _FakeResult(
                    first=self._deployments[0],
                    rows=list(self._deployments),
                )
            try:
                return next(self._results)
            except StopIteration:
                return _FakeResult(first=None, rows=[])

        empty_tables = (
            "deployment_replicas",
            "model_configs",
            "loaded_adapters",
            "external_sync_tasks",
            "external_sync_training_targets",
            "external_sync_trainings",
            "training_tasks",
            "generation_tasks",
            "evaluation_tasks",
            "deep_evaluation_tasks",
            "milvus_collections",
            "external_api_configs",
        )
        if any(table in sql for table in empty_tables):
            return _FakeResult(rowcount=1) if is_update else _FakeResult(rows=[])

        if is_update or sql.lstrip().startswith("delete "):
            return _FakeResult(rowcount=1)
        try:
            return next(self._results)
        except StopIteration:
            return _FakeResult(first=None, rows=[])

    def add(self, value):
        self.added.append(value)
        if isinstance(value, DeploymentDB) and value not in self._deployments:
            self._deployments.append(value)

    def delete(self, value):
        self.deleted.append(value)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def expire_all(self):
        return None

    def refresh(self, value):  # noqa: ARG002
        return None


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _deployment(**overrides):
    values = {
        "deployment_id": "deadbeef-0000-0000-0000-000000000000",
        "model_id": "model-1",
        "model_uid": "owned-model",
        "deployment_name": "deployment",
        "xinference_endpoint": SHARED_ENDPOINT,
        "deploy_mode": "shared",
        "inference_framework": "xinference",
        "container_name": None,
        "gpu_id": None,
        "config": {"runtime_managed": False, "read_only": True},
        "status": "running",
        "user_id": "user-1",
    }
    values.update(overrides)
    return DeploymentDB(**values)


def test_bind_existing_requires_admin_before_any_side_effect(monkeypatch):
    calls = []
    monkeypatch.setattr(
        deployment_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "bind_existing_model",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deployment_routes.bind_existing_model(
                deployment_routes.BindExistingModelRequest(
                    endpoint="https://inference.example.com",
                    model_uid="model-1",
                ),
                REGULAR_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert calls == []


def test_bind_route_ignores_client_container_and_gpu_identifiers_when_auth_disabled(
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(
        deployment_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=False),
    )
    monkeypatch.setattr(
        deployment_routes,
        "validate_user_outbound_url",
        lambda endpoint, user_id: endpoint.rstrip("/"),
    )

    def fake_bind(**kwargs):
        captured.update(kwargs)
        return {
            "deployment_id": "deployment-1",
            "model_id": "model-1",
            "model_uid": "model-1",
            "xinference_endpoint": SHARED_ENDPOINT,
            "replica": 1,
            "gpu_memory_utilization": 0.0,
            "deploy_mode": "shared",
            "container_name": None,
            "gpu_id": None,
            "inference_framework": "xinference",
            "config": {"runtime_managed": False, "read_only": True},
            "status": "running",
            "user_id": None,
        }

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "bind_existing_model",
        fake_bind,
    )

    asyncio.run(
        deployment_routes.bind_existing_model(
            deployment_routes.BindExistingModelRequest(
                endpoint=SHARED_ENDPOINT,
                model_uid="model-1",
                container_name="trainfactory-mysql",
                gpu_id=7,
            ),
            {"user_id": None, "username": "anonymous"},
        )
    )

    assert captured["container_name"] is None
    assert captured["gpu_id"] is None


def test_bound_record_is_explicitly_unmanaged_and_drops_runtime_identifiers(
    monkeypatch,
):
    session = _FakeSession([_FakeResult(first=None)])
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "list_models",
        lambda **kwargs: ([], 0),
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "register_model",
        lambda **kwargs: {"model_id": "model-1"},
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_create_config_for_deployment",
        lambda *args, **kwargs: None,
    )

    result = deployment_service_module.deployment_service.bind_existing_model(
        endpoint=SHARED_ENDPOINT,
        model_uid="model-1",
        container_name="trainfactory-mysql",
        gpu_id=7,
        user_id="admin-1",
        config={"runtime_managed": True, "read_only": False},
    )

    stored = session.added[-1]
    assert stored.container_name is None
    assert stored.gpu_id is None
    assert stored.config["runtime_managed"] is False
    assert stored.config["read_only"] is True
    assert result["container_name"] is None


def test_duplicate_endpoint_and_model_uid_is_rejected_before_registry_mutation(
    monkeypatch,
):
    existing = _deployment()
    session = _FakeSession([_FakeResult(first=existing)])
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    def unexpected(*args, **kwargs):  # noqa: ARG001
        pytest.fail("model registry mutated before duplicate binding check")

    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "list_models",
        unexpected,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "register_model",
        unexpected,
    )

    with pytest.raises(ValueError, match="already bound"):
        deployment_service_module.deployment_service.bind_existing_model(
            endpoint=f"{SHARED_ENDPOINT}/",
            model_uid="owned-model",
            user_id="admin-1",
        )


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_unmanaged_binding_lifecycle_is_read_only(monkeypatch, operation):
    deployment = _deployment()
    session = _FakeSession([_FakeResult(first=deployment)])
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "model_path": "/models/owned-model",
            "model_type": "embedding",
            "source_type": "external_bind",
        },
    )

    def unexpected(*args, **kwargs):  # noqa: ARG001
        pytest.fail("unmanaged binding performed a runtime operation")

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        unexpected,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "restart_container",
        unexpected,
    )

    method = getattr(
        deployment_service_module.deployment_service,
        f"{operation}_deployment",
    )
    with pytest.raises(ValueError, match="read-only"):
        method(deployment.deployment_id)


def test_deleting_unmanaged_binding_never_terminates_remote_model(monkeypatch):
    deployment = _deployment()
    session = _FakeSession(
        [
            _FakeResult(first=deployment),
            _FakeResult(rows=[]),  # external sync tasks
            _FakeResult(rows=[]),  # external sync training targets
            _FakeResult(rows=[]),  # model configs
        ]
    )
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )

    def unexpected(*args, **kwargs):  # noqa: ARG001
        pytest.fail("deleting an unmanaged binding terminated the remote model")

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        unexpected,
    )

    assert (
        deployment_service_module.deployment_service.delete_deployment(
            deployment.deployment_id
        )
        is True
    )
    assert session.deleted == [deployment]


def test_missing_model_provenance_fails_closed_before_shared_runtime_mutation(
    monkeypatch,
):
    deployment = _deployment(config={})
    session = _FakeSession([_FakeResult(first=deployment)])
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: None,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda *args, **kwargs: pytest.fail(
            "deployment with unverifiable provenance reached the runtime"
        ),
    )

    with pytest.raises(ValueError, match="read-only"):
        deployment_service_module.deployment_service.stop_deployment(
            deployment.deployment_id
        )


@pytest.mark.parametrize("operation", ["start", "stop", "restart", "delete"])
def test_forged_container_name_cannot_reach_docker_lifecycle(
    monkeypatch,
    operation,
):
    status = "pending" if operation == "start" else "running"
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-mysql",
        config={},
        status=status,
    )
    results = [_FakeResult(first=deployment)]
    if operation == "delete":
        results.append(_FakeResult(rows=[]))
    session = _FakeSession(results)
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "model_path": "/app/models/model-1",
            "model_type": "embedding",
        },
    )

    def unexpected(*args, **kwargs):  # noqa: ARG001
        pytest.fail("forged container name reached Docker")

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_start_container_deployment",
        unexpected,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "stop_container",
        unexpected,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "restart_container",
        unexpected,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "remove_container",
        unexpected,
    )

    method = getattr(
        deployment_service_module.deployment_service,
        f"{operation}_deployment",
    )
    with pytest.raises(ValueError, match="managed container name"):
        method(deployment.deployment_id)


def test_service_generated_container_name_is_accepted(monkeypatch):
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-xf-owned-model-deadbeef",
        config={},
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {"model_id": model_id, "model_name": "owned-model"},
    )

    deployment_service_module.deployment_service._require_managed_container(
        deployment
    )


def test_force_delete_model_rejects_forged_container_name_before_docker(
    monkeypatch,
):
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-mysql",
        config={},
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "source_type": "trained",
        },
    )
    monkeypatch.setattr(
        docker_deployer,
        "remove_container",
        lambda *_args, **_kwargs: pytest.fail(
            "forged container name reached Docker during model deletion"
        ),
    )

    with pytest.raises(ValueError, match="managed container name"):
        _remove_deployment_container(deployment, "model-1")


def test_force_delete_model_terminates_managed_shared_runtime(monkeypatch):
    deployment = _deployment(config={})
    terminated = []
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda endpoint, user_id: SimpleNamespace(
            terminate_model=lambda model_uid: terminated.append(
                (endpoint, user_id, model_uid)
            )
        ),
    )

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_require_replica_operation_ownership",
        lambda _claim: None,
    )
    _remove_deployment_container(
        deployment,
        "model-1",
        replica_operation_claim=deployment_service_module.ReplicaOperationClaim(
            deployment_id=deployment.deployment_id,
            token="delete-token",
            generation=1,
            operation="delete",
            replica_id=None,
        ),
    )

    assert terminated == [(SHARED_ENDPOINT, "user-1", "owned-model")]


def test_force_delete_model_keeps_shared_runtime_failure_retryable(monkeypatch):
    deployment = _deployment(config={})
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    class FailingClient:
        def terminate_model(self, _model_uid):
            raise OSError("endpoint unavailable")

        def get_model(self, _model_uid):
            return {"id": "owned-model"}

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: FailingClient(),
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_require_replica_operation_ownership",
        lambda _claim: None,
    )

    with pytest.raises(RuntimeError, match="deployment-1|deadbeef"):
        _remove_deployment_container(
            deployment,
            "model-1",
            replica_operation_claim=deployment_service_module.ReplicaOperationClaim(
                deployment_id=deployment.deployment_id,
                token="delete-token",
                generation=1,
                operation="delete",
                replica_id=None,
            ),
        )


@pytest.mark.parametrize("initial_status", ["pending", "failed", "stopped"])
def test_stop_does_not_release_transient_port_for_non_running_container(
    monkeypatch,
    initial_status,
):
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-xf-owned-model-deadbeef",
        config={},
        status=initial_status,
        port=10001,
    )
    session = _FakeSession([_FakeResult(first=deployment)])
    released_ports = []
    stopped_containers = []
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "container_exists_authoritative",
        lambda _container_name: False,
    )
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_require_managed_container",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "stop_container",
        lambda container_name: stopped_containers.append(container_name),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "release_port",
        lambda port: released_ports.append(port),
    )

    result = deployment_service_module.deployment_service.stop_deployment(
        deployment.deployment_id
    )

    assert result["status"] == "stopped"
    assert deployment.status == "stopped"
    assert released_ports == []
    assert stopped_containers == []
    # Durable parent fencing plus verified cleanup transitions.
    assert session.commits == (2 if initial_status == "stopped" else 4)


def test_repeat_stop_does_not_release_an_unrelated_transient_port_reservation(
    monkeypatch,
):
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-xf-owned-model-deadbeef",
        config={},
        status="stopped",
        port=10001,
    )
    session = _FakeSession([_FakeResult(first=deployment)])
    released_ports = []
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_require_managed_container",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "release_port",
        lambda port, owner_token=None: released_ports.append((port, owner_token)),
    )

    result = deployment_service_module.deployment_service.stop_deployment(
        deployment.deployment_id
    )

    assert result["status"] == "stopped"
    assert released_ports == []


def test_transient_port_reservations_can_only_be_released_by_their_owner(
    monkeypatch,
):
    deployer = DockerDeployer()
    monkeypatch.setattr(deployer, "_get_db_reserved_ports", lambda: set())
    monkeypatch.setattr(deployer, "_is_port_in_use", lambda _port: False)

    first_port = deployer.find_available_port(
        start=10001,
        end=10003,
        owner_token="create-a",
    )

    assert first_port == 10001
    assert deployer.release_port(first_port, owner_token="old-stopped-b") is False
    assert (
        deployer.find_available_port(
            start=10001,
            end=10003,
            owner_token="create-c",
        )
        == 10002
    )
    assert deployer.confirm_port(first_port, owner_token="create-a") is True


def test_explicit_container_port_does_not_confirm_a_transient_reservation(
    monkeypatch,
):
    confirmed_ports = []
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda _model_id: {"model_id": "model-1", "file_size": None},
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "create_deployment",
        lambda **_kwargs: {
            "deployment_id": "deployment-1",
            "container_name": "trainfactory-vllm-model-deadbeef",
            "xinference_endpoint": "",
        },
    )
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: _FakeSession([_FakeResult(first=None)]),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "confirm_port",
        lambda port, *, owner_token: confirmed_ports.append((port, owner_token)),
    )

    result = deployment_service_module.deployment_service.create_container_deployment(
        "model-1",
        gpu_id=0,
        gpu_memory_utilization=0.5,
        port=10001,
        auto_start=False,
    )

    assert result["deployment_id"] == "deployment-1"
    assert confirmed_ports == []


def test_auto_assigned_port_is_confirmed_by_the_reservation_owner(monkeypatch):
    reservations = []
    confirmations = []
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda _model_id: {"model_id": "model-1", "file_size": None},
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "find_available_port",
        lambda *, owner_token: reservations.append(owner_token) or 10001,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "confirm_port",
        lambda port, *, owner_token: confirmations.append((port, owner_token)) or True,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "create_deployment",
        lambda **_kwargs: {
            "deployment_id": "deployment-1",
            "container_name": "trainfactory-vllm-model-deadbeef",
            "xinference_endpoint": "",
        },
    )
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: _FakeSession([_FakeResult(first=None)]),
    )

    result = deployment_service_module.deployment_service.create_container_deployment(
        "model-1",
        gpu_id=0,
        gpu_memory_utilization=0.5,
        auto_start=False,
    )

    assert result["deployment_id"] == "deployment-1"
    assert len(reservations) == 1
    assert reservations[0]
    assert confirmations == [(10001, reservations[0])]


def test_failed_auto_assigned_deployment_releases_only_its_reservation(monkeypatch):
    reservations = []
    releases = []
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda _model_id: {"model_id": "model-1", "file_size": None},
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "find_available_port",
        lambda *, owner_token: reservations.append(owner_token) or 10001,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "release_port",
        lambda port, *, owner_token: releases.append((port, owner_token)) or True,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "create_deployment",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("create failed")),
    )

    with pytest.raises(RuntimeError, match="create failed"):
        deployment_service_module.deployment_service.create_container_deployment(
            "model-1",
            gpu_id=0,
            gpu_memory_utilization=0.5,
            auto_start=False,
        )

    assert len(reservations) == 1
    assert releases == [(10001, reservations[0])]


def test_stop_commits_container_cleanup_without_releasing_transient_port(monkeypatch):
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-xf-owned-model-deadbeef",
        config={},
        status="running",
        port=10001,
    )
    events = []

    class RecordingSession(_FakeSession):
        def commit(self):
            events.append(("commit", deployment.status))
            super().commit()

        def refresh(self, value):
            events.append(("refresh", value.status))

    session = RecordingSession([_FakeResult(first=deployment)])
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_require_managed_container",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )

    def stop_container(_container_name):
        events.append(("stop_container", deployment.status))
        return True

    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "stop_container_identity",
        stop_container,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "release_port",
        lambda _port: events.append(("release_port", deployment.status)),
    )

    deployment_service_module.deployment_service.stop_deployment(
        deployment.deployment_id
    )

    assert events == [
        ("commit", "running"),
        ("commit", "stopping"),
        ("commit", "stopping"),
        ("stop_container", "stopping"),
        ("commit", "stopped"),
        ("commit", "stopped"),
    ]


def test_stop_keeps_container_deployment_retryable_when_cleanup_fails(
    monkeypatch,
):
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-xf-owned-model-deadbeef",
        config={},
        port=10001,
    )
    session = _FakeSession([_FakeResult(first=deployment)])
    released_ports = []
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "source_type": "trained",
        },
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "stop_container_identity",
        lambda _name: False,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "release_port",
        lambda port: released_ports.append(port),
    )

    with pytest.raises(RuntimeError, match="stop container"):
        deployment_service_module.deployment_service.stop_deployment(
            deployment.deployment_id
        )

    assert deployment.status == "stopping"
    assert "stop container" in deployment.error_message.lower()
    assert released_ports == []


def test_stop_does_not_release_port_when_container_stop_raises(monkeypatch):
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-xf-owned-model-deadbeef",
        config={},
        port=10001,
    )
    session = _FakeSession([_FakeResult(first=deployment)])
    released_ports = []
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_require_managed_container",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "stop_container_identity",
        lambda _name: (_ for _ in ()).throw(OSError("docker unavailable")),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "release_port",
        lambda port: released_ports.append(port),
    )

    with pytest.raises(RuntimeError, match="docker unavailable"):
        deployment_service_module.deployment_service.stop_deployment(
            deployment.deployment_id
        )

    assert deployment.status == "stopping"
    assert "docker unavailable" in deployment.error_message.lower()
    assert released_ports == []


def test_stop_shared_deployment_does_not_release_container_port(monkeypatch):
    deployment = _deployment(config={}, port=10001)
    session = _FakeSession([_FakeResult(first=deployment)])
    released_ports = []
    terminated_models = []
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: SimpleNamespace(
            terminate_model=lambda model_uid: terminated_models.append(model_uid)
        ),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "release_port",
        lambda port: released_ports.append(port),
    )

    result = deployment_service_module.deployment_service.stop_deployment(
        deployment.deployment_id
    )

    assert result["status"] == "stopped"
    assert terminated_models == ["owned-model"]
    assert released_ports == []


def test_stop_unmanaged_container_does_not_release_reserved_port(monkeypatch):
    deployment = _deployment(
        deploy_mode="container",
        container_name="foreign-container",
        port=10001,
    )
    session = _FakeSession([_FakeResult(first=deployment)])
    released_ports = []
    stopped_containers = []
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "stop_container",
        lambda container_name: stopped_containers.append(container_name),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "release_port",
        lambda port: released_ports.append(port),
    )

    with pytest.raises(ValueError, match="read-only"):
        deployment_service_module.deployment_service.stop_deployment(
            deployment.deployment_id
        )

    assert released_ports == []
    assert stopped_containers == []


def test_stop_keeps_shared_deployment_retryable_when_termination_fails(
    monkeypatch,
):
    deployment = _deployment(config={})
    session = _FakeSession([_FakeResult(first=deployment)])
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "source_type": "trained",
        },
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )

    class FailingClient:
        def terminate_model(self, _model_uid):
            raise RuntimeError("runtime unavailable")

        def get_model(self, _model_uid):
            return {"id": "owned-model"}

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: FailingClient(),
    )

    with pytest.raises(RuntimeError, match="remains present"):
        deployment_service_module.deployment_service.stop_deployment(
            deployment.deployment_id
        )

    assert deployment.status == "stopping"
    assert "remains present" in deployment.error_message.lower()


def test_start_shared_failure_terminates_possibly_launched_runtime(monkeypatch):
    deployment = _deployment(
        config={},
        status="pending",
        model_uid=None,
        gpu_id=0,
    )
    session = _FakeSession([_FakeResult(first=deployment)])
    terminated = []
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "model_path": "/models/owned-model",
            "model_type": "embedding",
            "source_type": "trained",
        },
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_start_external_deployment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("launch response lost")
        ),
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda endpoint, user_id: SimpleNamespace(
            terminate_model=lambda model_uid: terminated.append(
                (endpoint, user_id, model_uid)
            )
        ),
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="launch response lost"):
        deployment_service_module.deployment_service.start_deployment(
            deployment.deployment_id
        )

    assert terminated == [
        (SHARED_ENDPOINT, "user-1", "owned-model-deadbeef")
    ]
    assert deployment.model_uid == "owned-model-deadbeef"
    assert deployment.status == "failed"


def test_start_container_cleanup_failure_remains_stopping_and_retryable(monkeypatch):
    deployment = _deployment(
        config={},
        status="pending",
        model_uid=None,
        deploy_mode="container",
        container_name="trainfactory-xf-owned-model-deadbeef",
        port=10001,
        gpu_id=0,
    )
    session = _FakeSession([_FakeResult(first=deployment)])
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "model_path": "/models/owned-model",
            "model_type": "embedding",
            "source_type": "trained",
        },
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_require_managed_container",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_start_container_deployment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("container readiness failed")
        ),
    )
    runtime_presence = iter((False, True))
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "container_exists_authoritative",
        lambda _container_name: next(runtime_presence),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "remove_container_identity",
        lambda _container_name: False,
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="container readiness failed"):
        deployment_service_module.deployment_service.start_deployment(
            deployment.deployment_id
        )

    assert deployment.model_uid == "owned-model-deadbeef"
    assert deployment.status == "stopping"
    assert "runtime cleanup failed" in deployment.error_message.lower()


def test_delete_keeps_container_record_when_cleanup_fails(monkeypatch):
    deployment = _deployment(
        deploy_mode="container",
        container_name="trainfactory-xf-owned-model-deadbeef",
        config={},
    )
    session = _FakeSession(
        [
            _FakeResult(first=deployment),
            _FakeResult(rows=[]),  # external sync tasks
            _FakeResult(rows=[]),  # external sync training targets
            _FakeResult(rows=[]),  # model configs
            _FakeResult(first=None),  # shared container lookup
        ]
    )
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "source_type": "trained",
        },
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "remove_container_identity",
        lambda _name: False,
    )

    with pytest.raises(RuntimeError, match="remove container"):
        deployment_service_module.deployment_service.delete_deployment(
            deployment.deployment_id
        )

    assert session.deleted == []


def test_delete_keeps_shared_record_when_termination_fails(monkeypatch):
    deployment = _deployment(config={})
    session = _FakeSession(
        [
            _FakeResult(first=deployment),
            _FakeResult(rows=[]),  # external sync tasks
            _FakeResult(rows=[]),  # external sync training targets
            _FakeResult(rows=[]),  # model configs
        ]
    )
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "source_type": "trained",
        },
    )

    class FailingClient:
        def terminate_model(self, _model_uid):
            raise RuntimeError("runtime unavailable")

        def get_model(self, _model_uid):
            return {"id": "owned-model"}

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: FailingClient(),
    )

    with pytest.raises(RuntimeError, match="remains present"):
        deployment_service_module.deployment_service.delete_deployment(
            deployment.deployment_id
        )

    assert session.deleted == []


@pytest.mark.parametrize("status", ["starting", "restarting", "stopping"])
def test_delete_terminates_shared_runtime_in_every_active_state(
    monkeypatch,
    status,
):
    deployment = _deployment(config={}, status=status)
    session = _FakeSession(
        [
            _FakeResult(first=deployment),
            _FakeResult(rows=[]),  # external sync tasks
            _FakeResult(rows=[]),  # external sync training targets
            _FakeResult(rows=[]),  # model configs
        ]
    )
    terminated = []
    monkeypatch.setattr(
        deployment_service_module,
        "get_session",
        lambda: session,
    )
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_name": "owned-model",
            "source_type": "trained",
        },
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda endpoint, user_id: SimpleNamespace(
            terminate_model=lambda model_uid: terminated.append(
                (endpoint, user_id, model_uid)
            )
        ),
    )

    assert deployment_service_module.deployment_service.delete_deployment(
        deployment.deployment_id
    ) is True

    assert terminated == [(SHARED_ENDPOINT, "user-1", "owned-model")]
    assert session.deleted == [deployment]


def test_sync_all_running_scopes_model_ids_to_each_endpoint(monkeypatch):
    first = _deployment(
        deployment_id="11111111-0000-0000-0000-000000000000",
        xinference_endpoint="https://inference-a.example.com",
        model_uid="shared-name",
    )
    second = _deployment(
        deployment_id="22222222-0000-0000-0000-000000000000",
        xinference_endpoint="https://inference-b.example.com",
        model_uid="shared-name",
    )
    session = _FakeSession([_FakeResult(rows=[first, second])])
    monkeypatch.setattr(deployment_service_module, "get_session", lambda: session)
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )
    deployments_by_id = {
        deployment.deployment_id: deployment for deployment in (first, second)
    }

    def persist_snapshot(snapshot, *, status=None, error_message=None):
        deployment = deployments_by_id[snapshot.deployment_id]
        if status is not None:
            deployment.status = status
        return True

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_write_legacy_status_sync_state",
        persist_snapshot,
    )

    class Client:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def list_models(self):
            if self.endpoint.endswith("inference-a.example.com"):
                return [{"id": "shared-name"}]
            return []

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda endpoint, **_kwargs: Client(endpoint),
    )

    updated = deployment_service_module.deployment_service.sync_all_running()

    assert updated == 1
    assert first.status == "running"
    assert second.status == "stopped"


def test_sync_all_running_keeps_endpoint_state_when_discovery_fails(monkeypatch):
    deployment = _deployment(
        xinference_endpoint="https://unavailable.example.com",
        model_uid="model-1",
    )
    session = _FakeSession([_FakeResult(rows=[deployment])])
    monkeypatch.setattr(deployment_service_module, "get_session", lambda: session)
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_sync_configs_from_deployments",
        lambda *_args, **_kwargs: None,
    )

    class FailingClient:
        def list_models(self):
            raise RuntimeError("endpoint unavailable")

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: FailingClient(),
    )

    updated = deployment_service_module.deployment_service.sync_all_running()

    assert updated == 0
    assert deployment.status == "running"


def test_managed_container_listing_covers_all_frameworks(monkeypatch):
    output = "\n".join(
        [
            "trainfactory-xf-embedding-deadbeef",
            "trainfactory-vllm-embedding-0123abcd",
            "trainfactory-sglang-embedding-fedcba98",
            "other-xf-embedding-deadbeef",
            "trainfactory-mysql",
            "trainfactory-xf-invalid-suffix",
        ]
    )
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "trainfactory")
    monkeypatch.setattr(
        docker_deployer,
        "_run_command",
        lambda *_args, **_kwargs: (True, output),
    )

    containers = docker_deployer.list_xinference_containers()

    assert containers == [
        "trainfactory-xf-embedding-deadbeef",
        "trainfactory-vllm-embedding-0123abcd",
        "trainfactory-sglang-embedding-fedcba98",
    ]


def test_orphan_cleanup_never_removes_non_managed_container(monkeypatch):
    removed = []
    monkeypatch.setattr(
        docker_deployer,
        "list_xinference_containers",
        lambda: [
            "trainfactory-xf-orphan-deadbeef",
            "trainfactory-xf-active-0123abcd",
        ],
    )
    monkeypatch.setattr(
        docker_deployer,
        "remove_container",
        lambda name: removed.append(name) or True,
    )

    count = docker_deployer.cleanup_orphan_containers(
        {"trainfactory-xf-active-0123abcd"}
    )

    assert count == 0
    assert removed == []


def test_regular_discovery_only_returns_owned_running_model_without_path(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        deployment_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_discovery_endpoint",
        lambda endpoint, user_id: endpoint.rstrip("/"),
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "list_deployments",
        lambda **kwargs: (
            [
                {
                    "deployment_id": "deployment-1",
                    "model_id": "model-1",
                    "model_uid": "owned-model",
                    "xinference_endpoint": SHARED_ENDPOINT,
                    "inference_framework": "xinference",
                    "status": "running",
                    "user_id": "user-1",
                }
            ],
            1,
        ),
    )

    def fake_request(method, url, user_id, **kwargs):  # noqa: ARG001
        calls.append(url)
        assert url == f"{SHARED_ENDPOINT}/v1/models"
        return _Response(
            {
                "data": [
                    {"id": "owned-model", "model_path": "/secret/owned"},
                    {"id": "foreign-model", "model_path": "/secret/foreign"},
                ]
            }
        )

    monkeypatch.setattr(deployment_routes, "request_user_outbound", fake_request)

    result = deployment_routes.discover_models(
        endpoint=SHARED_ENDPOINT,
        framework="xinference",
        current_user=REGULAR_USER,
    )

    assert calls == [f"{SHARED_ENDPOINT}/v1/models"]
    assert [model.model_uid for model in result.models] == ["owned-model"]
    assert result.models[0].model_path is None


def test_regular_discovery_without_owned_bindings_does_not_query_endpoint(
    monkeypatch,
):
    monkeypatch.setattr(
        deployment_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_discovery_endpoint",
        lambda endpoint, user_id: endpoint.rstrip("/"),
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "list_deployments",
        lambda **kwargs: ([], 0),
    )
    monkeypatch.setattr(
        deployment_routes,
        "request_user_outbound",
        lambda *args, **kwargs: pytest.fail("unowned endpoint was queried"),
    )

    result = deployment_routes.discover_models(
        endpoint=SHARED_ENDPOINT,
        framework="xinference",
        current_user=REGULAR_USER,
    )

    assert result.models == []


@pytest.fixture
def owned_shared_catalog(empty_inference_catalog, monkeypatch):
    """Keep the real ownership query, including foreign models at both aliases."""
    from sqlmodel import Session
    from train_factory.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "auth_enabled", True)
    with Session(empty_inference_catalog) as session:
        for endpoint in (SHARED_ENDPOINT, "https://gateway.example.com/inference"):
            for user_id, model_uid in (("user-1", "owned-model"), ("user-2", "foreign-model")):
                session.add(DeploymentDB(
                    model_id=f"registry-{user_id}", user_id=user_id, model_uid=model_uid,
                    xinference_endpoint=endpoint, inference_framework="xinference",
                    deploy_mode="shared", status="running",
                ))
        session.commit()


def test_regular_config_validation_filters_shared_model_enumeration(monkeypatch, owned_shared_catalog):
    monkeypatch.setattr(
        model_config_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
        raising=False,
    )
    async def fake_validate(**kwargs):  # noqa: ARG001
        return {
            "valid": True,
            "models": ["owned-model", "foreign-model"],
            "message": "connected",
        }

    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "validate_config_async",
        fake_validate,
    )

    result = asyncio.run(
        model_config_routes.validate_config(
            model_config_routes.ValidateConfigRequest(
                provider="xinference",
                api_endpoint=SHARED_ENDPOINT,
                model_name="owned-model",
            ),
            REGULAR_USER,
        )
    )

    assert result.models == ["owned-model"]


@pytest.mark.parametrize("provider", ["xinference", "custom"])
def test_regular_config_validation_rejects_unowned_shared_model(
    monkeypatch,
    provider,
    owned_shared_catalog,
):
    calls = []
    monkeypatch.setattr(
        model_config_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
        raising=False,
    )
    async def fake_validate(**kwargs):
        calls.append(kwargs)
        return {"valid": True, "models": ["foreign-model"]}

    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "validate_config_async",
        fake_validate,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            model_config_routes.validate_config(
                model_config_routes.ValidateConfigRequest(
                    provider=provider,
                    api_endpoint=SHARED_ENDPOINT,
                    model_name="foreign-model",
                ),
                REGULAR_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert calls == []


def test_custom_provider_alias_cannot_bypass_owned_shared_endpoint_scope(
    monkeypatch,
    owned_shared_catalog,
):
    endpoint = "https://gateway.example.com/inference"
    calls = []
    monkeypatch.setattr(
        model_config_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
        raising=False,
    )
    async def fake_validate(**kwargs):
        calls.append(kwargs)
        return {"valid": True, "models": ["foreign-model"]}

    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "validate_config_async",
        fake_validate,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            model_config_routes.validate_config(
                model_config_routes.ValidateConfigRequest(
                    provider="custom",
                    api_endpoint=endpoint,
                    model_name="foreign-model",
                ),
                REGULAR_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert calls == []


def test_auth_disabled_config_validation_keeps_full_model_list(monkeypatch):
    monkeypatch.setattr(
        model_config_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=False),
        raising=False,
    )

    async def fake_validate(**kwargs):  # noqa: ARG001
        return {"valid": True, "models": ["first", "second"]}

    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "validate_config_async",
        fake_validate,
    )

    result = asyncio.run(
        model_config_routes.validate_config(
            model_config_routes.ValidateConfigRequest(
                provider="xinference",
                api_endpoint=SHARED_ENDPOINT,
            ),
            {"user_id": None, "username": "anonymous"},
        )
    )

    assert result.models == ["first", "second"]


def test_regular_check_all_filters_shared_model_lists_and_details(monkeypatch, owned_shared_catalog):
    monkeypatch.setattr(
        model_config_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
        raising=False,
    )
    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "list_configs",
        lambda **kwargs: (
            [
                {
                    "config_id": "config-1",
                    "config_name": "shared",
                    "model_type": "embedding",
                    "provider": "xinference",
                    "api_endpoint": SHARED_ENDPOINT,
                    "model_name": "owned-model",
                    "user_id": "user-1",
                }
            ],
            1,
        ),
    )

    async def fake_check(config_id):  # noqa: ARG001
        return {
            "success": True,
            "models": ["owned-model", "foreign-model"],
            "model_details": [
                {"id": "owned-model", "model_path": "/secret/owned"},
                {"id": "foreign-model", "model_path": "/secret/foreign"},
            ],
        }

    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "check_connectivity",
        fake_check,
    )

    result = asyncio.run(
        model_config_routes.check_all_connectivity(
            model_type=None,
            current_user=REGULAR_USER,
        )
    )

    config_result = result["results"]["config-1"]
    assert config_result["models"] == ["owned-model"]
    assert config_result["model_details"] == [{"id": "owned-model"}]


def test_scoped_model_listing_replaces_errors_that_enumerate_other_models():
    result = model_config_routes._filter_shared_model_listing(
        {
            "success": False,
            "models": ["foreign-model"],
            "error": "owned-model missing; available: foreign-model",
        },
        {"owned-model"},
    )

    assert result["models"] == []
    assert result["error"] == "Authorized model is unavailable"
