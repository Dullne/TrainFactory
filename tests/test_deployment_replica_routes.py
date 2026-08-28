from __future__ import annotations

import asyncio
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from train_factory.api.routes import deployment_routes


DEPLOYMENT_ID = "11111111-1111-4111-8111-111111111111"
REPLICA_ID = "22222222-2222-4222-8222-222222222222"


def _child() -> dict[str, object]:
    return {
        "replica_id": REPLICA_ID,
        "deployment_id": DEPLOYMENT_ID,
        "replica_index": 0,
        "container_name": "private-container-name",
        "endpoint": "http://127.0.0.1:11000",
        "port": 11000,
        "gpu_ids": [0, 1],
        "status": "running",
        "health_status": "HEALTHY",
        "error_message": None,
        "created_at": None,
        "updated_at": None,
        "started_at": None,
        "stopped_at": None,
    }


def test_replica_routes_are_registered_before_parent_parameter_routes() -> None:
    paths = [route.path for route in deployment_routes.router.routes]

    replica_index = paths.index("/deployments/{deployment_id}/replicas")
    parent_index = paths.index("/deployments/{deployment_id}")
    assert replica_index < parent_index
    for action in ("start", "stop", "restart", "recreate"):
        assert (
            f"/deployments/{{deployment_id}}/replicas/{{replica_id}}/{action}"
            in paths
        )


def test_list_route_passes_authenticated_owner_and_hides_docker_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "list_replicas",
        lambda deployment_id, user_id: calls.append((deployment_id, user_id))
        or [_child()],
    )

    response = asyncio.run(
        deployment_routes.list_deployment_replicas(
            UUID(DEPLOYMENT_ID),
            {"user_id": "user-1"},
        )
    )

    assert calls == [(DEPLOYMENT_ID, "user-1")]
    assert response[0].replica_id == REPLICA_ID
    assert not hasattr(response[0], "container_name")
    assert "/app/models" not in response[0].model_dump_json()


@pytest.mark.parametrize("action", ["start", "stop", "restart", "recreate"])
def test_action_routes_pass_exact_parent_child_and_owner(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        f"{action}_replica",
        lambda deployment_id, replica_id, user_id: calls.append(
            (deployment_id, replica_id, user_id)
        )
        or _child(),
    )

    response = asyncio.run(
        getattr(deployment_routes, f"{action}_deployment_replica")(
            UUID(DEPLOYMENT_ID),
            UUID(REPLICA_ID),
            {"user_id": "user-1"},
        )
    )

    assert calls == [(DEPLOYMENT_ID, REPLICA_ID, "user-1")]
    assert response.replica_id == REPLICA_ID


@pytest.mark.parametrize("message", ["deployment access denied", "not found"])
def test_wrong_parent_child_and_foreign_owner_are_indistinguishable(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "start_replica",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_routes.DeploymentReplicaNotFoundError(message)
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deployment_routes.start_deployment_replica(
                UUID(DEPLOYMENT_ID),
                UUID(REPLICA_ID),
                {"user_id": "user-2"},
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Deployment replica not found"


def test_replica_route_error_distinguishes_not_found_state_and_validation() -> None:
    not_found = deployment_routes._replica_route_error(
        deployment_routes.DeploymentReplicaNotFoundError("deployment access denied")
    )
    invalid_state = deployment_routes._replica_route_error(
        deployment_routes.DeploymentReplicaStateConflictError(
            "deployment replica lifecycle is unavailable"
        )
    )
    invalid_request = deployment_routes._replica_route_error(
        ValueError("invalid deployment replica operation")
    )

    assert (not_found.status_code, not_found.detail) == (
        404,
        "Deployment replica not found",
    )
    assert invalid_state.status_code == 409
    assert invalid_request.status_code == 400


def test_busy_replica_operation_returns_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "start_replica",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_routes.ReplicaOperationBusyError(
                "deployment replica operation already in progress"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deployment_routes.start_deployment_replica(
                UUID(DEPLOYMENT_ID),
                UUID(REPLICA_ID),
                {"user_id": "user-1"},
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Deployment replica operation in progress"


@pytest.mark.parametrize("action", ["start", "stop", "restart", "delete"])
def test_busy_group_operation_returns_conflict(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {"user_id": "user-1"},
    )
    monkeypatch.setattr(
        deployment_routes,
        "verify_resource_ownership",
        lambda deployment, _user, _kind: deployment,
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_deployment_endpoint",
        lambda _deployment: None,
    )
    method_name = f"{action}_deployment"
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        method_name,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_routes.ReplicaOperationBusyError(
                "deployment replica operation already in progress"
            )
        ),
    )

    route = getattr(deployment_routes, method_name)
    args = [DEPLOYMENT_ID]
    kwargs = {"current_user": {"user_id": "user-1"}}
    if action == "restart":
        kwargs["request"] = None
    if action == "delete":
        kwargs["force"] = False

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(route(*args, **kwargs))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Deployment replica operation in progress"


def test_busy_status_sync_returns_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {"user_id": "user-1"},
    )
    monkeypatch.setattr(
        deployment_routes,
        "verify_resource_ownership",
        lambda deployment, _user, _kind: deployment,
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_deployment_endpoint",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "sync_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_routes.ReplicaOperationBusyError(
                "deployment replica operation already in progress"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deployment_routes.sync_deployment_status(
                DEPLOYMENT_ID,
                {"user_id": "user-1"},
            )
        )

    assert exc_info.value.status_code == 409


def test_missing_during_status_sync_returns_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {"user_id": "user-1"},
    )
    monkeypatch.setattr(
        deployment_routes,
        "verify_resource_ownership",
        lambda deployment, _user, _kind: deployment,
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_deployment_endpoint",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "sync_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_routes.DeploymentReplicaNotFoundError(
                "deployment not found"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deployment_routes.sync_deployment_status(
                DEPLOYMENT_ID,
                {"user_id": "user-1"},
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "deployment not found"


def test_busy_config_update_returns_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "user_id": "user-1",
            "external_api_config_id": None,
        },
    )
    monkeypatch.setattr(
        deployment_routes,
        "verify_resource_ownership",
        lambda deployment, _user, _kind: deployment,
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "update_deployment_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_routes.ReplicaOperationBusyError(
                "deployment config changed while acquiring locks"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deployment_routes.update_deployment_config(
                DEPLOYMENT_ID,
                deployment_routes.UpdateDeploymentConfigRequest(config={}),
                {"user_id": "user-1"},
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Deployment replica operation in progress"


def test_malformed_uuid_is_rejected_before_service_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "list_replicas",
        lambda *_args, **_kwargs: calls.append("called"),
    )
    app = FastAPI()
    app.include_router(deployment_routes.router)
    app.dependency_overrides[deployment_routes.get_current_user] = lambda: {
        "user_id": "user-1"
    }

    response = TestClient(app).get("/deployments/not-a-uuid/replicas")

    assert response.status_code == 422
    assert calls == []


def test_parent_response_exposes_typed_launch_config() -> None:
    payload = {
        "deployment_id": DEPLOYMENT_ID,
        "model_id": "model-1",
        "model_uid": "served",
        "deployment_name": "group",
        "xinference_endpoint": "http://127.0.0.1:11000",
        "replica": 1,
        "gpu_memory_utilization": 0.8,
        "deploy_mode": "container",
        "container_name": "group",
        "gpu_id": 0,
        "port": 11000,
        "inference_framework": "vllm",
        "enable_lora": False,
        "max_loras": 4,
        "max_lora_rank": 64,
        "config": {"launch_config": {"framework": "vllm"}},
        "status": "running",
        "error_message": None,
        "user_id": "user-1",
        "created_at": None,
        "updated_at": None,
        "started_at": None,
        "stopped_at": None,
        "replica_instances": [_child()],
    }

    response = deployment_routes._deployment_to_response(payload)

    assert response.launch_config is not None
    assert response.launch_config.framework == "vllm"
    assert response.replica_instances[0].replica_id == REPLICA_ID


@pytest.mark.parametrize("field", ["replica", "gpu_id", "port", "max_loras"])
def test_boolean_numeric_create_fields_are_rejected(field: str) -> None:
    payload = {
        "model_id": "model-1",
        "inference_framework": "vllm",
        "launch_config": {"framework": "vllm"},
        field: True,
    }

    with pytest.raises(ValidationError):
        deployment_routes.CreateContainerDeploymentRequest.model_validate(payload)


def test_extra_create_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        deployment_routes.CreateContainerDeploymentRequest.model_validate(
            {
                "model_id": "model-1",
                "inference_framework": "vllm",
                "launch_config": {"framework": "vllm"},
                "raw_server_args": ["--host", "attacker"],
            }
        )
