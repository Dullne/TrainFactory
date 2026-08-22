import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from train_factory.api.routes import deployment_routes
from train_factory.auth.dependencies import verify_resource_ownership
from train_factory.storage.services.model_registry_service import (
    model_registry_service,
)


def _invoke_create_route(route_name: str, background_tasks: BackgroundTasks):
    current_user = {"user_id": "user-1", "username": "alice"}

    if route_name == "shared":
        request = deployment_routes.CreateDeploymentRequest(model_id="model-1")
        return asyncio.run(
            deployment_routes.create_deployment(
                request=request,
                background_tasks=background_tasks,
                current_user=current_user,
                idempotency_key="request-1",
            )
        )
    if route_name == "quick":
        request = deployment_routes.QuickDeployRequest(
            xinference_endpoint="http://xinference:9997",
        )
        return asyncio.run(
            deployment_routes.quick_deploy(
                model_id="model-1",
                request=request,
                current_user=current_user,
            )
        )

    request = deployment_routes.CreateContainerDeploymentRequest(model_id="model-1")
    return asyncio.run(
        deployment_routes.create_container_deployment(
            request=request,
            current_user=current_user,
            idempotency_key="request-1",
        )
    )


@pytest.mark.parametrize(
    "resource",
    [
        {"model_id": "model-1"},
        {"model_id": "model-1", "user_id": None},
        {"model_id": "model-1", "user_id": "user-2"},
    ],
)
def test_authenticated_user_cannot_access_unowned_resource(resource):
    with pytest.raises(HTTPException) as exc_info:
        verify_resource_ownership(
            resource,
            {"user_id": "user-1", "username": "alice"},
            "Model",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize(
    "current_user",
    [
        {"user_id": "user-1", "username": "alice", "is_admin": False},
        {"user_id": "admin-1", "username": "admin", "is_admin": True},
    ],
)
def test_deployment_rejects_owned_model_without_managed_provenance(
    monkeypatch,
    tmp_path,
    current_user,
):
    model_path = tmp_path / "arbitrary" / "model"
    monkeypatch.setattr(
        deployment_routes,
        "get_settings",
        lambda: SimpleNamespace(
            auth_enabled=True,
            models_dir=tmp_path / "models",
            output_dir=tmp_path / "outputs",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        deployment_routes,
        "requires_tenant_provenance",
        lambda user: True,
        raising=False,
    )
    monkeypatch.setattr(
        model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_path": str(model_path),
            "source_type": "trained",
            "status": "available",
            "user_id": current_user["user_id"],
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        deployment_routes._verify_registered_model_ownership(
            "model-1",
            current_user,
        )

    assert exc_info.value.status_code == 403


def test_deployment_accepts_owned_api_downloaded_model(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    model_path = models_dir / "model-1"
    model_path.mkdir(parents=True)
    current_user = {"user_id": "user-1", "username": "alice", "is_admin": False}
    monkeypatch.setattr(
        deployment_routes,
        "get_settings",
        lambda: SimpleNamespace(
            auth_enabled=True,
            models_dir=models_dir,
            output_dir=tmp_path / "outputs",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        deployment_routes,
        "requires_tenant_provenance",
        lambda user: True,
        raising=False,
    )
    monkeypatch.setattr(
        model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_path": str(model_path),
            "source_type": "downloaded",
            "status": "available",
            "user_id": current_user["user_id"],
        },
    )

    model = deployment_routes._verify_registered_model_ownership(
        "model-1",
        current_user,
    )

    assert model["model_path"] == str(model_path)


@pytest.mark.parametrize("inside_managed_root", [True, False])
def test_deployment_enforces_route_local_training_output_root(
    monkeypatch,
    tmp_path,
    inside_managed_root,
):
    output_dir = tmp_path / "route-local-output"
    model_path = (
        output_dir / "task-1" / "model"
        if inside_managed_root
        else tmp_path / "outside-output" / "model"
    )
    current_user = {"user_id": "user-1", "username": "alice", "is_admin": False}
    monkeypatch.setattr(
        deployment_routes,
        "get_settings",
        lambda: SimpleNamespace(
            auth_enabled=True,
            models_dir=tmp_path / "models",
            output_dir=output_dir,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        deployment_routes,
        "requires_tenant_provenance",
        lambda user: True,
        raising=False,
    )
    monkeypatch.setattr(
        model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_path": str(model_path),
            "source_type": "trained",
            "source_task_id": "task-1",
            "status": "available",
            "user_id": current_user["user_id"],
        },
    )
    monkeypatch.setattr(
        deployment_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "final_model_path": str(model_path),
            "status": "succeeded",
            "user_id": current_user["user_id"],
        },
    )

    if inside_managed_root:
        model = deployment_routes._verify_registered_model_ownership(
            "model-1",
            current_user,
        )
        assert model["model_path"] == str(model_path)
    else:
        with pytest.raises(HTTPException) as exc_info:
            deployment_routes._verify_registered_model_ownership(
                "model-1",
                current_user,
            )
        assert exc_info.value.status_code == 403


def test_deployment_keeps_ownerless_direct_models_when_auth_is_disabled(monkeypatch):
    monkeypatch.setattr(
        deployment_routes,
        "requires_tenant_provenance",
        lambda user: False,
    )
    monkeypatch.setattr(
        model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_path": "/app/models/legacy",
            "source_type": "local",
            "status": "available",
            "user_id": None,
        },
    )

    model = deployment_routes._verify_registered_model_ownership(
        "model-1",
        {"user_id": None, "username": "anonymous"},
    )

    assert model["model_path"] == "/app/models/legacy"


@pytest.mark.parametrize("anonymous_user_id", [None, "anonymous"])
def test_auth_disabled_identity_can_access_ownerless_resource(anonymous_user_id):
    resource = {"model_id": "model-1", "user_id": None}

    assert (
        verify_resource_ownership(
            resource,
            {"user_id": anonymous_user_id, "username": "anonymous"},
            "Model",
        )
        is resource
    )


@pytest.mark.parametrize("route_name", ["shared", "quick", "container"])
@pytest.mark.parametrize("model_user_id", [None, "user-2"])
def test_deployment_create_routes_reject_unowned_models_before_side_effects(
    monkeypatch,
    route_name,
    model_user_id,
):
    background_tasks = BackgroundTasks()
    model_reads = []

    def get_model(model_id):
        model_reads.append(model_id)
        return {
            "model_id": model_id,
            "model_name": "Unowned model",
            "user_id": model_user_id,
        }

    def unexpected_side_effect(*args, **kwargs):
        pytest.fail("deployment side effect ran before model ownership verification")

    monkeypatch.setattr(model_registry_service, "get_model", get_model)
    monkeypatch.setattr(deployment_routes, "check_idempotency", unexpected_side_effect)
    monkeypatch.setattr(deployment_routes, "store_idempotency_response", unexpected_side_effect)
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "create_deployment",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "create_container_deployment",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "deploy_from_model",
        unexpected_side_effect,
    )

    with pytest.raises(HTTPException) as exc_info:
        _invoke_create_route(route_name, background_tasks)

    assert exc_info.value.status_code == 403
    assert model_reads == ["model-1"]
    assert background_tasks.tasks == []
