from __future__ import annotations

import asyncio
import inspect

import pytest
from fastapi import BackgroundTasks, HTTPException

from train_factory.api.routes import deployment_routes
from train_factory.api import server


USER = {"user_id": "user-1", "username": "alice"}


def _deployment(*, framework: str, status: str = "starting") -> dict:
    return {
        "deployment_id": "deployment-1",
        "model_id": "model-1",
        "model_uid": "served-model",
        "deployment_name": "async-deployment",
        "xinference_endpoint": "http://runtime:11000",
        "replica": 1,
        "gpu_memory_utilization": 0.8,
        "deploy_mode": "container",
        "container_name": "runtime",
        "gpu_id": 0,
        "port": 11000,
        "inference_framework": framework,
        "enable_lora": False,
        "max_loras": 4,
        "max_lora_rank": 64,
        "external_api_config_id": None,
        "config": None,
        "replica_instances": [],
        "status": status,
        "error_message": None,
        "user_id": USER["user_id"],
        "created_at": None,
        "updated_at": None,
        "started_at": None,
        "stopped_at": None,
    }


@pytest.fixture(autouse=True)
def route_dependencies(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        deployment_routes,
        "_verify_registered_model_ownership",
        lambda *_args, **_kwargs: {"model_id": "model-1"},
    )
    monkeypatch.setattr(
        deployment_routes,
        "check_idempotency",
        lambda *_args, **_kwargs: (False, None),
    )
    monkeypatch.setattr(
        deployment_routes,
        "store_idempotency_response",
        lambda *_args, **_kwargs: None,
    )


def test_primary_create_defers_vllm_start_to_managed_response_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    background = BackgroundTasks()

    def create_container_deployment(**kwargs):
        calls.append(kwargs)
        return _deployment(framework="vllm")

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "create_container_deployment",
        create_container_deployment,
    )

    response = asyncio.run(
        deployment_routes.create_deployment(
            request=deployment_routes.CreateDeploymentRequest(
                model_id="model-1",
                inference_framework="vllm",
            ),
            background_tasks=background,
            current_user=USER,
            idempotency_key="request-1",
        )
    )

    assert response.status == "starting"
    assert calls[0]["auto_start"] is True
    assert calls[0]["defer_start"] is True
    assert len(background.tasks) == 1
    assert (
        background.tasks[0].func
        is deployment_routes._run_registered_deferred_deployment_start
    )
    assert background.tasks[0].args == ("deployment-1",)


def test_explicit_container_create_defers_start_to_managed_response_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    background = BackgroundTasks()

    def create_container_deployment(**kwargs):
        calls.append(kwargs)
        return _deployment(framework="vllm")

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "create_container_deployment",
        create_container_deployment,
    )

    response = asyncio.run(
        deployment_routes.create_container_deployment(
            request=deployment_routes.CreateContainerDeploymentRequest(
                model_id="model-1",
                inference_framework="vllm",
            ),
            background_tasks=background,
            current_user=USER,
            idempotency_key="request-1",
        )
    )

    assert response.status == "starting"
    assert calls[0]["auto_start"] is True
    assert calls[0]["defer_start"] is True
    assert len(background.tasks) == 1
    assert (
        background.tasks[0].func
        is deployment_routes._run_registered_deferred_deployment_start
    )


def test_explicit_xinference_container_keeps_legacy_synchronous_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    background = BackgroundTasks()

    def create_container_deployment(**kwargs):
        calls.append(kwargs)
        return _deployment(framework="xinference", status="pending")

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "create_container_deployment",
        create_container_deployment,
    )

    response = asyncio.run(
        deployment_routes.create_container_deployment(
            request=deployment_routes.CreateContainerDeploymentRequest(
                model_id="model-1",
                inference_framework="xinference",
            ),
            background_tasks=background,
            current_user=USER,
            idempotency_key=None,
        )
    )

    assert response.status == "pending"
    assert calls[0]["auto_start"] is True
    assert calls[0]["defer_start"] is False
    assert background.tasks == []


def test_auto_start_false_does_not_schedule_background_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    background = BackgroundTasks()

    def create_container_deployment(**kwargs):
        calls.append(kwargs)
        return _deployment(framework="vllm", status="stopped")

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "create_container_deployment",
        create_container_deployment,
    )

    response = asyncio.run(
        deployment_routes.create_deployment(
            request=deployment_routes.CreateDeploymentRequest(
                model_id="model-1",
                inference_framework="vllm",
                auto_start=False,
            ),
            background_tasks=background,
            current_user=USER,
            idempotency_key=None,
        )
    )

    assert response.status == "stopped"
    assert calls[0]["defer_start"] is False
    assert background.tasks == []


@pytest.mark.parametrize("route_name", ["primary", "container"])
def test_create_routes_map_planning_busy_to_conflict(
    monkeypatch: pytest.MonkeyPatch,
    route_name: str,
) -> None:
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "create_container_deployment",
        lambda **_kwargs: (_ for _ in ()).throw(
            deployment_routes.ReplicaOperationBusyError(
                "registry model changed while planning deployment; retry"
            )
        ),
    )
    if route_name == "primary":
        def invoke():
            return deployment_routes.create_deployment(
                request=deployment_routes.CreateDeploymentRequest(
                    model_id="model-1",
                    inference_framework="vllm",
                ),
                background_tasks=BackgroundTasks(),
                current_user=USER,
                idempotency_key=None,
            )

    else:
        def invoke():
            return deployment_routes.create_container_deployment(
                request=deployment_routes.CreateContainerDeploymentRequest(
                    model_id="model-1",
                    inference_framework="vllm",
                ),
                background_tasks=BackgroundTasks(),
                current_user=USER,
                idempotency_key=None,
            )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(invoke())

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Deployment replica operation in progress"


def test_idempotency_cache_hit_reschedules_durable_deferred_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = _deployment(framework="vllm")
    cached["config"] = {"_deferred_auto_start": True}
    scheduled: list[str] = []
    monkeypatch.setattr(
        deployment_routes,
        "check_idempotency",
        lambda *_args, **_kwargs: (True, cached),
    )
    monkeypatch.setattr(
        deployment_routes,
        "_ensure_deferred_deployment_start",
        lambda deployment_id: scheduled.append(deployment_id),
    )

    response = asyncio.run(
        deployment_routes.create_deployment(
            request=deployment_routes.CreateDeploymentRequest(
                model_id="model-1",
                inference_framework="vllm",
            ),
            background_tasks=BackgroundTasks(),
            current_user=USER,
            idempotency_key="request-1",
        )
    )

    assert response.deployment_id == "deployment-1"
    assert scheduled == ["deployment-1"]


def test_startup_recovery_registers_and_drains_durable_deferred_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[str] = []

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "list_deferred_start_deployment_ids",
        lambda: ["deployment-1", "deployment-2"],
    )

    async def start(deployment_id: str) -> None:
        await asyncio.sleep(0)
        started.append(deployment_id)

    monkeypatch.setattr(
        deployment_routes,
        "_start_deferred_deployment_in_background",
        start,
    )

    async def scenario() -> None:
        await deployment_routes._resume_deferred_deployment_starts()
        await deployment_routes._drain_deferred_deployment_starts()

    asyncio.run(scenario())

    assert sorted(started) == ["deployment-1", "deployment-2"]
    assert deployment_routes._deferred_start_tasks == {}


def test_deferred_worker_recovers_from_transient_preflight_failure_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_attempts = 0
    started: list[str] = []

    def get_deployment(deployment_id: str) -> dict:
        nonlocal preflight_attempts
        preflight_attempts += 1
        if preflight_attempts == 1:
            raise RuntimeError("temporary database read failure")
        return {
            "deployment_id": deployment_id,
            "xinference_endpoint": "http://runtime:11000",
            "user_id": "user-1",
        }

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        get_deployment,
    )
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "start_deferred_deployment",
        lambda deployment_id, **_kwargs: started.append(deployment_id),
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_deployment_endpoint",
        lambda deployment: deployment,
    )
    monkeypatch.setattr(
        deployment_routes,
        "_DEFERRED_START_RETRY_SECONDS",
        0.001,
    )
    monkeypatch.setattr(deployment_routes, "_deferred_start_shutdown", False)

    async def scenario() -> None:
        first = deployment_routes._ensure_deferred_deployment_start("deployment-1")
        second = deployment_routes._ensure_deferred_deployment_start("deployment-1")
        assert second is first
        await asyncio.wait_for(first, timeout=0.2)
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert preflight_attempts == 2
    assert started == ["deployment-1"]
    assert deployment_routes._deferred_start_tasks == {}


def test_server_lifespan_resumes_and_drains_deferred_deployments() -> None:
    source = inspect.getsource(server.lifespan)

    assert "await _resume_deferred_deployment_starts()" in source
    assert "await _drain_deferred_deployment_starts()" in source


def test_deferred_worker_bounds_busy_retry_bursts_with_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    sleeps: list[float] = []
    deployment = {
        "deployment_id": "deployment-1",
        "xinference_endpoint": "http://runtime:11000",
        "user_id": "user-1",
        "config": {"_deferred_auto_start": True},
    }
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: deployment,
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_deployment_endpoint",
        lambda value: value,
    )

    def busy(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise deployment_routes.ReplicaOperationBusyError(
            "runtime recovery is still inspecting identities"
        )

    async def bounded_sleep(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) == 3:
            deployment_routes._deferred_start_shutdown = True

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "start_deferred_deployment",
        busy,
    )
    monkeypatch.setattr(deployment_routes.asyncio, "sleep", bounded_sleep)
    monkeypatch.setattr(deployment_routes, "_DEFERRED_START_RETRY_SECONDS", 0.25)
    monkeypatch.setattr(
        deployment_routes,
        "_DEFERRED_START_LEASE_RETRY_SECONDS",
        0.75,
    )
    monkeypatch.setattr(deployment_routes, "_deferred_start_shutdown", False)
    caplog.set_level("ERROR", logger=deployment_routes.__name__)

    asyncio.run(
        deployment_routes._start_deferred_deployment_in_background("deployment-1")
    )

    assert attempts == 3
    assert sleeps == [0.25, 0.5, 0.75]
    assert deployment["config"]["_deferred_auto_start"] is True
    assert "exhausted 3 busy attempts" in caplog.text
    assert "durable auto-start intent retained" in caplog.text


def test_deferred_worker_retries_after_busy_lease_without_second_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-1",
            "xinference_endpoint": "http://runtime:11000",
            "user_id": "user-1",
            "config": {"_deferred_auto_start": True},
        },
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_deployment_endpoint",
        lambda deployment: deployment,
    )

    def busy_until_lease_expires(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise deployment_routes.ReplicaOperationBusyError("fresh orphan claim")

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "start_deferred_deployment",
        busy_until_lease_expires,
    )
    monkeypatch.setattr(deployment_routes.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(deployment_routes, "_DEFERRED_START_RETRY_SECONDS", 0.25)
    monkeypatch.setattr(
        deployment_routes,
        "_DEFERRED_START_LEASE_RETRY_SECONDS",
        0.75,
        raising=False,
    )
    monkeypatch.setattr(deployment_routes, "_deferred_start_shutdown", False)

    asyncio.run(
        deployment_routes._start_deferred_deployment_in_background("deployment-1")
    )

    assert attempts == 4
    assert sleeps == [0.25, 0.5, 0.75]


def test_shutdown_drain_stops_a_worker_that_only_waits_on_foreign_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-1",
            "xinference_endpoint": "http://runtime:11000",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        deployment_routes,
        "_validate_deployment_endpoint",
        lambda deployment: deployment,
    )
    monkeypatch.setattr(
        deployment_routes,
        "_DEFERRED_START_RETRY_SECONDS",
        0.001,
    )

    def busy(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise deployment_routes.ReplicaOperationBusyError("busy")

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "start_deferred_deployment",
        busy,
    )

    async def scenario() -> None:
        deployment_routes._ensure_deferred_deployment_start("deployment-1")
        while attempts == 0:
            await asyncio.sleep(0)
        await asyncio.wait_for(
            deployment_routes._drain_deferred_deployment_starts(),
            timeout=0.2,
        )

    asyncio.run(scenario())

    assert attempts >= 1
    assert deployment_routes._deferred_start_tasks == {}
