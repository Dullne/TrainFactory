import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def external_inference_catalog(monkeypatch):
    from train_factory.storage.services import inference_authorization_service

    monkeypatch.setattr(
        inference_authorization_service, "registered_shared_models_for_endpoint",
        lambda *_args: (False, set()),
    )
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from train_factory.api.routes import external_api_config_routes, sync_routes
from train_factory.core.ssrf import SSRFError
from train_factory.deployment.deployment_service import deployment_service
from train_factory.storage.services.external_api_config_service import (
    external_api_config_service,
)
from train_factory.storage.services.external_sync_service import external_sync_service
from train_factory.storage.services.model_config_service import model_config_service
from train_factory.sync import sync_manager as sync_manager_module
from train_factory.sync import level2_handler, post_training_handler
from train_factory.sync import sync_worker


CURRENT_USER = {"user_id": "user-1", "username": "owner"}


@pytest.mark.parametrize(
    ("auth_enabled", "current_user", "resolved_user_id"),
    (
        (True, CURRENT_USER, "user-1"),
        (
            True,
            {"user_id": "admin-user", "username": "admin", "is_admin": True},
            "admin-user",
        ),
        (False, {"user_id": "disabled-auth-user", "username": "anonymous"}, None),
    ),
)
def test_list_tasks_pagination_forwards_resolved_tenant_and_page(
    monkeypatch,
    auth_enabled,
    current_user,
    resolved_user_id,
):
    service_calls = []

    def list_tasks(**kwargs):
        service_calls.append(kwargs)
        return [{"task_id": "sync-51"}], 51

    monkeypatch.setattr(
        sync_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=auth_enabled),
    )
    monkeypatch.setattr(external_sync_service, "list_tasks", list_tasks)

    response = asyncio.run(
        sync_routes.list_sync_tasks(
            external_api_config_id="external-api-1",
            limit=50,
            offset=50,
            current_user=current_user,
        )
    )

    assert service_calls == [
        {
            "user_id": resolved_user_id,
            "external_api_config_id": "external-api-1",
            "limit": 50,
            "offset": 50,
        }
    ]
    assert response == {"tasks": [{"task_id": "sync-51"}], "total": 51}


@pytest.mark.parametrize("query", ("limit=0", "limit=201", "offset=-1"))
def test_list_tasks_pagination_rejects_invalid_query_before_service(
    monkeypatch,
    query,
):
    service_calls = []
    monkeypatch.setattr(
        external_sync_service,
        "list_tasks",
        lambda **kwargs: service_calls.append(kwargs) or ([], 0),
    )
    app = FastAPI()
    app.include_router(sync_routes.router, prefix="/api/sync")
    app.dependency_overrides[sync_routes.get_current_user] = lambda: CURRENT_USER

    with TestClient(app) as client:
        response = client.get(f"/api/sync/tasks?{query}")

    assert response.status_code == 422
    assert service_calls == []


@pytest.mark.parametrize("status", ["deleting", "deleting_cascade"])
def test_deleting_sync_task_rejects_mutations(status):
    with pytest.raises(HTTPException) as exc_info:
        sync_routes._require_sync_task_mutable({"status": status})

    assert exc_info.value.status_code == 409


def test_update_sync_task_rechecks_deleting_parent_inside_operation_lock(
    monkeypatch,
):
    snapshots = iter(
        (
            {"task_id": "sync-update-fence", "user_id": "user-1", "status": "idle"},
            {
                "task_id": "sync-update-fence",
                "user_id": "user-1",
                "status": "deleting",
            },
        )
    )
    events = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: next(snapshots),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda *_args, **_kwargs: pytest.fail(
            "locked deleting parent must not be updated"
        ),
    )

    @asynccontextmanager
    async def operation_lock(task_id):
        events.append(("lock", task_id))
        yield

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.update_sync_task(
                "sync-update-fence",
                sync_routes.SyncTaskUpdateRequest(sync_interval_seconds=600),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [("lock", "sync-update-fence")]


def test_target_create_rechecks_deleting_parent_inside_operation_lock(monkeypatch):
    snapshots = iter(
        (
            {
                "task_id": "sync-target-fence",
                "user_id": "user-1",
                "status": "idle",
            },
            {
                "task_id": "sync-target-fence",
                "user_id": "user-1",
                "status": "deleting_cascade",
            },
        )
    )
    events = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: next(snapshots),
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_training_target",
        lambda **_kwargs: pytest.fail(
            "locked deleting parent must not gain a training target"
        ),
    )

    @asynccontextmanager
    async def operation_lock(task_id):
        events.append(("lock", task_id))
        yield

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.create_training_target(
                "sync-target-fence",
                sync_routes.TrainingTargetCreateRequest(target_name="Blocked"),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [("lock", "sync-target-fence")]


@pytest.mark.parametrize("operation", ["start", "stop"])
def test_sync_worker_control_revalidates_owner_and_forwards_it_to_update(
    monkeypatch,
    operation,
):
    events = []
    task_id = f"sync-{operation}-owner"
    task = {
        "task_id": task_id,
        "user_id": "user-1",
        "status": "idle",
        "is_active": operation == "stop",
    }

    def get_task(_task_id):
        events.append("get")
        return dict(task)

    def update_task(_task_id, **kwargs):
        events.append(("update", kwargs))
        return {**task, **kwargs}

    @asynccontextmanager
    async def operation_lock(_task_id):
        events.append("lock-enter")
        yield
        events.append("lock-exit")

    async def stop_worker(_task_id):
        events.append("stop-worker")

    monkeypatch.setattr(external_sync_service, "get_task", get_task)
    monkeypatch.setattr(external_sync_service, "update_task", update_task)
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "start_worker",
        lambda _task_id: events.append("start-worker"),
    )
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "stop_worker",
        stop_worker,
    )
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "get_worker_status",
        lambda _task_id: "running" if operation == "start" else "stopped",
    )

    if operation == "start":
        asyncio.run(sync_routes.start_sync(task_id, CURRENT_USER))
        assert events == [
            "get",
            "lock-enter",
            "get",
            ("update", {"expected_user_id": "user-1", "is_active": True}),
            "lock-exit",
            "start-worker",
        ]
    else:
        asyncio.run(sync_routes.stop_sync(task_id, CURRENT_USER))
        assert events == [
            "get",
            "lock-enter",
            "get",
            (
                "update",
                {
                    "expected_user_id": "user-1",
                    "is_active": False,
                    "status": "idle",
                },
            ),
            "stop-worker",
            "lock-exit",
        ]


def test_stop_sync_rejects_owner_drift_before_worker_side_effect(monkeypatch):
    snapshots = iter(
        (
            {
                "task_id": "sync-stop-drift",
                "user_id": "user-1",
                "status": "idle",
                "is_active": True,
            },
            {
                "task_id": "sync-stop-drift",
                "user_id": "user-2",
                "status": "idle",
                "is_active": True,
            },
        )
    )
    side_effects = []

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: next(snapshots),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda *_args, **_kwargs: side_effects.append("update"),
    )
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )

    async def stop_worker(_task_id):
        side_effects.append("stop-worker")

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "stop_worker",
        stop_worker,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.stop_sync(
                "sync-stop-drift",
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert side_effects == []


def test_authenticated_sync_config_discards_user_controlled_milvus_target(monkeypatch):
    monkeypatch.setattr(
        sync_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
    )

    validated = sync_routes._validate_generation_config(
        {
            "milvus_config": {
                "host": "127.0.0.1",
                "port": 2379,
                "token": "attacker-token",
            },
            "worker_config": {"concurrency": 2},
        },
        CURRENT_USER,
    )

    assert "milvus_config" not in validated
    assert validated["worker_config"] == {"concurrency": 2}


def test_auth_disabled_sync_config_keeps_legacy_milvus_target(monkeypatch):
    monkeypatch.setattr(
        sync_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=False),
    )
    legacy = {"host": "milvus-dev", "port": 19531, "token": "dev-token"}

    validated = sync_routes._validate_generation_config(
        {"milvus_config": legacy},
        {"user_id": None, "username": "anonymous"},
    )

    assert validated["milvus_config"] == legacy


def test_sync_worker_ignores_persisted_milvus_target_when_auth_is_enabled(
    monkeypatch,
):
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
        raising=False,
    )

    resolved = sync_worker._resolve_sync_milvus_connection_config(
        {
            "milvus_config": {
                "host": "169.254.169.254",
                "port": 443,
                "token": "persisted-secret",
            }
        }
    )

    assert resolved == {}


def test_create_sync_task_rejects_loopback_url_before_persist(monkeypatch):
    monkeypatch.setattr(
        external_sync_service,
        "create_task",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("blocked URL must not be persisted")
        ),
    )

    with pytest.raises(SSRFError):
        asyncio.run(
            sync_routes.create_sync_task(
                sync_routes.SyncTaskCreateRequest(
                    task_name="blocked",
                    external_api_url="http://127.0.0.1:8000/private",
                    is_active=False,
                ),
                CURRENT_USER,
            )
        )


def test_update_external_api_config_rejects_metadata_url_before_persist(monkeypatch):
    monkeypatch.setattr(
        external_api_config_service,
        "get_config",
        lambda _config_id: {"config_id": "api-1", "user_id": "user-1"},
    )
    monkeypatch.setattr(
        external_api_config_service,
        "update_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("blocked URL must not be persisted")
        ),
    )

    with pytest.raises(SSRFError):
        asyncio.run(
            external_api_config_routes.update_api_config(
                "api-1",
                external_api_config_routes.UpdateExternalApiConfigRequest(
                    api_url="http://169.254.169.254/latest/meta-data"
                ),
                CURRENT_USER,
            )
        )


def test_sync_worker_revalidates_persisted_url_before_any_cycle_side_effect(monkeypatch):
    async def fail_if_promoted(_config):
        raise AssertionError("runtime URL validation must happen before batch mutation")

    monkeypatch.setattr(sync_worker, "_promote_registered_batches", fail_if_promoted)

    with pytest.raises(SSRFError):
        asyncio.run(
            sync_worker.run_once(
                {
                    "task_id": "sync-ssrf",
                    "external_api_url": "http://localhost:8000/private",
                    "external_auth_config": {},
                }
            )
        )


def test_create_sync_task_accepts_owned_generation_model_config_and_forwards_it(
    monkeypatch,
):
    service_calls = []
    generation_config = {
        "eval_llm_config": {"config_id": "model-config-owned"}
    }
    monkeypatch.setattr(
        model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "model-config-owned",
            "user_id": "user-1",
            "api_endpoint": "https://models.example.com/v1",
        },
    )
    monkeypatch.setattr(
        sync_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_task",
        lambda **kwargs: service_calls.append(kwargs)
        or {"task_id": "sync-owned-create", "is_active": False},
    )

    response = asyncio.run(
        sync_routes.create_sync_task(
            sync_routes.SyncTaskCreateRequest(
                task_name="owned-generation-config",
                external_api_url="https://sync.example.com/data",
                generation_config=generation_config,
                is_active=False,
            ),
            CURRENT_USER,
        )
    )

    assert response["task"]["task_id"] == "sync-owned-create"
    assert service_calls[0]["user_id"] == "user-1"
    assert service_calls[0]["generation_config"] == generation_config


def test_update_sync_task_accepts_owned_generation_model_config_and_forwards_it(
    monkeypatch,
):
    service_calls = []
    generation_config = {
        "rerank_config": {"config_id": "model-config-owned"}
    }
    persisted_task = {
        "task_id": "sync-owned-update",
        "user_id": "user-1",
        "status": "idle",
        "is_active": False,
        "external_api_config_id": None,
        "base_deployment_id": None,
        "base_deployment_replica_id": None,
    }

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    monkeypatch.setattr(
        model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "model-config-owned",
            "user_id": "user-1",
            "api_endpoint": "https://models.example.com/v1",
        },
    )
    monkeypatch.setattr(
        sync_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(persisted_task),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda task_id, **kwargs: service_calls.append((task_id, kwargs))
        or {**persisted_task, **kwargs},
    )
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )
    monkeypatch.setattr(sync_manager_module.sync_manager, "_running", False)

    response = asyncio.run(
        sync_routes.update_sync_task(
            "sync-owned-update",
            sync_routes.SyncTaskUpdateRequest(
                generation_config=generation_config,
            ),
            CURRENT_USER,
        )
    )

    assert response["task"]["generation_config"] == generation_config
    assert service_calls == [
        (
            "sync-owned-update",
            {
                "expected_user_id": "user-1",
                "generation_config": generation_config,
            },
        )
    ]


@pytest.mark.parametrize(
    "config_key",
    ["llm_config", "eval_llm_config", "embedding_config", "rerank_config"],
)
def test_create_sync_task_rejects_foreign_generation_model_config(
    monkeypatch,
    config_key,
):
    monkeypatch.setattr(
        model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "model-config-foreign",
            "user_id": "user-2",
            "api_endpoint": "https://models.example.com/v1",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_task",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("foreign model config must not be persisted")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.create_sync_task(
                sync_routes.SyncTaskCreateRequest(
                    task_name="foreign-generation-config",
                    external_api_url="https://8.8.8.8/data",
                    generation_config={
                        config_key: {"config_id": "model-config-foreign"},
                    },
                    is_active=False,
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Not authorized to use this model config"


@pytest.mark.parametrize(
    "config_key",
    ["llm_config", "eval_llm_config", "embedding_config", "rerank_config"],
)
@pytest.mark.parametrize("owner_id", [None, "user-2"])
def test_update_sync_task_rejects_unowned_generation_model_config(
    monkeypatch,
    config_key,
    owner_id,
):
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {"task_id": "sync-owned", "user_id": "user-1"},
    )
    monkeypatch.setattr(
        model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "private-model-config",
            "user_id": owner_id,
            "api_endpoint": "https://models.example.com/v1",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unowned model config must not be persisted")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.update_sync_task(
                "sync-owned",
                sync_routes.SyncTaskUpdateRequest(
                    generation_config={
                        config_key: {"config_id": "private-model-config"},
                    },
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Not authorized to use this model config"
    assert "private-model-config" not in exc_info.value.detail


def test_create_sync_task_rejects_missing_generation_model_config(monkeypatch):
    monkeypatch.setattr(model_config_service, "get_config", lambda _config_id: None)
    monkeypatch.setattr(
        external_sync_service,
        "create_task",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing model config must not be persisted")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.create_sync_task(
                sync_routes.SyncTaskCreateRequest(
                    task_name="missing-generation-config",
                    external_api_url="https://8.8.8.8/data",
                    generation_config={
                        "embedding_config": {"config_id": "missing"},
                    },
                    is_active=False,
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 404


def test_create_sync_task_rejects_direct_generation_metadata_endpoint(monkeypatch):
    monkeypatch.setattr(
        external_sync_service,
        "create_task",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("blocked generation endpoint must not be persisted")
        ),
    )

    with pytest.raises(SSRFError):
        asyncio.run(
            sync_routes.create_sync_task(
                sync_routes.SyncTaskCreateRequest(
                    task_name="blocked-generation-endpoint",
                    external_api_url="https://sync.example.com/data",
                    generation_config={
                        "rerank_config": {
                            "endpoint": "http://169.254.169.254/latest/meta-data",
                            "model": "reranker",
                            "api_key": "secret",
                        },
                    },
                    is_active=False,
                ),
                CURRENT_USER,
            )
        )


@pytest.mark.parametrize("owner_id", [None, "user-2"])
def test_sync_worker_rejects_unowned_model_config_before_reading_api_key(
    monkeypatch,
    owner_id,
):
    class _SecretGuard(dict):
        def get(self, key, default=None):
            if key == "api_key":
                raise AssertionError("api_key was read before ownership validation")
            return super().get(key, default)

    monkeypatch.setattr(
        model_config_service,
        "get_config",
        lambda _config_id: _SecretGuard(
            config_id="model-config-unowned",
            user_id=owner_id,
            api_endpoint="https://models.example.com/v1",
            model_name="model",
        ),
    )

    with pytest.raises(PermissionError, match="another user|not owned"):
        sync_worker._resolve_config_from_gen_cfg(
            {"llm_config": {"config_id": "model-config-unowned"}},
            "llm_config",
            "llm",
            model_config_service,
            expected_user_id="user-1",
        )


def test_sync_worker_validates_persisted_model_endpoint_before_reading_api_key(
    monkeypatch,
):
    class _SecretGuard(dict):
        def get(self, key, default=None):
            if key == "api_key":
                raise AssertionError("api_key was read before endpoint validation")
            return super().get(key, default)

    monkeypatch.setattr(
        model_config_service,
        "get_config",
        lambda _config_id: _SecretGuard(
            config_id="model-config-owned",
            user_id="user-1",
            api_endpoint="http://169.254.169.254/latest/meta-data",
            model_name="model",
        ),
    )

    with pytest.raises(SSRFError):
        sync_worker._resolve_config_from_gen_cfg(
            {"llm_config": {"config_id": "model-config-owned"}},
            "llm_config",
            "llm",
            model_config_service,
            expected_user_id="user-1",
        )


@pytest.mark.parametrize("config_key", ["llm_config", "rerank_config"])
def test_sync_worker_preserves_model_config_id_for_background_revalidation(
    monkeypatch,
    config_key,
):
    monkeypatch.setattr(
        model_config_service,
        "get_config",
        lambda config_id: {
            "config_id": config_id,
            "user_id": "user-1",
            "api_endpoint": "https://models.example.com/v1",
            "model_name": "model",
            "api_key": "secret",
        },
    )
    monkeypatch.setattr(
        sync_worker,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )

    result = sync_worker._resolve_config_from_gen_cfg(
        {config_key: {"config_id": "model-config-owned"}},
        config_key,
        "llm" if config_key == "llm_config" else "rerank",
        model_config_service,
        expected_user_id="user-1",
    )

    assert result["config_id"] == "model-config-owned"


def test_sync_worker_validates_direct_generation_endpoint():
    with pytest.raises(SSRFError):
        sync_worker._resolve_config_from_gen_cfg(
            {
                "embedding_config": {
                    "endpoint": "http://127.0.0.1:9997/v1",
                    "model": "embedding",
                    "api_key": "secret",
                }
            },
            "embedding_config",
            "embedding",
            model_config_service,
            expected_user_id="user-1",
        )


def test_create_active_sync_task_starts_worker(monkeypatch):
    started = []
    monkeypatch.setattr(
        external_sync_service,
        "create_task",
        lambda **kwargs: {
            "task_id": "sync-active",
            "is_active": kwargs["is_active"],
        },
    )
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "start_worker",
        lambda task_id: started.append(task_id),
    )

    response = asyncio.run(
        sync_routes.create_sync_task(
            sync_routes.SyncTaskCreateRequest(
                task_name="active",
                external_api_url="https://8.8.8.8/data",
                is_active=True,
            ),
            CURRENT_USER,
        )
    )

    assert response["task"]["task_id"] == "sync-active"
    assert started == ["sync-active"]


def test_target_update_rejects_target_from_different_parent(monkeypatch):
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {"task_id": "task-owned", "user_id": "user-1"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target",
        lambda _target_id: {"target_id": "target-foreign", "task_id": "task-other"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_training_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("foreign target must not be updated")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.update_training_target(
                "task-owned",
                "target-foreign",
                sync_routes.TrainingTargetUpdateRequest(target_name="changed"),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 404


def test_active_training_target_rejects_user_delete(monkeypatch):
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": "task-owned",
            "user_id": "user-1",
            "status": "training",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target",
        lambda _target_id: {
            "target_id": "target-active",
            "task_id": "task-owned",
            "status": "training",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.delete_training_target(
                "task-owned",
                "target-active",
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409


def test_target_create_rejects_foreign_base_deployment(monkeypatch):
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": "task-owned",
            "user_id": "user-1",
            "external_api_config_id": None,
        },
    )
    monkeypatch.setattr(
        deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-foreign",
            "user_id": "user-2",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_training_target",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("foreign deployment must not be stored")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.create_training_target(
                "task-owned",
                sync_routes.TrainingTargetCreateRequest(
                    target_name="foreign",
                    base_deployment_id="deployment-foreign",
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403


def test_background_model_path_resolution_rejects_foreign_deployment(monkeypatch):
    monkeypatch.setattr(
        deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-foreign",
            "model_id": "model-foreign",
            "user_id": "user-2",
        },
    )

    with pytest.raises(PermissionError, match="another user"):
        level2_handler._resolve_base_model_path(
            "deployment-foreign",
            "sync-sec",
            expected_user_id="user-1",
        )


def test_post_training_target_resolution_rejects_foreign_deployment(monkeypatch):
    monkeypatch.setattr(
        deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-foreign",
            "user_id": "user-2",
        },
    )

    with pytest.raises(PermissionError, match="another user"):
        post_training_handler._resolve_target_deployment_id(
            {
                "task_id": "task-owned",
                "user_id": "user-1",
                "external_api_config_id": None,
            },
            {
                "task_id": "task-owned",
                "target_id": "target-owned",
                "base_deployment_id": "deployment-foreign",
            },
            "sync-sec",
        )


def test_post_training_explicit_missing_deployment_never_auto_falls_back(
    monkeypatch,
):
    discovered = []
    monkeypatch.setattr(
        deployment_service,
        "get_deployment",
        lambda _deployment_id: None,
    )
    monkeypatch.setattr(
        post_training_handler,
        "_find_compatible_deployment",
        lambda *_args, **_kwargs: discovered.append(True) or "deployment-fallback",
    )

    resolved = post_training_handler._resolve_deployment_id(
        {
            "task_id": "task-explicit",
            "user_id": "user-1",
            "base_deployment_id": "deployment-missing",
            "training_config": {"base_model_path": "/models/base"},
        },
        "sync-sec",
    )

    assert resolved is None
    assert discovered == []


@pytest.mark.parametrize(
    ("route", "manager_method"),
    (
        (sync_routes.sync_now, "run_once"),
        (sync_routes.trigger_generation, "trigger_generation"),
    ),
)
def test_reconciliation_unavailable_maps_to_redacted_503(
    monkeypatch,
    route,
    manager_method,
):
    class ReconciliationUnavailable(RuntimeError):
        pass

    monkeypatch.setattr(
        sync_manager_module,
        "SyncGenerationReconciliationError",
        ReconciliationUnavailable,
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": CURRENT_USER["user_id"],
            "status": "generating",
        },
    )

    async def unavailable(_task_id):
        raise ReconciliationUnavailable("credential-bearing-database-detail")

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        manager_method,
        unavailable,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(route("sync-sensitive-task", CURRENT_USER))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Sync generation state is temporarily unavailable"
    assert "credential-bearing" not in str(exc_info.value.detail)
