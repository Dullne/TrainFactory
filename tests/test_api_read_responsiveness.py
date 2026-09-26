"""Synchronous read dependencies must not stall unrelated API requests."""

import asyncio
import importlib
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from train_factory.api.routes import (
    dataset_routes,
    external_api_config_routes,
    generation_routes,
    model_config_routes,
    registry_routes,
    training_routes,
)
from train_factory.auth.dependencies import get_current_user
from train_factory.storage.services.external_api_config_service import (
    external_api_config_service,
)

model_config_service_module = importlib.import_module(
    "train_factory.storage.services.model_config_service"
)


CASES = [
    (training_routes, "training_task_service", "get_all_tasks", "get_task_stats", "/train"),
    (generation_routes, "generation_task_service", "get_all_tasks", "get_task_stats", "/tasks"),
    (registry_routes, "model_registry_service", "list_models", "get_stats", "/models"),
    (dataset_routes, "dataset_service", "list_datasets", "get_stats", "/datasets"),
]


@pytest.mark.parametrize("module,service_name,list_method,stats_method,path", CASES)
@pytest.mark.parametrize("stage", ("list", "stats"))
def test_slow_list_or_statistics_keeps_other_requests_responsive(
    monkeypatch, module, service_name, list_method, stats_method, path, stage,
):
    service = getattr(module, service_name)
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def query(kind, value, **kwargs):
        assert kwargs["user_id"] == "read-test-user"
        if kind == stage:
            entered.set()
            release.wait(2)
            returned.set()
        return value

    monkeypatch.setattr(service, list_method, lambda **kw: query("list", ([], 0), **kw))
    monkeypatch.setattr(service, stats_method, lambda **kw: query("stats", {}, **kw))
    app = FastAPI()
    app.include_router(module.router)

    async def identity():
        return {"user_id": "read-test-user", "username": "alice", "role": "user"}

    app.dependency_overrides[get_current_user] = identity

    @app.get("/probe-ping")
    async def ping():
        return {"available": True}

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            request = asyncio.create_task(client.get(path))
            try:
                async with asyncio.timeout(3):
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                response = await asyncio.wait_for(client.get("/probe-ping"), 1)
                assert response.json() == {"available": True}
                assert not returned.is_set(), "database query stalled the API event loop"
            finally:
                release.set()
                result = await request
            assert result.status_code == 200
            assert result.json()["total"] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "path, service_method, value",
    [
        ("/configs", "list_configs", ([], 0)),
        (
            "/configs/stats",
            "get_stats",
            {"total": 0, "by_type": {}, "by_provider": {}, "by_status": {}},
        ),
    ],
)
def test_slow_model_config_reads_keep_other_requests_responsive(
    monkeypatch, path, service_method, value,
):
    service = model_config_routes.model_config_service
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def query(**kwargs):
        assert kwargs["user_id"] == "read-test-user"
        entered.set()
        release.wait(2)
        returned.set()
        return value

    monkeypatch.setattr(service, service_method, query)
    app = FastAPI()
    app.include_router(model_config_routes.router)

    async def identity():
        return {"user_id": "read-test-user", "username": "alice", "role": "user"}

    app.dependency_overrides[get_current_user] = identity

    @app.get("/probe-ping")
    async def ping():
        return {"available": True}

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            request = asyncio.create_task(client.get(path))
            try:
                async with asyncio.timeout(3):
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                response = await asyncio.wait_for(client.get("/probe-ping"), 1)
                assert response.json() == {"available": True}
                assert not returned.is_set(), "database query stalled the API event loop"
            finally:
                release.set()
                result = await request
            assert result.status_code == 200

    asyncio.run(exercise())


def test_slow_external_api_config_list_keeps_other_requests_responsive(monkeypatch):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def list_configs(*, user_id):
        assert user_id == "read-test-user"
        entered.set()
        release.wait(2)
        returned.set()
        return [], 0

    monkeypatch.setattr(external_api_config_service, "list_configs", list_configs)
    app = FastAPI()
    app.include_router(external_api_config_routes.router)

    async def identity():
        return {"user_id": "read-test-user", "username": "alice", "role": "user"}

    app.dependency_overrides[get_current_user] = identity

    @app.get("/probe-ping")
    async def ping():
        return {"available": True}

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            request = asyncio.create_task(client.get("/api-configs"))
            try:
                async with asyncio.timeout(3):
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                response = await asyncio.wait_for(client.get("/probe-ping"), 1)
                assert response.json() == {"available": True}
                assert not returned.is_set(), "database query stalled the API event loop"
            finally:
                release.set()
                result = await request
            assert result.status_code == 200
            assert result.json()["total"] == 0

    asyncio.run(exercise())


def test_slow_model_config_connectivity_db_work_keeps_other_requests_responsive(monkeypatch):
    service = model_config_routes.model_config_service
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def get_config(_config_id):
        entered.set()
        release.wait(2)
        returned.set()
        return {
            "config_id": "config-1",
            "provider": "openai",
            "api_endpoint": "https://api.example.invalid/v1",
            "api_key": None,
            "model_name": "model-1",
            "user_id": "read-test-user",
            "source_type": "external_api",
            "deployment_id": None,
        }

    async def outbound(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, json=lambda: {"data": []})

    monkeypatch.setattr(service, "get_config", get_config)
    monkeypatch.setattr(service, "update_check_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "_sync_deployment_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_config_service_module,
        "normalize_api_endpoint",
        lambda *_args, **_kwargs: "https://api.example.invalid/v1",
    )
    monkeypatch.setattr(model_config_service_module, "async_request_user_outbound", outbound)

    app = FastAPI()

    @app.get("/check")
    async def check():
        return await service.check_connectivity("config-1")

    @app.get("/probe-ping")
    async def ping():
        return {"available": True}

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            request = asyncio.create_task(client.get("/check"))
            try:
                async with asyncio.timeout(3):
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                response = await asyncio.wait_for(client.get("/probe-ping"), 1)
                assert response.json() == {"available": True}
                assert not returned.is_set(), "database query stalled the API event loop"
            finally:
                release.set()
                result = await request
            assert result.status_code == 200
            assert result.json()["success"] is True

    asyncio.run(exercise())
