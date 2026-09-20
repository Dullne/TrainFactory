"""Synchronous read dependencies must not stall unrelated API requests."""

import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI

from train_factory.api.routes import (
    dataset_routes, generation_routes, registry_routes, training_routes,
)
from train_factory.auth.dependencies import get_current_user


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
