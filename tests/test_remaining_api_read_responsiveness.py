"""Remaining synchronous route dependencies must not stall the API loop."""

import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI

from train_factory.api.routes import adapter_routes, auth_routes, sync_routes
from train_factory.auth.dependencies import get_current_user
from train_factory.storage.services.external_sync_service import external_sync_service


CURRENT_USER = {"user_id": "read-test-user", "username": "alice", "role": "user"}


def _app_with_probe(router):
    app = FastAPI()
    app.include_router(router)

    async def identity():
        return CURRENT_USER

    app.dependency_overrides[get_current_user] = identity

    @app.get("/probe-ping")
    async def ping():
        return {"available": True}

    return app


def _run_responsiveness_probe(app, path, entered, release, returned, assert_result):
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(client.get(path))
            try:
                async with asyncio.timeout(3):
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                response = await asyncio.wait_for(client.get("/probe-ping"), 1)
                assert response.json() == {"available": True}
                assert not returned.is_set(), "synchronous service work stalled the API loop"
            finally:
                release.set()
                result = await request
            assert_result(result)

    asyncio.run(exercise())


def _assert_ok(response):
    assert response.status_code == 200


def test_slow_auth_user_lookup_keeps_other_requests_responsive(monkeypatch):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def get_user(_user_id):
        entered.set()
        release.wait(2)
        returned.set()
        return {
            "user_id": "read-test-user",
            "username": "alice",
            "email": None,
            "is_active": True,
            "is_admin": False,
            "created_at": None,
            "updated_at": None,
        }

    monkeypatch.setattr(auth_routes.user_service, "get_user", get_user)
    app = _app_with_probe(auth_routes.router)
    _run_responsiveness_probe(
        app,
        "/me",
        entered,
        release,
        returned,
        _assert_ok,
    )


def test_slow_adapter_lookup_keeps_other_requests_responsive(monkeypatch):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def get_deployment(_deployment_id):
        entered.set()
        release.wait(2)
        returned.set()
        return {"deployment_id": "deployment-1", "user_id": "read-test-user"}

    monkeypatch.setattr(adapter_routes.deployment_service, "get_deployment", get_deployment)
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "list_loaded_adapters",
        lambda *_args, **_kwargs: [],
    )
    app = _app_with_probe(adapter_routes.router)
    _run_responsiveness_probe(
        app,
        "/deployments/deployment-1/adapters",
        entered,
        release,
        returned,
        _assert_ok,
    )


def test_slow_sync_task_list_keeps_other_requests_responsive(monkeypatch):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def list_tasks(**_kwargs):
        entered.set()
        release.wait(2)
        returned.set()
        return [], 0

    monkeypatch.setattr(external_sync_service, "list_tasks", list_tasks)
    app = _app_with_probe(sync_routes.router)
    _run_responsiveness_probe(
        app,
        "/tasks",
        entered,
        release,
        returned,
        _assert_ok,
    )


def test_aborted_sync_deletion_resumes_worker_on_request_loop(monkeypatch):
    from fastapi import HTTPException

    from train_factory.sync import sync_manager as manager_module

    manager = manager_module.SyncManager()
    manager._running = True
    task_id = "sync-resume-after-abort"
    task = {
        "task_id": task_id,
        "user_id": CURRENT_USER["user_id"],
        "status": "idle",
        "is_active": True,
    }
    worker_loops = []

    async def worker(_task_id):
        worker_loops.append(asyncio.get_running_loop())
        await asyncio.Event().wait()

    def unavailable_snapshot(_task_id, _config):
        raise HTTPException(status_code=409, detail="Child snapshot unavailable")

    monkeypatch.setattr(manager, "_run_worker", worker)
    monkeypatch.setattr(manager_module, "sync_manager", manager)
    monkeypatch.setattr(external_sync_service, "get_task", lambda _id: dict(task))
    monkeypatch.setattr(
        sync_routes, "_snapshot_sync_deletion_children", unavailable_snapshot
    )

    async def exercise():
        try:
            with pytest.raises(HTTPException) as error:
                await sync_routes.delete_sync_task(task_id, True, CURRENT_USER)
            assert error.value.status_code == 409
            await asyncio.sleep(0)
            assert manager.get_worker_status(task_id) == "running"
            assert worker_loops == [asyncio.get_running_loop()]
        finally:
            await manager.stop_worker(task_id)

    asyncio.run(exercise())
