"""Manual sync must report source failure without losing pending job handoffs."""

import asyncio
import importlib

import pytest
from fastapi import HTTPException

from train_factory.api.routes import sync_routes
from train_factory.sync import sync_worker


@pytest.mark.parametrize("failure", ["fetch", "validation"])
@pytest.mark.parametrize("pending_generation", [False, True])
def test_manual_sync_reports_source_failure_and_preserves_handoff(
    monkeypatch, failure, pending_generation,
):
    manager_module = importlib.import_module("train_factory.sync.sync_manager")
    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    state = {
        "task_id": "manual-source-failure",
        "user_id": "owner",
        "status": "idle",
        "external_api_url": "https://source.invalid/items",
        "external_auth_config": {},
        "last_sync_at": None,
    }
    handed_off = []

    class Service:
        def get_task(self, _task_id):
            return dict(state)

        get_task_raw = get_task

        def update_task(self, _task_id, **updates):
            state.update(updates)
            return dict(state)

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def fetch_all_since(self, _since):
            if failure == "fetch":
                raise RuntimeError("private-upstream-credential-detail")
            return [{"id": "invalid-record-without-required-fields"}]

    async def promote(_config):
        return None

    async def check_thresholds(_config):
        if pending_generation:
            state["status"] = "generating"
            return {"gen_task_id": "pending-generation", "pipeline_config": {}}
        return None

    manager = manager_module.SyncManager()
    monkeypatch.setattr(manager_module, "sync_manager", manager)
    monkeypatch.setattr(service_module, "external_sync_service", Service())
    monkeypatch.setattr(sync_worker, "ExternalApiClient", Client)
    monkeypatch.setattr(sync_worker, "validate_user_outbound_url", lambda url, _uid: url)
    monkeypatch.setattr(sync_worker, "_promote_registered_batches", promote)
    monkeypatch.setattr(sync_worker, "_check_thresholds", check_thresholds)
    monkeypatch.setattr(sync_worker, "_update_api_config_status", lambda *_args: None)
    monkeypatch.setattr(manager, "_launch_generation", handed_off.append)

    async def exercise():
        for _attempt in range(2):
            state["status"] = "idle"
            with pytest.raises(HTTPException) as raised:
                await sync_routes.sync_now(state["task_id"], {"user_id": "owner"})
            assert raised.value.status_code == 502
            assert raised.value.detail == "Sync source fetch or validation failed"
            assert state["status"] == ("generating" if pending_generation else "error")
            assert state["last_sync_at"] is None

    asyncio.run(exercise())
    assert len(handed_off) == (2 if pending_generation else 0)
