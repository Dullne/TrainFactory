"""
Sync API integration tests.

Tests all external API config and sync config CRUD endpoints,
worker control, and runtime resolution.

Requires the API server running at TEST_API_BASE (default: http://localhost:18000/api).
Run: pytest tests/test_sync_api.py -v
"""

import os
import time

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SYNC_INTEGRATION") != "1",
    reason="set RUN_SYNC_INTEGRATION=1 and use an isolated API/database",
)

BASE = os.getenv("TEST_API_BASE", "http://localhost:18000/api").rstrip("/")
EXTERNAL_API_HOST = os.getenv("TEST_EXTERNAL_API_HOST", "localhost")
UNREACHABLE_API_URL = f"http://{EXTERNAL_API_HOST}:19999/api/texts"


# ── Helpers ──────────────────────────────────────────────


def create_api_config(**overrides):
    data = {
        "config_name": "pytest-api-cfg",
        "api_url": UNREACHABLE_API_URL,
        "auth_config": {"token": "pytest-token"},
        "description": "pytest test",
    }
    data.update(overrides)
    r = requests.post(f"{BASE}/sync/api-configs", json=data)
    assert r.status_code == 201, f"Create API config failed: {r.text}"
    return r.json()["config"]


def create_sync_config(**overrides):
    data = {
        "task_name": "pytest-sync-cfg",
        "is_active": False,
        "generation_threshold": 9999,
        "training_threshold": 9999,
        "sync_interval_seconds": 600,
        "generation_config": {},
    }
    data.update(overrides)
    r = requests.post(f"{BASE}/sync/tasks", json=data)
    assert r.status_code == 201, f"Create sync config failed: {r.text}"
    return r.json()["task"]


def delete_api_config(config_id):
    requests.delete(f"{BASE}/sync/api-configs/{config_id}")


def delete_sync_config(config_id):
    requests.post(f"{BASE}/sync/tasks/{config_id}/stop")
    requests.delete(f"{BASE}/sync/tasks/{config_id}")


# ── External API Config CRUD ─────────────────────────────


class TestExternalApiConfigCRUD:

    def test_create_api_config(self):
        cfg = create_api_config(config_name="test-create")
        try:
            assert cfg["config_id"]
            assert cfg["config_name"] == "test-create"
            assert cfg["api_url"] == UNREACHABLE_API_URL
            assert cfg["auth_config"]["token"] == "***"
            # Unreachable endpoint should be marked as error after auto-test.
            assert cfg["status"] == "error"
        finally:
            delete_api_config(cfg["config_id"])

    def test_list_api_configs(self):
        cfg = create_api_config(config_name="test-list")
        try:
            r = requests.get(f"{BASE}/sync/api-configs")
            assert r.status_code == 200
            data = r.json()
            assert data["total"] >= 1
            ids = [c["config_id"] for c in data["configs"]]
            assert cfg["config_id"] in ids
        finally:
            delete_api_config(cfg["config_id"])

    def test_get_api_config(self):
        cfg = create_api_config(config_name="test-get")
        try:
            r = requests.get(f"{BASE}/sync/api-configs/{cfg['config_id']}")
            assert r.status_code == 200
            got = r.json()["config"]
            assert got["config_name"] == "test-get"
            assert got["api_url"] == cfg["api_url"]
        finally:
            delete_api_config(cfg["config_id"])

    def test_update_api_config(self):
        cfg = create_api_config(config_name="test-update")
        try:
            updated_url = f"http://{EXTERNAL_API_HOST}:19998/api"
            r = requests.patch(
                f"{BASE}/sync/api-configs/{cfg['config_id']}",
                json={"config_name": "updated-name", "api_url": updated_url},
            )
            assert r.status_code == 200
            updated = r.json()["config"]
            assert updated["config_name"] == "updated-name"
            assert updated["api_url"] == updated_url
        finally:
            delete_api_config(cfg["config_id"])

    def test_delete_api_config(self):
        cfg = create_api_config(config_name="test-delete")
        r = requests.delete(f"{BASE}/sync/api-configs/{cfg['config_id']}")
        assert r.status_code == 200
        r = requests.get(f"{BASE}/sync/api-configs/{cfg['config_id']}")
        assert r.status_code == 404

    def test_delete_referenced_api_config_returns_409(self):
        api_cfg = create_api_config(config_name="test-409-ref")
        sync_cfg = create_sync_config(external_api_config_id=api_cfg["config_id"])
        try:
            r = requests.delete(f"{BASE}/sync/api-configs/{api_cfg['config_id']}")
            assert r.status_code == 409
            assert "referenced" in r.json()["detail"].lower()
        finally:
            delete_sync_config(sync_cfg["task_id"])
            delete_api_config(api_cfg["config_id"])


# ── Sync Config CRUD ─────────────────────────────────────


class TestSyncConfigCRUD:

    def test_create_with_api_ref(self):
        api_cfg = create_api_config(config_name="ref-api")
        try:
            sync_cfg = create_sync_config(external_api_config_id=api_cfg["config_id"])
            try:
                assert sync_cfg["external_api_config_id"] == api_cfg["config_id"]
            finally:
                delete_sync_config(sync_cfg["task_id"])
        finally:
            delete_api_config(api_cfg["config_id"])

    def test_create_with_direct_url(self):
        sync_cfg = create_sync_config(
            external_api_url="http://example.com/api",
            external_auth_config={"token": "direct"},
        )
        try:
            assert sync_cfg["external_api_config_id"] is None
        finally:
            delete_sync_config(sync_cfg["task_id"])

    def test_create_missing_api_returns_400(self):
        r = requests.post(
            f"{BASE}/sync/tasks",
            json={"task_name": "no-api", "generation_config": {}},
        )
        assert r.status_code == 400

    def test_get_sync_config(self):
        api_cfg = create_api_config(config_name="get-api")
        sync_cfg = create_sync_config(external_api_config_id=api_cfg["config_id"])
        try:
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}")
            assert r.status_code == 200
            got = r.json()["task"]
            assert got["task_name"] == "pytest-sync-cfg"
            assert got["external_api_config_id"] == api_cfg["config_id"]
        finally:
            delete_sync_config(sync_cfg["task_id"])
            delete_api_config(api_cfg["config_id"])

    def test_update_sync_config(self):
        api_cfg = create_api_config(config_name="upd-api")
        sync_cfg = create_sync_config(external_api_config_id=api_cfg["config_id"])
        try:
            r = requests.patch(
                f"{BASE}/sync/tasks/{sync_cfg['task_id']}",
                json={"generation_threshold": 42, "task_name": "updated-sync"},
            )
            assert r.status_code == 200
            updated = r.json()["task"]
            assert updated["generation_threshold"] == 42
            assert updated["task_name"] == "updated-sync"
        finally:
            delete_sync_config(sync_cfg["task_id"])
            delete_api_config(api_cfg["config_id"])

    def test_delete_sync_config(self):
        sync_cfg = create_sync_config(
            external_api_url="http://example.com/api",
            external_auth_config={"token": "del"},
        )
        r = requests.delete(f"{BASE}/sync/tasks/{sync_cfg['task_id']}")
        assert r.status_code == 200

    def test_get_sync_status(self):
        sync_cfg = create_sync_config(
            external_api_url="http://example.com/api",
            external_auth_config={"token": "status"},
        )
        try:
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/status")
            assert r.status_code == 200
            s = r.json()
            assert "worker_status" in s
            assert "pending_record_count" in s
            assert s["pending_record_count"] == 0
        finally:
            delete_sync_config(sync_cfg["task_id"])

    def test_list_batches_empty(self):
        sync_cfg = create_sync_config(
            external_api_url="http://example.com/api",
            external_auth_config={"token": "b"},
        )
        try:
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/batches")
            assert r.status_code == 200
            assert r.json()["total"] == 0
        finally:
            delete_sync_config(sync_cfg["task_id"])

    def test_list_generations_empty(self):
        sync_cfg = create_sync_config(
            external_api_url="http://example.com/api",
            external_auth_config={"token": "g"},
        )
        try:
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/generations")
            assert r.status_code == 200
            assert r.json()["total"] == 0
        finally:
            delete_sync_config(sync_cfg["task_id"])

    def test_list_trainings_empty(self):
        sync_cfg = create_sync_config(
            external_api_url="http://example.com/api",
            external_auth_config={"token": "t"},
        )
        try:
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/trainings")
            assert r.status_code == 200
            assert r.json()["total"] == 0
        finally:
            delete_sync_config(sync_cfg["task_id"])


# ── Worker Control ───────────────────────────────────────


class TestWorkerControl:

    def test_start_sync(self):
        sync_cfg = create_sync_config(
            external_api_url="http://example.com/api",
            external_auth_config={"token": "start"},
        )
        try:
            r = requests.post(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/start")
            assert r.status_code == 200
            assert r.json()["worker_status"] == "running"
            time.sleep(1)
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/status")
            assert r.json()["is_active"] is True
        finally:
            delete_sync_config(sync_cfg["task_id"])

    def test_stop_sync(self):
        sync_cfg = create_sync_config(
            external_api_url="http://example.com/api",
            external_auth_config={"token": "stop"},
        )
        try:
            requests.post(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/start")
            time.sleep(1)
            r = requests.post(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/stop")
            assert r.status_code == 200
            time.sleep(1)
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/status")
            assert r.json()["is_active"] is False
        finally:
            delete_sync_config(sync_cfg["task_id"])


# ── Runtime Resolution ───────────────────────────────────


class TestRuntimeResolution:

    def test_runtime_url_resolution(self):
        api_cfg = create_api_config(
            config_name="resolve-url",
            api_url="http://resolved-host:9000/api/data",
            auth_config={"token": "resolve-v1"},
        )
        sync_cfg = create_sync_config(external_api_config_id=api_cfg["config_id"])
        try:
            from train_factory.storage.services.external_sync_service import (
                external_sync_service,
            )

            raw = external_sync_service.get_task_raw(sync_cfg["task_id"])
            assert raw["external_api_url"] == "http://resolved-host:9000/api/data"
            assert raw["external_auth_config"]["token"] == "resolve-v1"
        finally:
            delete_sync_config(sync_cfg["task_id"])
            delete_api_config(api_cfg["config_id"])

    def test_runtime_token_update(self):
        api_cfg = create_api_config(
            config_name="resolve-token",
            auth_config={"token": "old-token"},
        )
        sync_cfg = create_sync_config(external_api_config_id=api_cfg["config_id"])
        try:
            from train_factory.storage.services.external_sync_service import (
                external_sync_service,
            )

            raw1 = external_sync_service.get_task_raw(sync_cfg["task_id"])
            assert raw1["external_auth_config"]["token"] == "old-token"

            requests.patch(
                f"{BASE}/sync/api-configs/{api_cfg['config_id']}",
                json={"auth_config": {"token": "new-token"}},
            )

            raw2 = external_sync_service.get_task_raw(sync_cfg["task_id"])
            assert raw2["external_auth_config"]["token"] == "new-token"
        finally:
            delete_sync_config(sync_cfg["task_id"])
            delete_api_config(api_cfg["config_id"])
