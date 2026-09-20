"""Development storage identity must not change database tenant ownership."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from train_factory.api.routes import sync_routes
from train_factory.config.settings import get_settings
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.external_api_config_entity import ExternalApiConfigDB
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncBatchDB,
    ExternalSyncGenerationDB,
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.services.external_api_config_service import external_api_config_service
from train_factory.storage.services.external_sync_service import external_sync_service
from train_factory.sync import sync_worker


@pytest.fixture
def development_storage(monkeypatch, tmp_path):
    settings = get_settings().model_copy(update={"auth_enabled": False})
    monkeypatch.setattr(sync_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(sync_worker, "get_settings", lambda: settings)
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(tmp_path / "sync"))
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    for entity in (
        DeploymentDB, ExternalApiConfigDB, ExternalSyncTaskDB, ExternalSyncBatchDB,
        ExternalSyncGenerationDB, ExternalSyncTrainingDB, ExternalSyncTrainingTargetDB,
    ):
        entity.__table__.create(engine)
    monkeypatch.setattr(external_api_config_service, "engine", engine)
    monkeypatch.setattr(external_sync_service, "engine", engine)
    yield settings, tmp_path / "sync"
    engine.dispose()


def _client(current_user):
    app = FastAPI()
    app.include_router(sync_routes.router, prefix="/api/sync")
    app.dependency_overrides[sync_routes.get_current_user] = lambda: current_user
    return TestClient(app)


def _batch(task_id, path, owner=""):
    return {
        "batch_id": "batch-one", "task_id": task_id, "user_id": owner,
        "storage_path": str(path), "record_count": 1,
    }


def test_new_development_task_references_empty_owner_config_and_full_file_lifecycle(
    development_storage,
):
    _settings, root = development_storage
    api = external_api_config_service.create_config(
        config_name="development-api", user_id="", api_url="https://8.8.8.8/items",
        auth_config={"token": "test-only-token"},
    )
    with _client({"user_id": None, "username": "anonymous"}) as client:
        response = client.post("/api/sync/tasks", json={
            "task_name": "development-task", "external_api_config_id": api["config_id"],
            "is_active": False, "generation_threshold": 99999, "training_threshold": 99999,
        })
        assert response.status_code == 201, response.text
    task = external_sync_service.get_task_raw(response.json()["task"]["task_id"])
    assert task["user_id"] == api["user_id"] == ""
    path = Path(sync_worker._save_batch(task, [{"id": "one", "text": "document"}]))
    assert path.parent == root / "anonymous" / task["task_id"]
    assert json.loads(path.read_text())["content"] == "document"
    batch, created = external_sync_service.create_batch(
        task_id=task["task_id"], user_id=task["user_id"], record_count=1,
        storage_path=str(path), since_time=None,
    )
    assert created and batch["user_id"] == ""
    assert sync_worker._select_generation_batches(task, [batch]) == [batch]
    merged = Path(sync_worker._merge_batches(task, [batch]))
    assert merged.parent == root / "anonymous" / task["task_id"] / "merged"
    assert json.loads(merged.read_text())["content"] == "document"
    snapshot, files = sync_routes._build_sync_batch_file_manifest(task["task_id"], "")
    assert [row["batch_id"] for row in snapshot] == [batch["batch_id"]]
    assert files == (path.resolve(),)
    assert sync_worker._delete_managed_merged_file(task["task_id"], "", str(merged))
    assert sync_worker._delete_managed_batch_files(task["task_id"], "", [batch]) == (1, 0)
    assert not path.exists() and not merged.exists()
    assert external_sync_service.get_task_raw(task["task_id"])["user_id"] == ""


@pytest.mark.parametrize("legacy_short", [False, True])
def test_legacy_empty_owner_batch_stays_owned_during_merge_and_cleanup(
    development_storage, legacy_short,
):
    _settings, root = development_storage
    task_id = "deadbeef-legacy-task"
    legacy_dir = root / (task_id[:8] if legacy_short else task_id)
    legacy_dir.mkdir(parents=True)
    path = legacy_dir / "batch_legacy.jsonl"
    path.write_text('{"content":"legacy document"}\n', encoding="utf-8")
    batch = _batch(task_id, path)
    task = {"task_id": task_id, "user_id": ""}
    assert sync_worker._resolve_managed_batch_path(task_id, "", batch) == path.resolve()
    assert sync_worker._select_generation_batches(task, [batch]) == [batch]
    merged = Path(sync_worker._merge_batches(task, [batch]))
    assert merged.parent == root / "anonymous" / task_id / "merged"
    assert json.loads(merged.read_text()) == {"content": "legacy document"}
    assert sync_worker._delete_managed_merged_file(task_id, "", str(merged))
    assert sync_worker._delete_managed_batch_files(task_id, "", [batch]) == (1, 0)


@pytest.mark.parametrize("operation", ["save", "resolve", "select", "merge", "delete_batch", "delete_merged"])
def test_empty_owner_compatibility_is_disabled_when_authentication_is_enabled(
    development_storage, operation,
):
    settings, root = development_storage
    settings.auth_enabled = True
    task = {"task_id": "old-task", "user_id": ""}
    path = root / "old-task" / "batch_old.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"content":"preserve"}\n', encoding="utf-8")
    batch = _batch(task["task_id"], path)
    calls = {
        "save": lambda: sync_worker._save_batch(task, [{"text": "new"}]),
        "resolve": lambda: sync_worker._resolve_managed_batch_path(task["task_id"], "", batch),
        "select": lambda: sync_worker._select_generation_batches(task, [batch]),
        "merge": lambda: sync_worker._merge_batches(task, [batch]),
        "delete_batch": lambda: sync_worker._delete_managed_batch_files(task["task_id"], "", [batch]),
        "delete_merged": lambda: sync_worker._delete_managed_merged_file(task["task_id"], "", str(path)),
    }
    with pytest.raises(ValueError, match="User ID path component"):
        calls[operation]()
    assert path.read_text() == '{"content":"preserve"}\n'


@pytest.mark.parametrize("stored_owner", ["", "anonymous"])
def test_authenticated_real_user_cannot_access_development_task(
    development_storage, stored_owner,
):
    settings, _root = development_storage
    task = external_sync_service.create_task(
        task_name="old-development-task", user_id=stored_owner, is_active=False,
    )
    settings.auth_enabled = True
    with _client({"user_id": "770516af-872f-49c4-bc2e-38aeadb13f20", "username": "anonymous"}) as client:
        response = client.get(f"/api/sync/tasks/{task['task_id']}")
    assert response.status_code == 403


@pytest.mark.parametrize("owner", [None, " ", "../tenant", "tenant/child", "CON"])
def test_development_mapping_does_not_accept_other_invalid_owner_values(
    development_storage, owner,
):
    with pytest.raises(ValueError, match="User ID path component"):
        sync_worker._save_batch({"task_id": "task-one", "user_id": owner}, [{"text": "new"}])


def test_development_mapping_does_not_equate_database_owners(development_storage):
    _settings, root = development_storage
    path = root / "anonymous" / "task-one" / "batch_one.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"content":"preserve"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not belong"):
        sync_worker._resolve_managed_batch_path("task-one", "", _batch("task-one", path, "anonymous"))
    assert path.exists()


def test_development_legacy_files_cannot_bypass_storage_quota(development_storage):
    settings, root = development_storage
    path = root / "oldtask1" / "batch_old.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 1000)
    settings.sync_storage_max_bytes_per_user = 1000
    with pytest.raises(sync_worker.SyncResourceLimitExceeded, match="Per-user"):
        sync_worker._save_batch({"task_id": "new-task", "user_id": ""}, [{"text": "new"}])
    assert list(root.rglob("batch_*.jsonl")) == [path]


def test_development_storage_rejects_symlinked_anonymous_directory(development_storage, tmp_path):
    _settings, root = development_storage
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "anonymous").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        sync_worker._save_batch({"task_id": "task-one", "user_id": ""}, [{"text": "new"}])
    assert list(outside.iterdir()) == []
