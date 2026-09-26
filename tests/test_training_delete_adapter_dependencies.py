import asyncio
import importlib
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

import train_factory.deployment.adapter_service as adapter_service_module
from train_factory.api.routes import (
    adapter_routes,
    registry_routes,
    sync_routes,
    training_routes,
)
from train_factory.sync import post_training_handler
from train_factory.sync.sync_manager import SyncManager
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
)
from train_factory.deployment.deployment_service import ReplicaOperationBusyError
from train_factory.storage.services.external_sync_service import (
    external_sync_service,
)
from train_factory.storage.services.training_task_service import (
    training_task_service,
)

USER = {"user_id": "user-1", "is_admin": False}
sync_manager_module = importlib.import_module("train_factory.sync.sync_manager")


@pytest.fixture(autouse=True)
def no_registry_path_dependencies(monkeypatch):
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "list_models_referencing_artifact_paths",
        lambda *_args, **_kwargs: [],
        raising=False,
    )


@pytest.fixture
def adapter_database(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    LoadedAdapterDB.__table__.create(engine)

    @contextmanager
    def get_test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(adapter_service_module, "get_session", get_test_session)
    return engine


def _add_adapter(
    engine,
    *,
    adapter_id: str,
    status: str,
    user_id: str | None = USER["user_id"],
    source_task_id: str | None = None,
    adapter_path: str = "/app/output/other/final_model",
) -> None:
    with Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                adapter_id=adapter_id,
                deployment_id=f"deployment-{adapter_id}",
                adapter_name=adapter_id,
                adapter_path=adapter_path,
                source_task_id=source_task_id,
                status=status,
                user_id=user_id,
            )
        )
        session.commit()


def _loaded_adapter(task_id: str) -> dict:
    return {
        "adapter_id": "adapter-1",
        "deployment_id": "deployment-1",
        "adapter_name": "adapter",
        "adapter_path": f"/app/output/{task_id}/final_model",
        "source_task_id": task_id,
        "source_model_id": None,
        "status": "loaded",
        "error_message": None,
        "user_id": USER["user_id"],
        "loaded_at": None,
        "unloaded_at": None,
    }


def _adapter_load_request(endpoint_name: str, task_id: str):
    if endpoint_name == "direct":
        return adapter_routes.load_adapter(
            "deployment-1",
            adapter_routes.LoadAdapterRequest(
                adapter_name="adapter",
                adapter_path=f"/app/output/{task_id}/final_model",
                source_task_id=task_id,
            ),
            USER,
        )
    return adapter_routes.load_adapter_from_task(
        "deployment-1",
        adapter_routes.LoadAdapterFromTaskRequest(
            task_id=task_id,
            adapter_name="adapter",
        ),
        USER,
    )


@pytest.mark.parametrize("endpoint_name", ["direct", "from-task"])
def test_adapter_load_holds_training_guard_while_resolving_and_loading(
    monkeypatch,
    endpoint_name,
):
    task_id = f"task-{endpoint_name}"
    events = []
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-1",
            "user_id": USER["user_id"],
        },
    )

    def begin_deletion(kind, requested_task_id):
        events.append(("begin", kind, requested_task_id))
        return SimpleNamespace(release=lambda: events.append(("release",)))

    monkeypatch.setattr(
        adapter_routes,
        "background_task_admission_service",
        SimpleNamespace(begin_deletion=begin_deletion),
        raising=False,
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_direct_adapter_path",
        lambda *_args: events.append(("resolve",))
        or f"/app/output/{task_id}/final_model",
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_owned_task_adapter_path",
        lambda *_args: events.append(("resolve",))
        or f"/app/output/{task_id}/final_model",
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda **_kwargs: events.append(("load",)) or _loaded_adapter(task_id),
    )

    response = asyncio.run(_adapter_load_request(endpoint_name, task_id))

    assert response.adapter_id == "adapter-1"
    assert events == [
        ("begin", "training", task_id),
        ("resolve",),
        ("load",),
        ("release",),
    ]


@pytest.mark.parametrize("endpoint_name", ["direct", "from-task"])
def test_adapter_load_maps_training_guard_conflict_to_409(
    monkeypatch,
    endpoint_name,
):
    task_id = f"task-{endpoint_name}"
    loaded = []
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-1",
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        adapter_routes,
        "background_task_admission_service",
        SimpleNamespace(
            begin_deletion=lambda *_args: (_ for _ in ()).throw(
                BackgroundTaskAlreadyExecuting("being deleted")
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda **kwargs: loaded.append(kwargs) or _loaded_adapter(task_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_adapter_load_request(endpoint_name, task_id))

    assert exc_info.value.status_code == 409
    assert loaded == []


@pytest.mark.parametrize("endpoint_name", ["direct", "from-task"])
def test_adapter_load_releases_training_guard_after_failure(
    monkeypatch,
    endpoint_name,
):
    task_id = f"task-{endpoint_name}"
    releases = []
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-1",
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        adapter_routes,
        "background_task_admission_service",
        SimpleNamespace(
            begin_deletion=lambda *_args: SimpleNamespace(
                release=lambda: releases.append(True)
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_direct_adapter_path",
        lambda *_args: f"/app/output/{task_id}/final_model",
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_owned_task_adapter_path",
        lambda *_args: f"/app/output/{task_id}/final_model",
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_adapter_load_request(endpoint_name, task_id))

    assert exc_info.value.status_code == 500
    assert releases == [True]


@pytest.mark.parametrize("endpoint_name", ["direct", "from-task"])
def test_cancelled_adapter_guard_acquisition_releases_delivered_guard(
    monkeypatch,
    endpoint_name,
):
    task_id = f"task-cancel-acquire-{endpoint_name}"
    acquisition_started = threading.Event()
    finish_acquisition = threading.Event()
    releases = []
    loads = []
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-1",
            "user_id": USER["user_id"],
        },
    )

    def begin_deletion(_kind, _task_id):
        acquisition_started.set()
        assert finish_acquisition.wait(5)
        return SimpleNamespace(release=lambda: releases.append(True))

    monkeypatch.setattr(
        adapter_routes,
        "background_task_admission_service",
        SimpleNamespace(begin_deletion=begin_deletion),
        raising=False,
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda **kwargs: loads.append(kwargs) or _loaded_adapter(task_id),
    )

    async def exercise():
        request = asyncio.create_task(_adapter_load_request(endpoint_name, task_id))
        try:
            assert await asyncio.to_thread(acquisition_started.wait, 2)
            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()
            assert releases == []

            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()
            assert releases == []

            finish_acquisition.set()
            with pytest.raises(asyncio.CancelledError):
                await request
        finally:
            finish_acquisition.set()
            if not request.done():
                request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request

    asyncio.run(exercise())

    assert releases == [True]
    assert loads == []


@pytest.mark.parametrize("endpoint_name", ["direct", "from-task"])
def test_cancelled_adapter_load_keeps_guard_until_worker_finishes(
    monkeypatch,
    endpoint_name,
):
    task_id = f"task-cancel-load-{endpoint_name}"
    load_started = threading.Event()
    finish_load = threading.Event()
    releases = []
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-1",
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        adapter_routes,
        "background_task_admission_service",
        SimpleNamespace(begin_deletion=lambda *_args: SimpleNamespace(release=lambda: releases.append(True))),
        raising=False,
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_direct_adapter_path",
        lambda *_args: f"/app/output/{task_id}/final_model",
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_owned_task_adapter_path",
        lambda *_args: f"/app/output/{task_id}/final_model",
    )

    def blocked_load(**_kwargs):
        load_started.set()
        assert finish_load.wait(5)
        return _loaded_adapter(task_id)

    monkeypatch.setattr(adapter_routes.adapter_service, "load_adapter", blocked_load)

    async def exercise():
        request = asyncio.create_task(_adapter_load_request(endpoint_name, task_id))
        try:
            assert await asyncio.to_thread(load_started.wait, 2)
            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()
            assert releases == []

            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()
            assert releases == []

            finish_load.set()
            with pytest.raises(asyncio.CancelledError):
                await request
        finally:
            finish_load.set()
            if not request.done():
                request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request

    asyncio.run(exercise())

    assert releases == [True]


@pytest.mark.parametrize("endpoint_name", ["direct", "from-task"])
def test_adapter_load_maps_replica_operation_conflict_to_409(
    monkeypatch,
    endpoint_name,
):
    task_id = f"task-{endpoint_name}"
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: {
            "deployment_id": "deployment-1",
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        adapter_routes,
        "background_task_admission_service",
        SimpleNamespace(
            begin_deletion=lambda *_args: SimpleNamespace(release=lambda: None)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_direct_adapter_path",
        lambda *_args: f"/app/output/{task_id}/final_model",
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_owned_task_adapter_path",
        lambda *_args: f"/app/output/{task_id}/final_model",
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda **_kwargs: (_ for _ in ()).throw(
            ReplicaOperationBusyError("deployment replica operation already in progress")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_adapter_load_request(endpoint_name, task_id))

    assert exc_info.value.status_code == 409


def test_active_training_output_consumers_match_source_or_normalized_path(
    adapter_database,
):
    task_id = "training-1"
    final_model_path = "/app/output/training-1/final_model/"
    _add_adapter(
        adapter_database,
        adapter_id="source-match",
        status="loaded",
        source_task_id=task_id,
    )
    _add_adapter(
        adapter_database,
        adapter_id="path-match-loading",
        status="loading",
        adapter_path=r"\app\output\training-1\.\final_model",
    )
    _add_adapter(
        adapter_database,
        adapter_id="path-match-unloading",
        status="unloading",
        adapter_path="/app/output/training-1/final_model",
    )
    _add_adapter(
        adapter_database,
        adapter_id="unloaded-match",
        status="unloaded",
        source_task_id=task_id,
        adapter_path="/app/output/training-1/final_model",
    )
    _add_adapter(
        adapter_database,
        adapter_id="failed-match",
        status="failed",
        source_task_id=task_id,
    )
    _add_adapter(
        adapter_database,
        adapter_id="foreign-owner-match",
        status="loaded",
        user_id="user-2",
        source_task_id=task_id,
        adapter_path="/app/output/training-1/final_model",
    )

    consumers = adapter_service_module.adapter_service.list_active_training_output_consumers(
        task_id,
        final_model_path,
        user_id=USER["user_id"],
    )

    assert {consumer["adapter_id"] for consumer in consumers} == {
        "source-match",
        "path-match-loading",
        "path-match-unloading",
    }


@pytest.mark.parametrize("status", ["loading", "loaded", "unloading"])
def test_training_delete_blocks_active_adapter_before_storage_cleanup(
    monkeypatch,
    adapter_database,
    status,
):
    task_id = f"training-{status}"
    final_model_path = f"/app/output/{task_id}/final_model"
    output_dir = f"/app/output/{task_id}"
    _add_adapter(
        adapter_database,
        adapter_id=f"adapter-{status}",
        status=status,
        source_task_id=task_id,
        adapter_path=final_model_path,
    )
    deleted = []
    cleaned = []
    releases = []
    monkeypatch.setattr(
        training_routes,
        "adapter_service",
        adapter_service_module.adapter_service,
        raising=False,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "status": "succeeded",
            "user_id": USER["user_id"],
            "output_dir": output_dir,
            "final_model_path": final_model_path,
            "process_pid": None,
            "process_status": None,
            "trained_model_registry_id": None,
        },
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_source_task",
        lambda _task_id, *, user_id: None,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        training_routes,
        "_safe_delete_path",
        lambda path, task: cleaned.append((path, task["task_id"])),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "delete_task",
        lambda requested_id: deleted.append(requested_id) or True,
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: SimpleNamespace(release=lambda: releases.append(True)),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(training_routes.delete_task(task_id, USER))

    assert exc_info.value.status_code == 409
    assert "adapter" in str(exc_info.value.detail).lower()
    assert cleaned == []
    assert deleted == []
    assert releases == [True]


def test_training_delete_allows_unloaded_adapter(monkeypatch, adapter_database):
    task_id = "training-unloaded"
    final_model_path = f"/app/output/{task_id}/final_model"
    output_dir = f"/app/output/{task_id}"
    _add_adapter(
        adapter_database,
        adapter_id="adapter-unloaded",
        status="unloaded",
        source_task_id=task_id,
        adapter_path=final_model_path,
    )
    deleted = []
    cleaned = []
    releases = []
    monkeypatch.setattr(
        training_routes,
        "adapter_service",
        adapter_service_module.adapter_service,
        raising=False,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "status": "succeeded",
            "user_id": USER["user_id"],
            "output_dir": output_dir,
            "final_model_path": final_model_path,
            "process_pid": None,
            "process_status": None,
            "trained_model_registry_id": None,
        },
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_source_task",
        lambda _task_id, *, user_id: None,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        training_routes,
        "_safe_delete_path",
        lambda path, task: cleaned.append((path, task["task_id"])),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "delete_task",
        lambda requested_id: deleted.append(requested_id) or True,
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: SimpleNamespace(release=lambda: releases.append(True)),
    )

    response = asyncio.run(training_routes.delete_task(task_id, USER))

    assert response == {"message": f"Task {task_id} deleted"}
    assert cleaned == [(output_dir, task_id)]
    assert deleted == [task_id]
    assert releases == [True]


def _registered_model(task_id: str) -> dict:
    return {
        "model_id": "model-1",
        "model_name": "registered",
        "version": "v1.0.0",
        "model_type": "embedding",
        "source_task_id": task_id,
        "model_path": f"/app/output/{task_id}/final_model",
        "base_model_path": None,
        "status": "ready",
        "is_latest": True,
        "user_id": USER["user_id"],
    }


def _patch_direct_registration(monkeypatch, task_id: str, events: list) -> None:
    model_path = f"/app/output/{task_id}/final_model"
    monkeypatch.setattr(registry_routes, "map_storage_path", lambda path: (path, None))
    monkeypatch.setattr(registry_routes, "unmap_storage_path", lambda path: path)
    monkeypatch.setattr(
        registry_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )
    monkeypatch.setattr(
        registry_routes,
        "check_idempotency",
        lambda *_args, **_kwargs: (False, None),
    )
    monkeypatch.setattr(
        registry_routes,
        "store_idempotency_response",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        registry_routes.training_task_service,
        "get_task",
        lambda requested_id: events.append(("resolve", requested_id))
        or {
            "task_id": task_id,
            "status": "succeeded",
            "is_lora": False,
            "final_model_path": model_path,
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "get_model_by_path",
        lambda _path: None,
    )


def test_direct_model_registration_holds_source_training_guard(monkeypatch):
    task_id = "training-direct-register"
    events = []
    _patch_direct_registration(monkeypatch, task_id, events)
    monkeypatch.setattr(
        registry_routes.background_task_admission_service,
        "begin_deletion",
        lambda kind, requested_id: events.append(("begin", kind, requested_id))
        or SimpleNamespace(release=lambda: events.append(("release",))),
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "register_model",
        lambda **_kwargs: events.append(("register",)) or _registered_model(task_id),
    )

    response = asyncio.run(
        registry_routes.register_model(
            registry_routes.RegisterModelRequest(
                model_name="registered",
                model_path=f"/app/output/{task_id}/final_model",
                model_type="embedding",
                source_task_id=task_id,
            ),
            USER,
            None,
        )
    )

    assert response.model_id == "model-1"
    assert events == [
        ("begin", "training", task_id),
        ("resolve", task_id),
        ("register",),
        ("release",),
    ]


def test_direct_model_registration_rejects_concurrent_training_delete(monkeypatch):
    task_id = "training-direct-register"
    events = []
    _patch_direct_registration(monkeypatch, task_id, events)
    registrations = []
    monkeypatch.setattr(
        registry_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: (_ for _ in ()).throw(
            BackgroundTaskAlreadyExecuting("being deleted")
        ),
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "register_model",
        lambda **kwargs: registrations.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            registry_routes.register_model(
                registry_routes.RegisterModelRequest(
                    model_name="registered",
                    model_path=f"/app/output/{task_id}/final_model",
                    model_type="embedding",
                    source_task_id=task_id,
                ),
                USER,
                None,
            )
        )

    assert exc_info.value.status_code == 409
    assert events == []
    assert registrations == []


def test_training_delete_blocks_dependent_two_stage_task(monkeypatch):
    task_id = "training-parent"
    output_dir = f"/app/output/{task_id}"
    cleaned = []
    deleted = []
    releases = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "status": "succeeded",
            "user_id": USER["user_id"],
            "output_dir": output_dir,
            "final_model_path": f"{output_dir}/final_model",
            "process_pid": None,
            "process_status": None,
            "trained_model_registry_id": None,
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda requested_id, requested_output: [
            {
                "task_id": "training-child",
                "parent_task_id": requested_id,
                "sft_checkpoint_path": f"{requested_output}/checkpoint-10",
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_source_task",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes.adapter_service,
        "list_active_training_output_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes,
        "_safe_delete_path",
        lambda *_args: cleaned.append(True),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "delete_task",
        lambda requested_id: deleted.append(requested_id) or True,
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: SimpleNamespace(release=lambda: releases.append(True)),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(training_routes.delete_task(task_id, USER))

    assert exc_info.value.status_code == 409
    assert "training task" in str(exc_info.value.detail).lower()
    assert cleaned == []
    assert deleted == []
    assert releases == [True]


def test_training_delete_blocks_source_less_registry_path_reference(monkeypatch):
    task_id = "training-source-less-model"
    output_dir = f"/app/output/{task_id}"
    final_model_path = f"{output_dir}/final_model"
    cleaned = []
    deleted = []
    releases = []
    task = {
        "task_id": task_id,
        "status": "succeeded",
        "user_id": USER["user_id"],
        "output_dir": output_dir,
        "final_model_path": final_model_path,
        "process_pid": None,
        "process_status": None,
        "trained_model_registry_id": None,
    }
    source_less_model = {
        **_registered_model(task_id),
        "source_task_id": None,
        "model_path": f"{final_model_path}/export",
    }
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_source_task",
        lambda *_args, **_kwargs: None,
    )

    def list_path_references(paths, *, user_id):
        normalized_paths = set(paths)
        if (
            output_dir in normalized_paths
            and final_model_path in normalized_paths
            and user_id == USER["user_id"]
        ):
            return [source_less_model]
        return []

    monkeypatch.setattr(
        training_routes.model_registry_service,
        "list_models_referencing_artifact_paths",
        list_path_references,
    )
    monkeypatch.setattr(
        training_routes.adapter_service,
        "list_active_training_output_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes,
        "_safe_delete_path",
        lambda *_args: cleaned.append(True),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "delete_task",
        lambda requested_id: deleted.append(requested_id) or True,
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: SimpleNamespace(release=lambda: releases.append(True)),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(training_routes.delete_task(task_id, USER))

    assert exc_info.value.status_code == 409
    assert cleaned == []
    assert deleted == []
    assert releases == [True]


def test_sync_training_conflict_helper_uses_registry_path_reference(monkeypatch):
    task_id = "sync-training-source-less-model"
    output_dir = f"/app/output/{task_id}"
    final_model_path = f"{output_dir}/final_model"
    task = {
        "task_id": task_id,
        "status": "succeeded",
        "user_id": USER["user_id"],
        "output_dir": output_dir,
        "final_model_path": final_model_path,
        "trained_model_registry_id": None,
    }
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_source_task",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "list_models_referencing_artifact_paths",
        lambda paths, *, user_id: [
            {
                **_registered_model(task_id),
                "source_task_id": None,
                "model_path": f"{final_model_path}/export",
            }
        ]
        if user_id == USER["user_id"] and output_dir in set(paths)
        else [],
    )
    monkeypatch.setattr(
        training_routes.adapter_service,
        "list_active_training_output_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda *_args, **_kwargs: [],
    )

    conflicts = sync_routes._list_sync_training_artifact_conflicts(
        [{"training_task_id": task_id}]
    )

    assert conflicts == [task_id]


def test_training_dependencies_ignore_foreign_registry_id(monkeypatch):
    task_id = "training-foreign-registry"
    task = {
        "task_id": task_id,
        "user_id": USER["user_id"],
        "output_dir": f"/app/output/{task_id}",
        "final_model_path": f"/app/output/{task_id}/final_model",
        "trained_model_registry_id": "foreign-model",
    }
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model",
        lambda _model_id: {
            **_registered_model(task_id),
            "model_id": "foreign-model",
            "user_id": "user-2",
        },
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_source_task",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes.adapter_service,
        "list_active_training_output_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda *_args, **_kwargs: [],
    )

    dependencies = training_routes._get_training_artifact_dependencies(task)

    assert dependencies["registered_models"] == []


def test_two_stage_creation_holds_parent_guard_until_child_is_persisted(monkeypatch):
    parent_task_id = "training-parent"
    events = []
    execution_lease = SimpleNamespace(release=lambda: None)
    monkeypatch.setattr(
        training_routes,
        "get_settings",
        lambda: SimpleNamespace(
            auth_enabled=True,
            get_task_output_dir=lambda task_id: Path("/app/output") / task_id,
        ),
    )
    monkeypatch.setattr(
        training_routes,
        "check_idempotency",
        lambda *_args, **_kwargs: (False, None),
    )
    monkeypatch.setattr(
        training_routes,
        "store_idempotency_response",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_normalize_training_config_paths",
        lambda config, _user_id: config,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_training_parent_checkpoint",
        lambda requested_parent, *_args: events.append(
            ("validate", requested_parent)
        ),
    )

    def begin_deletion(kind, requested_id):
        events.append(("begin", kind, requested_id))
        return SimpleNamespace(release=lambda: events.append(("release",)))

    def admit_execution(
        kind,
        task_id,
        admission_user_id,
        operation,
        *args,
        **kwargs,
    ):
        events.append(("admit", kind, task_id, admission_user_id))
        return operation(*args, **kwargs), execution_lease

    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        begin_deletion,
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "admit_execution",
        admit_execution,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "create_task",
        lambda **_kwargs: events.append(("persist",))
        or {"task_id": "training-child", "task_name": "child"},
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_output_dir",
        lambda *_args, **_kwargs: events.append(("update-output",)) or True,
    )
    background_tasks = BackgroundTasks()

    response = asyncio.run(
        training_routes.create_training_task(
            training_routes.TrainingRequest(
                task_name="child",
                model_type="llm",
                training_method="dpo",
                base_model_path="/app/models/base",
                datasets=[{"path": "/app/data/train.jsonl", "split": "train"}],
                parent_task_id=parent_task_id,
                sft_checkpoint_path=(
                    f"/app/output/{parent_task_id}/checkpoint-10"
                ),
            ),
            background_tasks,
            USER,
            None,
        )
    )

    assert response.task_id == "training-child"
    assert events == [
        ("begin", "training", parent_task_id),
        ("validate", parent_task_id),
        ("admit", "training", None, USER["user_id"]),
        ("persist",),
        ("release",),
        ("update-output",),
    ]
    assert len(background_tasks.tasks) == 1


def test_managed_checkpoint_requires_parent_when_auth_is_disabled(monkeypatch):
    monkeypatch.setattr(
        training_routes,
        "_resource_ownership_required",
        lambda _user_id: False,
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_training_parent_checkpoint(
            None,
            "/app/output/parent/checkpoint-10",
            {"user_id": "anonymous"},
        )

    assert exc_info.value.status_code == 400
    assert "parent_task_id" in str(exc_info.value.detail)


def test_manual_adapter_retry_holds_training_guard_until_worker_finishes(
    monkeypatch,
):
    sync_task_id = "sync-1"
    training_task_id = "training-1"
    events = []
    guard_active = False

    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": sync_task_id,
            "user_id": USER["user_id"],
            "status": "idle",
        },
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _task_id: {
            "task_id": sync_task_id,
            "training_task_id": training_task_id,
            "user_id": USER["user_id"],
            "status": "adapter_load_failed",
        },
        raising=False,
    )

    def get_training_task(_task_id):
        assert guard_active is True
        events.append(("read",))
        return {
            "task_id": training_task_id,
            "user_id": USER["user_id"],
            "final_model_path": "/app/output/training-1/final_model",
            "model_registry_id": "model-1",
        }

    monkeypatch.setattr(
        training_task_service,
        "get_task",
        get_training_task,
        raising=False,
    )

    def begin_deletion(kind, requested_task_id):
        nonlocal guard_active
        events.append(("begin", kind, requested_task_id))
        guard_active = True

        def release():
            nonlocal guard_active
            guard_active = False
            events.append(("release",))

        return SimpleNamespace(release=release)

    def run_with_deletion_guard(guard, operation, *args):
        try:
            return operation(*args)
        finally:
            guard.release()

    monkeypatch.setattr(
        sync_routes,
        "background_task_admission_service",
        SimpleNamespace(
            begin_deletion=begin_deletion,
            run_with_deletion_guard=run_with_deletion_guard,
        ),
        raising=False,
    )

    def load_adapter(*args):
        assert guard_active is True
        events.append(("load", *args))

    monkeypatch.setattr(
        post_training_handler,
        "load_adapter_for_training",
        load_adapter,
    )

    class ImmediateThread:
        def __init__(self, *, target, args, daemon):
            assert daemon is True
            self._target = target
            self._args = args

        def start(self):
            events.append(("start",))
            self._target(*self._args)

    monkeypatch.setattr("threading.Thread", ImmediateThread)

    response = asyncio.run(
        sync_routes.retry_adapter_load(
            sync_task_id,
            training_task_id,
            True,
            USER,
        )
    )

    assert response == {"message": "Adapter loading retry started"}
    assert events == [
        ("begin", "training", training_task_id),
        ("read",),
        ("start",),
        (
            "load",
            sync_task_id,
            training_task_id,
            "/app/output/training-1/final_model",
            "model-1",
            True,
        ),
        ("release",),
    ]
    assert guard_active is False


def test_manual_adapter_retry_rejects_concurrent_training_deletion(monkeypatch):
    sync_task_id = "sync-1"
    training_task_id = "training-1"
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": sync_task_id,
            "user_id": USER["user_id"],
            "status": "idle",
        },
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _task_id: {
            "task_id": sync_task_id,
            "training_task_id": training_task_id,
            "user_id": USER["user_id"],
            "status": "adapter_load_failed",
        },
        raising=False,
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda _task_id: pytest.fail(
            "adapter path must not be read without the training guard"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        sync_routes,
        "background_task_admission_service",
        SimpleNamespace(
            begin_deletion=lambda *_args: (_ for _ in ()).throw(
                BackgroundTaskAlreadyExecuting("being deleted")
            )
        ),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.retry_adapter_load(
                sync_task_id,
                training_task_id,
                True,
                USER,
            )
        )

    assert exc_info.value.status_code == 409


def test_cancelled_manual_adapter_retry_releases_acquired_guard_before_unlock(
    monkeypatch,
):
    sync_task_id = "sync-cancelled-retry"
    training_task_id = "training-cancelled-retry"
    acquisition_started = threading.Event()
    finish_acquisition = threading.Event()
    releases = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": sync_task_id,
            "user_id": USER["user_id"],
            "status": "idle",
        },
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _task_id: {
            "task_id": sync_task_id,
            "training_task_id": training_task_id,
            "user_id": USER["user_id"],
            "status": "adapter_load_failed",
        },
        raising=False,
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda _task_id: pytest.fail("cancelled retry must not read artifacts"),
        raising=False,
    )

    def begin_deletion(_kind, _task_id):
        acquisition_started.set()
        assert finish_acquisition.wait(5)
        return SimpleNamespace(release=lambda: releases.append(True))

    monkeypatch.setattr(
        sync_routes,
        "background_task_admission_service",
        SimpleNamespace(begin_deletion=begin_deletion),
        raising=False,
    )

    async def exercise():
        manager = SyncManager()
        monkeypatch.setattr(sync_manager_module, "sync_manager", manager)
        request = asyncio.create_task(
            sync_routes.retry_adapter_load(
                sync_task_id,
                training_task_id,
                True,
                USER,
            )
        )
        competitor_entered = asyncio.Event()

        async def compete_for_lock():
            async with manager.task_operation_lock(sync_task_id):
                competitor_entered.set()

        competitor = None
        try:
            assert await asyncio.to_thread(acquisition_started.wait, 2)
            competitor = asyncio.create_task(compete_for_lock())
            await asyncio.sleep(0)
            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()
            assert not competitor_entered.is_set()
            assert releases == []

            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()
            assert not competitor_entered.is_set()
            assert releases == []

            finish_acquisition.set()
            with pytest.raises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(competitor, timeout=2)
        finally:
            finish_acquisition.set()
            if not request.done():
                request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            if competitor is not None and not competitor.done():
                competitor.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await competitor

    asyncio.run(exercise())

    assert releases == [True]
