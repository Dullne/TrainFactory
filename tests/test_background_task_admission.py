"""Admission-control regressions for long-running API background jobs."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def external_inference_catalog(monkeypatch):
    from train_factory.storage.services import inference_authorization_service

    monkeypatch.setattr(
        inference_authorization_service, "registered_shared_models_for_endpoint",
        lambda *_args: (False, set()),
    )
import yaml
from fastapi import BackgroundTasks, HTTPException
from sqlmodel import Session, SQLModel, create_engine

from train_factory.api.routes import (
    deep_evaluation_routes,
    evaluation_routes,
    generation_routes,
    registry_routes,
    sync_routes,
    training_routes,
)
from train_factory.api import server
from train_factory.config.settings import Settings
from train_factory.storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    BackgroundTaskAdmissionService,
    BackgroundTaskCapacityExceeded,
)
from train_factory.storage.services.dataset_service import (
    DatasetConsumptionUnavailableError,
)
from train_factory.storage.services import background_task_admission_service as admission_module
from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.training_task_entity import TrainingTaskDB


USER = {"user_id": "user-1", "username": "user", "role": "user"}


def _request() -> evaluation_routes.CreateEvaluationRequest:
    return evaluation_routes.CreateEvaluationRequest(
        model_configs=[
            evaluation_routes.ModelConfig(
                endpoint="https://example.com/v1/rerank",
                name="model",
            )
        ],
        dataset_configs=[
            evaluation_routes.DatasetConfig(type="mteb", name="T2Reranking")
        ],
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    (
        ((8, 0), "global"),
        ((1, 2), "per-user"),
    ),
)
def test_admission_rejects_at_global_and_per_user_limits(
    monkeypatch,
    counts,
    message,
):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_counts", lambda _user_id: counts)
    called = []

    with pytest.raises(BackgroundTaskCapacityExceeded, match=message):
        service.admit("user-1", lambda: called.append(True))

    assert called == []


def test_admission_executes_database_transition_while_holding_slot(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_counts", lambda _user_id: (0, 0))

    result = service.admit("user-1", lambda value: {"value": value}, 42)

    assert result == {"value": 42}


def test_execution_lease_blocks_restart_until_worker_actually_exits(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})

    result, lease = service.admit_execution(
        "evaluation",
        None,
        "user-1",
        lambda: {"task_id": "task-1"},
    )

    assert result["task_id"] == "task-1"
    assert service.is_executing("evaluation", "task-1") is True
    with pytest.raises(BackgroundTaskAlreadyExecuting):
        service.begin_deletion("evaluation", "task-1")
    with pytest.raises(BackgroundTaskAlreadyExecuting, match="already executing"):
        service.admit_execution(
            "evaluation",
            "task-1",
            "user-1",
            lambda: True,
        )

    lease.release()
    assert service.is_executing("evaluation", "task-1") is False
    claimed, replacement = service.admit_execution(
        "evaluation",
        "task-1",
        "user-1",
        lambda: True,
    )
    assert claimed is True
    replacement.release()
    deletion_guard = service.begin_deletion("evaluation", "task-1")
    with pytest.raises(BackgroundTaskAlreadyExecuting):
        service.admit_execution(
            "evaluation",
            "task-1",
            "user-1",
            lambda: True,
        )
    deletion_guard.release()


def test_execution_lease_supports_training_workers(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})

    result, lease = service.admit_execution(
        "training",
        None,
        "user-1",
        lambda: {"task_id": "training-1"},
    )

    assert result["task_id"] == "training-1"
    assert service.is_executing("training", "training-1") is True
    lease.release()
    assert service.is_executing("training", "training-1") is False


def test_admit_execution_allows_operation_task_id_keyword(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})

    result, lease = service.admit_execution(
        "training",
        "sync-training-1",
        "user-1",
        lambda *, task_id: {"task_id": task_id},
        task_id="sync-training-1",
    )

    assert result == {"task_id": "sync-training-1"}
    lease.release()


def test_training_delete_rejects_pending_task_before_deletion_guard(monkeypatch):
    deleted = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "training-pending",
            "status": "pending",
            "user_id": USER["user_id"],
            "process_pid": None,
            "process_status": None,
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "delete_task",
        lambda task_id: deleted.append(task_id) or True,
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: pytest.fail("pending task reached the deletion guard"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(training_routes.delete_task("training-pending", USER))

    assert exc_info.value.status_code == 400
    assert deleted == []


def test_training_delete_maps_live_worker_to_409_without_deleting(monkeypatch):
    deleted = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "training-failed-live",
            "status": "failed",
            "user_id": USER["user_id"],
            "process_pid": None,
            "process_status": None,
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "delete_task",
        lambda task_id: deleted.append(task_id) or True,
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: (_ for _ in ()).throw(
            BackgroundTaskAlreadyExecuting("still executing")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(training_routes.delete_task("training-failed-live", USER))

    assert exc_info.value.status_code == 409
    assert deleted == []


def test_training_delete_releases_deletion_guard_after_success(monkeypatch):
    guard_releases = []
    guard_calls = []
    deleted = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "training-failed",
            "status": "failed",
            "user_id": USER["user_id"],
            "process_pid": None,
            "process_status": None,
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "delete_task",
        lambda task_id: deleted.append(task_id) or True,
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_source_task",
        lambda _task_id, *, user_id: None,
    )
    monkeypatch.setattr(
        training_routes.adapter_service,
        "list_active_training_output_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda *_args: [],
    )

    def begin_deletion(kind, task_id):
        guard_calls.append((kind, task_id))
        return SimpleNamespace(release=lambda: guard_releases.append(True))

    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        begin_deletion,
    )

    response = asyncio.run(training_routes.delete_task("training-failed", USER))

    assert response == {"message": "Task training-failed deleted"}
    assert guard_calls == [("training", "training-failed")]
    assert deleted == ["training-failed"]
    assert guard_releases == [True]


def test_training_delete_preserves_registered_model_artifacts(monkeypatch):
    task_id = "training-registered"
    output_dir = "/app/output/training-registered"
    deleted = []
    cleaned = []
    releases = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "status": "succeeded",
            "user_id": USER["user_id"],
            "output_dir": output_dir,
            "process_pid": None,
            "process_status": None,
            "trained_model_registry_id": "model-1",
        },
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "model_path": f"{output_dir}/final_model",
            "source_task_id": task_id,
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        training_routes.adapter_service,
        "list_active_training_output_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "delete_task",
        lambda requested_id: deleted.append(requested_id) or True,
    )
    monkeypatch.setattr(
        training_routes,
        "_safe_delete_path",
        lambda path, task: cleaned.append((path, task["task_id"])),
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: SimpleNamespace(release=lambda: releases.append(True)),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(training_routes.delete_task(task_id, USER))

    assert exc_info.value.status_code == 409
    assert deleted == []
    assert cleaned == []
    assert releases == [True]


def test_training_delete_keeps_record_when_output_cleanup_fails(monkeypatch):
    task_id = "training-cleanup-failure"
    deleted = []
    releases = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "status": "failed",
            "user_id": USER["user_id"],
            "output_dir": f"/app/output/{task_id}",
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
        training_routes.model_registry_service,
        "list_models_referencing_artifact_paths",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes.adapter_service,
        "list_active_training_output_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "list_artifact_consumers",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        training_routes,
        "_safe_delete_path",
        lambda *_args: (_ for _ in ()).throw(OSError("unlink denied")),
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

    with pytest.raises(OSError, match="unlink denied"):
        asyncio.run(training_routes.delete_task(task_id, USER))

    assert deleted == []
    assert releases == [True]


def test_register_from_training_task_rejects_concurrent_deletion(monkeypatch):
    registered = []
    monkeypatch.setattr(
        registry_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "training-deleting",
            "status": "succeeded",
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "register_from_task",
        lambda **kwargs: registered.append(kwargs),
    )
    monkeypatch.setattr(
        registry_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: (_ for _ in ()).throw(
            BackgroundTaskAlreadyExecuting("being deleted")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            registry_routes.register_from_task(
                "training-deleting",
                registry_routes.RegisterFromTaskRequest(model_name="model"),
                USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert registered == []


def test_standard_evaluation_delete_rejects_live_cancelled_worker(monkeypatch):
    deleted = []
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "evaluation-1",
            "status": "cancelled",
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "delete_task",
        lambda task_id: deleted.append(task_id) or True,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: (_ for _ in ()).throw(
            BackgroundTaskAlreadyExecuting("still executing")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.delete_evaluation_task("evaluation-1", USER)
        )

    assert exc_info.value.status_code == 409
    assert deleted == []


def test_deep_evaluation_delete_rejects_live_cancelled_worker(monkeypatch):
    deleted = []
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "deep-1",
            "status": "cancelled",
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "delete_task",
        lambda task_id: deleted.append(task_id) or True,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: (_ for _ in ()).throw(
            BackgroundTaskAlreadyExecuting("still executing")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.delete_deep_evaluation_task("deep-1", USER)
        )

    assert exc_info.value.status_code == 409
    assert deleted == []


def test_generation_delete_rejects_live_stopped_worker(monkeypatch):
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: {
            "task_id": "generation-1",
            "status": "stopped",
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        generation_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: (_ for _ in ()).throw(
            BackgroundTaskAlreadyExecuting("still executing")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.delete_task(
                "generation-1",
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409


def test_execution_wrapper_releases_lease_after_failure(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})
    _, lease = service.admit_execution(
        "evaluation",
        None,
        "user-1",
        lambda: {"task_id": "task-1"},
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        service.run_sync(lease, lambda: (_ for _ in ()).throw(RuntimeError("worker failed")))

    _, replacement = service.admit_execution(
        "evaluation",
        "task-1",
        "user-1",
        lambda: True,
    )
    replacement.release()


def test_sync_manager_launch_releases_generation_execution_lease(monkeypatch):
    from train_factory.api.routes import generation_routes as generation_module
    from train_factory.storage.entities.generation_task_entity import (
        GenerationStatus,
    )
    from train_factory.sync.sync_manager import SyncManager

    calls = []
    releases = []
    admission = BackgroundTaskAdmissionService(global_limit=1, per_user_limit=1)
    monkeypatch.setattr(admission_module, "background_task_admission_service", admission)

    async def fake_generation(
        task_id,
        config,
        *,
        expected_run_token,
        sync_handoff,
    ):
        assert sync_handoff["gen_task_id"] == task_id
        assert sync_handoff["generation_run_token"] == expected_run_token
        calls.append((task_id, config, expected_run_token))

    monkeypatch.setattr(
        generation_module,
        "_run_generation_task",
        fake_generation,
    )
    monkeypatch.setattr(
        generation_module.generation_task_service,
        "get_task_raw",
        lambda _task_id: {
            "status": GenerationStatus.PENDING,
            "run_token": "sync-generation-token",
        },
    )
    lease = SimpleNamespace(release=lambda: releases.append(True))

    async def launch():
        SyncManager()._launch_generation(
            {
                "gen_task_id": "sync-generation-1",
                "generation_run_token": "sync-generation-token",
                "pipeline_config": "config",
                "execution_lease": lease,
            }
        )
        # Completion belongs to a dedicated worker pool, so two event-loop
        # ticks no longer imply the pipeline and its lease have finished.
        launched = [task for task in asyncio.all_tasks()
                    if task.get_name() == "gen-pipeline-sync-gen"]
        assert len(launched) == 1
        await asyncio.wait_for(launched[0], 3)

    try:
        asyncio.run(launch())
    finally:
        admission.shutdown_async_workers()

    assert calls == [
        ("sync-generation-1", "config", "sync-generation-token")
    ]
    assert releases == [True]


def test_cancelled_sync_cycle_finalizes_late_generation_handoff(monkeypatch):
    import concurrent.futures
    import threading

    from train_factory.api.routes import generation_routes as generation_module
    from train_factory.sync.sync_manager import SyncManager

    started = threading.Event()
    finish = threading.Event()
    releases = []

    async def fake_generation(task_id, config):
        pytest.fail(f"cancelled cycle launched {task_id} with {config}")

    monkeypatch.setattr(generation_module, "_run_generation_task", fake_generation)
    lease = SimpleNamespace(release=lambda: releases.append(True))

    def cycle(_config):
        started.set()
        assert finish.wait(2)
        return {
            "gen_task_id": "cancelled-handoff-1",
            "pipeline_config": "config",
            "execution_lease": lease,
        }

    manager = SyncManager()
    manager._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(
        manager,
        "_finalize_unlaunched_generation",
        lambda result, _reason: result["execution_lease"].release(),
    )

    async def exercise():
        task = asyncio.create_task(
            manager._run_cycle_with_handoff(
                asyncio.get_running_loop(),
                cycle,
                {},
            )
        )
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(20):
            if releases:
                break
            await asyncio.sleep(0.01)

    try:
        asyncio.run(exercise())
    finally:
        manager._executor.shutdown(wait=True)

    assert releases == [True]


def test_sync_launch_failure_releases_lease_and_fails_pending_task(monkeypatch):
    import importlib

    manager_module = importlib.import_module("train_factory.sync.sync_manager")
    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    updates = []
    recoveries = []
    monkeypatch.setattr(
        generation_module.generation_task_service,
        "update_status",
        lambda *args, **kwargs: updates.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        generation_module.generation_task_service,
        "get_task_raw",
        lambda _task_id: {
            "status": "pending",
            "run_token": "launch-failure-token",
        },
    )
    monkeypatch.setattr(
        sync_service_module.external_sync_service,
        "fail_generation_and_restore_batches",
        lambda task_id, reason: recoveries.append((task_id, reason)) or {
            "recovered": True,
        },
    )
    monkeypatch.setattr(
        manager_module.asyncio,
        "create_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("event loop rejected task")
        ),
    )
    releases = []
    lease = SimpleNamespace(release=lambda: releases.append(True))

    with pytest.raises(RuntimeError, match="event loop rejected task"):
        manager_module.SyncManager()._launch_generation(
            {
                "gen_task_id": "launch-failure-1",
                "generation_run_token": "launch-failure-token",
                "pipeline_config": "config",
                "execution_lease": lease,
            }
        )

    assert releases == [True]
    assert updates[0][0][0] == "launch-failure-1"
    assert updates[0][0][1] == "failed"
    assert updates[0][1]["expected_run_token"] == "launch-failure-token"
    assert recoveries == [
        (
            "launch-failure-1",
            "Sync generation could not be launched on the main event loop",
        )
    ]


def test_old_sync_worker_cleanup_does_not_remove_replacement_worker():
    from train_factory.sync.sync_manager import SyncManager

    manager = SyncManager()
    manager._running = False
    replacement = SimpleNamespace(done=lambda: False)
    manager._workers["sync-1"] = replacement

    asyncio.run(manager._run_worker("sync-1"))

    assert manager._workers["sync-1"] is replacement


@pytest.mark.parametrize(
    ("method_name", "cycle_name"),
    (
        ("run_once", "_run_sync_cycle"),
        ("trigger_generation", "_run_generation_cycle"),
    ),
)
def test_orphan_generating_manual_entry_recovers_and_runs_cycle(
    monkeypatch,
    method_name,
    cycle_name,
):
    import concurrent.futures
    import importlib

    from train_factory.sync import sync_manager as manager_module

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    state = {"status": "generating"}
    events = []

    def get_task_raw(task_id):
        events.append(("read", task_id, state["status"]))
        return {
            "task_id": "sync-1",
            "is_active": True,
            "status": state["status"],
            "sync_interval_seconds": 1,
        }

    def reset_generating(task_id):
        events.append(("reset", task_id))
        state["status"] = "idle"
        return True

    service = SimpleNamespace(
        get_task_raw=get_task_raw,
        reset_generating_task_if_no_pending_generation=reset_generating,
    )
    monkeypatch.setattr(sync_module, "external_sync_service", service)
    monkeypatch.setattr(
        manager_module,
        "_reconcile_terminal_sync_generation",
        lambda task_id: events.append(("reconcile", task_id)),
        raising=False,
    )

    def run_cycle(config):
        events.append(("cycle", config["task_id"], config["status"]))

    monkeypatch.setattr(manager_module, cycle_name, run_cycle)
    manager = manager_module.SyncManager()
    manager._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        asyncio.run(getattr(manager, method_name)("sync-1"))
    finally:
        manager._executor.shutdown(wait=True)

    assert events == [
        ("read", "sync-1", "generating"),
        ("reconcile", "sync-1"),
        ("reset", "sync-1"),
        ("read", "sync-1", "idle"),
        ("cycle", "sync-1", "idle"),
    ]


def test_generating_sync_worker_reconciles_resets_and_runs_cycle(monkeypatch):
    import concurrent.futures
    import importlib

    from train_factory.sync import sync_manager as manager_module

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    state = {"status": "generating"}
    events = []

    def get_task_raw(task_id):
        events.append(("read", task_id, state["status"]))
        return {
            "task_id": task_id,
            "is_active": True,
            "status": state["status"],
            "sync_interval_seconds": 1,
        }

    def reset_generating(task_id):
        events.append(("reset", task_id))
        state["status"] = "idle"
        return True

    service = SimpleNamespace(
        get_task_raw=get_task_raw,
        reset_generating_task_if_no_pending_generation=reset_generating,
    )
    monkeypatch.setattr(sync_module, "external_sync_service", service)
    monkeypatch.setattr(
        manager_module,
        "_reconcile_terminal_sync_generation",
        lambda task_id: events.append(("reconcile", task_id)),
    )
    monkeypatch.setattr(
        manager_module,
        "_run_sync_cycle",
        lambda config: events.append(
            ("cycle", config["task_id"], config["status"])
        ),
    )
    manager = manager_module.SyncManager()
    manager._running = True
    manager._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def finish_cycle(_interval):
        manager._running = False

    monkeypatch.setattr(manager_module.asyncio, "sleep", finish_cycle)

    try:
        asyncio.run(manager._run_worker("sync-1"))
    finally:
        manager._executor.shutdown(wait=True)

    assert events == [
        ("read", "sync-1", "generating"),
        ("read", "sync-1", "generating"),
        ("reconcile", "sync-1"),
        ("reset", "sync-1"),
        ("read", "sync-1", "idle"),
        ("cycle", "sync-1", "idle"),
    ]


def test_generating_sync_worker_preserves_pending_generation(monkeypatch):
    import concurrent.futures
    import importlib

    from train_factory.sync import sync_manager as manager_module

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    events = []

    def get_task_raw(task_id):
        events.append(("read", task_id))
        return {
            "task_id": task_id,
            "is_active": True,
            "status": "generating",
            "sync_interval_seconds": 1,
        }

    service = SimpleNamespace(
        get_task_raw=get_task_raw,
        reset_generating_task_if_no_pending_generation=lambda task_id: (
            events.append(("reset", task_id)) or False
        ),
    )
    monkeypatch.setattr(sync_module, "external_sync_service", service)
    monkeypatch.setattr(
        manager_module,
        "_reconcile_terminal_sync_generation",
        lambda task_id: events.append(("reconcile", task_id)),
    )
    monkeypatch.setattr(
        manager_module,
        "_run_sync_cycle",
        lambda _config: pytest.fail("pending generation must skip the sync cycle"),
    )
    manager = manager_module.SyncManager()
    manager._running = True
    manager._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def finish_cycle(_interval):
        manager._running = False

    monkeypatch.setattr(manager_module.asyncio, "sleep", finish_cycle)

    try:
        asyncio.run(manager._run_worker("sync-pending"))
    finally:
        manager._executor.shutdown(wait=True)

    assert events == [
        ("read", "sync-pending"),
        ("read", "sync-pending"),
        ("reconcile", "sync-pending"),
        ("reset", "sync-pending"),
        ("read", "sync-pending"),
    ]


@pytest.mark.parametrize(
    ("failure_step", "error_type"),
    (("reconcile", "RuntimeError"), ("reset", "LookupError")),
)
def test_generating_sync_worker_preserves_state_when_reconciliation_fails(
    monkeypatch,
    caplog,
    failure_step,
    error_type,
):
    import concurrent.futures
    import importlib
    import logging

    from train_factory.sync import sync_manager as manager_module

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    secret = "database-secret-detail"
    state = {"status": "generating"}
    events = []
    error_updates = []
    ordinary_updates = []

    def get_task_raw(task_id):
        events.append(("read", task_id))
        return {
            "task_id": task_id,
            "is_active": True,
            "status": state["status"],
            "sync_interval_seconds": 1,
        }

    def reconcile(task_id):
        events.append(("reconcile", task_id))
        if failure_step == "reconcile":
            raise RuntimeError(secret)

    def reset_generating(task_id):
        events.append(("reset", task_id))
        if failure_step == "reset":
            raise LookupError(secret)
        return False

    service = SimpleNamespace(
        get_task_raw=get_task_raw,
        reset_generating_task_if_no_pending_generation=reset_generating,
        update_task=lambda *args, **kwargs: ordinary_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(sync_module, "external_sync_service", service)
    monkeypatch.setattr(manager_module, "_reconcile_terminal_sync_generation", reconcile)
    monkeypatch.setattr(
        manager_module,
        "_set_task_error",
        lambda *args, **kwargs: error_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        manager_module,
        "_run_sync_cycle",
        lambda _config: pytest.fail("failed reconciliation must skip the sync cycle"),
    )
    manager = manager_module.SyncManager()
    manager._running = True
    manager._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def finish_cycle(_interval):
        manager._running = False

    monkeypatch.setattr(manager_module.asyncio, "sleep", finish_cycle)
    caplog.set_level(logging.WARNING, logger=manager_module.__name__)

    try:
        asyncio.run(manager._run_worker("sync-secret-task"))
    finally:
        manager._executor.shutdown(wait=True)

    assert state["status"] == "generating"
    assert error_updates == []
    assert ordinary_updates == []
    assert secret not in caplog.text
    assert error_type in caplog.text
    assert "sync-sec" in caplog.text


@pytest.mark.parametrize("second_method", ("run_once", "trigger_generation"))
def test_manual_admission_rejects_concurrent_same_task_without_queueing(
    monkeypatch,
    second_method,
):
    import concurrent.futures
    import importlib

    from train_factory.sync import sync_manager as manager_module

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    service = SimpleNamespace(
        get_task_raw=lambda task_id: {
            "task_id": task_id,
            "is_active": True,
            "status": "idle",
        }
    )
    monkeypatch.setattr(sync_module, "external_sync_service", service)
    manager = manager_module.SyncManager()
    manager._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    started = asyncio.Event()
    release = asyncio.Event()
    cycle_calls = []

    async def blocking_cycle(_loop, operation, config):
        cycle_calls.append((operation, config["task_id"]))
        started.set()
        await release.wait()

    monkeypatch.setattr(manager, "_run_cycle_with_handoff", blocking_cycle)

    async def exercise():
        first = asyncio.create_task(manager.run_once("sync-shared"))
        await asyncio.wait_for(started.wait(), timeout=1)
        try:
            with pytest.raises(manager_module.SyncTaskBusyError):
                await asyncio.wait_for(
                    getattr(manager, second_method)("sync-shared"),
                    timeout=0.2,
                )
        finally:
            release.set()
            await first

    try:
        asyncio.run(exercise())
    finally:
        manager._executor.shutdown(wait=True)

    assert len(cycle_calls) == 1


@pytest.mark.parametrize("failure", ("exception", "cancelled"))
def test_manual_admission_releases_claim_after_cycle_failure(monkeypatch, failure):
    import concurrent.futures
    import importlib

    from train_factory.sync import sync_manager as manager_module

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    service = SimpleNamespace(
        get_task_raw=lambda task_id: {
            "task_id": task_id,
            "is_active": True,
            "status": "idle",
        }
    )
    monkeypatch.setattr(sync_module, "external_sync_service", service)
    manager = manager_module.SyncManager()
    manager._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    attempts = []

    async def fail_then_succeed(_loop, _operation, config):
        attempts.append(config["task_id"])
        if len(attempts) == 1:
            if failure == "cancelled":
                raise asyncio.CancelledError()
            raise RuntimeError("cycle failed")

    monkeypatch.setattr(manager, "_run_cycle_with_handoff", fail_then_succeed)

    async def exercise():
        expected = asyncio.CancelledError if failure == "cancelled" else RuntimeError
        with pytest.raises(expected):
            await manager.run_once("sync-release")
        assert "sync-release" not in manager._manual_task_ids
        await manager.run_once("sync-release")

    try:
        asyncio.run(exercise())
    finally:
        manager._executor.shutdown(wait=True)

    assert attempts == ["sync-release", "sync-release"]
    assert "sync-release" not in manager._manual_task_ids


def test_manual_admission_cancellation_waits_for_sync_reconciliation(monkeypatch):
    import concurrent.futures
    import importlib
    import threading

    from train_factory.sync import sync_manager as manager_module

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    task_id = "sync-cancel-reconcile"
    service = SimpleNamespace(
        get_task_raw=lambda requested_task_id: {
            "task_id": requested_task_id,
            "is_active": True,
            "status": "generating",
        }
    )
    monkeypatch.setattr(sync_module, "external_sync_service", service)
    started = threading.Event()
    release = threading.Event()
    reconcile_calls = []
    cycle_calls = []

    def blocking_reconciliation(_service, requested_task_id):
        reconcile_calls.append(requested_task_id)
        if len(reconcile_calls) == 1:
            started.set()
            assert release.wait(5)
        return {
            "task_id": requested_task_id,
            "is_active": True,
            "status": "idle",
        }

    monkeypatch.setattr(
        manager_module,
        "_reconcile_sync_generation_state",
        blocking_reconciliation,
    )
    monkeypatch.setattr(
        manager_module,
        "_run_sync_cycle",
        lambda config: cycle_calls.append(config["task_id"]),
    )
    manager = manager_module.SyncManager()
    manager._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def exercise():
        operation = asyncio.create_task(manager.run_once(task_id))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task_lock = manager._get_task_lock(task_id)
            assert task_id in manager._manual_task_ids
            assert task_lock.locked()

            operation.cancel()
            await asyncio.sleep(0)

            assert not operation.done()
            assert task_id in manager._manual_task_ids
            assert task_lock.locked()
            with pytest.raises(manager_module.SyncTaskBusyError):
                await manager.trigger_generation(task_id)

            release.set()
            with pytest.raises(asyncio.CancelledError):
                await operation

            assert task_id not in manager._manual_task_ids
            assert not task_lock.locked()
            await manager.run_once(task_id)
        finally:
            release.set()
            if not operation.done():
                operation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await operation

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        manager._executor.shutdown(wait=True)

    assert reconcile_calls == [task_id, task_id]
    assert cycle_calls == [task_id]
    assert task_id not in manager._manual_task_ids


def test_sync_generation_creation_returns_admitted_execution_lease(
    monkeypatch,
    tmp_path,
):
    import importlib

    from train_factory.sync import sync_worker

    asset_module = importlib.import_module(
        "train_factory.storage.services.dataset_asset_service"
    )
    lineage_module = importlib.import_module(
        "train_factory.storage.services.dataset_lineage_service"
    )
    dataset_module = importlib.import_module(
        "train_factory.storage.services.dataset_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )

    input_path = tmp_path / "merged.jsonl"
    input_path.write_text('{"content":"document"}\n', encoding="utf-8")
    monkeypatch.setenv("GENERATION_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(sync_worker, "_restore_stopped_queued_batches", lambda _id: 0)
    monkeypatch.setattr(
        sync_worker,
        "_select_generation_batches",
        lambda _config, batches: batches,
    )
    monkeypatch.setattr(
        sync_worker,
        "_merge_batches",
        lambda merge_config, _batches: merge_config["_generation_merged_path"],
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )

    sync_calls = []
    fake_sync_service = SimpleNamespace(
        update_task=lambda *args, **kwargs: sync_calls.append(
            ("update_task", args, kwargs)
        ),
        get_pending_batches=lambda _task_id: [
            {"batch_id": "batch-1", "record_count": 1}
        ],
        has_pending_generation=lambda _task_id: False,
        get_all_completed_generation_datasets=lambda _task_id: [],
        create_generation_and_claim_batches=lambda **kwargs: sync_calls.append(
            ("create_generation_and_claim_batches", kwargs)
        )
        or {"generation_task_id": kwargs["generation_task_id"]},
        fail_generation_and_restore_batches=lambda *_args: {
            "tracking_found": False,
            "recovered": False,
        },
    )
    monkeypatch.setattr(
        sync_module,
        "external_sync_service",
        fake_sync_service,
    )
    monkeypatch.setattr(
        dataset_module.dataset_service,
        "create_dataset",
        lambda **kwargs: {"dataset_id": kwargs["dataset_id"]},
    )
    monkeypatch.setattr(
        lineage_module.dataset_lineage_service,
        "create_edge",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "create_asset",
        lambda **_kwargs: None,
    )

    generation_calls = []
    generation_run_token = "sync-generation-creation-token"
    fake_generation_service = SimpleNamespace(
        create_task=lambda **kwargs: generation_calls.append(kwargs)
        or {"task_id": kwargs["task_id"]},
        get_task_raw=lambda task_id: {
            "task_id": task_id,
            "status": GenerationStatus.PENDING,
            "run_token": generation_run_token,
        },
        set_output=lambda *_args, **_kwargs: True,
        update_status=lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        generation_module,
        "generation_task_service",
        fake_generation_service,
    )

    lease = SimpleNamespace(release=lambda: None)
    admissions = []

    def admit(kind, task_id, admission_user_id, operation, *args, **kwargs):
        admissions.append((kind, task_id, admission_user_id))
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(
        admission_module.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        admission_module.background_task_admission_service,
        "assert_capacity_available",
        lambda _user_id: None,
    )

    result = asyncio.run(
        sync_worker._trigger_generation(
            {
                "task_id": "sync-1",
                "user_id": "user-1",
                "generation_mode": "qa_extraction",
                "generation_config": {},
            }
        )
    )

    assert admissions == [("generation", None, "user-1")]
    planned_generation_id = generation_calls[0]["task_id"]
    raw_dataset_id = generation_calls[0]["source_dataset_id"]
    assert result["execution_lease"] is lease
    assert result["gen_task_id"] == planned_generation_id
    assert result["generation_run_token"] == generation_run_token
    assert result["sync_task_id"] == "sync-1"
    assert result["user_id"] == "user-1"
    assert result["raw_dataset_id"] == raw_dataset_id
    assert Path(result["merged_path"]).name.startswith("merged_")
    claim = next(
        call for call in sync_calls if call[0] == "create_generation_and_claim_batches"
    )
    assert claim[1]["input_batch_ids"] == ["batch-1"]
    assert claim[1]["dataset_id"] == raw_dataset_id


def test_admission_counts_active_jobs_across_all_task_tables(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'admission.db'}")
    SQLModel.metadata.create_all(
        engine,
        tables=[
            EvaluationTaskDB.__table__,
            GenerationTaskDB.__table__,
            TrainingTaskDB.__table__,
        ],
    )
    with Session(engine) as session:
        session.add_all(
            [
                EvaluationTaskDB(user_id="user-1", status="pending"),
                EvaluationTaskDB(user_id="user-2", status="running"),
                EvaluationTaskDB(user_id="user-1", status="failed"),
                GenerationTaskDB(
                    task_name="generation",
                    input_path="/managed/input.jsonl",
                    llm_config={},
                    steps_config={},
                    user_id="user-1",
                    status="running",
                ),
                GenerationTaskDB(
                    task_name="publishing-generation",
                    input_path="/managed/publishing-input.jsonl",
                    llm_config={},
                    steps_config={},
                    user_id="user-1",
                    status="publishing",
                ),
                GenerationTaskDB(
                    task_name="restarting-generation",
                    input_path="/managed/restarting-input.jsonl",
                    llm_config={},
                    steps_config={},
                    user_id="user-1",
                    status="restarting",
                ),
                TrainingTaskDB(user_id="user-1", status="pending"),
                TrainingTaskDB(user_id="user-3", status="running"),
                TrainingTaskDB(user_id="user-1", status="succeeded"),
            ]
        )
        session.commit()

    @contextmanager
    def fake_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(admission_module, "get_session", fake_session)
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)

    assert service._active_counts("user-1") == (7, 5)
    assert service._active_counts("user-2") == (7, 1)
    assert service._active_counts("user-3") == (7, 1)


def test_training_create_maps_capacity_to_429_before_task_creation(monkeypatch):
    monkeypatch.setattr(training_routes, "check_idempotency", lambda *_args: (False, None))
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
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "background_task_admission_service",
        SimpleNamespace(
            admit_execution=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                BackgroundTaskCapacityExceeded("per-user active task limit exceeded")
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "create_task",
        lambda **_kwargs: pytest.fail("capacity rejection reached task creation"),
    )
    request = training_routes.TrainingRequest(
        task_name="capacity-test",
        base_model_path="/app/models/model-1",
        datasets=[{"path": "/app/data/train.jsonl", "split": "train"}],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.create_training_task(
                request,
                BackgroundTasks(),
                USER,
                None,
            )
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "5"}


def test_training_create_releases_lease_when_background_scheduling_fails(
    monkeypatch,
):
    releases = []
    failures = []
    lease = SimpleNamespace(release=lambda: releases.append(True))

    def admit(_kind, _task_id, _admission_user_id, operation, *args, **kwargs):
        return operation(*args, **kwargs), lease

    class FailingBackgroundTasks:
        def add_task(self, *_args, **_kwargs):
            raise RuntimeError("background queue unavailable")

    monkeypatch.setattr(training_routes, "check_idempotency", lambda *_args: (False, None))
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
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_set_task_scoped_output",
        lambda task_id, _config: f"/app/output/{task_id}",
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "create_task",
        lambda **_kwargs: {
            "task_id": "training-unscheduled",
            "task_name": "unscheduled",
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_output_dir",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        training_routes,
        "_fail_unscheduled_training",
        lambda task_id, run_token: failures.append((task_id, run_token)),
    )
    monkeypatch.setattr(
        training_routes,
        "store_idempotency_response",
        lambda *_args, **_kwargs: pytest.fail(
            "an unscheduled task must not cache an idempotency response"
        ),
    )
    request = training_routes.TrainingRequest(
        task_name="unscheduled",
        base_model_path="/app/models/model-1",
        datasets=[{"path": "/app/data/train.jsonl", "split": "train"}],
    )

    with pytest.raises(RuntimeError, match="background queue unavailable"):
        asyncio.run(
            training_routes.create_training_task(
                request,
                FailingBackgroundTasks(),
                USER,
                "request-1",
            )
        )

    assert releases == [True]
    assert len(failures) == 1
    assert failures[0][0] == "training-unscheduled"
    assert failures[0][1]


def test_training_resume_releases_lease_when_background_scheduling_fails(
    monkeypatch,
):
    releases = []
    failures = []
    lease = SimpleNamespace(release=lambda: releases.append(True))
    task = {
        "task_id": "training-resume-unscheduled",
        "user_id": USER["user_id"],
        "status": "failed",
        "base_model_path": "/app/models/model-1",
        "train_dataset_path": "/app/data/train.jsonl",
        "output_dir": "/app/output/training-resume-unscheduled",
        "process_pid": None,
        "process_status": None,
        "training_params": {
            "base_model_path": "/app/models/model-1",
            "train_dataset_path": "/app/data/train.jsonl",
        },
    }

    def admit(_kind, _task_id, _admission_user_id, operation, *args, **kwargs):
        return operation(*args, **kwargs), lease

    class FailingBackgroundTasks:
        def add_task(self, *_args, **_kwargs):
            raise RuntimeError("background queue unavailable")

    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "find_latest_checkpoint",
        lambda _task_id: "/app/output/training-resume-unscheduled/checkpoint-1",
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "reset_for_resume",
        lambda _task_id, _run_token, **_kwargs: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_output_dir",
        lambda *_args, **_kwargs: True,
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
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_resolve_server_managed_task_output",
        lambda *_args, **_kwargs: task["output_dir"],
    )
    monkeypatch.setattr(
        training_routes,
        "_resolve_resume_checkpoint",
        lambda *_args, **_kwargs: (
            "/app/output/training-resume-unscheduled/checkpoint-1"
        ),
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        training_routes,
        "_fail_unscheduled_training",
        lambda task_id, run_token: failures.append((task_id, run_token)),
    )

    with pytest.raises(RuntimeError, match="background queue unavailable"):
        asyncio.run(
            training_routes.resume_task(
                "training-resume-unscheduled",
                FailingBackgroundTasks(),
                USER,
            )
        )

    assert releases == [True]
    assert len(failures) == 1
    assert failures[0][0] == "training-resume-unscheduled"
    assert failures[0][1]


def test_standard_evaluation_create_maps_capacity_to_429(monkeypatch):
    monkeypatch.setattr(
        evaluation_routes,
        "_validate_model_configs",
        lambda configs, _user_id: configs,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "_validate_dataset_configs",
        lambda configs, _user: configs,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BackgroundTaskCapacityExceeded("per-user active task limit exceeded")
        ),
    )
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "create_task",
        lambda **_kwargs: pytest.fail("capacity rejection reached task creation"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.create_evaluation_task(
                _request(),
                BackgroundTasks(),
                USER,
            )
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "5"}


@pytest.mark.parametrize(
    ("route_module", "create_operation"),
    (
        (evaluation_routes, "standard"),
        (deep_evaluation_routes, "deep"),
    ),
)
def test_evaluation_create_releases_lease_when_background_scheduling_fails(
    monkeypatch,
    route_module,
    create_operation,
):
    releases = []
    updates = []
    lease = SimpleNamespace(release=lambda: releases.append(True))

    def admit(_kind, _task_id, _admission_user_id, operation, *args, **kwargs):
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(
        route_module.background_task_admission_service,
        "admit_execution",
        admit,
    )

    class FailingBackgroundTasks:
        def add_task(self, *_args, **_kwargs):
            raise RuntimeError("background queue unavailable")

    if create_operation == "standard":
        monkeypatch.setattr(
            evaluation_routes,
            "_validate_model_configs",
            lambda configs, _user_id: configs,
        )
        monkeypatch.setattr(
            evaluation_routes,
            "_validate_dataset_configs",
            lambda configs, _user: configs,
        )
        monkeypatch.setattr(
            evaluation_routes.evaluation_task_service,
            "create_task",
            lambda **_kwargs: {"task_id": "standard-unscheduled"},
        )
        monkeypatch.setattr(
            evaluation_routes.evaluation_task_service,
            "update_status",
            lambda *args: updates.append(args) or True,
        )
        coroutine = evaluation_routes.create_evaluation_task(
            _request(),
            FailingBackgroundTasks(),
            USER,
        )
    else:
        monkeypatch.setattr(
            deep_evaluation_routes,
            "_resolve_deep_evaluation_dataset_configs",
            lambda *_args, **_kwargs: [
                {"dataset_id": "dataset-1", "dataset_name": "dataset"}
            ],
        )
        monkeypatch.setattr(
            deep_evaluation_routes,
            "validate_user_outbound_url",
            lambda url, *_args, **_kwargs: url,
        )
        monkeypatch.setattr(
            deep_evaluation_routes.deep_evaluation_task_service,
            "create_task",
            lambda **_kwargs: {"task_id": "deep-unscheduled"},
        )
        monkeypatch.setattr(
            deep_evaluation_routes.deep_evaluation_task_service,
            "update_status",
            lambda *args: updates.append(args) or True,
        )
        request = deep_evaluation_routes.CreateDeepEvaluationTaskRequest(
            model_configs=[
                {
                    "group_name": "group-1",
                    "embedding": {
                        "endpoint": "https://example.com/v1",
                        "model_name": "embedding",
                    },
                }
            ],
            dataset_configs=[{"dataset_id": "dataset-1"}],
        )
        coroutine = deep_evaluation_routes.create_deep_evaluation_task(
            request,
            FailingBackgroundTasks(),
            USER,
        )

    with pytest.raises(RuntimeError, match="background queue unavailable"):
        asyncio.run(coroutine)

    assert releases == [True]
    assert updates
    assert updates[0][1] == "failed"


def test_deep_evaluation_create_maps_capacity_to_429(monkeypatch):
    request = deep_evaluation_routes.CreateDeepEvaluationTaskRequest(
        model_configs=[
            {
                "group_name": "group-1",
                "embedding": {
                    "endpoint": "https://example.com/v1",
                    "model_name": "embedding",
                },
            }
        ],
        dataset_configs=[{"dataset_id": "dataset-1"}],
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "_resolve_deep_evaluation_dataset_configs",
        lambda *_args, **_kwargs: [
            {"dataset_id": "dataset-1", "dataset_name": "dataset"}
        ],
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, *_args, **_kwargs: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BackgroundTaskCapacityExceeded("per-user active task limit exceeded")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                request,
                BackgroundTasks(),
                USER,
            )
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "5"}


def test_deep_evaluation_create_maps_dataset_fence_race_to_409(monkeypatch):
    request = deep_evaluation_routes.CreateDeepEvaluationTaskRequest(
        model_configs=[
            {
                "group_name": "group-1",
                "embedding": {
                    "endpoint": "https://example.com/v1",
                    "model_name": "embedding",
                },
            }
        ],
        dataset_configs=[{"dataset_id": "dataset-1"}],
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "_resolve_deep_evaluation_dataset_configs",
        lambda *_args, **_kwargs: [
            {"dataset_id": "dataset-1", "dataset_name": "dataset"}
        ],
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, *_args, **_kwargs: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DatasetConsumptionUnavailableError("dataset is being deleted")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                request,
                BackgroundTasks(),
                USER,
            )
        )

    assert exc_info.value.status_code == 409


def test_generation_create_maps_capacity_to_429(monkeypatch, tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("document", encoding="utf-8")
    request = generation_routes.CreateTaskRequest(
        task_name="capacity",
        dataset_id="dataset-1",
        input_format="txt",
        generation_mode="qa_extraction",
        llm_config={
            "endpoint": "https://example.com/v1",
            "model": "model",
        },
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_owned_generation_dataset",
        lambda *_args, **_kwargs: {
            "dataset_id": "dataset-1",
            "storage_path": str(input_path),
            "extra_metadata": {},
        },
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "validate_user_outbound_url",
        lambda url, *_args, **_kwargs: url,
    )
    monkeypatch.setattr(
        generation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BackgroundTaskCapacityExceeded("per-user active task limit exceeded")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.create_task(
                request,
                BackgroundTasks(),
                USER,
            )
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "5"}


@pytest.mark.parametrize(
    ("route", "manager_method"),
    (
        (sync_routes.sync_now, "run_once"),
        (sync_routes.trigger_generation, "trigger_generation"),
    ),
)
def test_manual_sync_generation_maps_capacity_to_429(
    monkeypatch,
    route,
    manager_method,
):
    import importlib

    sync_service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    manager_module = importlib.import_module("train_factory.sync.sync_manager")
    monkeypatch.setattr(
        sync_service_module.external_sync_service,
        "get_task",
        lambda task_id: {"task_id": task_id, "user_id": USER["user_id"]},
    )

    async def reject(_task_id):
        raise BackgroundTaskCapacityExceeded("per-user active task limit exceeded")

    monkeypatch.setattr(manager_module.sync_manager, manager_method, reject)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(route("sync-1", USER))

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "5"}


def test_standard_evaluation_resume_rejects_lost_status_race(monkeypatch):
    task = {
        "task_id": "task-1",
        "user_id": USER["user_id"],
        "status": "failed",
        "model_configs": [
            {"endpoint": "https://example.com/v1/rerank", "name": "model"}
        ],
        "dataset_configs": [{"type": "mteb", "name": "T2Reranking"}],
        "max_samples": 10,
        "batch_size": 50,
        "workers": 8,
        "model_workers": 2,
        "results": {},
    }
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "_validate_model_configs",
        lambda configs, _user_id: configs,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "_validate_dataset_configs",
        lambda configs, _user: configs,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda _kind, _task_id, _user_id, operation, *args, **kwargs: (
            operation(*args, **kwargs),
            SimpleNamespace(release=lambda: None),
        ),
    )
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "reset_for_resume",
        lambda _task_id, **_kwargs: False,
    )
    background = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.resume_evaluation_task(
                "task-1",
                background,
                USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert background.tasks == []


def test_background_task_limits_have_bounded_defaults_and_deployment_wiring():
    fields = Settings.model_fields
    assert fields["background_task_max_active_global"].default == 8
    assert fields["background_task_max_active_per_user"].default == 2
    assert fields["training_allow_cpu_fallback"].default is False

    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load(
        (root / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    env_example = (root / ".env.example").read_text(encoding="utf-8")
    expected = {
        "BACKGROUND_TASK_MAX_ACTIVE_GLOBAL": "8",
        "BACKGROUND_TASK_MAX_ACTIVE_PER_USER": "2",
        "TRAINING_ALLOW_CPU_FALLBACK": "false",
    }
    for service_name in ("train-factory-api", "train-factory-api-dev"):
        environment = compose["services"][service_name]["environment"]
        for name, default in expected.items():
            assert f"{name}={default}" in env_example
            assert environment[name] == f"${{{name}:-{default}}}"


@pytest.mark.parametrize(
    ("cleanup", "service_module_name", "service_name"),
    (
        (
            server.cleanup_orphan_evaluation_tasks,
            "train_factory.storage.services.evaluation_task_service",
            "evaluation_task_service",
        ),
        (
            server.cleanup_orphan_deep_evaluation_tasks,
            "train_factory.storage.services.deep_evaluation_task_service",
            "deep_evaluation_task_service",
        ),
    ),
)
def test_restart_cleanup_finalizes_pending_and_running_evaluations(
    monkeypatch,
    cleanup,
    service_module_name,
    service_name,
):
    import importlib

    service = getattr(importlib.import_module(service_module_name), service_name)
    queried = []
    attempted = []
    updates = []

    def get_all_tasks(*, status, limit, offset):
        queried.append((status, limit, offset))
        if status == "pending" and offset == 0:
            return (
                [{"task_id": f"pending-{index}"} for index in range(1000)],
                1001,
            )
        if status == "pending" and offset == 1000:
            return ([{"task_id": "pending-1000"}], 1001)
        return ([{"task_id": "running-task"}], 1)

    monkeypatch.setattr(service, "get_all_tasks", get_all_tasks)
    def update_status(task_id, status, error_message):
        attempted.append(task_id)
        if task_id == "pending-500":
            raise RuntimeError("row update failed")
        if task_id == "pending-501":
            return False
        updates.append((task_id, status, error_message))
        return True

    monkeypatch.setattr(service, "update_status", update_status)

    cleanup()

    assert queried == [
        ("pending", 1000, 0),
        ("pending", 1000, 1000),
        ("running", 1000, 0),
    ]
    assert len(attempted) == 1002
    assert len(updates) == 1000
    assert updates[0][:2] == ("pending-0", "failed")
    assert updates[-1][:2] == ("running-task", "failed")


def test_restart_cleanup_stops_pending_and_running_generation_tasks(monkeypatch):
    from train_factory.storage.entities.generation_task_entity import GenerationStatus
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )
    from train_factory.storage.services.generation_task_service import (
        generation_task_service,
    )
    from train_factory.storage.services.generation_publication_service import (
        generation_publication_service,
    )

    queried = []
    claimed = []
    finalized = []
    compensations = []

    def get_all_tasks(*, status, limit, offset):
        queried.append((status, limit, offset))
        if status in {
            GenerationStatus.RECOVERING,
            GenerationStatus.RESTARTING,
        }:
            return ([], 0)
        return ([{"task_id": f"{status}-task", "status": status}], 1)

    monkeypatch.setattr(generation_task_service, "get_all_tasks", get_all_tasks)
    monkeypatch.setattr(
        generation_task_service,
        "get_task_raw",
        lambda task_id: {
            "task_id": task_id,
            "status": task_id.removesuffix("-task"),
            "run_token": f"token-{task_id}",
        },
    )

    def claim_orphan_recovery(
        task_id,
        *,
        expected_status,
        expected_run_token,
    ):
        assert expected_run_token == f"token-{task_id}"
        claimed.append((task_id, expected_status))
        if task_id == f"{GenerationStatus.PENDING}-task":
            raise RuntimeError("row update failed")
        return {
            "run_token": expected_run_token,
            "source_status": expected_status,
        }

    def finish_orphan_recovery(
        task_id,
        *,
        expected_run_token,
        terminal_status,
        error_message,
    ):
        assert expected_run_token == f"token-{task_id}"
        finalized.append((task_id, terminal_status, error_message))
        return False

    monkeypatch.setattr(
        generation_task_service,
        "claim_orphan_recovery",
        claim_orphan_recovery,
    )
    monkeypatch.setattr(
        generation_task_service,
        "finish_orphan_recovery",
        finish_orphan_recovery,
    )
    monkeypatch.setattr(
        generation_publication_service,
        "compensate_attempt",
        lambda **kwargs: compensations.append(kwargs) or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_generation_and_restore_batches",
        lambda *_args, **_kwargs: {"tracking_found": False, "recovered": False},
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_pending_generations",
        lambda **_kwargs: ([], 0),
    )

    server.cleanup_orphan_generation_tasks()

    assert queried == [
        (GenerationStatus.PENDING, 1000, 0),
        (GenerationStatus.RUNNING, 1000, 0),
        (GenerationStatus.STOPPING, 1000, 0),
        (GenerationStatus.PUBLISHING, 1000, 0),
        (GenerationStatus.RECOVERING, 1000, 0),
        (GenerationStatus.RESTARTING, 1000, 0),
    ]
    assert claimed == [
        (f"{GenerationStatus.PENDING}-task", GenerationStatus.PENDING),
        (f"{GenerationStatus.RUNNING}-task", GenerationStatus.RUNNING),
        (f"{GenerationStatus.STOPPING}-task", GenerationStatus.STOPPING),
        (f"{GenerationStatus.PUBLISHING}-task", GenerationStatus.PUBLISHING),
    ]
    assert [(task_id, status) for task_id, status, _ in finalized] == [
        (f"{GenerationStatus.RUNNING}-task", GenerationStatus.STOPPED),
        (f"{GenerationStatus.STOPPING}-task", GenerationStatus.STOPPED),
        (f"{GenerationStatus.PUBLISHING}-task", GenerationStatus.FAILED),
    ]
    assert compensations == [
        {
            "task_id": f"{GenerationStatus.PUBLISHING}-task",
            "expected_run_token": (
                f"token-{GenerationStatus.PUBLISHING}-task"
            ),
            "recovery_run_token": (
                f"token-{GenerationStatus.PUBLISHING}-task"
            ),
            "user_id": None,
        }
    ]
