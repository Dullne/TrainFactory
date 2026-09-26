"""Regression tests for sync worker recovery after an aborted deletion."""

import asyncio
import importlib
import threading
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException

from train_factory.api.routes import sync_routes
from train_factory.storage.services.dataset_service import dataset_service
from train_factory.storage.services.evaluation_task_service import (
    evaluation_task_service,
)
from train_factory.storage.services.external_sync_service import (
    external_sync_service,
)
from train_factory.storage.services.generation_task_service import (
    generation_task_service,
)
from train_factory.storage.services.training_task_service import (
    training_task_service,
)
from train_factory.sync.sync_manager import SyncManager

sync_manager_module = importlib.import_module("train_factory.sync.sync_manager")


async def _install_running_manager(monkeypatch, task_id, *, start_worker):
    manager = SyncManager()
    manager._running = True
    worker_started = asyncio.Event()

    async def idle_worker(_task_id):
        worker_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_run_worker", idle_worker)
    monkeypatch.setattr(sync_manager_module, "sync_manager", manager)

    if start_worker:
        manager.start_worker(task_id)
        await worker_started.wait()

    return manager


def _configure_parent(monkeypatch, task_id, user_id, *, is_active):
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": is_active,
        "status": "idle",
    }
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(config),
    )
    return config


@pytest.mark.parametrize("abort_kind", ["active_child", "snapshot_exception"])
def test_aborted_delete_restores_active_worker_on_api_loop(monkeypatch, abort_kind):
    """Removing the loop-side recovery makes the real worker remain absent."""
    task_id = f"sync-recovery-{abort_kind}"
    user_id = "sync-recovery-user"
    _configure_parent(monkeypatch, task_id, user_id, is_active=True)

    if abort_kind == "active_child":
        monkeypatch.setattr(
            external_sync_service,
            "list_generations",
            lambda **_kwargs: (
                [
                    {
                        "id": 1,
                        "task_id": task_id,
                        "user_id": user_id,
                        "generation_task_id": "generation-active",
                        "status": "running",
                    }
                ],
                1,
            ),
        )
        monkeypatch.setattr(
            external_sync_service,
            "list_trainings",
            lambda **_kwargs: ([], 0),
        )
        monkeypatch.setattr(
            generation_task_service,
            "get_task",
            lambda _task_id: {"status": "running", "user_id": user_id},
        )
        monkeypatch.setattr(
            external_sync_service,
            "begin_task_deletion",
            lambda *_args, **_kwargs: pytest.fail("an active child must reject before delete intent"),
        )
        expected_error = HTTPException
    else:

        class SnapshotFailure(RuntimeError):
            pass

        def fail_snapshot(**_kwargs):
            raise SnapshotFailure("snapshot unavailable")

        monkeypatch.setattr(
            external_sync_service,
            "list_generations",
            fail_snapshot,
        )
        expected_error = SnapshotFailure

    async def exercise():
        manager = await _install_running_manager(
            monkeypatch,
            task_id,
            start_worker=True,
        )
        original_worker = manager._workers[task_id]
        lock_release_status = []
        real_operation_lock = manager.task_operation_lock

        @asynccontextmanager
        async def observed_operation_lock(requested_task_id):
            async with real_operation_lock(requested_task_id):
                try:
                    yield
                finally:
                    lock_release_status.append(manager.get_worker_status(requested_task_id))

        monkeypatch.setattr(
            manager,
            "task_operation_lock",
            observed_operation_lock,
        )
        try:
            with pytest.raises(expected_error) as exc_info:
                await sync_routes.delete_sync_task(
                    task_id,
                    cascade=True,
                    current_user={"user_id": user_id, "is_admin": False},
                )

            if abort_kind == "active_child":
                assert exc_info.value.status_code == 409
            assert manager.get_worker_status(task_id) == "running"
            assert manager._workers[task_id] is not original_worker
            assert lock_release_status == ["running"]
        finally:
            await manager.stop()

    asyncio.run(exercise())


def test_aborted_delete_does_not_start_worker_for_inactive_task(monkeypatch):
    task_id = "sync-recovery-inactive"
    user_id = "sync-recovery-user"
    _configure_parent(monkeypatch, task_id, user_id, is_active=False)
    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: (
            [
                {
                    "id": 1,
                    "task_id": task_id,
                    "user_id": user_id,
                    "generation_task_id": "generation-active",
                    "status": "running",
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda _task_id: {"status": "running", "user_id": user_id},
    )

    async def exercise():
        manager = await _install_running_manager(
            monkeypatch,
            task_id,
            start_worker=False,
        )
        try:
            with pytest.raises(HTTPException) as exc_info:
                await sync_routes.delete_sync_task(
                    task_id,
                    cascade=True,
                    current_user={"user_id": user_id, "is_admin": False},
                )

            assert exc_info.value.status_code == 409
            assert manager.get_worker_status(task_id) == "not_found"
        finally:
            await manager.stop()

    asyncio.run(exercise())


def test_successful_delete_leaves_worker_stopped(monkeypatch):
    task_id = "sync-recovery-success"
    user_id = "sync-recovery-user"
    original = _configure_parent(
        monkeypatch,
        task_id,
        user_id,
        is_active=True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda *_args, **_kwargs: {
            **original,
            "is_active": False,
            "status": "deleting",
            "_deletion_identity": "delete-success",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "finalize_task_deletion",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        dataset_service,
        "list_external_storage_reference_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_artifact_reference_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )

    async def exercise():
        manager = await _install_running_manager(
            monkeypatch,
            task_id,
            start_worker=True,
        )
        try:
            result = await sync_routes.delete_sync_task(
                task_id,
                cascade=False,
                current_user={"user_id": user_id, "is_admin": False},
            )

            assert result == {"message": "Sync task deleted"}
            assert manager.get_worker_status(task_id) == "not_found"
        finally:
            await manager.stop()

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_stage", ["lock_wait", "latest_read"])
def test_cancelled_delete_before_snapshot_restores_active_worker(monkeypatch, cancel_stage):
    """Cancellation after stopping the worker must finish deletion or recovery."""
    task_id = f"sync-recovery-cancel-{cancel_stage}"
    user_id = "sync-recovery-user"
    config = _configure_parent(monkeypatch, task_id, user_id, is_active=True)
    finish_latest_read = threading.Event()
    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: (
            [
                {
                    "id": 1,
                    "task_id": task_id,
                    "user_id": user_id,
                    "generation_task_id": "generation-active",
                    "status": "running",
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(external_sync_service, "list_trainings", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda _task_id: {"status": "running", "user_id": user_id},
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda *_args, **_kwargs: pytest.fail("active children must prevent deletion"),
    )

    async def exercise():
        manager = await _install_running_manager(monkeypatch, task_id, start_worker=True)
        original_worker = manager._workers[task_id]
        loop = asyncio.get_running_loop()
        boundary_reached = asyncio.Event()
        lock_held = asyncio.Event()
        release_lock = asyncio.Event()
        lock_release_status = []
        real_operation_lock = manager.task_operation_lock
        reads = 0

        def get_task(_task_id):
            nonlocal reads
            reads += 1
            if reads == 2 and cancel_stage == "latest_read":
                loop.call_soon_threadsafe(boundary_reached.set)
                assert finish_latest_read.wait(5)
            return dict(config)

        monkeypatch.setattr(external_sync_service, "get_task", get_task)

        @asynccontextmanager
        async def observed_operation_lock(requested_task_id):
            if cancel_stage == "lock_wait":
                boundary_reached.set()
            async with real_operation_lock(requested_task_id):
                try:
                    yield
                finally:
                    lock_release_status.append(manager.get_worker_status(requested_task_id))

        monkeypatch.setattr(manager, "task_operation_lock", observed_operation_lock)

        async def hold_task_lock():
            async with real_operation_lock(task_id):
                lock_held.set()
                await release_lock.wait()

        lock_holder = None
        if cancel_stage == "lock_wait":
            lock_holder = asyncio.create_task(hold_task_lock())
            await asyncio.wait_for(lock_held.wait(), timeout=2)
        delete_request = asyncio.create_task(
            sync_routes.delete_sync_task(
                task_id,
                cascade=True,
                current_user={"user_id": user_id, "is_admin": False},
            )
        )
        try:
            await asyncio.wait_for(boundary_reached.wait(), timeout=2)
            assert manager.get_worker_status(task_id) == "not_found"
            delete_request.cancel("first cancellation")
            # Yield to deliver cancellation while the exact boundary stays blocked.
            await asyncio.sleep(0)
            delete_request.cancel("repeated cancellation")
            await asyncio.sleep(0)
            release_lock.set()
            finish_latest_read.set()

            with pytest.raises(asyncio.CancelledError) as exc_info:
                await asyncio.wait_for(delete_request, timeout=2)
            assert exc_info.value.args == ("first cancellation",)
            assert config["is_active"] is True
            assert manager.get_worker_status(task_id) == "running"
            assert manager._workers[task_id] is not original_worker
            assert lock_release_status == ["running"]
        finally:
            release_lock.set()
            finish_latest_read.set()
            if not delete_request.done():
                delete_request.cancel()
            try:
                await delete_request
            except (asyncio.CancelledError, HTTPException):
                pass
            if lock_holder is not None:
                await asyncio.wait_for(lock_holder, timeout=2)
            await manager.stop()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "latest_state",
    ["active", "inactive", "deleting", "deleting_cascade", "deleted", "unavailable"],
)
def test_delete_latest_read_failure_restarts_worker_with_fresh_state(monkeypatch, latest_state):
    """Read failures must restore a worker without reviving stale task state."""
    task_id = f"sync-recovery-read-failure-{latest_state}"
    user_id = "sync-recovery-user"
    config = _configure_parent(monkeypatch, task_id, user_id, is_active=True)
    latest = {
        **config,
        "is_active": latest_state != "inactive",
        "status": latest_state if latest_state.startswith("deleting") else "idle",
        "sync_interval_seconds": 300,
        "source_cursor": "fresh-cursor",
    }
    failure = RuntimeError("latest task read unavailable")
    raw_read = threading.Event()
    cycle_finished = threading.Event()
    worker_error_handled = threading.Event()
    cycles = []
    reads = 0

    def get_task(_task_id):
        nonlocal reads
        reads += 1
        if reads == 2:
            raise failure
        return dict(config)

    def get_task_raw(_task_id):
        raw_read.set()
        if latest_state == "unavailable":
            raise RuntimeError("database still unavailable")
        return None if latest_state == "deleted" else dict(latest)

    def run_sync_cycle(current):
        cycles.append(current)
        cycle_finished.set()

    monkeypatch.setattr(external_sync_service, "get_task", get_task)
    monkeypatch.setattr(external_sync_service, "get_task_raw", get_task_raw)
    monkeypatch.setattr(sync_manager_module, "_run_sync_cycle", run_sync_cycle)
    monkeypatch.setattr(
        sync_manager_module,
        "_set_task_error",
        lambda *_args: worker_error_handled.set(),
    )

    async def exercise():
        manager = await _install_running_manager(monkeypatch, task_id, start_worker=True)
        original_worker = manager._workers[task_id]
        # Only the original worker is idle; exercise the production replacement loop.
        monkeypatch.setattr(manager, "_run_worker", SyncManager._run_worker.__get__(manager))
        lock_release_status = []
        restored_workers = []
        real_operation_lock = manager.task_operation_lock

        @asynccontextmanager
        async def observed_operation_lock(requested_task_id):
            async with real_operation_lock(requested_task_id):
                try:
                    yield
                finally:
                    lock_release_status.append(manager.get_worker_status(requested_task_id))
                    restored_workers.append(manager._workers.get(requested_task_id))

        monkeypatch.setattr(manager, "task_operation_lock", observed_operation_lock)
        try:
            with pytest.raises(RuntimeError) as exc_info:
                await sync_routes.delete_sync_task(
                    task_id,
                    cascade=True,
                    current_user={"user_id": user_id, "is_admin": False},
                )
            assert exc_info.value is failure
            assert lock_release_status == ["running"]
            restored_worker = restored_workers[0]
            assert restored_worker is not original_worker
            if latest_state == "active":
                assert await asyncio.to_thread(cycle_finished.wait, 2)
                assert manager.get_worker_status(task_id) == "running"
                assert [cycle["source_cursor"] for cycle in cycles] == ["fresh-cursor"]
            elif latest_state == "unavailable":
                assert await asyncio.to_thread(worker_error_handled.wait, 2)
                assert manager.get_worker_status(task_id) == "running"
                assert not cycles
            else:
                await asyncio.wait_for(asyncio.shield(restored_worker), timeout=2)
                assert manager.get_worker_status(task_id) == "not_found"
                assert not cycles
            assert raw_read.is_set()
            assert config["is_active"] is True
            assert config["status"] == "idle"
        finally:
            await manager.stop()

    asyncio.run(exercise())


def test_cancelled_delete_holds_lock_until_abort_recovery_finishes(monkeypatch):
    task_id = "sync-recovery-cancelled"
    user_id = "sync-recovery-user"
    snapshot_started = threading.Event()
    finish_snapshot = threading.Event()
    _configure_parent(monkeypatch, task_id, user_id, is_active=True)

    def blocked_generation_snapshot(**_kwargs):
        snapshot_started.set()
        assert finish_snapshot.wait(5)
        return (
            [
                {
                    "id": 1,
                    "task_id": task_id,
                    "user_id": user_id,
                    "generation_task_id": "generation-active",
                    "status": "running",
                }
            ],
            1,
        )

    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        blocked_generation_snapshot,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda _task_id: {"status": "running", "user_id": user_id},
    )

    async def exercise():
        manager = await _install_running_manager(
            monkeypatch,
            task_id,
            start_worker=True,
        )
        delete_request = asyncio.create_task(
            sync_routes.delete_sync_task(
                task_id,
                cascade=True,
                current_user={"user_id": user_id, "is_admin": False},
            )
        )
        competing_operation = None
        competitor_entered = asyncio.Event()
        status_when_competitor_entered = []

        async def compete_for_task_lock():
            async with manager.task_operation_lock(task_id):
                status_when_competitor_entered.append(manager.get_worker_status(task_id))
                competitor_entered.set()

        try:
            assert await asyncio.to_thread(snapshot_started.wait, 2)
            competing_operation = asyncio.create_task(compete_for_task_lock())
            await asyncio.sleep(0)

            delete_request.cancel()
            await asyncio.sleep(0.05)
            assert not delete_request.done()
            assert not competitor_entered.is_set()

            delete_request.cancel()
            await asyncio.sleep(0.05)
            assert not delete_request.done()
            assert not competitor_entered.is_set()

            finish_snapshot.set()
            with pytest.raises(asyncio.CancelledError):
                await delete_request
            await asyncio.wait_for(competing_operation, timeout=2)

            assert status_when_competitor_entered == ["running"]
        finally:
            finish_snapshot.set()
            if not delete_request.done():
                delete_request.cancel()
            try:
                await delete_request
            except (asyncio.CancelledError, HTTPException):
                pass
            if competing_operation is not None and not competing_operation.done():
                competing_operation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await competing_operation
            await manager.stop()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("operation", "initially_active", "expected_active", "expected_worker"),
    [
        ("start", False, True, "running"),
        ("stop", True, False, "not_found"),
        ("update", True, False, "not_found"),
    ],
)
def test_cancelled_sync_state_mutation_finishes_worker_handoff(
    monkeypatch,
    operation,
    initially_active,
    expected_active,
    expected_worker,
):
    task_id = f"sync-cancel-{operation}"
    user_id = "sync-recovery-user"
    mutation_started = threading.Event()
    finish_mutation = threading.Event()
    state = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": initially_active,
        "status": "idle",
    }
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(state),
    )

    def blocked_update(_task_id, **updates):
        mutation_started.set()
        assert finish_mutation.wait(5)
        state.update({key: value for key, value in updates.items() if key != "expected_user_id"})
        return dict(state)

    monkeypatch.setattr(external_sync_service, "update_task", blocked_update)
    monkeypatch.setattr(
        sync_routes,
        "_validate_external_api_config_reference",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sync_routes,
        "_validate_base_deployment_reference",
        lambda *_args, **_kwargs: None,
    )

    async def exercise():
        manager = await _install_running_manager(
            monkeypatch,
            task_id,
            start_worker=initially_active,
        )
        if operation == "start":
            call = sync_routes.start_sync(
                task_id,
                {"user_id": user_id, "is_admin": False},
            )
        elif operation == "stop":
            call = sync_routes.stop_sync(
                task_id,
                {"user_id": user_id, "is_admin": False},
            )
        else:
            call = sync_routes.update_sync_task(
                task_id,
                sync_routes.SyncTaskUpdateRequest(is_active=False),
                {"user_id": user_id, "is_admin": False},
            )
        request = asyncio.create_task(call)
        try:
            assert await asyncio.to_thread(mutation_started.wait, 2)
            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()

            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()

            finish_mutation.set()
            with pytest.raises(asyncio.CancelledError):
                await request

            assert state["is_active"] is expected_active
            assert manager.get_worker_status(task_id) == expected_worker
        finally:
            finish_mutation.set()
            if not request.done():
                request.cancel()
            try:
                await request
            except asyncio.CancelledError:
                pass
            await manager.stop()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_cancelled_training_target_mutation_keeps_task_lock(monkeypatch, operation):
    task_id = f"sync-target-cancel-{operation}"
    target_id = "target-1"
    user_id = "sync-recovery-user"
    mutation_started = threading.Event()
    finish_mutation = threading.Event()
    mutation_finished = threading.Event()
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": False,
        "status": "idle",
        "external_api_config_id": None,
    }
    target = {
        "target_id": target_id,
        "task_id": task_id,
        "user_id": user_id,
        "status": "idle",
    }
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(config),
    )
    monkeypatch.setattr(
        sync_routes,
        "_validate_base_deployment_reference",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sync_routes,
        "_get_or_materialize_training_target_for_write",
        lambda *_args, **_kwargs: dict(target),
    )

    def blocked_mutation(*_args, **_kwargs):
        mutation_started.set()
        assert finish_mutation.wait(5)
        mutation_finished.set()
        if operation == "delete":
            return True
        return dict(target)

    monkeypatch.setattr(
        external_sync_service,
        f"{operation}_training_target",
        blocked_mutation,
    )

    async def exercise():
        manager = await _install_running_manager(
            monkeypatch,
            task_id,
            start_worker=False,
        )
        if operation == "create":
            call = sync_routes.create_training_target(
                task_id,
                sync_routes.TrainingTargetCreateRequest(target_name="target"),
                {"user_id": user_id, "is_admin": False},
            )
        elif operation == "update":
            call = sync_routes.update_training_target(
                task_id,
                target_id,
                sync_routes.TrainingTargetUpdateRequest(target_name="updated"),
                {"user_id": user_id, "is_admin": False},
            )
        else:
            call = sync_routes.delete_training_target(
                task_id,
                target_id,
                {"user_id": user_id, "is_admin": False},
            )
        request = asyncio.create_task(call)
        competitor_entered = asyncio.Event()

        async def compete_for_lock():
            async with manager.task_operation_lock(task_id):
                competitor_entered.set()

        competitor = None
        try:
            assert await asyncio.to_thread(mutation_started.wait, 2)
            competitor = asyncio.create_task(compete_for_lock())
            await asyncio.sleep(0)
            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()
            assert not competitor_entered.is_set()

            request.cancel()
            await asyncio.sleep(0.05)
            assert not request.done()
            assert not competitor_entered.is_set()

            finish_mutation.set()
            with pytest.raises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(competitor, timeout=2)
            assert mutation_finished.is_set()
        finally:
            finish_mutation.set()
            if not request.done():
                request.cancel()
            try:
                await request
            except asyncio.CancelledError:
                pass
            if competitor is not None and not competitor.done():
                competitor.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await competitor
            await manager.stop()

    asyncio.run(exercise())
