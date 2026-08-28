import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import importlib
import threading
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from train_factory.api.routes import training_routes
from train_factory.enums import TrainingStatus
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)

service_module = importlib.import_module(
    "train_factory.storage.services.training_task_service"
)


def _training_service():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[
            ModelArtifactMembershipGateDB.__table__,
            ModelRegistryDB.__table__,
            TrainingTaskDB.__table__,
        ],
    )
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    service = service_module.TrainingTaskService()
    return service, engine, test_session


def test_claim_preparing_is_atomic_and_does_not_resurrect_stopped(monkeypatch):
    service, engine, test_session = _training_service()
    monkeypatch.setattr(service_module, "get_session", test_session)
    with Session(engine) as session:
        session.add_all(
            [
                TrainingTaskDB(
                    task_id="pending",
                    status=TrainingStatus.PENDING.value,
                    run_token="pending-run",
                ),
                TrainingTaskDB(
                    task_id="stopped",
                    status=TrainingStatus.STOPPED.value,
                    run_token="stopped-run",
                ),
                TrainingTaskDB(
                    task_id="failed",
                    status=TrainingStatus.FAILED.value,
                    run_token="failed-run",
                ),
                TrainingTaskDB(
                    task_id="status-evidence",
                    status=TrainingStatus.RUNNING.value,
                    process_status="running",
                ),
                TrainingTaskDB(
                    task_id="create-time-evidence",
                    status=TrainingStatus.RUNNING.value,
                    process_create_time=1234.5,
                ),
            ]
        )
        session.commit()

    assert service.claim_preparing("pending", "wrong-run") is False
    assert service.claim_preparing("pending", "pending-run") is True
    assert service.claim_preparing("pending", "pending-run") is False
    assert service.claim_preparing("stopped", "stopped-run") is False
    assert service.stop_if_active("pending") is True
    assert service.stop_if_active("pending") is False
    assert service.get_task("pending")["status"] == TrainingStatus.STOPPED.value
    assert service.get_task("pending")["process_status"] is None
    assert service.get_task("stopped")["status"] == TrainingStatus.STOPPED.value
    assert service.stop_if_active("status-evidence") is True
    assert service.get_task("status-evidence")["process_status"] == "stopping"
    assert service.stop_if_active("create-time-evidence") is True
    assert service.get_task("create-time-evidence")["process_status"] == "stopping"
    assert service.reset_for_resume("pending", "resumed-run") is True
    assert service.claim_preparing("pending", "pending-run") is False
    assert service.claim_preparing("pending", "resumed-run") is True
    assert service.reset_for_resume("failed", "failed-resume") is True
    assert service.reset_for_resume("failed", "second-resume") is False
    assert service.get_task("failed")["status"] == TrainingStatus.PENDING.value
    assert service.get_task("failed")["run_token"] == "failed-resume"


def test_old_attempt_cannot_write_or_release_new_attempt_state(monkeypatch):
    service, engine, test_session = _training_service()
    monkeypatch.setattr(service_module, "get_session", test_session)
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="resumed-task",
                status=TrainingStatus.RUNNING.value,
                run_token="new-run",
                process_pid=222,
                process_status="running",
                progress=5,
            )
        )
        session.commit()

    assert service.update_task_status(
        "resumed-task",
        TrainingStatus.FAILED.value,
        "late old failure",
        run_token="old-run",
    ) is False
    assert service.update_task_progress(
        "resumed-task",
        99,
        run_token="old-run",
    ) is False
    assert service.complete_task(
        "resumed-task",
        TrainingStatus.SUCCEEDED.value,
        final_model_path="/tmp/old-result",
        run_token="old-run",
    ) is False
    assert service.update_process_info(
        "resumed-task",
        process_pid=None,
        process_status=None,
        run_token="old-run",
    ) is False

    task = service.get_task("resumed-task")
    assert task["status"] == TrainingStatus.RUNNING.value
    assert task["run_token"] == "new-run"
    assert task["progress"] == 5
    assert task["process_pid"] == 222
    assert task["final_model_path"] is None


def test_task_pagination_has_unique_tie_break_for_equal_created_at(monkeypatch):
    service, engine, test_session = _training_service()
    monkeypatch.setattr(service_module, "get_session", test_session)
    created_at = datetime(2026, 1, 1)
    with Session(engine) as session:
        session.add_all(
            [
                TrainingTaskDB(
                    task_id=f"stable-page-{index:04d}",
                    status=TrainingStatus.PENDING.value,
                    created_at=created_at,
                )
                for index in range(1001)
            ]
        )
        session.commit()

    first, total = service.get_all_tasks(
        status=TrainingStatus.PENDING.value,
        limit=1000,
        offset=0,
    )
    second, second_total = service.get_all_tasks(
        status=TrainingStatus.PENDING.value,
        limit=1000,
        offset=1000,
    )

    task_ids = [task["task_id"] for task in first + second]
    assert total == second_total == 1001
    assert task_ids == [f"stable-page-{index:04d}" for index in range(1000, -1, -1)]


def test_completion_does_not_overwrite_a_stopped_task(monkeypatch):
    service, engine, test_session = _training_service()
    monkeypatch.setattr(service_module, "get_session", test_session)
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="stopped-at-completion",
                status=TrainingStatus.RUNNING.value,
            )
        )
        session.commit()

    assert service.stop_if_active("stopped-at-completion") is True
    assert (
        service.complete_task(
            "stopped-at-completion",
            TrainingStatus.SUCCEEDED.value,
            final_model_path="/tmp/should-not-be-persisted",
            final_metrics={"loss": 0.1},
        )
        is False
    )
    task = service.get_task("stopped-at-completion")
    assert task["status"] == TrainingStatus.STOPPED.value
    assert task["final_model_path"] is None
    assert task["final_metrics"] is None


def test_child_skips_registration_when_completion_loses_to_stop(monkeypatch):
    registrations = []

    class CompletionLostService:
        def get_task(self, _task_id):
            return {
                "task_id": "completion-lost",
                "task_name": "completion-lost",
                "user_id": "user-1",
                "status": TrainingStatus.RUNNING.value,
                "run_token": "completion-run",
            }

        def update_task_progress(self, *_args, **_kwargs):
            return True

        def complete_task(self, *_args, **_kwargs):
            return False

        def update_task_status(self, *_args, **_kwargs):
            raise AssertionError("a stopped task must not be marked failed")

    monkeypatch.setattr(
        training_routes,
        "training_task_service",
        CompletionLostService(),
    )
    monkeypatch.setattr(
        training_routes,
        "train_with_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            save_dir="/tmp/completion-lost",
            final_metrics={"loss": 0.1},
        ),
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "register_from_task",
        lambda *_args, **_kwargs: registrations.append(True),
    )

    training_routes._run_training_task_worker(
        "completion-lost",
        {
            "task_id": "completion-lost",
            "user_id": "user-1",
            "_run_token": "completion-run",
        },
        None,
    )

    assert registrations == []


def test_child_from_old_attempt_exits_before_training(monkeypatch):
    training_calls = []

    class ResumedService:
        def get_task(self, _task_id):
            return {
                "task_id": "resumed-task",
                "user_id": "user-1",
                "status": TrainingStatus.RUNNING.value,
                "run_token": "new-run",
            }

    monkeypatch.setattr(training_routes, "training_task_service", ResumedService())
    monkeypatch.setattr(
        training_routes,
        "train_with_config",
        lambda *_args, **_kwargs: training_calls.append(True),
    )

    training_routes._run_training_task_worker(
        "resumed-task",
        {"task_id": "resumed-task", "_run_token": "old-run"},
        None,
    )

    assert training_calls == []


def test_child_failure_does_not_compensate_when_failure_cas_loses(monkeypatch):
    recovered_failures = []
    status_updates = []

    class FailureCasLostService:
        def get_task(self, _task_id):
            return {
                "task_id": "child-cas-lost",
                "user_id": "user-1",
                "status": TrainingStatus.RUNNING.value,
                "run_token": "child-run",
            }

        def update_task_progress(self, *_args, **_kwargs):
            return True

        def update_task_status(self, *args, **kwargs):
            status_updates.append((args, kwargs))
            return False

    monkeypatch.setattr(training_routes, "training_task_service", FailureCasLostService())
    monkeypatch.setattr(
        training_routes,
        "train_with_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker boom")),
    )
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda task_id, reason: recovered_failures.append((task_id, reason)),
    )

    training_routes._run_training_task_worker(
        "child-cas-lost",
        {"task_id": "child-cas-lost", "_run_token": "child-run"},
        None,
    )

    assert len(status_updates) == 1
    assert recovered_failures == []


def test_child_failure_defers_recovery_after_failure_cas_succeeds(monkeypatch):
    recovered_failures = []

    class FailurePersistedService:
        def get_task(self, _task_id):
            return {
                "task_id": "child-failure-deferred",
                "user_id": "user-1",
                "status": TrainingStatus.RUNNING.value,
                "run_token": "child-run",
            }

        def update_task_progress(self, *_args, **_kwargs):
            return True

        def update_task_status(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(training_routes, "training_task_service", FailurePersistedService())
    monkeypatch.setattr(
        training_routes,
        "train_with_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker boom")),
    )
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda task_id, reason: recovered_failures.append((task_id, reason)),
    )

    training_routes._run_training_task_worker(
        "child-failure-deferred",
        {"task_id": "child-failure-deferred", "_run_token": "child-run"},
        None,
    )

    assert recovered_failures == []


def test_child_success_defers_post_completion_work_to_parent(monkeypatch):
    registrations = []
    sync_callbacks = []

    class SuccessfulChildService:
        status = TrainingStatus.RUNNING.value

        def get_task(self, _task_id):
            return {
                "task_id": "child-success",
                "task_name": "child-success",
                "user_id": "user-1",
                "status": self.status,
                "run_token": "child-run",
            }

        def update_task_progress(self, *_args, **_kwargs):
            return True

        def complete_task(self, *_args, **_kwargs):
            self.status = TrainingStatus.SUCCEEDED.value
            return True

    post_training_module = importlib.import_module(
        "train_factory.sync.post_training_handler"
    )
    monkeypatch.setattr(training_routes, "training_task_service", SuccessfulChildService())
    monkeypatch.setattr(
        training_routes,
        "train_with_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            save_dir="/tmp/child-success",
            final_metrics={"loss": 0.1},
        ),
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "register_from_task",
        lambda **kwargs: registrations.append(kwargs) or {"model_id": "model-1"},
    )
    monkeypatch.setattr(
        post_training_module,
        "on_training_completed",
        lambda **kwargs: sync_callbacks.append(kwargs),
    )

    training_routes._run_training_task_worker(
        "child-success",
        {"task_id": "child-success", "_run_token": "child-run"},
        None,
    )

    assert registrations == []
    assert sync_callbacks == []


def test_unscheduled_training_recovers_only_after_persisting_failure(monkeypatch):
    recovered_failures = []
    status_updates = []

    class FailurePersistedService:
        def update_task_status(self, *args, **kwargs):
            status_updates.append((args, kwargs))
            return True

    monkeypatch.setattr(training_routes, "training_task_service", FailurePersistedService())
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda task_id, reason: recovered_failures.append((task_id, reason)),
    )

    training_routes._fail_unscheduled_training("unscheduled-persisted", "unscheduled-run")

    assert len(status_updates) == 1
    assert recovered_failures == [
        (
            "unscheduled-persisted",
            "Task could not be scheduled for background execution",
        )
    ]


@pytest.mark.parametrize("response_mode", ["cas-false", "database-error"])
def test_unscheduled_response_lost_recovers_current_failed_attempt(
    monkeypatch,
    response_mode,
):
    status_updates = []
    task_reads = []
    recovered_failures = []

    class ResponseLostService:
        task = {
            "task_id": "unscheduled-response-lost",
            "status": TrainingStatus.PENDING.value,
            "run_token": "unscheduled-run",
        }

        def update_task_status(self, *args, **kwargs):
            status_updates.append((args, kwargs))
            self.task["status"] = TrainingStatus.FAILED.value
            if response_mode == "database-error":
                raise RuntimeError("database response was lost")
            return False

        def get_task(self, task_id):
            task_reads.append(task_id)
            return dict(self.task)

    monkeypatch.setattr(training_routes, "training_task_service", ResponseLostService())
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda task_id, reason: recovered_failures.append((task_id, reason)),
    )

    training_routes._fail_unscheduled_training(
        "unscheduled-response-lost",
        "unscheduled-run",
    )

    assert len(status_updates) == 1
    assert task_reads == [
        "unscheduled-response-lost",
        "unscheduled-response-lost",
    ]
    assert recovered_failures == [
        (
            "unscheduled-response-lost",
            "Task could not be scheduled for background execution",
        )
    ]


@pytest.mark.parametrize("failure_mode", ["cas-false", "database-error"])
def test_unscheduled_active_failure_exhaustion_raises_without_recovery(
    monkeypatch,
    failure_mode,
):
    status_updates = []
    recovered_failures = []

    class ActiveAttemptService:
        def update_task_status(self, *args, **kwargs):
            status_updates.append((args, kwargs))
            if failure_mode == "database-error":
                raise RuntimeError("database unavailable")
            return False

        def get_task(self, _task_id):
            return {
                "task_id": "unscheduled-active",
                "status": TrainingStatus.PENDING.value,
                "run_token": "unscheduled-run",
            }

    monkeypatch.setattr(training_routes, "training_task_service", ActiveAttemptService())
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda task_id, reason: recovered_failures.append((task_id, reason)),
    )

    with pytest.raises(training_routes._TrainingFailurePersistenceError) as exc_info:
        training_routes._fail_unscheduled_training(
            "unscheduled-active",
            "unscheduled-run",
        )

    assert len(status_updates) == 3
    assert recovered_failures == []
    if failure_mode == "database-error":
        assert str(exc_info.value.__cause__) == "database unavailable"
    else:
        assert "rejected while the attempt remained active" in str(
            exc_info.value.__cause__
        )


@pytest.mark.parametrize(
    "winner",
    ["stopped", "succeeded", "new-token", "missing"],
)
def test_unscheduled_terminal_or_inactive_winner_does_not_recover(
    monkeypatch,
    winner,
):
    status_updates = []
    task_reads = []
    recovered_failures = []

    class WinnerService:
        def update_task_status(self, *args, **kwargs):
            status_updates.append((args, kwargs))
            raise RuntimeError("database response was lost")

        def get_task(self, task_id):
            task_reads.append(task_id)
            if winner == "missing":
                return None
            if winner == "stopped":
                status = TrainingStatus.STOPPED.value
            elif winner == "succeeded":
                status = TrainingStatus.SUCCEEDED.value
            else:
                status = TrainingStatus.RUNNING.value
            return {
                "task_id": task_id,
                "status": status,
                "run_token": (
                    "unscheduled-run"
                    if winner in {"stopped", "succeeded"}
                    else "replacement-run"
                ),
            }

    monkeypatch.setattr(training_routes, "training_task_service", WinnerService())
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda task_id, reason: recovered_failures.append((task_id, reason)),
    )

    training_routes._fail_unscheduled_training(
        "unscheduled-winner",
        "unscheduled-run",
    )

    assert len(status_updates) == 1
    assert task_reads == ["unscheduled-winner", "unscheduled-winner"]
    assert recovered_failures == []


def test_resume_waits_for_previous_process_cleanup(monkeypatch):
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "cleanup-pending",
            "user_id": "user-1",
            "status": TrainingStatus.STOPPED.value,
            "process_pid": 1234,
            "process_status": "stopping",
        },
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.resume_task(
                "cleanup-pending",
                background_tasks,
                {"user_id": "user-1", "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 409
    assert background_tasks.tasks == []


def test_delete_waits_for_terminal_process_cleanup(monkeypatch):
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "success-cleanup-pending",
            "user_id": "user-1",
            "status": TrainingStatus.SUCCEEDED.value,
            "process_pid": 1234,
            "process_status": "running",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.delete_task(
                "success-cleanup-pending",
                {"user_id": "user-1", "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 400


class _RaceTrainingService:
    def __init__(self, task_id: str):
        self.task = {
            "task_id": task_id,
            "task_name": task_id,
            "user_id": "user-1",
            "status": TrainingStatus.PENDING.value,
            "process_pid": None,
            "process_status": None,
            "process_create_time": None,
            "run_token": None,
        }
        self._lock = threading.Lock()
        self.claim_calls = 0

    def get_task(self, _task_id):
        with self._lock:
            return dict(self.task)

    def claim_preparing(self, _task_id, run_token):
        with self._lock:
            self.claim_calls += 1
            if self.task["status"] != TrainingStatus.PENDING.value:
                return False
            if self.task["run_token"] not in {None, run_token}:
                return False
            self.task["status"] = TrainingStatus.PREPARING.value
            self.task["run_token"] = run_token
            return True

    def stop_if_active(self, _task_id):
        with self._lock:
            if self.task["status"] not in {
                TrainingStatus.PENDING.value,
                TrainingStatus.PREPARING.value,
                TrainingStatus.RUNNING.value,
                TrainingStatus.EVALUATING.value,
            }:
                return False
            self.task["status"] = TrainingStatus.STOPPED.value
            return True

    def update_task_status(
        self,
        _task_id,
        status,
        _error_message=None,
        *,
        run_token=None,
    ):
        with self._lock:
            if run_token is not None and self.task["run_token"] != run_token:
                return False
            current = self.task["status"]
            if current == TrainingStatus.STOPPED.value and status != current:
                raise ValueError(f"invalid transition: {current} -> {status}")
            self.task["status"] = status
            return True

    def update_task_execution_config(self, *_args, **kwargs):
        with self._lock:
            return kwargs.get("run_token") == self.task["run_token"]

    def update_process_info(
        self,
        _task_id,
        process_pid=None,
        process_status=None,
        process_create_time=None,
        *,
        run_token=None,
        expected_process_pid=None,
        expected_process_create_time=None,
    ):
        with self._lock:
            if run_token is not None and self.task["run_token"] != run_token:
                return False
            if (
                expected_process_pid is not None
                and self.task["process_pid"] != expected_process_pid
            ):
                return False
            if (
                expected_process_create_time is not None
                and self.task["process_create_time"]
                != expected_process_create_time
            ):
                return False
            self.task["process_pid"] = process_pid
            self.task["process_status"] = process_status
            self.task["process_create_time"] = process_create_time
        return True


def _patch_training_setup(monkeypatch, service, allocation):
    monkeypatch.setattr(training_routes, "training_task_service", service)
    monkeypatch.setattr(
        training_routes,
        "_apply_llm_training_safety_defaults",
        lambda config: dict(config),
    )
    monkeypatch.setattr(
        training_routes,
        "_apply_max_length_to_nested_configs",
        lambda config: dict(config),
    )
    monkeypatch.setattr(
        training_routes,
        "_normalize_training_config_paths",
        lambda config, _user_id: dict(config),
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_training_parent_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        allocation,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda _task_id: True,
    )


@pytest.mark.parametrize("claim_error", [None, RuntimeError("claim failed")])
def test_duplicate_same_token_worker_does_not_touch_owned_attempt_resources(
    monkeypatch,
    claim_error,
):
    process_info_updates = []
    status_updates = []
    released_leases = []

    class ExistingAttemptService(_RaceTrainingService):
        def claim_preparing(self, *_args, **_kwargs):
            if claim_error is not None:
                raise claim_error
            return False

        def update_task_status(self, *args, **kwargs):
            status_updates.append((args, kwargs))
            return False

        def update_process_info(self, *args, **kwargs):
            process_info_updates.append((args, kwargs))
            return True

    service = ExistingAttemptService("existing-attempt")
    with service._lock:
        service.task.update(
            status=TrainingStatus.RUNNING.value,
            run_token="shared-run",
            process_pid=9876,
            process_status="running",
        )
    monkeypatch.setattr(training_routes, "training_task_service", service)
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda lease_id: released_leases.append(lease_id) or True,
    )

    training_routes.run_training_task(
        "existing-attempt",
        {
            "task_id": "existing-attempt",
            "user_id": "user-1",
            "_run_token": "shared-run",
        },
    )

    assert status_updates == []
    assert process_info_updates == []
    assert released_leases == []
    task = service.get_task("existing-attempt")
    assert task["process_pid"] == 9876
    assert task["process_status"] == "running"


class _LifecycleFakeProcess:
    pid = 4321

    def __init__(self, on_join):
        self._on_join = on_join
        self._popen = None
        self.alive = True
        self.exitcode = None
        self.events = []
        self.join_count = 0

    def start(self):
        self._popen = object()
        self.events.append(("start", None))

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_count += 1
        self.events.append(("join", timeout))
        self._on_join(self)

    def terminate(self):
        self.events.append(("terminate", None))
        self.alive = False
        self.exitcode = -15

    def kill(self):
        self.events.append(("kill", None))
        self.alive = False
        self.exitcode = -9


def _run_lifecycle_case(
    monkeypatch,
    service,
    process,
    release=None,
    task_id=None,
    recover=None,
):
    _patch_training_setup(
        monkeypatch,
        service,
        lambda *_args, **_kwargs: "cuda:0",
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes.multiprocessing,
        "get_context",
        lambda _method: SimpleNamespace(Process=lambda **_kwargs: process),
    )
    monkeypatch.setattr(
        training_routes,
        "capture_process_create_time",
        lambda _pid: 1234.5,
    )
    if release is not None:
        monkeypatch.setattr(
            training_routes.gpu_resource_manager,
            "release_gpus_for_task",
            release,
        )
    recovered_failures = []
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        recover
        or (lambda task_id, reason: recovered_failures.append((task_id, reason))),
    )

    lifecycle_task_id = task_id or service.task["task_id"]
    training_routes.run_training_task(
        lifecycle_task_id,
        {
            "task_id": lifecycle_task_id,
            "user_id": "user-1",
            "gpu_ids": [0],
            "_run_token": "lifecycle-run",
        },
    )
    return recovered_failures


def test_active_training_survives_multiple_monitor_cycles(monkeypatch):
    service = _RaceTrainingService("long-running-training")

    def exit_after_four_monitor_cycles(process):
        if process.join_count == 4:
            process.alive = False
            process.exitcode = 0

    process = _LifecycleFakeProcess(exit_after_four_monitor_cycles)

    recovered_failures = _run_lifecycle_case(monkeypatch, service, process)

    assert process.join_count == 4
    assert all(event != "terminate" for event, _timeout in process.events)
    assert service.get_task("long-running-training")["status"] == TrainingStatus.FAILED.value
    assert len(recovered_failures) == 1


@pytest.mark.parametrize(
    "terminal_status",
    [TrainingStatus.SUCCEEDED.value, TrainingStatus.FAILED.value],
)
def test_terminal_training_gets_a_bounded_process_drain(monkeypatch, terminal_status):
    service = _RaceTrainingService(f"fast-{terminal_status}")

    def finish_status_then_exit_during_drain(process):
        if process.join_count == 1:
            with service._lock:
                service.task["status"] = terminal_status
        elif process.join_count == 2:
            process.alive = False
            process.exitcode = 0 if terminal_status == TrainingStatus.SUCCEEDED.value else 1

    process = _LifecycleFakeProcess(finish_status_then_exit_during_drain)

    recovered_failures = _run_lifecycle_case(monkeypatch, service, process)

    assert process.join_count == 2
    assert all(event != "terminate" for event, _timeout in process.events)
    assert service.get_task(service.task["task_id"])["status"] == terminal_status
    if terminal_status == TrainingStatus.FAILED.value:
        assert len(recovered_failures) == 1
    else:
        assert recovered_failures == []


def test_terminal_drain_escalates_from_terminate_to_kill(monkeypatch):
    service = _RaceTrainingService("terminal-drain-stuck")
    with service._lock:
        service.task["status"] = TrainingStatus.SUCCEEDED.value
        service.task["run_token"] = "terminal-run"

    class StuckUntilKilledProcess(_LifecycleFakeProcess):
        def terminate(self):
            self.events.append(("terminate", None))

    process = StuckUntilKilledProcess(lambda _process: None)
    process.start()
    monkeypatch.setattr(training_routes, "training_task_service", service)
    monkeypatch.setattr(
        training_routes,
        "_TRAINING_PROCESS_TERMINAL_DRAIN_SECONDS",
        0.25,
    )

    task, confirmed_exited = training_routes._monitor_training_process(
        process,
        "terminal-drain-stuck",
        "terminal-run",
    )

    assert task["status"] == TrainingStatus.SUCCEEDED.value
    assert confirmed_exited is True
    assert process.events == [
        ("start", None),
        ("join", 0.25),
        ("terminate", None),
        ("join", 30),
        ("kill", None),
        ("join", 10),
    ]
    assert process.is_alive() is False


@pytest.mark.parametrize("kill_behavior", ["still-alive", "raises"])
def test_unconfirmed_process_exit_preserves_pid_and_gpu_lease(monkeypatch, kill_behavior):
    service = _RaceTrainingService(f"unkillable-{kill_behavior}")
    released_leases = []

    def stop_on_first_monitor_join(process):
        if process.join_count == 1:
            with service._lock:
                service.task["status"] = TrainingStatus.STOPPED.value

    class UnkillableProcess(_LifecycleFakeProcess):
        def terminate(self):
            self.events.append(("terminate", None))

        def kill(self):
            self.events.append(("kill", None))
            if kill_behavior == "raises":
                raise RuntimeError("kill failed")

    process = UnkillableProcess(stop_on_first_monitor_join)

    recovered_failures = _run_lifecycle_case(
        monkeypatch,
        service,
        process,
        release=lambda lease_id: released_leases.append(lease_id) or True,
    )

    task = service.get_task(service.task["task_id"])
    assert process.is_alive() is True
    assert task["process_pid"] == process.pid
    assert task["process_status"] == "running"
    assert released_leases == []
    assert recovered_failures == []


def test_confirmed_process_exit_clears_pid_and_releases_owned_lease(monkeypatch):
    service = _RaceTrainingService("confirmed-process-exit")
    released_leases = []

    def exit_on_first_monitor_join(process):
        process.alive = False
        process.exitcode = 0

    process = _LifecycleFakeProcess(exit_on_first_monitor_join)

    _run_lifecycle_case(
        monkeypatch,
        service,
        process,
        release=lambda lease_id: released_leases.append(lease_id) or True,
    )

    task = service.get_task("confirmed-process-exit")
    assert task["process_pid"] is None
    assert task["process_status"] is None
    assert released_leases == ["training:confirmed-process-exit:lifecycle-run"]


def test_parent_recovers_child_failure_after_cleanup_attempts(monkeypatch):
    events = []

    class ParentFailureService(_RaceTrainingService):
        def update_process_info(
            self,
            task_id,
            process_pid=None,
            process_status=None,
            *,
            run_token=None,
            **process_identity,
        ):
            if process_pid is None and process_status is None:
                events.append("clear-pid")
            return super().update_process_info(
                task_id,
                process_pid,
                process_status,
                run_token=run_token,
                **process_identity,
            )

    service = ParentFailureService("parent-child-failure")

    def fail_and_exit(process):
        with service._lock:
            service.task["status"] = TrainingStatus.FAILED.value
            service.task["error_message"] = "worker boom"
        process.alive = False
        process.exitcode = 1

    process = _LifecycleFakeProcess(fail_and_exit)

    def recover(_task_id, reason):
        assert process.is_alive() is False
        assert events == ["clear-pid", "release-gpu"]
        events.append("recover")
        assert reason == "worker boom"

    _run_lifecycle_case(
        monkeypatch,
        service,
        process,
        release=lambda _lease_id: events.append("release-gpu") or True,
        recover=recover,
    )

    assert events == ["clear-pid", "release-gpu", "recover"]


@pytest.mark.parametrize(
    "cleanup_failure",
    ["pid-false", "pid-raises", "gpu-false", "gpu-raises"],
)
def test_success_finalizer_runs_after_cleanup_attempt_even_if_cleanup_fails(
    monkeypatch,
    cleanup_failure,
):
    events = []

    class CleanupFailureService(_RaceTrainingService):
        def update_process_info(
            self,
            task_id,
            process_pid=None,
            process_status=None,
            *,
            run_token=None,
            **process_identity,
        ):
            if process_pid is None and process_status is None:
                events.append("clear-pid")
                if cleanup_failure == "pid-false":
                    return False
                if cleanup_failure == "pid-raises":
                    raise RuntimeError("pid cleanup unavailable")
            return super().update_process_info(
                task_id,
                process_pid,
                process_status,
                run_token=run_token,
                **process_identity,
            )

    service = CleanupFailureService(f"success-{cleanup_failure}")

    def succeed_and_exit(process):
        with service._lock:
            service.task["status"] = TrainingStatus.SUCCEEDED.value
            service.task["final_model_path"] = "/tmp/cleanup-failure-success"
        process.alive = False
        process.exitcode = 0

    process = _LifecycleFakeProcess(succeed_and_exit)
    post_training_module = importlib.import_module(
        "train_factory.sync.post_training_handler"
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "register_from_task",
        lambda **_kwargs: events.append("register") or {"model_id": "model-1"},
    )
    monkeypatch.setattr(
        post_training_module,
        "on_training_completed",
        lambda **_kwargs: events.append("sync"),
    )

    def release(_lease_id):
        events.append("release-gpu")
        if cleanup_failure == "gpu-false":
            return False
        if cleanup_failure == "gpu-raises":
            raise RuntimeError("gpu cleanup unavailable")
        return True

    _run_lifecycle_case(
        monkeypatch,
        service,
        process,
        release=release,
    )

    assert events == ["clear-pid", "release-gpu", "register", "sync"]


def test_success_finalizer_waits_for_confirmed_process_exit(monkeypatch):
    registrations = []
    sync_callbacks = []
    released_leases = []
    service = _RaceTrainingService("success-process-still-alive")

    def succeed_without_exiting(process):
        with service._lock:
            service.task["status"] = TrainingStatus.SUCCEEDED.value
            service.task["final_model_path"] = "/tmp/process-still-alive"

    class UnkillableSuccessfulProcess(_LifecycleFakeProcess):
        def terminate(self):
            self.events.append(("terminate", None))

        def kill(self):
            self.events.append(("kill", None))

    process = UnkillableSuccessfulProcess(succeed_without_exiting)
    post_training_module = importlib.import_module(
        "train_factory.sync.post_training_handler"
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "register_from_task",
        lambda **kwargs: registrations.append(kwargs),
    )
    monkeypatch.setattr(
        post_training_module,
        "on_training_completed",
        lambda **kwargs: sync_callbacks.append(kwargs),
    )

    _run_lifecycle_case(
        monkeypatch,
        service,
        process,
        release=lambda lease_id: released_leases.append(lease_id) or True,
    )

    task = service.get_task("success-process-still-alive")
    assert task["process_pid"] == process.pid
    assert task["process_status"] == "running"
    assert released_leases == []
    assert registrations == []
    assert sync_callbacks == []


def test_parent_finalizes_success_after_pid_clear_and_gpu_release(monkeypatch):
    events = []
    callback_payloads = []

    class ParentFinalizationService(_RaceTrainingService):
        def update_process_info(
            self,
            task_id,
            process_pid=None,
            process_status=None,
            *,
            run_token=None,
            **process_identity,
        ):
            if process_pid is None and process_status is None:
                events.append("clear-pid")
            return super().update_process_info(
                task_id,
                process_pid,
                process_status,
                run_token=run_token,
                **process_identity,
            )

    service = ParentFinalizationService("parent-finalization")

    def complete_and_exit(process):
        with service._lock:
            service.task["status"] = TrainingStatus.SUCCEEDED.value
            service.task["final_model_path"] = "/tmp/parent-finalization"
        process.alive = False
        process.exitcode = 0

    process = _LifecycleFakeProcess(complete_and_exit)
    post_training_module = importlib.import_module(
        "train_factory.sync.post_training_handler"
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "register_from_task",
        lambda **_kwargs: events.append("register") or {"model_id": "model-1"},
    )

    def slow_sync_callback(**kwargs):
        assert events == ["clear-pid", "release-gpu", "register"]
        assert process.is_alive() is False
        events.append("sync")
        callback_payloads.append(kwargs)

    monkeypatch.setattr(
        post_training_module,
        "on_training_completed",
        slow_sync_callback,
    )

    _run_lifecycle_case(
        monkeypatch,
        service,
        process,
        release=lambda _lease_id: events.append("release-gpu") or True,
    )

    assert events == ["clear-pid", "release-gpu", "register", "sync"]
    assert callback_payloads == [
        {
            "training_task_id": "parent-finalization",
            "final_model_path": "/tmp/parent-finalization",
            "model_registry_id": "model-1",
        }
    ]
    assert all(event not in {"terminate", "kill"} for event, _ in process.events)


@pytest.mark.parametrize("winner", ["failed", "new-token-succeeded"])
def test_parent_post_completion_requires_current_succeeded_attempt(monkeypatch, winner):
    registrations = []
    sync_callbacks = []
    service = _RaceTrainingService(f"parent-no-finalize-{winner}")

    def finish_other_state(process):
        with service._lock:
            if winner == "failed":
                service.task["status"] = TrainingStatus.FAILED.value
            else:
                service.task["status"] = TrainingStatus.SUCCEEDED.value
                service.task["run_token"] = "replacement-run"
                service.task["final_model_path"] = "/tmp/replacement"
        process.alive = False
        process.exitcode = 1 if winner == "failed" else 0

    process = _LifecycleFakeProcess(finish_other_state)
    post_training_module = importlib.import_module(
        "train_factory.sync.post_training_handler"
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "register_from_task",
        lambda **kwargs: registrations.append(kwargs),
    )
    monkeypatch.setattr(
        post_training_module,
        "on_training_completed",
        lambda **kwargs: sync_callbacks.append(kwargs),
    )

    _run_lifecycle_case(monkeypatch, service, process)

    assert registrations == []
    assert sync_callbacks == []


def test_pid_persisted_attempt_transitions_to_running_and_sets_started_at(monkeypatch):
    task_id = "running-transition"
    run_token = "lifecycle-run"
    service, engine, test_session = _training_service()
    monkeypatch.setattr(service_module, "get_session", test_session)
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id=task_id,
                user_id="user-1",
                status=TrainingStatus.PENDING.value,
                run_token=run_token,
            )
        )
        session.commit()

    def stop_after_running(process):
        if process.join_count == 1:
            assert service.stop_if_active(task_id) is True

    process = _LifecycleFakeProcess(stop_after_running)
    _run_lifecycle_case(monkeypatch, service, process, task_id=task_id)

    task = service.get_task(task_id)
    assert task["status"] == TrainingStatus.STOPPED.value
    assert task["started_at"] is not None


@pytest.mark.parametrize("cas_winner", ["stopped", "succeeded", "new-token"])
def test_running_cas_loss_preserves_winning_state_without_compensation(
    monkeypatch,
    cas_winner,
):
    recovered_failures = []

    class RunningCasLostService(_RaceTrainingService):
        running_cas_calls = 0

        def update_task_status(self, _task_id, status, *args, **kwargs):
            if status != TrainingStatus.RUNNING.value:
                return super().update_task_status(_task_id, status, *args, **kwargs)
            with self._lock:
                self.running_cas_calls += 1
                if cas_winner == "new-token":
                    self.task["run_token"] = "replacement-run"
                    self.task["status"] = TrainingStatus.RUNNING.value
                else:
                    self.task["status"] = cas_winner
            return False

    service = RunningCasLostService(f"running-cas-{cas_winner}")

    def finish_without_a_running_transition(process):
        if service.running_cas_calls == 0 or cas_winner == "succeeded":
            process.alive = False
            process.exitcode = 0

    process = _LifecycleFakeProcess(finish_without_a_running_transition)
    _patch_training_setup(
        monkeypatch,
        service,
        lambda *_args, **_kwargs: "cuda:0",
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes.multiprocessing,
        "get_context",
        lambda _method: SimpleNamespace(Process=lambda **_kwargs: process),
    )
    monkeypatch.setattr(
        training_routes,
        "capture_process_create_time",
        lambda _pid: 1234.5,
    )
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda task_id, reason: recovered_failures.append((task_id, reason)),
    )

    training_routes.run_training_task(
        service.task["task_id"],
        {
            "task_id": service.task["task_id"],
            "user_id": "user-1",
            "gpu_ids": [0],
            "_run_token": "lifecycle-run",
        },
    )

    task = service.get_task(service.task["task_id"])
    assert service.running_cas_calls == 1
    assert task["status"] == (
        TrainingStatus.RUNNING.value if cas_winner == "new-token" else cas_winner
    )
    if cas_winner == "new-token":
        assert task["run_token"] == "replacement-run"
    assert recovered_failures == []


def test_monitor_observes_stopped_status_before_bounded_cleanup(monkeypatch):
    service = _RaceTrainingService("stopped-while-monitored")

    def stop_during_second_monitor_cycle(process):
        if process.join_count == 2:
            with service._lock:
                service.task["status"] = TrainingStatus.STOPPED.value

    process = _LifecycleFakeProcess(stop_during_second_monitor_cycle)

    recovered_failures = _run_lifecycle_case(monkeypatch, service, process)

    event_names = [event for event, _timeout in process.events]
    assert event_names[:3] == ["start", "join", "join"]
    assert "terminate" in event_names
    assert "kill" not in event_names
    assert service.get_task("stopped-while-monitored")["status"] == TrainingStatus.STOPPED.value
    assert recovered_failures == []


@pytest.mark.parametrize("inactive_reason", ["cancelled", "missing", "token-mismatch"])
def test_monitor_cleans_process_when_attempt_becomes_inactive(monkeypatch, inactive_reason):
    class MonitorStateService(_RaceTrainingService):
        missing = False

        def get_task(self, task_id):
            if self.missing:
                return None
            return super().get_task(task_id)

        def update_process_info(self, *args, **kwargs):
            if self.missing:
                return False
            return super().update_process_info(*args, **kwargs)

    service = MonitorStateService(f"inactive-{inactive_reason}")

    def deactivate_during_second_monitor_cycle(process):
        if process.join_count != 2:
            return
        if inactive_reason == "missing":
            service.missing = True
            return
        with service._lock:
            if inactive_reason == "cancelled":
                service.task["status"] = TrainingStatus.CANCELLED.value
            else:
                service.task["run_token"] = "replacement-run"

    process = _LifecycleFakeProcess(deactivate_during_second_monitor_cycle)

    recovered_failures = _run_lifecycle_case(monkeypatch, service, process)

    event_names = [event for event, _timeout in process.events]
    assert event_names[:3] == ["start", "join", "join"]
    assert "terminate" in event_names
    assert "kill" not in event_names
    assert recovered_failures == []


def test_natural_process_exit_while_database_is_active_marks_failed(monkeypatch):
    service = _RaceTrainingService("active-process-exit")

    def exit_during_second_monitor_cycle(process):
        if process.join_count == 2:
            process.alive = False
            process.exitcode = 17

    process = _LifecycleFakeProcess(exit_during_second_monitor_cycle)

    recovered_failures = _run_lifecycle_case(monkeypatch, service, process)

    assert process.join_count == 2
    assert all(event != "terminate" for event, _timeout in process.events)
    task = service.get_task("active-process-exit")
    assert task["status"] == TrainingStatus.FAILED.value
    assert len(recovered_failures) == 1
    assert "exit_code=17" in recovered_failures[0][1]


def test_natural_failure_recovery_runs_once_after_cleanup_attempts(monkeypatch):
    events = []

    class NaturalFailureService(_RaceTrainingService):
        def update_task_status(
            self,
            task_id,
            status,
            error_message=None,
            *,
            run_token=None,
        ):
            updated = super().update_task_status(
                task_id,
                status,
                error_message,
                run_token=run_token,
            )
            if updated and status == TrainingStatus.FAILED.value:
                with self._lock:
                    self.task["error_message"] = error_message
            return updated

        def update_process_info(
            self,
            task_id,
            process_pid=None,
            process_status=None,
            *,
            run_token=None,
            **process_identity,
        ):
            if process_pid is None and process_status is None:
                events.append("clear-pid")
            return super().update_process_info(
                task_id,
                process_pid,
                process_status,
                run_token=run_token,
                **process_identity,
            )

    service = NaturalFailureService("natural-failure-order")

    def exit_naturally(process):
        process.alive = False
        process.exitcode = 17

    process = _LifecycleFakeProcess(exit_naturally)

    _run_lifecycle_case(
        monkeypatch,
        service,
        process,
        release=lambda _lease_id: events.append("release-gpu") or True,
        recover=lambda _task_id, _reason: events.append("recover"),
    )

    assert events == ["clear-pid", "release-gpu", "recover"]


def test_unpersistable_natural_failure_preserves_recovery_evidence_and_raises(
    monkeypatch,
):
    failed_update_calls = []
    released_leases = []

    class DatabaseUnavailableService(_RaceTrainingService):
        def update_task_status(self, task_id, status, *args, **kwargs):
            if status == TrainingStatus.FAILED.value:
                failed_update_calls.append((task_id, status))
                raise RuntimeError("database unavailable")
            return super().update_task_status(task_id, status, *args, **kwargs)

    service = DatabaseUnavailableService("ghost-active")

    def exit_naturally(process):
        process.alive = False
        process.exitcode = 17

    process = _LifecycleFakeProcess(exit_naturally)

    with pytest.raises(
        RuntimeError,
        match="Could not persist FAILED status for current training attempt",
    ) as exc_info:
        _run_lifecycle_case(
            monkeypatch,
            service,
            process,
            release=lambda lease_id: released_leases.append(lease_id) or True,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "database unavailable"
    assert len(failed_update_calls) == 3
    task = service.get_task("ghost-active")
    assert task["status"] == TrainingStatus.RUNNING.value
    assert task["process_pid"] == process.pid
    assert task["process_status"] == "running"
    assert released_leases == []


def test_failure_persistence_error_accepts_a_terminal_winner(monkeypatch):
    failed_update_calls = []
    released_leases = []

    class TerminalWinnerService(_RaceTrainingService):
        def update_task_status(self, task_id, status, *args, **kwargs):
            if status == TrainingStatus.FAILED.value:
                failed_update_calls.append((task_id, status))
                with self._lock:
                    self.task["status"] = TrainingStatus.STOPPED.value
                raise RuntimeError("database response was lost")
            return super().update_task_status(task_id, status, *args, **kwargs)

    service = TerminalWinnerService("failure-terminal-winner")

    def exit_naturally(process):
        process.alive = False
        process.exitcode = 17

    process = _LifecycleFakeProcess(exit_naturally)

    recovered_failures = _run_lifecycle_case(
        monkeypatch,
        service,
        process,
        release=lambda lease_id: released_leases.append(lease_id) or True,
    )

    task = service.get_task("failure-terminal-winner")
    assert task["status"] == TrainingStatus.STOPPED.value
    assert task["process_pid"] is None
    assert task["process_status"] is None
    assert failed_update_calls == [
        ("failure-terminal-winner", TrainingStatus.FAILED.value)
    ]
    assert released_leases == [
        "training:failure-terminal-winner:lifecycle-run"
    ]
    assert recovered_failures == []


def test_rejected_pid_persistence_cleans_process_before_monitoring(monkeypatch):
    status_updates = []

    class PidPersistenceRejectedService(_RaceTrainingService):
        def update_process_info(
            self,
            _task_id,
            process_pid=None,
            process_status=None,
            *,
            run_token=None,
            **process_identity,
        ):
            if process_pid is not None:
                return False
            return super().update_process_info(
                _task_id,
                process_pid,
                process_status,
                run_token=run_token,
                **process_identity,
            )

        def update_task_status(self, *args, **kwargs):
            status_updates.append((args, kwargs))
            return super().update_task_status(*args, **kwargs)

    service = PidPersistenceRejectedService("pid-persistence-rejected")
    process = _LifecycleFakeProcess(lambda _process: None)

    recovered_failures = _run_lifecycle_case(monkeypatch, service, process)

    event_names = [event for event, _timeout in process.events]
    assert event_names[:2] == ["start", "terminate"]
    assert service.get_task("pid-persistence-rejected")["status"] == TrainingStatus.PREPARING.value
    assert recovered_failures == []
    assert status_updates == []


def test_rejected_natural_failure_preserves_recovery_evidence_and_raises(monkeypatch):
    failed_update_calls = []
    recovered_failures = []
    released_leases = []

    class FailureCasRejectedService(_RaceTrainingService):
        def update_task_status(self, task_id, status, *args, **kwargs):
            if status == TrainingStatus.FAILED.value:
                failed_update_calls.append((task_id, status))
                return False
            return super().update_task_status(task_id, status, *args, **kwargs)

    service = FailureCasRejectedService("ghost-active-cas-rejected")

    def exit_naturally(process):
        process.alive = False
        process.exitcode = 17

    process = _LifecycleFakeProcess(exit_naturally)

    with pytest.raises(training_routes._TrainingFailurePersistenceError) as exc_info:
        _run_lifecycle_case(
            monkeypatch,
            service,
            process,
            release=lambda lease_id: released_leases.append(lease_id) or True,
            recover=lambda task_id, reason: recovered_failures.append(
                (task_id, reason)
            ),
        )

    assert "rejected while the attempt remained active" in str(
        exc_info.value.__cause__
    )
    assert len(failed_update_calls) == 3
    task = service.get_task("ghost-active-cas-rejected")
    assert task["status"] == TrainingStatus.RUNNING.value
    assert task["process_pid"] == process.pid
    assert task["process_status"] == "running"
    assert released_leases == []
    assert recovered_failures == []


def test_stop_during_validation_prevents_gpu_allocation(monkeypatch):
    task_id = "stop-during-validation"
    service = _RaceTrainingService(task_id)
    validation_started = threading.Event()
    finish_validation = threading.Event()
    allocations = []
    _patch_training_setup(
        monkeypatch,
        service,
        lambda *_args, **_kwargs: allocations.append(True) or None,
    )

    def blocking_validation(*_args, **_kwargs):
        validation_started.set()
        assert finish_validation.wait(timeout=5)

    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        blocking_validation,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            training_routes.run_training_task,
            task_id,
            {"task_id": task_id, "user_id": "user-1", "gpu_ids": [0]},
        )
        assert validation_started.wait(timeout=5)
        try:
            asyncio.run(training_routes.stop_task(task_id, {"user_id": "user-1"}))
        finally:
            finish_validation.set()
        worker.result(timeout=5)

    assert service.claim_calls == 1
    assert allocations == []
    task = service.get_task(task_id)
    assert task["status"] == TrainingStatus.STOPPED.value
    assert task["process_pid"] is None
    assert task["process_status"] is None
    assert task["process_create_time"] is None


def test_worker_rejects_legacy_qlora_before_gpu_allocation(monkeypatch):
    task_id = "legacy-qlora-worker"
    service = _RaceTrainingService(task_id)
    allocations = []
    recovered = []
    _patch_training_setup(
        monkeypatch,
        service,
        lambda *_args, **_kwargs: allocations.append(True) or "cuda:0",
    )
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda task_id, reason: recovered.append((task_id, reason)),
    )

    training_routes.run_training_task(
        task_id,
        {
            "task_id": task_id,
            "user_id": "user-1",
            "tuner_type": " QLoRA ",
            "_run_token": "legacy-qlora-run",
        },
    )

    assert allocations == []
    assert service.get_task(task_id)["status"] == TrainingStatus.FAILED.value
    assert recovered == [
        (
            task_id,
            "QLoRA 4-bit quantization is not implemented end-to-end. "
            "Use 'lora' or 'full' instead.",
        )
    ]


def test_stop_while_process_starts_is_observed_before_join(monkeypatch):
    task_id = "stop-during-process-start"
    service = _RaceTrainingService(task_id)
    process_started = threading.Event()
    finish_start = threading.Event()
    events = []
    _patch_training_setup(
        monkeypatch,
        service,
        lambda *_args, **_kwargs: "cuda:0",
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        lambda *_args, **_kwargs: None,
    )

    class FakeProcess:
        pid = 4321
        exitcode = None

        def __init__(self):
            self._popen = object()
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append("terminate")
            self.alive = False

        def join(self, timeout=None):
            del timeout
            events.append("join")
            if self.alive:
                raise AssertionError("parent joined a stopped task before terminating it")

    process = FakeProcess()
    monkeypatch.setattr(
        training_routes.multiprocessing,
        "get_context",
        lambda _method: SimpleNamespace(Process=lambda **_kwargs: process),
    )

    def blocking_start(_process, _task_id, _cuda_visible):
        process_started.set()
        assert finish_start.wait(timeout=5)

    monkeypatch.setattr(training_routes, "_start_training_process", blocking_start)

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            training_routes.run_training_task,
            task_id,
            {"task_id": task_id, "user_id": "user-1", "gpu_ids": [0]},
        )
        assert process_started.wait(timeout=5)
        try:
            asyncio.run(training_routes.stop_task(task_id, {"user_id": "user-1"}))
        finally:
            finish_start.set()
        worker.result(timeout=5)

    assert service.get_task(task_id)["status"] == TrainingStatus.STOPPED.value
    assert events[0] == "terminate"
    assert process.is_alive() is False
