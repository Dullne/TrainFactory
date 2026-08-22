import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import importlib
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import generation_routes
from train_factory.generation.pipeline import PipelineConfig
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)


@pytest.fixture
def generation_service(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'generation-stop-race.db'}")
    SQLModel.metadata.create_all(
        engine,
        tables=[GenerationTaskDB.__table__, DatasetDB.__table__],
    )

    service_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)
    return service_module.GenerationTaskService(), engine


def _add_generation_task(engine, task_id: str, status: str, **overrides) -> None:
    values = {
        "task_id": task_id,
        "task_name": task_id,
        "input_path": "/managed/input.jsonl",
        "llm_config": {},
        "steps_config": {},
        "status": status,
    }
    values.update(overrides)
    with Session(engine) as session:
        session.add(GenerationTaskDB(**values))
        session.commit()


def test_claim_running_is_a_single_pending_to_running_transition(generation_service):
    service, engine = generation_service
    _add_generation_task(engine, "pending-task", GenerationStatus.PENDING)

    assert hasattr(service, "claim_running"), "worker startup requires an atomic claim"
    run_token = service.claim_running("pending-task")
    assert run_token == service.get_task_raw("pending-task")["run_token"]
    assert service.claim_running("pending-task") is None

    task = service.get_task("pending-task")
    assert task["status"] == GenerationStatus.RUNNING
    assert task["started_at"] is not None


def test_claim_running_does_not_resurrect_a_stopped_task(generation_service):
    service, engine = generation_service
    _add_generation_task(engine, "stopped-task", GenerationStatus.STOPPED)

    assert hasattr(service, "claim_running"), "worker startup requires an atomic claim"
    assert service.claim_running("stopped-task") is None
    assert service.get_task("stopped-task")["status"] == GenerationStatus.STOPPED


@pytest.mark.parametrize(
    ("source_status", "target_status", "error_message"),
    (
        (GenerationStatus.PENDING, GenerationStatus.RUNNING, None),
        (GenerationStatus.RUNNING, GenerationStatus.PUBLISHING, None),
        (GenerationStatus.PUBLISHING, GenerationStatus.COMPLETED, None),
        (GenerationStatus.RUNNING, GenerationStatus.FAILED, "pipeline failed"),
        (GenerationStatus.RUNNING, GenerationStatus.STOPPING, None),
        (GenerationStatus.STOPPING, GenerationStatus.STOPPED, None),
    ),
)
def test_active_status_transitions_are_single_cas_with_lifecycle_fields(
    generation_service,
    source_status,
    target_status,
    error_message,
):
    service, engine = generation_service
    task_id = f"{source_status}-to-{target_status}"
    initial_started_at = (
        datetime(2026, 1, 2, 3, 4, 5)
        if source_status in {
            GenerationStatus.RUNNING,
            GenerationStatus.STOPPING,
            GenerationStatus.PUBLISHING,
        }
        else None
    )
    _add_generation_task(
        engine,
        task_id,
        source_status,
        started_at=initial_started_at,
        error_message=(
            "stale error"
            if target_status == GenerationStatus.COMPLETED
            else None
        ),
    )

    updated = service.update_status(task_id, target_status, error_message)

    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        assert updated is True
        assert task.status == target_status
        if target_status == GenerationStatus.RUNNING:
            assert task.started_at is not None
            assert task.completed_at is None
        elif target_status in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
        }:
            assert task.started_at == initial_started_at
            assert task.completed_at is not None
        else:
            assert task.started_at == initial_started_at
            assert task.completed_at is None
        if target_status == GenerationStatus.COMPLETED:
            assert task.error_message is None
        elif error_message is not None:
            assert task.error_message == error_message


def test_late_worker_completion_cannot_overwrite_committed_stop(generation_service):
    service, engine = generation_service
    task_id = "stop-before-completion"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.RUNNING,
        started_at=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert service.update_status(task_id, GenerationStatus.STOPPING) is True
    assert service.update_status(task_id, GenerationStatus.STOPPED) is True
    assert service.update_status(task_id, GenerationStatus.COMPLETED) is False

    task = service.get_task(task_id)
    assert task["status"] == GenerationStatus.STOPPED


def test_restart_claim_atomically_resets_lifecycle_and_attempt_artifacts(
    generation_service,
):
    service, engine = generation_service
    task_id = "atomic-restart"
    output_path = "/managed/generated.jsonl"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.FAILED,
        progress=87.5,
        total_docs=12,
        processed_docs=11,
        output_sample_count=9,
        error_message="old failure",
        started_at=datetime(2026, 1, 2, 3, 4, 5),
        completed_at=datetime(2026, 1, 2, 4, 5, 6),
        output_path=output_path,
        qa_output_path="/managed/qa-checkpoint.jsonl",
    )

    assert service.update_status(task_id, GenerationStatus.PENDING) is True

    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        assert task.status == GenerationStatus.PENDING
        assert task.progress == 0.0
        assert task.processed_docs == 0
        assert task.output_sample_count == 0
        assert task.error_message is None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.total_docs == 0
        assert task.output_path is None
        assert task.qa_output_path is None


def test_update_status_rejects_unknown_target_without_mutating_task(
    generation_service,
):
    service, engine = generation_service
    task_id = "unknown-target"
    _add_generation_task(engine, task_id, GenerationStatus.PENDING)

    with pytest.raises(ValueError, match="Invalid status"):
        service.update_status(task_id, "not-a-generation-status")

    assert service.get_task(task_id)["status"] == GenerationStatus.PENDING


def test_artifact_reference_scan_is_all_tenant_and_excludes_current_task(
    generation_service,
):
    service, engine = generation_service
    shared_path = "/managed/shared-output"
    nested_path = f"{shared_path}/part-0001.jsonl"
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id="deleting-task",
                task_name="deleting-task",
                input_path="/managed/source-a.jsonl",
                output_path=shared_path,
                llm_config={},
                steps_config={},
                status=GenerationStatus.DELETING_CASCADE,
                user_id="tenant-a",
            )
        )
        session.add(
            GenerationTaskDB(
                task_id="foreign-task",
                task_name="foreign-task",
                input_path="/managed/source-b.jsonl",
                qa_output_path=nested_path,
                llm_config={},
                steps_config={},
                status=GenerationStatus.COMPLETED,
                user_id="tenant-b",
            )
        )
        session.commit()

    consumers = service.list_artifact_reference_consumers(
        [shared_path],
        exclude_task_ids=("deleting-task",),
    )

    assert consumers == [
        {
            "task_id": "foreign-task",
            "field": "qa_output_path",
            "reference": nested_path,
        }
    ]


@pytest.fixture(autouse=True)
def isolate_generation_worker_state():
    generation_routes._running_pipelines.clear()
    generation_routes._running_pipeline_tokens.clear()
    cancellation_requests = getattr(
        generation_routes,
        "_generation_cancellation_requests",
        None,
    )
    if cancellation_requests is not None:
        cancellation_requests.clear()
    yield
    generation_routes._running_pipelines.clear()
    generation_routes._running_pipeline_tokens.clear()
    if cancellation_requests is not None:
        cancellation_requests.clear()


class _RaceTaskService:
    def __init__(self, task_id: str):
        self.run_token = f"run-token-{task_id}"
        self.task = {
            "task_id": task_id,
            "task_name": task_id,
            "status": GenerationStatus.PENDING,
            "user_id": "user-1",
            "input_path": "/managed/input.jsonl",
            "llm_config": {},
            "auto_register_dataset": False,
        }
        self.claim_calls = 0
        self._lock = threading.Lock()

    def get_task(self, _task_id):
        with self._lock:
            return dict(self.task)

    def get_task_raw(self, _task_id):
        with self._lock:
            return {**self.task, "run_token": self.run_token}

    def claim_running(self, _task_id, *, expected_run_token=None):
        with self._lock:
            self.claim_calls += 1
            if (
                expected_run_token is not None
                and expected_run_token != self.run_token
            ):
                return None
            if self.task["status"] != GenerationStatus.PENDING:
                return None
            self.task["status"] = GenerationStatus.RUNNING
            return self.run_token

    def update_status(
        self,
        _task_id,
        status,
        error_message=None,
        *,
        expected_run_token=None,
        **_kwargs,
    ):
        del error_message
        with self._lock:
            if (
                expected_run_token is not None
                and expected_run_token != self.run_token
            ):
                return False
            current = self.task["status"]
            allowed = {
                GenerationStatus.PENDING: {
                    GenerationStatus.RUNNING,
                    GenerationStatus.FAILED,
                    GenerationStatus.STOPPED,
                },
                GenerationStatus.RUNNING: {
                    GenerationStatus.STOPPING,
                    GenerationStatus.PUBLISHING,
                    GenerationStatus.FAILED,
                },
                GenerationStatus.STOPPING: {GenerationStatus.STOPPED},
                GenerationStatus.PUBLISHING: {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                },
                GenerationStatus.RECOVERING: {
                    GenerationStatus.FAILED,
                    GenerationStatus.STOPPED,
                },
                GenerationStatus.STOPPED: {GenerationStatus.PENDING},
            }
            if status != current and status not in allowed.get(current, set()):
                raise ValueError(f"invalid transition: {current} -> {status}")
            self.task["status"] = status
            if status == GenerationStatus.PENDING:
                self.run_token = f"restarted-{self.run_token}"
            return True

    def update_progress(self, *_args, **_kwargs):
        return True

    def complete_publication(
        self,
        _task_id,
        *,
        expected_run_token,
        dataset_bindings,
        artifact_updates,
        **_kwargs,
    ):
        if (
            expected_run_token != self.run_token
            or self.task["status"] != GenerationStatus.PUBLISHING
        ):
            return False
        self.task.update(dataset_bindings)
        self.task.update(artifact_updates)
        self.task["status"] = GenerationStatus.COMPLETED
        return True


class _CasLosingTaskService(_RaceTaskService):
    def __init__(self, task_id: str, losing_target: str, winning_status: str):
        super().__init__(task_id)
        self.losing_target = losing_target
        self.winning_status = winning_status

    def update_status(self, task_id, status, error_message=None, **kwargs):
        if status == self.losing_target:
            with self._lock:
                self.task["status"] = self.winning_status
            return False
        return super().update_status(task_id, status, error_message, **kwargs)


def _patch_worker_dependencies(monkeypatch, service, finalized):
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_refresh_owned_pipeline_model_configs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )
    monkeypatch.setattr(
        generation_routes,
        "get_generation_input_allowed_dirs",
        lambda: ["/managed"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_stopped_generation",
        lambda task_id, reason: finalized.append((task_id, reason)),
        raising=False,
    )


def _patch_pipeline_result(monkeypatch, result):
    class ResultPipeline:
        def __init__(self, *_args, **_kwargs):
            self.stop_called = False

        def stop(self):
            self.stop_called = True

        async def run(self):
            if isinstance(result, BaseException):
                raise result
            return result

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        ResultPipeline,
    )


def test_stop_records_cancellation_without_a_registered_pipeline(monkeypatch):
    task_id = "stop-without-pipeline"
    service = _RaceTaskService(task_id)
    finalized = []
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "_finalize_stopped_generation",
        lambda stopped_task_id, reason: finalized.append((stopped_task_id, reason)),
    )

    asyncio.run(generation_routes.stop_task(task_id, {"user_id": "user-1"}))

    assert hasattr(
        generation_routes,
        "_is_generation_cancellation_requested",
    ), "the stop route must leave a cancellation token for a not-yet-registered worker"
    assert generation_routes._is_generation_cancellation_requested(task_id) is True
    assert service.get_task(task_id)["status"] == GenerationStatus.STOPPED
    assert finalized == [(task_id, "Generation stopped by user")]


@pytest.mark.parametrize(
    ("winning_status", "expect_conflict"),
    (
        (GenerationStatus.STOPPED, False),
        (GenerationStatus.COMPLETED, True),
    ),
)
def test_stop_cas_loss_clears_cancellation_and_reports_winning_status(
    monkeypatch,
    winning_status,
    expect_conflict,
):
    task_id = f"stop-lost-to-{winning_status}"
    service = _CasLosingTaskService(
        task_id,
        GenerationStatus.STOPPING,
        winning_status,
    )
    service.task["status"] = GenerationStatus.RUNNING
    finalized = []
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "_finalize_stopped_generation",
        lambda stopped_task_id, reason: finalized.append((stopped_task_id, reason)),
    )

    if expect_conflict:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                generation_routes.stop_task(task_id, {"user_id": "user-1"})
            )
        assert exc_info.value.status_code == 409
        assert winning_status in str(exc_info.value.detail)
    else:
        assert asyncio.run(
            generation_routes.stop_task(task_id, {"user_id": "user-1"})
        ) == {"status": "stopped"}

    assert generation_routes._is_generation_cancellation_requested(task_id) is False
    assert service.get_task(task_id)["status"] == winning_status
    assert finalized == []


@pytest.mark.parametrize(
    "terminal_status",
    (GenerationStatus.COMPLETED, GenerationStatus.FAILED),
)
def test_stop_rejects_a_task_that_was_already_terminal(
    monkeypatch,
    terminal_status,
):
    task_id = f"already-{terminal_status}"
    service = _RaceTaskService(task_id)
    service.task["status"] = terminal_status
    monkeypatch.setattr(generation_routes, "generation_task_service", service)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(generation_routes.stop_task(task_id, {"user_id": "user-1"}))

    assert exc_info.value.status_code == 409
    assert terminal_status in str(exc_info.value.detail)
    assert generation_routes._is_generation_cancellation_requested(task_id) is False


def test_internal_stop_does_not_finalize_when_cas_loses_to_completion(monkeypatch):
    task_id = "internal-stop-lost"
    service = _CasLosingTaskService(
        task_id,
        GenerationStatus.STOPPING,
        GenerationStatus.COMPLETED,
    )
    service.task["status"] = GenerationStatus.RUNNING
    finalized = []
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "_finalize_stopped_generation",
        lambda stopped_task_id, reason: finalized.append((stopped_task_id, reason)),
    )

    generation_routes._stop_generation_execution(task_id, "worker stopped")

    assert service.get_task(task_id)["status"] == GenerationStatus.COMPLETED
    assert finalized == []


def test_stop_before_running_claim_prevents_worker_start(monkeypatch):
    task_id = "stop-before-claim"
    service = _RaceTaskService(task_id)
    finalized = []
    validation_entered = threading.Event()
    allow_validation_to_finish = threading.Event()
    pipeline_constructions = []
    _patch_worker_dependencies(monkeypatch, service, finalized)

    def pause_during_revalidation(_task):
        validation_entered.set()
        assert allow_validation_to_finish.wait(timeout=5)
        return {}

    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        pause_during_revalidation,
    )

    class ForbiddenPipeline:
        def __init__(self, *_args, **_kwargs):
            pipeline_constructions.append(True)

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        ForbiddenPipeline,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            asyncio.run,
            generation_routes._run_generation_task(
                task_id,
                PipelineConfig(input_path="/managed/input.jsonl"),
            ),
        )
        assert validation_entered.wait(timeout=5)
        try:
            asyncio.run(generation_routes.stop_task(task_id, {"user_id": "user-1"}))
        finally:
            allow_validation_to_finish.set()
        worker.result(timeout=5)

    assert service.claim_calls == 0
    assert pipeline_constructions == []
    assert service.get_task(task_id)["status"] == GenerationStatus.STOPPED
    assert finalized == [
        (task_id, "Generation stopped by user"),
        (task_id, "Generation stopped during worker validation"),
    ]
    assert generation_routes._is_generation_cancellation_requested(task_id) is False


def test_stop_between_claim_and_pipeline_registration_prevents_run(monkeypatch):
    task_id = "stop-before-registration"
    service = _RaceTaskService(task_id)
    finalized = []
    pipeline_constructed = threading.Event()
    allow_pipeline_registration = threading.Event()
    pipelines = []
    _patch_worker_dependencies(monkeypatch, service, finalized)
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )

    class PausingPipeline:
        def __init__(self, *_args, **_kwargs):
            self.stop_called = False
            self.run_called = False
            pipelines.append(self)
            pipeline_constructed.set()
            assert allow_pipeline_registration.wait(timeout=5)

        def stop(self):
            self.stop_called = True

        async def run(self):
            self.run_called = True
            raise AssertionError("a stopped pipeline must not run")

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        PausingPipeline,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            asyncio.run,
            generation_routes._run_generation_task(
                task_id,
                PipelineConfig(input_path="/managed/input.jsonl"),
            ),
        )
        assert pipeline_constructed.wait(timeout=5)
        try:
            assert task_id not in generation_routes._running_pipelines
            asyncio.run(generation_routes.stop_task(task_id, {"user_id": "user-1"}))
        finally:
            allow_pipeline_registration.set()
        worker.result(timeout=5)

    assert service.claim_calls == 1
    assert pipelines[0].stop_called is True
    assert pipelines[0].run_called is False
    assert service.get_task(task_id)["status"] == GenerationStatus.STOPPED
    assert finalized == [
        (task_id, "Generation stopped before pipeline execution"),
    ]
    assert task_id not in generation_routes._running_pipelines
    assert generation_routes._is_generation_cancellation_requested(task_id) is False


def test_worker_completion_cas_loss_skips_sync_success_finalize(monkeypatch):
    task_id = "completion-cas-lost"
    service = _CasLosingTaskService(
        task_id,
        GenerationStatus.PUBLISHING,
        GenerationStatus.STOPPED,
    )
    stopped_finalizations = []
    sync_success_finalizations = []
    sync_failure_finalizations = []
    _patch_worker_dependencies(monkeypatch, service, stopped_finalizations)
    _patch_pipeline_result(
        monkeypatch,
        SimpleNamespace(
            success=True,
            total_docs=0,
            processed_docs=0,
            output_samples=0,
            output_path=None,
            details={},
            error=None,
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_tracking",
        lambda *args, **kwargs: sync_success_finalizations.append((args, kwargs))
        or True,
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_failure_tracking",
        lambda *args, **kwargs: sync_failure_finalizations.append((args, kwargs))
        or True,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
        )
    )

    assert service.get_task(task_id)["status"] == GenerationStatus.STOPPED
    assert sync_success_finalizations == []
    assert sync_failure_finalizations == []
    assert len(stopped_finalizations) == 1
    assert stopped_finalizations[0][0] == task_id


def test_worker_failed_result_cas_loss_skips_sync_retry(monkeypatch):
    task_id = "failed-result-cas-lost"
    service = _CasLosingTaskService(
        task_id,
        GenerationStatus.FAILED,
        GenerationStatus.STOPPED,
    )
    stopped_finalizations = []
    sync_failure_finalizations = []
    _patch_worker_dependencies(monkeypatch, service, stopped_finalizations)
    _patch_pipeline_result(
        monkeypatch,
        SimpleNamespace(
            success=False,
            total_docs=1,
            processed_docs=0,
            output_samples=0,
            output_path=None,
            details={},
            error="pipeline returned failure",
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_failure_tracking",
        lambda *args, **kwargs: sync_failure_finalizations.append((args, kwargs))
        or True,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
        )
    )

    assert service.get_task(task_id)["status"] == GenerationStatus.STOPPED
    assert sync_failure_finalizations == []
    assert len(stopped_finalizations) == 1
    assert stopped_finalizations[0][0] == task_id


def test_worker_exception_cas_loss_skips_sync_retry(monkeypatch):
    task_id = "exception-cas-lost"
    service = _CasLosingTaskService(
        task_id,
        GenerationStatus.FAILED,
        GenerationStatus.STOPPED,
    )
    stopped_finalizations = []
    sync_failure_finalizations = []
    _patch_worker_dependencies(monkeypatch, service, stopped_finalizations)
    _patch_pipeline_result(monkeypatch, RuntimeError("pipeline crashed"))
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_failure_tracking",
        lambda *args, **kwargs: sync_failure_finalizations.append((args, kwargs))
        or True,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
        )
    )

    assert service.get_task(task_id)["status"] == GenerationStatus.STOPPED
    assert sync_failure_finalizations == []
    assert len(stopped_finalizations) == 1
    assert stopped_finalizations[0][0] == task_id


def test_recovery_claim_prevents_old_worker_failure_and_allows_repair(
    monkeypatch,
):
    from train_factory.api import server

    generation_service_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    publication_module = importlib.import_module(
        "train_factory.storage.services.generation_publication_service"
    )
    sync_service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    task_id = "worker-loses-to-recovery"
    service = _RaceTaskService(task_id)
    pipeline_entered = threading.Event()
    release_pipeline = threading.Event()
    sync_failures = []
    compensations = []
    finished_recoveries = []
    _patch_worker_dependencies(monkeypatch, service, [])

    class FailingPipeline:
        def __init__(self, *_args, **_kwargs):
            self.stop_called = False

        def stop(self):
            self.stop_called = True

        async def run(self):
            pipeline_entered.set()
            assert release_pipeline.wait(timeout=5)
            raise RuntimeError("old worker crashed during recovery")

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        FailingPipeline,
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_failure_tracking",
        lambda *_args, **_kwargs: sync_failures.append(True) or True,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            asyncio.run,
            generation_routes._run_generation_task(
                task_id,
                PipelineConfig(input_path="/managed/input.jsonl"),
                expected_run_token=service.run_token,
            ),
        )
        assert pipeline_entered.wait(timeout=5)
        with service._lock:
            assert service.task["status"] == GenerationStatus.RUNNING
            service.task["status"] = GenerationStatus.RECOVERING
        release_pipeline.set()
        worker.result(timeout=5)

    assert service.get_task(task_id)["status"] == GenerationStatus.RECOVERING
    assert sync_failures == []

    def get_all_tasks(*, status, **_kwargs):
        tasks = [service.get_task(task_id)] if status == GenerationStatus.RECOVERING else []
        return tasks, len(tasks)

    def finish_orphan_recovery(
        _task_id,
        *,
        expected_run_token,
        terminal_status,
        **_kwargs,
    ):
        assert expected_run_token == service.run_token
        with service._lock:
            if service.task["status"] != GenerationStatus.RECOVERING:
                return False
            service.task["status"] = terminal_status
        finished_recoveries.append(terminal_status)
        return True

    service.get_all_tasks = get_all_tasks
    service.claim_orphan_recovery = lambda *_args, **_kwargs: pytest.fail(
        "an existing recovery claim must not be reclaimed"
    )
    service.finish_orphan_recovery = finish_orphan_recovery
    monkeypatch.setattr(
        generation_service_module,
        "generation_task_service",
        service,
    )
    monkeypatch.setattr(
        sync_service_module,
        "external_sync_service",
        SimpleNamespace(
            list_pending_generations=lambda **_kwargs: ([], 0),
            fail_generation_and_restore_batches=lambda *_args: {
                "tracking_found": False,
                "recovered": False,
            },
        ),
    )
    monkeypatch.setattr(
        publication_module.generation_publication_service,
        "compensate_attempt",
        lambda **kwargs: compensations.append(kwargs) or True,
    )

    server.cleanup_orphan_generation_tasks()

    assert finished_recoveries == [GenerationStatus.FAILED]
    assert service.get_task(task_id)["status"] == GenerationStatus.FAILED
    assert compensations == [
        {
            "task_id": task_id,
            "expected_run_token": service.run_token,
            "recovery_run_token": service.run_token,
            "user_id": "user-1",
        }
    ]
