import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib
import logging
import sys
import threading
from types import ModuleType, SimpleNamespace
from uuid import UUID

import pytest
from sqlmodel import Session, SQLModel, create_engine

from train_factory.api.routes import deep_evaluation_routes, evaluation_routes
from train_factory.deep_evaluation import deep_evaluation_runner
from train_factory.evaluation import evaluation_runner
from train_factory.storage.entities.evaluation_task_entity import (
    EvaluationFramework,
    EvaluationStatus,
    EvaluationTaskDB,
)
from train_factory.storage.services.background_task_admission_service import (
    BackgroundTaskAdmissionService,
)


@pytest.fixture
def evaluation_services(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'evaluation-cancel-race.db'}")
    SQLModel.metadata.create_all(engine, tables=[EvaluationTaskDB.__table__])

    standard_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    deep_module = importlib.import_module(
        "train_factory.storage.services.deep_evaluation_task_service"
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(standard_module, "get_session", test_session)
    monkeypatch.setattr(deep_module, "get_session", test_session)
    return (
        standard_module.EvaluationTaskService(),
        deep_module.DeepEvaluationTaskService(),
        engine,
    )


def _add_evaluation_task(
    engine,
    task_id: str,
    framework: str,
    status: str,
    *,
    run_token: str | None = None,
) -> None:
    with Session(engine) as session:
        session.add(
            EvaluationTaskDB(
                task_id=task_id,
                task_name=task_id,
                eval_framework=framework,
                model_configs=[],
                dataset_configs=[],
                model_progress={},
                status=status,
                run_token=run_token,
            )
        )
        session.commit()


def test_evaluation_services_claim_each_pending_task_once(evaluation_services):
    standard, deep, engine = evaluation_services
    cases = (
        (standard, "standard-pending", EvaluationFramework.MTEB),
        (deep, "deep-pending", EvaluationFramework.DEEPEVAL),
    )
    for _service, task_id, framework in cases:
        _add_evaluation_task(engine, task_id, framework, EvaluationStatus.PENDING)

    for service, task_id, _framework in cases:
        assert hasattr(service, "claim_running"), "worker startup requires a DB claim"
        assert service.claim_running(task_id) is True
        assert service.claim_running(task_id) is False
        assert service.update_status(task_id, EvaluationStatus.RUNNING) is True
        assert service.get_task(task_id)["status"] == EvaluationStatus.RUNNING


def test_standard_create_and_legacy_claim_assign_uuid_run_tokens(
    evaluation_services,
):
    standard, _deep, engine = evaluation_services
    created = standard.create_task(model_configs=[], dataset_configs=[])
    UUID(created["run_token"])

    _add_evaluation_task(
        engine,
        "legacy-null-token",
        EvaluationFramework.MTEB,
        EvaluationStatus.PENDING,
    )
    assert standard.get_task("legacy-null-token")["run_token"] is None
    assert standard.claim_running("legacy-null-token") is True
    UUID(standard.get_task("legacy-null-token")["run_token"])


def test_standard_cancel_preserves_and_resume_rotates_run_token(
    evaluation_services,
):
    standard, _deep, engine = evaluation_services
    _add_evaluation_task(
        engine,
        "token-lifecycle",
        EvaluationFramework.MTEB,
        EvaluationStatus.RUNNING,
        run_token="old-attempt-token",
    )

    assert standard.cancel_task("token-lifecycle") is True
    assert standard.get_task("token-lifecycle")["run_token"] == "old-attempt-token"
    assert standard.reset_for_resume("token-lifecycle") is True
    pending = standard.get_task("token-lifecycle")
    assert pending["run_token"] != "old-attempt-token"
    UUID(pending["run_token"])
    assert standard.claim_running("token-lifecycle") is True
    assert standard.get_task("token-lifecycle")["run_token"] == pending["run_token"]


def test_stale_attempt_cannot_write_or_complete_resumed_evaluation(
    evaluation_services,
):
    standard, _deep, engine = evaluation_services
    task_id = "stale-evaluation-attempt"
    _add_evaluation_task(
        engine,
        task_id,
        EvaluationFramework.MTEB,
        EvaluationStatus.CANCELLED,
        run_token="old-attempt-token",
    )

    assert standard.reset_for_resume(task_id) is True
    new_token = standard.get_task(task_id)["run_token"]
    assert standard.claim_running(task_id) is True
    assert not standard.save_partial_results(
        task_id,
        {"old-model": {"_error": {"error": "stale"}}},
        run_token="old-attempt-token",
    )
    assert not standard.save_model_dataset_result(
        task_id,
        "old-model",
        "mteb:T2Reranking",
        {"NDCG@10": 0.1},
        run_token="old-attempt-token",
    )
    assert not standard.complete_task(
        task_id,
        EvaluationStatus.FAILED,
        results={"old-model": {"_error": {"error": "stale"}}},
        run_token="old-attempt-token",
    )

    stored = standard.get_task(task_id)
    assert stored["run_token"] == new_token
    assert stored["status"] == EvaluationStatus.RUNNING
    assert stored["results"] is None


@pytest.mark.parametrize(
    "target_status",
    (
        EvaluationStatus.RUNNING,
        EvaluationStatus.FAILED,
        EvaluationStatus.CANCELLED,
        EvaluationStatus.COMPLETED,
        EvaluationStatus.SUCCEEDED,
    ),
)
def test_explicit_run_token_fences_every_standard_status_transition(
    evaluation_services,
    target_status,
):
    standard, _deep, engine = evaluation_services
    task_id = f"status-token-fence-{target_status}"
    _add_evaluation_task(
        engine,
        task_id,
        EvaluationFramework.MTEB,
        EvaluationStatus.RUNNING,
        run_token="current-attempt-token",
    )

    assert not standard.update_status(
        task_id,
        target_status,
        run_token="stale-attempt-token",
    )
    assert standard.get_task(task_id)["status"] == EvaluationStatus.RUNNING

    assert standard.update_status(
        task_id,
        target_status,
        run_token="current-attempt-token",
    )
    assert standard.get_task(task_id)["status"] == target_status


def test_resume_preserves_success_committed_after_route_snapshot(
    evaluation_services,
):
    standard, _deep, engine = evaluation_services
    task_id = "late-success-before-resume-cas"
    _add_evaluation_task(
        engine,
        task_id,
        EvaluationFramework.MTEB,
        EvaluationStatus.CANCELLED,
        run_token="old-attempt-token",
    )
    stale_route_results = {}
    canonical_progress = {
        "model-one": {
            "mteb:T2Reranking": {"progress": 0, "status": "pending"}
        }
    }

    assert standard.save_partial_results(
        task_id,
        {
            "model-one": {
                "mteb:T2Reranking": {"NDCG@10": 0.75},
            }
        },
        run_token="old-attempt-token",
    )
    assert standard.reset_for_resume(
        task_id,
        results=stale_route_results,
        model_progress=canonical_progress,
    )

    stored = standard.get_task(task_id)
    assert stored["results"] == {
        "model-one": {"mteb:T2Reranking": {"NDCG@10": 0.75}}
    }
    assert stored["model_progress"] == {
        "model-one": {
            "mteb:T2Reranking": {"progress": 100, "status": "completed"}
        }
    }


def test_evaluation_services_do_not_resurrect_cancelled_tasks(evaluation_services):
    standard, deep, engine = evaluation_services
    cases = (
        (standard, "standard-cancelled", EvaluationFramework.MTEB),
        (deep, "deep-cancelled", EvaluationFramework.DEEPEVAL),
    )
    for _service, task_id, framework in cases:
        _add_evaluation_task(engine, task_id, framework, EvaluationStatus.CANCELLED)

    for service, task_id, _framework in cases:
        assert service.claim_running(task_id) is False
        assert service.update_status(task_id, EvaluationStatus.RUNNING) is False
        assert service.get_task(task_id)["status"] == EvaluationStatus.CANCELLED


def test_evaluation_services_cancel_only_active_tasks(evaluation_services):
    standard, deep, engine = evaluation_services
    cases = (
        (
            standard,
            "standard-pending",
            EvaluationFramework.MTEB,
            EvaluationStatus.PENDING,
        ),
        (
            standard,
            "standard-running",
            EvaluationFramework.MTEB,
            EvaluationStatus.RUNNING,
        ),
        (
            deep,
            "deep-pending",
            EvaluationFramework.DEEPEVAL,
            EvaluationStatus.PENDING,
        ),
        (
            deep,
            "deep-running",
            EvaluationFramework.DEEPEVAL,
            EvaluationStatus.RUNNING,
        ),
    )
    for _service, task_id, framework, status in cases:
        _add_evaluation_task(engine, task_id, framework, status)

    for service, task_id, _framework, _status in cases:
        assert hasattr(service, "cancel_task"), "cancel requires an active-state CAS"
        assert service.cancel_task(task_id) is True
        assert service.cancel_task(task_id) is False
        assert service.get_task(task_id)["status"] == EvaluationStatus.CANCELLED


@pytest.mark.parametrize(
    ("service_index", "framework", "terminal_status"),
    (
        (0, EvaluationFramework.MTEB, EvaluationStatus.SUCCEEDED),
        (1, EvaluationFramework.DEEPEVAL, EvaluationStatus.COMPLETED),
    ),
)
def test_runner_completion_cannot_overwrite_committed_cancellation(
    evaluation_services,
    service_index,
    framework,
    terminal_status,
):
    services = evaluation_services[:2]
    engine = evaluation_services[2]
    service = services[service_index]
    task_id = f"{framework}-cancel-before-completion"
    _add_evaluation_task(engine, task_id, framework, EvaluationStatus.RUNNING)

    # Reproduce the runner/cancel ordering: evaluation work has finished, but
    # the cancel request commits before the runner attempts its terminal write.
    assert service.cancel_task(task_id) is True
    assert (
        service.complete_task(
            task_id,
            terminal_status,
            results={"metric": 1.0},
            report_path="/tmp/must-not-be-persisted.json",
        )
        is False
    )

    task = service.get_task(task_id)
    assert task["status"] == EvaluationStatus.CANCELLED
    assert task.get("results") is None
    assert task.get("results_path") is None


def test_standard_resume_returns_to_pending_for_worker_claim(evaluation_services):
    standard, _deep, engine = evaluation_services
    task_id = "standard-resume"
    _add_evaluation_task(
        engine,
        task_id,
        EvaluationFramework.MTEB,
        EvaluationStatus.FAILED,
    )

    assert standard.reset_for_resume(task_id) is True
    assert standard.get_task(task_id)["status"] == EvaluationStatus.PENDING


class _RouteRaceService:
    def __init__(
        self,
        task_id: str,
        claim_entered: threading.Event | None = None,
        release_claim: threading.Event | None = None,
    ):
        self.task = {
            "task_id": task_id,
            "task_name": task_id,
            "status": EvaluationStatus.PENDING,
            "user_id": "user-1",
        }
        self.claim_entered = claim_entered
        self.release_claim = release_claim
        self._lock = threading.Lock()

    def get_task(self, _task_id, **_kwargs):
        with self._lock:
            return dict(self.task)

    def claim_running(self, _task_id, *, run_token=None):
        with self._lock:
            if self.task["status"] != EvaluationStatus.PENDING:
                return False
            self.task["status"] = EvaluationStatus.RUNNING
            self.task["run_token"] = run_token
        if self.claim_entered is not None:
            self.claim_entered.set()
        if self.release_claim is not None:
            assert self.release_claim.wait(timeout=5)
        return True

    def cancel_task(self, _task_id):
        with self._lock:
            if self.task["status"] not in {
                EvaluationStatus.PENDING,
                EvaluationStatus.RUNNING,
            }:
                return False
            self.task["status"] = EvaluationStatus.CANCELLED
            return True

    def update_status(self, _task_id, status, _error_message=None):
        if status == EvaluationStatus.CANCELLED:
            return self.cancel_task(_task_id)
        with self._lock:
            self.task["status"] = status
            return True


_ROUTE_CASES = {
    "standard": {
        "module": evaluation_routes,
        "service_attr": "evaluation_task_service",
        "cancel_route": "cancel_evaluation_task",
        "signal_attr": "cancel_evaluation",
        "clear_attr": "clear_evaluation_cancellation",
        "runner_attr": "run_evaluation_task",
        "wrapper_attr": "_run_claimed_evaluation_task",
    },
    "deep": {
        "module": deep_evaluation_routes,
        "service_attr": "deep_evaluation_task_service",
        "cancel_route": "cancel_deep_evaluation_task",
        "signal_attr": "cancel_deep_evaluation",
        "clear_attr": "clear_deep_cancellation",
        "runner_attr": "run_deep_evaluation_task",
        "wrapper_attr": "_run_claimed_deep_evaluation_task",
    },
}


def _patch_route_runtime(monkeypatch, case, service, runner_calls, cancelled):
    module = case["module"]
    monkeypatch.setattr(module, case["service_attr"], service)
    monkeypatch.setattr(
        module,
        case["signal_attr"],
        lambda task_id: cancelled.add(task_id),
    )
    monkeypatch.setattr(
        module,
        case["clear_attr"],
        lambda task_id: cancelled.discard(task_id),
    )
    monkeypatch.setattr(
        module,
        case["runner_attr"],
        lambda task_id, *args, **kwargs: runner_calls.append(
            (task_id, args, kwargs)
        ),
    )


def _call_route_wrapper(case, task_id: str):
    wrapper = getattr(case["module"], case["wrapper_attr"])
    if case["module"] is evaluation_routes:
        return wrapper(task_id, {"model_configs": [], "dataset_configs": []})
    return wrapper(task_id, existing_results={})


@pytest.mark.parametrize("case_name", ("standard", "deep"))
@pytest.mark.parametrize("concurrent_change", (None, "cancelled", "new_attempt"))
def test_claimed_startup_read_failure_finalizes_only_its_running_attempt(
    case_name, concurrent_change, evaluation_services, monkeypatch,
):
    standard, deep, engine = evaluation_services
    service = standard if case_name == "standard" else deep
    framework = EvaluationFramework.MTEB if case_name == "standard" else EvaluationFramework.DEEPEVAL
    case = _ROUTE_CASES[case_name]
    task_id = f"startup-{case_name}-{concurrent_change}"
    _add_evaluation_task(engine, task_id, framework, EvaluationStatus.PENDING)
    runner_calls = []
    cancelled = set()
    _patch_route_runtime(monkeypatch, case, service, runner_calls, cancelled)
    real_get_task = service.get_task

    def fail_after_claim(_task_id, **_kwargs):
        assert real_get_task(task_id)["status"] == EvaluationStatus.RUNNING
        if concurrent_change == "cancelled":
            assert service.cancel_task(task_id)
        elif concurrent_change == "new_attempt":
            # Another worker owns a later attempt before this worker handles
            # its failed read; the old worker must not fail that attempt.
            from sqlalchemy import update
            with Session(engine) as session:
                session.exec(update(EvaluationTaskDB).where(
                    EvaluationTaskDB.task_id == task_id,
                ).values(run_token="later-worker-token"))
                session.commit()
        raise RuntimeError("transient startup read failure")

    monkeypatch.setattr(service, "get_task", fail_after_claim)
    admission = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(admission, "_active_task_owners", lambda: {})
    _result, lease = admission.admit_execution(
        "evaluation", task_id, "user-1", lambda: True,
    )

    with pytest.raises(RuntimeError, match="transient startup read failure"):
        admission.run_sync(lease, _call_route_wrapper, case, task_id)

    stored = real_get_task(task_id)
    assert not admission.is_executing("evaluation", task_id)
    expected = {
        None: EvaluationStatus.FAILED,
        "cancelled": EvaluationStatus.CANCELLED,
        "new_attempt": EvaluationStatus.RUNNING,
    }[concurrent_change]
    assert stored["status"] == expected
    assert runner_calls == []
    if concurrent_change is None:
        assert stored["completed_at"] is not None
        assert stored["error_message"]
    else:
        assert stored["error_message"] is None


@pytest.mark.parametrize("case_name", ("standard", "deep"))
@pytest.mark.parametrize("failure_phase", ("claim", "runner", "cleanup"))
def test_startup_failure_handling_preserves_exception_and_runner_boundary(
    case_name, failure_phase, evaluation_services, monkeypatch,
):
    standard, deep, engine = evaluation_services
    service = standard if case_name == "standard" else deep
    framework = EvaluationFramework.MTEB if case_name == "standard" else EvaluationFramework.DEEPEVAL
    case = _ROUTE_CASES[case_name]
    task_id = f"boundary-{case_name}-{failure_phase}"
    _add_evaluation_task(engine, task_id, framework, EvaluationStatus.PENDING)
    _patch_route_runtime(monkeypatch, case, service, [], set())
    real_get_task = service.get_task

    def original_failure(*_args, **_kwargs):
        raise RuntimeError("original operation failed")

    if failure_phase == "claim":
        monkeypatch.setattr(service, "claim_running", original_failure)
    elif failure_phase == "runner":
        monkeypatch.setattr(case["module"], case["runner_attr"], original_failure)
    else:
        monkeypatch.setattr(service, "get_task", original_failure)

        def cleanup_failure(*_args, **_kwargs):
            raise OSError("database remains unavailable")

        monkeypatch.setattr(service, "fail_claimed_startup", cleanup_failure)

    with pytest.raises(RuntimeError, match="original operation failed"):
        _call_route_wrapper(case, task_id)

    expected = EvaluationStatus.PENDING if failure_phase == "claim" else EvaluationStatus.RUNNING
    assert real_get_task(task_id)["status"] == expected


@pytest.mark.parametrize("case_name", ("standard", "deep"))
@pytest.mark.parametrize("concurrent_change", (None, "cancelled", "new_attempt", "new_attempt_visible"))
def test_real_runner_first_read_failure_uses_claim_token_without_adopting_later_attempt(
    case_name, concurrent_change, evaluation_services, monkeypatch,
):
    standard, deep, engine = evaluation_services
    service = standard if case_name == "standard" else deep
    framework = EvaluationFramework.MTEB if case_name == "standard" else EvaluationFramework.DEEPEVAL
    case = _ROUTE_CASES[case_name]
    task_id = f"runner-first-read-{case_name}-{concurrent_change}"
    _add_evaluation_task(engine, task_id, framework, EvaluationStatus.PENDING)
    monkeypatch.setattr(case["module"], case["service_attr"], service)
    service_module = importlib.import_module("train_factory.storage.services.evaluation_task_service")
    monkeypatch.setattr(service_module, "evaluation_task_service", standard)
    monkeypatch.setattr(deep_evaluation_runner, "deep_evaluation_task_service", deep)
    monkeypatch.setattr(evaluation_routes, "is_evaluation_cancelled", lambda _id: False)
    monkeypatch.setattr(evaluation_runner, "is_cancelled", lambda _id: False)
    monkeypatch.setattr(deep_evaluation_runner, "is_cancelled", lambda _id: False)
    real_get_task = service.get_task
    reads = 0

    def fail_runner_read(_task_id, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            if concurrent_change == "cancelled":
                assert service.cancel_task(task_id)
            elif concurrent_change in {"new_attempt", "new_attempt_visible"}:
                from sqlalchemy import update
                with Session(engine) as session:
                    session.exec(update(EvaluationTaskDB).where(
                        EvaluationTaskDB.task_id == task_id,
                    ).values(run_token="later-worker-token"))
                    session.commit()
            if concurrent_change != "new_attempt_visible":
                raise RuntimeError("runner initial database read failed")
        return real_get_task(_task_id, **kwargs)

    monkeypatch.setattr(service, "get_task", fail_runner_read)
    _call_route_wrapper(case, task_id)

    stored = real_get_task(task_id)
    expected = {
        None: EvaluationStatus.FAILED,
        "cancelled": EvaluationStatus.CANCELLED,
        "new_attempt": EvaluationStatus.RUNNING,
        "new_attempt_visible": EvaluationStatus.RUNNING,
    }[concurrent_change]
    assert reads == 2
    assert stored["status"] == expected
    if concurrent_change is None:
        assert stored["completed_at"] is not None
        assert "runner initial database read failed" in stored["error_message"]
    else:
        assert stored["error_message"] is None


@pytest.mark.parametrize("case_name", ("standard", "deep"))
def test_pending_cancel_prevents_evaluation_runner_start(
    case_name,
    monkeypatch,
):
    case = _ROUTE_CASES[case_name]
    task_id = f"{case_name}-pending-cancel"
    service = _RouteRaceService(task_id)
    runner_calls = []
    cancelled = set()
    _patch_route_runtime(monkeypatch, case, service, runner_calls, cancelled)

    cancel_route = getattr(case["module"], case["cancel_route"])
    asyncio.run(cancel_route(task_id, {"user_id": "user-1"}))

    assert hasattr(
        case["module"],
        case["wrapper_attr"],
    ), "background scheduling requires a claim-aware wrapper"
    _call_route_wrapper(case, task_id)

    assert runner_calls == []
    assert service.get_task(task_id)["status"] == EvaluationStatus.CANCELLED
    assert task_id not in cancelled


def test_standard_wrapper_rechecks_cancel_signal_after_claim(monkeypatch):
    task_id = "standard-signalled-after-claim"
    service = _RouteRaceService(task_id)
    runner_calls = []
    cancelled = set()
    case = _ROUTE_CASES["standard"]
    _patch_route_runtime(monkeypatch, case, service, runner_calls, cancelled)
    monkeypatch.setattr(
        evaluation_routes,
        "is_evaluation_cancelled",
        lambda _task_id: True,
        raising=False,
    )

    _call_route_wrapper(case, task_id)

    assert runner_calls == []
    assert service.get_task(task_id)["status"] == EvaluationStatus.RUNNING


@pytest.mark.parametrize("case_name", ("standard", "deep"))
def test_cancel_after_claim_before_runner_registration_prevents_start(
    case_name,
    monkeypatch,
):
    case = _ROUTE_CASES[case_name]
    task_id = f"{case_name}-cancel-before-runner"
    claim_entered = threading.Event()
    release_claim = threading.Event()
    service = _RouteRaceService(task_id, claim_entered, release_claim)
    runner_calls = []
    cancelled = set()
    _patch_route_runtime(monkeypatch, case, service, runner_calls, cancelled)

    assert hasattr(
        case["module"],
        case["wrapper_attr"],
    ), "background scheduling requires a claim-aware wrapper"

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(_call_route_wrapper, case, task_id)
        assert claim_entered.wait(timeout=5)
        try:
            cancel_route = getattr(case["module"], case["cancel_route"])
            asyncio.run(cancel_route(task_id, {"user_id": "user-1"}))
        finally:
            release_claim.set()
        worker.result(timeout=5)

    assert runner_calls == []
    assert service.get_task(task_id)["status"] == EvaluationStatus.CANCELLED
    assert task_id not in cancelled


class _TerminalRaceService:
    def __init__(self, task):
        self.task = dict(task)
        self.completion_calls = []

    def get_task(self, _task_id, **_kwargs):
        return dict(self.task)

    def update_status(self, _task_id, status, *_args, **_kwargs):
        # This models the old read/modify/write terminal path: a cancellation
        # committed after the runner's final check could still be overwritten.
        if status != EvaluationStatus.RUNNING:
            self.task["status"] = status
        return True

    def init_model_progress(self, *_args, **_kwargs):
        return True

    def update_model_progress(self, *_args, **_kwargs):
        return True

    def update_progress(self, *_args, **_kwargs):
        return True

    def save_partial_results(self, *_args, **_kwargs):
        return True

    def save_model_dataset_result(
        self,
        _task_id,
        model_name,
        dataset_name,
        result,
        **_kwargs,
    ):
        self.task.setdefault("results", {}).setdefault(model_name, {})[
            dataset_name
        ] = result
        return True

    def update_results(self, *_args, **_kwargs):
        return True

    def complete_task(self, task_id, status, **kwargs):
        self.completion_calls.append((task_id, status, kwargs))
        # Cancellation wins immediately before the conditional terminal write.
        self.task["status"] = EvaluationStatus.CANCELLED
        return False


class _RunnerStartGateService:
    def __init__(self):
        self.cas_entered = threading.Event()
        self.release_cas = threading.Event()
        self.progress_mutations = 0
        self.connection_attempts = 0
        self.outbound_validations = 0
        self._lock = threading.Lock()
        self.task = {
            "task_id": "cancel-during-runner-start",
            "status": EvaluationStatus.RUNNING,
            "run_token": "runner-attempt",
            "user_id": None,
            "model_configs": [
                {
                    "name": "model-one",
                    "model_name": "served-one",
                    "endpoint": "https://one.example.test",
                    "_evaluation_identity": {
                        "schema_version": 2,
                        "result_key": "model-one",
                    },
                }
            ],
            "dataset_configs": [
                {
                    "type": "mteb",
                    "name": "T2Reranking",
                    "_evaluation_identity": {
                        "schema_version": 2,
                        "result_key": "mteb:T2Reranking",
                    },
                }
            ],
            "results": {},
            "model_progress": {},
            "max_samples": 1,
            "batch_size": 1,
            "workers": 1,
            "model_workers": 1,
        }

    def get_task(self, _task_id, **_kwargs):
        with self._lock:
            return dict(self.task)

    def update_status(self, _task_id, status, *_args, **_kwargs):
        assert status == EvaluationStatus.RUNNING
        self.cas_entered.set()
        assert self.release_cas.wait(timeout=5)
        with self._lock:
            return self.task["status"] == EvaluationStatus.RUNNING

    def cancel_task(self):
        with self._lock:
            self.task["status"] = EvaluationStatus.CANCELLED

    def init_model_progress(self, *_args, **_kwargs):
        self.progress_mutations += 1
        return True

    def update_model_progress(self, *_args, **_kwargs):
        self.progress_mutations += 1
        return True

    def save_partial_results(self, *_args, **_kwargs):
        return False

    def complete_task(self, *_args, **_kwargs):
        return False


def test_runner_honors_running_cas_before_progress_or_outbound_io(
    monkeypatch,
):
    service = _RunnerStartGateService()
    service_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    monkeypatch.setattr(service_module, "evaluation_task_service", service)
    monkeypatch.setattr(
        evaluation_runner,
        "is_cancelled",
        lambda _task_id: service.get_task(_task_id)["status"]
        == EvaluationStatus.CANCELLED,
    )
    monkeypatch.setattr(evaluation_runner, "_cleanup_cancelled", lambda _task_id: None)
    monkeypatch.setattr(
        evaluation_runner,
        "task_requires_dataset_provenance",
        lambda _user_id: False,
    )
    def validate_endpoint(endpoint, _user_id):
        service.outbound_validations += 1
        return endpoint

    monkeypatch.setattr(
        evaluation_runner,
        "validate_user_outbound_url",
        validate_endpoint,
    )

    class FakeReranker:
        def __init__(self, *_args, **_kwargs):
            service.connection_attempts += 1

        def test_connection(self):
            return True

    monkeypatch.setattr(evaluation_runner, "SecureAPIReranker", FakeReranker)

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            evaluation_runner.run_evaluation_task,
            service.task["task_id"],
            {},
        )
        assert service.cas_entered.wait(timeout=5)
        service.cancel_task()
        service.release_cas.set()
        worker.result(timeout=5)

    assert service.progress_mutations == 0
    assert service.connection_attempts == 0
    assert service.outbound_validations == 0


def test_runner_aborts_immediately_for_cancelled_persisted_task(monkeypatch):
    service = _RunnerStartGateService()
    service.task["status"] = EvaluationStatus.CANCELLED
    service.release_cas.set()
    service_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    monkeypatch.setattr(service_module, "evaluation_task_service", service)
    monkeypatch.setattr(evaluation_runner, "is_cancelled", lambda _task_id: True)
    monkeypatch.setattr(evaluation_runner, "_cleanup_cancelled", lambda _task_id: None)
    monkeypatch.setattr(
        evaluation_runner,
        "task_requires_dataset_provenance",
        lambda _user_id: False,
    )
    def validate_endpoint(endpoint, _user_id):
        service.outbound_validations += 1
        return endpoint

    monkeypatch.setattr(
        evaluation_runner,
        "validate_user_outbound_url",
        validate_endpoint,
    )

    class FakeReranker:
        def __init__(self, *_args, **_kwargs):
            service.connection_attempts += 1

        def test_connection(self):
            return True

    monkeypatch.setattr(evaluation_runner, "SecureAPIReranker", FakeReranker)

    evaluation_runner.run_evaluation_task(service.task["task_id"], {})

    assert not service.cas_entered.is_set()
    assert service.progress_mutations == 0
    assert service.connection_attempts == 0
    assert service.outbound_validations == 0


def test_standard_runner_discards_completion_when_cancel_wins_final_cas(
    monkeypatch,
    caplog,
):
    task_id = "standard-final-cancel-race"
    service = _TerminalRaceService(
        {
            "task_id": task_id,
            "status": EvaluationStatus.RUNNING,
            "user_id": None,
            "model_configs": [
                {
                    "name": "model-1",
                    "model_name": "model-1",
                    "endpoint": "http://127.0.0.1:9999",
                    "inference_framework": "xinference",
                }
            ],
            "dataset_configs": [{"type": "mteb", "name": "T2Reranking"}],
            "max_samples": 1,
            "batch_size": 1,
            "workers": 1,
            "model_workers": 1,
        }
    )
    service_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    monkeypatch.setattr(service_module, "evaluation_task_service", service)
    monkeypatch.setattr(evaluation_runner, "is_cancelled", lambda _task_id: False)
    monkeypatch.setattr(evaluation_runner, "_cleanup_cancelled", lambda _task_id: None)
    monkeypatch.setattr(
        evaluation_runner,
        "task_requires_dataset_provenance",
        lambda _user_id: False,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )

    class FakeReranker:
        def __init__(self, *_args, **_kwargs):
            pass

        def test_connection(self):
            return True

    class FakeEvaluator:
        def __init__(self, *_args, **_kwargs):
            pass

        def evaluate(self, **_kwargs):
            return {"NDCG@10": 1.0}

    fake_qwen_evaluation = ModuleType("qwen3_rerank_trainer.evaluation")
    fake_qwen_evaluation.MTEBRerankEvaluator = FakeEvaluator
    monkeypatch.setitem(
        sys.modules,
        "qwen3_rerank_trainer.evaluation",
        fake_qwen_evaluation,
    )
    monkeypatch.setattr(evaluation_runner, "SecureAPIReranker", FakeReranker)

    with caplog.at_level(logging.INFO):
        evaluation_runner.run_evaluation_task(task_id, {})

    assert len(service.completion_calls) == 1
    assert service.task["status"] == EvaluationStatus.CANCELLED
    assert "Discarding completion for inactive evaluation task" in caplog.text
    assert "completed successfully" not in caplog.text


def test_deep_runner_discards_completion_when_cancel_wins_final_cas(
    monkeypatch,
    caplog,
    tmp_path,
):
    task_id = "deep-final-cancel-race"
    service = _TerminalRaceService(
        {
            "task_id": task_id,
            "status": EvaluationStatus.RUNNING,
            "user_id": None,
            "model_configs": [
                {
                    "group_name": "group-1",
                    "embedding": {
                        "endpoint": "http://127.0.0.1:9999",
                        "model_name": "model-1",
                    },
                }
            ],
            "dataset_configs": [{"dataset_id": "dataset-1"}],
            "field_mapping": {},
            "metrics": ["mrr"],
            "worker_groups": {"model_workers": 1},
            "max_samples": 1,
        }
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "deep_evaluation_task_service",
        service,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "validate_deep_evaluation_resource_config",
        lambda _task: None,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_require_task_collection_ownership",
        lambda _task: None,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_resolve_task_dataset_records",
        lambda *_args, **_kwargs: [
            {
                "dataset_id": "dataset-1",
                "dataset_name": "dataset-1",
                "storage_path": "/managed/dataset-1.jsonl",
            }
        ],
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_resolve_dataset_source",
        lambda *_args, **_kwargs: ("/managed/dataset-1.jsonl", "jsonl"),
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_iter_rows",
        lambda *_args, **_kwargs: iter(
            [{"query": "q", "positives": ["p"], "negatives": ["n"]}]
        ),
    )
    monkeypatch.setattr(deep_evaluation_runner, "is_cancelled", lambda _task_id: False)
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_cleanup_cancelled",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "get_settings",
        lambda: SimpleNamespace(output_dir=tmp_path),
    )
    existing_results = {
        "group-1": {
            "retrieval": {
                "embedding": {
                    "summary": {
                        "datasets": {
                            "dataset-1": {
                                "summary": {
                                    "metrics": {
                                        "mrr": {
                                            "mean": 1.0,
                                            "min": 1.0,
                                            "max": 1.0,
                                            "count": 1,
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "llm": {},
        }
    }

    with caplog.at_level(logging.INFO):
        deep_evaluation_runner.run_deep_evaluation_task(
            task_id,
            existing_results=existing_results,
        )

    assert len(service.completion_calls) == 1
    assert service.task["status"] == EvaluationStatus.CANCELLED
    assert "Discarding completion for inactive deep evaluation task" in caplog.text
