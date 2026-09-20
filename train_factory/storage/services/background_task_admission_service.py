"""Admission control for long-running training, evaluation, and generation jobs."""

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from typing import Any, Callable, Optional, TypeVar

from sqlmodel import select

from ...config import settings
from ..database import get_session
from ..entities.evaluation_task_entity import EvaluationTaskDB
from ..entities.generation_task_entity import GenerationTaskDB
from ..entities.training_task_entity import TrainingTaskDB


T = TypeVar("T")
_ACTIVE_STATUSES = ("pending", "running")
_GENERATION_ACTIVE_STATUSES = (
    "pending",
    "running",
    "stopping",
    "publishing",
    "recovering",
    "restarting",
)
_TRAINING_ACTIVE_STATUSES = ("pending", "preparing", "running", "evaluating")
_TASK_KINDS = {"evaluation", "generation", "training"}


class BackgroundTaskCapacityExceeded(ValueError):
    """Raised when another long-running API task cannot be admitted."""


class BackgroundTaskAlreadyExecuting(ValueError):
    """Raised when a cancelled task still has a live worker."""


class BackgroundTaskExecutionLease:
    """Idempotently remove a worker from the live-execution registry."""

    def __init__(self, service: "BackgroundTaskAdmissionService", task_key: str):
        self._service = service
        self._task_key = task_key
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._service._finish_execution(self._task_key)


class BackgroundTaskDeletionGuard:
    """Idempotently unblock a task after its delete attempt finishes."""

    def __init__(self, service: "BackgroundTaskAdmissionService", task_key: str):
        self._service = service
        self._task_key = task_key
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._service._finish_deletion(self._task_key)


class BackgroundTaskAdmissionService:
    """Serialize capacity checks with the database transition they protect."""

    def __init__(
        self,
        *,
        global_limit: Optional[int] = None,
        per_user_limit: Optional[int] = None,
    ) -> None:
        self._global_limit = global_limit
        self._per_user_limit = per_user_limit
        self._lock = threading.RLock()
        self._executing: dict[str, Optional[str]] = {}
        self._deleting: set[str] = set()
        self._async_executor: Optional[ThreadPoolExecutor] = None
        self._async_workers_stopped = False

    def _active_task_owners(self) -> dict[str, Optional[str]]:
        with get_session() as session:
            evaluation_rows = session.exec(
                select(EvaluationTaskDB.task_id, EvaluationTaskDB.user_id)
                .where(EvaluationTaskDB.status.in_(_ACTIVE_STATUSES))
            ).all()
            generation_rows = session.exec(
                select(GenerationTaskDB.task_id, GenerationTaskDB.user_id)
                .where(GenerationTaskDB.status.in_(_GENERATION_ACTIVE_STATUSES))
            ).all()
            training_rows = session.exec(
                select(TrainingTaskDB.task_id, TrainingTaskDB.user_id)
                .where(TrainingTaskDB.status.in_(_TRAINING_ACTIVE_STATUSES))
            ).all()

        owners = {
            f"evaluation:{task_id}": owner for task_id, owner in evaluation_rows
        }
        owners.update(
            {
                f"generation:{task_id}": owner
                for task_id, owner in generation_rows
            }
        )
        owners.update(
            {
                f"training:{task_id}": owner
                for task_id, owner in training_rows
            }
        )
        return owners

    def _active_counts(self, user_id: Optional[str]) -> tuple[int, int]:
        with self._lock:
            owners = self._active_task_owners()
            owners.update(self._executing)
            counts = Counter(owners.values())
            return len(owners), counts[user_id]

    def _limits(self) -> tuple[int, int]:
        global_limit = (
            self._global_limit
            if self._global_limit is not None
            else settings.background_task_max_active_global
        )
        per_user_limit = (
            self._per_user_limit
            if self._per_user_limit is not None
            else settings.background_task_max_active_per_user
        )
        return global_limit, per_user_limit

    def _assert_capacity(self, user_id: Optional[str]) -> None:
        global_limit, per_user_limit = self._limits()
        active_global, active_user = self._active_counts(user_id)
        if active_global >= global_limit:
            raise BackgroundTaskCapacityExceeded(
                "Background task global active limit exceeded"
            )
        if active_user >= per_user_limit:
            raise BackgroundTaskCapacityExceeded(
                "Background task per-user active limit exceeded"
            )

    def admit(
        self,
        admission_user_id: Optional[str],
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run one DB create/CAS only when active-task capacity is available."""
        with self._lock:
            self._assert_capacity(admission_user_id)
            return operation(*args, **kwargs)

    def assert_capacity_available(self, admission_user_id: Optional[str]) -> None:
        """Fail fast before expensive preparation; final admission is authoritative."""
        with self._lock:
            self._assert_capacity(admission_user_id)

    def admit_execution(
        self,
        task_kind: str,
        requested_task_id: Optional[str],
        admission_user_id: Optional[str],
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[T, Optional[BackgroundTaskExecutionLease]]:
        """Admit a DB transition and reserve its live worker until exit."""
        if task_kind not in _TASK_KINDS:
            raise ValueError(f"Unsupported background task kind: {task_kind}")
        requested_key = (
            f"{task_kind}:{requested_task_id}" if requested_task_id else None
        )
        with self._lock:
            if requested_key and (
                requested_key in self._executing
                or requested_key in self._deleting
            ):
                raise BackgroundTaskAlreadyExecuting(
                    "Background task is already executing or being deleted"
                )
            self._assert_capacity(admission_user_id)
            result = operation(*args, **kwargs)
            if result is False or result is None:
                return result, None
            resolved_task_id = requested_task_id
            if resolved_task_id is None and isinstance(result, dict):
                resolved_task_id = result.get("task_id")
            if not resolved_task_id:
                raise ValueError("Admitted background operation did not return a task ID")
            task_key = f"{task_kind}:{resolved_task_id}"
            if task_key in self._executing or task_key in self._deleting:
                raise BackgroundTaskAlreadyExecuting(
                    "Background task is already executing or being deleted"
                )
            self._executing[task_key] = admission_user_id
            return result, BackgroundTaskExecutionLease(self, task_key)

    def _finish_execution(self, task_key: str) -> None:
        with self._lock:
            self._executing.pop(task_key, None)

    def is_executing(self, task_kind: str, task_id: str) -> bool:
        """Return whether this process still has a live worker for the task."""
        if task_kind not in _TASK_KINDS:
            raise ValueError(f"Unsupported background task kind: {task_kind}")
        with self._lock:
            return f"{task_kind}:{task_id}" in self._executing

    def get_executing_task_ids(self, task_kind: str) -> set[str]:
        """Return a thread-safe snapshot of live worker task IDs for one kind."""
        if task_kind not in _TASK_KINDS:
            raise ValueError(f"Unsupported background task kind: {task_kind}")
        prefix = f"{task_kind}:"
        with self._lock:
            return {
                task_key[len(prefix) :]
                for task_key in self._executing
                if task_key.startswith(prefix)
            }

    def begin_deletion(
        self,
        task_kind: str,
        task_id: str,
    ) -> BackgroundTaskDeletionGuard:
        """Atomically block a task from starting while it is being deleted."""
        if task_kind not in _TASK_KINDS:
            raise ValueError(f"Unsupported background task kind: {task_kind}")
        task_key = f"{task_kind}:{task_id}"
        with self._lock:
            if task_key in self._executing or task_key in self._deleting:
                raise BackgroundTaskAlreadyExecuting(
                    "Background task is already executing or being deleted"
                )
            self._deleting.add(task_key)
        return BackgroundTaskDeletionGuard(self, task_key)

    def _finish_deletion(self, task_key: str) -> None:
        with self._lock:
            self._deleting.discard(task_key)

    @staticmethod
    def run_sync(
        lease: BackgroundTaskExecutionLease,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        try:
            return operation(*args, **kwargs)
        finally:
            lease.release()

    @staticmethod
    def run_with_deletion_guard(
        guard: BackgroundTaskDeletionGuard,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run maintenance work while blocking execution and deletion."""
        try:
            return operation(*args, **kwargs)
        finally:
            guard.release()

    def start_async_workers(self) -> None:
        """Allow submission when an application lifespan starts."""
        with self._lock:
            if self._async_workers_stopped and self._async_executor is not None:
                raise RuntimeError("Background workers are still shutting down")
            self._async_workers_stopped = False

    def shutdown_async_workers(self) -> None:
        """Cancel queued work and join running workers during app shutdown."""
        with self._lock:
            self._async_workers_stopped = True
            executor = self._async_executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            with self._lock:
                if self._async_executor is executor:
                    self._async_executor = None

    async def run_async(
        self,
        lease: BackgroundTaskExecutionLease,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a generation coroutine on a worker-owned event loop.

        Pipelines also perform synchronous database, file and Milvus work. A
        BackgroundTask coroutine alone would run all of that on the API loop.
        Keep the execution lease until the real worker exits, even if its API
        waiter is cancelled; queued workers cancelled before start must not run.
        """
        state_lock = threading.Lock()
        started = False
        abandoned = False

        def execute():
            nonlocal started
            with state_lock:
                if abandoned:
                    return None
                started = True
            try:
                return asyncio.run(operation(*args, **kwargs))
            finally:
                lease.release()

        try:
            with self._lock:
                if self._async_workers_stopped:
                    raise RuntimeError("Background workers are shutting down")
                if self._async_executor is None:
                    self._async_executor = ThreadPoolExecutor(
                        max_workers=self._limits()[0],
                        thread_name_prefix="generation",
                    )
                # Long jobs have their own bounded pool; deployment stop and
                # other short API operations retain the loop's default pool.
                context = contextvars.copy_context()
                future = self._async_executor.submit(context.run, execute)
            future.add_done_callback(
                lambda completed: lease.release() if completed.cancelled() else None
            )
            return await asyncio.wrap_future(future)
        finally:
            with state_lock:
                if not started:
                    abandoned = True
                    lease.release()


background_task_admission_service = BackgroundTaskAdmissionService()
