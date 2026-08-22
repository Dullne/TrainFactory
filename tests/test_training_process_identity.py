"""Process identity fencing for durable training subprocess cleanup."""

import asyncio
from contextlib import contextmanager
import importlib
import sys
from types import ModuleType, SimpleNamespace

import psutil
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateTable
from sqlmodel import Session, SQLModel, create_engine

from train_factory.core.process_identity import (
    capture_process_create_time,
    terminate_process_if_matches,
)

from train_factory.api import server
from train_factory.api.routes import training_routes
from train_factory.storage.entities.training_task_entity import TrainingTaskDB


def _import_migration(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "alembic":
            raise
    alembic_stub = ModuleType("alembic")
    alembic_stub.op = object()
    sys.modules["alembic"] = alembic_stub
    try:
        return importlib.import_module(module_name)
    finally:
        sys.modules.pop("alembic", None)


class _FakeProcess:
    def __init__(
        self,
        *,
        create_time: float = 1234.5,
        wait_effects: tuple[object, ...] = (0,),
        terminate_effect: Exception | None = None,
        kill_effect: Exception | None = None,
    ) -> None:
        self._create_time = create_time
        self._wait_effects = list(wait_effects)
        self._terminate_effect = terminate_effect
        self._kill_effect = kill_effect
        self.terminated = 0
        self.killed = 0
        self.wait_timeouts: list[float] = []

    def create_time(self) -> float:
        return self._create_time

    def terminate(self) -> None:
        self.terminated += 1
        if self._terminate_effect is not None:
            raise self._terminate_effect

    def kill(self) -> None:
        self.killed += 1
        if self._kill_effect is not None:
            raise self._kill_effect

    def wait(self, timeout: float) -> int:
        self.wait_timeouts.append(timeout)
        effect = self._wait_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return int(effect)


def _patch_process(monkeypatch, fake: _FakeProcess) -> None:
    monkeypatch.setattr(psutil, "Process", lambda _pid: fake)


def test_capture_process_create_time_reads_the_spawned_process(monkeypatch):
    fake = _FakeProcess(create_time=4567.25)
    _patch_process(monkeypatch, fake)

    assert capture_process_create_time(321) == 4567.25


def test_verified_process_identity_is_terminated_and_confirmed(monkeypatch):
    fake = _FakeProcess(create_time=1234.5)
    _patch_process(monkeypatch, fake)

    assert terminate_process_if_matches(321, 1234.5) is True
    assert fake.terminated == 1
    assert fake.killed == 0


def test_reused_pid_is_never_signalled(monkeypatch):
    fake = _FakeProcess(create_time=9999.0)
    _patch_process(monkeypatch, fake)

    assert terminate_process_if_matches(321, 1234.5) is False
    assert fake.terminated == 0
    assert fake.killed == 0


@pytest.mark.parametrize("expected_create_time", [None, 0.0, -1.0])
def test_legacy_or_invalid_identity_is_preserved(monkeypatch, expected_create_time):
    fake = _FakeProcess(create_time=1234.5)
    _patch_process(monkeypatch, fake)

    assert terminate_process_if_matches(321, expected_create_time) is False
    assert fake.terminated == 0


def test_missing_process_is_safe_to_reconcile(monkeypatch):
    def missing(_pid):
        raise psutil.NoSuchProcess(321)

    monkeypatch.setattr(psutil, "Process", missing)

    assert terminate_process_if_matches(321, 1234.5) is True


def test_missing_legacy_process_is_safe_to_reconcile(monkeypatch):
    def missing(_pid):
        raise psutil.NoSuchProcess(321)

    monkeypatch.setattr(psutil, "Process", missing)

    assert terminate_process_if_matches(321, None) is True


def test_access_denied_is_fail_closed(monkeypatch):
    def denied(_pid):
        raise psutil.AccessDenied(321)

    monkeypatch.setattr(psutil, "Process", denied)

    assert terminate_process_if_matches(321, 1234.5) is False


def test_terminate_os_error_is_fail_closed(monkeypatch):
    fake = _FakeProcess(terminate_effect=OSError("operation unavailable"))
    _patch_process(monkeypatch, fake)

    assert terminate_process_if_matches(321, 1234.5) is False
    assert fake.terminated == 1
    assert fake.killed == 0


def test_timeout_escalates_to_cross_platform_kill(monkeypatch):
    fake = _FakeProcess(
        wait_effects=(psutil.TimeoutExpired(1, pid=321), 0),
    )
    _patch_process(monkeypatch, fake)

    assert terminate_process_if_matches(
        321,
        1234.5,
        terminate_timeout=1,
        kill_timeout=2,
    ) is True
    assert fake.terminated == 1
    assert fake.killed == 1
    assert fake.wait_timeouts == [1, 2]


def test_unresponsive_process_keeps_its_durable_evidence(monkeypatch):
    fake = _FakeProcess(
        wait_effects=(
            psutil.TimeoutExpired(1, pid=321),
            psutil.TimeoutExpired(2, pid=321),
        ),
    )
    _patch_process(monkeypatch, fake)

    assert terminate_process_if_matches(321, 1234.5) is False
    assert fake.terminated == 1
    assert fake.killed == 1


def test_nonblocking_stop_only_signals_a_verified_process(monkeypatch):
    fake = _FakeProcess()
    _patch_process(monkeypatch, fake)

    assert terminate_process_if_matches(
        321,
        1234.5,
        wait_for_exit=False,
    ) is True
    assert fake.terminated == 1
    assert fake.killed == 0
    assert fake.wait_timeouts == []


def _install_startup_cleanup_fakes(
    monkeypatch,
    *,
    active_tasks: list[dict] | None = None,
    process_tasks: list[dict] | None = None,
):
    training_service_module = importlib.import_module(
        "train_factory.storage.services.training_task_service"
    )
    gpu_module = importlib.import_module("train_factory.core.gpu_resource_manager")
    active_tasks = list(active_tasks or [])
    process_tasks = list(process_tasks or active_tasks)
    status_updates: list[tuple] = []
    process_updates: list[tuple] = []
    blanket_clears: list[bool] = []
    released: list[str] = []
    force_released: list[str] = []

    class TrainingService:
        @staticmethod
        def get_all_tasks(*, status, limit, offset):
            matching = [task for task in active_tasks if task["status"] == status]
            return matching[offset : offset + limit], len(matching)

        @staticmethod
        def get_tasks_with_process_info(*, limit, offset):
            return process_tasks[offset : offset + limit], len(process_tasks)

        @staticmethod
        def update_task_status(task_id, status, **kwargs):
            status_updates.append((task_id, status, kwargs))
            return True

        @staticmethod
        def update_process_info(task_id, **kwargs):
            process_updates.append((task_id, kwargs))
            return True

        @staticmethod
        def clear_stale_process_info():
            blanket_clears.append(True)
            return 1

    manager = SimpleNamespace(
        release_gpus_for_task=lambda lease_id: released.append(lease_id) or True,
        force_release_gpu_for_task=lambda task_id: force_released.append(task_id)
        or True,
        cleanup_stale_allocations=lambda **_kwargs: 0,
    )
    monkeypatch.setattr(
        training_service_module,
        "training_task_service",
        TrainingService(),
    )
    monkeypatch.setattr(gpu_module, "gpu_resource_manager", manager)
    return status_updates, process_updates, blanket_clears, released, force_released


def _active_task(**overrides) -> dict:
    task = {
        "task_id": "11111111-1111-1111-1111-111111111111",
        "status": "running",
        "user_id": "owner",
        "run_token": "22222222-2222-2222-2222-222222222222",
        "process_pid": 321,
        "process_status": "running",
        "process_create_time": 1234.5,
    }
    task.update(overrides)
    return task


@pytest.mark.parametrize("failure", ["mismatch", "access-denied", "legacy"])
def test_startup_cleanup_preserves_unverified_process_and_gpu_lease(
    monkeypatch,
    failure,
):
    task = _active_task()
    if failure == "mismatch":
        _patch_process(monkeypatch, _FakeProcess(create_time=9999.0))
    elif failure == "access-denied":
        monkeypatch.setattr(
            psutil,
            "Process",
            lambda _pid: (_ for _ in ()).throw(psutil.AccessDenied(321)),
        )
    else:
        task["process_create_time"] = None
        _patch_process(monkeypatch, _FakeProcess())

    status_updates, process_updates, blanket, released, force_released = (
        _install_startup_cleanup_fakes(monkeypatch, active_tasks=[task])
    )

    server.cleanup_orphan_tasks()

    assert status_updates == []
    assert process_updates == []
    assert blanket == []
    assert released == []
    assert force_released == []


def test_startup_cleanup_reconciles_only_after_verified_exit(monkeypatch):
    task = _active_task()
    fake = _FakeProcess()
    _patch_process(monkeypatch, fake)
    status_updates, process_updates, blanket, released, force_released = (
        _install_startup_cleanup_fakes(monkeypatch, active_tasks=[task])
    )

    server.cleanup_orphan_tasks()

    assert fake.terminated == 1
    assert status_updates == [
        (
            task["task_id"],
            "failed",
            {
                "error_message": "Task interrupted by server restart",
                "run_token": task["run_token"],
            },
        )
    ]
    assert process_updates == [
        (
            task["task_id"],
            {
                "process_pid": None,
                "process_status": None,
                "process_create_time": None,
                "run_token": task["run_token"],
                "expected_process_pid": task["process_pid"],
                "expected_process_create_time": task["process_create_time"],
            },
        )
    ]
    assert blanket == []
    assert released == [
        f"training:{task['task_id']}:{task['run_token']}"
    ]
    assert force_released == []


def test_startup_cleanup_handles_already_exited_process(monkeypatch):
    task = _active_task()
    task["process_create_time"] = None
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda _pid: (_ for _ in ()).throw(psutil.NoSuchProcess(321)),
    )
    status_updates, process_updates, _blanket, released, _force_released = (
        _install_startup_cleanup_fakes(monkeypatch, active_tasks=[task])
    )

    server.cleanup_orphan_tasks()

    assert len(status_updates) == 1
    assert len(process_updates) == 1
    assert released == [f"training:{task['task_id']}:{task['run_token']}"]


def test_startup_cleanup_reconciles_terminal_process_evidence(monkeypatch):
    task = _active_task(status="succeeded")
    fake = _FakeProcess()
    _patch_process(monkeypatch, fake)
    status_updates, process_updates, blanket, released, _force_released = (
        _install_startup_cleanup_fakes(
            monkeypatch,
            active_tasks=[],
            process_tasks=[task],
        )
    )

    server.cleanup_orphan_tasks()

    assert fake.terminated == 1
    assert status_updates == []
    assert len(process_updates) == 1
    assert blanket == []
    assert released == [f"training:{task['task_id']}:{task['run_token']}"]


def test_startup_cleanup_isolates_process_signal_failures(monkeypatch):
    first = _active_task(process_pid=321)
    second = _active_task(
        task_id="33333333-3333-3333-3333-333333333333",
        run_token="44444444-4444-4444-4444-444444444444",
        process_pid=654,
    )
    processes = {
        321: _FakeProcess(terminate_effect=OSError("operation unavailable")),
        654: _FakeProcess(),
    }
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])
    status_updates, process_updates, _blanket, released, _force_released = (
        _install_startup_cleanup_fakes(
            monkeypatch,
            active_tasks=[first, second],
        )
    )

    server.cleanup_orphan_tasks()

    assert [call[0] for call in status_updates] == [second["task_id"]]
    assert [call[0] for call in process_updates] == [second["task_id"]]
    assert released == [f"training:{second['task_id']}:{second['run_token']}"]


def test_stop_endpoint_only_signals_matching_process_identity(monkeypatch):
    task = _active_task()
    stopped = {**task, "status": "stopped"}
    reads = iter((task, stopped))
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: next(reads),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "stop_if_active",
        lambda _task_id: True,
    )
    monkeypatch.setattr(
        training_routes,
        "_recover_sync_training_failure",
        lambda *_args: None,
    )
    fake = _FakeProcess()
    _patch_process(monkeypatch, fake)
    monkeypatch.setattr(
        training_routes.os,
        "kill",
        lambda *_args: pytest.fail("raw PID signalling bypassed identity fencing"),
        raising=False,
    )

    response = asyncio.run(
        training_routes.stop_task(
            task["task_id"],
            current_user={"user_id": "owner", "role": "admin"},
        )
    )

    assert response == {"message": f"Task {task['task_id']} stopped"}
    assert fake.terminated == 1
    assert fake.wait_timeouts == []


def test_training_process_identity_migration_follows_current_head(monkeypatch):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity"
    )

    class Operations:
        def __init__(self):
            self.added_columns = []

        @staticmethod
        def get_bind():
            return object()

        def add_column(self, table, column):
            self.added_columns.append((table, column))

    class Inspector:
        @staticmethod
        def get_table_names():
            return ["training_tasks"]

        @staticmethod
        def get_columns(table):
            assert table == "training_tasks"
            return [
                {"name": "id", "primary_key": True},
                {"name": "process_pid", "primary_key": False},
            ]

        @staticmethod
        def get_pk_constraint(table):
            assert table == "training_tasks"
            return {"name": "PRIMARY", "constrained_columns": ["id"]}

        @staticmethod
        def get_unique_constraints(table):
            assert table == "training_tasks"
            return []

        @staticmethod
        def get_indexes(table):
            assert table == "training_tasks"
            return []

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert migration.revision == "048_add_training_process_identity"
    assert migration.down_revision == "047_milvus_hybrid_intent"
    assert len(operations.added_columns) == 1
    table, column = operations.added_columns[0]
    assert table == "training_tasks"
    assert column.name == "process_create_time"
    assert isinstance(column.type, sa.Double)


def test_training_process_identity_uses_mysql_double_precision():
    ddl = str(
        CreateTable(TrainingTaskDB.__table__).compile(dialect=mysql.dialect())
    )

    assert "process_create_time DOUBLE" in ddl


def test_process_info_compare_and_clear_uses_pid_and_create_time(monkeypatch):
    service_module = importlib.import_module(
        "train_factory.storage.services.training_task_service"
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[TrainingTaskDB.__table__])

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)
    service = service_module.TrainingTaskService()
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="identity-fenced",
                status="running",
                run_token="run-1",
                process_pid=321,
                process_status="running",
                process_create_time=1234.5,
            )
        )
        session.commit()

    assert service.update_process_info(
        "identity-fenced",
        process_pid=None,
        process_status=None,
        process_create_time=None,
        run_token="run-1",
        expected_process_pid=999,
        expected_process_create_time=1234.5,
    ) is False
    assert service.get_task("identity-fenced")["process_pid"] == 321

    assert service.update_process_info(
        "identity-fenced",
        process_pid=None,
        process_status=None,
        process_create_time=None,
        run_token="run-1",
        expected_process_pid=321,
        expected_process_create_time=1234.5,
    ) is True
    task = service.get_task("identity-fenced")
    assert task["process_pid"] is None
    assert task["process_status"] is None
    assert task["process_create_time"] is None


def test_service_lists_terminal_tasks_with_process_evidence(monkeypatch):
    service_module = importlib.import_module(
        "train_factory.storage.services.training_task_service"
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[TrainingTaskDB.__table__])

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)
    service = service_module.TrainingTaskService()
    with Session(engine) as session:
        session.add_all(
            [
                TrainingTaskDB(task_id="clean", status="succeeded"),
                TrainingTaskDB(
                    task_id="terminal-with-process",
                    status="succeeded",
                    process_pid=321,
                    process_status="running",
                    process_create_time=1234.5,
                ),
            ]
        )
        session.commit()

    tasks, total = service.get_tasks_with_process_info(limit=100, offset=0)

    assert total == 1
    assert [task["task_id"] for task in tasks] == ["terminal-with-process"]
