"""Training cannot consume a GPU before its process identity is durable."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api import server
from train_factory.api.routes import training_routes
from train_factory.core.gpu_resource_manager import GPUResourceManager
from train_factory.storage.entities.training_task_entity import TrainingTaskDB

service_module = importlib.import_module("train_factory.storage.services.training_task_service")
database_module = importlib.import_module("train_factory.storage.database")
gpu_module = importlib.import_module("train_factory.core.gpu_resource_manager")


@pytest.fixture
def registration_state(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'training.db'}")
    SQLModel.metadata.create_all(engine)

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", get_session)
    monkeypatch.setattr(database_module, "get_session", get_session)
    event_module = importlib.import_module("train_factory.storage.services.training_task_event_service")
    monkeypatch.setattr(event_module, "get_session", get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda self: 1)
    manager = GPUResourceManager()
    monkeypatch.setattr(gpu_module, "gpu_resource_manager", manager)
    task_id, token = str(uuid4()), str(uuid4())
    with get_session() as session:
        session.add(TrainingTaskDB(
            task_id=task_id, task_name="registration-test", status="preparing",
            run_token=token, training_params={"device": "cuda:0"},
        ))
        session.commit()
    try:
        yield service_module.training_task_service, manager, task_id, token, get_session
    finally:
        engine.dispose()


def test_child_persists_identity_before_entering_training(registration_state, monkeypatch):
    service, manager, task_id, token, _sessions = registration_state
    observed = []

    def train(*_args, **_kwargs):
        observed.append((
            service.get_task(task_id),
            manager.allocate_gpus_for_task("another-attempt", "cuda:0"),
        ))
        return SimpleNamespace(save_dir="/tmp/no-real-model", final_metrics={})

    monkeypatch.setattr(training_routes, "train_with_config", train)
    training_routes._run_training_task_worker(task_id, {"_run_token": token}, None)
    assert len(observed) == 1
    registered, competing_allocation = observed[0]
    assert registered["process_pid"] == os.getpid()
    assert registered["process_create_time"] > 0
    assert registered["process_status"] == "running"
    assert competing_allocation is None
    # A child may be orphaned before the parent performs its RUNNING update.
    assert service.get_task(task_id)["status"] == "succeeded"


@pytest.mark.parametrize("state", ["preparing", "running", "failed", "stopped", "succeeded"])
@pytest.mark.parametrize("evidence", [{"process_pid": 54321}, {"process_status": "stopping"}, {"process_create_time": 1234.5}])
def test_sync_recovery_preserves_claim_until_process_exit_is_confirmed(monkeypatch, state, evidence):
    sync_module = importlib.import_module("train_factory.storage.services.external_sync_service")
    task = {"task_id": "owned-training", "status": state, **evidence}
    restored = []
    completed = []
    monkeypatch.setattr(server, "_iter_pending_sync_trainings", lambda _service: iter([
        {"training_task_id": "owned-training"},
    ]))
    monkeypatch.setattr(service_module.training_task_service, "get_task", lambda _task_id: task)
    monkeypatch.setattr(sync_module.external_sync_service, "fail_training_and_restore_claim", lambda *_a: restored.append(True))
    post_training = importlib.import_module("train_factory.sync.post_training_handler")
    monkeypatch.setattr(post_training, "on_training_completed", lambda **_kw: completed.append(True))
    server.cleanup_orphan_sync_trainings()
    assert restored == []
    assert completed == []


@pytest.mark.parametrize("failure", [False, RuntimeError("database unavailable")])
def test_child_never_trains_when_registration_fails(registration_state, monkeypatch, failure):
    service, _manager, task_id, token, _sessions = registration_state
    training = []

    def register(*_args, **_kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(service, "register_training_process", register, raising=False)
    monkeypatch.setattr(training_routes, "train_with_config", lambda *_a, **_kw: training.append(True))
    training_routes._run_training_task_worker(task_id, {"_run_token": token}, None)
    assert training == []


def test_stale_recovery_snapshot_cannot_finalize_a_registered_child(registration_state, monkeypatch):
    service, manager, task_id, token, sessions = registration_state
    snapshot = service.get_task(task_id)
    # Force the child registration to win AFTER recovery collected its snapshot.
    def pages(operation, *args):
        if args == ("preparing",):
            return [snapshot]
        if not args:
            assert service.register_training_process(task_id, 54321, 1234.5, run_token=token)
        return []

    monkeypatch.setattr(server, "_collect_task_pages", pages)
    monkeypatch.setattr(server, "terminate_process_if_matches", lambda *_a: pytest.fail("stale snapshot must not signal an unobserved process"))
    server.cleanup_orphan_tasks()
    current = service.get_task(task_id)
    assert current["status"] == "running"
    assert current["process_pid"] == 54321
    assert manager.allocate_gpus_for_task("another-attempt", "cuda:0") is None


@pytest.mark.parametrize("winner", ["stopped", "cancelled", "new_attempt", "other_process", "stopping_process", "recovery"])
def test_registration_cannot_overwrite_a_concurrent_winner(registration_state, monkeypatch, winner):
    service, _manager, task_id, token, sessions = registration_state
    training = []

    def concurrent_transition(_config):
        if winner == "recovery":
            server.cleanup_orphan_tasks()
            return
        with sessions() as session:
            task = session.exec(select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)).one()
            if winner in {"stopped", "cancelled"}:
                task.status = winner
            elif winner == "new_attempt":
                task.run_token = str(uuid4())
            elif winner == "other_process":
                task.process_pid, task.process_create_time = 54321, 1234.5
                task.process_status = "running"
            else:
                task.process_pid = os.getpid()
                task.process_create_time = training_routes.capture_process_create_time(os.getpid())
                task.process_status = "stopping"
            session.add(task)
            session.commit()

    monkeypatch.setattr(training_routes, "_validate_training_resource_limits", concurrent_transition)
    monkeypatch.setattr(training_routes, "train_with_config", lambda *_a, **_kw: training.append(True))
    training_routes._run_training_task_worker(task_id, {"_run_token": token}, None)
    assert training == []
    current = service.get_task(task_id)
    if winner in {"stopped", "cancelled", "recovery"}:
        assert current["status"] == ("failed" if winner == "recovery" else winner)
        assert current["process_pid"] is None
    elif winner == "other_process":
        assert current["process_pid"] == 54321
    elif winner == "new_attempt":
        assert current["run_token"] != token
        assert current["process_pid"] is None
    else:
        assert current["process_status"] == "stopping"


def test_child_accepts_its_already_persisted_parent_registration(registration_state, monkeypatch):
    service, _manager, task_id, token, _sessions = registration_state
    assert service.update_process_info(
        task_id, process_pid=os.getpid(),
        process_create_time=training_routes.capture_process_create_time(os.getpid()),
        process_status="running", run_token=token,
    )
    training = []

    def train(*_args, **_kwargs):
        training.append(True)
        return SimpleNamespace(save_dir="/tmp/no-real-model", final_metrics={})

    monkeypatch.setattr(training_routes, "train_with_config", train)
    training_routes._run_training_task_worker(task_id, {"_run_token": token}, None)
    assert training == [True]


def test_parent_kills_child_before_releasing_gpu_when_registration_loses(monkeypatch):
    from test_training_stop_race import _LifecycleFakeProcess, _RaceTrainingService, _run_lifecycle_case

    class LostRegistrationService(_RaceTrainingService):
        def register_training_process(self, *_args, **_kwargs):
            self.task["status"] = "stopped"
            return False

    def exit_process(process):
        process.alive = False
        process.exitcode = 0

    service = LostRegistrationService("lost-registration")
    process = _LifecycleFakeProcess(exit_process)
    release_states = []
    _run_lifecycle_case(
        monkeypatch, service, process,
        release=lambda _lease: release_states.append(process.is_alive()) or True,
    )
    assert service.task["status"] == "stopped"
    assert ("terminate", None) in process.events
    assert release_states == [False]


def test_parent_cleans_self_registered_child_that_exits_before_identity_capture(monkeypatch):
    from test_training_stop_race import _LifecycleFakeProcess, _RaceTrainingService, _run_lifecycle_case

    class CompletedService(_RaceTrainingService):
        def update_task_status(self, *args, **kwargs):
            if self.task["status"] == "succeeded":
                raise ValueError("terminal task cannot become failed")
            return super().update_task_status(*args, **kwargs)

    service = CompletedService("fast-child")

    class FastChild(_LifecycleFakeProcess):
        def start(self):
            super().start()
            service.task.update(
                status="succeeded", process_pid=self.pid,
                process_create_time=1234.5, process_status="running",
            )
            self.alive = False
            self.exitcode = 0
            monkeypatch.setattr(training_routes, "capture_process_create_time", lambda _pid: (_ for _ in ()).throw(ProcessLookupError("child exited")))

    process = FastChild(lambda _process: None)
    monkeypatch.setattr(training_routes, "_finalize_successful_training", lambda *_a: True)
    _run_lifecycle_case(monkeypatch, service, process)
    assert service.task["status"] == "succeeded"
    assert service.task["process_pid"] is None
    assert service.task["process_create_time"] is None


@pytest.mark.parametrize("concurrent", [False, True])
def test_parent_and_child_registration_emit_one_durable_running_event(registration_state, concurrent):
    from train_factory.storage.entities.training_task_event_entity import TrainingTaskEventDB

    service, _manager, task_id, token, sessions = registration_state

    def register():
        return service.register_training_process(task_id, 54321, 1234.5, run_token=token)

    if concurrent:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: register(), range(2)))
    else:
        results = [register(), register()]
    assert results == [True, True]
    with sessions() as session:
        events = session.exec(select(TrainingTaskEventDB).where(
            TrainingTaskEventDB.task_id == task_id,
            TrainingTaskEventDB.event_type == "status_changed",
        )).all()
        assert [event.payload for event in events] == [
            {"from": "preparing", "to": "running", "error_message": None},
        ]
