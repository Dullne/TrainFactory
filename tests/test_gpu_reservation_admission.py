"""GPU exclusion across training, durable runtime state, and API restart."""

import importlib
from contextlib import contextmanager
from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api import server
from train_factory.core.gpu_resource_manager import GPUResourceManager
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.training_task_entity import TrainingTaskDB


gpu_module = importlib.import_module("train_factory.core.gpu_resource_manager")
database_module = importlib.import_module("train_factory.storage.database")
training_module = importlib.import_module("train_factory.storage.services.training_task_service")
deployment_module = importlib.import_module("train_factory.deployment.deployment_service")


@pytest.fixture
def gpu_state(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'gpu.db'}")
    SQLModel.metadata.create_all(engine)

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(database_module, "get_session", get_session)
    monkeypatch.setattr(training_module, "get_session", get_session)
    monkeypatch.setattr(deployment_module, "get_session", get_session)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    manager = GPUResourceManager()
    monkeypatch.setattr(gpu_module, "gpu_resource_manager", manager)
    monkeypatch.setattr(deployment_module, "gpu_resource_manager", manager, raising=False)
    return manager, engine


def _training(engine, *, device="cuda:0", status="running"):
    task_id, run_token = str(uuid4()), str(uuid4())
    with Session(engine) as session:
        session.add(TrainingTaskDB(
            task_id=task_id, task_name=task_id, run_token=run_token,
            status=status, process_pid=321, process_create_time=1234.5,
            process_status="running", training_params={"device": device},
        ))
        session.commit()
    return task_id, run_token


def _deployment(engine, *, status="running", gpu_ids=(0,), mode="container"):
    deployment_id = str(uuid4())
    with Session(engine) as session:
        session.add(DeploymentDB(
            deployment_id=deployment_id, model_id="model-1",
            deployment_name=deployment_id, deploy_mode=mode, status=status,
            gpu_id=gpu_ids[0] if gpu_ids else None,
            xinference_endpoint="http://example.invalid", config={"replica_schema_version": 1},
        ))
        session.flush()
        session.add(DeploymentReplicaDB(
            deployment_id=deployment_id, replica_index=0,
            container_name=deployment_id, endpoint="http://example.invalid",
            port=12000 + len(session.exec(select(DeploymentReplicaDB)).all()),
            gpu_ids=list(gpu_ids), status=status,
        ))
        session.commit()
    return deployment_id


@pytest.mark.parametrize("status", ["running", "failed", "succeeded"])
def test_restart_keeps_unverified_training_gpu_excluded(gpu_state, monkeypatch, status):
    manager, engine = gpu_state
    _training(engine, status=status)
    monkeypatch.setattr(server, "terminate_process_if_matches", lambda *_args: False)

    server.cleanup_orphan_tasks()

    assert manager.allocate_gpus_for_task("new-training", "cuda:0") is None
    assert manager.allocate_gpus_for_task("other-training", "cuda:1") == "cuda:1"


@pytest.mark.parametrize("device", [None, "auto", "cuda:broken", "cuda:0,bad"])
def test_unknown_orphan_devices_quarantine_all_gpu_admission(gpu_state, device):
    manager, engine = gpu_state
    _training(engine, device=device)

    assert manager.allocate_gpus_for_task("new-training", "auto") is None
    assert manager.allocate_gpus_for_task("cpu-training", "cpu") == "cpu"


def test_verified_process_exit_releases_recovered_gpu(gpu_state, monkeypatch):
    manager, engine = gpu_state
    _training(engine)
    assert manager.allocate_gpus_for_task("new-training", "cuda:0") is None
    monkeypatch.setattr(server, "terminate_process_if_matches", lambda *_args: True)

    server.cleanup_orphan_tasks()

    assert manager.allocate_gpus_for_task("new-training", "cuda:0") == "cuda:0"


@pytest.mark.parametrize("status", ["starting", "running", "stopping", "restarting", "failed"])
def test_training_excludes_existing_deployment_even_after_manager_restart(gpu_state, status):
    manager, engine = gpu_state
    _deployment(engine, status=status)

    assert manager.allocate_gpus_for_task("new-training", "cuda:0") is None
    assert manager.allocate_gpus_for_task("new-training", "auto") == "cuda:1"


def test_stopped_and_remote_deployments_do_not_reserve_local_gpu(gpu_state):
    manager, engine = gpu_state
    _deployment(engine, status="stopped")
    _deployment(engine, mode="shared", gpu_ids=(1,))

    assert manager.allocate_gpus_for_task("new-training", "auto:2") == "cuda:0,cuda:1"


def test_gpu_allocation_fails_closed_when_durable_state_is_unavailable(gpu_state, monkeypatch):
    manager, _engine = gpu_state

    @contextmanager
    def unavailable():
        raise RuntimeError("database unavailable")
        yield

    monkeypatch.setattr(database_module, "get_session", unavailable)
    assert manager.allocate_gpus_for_task("new-training", "auto") is None


@pytest.mark.parametrize("endpoint", ["http://xinference:9997", "http://127.0.0.1:9997", "http://localhost:9997", "http://trainfactory-xinference:9997"])
def test_local_shared_xinference_reserves_host_gpu(gpu_state, endpoint):
    manager, engine = gpu_state
    with Session(engine) as session:
        session.add(DeploymentDB(
            deployment_id="shared-local", model_id="model-1", deploy_mode="shared",
            xinference_endpoint=endpoint, status="running", gpu_id=0,
        ))
        session.commit()
    assert manager.allocate_gpus_for_task("training-attempt", "cuda:0") is None
    assert manager.allocate_gpus_for_task("other-attempt", "cuda:1") == "cuda:1"


def test_running_local_shared_unknown_assignment_quarantines_all_gpus(gpu_state):
    manager, engine = gpu_state
    with Session(engine) as session:
        session.add(DeploymentDB(
            deployment_id="shared-local", model_id="model-1", deploy_mode="shared",
            xinference_endpoint="http://xinference:9997", status="running", gpu_id=None,
        ))
        session.commit()
    assert manager.allocate_gpus_for_task("training-attempt", "auto") is None


def test_unresolved_cpu_training_does_not_quarantine_gpus(gpu_state):
    manager, engine = gpu_state
    _training(engine, device="cpu")
    assert manager.allocate_gpus_for_task("training-attempt", "auto:2") == "cuda:0,cuda:1"
