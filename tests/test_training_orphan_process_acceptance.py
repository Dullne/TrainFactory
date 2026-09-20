"""CPU-only Linux container acceptance of a genuinely orphaned training worker.

The launcher exits without touching the database. Its fresh-interpreter child
runs the production worker with only the training body replaced by a wait.
GPU 0 below is a scheduler identifier; this test never opens a CUDA device.
"""

import ctypes
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import psutil
import pytest


def _write_json(path, value):
    temporary = Path(str(path) + ".writing")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def _launch_worker(database_path, task_id, token, identity_path, ready_path):
    # Popen does not join children when this short-lived launcher exits.
    process = subprocess.Popen([
        sys.executable, __file__, "--worker", database_path, task_id, token,
        ready_path,
    ])
    _write_json(identity_path, {
        "pid": process.pid,
        "create_time": psutil.Process(process.pid).create_time(),
        "launcher_pid": os.getpid(),
    })


def _run_worker(database_path, task_id, token, ready_path):
    from sqlmodel import create_engine
    from train_factory.api.routes import training_routes
    from train_factory.storage import database

    # A fresh interpreter and a fresh engine: no inherited SQLite connections.
    database._engine = create_engine(f"sqlite:///{database_path}")

    def wait_instead_of_training(*_args, **_kwargs):
        task = training_routes.training_task_service.get_task(task_id)
        _write_json(ready_path, {
            "pid": os.getpid(),
            "create_time": psutil.Process().create_time(),
            "registered_pid": task["process_pid"],
            "registered_create_time": task["process_create_time"],
            "registered_process_status": task["process_status"],
            "registered_task_status": task["status"],
            "registered_run_token": task["run_token"],
        })
        # No model, dataset, GPU, or network operation is performed.
        while True:
            time.sleep(0.1)

    training_routes.train_with_config = wait_instead_of_training
    training_routes._run_training_task_worker(task_id, {"_run_token": token}, "")


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/.dockerenv").exists(),
    reason="Real orphan-process acceptance runs only inside a Linux Docker container",
)
def test_startup_reaps_registered_orphan_before_releasing_durable_gpu(monkeypatch, tmp_path):
    from sqlmodel import Session, SQLModel, create_engine
    from train_factory.api import server
    from train_factory.core.gpu_resource_manager import GPUResourceManager
    from train_factory.core.process_identity import terminate_process_if_matches
    from train_factory.storage.entities.training_task_entity import TrainingTaskDB
    from train_factory.storage.services.training_task_service import training_task_service

    database = importlib.import_module("train_factory.storage.database")
    gpu_module = importlib.import_module("train_factory.core.gpu_resource_manager")
    database_path = tmp_path / "orphan.sqlite"
    identity_path, ready_path = tmp_path / "child.json", tmp_path / "training-entered.json"
    engine = create_engine(f"sqlite:///{database_path}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 1)
    manager = GPUResourceManager()
    monkeypatch.setattr(gpu_module, "gpu_resource_manager", manager)
    task_id, token = str(uuid4()), str(uuid4())
    with Session(engine) as session:
        session.add(TrainingTaskDB(
            task_id=task_id, task_name="real-orphan-acceptance", run_token=token,
            status="preparing", training_params={"device": "cuda:0"},
        ))
        session.commit()

    # Adopt just-created grandchildren so psutil's real wait can reap them.
    # Restore the previous setting afterward; never signal unrelated processes.
    libc = ctypes.CDLL(None, use_errno=True)
    previous_subreaper = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous_subreaper), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    launcher = None
    identity = None
    try:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", MAX_GPUS="0")
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        for key in ("BASE_DIR", "TRAINING_CACHE", "MODELS_DIR", "DATASETS_DIR", "OUTPUT_DIR", "LOCAL_CACHE_DIR"):
            directory = tmp_path / key.lower()
            directory.mkdir()
            environment[key] = str(directory)
        with (tmp_path / "child.log").open("w", encoding="utf-8") as log:
            launcher = subprocess.Popen([
                sys.executable, __file__, "--launcher", str(database_path),
                task_id, token, str(identity_path), str(ready_path),
            ], env=environment, stdout=log, stderr=subprocess.STDOUT)
            assert launcher.wait(timeout=15) == 0
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            assert identity["launcher_pid"] == launcher.pid
            assert identity["pid"] not in {os.getpid(), launcher.pid}
            deadline = time.monotonic() + 30
            while not ready_path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert ready_path.exists(), (tmp_path / "child.log").read_text(encoding="utf-8")

        entered = json.loads(ready_path.read_text(encoding="utf-8"))
        child = psutil.Process(identity["pid"])
        assert child.ppid() == os.getpid(), "Worker was not orphaned and adopted by the test"
        assert child.create_time() == pytest.approx(identity["create_time"], rel=0, abs=0.01)
        assert entered["pid"] == entered["registered_pid"] == child.pid
        assert entered["registered_create_time"] == pytest.approx(child.create_time(), rel=0, abs=0.01)
        assert entered["registered_process_status"] == "running"
        assert entered["registered_task_status"] == "running"
        assert entered["registered_run_token"] == token
        # Simulate a restarted API: no inherited in-memory GPU leases exist.
        assert manager.task_gpus == {}
        assert manager.allocate_gpus_for_task("competing-before-cleanup", "cuda:0") is None
        checkpoints = []

        def observe_real_termination(pid, create_time):
            assert pid == identity["pid"]
            assert create_time == pytest.approx(identity["create_time"], rel=0, abs=0.01)
            assert psutil.Process(pid).create_time() == pytest.approx(create_time, rel=0, abs=0.01)
            assert manager.allocate_gpus_for_task("competing-during-cleanup", "cuda:0") is None
            checkpoints.append("live-worker-still-reserved")
            confirmed = terminate_process_if_matches(pid, create_time)
            assert confirmed
            assert not psutil.pid_exists(pid), "Termination returned before the owned worker was reaped"
            # The durable row must still hold the reservation until cleanup
            # commits its process-info clear, even after the OS process exits.
            assert manager.allocate_gpus_for_task("competing-before-db-clear", "cuda:0") is None
            checkpoints.append("exited-worker-still-reserved")
            return confirmed

        monkeypatch.setattr(server, "terminate_process_if_matches", observe_real_termination)
        server.cleanup_orphan_tasks()
        assert checkpoints == ["live-worker-still-reserved", "exited-worker-still-reserved"]
        recovered = training_task_service.get_task(task_id)
        assert recovered["status"] == "failed"
        assert recovered["process_pid"] is None
        assert recovered["process_create_time"] is None
        assert recovered["process_status"] is None
        assert manager.allocate_gpus_for_task("competing-after-cleanup", "cuda:0") == "cuda:0"
    finally:
        # Even assertion failures only terminate the exact child created here.
        try:
            if identity is None and identity_path.exists():
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if identity is not None:
                assert terminate_process_if_matches(identity["pid"], identity["create_time"])
            if launcher is not None and launcher.poll() is None:
                launcher_identity = psutil.Process(launcher.pid).create_time()
                assert terminate_process_if_matches(launcher.pid, launcher_identity)
                launcher.wait(timeout=5)
        finally:
            libc.prctl(36, previous_subreaper.value, 0, 0, 0)
            engine.dispose()


if __name__ == "__main__":
    if sys.argv[1] == "--launcher":
        _launch_worker(*sys.argv[2:])
    elif sys.argv[1] == "--worker":
        _run_worker(*sys.argv[2:])
    else:
        raise SystemExit("Unknown isolated acceptance mode")
