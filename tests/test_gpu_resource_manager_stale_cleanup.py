import logging
from datetime import datetime, timedelta

import pytest

from train_factory.core.gpu_resource_manager import GPUResourceManager
from train_factory.storage.services.training_task_service import training_task_service


TASK_ID = "11111111-1111-4111-8111-11111111111a"
RUN_TOKEN = "22222222-2222-4222-8222-22222222222b"
OTHER_RUN_TOKEN = "33333333-3333-4333-8333-33333333333c"


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(GPUResourceManager, "_detect_max_gpus", lambda _self: 2)
    return GPUResourceManager()


def _lease(run_token=RUN_TOKEN):
    return f"training:{TASK_ID}:{run_token}"


def _task(
    *,
    status,
    run_token=RUN_TOKEN,
    process_pid=None,
    process_status=None,
):
    return {
        "task_id": TASK_ID,
        "status": status,
        "run_token": run_token,
        "process_pid": process_pid,
        "process_status": process_status,
    }


def _seed_stale_lease(manager, lease, gpu_id=0):
    manager.gpu_allocations[gpu_id] = lease
    manager.task_gpus[lease] = {gpu_id}
    manager.allocation_times[lease] = datetime.now() - timedelta(days=2)


def _assert_lease_retained(manager, lease, gpu_id=0):
    assert manager.gpu_allocations == {gpu_id: lease}
    assert manager.task_gpus == {lease: {gpu_id}}
    assert set(manager.allocation_times) == {lease}


def _assert_lease_released(manager):
    assert manager.gpu_allocations == {}
    assert manager.task_gpus == {}
    assert manager.allocation_times == {}


def test_gpu_allocation_logs_hide_training_run_token(manager, caplog):
    lease = _lease()

    with caplog.at_level(logging.INFO):
        assert manager.allocate_gpus_for_task(lease, "cuda:0") == "cuda:0"
        assert manager.release_gpus_for_task(lease)

    assert TASK_ID in caplog.text
    assert RUN_TOKEN not in caplog.text


@pytest.mark.parametrize(
    ("lease", "expected_label"),
    [
        (_lease(), f"training:{TASK_ID}"),
        (f"training:{RUN_TOKEN}", "training:<invalid>"),
    ],
)
def test_simple_gpu_status_never_exposes_training_run_token(
    monkeypatch,
    manager,
    lease,
    expected_label,
):
    monkeypatch.setattr(
        manager,
        "_get_gpu_memory_info",
        lambda _gpu_id: {
            "gpu_name": "test-gpu",
            "memory": {
                "total_gb": 1,
                "used_gb": 0,
                "free_gb": 1,
                "usage_percent": 0,
            },
            "utilization": {"gpu_percent": 0},
            "temperature": None,
            "power": {"usage_w": None, "limit_w": None},
        },
    )
    assert manager.allocate_gpus_for_task(lease, "cuda:0") == "cuda:0"

    allocated_task = manager.get_gpu_info_simple()[0]["allocated_task"]

    assert allocated_task == expected_label
    assert RUN_TOKEN not in allocated_task


@pytest.mark.parametrize("status", ["preparing", "running", "evaluating"])
def test_cleanup_retains_matching_active_training_attempt(
    monkeypatch,
    manager,
    status,
):
    lease = _lease()
    _seed_stale_lease(manager, lease)
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: _task(status=status) if task_id == TASK_ID else None,
    )

    cleaned = manager.cleanup_stale_allocations(max_age_hours=24)

    assert cleaned == 0
    _assert_lease_retained(manager, lease)


def test_cleanup_releases_old_attempt_when_run_token_no_longer_matches(
    monkeypatch,
    manager,
):
    lease = _lease()
    _seed_stale_lease(manager, lease)
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: (
            _task(
                status="running",
                run_token=OTHER_RUN_TOKEN,
                process_pid=4321,
                process_status="running",
            )
            if task_id == TASK_ID
            else None
        ),
    )

    cleaned = manager.cleanup_stale_allocations(max_age_hours=24)

    assert cleaned == 1
    _assert_lease_released(manager)


def test_cleanup_retains_lease_when_task_lookup_fails(
    monkeypatch,
    manager,
    caplog,
):
    lease = _lease()
    _seed_stale_lease(manager, lease)

    def fail_lookup(_task_id):
        raise OSError("database unavailable")

    monkeypatch.setattr(training_task_service, "get_task", fail_lookup)

    with caplog.at_level(
        logging.WARNING,
        logger="train_factory.core.gpu_resource_manager",
    ):
        cleaned = manager.cleanup_stale_allocations(max_age_hours=24)

    assert cleaned == 0
    _assert_lease_retained(manager, lease)
    assert "database unavailable" in caplog.text


@pytest.mark.parametrize(
    ("process_pid", "process_status"),
    [
        (4321, None),
        (None, "running"),
        (None, "stopping"),
    ],
)
def test_cleanup_retains_terminal_task_with_live_process_evidence(
    monkeypatch,
    manager,
    process_pid,
    process_status,
):
    lease = _lease()
    _seed_stale_lease(manager, lease)
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: (
            _task(
                status="succeeded",
                process_pid=process_pid,
                process_status=process_status,
            )
            if task_id == TASK_ID
            else None
        ),
    )

    cleaned = manager.cleanup_stale_allocations(max_age_hours=24)

    assert cleaned == 0
    _assert_lease_retained(manager, lease)


def test_cleanup_releases_terminal_task_without_process_evidence(
    monkeypatch,
    manager,
):
    lease = _lease()
    _seed_stale_lease(manager, lease)
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: _task(status="succeeded") if task_id == TASK_ID else None,
    )

    cleaned = manager.cleanup_stale_allocations(max_age_hours=24)

    assert cleaned == 1
    _assert_lease_released(manager)


def test_cleanup_releases_lease_when_task_is_missing(monkeypatch, manager):
    lease = _lease()
    _seed_stale_lease(manager, lease)
    monkeypatch.setattr(training_task_service, "get_task", lambda _task_id: None)

    cleaned = manager.cleanup_stale_allocations(max_age_hours=24)

    assert cleaned == 1
    _assert_lease_released(manager)


@pytest.mark.parametrize(
    "lease",
    [
        "training:missing-token",
        f"training:{RUN_TOKEN}",
        f"training:not-a-uuid:{RUN_TOKEN}",
        f"training:{TASK_ID}:not-a-uuid",
        f"training:{TASK_ID}:{RUN_TOKEN}:unexpected",
        f"training:{TASK_ID.upper()}:{RUN_TOKEN}",
    ],
)
def test_cleanup_retains_malformed_training_lease_and_warns(
    monkeypatch,
    manager,
    caplog,
    lease,
):
    _seed_stale_lease(manager, lease)
    lookups = []
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: lookups.append(task_id),
    )

    with caplog.at_level(
        logging.WARNING,
        logger="train_factory.core.gpu_resource_manager",
    ):
        cleaned = manager.cleanup_stale_allocations(max_age_hours=24)

    assert cleaned == 0
    _assert_lease_retained(manager, lease)
    assert lookups == []
    assert "training:<invalid>" in caplog.text
    assert lease not in caplog.text
    assert RUN_TOKEN not in caplog.text
