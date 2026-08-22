from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from train_factory.enums.sync_status import SyncStatus
from train_factory.storage.services.external_sync_service import external_sync_service
from train_factory.sync import level2_handler, sync_worker


@pytest.fixture
def immediate_to_thread(monkeypatch):
    async def _immediate(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(sync_worker.asyncio, "to_thread", _immediate)


def test_check_thresholds_level1_generation_first(monkeypatch):
    config = {
        "task_id": "task-level1",
        "generation_threshold": 10,
        "pending_record_count": 10,
    }
    called: List[Dict[str, Any]] = []

    async def _fake_trigger_generation(cfg):
        called.append(cfg)
        return {"generation_task_id": "gen-1", "task_id": cfg["task_id"]}

    monkeypatch.setattr(sync_worker, "_trigger_generation", _fake_trigger_generation)
    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda task_id, is_active=None: (_ for _ in ()).throw(
            AssertionError("level1 reached should return before listing training targets")
        ),
    )

    result = asyncio.run(sync_worker._check_thresholds(config))
    assert result == {"generation_task_id": "gen-1", "task_id": "task-level1"}
    assert called == [config]


def test_check_thresholds_legacy_training_when_no_targets(monkeypatch, immediate_to_thread):
    config = {
        "task_id": "task-legacy",
        "generation_threshold": 1000,
        "pending_record_count": 0,
        "training_threshold": 50,
        "pending_training_samples": 50,
        "status": SyncStatus.IDLE,
    }
    triggered: List[Dict[str, Any]] = []

    monkeypatch.setattr(external_sync_service, "list_training_targets_raw", lambda task_id, is_active=None: [])
    monkeypatch.setattr(level2_handler, "_trigger_training", lambda cfg: triggered.append(cfg))
    monkeypatch.setattr(
        level2_handler,
        "_schedule_next_training",
        lambda task_id: (_ for _ in ()).throw(AssertionError("legacy path should not schedule target")),
    )

    result = asyncio.run(sync_worker._check_thresholds(config))
    assert result is None
    assert triggered == [config]


def test_check_thresholds_multitarget_schedules_only_one(monkeypatch, immediate_to_thread):
    config = {
        "task_id": "task-multi",
        "generation_threshold": 1000,
        "pending_record_count": 0,
        "status": SyncStatus.IDLE,
    }
    scheduled: List[str] = []

    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda task_id, is_active=None: [
            {
                "target_id": "t-1",
                "target_name": "first",
                "is_active": True,
                "training_threshold": 100,
                "pending_training_samples": 50,
            },
            {
                "target_id": "t-2",
                "target_name": "second",
                "is_active": True,
                "training_threshold": 30,
                "pending_training_samples": 30,
            },
            {
                "target_id": "t-3",
                "target_name": "third",
                "is_active": True,
                "training_threshold": 10,
                "pending_training_samples": 10,
            },
        ],
    )
    monkeypatch.setattr(
        level2_handler,
        "_trigger_training",
        lambda cfg: (_ for _ in ()).throw(AssertionError("multi-target path should not trigger legacy training")),
    )
    monkeypatch.setattr(level2_handler, "_schedule_next_training", lambda task_id: scheduled.append(task_id))

    result = asyncio.run(sync_worker._check_thresholds(config))
    assert result is None
    assert scheduled == ["task-multi"]


@pytest.mark.parametrize("status", [SyncStatus.TRAINING, SyncStatus.LOADING_ADAPTER, SyncStatus.GENERATING])
@pytest.mark.parametrize("mode", ["legacy", "target"])
def test_check_thresholds_busy_status_blocks_training(monkeypatch, immediate_to_thread, status, mode):
    config = {
        "task_id": f"task-busy-{mode}",
        "generation_threshold": 1000,
        "pending_record_count": 0,
        "training_threshold": 20,
        "pending_training_samples": 20,
        "status": status,
    }
    triggered: List[Dict[str, Any]] = []
    scheduled: List[str] = []

    if mode == "legacy":
        monkeypatch.setattr(external_sync_service, "list_training_targets_raw", lambda task_id, is_active=None: [])
    else:
        monkeypatch.setattr(
            external_sync_service,
            "list_training_targets_raw",
            lambda task_id, is_active=None: [
                {
                    "target_id": "t-ready",
                    "target_name": "ready",
                    "is_active": True,
                    "training_threshold": 20,
                    "pending_training_samples": 20,
                }
            ],
        )

    monkeypatch.setattr(level2_handler, "_trigger_training", lambda cfg: triggered.append(cfg))
    monkeypatch.setattr(level2_handler, "_schedule_next_training", lambda task_id: scheduled.append(task_id))

    result = asyncio.run(sync_worker._check_thresholds(config))
    assert result is None
    assert triggered == []
    assert scheduled == []


def test_check_thresholds_targets_all_inactive_do_not_schedule(monkeypatch, immediate_to_thread):
    config = {
        "task_id": "task-inactive",
        "generation_threshold": 1000,
        "pending_record_count": 0,
        "training_threshold": 10,
        "pending_training_samples": 10,
        "status": SyncStatus.IDLE,
    }
    triggered: List[Dict[str, Any]] = []
    scheduled: List[str] = []

    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda task_id, is_active=None: [
            {
                "target_id": "t-off-1",
                "target_name": "off-1",
                "is_active": False,
                "training_threshold": 10,
                "pending_training_samples": 10,
            },
            {
                "target_id": "t-off-2",
                "target_name": "off-2",
                "is_active": False,
                "training_threshold": 5,
                "pending_training_samples": 5,
            },
        ],
    )
    monkeypatch.setattr(level2_handler, "_trigger_training", lambda cfg: triggered.append(cfg))
    monkeypatch.setattr(level2_handler, "_schedule_next_training", lambda task_id: scheduled.append(task_id))

    result = asyncio.run(sync_worker._check_thresholds(config))
    assert result is None
    assert triggered == []
    assert scheduled == []
