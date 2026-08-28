from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.exc import IntegrityError

from train_factory.api.routes.sync_routes import (
    _get_task_training_targets,
    _legacy_training_target_id,
)
from train_factory.enums.sync_status import (
    BatchStatus,
    SyncGenerationStatus,
    SyncStatus,
    SyncTrainingStatus,
    TrainingTargetStatus,
)
from train_factory.enums.training_status import TrainingStatus
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.services.external_sync_service import external_sync_service
from train_factory.sync import level2_handler
from train_factory.sync.sync_manager import SyncManager


class _FakeExternalSyncService:
    def __init__(self):
        self.all_targets: List[Dict[str, Any]] = []
        self.active_targets: List[Dict[str, Any]] = []
        self.created_targets: List[Dict[str, Any]] = []
        self.increment_target_calls: List[tuple[str, int]] = []
        self.increment_pending_calls: List[tuple[str, int]] = []
        self.updated_generations: List[Dict[str, Any]] = []
        self.updated_batches: List[tuple[str, str]] = []
        self.updated_tasks: List[Dict[str, Any]] = []
        self.completion_calls: List[Dict[str, Any]] = []
        self.generation_by_task_id: Dict[str, Dict[str, Any]] = {}
        self.task_raw_map: Dict[str, Dict[str, Any]] = {}
        self.raise_integrity_on_create = False

    def list_training_targets(self, task_id: str, is_active: Optional[bool] = True, **kwargs):  # noqa: ARG002
        if is_active is None:
            return list(self.all_targets)
        return list(self.active_targets)

    def create_training_target(self, **kwargs):
        if self.raise_integrity_on_create:
            raise IntegrityError("insert", {}, Exception("duplicate"))
        self.created_targets.append(kwargs)
        return {"target_id": kwargs["target_id"]}

    def migrate_legacy_training_target(self, **kwargs):
        if self.raise_integrity_on_create:
            raise IntegrityError("insert", {}, Exception("duplicate"))
        self.created_targets.append(kwargs)
        return {"target_id": kwargs["target_id"]}

    def increment_target_pending_samples(self, target_id: str, delta: int):
        self.increment_target_calls.append((target_id, delta))

    def increment_pending_training_samples(self, task_id: str, delta: int):
        self.increment_pending_calls.append((task_id, delta))

    def get_generation_by_task_id(self, generation_task_id: str):
        return self.generation_by_task_id.get(generation_task_id)

    def update_generation_status(self, generation_task_id: str, status: str, **kwargs):
        self.updated_generations.append(
            {"generation_task_id": generation_task_id, "status": status, **kwargs}
        )

    def update_batch_status(self, batch_id: str, status: str):
        self.updated_batches.append((batch_id, status))

    def complete_generation_and_consume_batches(
        self,
        generation_task_id: str,
        output_dataset_id: Optional[str],
        output_sample_count: int,
    ):
        sync_gen = self.generation_by_task_id.get(generation_task_id)
        if not sync_gen:
            return {"tracking_found": False, "completed": False}

        config_id = sync_gen["task_id"]
        self.completion_calls.append(
            {
                "generation_task_id": generation_task_id,
                "output_dataset_id": output_dataset_id,
                "output_sample_count": output_sample_count,
            }
        )
        self.update_generation_status(
            generation_task_id,
            SyncGenerationStatus.COMPLETED,
            output_dataset_id=output_dataset_id,
            output_sample_count=output_sample_count,
        )
        for batch_id in sync_gen.get("input_batch_ids") or []:
            self.update_batch_status(batch_id, BatchStatus.GENERATION_DONE)

        if output_dataset_id and output_sample_count > 0:
            if self.all_targets:
                for target in self.all_targets:
                    if target.get("is_active", True) and target.get("data_phase") == "final":
                        self.increment_target_pending_samples(
                            target["target_id"], output_sample_count
                        )
            else:
                self.increment_pending_training_samples(config_id, output_sample_count)
                config = self.task_raw_map.get(config_id)
                if config:
                    config["pending_training_samples"] = int(
                        config.get("pending_training_samples", 0)
                    ) + output_sample_count

        config = self.task_raw_map.get(config_id)
        parent_status_updated = bool(
            config and config.get("status") == SyncStatus.GENERATING
        )
        if parent_status_updated:
            config["status"] = SyncStatus.IDLE
            self.update_task(config_id, status=SyncStatus.IDLE)

        return {
            "tracking_found": True,
            "completed": True,
            "already_completed": False,
            "task_id": config_id,
            "completed_batch_count": len(sync_gen.get("input_batch_ids") or []),
            "credited_sample_count": output_sample_count,
            "credited_target_count": len(self.increment_target_calls),
            "parent_status_updated": parent_status_updated,
            "uses_training_targets": bool(self.all_targets),
        }

    def get_task_raw(self, task_id: str):
        return self.task_raw_map.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        self.updated_tasks.append({"task_id": task_id, **kwargs})


@pytest.fixture
def fake_external_sync_service(monkeypatch):
    fake = _FakeExternalSyncService()
    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        fake,
    )
    return fake


def test_legacy_training_target_id_is_deterministic():
    task_id = "task-123"
    assert _legacy_training_target_id(task_id) == _legacy_training_target_id(task_id)
    assert _legacy_training_target_id("task-abc") != _legacy_training_target_id(task_id)


def test_get_task_training_targets_returns_existing_active_targets(fake_external_sync_service):
    fake_external_sync_service.all_targets = [{"target_id": "t1"}]
    fake_external_sync_service.active_targets = [{"target_id": "t1", "is_active": True}]

    targets = _get_task_training_targets("task-1", {"training_threshold": 1000})

    assert targets == [{"target_id": "t1", "is_active": True}]
    assert fake_external_sync_service.created_targets == []


def test_get_task_training_targets_returns_read_only_legacy_view(fake_external_sync_service):
    config = {
        "training_threshold": 1200,
        "pending_training_samples": 88,
        "base_deployment_id": "dep-1",
        "training_config": {
            "model_type": "llm",
            "training_method": "dpo",
            "base_model_path": "/models/base",
        },
    }

    targets = _get_task_training_targets("task-legacy", config)

    assert fake_external_sync_service.created_targets == []
    assert fake_external_sync_service.increment_target_calls == []
    assert len(targets) == 1
    assert targets[0]["target_id"] == _legacy_training_target_id("task-legacy")
    assert targets[0]["target_name"] == "LLM (legacy)"
    assert targets[0]["training_threshold"] == 1200
    assert targets[0]["pending_training_samples"] == 88


def test_get_task_training_targets_never_enters_migration_write_path(fake_external_sync_service):
    fake_external_sync_service.raise_integrity_on_create = True
    config = {
        "training_threshold": 1000,
        "training_config": {"model_type": "embedding", "training_method": "sft"},
    }

    targets = _get_task_training_targets("task-idem", config)

    assert targets[0]["target_id"] == _legacy_training_target_id("task-idem")
    assert fake_external_sync_service.created_targets == []
    assert fake_external_sync_service.increment_target_calls == []


def test_on_generation_completed_target_mode_routes_to_targets_only(
    fake_external_sync_service,
    monkeypatch,
):
    config_id = "cfg-target"
    generation_task_id = "gen-1"
    fake_external_sync_service.generation_by_task_id[generation_task_id] = {
        "task_id": config_id,
        "input_batch_ids": ["b-1", "b-2"],
    }
    fake_external_sync_service.all_targets = [
        {"target_id": "t-final", "is_active": True, "data_phase": "final"},
        {"target_id": "t-qa", "is_active": True, "data_phase": "qa"},
    ]
    fake_external_sync_service.task_raw_map[config_id] = {"status": SyncStatus.GENERATING}

    scheduled: List[str] = []
    monkeypatch.setattr(level2_handler, "_schedule_next_training", lambda cid: scheduled.append(cid))
    monkeypatch.setattr(level2_handler, "_trigger_training", lambda cfg: (_ for _ in ()).throw(AssertionError("should not trigger legacy training")))  # noqa: E501

    level2_handler.on_generation_completed(generation_task_id, "ds-1", 50)

    assert fake_external_sync_service.updated_generations == [
        {
            "generation_task_id": generation_task_id,
            "status": SyncGenerationStatus.COMPLETED,
            "output_dataset_id": "ds-1",
            "output_sample_count": 50,
        }
    ]
    assert fake_external_sync_service.updated_batches == [
        ("b-1", BatchStatus.GENERATION_DONE),
        ("b-2", BatchStatus.GENERATION_DONE),
    ]
    assert fake_external_sync_service.increment_pending_calls == []
    assert fake_external_sync_service.increment_target_calls == [("t-final", 50)]
    assert scheduled == [config_id]
    assert fake_external_sync_service.updated_tasks == [{"task_id": config_id, "status": SyncStatus.IDLE}]


def test_on_generation_completed_legacy_mode_triggers_training(
    fake_external_sync_service,
    monkeypatch,
):
    config_id = "cfg-legacy"
    generation_task_id = "gen-2"
    fake_external_sync_service.generation_by_task_id[generation_task_id] = {
        "task_id": config_id,
        "input_batch_ids": [],
    }
    fake_external_sync_service.all_targets = []
    fake_external_sync_service.task_raw_map[config_id] = {
        "status": SyncStatus.IDLE,
        "pending_training_samples": 100,
        "training_threshold": 100,
    }

    triggered: List[Dict[str, Any]] = []
    scheduled: List[str] = []
    monkeypatch.setattr(level2_handler, "_trigger_training", lambda cfg: triggered.append(cfg))
    monkeypatch.setattr(level2_handler, "_schedule_next_training", lambda cid: scheduled.append(cid))

    level2_handler.on_generation_completed(generation_task_id, "ds-2", 12)

    assert fake_external_sync_service.increment_pending_calls == [(config_id, 12)]
    assert fake_external_sync_service.increment_target_calls == []
    assert len(triggered) == 1
    assert scheduled == [config_id]


def test_on_generation_completed_legacy_mode_recovers_to_idle_when_not_reached(
    fake_external_sync_service,
    monkeypatch,
):
    config_id = "cfg-legacy-idle"
    generation_task_id = "gen-3"
    fake_external_sync_service.generation_by_task_id[generation_task_id] = {
        "task_id": config_id,
        "input_batch_ids": [],
    }
    fake_external_sync_service.all_targets = []
    fake_external_sync_service.task_raw_map[config_id] = {
        "status": SyncStatus.GENERATING,
        "pending_training_samples": 10,
        "training_threshold": 100,
    }

    monkeypatch.setattr(level2_handler, "_trigger_training", lambda cfg: (_ for _ in ()).throw(AssertionError("should not trigger")))  # noqa: E501
    monkeypatch.setattr(level2_handler, "_schedule_next_training", lambda cid: (_ for _ in ()).throw(AssertionError("should not schedule")))  # noqa: E501

    level2_handler.on_generation_completed(generation_task_id, "ds-3", 5)

    assert fake_external_sync_service.increment_pending_calls == [(config_id, 5)]
    assert fake_external_sync_service.updated_tasks == [{"task_id": config_id, "status": SyncStatus.IDLE}]


def test_trigger_training_prefers_highest_priority_target(monkeypatch):
    from train_factory.storage.services.external_sync_service import external_sync_service
    import train_factory.sync.sync_manager as sync_manager_module

    config_id = "cfg-manager"
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda task_id: {"task_id": task_id, "status": SyncStatus.IDLE},
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda task_id, is_active=True: [
            {"target_id": "t-low", "priority": 10, "sort_order": 0},
            {"target_id": "t-high", "priority": 0, "sort_order": 5},
            {
                "target_id": "t-high2",
                "priority": 0,
                "sort_order": 1,
                "training_config": {"api_key": "target-secret"},
            },
        ],
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets",
        lambda *_args, **_kwargs: pytest.fail(
            "manual training must not consume redacted public target DTOs"
        ),
    )

    called: List[tuple[str, Dict[str, Any]]] = []
    monkeypatch.setattr(
        level2_handler,
        "_trigger_training_for_target",
        lambda cid, target: called.append((cid, target)),
    )
    monkeypatch.setattr(
        level2_handler,
        "_trigger_training",
        lambda cfg: (_ for _ in ()).throw(AssertionError("should not use legacy trigger")),
    )

    class _ImmediateLoop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    monkeypatch.setattr(sync_manager_module.asyncio, "get_event_loop", lambda: _ImmediateLoop())

    manager = SyncManager()
    import asyncio
    asyncio.run(manager.trigger_training(config_id))
    assert called[0][0] == config_id
    assert called[0][1]["target_id"] == "t-high2"
    assert called[0][1]["training_config"]["api_key"] == "target-secret"


def test_trigger_training_falls_back_to_legacy_when_no_targets(monkeypatch):
    from train_factory.storage.services.external_sync_service import external_sync_service
    import train_factory.sync.sync_manager as sync_manager_module

    config_id = "cfg-legacy-manager"
    config = {"task_id": config_id, "status": SyncStatus.IDLE}
    monkeypatch.setattr(external_sync_service, "get_task_raw", lambda task_id: config)
    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda task_id, is_active=True: [],
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets",
        lambda *_args, **_kwargs: pytest.fail(
            "manual training must not consume redacted public target DTOs"
        ),
    )

    called: List[Dict[str, Any]] = []
    monkeypatch.setattr(level2_handler, "_trigger_training", lambda cfg: called.append(cfg))
    monkeypatch.setattr(
        level2_handler,
        "_trigger_training_for_target",
        lambda cid, target: (_ for _ in ()).throw(AssertionError("should not use target trigger")),
    )

    class _ImmediateLoop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    monkeypatch.setattr(sync_manager_module.asyncio, "get_event_loop", lambda: _ImmediateLoop())

    manager = SyncManager()
    import asyncio

    asyncio.run(manager.trigger_training(config_id))
    assert called == [config]


def test_schedule_next_training_claims_one_target_across_concurrent_callbacks(
    tmp_path,
    monkeypatch,
):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'sync-claim.db'}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    SQLModel.metadata.create_all(engine)
    config_id = "cfg-concurrent-claim"
    target_id = "target-concurrent-claim"
    with Session(engine) as session:
        session.add(ExternalSyncTaskDB(
            task_id=config_id,
            task_name="Concurrent claim",
            user_id="user-1",
        ))
        session.add(ExternalSyncTrainingTargetDB(
            target_id=target_id,
            task_id=config_id,
            target_name="Embedding",
            training_threshold=1,
            pending_training_samples=10,
            status=TrainingTargetStatus.READY,
        ))
        session.commit()

    monkeypatch.setattr(external_sync_service, "engine", engine)
    launched: List[str] = []

    def claim_and_record(_config_id, target, **kwargs):
        claimed = external_sync_service.create_training_with_claim(
            task_id=config_id,
            training_task_id=f"training-{threading.get_ident()}",
            user_id="user-1",
            input_dataset_ids=[],
            total_samples=10,
            target_id=target["target_id"],
            require_threshold=kwargs.get("require_threshold", False),
        )
        if claimed:
            launched.append(target["target_id"])
        return bool(claimed)

    monkeypatch.setattr(
        level2_handler,
        "_trigger_training_for_target",
        claim_and_record,
    )

    start = threading.Barrier(3)
    errors: List[BaseException] = []

    def schedule() -> None:
        try:
            start.wait()
            level2_handler._schedule_next_training(config_id)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=schedule) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert launched == [target_id]


def test_claim_training_target_consumes_snapshot_without_losing_concurrent_increment(
    tmp_path,
    monkeypatch,
):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'sync-pending-snapshot.db'}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    SQLModel.metadata.create_all(engine)
    config_id = "cfg-pending-snapshot"
    target_id = "target-pending-snapshot"
    with Session(engine) as session:
        session.add(ExternalSyncTaskDB(
            task_id=config_id,
            task_name="Pending snapshot",
            user_id="user-1",
        ))
        session.add(ExternalSyncTrainingTargetDB(
            target_id=target_id,
            task_id=config_id,
            target_name="Embedding",
            training_threshold=1,
            pending_training_samples=10,
            total_training_samples=10,
            status=TrainingTargetStatus.READY,
        ))
        session.commit()

    monkeypatch.setattr(external_sync_service, "engine", engine)
    start = threading.Barrier(3)
    claimed: List[Dict[str, Any]] = []
    errors: List[BaseException] = []

    def claim() -> None:
        try:
            start.wait()
            result = external_sync_service.claim_training_target(config_id)
            if result:
                claimed.append(result)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def increment() -> None:
        try:
            start.wait()
            external_sync_service.increment_target_pending_samples(target_id, 5)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=claim), threading.Thread(target=increment)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert len(claimed) == 1
    target = external_sync_service.get_training_target(target_id)
    assert target is not None
    assert claimed[0]["_claim_pending_samples"] + target["pending_training_samples"] == 15
    assert target["status"] == TrainingTargetStatus.TRAINING


def test_release_training_target_claim_restores_only_claimed_snapshot(
    tmp_path,
    monkeypatch,
):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'sync-pending-compensation.db'}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    SQLModel.metadata.create_all(engine)
    config_id = "cfg-pending-compensation"
    target_id = "target-pending-compensation"
    with Session(engine) as session:
        session.add(ExternalSyncTaskDB(
            task_id=config_id,
            task_name="Pending compensation",
            user_id="user-1",
        ))
        session.add(ExternalSyncTrainingTargetDB(
            target_id=target_id,
            task_id=config_id,
            target_name="Embedding",
            training_threshold=1,
            pending_training_samples=10,
            total_training_samples=10,
            status=TrainingTargetStatus.READY,
        ))
        session.commit()

    monkeypatch.setattr(external_sync_service, "engine", engine)
    claimed = external_sync_service.claim_training_target(config_id)
    assert claimed is not None
    external_sync_service.increment_target_pending_samples(target_id, 5)

    released = external_sync_service.release_training_target_claim(
        target_id,
        claimed["_claim_previous_status"],
        claimed.get("_claim_previous_training_id"),
        claimed_pending_samples=claimed["_claim_pending_samples"],
    )

    assert released is True
    assert external_sync_service.release_training_target_claim(
        target_id,
        claimed["_claim_previous_status"],
        claimed.get("_claim_previous_training_id"),
        claimed_pending_samples=claimed["_claim_pending_samples"],
    ) is False
    target = external_sync_service.get_training_target(target_id)
    assert target is not None
    assert target["pending_training_samples"] == 15
    assert target["status"] == TrainingTargetStatus.READY


def test_trigger_training_for_target_releases_claim_when_creation_fails(
    tmp_path,
    monkeypatch,
):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'sync-claim-rollback.db'}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    SQLModel.metadata.create_all(engine)
    config_id = "cfg-claim-rollback"
    target_id = "target-claim-rollback"
    with Session(engine) as session:
        session.add(ExternalSyncTaskDB(
            task_id=config_id,
            task_name="Claim rollback",
            user_id="user-1",
        ))
        session.add(ExternalSyncTrainingTargetDB(
            target_id=target_id,
            task_id=config_id,
            target_name="Embedding",
            status=TrainingTargetStatus.READY,
        ))
        session.commit()

    monkeypatch.setattr(external_sync_service, "engine", engine)
    monkeypatch.setattr(
        level2_handler,
        "_launch_training_for_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("create failed")),
    )

    with pytest.raises(RuntimeError, match="create failed"):
        level2_handler._trigger_training_for_target(
            config_id,
            {
                "target_id": target_id,
                "target_name": "Embedding",
            },
        )

    target = external_sync_service.get_training_target(target_id)
    assert target is not None
    assert target["status"] == TrainingTargetStatus.READY
    assert target["current_training_id"] is None


def test_target_training_thread_start_failure_rolls_back_launch_state(monkeypatch):
    from train_factory.storage.services.dataset_service import dataset_service
    from train_factory.storage.services.training_task_service import training_task_service

    config_id = "cfg-thread-start-failure"
    target_id = "target-thread-start-failure"
    task_id = "training-thread-start-failure"
    task_status_updates: List[tuple[str, str, Optional[str]]] = []
    sync_training_updates: List[tuple[str, str]] = []
    parent_status_updates: List[tuple[str, str]] = []
    released_claims: List[tuple[str, str, Optional[str], int]] = []
    recovered_claims: List[tuple[str, str]] = []
    reset_targets: List[str] = []
    admissions: List[tuple[str, Optional[str], str]] = []
    lease_releases: List[bool] = []
    parent_guard_releases: List[bool] = []
    parent_validation_calls: List[tuple[Optional[str], Optional[str], str]] = []
    parent_guard_active = False
    execution_lease = type(
        "Lease",
        (),
        {"release": lambda self: lease_releases.append(True)},
    )()

    def admit(kind, admitted_task_id, admission_user_id, operation, *args, **kwargs):
        admissions.append((kind, admitted_task_id, admission_user_id))
        return operation(*args, **kwargs), execution_lease

    def begin_deletion(kind, parent_task_id):
        nonlocal parent_guard_active
        assert (kind, parent_task_id) == ("training", "parent-target")
        parent_guard_active = True

        def release():
            nonlocal parent_guard_active
            parent_guard_active = False
            parent_guard_releases.append(True)

        return SimpleNamespace(release=release)

    def validate_parent(parent_task_id, checkpoint_path, sync_user_id):
        assert parent_guard_active is True
        parent_validation_calls.append(
            (parent_task_id, checkpoint_path, sync_user_id)
        )

    monkeypatch.setattr(
        level2_handler,
        "background_task_admission_service",
        type(
            "Admission",
            (),
            {
                "admit_execution": staticmethod(admit),
                "begin_deletion": staticmethod(begin_deletion),
                "run_sync": staticmethod(lambda *_args, **_kwargs: None),
            },
        )(),
        raising=False,
    )
    monkeypatch.setattr(
        level2_handler,
        "_validate_sync_training_parent_checkpoint",
        validate_parent,
    )

    monkeypatch.setattr(
        external_sync_service,
        "get_all_completed_generation_datasets",
        lambda _config_id: [{"dataset_id": "dataset-1", "sample_count": 12}],
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {"storage_path": "/app/data/dataset-1.jsonl"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _config_id: {
            "task_id": config_id,
            "user_id": "user-1",
            "status": SyncStatus.IDLE,
        },
    )
    monkeypatch.setattr(
        level2_handler,
        "_build_sync_target_output_dir",
        lambda *_args, **_kwargs: "/app/output/sync/failure",
    )

    from train_factory.api.routes import training_routes

    monkeypatch.setattr(
        training_routes,
        "_normalize_training_config_paths",
        lambda config, _user_id: {
            **config,
            "base_model_path": "/app/models/base",
            "train_dataset_path": "/app/data/dataset-1.jsonl",
        },
    )
    monkeypatch.setattr(
        training_task_service,
        "create_task",
        lambda **_kwargs: {"task_id": task_id},
    )
    monkeypatch.setattr(level2_handler.uuid, "uuid4", lambda: task_id)
    monkeypatch.setattr(
        training_task_service,
        "update_task_output_dir",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        training_task_service,
        "update_task_status",
        lambda updated_task_id, status, error_message=None: task_status_updates.append(
            (updated_task_id, status, error_message)
        ) or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_training_with_claim",
        lambda **kwargs: {"training_task_id": kwargs["training_task_id"]},
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_training_and_restore_claim",
        lambda failed_task_id, reason: recovered_claims.append(
            (failed_task_id, reason)
        ) or {"recovered": True},
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_training_status",
        lambda updated_task_id, status, **_kwargs: sync_training_updates.append(
            (updated_task_id, status)
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_training_target",
        lambda _target_id, **kwargs: {"target_id": target_id, **kwargs},
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda updated_config_id, **kwargs: parent_status_updates.append(
            (updated_config_id, kwargs["status"])
        ) or {"task_id": updated_config_id, **kwargs},
    )
    monkeypatch.setattr(
        external_sync_service,
        "compare_and_set_task_status",
        lambda updated_config_id, expected_status, status: parent_status_updates.append(
            (updated_config_id, status)
        ) or expected_status == SyncStatus.TRAINING,
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "release_training_target_claim",
        lambda released_target_id, previous_status, previous_training_id=None,
        claimed_pending_samples=0: released_claims.append(
            (
                released_target_id,
                previous_status,
                previous_training_id,
                claimed_pending_samples,
            )
        ) or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "reset_target_pending_samples",
        lambda reset_target_id: reset_targets.append(reset_target_id),
    )

    class _FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading, "Thread", _FailingThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        level2_handler._trigger_training_for_target(
            config_id,
            {
                "target_id": target_id,
                "target_name": "Embedding",
                "model_type": "embedding",
                "training_method": "sft",
                "training_config": {
                    "parent_task_id": "parent-target",
                    "sft_checkpoint_path": (
                        "/app/output/parent-target/checkpoint-10"
                    ),
                },
                "base_model_path": "/app/models/base",
                "data_phase": "final",
                "total_trainings": 0,
                "_claim_previous_status": TrainingTargetStatus.READY,
                "_claim_previous_training_id": "previous-training-id",
                "_claim_pending_samples": 12,
            },
        )

    assert task_status_updates == [
        (task_id, TrainingStatus.FAILED.value, "Training thread failed to start: thread start failed")
    ]
    assert sync_training_updates == []
    assert parent_status_updates == []
    assert recovered_claims == [
        (task_id, "Training thread failed to start: thread start failed")
    ]
    assert released_claims == []
    assert reset_targets == []
    assert admissions == [("training", task_id, "user-1")]
    assert lease_releases == [True]
    assert parent_validation_calls == [
        (
            "parent-target",
            "/app/output/parent-target/checkpoint-10",
            "user-1",
        )
    ]
    assert parent_guard_releases == [True]


def test_legacy_training_thread_start_failure_compensates_created_records(monkeypatch):
    import importlib

    from train_factory.api.routes import training_routes
    from train_factory.storage.services.dataset_service import dataset_service
    from train_factory.storage.services.training_task_service import training_task_service

    settings_module = importlib.import_module("train_factory.config.settings")

    config_id = "cfg-legacy-thread-start-failure"
    task_id = "legacy-training-thread-start-failure"
    task_status_updates: List[tuple[str, str, Optional[str]]] = []
    sync_training_updates: List[tuple[str, str]] = []
    parent_status_updates: List[Dict[str, Any]] = []
    reset_tasks: List[str] = []
    created_task_configs: List[Dict[str, Any]] = []
    admissions: List[tuple[str, Optional[str], str]] = []
    lease_releases: List[bool] = []
    parent_guard_releases: List[bool] = []
    parent_validation_calls: List[tuple[Optional[str], Optional[str], str]] = []
    parent_guard_active = False
    recovered_claims: List[tuple[str, str]] = []
    execution_lease = type(
        "Lease",
        (),
        {"release": lambda self: lease_releases.append(True)},
    )()

    def admit(kind, admitted_task_id, admission_user_id, operation, *args, **kwargs):
        admissions.append((kind, admitted_task_id, admission_user_id))
        return operation(*args, **kwargs), execution_lease

    def begin_deletion(kind, parent_task_id):
        nonlocal parent_guard_active
        assert (kind, parent_task_id) == ("training", "parent-legacy")
        parent_guard_active = True

        def release():
            nonlocal parent_guard_active
            parent_guard_active = False
            parent_guard_releases.append(True)

        return SimpleNamespace(release=release)

    def validate_parent(parent_task_id, checkpoint_path, sync_user_id):
        assert parent_guard_active is True
        parent_validation_calls.append(
            (parent_task_id, checkpoint_path, sync_user_id)
        )

    monkeypatch.setattr(
        level2_handler,
        "background_task_admission_service",
        type(
            "Admission",
            (),
            {
                "admit_execution": staticmethod(admit),
                "begin_deletion": staticmethod(begin_deletion),
                "run_sync": staticmethod(lambda *_args, **_kwargs: None),
            },
        )(),
        raising=False,
    )
    monkeypatch.setattr(
        level2_handler,
        "_validate_sync_training_parent_checkpoint",
        validate_parent,
    )

    monkeypatch.setattr(
        external_sync_service,
        "get_all_completed_generation_datasets",
        lambda _config_id: [
            {
                "output_dataset_id": "dataset-legacy-1",
                "output_sample_count": 20,
            }
        ],
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "storage_path": None,
            "storage_uri": "s3://trainfactory/dataset-legacy-1.jsonl",
        },
    )
    monkeypatch.setattr(
        training_routes,
        "_normalize_training_config_paths",
        lambda config, _user_id: config,
    )

    class _Settings:
        @staticmethod
        def get_task_output_dir(_task_id):
            return "/app/output/legacy-failure"

    monkeypatch.setattr(settings_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(level2_handler.uuid, "uuid4", lambda: task_id)
    monkeypatch.setattr(
        training_task_service,
        "create_task",
        lambda **kwargs: created_task_configs.append(kwargs) or {"task_id": task_id},
    )
    monkeypatch.setattr(
        training_task_service,
        "update_task_output_dir",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        training_task_service,
        "update_task_status",
        lambda updated_task_id, status, error_message=None: task_status_updates.append(
            (updated_task_id, status, error_message)
        ) or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_training_with_claim",
        lambda **kwargs: {"training_task_id": kwargs["training_task_id"]},
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_training_and_restore_claim",
        lambda failed_task_id, reason: recovered_claims.append(
            (failed_task_id, reason)
        ) or {"recovered": True},
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_training_status",
        lambda updated_task_id, status, **_kwargs: sync_training_updates.append(
            (updated_task_id, status)
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda updated_config_id, **kwargs: parent_status_updates.append(
            {"task_id": updated_config_id, **kwargs}
        ) or {"task_id": updated_config_id, **kwargs},
    )
    monkeypatch.setattr(
        external_sync_service,
        "compare_and_set_task_status",
        lambda updated_config_id, expected_status, status, error_message=None: parent_status_updates.append(
            {
                "task_id": updated_config_id,
                "expected_status": expected_status,
                "status": status,
                "error_message": error_message,
            }
        ) or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "reset_pending_training_samples",
        lambda reset_task_id: reset_tasks.append(reset_task_id),
    )

    class _FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("legacy thread start failed")

    monkeypatch.setattr(threading, "Thread", _FailingThread)

    level2_handler._trigger_training(
        {
            "task_id": config_id,
            "user_id": "user-1",
            "total_trainings": 0,
            "current_training_id": None,
            "training_config": {
                "base_model_path": "/app/models/base",
                "model_type": "embedding",
                "training_method": "sft",
                "parent_task_id": "parent-legacy",
                "sft_checkpoint_path": (
                    "/app/output/parent-legacy/checkpoint-10"
                ),
            },
        }
    )

    failure_message = "Training thread failed to start: legacy thread start failed"
    assert task_status_updates == [
        (task_id, TrainingStatus.FAILED.value, failure_message)
    ]
    assert sync_training_updates == []
    assert created_task_configs[0]["train_dataset_path"] == (
        "s3://trainfactory/dataset-legacy-1.jsonl"
    )
    assert created_task_configs[0]["training_params"]["dataset_configs"][0]["path"] == (
        "s3://trainfactory/dataset-legacy-1.jsonl"
    )
    assert recovered_claims == [(task_id, failure_message)]
    assert parent_status_updates == [
        {
            "task_id": config_id,
            "expected_status": SyncStatus.TRAINING,
            "status": SyncStatus.ERROR,
            "error_message": f"Trigger training failed: {failure_message}",
        }
    ]
    assert reset_tasks == []
    assert admissions == [("training", task_id, "user-1")]
    assert lease_releases == [True]
    assert parent_validation_calls == [
        (
            "parent-legacy",
            "/app/output/parent-legacy/checkpoint-10",
            "user-1",
        )
    ]
    assert parent_guard_releases == [True]
def test_delete_training_target_rejects_pending_tracking(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'target-delete-guard.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    config_id = "cfg-target-delete"
    target_id = "target-delete"
    training_task_id = "training-target-delete"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=config_id,
                task_name="Target delete guard",
                user_id="user-1",
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id=target_id,
                task_id=config_id,
                target_name="Embedding",
                status=TrainingTargetStatus.ERROR,
            )
        )
        session.add(
            ExternalSyncTrainingDB(
                task_id=config_id,
                training_task_id=training_task_id,
                user_id="user-1",
                target_id=target_id,
                status=SyncTrainingStatus.PENDING,
            )
        )
        session.commit()

    monkeypatch.setattr(external_sync_service, "engine", engine)

    with pytest.raises(ValueError, match="active training"):
        external_sync_service.delete_training_target(target_id)

    assert external_sync_service.get_training_target(target_id) is not None
def test_sync_parent_validation_allows_auth_disabled_owner_without_parent(
    monkeypatch,
):
    import importlib

    from train_factory.api.routes import training_routes

    settings_module = importlib.import_module("train_factory.config.settings")
    calls = []
    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=False),
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_training_parent_checkpoint",
        lambda parent_task_id, checkpoint_path, current_user: calls.append(
            (parent_task_id, checkpoint_path, current_user)
        ),
    )

    level2_handler._validate_sync_training_parent_checkpoint(None, None, "")

    assert calls == [(None, None, {"user_id": "", "is_admin": False})]
