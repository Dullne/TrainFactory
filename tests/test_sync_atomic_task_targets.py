from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncBatchDB,
    ExternalSyncGenerationDB,
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.storage.services.external_sync_service import ExternalSyncService
from train_factory.api.routes import sync_routes
from train_factory.storage.services.external_sync_service import external_sync_service
from train_factory.sync import sync_manager as sync_manager_module
from train_factory.sync import level2_handler


def _service() -> tuple[ExternalSyncService, object]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in (
        ModelRegistryDB.__table__,
        DeploymentDB.__table__,
        DeploymentReplicaDB.__table__,
        ExternalSyncTaskDB.__table__,
        ExternalSyncBatchDB.__table__,
        ExternalSyncGenerationDB.__table__,
        ExternalSyncTrainingDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
    ):
        table.create(engine)
    service = ExternalSyncService()
    service.engine = engine
    return service, engine


def _file_service(tmp_path) -> tuple[ExternalSyncService, object]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'atomic-task-targets.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    for table in (
        ModelRegistryDB.__table__,
        DeploymentDB.__table__,
        DeploymentReplicaDB.__table__,
        ExternalSyncTaskDB.__table__,
        ExternalSyncBatchDB.__table__,
        ExternalSyncGenerationDB.__table__,
        ExternalSyncTrainingDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
    ):
        table.create(engine)
    service = ExternalSyncService()
    service.engine = engine
    return service, engine


def _target(target_id: str, name: str, deployment_id: str | None = None) -> dict:
    return {
        "target_id": target_id,
        "target_name": name,
        "model_type": "embedding",
        "data_phase": "final",
        "training_method": "sft",
        "training_config": {"epochs": 1},
        "base_model_path": "models/base",
        "base_deployment_id": deployment_id,
        "base_deployment_replica_id": None,
        "training_threshold": 100,
        "priority": 0,
        "sort_order": 0,
    }


def test_atomic_task_target_update_rolls_back_when_later_target_is_invalid():
    service, engine = _service()
    created = service.create_task(
        task_name="before",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        is_active=False,
        training_targets=[
            _target("target-a", "A"),
            _target("target-b", "B"),
        ],
    )
    task_id = created["task_id"]
    with Session(engine) as session:
        session.add_all(
            [
                DeploymentDB(
                    deployment_id="deployment-a-good",
                    model_id="model-a",
                    deployment_name="good",
                    xinference_endpoint="http://127.0.0.1:11000",
                    user_id="user-1",
                ),
                DeploymentDB(
                    deployment_id="deployment-z-bad",
                    model_id="model-z",
                    deployment_name="bad",
                    xinference_endpoint="http://127.0.0.1:12000",
                    user_id="user-2",
                ),
            ]
        )
        session.commit()

    with pytest.raises(ValueError, match="^Runtime dependency is unavailable$"):
        service.update_task(
            task_id,
            expected_user_id="user-1",
            task_name="after",
            training_targets=[
                _target("target-a", "A changed", "deployment-a-good"),
                _target("target-b", "B changed", "deployment-z-bad"),
            ],
        )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        targets = session.exec(
            select(ExternalSyncTrainingTargetDB)
            .where(ExternalSyncTrainingTargetDB.task_id == task_id)
            .order_by(ExternalSyncTrainingTargetDB.target_id)
        ).all()
        assert task.task_name == "before"
        assert [(target.target_id, target.target_name, target.base_deployment_id) for target in targets] == [
            ("target-a", "A", None),
            ("target-b", "B", None),
        ]


def test_legacy_pending_migration_rolls_back_then_retries_without_duplication(
    monkeypatch,
):
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-legacy",
                task_name="legacy",
                user_id="user-1",
                external_api_url="https://sync.example.com/data",
                training_threshold=100,
                training_config={"model_type": "embedding"},
                pending_training_samples=88,
                is_active=False,
            )
        )
        session.commit()

    migration = {
        "task_id": "sync-legacy",
        "target_id": "legacy-sync-legacy",
        "target_name": "EMBEDDING (migrated)",
        "model_type": "embedding",
        "data_phase": "final",
        "training_method": "sft",
        "training_config": {"model_type": "embedding"},
        "base_model_path": "models/base",
        "base_deployment_id": None,
        "base_deployment_replica_id": None,
        "training_threshold": 100,
        "expected_user_id": "user-1",
    }

    original_commit = Session.commit
    with monkeypatch.context() as scoped:
        scoped.setattr(
            Session,
            "commit",
            lambda _session: (_ for _ in ()).throw(RuntimeError("commit failed")),
        )
        with pytest.raises(RuntimeError, match="commit failed"):
            service.migrate_legacy_training_target(**migration)

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == "sync-legacy"
            )
        ).one()
        assert task.pending_training_samples == 88
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []

    assert Session.commit is original_commit
    migrated = service.migrate_legacy_training_target(**migration)
    repeated = service.migrate_legacy_training_target(**migration)

    assert migrated["pending_training_samples"] == 88
    assert repeated["target_id"] == migrated["target_id"]
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == "sync-legacy"
            )
        ).one()
        targets = session.exec(select(ExternalSyncTrainingTargetDB)).all()
        assert task.pending_training_samples == 0
        assert len(targets) == 1
        assert targets[0].pending_training_samples == 88
        assert targets[0].total_training_samples == 88
        assert targets[0].training_config == {"model_type": "embedding"}
        assert targets[0].training_threshold == 100


def test_update_route_passes_complete_target_set_to_one_service_call(monkeypatch):
    stored = {
        "task_id": "sync-route-atomic",
        "task_name": "before",
        "user_id": "user-1",
        "status": "idle",
        "is_active": False,
        "external_api_config_id": None,
        "base_deployment_id": None,
        "base_deployment_replica_id": None,
    }
    update_calls = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(stored),
    )

    def update_task(_task_id, **kwargs):
        update_calls.append(kwargs)
        return {**stored, **kwargs}

    monkeypatch.setattr(external_sync_service, "update_task", update_task)

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )

    response = asyncio.run(
        sync_routes.update_sync_task(
            "sync-route-atomic",
            sync_routes.SyncTaskUpdateRequest(
                task_name="after",
                training_targets=[
                    {
                        **_target("target-a", "A changed"),
                    },
                    {
                        **_target("target-c", "C new"),
                    },
                ],
            ),
            {"user_id": "user-1", "username": "owner"},
        )
    )

    assert len(update_calls) == 1
    assert update_calls[0]["expected_user_id"] == "user-1"
    assert [
        item["target_id"] for item in update_calls[0]["training_targets"]
    ] == ["target-a", "target-c"]
    assert response["task"]["task_name"] == "after"


def test_atomic_task_target_update_explicit_empty_list_deletes_all_targets():
    service, engine = _service()
    created = service.create_task(
        task_name="clear",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        is_active=False,
        training_targets=[_target("target-a", "A"), _target("target-b", "B")],
    )

    updated = service.update_task(
        created["task_id"],
        expected_user_id="user-1",
        training_targets=[],
    )

    assert updated["training_targets"] == []
    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []


def test_atomic_task_target_update_omitted_targets_preserves_existing_set():
    service, engine = _service()
    created = service.create_task(
        task_name="before",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        is_active=False,
        training_targets=[_target("target-a", "A"), _target("target-b", "B")],
    )

    updated = service.update_task(
        created["task_id"],
        expected_user_id="user-1",
        task_name="after",
    )

    assert updated["task_name"] == "after"
    assert "training_targets" not in updated
    with Session(engine) as session:
        targets = session.exec(
            select(ExternalSyncTrainingTargetDB).order_by(
                ExternalSyncTrainingTargetDB.target_id
            )
        ).all()
        assert [(target.target_id, target.target_name) for target in targets] == [
            ("target-a", "A"),
            ("target-b", "B"),
        ]


def test_atomic_task_target_update_applies_create_update_delete_together():
    service, engine = _service()
    created = service.create_task(
        task_name="before",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        is_active=False,
        training_targets=[_target("target-a", "A"), _target("target-b", "B")],
    )
    with Session(engine) as session:
        target_a = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == "target-a"
            )
        ).one()
        target_a.pending_training_samples = 17
        target_a.total_training_samples = 29
        session.add(target_a)
        session.commit()

    updated = service.update_task(
        created["task_id"],
        expected_user_id="user-1",
        task_name="after",
        training_targets=[
            _target("target-a", "A changed"),
            _target("target-c", "C new"),
        ],
    )

    assert updated["task_name"] == "after"
    assert [item["target_id"] for item in updated["training_targets"]] == [
        "target-a",
        "target-c",
    ]
    with Session(engine) as session:
        targets = {
            target.target_id: target
            for target in session.exec(select(ExternalSyncTrainingTargetDB)).all()
        }
        assert set(targets) == {"target-a", "target-c"}
        assert targets["target-a"].target_name == "A changed"
        assert targets["target-a"].pending_training_samples == 17
        assert targets["target-a"].total_training_samples == 29
        assert targets["target-c"].pending_training_samples == 0


def test_atomic_task_target_update_rejects_duplicate_ids_without_mutation():
    service, engine = _service()
    created = service.create_task(
        task_name="before",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        is_active=False,
        training_targets=[_target("target-a", "A")],
    )

    with pytest.raises(ValueError, match="Duplicate training target_id"):
        service.update_task(
            created["task_id"],
            expected_user_id="user-1",
            task_name="after",
            training_targets=[
                _target("target-a", "A changed"),
                _target("target-a", "duplicate"),
            ],
        )

    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        assert task.task_name == "before"
        assert target.target_name == "A"


def test_legacy_pending_migration_is_concurrent_and_repeat_safe(tmp_path):
    service, engine = _file_service(tmp_path)
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-legacy-race",
                task_name="legacy-race",
                user_id="user-1",
                external_api_url="https://sync.example.com/data",
                training_threshold=100,
                training_config={"model_type": "embedding"},
                pending_training_samples=88,
                is_active=False,
            )
        )
        session.commit()
    target_id = sync_routes._legacy_training_target_id("sync-legacy-race")
    migration = {
        "task_id": "sync-legacy-race",
        "target_id": target_id,
        "target_name": "EMBEDDING (migrated)",
        "training_config": {"model_type": "embedding"},
        "training_threshold": 100,
        "expected_user_id": "user-1",
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _index: service.migrate_legacy_training_target(**migration),
                range(2),
            )
        )

    assert {result["target_id"] for result in results} == {target_id}
    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        targets = session.exec(select(ExternalSyncTrainingTargetDB)).all()
        assert task.pending_training_samples == 0
        assert task.training_config is None
        assert task.training_threshold == 0
        assert len(targets) == 1
        assert targets[0].pending_training_samples == 88
        assert targets[0].total_training_samples == 88


def test_legacy_materialize_existing_uuid_transfers_parent_pending_once():
    service, engine = _service()
    task_id = "sync-existing-legacy"
    target_id = sync_routes._legacy_training_target_id(task_id)
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="existing legacy",
                    user_id="user-1",
                    external_api_url="https://sync.example.com/data",
                    training_threshold=100,
                    training_config={"model_type": "embedding"},
                    pending_training_samples=88,
                    is_active=False,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id=target_id,
                    task_id=task_id,
                    target_name="existing legacy target",
                    pending_training_samples=7,
                    total_training_samples=10,
                ),
            ]
        )
        session.commit()

    migration = {
        "task_id": task_id,
        "target_id": target_id,
        "target_name": "EMBEDDING (legacy)",
        "training_config": {"model_type": "embedding"},
        "training_threshold": 100,
        "expected_user_id": "user-1",
    }
    first = service.migrate_legacy_training_target(**migration)
    second = service.migrate_legacy_training_target(**migration)

    assert first["pending_training_samples"] == 95
    assert second["pending_training_samples"] == 95
    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        assert task.pending_training_samples == 0
        assert task.training_config is None
        assert task.training_threshold == 0
        assert target.pending_training_samples == 95
        assert target.total_training_samples == 98


def test_update_request_distinguishes_omitted_targets_from_explicit_empty_list():
    omitted = sync_routes.SyncTaskUpdateRequest(task_name="keep targets")
    cleared = sync_routes.SyncTaskUpdateRequest(training_targets=[])

    assert "training_targets" not in omitted.model_dump(exclude_unset=True)
    assert cleared.model_dump(exclude_unset=True)["training_targets"] == []


def test_update_route_preserves_explicit_null_for_clearable_configs(monkeypatch):
    stored = {
        "task_id": "sync-clear-configs",
        "task_name": "clear configs",
        "user_id": "user-1",
        "status": "idle",
        "is_active": False,
        "external_api_config_id": None,
        "generation_config": {"llm_config": {"config_id": "llm-1"}},
        "training_config": {"model_type": "embedding"},
        "base_deployment_id": None,
        "base_deployment_replica_id": None,
    }
    calls = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(stored),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda _task_id, **kwargs: calls.append(kwargs) or {**stored, **kwargs},
    )

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )

    response = asyncio.run(
        sync_routes.update_sync_task(
            "sync-clear-configs",
            sync_routes.SyncTaskUpdateRequest(
                generation_config=None,
                training_config=None,
            ),
            {"user_id": "user-1", "username": "owner"},
        )
    )

    assert calls == [
        {
            "expected_user_id": "user-1",
            "generation_config": None,
            "training_config": None,
        }
    ]
    assert response["task"]["generation_config"] is None
    assert response["task"]["training_config"] is None


def test_service_clears_explicit_null_configs_but_omission_preserves_them():
    service, engine = _service()
    created = service.create_task(
        task_name="clear configs",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        generation_config={"llm_config": {"config_id": "llm-1"}},
        training_config={"model_type": "embedding"},
        is_active=False,
    )

    omitted = service.update_task(
        created["task_id"],
        expected_user_id="user-1",
        task_name="still configured",
    )
    assert omitted["generation_config"] == {
        "llm_config": {"config_id": "llm-1"}
    }
    assert omitted["training_config"] == {"model_type": "embedding"}

    cleared = service.update_task(
        created["task_id"],
        expected_user_id="user-1",
        generation_config=None,
        training_config=None,
    )
    assert cleared["generation_config"] is None
    assert cleared["training_config"] is None
    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        assert task.generation_config is None
        assert task.training_config is None


def test_get_legacy_training_targets_is_repeatable_and_has_zero_mutations(
    monkeypatch,
):
    class ReadOnlyLegacyService:
        def __init__(self):
            self.migration_calls = []

        def list_training_targets(self, _task_id, is_active=True):
            return []

        def migrate_legacy_training_target(self, **kwargs):
            self.migration_calls.append(kwargs)
            return {"target_id": kwargs["target_id"]}

    service = ReadOnlyLegacyService()
    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        service,
    )
    config = {
        "task_id": "sync-legacy-read",
        "task_name": "legacy read",
        "user_id": "user-1",
        "training_threshold": 100,
        "pending_training_samples": 88,
        "total_training_samples": 144,
        "total_trainings": 2,
        "training_config": {
            "model_type": "embedding",
            "training_method": "sft",
            "base_model_path": "models/base",
        },
        "base_deployment_id": None,
        "base_deployment_replica_id": None,
        "status": "idle",
        "is_active": True,
        "created_at": "2026-08-20T00:00:00",
        "updated_at": "2026-08-21T00:00:00",
    }
    before = copy.deepcopy(config)

    first = sync_routes._get_task_training_targets("sync-legacy-read", config)
    second = sync_routes._get_task_training_targets("sync-legacy-read", config)

    assert service.migration_calls == []
    assert config == before
    assert first == second
    assert first == [
        {
            "target_id": sync_routes._legacy_training_target_id(
                "sync-legacy-read"
            ),
            "task_id": "sync-legacy-read",
            "target_name": "EMBEDDING (legacy)",
            "model_type": "embedding",
            "data_phase": "final",
            "training_method": "sft",
            "training_config": config["training_config"],
            "base_model_path": "models/base",
            "base_deployment_id": None,
            "base_deployment_replica_id": None,
            "training_threshold": 100,
            "pending_training_samples": 88,
            "total_training_samples": 144,
            "total_trainings": 2,
            "current_adapter_name": None,
            "current_adapter_id": None,
            "current_training_id": None,
            "priority": 0,
            "status": "idle",
            "is_active": True,
            "sort_order": 0,
            "created_at": "2026-08-20T00:00:00",
            "updated_at": "2026-08-21T00:00:00",
        }
    ]


def test_get_synthetic_legacy_target_can_be_patched_via_controlled_write(
    monkeypatch,
):
    service, engine = _service()
    created = service.create_task(
        task_name="legacy patch",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        training_threshold=100,
        training_config={
            "model_type": "embedding",
            "training_method": "sft",
            "base_model_path": "models/base",
        },
        is_active=False,
    )
    task_id = created["task_id"]
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == task_id
            )
        ).one()
        task.pending_training_samples = 88
        session.add(task)
        session.commit()

    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        service,
    )

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )

    config = service.get_task(task_id)
    synthetic = sync_routes._get_task_training_targets(task_id, config)
    assert len(synthetic) == 1
    target_id = synthetic[0]["target_id"]
    assert target_id == sync_routes._legacy_training_target_id(task_id)
    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []
        assert session.exec(select(ExternalSyncTaskDB)).one().pending_training_samples == 88

    response = asyncio.run(
        sync_routes.update_training_target(
            task_id,
            target_id,
            sync_routes.TrainingTargetUpdateRequest(target_name="legacy renamed"),
            {"user_id": "user-1", "username": "owner"},
        )
    )

    assert response["target"]["target_name"] == "legacy renamed"
    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        assert task.pending_training_samples == 0
        assert target.target_id == target_id
        assert target.target_name == "legacy renamed"
        assert target.pending_training_samples == 88
        assert target.training_config == {
            "model_type": "embedding",
            "training_method": "sft",
            "base_model_path": "models/base",
        }
        assert target.training_threshold == 100


def test_get_synthetic_legacy_target_can_be_deleted_via_existing_semantics(
    monkeypatch,
):
    service, engine = _service()
    created = service.create_task(
        task_name="legacy delete",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        training_threshold=100,
        training_config={"model_type": "embedding"},
        is_active=False,
    )
    task_id = created["task_id"]
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == task_id
            )
        ).one()
        task.pending_training_samples = 88
        session.add(task)
        session.commit()

    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        service,
    )

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )

    config = service.get_task(task_id)
    target_id = sync_routes._get_task_training_targets(task_id, config)[0][
        "target_id"
    ]
    response = asyncio.run(
        sync_routes.delete_training_target(
            task_id,
            target_id,
            {"user_id": "user-1", "username": "owner"},
        )
    )

    assert response == {"message": "Training target deleted"}
    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        assert task.pending_training_samples == 0
        assert task.training_config is None
        assert task.training_threshold == 0
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []
    config = service.get_task(task_id)
    assert sync_routes._get_task_training_targets(task_id, config) == []
    assert sync_routes._get_task_training_targets(task_id, config) == []


def test_atomic_empty_target_set_deletes_synthetic_without_resurrection(
    monkeypatch,
):
    service, engine = _service()
    created = service.create_task(
        task_name="legacy atomic delete",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        training_threshold=100,
        training_config={"model_type": "embedding"},
        is_active=False,
    )
    task_id = created["task_id"]
    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        task.pending_training_samples = 88
        session.add(task)
        session.commit()

    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        service,
    )

    before = service.get_task(task_id)
    assert len(sync_routes._get_task_training_targets(task_id, before)) == 1

    updated = service.update_task(
        task_id,
        expected_user_id="user-1",
        training_targets=[],
    )

    assert updated["training_targets"] == []
    assert updated["pending_training_samples"] == 0
    assert updated["training_config"] is None
    assert updated["training_threshold"] == 0
    after = service.get_task(task_id)
    assert sync_routes._get_task_training_targets(task_id, after) == []
    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []


def test_delete_persisted_legacy_uuid_clears_only_legacy_training_sources(
    monkeypatch,
):
    service, engine = _service()
    task_id = "sync-persisted-legacy-delete"
    target_id = sync_routes._legacy_training_target_id(task_id)
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="persisted legacy",
                    user_id="user-1",
                    external_api_url="https://sync.example.com/data",
                    training_threshold=100,
                    training_config={"model_type": "embedding"},
                    base_deployment_id="deployment-preserved",
                    pending_training_samples=0,
                    is_active=False,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id=target_id,
                    task_id=task_id,
                    target_name="legacy persisted",
                    pending_training_samples=88,
                ),
            ]
        )
        session.commit()

    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        service,
    )

    assert service.delete_training_target(
        target_id,
        task_id=task_id,
        expected_user_id="user-1",
    )

    config = service.get_task(task_id)
    assert config["training_config"] is None
    assert config["training_threshold"] == 0
    assert config["base_deployment_id"] == "deployment-preserved"
    assert sync_routes._get_task_training_targets(task_id, config) == []


def test_delete_ordinary_target_preserves_parent_training_settings(monkeypatch):
    service, _engine = _service()
    created = service.create_task(
        task_name="ordinary target",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        training_threshold=321,
        training_config={"model_type": "reranker", "epochs": 3},
        is_active=False,
        training_targets=[_target("ordinary-target", "ordinary")],
    )
    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        service,
    )

    assert service.delete_training_target(
        "ordinary-target",
        task_id=created["task_id"],
        expected_user_id="user-1",
    )

    config = service.get_task(created["task_id"])
    assert config["training_config"] == {
        "model_type": "reranker",
        "epochs": 3,
    }
    assert config["training_threshold"] == 321
    synthetic = sync_routes._get_task_training_targets(
        created["task_id"],
        config,
    )
    assert [target["target_id"] for target in synthetic] == [
        sync_routes._legacy_training_target_id(created["task_id"])
    ]


def test_non_synthetic_missing_target_is_404_without_legacy_materialization(
    monkeypatch,
):
    service, _engine = _service()
    created = service.create_task(
        task_name="legacy missing",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        training_threshold=100,
        training_config={"model_type": "embedding"},
        is_active=False,
    )
    migration_calls = []
    original_migrate = service.migrate_legacy_training_target

    def record_migration(**kwargs):
        migration_calls.append(kwargs)
        return original_migrate(**kwargs)

    monkeypatch.setattr(service, "migrate_legacy_training_target", record_migration)
    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        service,
    )

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "task_operation_lock",
        operation_lock,
    )

    with pytest.raises(sync_routes.HTTPException) as exc_info:
        asyncio.run(
            sync_routes.delete_training_target(
                created["task_id"],
                "not-the-synthetic-target",
                {"user_id": "user-1", "username": "owner"},
            )
        )

    assert exc_info.value.status_code == 404
    assert migration_calls == []


def test_foreign_synthetic_id_collision_is_404_without_legacy_materialization(
    monkeypatch,
):
    service, _engine = _service()
    owned = service.create_task(
        task_name="legacy owned",
        user_id="user-1",
        external_api_url="https://sync.example.com/owned",
        training_threshold=100,
        training_config={"model_type": "embedding"},
        is_active=False,
    )
    target_id = sync_routes._legacy_training_target_id(owned["task_id"])
    service.create_task(
        task_name="foreign parent",
        user_id="user-1",
        external_api_url="https://sync.example.com/foreign",
        is_active=False,
        training_targets=[_target(target_id, "foreign collision")],
    )
    migration_calls = []
    original_migrate = service.migrate_legacy_training_target

    def record_migration(**kwargs):
        migration_calls.append(kwargs)
        return original_migrate(**kwargs)

    monkeypatch.setattr(service, "migrate_legacy_training_target", record_migration)
    monkeypatch.setattr(
        "train_factory.storage.services.external_sync_service.external_sync_service",
        service,
    )

    with pytest.raises(sync_routes.HTTPException) as exc_info:
        asyncio.run(
            sync_routes.update_training_target(
                owned["task_id"],
                target_id,
                sync_routes.TrainingTargetUpdateRequest(target_name="stolen"),
                {"user_id": "user-1", "username": "owner"},
            )
        )

    assert exc_info.value.status_code == 404
    assert migration_calls == []


def test_atomic_update_locks_union_of_current_and_requested_target_ids_once():
    service, engine = _service()
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-a",
                    task_name="A",
                    user_id="user-1",
                    external_api_url="https://sync.example.com/a",
                    is_active=False,
                ),
                ExternalSyncTaskDB(
                    task_id="sync-b",
                    task_name="B",
                    user_id="user-1",
                    external_api_url="https://sync.example.com/b",
                    is_active=False,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-a",
                    task_id="sync-a",
                    target_name="A target",
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-b",
                    task_id="sync-b",
                    target_name="B target",
                ),
            ]
        )
        session.commit()

    target_selects = []

    def record_target_select(
        _connection,
        _cursor,
        statement,
        parameters,
        _context,
        _executemany,
    ):
        normalized = statement.lower()
        if (
            normalized.lstrip().startswith("select")
            and "from external_sync_training_targets" in normalized
            and "order by external_sync_training_targets.target_id" in normalized
        ):
            target_selects.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", record_target_select)
    try:
        with pytest.raises(
            ValueError,
            match="belongs to another sync task",
        ):
            service.update_task(
                "sync-a",
                expected_user_id="user-1",
                training_targets=[_target("target-b", "B stolen")],
            )
    finally:
        event.remove(engine, "before_cursor_execute", record_target_select)

    assert len(target_selects) == 1
    assert tuple(
        value
        for value in target_selects[0][1]
        if isinstance(value, str) and value.startswith("target-")
    ) == ("target-a", "target-b")


def test_concurrent_cross_task_target_id_swap_fails_without_deadlock(tmp_path):
    service, engine = _file_service(tmp_path)
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-a",
                    task_name="A",
                    user_id="user-1",
                    external_api_url="https://sync.example.com/a",
                    is_active=False,
                ),
                ExternalSyncTaskDB(
                    task_id="sync-b",
                    task_name="B",
                    user_id="user-1",
                    external_api_url="https://sync.example.com/b",
                    is_active=False,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-a",
                    task_id="sync-a",
                    target_name="A target",
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-b",
                    task_id="sync-b",
                    target_name="B target",
                ),
            ]
        )
        session.commit()

    def swap(task_id, requested_id):
        try:
            service.update_task(
                task_id,
                expected_user_id="user-1",
                training_targets=[_target(requested_id, "swap")],
            )
        except ValueError as exc:
            return str(exc)
        return "unexpected success"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(swap, "sync-a", "target-b")
        second = pool.submit(swap, "sync-b", "target-a")
        errors = [first.result(timeout=10), second.result(timeout=10)]

    assert all("belongs to another sync task" in error for error in errors)


def test_explicit_target_trigger_migrates_synthetic_legacy_target_for_write(
    monkeypatch,
):
    task_id = "sync-legacy-trigger"
    target_id = sync_routes._legacy_training_target_id(task_id)
    parent = {
        "task_id": task_id,
        "task_name": "legacy trigger",
        "user_id": "user-1",
        "status": "idle",
        "is_active": False,
        "external_api_config_id": None,
        "training_threshold": 100,
        "pending_training_samples": 88,
        "training_config": {
            "model_type": "embedding",
            "training_method": "sft",
            "base_model_path": "models/base",
        },
        "base_deployment_id": None,
        "base_deployment_replica_id": None,
    }
    persisted_target = {}
    migration_calls = []
    trigger_calls = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(parent),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target",
        lambda _target_id: dict(persisted_target) if persisted_target else None,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target_raw",
        lambda _target_id: dict(persisted_target) if persisted_target else None,
    )

    def migrate(**kwargs):
        migration_calls.append(kwargs)
        persisted_target.update(
            {
                "target_id": kwargs["target_id"],
                "task_id": kwargs["task_id"],
                "target_name": kwargs["target_name"],
                "base_deployment_id": kwargs["base_deployment_id"],
                "base_deployment_replica_id": kwargs[
                    "base_deployment_replica_id"
                ],
                "status": "idle",
            }
        )
        return dict(persisted_target)

    monkeypatch.setattr(
        external_sync_service,
        "migrate_legacy_training_target",
        migrate,
    )
    monkeypatch.setattr(
        level2_handler,
        "_trigger_training_for_target",
        lambda actual_task_id, target: trigger_calls.append(
            (actual_task_id, target["target_id"])
        )
        or True,
    )

    response = asyncio.run(
        sync_routes.trigger_target_training(
            task_id,
            target_id,
            {"user_id": "user-1", "username": "owner"},
        )
    )

    assert len(migration_calls) == 1
    assert migration_calls[0]["expected_user_id"] == "user-1"
    assert trigger_calls == [(task_id, target_id)]
    assert "Training triggered" in response["message"]


def test_atomic_patch_of_synthetic_legacy_target_transfers_pending_once():
    service, engine = _service()
    created = service.create_task(
        task_name="legacy edit",
        user_id="user-1",
        external_api_url="https://sync.example.com/data",
        training_threshold=100,
        training_config={
            "model_type": "embedding",
            "training_method": "sft",
            "base_model_path": "models/base",
        },
        is_active=False,
    )
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == created["task_id"]
            )
        ).one()
        task.pending_training_samples = 88
        task.total_training_samples = 88
        session.add(task)
        session.commit()
    target_id = sync_routes._legacy_training_target_id(created["task_id"])

    service.update_task(
        created["task_id"],
        expected_user_id="user-1",
        training_config=None,
        training_targets=[
            _target(target_id, "EMBEDDING (legacy)"),
        ],
    )

    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        assert task.pending_training_samples == 0
        assert target.pending_training_samples == 88
        assert target.total_training_samples == 88
