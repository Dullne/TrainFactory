from __future__ import annotations

import asyncio
import importlib
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, func, select

from train_factory.api.routes import sync_routes
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncBatchDB,
    ExternalSyncGenerationDB,
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.entities.external_api_config_entity import (
    ExternalApiConfigDB,
)
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.storage.services.external_sync_service import (
    ExternalSyncService,
    external_sync_service,
)
from train_factory.sync import sync_manager as sync_manager_module
from train_factory.sync import post_training_handler


CURRENT_USER = {"user_id": "user-1", "username": "owner"}
adapter_service_module = importlib.import_module(
    "train_factory.deployment.adapter_service"
)


def _sync_service() -> tuple[ExternalSyncService, object]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in (
        ExternalApiConfigDB.__table__,
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
    with Session(engine) as session:
        session.add_all(
            [
                ExternalApiConfigDB(
                    config_id=config_id,
                    config_name=config_id,
                    user_id="user-1",
                    api_url=f"https://{config_id}.example.com/api",
                    auth_config={},
                )
                for config_id in ("api-config-a", "api-config-b")
            ]
        )
        session.commit()
    service = ExternalSyncService()
    service.engine = engine
    return service, engine


def _file_sync_service(tmp_path) -> tuple[ExternalSyncService, object]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'sync-atomic.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    for table in (
        ExternalApiConfigDB.__table__,
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
    with Session(engine) as session:
        session.add_all(
            [
                ExternalApiConfigDB(
                    config_id=config_id,
                    config_name=config_id,
                    user_id="user-1",
                    api_url=f"https://{config_id}.example.com/api",
                    auth_config={},
                )
                for config_id in ("api-config-a", "api-config-b")
            ]
        )
        session.commit()
    service = ExternalSyncService()
    service.engine = engine
    return service, engine


def _add_binding_runtime(
    session: Session,
    *,
    deployment_id: str,
    replica_id: str,
    port: int,
) -> None:
    session.add_all(
        [
            DeploymentDB(
                deployment_id=deployment_id,
                model_id=f"model-{deployment_id}",
                deployment_name=deployment_id,
                xinference_endpoint=f"http://127.0.0.1:{port}",
                user_id="user-1",
            ),
            DeploymentReplicaDB(
                replica_id=replica_id,
                deployment_id=deployment_id,
                replica_index=0,
                container_name=f"container-{replica_id}",
                endpoint=f"http://127.0.0.1:{port}",
                port=port,
            ),
        ]
    )


def test_service_rolls_back_parent_and_all_targets_when_one_target_insert_fails():
    service, engine = _sync_service()
    duplicate_target_id = "11111111-1111-1111-1111-111111111111"

    with pytest.raises(IntegrityError):
        service.create_task(
            task_name="atomic",
            user_id="user-1",
            external_api_url="https://sync.example.com/data",
            is_active=False,
            training_targets=[
                {
                    "target_id": duplicate_target_id,
                    "target_name": "embedding",
                    "model_type": "embedding",
                },
                {
                    "target_id": duplicate_target_id,
                    "target_name": "rerank",
                    "model_type": "rerank",
                },
            ],
        )

    with Session(engine) as session:
        assert session.exec(select(func.count()).select_from(ExternalSyncTaskDB)).one() == 0
        assert (
            session.exec(
                select(func.count()).select_from(ExternalSyncTrainingTargetDB)
            ).one()
            == 0
        )


def test_service_rolls_back_first_api_config_claim_with_failed_task_insert():
    service, engine = _sync_service()
    duplicate_target_id = "11111111-1111-1111-1111-111111111111"
    with Session(engine) as session:
        session.add_all(
            [
                DeploymentDB(
                    deployment_id="deployment-atomic-claim",
                    model_id="model-a",
                    deployment_name="deployment-atomic-claim",
                    xinference_endpoint="http://127.0.0.1:11000",
                    user_id="user-1",
                    config={"preserved": True},
                ),
                DeploymentDB(
                    deployment_id="deployment-target-atomic-claim",
                    model_id="model-b",
                    deployment_name="deployment-target-atomic-claim",
                    xinference_endpoint="http://127.0.0.1:12000",
                    user_id="user-1",
                    config={"target_preserved": True},
                ),
            ]
        )
        session.commit()

    with pytest.raises(IntegrityError):
        service.create_task(
            task_name="atomic-api-claim",
            user_id="user-1",
            external_api_config_id="api-config-a",
            external_api_url="https://sync.example.com/data",
            base_deployment_id="deployment-atomic-claim",
            is_active=False,
            training_targets=[
                {
                    "target_id": duplicate_target_id,
                    "target_name": "embedding",
                    "model_type": "embedding",
                    "base_deployment_id": "deployment-target-atomic-claim",
                },
                {
                    "target_id": duplicate_target_id,
                    "target_name": "rerank",
                    "model_type": "rerank",
                    "base_deployment_id": "deployment-target-atomic-claim",
                },
            ],
        )

    with Session(engine) as session:
        deployment = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-atomic-claim"
            )
        ).one()
        assert deployment.external_api_config_id is None
        assert deployment.config == {"preserved": True}
        target_deployment = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id
                == "deployment-target-atomic-claim"
            )
        ).one()
        assert target_deployment.external_api_config_id is None
        assert target_deployment.config == {"target_preserved": True}
        assert session.exec(select(ExternalSyncTaskDB)).all() == []


def test_service_claims_task_and_target_deployments_in_one_transaction():
    service, engine = _sync_service()
    with Session(engine) as session:
        session.add_all(
            [
                DeploymentDB(
                    deployment_id="deployment-task-claim",
                    model_id="model-a",
                    deployment_name="deployment-task-claim",
                    xinference_endpoint="http://127.0.0.1:11000",
                    user_id="user-1",
                    config={"task_preserved": True},
                ),
                DeploymentDB(
                    deployment_id="deployment-target-claim",
                    model_id="model-b",
                    deployment_name="deployment-target-claim",
                    xinference_endpoint="http://127.0.0.1:12000",
                    user_id="user-1",
                    config={"target_preserved": True},
                ),
            ]
        )
        session.commit()

    created = service.create_task(
        task_name="atomic-multi-claim",
        user_id="user-1",
        external_api_config_id="api-config-a",
        external_api_url="https://sync.example.com/data",
        base_deployment_id="deployment-task-claim",
        training_targets=[
            {
                "target_name": "rerank",
                "model_type": "rerank",
                "base_deployment_id": "deployment-target-claim",
            }
        ],
        is_active=False,
    )

    assert len(created["training_targets"]) == 1
    with Session(engine) as session:
        deployments = {
            deployment.deployment_id: deployment
            for deployment in session.exec(select(DeploymentDB)).all()
        }
        assert deployments[
            "deployment-task-claim"
        ].external_api_config_id == "api-config-a"
        assert deployments["deployment-task-claim"].config == {
            "task_preserved": True,
            "external_api_config_id": "api-config-a",
        }
        assert deployments[
            "deployment-target-claim"
        ].external_api_config_id == "api-config-a"
        assert deployments["deployment-target-claim"].config == {
            "target_preserved": True,
            "external_api_config_id": "api-config-a",
        }
        assert len(session.exec(select(ExternalSyncTaskDB)).all()) == 1
        assert len(session.exec(select(ExternalSyncTrainingTargetDB)).all()) == 1


def test_service_rechecks_deployment_owner_and_api_config_under_lock():
    service, engine = _sync_service()
    with Session(engine) as session:
        session.add_all(
            [
                DeploymentDB(
                    deployment_id="deployment-other-owner",
                    model_id="model-a",
                    deployment_name="deployment-other-owner",
                    xinference_endpoint="http://127.0.0.1:11000",
                    user_id="user-2",
                ),
                DeploymentDB(
                    deployment_id="deployment-other-api",
                    model_id="model-b",
                    deployment_name="deployment-other-api",
                    xinference_endpoint="http://127.0.0.1:12000",
                    user_id="user-1",
                    external_api_config_id="api-config-b",
                    config={"external_api_config_id": "api-config-b"},
                ),
            ]
        )
        session.commit()

    with pytest.raises(ValueError, match="^Runtime dependency is unavailable$"):
        service.create_task(
            task_name="owner-race",
            user_id="user-1",
            external_api_config_id="api-config-a",
            external_api_url="https://sync.example.com/data",
            base_deployment_id="deployment-other-owner",
            is_active=False,
        )
    with pytest.raises(ValueError, match="api_config mismatch"):
        service.create_task(
            task_name="api-config-race",
            user_id="user-1",
            external_api_config_id="api-config-a",
            external_api_url="https://sync.example.com/data",
            base_deployment_id="deployment-other-api",
            is_active=False,
        )

    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTaskDB)).all() == []


def test_concurrent_first_api_config_claim_has_one_atomic_winner(tmp_path):
    service, engine = _file_sync_service(tmp_path)
    with Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="deployment-concurrent-claim",
                model_id="model-a",
                deployment_name="deployment-concurrent-claim",
                xinference_endpoint="http://127.0.0.1:11000",
                user_id="user-1",
                config={"preserved": True},
            )
        )
        session.commit()

    start = threading.Barrier(2)

    def claim(api_config_id):
        start.wait(timeout=5)
        try:
            task = service.create_task(
                task_name=f"claim-{api_config_id}",
                user_id="user-1",
                external_api_config_id=api_config_id,
                external_api_url="https://sync.example.com/data",
                base_deployment_id="deployment-concurrent-claim",
                is_active=False,
            )
            return ("created", api_config_id, task["task_id"])
        except ValueError as exc:
            return ("conflict", api_config_id, str(exc))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, ("api-config-a", "api-config-b")))

    created = [result for result in results if result[0] == "created"]
    conflicts = [result for result in results if result[0] == "conflict"]
    assert len(created) == 1
    assert len(conflicts) == 1
    assert "api_config mismatch" in conflicts[0][2]

    with Session(engine) as session:
        deployment = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-concurrent-claim"
            )
        ).one()
        tasks = session.exec(select(ExternalSyncTaskDB)).all()
        assert deployment.external_api_config_id == created[0][1]
        assert deployment.config == {
            "preserved": True,
            "external_api_config_id": created[0][1],
        }
        assert len(tasks) == 1
        assert tasks[0].external_api_config_id == created[0][1]


def test_update_task_rechecks_every_target_deployment_api_config():
    service, engine = _sync_service()
    with Session(engine) as session:
        task = ExternalSyncTaskDB(
            task_id="sync-source-update",
            task_name="source-update",
            user_id="user-1",
            external_api_config_id="api-config-a",
            external_api_url="https://sync.example.com/data",
            is_active=False,
        )
        session.add_all(
            [
                DeploymentDB(
                    deployment_id="deployment-target-source-update",
                    model_id="model-a",
                    deployment_name="deployment-target-source-update",
                    xinference_endpoint="http://127.0.0.1:11000",
                    user_id="user-1",
                    external_api_config_id="api-config-a",
                    config={"external_api_config_id": "api-config-a"},
                ),
                task,
                ExternalSyncTrainingTargetDB(
                    target_id="target-source-update",
                    task_id=task.task_id,
                    target_name="rerank",
                    model_type="rerank",
                    base_deployment_id="deployment-target-source-update",
                ),
            ]
        )
        session.commit()

    with pytest.raises(ValueError, match="api_config mismatch"):
        service.update_task(
            "sync-source-update",
            external_api_config_id="api-config-b",
        )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == "sync-source-update"
            )
        ).one()
        deployment = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id
                == "deployment-target-source-update"
            )
        ).one()
        assert task.external_api_config_id == "api-config-a"
        assert deployment.external_api_config_id == "api-config-a"
        assert deployment.config == {
            "external_api_config_id": "api-config-a"
        }


def test_update_task_claims_unbound_target_deployments_atomically():
    service, engine = _sync_service()
    with Session(engine) as session:
        task = ExternalSyncTaskDB(
            task_id="sync-target-claim",
            task_name="target-claim",
            user_id="user-1",
            external_api_url="https://sync.example.com/data",
            is_active=False,
        )
        session.add_all(
            [
                DeploymentDB(
                    deployment_id="deployment-target-claim-on-update",
                    model_id="model-a",
                    deployment_name="deployment-target-claim-on-update",
                    xinference_endpoint="http://127.0.0.1:11000",
                    user_id="user-1",
                    config={"preserved": True},
                ),
                task,
                ExternalSyncTrainingTargetDB(
                    target_id="target-claim-on-update",
                    task_id=task.task_id,
                    target_name="rerank",
                    model_type="rerank",
                    base_deployment_id="deployment-target-claim-on-update",
                ),
            ]
        )
        session.commit()

    updated = service.update_task(
        "sync-target-claim",
        external_api_config_id="api-config-a",
    )

    assert updated["external_api_config_id"] == "api-config-a"
    with Session(engine) as session:
        deployment = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id
                == "deployment-target-claim-on-update"
            )
        ).one()
        assert deployment.external_api_config_id == "api-config-a"
        assert deployment.config == {
            "preserved": True,
            "external_api_config_id": "api-config-a",
        }


def test_update_task_rejects_binding_drift_while_training_claim_is_active():
    service, engine = _sync_service()
    with Session(engine) as session:
        _add_binding_runtime(
            session,
            deployment_id="deployment-binding-d1",
            replica_id="replica-binding-r1",
            port=11001,
        )
        _add_binding_runtime(
            session,
            deployment_id="deployment-binding-d2",
            replica_id="replica-binding-r2",
            port=12001,
        )
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-binding-freeze",
                task_name="binding-freeze",
                user_id="user-1",
                base_deployment_id="deployment-binding-d1",
                base_deployment_replica_id="replica-binding-r1",
                training_config={},
                training_threshold=1,
                pending_training_samples=10,
                is_active=False,
            )
        )
        session.commit()

    claim = service.create_training_with_claim(
        task_id="sync-binding-freeze",
        training_task_id="training-binding-freeze",
        user_id="user-1",
        input_dataset_ids=["dataset-binding-freeze"],
        total_samples=10,
        require_threshold=True,
    )
    assert claim is not None
    assert claim["target_config_snapshot"]["base_deployment_id"] == (
        "deployment-binding-d1"
    )
    assert claim["target_config_snapshot"]["base_deployment_replica_id"] == (
        "replica-binding-r1"
    )

    with pytest.raises(ValueError, match="binding"):
        service.update_task(
            "sync-binding-freeze",
            base_deployment_id="deployment-binding-d2",
            base_deployment_replica_id="replica-binding-r2",
        )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == "sync-binding-freeze"
            )
        ).one()
        assert task.base_deployment_id == "deployment-binding-d1"
        assert task.base_deployment_replica_id == "replica-binding-r1"


def test_update_target_rejects_binding_drift_while_training_claim_is_active():
    service, engine = _sync_service()
    with Session(engine) as session:
        _add_binding_runtime(
            session,
            deployment_id="deployment-target-d1",
            replica_id="replica-target-r1",
            port=13001,
        )
        _add_binding_runtime(
            session,
            deployment_id="deployment-target-d2",
            replica_id="replica-target-r2",
            port=14001,
        )
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-target-binding-freeze",
                    task_name="target-binding-freeze",
                    user_id="user-1",
                    is_active=False,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-binding-freeze",
                    task_id="sync-target-binding-freeze",
                    target_name="embedding",
                    base_deployment_id="deployment-target-d1",
                    base_deployment_replica_id="replica-target-r1",
                    training_threshold=1,
                    pending_training_samples=10,
                ),
            ]
        )
        session.commit()

    claim = service.create_training_with_claim(
        task_id="sync-target-binding-freeze",
        target_id="target-binding-freeze",
        training_task_id="training-target-binding-freeze",
        user_id="user-1",
        input_dataset_ids=["dataset-target-binding-freeze"],
        total_samples=10,
        require_threshold=True,
    )
    assert claim is not None

    with pytest.raises(ValueError, match="binding"):
        service.update_training_target(
            "target-binding-freeze",
            task_id="sync-target-binding-freeze",
            base_deployment_id="deployment-target-d2",
            base_deployment_replica_id="replica-target-r2",
        )

    target = service.get_training_target_raw("target-binding-freeze")
    assert target["base_deployment_id"] == "deployment-target-d1"
    assert target["base_deployment_replica_id"] == "replica-target-r1"


@pytest.mark.parametrize("target_overrides_task_binding", [True, False])
def test_training_claim_snapshots_effective_target_replica_binding(
    target_overrides_task_binding,
):
    service, engine = _sync_service()
    with Session(engine) as session:
        _add_binding_runtime(
            session,
            deployment_id="deployment-snapshot-d1",
            replica_id="replica-snapshot-r1",
            port=15001,
        )
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-binding-snapshot",
                    task_name="binding-snapshot",
                    user_id="user-1",
                    base_deployment_id="deployment-snapshot-d1",
                    base_deployment_replica_id="replica-snapshot-r1",
                    is_active=False,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-binding-snapshot",
                    task_id="sync-binding-snapshot",
                    target_name="embedding",
                    base_deployment_id=(
                        "deployment-snapshot-d1"
                        if target_overrides_task_binding
                        else None
                    ),
                    base_deployment_replica_id=(
                        "replica-snapshot-r1"
                        if target_overrides_task_binding
                        else None
                    ),
                    training_threshold=1,
                    pending_training_samples=10,
                ),
            ]
        )
        session.commit()

    claim = service.create_training_with_claim(
        task_id="sync-binding-snapshot",
        target_id="target-binding-snapshot",
        training_task_id="training-binding-snapshot",
        user_id="user-1",
        input_dataset_ids=["dataset-binding-snapshot"],
        total_samples=10,
        require_threshold=True,
    )

    assert claim["target_config_snapshot"]["base_deployment_id"] == (
        "deployment-snapshot-d1"
    )
    assert claim["target_config_snapshot"]["base_deployment_replica_id"] == (
        "replica-snapshot-r1"
    )


def test_adapter_loaded_training_freezes_binding_until_adapter_is_unloaded():
    service, engine = _sync_service()
    with Session(engine) as session:
        _add_binding_runtime(
            session,
            deployment_id="deployment-loaded-d1",
            replica_id="replica-loaded-r1",
            port=16001,
        )
        _add_binding_runtime(
            session,
            deployment_id="deployment-loaded-d2",
            replica_id="replica-loaded-r2",
            port=17001,
        )
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-loaded-binding",
                    task_name="loaded-binding",
                    user_id="user-1",
                    base_deployment_id="deployment-loaded-d1",
                    base_deployment_replica_id="replica-loaded-r1",
                    is_active=False,
                ),
                ExternalSyncTrainingDB(
                    task_id="sync-loaded-binding",
                    training_task_id="training-loaded-binding",
                    user_id="user-1",
                    status="adapter_loaded",
                    target_config_snapshot={
                        "base_deployment_id": "deployment-loaded-d1",
                        "base_deployment_replica_id": "replica-loaded-r1",
                    },
                ),
            ]
        )
        session.commit()

    with pytest.raises(ValueError, match="binding"):
        service.update_task(
            "sync-loaded-binding",
            base_deployment_id="deployment-loaded-d2",
            base_deployment_replica_id="replica-loaded-r2",
        )

    assert (
        service.mark_all_trainings_adapter_unloaded("sync-loaded-binding")
        == 1
    )
    updated = service.update_task(
        "sync-loaded-binding",
        base_deployment_id="deployment-loaded-d2",
        base_deployment_replica_id="replica-loaded-r2",
    )

    assert updated["base_deployment_id"] == "deployment-loaded-d2"
    assert updated["base_deployment_replica_id"] == "replica-loaded-r2"


def test_task_patch_cannot_remove_target_that_still_owns_loaded_adapter():
    service, engine = _sync_service()
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-loaded-target-owner",
                    task_name="loaded-target-owner",
                    user_id="user-1",
                    is_active=False,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-loaded-owner",
                    task_id="sync-loaded-target-owner",
                    target_name="embedding",
                    current_adapter_name="adapter-loaded-owner",
                    current_adapter_id="adapter-loaded-owner-id",
                    current_training_id="training-loaded-owner",
                ),
                ExternalSyncTrainingDB(
                    task_id="sync-loaded-target-owner",
                    training_task_id="training-loaded-owner",
                    target_id="target-loaded-owner",
                    user_id="user-1",
                    loaded_adapter_name="adapter-loaded-owner",
                    loaded_adapter_id="adapter-loaded-owner-id",
                    status="adapter_loaded",
                ),
            ]
        )
        session.commit()

    with pytest.raises(ValueError, match="binding|adapter"):
        service.update_task(
            "sync-loaded-target-owner",
            training_targets=[],
        )

    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id
                == "target-loaded-owner"
            )
        ).one()
        assert target.current_adapter_id == "adapter-loaded-owner-id"


def test_update_task_rolls_back_earlier_target_claim_on_later_conflict():
    service, engine = _sync_service()
    with Session(engine) as session:
        task = ExternalSyncTaskDB(
            task_id="sync-target-claim-rollback",
            task_name="target-claim-rollback",
            user_id="user-1",
            external_api_url="https://sync.example.com/data",
            is_active=False,
        )
        session.add_all(
            [
                DeploymentDB(
                    deployment_id="deployment-a-unbound",
                    model_id="model-a",
                    deployment_name="deployment-a-unbound",
                    xinference_endpoint="http://127.0.0.1:11000",
                    user_id="user-1",
                    config={"preserved": True},
                ),
                DeploymentDB(
                    deployment_id="deployment-b-conflict",
                    model_id="model-b",
                    deployment_name="deployment-b-conflict",
                    xinference_endpoint="http://127.0.0.1:12000",
                    user_id="user-1",
                    external_api_config_id="api-config-b",
                    config={"external_api_config_id": "api-config-b"},
                ),
                task,
                ExternalSyncTrainingTargetDB(
                    target_id="target-a-unbound",
                    task_id=task.task_id,
                    target_name="embedding",
                    model_type="embedding",
                    base_deployment_id="deployment-a-unbound",
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-b-conflict",
                    task_id=task.task_id,
                    target_name="rerank",
                    model_type="rerank",
                    base_deployment_id="deployment-b-conflict",
                ),
            ]
        )
        session.commit()

    with pytest.raises(ValueError, match="api_config mismatch"):
        service.update_task(
            "sync-target-claim-rollback",
            external_api_config_id="api-config-a",
        )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == "sync-target-claim-rollback"
            )
        ).one()
        first_deployment = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-a-unbound"
            )
        ).one()
        assert task.external_api_config_id is None
        assert first_deployment.external_api_config_id is None
        assert first_deployment.config == {"preserved": True}


def test_service_rejects_orphan_replica_binding_without_any_mutation():
    service, engine = _sync_service()

    with pytest.raises(ValueError, match="requires base_deployment_id"):
        service.create_task(
            task_name="orphan-binding",
            user_id="user-1",
            external_api_url="https://sync.example.com/data",
            base_deployment_replica_id="replica-without-parent",
            is_active=False,
        )

    with Session(engine) as session:
        assert session.exec(select(func.count()).select_from(ExternalSyncTaskDB)).one() == 0
        assert (
            session.exec(
                select(func.count()).select_from(ExternalSyncTrainingTargetDB)
            ).one()
            == 0
        )


@pytest.mark.parametrize("binding_scope", ["task", "target"])
def test_service_rejects_replica_bound_to_another_deployment_without_mutation(
    binding_scope,
):
    service, engine = _sync_service()
    with Session(engine) as session:
        session.add_all(
            [
                DeploymentDB(
                    deployment_id="deployment-a",
                    model_id="model-a",
                    deployment_name="deployment-a",
                    xinference_endpoint="http://127.0.0.1:11000",
                    user_id="user-1",
                ),
                DeploymentDB(
                    deployment_id="deployment-b",
                    model_id="model-b",
                    deployment_name="deployment-b",
                    xinference_endpoint="http://127.0.0.1:12000",
                    user_id="user-1",
                ),
            ]
        )
        session.commit()
        session.add(
            DeploymentReplicaDB(
                replica_id="deployment-a-r1",
                deployment_id="deployment-a",
                replica_index=1,
                container_name="deployment-a-r1",
                endpoint="http://127.0.0.1:11001",
                port=11001,
                gpu_ids=[0],
            )
        )
        session.commit()

    task_binding = {}
    target_bindings = []
    if binding_scope == "task":
        task_binding = {
            "base_deployment_id": "deployment-b",
            "base_deployment_replica_id": "deployment-a-r1",
        }
    else:
        target_bindings = [
            {
                "target_name": "rerank",
                "base_deployment_id": "deployment-b",
                "base_deployment_replica_id": "deployment-a-r1",
            }
        ]

    with pytest.raises(ValueError, match="does not belong to deployment"):
        service.create_task(
            task_name="cross-deployment-binding",
            user_id="user-1",
            external_api_url="https://sync.example.com/data",
            is_active=False,
            training_targets=target_bindings,
            **task_binding,
        )

    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTaskDB)).all() == []
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []


def test_target_deployment_override_never_inherits_task_replica(monkeypatch):
    config = {
        "task_id": "sync-cross-binding",
        "user_id": "user-1",
        "external_api_config_id": None,
        "base_deployment_id": "deployment-a",
        "base_deployment_replica_id": "deployment-a-r1",
        "training_config": {},
        "total_trainings": 0,
    }
    training = {
        "task_id": config["task_id"],
        "target_id": "target-b",
        "training_round": 1,
    }
    target = {
        "target_id": "target-b",
        "task_id": config["task_id"],
        "model_type": "rerank",
        "base_deployment_id": "deployment-b",
        "base_deployment_replica_id": None,
    }
    adapter_calls = []

    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: dict(config),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _training_id: dict(training),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target_raw",
        lambda _target_id: dict(target),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda _task_id, **_kwargs: [dict(target)],
    )
    monkeypatch.setattr(external_sync_service, "update_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        external_sync_service,
        "update_training_status",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        external_sync_service,
        "mark_all_trainings_adapter_unloaded",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_training_target",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        post_training_handler,
        "_find_compatible_deployment",
        lambda *_args, **_kwargs: None,
    )

    from train_factory.deployment.deployment_service import deployment_service

    monkeypatch.setattr(
        deployment_service,
        "get_deployment",
        lambda deployment_id: {
            "deployment_id": deployment_id,
            "user_id": "user-1",
            "external_api_config_id": None,
            "config": {},
        },
    )

    class RecordingAdapterService:
        def sync_loaded_adapters(self, deployment_id, **kwargs):
            adapter_calls.append(("sync", deployment_id, kwargs))

        def list_loaded_adapters(self, deployment_id, **kwargs):
            adapter_calls.append(("list", deployment_id, kwargs))
            return []

        def load_adapter(self, **kwargs):
            adapter_calls.append(("load", kwargs["deployment_id"], kwargs))
            return {"adapter_id": "adapter-b"}

    monkeypatch.setattr(
        adapter_service_module,
        "AdapterService",
        RecordingAdapterService,
    )

    post_training_handler.load_adapter_for_training(
        config["task_id"],
        "training-b",
        "/models/adapter-b",
        target_id=target["target_id"],
    )

    assert [call[1] for call in adapter_calls] == [
        "deployment-b",
        "deployment-b",
    ]
    assert all(
        call[2].get("deployment_replica_id") is None
        for call in adapter_calls
    )
    assert all(
        call[2].get("deployment_replica_id") != "deployment-a-r1"
        for call in adapter_calls
    )


def test_create_route_validates_every_target_before_persisting(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        sync_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_task",
        lambda **kwargs: persisted.append(kwargs) or {"task_id": "unexpected"},
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.create_sync_task(
                sync_routes.SyncTaskCreateRequest(
                    task_name="invalid-target",
                    external_api_url="https://sync.example.com/data",
                    is_active=True,
                    training_targets=[
                        sync_routes.TrainingTargetCreateRequest(
                            target_name="valid",
                        ),
                        sync_routes.TrainingTargetCreateRequest(
                            target_name="orphan-replica",
                            base_deployment_replica_id=(
                                "22222222-2222-2222-2222-222222222222"
                            ),
                        ),
                    ],
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 400
    assert "deployment" in exc_info.value.detail.lower()
    assert persisted == []


def test_create_route_passes_validated_targets_to_single_service_call(monkeypatch):
    calls = []
    started = []
    monkeypatch.setattr(
        sync_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )

    def create_task(**kwargs):
        calls.append(kwargs)
        return {
            "task_id": "sync-atomic",
            "is_active": kwargs["is_active"],
            "training_targets": kwargs["training_targets"],
        }

    monkeypatch.setattr(external_sync_service, "create_task", create_task)
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "start_worker",
        lambda task_id: started.append(task_id),
    )

    response = asyncio.run(
        sync_routes.create_sync_task(
            sync_routes.SyncTaskCreateRequest(
                task_name="atomic",
                external_api_url="https://sync.example.com/data",
                is_active=True,
                training_targets=[
                    sync_routes.TrainingTargetCreateRequest(
                        target_name="embedding",
                        model_type="embedding",
                    ),
                    sync_routes.TrainingTargetCreateRequest(
                        target_name="rerank",
                        model_type="rerank",
                    ),
                ],
            ),
            CURRENT_USER,
        )
    )

    assert len(calls) == 1
    assert [item["target_name"] for item in calls[0]["training_targets"]] == [
        "embedding",
        "rerank",
    ]
    assert response["task"]["task_id"] == "sync-atomic"
    assert started == ["sync-atomic"]


def test_create_route_maps_binding_race_to_conflict_without_starting_worker(
    monkeypatch,
):
    started = []
    monkeypatch.setattr(
        sync_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )
    monkeypatch.setattr(
        sync_routes,
        "_validate_base_deployment_reference",
        lambda *_args, **_kwargs: {"deployment_id": "deployment-deleting"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_task",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("Deployment deletion is in progress")
        ),
    )
    monkeypatch.setattr(
        sync_manager_module.sync_manager,
        "start_worker",
        lambda task_id: started.append(task_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.create_sync_task(
                sync_routes.SyncTaskCreateRequest(
                    task_name="binding-race",
                    external_api_url="https://sync.example.com/data",
                    base_deployment_id="deployment-deleting",
                    is_active=True,
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Deployment deletion is in progress"
    assert started == []


def test_sync_task_update_preserves_explicit_null_replica_unbinding(monkeypatch):
    stored = {
        "task_id": "sync-bound",
        "task_name": "bound",
        "user_id": "user-1",
        "status": "idle",
        "is_active": False,
        "external_api_config_id": None,
        "base_deployment_id": "deployment-1",
        "base_deployment_replica_id": "replica-1",
    }
    updates = []
    monkeypatch.setattr(external_sync_service, "get_task", lambda _task_id: dict(stored))

    def update_task(_task_id, **kwargs):
        updates.append(kwargs)
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
            "sync-bound",
            sync_routes.SyncTaskUpdateRequest(
                base_deployment_id=None,
                base_deployment_replica_id=None,
            ),
            CURRENT_USER,
        )
    )

    assert updates == [
        {
            "expected_user_id": "user-1",
            "base_deployment_id": None,
            "base_deployment_replica_id": None,
        }
    ]
    assert response["task"]["base_deployment_id"] is None
    assert response["task"]["base_deployment_replica_id"] is None


def test_sync_task_update_does_not_write_null_to_non_binding_columns(monkeypatch):
    stored = {
        "task_id": "sync-bound",
        "task_name": "bound",
        "user_id": "user-1",
        "status": "idle",
        "is_active": False,
    }
    updates = []
    monkeypatch.setattr(external_sync_service, "get_task", lambda _task_id: dict(stored))
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda _task_id, **kwargs: updates.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.update_sync_task(
                "sync-bound",
                sync_routes.SyncTaskUpdateRequest(task_name=None),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 400
    assert updates == []


def test_training_target_update_preserves_explicit_null_replica_unbinding(monkeypatch):
    parent = {
        "task_id": "sync-bound",
        "user_id": "user-1",
        "status": "idle",
        "is_active": False,
        "external_api_config_id": None,
    }
    target = {
        "target_id": "target-bound",
        "task_id": "sync-bound",
        "target_name": "bound",
        "status": "idle",
        "base_deployment_id": "deployment-1",
        "base_deployment_replica_id": "replica-1",
    }
    updates = []
    monkeypatch.setattr(external_sync_service, "get_task", lambda _task_id: dict(parent))
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target",
        lambda _target_id: dict(target),
    )

    def update_target(_target_id, **kwargs):
        updates.append(kwargs)
        return {**target, **kwargs}

    monkeypatch.setattr(
        external_sync_service,
        "update_training_target",
        update_target,
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
        sync_routes.update_training_target(
            "sync-bound",
            "target-bound",
            sync_routes.TrainingTargetUpdateRequest(
                base_deployment_id=None,
                base_deployment_replica_id=None,
            ),
            CURRENT_USER,
        )
    )

    assert updates == [
        {
            "task_id": "sync-bound",
            "expected_user_id": "user-1",
            "base_deployment_id": None,
            "base_deployment_replica_id": None,
        }
    ]
    assert response["target"]["base_deployment_id"] is None
    assert response["target"]["base_deployment_replica_id"] is None


def test_training_target_patch_allows_non_binding_rename_while_training(
    monkeypatch,
):
    parent = {
        "task_id": "sync-active-rename",
        "user_id": "user-1",
        "status": "training",
        "is_active": False,
        "external_api_config_id": None,
    }
    target = {
        "target_id": "target-active-rename",
        "task_id": "sync-active-rename",
        "target_name": "before",
        "status": "training",
        "base_deployment_id": "deployment-1",
        "base_deployment_replica_id": "replica-1",
    }
    updates = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(parent),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target",
        lambda _target_id: dict(target),
    )

    def update_target(_target_id, **kwargs):
        updates.append(kwargs)
        return {**target, **kwargs}

    monkeypatch.setattr(
        external_sync_service,
        "update_training_target",
        update_target,
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
        sync_routes.update_training_target(
            "sync-active-rename",
            "target-active-rename",
            sync_routes.TrainingTargetUpdateRequest(target_name="after"),
            CURRENT_USER,
        )
    )

    assert updates == [
        {
            "task_id": "sync-active-rename",
            "expected_user_id": "user-1",
            "target_name": "after",
        }
    ]
    assert response["target"]["target_name"] == "after"


def test_replica_reference_without_parent_deployment_is_rejected():
    with pytest.raises(HTTPException) as exc_info:
        sync_routes._validate_base_deployment_reference(
            None,
            CURRENT_USER,
            base_deployment_replica_id="replica-without-parent",
        )

    assert exc_info.value.status_code == 400
    assert "deployment" in exc_info.value.detail.lower()
