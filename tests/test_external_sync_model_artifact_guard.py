from __future__ import annotations

import importlib

import pytest
import sqlalchemy as sa
from sqlmodel import Session, SQLModel, select

from train_factory.enums.sync_status import SyncStatus
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.storage.services.external_sync_service import ExternalSyncService


sync_module = importlib.import_module(
    "train_factory.storage.services.external_sync_service"
)
runtime_module = importlib.import_module(
    "train_factory.storage.services.runtime_dependency_service"
)


@pytest.fixture
def sync_model_guard(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'sync-model-guard.db'}")
    SQLModel.metadata.create_all(engine)
    model_path = tmp_path / "models" / "managed-model"
    model_path.mkdir(parents=True)
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.add(
            ModelRegistryDB(
                model_id="managed-model",
                model_name="managed-model",
                model_type="embedding",
                model_path=str(model_path),
                status="available",
                user_id="user-1",
            )
        )
        session.commit()
    service = ExternalSyncService()
    service.engine = engine
    return engine, service, str(model_path)


def _mark_model_deleting(engine) -> None:
    with Session(engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == "managed-model"
            )
        ).one()
        model.status = "deleting"
        session.add(model)
        session.commit()


@pytest.mark.parametrize("reference_kind", ["target", "legacy-parent"])
def test_active_sync_create_rejects_path_under_deleting_model(
    sync_model_guard,
    reference_kind: str,
) -> None:
    engine, service, model_path = sync_model_guard
    _mark_model_deleting(engine)
    kwargs = {}
    if reference_kind == "target":
        kwargs["training_targets"] = [
            {
                "target_name": "managed-target",
                "base_model_path": f"{model_path}/child",
            }
        ]
    else:
        kwargs["training_config"] = {
            "base_model_path": f"{model_path}/legacy-child"
        }

    with pytest.raises(ValueError, match="being deleted"):
        service.create_task(
            task_name=f"create-{reference_kind}",
            user_id="user-1",
            is_active=True,
            **kwargs,
        )

    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTaskDB)).all() == []
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []


@pytest.mark.parametrize("reference_kind", ["target", "legacy-parent"])
def test_inactive_sync_cannot_activate_managed_path_while_model_is_deleting(
    sync_model_guard,
    reference_kind: str,
) -> None:
    engine, service, model_path = sync_model_guard
    with Session(engine) as session:
        task = ExternalSyncTaskDB(
            task_id=f"activate-{reference_kind}",
            task_name=f"activate-{reference_kind}",
            user_id="user-1",
            is_active=False,
            training_config=(
                {"base_model_path": f"{model_path}/legacy-child"}
                if reference_kind == "legacy-parent"
                else None
            ),
        )
        session.add(task)
        if reference_kind == "target":
            session.add(
                ExternalSyncTrainingTargetDB(
                    target_id="activate-target",
                    task_id=task.task_id,
                    target_name="activate-target",
                    base_model_path=f"{model_path}/target-child",
                    is_active=True,
                )
            )
        session.commit()
    _mark_model_deleting(engine)

    with pytest.raises(ValueError, match="being deleted"):
        service.update_task(
            f"activate-{reference_kind}",
            is_active=True,
            expected_user_id="user-1",
        )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == f"activate-{reference_kind}"
            )
        ).one()
        assert task.is_active is False


@pytest.mark.parametrize("operation", ["create-target", "update-target", "replace"])
def test_active_sync_target_path_writer_rejects_deleting_model(
    sync_model_guard,
    operation: str,
) -> None:
    engine, service, model_path = sync_model_guard
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="target-writer",
                task_name="target-writer",
                user_id="user-1",
                is_active=True,
            )
        )
        if operation != "create-target":
            session.add(
                ExternalSyncTrainingTargetDB(
                    target_id="existing-target",
                    task_id="target-writer",
                    target_name="existing-target",
                    base_model_path="/external/original-model",
                    is_active=True,
                )
            )
        session.commit()
    _mark_model_deleting(engine)

    with pytest.raises(ValueError, match="being deleted"):
        if operation == "create-target":
            service.create_training_target(
                "target-writer",
                "new-target",
                target_id="new-target",
                base_model_path=f"{model_path}/new-target",
                expected_user_id="user-1",
            )
        elif operation == "update-target":
            service.update_training_target(
                "existing-target",
                task_id="target-writer",
                base_model_path=f"{model_path}/updated-target",
                expected_user_id="user-1",
            )
        else:
            service.update_task(
                "target-writer",
                training_targets=[
                    {
                        "target_id": "existing-target",
                        "target_name": "existing-target",
                        "base_model_path": f"{model_path}/replacement-target",
                    }
                ],
                expected_user_id="user-1",
            )

    with Session(engine) as session:
        targets = session.exec(
            select(ExternalSyncTrainingTargetDB).order_by(
                ExternalSyncTrainingTargetDB.target_id
            )
        ).all()
        if operation == "create-target":
            assert targets == []
        else:
            assert len(targets) == 1
            assert targets[0].base_model_path == "/external/original-model"


def test_cancel_deletion_cannot_reactivate_managed_path_while_model_is_deleting(
    sync_model_guard,
) -> None:
    engine, service, model_path = sync_model_guard
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="cancel-delete",
                task_name="cancel-delete",
                user_id="user-1",
                status=SyncStatus.DELETING,
                is_active=False,
                training_config={
                    "base_model_path": f"{model_path}/legacy-child"
                },
            )
        )
        session.commit()
    _mark_model_deleting(engine)

    with pytest.raises(ValueError, match="being deleted"):
        service.cancel_task_deletion(
            "cancel-delete",
            status=SyncStatus.IDLE,
            is_active=True,
            expected_user_id="user-1",
            expected_deleting_status=SyncStatus.DELETING,
        )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == "cancel-delete"
            )
        ).one()
        assert task.status == SyncStatus.DELETING
        assert task.is_active is False


def test_sync_path_writer_detects_target_signature_drift(
    monkeypatch: pytest.MonkeyPatch,
    sync_model_guard,
) -> None:
    engine, service, model_path = sync_model_guard
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="signature-drift",
                task_name="signature-drift",
                user_id="user-1",
                is_active=True,
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="signature-target",
                task_id="signature-drift",
                target_name="signature-target",
                base_model_path="/external/original-model",
                is_active=True,
            )
        )
        session.commit()

    original_lock = sync_module.lock_runtime_dependencies

    def drift_after_dependency_lock(
        session,
        references,
        *,
        expected_user_id=None,
    ):
        locked = original_lock(
            session,
            references,
            expected_user_id=expected_user_id,
        )
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == "signature-target"
            )
        ).one()
        target.base_model_path = "/external/concurrent-drift"
        session.add(target)
        session.flush()
        return locked

    monkeypatch.setattr(
        sync_module,
        "lock_runtime_dependencies",
        drift_after_dependency_lock,
    )

    with pytest.raises(
        runtime_module.RuntimeDependencyChangedError,
        match="target",
    ):
        service.update_training_target(
            "signature-target",
            task_id="signature-drift",
            base_model_path=f"{model_path}/prospective-model",
            expected_user_id="user-1",
        )

    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == "signature-target"
            )
        ).one()
        assert target.base_model_path == "/external/original-model"


def test_nonbinding_sync_metadata_update_does_not_take_model_membership_gate(
    monkeypatch: pytest.MonkeyPatch,
    sync_model_guard,
) -> None:
    engine, service, model_path = sync_model_guard
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="metadata-only",
                task_name="metadata-only",
                user_id="user-1",
                is_active=True,
                training_config={
                    "base_model_path": f"{model_path}/legacy-child"
                },
            )
        )
        session.commit()
    _mark_model_deleting(engine)
    monkeypatch.setattr(
        sync_module,
        "lock_model_artifact_membership",
        lambda _session: pytest.fail(
            "metadata-only update must not lock model membership"
        ),
        raising=False,
    )

    updated = service.update_task(
        "metadata-only",
        task_name="renamed-only",
        expected_user_id="user-1",
    )

    assert updated is not None
    assert updated["task_name"] == "renamed-only"


def test_sync_path_writer_locks_api_then_gate_then_dependencies_then_task(
    monkeypatch: pytest.MonkeyPatch,
    sync_model_guard,
) -> None:
    engine, service, model_path = sync_model_guard
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="ordered-path-writer",
                task_name="ordered-path-writer",
                user_id="user-1",
                is_active=True,
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="ordered-path-target",
                task_id="ordered-path-writer",
                target_name="ordered-path-target",
                base_model_path="/external/original-model",
                is_active=True,
            )
        )
        session.commit()

    events: list[str] = []
    original_gate = sync_module.lock_model_artifact_membership
    original_api = sync_module._lock_external_api_configs
    original_dependencies = sync_module.lock_runtime_dependencies
    original_task = sync_module.lock_runtime_task_after_dependencies

    def record_gate(session):
        events.append("gate")
        return original_gate(session)

    def record_api(session, config_ids, *, expected_user_id):
        events.append("api")
        return original_api(
            session,
            config_ids,
            expected_user_id=expected_user_id,
        )

    def record_dependencies(session, references, *, expected_user_id=None):
        events.append("dependencies")
        return original_dependencies(
            session,
            references,
            expected_user_id=expected_user_id,
        )

    def record_task(session, **kwargs):
        events.append("task")
        return original_task(session, **kwargs)

    monkeypatch.setattr(
        sync_module,
        "lock_model_artifact_membership",
        record_gate,
    )
    monkeypatch.setattr(
        sync_module,
        "_lock_external_api_configs",
        record_api,
    )
    monkeypatch.setattr(
        sync_module,
        "lock_runtime_dependencies",
        record_dependencies,
    )
    monkeypatch.setattr(
        sync_module,
        "lock_runtime_task_after_dependencies",
        record_task,
    )

    service.update_training_target(
        "ordered-path-target",
        task_id="ordered-path-writer",
        base_model_path=f"{model_path}/prospective-model",
        expected_user_id="user-1",
    )

    assert events[:4] == ["api", "gate", "dependencies", "task"]


def test_sync_path_create_locks_api_then_gate_then_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    sync_model_guard,
) -> None:
    _engine, service, model_path = sync_model_guard
    events: list[str] = []
    original_api = sync_module._lock_external_api_configs
    original_gate = sync_module.lock_model_artifact_membership
    original_dependencies = sync_module.lock_runtime_dependencies

    def record_api(session, config_ids, *, expected_user_id):
        events.append("api")
        return original_api(
            session,
            config_ids,
            expected_user_id=expected_user_id,
        )

    def record_gate(session):
        events.append("gate")
        return original_gate(session)

    def record_dependencies(session, references, *, expected_user_id=None):
        events.append("dependencies")
        return original_dependencies(
            session,
            references,
            expected_user_id=expected_user_id,
        )

    monkeypatch.setattr(
        sync_module,
        "_lock_external_api_configs",
        record_api,
    )
    monkeypatch.setattr(
        sync_module,
        "lock_model_artifact_membership",
        record_gate,
    )
    monkeypatch.setattr(
        sync_module,
        "lock_runtime_dependencies",
        record_dependencies,
    )

    service.create_task(
        task_name="ordered-path-create",
        user_id="user-1",
        training_config={
            "base_model_path": f"{model_path}/prospective-model"
        },
        is_active=True,
    )

    assert events[:3] == ["api", "gate", "dependencies"]


def test_inactive_target_cannot_activate_managed_path_while_model_is_deleting(
    sync_model_guard,
) -> None:
    engine, service, model_path = sync_model_guard
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="activate-inactive-target",
                task_name="activate-inactive-target",
                user_id="user-1",
                is_active=True,
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="inactive-managed-target",
                task_id="activate-inactive-target",
                target_name="inactive-managed-target",
                base_model_path=f"{model_path}/inactive-target",
                is_active=False,
            )
        )
        session.commit()
    _mark_model_deleting(engine)

    with pytest.raises(ValueError, match="being deleted"):
        service.update_training_target(
            "inactive-managed-target",
            task_id="activate-inactive-target",
            is_active=True,
            expected_user_id="user-1",
        )

    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id
                == "inactive-managed-target"
            )
        ).one()
        assert target.is_active is False
