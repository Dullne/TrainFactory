from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlmodel import Session, select

from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_registry_entity import (
    ModelRegistryDB,
    ModelVersionDB,
)
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.entities.generation_task_entity import GenerationTaskDB
from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.milvus_collection_entity import (
    MilvusCollectionDB,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)


deployment_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)
registry_module = importlib.import_module(
    "train_factory.storage.services.model_registry_service"
)
docker_module = importlib.import_module("train_factory.deployment.docker_deployer")


@pytest.fixture
def claimed_model(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'claimed-model.db'}")
    for table in (
        ModelRegistryDB.__table__,
        ModelArtifactMembershipGateDB.__table__,
        ModelVersionDB.__table__,
        DeploymentDB.__table__,
        DeploymentReplicaDB.__table__,
        LoadedAdapterDB.__table__,
        ModelConfigDB.__table__,
        TrainingTaskDB.__table__,
        GenerationTaskDB.__table__,
        EvaluationTaskDB.__table__,
        ExternalSyncTaskDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
        ExternalSyncTrainingDB.__table__,
        MilvusCollectionDB.__table__,
    ):
        table.create(engine)
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(deployment_module, "get_session", get_session)
    monkeypatch.setattr(registry_module, "get_session", get_session)
    monkeypatch.setattr(registry_module.settings, "models_dir", tmp_path / "models")

    model_id = "model-claimed"
    model_path = tmp_path / "models" / model_id
    model_path.mkdir(parents=True)
    (model_path / "weights.bin").write_bytes(b"weights")
    with Session(engine) as session:
        session.add(
            ModelRegistryDB(
                model_id=model_id,
                model_name="claimed-model",
                model_type="llm",
                model_path=str(model_path),
                source_type="downloaded",
            )
        )
        session.add(
            DeploymentDB(
                deployment_id="deployment-claimed",
                model_id=model_id,
                deployment_name="claimed-deployment",
                xinference_endpoint="http://127.0.0.1:11000",
                deploy_mode="container",
                container_name="trainfactory-vllm-claimed-deployment",
                inference_framework="vllm",
                config={"replica_schema_version": 1},
                status="running",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="replica-claimed",
                deployment_id="deployment-claimed",
                replica_index=0,
                container_name="trainfactory-vllm-claimed-deployment",
                endpoint="http://127.0.0.1:11000",
                port=11000,
                gpu_ids=[0],
                status="running",
                health_status="HEALTHY",
            )
        )
        session.commit()

    return engine, model_id, model_path


@pytest.fixture
def adapter_model_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'adapter-dependency.db'}")
    for table in (
        ModelRegistryDB.__table__,
        ModelArtifactMembershipGateDB.__table__,
        ModelVersionDB.__table__,
        DeploymentDB.__table__,
        DeploymentReplicaDB.__table__,
        LoadedAdapterDB.__table__,
        ModelConfigDB.__table__,
        TrainingTaskDB.__table__,
        GenerationTaskDB.__table__,
        EvaluationTaskDB.__table__,
        ExternalSyncTaskDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
        ExternalSyncTrainingDB.__table__,
        MilvusCollectionDB.__table__,
    ):
        table.create(engine)
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(deployment_module, "get_session", get_session)
    monkeypatch.setattr(registry_module, "get_session", get_session)
    monkeypatch.setattr(registry_module.settings, "models_dir", tmp_path / "models")

    model_id = "adapter-model"
    model_path = tmp_path / "models" / model_id
    model_path.mkdir(parents=True)
    (model_path / "adapter.bin").write_bytes(b"adapter")
    with Session(engine) as session:
        session.add(
            ModelRegistryDB(
                model_id=model_id,
                model_name="managed-adapter",
                model_type="llm",
                model_path=str(model_path),
                source_type="downloaded",
                is_adapter=True,
            )
        )
        session.add(
            DeploymentDB(
                deployment_id="adapter-target",
                model_id="base-model",
                deployment_name="adapter-target",
                xinference_endpoint="http://127.0.0.1:12000",
                deploy_mode="container",
                inference_framework="vllm",
                status="running",
                enable_lora=True,
                user_id="user-1",
            )
        )
        session.commit()

    return engine, model_id, model_path


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("adapter_status", ["loading", "loaded", "unloading"])
@pytest.mark.parametrize("reference_kind", ["source_model_id", "legacy_path"])
def test_delete_model_rejects_active_loaded_adapter_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    adapter_model_dependency,
    force: bool,
    adapter_status: str,
    reference_kind: str,
) -> None:
    engine, model_id, model_path = adapter_model_dependency
    with Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                deployment_id="adapter-target",
                adapter_name=f"active-{adapter_status}-{reference_kind}",
                adapter_path=(
                    str(model_path)
                    if reference_kind == "source_model_id"
                    else str(model_path).replace("\\", "/") + "/."
                ),
                source_model_id=(
                    model_id if reference_kind == "source_model_id" else None
                ),
                status=adapter_status,
                user_id="user-1",
            )
        )
        session.commit()

    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: cleanup_calls.append("runtime"),
    )
    monkeypatch.setattr(
        registry_module.shutil,
        "rmtree",
        lambda _path: cleanup_calls.append("file"),
    )

    with pytest.raises(ValueError, match="active loaded adapter"):
        registry_module.ModelRegistryService().delete_model(model_id, force=force)

    assert cleanup_calls == []
    assert (model_path / "adapter.bin").read_bytes() == b"adapter"
    with Session(engine) as session:
        assert session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).one()
        assert session.exec(
            select(LoadedAdapterDB).where(
                LoadedAdapterDB.source_model_id == model_id
                if reference_kind == "source_model_id"
                else LoadedAdapterDB.source_model_id.is_(None)
            )
        ).one()


@pytest.mark.parametrize(
    ("operation", "replica_id"),
    [("recreate", "replica-claimed"), ("delete", None)],
)
def test_force_delete_model_fails_closed_while_replica_lifecycle_is_claimed(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    claimed_model,
    operation: str,
    replica_id: str | None,
) -> None:
    engine, model_id, model_path = claimed_model
    service = deployment_module.DeploymentService()
    claim = service._claim_replica_operation(
        "deployment-claimed",
        operation=operation,
        replica_id=replica_id,
        user_id="user-1",
    )
    request.addfinalizer(lambda: service._release_replica_operation(claim))
    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda deployment, *_args, **_kwargs: cleanup_calls.append(
            deployment.deployment_id
        ),
    )

    with pytest.raises(
        deployment_module.ReplicaOperationBusyError,
        match="already in progress",
    ):
        registry_module.ModelRegistryService().delete_model(model_id, force=True)

    assert cleanup_calls == []
    assert (model_path / "weights.bin").read_bytes() == b"weights"
    with Session(engine) as session:
        assert session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).one()
        assert session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-claimed"
            )
        ).one()
        assert session.exec(select(DeploymentReplicaDB)).one()


def test_force_delete_model_refences_claim_before_first_container_removal(
    monkeypatch: pytest.MonkeyPatch,
    claimed_model,
) -> None:
    engine, model_id, model_path = claimed_model
    with Session(engine) as session:
        session.add(
            DeploymentReplicaDB(
                replica_id="replica-second",
                deployment_id="deployment-claimed",
                replica_index=1,
                container_name="trainfactory-vllm-claimed-deployment-r1",
                endpoint="http://127.0.0.1:11001",
                port=11001,
                gpu_ids=[1],
                status="running",
                health_status="HEALTHY",
            )
        )
        session.commit()

    class ClaimReplacingDocker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.replaced = False

        def container_exists(self, container_name: str) -> bool:
            self.calls.append(("exists", container_name))
            return True

        def container_exists_authoritative(self, container_name: str) -> bool:
            return self.container_exists(container_name)

        def get_managed_container_id(
            self,
            container_name: str,
            *,
            deployment_id: str,
            replica_id: str,
        ) -> str:
            self.calls.append(("inspect", container_name))
            if not self.replaced:
                self.replaced = True
                with Session(engine) as session:
                    parent = session.exec(
                        select(DeploymentDB).where(
                            DeploymentDB.deployment_id == deployment_id
                        )
                    ).one()
                    parent.replica_operation_token = "replacement-owner"
                    parent.replica_operation_generation += 1
                    session.add(parent)
                    session.commit()
            return f"sha256:{replica_id}"

        def remove_container_identity(self, container_id: str) -> bool:
            self.calls.append(("remove", container_id))
            return True

    docker = ClaimReplacingDocker()
    monkeypatch.setattr(docker_module, "docker_deployer", docker)
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_register_claim_heartbeat",
        lambda _claim: None,
    )

    with pytest.raises(
        deployment_module.ReplicaOperationLostError,
        match="ownership was lost",
    ):
        registry_module.ModelRegistryService().delete_model(model_id, force=True)

    assert [call for call in docker.calls if call[0] == "remove"] == []
    assert [call for call in docker.calls if call[0] == "inspect"] == [
        ("inspect", "trainfactory-vllm-claimed-deployment")
    ]
    assert (model_path / "weights.bin").read_bytes() == b"weights"
    with Session(engine) as session:
        assert session.exec(select(ModelRegistryDB)).one().model_id == model_id
        assert len(session.exec(select(DeploymentReplicaDB)).all()) == 2


def test_successful_repeated_model_delete_stops_and_removes_heartbeat_entry(
    monkeypatch: pytest.MonkeyPatch,
    claimed_model,
) -> None:
    _engine, model_id, _model_path = claimed_model
    service = deployment_module.deployment_service
    stopped: list[str] = []
    joined: list[float | None] = []
    tokens: list[str] = []

    def register_without_thread(claim) -> None:
        tokens.append(claim.token)
        service._claim_heartbeats[claim.token] = (
            SimpleNamespace(set=lambda: stopped.append(claim.token)),
            SimpleNamespace(join=lambda timeout=None: joined.append(timeout)),
        )

    monkeypatch.setattr(service, "_register_claim_heartbeat", register_without_thread)
    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: None,
    )

    registry = registry_module.ModelRegistryService()
    assert registry.delete_model(model_id, force=True) is True
    assert registry.delete_model(model_id, force=True) is False
    assert stopped == tokens
    assert joined == [1.0]
    assert service._claim_heartbeats == {}


def test_force_delete_runs_runtime_and_file_cleanup_outside_session(
    monkeypatch: pytest.MonkeyPatch,
    claimed_model,
) -> None:
    _engine, model_id, _model_path = claimed_model
    original_get_session = registry_module.get_session
    active_sessions = 0
    observations: list[tuple[str, int]] = []

    @contextmanager
    def tracked_get_session():
        nonlocal active_sessions
        with original_get_session() as session:
            active_sessions += 1
            try:
                yield session
            finally:
                active_sessions -= 1

    monkeypatch.setattr(registry_module, "get_session", tracked_get_session)
    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: observations.append(
            ("runtime", active_sessions)
        ),
    )
    monkeypatch.setattr(
        registry_module.shutil,
        "rmtree",
        lambda _path: observations.append(("file", active_sessions)),
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_register_claim_heartbeat",
        lambda _claim: None,
    )

    assert registry_module.ModelRegistryService().delete_model(
        model_id,
        force=True,
    ) is True
    assert observations == [("runtime", 0), ("file", 0)]


def test_force_delete_final_cas_loss_retains_new_owner_and_database_records(
    monkeypatch: pytest.MonkeyPatch,
    claimed_model,
) -> None:
    engine, model_id, model_path = claimed_model
    original_rmtree = registry_module.shutil.rmtree

    with Session(engine) as session:
        session.add(
            ModelVersionDB(
                version_id="version-claimed",
                model_id=model_id,
                version="v2",
                model_path=str(model_path),
            )
        )
        session.commit()

    def replace_owner_after_file_cleanup(path) -> None:
        original_rmtree(path)
        with Session(engine) as session:
            parent = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == "deployment-claimed"
                )
            ).one()
            parent.replica_operation_token = "replacement-owner"
            parent.replica_operation_generation += 1
            session.add(parent)
            session.commit()

    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(registry_module.shutil, "rmtree", replace_owner_after_file_cleanup)
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_register_claim_heartbeat",
        lambda _claim: None,
    )

    with pytest.raises(
        deployment_module.ReplicaOperationLostError,
        match="external cleanup.*database records retained",
    ):
        registry_module.ModelRegistryService().delete_model(model_id, force=True)

    assert not model_path.exists()
    with Session(engine) as session:
        assert session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).one()
        parent = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-claimed"
            )
        ).one()
        assert parent.replica_operation_token == "replacement-owner"
        assert parent.replica_operation_generation > 1
        assert session.exec(select(DeploymentReplicaDB)).one()
        assert session.exec(select(ModelVersionDB)).one().version_id == "version-claimed"

    for operation in ("start", "restart", "recreate", "load_adapter"):
        with pytest.raises(ValueError, match="being deleted"):
            deployment_module.deployment_service._claim_replica_operation(
                "deployment-claimed",
                operation=operation,
                replica_id=None,
                user_id="user-1",
            )
