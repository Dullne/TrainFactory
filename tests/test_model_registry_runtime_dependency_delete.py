"""Runtime dependency fences for registry-model force deletion."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import importlib
import inspect
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.evaluation_task_entity import (
    EvaluationFramework,
    EvaluationStatus,
    EvaluationTaskDB,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import (
    DeploymentReplicaDB,
)
from train_factory.storage.entities.milvus_collection_entity import (
    MilvusCollectionDB,
)
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_registry_entity import (
    MODEL_DELETE_INTENT_METADATA_KEY,
    ModelRegistryDB,
    ModelVersionDB,
)
from train_factory.api.routes import registry_routes


registry_module = importlib.import_module(
    "train_factory.storage.services.model_registry_service"
)
runtime_module = importlib.import_module(
    "train_factory.storage.services.runtime_dependency_service"
)
generation_module = importlib.import_module(
    "train_factory.storage.services.generation_task_service"
)
milvus_module = importlib.import_module(
    "train_factory.storage.services.milvus_collection_service"
)
deployment_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)
_ORIGINAL_REMOVE_DEPLOYMENT_CONTAINER = (
    registry_module._remove_deployment_container
)


@pytest.fixture
def registry_runtime_dependencies(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'registry-runtime-dependencies.db'}"
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()

    @contextmanager
    def test_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    models_dir = tmp_path / "models"
    monkeypatch.setattr(registry_module, "get_session", test_session)
    monkeypatch.setattr(generation_module, "get_session", test_session)
    monkeypatch.setattr(milvus_module, "get_session", test_session)
    monkeypatch.setattr(deployment_module, "get_session", test_session)
    monkeypatch.setattr(registry_module.settings, "models_dir", models_dir)

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

    model_id = "runtime-guarded-model"
    model_path = models_dir / model_id
    model_path.mkdir(parents=True)
    with Session(engine) as session:
        session.add(
            ModelRegistryDB(
                model_id=model_id,
                model_name=model_id,
                model_type="embedding",
                model_path=str(model_path),
                source_type="trained",
                status="available",
                user_id="user-1",
            )
        )
        session.add(
            ModelConfigDB(
                config_id="runtime-config",
                config_name="runtime-config",
                source_type="local_deployed",
                registry_id=model_id,
                model_type="embedding",
                provider="openai-compatible",
                api_endpoint="http://inference.invalid/v1",
                model_name=model_id,
                user_id="user-1",
            )
        )
        session.commit()

    return SimpleNamespace(
        engine=engine,
        model_id=model_id,
        cleanup_calls=cleanup_calls,
    )


def _assert_registry_records_retained(fixture) -> None:
    with Session(fixture.engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == fixture.model_id
            )
        ).one()
        assert model.status == "available"
        assert model.extra_metadata is None
        assert session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "runtime-config"
            )
        ).one()


def _make_runtime_config_legacy_mismatch(fixture) -> None:
    with Session(fixture.engine) as session:
        session.add(
            ModelRegistryDB(
                model_id="legacy-config-registry",
                model_name="legacy-config-registry",
                model_type="embedding",
                model_path="/external/legacy-config-registry",
                status="available",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentDB(
                deployment_id="legacy-config-deployment",
                deployment_name="legacy-config-deployment",
                model_id=fixture.model_id,
                xinference_endpoint="http://inference.invalid:8099",
                status="stopped",
                user_id="user-1",
            )
        )
        config = session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "runtime-config"
            )
        ).one()
        config.registry_id = "legacy-config-registry"
        config.deployment_id = "legacy-config-deployment"
        session.add(config)
        session.commit()


def test_force_delete_rejects_active_generation_before_any_cleanup(
    registry_runtime_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = registry_runtime_dependencies
    _make_runtime_config_legacy_mismatch(fixture)
    claim_calls: list[str] = []
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_claim_replica_operation",
        lambda deployment_id, **_kwargs: claim_calls.append(deployment_id),
    )
    with Session(fixture.engine) as session:
        session.add(
            GenerationTaskDB(
                task_id="active-generation",
                task_name="active-generation",
                input_path="/managed/input.jsonl",
                llm_config={},
                steps_config={},
                embedding_config_id="runtime-config",
                status=GenerationStatus.RUNNING,
                user_id="user-1",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="active runtime task"):
        registry_module.ModelRegistryService().delete_model(
            fixture.model_id,
            force=True,
        )

    assert fixture.cleanup_calls == []
    assert claim_calls == []
    _assert_registry_records_retained(fixture)


@pytest.mark.parametrize(
    ("consumer_kind", "model_configs"),
    [
        (
            EvaluationFramework.MTEB,
            [{"config_id": "runtime-config"}],
        ),
        (
            EvaluationFramework.DEEPEVAL,
            [{"embedding": {"config_id": "runtime-config"}}],
        ),
    ],
)
def test_force_delete_rejects_active_evaluation_before_any_cleanup(
    registry_runtime_dependencies,
    consumer_kind: str,
    model_configs: list[dict],
) -> None:
    fixture = registry_runtime_dependencies
    with Session(fixture.engine) as session:
        session.add(
            EvaluationTaskDB(
                task_id=f"active-{consumer_kind}",
                task_name=f"active-{consumer_kind}",
                eval_framework=consumer_kind,
                model_configs=model_configs,
                status=EvaluationStatus.RUNNING,
                user_id="user-1",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="active runtime task"):
        registry_module.ModelRegistryService().delete_model(
            fixture.model_id,
            force=True,
        )

    assert fixture.cleanup_calls == []
    _assert_registry_records_retained(fixture)


def test_force_delete_rejects_active_external_sync_before_any_cleanup(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    with Session(fixture.engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="active-external-sync",
                task_name="active-external-sync",
                generation_config={
                    "embedding_config": {"config_id": "runtime-config"}
                },
                is_active=True,
                user_id="user-1",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="active runtime task"):
        registry_module.ModelRegistryService().delete_model(
            fixture.model_id,
            force=True,
        )

    assert fixture.cleanup_calls == []
    _assert_registry_records_retained(fixture)


@pytest.mark.parametrize("reference_kind", ["target", "legacy-parent"])
def test_force_delete_rejects_active_external_sync_model_path(
    registry_runtime_dependencies,
    reference_kind: str,
) -> None:
    fixture = registry_runtime_dependencies
    with Session(fixture.engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == fixture.model_id
            )
        ).one()
        alias_path = os.path.join(
            model.model_path,
            "alias-segment",
            "..",
            "training-base",
        )
        task = ExternalSyncTaskDB(
            task_id=f"active-path-sync-{reference_kind}",
            task_name=f"active-path-sync-{reference_kind}",
            training_config=(
                {"base_model_path": alias_path}
                if reference_kind == "legacy-parent"
                else None
            ),
            is_active=True,
            user_id="user-1",
        )
        session.add(task)
        if reference_kind == "target":
            session.add(
                ExternalSyncTrainingTargetDB(
                    target_id="active-path-target",
                    task_id=task.task_id,
                    target_name="active-path-target",
                    base_model_path=alias_path,
                    is_active=True,
                )
            )
        session.commit()

    with pytest.raises(ValueError, match="external sync"):
        registry_module.ModelRegistryService().delete_model(
            fixture.model_id,
            force=True,
        )

    assert fixture.cleanup_calls == []
    _assert_registry_records_retained(fixture)


@pytest.mark.parametrize("inactive_scope", ["task", "target"])
def test_force_delete_allows_inactive_external_sync_model_path(
    registry_runtime_dependencies,
    inactive_scope: str,
) -> None:
    fixture = registry_runtime_dependencies
    with Session(fixture.engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == fixture.model_id
            )
        ).one()
        managed_path = os.path.join(model.model_path, "training-base")
        task = ExternalSyncTaskDB(
            task_id=f"inactive-path-sync-{inactive_scope}",
            task_name=f"inactive-path-sync-{inactive_scope}",
            training_config=(
                {"base_model_path": managed_path}
                if inactive_scope == "task"
                else None
            ),
            is_active=inactive_scope != "task",
            user_id="user-1",
        )
        session.add(task)
        if inactive_scope == "target":
            session.add(
                ExternalSyncTrainingTargetDB(
                    target_id="inactive-path-target",
                    task_id=task.task_id,
                    target_name="inactive-path-target",
                    base_model_path=managed_path,
                    is_active=False,
                )
            )
        session.commit()

    assert registry_module.ModelRegistryService().delete_model(
        fixture.model_id,
        force=True,
    ) is True
    assert fixture.cleanup_calls == ["file"]


def test_force_delete_rejects_milvus_binding_before_any_cleanup(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _make_runtime_config_legacy_mismatch(fixture)
    with Session(fixture.engine) as session:
        session.add(
            MilvusCollectionDB(
                collection_name="durably-bound-collection",
                embedding_config_id="runtime-config",
                status="active",
                user_id="user-1",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="Milvus collection"):
        registry_module.ModelRegistryService().delete_model(
            fixture.model_id,
            force=True,
        )

    assert fixture.cleanup_calls == []
    _assert_registry_records_retained(fixture)


def _mark_model_delete_owner(
    engine,
    *,
    model_id: str,
    delete_token: str,
) -> None:
    with Session(engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).one()
        model.status = "deleting"
        model.extra_metadata = {
            MODEL_DELETE_INTENT_METADATA_KEY: {
                "token": delete_token,
            }
        }
        session.add(model)
        session.commit()


def _add_claimed_deployment(engine, *, model_id: str) -> None:
    with Session(engine) as session:
        config = session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "runtime-config"
            )
        ).one()
        config.deployment_id = "claimed-deployment"
        session.add(config)
        session.add(
            DeploymentDB(
                deployment_id="claimed-deployment",
                deployment_name="claimed-deployment",
                model_id=model_id,
                xinference_endpoint="http://inference.invalid:8000",
                replica_operation_token="deployment-delete-token",
                replica_operation_kind="delete",
                replica_operation_generation=7,
                user_id="user-1",
            )
        )
        session.commit()


def test_model_delete_scope_rejects_wrong_model_owner_token(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    _add_claimed_deployment(fixture.engine, model_id=fixture.model_id)

    with Session(fixture.engine) as session:
        with pytest.raises(
            runtime_module.RuntimeDependencyUnavailableError,
            match="(?i)model deletion ownership",
        ):
            runtime_module._lock_model_delete_dependency_scope(
                session,
                model_id=fixture.model_id,
                delete_token="wrong-model-delete-token",
                deployment_claims={
                    "claimed-deployment": ("deployment-delete-token", 7)
                },
                execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
            )


def test_model_delete_scope_rejects_stale_deployment_claim_generation(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    _add_claimed_deployment(fixture.engine, model_id=fixture.model_id)

    with Session(fixture.engine) as session:
        with pytest.raises(
            runtime_module.RuntimeDependencyUnavailableError,
            match="(?i)deployment deletion ownership",
        ):
            runtime_module._lock_model_delete_dependency_scope(
                session,
                model_id=fixture.model_id,
                delete_token="real-model-delete-token",
                deployment_claims={
                    "claimed-deployment": ("deployment-delete-token", 6)
                },
                execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
            )


def test_model_delete_phase_a_rejects_any_existing_deployment_claim(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    _add_claimed_deployment(fixture.engine, model_id=fixture.model_id)

    with Session(fixture.engine) as session:
        with pytest.raises(
            runtime_module.RuntimeDependencyUnavailableError,
            match="(?i)deployment deletion ownership",
        ):
            runtime_module._lock_model_delete_dependency_scope(
                session,
                model_id=fixture.model_id,
                delete_token="real-model-delete-token",
                deployment_claims={},
                execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
            )


def test_model_delete_scope_rejects_inactive_external_sync_binding(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _make_runtime_config_legacy_mismatch(fixture)
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    with Session(fixture.engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="inactive-sync-binding",
                task_name="inactive-sync-binding",
                base_deployment_id="legacy-config-deployment",
                is_active=False,
                user_id="user-1",
            )
        )
        session.commit()

    with Session(fixture.engine) as session:
        with pytest.raises(
            runtime_module.RuntimeDependencyUnavailableError,
            match="external sync",
        ):
            scope = runtime_module._lock_model_delete_dependency_scope(
                session,
                model_id=fixture.model_id,
                delete_token="real-model-delete-token",
                deployment_claims={},
                execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
            )
            runtime_module._guard_model_delete_runtime_dependencies(
                session,
                scope,
                execution_snapshot=(
                    runtime_module.RuntimeExecutionSnapshot()
                ),
            )


def test_model_delete_scope_rejects_unrelated_deleting_parent(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    with Session(fixture.engine) as session:
        session.add(
            ModelRegistryDB(
                model_id="other-deleting-model",
                model_name="other-deleting-model",
                model_type="embedding",
                model_path="/external/other-deleting-model",
                status="deleting",
                extra_metadata={
                    MODEL_DELETE_INTENT_METADATA_KEY: {
                        "token": "other-delete-token"
                    }
                },
                user_id="user-1",
            )
        )
        session.add(
            DeploymentDB(
                deployment_id="other-deleting-deployment",
                deployment_name="other-deleting-deployment",
                model_id="other-deleting-model",
                xinference_endpoint="http://inference.invalid:8200",
                replica_operation_token="other-deployment-token",
                replica_operation_kind="delete",
                replica_operation_generation=3,
                user_id="user-1",
            )
        )
        config = session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "runtime-config"
            )
        ).one()
        config.deployment_id = "other-deleting-deployment"
        session.add(config)
        session.commit()

    with Session(fixture.engine) as session:
        with pytest.raises(
            runtime_module.RuntimeDependencyUnavailableError,
            match="unrelated model deletion",
        ):
            runtime_module._lock_model_delete_dependency_scope(
                session,
                model_id=fixture.model_id,
                delete_token="real-model-delete-token",
                deployment_claims={},
                execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
            )


def test_phase_a_intent_blocks_prospective_task_and_milvus_writers(
    registry_runtime_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = registry_runtime_dependencies
    original_scope_lock = registry_module._lock_model_delete_dependency_scope
    writer_checks: list[str] = []

    def lock_after_writer_probes(session, **kwargs):
        if not writer_checks:
            with pytest.raises(
                runtime_module.RuntimeDependencyUnavailableError,
                match="Model is being deleted",
            ):
                generation_module.GenerationTaskService().create_task(
                    task_id="prospective-generation",
                    task_name="prospective-generation",
                    input_path="/managed/input.jsonl",
                    llm_config={},
                    steps_config={},
                    embedding_config_id="runtime-config",
                    user_id="user-1",
                )
            writer_checks.append("generation")

            with pytest.raises(
                milvus_module.MilvusCollectionUnavailableError,
                match="Embedding config is unavailable",
            ):
                milvus_module.MilvusCollectionService().register_collection(
                    "prospective-collection",
                    embedding_config_id="runtime-config",
                    user_id="user-1",
                )
            writer_checks.append("milvus")
        return original_scope_lock(session, **kwargs)

    monkeypatch.setattr(
        registry_module,
        "_lock_model_delete_dependency_scope",
        lock_after_writer_probes,
    )

    assert registry_module.ModelRegistryService().delete_model(
        fixture.model_id,
        force=True,
    ) is True
    assert writer_checks == ["generation", "milvus"]
    with Session(fixture.engine) as session:
        assert session.exec(select(GenerationTaskDB)).all() == []
        assert session.exec(select(MilvusCollectionDB)).all() == []


def test_phase_a_commits_intent_before_dependency_union_lock() -> None:
    source = inspect.getsource(
        registry_module.ModelRegistryService.delete_model
    )
    phase_a2 = source.index("# Phase A2 starts a fresh transaction")
    intent_commit = source.rfind("session.commit()", 0, phase_a2)
    union_lock = source.index(
        "dependency_scope = _lock_model_delete_dependency_scope(",
        phase_a2,
    )

    assert intent_commit != -1
    assert intent_commit < phase_a2 < union_lock
    assert "_lock_model_delete_dependency_scope(" not in source[
        intent_commit:phase_a2
    ]


def test_force_delete_uses_legacy_config_union_in_every_dependency_phase(
    registry_runtime_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = registry_runtime_dependencies
    with Session(fixture.engine) as session:
        session.add(
            ModelRegistryDB(
                model_id="other-registry-model",
                model_name="other-registry-model",
                model_type="embedding",
                model_path="/external/other-registry-model",
                status="available",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentDB(
                deployment_id="slated-deployment",
                deployment_name="slated-deployment",
                model_id=fixture.model_id,
                xinference_endpoint="http://inference.invalid:8100",
                inference_framework="vllm",
                deploy_mode="container",
                container_name="trainfactory-vllm-slated-deployment",
                config={"replica_schema_version": 1},
                status="stopped",
                user_id="user-1",
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="slated-deployment-r0",
                deployment_id="slated-deployment",
                replica_index=0,
                container_name="trainfactory-vllm-slated-deployment",
                endpoint="http://inference.invalid:8100",
                port=8100,
                gpu_ids=[0],
                status="stopped",
            )
        )
        config = session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "runtime-config"
            )
        ).one()
        # Legacy mismatch: registry_id points elsewhere, but the deployment is
        # in the target model's force-delete set. It must still be deleted.
        config.registry_id = "other-registry-model"
        config.deployment_id = "slated-deployment"
        session.add(config)
        session.commit()

    original_scope_lock = registry_module._lock_model_delete_dependency_scope
    phase_config_ids: list[tuple[str, ...]] = []

    def record_scope_configs(session, **kwargs):
        scope = original_scope_lock(session, **kwargs)
        phase_config_ids.append(
            tuple(config.config_id for config in scope.configs)
        )
        return scope

    monkeypatch.setattr(
        registry_module,
        "_lock_model_delete_dependency_scope",
        record_scope_configs,
    )

    assert registry_module.ModelRegistryService().delete_model(
        fixture.model_id,
        force=True,
    ) is True

    assert phase_config_ids == [("runtime-config",)] * 5
    with Session(fixture.engine) as session:
        assert session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "runtime-config"
            )
        ).first() is None
        assert session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == "other-registry-model"
            )
        ).one()


def test_zero_child_container_cleanup_requires_exact_claim_before_remove(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[str] = []
    claim = deployment_module.ReplicaOperationClaim(
        deployment_id="zero-child-deployment",
        token="stale-delete-token",
        generation=4,
        operation="delete",
        replica_id=None,
    )
    deployment = DeploymentDB(
        deployment_id=claim.deployment_id,
        deployment_name=claim.deployment_id,
        model_id="runtime-guarded-model",
        xinference_endpoint="http://inference.invalid:8300",
        inference_framework="vllm",
        deploy_mode="container",
        container_name="trainfactory-vllm-zero-child",
        config={},
        status="stopped",
    )

    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_require_managed_container",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_require_replica_operation_ownership",
        lambda _claim: (_ for _ in ()).throw(
            deployment_module.ReplicaOperationLostError("claim replaced")
        ),
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "remove_container",
        lambda _name: cleanup_calls.append("remove") or True,
    )

    with pytest.raises(
        deployment_module.ReplicaOperationLostError,
        match="claim replaced",
    ):
        registry_module._remove_deployment_container(
            deployment,
            deployment.model_id,
            replicas=(),
            replica_operation_claim=claim,
        )

    assert cleanup_calls == []


def test_canonical_zero_child_container_never_mutates_unproven_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[str] = []
    claim = deployment_module.ReplicaOperationClaim(
        deployment_id="canonical-zero-child",
        token="delete-token",
        generation=5,
        operation="delete",
        replica_id=None,
    )
    deployment = DeploymentDB(
        deployment_id=claim.deployment_id,
        deployment_name=claim.deployment_id,
        model_id="runtime-guarded-model",
        xinference_endpoint="http://inference.invalid:8310",
        inference_framework="vllm",
        deploy_mode="container",
        container_name="trainfactory-vllm-canonical-zero-child",
        config={"replica_schema_version": 1},
        status="stopped",
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_uses_replica_lifecycle",
        lambda _deployment: True,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_require_managed_container",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_require_replica_operation_ownership",
        lambda _claim: None,
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "container_exists_authoritative",
        lambda _name: True,
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "remove_container",
        lambda _name: cleanup_calls.append("name-remove") or True,
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "remove_container_identity",
        lambda _identity: cleanup_calls.append("id-remove") or True,
    )

    with pytest.raises(RuntimeError, match="replica identity"):
        registry_module._remove_deployment_container(
            deployment,
            deployment.model_id,
            replicas=(),
            replica_operation_claim=claim,
        )

    assert cleanup_calls == []


def test_replica_cleanup_treats_docker_probe_failure_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[str] = []
    claim = deployment_module.ReplicaOperationClaim(
        deployment_id="probe-failure-deployment",
        token="delete-token",
        generation=6,
        operation="delete",
        replica_id=None,
    )
    deployment = DeploymentDB(
        deployment_id=claim.deployment_id,
        deployment_name=claim.deployment_id,
        model_id="runtime-guarded-model",
        xinference_endpoint="http://inference.invalid:8320",
        inference_framework="vllm",
        deploy_mode="container",
        container_name="trainfactory-vllm-probe-failure",
        config={"replica_schema_version": 1},
        status="stopped",
    )
    replica = DeploymentReplicaDB(
        replica_id="probe-failure-replica",
        deployment_id=claim.deployment_id,
        replica_index=0,
        container_name="trainfactory-vllm-probe-failure",
        endpoint="http://inference.invalid:8320",
        port=8320,
        gpu_ids=[0],
        status="stopped",
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_expected_replica_container_name",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_require_replica_operation_ownership",
        lambda _claim: None,
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "_run_command",
        lambda *_args, **_kwargs: (False, "daemon unavailable"),
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "remove_container_identity",
        lambda _identity: cleanup_calls.append("id-remove") or True,
    )

    with pytest.raises(RuntimeError, match="replica cleanup failed"):
        registry_module._remove_deployment_container(
            deployment,
            deployment.model_id,
            replicas=(replica,),
            replica_operation_claim=claim,
        )

    assert cleanup_calls == []


def test_force_delete_retains_records_when_container_probe_is_unknown(
    registry_runtime_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = registry_runtime_dependencies
    container_name = "trainfactory-vllm-unknown-probe"
    with Session(fixture.engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="unknown-probe-deployment",
                deployment_name="unknown-probe-deployment",
                model_id=fixture.model_id,
                xinference_endpoint="http://inference.invalid:8330",
                inference_framework="vllm",
                deploy_mode="container",
                container_name=container_name,
                config={"replica_schema_version": 1},
                status="stopped",
                user_id="user-1",
            )
        )
        config = session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "runtime-config"
            )
        ).one()
        config.deployment_id = "unknown-probe-deployment"
        session.add(config)
        session.commit()

    docker_mutations: list[str] = []
    monkeypatch.setattr(
        registry_module,
        "_remove_deployment_container",
        _ORIGINAL_REMOVE_DEPLOYMENT_CONTAINER,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_register_claim_heartbeat",
        lambda _claim: None,
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "_run_command",
        lambda *_args, **_kwargs: (False, "daemon unavailable"),
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "remove_container",
        lambda _name: docker_mutations.append("name-remove") or True,
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "remove_container_identity",
        lambda _identity: docker_mutations.append("id-remove") or True,
    )

    with pytest.raises(RuntimeError, match="replica identity is unknown"):
        registry_module.ModelRegistryService().delete_model(
            fixture.model_id,
            force=True,
        )

    assert docker_mutations == []
    assert fixture.cleanup_calls == []
    with Session(fixture.engine) as session:
        assert session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == fixture.model_id
            )
        ).one()
        assert session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == "unknown-probe-deployment"
            )
        ).one()
        assert session.exec(
            select(ModelConfigDB).where(
                ModelConfigDB.config_id == "runtime-config"
            )
        ).one()


def test_shared_runtime_cleanup_requires_exact_claim_before_terminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[str] = []
    claim = deployment_module.ReplicaOperationClaim(
        deployment_id="shared-deployment",
        token="stale-delete-token",
        generation=9,
        operation="delete",
        replica_id=None,
    )
    deployment = DeploymentDB(
        deployment_id=claim.deployment_id,
        deployment_name=claim.deployment_id,
        model_id="runtime-guarded-model",
        xinference_endpoint="http://xinference.invalid:9997",
        inference_framework="xinference",
        deploy_mode="shared",
        model_uid="shared-model-uid",
        config={"replica_schema_version": 1},
        status="running",
    )
    client = SimpleNamespace(
        terminate_model=lambda _uid: cleanup_calls.append("terminate")
    )

    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_require_replica_operation_ownership",
        lambda _claim: (_ for _ in ()).throw(
            deployment_module.ReplicaOperationLostError("claim replaced")
        ),
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: client,
    )

    with pytest.raises(
        deployment_module.ReplicaOperationLostError,
        match="claim replaced",
    ):
        registry_module._remove_deployment_container(
            deployment,
            deployment.model_id,
            replicas=(),
            replica_operation_claim=claim,
        )

    assert cleanup_calls == []


def test_replica_cleanup_stops_before_later_member_after_claim_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[str] = []
    ownership_checks: list[int] = []
    claim = deployment_module.ReplicaOperationClaim(
        deployment_id="replicated-deployment",
        token="delete-token",
        generation=4,
        operation="delete",
        replica_id=None,
    )
    deployment = DeploymentDB(
        deployment_id=claim.deployment_id,
        deployment_name=claim.deployment_id,
        model_id="runtime-guarded-model",
        xinference_endpoint="http://inference.invalid:8300",
        inference_framework="vllm",
        deploy_mode="container",
        container_name="trainfactory-vllm-replicated",
        config={"replica_schema_version": 1},
        status="stopped",
    )
    replicas = [
        DeploymentReplicaDB(
            replica_id=f"replica-{index}",
            deployment_id=claim.deployment_id,
            replica_index=index,
            container_name=f"replica-container-{index}",
            endpoint=f"http://inference.invalid:{8301 + index}",
            port=8301 + index,
            gpu_ids=[index],
            status="stopped",
        )
        for index in range(2)
    ]

    def require_ownership(_claim) -> None:
        ownership_checks.append(len(ownership_checks) + 1)
        if len(ownership_checks) == 2:
            raise deployment_module.ReplicaOperationLostError(
                "claim replaced"
            )

    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_expected_replica_container_name",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_require_replica_operation_ownership",
        require_ownership,
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "container_exists_authoritative",
        lambda _name: True,
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "get_managed_container_id",
        lambda name, **_kwargs: f"sha256:{name}",
    )
    monkeypatch.setattr(
        deployment_module.docker_deployer,
        "remove_container_identity",
        lambda container_id: removed.append(container_id) or True,
    )

    with pytest.raises(
        deployment_module.ReplicaOperationLostError,
        match="claim replaced",
    ):
        registry_module._remove_deployment_container(
            deployment,
            deployment.model_id,
            replicas=replicas,
            replica_operation_claim=claim,
        )

    assert removed == ["sha256:replica-container-0"]


def test_shared_runtime_cleanup_accepts_authoritatively_absent_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    claim = deployment_module.ReplicaOperationClaim(
        deployment_id="shared-deployment",
        token="delete-token",
        generation=10,
        operation="delete",
        replica_id=None,
    )
    deployment = DeploymentDB(
        deployment_id=claim.deployment_id,
        deployment_name=claim.deployment_id,
        model_id="runtime-guarded-model",
        xinference_endpoint="http://xinference.invalid:9997",
        inference_framework="xinference",
        deploy_mode="shared",
        model_uid="already-absent-model",
        config={"replica_schema_version": 1},
        status="running",
    )

    def terminate_model(_uid: str) -> None:
        calls.append("terminate")
        raise RuntimeError("404 not found")

    client = SimpleNamespace(
        terminate_model=terminate_model,
        get_model=lambda _uid: calls.append("get") or None,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_require_replica_operation_ownership",
        lambda _claim: calls.append("renew"),
    )
    monkeypatch.setattr(
        deployment_module.deployment_service,
        "_get_xinference_client",
        lambda *_args, **_kwargs: client,
    )

    registry_module._remove_deployment_container(
        deployment,
        deployment.model_id,
        replicas=(),
        replica_operation_claim=claim,
    )

    assert calls == ["renew", "terminate", "renew", "get"]


def test_model_delete_scope_locks_lower_id_overlap_owner_in_sorted_union(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    with Session(fixture.engine) as session:
        target = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == fixture.model_id
            )
        ).one()
        session.add(
            ModelRegistryDB(
                model_id="aaa-overlap-owner",
                model_name="aaa-overlap-owner",
                model_type="embedding",
                model_path=f"{target.model_path}/legacy-child",
                status="available",
                user_id="user-1",
            )
        )
        session.commit()

    locked_model_id_sets: list[tuple[str, ...]] = []
    with Session(fixture.engine) as session:
        original_exec = session.exec

        def recording_exec(statement, *args, **kwargs):
            from_tables = statement.get_final_froms()
            if (
                statement._for_update_arg is not None
                and from_tables
                and from_tables[0].name == "model_registry"
            ):
                params = statement.compile().params
                model_ids = next(
                    value
                    for key, value in params.items()
                    if key.startswith("model_id_")
                )
                locked_model_id_sets.append(tuple(model_ids))
            return original_exec(statement, *args, **kwargs)

        session.exec = recording_exec
        runtime_module._lock_model_delete_dependency_scope(
            session,
            model_id=fixture.model_id,
            delete_token="real-model-delete-token",
            deployment_claims={},
            execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
        )

    assert locked_model_id_sets == [
        ("aaa-overlap-owner", fixture.model_id)
    ]


def test_model_delete_scope_locks_version_owner_in_sorted_model_union(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    with Session(fixture.engine) as session:
        target = session.exec(
            select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == fixture.model_id
            )
        ).one()
        session.add(
            ModelRegistryDB(
                model_id="aaa-version-owner",
                model_name="aaa-version-owner",
                model_type="embedding",
                model_path="/external/version-owner",
                status="available",
                user_id="user-1",
            )
        )
        session.add(
            ModelVersionDB(
                version_id="overlapping-version",
                model_id="aaa-version-owner",
                version="v1",
                model_path=f"{target.model_path}/legacy-version",
            )
        )
        session.commit()

    locked_model_id_sets: list[tuple[str, ...]] = []
    with Session(fixture.engine) as session:
        original_exec = session.exec

        def recording_exec(statement, *args, **kwargs):
            from_tables = statement.get_final_froms()
            if (
                statement._for_update_arg is not None
                and from_tables
                and from_tables[0].name == "model_registry"
            ):
                params = statement.compile().params
                model_ids = next(
                    value
                    for key, value in params.items()
                    if key.startswith("model_id_")
                )
                locked_model_id_sets.append(tuple(model_ids))
            return original_exec(statement, *args, **kwargs)

        session.exec = recording_exec
        runtime_module._lock_model_delete_dependency_scope(
            session,
            model_id=fixture.model_id,
            delete_token="real-model-delete-token",
            deployment_claims={},
            execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
        )

    assert locked_model_id_sets == [
        ("aaa-version-owner", fixture.model_id)
    ]


def test_model_delete_scope_locks_dependencies_then_tasks_then_milvus(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    _add_claimed_deployment(fixture.engine, model_id=fixture.model_id)

    lock_layers: list[str] = []
    with Session(fixture.engine) as session:
        original_exec = session.exec

        def recording_exec(statement, *args, **kwargs):
            from_tables = statement.get_final_froms()
            if statement._for_update_arg is not None and from_tables:
                lock_layers.append(from_tables[0].name)
            return original_exec(statement, *args, **kwargs)

        session.exec = recording_exec
        scope = runtime_module._lock_model_delete_dependency_scope(
            session,
            model_id=fixture.model_id,
            delete_token="real-model-delete-token",
            deployment_claims={
                "claimed-deployment": ("deployment-delete-token", 7)
            },
            execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
        )
        runtime_module._guard_model_delete_runtime_dependencies(
            session,
            scope,
            execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
        )

    assert lock_layers == [
        "model_registry",
        "deployments",
        "model_configs",
        "deployment_replicas",
        "milvus_collections",
    ]


def test_model_delete_locks_bound_and_runtime_sync_tasks_as_one_sorted_union(
    registry_runtime_dependencies,
) -> None:
    fixture = registry_runtime_dependencies
    _make_runtime_config_legacy_mismatch(fixture)
    _mark_model_delete_owner(
        fixture.engine,
        model_id=fixture.model_id,
        delete_token="real-model-delete-token",
    )
    with Session(fixture.engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="z-bound-sync",
                task_name="z-bound-sync",
                is_active=False,
                user_id="user-1",
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="bound-target",
                task_id="z-bound-sync",
                target_name="bound-target",
                base_deployment_id="legacy-config-deployment",
            )
        )
        session.add(
            ExternalSyncTrainingDB(
                task_id="z-bound-sync",
                training_task_id="bound-training",
                user_id="user-1",
            )
        )
        session.add(
            ExternalSyncTaskDB(
                task_id="a-runtime-sync",
                task_name="a-runtime-sync",
                generation_config={
                    "embedding_config": {"config_id": "runtime-config"}
                },
                is_active=True,
                user_id="user-1",
            )
        )
        session.commit()

    sync_lock_params: list[dict] = []
    lock_tables: list[str] = []
    with Session(fixture.engine) as session:
        original_exec = session.exec

        def recording_exec(statement, *args, **kwargs):
            from_tables = statement.get_final_froms()
            if statement._for_update_arg is not None and from_tables:
                table_name = from_tables[0].name
                lock_tables.append(table_name)
                if table_name == "external_sync_tasks":
                    sync_lock_params.append(statement.compile().params)
            return original_exec(statement, *args, **kwargs)

        session.exec = recording_exec
        scope = runtime_module._lock_model_delete_dependency_scope(
            session,
            model_id=fixture.model_id,
            delete_token="real-model-delete-token",
            deployment_claims={},
            execution_snapshot=runtime_module.RuntimeExecutionSnapshot(),
        )
        with pytest.raises(
            runtime_module.RuntimeDependencyUnavailableError,
            match="external sync",
        ):
            runtime_module._guard_model_delete_runtime_dependencies(
                session,
                scope,
                execution_snapshot=(
                    runtime_module.RuntimeExecutionSnapshot()
                ),
                training_execution_task_ids=(),
            )

    assert len(sync_lock_params) == 1
    locked_sync_ids = {
        item
        for params in sync_lock_params
        for value in params.values()
        for item in (value if isinstance(value, (list, tuple)) else (value,))
        if item in {"a-runtime-sync", "z-bound-sync"}
    }
    assert locked_sync_ids == {"a-runtime-sync", "z-bound-sync"}
    assert lock_tables.index("external_sync_tasks") < lock_tables.index(
        "external_sync_training_targets"
    ) < lock_tables.index("external_sync_trainings")


def test_registry_delete_route_maps_runtime_dependency_conflict_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "get_model",
        lambda _model_id: {
            "model_id": "guarded-model",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "delete_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime_module.RuntimeDependencyUnavailableError(
                "Model dependency is referenced by an active runtime task"
            )
        ),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            registry_routes.delete_model(
                "guarded-model",
                force=True,
                current_user={"user_id": "user-1"},
            )
        )

    assert error.value.status_code == 409
    assert error.value.detail == (
        "Model dependency is referenced by an active runtime task"
    )


def test_registry_delete_route_maps_replica_operation_busy_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "get_model",
        lambda _model_id: {
            "model_id": "busy-model",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "delete_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_module.ReplicaOperationBusyError(
                "model deletion is already in progress"
            )
        ),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            registry_routes.delete_model(
                "busy-model",
                force=True,
                current_user={"user_id": "user-1"},
            )
        )

    assert error.value.status_code == 409
    assert error.value.detail == "model deletion is already in progress"
