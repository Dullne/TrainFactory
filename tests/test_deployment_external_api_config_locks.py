from __future__ import annotations

import importlib
from contextlib import contextmanager

import pytest
import sqlalchemy as sa
from sqlmodel import Session, select

from train_factory.deployment.replica_planner import DeploymentPlan, ReplicaPlan
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.external_api_config_entity import ExternalApiConfigDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB


deployment_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)


@pytest.fixture
def external_api_writer_db(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'deployment-api-locks.db'}")
    for table in (
        ExternalApiConfigDB.__table__,
        ModelArtifactMembershipGateDB.__table__,
        ModelRegistryDB.__table__,
        DeploymentDB.__table__,
        DeploymentReplicaDB.__table__,
    ):
        table.create(engine)

    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.add(
            ModelRegistryDB(
                model_id="model-1",
                model_name="Model",
                model_type="llm",
                model_path="/app/models/model-1",
                source_type="trained",
                status="available",
                user_id="user-1",
            )
        )
        session.add(
            ExternalApiConfigDB(
                config_id="api-foreign",
                config_name="foreign",
                user_id="user-2",
                api_url="https://foreign.invalid/v1",
                auth_config={},
            )
        )
        for suffix in ("a", "b"):
            session.add(
                ExternalApiConfigDB(
                    config_id=f"api-{suffix}",
                    config_name=f"owned-{suffix}",
                    user_id="user-1",
                    api_url=f"https://owned-{suffix}.invalid/v1",
                    auth_config={},
                )
            )
        session.commit()

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    model = {
        "model_id": "model-1",
        "model_name": "Model",
        "model_path": "/app/models/model-1",
        "model_type": "llm",
        "source_type": "trained",
        "file_size": None,
        "user_id": "user-1",
    }
    monkeypatch.setattr(deployment_module, "get_session", get_session)
    monkeypatch.setattr(
        deployment_module.model_registry_service,
        "get_model",
        lambda _model_id: dict(model),
    )
    monkeypatch.setattr(
        deployment_module.model_registry_service,
        "list_models",
        lambda **_kwargs: (
            [
                {
                    **model,
                    "model_name": "Model",
                    "model_path": "https://inference.invalid",
                }
            ],
            1,
        ),
    )
    service = deployment_module.DeploymentService()
    monkeypatch.setattr(
        service,
        "_create_config_for_deployment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(service, "_register_claim_heartbeat", lambda _claim: None)
    return service, engine, model


def _invoke_external_api_writer(
    service,
    engine,
    model: dict,
    *,
    writer: str,
    config_id: str,
    user_id: str,
):
    config = {"external_api_config_id": config_id}
    if writer == "create":
        return service.create_deployment(
            model_id="model-1",
            xinference_endpoint="http://xinference:9997",
            config=config,
            external_api_config_id=config_id,
            user_id=user_id,
        )
    if writer == "bind":
        return service.bind_existing_model(
            endpoint="https://inference.invalid",
            model_uid="Model",
            config=config,
            user_id=user_id,
        )
    if writer == "replica_plan":
        return service._persist_replica_plan(
            deployment_id="deployment-replica",
            model=model,
            deployment_name="replica",
            replica_count=1,
            gpu_memory_utilization=0.8,
            config={
                **config,
                "launch_config": {"framework": "vllm", "gpu_pool": [0]},
            },
            user_id=user_id,
            inference_framework="vllm",
            enable_lora=False,
            max_loras=4,
            max_lora_rank=64,
            external_api_config_id=config_id,
            plan=DeploymentPlan(
                replicas=(
                    ReplicaPlan(
                        replica_index=0,
                        container_name="trainfactory-vllm-model-replica",
                        gpu_ids=(0,),
                        port=11000,
                        endpoint="http://replica:11000",
                    ),
                ),
                required_gpus_per_replica=1,
                reservation_owner="reservation-owner",
            ),
            auto_start=False,
            defer_start=False,
        )
    if writer == "update":
        with Session(engine) as session:
            session.add(
                DeploymentDB(
                    deployment_id="deployment-update",
                    model_id="model-1",
                    deployment_name="update",
                    xinference_endpoint="http://xinference:9997",
                    config={},
                    user_id=user_id,
                )
            )
            session.commit()
        return service.update_deployment_config(
            "deployment-update",
            config,
            user_id=user_id,
        )
    raise AssertionError(f"unknown writer: {writer}")


@pytest.mark.parametrize(
    "writer",
    ["create", "bind", "replica_plan", "update"],
)
@pytest.mark.parametrize(
    "config_id",
    ["api-missing", "api-foreign"],
)
def test_deployment_external_api_writer_rejects_missing_or_foreign_config(
    external_api_writer_db,
    writer: str,
    config_id: str,
) -> None:
    service, engine, model = external_api_writer_db

    with pytest.raises(ValueError, match="External API config"):
        _invoke_external_api_writer(
            service,
            engine,
            model,
            writer=writer,
            config_id=config_id,
            user_id="user-1",
        )

    with Session(engine) as session:
        deployments = list(session.exec(select(DeploymentDB)).all())
        if writer == "update":
            assert len(deployments) == 1
            assert deployments[0].external_api_config_id is None
            assert deployments[0].config == {}
        else:
            assert deployments == []


@pytest.mark.parametrize("writer", ["create", "replica_plan"])
def test_deployment_writer_rejects_conflicting_external_api_references(
    external_api_writer_db,
    writer: str,
) -> None:
    service, engine, model = external_api_writer_db
    if writer == "create":
        def invoke():
            return service.create_deployment(
                model_id="model-1",
                xinference_endpoint="http://xinference:9997",
                config={"external_api_config_id": "api-b"},
                external_api_config_id="api-a",
                user_id="user-1",
            )

    else:
        def invoke():
            return service._persist_replica_plan(
                deployment_id="deployment-conflict",
                model=model,
                deployment_name="conflict",
                replica_count=1,
                gpu_memory_utilization=0.8,
                config={
                    "external_api_config_id": "api-b",
                    "launch_config": {"framework": "vllm", "gpu_pool": [0]},
                },
                user_id="user-1",
                inference_framework="vllm",
                enable_lora=False,
                max_loras=4,
                max_lora_rank=64,
                external_api_config_id="api-a",
                plan=DeploymentPlan(
                    replicas=(
                        ReplicaPlan(
                            replica_index=0,
                            container_name="trainfactory-vllm-conflict",
                            gpu_ids=(0,),
                            port=11001,
                            endpoint="http://replica:11001",
                        ),
                    ),
                    required_gpus_per_replica=1,
                    reservation_owner="reservation-owner",
                ),
                auto_start=False,
                defer_start=False,
            )

    with pytest.raises(ValueError, match="external_api_config_id mismatch"):
        invoke()

    with Session(engine) as session:
        assert session.exec(select(DeploymentDB)).all() == []


def test_external_api_rows_are_locked_in_sql_config_id_order(
    external_api_writer_db,
) -> None:
    service, engine, _model = external_api_writer_db
    statements: list[str] = []

    def record(_connection, _cursor, statement, _params, _context, _many):
        statements.append(" ".join(statement.lower().split()))

    sa.event.listen(engine, "before_cursor_execute", record)
    try:
        with Session(engine) as session:
            locked = service._lock_external_api_configs(
                session,
                ("api-b", "api-a", "api-b"),
                expected_user_id="user-1",
            )
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)

    assert [config.config_id for config in locked] == ["api-a", "api-b"]
    api_lock_sql = next(
        statement
        for statement in statements
        if " from external_api_configs " in f" {statement} "
    )
    assert "order by external_api_configs.config_id" in api_lock_sql


@pytest.mark.parametrize(
    "writer",
    ["create", "bind", "replica_plan", "update"],
)
def test_deployment_writers_lock_api_then_model_then_deployment(
    external_api_writer_db,
    writer: str,
) -> None:
    service, engine, model = external_api_writer_db
    statements: list[str] = []

    def record(_connection, _cursor, statement, _params, _context, _many):
        statements.append(" ".join(statement.lower().split()))

    sa.event.listen(engine, "before_cursor_execute", record)
    try:
        _invoke_external_api_writer(
            service,
            engine,
            model,
            writer=writer,
            config_id="api-a",
            user_id="user-1",
        )
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)

    api_index = next(
        index
        for index, statement in enumerate(statements)
        if " from external_api_configs " in f" {statement} "
        and "order by external_api_configs.config_id" in statement
    )
    gate_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("update model_artifact_membership_gate")
    )
    model_index = next(
        index
        for index, statement in enumerate(statements)
        if index > gate_index
        and " from model_registry " in f" {statement} "
    )
    deployment_indexes = [
        index
        for index, statement in enumerate(statements)
        if statement.startswith("insert into deployments")
        or " from deployments " in f" {statement} "
    ]
    deployment_write_index = next(
        index for index in deployment_indexes if index > model_index
    )
    assert api_index < gate_index < model_index < deployment_write_index


def test_create_deployment_builds_record_from_locked_model_snapshot(
    external_api_writer_db,
) -> None:
    service, engine, _model = external_api_writer_db
    with Session(engine) as session:
        model = session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == "model-1")
        ).one()
        model.model_name = "Trusted Model"
        session.add(model)
        session.commit()

    created = service.create_deployment(
        model_id="model-1",
        xinference_endpoint="http://xinference:9997",
        user_id="user-1",
    )

    assert created["deployment_name"] == "Trusted Model-deployment"
