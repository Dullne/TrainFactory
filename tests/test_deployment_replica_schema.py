from __future__ import annotations

import importlib
from datetime import datetime

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import inspect

from train_factory.storage.entities.deployment_replica_entity import (
    DeploymentReplicaDB,
)
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.entities.deployment_entity import DeploymentDB


MIGRATION_054 = importlib.import_module(
    "train_factory.storage.migrations.versions.054_add_deployment_replicas"
)
MIGRATION_055 = importlib.import_module(
    "train_factory.storage.migrations.versions.055_bind_configs_and_adapters_to_replicas"
)
MIGRATION_056 = importlib.import_module(
    "train_factory.storage.migrations.versions.056_bind_sync_targets_to_replicas"
)
MIGRATION_057 = importlib.import_module(
    "train_factory.storage.migrations.versions.057_add_replica_operation_fence"
)


def _unique_shapes(table: sa.Table) -> set[tuple[str, ...]]:
    return {
        tuple(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, sa.UniqueConstraint)
    }


def _index_shapes(table: sa.Table) -> set[tuple[str, ...]]:
    return {tuple(index.columns.keys()) for index in table.indexes}


def test_replica_entity_has_canonical_schema_shape() -> None:
    table = DeploymentReplicaDB.__table__

    assert table.name == "deployment_replicas"
    assert set(table.columns.keys()) == {
        "id",
        "replica_id",
        "deployment_id",
        "replica_index",
        "container_name",
        "endpoint",
        "port",
        "gpu_ids",
        "status",
        "health_status",
        "error_message",
        "created_at",
        "updated_at",
        "started_at",
        "stopped_at",
    }
    assert table.c.id.primary_key is True
    assert table.c.replica_id.nullable is False
    assert table.c.deployment_id.nullable is False
    assert table.c.replica_index.nullable is False
    assert table.c.container_name.nullable is False
    assert table.c.endpoint.nullable is False
    assert table.c.port.nullable is False
    assert table.c.gpu_ids.nullable is False
    assert table.c.status.nullable is False
    assert table.c.health_status.nullable is False
    assert table.c.error_message.nullable is True

    assert {
        ("replica_id",),
        ("deployment_id", "replica_index"),
        ("container_name",),
        ("port",),
    }.issubset(_unique_shapes(table))
    assert {
        ("deployment_id",),
        ("deployment_id", "status"),
    }.issubset(_index_shapes(table))

    foreign_keys = list(table.c.deployment_id.foreign_keys)
    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "deployments.deployment_id"
    assert foreign_keys[0].ondelete == "CASCADE"


def test_deployment_entity_has_durable_replica_operation_fence() -> None:
    table = DeploymentDB.__table__

    assert table.c.replica_operation_token.nullable is True
    assert table.c.replica_operation_token.type.length == 36
    assert table.c.replica_operation_kind.nullable is True
    assert table.c.replica_operation_kind.type.length == 16
    assert table.c.replica_operation_replica_id.nullable is True
    assert table.c.replica_operation_replica_id.type.length == 36
    assert table.c.replica_operation_generation.nullable is False
    assert table.c.replica_operation_started_at.nullable is True
    assert table.c.replica_operation_heartbeat_at.nullable is True


def test_replica_to_dict_has_stable_public_types() -> None:
    now = datetime(2026, 8, 22, 12, 0, 0)
    replica = DeploymentReplicaDB(
        replica_id="11111111-1111-1111-1111-111111111111",
        deployment_id="22222222-2222-2222-2222-222222222222",
        replica_index=0,
        container_name="tf-deploy-22222222-r0",
        endpoint="http://127.0.0.1:18080",
        port=18080,
        gpu_ids=[3, 1],
        status="running",
        health_status="HEALTHY",
        error_message=None,
        created_at=now,
        updated_at=now,
        started_at=now,
    )

    assert replica.to_dict() == {
        "replica_id": "11111111-1111-1111-1111-111111111111",
        "deployment_id": "22222222-2222-2222-2222-222222222222",
        "replica_index": 0,
        "container_name": "tf-deploy-22222222-r0",
        "endpoint": "http://127.0.0.1:18080",
        "port": 18080,
        "gpu_ids": [3, 1],
        "status": "running",
        "health_status": "HEALTHY",
        "error_message": None,
        "created_at": "2026-08-22T12:00:00",
        "updated_at": "2026-08-22T12:00:00",
        "started_at": "2026-08-22T12:00:00",
        "stopped_at": None,
    }


def test_downstream_entities_have_nullable_indexed_replica_bindings() -> None:
    for table in (ModelConfigDB.__table__, LoadedAdapterDB.__table__):
        column = table.c.deployment_replica_id
        assert column.nullable is True
        assert column.type.length == 36
        assert ("deployment_replica_id",) in _index_shapes(table)
        foreign_keys = list(column.foreign_keys)
        assert len(foreign_keys) == 1
        assert foreign_keys[0].target_fullname == "deployment_replicas.replica_id"
        assert foreign_keys[0].ondelete == "SET NULL"


def test_sync_entities_have_nullable_indexed_replica_bindings() -> None:
    for table in (
        ExternalSyncTaskDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
    ):
        column = table.c.base_deployment_replica_id
        assert column.nullable is True
        assert column.type.length == 36
        assert ("base_deployment_replica_id",) in _index_shapes(table)
        foreign_keys = list(column.foreign_keys)
        assert len(foreign_keys) == 1
        assert foreign_keys[0].target_fullname == "deployment_replicas.replica_id"
        assert foreign_keys[0].ondelete == "SET NULL"


def _create_legacy_tables(connection: sa.Connection) -> None:
    connection.execute(
        sa.text(
            """
            CREATE TABLE deployments (
                id INTEGER PRIMARY KEY,
                deployment_id VARCHAR(36) NOT NULL UNIQUE,
                deploy_mode VARCHAR(32) NOT NULL,
                inference_framework VARCHAR(32) NOT NULL,
                container_name VARCHAR(255),
                xinference_endpoint VARCHAR(512) NOT NULL,
                port INTEGER,
                gpu_id INTEGER,
                config JSON,
                status VARCHAR(50) NOT NULL,
                health_status VARCHAR(32) NOT NULL,
                error_message TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                started_at DATETIME,
                stopped_at DATETIME
            )
            """
        )
    )
    connection.execute(
        sa.text(
            """
            CREATE TABLE model_configs (
                id INTEGER PRIMARY KEY,
                config_id VARCHAR(36) NOT NULL UNIQUE
            )
            """
        )
    )
    connection.execute(
        sa.text(
            """
            CREATE TABLE loaded_adapters (
                id INTEGER PRIMARY KEY,
                adapter_id VARCHAR(36) NOT NULL UNIQUE
            )
            """
        )
    )


def _insert_deployment(
    connection: sa.Connection,
    *,
    deployment_id: str,
    deploy_mode: str,
    framework: str,
    container_name: str | None,
    endpoint: str,
    port: int | None,
    gpu_id: int | None,
) -> None:
    connection.execute(
        sa.text(
            """
            INSERT INTO deployments (
                deployment_id, deploy_mode, inference_framework,
                container_name, xinference_endpoint, port, gpu_id,
                status, health_status, created_at, updated_at
            ) VALUES (
                :deployment_id, :deploy_mode, :framework,
                :container_name, :endpoint, :port, :gpu_id,
                'running', 'HEALTHY', :created_at, :updated_at
            )
            """
        ),
        {
            "deployment_id": deployment_id,
            "deploy_mode": deploy_mode,
            "framework": framework,
            "container_name": container_name,
            "endpoint": endpoint,
            "port": port,
            "gpu_id": gpu_id,
            "created_at": datetime(2026, 8, 22, 12, 0, 0),
            "updated_at": datetime(2026, 8, 22, 12, 0, 0),
        },
    )


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def test_054_creates_empty_replica_table_without_adopting_legacy_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _create_legacy_tables(connection)
        _insert_deployment(
            connection,
            deployment_id="vllm-deployment",
            deploy_mode="container",
            framework="vllm",
            container_name="vllm-container",
            endpoint="http://127.0.0.1:18080",
            port=18080,
            gpu_id=2,
        )
        _insert_deployment(
            connection,
            deployment_id="xinference-container",
            deploy_mode="container",
            framework="xinference",
            container_name="xf-container",
            endpoint="http://127.0.0.1:18081",
            port=18081,
            gpu_id=3,
        )
        _insert_deployment(
            connection,
            deployment_id="xinference-shared",
            deploy_mode="shared",
            framework="xinference",
            container_name="xinference",
            endpoint="http://xinference:9997",
            port=None,
            gpu_id=None,
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()

        rows = connection.execute(
            sa.text(
                "SELECT deployment_id, replica_index, container_name, endpoint, "
                "port, gpu_ids FROM deployment_replicas ORDER BY port"
            )
        ).mappings().all()
        assert rows == []
        replica_foreign_keys = inspect(connection).get_foreign_keys(
            "deployment_replicas"
        )
        assert len(replica_foreign_keys) == 1
        assert replica_foreign_keys[0]["options"]["ondelete"].upper() == "CASCADE"

        MIGRATION_054.upgrade()
        count = connection.scalar(sa.text("SELECT COUNT(*) FROM deployment_replicas"))
        assert count == 0

        deployments = sa.Table("deployments", sa.MetaData(), autoload_with=connection)
        configs = {
            row["deployment_id"]: row["config"]
            for row in connection.execute(
                sa.select(deployments.c.deployment_id, deployments.c.config)
            ).mappings()
        }
        assert configs["vllm-deployment"] is None
        assert configs["xinference-container"] is None
        assert configs["xinference-shared"] is None


def test_054_reentry_preserves_canonical_rows_created_after_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _create_legacy_tables(connection)
        _insert_deployment(
            connection,
            deployment_id="vllm-deployment",
            deploy_mode="container",
            framework="vllm",
            container_name="vllm-container",
            endpoint="http://127.0.0.1:18080",
            port=18080,
            gpu_id=2,
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))
        MIGRATION_054.upgrade()
        connection.execute(
            sa.text(
                "INSERT INTO deployment_replicas ("
                "replica_id, deployment_id, replica_index, container_name, "
                "endpoint, port, gpu_ids, status, health_status, created_at, updated_at"
                ") VALUES ("
                "'replica-new', 'vllm-deployment', 0, 'managed-new', "
                "'http://127.0.0.1:18082', 18082, '[7]', 'running', 'HEALTHY', "
                ":created_at, :updated_at)"
            ),
            {
                "created_at": datetime(2026, 8, 23, 12, 0, 0),
                "updated_at": datetime(2026, 8, 23, 12, 0, 0),
            },
        )

        MIGRATION_054.upgrade()
        row = connection.execute(
            sa.text(
                "SELECT replica_id, container_name, gpu_ids "
                "FROM deployment_replicas"
            )
        ).mappings().one()
        assert row["replica_id"] == "replica-new"
        assert row["container_name"] == "managed-new"
        assert MIGRATION_054.decode_gpu_ids(row["gpu_ids"]) == [7]


def test_054_does_not_adopt_invalid_legacy_container_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_tables(connection)
        _insert_deployment(
            connection,
            deployment_id="broken-deployment",
            deploy_mode="container",
            framework="vllm",
            container_name="broken-container",
            endpoint="http://127.0.0.1:18080",
            port=None,
            gpu_id=0,
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()

        assert "deployment_replicas" in inspect(connection).get_table_names()
        assert connection.scalar(
            sa.text("SELECT COUNT(*) FROM deployment_replicas")
        ) == 0


def test_055_adds_nullable_replica_bindings_with_indexes_and_foreign_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _create_legacy_tables(connection)
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))
        MIGRATION_054.upgrade()
        monkeypatch.setattr(MIGRATION_055, "op", _operations(connection))

        MIGRATION_055.upgrade()

        reflected = inspect(connection)
        for table_name in ("model_configs", "loaded_adapters"):
            columns = {column["name"]: column for column in reflected.get_columns(table_name)}
            assert columns["deployment_replica_id"]["nullable"] is True
            indexes = {
                index["name"]: tuple(index["column_names"])
                for index in reflected.get_indexes(table_name)
            }
            assert (
                indexes[f"idx_{table_name}_deployment_replica_id"]
                == ("deployment_replica_id",)
            )
            foreign_keys = reflected.get_foreign_keys(table_name)
            replica_fk = [
                foreign_key
                for foreign_key in foreign_keys
                if foreign_key["constrained_columns"] == ["deployment_replica_id"]
            ]
            assert len(replica_fk) == 1
            assert replica_fk[0]["referred_table"] == "deployment_replicas"
            assert replica_fk[0]["referred_columns"] == ["replica_id"]
            assert replica_fk[0]["options"]["ondelete"].upper() == "SET NULL"


def test_migration_chain_is_linear() -> None:
    assert MIGRATION_054.revision == "054_add_deployment_replicas"
    assert MIGRATION_054.down_revision == "053_validate_lifecycle_schema"
    assert MIGRATION_055.revision == "055_bind_configs_and_adapters_to_replicas"
    assert MIGRATION_055.down_revision == MIGRATION_054.revision
    assert MIGRATION_056.revision == "056_bind_sync_targets_to_replicas"
    assert MIGRATION_056.down_revision == MIGRATION_055.revision
    assert MIGRATION_057.revision == "057_add_replica_operation_fence"
    assert MIGRATION_057.down_revision == MIGRATION_056.revision


def test_057_adds_durable_replica_operation_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE deployments ("
                "id INTEGER PRIMARY KEY, deployment_id VARCHAR(36) NOT NULL UNIQUE)"
            )
        )
        monkeypatch.setattr(MIGRATION_057, "op", _operations(connection))

        MIGRATION_057.upgrade()
        MIGRATION_057.upgrade()

        connection.execute(
            sa.text(
                "INSERT INTO deployments (id, deployment_id) "
                "VALUES (1, 'deployment-1')"
            )
        )

        columns = {
            item["name"]: item for item in inspect(connection).get_columns("deployments")
        }
        assert columns["replica_operation_token"]["nullable"] is True
        assert columns["replica_operation_kind"]["nullable"] is True
        assert columns["replica_operation_replica_id"]["nullable"] is True
        assert columns["replica_operation_generation"]["nullable"] is False
        assert columns["replica_operation_started_at"]["nullable"] is True
        assert columns["replica_operation_heartbeat_at"]["nullable"] is True
        generation = connection.scalar(
            sa.text("SELECT replica_operation_generation FROM deployments LIMIT 1")
        )
        assert generation == 0
