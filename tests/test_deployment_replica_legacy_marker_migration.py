from __future__ import annotations

import importlib
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import inspect

MIGRATION_054 = importlib.import_module("train_factory.storage.migrations.versions.054_add_deployment_replicas")

BACKUP_KEY = "_operator_054_legacy_replica_schema_marker_backup"
BACKUP_PAYLOAD = {
    "migration_revision": "054_add_deployment_replicas",
    "replica_schema_version": 1,
}


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _decode_config(value):
    return json.loads(value) if isinstance(value, str) else value


class _InspectorOverride:
    def __init__(
        self,
        delegate,
        *,
        dialect_name: str | None = None,
        id_autoincrement=...,
        referred_schema=...,
        default_schema_name=...,
        index_metadata: dict | None = None,
    ) -> None:
        self._delegate = delegate
        self.bind = SimpleNamespace(
            dialect=SimpleNamespace(
                name=dialect_name or delegate.bind.dialect.name,
            )
        )
        self.default_schema_name = delegate.default_schema_name if default_schema_name is ... else default_schema_name
        self._id_autoincrement = id_autoincrement
        self._referred_schema = referred_schema
        self._index_metadata = index_metadata

    def get_columns(self, table_name: str):
        columns = [dict(column) for column in self._delegate.get_columns(table_name)]
        if table_name == MIGRATION_054.TABLE_NAME and self._id_autoincrement is not ...:
            id_column = next(column for column in columns if column["name"] == "id")
            id_column["autoincrement"] = self._id_autoincrement
        return columns

    def get_foreign_keys(self, table_name: str):
        foreign_keys = [dict(foreign_key) for foreign_key in self._delegate.get_foreign_keys(table_name)]
        if table_name == MIGRATION_054.TABLE_NAME and self._referred_schema is not ...:
            for foreign_key in foreign_keys:
                foreign_key["referred_schema"] = self._referred_schema
        return foreign_keys

    def get_indexes(self, table_name: str):
        indexes = [dict(index) for index in self._delegate.get_indexes(table_name)]
        if table_name == MIGRATION_054.TABLE_NAME and self._index_metadata is not None:
            target = next(index for index in indexes if index["name"] == "idx_deployment_replica_deployment_status")
            target.update(self._index_metadata)
        return indexes

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def _create_deployments(connection: sa.Connection) -> None:
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


def _create_deployments_without_config(connection: sa.Connection) -> None:
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


def _insert_deployment(
    connection: sa.Connection,
    *,
    deployment_id: str,
    deploy_mode: str,
    port: int | None,
    gpu_id: int | None,
    config: dict,
    framework: str = "vllm",
) -> None:
    now = datetime(2026, 8, 23, 12, 0, 0)
    connection.execute(
        sa.text(
            """
            INSERT INTO deployments (
                deployment_id, deploy_mode, inference_framework,
                container_name, xinference_endpoint, port, gpu_id, config,
                status, health_status, created_at, updated_at
            ) VALUES (
                :deployment_id, :deploy_mode, :framework,
                :container_name, :endpoint, :port, :gpu_id, :config,
                'running', 'HEALTHY', :now, :now
            )
            """
        ),
        {
            "deployment_id": deployment_id,
            "deploy_mode": deploy_mode,
            "framework": framework,
            "container_name": f"container-{deployment_id}",
            "endpoint": f"http://127.0.0.1:{port or 18099}",
            "port": port,
            "gpu_id": gpu_id,
            "config": json.dumps(config),
            "now": now,
        },
    )


def test_054_preserves_legacy_parent_config_without_claiming_container_ownership(
    monkeypatch,
):
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="container-deployment",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"preserve": "value"},
        )
        _insert_deployment(
            connection,
            deployment_id="shared-deployment",
            deploy_mode="shared",
            port=None,
            gpu_id=None,
            config={"shared": True},
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()
        MIGRATION_054.upgrade()

        configs = {
            row["deployment_id"]: _decode_config(row["config"])
            for row in connection.execute(sa.text("SELECT deployment_id, config FROM deployments")).mappings()
        }
        assert configs["container-deployment"] == {"preserve": "value"}
        assert configs["shared-deployment"] == {"shared": True}
        assert connection.scalar(sa.text("SELECT COUNT(*) FROM deployment_replicas")) == 0

        MIGRATION_054.downgrade()

        configs = {
            row["deployment_id"]: _decode_config(row["config"])
            for row in connection.execute(sa.text("SELECT deployment_id, config FROM deployments")).mappings()
        }
        assert configs["container-deployment"] == {"preserve": "value"}
        assert configs["shared-deployment"] == {"shared": True}


def test_054_keeps_all_legacy_containers_outside_replica_lifecycle(
    monkeypatch,
):
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="vllm-container",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"framework": "vllm"},
        )
        _insert_deployment(
            connection,
            deployment_id="xinference-container",
            deploy_mode="container",
            port=18081,
            gpu_id=1,
            config={"framework": "xinference"},
            framework="xinference",
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()

        replica_parents = list(
            connection.scalars(sa.text("SELECT deployment_id FROM deployment_replicas " "ORDER BY deployment_id"))
        )
        configs = {
            row["deployment_id"]: _decode_config(row["config"])
            for row in connection.execute(sa.text("SELECT deployment_id, config FROM deployments")).mappings()
        }
        assert replica_parents == []
        assert configs["vllm-container"] == {"framework": "vllm"}
        assert configs["xinference-container"] == {"framework": "xinference"}


def test_054_refuses_schema_without_parent_config_column(monkeypatch):
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments_without_config(connection)
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        with pytest.raises(
            RuntimeError,
            match="Required deployments columns are missing: config",
        ):
            MIGRATION_054.upgrade()


def test_054_preserves_unknown_legacy_marker_without_adopting_the_container(monkeypatch):
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="future-schema",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"replica_schema_version": 2},
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()

        assert "deployment_replicas" in inspect(connection).get_table_names()
        assert connection.scalar(sa.text("SELECT COUNT(*) FROM deployment_replicas")) == 0
        config = connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'future-schema'"))
        assert _decode_config(config) == {"replica_schema_version": 2}


def test_054_backs_up_only_preexisting_markers_that_would_claim_replica_lifecycle(
    monkeypatch,
):
    engine = sa.create_engine("sqlite://")
    cases = {
        "vllm-collision": ("vllm", {"replica_schema_version": 1, "keep": "v"}),
        "sglang-collision": (
            "sglang",
            {
                "replica_schema_version": 1,
                "keep": "s",
                "nested": {"replica_schema_version": 1},
            },
        ),
        "xinference-marker": ("xinference", {"replica_schema_version": 1}),
        "boolean-marker": ("vllm", {"replica_schema_version": True}),
        "string-marker": ("vllm", {"replica_schema_version": "1"}),
        "future-marker": ("vllm", {"replica_schema_version": 2}),
        "nested-only": (
            "vllm",
            {"nested": {"replica_schema_version": 1}, "keep": "nested"},
        ),
    }
    with engine.begin() as connection:
        _create_deployments(connection)
        for offset, (deployment_id, (framework, config)) in enumerate(cases.items()):
            _insert_deployment(
                connection,
                deployment_id=deployment_id,
                deploy_mode="container",
                port=18080 + offset,
                gpu_id=offset,
                config=config,
                framework=framework,
            )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()

        configs = {
            row["deployment_id"]: _decode_config(row["config"])
            for row in connection.execute(sa.text("SELECT deployment_id, config FROM deployments")).mappings()
        }
        assert configs["vllm-collision"] == {
            BACKUP_KEY: BACKUP_PAYLOAD,
            "keep": "v",
        }
        assert configs["sglang-collision"] == {
            BACKUP_KEY: BACKUP_PAYLOAD,
            "keep": "s",
            "nested": {"replica_schema_version": 1},
        }
        for deployment_id in (
            "xinference-marker",
            "boolean-marker",
            "string-marker",
            "future-marker",
            "nested-only",
        ):
            assert configs[deployment_id] == cases[deployment_id][1]


def test_054_second_run_preserves_post_migration_replica_marker(monkeypatch):
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="canonical-after-upgrade",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"replica_schema_version": 1, "keep": "collision"},
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()
        connection.execute(
            sa.text("UPDATE deployments SET config = :config " "WHERE deployment_id = 'canonical-after-upgrade'"),
            {"config": json.dumps({"replica_schema_version": 1, "keep": "canonical"})},
        )

        MIGRATION_054.upgrade()

        config = connection.scalar(
            sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'canonical-after-upgrade'")
        )
        assert _decode_config(config) == {
            "replica_schema_version": 1,
            "keep": "canonical",
        }


def test_054_marker_backup_is_preserved_when_index_creation_is_retried(
    monkeypatch,
):
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="retry-collision",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"replica_schema_version": 1, "keep": "value"},
        )
        delegate = _operations(connection)

        class FailSecondIndexOnce:
            failed = False

            def create_index(self, name, *args, **kwargs):
                if name == "idx_deployment_replica_deployment_status" and not self.failed:
                    self.failed = True
                    raise RuntimeError("fixed second index failure")
                return delegate.create_index(name, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(delegate, name)

        monkeypatch.setattr(MIGRATION_054, "op", FailSecondIndexOnce())

        with pytest.raises(RuntimeError, match="fixed second index failure"):
            MIGRATION_054.upgrade()
        reflected = inspect(connection)
        assert "deployment_replicas" in reflected.get_table_names()
        indexes = {
            index["name"]: tuple(index.get("column_names") or ())
            for index in reflected.get_indexes("deployment_replicas")
        }
        assert indexes == {
            "idx_deployment_replica_deployment_id": ("deployment_id",),
        }
        config = connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'retry-collision'"))
        assert _decode_config(config) == {BACKUP_KEY: BACKUP_PAYLOAD, "keep": "value"}

        MIGRATION_054.upgrade()

        indexes = {
            index["name"]: tuple(index.get("column_names") or ())
            for index in inspect(connection).get_indexes("deployment_replicas")
        }
        assert indexes == {
            "idx_deployment_replica_deployment_id": ("deployment_id",),
            "idx_deployment_replica_deployment_status": (
                "deployment_id",
                "status",
            ),
        }
        config = connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'retry-collision'"))
        assert _decode_config(config) == {
            BACKUP_KEY: BACKUP_PAYLOAD,
            "keep": "value",
        }


def test_054_create_table_failure_retry_preserves_owned_backup(monkeypatch) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.connect() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="create-retry-collision",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"replica_schema_version": 1, "keep": "value"},
        )
        connection.commit()
        delegate = _operations(connection)

        class FailReplicaTableOnce:
            failed = False

            def create_table(self, name, *args, **kwargs):
                if name == MIGRATION_054.TABLE_NAME and not self.failed:
                    self.failed = True
                    raise RuntimeError("fixed replica table creation failure")
                return delegate.create_table(name, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(delegate, name)

        monkeypatch.setattr(MIGRATION_054, "op", FailReplicaTableOnce())

        with pytest.raises(RuntimeError, match="fixed replica table creation failure"):
            MIGRATION_054.upgrade()
        connection.commit()

        assert MIGRATION_054.TABLE_NAME not in inspect(connection).get_table_names()
        config = connection.scalar(
            sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'create-retry-collision'")
        )
        assert _decode_config(config) == {
            BACKUP_KEY: BACKUP_PAYLOAD,
            "keep": "value",
        }

        monkeypatch.setattr(MIGRATION_054, "op", delegate)
        MIGRATION_054.upgrade()
        connection.commit()

        assert MIGRATION_054.TABLE_NAME in inspect(connection).get_table_names()
        config = connection.scalar(
            sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'create-retry-collision'")
        )
        assert _decode_config(config) == {
            BACKUP_KEY: BACKUP_PAYLOAD,
            "keep": "value",
        }


def test_054_reentry_rejects_same_name_index_with_wrong_shape(monkeypatch):
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments(connection)
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))
        MIGRATION_054.upgrade()
        connection.execute(sa.text("DROP INDEX idx_deployment_replica_deployment_status"))
        connection.execute(
            sa.text("CREATE INDEX idx_deployment_replica_deployment_status " "ON deployment_replicas (status)")
        )

        with pytest.raises(
            RuntimeError,
            match="Existing deployment_replicas indexes are invalid",
        ):
            MIGRATION_054.upgrade()

        indexes = {
            index["name"]: tuple(index.get("column_names") or ())
            for index in inspect(connection).get_indexes("deployment_replicas")
        }
        assert indexes["idx_deployment_replica_deployment_status"] == ("status",)


def _create_empty_replica_schema(connection: sa.Connection, monkeypatch) -> None:
    _create_deployments(connection)
    monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))
    MIGRATION_054.upgrade()


def test_054_sqlite_integer_primary_key_reflection_is_accepted(monkeypatch) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_empty_replica_schema(connection, monkeypatch)

        id_column = next(
            column for column in inspect(connection).get_columns("deployment_replicas") if column["name"] == "id"
        )
        assert id_column["primary_key"] == 1
        MIGRATION_054._validate_replica_table_core(connection)


@pytest.mark.parametrize("autoincrement", [False, None, "auto"])
def test_054_mysql_reentry_rejects_non_autoincrement_primary_key(
    monkeypatch,
    autoincrement,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_empty_replica_schema(connection, monkeypatch)
        reflected = _InspectorOverride(
            inspect(connection),
            dialect_name="mysql",
            id_autoincrement=autoincrement,
        )
        monkeypatch.setattr(MIGRATION_054, "inspect", lambda _bind: reflected)

        with pytest.raises(
            RuntimeError,
            match="deployment_replicas.id autoincrement is invalid",
        ):
            MIGRATION_054._validate_replica_table_core(connection)


def test_054_mysql_reentry_accepts_autoincrement_primary_key(monkeypatch) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_empty_replica_schema(connection, monkeypatch)
        reflected = _InspectorOverride(
            inspect(connection),
            dialect_name="mysql",
            id_autoincrement=True,
        )
        monkeypatch.setattr(MIGRATION_054, "inspect", lambda _bind: reflected)

        MIGRATION_054._validate_replica_table_core(connection)


@pytest.mark.parametrize("referred_schema", [None, "train_factory"])
def test_054_reentry_accepts_local_foreign_key_schema(
    monkeypatch,
    referred_schema,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_empty_replica_schema(connection, monkeypatch)
        reflected = _InspectorOverride(
            inspect(connection),
            referred_schema=referred_schema,
            default_schema_name="train_factory",
        )
        monkeypatch.setattr(MIGRATION_054, "inspect", lambda _bind: reflected)

        MIGRATION_054._validate_replica_table_core(connection)


def test_054_reentry_rejects_cross_database_foreign_key(monkeypatch) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_empty_replica_schema(connection, monkeypatch)
        reflected = _InspectorOverride(
            inspect(connection),
            referred_schema="other_database",
            default_schema_name="train_factory",
        )
        monkeypatch.setattr(MIGRATION_054, "inspect", lambda _bind: reflected)

        with pytest.raises(
            RuntimeError,
            match="Existing deployment_replicas foreign key is invalid",
        ):
            MIGRATION_054._validate_replica_table_core(connection)


@pytest.mark.parametrize(
    "index_metadata",
    [
        {"dialect_options": {"sqlite_where": sa.text("status = 'running'")}},
        {"type": "FULLTEXT"},
        {"prefix": "FULLTEXT"},
        {"partial": True},
        {"mysql_prefix": "FULLTEXT"},
    ],
    ids=["partial-where", "type", "prefix", "partial", "mysql-prefix"],
)
def test_054_reentry_rejects_semantic_expected_index_metadata(
    monkeypatch,
    index_metadata,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_empty_replica_schema(connection, monkeypatch)
        reflected = _InspectorOverride(
            inspect(connection),
            index_metadata=index_metadata,
        )
        monkeypatch.setattr(MIGRATION_054, "inspect", lambda _bind: reflected)

        with pytest.raises(
            RuntimeError,
            match="Existing deployment_replicas indexes are invalid",
        ):
            MIGRATION_054._validate_replica_table(connection)


@pytest.mark.parametrize(
    "index_metadata",
    [
        {},
        {"dialect_options": {}},
        {
            "dialect_options": {"mysql_prefix": None, "mysql_length": {}},
            "type": None,
            "prefix": None,
            "partial": False,
        },
    ],
    ids=["absent", "empty", "mysql-none"],
)
def test_054_reentry_accepts_empty_expected_index_metadata(
    monkeypatch,
    index_metadata,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_empty_replica_schema(connection, monkeypatch)
        reflected = _InspectorOverride(
            inspect(connection),
            index_metadata=index_metadata,
        )
        monkeypatch.setattr(MIGRATION_054, "inspect", lambda _bind: reflected)

        MIGRATION_054._validate_replica_table(connection)


def test_054_uses_stable_operator_owned_marker_backup_key() -> None:
    assert getattr(MIGRATION_054, "LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY", None) == BACKUP_KEY


@pytest.mark.parametrize(
    ("backup_value", "include_marker"),
    [
        ("caller-value", True),
        (BACKUP_PAYLOAD, True),
        ({**BACKUP_PAYLOAD, "caller_extra": True}, False),
    ],
    ids=["arbitrary-value", "exact-payload-with-marker", "extra-field"],
)
def test_054_upgrade_rejects_caller_owned_backup_key_without_mutation(
    monkeypatch,
    backup_value,
    include_marker,
) -> None:
    engine = sa.create_engine("sqlite://")
    original_config = {
        BACKUP_KEY: backup_value,
        "keep": "value",
    }
    if include_marker:
        original_config["replica_schema_version"] = 1
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="spoofed-backup",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config=original_config,
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        with pytest.raises(
            RuntimeError,
            match="legacy replica schema marker backup",
        ):
            MIGRATION_054.upgrade()

        assert "deployment_replicas" not in inspect(connection).get_table_names()
        config = connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'spoofed-backup'"))
        assert _decode_config(config) == original_config


def test_054_ignores_xinference_config_using_same_backup_key(monkeypatch) -> None:
    engine = sa.create_engine("sqlite://")
    original_config = {
        "replica_schema_version": 1,
        BACKUP_KEY: "xinference-owned-value",
        "keep": "xinference",
    }
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="xinference-reserved-key",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config=original_config,
            framework="xinference",
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()
        config = connection.scalar(
            sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'xinference-reserved-key'")
        )
        assert _decode_config(config) == original_config

        MIGRATION_054.downgrade()

        assert MIGRATION_054.TABLE_NAME not in inspect(connection).get_table_names()
        config = connection.scalar(
            sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'xinference-reserved-key'")
        )
        assert _decode_config(config) == original_config


def test_054_upgrade_downgrade_restores_legacy_marker_after_table_drop(
    monkeypatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="roundtrip-collision",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"replica_schema_version": 1, "keep": "value"},
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()

        upgraded_config = _decode_config(
            connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'roundtrip-collision'"))
        )
        assert upgraded_config == {BACKUP_KEY: BACKUP_PAYLOAD, "keep": "value"}

        delegate = _operations(connection)

        class AssertBackupBeforeDrop:
            def drop_table(self, name, *args, **kwargs):
                config = _decode_config(
                    connection.scalar(
                        sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'roundtrip-collision'")
                    )
                )
                assert config == {BACKUP_KEY: BACKUP_PAYLOAD, "keep": "value"}
                return delegate.drop_table(name, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(delegate, name)

        monkeypatch.setattr(MIGRATION_054, "op", AssertBackupBeforeDrop())
        MIGRATION_054.downgrade()

        assert "deployment_replicas" not in inspect(connection).get_table_names()
        restored_config = _decode_config(
            connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'roundtrip-collision'"))
        )
        assert restored_config == {
            "replica_schema_version": 1,
            "keep": "value",
        }


def test_054_downgrade_retries_marker_restore_after_table_was_dropped(
    monkeypatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="partial-downgrade",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"replica_schema_version": 1, "keep": "value"},
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))

        MIGRATION_054.upgrade()
        assert _decode_config(
            connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'partial-downgrade'"))
        ) == {BACKUP_KEY: BACKUP_PAYLOAD, "keep": "value"}

        apply_restores = MIGRATION_054._apply_preexisting_replica_schema_marker_restores

        def fail_restore_once(*args, **kwargs):
            raise RuntimeError("fixed restore write failure")

        monkeypatch.setattr(
            MIGRATION_054,
            "_apply_preexisting_replica_schema_marker_restores",
            fail_restore_once,
        )
        with pytest.raises(RuntimeError, match="fixed restore write failure"):
            MIGRATION_054.downgrade()

        assert MIGRATION_054.TABLE_NAME not in inspect(connection).get_table_names()
        assert _decode_config(
            connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'partial-downgrade'"))
        ) == {BACKUP_KEY: BACKUP_PAYLOAD, "keep": "value"}

        monkeypatch.setattr(
            MIGRATION_054,
            "_apply_preexisting_replica_schema_marker_restores",
            apply_restores,
        )

        MIGRATION_054.downgrade()

        config = _decode_config(
            connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'partial-downgrade'"))
        )
        assert config == {"replica_schema_version": 1, "keep": "value"}


def test_054_downgrade_rejects_tampered_backup_before_dropping_table(
    monkeypatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="tampered-restore",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"replica_schema_version": 1, "keep": "value"},
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))
        MIGRATION_054.upgrade()
        tampered_config = {
            BACKUP_KEY: {**BACKUP_PAYLOAD, "caller_extra": True},
            "keep": "value",
        }
        connection.execute(
            sa.text("UPDATE deployments SET config = :config " "WHERE deployment_id = 'tampered-restore'"),
            {"config": json.dumps(tampered_config)},
        )

        with pytest.raises(
            RuntimeError,
            match="legacy replica schema marker backup value is invalid",
        ):
            MIGRATION_054.downgrade()

        assert MIGRATION_054.TABLE_NAME in inspect(connection).get_table_names()
        config = connection.scalar(
            sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'tampered-restore'")
        )
        assert _decode_config(config) == tampered_config

        connection.execute(
            sa.text("UPDATE deployments SET config = :config " "WHERE deployment_id = 'tampered-restore'"),
            {"config": json.dumps({BACKUP_KEY: BACKUP_PAYLOAD, "keep": "value"})},
        )
        MIGRATION_054.downgrade()

        assert MIGRATION_054.TABLE_NAME not in inspect(connection).get_table_names()
        config = connection.scalar(
            sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'tampered-restore'")
        )
        assert _decode_config(config) == {"replica_schema_version": 1, "keep": "value"}


@pytest.mark.parametrize("restored_marker", [1, 2], ids=["matching", "conflicting"])
def test_054_downgrade_rejects_two_key_restore_before_dropping_table(
    monkeypatch,
    restored_marker,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_deployments(connection)
        _insert_deployment(
            connection,
            deployment_id="two-key-restore",
            deploy_mode="container",
            port=18080,
            gpu_id=0,
            config={"replica_schema_version": 1, "keep": "value"},
        )
        monkeypatch.setattr(MIGRATION_054, "op", _operations(connection))
        MIGRATION_054.upgrade()
        conflicting_config = {
            "replica_schema_version": restored_marker,
            BACKUP_KEY: BACKUP_PAYLOAD,
            "keep": "value",
        }
        connection.execute(
            sa.text("UPDATE deployments SET config = :config " "WHERE deployment_id = 'two-key-restore'"),
            {"config": json.dumps(conflicting_config)},
        )

        with pytest.raises(
            RuntimeError,
            match="legacy replica schema marker restore conflicts",
        ):
            MIGRATION_054.downgrade()

        assert MIGRATION_054.TABLE_NAME in inspect(connection).get_table_names()
        config = _decode_config(
            connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'two-key-restore'"))
        )
        assert config == conflicting_config

        connection.execute(
            sa.text("UPDATE deployments SET config = :config " "WHERE deployment_id = 'two-key-restore'"),
            {"config": json.dumps({BACKUP_KEY: BACKUP_PAYLOAD, "keep": "value"})},
        )
        MIGRATION_054.downgrade()

        assert MIGRATION_054.TABLE_NAME not in inspect(connection).get_table_names()
        config = _decode_config(
            connection.scalar(sa.text("SELECT config FROM deployments " "WHERE deployment_id = 'two-key-restore'"))
        )
        assert config == {"replica_schema_version": 1, "keep": "value"}
