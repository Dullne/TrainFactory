"""Add canonical lifecycle records for independent deployment replicas.

Revision ID: 054_add_deployment_replicas
Revises: 053_validate_lifecycle_schema
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "054_add_deployment_replicas"
down_revision = "053_validate_lifecycle_schema"
branch_labels = None
depends_on = None

TABLE_NAME = "deployment_replicas"
REPLICA_SCHEMA_VERSION_KEY = "replica_schema_version"
REPLICA_SCHEMA_VERSION = 1
LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY = "_operator_054_legacy_replica_schema_marker_backup"
LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_PAYLOAD = {
    "migration_revision": revision,
    "replica_schema_version": REPLICA_SCHEMA_VERSION,
}
REQUIRED_DEPLOYMENT_COLUMNS = {
    "deployment_id",
    "deploy_mode",
    "inference_framework",
    "container_name",
    "xinference_endpoint",
    "port",
    "gpu_id",
    "config",
    "status",
    "health_status",
    "error_message",
    "created_at",
    "updated_at",
    "started_at",
    "stopped_at",
}
REPLICA_COLUMNS = {
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
NULLABLE_REPLICA_COLUMNS = {
    "error_message",
    "started_at",
    "stopped_at",
}
EXPECTED_REPLICA_INDEXES = {
    "idx_deployment_replica_deployment_id": ("deployment_id",),
    "idx_deployment_replica_deployment_status": ("deployment_id", "status"),
}


def decode_gpu_ids(value):
    """Decode reflected JSON values consistently across SQLite and MySQL."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            raise RuntimeError("deployment replica gpu_ids is invalid") from None
    if (
        not isinstance(value, list)
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in value)
        or len(set(value)) != len(value)
    ):
        raise RuntimeError("deployment replica gpu_ids is invalid")
    return value


def _legacy_container_rows(bind) -> list[Mapping]:
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    if "deployments" not in tables:
        raise RuntimeError("Required deployments table is missing")
    columns = {column["name"] for column in inspector.get_columns("deployments")}
    missing = sorted(REQUIRED_DEPLOYMENT_COLUMNS - columns)
    if missing:
        raise RuntimeError("Required deployments columns are missing: " + ", ".join(missing))

    deployments = sa.Table("deployments", sa.MetaData(), autoload_with=bind)
    rows = list(
        bind.execute(
            sa.select(*(deployments.c[name] for name in sorted(REQUIRED_DEPLOYMENT_COLUMNS)))
            .where(deployments.c.deploy_mode == "container")
            .where(deployments.c.inference_framework.in_(("vllm", "sglang")))
            .order_by(deployments.c.deployment_id)
        ).mappings()
    )

    container_names: set[str] = set()
    ports: set[int] = set()
    for row in rows:
        deployment_id = row["deployment_id"]
        container_name = row["container_name"]
        endpoint = row["xinference_endpoint"]
        port = row["port"]
        gpu_id = row["gpu_id"]
        if (
            not isinstance(deployment_id, str)
            or not deployment_id
            or not isinstance(container_name, str)
            or not container_name
            or not isinstance(endpoint, str)
            or not endpoint
            or type(port) is not int
            or not 1024 <= port <= 65535
            or type(gpu_id) is not int
            or gpu_id < 0
        ):
            raise RuntimeError(
                "Refusing deployment replica migration: legacy container " "deployment identity is invalid"
            )
        if container_name in container_names or port in ports:
            raise RuntimeError(
                "Refusing deployment replica migration: legacy container " "deployment identity is not unique"
            )
        container_names.add(container_name)
        ports.add(port)
    return rows


def _validate_parent_schema(bind) -> None:
    """Validate the parent shape without adopting pre-migration containers.

    Legacy containers lack the immutable ownership labels and typed launch
    configuration required by the replica lifecycle. They must remain on the
    legacy lifecycle until an operator recreates them explicitly.
    """

    inspector = inspect(bind)
    if "deployments" not in set(inspector.get_table_names()):
        raise RuntimeError("Required deployments table is missing")
    columns = {column["name"] for column in inspector.get_columns("deployments")}
    missing = sorted(REQUIRED_DEPLOYMENT_COLUMNS - columns)
    if missing:
        raise RuntimeError("Required deployments columns are missing: " + ", ".join(missing))


def _clear_preexisting_replica_schema_marker_collisions(bind) -> None:
    """Back up legacy free-form values mistaken for lifecycle state."""

    deployments = sa.Table("deployments", sa.MetaData(), autoload_with=bind)
    rows = list(
        bind.execute(
            sa.select(
                deployments.c.deployment_id,
                deployments.c.inference_framework,
                deployments.c.config,
            ).order_by(deployments.c.deployment_id)
        )
    )
    pending_updates = []
    for deployment_id, framework, value in rows:
        if framework not in ("vllm", "sglang"):
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                continue
        if not isinstance(value, Mapping):
            continue
        config = dict(value)
        if LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY in config:
            if REPLICA_SCHEMA_VERSION_KEY in config:
                raise RuntimeError("legacy replica schema marker backup conflicts with schema marker")
            if config[LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY] != (LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_PAYLOAD):
                raise RuntimeError("legacy replica schema marker backup value is invalid")
            continue
        marker = config.get(REPLICA_SCHEMA_VERSION_KEY)
        if type(marker) is not int or marker != REPLICA_SCHEMA_VERSION:
            continue
        config[LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY] = dict(LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_PAYLOAD)
        config.pop(REPLICA_SCHEMA_VERSION_KEY)
        pending_updates.append((deployment_id, config))

    for deployment_id, config in pending_updates:
        bind.execute(sa.update(deployments).where(deployments.c.deployment_id == deployment_id).values(config=config))


def _create_replica_table() -> None:
    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("replica_id", sa.String(length=36), nullable=False),
        sa.Column("deployment_id", sa.String(length=36), nullable=False),
        sa.Column("replica_index", sa.Integer(), nullable=False),
        sa.Column("container_name", sa.String(length=255), nullable=False),
        sa.Column("endpoint", sa.String(length=512), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("gpu_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("health_status", sa.String(length=32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("stopped_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["deployment_id"],
            ["deployments.deployment_id"],
            name="fk_deployment_replicas_deployment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_deployment_replicas"),
        sa.UniqueConstraint("replica_id", name="uq_deployment_replica_id"),
        sa.UniqueConstraint(
            "deployment_id",
            "replica_index",
            name="uq_deployment_replica_index",
        ),
        sa.UniqueConstraint(
            "container_name",
            name="uq_deployment_replica_container_name",
        ),
        sa.UniqueConstraint("port", name="uq_deployment_replica_port"),
    )


def _column_type_matches(name: str, actual) -> bool:
    integer_columns = {"id", "replica_index", "port"}
    string_lengths = {
        "replica_id": 36,
        "deployment_id": 36,
        "container_name": 255,
        "endpoint": 512,
        "status": 50,
        "health_status": 32,
    }
    datetime_columns = {"created_at", "updated_at", "started_at", "stopped_at"}
    if name in integer_columns:
        return isinstance(actual, sa.Integer) and not isinstance(actual, sa.Boolean)
    if name in string_lengths:
        return (
            isinstance(actual, sa.String) and not isinstance(actual, sa.Text) and actual.length == string_lengths[name]
        )
    if name == "gpu_ids":
        return isinstance(actual, sa.JSON)
    if name == "error_message":
        return isinstance(actual, sa.Text)
    if name in datetime_columns:
        return isinstance(actual, sa.DateTime)
    return False


def _validate_replica_table_core(bind) -> None:
    inspector = inspect(bind)
    columns = {column["name"]: column for column in inspector.get_columns(TABLE_NAME)}
    if set(columns) != REPLICA_COLUMNS:
        raise RuntimeError("Existing deployment_replicas column set is invalid")
    for name, column in columns.items():
        if not _column_type_matches(name, column.get("type")):
            raise RuntimeError(f"Existing deployment_replicas.{name} type is invalid")
        expected_nullable = name in NULLABLE_REPLICA_COLUMNS
        if column.get("nullable") is not expected_nullable:
            raise RuntimeError(f"Existing deployment_replicas.{name} nullability is invalid")

    dialect_name = str(getattr(getattr(getattr(inspector, "bind", None), "dialect", None), "name", "")).lower()
    if dialect_name in {"mysql", "mariadb"} and columns["id"].get("autoincrement") is not True:
        raise RuntimeError("Existing deployment_replicas.id autoincrement is invalid")

    primary_key = inspector.get_pk_constraint(TABLE_NAME)
    if tuple(primary_key.get("constrained_columns") or ()) != ("id",):
        raise RuntimeError("Existing deployment_replicas primary key is invalid")

    unique_shapes = {
        tuple(constraint.get("column_names") or ()) for constraint in inspector.get_unique_constraints(TABLE_NAME)
    }
    expected_unique = {
        ("replica_id",),
        ("deployment_id", "replica_index"),
        ("container_name",),
        ("port",),
    }
    if unique_shapes != expected_unique:
        raise RuntimeError("Existing deployment_replicas uniqueness is invalid")

    foreign_keys = inspector.get_foreign_keys(TABLE_NAME)
    default_schema_name = getattr(inspector, "default_schema_name", None)
    allowed_referred_schemas = {None}
    if default_schema_name is not None:
        allowed_referred_schemas.add(default_schema_name)
    matching = [
        foreign_key
        for foreign_key in foreign_keys
        if tuple(foreign_key.get("constrained_columns") or ()) == ("deployment_id",)
        and foreign_key.get("referred_table") == "deployments"
        and foreign_key.get("referred_schema") in allowed_referred_schemas
        and tuple(foreign_key.get("referred_columns") or ()) == ("deployment_id",)
        and str((foreign_key.get("options") or {}).get("ondelete", "")).upper() == "CASCADE"
    ]
    if len(foreign_keys) != 1 or len(matching) != 1:
        raise RuntimeError("Existing deployment_replicas foreign key is invalid")


def _replica_indexes(bind) -> dict[str, Mapping]:
    indexes = {}
    for reflected in inspect(bind).get_indexes(TABLE_NAME):
        name = reflected.get("name")
        if not isinstance(name, str) or not name or name in indexes:
            raise RuntimeError("Existing deployment_replicas indexes are invalid")
        indexes[name] = dict(reflected)
    return indexes


def _has_index_semantics(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_index_semantics(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_has_index_semantics(item) for item in value)
    if isinstance(value, (int, float)) and value == 0:
        return False
    return True


def _validate_expected_replica_indexes(
    indexes: Mapping[str, Mapping],
    *,
    require_all: bool,
) -> None:
    for name, shape in EXPECTED_REPLICA_INDEXES.items():
        actual = indexes.get(name)
        if actual is None:
            if require_all:
                raise RuntimeError("Existing deployment_replicas indexes are invalid")
            continue
        if (
            tuple(actual.get("column_names") or ()) != shape
            or actual.get("unique") not in (False, 0)
            or any(
                _has_index_semantics(value)
                for key, value in actual.items()
                if key not in {"name", "column_names", "unique"}
            )
        ):
            raise RuntimeError("Existing deployment_replicas indexes are invalid")


def _ensure_replica_indexes(bind) -> None:
    indexes = _replica_indexes(bind)
    _validate_expected_replica_indexes(indexes, require_all=False)
    for name, shape in EXPECTED_REPLICA_INDEXES.items():
        if name not in indexes:
            op.create_index(name, TABLE_NAME, list(shape), unique=False)
    _validate_expected_replica_indexes(
        _replica_indexes(bind),
        require_all=True,
    )


def _validate_replica_table(bind) -> None:
    _validate_replica_table_core(bind)
    _validate_expected_replica_indexes(
        _replica_indexes(bind),
        require_all=True,
    )


def _backfill_replica_zero(bind, rows: list[Mapping]) -> None:
    table = sa.Table(TABLE_NAME, sa.MetaData(), autoload_with=bind)
    existing = {
        row["deployment_id"]: row for row in bind.execute(sa.select(table).where(table.c.replica_index == 0)).mappings()
    }
    for source in rows:
        expected = {
            "deployment_id": source["deployment_id"],
            "replica_index": 0,
            "container_name": source["container_name"],
            "endpoint": source["xinference_endpoint"],
            "port": source["port"],
            "gpu_ids": [source["gpu_id"]],
            "status": source["status"],
            "health_status": source["health_status"],
            "error_message": source["error_message"],
            "created_at": source["created_at"],
            "updated_at": source["updated_at"],
            "started_at": source["started_at"],
            "stopped_at": source["stopped_at"],
        }
        current = existing.get(source["deployment_id"])
        if current is not None:
            mismatched = any(
                (decode_gpu_ids(current[key]) != value if key == "gpu_ids" else current[key] != value)
                for key, value in expected.items()
            )
            if mismatched:
                raise RuntimeError("Existing deployment replica zero does not match legacy deployment")
            continue
        bind.execute(
            sa.insert(table).values(
                replica_id=str(uuid.uuid4()),
                **expected,
            )
        )

    count = bind.scalar(sa.select(sa.func.count()).select_from(table))
    if count != len(rows):
        raise RuntimeError("Deployment replica backfill row count is invalid")


def _deployment_configs(bind, deployment_ids: list[str]) -> tuple[sa.Table, dict]:
    inspector = inspect(bind)
    deployment_columns = {column["name"] for column in inspector.get_columns("deployments")}
    if "config" not in deployment_columns:
        raise RuntimeError("Required deployments columns are missing: config")

    deployments = sa.Table("deployments", sa.MetaData(), autoload_with=bind)
    configs = {}
    for deployment_id, value in bind.execute(
        sa.select(deployments.c.deployment_id, deployments.c.config).where(
            deployments.c.deployment_id.in_(deployment_ids)
        )
    ):
        if value is None:
            config = {}
        elif isinstance(value, str):
            try:
                config = json.loads(value)
            except (TypeError, ValueError):
                raise RuntimeError("deployment config is invalid") from None
        elif isinstance(value, Mapping):
            config = dict(value)
        else:
            raise RuntimeError("deployment config is invalid")
        if not isinstance(config, dict):
            raise RuntimeError("deployment config is invalid")
        configs[deployment_id] = config
    if set(configs) != set(deployment_ids):
        raise RuntimeError("deployment replica parent config is missing")
    return deployments, configs


def _validate_replica_schema_configs(configs: Mapping[str, dict]) -> None:
    for config in configs.values():
        current = config.get(REPLICA_SCHEMA_VERSION_KEY)
        if current is not None and (type(current) is not int or current != REPLICA_SCHEMA_VERSION):
            raise RuntimeError("deployment replica schema marker is invalid")


def _set_replica_schema_marker(bind, deployment_ids: list[str]) -> None:
    deployments, configs = _deployment_configs(bind, deployment_ids)
    _validate_replica_schema_configs(configs)

    for deployment_id, config in configs.items():
        if config.get(REPLICA_SCHEMA_VERSION_KEY) == REPLICA_SCHEMA_VERSION:
            continue
        updated = dict(config)
        updated[REPLICA_SCHEMA_VERSION_KEY] = REPLICA_SCHEMA_VERSION
        bind.execute(sa.update(deployments).where(deployments.c.deployment_id == deployment_id).values(config=updated))


def _clear_replica_schema_marker(bind, deployment_ids: list[str]) -> None:
    deployments, configs = _deployment_configs(bind, deployment_ids)
    _validate_replica_schema_configs(configs)

    for deployment_id, config in configs.items():
        if REPLICA_SCHEMA_VERSION_KEY not in config:
            continue
        updated = dict(config)
        updated.pop(REPLICA_SCHEMA_VERSION_KEY)
        bind.execute(sa.update(deployments).where(deployments.c.deployment_id == deployment_id).values(config=updated))


def _plan_preexisting_replica_schema_marker_restores(bind) -> tuple[sa.Table, list[tuple[str, dict]]]:
    deployments = sa.Table("deployments", sa.MetaData(), autoload_with=bind)
    rows = list(
        bind.execute(
            sa.select(
                deployments.c.deployment_id,
                deployments.c.inference_framework,
                deployments.c.config,
            ).order_by(deployments.c.deployment_id)
        )
    )
    pending_updates = []
    for deployment_id, framework, value in rows:
        if framework not in ("vllm", "sglang"):
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                continue
        if not isinstance(value, Mapping):
            continue
        config = dict(value)
        if LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY not in config:
            continue
        backup = config[LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY]
        if backup != LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_PAYLOAD:
            raise RuntimeError("legacy replica schema marker backup value is invalid")
        if REPLICA_SCHEMA_VERSION_KEY in config:
            raise RuntimeError("legacy replica schema marker restore conflicts")
        updated = dict(config)
        updated[REPLICA_SCHEMA_VERSION_KEY] = backup[REPLICA_SCHEMA_VERSION_KEY]
        updated.pop(LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY)
        pending_updates.append((deployment_id, updated))
    return deployments, pending_updates


def _apply_preexisting_replica_schema_marker_restores(
    bind,
    deployments: sa.Table,
    pending_updates: list[tuple[str, dict]],
) -> None:
    for deployment_id, updated in pending_updates:
        bind.execute(sa.update(deployments).where(deployments.c.deployment_id == deployment_id).values(config=updated))


def upgrade() -> None:
    bind = op.get_bind()
    _validate_parent_schema(bind)
    if TABLE_NAME not in set(inspect(bind).get_table_names()):
        _clear_preexisting_replica_schema_marker_collisions(bind)
        _create_replica_table()
    _validate_replica_table_core(bind)
    _ensure_replica_indexes(bind)
    _validate_replica_table(bind)


def downgrade() -> None:
    bind = op.get_bind()
    _validate_parent_schema(bind)
    deployments, pending_restores = _plan_preexisting_replica_schema_marker_restores(bind)
    if TABLE_NAME in set(inspect(bind).get_table_names()):
        _validate_replica_table(bind)
        table = sa.Table(TABLE_NAME, sa.MetaData(), autoload_with=bind)
        replica_count = bind.scalar(sa.select(sa.func.count()).select_from(table))
        if replica_count:
            raise RuntimeError("Refusing deployment replica downgrade with canonical replica state")
        op.drop_table(TABLE_NAME)
    _apply_preexisting_replica_schema_marker_restores(bind, deployments, pending_restores)
