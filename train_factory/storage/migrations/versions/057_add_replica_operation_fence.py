"""Add a durable CAS fence for deployment replica lifecycle operations.

Revision ID: 057_add_replica_operation_fence
Revises: 056_bind_sync_targets_to_replicas
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "057_add_replica_operation_fence"
down_revision = "056_bind_sync_targets_to_replicas"
branch_labels = None
depends_on = None

TABLE_NAME = "deployments"
COLUMNS = {
    "replica_operation_token": (sa.String(length=36), True, None),
    "replica_operation_kind": (sa.String(length=16), True, None),
    "replica_operation_replica_id": (sa.String(length=36), True, None),
    "replica_operation_generation": (sa.Integer(), False, "0"),
    "replica_operation_started_at": (sa.DateTime(), True, None),
    "replica_operation_heartbeat_at": (sa.DateTime(), True, None),
}


def _columns(bind):
    reflected = inspect(bind)
    if TABLE_NAME not in set(reflected.get_table_names()):
        raise RuntimeError("Required deployments table is missing")
    return {item["name"]: item for item in reflected.get_columns(TABLE_NAME)}


def _normalize_default(default):
    if default is None:
        return None
    normalized = str(default).strip()
    while len(normalized) >= 2 and normalized[0] == "(" and normalized[-1] == ")":
        normalized = normalized[1:-1].strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {"'", '"'}
    ):
        normalized = normalized[1:-1]
    return normalized


def _validate_column(name: str, column) -> None:
    expected_type, nullable, default = COLUMNS[name]
    actual_type = column.get("type")
    actual_visit_name = getattr(actual_type, "__visit_name__", "").lower()
    expected_visit_name = getattr(expected_type, "__visit_name__", "").lower()
    if isinstance(expected_type, sa.String):
        valid_type = (
            actual_visit_name in {"string", "varchar"}
            and getattr(actual_type, "length", None) == expected_type.length
        )
    else:
        valid_type = actual_visit_name == expected_visit_name
    if (
        not valid_type
        or column.get("nullable") is not nullable
        or _normalize_default(column.get("default")) != default
    ):
        raise RuntimeError(f"deployments.{name} shape is invalid")


def _validate_present(columns) -> None:
    for name in COLUMNS:
        column = columns.get(name)
        if column is not None:
            _validate_column(name, column)


def _validate(bind) -> None:
    columns = _columns(bind)
    for name in COLUMNS:
        column = columns.get(name)
        if column is None:
            raise RuntimeError(f"deployments.{name} shape is invalid")
        _validate_column(name, column)


def upgrade() -> None:
    bind = op.get_bind()
    columns = _columns(bind)
    _validate_present(columns)
    missing = [name for name in COLUMNS if name not in columns]
    if missing:
        with op.batch_alter_table(TABLE_NAME) as batch_op:
            for name in missing:
                column_type, nullable, default = COLUMNS[name]
                batch_op.add_column(
                    sa.Column(
                        name,
                        column_type,
                        nullable=nullable,
                        server_default=sa.text(default) if default is not None else None,
                    )
                )
    _validate(bind)


def downgrade() -> None:
    bind = op.get_bind()
    columns = _columns(bind)
    token = columns.get("replica_operation_token")
    if token is not None:
        table = sa.Table(TABLE_NAME, sa.MetaData(), autoload_with=bind)
        active = bind.scalar(
            sa.select(sa.func.count()).select_from(table).where(table.c.replica_operation_token.is_not(None))
        )
        if active:
            raise RuntimeError("Refusing downgrade while replica operations are claimed")
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        for name in reversed(tuple(COLUMNS)):
            if name in columns:
                batch_op.drop_column(name)
