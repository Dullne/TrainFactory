"""Bind local model configs and loaded adapters to explicit replicas.

Revision ID: 055_bind_configs_and_adapters_to_replicas
Revises: 054_add_deployment_replicas
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "055_bind_configs_and_adapters_to_replicas"
down_revision = "054_add_deployment_replicas"
branch_labels = None
depends_on = None

TARGETS = {
    "model_configs": (
        "idx_model_configs_deployment_replica_id",
        "fk_model_configs_deployment_replica_id",
    ),
    "loaded_adapters": (
        "idx_loaded_adapters_deployment_replica_id",
        "fk_loaded_adapters_deployment_replica_id",
    ),
}
COLUMN_NAME = "deployment_replica_id"


def _snapshot(bind, table_name: str):
    inspector = inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        raise RuntimeError(f"Required {table_name} table is missing")
    return {
        "columns": {
            column["name"]: column for column in inspector.get_columns(table_name)
        },
        "indexes": {index["name"]: index for index in inspector.get_indexes(table_name)},
        "foreign_keys": inspector.get_foreign_keys(table_name),
    }


def _validate_snapshot(table_name: str, snapshot) -> None:
    index_name, foreign_key_name = TARGETS[table_name]
    column = snapshot["columns"].get(COLUMN_NAME)
    if column is None:
        raise RuntimeError(f"{table_name}.{COLUMN_NAME} is missing")
    column_type = column.get("type")
    if (
        getattr(column_type, "__visit_name__", "").lower() != "string"
        and getattr(column_type, "__visit_name__", "").lower() != "varchar"
    ) or getattr(column_type, "length", None) != 36 or column.get("nullable") is not True:
        raise RuntimeError(f"{table_name}.{COLUMN_NAME} shape is invalid")

    index = snapshot["indexes"].get(index_name)
    related_indexes = [
        item
        for item in snapshot["indexes"].values()
        if COLUMN_NAME in tuple(item.get("column_names") or ())
    ]
    if (
        index is None
        or tuple(index.get("column_names") or ()) != (COLUMN_NAME,)
        or index.get("unique") not in (False, 0)
        or len(related_indexes) != 1
    ):
        raise RuntimeError(f"{table_name}.{COLUMN_NAME} index is invalid")

    related_foreign_keys = [
        foreign_key
        for foreign_key in snapshot["foreign_keys"]
        if COLUMN_NAME in tuple(foreign_key.get("constrained_columns") or ())
    ]
    matching = [
        foreign_key
        for foreign_key in related_foreign_keys
        if foreign_key.get("name") == foreign_key_name
        and tuple(foreign_key.get("constrained_columns") or ()) == (COLUMN_NAME,)
        and foreign_key.get("referred_table") == "deployment_replicas"
        and tuple(foreign_key.get("referred_columns") or ()) == ("replica_id",)
        and str((foreign_key.get("options") or {}).get("ondelete", "")).upper()
        == "SET NULL"
    ]
    if len(related_foreign_keys) != 1 or len(matching) != 1:
        raise RuntimeError(f"{table_name}.{COLUMN_NAME} foreign key is invalid")


def _validate_existing(bind, table_name: str) -> None:
    _validate_snapshot(table_name, _snapshot(bind, table_name))


def _validate_upgrade_snapshot(table_name: str, snapshot) -> None:
    if COLUMN_NAME in snapshot["columns"]:
        _validate_snapshot(table_name, snapshot)
        return

    index_name, foreign_key_name = TARGETS[table_name]
    if index_name in snapshot["indexes"]:
        raise RuntimeError(f"{table_name}.{COLUMN_NAME} index is invalid")
    if any(
        foreign_key.get("name") == foreign_key_name
        for foreign_key in snapshot["foreign_keys"]
    ):
        raise RuntimeError(f"{table_name}.{COLUMN_NAME} foreign key is invalid")


def upgrade() -> None:
    bind = op.get_bind()
    if "deployment_replicas" not in set(inspect(bind).get_table_names()):
        raise RuntimeError("Required deployment_replicas table is missing")

    snapshots = {
        table_name: _snapshot(bind, table_name) for table_name in TARGETS
    }
    for table_name, snapshot in snapshots.items():
        _validate_upgrade_snapshot(table_name, snapshot)

    for table_name, (index_name, foreign_key_name) in TARGETS.items():
        if COLUMN_NAME not in snapshots[table_name]["columns"]:
            with op.batch_alter_table(table_name) as batch_op:
                batch_op.add_column(
                    sa.Column(COLUMN_NAME, sa.String(length=36), nullable=True)
                )
                batch_op.create_index(index_name, [COLUMN_NAME], unique=False)
                batch_op.create_foreign_key(
                    foreign_key_name,
                    "deployment_replicas",
                    [COLUMN_NAME],
                    ["replica_id"],
                    ondelete="SET NULL",
                )

    for table_name in TARGETS:
        _validate_existing(bind, table_name)


def downgrade() -> None:
    bind = op.get_bind()
    table_names = tuple(reversed(tuple(TARGETS)))
    snapshots = {
        table_name: _snapshot(bind, table_name) for table_name in table_names
    }
    present_tables = [
        table_name
        for table_name in table_names
        if COLUMN_NAME in snapshots[table_name]["columns"]
    ]

    for table_name in present_tables:
        _validate_snapshot(table_name, snapshots[table_name])

    reflected_tables = {
        table_name: sa.Table(table_name, sa.MetaData(), autoload_with=bind)
        for table_name in present_tables
    }
    for table_name, table in reflected_tables.items():
        if bind.scalar(
            sa.select(sa.func.count())
            .select_from(table)
            .where(table.c.deployment_replica_id.is_not(None))
        ):
            raise RuntimeError(
                "Refusing replica binding downgrade while explicit bindings exist"
            )

    for table_name in present_tables:
        index_name, foreign_key_name = TARGETS[table_name]
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.drop_constraint(foreign_key_name, type_="foreignkey")
            batch_op.drop_index(index_name)
            batch_op.drop_column(COLUMN_NAME)
