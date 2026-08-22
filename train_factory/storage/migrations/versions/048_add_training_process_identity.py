"""Persist training subprocess creation time for PID reuse fencing.

Revision ID: 048_add_training_process_identity
Revises: 047_milvus_hybrid_intent
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "048_add_training_process_identity"
down_revision = "047_milvus_hybrid_intent"
branch_labels = None
depends_on = None

TABLE_NAME = "training_tasks"
COLUMN_NAME = "process_create_time"


def _schema_snapshot(bind, *, require_table: bool):
    inspector = inspect(bind)
    if TABLE_NAME not in inspector.get_table_names():
        if require_table:
            raise RuntimeError(
                f"Required table {TABLE_NAME!r} is missing; "
                "refusing to stamp the training process identity migration"
            )
        return None
    reflected_columns = inspector.get_columns(TABLE_NAME)
    get_primary_key = getattr(inspector, "get_pk_constraint", None)
    if get_primary_key is None:
        if any(
            "primary_key" not in column
            or column["primary_key"] not in (False, 0, True, 1)
            for column in reflected_columns
        ):
            raise RuntimeError(
                "Training process identity primary key metadata is invalid"
            )
        primary_key = {
            "constrained_columns": [
                column["name"]
                for column in reflected_columns
                if column.get("primary_key") in (True, 1)
            ]
        }
    else:
        try:
            primary_key = get_primary_key(TABLE_NAME)
        except Exception:
            raise RuntimeError(
                "Training process identity primary key metadata is invalid"
            ) from None
    primary_key_columns = primary_key.get("constrained_columns") if isinstance(primary_key, dict) else None
    if (
        not isinstance(primary_key_columns, (list, tuple))
        or any(not isinstance(name, str) or not name for name in primary_key_columns)
        or len(set(primary_key_columns)) != len(primary_key_columns)
        or not set(primary_key_columns).issubset(
            {column.get("name") for column in reflected_columns}
        )
    ):
        raise RuntimeError("Training process identity primary key metadata is invalid")
    return {
        "columns": {column["name"]: column for column in reflected_columns},
        "primary_key_columns": tuple(primary_key_columns),
        "unique_constraints": inspector.get_unique_constraints(TABLE_NAME),
        "indexes": inspector.get_indexes(TABLE_NAME),
    }


def _validate_existing_schema(snapshot) -> None:
    if snapshot is None:
        return
    column = snapshot["columns"].get(COLUMN_NAME)
    if column is None:
        return
    if column.get("computed") is not None:
        raise RuntimeError(
            "Refusing training process identity migration: existing column "
            f"{TABLE_NAME}.{COLUMN_NAME} must be a plain column, not a computed column"
        )
    if column.get("identity") is not None:
        raise RuntimeError(
            "Refusing training process identity migration: existing column "
            f"{TABLE_NAME}.{COLUMN_NAME} must be a plain column, not an identity column"
        )
    if column.get("default") is not None:
        raise RuntimeError(
            "Refusing training process identity migration: existing column "
            f"{TABLE_NAME}.{COLUMN_NAME} must not have a server default"
        )
    primary_key_columns = snapshot.get("primary_key_columns")
    marker = column.get("primary_key")
    if primary_key_columns is None:
        if marker not in (False, 0):
            raise RuntimeError("Training process identity primary key metadata is invalid")
        primary_key_columns = ()
    if not set(primary_key_columns).issubset(snapshot["columns"]):
        raise RuntimeError("Training process identity primary key metadata is invalid")
    if marker not in (None, False, 0) or COLUMN_NAME in primary_key_columns:
        raise RuntimeError(
            "Refusing training process identity migration: existing column "
            f"{TABLE_NAME}.{COLUMN_NAME} must not be a primary key"
        )
    column_type = column.get("type")
    visit_name = getattr(column_type, "__visit_name__", "").lower()
    if visit_name != "double" or column.get("nullable") is not True:
        raise RuntimeError(
            "Refusing training process identity migration: existing column "
            f"{TABLE_NAME}.{COLUMN_NAME} must be nullable DOUBLE"
        )

    for constraint in snapshot["unique_constraints"]:
        columns = tuple(constraint.get("column_names") or ())
        if COLUMN_NAME in columns:
            name = constraint.get("name") or "<unnamed>"
            raise RuntimeError(
                "Refusing training process identity migration: unexpected "
                f"unique constraint {name!r} references {TABLE_NAME}.{COLUMN_NAME}"
            )

    for index in snapshot["indexes"]:
        columns = tuple(index.get("column_names") or ())
        if COLUMN_NAME not in columns:
            continue
        name = index.get("name") or "<unnamed>"
        unique = index.get("unique")
        if unique in (True, 1):
            raise RuntimeError(
                "Refusing training process identity migration: unexpected index "
                f"{name!r} references {TABLE_NAME}.{COLUMN_NAME}; this is an "
                "unexpected unique index"
            )
        if unique not in (False, 0):
            raise RuntimeError(
                "Refusing training process identity migration: index "
                f"{name!r} has unknown uniqueness metadata for "
                f"{TABLE_NAME}.{COLUMN_NAME}"
            )


def _validate_downgrade_indexes(snapshot) -> None:
    for index in snapshot["indexes"]:
        columns = tuple(index.get("column_names") or ())
        if COLUMN_NAME not in columns:
            continue
        name = index.get("name") or "<unnamed>"
        raise RuntimeError(
            "Refusing training process identity downgrade: unexpected index "
            f"{name!r} references {TABLE_NAME}.{COLUMN_NAME}"
        )


def upgrade() -> None:
    snapshot = _schema_snapshot(op.get_bind(), require_table=True)
    _validate_existing_schema(snapshot)
    if COLUMN_NAME not in snapshot["columns"]:
        op.add_column(
            TABLE_NAME,
            sa.Column(COLUMN_NAME, sa.Double(), nullable=True),
        )


def downgrade() -> None:
    snapshot = _schema_snapshot(op.get_bind(), require_table=False)
    _validate_existing_schema(snapshot)
    if snapshot is None:
        return
    _validate_downgrade_indexes(snapshot)
    if COLUMN_NAME in snapshot["columns"]:
        op.drop_column(TABLE_NAME, COLUMN_NAME)
