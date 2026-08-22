"""Add an immutable execution-attempt fence to generation tasks.

Revision ID: 051_add_generation_run_token
Revises: 050_add_evaluation_run_token
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "051_add_generation_run_token"
down_revision = "050_add_evaluation_run_token"
branch_labels = None
depends_on = None

TABLE_NAME = "generation_tasks"
COLUMN_NAME = "run_token"
INDEX_NAME = "idx_gen_task_run_token"


def _schema_snapshot(bind, *, require_table: bool):
    inspector = inspect(bind)
    if TABLE_NAME not in inspector.get_table_names():
        if require_table:
            raise RuntimeError(
                f"Required table {TABLE_NAME!r} is missing; "
                "refusing to stamp the generation run-token migration"
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
                "Generation run-token primary key metadata is invalid"
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
                "Generation run-token primary key metadata is invalid"
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
        raise RuntimeError("Generation run-token primary key metadata is invalid")
    return {
        "columns": {column["name"]: column for column in reflected_columns},
        "primary_key_columns": tuple(primary_key_columns),
        "indexes": {
            index["name"]: index
            for index in inspector.get_indexes(TABLE_NAME)
        },
        "unique_constraints": inspector.get_unique_constraints(TABLE_NAME),
    }


def _validate_existing_schema(snapshot) -> None:
    if snapshot is None:
        return
    column = snapshot["columns"].get(COLUMN_NAME)
    if column is not None:
        if column.get("computed") is not None:
            raise RuntimeError(
                "Refusing generation run-token migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must be a plain column, not a "
                "computed column"
            )
        if column.get("identity") is not None:
            raise RuntimeError(
                "Refusing generation run-token migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must be a plain column, not an "
                "identity column"
            )
        if column.get("default") is not None:
            raise RuntimeError(
                "Refusing generation run-token migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must not have a server default"
            )
        primary_key_columns = snapshot.get("primary_key_columns")
        marker = column.get("primary_key")
        if primary_key_columns is None:
            if marker not in (False, 0):
                raise RuntimeError("Generation run-token primary key metadata is invalid")
            primary_key_columns = ()
        if not set(primary_key_columns).issubset(snapshot["columns"]):
            raise RuntimeError("Generation run-token primary key metadata is invalid")
        if marker not in (None, False, 0) or COLUMN_NAME in primary_key_columns:
            raise RuntimeError(
                "Refusing generation run-token migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must not be a primary key"
            )
        column_type = column.get("type")
        visit_name = getattr(column_type, "__visit_name__", "").lower()
        if (
            visit_name != "varchar"
            or getattr(column_type, "length", None) != 36
            or column.get("nullable") is not True
        ):
            raise RuntimeError(
                "Refusing generation run-token migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must be nullable VARCHAR(36)"
            )

    for constraint in snapshot["unique_constraints"]:
        columns = tuple(constraint.get("column_names") or ())
        if COLUMN_NAME in columns:
            name = constraint.get("name") or "<unnamed>"
            raise RuntimeError(
                "Refusing generation run-token migration: unexpected unique "
                f"constraint {name!r} references {TABLE_NAME}.{COLUMN_NAME}"
            )

    index = snapshot["indexes"].get(INDEX_NAME)
    if index is not None:
        actual_columns = tuple(index.get("column_names") or ())
        unique = index.get("unique")
        if actual_columns != (COLUMN_NAME,) or unique not in (False, 0):
            raise RuntimeError(
                "Refusing generation run-token migration: existing index "
                f"{TABLE_NAME}.{INDEX_NAME} must be a non-unique index on "
                f"{(COLUMN_NAME,)!r} in that exact order"
            )

    for name, auxiliary_index in snapshot["indexes"].items():
        if name == INDEX_NAME:
            continue
        columns = tuple(auxiliary_index.get("column_names") or ())
        if COLUMN_NAME not in columns:
            continue
        display_name = name or "<unnamed>"
        unique = auxiliary_index.get("unique")
        if unique in (True, 1):
            raise RuntimeError(
                "Refusing generation run-token migration: unexpected index "
                f"{display_name!r} references {TABLE_NAME}.{COLUMN_NAME}; this "
                "is an unexpected unique index"
            )
        if unique not in (False, 0):
            raise RuntimeError(
                "Refusing generation run-token migration: index "
                f"{display_name!r} has unknown uniqueness metadata for "
                f"{TABLE_NAME}.{COLUMN_NAME}"
            )


def _validate_downgrade_indexes(snapshot) -> None:
    for name, index in snapshot["indexes"].items():
        if name == INDEX_NAME:
            continue
        columns = tuple(index.get("column_names") or ())
        if COLUMN_NAME not in columns:
            continue
        display_name = name or "<unnamed>"
        raise RuntimeError(
            "Refusing generation run-token downgrade: unexpected index "
            f"{display_name!r} references {TABLE_NAME}.{COLUMN_NAME}"
        )


def upgrade() -> None:
    snapshot = _schema_snapshot(op.get_bind(), require_table=True)
    _validate_existing_schema(snapshot)
    if COLUMN_NAME not in snapshot["columns"]:
        op.add_column(
            TABLE_NAME,
            sa.Column(COLUMN_NAME, sa.String(length=36), nullable=True),
        )

    if INDEX_NAME not in snapshot["indexes"]:
        op.create_index(INDEX_NAME, TABLE_NAME, [COLUMN_NAME], unique=False)


def downgrade() -> None:
    snapshot = _schema_snapshot(op.get_bind(), require_table=False)
    _validate_existing_schema(snapshot)
    if snapshot is None:
        return
    _validate_downgrade_indexes(snapshot)

    if INDEX_NAME in snapshot["indexes"]:
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
    if COLUMN_NAME in snapshot["columns"]:
        op.drop_column(TABLE_NAME, COLUMN_NAME)
