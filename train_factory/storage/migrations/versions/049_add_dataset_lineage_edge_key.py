"""Add a deterministic unique identity to dataset lineage edges.

Revision ID: 049_dataset_lineage_edge_key
Revises: 048_add_training_process_identity
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Optional

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "049_dataset_lineage_edge_key"
down_revision = "048_add_training_process_identity"
branch_labels = None
depends_on = None

TABLE_NAME = "dataset_lineage_edges"
COLUMN_NAME = "edge_key"
CONSTRAINT_NAME = "uq_lineage_edge_key"
UNIQUE_NAMING_CONVENTION = {
    "uq": "uq_%(table_name)s_%(column_0_name)s",
}


def _schema_snapshot(bind, *, require_table: bool):
    inspector = inspect(bind)
    if TABLE_NAME not in inspector.get_table_names():
        if require_table:
            raise RuntimeError(
                f"Required table {TABLE_NAME!r} is missing; "
                "refusing to stamp the lineage identity migration"
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
            raise RuntimeError("Lineage identity primary key metadata is invalid")
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
                "Lineage identity primary key metadata is invalid"
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
        raise RuntimeError("Lineage identity primary key metadata is invalid")
    return {
        "columns": {column["name"]: column for column in reflected_columns},
        "primary_key_columns": tuple(primary_key_columns),
        "unique_constraints": inspector.get_unique_constraints(TABLE_NAME),
        "indexes": inspector.get_indexes(TABLE_NAME),
    }


def _raise_malformed_uniqueness(object_type: str, name) -> None:
    display_name = name if name is not None else "<unnamed>"
    raise RuntimeError(
        "Refusing lineage identity migration: existing uniqueness object "
        f"{object_type} {display_name!r} must be unique on {(COLUMN_NAME,)!r} "
        "in that exact order"
    )


def _validate_existing_schema(snapshot) -> None:
    if snapshot is None:
        return

    column = snapshot["columns"].get(COLUMN_NAME)
    if column is not None:
        if column.get("computed") is not None:
            raise RuntimeError(
                "Refusing lineage identity migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must be a plain column, not a "
                "computed column"
            )
        if column.get("identity") is not None:
            raise RuntimeError(
                "Refusing lineage identity migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must be a plain column, not an "
                "identity column"
            )
        if column.get("default") is not None:
            raise RuntimeError(
                "Refusing lineage identity migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must not have a server default"
            )
        primary_key_columns = snapshot.get("primary_key_columns")
        marker = column.get("primary_key")
        if primary_key_columns is None:
            if marker not in (False, 0):
                raise RuntimeError("Lineage identity primary key metadata is invalid")
            primary_key_columns = ()
        if not set(primary_key_columns).issubset(snapshot["columns"]):
            raise RuntimeError("Lineage identity primary key metadata is invalid")
        if marker not in (None, False, 0) or COLUMN_NAME in primary_key_columns:
            raise RuntimeError(
                "Refusing lineage identity migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must not be a primary key"
            )
        column_type = column.get("type")
        visit_name = getattr(column_type, "__visit_name__", "").lower()
        if visit_name != "varchar" or getattr(column_type, "length", None) != 64:
            raise RuntimeError(
                "Refusing lineage identity migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} must be VARCHAR(64)"
            )
        nullable = column.get("nullable")
        if nullable is not True and nullable is not False:
            raise RuntimeError(
                "Refusing lineage identity migration: existing column "
                f"{TABLE_NAME}.{COLUMN_NAME} nullable metadata must identify a "
                "partial or complete schema"
            )

    for constraint in snapshot["unique_constraints"]:
        columns = tuple(constraint.get("column_names") or ())
        name = constraint.get("name")
        if name != CONSTRAINT_NAME and COLUMN_NAME not in columns:
            continue
        if columns != (COLUMN_NAME,):
            _raise_malformed_uniqueness("constraint", name)

    for index in snapshot["indexes"]:
        columns = tuple(index.get("column_names") or ())
        name = index.get("name")
        unique = index.get("unique")
        if COLUMN_NAME in columns and unique not in (True, 1, False, 0):
            display_name = name if name is not None else "<unnamed>"
            raise RuntimeError(
                "Refusing lineage identity migration: index "
                f"{display_name!r} has unknown uniqueness metadata for "
                f"{TABLE_NAME}.{COLUMN_NAME}"
            )
        is_unique = unique in (True, 1)
        if name != CONSTRAINT_NAME and not (is_unique and COLUMN_NAME in columns):
            continue
        if name is None or columns != (COLUMN_NAME,) or not is_unique:
            _raise_malformed_uniqueness("index", name)


def _validate_downgrade_indexes(snapshot) -> None:
    for index in snapshot["indexes"]:
        columns = tuple(index.get("column_names") or ())
        if COLUMN_NAME not in columns:
            continue
        if index.get("unique") in (True, 1) and columns == (COLUMN_NAME,):
            continue
        name = index.get("name") or "<unnamed>"
        raise RuntimeError(
            "Refusing lineage identity downgrade: unexpected index "
            f"{name!r} references {TABLE_NAME}.{COLUMN_NAME}"
        )


def _build_edge_key(
    from_dataset_id: Optional[str],
    to_dataset_id: str,
    relation_type: str,
) -> str:
    canonical = json.dumps(
        [from_dataset_id, to_dataset_id, relation_type],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unique_edge_key_objects(bind) -> list[tuple[str, str]]:
    inspector = inspect(bind)
    constraints = inspector.get_unique_constraints(TABLE_NAME)
    indexes = inspector.get_indexes(TABLE_NAME)
    objects: list[tuple[str, str]] = []
    constraint_names: set[str] = set()
    duplicate_index_names: set[str] = set()
    for constraint in constraints:
        if tuple(constraint.get("column_names") or ()) != (COLUMN_NAME,):
            continue
        name = constraint.get("name") or (f"uq_{TABLE_NAME}_{COLUMN_NAME}")
        objects.append(("constraint", name))
        constraint_names.add(name)
        duplicate_index_name = constraint.get("duplicates_index")
        if duplicate_index_name:
            duplicate_index_names.add(duplicate_index_name)
    for index in indexes:
        if not index.get("unique") or tuple(index.get("column_names") or ()) != (
            COLUMN_NAME,
        ):
            continue
        name = index.get("name")
        if not name:
            raise RuntimeError("Cannot safely replace an unnamed unique edge_key index")
        if (
            name in constraint_names
            or name in duplicate_index_names
            or index.get("duplicates_constraint") in constraint_names
        ):
            continue
        item = ("index", name)
        if item not in objects:
            objects.append(item)
    return objects


def _drop_unique_edge_key_objects(batch_op, objects) -> None:
    for object_type, name in objects:
        if object_type == "constraint":
            batch_op.drop_constraint(name, type_="unique")
        else:
            batch_op.drop_index(name)


class _ExactJsonNumber(str):
    """A parsed JSON number whose original decimal lexeme is intact."""


class _InvalidJsonValue(ValueError):
    """Raised internally when op_params cannot be compared losslessly."""


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJsonValue("duplicate object key")
        result[key] = value
    return result


def _reject_non_finite_constant(_value: str):
    raise _InvalidJsonValue("non-finite JSON number")


def _canonical_json_value(value: Any) -> tuple[Any, ...]:
    if isinstance(value, _ExactJsonNumber):
        return ("number", str(value))
    if value is None:
        return ("json-null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, int):
        return ("number", str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _InvalidJsonValue("non-finite decoded number")
        return ("number", repr(value))
    if isinstance(value, list):
        return ("array", tuple(_canonical_json_value(item) for item in value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise _InvalidJsonValue("non-string object key")
        return (
            "object",
            tuple(
                sorted(
                    (key, _canonical_json_value(item)) for key, item in value.items()
                )
            ),
        )
    raise _InvalidJsonValue("unsupported decoded JSON value")


def _canonical_op_params(value: Any) -> tuple[Any, ...]:
    """Build a lossless, order-independent comparison token for op_params.

    The migration reads the nullable JSON column through raw SQL, so a top-level
    Python ``None`` represents SQL NULL.  JSON text ``"null"`` represents JSON
    null and intentionally remains distinct: absence of provenance parameters is
    not silently equated with an explicitly stored JSON null value.
    """

    if value is None:
        return ("sql-null",)
    try:
        if isinstance(value, str):
            value = json.loads(
                value,
                parse_int=_ExactJsonNumber,
                parse_float=_ExactJsonNumber,
                parse_constant=_reject_non_finite_constant,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        return ("json", _canonical_json_value(value))
    except (json.JSONDecodeError, _InvalidJsonValue):
        raise RuntimeError(
            "Cannot safely compare op_params: malformed or unsupported JSON"
        ) from None


def _lineage_triple_description(row) -> str:
    return (
        "(from_dataset_id="
        f"{row['from_dataset_id']!r}, to_dataset_id={row['to_dataset_id']!r}, "
        f"relation_type={row['relation_type']!r})"
    )


def _canonical_provenance(row) -> dict[str, Any]:
    try:
        canonical_params = _canonical_op_params(row["op_params"])
    except RuntimeError:
        raise RuntimeError(
            "Refusing to discard duplicate lineage triple "
            f"{_lineage_triple_description(row)}: row {row['id']} has "
            "malformed or unsupported op_params"
        ) from None
    return {
        "op_task_type": row["op_task_type"],
        "op_task_id": row["op_task_id"],
        "op_params": canonical_params,
    }


def _build_backfill_plan(bind) -> list[tuple[str, Any, Optional[str]]]:
    available_columns = {
        column["name"] for column in inspect(bind).get_columns(TABLE_NAME)
    }

    def selected(column_name: str) -> str:
        return (
            column_name
            if column_name in available_columns
            else f"NULL AS {column_name}"
        )

    rows = list(
        bind.execute(
            sa.text(
                "SELECT id, edge_id, from_dataset_id, to_dataset_id, relation_type, "
                f"{selected('op_task_type')}, {selected('op_task_id')}, "
                f"{selected('op_params')} "
                f"FROM {TABLE_NAME} ORDER BY id"
            )
        ).mappings()
    )
    seen: dict[str, Any] = {}
    plan: list[tuple[str, Any, Optional[str]]] = []
    for row in rows:
        edge_key = _build_edge_key(
            row["from_dataset_id"],
            row["to_dataset_id"],
            row["relation_type"],
        )
        if edge_key in seen:
            first = seen[edge_key]
            first_provenance = _canonical_provenance(first)
            duplicate_provenance = _canonical_provenance(row)
            if duplicate_provenance != first_provenance:
                differing_fields = [
                    field_name
                    for field_name in first_provenance
                    if first_provenance[field_name] != duplicate_provenance[field_name]
                ]
                raise RuntimeError(
                    "Refusing to discard conflicting provenance for duplicate "
                    f"lineage triple {_lineage_triple_description(row)}: rows "
                    f"{first['id']} and {row['id']}; differing fields: "
                    f"{', '.join(differing_fields)}"
                )
            raise RuntimeError(
                "Refusing lineage identity migration: duplicate lineage triple "
                f"{_lineage_triple_description(row)} is represented by distinct "
                f"rows {first['id']} (edge_id={first['edge_id']!r}) and "
                f"{row['id']} (edge_id={row['edge_id']!r}); resolve the duplicate "
                "edges explicitly before retrying"
            )
        seen[edge_key] = row
        plan.append(("update", row["id"], edge_key))
    return plan


def _apply_backfill_plan(
    bind,
    plan: list[tuple[str, Any, Optional[str]]],
) -> None:
    for operation, row_id, edge_key in plan:
        if operation != "update":
            raise RuntimeError(f"Unsupported lineage backfill operation: {operation}")
        bind.execute(
            sa.text(
                f"UPDATE {TABLE_NAME} SET {COLUMN_NAME} = :edge_key " "WHERE id = :id"
            ),
            {"edge_key": edge_key, "id": row_id},
        )


def _backfill_and_deduplicate(bind) -> None:
    _apply_backfill_plan(bind, _build_backfill_plan(bind))


def upgrade() -> None:
    bind = op.get_bind()
    snapshot = _schema_snapshot(bind, require_table=True)
    _validate_existing_schema(snapshot)
    columns = snapshot["columns"]
    backfill_plan = _build_backfill_plan(bind)
    if COLUMN_NAME not in columns:
        op.add_column(
            TABLE_NAME,
            sa.Column(COLUMN_NAME, sa.String(64), nullable=True),
        )

    existing_unique_objects = _unique_edge_key_objects(bind)
    if existing_unique_objects:
        with op.batch_alter_table(
            TABLE_NAME,
            naming_convention=UNIQUE_NAMING_CONVENTION,
        ) as batch_op:
            _drop_unique_edge_key_objects(
                batch_op,
                existing_unique_objects,
            )

    _apply_backfill_plan(bind, backfill_plan)

    inspector = inspect(bind)
    edge_key_column = next(
        column
        for column in inspector.get_columns(TABLE_NAME)
        if column["name"] == COLUMN_NAME
    )
    needs_not_null = edge_key_column["nullable"]
    needs_unique = not _unique_edge_key_objects(bind)
    if needs_not_null or needs_unique:
        with op.batch_alter_table(TABLE_NAME) as batch_op:
            if needs_not_null:
                batch_op.alter_column(
                    COLUMN_NAME,
                    existing_type=sa.String(64),
                    nullable=False,
                )
            if needs_unique:
                batch_op.create_unique_constraint(
                    CONSTRAINT_NAME,
                    [COLUMN_NAME],
                )


def downgrade() -> None:
    bind = op.get_bind()
    snapshot = _schema_snapshot(bind, require_table=False)
    _validate_existing_schema(snapshot)
    if snapshot is None:
        return
    if COLUMN_NAME not in snapshot["columns"]:
        return
    _validate_downgrade_indexes(snapshot)

    unique_objects = _unique_edge_key_objects(bind)
    with op.batch_alter_table(
        TABLE_NAME,
        naming_convention=UNIQUE_NAMING_CONVENTION,
    ) as batch_op:
        _drop_unique_edge_key_objects(batch_op, unique_objects)
        batch_op.drop_column(COLUMN_NAME)
