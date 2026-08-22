"""Validate and safely repair lifecycle schema migrations 048 through 052.

Revision ID: 053_validate_lifecycle_schema
Revises: 052_generation_publication_staging
"""

from __future__ import annotations

import importlib
import hashlib
import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "053_validate_lifecycle_schema"
down_revision = "052_generation_publication_staging"
branch_labels = None
depends_on = None

LIFECYCLE_MIGRATION_MODULES = (
    "train_factory.storage.migrations.versions." "048_add_training_process_identity",
    "train_factory.storage.migrations.versions." "049_add_dataset_lineage_edge_key",
    "train_factory.storage.migrations.versions." "050_add_evaluation_run_token",
    "train_factory.storage.migrations.versions." "051_add_generation_run_token",
    "train_factory.storage.migrations.versions." "052_generation_publication_staging",
)
_RAW_SNAPSHOT_FAILED = object()


def _load_lifecycle_migrations():
    loaded = []
    for module_name in LIFECYCLE_MIGRATION_MODULES:
        try:
            loaded.append(importlib.import_module(module_name))
        except Exception:
            raise RuntimeError(
                f"Unable to load lifecycle migration {module_name}"
            ) from None
    modules = tuple(loaded)
    for previous, current in zip(modules, modules[1:]):
        if current.down_revision != previous.revision:
            raise RuntimeError("Lifecycle migration revision chain is not continuous")
    if modules[-1].revision != down_revision:
        raise RuntimeError(
            "Lifecycle migration chain does not terminate at revision 052"
        )
    return modules


def _summary(revision_id, category, object_name, *, count=1):
    return {
        "revision": revision_id,
        "category": category,
        "object_name": object_name,
        "count": count,
    }


def _digest(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _column_is_plain(column, primary_key_columns, column_name: str) -> bool:
    return (
        column.get("computed") is None
        and column.get("identity") is None
        and column.get("default") is None
        and column.get("primary_key") in (None, False, 0)
        and column_name not in primary_key_columns
    )


def _primary_key_columns(snapshot, revision_id, table_name, errors):
    primary_key = snapshot.get("primary_key")
    columns = (
        primary_key.get("constrained_columns")
        if isinstance(primary_key, dict)
        else None
    )
    reflected_names = [column.get("name") for column in snapshot.get("columns", ())]
    if (
        not isinstance(columns, (list, tuple))
        or any(not isinstance(name, str) or not name for name in columns)
        or len(set(columns)) != len(columns)
        or len(set(reflected_names)) != len(reflected_names)
        or not set(columns).issubset(reflected_names)
    ):
        errors.append(
            _summary(revision_id, "primary_key_metadata", table_name)
        )
        return None
    return tuple(columns)


def _snapshot_052(module, bind, errors):
    """Reflect 052 objects as raw lists so duplicate names remain visible."""

    try:
        reflected = inspect(bind)
        available_tables = set(reflected.get_table_names())
    except Exception:
        errors.append(
            _summary(
                module.revision,
                "raw_snapshot_failed",
                module.revision,
            )
        )
        return _RAW_SNAPSHOT_FAILED
    snapshot = {}
    for table_name in module.COLUMN_SPECS:
        if table_name not in available_tables:
            continue
        try:
            snapshot[table_name] = {
                "columns": list(reflected.get_columns(table_name)),
                "indexes": list(reflected.get_indexes(table_name)),
                "unique_constraints": list(
                    reflected.get_unique_constraints(table_name)
                ),
                "primary_key": reflected.get_pk_constraint(table_name),
            }
        except Exception:
            errors.append(
                _summary(
                    module.revision,
                    "raw_snapshot_failed",
                    table_name,
                )
            )
            snapshot[table_name] = _RAW_SNAPSHOT_FAILED
    return snapshot


def _snapshot_standard_raw(module, bind, errors):
    try:
        reflected = inspect(bind)
        if module.TABLE_NAME not in set(reflected.get_table_names()):
            return None
        return {
            "columns": list(reflected.get_columns(module.TABLE_NAME)),
            "indexes": list(reflected.get_indexes(module.TABLE_NAME)),
            "unique_constraints": list(
                reflected.get_unique_constraints(module.TABLE_NAME)
            ),
            "primary_key": reflected.get_pk_constraint(module.TABLE_NAME),
        }
    except Exception:
        errors.append(
            _summary(
                module.revision,
                "raw_snapshot_failed",
                module.TABLE_NAME,
            )
        )
        return _RAW_SNAPSHOT_FAILED


def _validate_standard_raw(module, snapshot, repairs, errors):
    if snapshot is _RAW_SNAPSHOT_FAILED:
        return
    revision_id = module.revision
    table_name = module.TABLE_NAME
    column_name = module.COLUMN_NAME
    column_object = f"{table_name}.{column_name}"
    if snapshot is None:
        errors.append(_summary(revision_id, "missing_base_table", table_name))
        return
    primary_key_columns = _primary_key_columns(
        snapshot, revision_id, table_name, errors
    )
    if primary_key_columns is None:
        return

    matching_columns = [
        column for column in snapshot["columns"] if column.get("name") == column_name
    ]
    if len(matching_columns) > 1:
        errors.append(_summary(revision_id, "duplicate_column_name", column_object))
    if not matching_columns:
        repairs.append(_summary(revision_id, "missing_column", column_object))
    for column in matching_columns:
        column_type = column.get("type")
        visit_name = getattr(column_type, "__visit_name__", "").lower()
        expected_type = "double" if revision_id.startswith("048_") else "varchar"
        valid_length = (
            True
            if expected_type == "double"
            else getattr(column_type, "length", None) == 36
        )
        if (
            not _column_is_plain(column, primary_key_columns, column_name)
            or visit_name != expected_type
            or not valid_length
            or column.get("nullable") is not True
        ):
            errors.append(_summary(revision_id, "column_shape", column_object))

    for constraint in snapshot["unique_constraints"]:
        columns = tuple(constraint.get("column_names") or ())
        if column_name not in columns:
            continue
        name = constraint.get("name") or "<unnamed>"
        errors.append(
            _summary(
                revision_id,
                "unexpected_unique_constraint",
                f"{table_name}.{name}",
            )
        )

    expected_index_name = getattr(module, "INDEX_NAME", None)
    relevant_names = {}
    expected_count = 0
    for index in snapshot["indexes"]:
        name = index.get("name")
        columns = tuple(index.get("column_names") or ())
        unique = index.get("unique")
        is_expected = name == expected_index_name and expected_index_name is not None
        overlaps = column_name in columns
        if not is_expected and not overlaps:
            continue
        object_name = f"{table_name}.{name or '<unnamed>'}"
        if name is not None:
            relevant_names[name] = relevant_names.get(name, 0) + 1
        if is_expected:
            expected_count += 1
            if columns != (column_name,) or unique not in (False, 0):
                errors.append(_summary(revision_id, "index_shape", object_name))
            continue
        if unique in (True, 1):
            category = "unexpected_unique_index"
        elif unique not in (False, 0):
            category = "unknown_index_uniqueness"
        elif expected_index_name is not None:
            category = "unexpected_index"
        elif name is None:
            category = "unnamed_index"
        else:
            continue
        errors.append(_summary(revision_id, category, object_name))

    for name, count in relevant_names.items():
        if count > 1:
            errors.append(
                _summary(
                    revision_id,
                    "duplicate_index_name",
                    f"{table_name}.{name}",
                )
            )
    if expected_index_name is not None and expected_count == 0:
        repairs.append(
            _summary(
                revision_id,
                "missing_index",
                f"{table_name}.{expected_index_name}",
            )
        )


def _validate_052(module, snapshot, repairs, errors):
    if snapshot is _RAW_SNAPSHOT_FAILED:
        return
    revision_id = module.revision
    for table_name in module.COLUMN_SPECS:
        table_schema = snapshot.get(table_name)
        if table_schema is _RAW_SNAPSHOT_FAILED:
            continue
        if table_schema is None:
            errors.append(_summary(revision_id, "missing_base_table", table_name))
            continue
        primary_key_columns = _primary_key_columns(
            table_schema, revision_id, table_name, errors
        )
        if primary_key_columns is None:
            continue

        columns_by_name = {}
        for column in table_schema["columns"]:
            columns_by_name.setdefault(column.get("name"), []).append(column)
        target_columns = {
            column_name for column_name, _length in module.COLUMN_SPECS[table_name]
        }
        for column_name, expected_length in module.COLUMN_SPECS[table_name]:
            object_name = f"{table_name}.{column_name}"
            matching_columns = columns_by_name.get(column_name, [])
            if len(matching_columns) > 1:
                errors.append(
                    _summary(
                        revision_id,
                        "duplicate_column_name",
                        object_name,
                    )
                )
                continue
            if not matching_columns:
                repairs.append(_summary(revision_id, "missing_column", object_name))
                continue
            column = matching_columns[0]
            column_type = column.get("type")
            if (
                not _column_is_plain(column, primary_key_columns, column_name)
                or getattr(column_type, "__visit_name__", "").lower() != "varchar"
                or getattr(column_type, "length", None) != expected_length
                or column.get("nullable") is not True
            ):
                errors.append(_summary(revision_id, "column_shape", object_name))

        expected_indexes = dict(module.INDEX_SPECS[table_name])
        found_expected = set()
        for index in table_schema["indexes"]:
            name = index.get("name")
            columns = tuple(index.get("column_names") or ())
            unique = index.get("unique")
            object_name = f"{table_name}.{name or '<unnamed>'}"
            if name in expected_indexes:
                if name in found_expected:
                    errors.append(
                        _summary(
                            revision_id,
                            "duplicate_index_name",
                            object_name,
                        )
                    )
                found_expected.add(name)
                if columns != expected_indexes[name] or unique not in (False, 0):
                    errors.append(
                        _summary(
                            revision_id,
                            "index_shape",
                            object_name,
                        )
                    )
                continue
            if not target_columns.intersection(columns):
                continue
            if name is None:
                errors.append(_summary(revision_id, "unnamed_index", object_name))
            elif unique in (True, 1):
                errors.append(
                    _summary(
                        revision_id,
                        "unexpected_unique_index",
                        object_name,
                    )
                )
            elif unique not in (False, 0):
                errors.append(
                    _summary(
                        revision_id,
                        "unknown_index_uniqueness",
                        object_name,
                    )
                )
            else:
                errors.append(
                    _summary(
                        revision_id,
                        "unexpected_index",
                        object_name,
                    )
                )

        for constraint in table_schema["unique_constraints"]:
            columns = tuple(constraint.get("column_names") or ())
            if not target_columns.intersection(columns):
                continue
            name = constraint.get("name") or "<unnamed>"
            errors.append(
                _summary(
                    revision_id,
                    "unexpected_unique_constraint",
                    f"{table_name}.{name}",
                )
            )

        for index_name in expected_indexes:
            if index_name not in found_expected:
                repairs.append(
                    _summary(
                        revision_id,
                        "missing_index",
                        f"{table_name}.{index_name}",
                    )
                )


def _collect_standard_schema_plan(module, snapshot, repairs, errors):
    revision_id = module.revision
    if snapshot is None:
        errors.append(_summary(revision_id, "missing_base_table", module.TABLE_NAME))
        return
    try:
        module._validate_existing_schema(snapshot)
    except Exception:
        errors.append(
            _summary(
                revision_id,
                "schema_incompatible",
                f"{module.TABLE_NAME}.{module.COLUMN_NAME}",
            )
        )

    columns = snapshot["columns"]
    if module.COLUMN_NAME not in columns:
        repairs.append(
            _summary(
                revision_id,
                "missing_column",
                f"{module.TABLE_NAME}.{module.COLUMN_NAME}",
            )
        )

    index_name = getattr(module, "INDEX_NAME", None)
    if index_name is not None:
        raw_indexes = snapshot["indexes"]
        existing_names = (
            set(raw_indexes)
            if isinstance(raw_indexes, dict)
            else {item.get("name") for item in raw_indexes}
        )
        if index_name not in existing_names:
            repairs.append(
                _summary(
                    revision_id,
                    "missing_index",
                    f"{module.TABLE_NAME}.{index_name}",
                )
            )


def _historical_snapshot(module, bind, errors, **kwargs):
    try:
        return module._schema_snapshot(bind, **kwargs)
    except Exception:
        errors.append(
            _summary(
                module.revision,
                "historical_snapshot_failed",
                module.revision,
            )
        )
        return None


def _validate_049_raw_column(module, snapshot, repairs, errors):
    revision_id = module.revision
    object_name = f"{module.TABLE_NAME}.{module.COLUMN_NAME}"
    if snapshot is None:
        errors.append(_summary(revision_id, "missing_base_table", module.TABLE_NAME))
        return 0
    matching_columns = [
        column
        for column in snapshot["columns"]
        if column.get("name") == module.COLUMN_NAME
    ]
    if len(matching_columns) > 1:
        errors.append(_summary(revision_id, "duplicate_column_name", object_name))
        return len(matching_columns)
    if not matching_columns:
        repairs.append(_summary(revision_id, "missing_column", object_name))
        return 0

    column = matching_columns[0]
    primary_key_columns = _primary_key_columns(
        snapshot, revision_id, module.TABLE_NAME, errors
    )
    if primary_key_columns is None:
        return 1
    column_type = column.get("type")
    nullable = column.get("nullable")
    if (
        not _column_is_plain(column, primary_key_columns, module.COLUMN_NAME)
        or getattr(column_type, "__visit_name__", "").lower() != "varchar"
        or getattr(column_type, "length", None) != 64
        or nullable not in (True, False)
    ):
        errors.append(_summary(revision_id, "column_shape", object_name))
    elif nullable is True:
        repairs.append(_summary(revision_id, "nullable_column", object_name))
    return 1


def _build_private_049_plan(
    bind,
    module,
    snapshot,
    *,
    has_stored_column=None,
    lock_rows=False,
):
    required_columns = {
        "id",
        "edge_id",
        "from_dataset_id",
        "to_dataset_id",
        "relation_type",
    }
    if snapshot is None or not required_columns.issubset(snapshot["columns"]):
        return None
    legacy_plan = tuple(module._build_backfill_plan(bind))
    legacy_by_id = {}
    for operation, row_id, expected_key in legacy_plan:
        if operation != "update" or row_id in legacy_by_id:
            raise RuntimeError("Legacy lineage planner returned an invalid plan")
        legacy_by_id[row_id] = expected_key

    if has_stored_column is None:
        has_stored_column = module.COLUMN_NAME in snapshot["columns"]
    stored_expression = (
        module.COLUMN_NAME if has_stored_column else f"NULL AS {module.COLUMN_NAME}"
    )
    statement = (
        "SELECT id, from_dataset_id, to_dataset_id, relation_type, "
        f"{stored_expression} FROM {module.TABLE_NAME} ORDER BY id"
    )
    dialect_name = getattr(getattr(bind, "dialect", None), "name", "").lower()
    if lock_rows and dialect_name == "mysql":
        # This row/gap lock closes the MySQL DML race. Deployment still needs
        # publication quiescence and a real-MySQL runbook check around DDL.
        statement += " FOR UPDATE"
    rows = list(bind.execute(sa.text(statement)).mappings())
    rows_by_id = {}
    for row in rows:
        row_id = row["id"]
        if row_id in rows_by_id:
            raise RuntimeError("Lineage scan returned duplicate row identities")
        rows_by_id[row_id] = row
    if set(rows_by_id) != set(legacy_by_id):
        raise RuntimeError("Lineage identities changed between validation scans")

    rich_plan = []
    for row_id, row in rows_by_id.items():
        expected_key = module._build_edge_key(
            row["from_dataset_id"],
            row["to_dataset_id"],
            row["relation_type"],
        )
        if legacy_by_id[row_id] != expected_key:
            raise RuntimeError("Lineage source fields changed between validation scans")
        if row[module.COLUMN_NAME] == expected_key:
            continue
        rich_plan.append(
            {
                "operation": "update",
                "id": row_id,
                "edge_key": expected_key,
                "old_edge_key": row[module.COLUMN_NAME],
                "from_dataset_id": row["from_dataset_id"],
                "to_dataset_id": row["to_dataset_id"],
                "relation_type": row["relation_type"],
            }
        )
    return tuple(rich_plan)


def _apply_private_049_plan(bind, module, rich_plan):
    statement = sa.text(
        f"UPDATE {module.TABLE_NAME} SET {module.COLUMN_NAME} = :edge_key "
        "WHERE id = :id "
        f"AND (({module.COLUMN_NAME} = :old_edge_key) "
        f"OR ({module.COLUMN_NAME} IS NULL AND :old_edge_key IS NULL)) "
        "AND ((from_dataset_id = :from_dataset_id) "
        "OR (from_dataset_id IS NULL AND :from_dataset_id IS NULL)) "
        "AND ((to_dataset_id = :to_dataset_id) "
        "OR (to_dataset_id IS NULL AND :to_dataset_id IS NULL)) "
        "AND ((relation_type = :relation_type) "
        "OR (relation_type IS NULL AND :relation_type IS NULL))"
    )
    for item in rich_plan:
        if item["operation"] != "update":
            raise RuntimeError("Unsupported private lineage repair operation") from None
        result = bind.execute(
            statement,
            {
                "id": item["id"],
                "edge_key": item["edge_key"],
                "old_edge_key": item["old_edge_key"],
                "from_dataset_id": item["from_dataset_id"],
                "to_dataset_id": item["to_dataset_id"],
                "relation_type": item["relation_type"],
            },
        )
        if result.rowcount != 1:
            raise RuntimeError(
                "Conditional lineage repair did not match exactly one row"
            ) from None


def _validate_049_uniqueness(module, snapshot, repairs, errors):
    if snapshot is None:
        return
    revision_id = module.revision
    target_columns = (module.COLUMN_NAME,)
    relevant = []
    constraints_by_name = {}
    indexes_by_name = {}
    raw_constraints = list(snapshot["unique_constraints"])
    raw_indexes = list(snapshot["indexes"])

    constraint_seeds = {
        id(constraint)
        for constraint in raw_constraints
        if constraint.get("name") == module.CONSTRAINT_NAME
        or module.COLUMN_NAME in tuple(constraint.get("column_names") or ())
    }
    index_seeds = {
        id(index)
        for index in raw_indexes
        if index.get("name") == module.CONSTRAINT_NAME
        or (
            module.COLUMN_NAME in tuple(index.get("column_names") or ())
            and index.get("unique") not in (False, 0)
        )
    }
    included_constraints = set(constraint_seeds)
    included_indexes = set(index_seeds)
    while True:
        previous = (len(included_constraints), len(included_indexes))
        included_constraint_names = {
            constraint.get("name")
            for constraint in raw_constraints
            if id(constraint) in included_constraints
            and constraint.get("name") is not None
        }
        included_index_names = {
            index.get("name")
            for index in raw_indexes
            if id(index) in included_indexes and index.get("name") is not None
        }
        for constraint in raw_constraints:
            duplicate_name = constraint.get("duplicates_index")
            if id(constraint) in included_constraints and duplicate_name is not None:
                included_indexes.update(
                    id(index)
                    for index in raw_indexes
                    if index.get("name") == duplicate_name
                )
            if duplicate_name in included_index_names:
                included_constraints.add(id(constraint))
        for index in raw_indexes:
            duplicate_name = index.get("duplicates_constraint")
            if id(index) in included_indexes and duplicate_name is not None:
                included_constraints.update(
                    id(constraint)
                    for constraint in raw_constraints
                    if constraint.get("name") == duplicate_name
                )
            if duplicate_name in included_constraint_names:
                included_indexes.add(id(index))
        if previous == (len(included_constraints), len(included_indexes)):
            break

    for constraint in raw_constraints:
        if id(constraint) not in included_constraints:
            continue
        name = constraint.get("name")
        columns = tuple(constraint.get("column_names") or ())
        object_name = f"{module.TABLE_NAME}.{name or '<unnamed>'}"
        if name is None:
            errors.append(
                _summary(
                    revision_id,
                    "unnamed_lineage_uniqueness",
                    object_name,
                )
            )
        if columns != target_columns:
            errors.append(
                _summary(
                    revision_id,
                    "lineage_uniqueness_shape",
                    object_name,
                )
            )
        token = ("constraint", name, id(constraint))
        relevant.append((token, constraint))
        if name is not None:
            constraints_by_name.setdefault(name, []).append(token)

    for index in raw_indexes:
        if id(index) not in included_indexes:
            continue
        name = index.get("name")
        columns = tuple(index.get("column_names") or ())
        unique = index.get("unique")
        object_name = f"{module.TABLE_NAME}.{name or '<unnamed>'}"
        if name is None:
            errors.append(
                _summary(
                    revision_id,
                    "unnamed_lineage_uniqueness",
                    object_name,
                )
            )
        if unique not in (True, 1, False, 0):
            errors.append(
                _summary(
                    revision_id,
                    "unknown_lineage_uniqueness",
                    object_name,
                )
            )
        if columns != target_columns or unique not in (True, 1):
            errors.append(
                _summary(
                    revision_id,
                    "lineage_uniqueness_shape",
                    object_name,
                )
            )
        token = ("index", name, id(index))
        relevant.append((token, index))
        if name is not None:
            indexes_by_name.setdefault(name, []).append(token)

    parents = {token: token for token, _item in relevant}
    items_by_token = dict(relevant)

    def root(token):
        while parents[token] != token:
            parents[token] = parents[parents[token]]
            token = parents[token]
        return token

    def union(left, right):
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    same_type_name_collisions = {
        name for name, tokens in constraints_by_name.items() if len(tokens) > 1
    }
    same_type_name_collisions.update(
        name for name, tokens in indexes_by_name.items() if len(tokens) > 1
    )
    for name in sorted(same_type_name_collisions):
        errors.append(
            _summary(
                revision_id,
                "lineage_raw_name_collision",
                f"{module.TABLE_NAME}.{name}",
            )
        )

    valid_mirror_pairs = set()
    candidate_mirrors = []
    for constraint_token, constraint in relevant:
        if constraint_token[0] != "constraint":
            continue
        duplicate_name = constraint.get("duplicates_index")
        if duplicate_name is None:
            continue
        candidates = indexes_by_name.get(duplicate_name, ())
        valid_target = len(candidates) == 1
        if valid_target:
            index_token = candidates[0]
            index = items_by_token[index_token]
            index_backlink = index.get("duplicates_constraint")
            valid_target = (
                tuple(constraint.get("column_names") or ()) == target_columns
                and tuple(index.get("column_names") or ()) == target_columns
                and index.get("unique") in (True, 1)
                and index_backlink in (None, constraint_token[1])
            )
        if not valid_target:
            errors.append(
                _summary(
                    revision_id,
                    "lineage_mirror_reference",
                    f"{module.TABLE_NAME}.{constraint_token[1] or '<unnamed>'}",
                )
            )
            continue
        candidate_mirrors.append((constraint_token, index_token))

    mirror_claims = {}
    for constraint_token, index_token in candidate_mirrors:
        mirror_claims.setdefault(index_token, []).append(constraint_token)
    for constraint_token, index_token in candidate_mirrors:
        if len(mirror_claims[index_token]) != 1:
            errors.append(
                _summary(
                    revision_id,
                    "lineage_mirror_reference",
                    f"{module.TABLE_NAME}.{constraint_token[1] or '<unnamed>'}",
                )
            )
            continue
        union(constraint_token, index_token)
        valid_mirror_pairs.add(frozenset((constraint_token, index_token)))

    for index_token, index in relevant:
        if index_token[0] != "index":
            continue
        duplicate_name = index.get("duplicates_constraint")
        if duplicate_name is None:
            continue
        candidates = constraints_by_name.get(duplicate_name, ())
        valid_source = len(candidates) == 1
        if valid_source:
            constraint_token = candidates[0]
            constraint = items_by_token[constraint_token]
            valid_source = (
                constraint.get("duplicates_index") == index_token[1]
                and tuple(constraint.get("column_names") or ()) == target_columns
                and tuple(index.get("column_names") or ()) == target_columns
                and index.get("unique") in (True, 1)
            )
        if not valid_source:
            errors.append(
                _summary(
                    revision_id,
                    "lineage_mirror_reference",
                    f"{module.TABLE_NAME}.{index_token[1] or '<unnamed>'}",
                )
            )

    cross_type_name_collisions = set(constraints_by_name).intersection(indexes_by_name)
    for name in sorted(cross_type_name_collisions):
        constraint_tokens = constraints_by_name[name]
        index_tokens = indexes_by_name[name]
        allowed_pair = (
            len(constraint_tokens) == 1
            and len(index_tokens) == 1
            and frozenset((constraint_tokens[0], index_tokens[0])) in valid_mirror_pairs
        )
        if allowed_pair:
            continue
        errors.append(
            _summary(
                revision_id,
                "lineage_raw_name_collision",
                f"{module.TABLE_NAME}.{name}",
            )
        )

    logical_groups = {}
    for token in parents:
        logical_groups.setdefault(root(token), []).append(token)
    for tokens in logical_groups.values():
        if len(tokens) == 1:
            canonical = tokens[0][1] == module.CONSTRAINT_NAME
        else:
            canonical = any(
                token[0] == "constraint" and token[1] == module.CONSTRAINT_NAME
                for token in tokens
            )
        if not canonical:
            errors.append(
                _summary(
                    revision_id,
                    "noncanonical_lineage_uniqueness",
                    f"{module.TABLE_NAME}.{module.COLUMN_NAME}",
                )
            )

    logical_objects = set(logical_groups)
    if len(logical_objects) > 1:
        errors.append(
            _summary(
                revision_id,
                "multiple_lineage_uniqueness",
                f"{module.TABLE_NAME}.{module.COLUMN_NAME}",
                count=len(logical_objects),
            )
        )
    managed_column_present = any(
        column.get("name") == module.COLUMN_NAME for column in snapshot["columns"]
    )
    if not logical_objects and managed_column_present:
        repairs.append(
            _summary(
                revision_id,
                "missing_unique_constraint",
                f"{module.TABLE_NAME}.{module.CONSTRAINT_NAME}",
            )
        )


def _collect_049_plan(
    bind,
    module,
    repair_snapshot,
    raw_snapshot,
    repairs,
    errors,
):
    revision_id = module.revision
    managed_column_count = _validate_049_raw_column(
        module,
        raw_snapshot,
        repairs,
        errors,
    )
    if repair_snapshot is None or raw_snapshot is None:
        return (), _digest(())
    _validate_049_uniqueness(module, raw_snapshot, repairs, errors)

    required_columns = {
        "id",
        "edge_id",
        "from_dataset_id",
        "to_dataset_id",
        "relation_type",
    }
    missing = sorted(required_columns - set(repair_snapshot["columns"]))
    for column_name in missing:
        errors.append(
            _summary(
                revision_id,
                "missing_base_column",
                f"{module.TABLE_NAME}.{column_name}",
            )
        )
    if missing:
        return (), _digest(())

    try:
        private_plan = tuple(
            _build_private_049_plan(
                bind,
                module,
                repair_snapshot,
                has_stored_column=managed_column_count == 1,
            )
            or ()
        )
    except Exception:
        errors.append(
            _summary(
                revision_id,
                "lineage_data_incompatible",
                module.TABLE_NAME,
            )
        )
        return (), _digest(())

    plan_digest = _digest(private_plan)
    if private_plan:
        if managed_column_count <= 1:
            category = (
                "stored_lineage_key_mismatch"
                if managed_column_count == 1
                else "lineage_key_backfill"
            )
            repairs.append(
                _summary(
                    revision_id,
                    category,
                    f"{module.TABLE_NAME}.{module.COLUMN_NAME}",
                )
            )
    return private_plan, plan_digest


def _build_preflight(bind):
    modules = _load_lifecycle_migrations()
    migration_048, migration_049, migration_050, migration_051, migration_052 = modules
    repairs = []
    errors = []

    _historical_snapshot(
        migration_048,
        bind,
        errors,
        require_table=False,
    )
    snapshot_049 = _historical_snapshot(
        migration_049,
        bind,
        errors,
        require_table=False,
    )
    _historical_snapshot(
        migration_050,
        bind,
        errors,
        require_table=False,
    )
    _historical_snapshot(
        migration_051,
        bind,
        errors,
        require_table=False,
    )
    _historical_snapshot(
        migration_052,
        bind,
        errors,
        require_all_tables=False,
    )
    raw_snapshot_049 = _snapshot_standard_raw(migration_049, bind, errors)
    snapshots = {
        migration_048.revision: _snapshot_standard_raw(
            migration_048,
            bind,
            errors,
        ),
        migration_049.revision: snapshot_049,
        migration_050.revision: _snapshot_standard_raw(
            migration_050,
            bind,
            errors,
        ),
        migration_051.revision: _snapshot_standard_raw(
            migration_051,
            bind,
            errors,
        ),
        migration_052.revision: _snapshot_052(migration_052, bind, errors),
    }

    _validate_standard_raw(
        migration_048,
        snapshots[migration_048.revision],
        repairs,
        errors,
    )
    if raw_snapshot_049 is _RAW_SNAPSHOT_FAILED:
        _private_049 = ()
        private_049_digest = _digest(())
    else:
        _private_049, private_049_digest = _collect_049_plan(
            bind,
            migration_049,
            snapshots[migration_049.revision],
            raw_snapshot_049,
            repairs,
            errors,
        )
    _validate_standard_raw(
        migration_050,
        snapshots[migration_050.revision],
        repairs,
        errors,
    )
    _validate_standard_raw(
        migration_051,
        snapshots[migration_051.revision],
        repairs,
        errors,
    )
    _validate_052(
        migration_052,
        snapshots[migration_052.revision],
        repairs,
        errors,
    )

    repairs = tuple(repairs)
    errors = tuple(errors)
    public_plan = {
        "repairs": repairs,
        "errors": errors,
        "repair_count": len(repairs),
    }
    private_fingerprint = (len(_private_049), private_049_digest)
    return public_plan, private_fingerprint


def build_preflight_plan(bind=None):
    """Return a read-only, data-safe summary for lifecycle repair."""

    if bind is None:
        bind = op.get_bind()
    public_plan, _private_fingerprint = _build_preflight(bind)
    return public_plan


def _raise_preflight_errors(plan, *, phase):
    if plan["errors"]:
        summary = ", ".join(
            f"{item['revision']}:{item['category']}:{item['object_name']}"
            for item in plan["errors"]
        )
        raise RuntimeError(f"Lifecycle {phase} failed: {summary}") from None


def upgrade() -> None:
    bind = op.get_bind()
    plan, initial_private_fingerprint = _build_preflight(bind)
    _raise_preflight_errors(plan, phase="preflight")

    modules = _load_lifecycle_migrations()
    migration_048, migration_049, migration_050, migration_051, migration_052 = modules
    try:
        snapshot_049 = migration_049._schema_snapshot(
            bind,
            require_table=False,
        )
        private_049_plan = tuple(
            _build_private_049_plan(
                bind,
                migration_049,
                snapshot_049,
                lock_rows=True,
            )
            or ()
        )
    except Exception:
        raise RuntimeError(
            "Lifecycle private lineage recheck failed: lineage_data_incompatible"
        ) from None
    current_private_fingerprint = (
        len(private_049_plan),
        _digest(private_049_plan),
    )
    if current_private_fingerprint != initial_private_fingerprint:
        raise RuntimeError(
            "Lifecycle private lineage recheck failed: plan_changed"
        ) from None

    revisions_with_repairs = {item["revision"] for item in plan["repairs"]}
    lineage_repairs = tuple(
        item for item in plan["repairs"] if item["revision"] == migration_049.revision
    )
    lineage_data_only = bool(lineage_repairs) and all(
        item["category"] == "stored_lineage_key_mismatch" for item in lineage_repairs
    )
    repair_order = (
        migration_049,
        migration_048,
        migration_050,
        migration_051,
        migration_052,
    )
    original_operations = {module: module.op for module in modules}
    original_049_builder = migration_049._build_backfill_plan
    original_049_apply = migration_049._apply_backfill_plan
    legacy_049_plan = [
        (item["operation"], item["id"], item["edge_key"]) for item in private_049_plan
    ]
    try:
        for module in modules:
            module.op = op
        if migration_049.revision in revisions_with_repairs:
            migration_049._build_backfill_plan = lambda actual_bind: list(
                legacy_049_plan
            )
            migration_049._apply_backfill_plan = (
                lambda actual_bind, _legacy_plan: _apply_private_049_plan(
                    actual_bind,
                    migration_049,
                    private_049_plan,
                )
            )
        for module in repair_order:
            if module.revision not in revisions_with_repairs:
                continue
            try:
                if module is migration_049 and lineage_data_only:
                    _apply_private_049_plan(
                        bind,
                        migration_049,
                        private_049_plan,
                    )
                else:
                    module.upgrade()
            except Exception:
                raise RuntimeError(
                    f"Lifecycle repair failed for {module.revision}"
                ) from None
    finally:
        migration_049._build_backfill_plan = original_049_builder
        migration_049._apply_backfill_plan = original_049_apply
        for module, historical_op in original_operations.items():
            module.op = historical_op

    final_plan = build_preflight_plan(bind)
    _raise_preflight_errors(final_plan, phase="postflight")
    if final_plan["repairs"]:
        summary = ", ".join(
            f"{item['revision']}:{item['category']}:{item['object_name']}"
            for item in final_plan["repairs"]
        )
        raise RuntimeError(f"Lifecycle postflight incomplete: {summary}") from None


def downgrade() -> None:
    pass
