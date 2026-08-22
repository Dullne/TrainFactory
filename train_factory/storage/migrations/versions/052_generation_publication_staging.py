"""Add attempt-owned staging metadata for generated publication products.

Revision ID: 052_generation_publication_staging
Revises: 051_add_generation_run_token
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "052_generation_publication_staging"
down_revision = "051_add_generation_run_token"
branch_labels = None
depends_on = None


COLUMN_SPECS = {
    "datasets": (
        ("generation_run_token", 36),
    ),
    "milvus_collections": (
        ("generation_task_id", 36),
        ("generation_run_token", 36),
    ),
    "collection_dataset_links": (
        ("generation_run_token", 36),
    ),
}

INDEX_SPECS = {
    "datasets": (
        (
            "idx_dataset_generation_run_token",
            ("generation_run_token",),
        ),
    ),
    "milvus_collections": (
        (
            "idx_milvus_generation_attempt",
            ("generation_task_id", "generation_run_token"),
        ),
    ),
    "collection_dataset_links": (
        (
            "idx_coll_link_generation_run_token",
            ("generation_run_token",),
        ),
    ),
}


def _schema_snapshot(bind, *, require_all_tables: bool):
    inspector = inspect(bind)
    available_tables = set(inspector.get_table_names())
    missing_tables = sorted(set(COLUMN_SPECS) - available_tables)
    if require_all_tables and missing_tables:
        raise RuntimeError(
            "Required generation publication tables are missing: "
            + ", ".join(missing_tables)
        )

    snapshot = {}
    for table_name in COLUMN_SPECS:
        if table_name not in available_tables:
            continue
        reflected_columns = inspector.get_columns(table_name)
        get_primary_key = getattr(inspector, "get_pk_constraint", None)
        if get_primary_key is None:
            if any(
                "primary_key" not in column
                or column["primary_key"] not in (False, 0, True, 1)
                for column in reflected_columns
            ):
                raise RuntimeError(
                    "Generation publication primary key metadata is invalid"
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
                primary_key = get_primary_key(table_name)
            except Exception:
                raise RuntimeError(
                    "Generation publication primary key metadata is invalid"
                ) from None
        primary_key_columns = (
            primary_key.get("constrained_columns")
            if isinstance(primary_key, dict)
            else None
        )
        if (
            not isinstance(primary_key_columns, (list, tuple))
            or any(
                not isinstance(name, str) or not name
                for name in primary_key_columns
            )
            or len(set(primary_key_columns)) != len(primary_key_columns)
            or not set(primary_key_columns).issubset(
                {column.get("name") for column in reflected_columns}
            )
        ):
            raise RuntimeError(
                "Generation publication primary key metadata is invalid"
            )
        snapshot[table_name] = {
            "columns": {column["name"]: column for column in reflected_columns},
            "primary_key_columns": tuple(primary_key_columns),
            "indexes": {
                index["name"]: index
                for index in inspector.get_indexes(table_name)
            },
        }
    return snapshot


def _validate_existing_schema(snapshot) -> None:
    for table_name, table_schema in snapshot.items():
        existing_columns = table_schema["columns"]
        for column_name, expected_length in COLUMN_SPECS[table_name]:
            column = existing_columns.get(column_name)
            if column is None:
                continue
            primary_key_columns = table_schema.get("primary_key_columns")
            marker = column.get("primary_key")
            if primary_key_columns is None:
                if marker not in (False, 0):
                    raise RuntimeError(
                        "Generation publication primary key metadata is invalid"
                    )
                primary_key_columns = ()
            if not set(primary_key_columns).issubset(existing_columns):
                raise RuntimeError(
                    "Generation publication primary key metadata is invalid"
                )
            if (
                marker not in (None, False, 0)
                or column_name in primary_key_columns
            ):
                raise RuntimeError(
                    "Refusing generation publication migration: existing "
                    f"column {table_name}.{column_name} must not be a primary key"
                )
            column_type = column.get("type")
            visit_name = getattr(column_type, "__visit_name__", "").lower()
            if (
                visit_name != "varchar"
                or getattr(column_type, "length", None) != expected_length
                or column.get("nullable") is not True
            ):
                raise RuntimeError(
                    "Refusing generation publication migration: existing "
                    f"column {table_name}.{column_name} must be nullable "
                    f"VARCHAR({expected_length})"
                )

        existing_indexes = table_schema["indexes"]
        for index_name, expected_columns in INDEX_SPECS[table_name]:
            index = existing_indexes.get(index_name)
            if index is None:
                continue
            actual_columns = tuple(index.get("column_names") or ())
            unique = index.get("unique")
            if actual_columns != expected_columns or unique not in (False, 0):
                raise RuntimeError(
                    "Refusing generation publication migration: existing "
                    f"index {table_name}.{index_name} must be a non-unique "
                    f"index on {expected_columns!r} in that exact order"
                )


def upgrade() -> None:
    bind = op.get_bind()
    snapshot = _schema_snapshot(bind, require_all_tables=True)
    _validate_existing_schema(snapshot)

    for table_name, column_specs in COLUMN_SPECS.items():
        existing_columns = snapshot[table_name]["columns"]
        for column_name, length in column_specs:
            if column_name in existing_columns:
                continue
            op.add_column(
                table_name,
                sa.Column(
                    column_name,
                    sa.String(length=length),
                    nullable=True,
                ),
            )

    for table_name, index_specs in INDEX_SPECS.items():
        existing_indexes = snapshot[table_name]["indexes"]
        for index_name, column_names in index_specs:
            if index_name in existing_indexes:
                continue
            op.create_index(
                index_name,
                table_name,
                list(column_names),
                unique=False,
            )


def downgrade() -> None:
    bind = op.get_bind()
    snapshot = _schema_snapshot(bind, require_all_tables=False)
    _validate_existing_schema(snapshot)

    for table_name in reversed(tuple(COLUMN_SPECS)):
        if table_name not in snapshot:
            continue

        existing_indexes = snapshot[table_name]["indexes"]
        for index_name, _column_names in INDEX_SPECS[table_name]:
            if index_name in existing_indexes:
                op.drop_index(index_name, table_name=table_name)

        existing_columns = snapshot[table_name]["columns"]
        for column_name, _length in reversed(COLUMN_SPECS[table_name]):
            if column_name in existing_columns:
                op.drop_column(table_name, column_name)
