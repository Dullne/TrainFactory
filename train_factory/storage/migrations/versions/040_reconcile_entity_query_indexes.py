"""Reconcile entity uniqueness constraints and query indexes.

Revision ID: 040_reconcile_entity_query_indexes
Revises: 039_add_user_token_version
"""

from alembic import op
from sqlalchemy import inspect, text


revision = "040_reconcile_entity_query_indexes"
down_revision = "039_add_user_token_version"
branch_labels = None
depends_on = None


QUERY_INDEXES = (
    (
        "training_tasks",
        "idx_task_type_status",
        ("model_type", "status"),
    ),
    (
        "model_registry",
        "ix_model_registry_source_task_id",
        ("source_task_id",),
    ),
    (
        "model_registry",
        "idx_model_user_created",
        ("user_id", "created_at"),
    ),
    (
        "model_registry",
        "idx_model_type_status",
        ("model_type", "status"),
    ),
)

UNIQUE_CONSTRAINTS = (
    (
        "training_tasks",
        "uq_task_user_name",
        ("user_id", "task_name"),
    ),
    (
        "model_registry",
        "uq_model_user_name_version",
        ("user_id", "model_name", "version"),
    ),
    (
        "model_versions",
        "uq_version_model_version",
        ("model_id", "version"),
    ),
    (
        "deployments",
        "uq_deployment_user_name",
        ("user_id", "deployment_name"),
    ),
    (
        "model_configs",
        "uq_config_user_name",
        ("user_id", "config_name"),
    ),
    (
        "datasets",
        "uq_dataset_user_name",
        ("user_id", "dataset_name"),
    ),
)


def _has_columns(definitions, columns: tuple[str, ...]) -> bool:
    return any(tuple(item.get("column_names") or ()) == columns for item in definitions)


def _create_query_index(
    bind,
    index_name: str,
    table_name: str,
    columns: tuple[str, ...] | list[str],
    unique: bool = False,
    **_mysql_options,
) -> None:
    if bind.dialect.name != "mysql":
        op.create_index(index_name, table_name, list(columns), unique=unique)
        return
    quote = bind.dialect.identifier_preparer.quote
    quoted_columns = ", ".join(quote(column) for column in columns)
    op.execute(
        text(
            f"ALTER TABLE {quote(table_name)} ADD INDEX {quote(index_name)} "
            f"({quoted_columns}), ALGORITHM=INPLACE, LOCK=NONE"
        )
    )


def _assert_no_duplicate_rows(
    bind,
    table_name: str,
    columns: tuple[str, ...],
    constraint_name: str,
) -> None:
    quote = bind.dialect.identifier_preparer.quote
    quoted_table = quote(table_name)
    quoted_columns = tuple(quote(column) for column in columns)
    column_list = ", ".join(quoted_columns)
    non_null_filter = " AND ".join(
        f"{column} IS NOT NULL" for column in quoted_columns
    )
    rows = bind.execute(
        text(
            f"SELECT {column_list}, COUNT(*) AS duplicate_count "
            f"FROM {quoted_table} "
            f"WHERE {non_null_filter} "
            f"GROUP BY {column_list} "
            "HAVING COUNT(*) > 1 LIMIT 5"
        )
    ).fetchall()
    if rows:
        raise RuntimeError(
            f"Cannot create {constraint_name} on {table_name}{columns}: "
            f"duplicate rows exist (up to 5 shown): {rows!r}"
        )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    table_names = set(inspector.get_table_names())

    for table_name, index_name, columns in QUERY_INDEXES:
        if table_name not in table_names:
            continue
        if not _has_columns(inspector.get_indexes(table_name), columns):
            _create_query_index(
                bind,
                index_name,
                table_name,
                list(columns),
                unique=False,
                # MySQL 8.0：INPLACE + NONE 锁，避免大表启动迁移长时间阻塞写入
                mysql_algorithm="INPLACE",
                mysql_lock="NONE",
            )

    for table_name, constraint_name, columns in UNIQUE_CONSTRAINTS:
        if table_name not in table_names:
            continue
        existing = inspector.get_unique_constraints(table_name)
        if _has_columns(existing, columns):
            continue
        _assert_no_duplicate_rows(bind, table_name, columns, constraint_name)
        op.create_unique_constraint(constraint_name, table_name, list(columns))


def downgrade() -> None:
    # Some indexes and constraints may have been adopted from init.sql. Without
    # provenance metadata, dropping them would weaken schemas created from the
    # current baseline, so downgrade intentionally preserves them.
    pass
