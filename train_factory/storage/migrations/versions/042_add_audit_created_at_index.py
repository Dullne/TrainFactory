"""Add an audit timestamp index for retention cleanup.

Revision ID: 042_add_audit_created_at_index
Revises: 041_set_dataset_status_default
"""

from alembic import op
from sqlalchemy import inspect, text


revision = "042_add_audit_created_at_index"
down_revision = "041_set_dataset_status_default"
branch_labels = None
depends_on = None

INDEX_NAME = "idx_audit_created_at"


def _create_audit_index(
    bind,
    index_name: str,
    table_name: str,
    columns: list[str],
    unique: bool = False,
    **_mysql_options,
) -> None:
    if bind.dialect.name != "mysql":
        op.create_index(index_name, table_name, columns, unique=unique)
        return
    quote = bind.dialect.identifier_preparer.quote
    quoted_columns = ", ".join(quote(column) for column in columns)
    op.execute(
        text(
            f"ALTER TABLE {quote(table_name)} ADD INDEX {quote(index_name)} "
            f"({quoted_columns}), ALGORITHM=INPLACE, LOCK=NONE"
        )
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "audit_logs" not in inspector.get_table_names():
        return
    if any(
        tuple(index.get("column_names") or ()) == ("created_at",)
        for index in inspector.get_indexes("audit_logs")
    ):
        return
    _create_audit_index(
        bind,
        INDEX_NAME,
        "audit_logs",
        ["created_at"],
        unique=False,
        # MySQL 8.0：INPLACE + NONE 锁，audit_logs 是持续写入的高增长表
        mysql_algorithm="INPLACE",
        mysql_lock="NONE",
    )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if "audit_logs" not in inspector.get_table_names():
        return
    if any(
        index.get("name") == INDEX_NAME
        for index in inspector.get_indexes("audit_logs")
    ):
        op.drop_index(INDEX_NAME, table_name="audit_logs")
