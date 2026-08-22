"""Alembic version-table compatibility helpers."""

from sqlalchemy import (
    Column,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    inspect,
    text,
)


VERSION_NUM_LENGTH = 64


def ensure_version_table_capacity(connection) -> None:
    """Create or widen Alembic's version table for long revision identifiers."""
    inspector = inspect(connection)
    if "alembic_version" not in inspector.get_table_names():
        Table(
            "alembic_version",
            MetaData(),
            Column("version_num", String(VERSION_NUM_LENGTH), nullable=False),
            PrimaryKeyConstraint("version_num", name="alembic_version_pkc"),
        ).create(connection)
        return

    version_column = next(
        (
            column
            for column in inspector.get_columns("alembic_version")
            if column["name"] == "version_num"
        ),
        None,
    )
    length = getattr(version_column["type"], "length", None) if version_column else None
    if isinstance(length, int) and length < VERSION_NUM_LENGTH:
        connection.execute(
            text(
                "ALTER TABLE alembic_version "
                f"MODIFY version_num VARCHAR({VERSION_NUM_LENGTH}) NOT NULL"
            )
        )
