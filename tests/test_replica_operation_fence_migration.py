from __future__ import annotations

from contextlib import contextmanager
import importlib

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
import sqlalchemy as sa


MIGRATION = importlib.import_module(
    "train_factory.storage.migrations.versions.057_add_replica_operation_fence"
)


def _canonical_column(name: str) -> dict[str, object]:
    column_type, nullable, default = MIGRATION.COLUMNS[name]
    return {
        "name": name,
        "type": column_type,
        "nullable": nullable,
        "default": default,
    }


class _Inspector:
    def __init__(self, columns: dict[str, dict[str, object]]) -> None:
        self._columns = columns

    def get_table_names(self) -> list[str]:
        return [MIGRATION.TABLE_NAME]

    def get_columns(self, table_name: str) -> list[dict[str, object]]:
        assert table_name == MIGRATION.TABLE_NAME
        return list(self._columns.values())


class _BatchOperations:
    def __init__(
        self,
        columns: dict[str, dict[str, object]],
        added_columns: list[str],
    ) -> None:
        self._columns = columns
        self._added_columns = added_columns

    def add_column(self, column: sa.Column) -> None:
        self._added_columns.append(column.name)
        server_default = column.server_default
        self._columns[column.name] = {
            "name": column.name,
            "type": column.type,
            "nullable": column.nullable,
            "default": None if server_default is None else str(server_default.arg),
        }


class _Operations:
    def __init__(self, columns: dict[str, dict[str, object]]) -> None:
        self._columns = columns
        self.added_columns: list[str] = []
        self.bind = object()

    def get_bind(self) -> object:
        return self.bind

    @contextmanager
    def batch_alter_table(self, table_name: str):
        assert table_name == MIGRATION.TABLE_NAME
        yield _BatchOperations(self._columns, self.added_columns)


def _install_migration_boundary(
    monkeypatch: pytest.MonkeyPatch,
    columns: dict[str, dict[str, object]],
) -> _Operations:
    operations = _Operations(columns)
    monkeypatch.setattr(MIGRATION, "op", operations)
    monkeypatch.setattr(
        MIGRATION,
        "inspect",
        lambda bind: _Inspector(columns) if bind is operations.bind else None,
    )
    return operations


def _malformed_cases() -> list[object]:
    cases: list[object] = []
    for name, (column_type, nullable, default) in MIGRATION.COLUMNS.items():
        wrong_type: sa.types.TypeEngine
        if isinstance(column_type, sa.String):
            wrong_type = sa.Integer()
        elif isinstance(column_type, sa.Integer):
            wrong_type = sa.BigInteger()
        else:
            wrong_type = sa.Date()
        cases.append(pytest.param(name, "type", wrong_type, id=f"{name}-type"))
        cases.append(
            pytest.param(
                name,
                "nullable",
                not nullable,
                id=f"{name}-nullable",
            )
        )
        cases.append(
            pytest.param(
                name,
                "default",
                "1" if default == "0" else "0",
                id=f"{name}-default",
            )
        )
        if isinstance(column_type, sa.String):
            cases.append(
                pytest.param(
                    name,
                    "type",
                    sa.String(length=column_type.length + 1),
                    id=f"{name}-length",
                )
            )
    return cases


@pytest.mark.parametrize("column_name, field, wrong_value", _malformed_cases())
def test_upgrade_rejects_a_malformed_present_column_before_any_add(
    monkeypatch: pytest.MonkeyPatch,
    column_name: str,
    field: str,
    wrong_value: object,
) -> None:
    columns = {column_name: _canonical_column(column_name)}
    columns[column_name][field] = wrong_value
    operations = _install_migration_boundary(monkeypatch, columns)

    with pytest.raises(
        RuntimeError,
        match=rf"deployments\.{column_name} shape is invalid",
    ):
        MIGRATION.upgrade()

    assert operations.added_columns == []


def test_upgrade_completes_a_clean_partial_shape_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    present = {
        "replica_operation_token",
        "replica_operation_generation",
        "replica_operation_heartbeat_at",
    }
    columns = {name: _canonical_column(name) for name in present}
    operations = _install_migration_boundary(monkeypatch, columns)
    expected_missing = [name for name in MIGRATION.COLUMNS if name not in present]

    MIGRATION.upgrade()

    assert operations.added_columns == expected_missing
    assert set(columns) == set(MIGRATION.COLUMNS)
    for name, (expected_type, nullable, default) in MIGRATION.COLUMNS.items():
        column = columns[name]
        assert type(column["type"]) is type(expected_type)
        assert getattr(column["type"], "length", None) == getattr(
            expected_type,
            "length",
            None,
        )
        assert column["nullable"] is nullable
        assert column["default"] == default

    MIGRATION.upgrade()

    assert operations.added_columns == expected_missing


def test_upgrade_is_idempotent_with_real_sqlite_reflection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE deployments ("
                "id INTEGER PRIMARY KEY, deployment_id VARCHAR(36) NOT NULL UNIQUE)"
            )
        )
        context = MigrationContext.configure(connection)
        monkeypatch.setattr(MIGRATION, "op", Operations(context))

        MIGRATION.upgrade()
        MIGRATION.upgrade()

        columns = {
            item["name"]: item
            for item in sa.inspect(connection).get_columns(MIGRATION.TABLE_NAME)
        }
        for name, (expected_type, nullable, default) in MIGRATION.COLUMNS.items():
            column = columns[name]
            actual_type = column["type"]
            actual_visit_name = getattr(actual_type, "__visit_name__", "").lower()
            expected_visit_name = getattr(
                expected_type,
                "__visit_name__",
                "",
            ).lower()
            if isinstance(expected_type, sa.String):
                assert actual_visit_name == "varchar"
                assert actual_type.length == expected_type.length
            else:
                assert actual_visit_name == expected_visit_name
            assert column["nullable"] is nullable
            assert MIGRATION._normalize_default(column["default"]) == default
