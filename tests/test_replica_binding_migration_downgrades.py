from __future__ import annotations

import importlib
from contextlib import contextmanager

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_055 = importlib.import_module(
    "train_factory.storage.migrations.versions.055_bind_configs_and_adapters_to_replicas"
)
MIGRATION_056 = importlib.import_module(
    "train_factory.storage.migrations.versions.056_bind_sync_targets_to_replicas"
)


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


class _RecordingBatchOperations:
    def __init__(self, table_name: str, delegate, mutation_calls: list[tuple]) -> None:
        self._table_name = table_name
        self._delegate = delegate
        self._mutation_calls = mutation_calls

    def add_column(self, column: sa.Column) -> None:
        self._mutation_calls.append(
            (
                self._table_name,
                "add_column",
                column.name,
                getattr(column.type, "length", None),
                column.nullable,
            )
        )
        self._delegate.add_column(column)

    def create_index(
        self,
        name: str,
        columns: list[str],
        *,
        unique: bool,
    ) -> None:
        self._mutation_calls.append(
            (self._table_name, "create_index", name, tuple(columns), unique)
        )
        self._delegate.create_index(name, columns, unique=unique)

    def create_foreign_key(
        self,
        name: str,
        referent_table: str,
        local_columns: list[str],
        remote_columns: list[str],
        *,
        ondelete: str,
    ) -> None:
        self._mutation_calls.append(
            (
                self._table_name,
                "create_foreign_key",
                name,
                referent_table,
                tuple(local_columns),
                tuple(remote_columns),
                ondelete,
            )
        )
        self._delegate.create_foreign_key(
            name,
            referent_table,
            local_columns,
            remote_columns,
            ondelete=ondelete,
        )

    def drop_constraint(self, name: str, *, type_: str) -> None:
        self._mutation_calls.append(
            (self._table_name, "drop_constraint", name, type_)
        )
        self._delegate.drop_constraint(name, type_=type_)

    def drop_index(self, name: str) -> None:
        self._mutation_calls.append((self._table_name, "drop_index", name))
        self._delegate.drop_index(name)

    def drop_column(self, name: str) -> None:
        self._mutation_calls.append((self._table_name, "drop_column", name))
        self._delegate.drop_column(name)


class _RecordingOperations:
    def __init__(self, connection: sa.Connection) -> None:
        self._delegate = _operations(connection)
        self.mutation_calls: list[tuple] = []

    def get_bind(self) -> sa.Connection:
        return self._delegate.get_bind()

    @contextmanager
    def batch_alter_table(self, table_name: str):
        with self._delegate.batch_alter_table(table_name) as batch_op:
            yield _RecordingBatchOperations(
                table_name,
                batch_op,
                self.mutation_calls,
            )


def _create_replica_binding_targets(connection: sa.Connection, migration) -> None:
    connection.execute(
        sa.text(
            "CREATE TABLE deployment_replicas ("
            "replica_id VARCHAR(36) PRIMARY KEY)"
        )
    )
    for table_name in migration.TARGETS:
        connection.execute(
            sa.text(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY)")
        )


def _assert_mixed_binding_downgrade_has_no_mutations(
    monkeypatch: pytest.MonkeyPatch,
    migration,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_replica_binding_targets(connection, migration)
        monkeypatch.setattr(migration, "op", _operations(connection))
        migration.upgrade()

        later_table = next(iter(migration.TARGETS))
        connection.execute(
            sa.text(
                f"INSERT INTO {later_table} (id, {migration.COLUMN_NAME}) "
                "VALUES (1, 'replica-bound')"
            )
        )
        recording_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", recording_operations)

        with pytest.raises(
            RuntimeError,
            match="Refusing replica binding downgrade while explicit bindings exist",
        ):
            migration.downgrade()

        assert recording_operations.mutation_calls == []


def _expected_drop_calls(migration) -> list[tuple]:
    calls: list[tuple] = []
    for table_name in reversed(tuple(migration.TARGETS)):
        index_name, foreign_key_name = migration.TARGETS[table_name]
        calls.extend(
            [
                (
                    table_name,
                    "drop_constraint",
                    foreign_key_name,
                    "foreignkey",
                ),
                (table_name, "drop_index", index_name),
                (table_name, "drop_column", migration.COLUMN_NAME),
            ]
        )
    return calls


def _expected_add_calls(migration, table_names=None) -> list[tuple]:
    calls: list[tuple] = []
    selected = migration.TARGETS if table_names is None else table_names
    for table_name in selected:
        index_name, foreign_key_name = migration.TARGETS[table_name]
        calls.extend(
            [
                (table_name, "add_column", migration.COLUMN_NAME, 36, True),
                (
                    table_name,
                    "create_index",
                    index_name,
                    (migration.COLUMN_NAME,),
                    False,
                ),
                (
                    table_name,
                    "create_foreign_key",
                    foreign_key_name,
                    "deployment_replicas",
                    (migration.COLUMN_NAME,),
                    ("replica_id",),
                    "SET NULL",
                ),
            ]
        )
    return calls


def _drop_binding_column(connection: sa.Connection, migration, table_name: str) -> None:
    index_name, foreign_key_name = migration.TARGETS[table_name]
    with _operations(connection).batch_alter_table(table_name) as batch_op:
        batch_op.drop_constraint(foreign_key_name, type_="foreignkey")
        batch_op.drop_index(index_name)
        batch_op.drop_column(migration.COLUMN_NAME)


@pytest.mark.parametrize(
    "migration",
    [MIGRATION_055, MIGRATION_056],
    ids=["055", "056"],
)
def test_upgrade_preflights_malformed_later_table_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
    migration,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_replica_binding_targets(connection, migration)
        monkeypatch.setattr(migration, "op", _operations(connection))
        migration.upgrade()

        first_table, later_table = tuple(migration.TARGETS)
        _drop_binding_column(connection, migration, first_table)
        index_name, _foreign_key_name = migration.TARGETS[later_table]
        connection.execute(sa.text(f"DROP INDEX {index_name}"))
        connection.execute(
            sa.text(f"CREATE UNIQUE INDEX {index_name} ON {later_table} (id)")
        )
        recording_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", recording_operations)

        with pytest.raises(
            RuntimeError,
            match=rf"{later_table}\.{migration.COLUMN_NAME} index is invalid",
        ):
            migration.upgrade()

        assert recording_operations.mutation_calls == []


@pytest.mark.parametrize(
    "migration",
    [MIGRATION_055, MIGRATION_056],
    ids=["055", "056"],
)
def test_upgrade_preflights_conflicting_later_index_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
    migration,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_replica_binding_targets(connection, migration)
        later_table = tuple(migration.TARGETS)[-1]
        index_name, _foreign_key_name = migration.TARGETS[later_table]
        connection.execute(
            sa.text(f"CREATE INDEX {index_name} ON {later_table} (id)")
        )
        recording_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", recording_operations)

        with pytest.raises(
            RuntimeError,
            match=rf"{later_table}\.{migration.COLUMN_NAME} index is invalid",
        ):
            migration.upgrade()

        assert recording_operations.mutation_calls == []


@pytest.mark.parametrize(
    "migration",
    [MIGRATION_055, MIGRATION_056],
    ids=["055", "056"],
)
def test_clean_upgrade_performs_exactly_three_actions_per_missing_table(
    monkeypatch: pytest.MonkeyPatch,
    migration,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_replica_binding_targets(connection, migration)
        recording_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", recording_operations)

        migration.upgrade()

        assert recording_operations.mutation_calls == _expected_add_calls(migration)


@pytest.mark.parametrize(
    "migration",
    [MIGRATION_055, MIGRATION_056],
    ids=["055", "056"],
)
def test_partial_upgrade_only_fills_missing_table_and_then_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    migration,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_replica_binding_targets(connection, migration)
        monkeypatch.setattr(migration, "op", _operations(connection))
        migration.upgrade()

        missing_table = tuple(migration.TARGETS)[-1]
        _drop_binding_column(connection, migration, missing_table)
        recording_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", recording_operations)

        migration.upgrade()

        assert recording_operations.mutation_calls == _expected_add_calls(
            migration,
            (missing_table,),
        )
        reentry_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", reentry_operations)

        migration.upgrade()

        assert reentry_operations.mutation_calls == []


def test_055_downgrade_preflights_all_tables_before_mutating_mixed_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_mixed_binding_downgrade_has_no_mutations(monkeypatch, MIGRATION_055)


def test_056_downgrade_preflights_all_tables_before_mutating_mixed_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_mixed_binding_downgrade_has_no_mutations(monkeypatch, MIGRATION_056)


@pytest.mark.parametrize(
    "migration",
    [MIGRATION_055, MIGRATION_056],
    ids=["055", "056"],
)
def test_downgrade_rejects_malformed_later_table_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
    migration,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_replica_binding_targets(connection, migration)
        monkeypatch.setattr(migration, "op", _operations(connection))
        migration.upgrade()

        later_table = next(iter(migration.TARGETS))
        index_name, _foreign_key_name = migration.TARGETS[later_table]
        connection.execute(sa.text(f"DROP INDEX {index_name}"))
        connection.execute(
            sa.text(f"CREATE UNIQUE INDEX {index_name} ON {later_table} (id)")
        )
        recording_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", recording_operations)

        with pytest.raises(
            RuntimeError,
            match=rf"{later_table}\.{migration.COLUMN_NAME} index is invalid",
        ):
            migration.downgrade()

        assert recording_operations.mutation_calls == []


@pytest.mark.parametrize(
    "migration",
    [MIGRATION_055, MIGRATION_056],
    ids=["055", "056"],
)
def test_downgrade_rejects_extra_binding_index_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
    migration,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_replica_binding_targets(connection, migration)
        monkeypatch.setattr(migration, "op", _operations(connection))
        migration.upgrade()

        later_table = next(iter(migration.TARGETS))
        connection.execute(
            sa.text(
                f"CREATE INDEX unexpected_replica_binding_index "
                f"ON {later_table} ({migration.COLUMN_NAME})"
            )
        )
        recording_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", recording_operations)

        with pytest.raises(
            RuntimeError,
            match=rf"{later_table}\.{migration.COLUMN_NAME} index is invalid",
        ):
            migration.downgrade()

        assert recording_operations.mutation_calls == []


@pytest.mark.parametrize(
    "migration",
    [MIGRATION_055, MIGRATION_056],
    ids=["055", "056"],
)
def test_clean_downgrade_performs_exactly_three_actions_per_present_table(
    monkeypatch: pytest.MonkeyPatch,
    migration,
) -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_replica_binding_targets(connection, migration)
        monkeypatch.setattr(migration, "op", _operations(connection))
        migration.upgrade()
        recording_operations = _RecordingOperations(connection)
        monkeypatch.setattr(migration, "op", recording_operations)

        migration.downgrade()

        assert recording_operations.mutation_calls == _expected_drop_calls(migration)
        reflected = sa.inspect(connection)
        for table_name in migration.TARGETS:
            columns = {
                column["name"] for column in reflected.get_columns(table_name)
            }
            assert migration.COLUMN_NAME not in columns
