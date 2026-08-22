import importlib
import sys
from types import ModuleType

import sqlalchemy as sa

from train_factory.auth.schemas import UserResponse
from train_factory.storage.entities.user_entity import UserDB


def _import_migration(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "alembic":
            raise

    alembic_stub = ModuleType("alembic")
    alembic_stub.op = object()
    sys.modules["alembic"] = alembic_stub
    try:
        return importlib.import_module(module_name)
    finally:
        sys.modules.pop("alembic", None)


migration = _import_migration(
    "train_factory.storage.migrations.versions.038_add_auth_and_audit_tables"
)


class RecordingOperations:
    def __init__(self):
        self.created_tables = []
        self.created_indexes = []
        self.added_columns = []
        self.dropped_columns = []

    def get_bind(self):
        return object()

    def create_table(self, name, *columns):
        self.created_tables.append((name, columns))

    def create_index(self, name, table, columns, unique=False):
        self.created_indexes.append((name, table, tuple(columns), unique))

    def add_column(self, table, column):
        self.added_columns.append((table, column))

    def drop_column(self, table, column):
        self.dropped_columns.append((table, column))


class RecordingInspector:
    def __init__(self, column_names):
        self.column_names = column_names

    def get_columns(self, table):
        assert table == "users"
        return [{"name": name} for name in self.column_names]


def _token_version_migration():
    return _import_migration(
        "train_factory.storage.migrations.versions.039_add_user_token_version"
    )


def test_auth_migration_creates_missing_tables_and_indexes(monkeypatch):
    operations = RecordingOperations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "_table_exists", lambda _name: False)

    migration.upgrade()

    assert [name for name, _columns in operations.created_tables] == [
        "users",
        "audit_logs",
    ]
    assert set(operations.created_indexes) == {
        ("ix_users_email", "users", ("email",), True),
        ("ix_users_is_active", "users", ("is_active",), False),
        ("ix_users_user_id", "users", ("user_id",), True),
        ("ix_users_username", "users", ("username",), True),
        ("idx_audit_action", "audit_logs", ("action",), False),
        (
            "idx_audit_resource",
            "audit_logs",
            ("resource_type", "resource_id"),
            False,
        ),
        (
            "idx_audit_user_time",
            "audit_logs",
            ("user_id", "created_at"),
            False,
        ),
        ("ix_audit_logs_log_id", "audit_logs", ("log_id",), False),
        ("ix_audit_logs_user_id", "audit_logs", ("user_id",), False),
    }


def test_auth_migration_skips_tables_that_already_exist(monkeypatch):
    operations = RecordingOperations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "_table_exists", lambda _name: True)

    migration.upgrade()

    assert operations.created_tables == []
    assert operations.created_indexes == []


def test_auth_migration_downgrade_preserves_adopted_tables(monkeypatch):
    operations = RecordingOperations()
    monkeypatch.setattr(migration, "op", operations)

    migration.downgrade()

    assert operations.created_tables == []
    assert operations.created_indexes == []


def test_token_version_migration_has_expected_revision_chain():
    token_migration = _token_version_migration()

    assert token_migration.revision == "039_add_user_token_version"
    assert token_migration.down_revision == "038_add_auth_and_audit_tables"


def test_token_version_migration_adds_missing_column(monkeypatch):
    token_migration = _token_version_migration()
    operations = RecordingOperations()
    monkeypatch.setattr(token_migration, "op", operations)
    monkeypatch.setattr(
        token_migration,
        "inspect",
        lambda _bind: RecordingInspector([]),
    )

    token_migration.upgrade()

    assert len(operations.added_columns) == 1
    table, column = operations.added_columns[0]
    assert table == "users"
    assert column.name == "token_version"
    assert isinstance(column.type, sa.Integer)
    assert column.nullable is False
    assert column.server_default.arg == "0"


def test_token_version_migration_skips_existing_column(monkeypatch):
    token_migration = _token_version_migration()
    operations = RecordingOperations()
    monkeypatch.setattr(token_migration, "op", operations)
    monkeypatch.setattr(
        token_migration,
        "inspect",
        lambda _bind: RecordingInspector(["token_version"]),
    )

    token_migration.upgrade()

    assert operations.added_columns == []


def test_token_version_migration_drops_existing_column(monkeypatch):
    token_migration = _token_version_migration()
    operations = RecordingOperations()
    monkeypatch.setattr(token_migration, "op", operations)
    monkeypatch.setattr(
        token_migration,
        "inspect",
        lambda _bind: RecordingInspector(["token_version"]),
    )

    token_migration.downgrade()

    assert operations.dropped_columns == [("users", "token_version")]


def test_token_version_migration_skips_missing_column_on_downgrade(monkeypatch):
    token_migration = _token_version_migration()
    operations = RecordingOperations()
    monkeypatch.setattr(token_migration, "op", operations)
    monkeypatch.setattr(
        token_migration,
        "inspect",
        lambda _bind: RecordingInspector([]),
    )

    token_migration.downgrade()

    assert operations.dropped_columns == []


def test_user_entity_defaults_and_serializes_token_version():
    user = UserDB(username="alice", hashed_password="not-a-real-password")

    assert user.token_version == 0
    assert user.to_dict()["token_version"] == 0


def test_public_user_response_excludes_token_version():
    assert "token_version" not in UserResponse.model_fields
