import re
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import sqlalchemy as sa
import pytest
from sqlalchemy.dialects import mysql as mysql_types

from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.external_sync_entity import ExternalSyncTrainingDB
from train_factory.storage.entities.generation_task_entity import GenerationTaskDB
from train_factory.storage.entities.milvus_collection_entity import MilvusCollectionDB


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INIT_SQL_PATH = PROJECT_ROOT / "docker" / "init.sql"


def _execute_migration_env(
    monkeypatch,
    environment,
    *,
    identity_override=None,
    captured=None,
    offline=False,
    settings_url=None,
):
    captured = captured if captured is not None else {"events": []}
    captured.setdefault("events", [])

    class _Transaction:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    config = SimpleNamespace(
        config_file_name=None,
        config_ini_section="alembic",
        get_section=lambda _name: {},
    )
    context = SimpleNamespace(
        config=config,
        is_offline_mode=lambda: offline,
        configure=lambda **kwargs: (captured["events"].append("configure"), captured.update(kwargs)),
        begin_transaction=lambda: _Transaction(),
        run_migrations=lambda: None,
    )
    alembic_stub = ModuleType("alembic")
    alembic_stub.context = context
    monkeypatch.setitem(sys.modules, "alembic", alembic_stub)

    class _Result:
        def mappings(self):
            return self

        def one(self):
            parsed = sa.engine.make_url(environment["TRAINFACTORY_TEST_MYSQL_URL"])
            result = {
                "server_uuid": environment["TRAINFACTORY_MYSQL_SERVER_UUID"],
                "database_name": parsed.database,
                "marker_exists": 1,
            }
            result.update(identity_override or {})
            return result

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args, **_kwargs):
            captured["events"].append("identity")
            return _Result()

        def in_transaction(self):
            return False

    class _Engine:
        def connect(self):
            return _Connection()

    def fake_engine_from_config(configuration, **_kwargs):
        captured["url"] = configuration["sqlalchemy.url"]
        return _Engine()

    monkeypatch.setattr(sa, "engine_from_config", fake_engine_from_config)
    version_table_stub = ModuleType(
        "train_factory.storage.migrations.version_table"
    )
    version_table_stub.ensure_version_table_capacity = (
        lambda _connection: captured["events"].append("ensure")
    )
    monkeypatch.setitem(
        sys.modules,
        "train_factory.storage.migrations.version_table",
        version_table_stub,
    )

    class _Settings:
        @property
        def mysql_url(self):
            if settings_url is None:
                raise AssertionError("private dotenv settings were accessed")
            return settings_url

    config_stub = ModuleType("train_factory.config")
    config_stub.settings = _Settings()
    monkeypatch.setitem(sys.modules, "train_factory.config", config_stub)
    for name in (
        "TRAINFACTORY_MYSQL_RUN_ID",
        "TRAINFACTORY_MYSQL_SERVER_UUID",
        "TRAINFACTORY_TEST_MYSQL_URL",
        "MYSQL_URL",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    module_name = f"_migration_env_contract_{id(environment)}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "train_factory" / "storage" / "migrations" / "env.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return captured


def test_migration_env_uses_only_closed_runner_url_without_settings(monkeypatch):
    url = (
        "mysql+pymysql://root:private-url-canary@127.0.0.1:49153/"
        f"tf_migration_{'ab' * 16}_{'a' * 16}"
    )
    captured = _execute_migration_env(
        monkeypatch,
        {
            "TRAINFACTORY_MYSQL_RUN_ID": "ab" * 16,
            "TRAINFACTORY_MYSQL_SERVER_UUID": "12345678-1234-1234-1234-123456789abc",
            "TRAINFACTORY_TEST_MYSQL_URL": url,
        },
    )

    assert captured["url"] == url


@pytest.mark.parametrize(
    "environment",
    (
        {
            "TRAINFACTORY_TEST_MYSQL_URL": (
                "mysql+pymysql://root:test@127.0.0.1:49153/"
                "tf_migration_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ),
            "MYSQL_URL": "mysql+pymysql://root:private-fallback@production:3306/mysql",
        },
        {
            "TRAINFACTORY_MYSQL_RUN_ID": "ab" * 16,
            "TRAINFACTORY_MYSQL_SERVER_UUID": "12345678-1234-1234-1234-123456789abc",
            "TRAINFACTORY_TEST_MYSQL_URL": (
                "mysql+pymysql://root:test@production:3306/"
                f"tf_migration_{'ab' * 16}_{'a' * 16}"
            ),
        },
        {
            "TRAINFACTORY_MYSQL_RUN_ID": "ab" * 16,
            "TRAINFACTORY_MYSQL_SERVER_UUID": "12345678-1234-1234-1234-123456789abc",
            "TRAINFACTORY_TEST_MYSQL_URL": (
                "mysql+pymysql://root:test@127.0.0.1:49153/mysql"
            ),
        },
    ),
)
def test_migration_env_rejects_partial_or_invalid_runner_seam_without_fallback(
    monkeypatch,
    environment,
):
    with pytest.raises(RuntimeError) as exc_info:
        _execute_migration_env(monkeypatch, environment)

    assert str(exc_info.value) == "Controlled migration URL is invalid"
    assert "private" not in repr(exc_info.value)


def test_migration_env_identity_mismatch_precedes_version_table_or_migrations(
    monkeypatch,
):
    environment = {
        "TRAINFACTORY_MYSQL_RUN_ID": "ab" * 16,
        "TRAINFACTORY_MYSQL_SERVER_UUID": "12345678-1234-1234-1234-123456789abc",
        "TRAINFACTORY_TEST_MYSQL_URL": (
            "mysql+pymysql://root:test@127.0.0.1:49153/"
            f"tf_migration_{'ab' * 16}_{'a' * 16}"
        ),
    }
    captured = {"events": []}
    with pytest.raises(RuntimeError) as exc_info:
        _execute_migration_env(
            monkeypatch,
            environment,
            identity_override={"server_uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"},
            captured=captured,
        )

    assert str(exc_info.value) == "Controlled migration database identity mismatch"
    assert captured["events"] == ["identity"]


def test_migration_env_controlled_offline_mode_fails_closed(monkeypatch):
    environment = {
        "TRAINFACTORY_MYSQL_RUN_ID": "ab" * 16,
        "TRAINFACTORY_MYSQL_SERVER_UUID": "12345678-1234-1234-1234-123456789abc",
        "TRAINFACTORY_TEST_MYSQL_URL": (
            "mysql+pymysql://root:test@127.0.0.1:49153/"
            f"tf_migration_{'ab' * 16}_{'a' * 16}"
        ),
    }

    with pytest.raises(RuntimeError) as exc_info:
        _execute_migration_env(monkeypatch, environment, offline=True)

    assert str(exc_info.value) == "Controlled migrations require an online connection"


def test_migration_env_normal_path_preserves_application_settings_url(monkeypatch):
    expected = "mysql+pymysql://app@database:3306/train_factory"

    captured = _execute_migration_env(
        monkeypatch,
        {},
        settings_url=expected,
    )

    assert captured["url"] == expected


def test_migration_env_cold_controlled_process_does_not_import_application_modules():
    script = r'''
import importlib.abc
import runpy
import sys
from types import ModuleType, SimpleNamespace
import sqlalchemy as sa

blocked = ("train_factory.config", "train_factory.storage.entities", "train_factory.core", "torch")
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == blocked or any(fullname == item or fullname.startswith(item + ".") for item in blocked):
            raise RuntimeError("application import was attempted")
        return None
sys.meta_path.insert(0, Blocker())

class Result:
    def mappings(self): return self
    def one(self):
        return {"server_uuid": "12345678-1234-1234-1234-123456789abc", "database_name": "tf_migration_" + "ab" * 16 + "_" + "a" * 16, "marker_exists": 1}
class Connection:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, *args, **kwargs): return Result()
    def in_transaction(self): return False
class Engine:
    def connect(self): return Connection()
sa.engine_from_config = lambda *args, **kwargs: Engine()

class Transaction:
    def __enter__(self): return self
    def __exit__(self, *args): return False
context = SimpleNamespace(
    config=SimpleNamespace(config_file_name=None, config_ini_section="alembic", get_section=lambda name: {}),
    is_offline_mode=lambda: False,
    configure=lambda **kwargs: None,
    begin_transaction=lambda: Transaction(),
    run_migrations=lambda: None,
)
alembic = ModuleType("alembic")
alembic.context = context
sys.modules["alembic"] = alembic
version_table = ModuleType("train_factory.storage.migrations.version_table")
version_table.ensure_version_table_capacity = lambda connection: None
sys.modules["train_factory.storage.migrations.version_table"] = version_table
runpy.run_path("train_factory/storage/migrations/env.py", run_name="_cold_controlled_env")
'''
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TRAINFACTORY_MYSQL_RUN_ID": "ab" * 16,
        "TRAINFACTORY_MYSQL_SERVER_UUID": "12345678-1234-1234-1234-123456789abc",
        "TRAINFACTORY_TEST_MYSQL_URL": (
            "mysql+pymysql://root:test@127.0.0.1:49153/"
            f"tf_migration_{'ab' * 16}_{'a' * 16}"
        ),
    }

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


def _import_migration(module_name: str):
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


def _table_definition(sql: str, table_name: str) -> str:
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS {re.escape(table_name)}\s*\((.*?)\)\s*ENGINE=",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert match is not None, f"missing {table_name} from init.sql"
    return re.sub(r"\s+", " ", match.group(1)).lower()


def test_fresh_alembic_version_table_supports_long_revision_ids():
    from train_factory.storage.migrations.version_table import (
        ensure_version_table_capacity,
    )

    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        ensure_version_table_capacity(connection)
        column = sa.inspect(connection).get_columns("alembic_version")[0]

    assert column["name"] == "version_num"
    assert column["type"].length == 64


def test_init_sql_keeps_existing_tables_aligned_with_current_entities():
    sql = INIT_SQL_PATH.read_text(encoding="utf-8")

    expected_schema = {
        "training_tasks": (
            "run_token varchar(36)",
            "index idx_training_run_token (run_token)",
            "unique key uq_task_user_name (user_id, task_name)",
            "index idx_task_user_created (user_id, created_at)",
            "index idx_task_type_status (model_type, status)",
        ),
        "model_registry": (
            "unique key uq_model_user_name_version (user_id, model_name, version)",
            "index ix_model_registry_source_task_id (source_task_id)",
            "index idx_model_user_created (user_id, created_at)",
            "index idx_model_type_status (model_type, status)",
        ),
        "model_versions": (
            "unique key uq_version_model_version (model_id, version)",
            "index idx_version_model_created (model_id, created_at)",
        ),
        "deployments": (
            "external_api_config_id varchar(36)",
            "unique key uq_deployment_user_name (user_id, deployment_name)",
            "index idx_deployment_model_status (model_id, status)",
            "index idx_deployment_user_created (user_id, created_at)",
            "index idx_deployment_external_api_config_id (external_api_config_id)",
            "index idx_deployment_external_api_status (external_api_config_id, status)",
        ),
        "model_configs": (
            "container_name varchar(255)",
            "inference_framework varchar(32)",
            "unique key uq_config_user_name (user_id, config_name)",
            "index idx_config_user_type (user_id, model_type)",
            "index idx_config_provider_status (provider, status)",
        ),
        "datasets": (
            "source_dataset_id varchar(36)",
            "status varchar(50) not null default 'registered'",
            "deletion_owner varchar(128)",
            "unique key uq_dataset_user_name (user_id, dataset_name)",
            "index idx_dataset_source_dataset (source_dataset_id)",
            "index idx_dataset_user_created (user_id, created_at)",
            "index idx_dataset_type_status (dataset_type, status)",
        ),
        "dataset_lineage_edges": (
            "edge_key varchar(64) not null",
            "unique key uq_lineage_edge_key (edge_key)",
        ),
        "evaluation_tasks": (
            "run_token varchar(36)",
            "eval_framework varchar(32)",
            "field_mapping json",
            "metrics json",
            "llm_config json",
            "worker_groups json",
            "total_samples int",
            "processed_samples int",
            "results_path varchar(1024)",
            "index idx_eval_run_token (run_token)",
            "index idx_eval_framework (eval_framework)",
        ),
    }

    for table_name, fragments in expected_schema.items():
        definition = _table_definition(sql, table_name)
        for fragment in fragments:
            assert fragment in definition, f"{table_name} is missing: {fragment}"


def test_evaluation_run_token_migration_follows_current_head():
    module_name = (
        "train_factory.storage.migrations.versions."
        "050_add_evaluation_run_token"
    )
    assert importlib.util.find_spec(module_name) is not None
    migration = _import_migration(module_name)

    assert migration.revision == "050_add_evaluation_run_token"
    assert migration.down_revision == "049_dataset_lineage_edge_key"
    assert EvaluationTaskDB.__table__.c.run_token.type.length == 36


def test_generation_run_token_migration_follows_evaluation_token(monkeypatch):
    module_name = (
        "train_factory.storage.migrations.versions."
        "051_add_generation_run_token"
    )
    assert importlib.util.find_spec(module_name) is not None
    migration = _import_migration(module_name)

    class Operations:
        def __init__(self):
            self.added_columns = []
            self.created_indexes = []

        @staticmethod
        def get_bind():
            return object()

        def add_column(self, table, column):
            self.added_columns.append((table, column))

        def create_index(self, name, table, columns, unique=False, **_kwargs):
            self.created_indexes.append((name, table, tuple(columns), unique))

    class Inspector:
        @staticmethod
        def get_table_names():
            return ["generation_tasks"]

        @staticmethod
        def get_columns(table):
            assert table == "generation_tasks"
            return [{"name": "task_id"}]

        @staticmethod
        def get_pk_constraint(table):
            assert table == "generation_tasks"
            return {"constrained_columns": []}

        @staticmethod
        def get_indexes(table):
            assert table == "generation_tasks"
            return []

        @staticmethod
        def get_unique_constraints(table):
            assert table == "generation_tasks"
            return []

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert migration.revision == "051_add_generation_run_token"
    assert migration.down_revision == "050_add_evaluation_run_token"
    assert GenerationTaskDB.__table__.c.run_token.type.length == 36
    assert len(operations.added_columns) == 1
    table, column = operations.added_columns[0]
    assert table == "generation_tasks"
    assert column.name == "run_token"
    assert isinstance(column.type, sa.String)
    assert column.type.length == 36
    assert operations.created_indexes == [
        ("idx_gen_task_run_token", "generation_tasks", ("run_token",), False)
    ]


def test_generation_run_token_head_descends_from_generation_table_creation():
    pytest.importorskip("alembic")
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(
        str(
            PROJECT_ROOT
            / "train_factory"
            / "storage"
            / "migrations"
            / "alembic.ini"
        )
    )
    script = ScriptDirectory.from_config(config)
    revisions = {
        revision.revision
        for revision in script.walk_revisions(
            base="base",
            head="051_add_generation_run_token",
        )
    }

    assert "012_add_generation_tasks" in revisions


def test_lifecycle_schema_validation_migration_is_current_head_and_downgrade_is_noop(
    monkeypatch,
):
    module_name = (
        "train_factory.storage.migrations.versions."
        "053_validate_lifecycle_schema"
    )
    assert importlib.util.find_spec(module_name) is not None
    migration = _import_migration(module_name)

    assert migration.revision == "053_validate_lifecycle_schema"
    assert migration.down_revision == "052_generation_publication_staging"
    expected_modules = (
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity",
        "train_factory.storage.migrations.versions."
        "049_add_dataset_lineage_edge_key",
        "train_factory.storage.migrations.versions."
        "050_add_evaluation_run_token",
        "train_factory.storage.migrations.versions."
        "051_add_generation_run_token",
        "train_factory.storage.migrations.versions."
        "052_generation_publication_staging",
    )
    assert migration.LIFECYCLE_MIGRATION_MODULES == expected_modules
    alembic_stub = ModuleType("alembic")
    alembic_stub.op = object()
    monkeypatch.setitem(sys.modules, "alembic", alembic_stub)
    loaded = migration._load_lifecycle_migrations()
    monkeypatch.delitem(sys.modules, "alembic")
    assert tuple(module.__name__ for module in loaded) == expected_modules
    assert tuple(module.revision for module in loaded) == (
        "048_add_training_process_identity",
        "049_dataset_lineage_edge_key",
        "050_add_evaluation_run_token",
        "051_add_generation_run_token",
        "052_generation_publication_staging",
    )
    assert tuple(module.down_revision for module in loaded[1:]) == tuple(
        module.revision for module in loaded[:-1]
    )

    class Operations:
        mutations = []

        def __getattr__(self, name):
            def record(*args, **kwargs):
                self.mutations.append((name, args, kwargs))

            return record

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    migration.downgrade()
    assert operations.mutations == []


def test_lifecycle_schema_validation_migration_is_alembic_head():
    pytest.importorskip("alembic")
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(
        str(
            PROJECT_ROOT
            / "train_factory"
            / "storage"
            / "migrations"
            / "alembic.ini"
        )
    )
    script = ScriptDirectory.from_config(config)
    assert script.get_current_head() == "053_validate_lifecycle_schema"


def test_lifecycle_schema_validation_sanitizes_whitelisted_import_errors(
    monkeypatch,
):
    module_name = (
        "train_factory.storage.migrations.versions."
        "053_validate_lifecycle_schema"
    )
    migration = _import_migration(module_name)
    actual_import = migration.importlib.import_module

    def fail_one_import(import_name):
        if import_name.endswith("050_add_evaluation_run_token"):
            raise ValueError("import-private-canary")
        return actual_import(import_name)

    monkeypatch.setattr(migration.importlib, "import_module", fail_one_import)
    alembic_stub = ModuleType("alembic")
    alembic_stub.op = object()
    monkeypatch.setitem(sys.modules, "alembic", alembic_stub)

    with pytest.raises(RuntimeError, match="Unable to load lifecycle migration") as exc_info:
        migration._load_lifecycle_migrations()

    assert exc_info.value.__cause__ is None
    assert "import-private-canary" not in str(exc_info.value)


def test_deployment_link_migration_adopts_current_init_schema(monkeypatch):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "035_add_deployment_external_api_config_id"
    )

    class Operations:
        def __init__(self):
            self.added_columns = []
            self.created_indexes = []
            self.executed = []
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))

        def get_bind(self):
            return self.bind

        def add_column(self, table, column):
            self.added_columns.append((table, column.name))

        def create_index(self, name, table, columns, unique=False, **kwargs):
            self.created_indexes.append((name, table, tuple(columns), unique))

        def execute(self, statement):
            self.executed.append(str(statement))

    class Inspector:
        def get_columns(self, table):
            assert table == "deployments"
            return [{"name": "external_api_config_id"}]

        def get_indexes(self, table):
            assert table == "deployments"
            return [
                {"name": "idx_deployment_external_api_config_id"},
                {"name": "idx_deployment_external_api_status"},
            ]

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert operations.added_columns == []
    assert operations.created_indexes == []
    assert len(operations.executed) == 1


def test_schema_reconciliation_adds_only_missing_query_indexes(monkeypatch):
    from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect

    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "040_reconcile_entity_query_indexes"
    )

    class Operations:
        def __init__(self):
            self.created_indexes = []
            self.created_constraints = []

        def get_bind(self):
            return SimpleNamespace(dialect=sqlite_dialect())

        def create_index(self, name, table, columns, unique=False, **kwargs):
            self.created_indexes.append((name, table, tuple(columns), unique))

        def create_unique_constraint(self, name, table, columns):
            self.created_constraints.append((name, table, tuple(columns)))

    class Inspector:
        def get_table_names(self):
            return [
                "training_tasks",
                "model_registry",
                "model_versions",
                "deployments",
                "model_configs",
                "datasets",
            ]

        def get_indexes(self, _table):
            return []

        def get_unique_constraints(self, _table):
            return []

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())
    monkeypatch.setattr(migration, "_assert_no_duplicate_rows", lambda *_args: None)

    migration.upgrade()

    assert set(operations.created_indexes) == {
        (
            "idx_task_type_status",
            "training_tasks",
            ("model_type", "status"),
            False,
        ),
        (
            "ix_model_registry_source_task_id",
            "model_registry",
            ("source_task_id",),
            False,
        ),
        (
            "idx_model_user_created",
            "model_registry",
            ("user_id", "created_at"),
            False,
        ),
        (
            "idx_model_type_status",
            "model_registry",
            ("model_type", "status"),
            False,
        ),
    }
    assert set(operations.created_constraints) == {
        ("uq_task_user_name", "training_tasks", ("user_id", "task_name")),
        (
            "uq_model_user_name_version",
            "model_registry",
            ("user_id", "model_name", "version"),
        ),
        (
            "uq_version_model_version",
            "model_versions",
            ("model_id", "version"),
        ),
        (
            "uq_deployment_user_name",
            "deployments",
            ("user_id", "deployment_name"),
        ),
        ("uq_config_user_name", "model_configs", ("user_id", "config_name")),
        ("uq_dataset_user_name", "datasets", ("user_id", "dataset_name")),
    }


def test_schema_reconciliation_uses_mysql_online_index_ddl(monkeypatch):
    from sqlalchemy.dialects.mysql import dialect as mysql_dialect

    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "040_reconcile_entity_query_indexes"
    )
    class Operations:
        def __init__(self):
            self.bind = SimpleNamespace(dialect=mysql_dialect())
            self.executed = []

        def get_bind(self):
            return self.bind

        def execute(self, statement):
            self.executed.append(str(statement))

        def create_index(self, name, table, columns, unique=False, **kwargs):
            metadata = sa.MetaData()
            target = sa.Table(
                table,
                metadata,
                *(sa.Column(column, sa.String()) for column in columns),
            )
            sa.Index(name, *(target.c[column] for column in columns), unique=unique, **kwargs)

        @staticmethod
        def create_unique_constraint(*_args):
            raise AssertionError("existing constraint must be adopted")

    operations = Operations()

    class Inspector:
        @staticmethod
        def get_table_names():
            return ["training_tasks"]

        @staticmethod
        def get_indexes(_table):
            return []

        @staticmethod
        def get_unique_constraints(_table):
            return [{"column_names": ["user_id", "task_name"]}]

    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert operations.executed == [
        "ALTER TABLE training_tasks ADD INDEX idx_task_type_status "
        "(model_type, status), ALGORITHM=INPLACE, LOCK=NONE"
    ]


def test_schema_reconciliation_accepts_equivalent_existing_indexes(monkeypatch):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "040_reconcile_entity_query_indexes"
    )

    class Operations:
        def __init__(self):
            self.created_indexes = []
            self.created_constraints = []

        def get_bind(self):
            return object()

        def create_index(self, name, table, columns, unique=False, **kwargs):
            self.created_indexes.append((name, table, tuple(columns), unique))

        def create_unique_constraint(self, name, table, columns):
            self.created_constraints.append((name, table, tuple(columns)))

    expected_columns = {
        "training_tasks": [("model_type", "status")],
        "model_registry": [
            ("source_task_id",),
            ("user_id", "created_at"),
            ("model_type", "status"),
        ],
    }

    class Inspector:
        def get_table_names(self):
            return [
                "training_tasks",
                "model_registry",
                "model_versions",
                "deployments",
                "model_configs",
                "datasets",
            ]

        def get_indexes(self, table):
            return [
                {"name": f"legacy_{index}", "column_names": list(columns)}
                for index, columns in enumerate(expected_columns[table])
            ]

        def get_unique_constraints(self, table):
            constraints = {
                "training_tasks": [("user_id", "task_name")],
                "model_registry": [("user_id", "model_name", "version")],
                "model_versions": [("model_id", "version")],
                "deployments": [("user_id", "deployment_name")],
                "model_configs": [("user_id", "config_name")],
                "datasets": [("user_id", "dataset_name")],
            }
            return [
                {"name": "legacy_unique", "column_names": list(columns)}
                for columns in constraints[table]
            ]

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert operations.created_indexes == []
    assert operations.created_constraints == []


def test_dataset_status_default_migration_follows_schema_reconciliation(monkeypatch):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "041_set_dataset_status_default"
    )

    class Operations:
        def __init__(self):
            self.alterations = []

        def alter_column(self, table, column, **kwargs):
            self.alterations.append((table, column, kwargs))

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)

    migration.upgrade()

    assert migration.revision == "041_set_dataset_status_default"
    assert migration.down_revision == "040_reconcile_entity_query_indexes"
    assert len(operations.alterations) == 1
    table, column, kwargs = operations.alterations[0]
    assert (table, column) == ("datasets", "status")
    assert isinstance(kwargs["existing_type"], sa.String)
    assert kwargs["existing_type"].length == 50
    assert kwargs["existing_nullable"] is False
    assert kwargs["server_default"] == "registered"


def test_audit_created_at_index_migration_follows_dataset_default(monkeypatch):
    from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect

    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "042_add_audit_created_at_index"
    )

    class Operations:
        def __init__(self):
            self.created_indexes = []

        def get_bind(self):
            return SimpleNamespace(dialect=sqlite_dialect())

        def create_index(self, name, table, columns, unique=False, **kwargs):
            self.created_indexes.append((name, table, tuple(columns), unique))

    class Inspector:
        @staticmethod
        def get_table_names():
            return ["audit_logs"]

        @staticmethod
        def get_indexes(table):
            assert table == "audit_logs"
            return [{"name": "idx_audit_user_time", "column_names": ["user_id", "created_at"]}]

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert migration.revision == "042_add_audit_created_at_index"
    assert migration.down_revision == "041_set_dataset_status_default"
    assert operations.created_indexes == [
        ("idx_audit_created_at", "audit_logs", ("created_at",), False)
    ]


def test_audit_created_at_index_uses_mysql_online_ddl(monkeypatch):
    from sqlalchemy.dialects.mysql import dialect as mysql_dialect

    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "042_add_audit_created_at_index"
    )
    class Operations:
        def __init__(self):
            self.bind = SimpleNamespace(dialect=mysql_dialect())
            self.executed = []

        def get_bind(self):
            return self.bind

        def execute(self, statement):
            self.executed.append(str(statement))

        def create_index(self, name, table, columns, unique=False, **kwargs):
            metadata = sa.MetaData()
            target = sa.Table(
                table,
                metadata,
                *(sa.Column(column, sa.String()) for column in columns),
            )
            sa.Index(name, *(target.c[column] for column in columns), unique=unique, **kwargs)

    operations = Operations()

    class Inspector:
        @staticmethod
        def get_table_names():
            return ["audit_logs"]

        @staticmethod
        def get_indexes(_table):
            return []

    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert operations.executed == [
        "ALTER TABLE audit_logs ADD INDEX idx_audit_created_at (created_at), "
        "ALGORITHM=INPLACE, LOCK=NONE"
    ]


def test_training_run_token_migration_follows_audit_index(monkeypatch):
    migration = _import_migration(
        "train_factory.storage.migrations.versions.043_add_training_run_token"
    )

    class Operations:
        def __init__(self):
            self.added_columns = []
            self.created_indexes = []

        def get_bind(self):
            return object()

        def add_column(self, table, column):
            self.added_columns.append((table, column))

        def create_index(self, name, table, columns, unique=False, **kwargs):
            self.created_indexes.append((name, table, tuple(columns), unique))

    class Inspector:
        @staticmethod
        def get_table_names():
            return ["training_tasks"]

        @staticmethod
        def get_columns(table):
            assert table == "training_tasks"
            return [{"name": "task_id"}]

        @staticmethod
        def get_indexes(table):
            assert table == "training_tasks"
            return []

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert migration.revision == "043_add_training_run_token"
    assert migration.down_revision == "042_add_audit_created_at_index"
    assert len(operations.added_columns) == 1
    table, column = operations.added_columns[0]
    assert table == "training_tasks"
    assert column.name == "run_token"
    assert isinstance(column.type, sa.String)
    assert column.type.length == 36
    assert operations.created_indexes == [
        ("idx_training_run_token", "training_tasks", ("run_token",), False)
    ]


def test_deletion_fence_owner_migration_backfills_retriable_tokens(monkeypatch):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "045_add_deletion_fence_owners"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE datasets ("
                "id INTEGER PRIMARY KEY, dataset_id VARCHAR(36), "
                "status VARCHAR(50))"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE milvus_collections ("
                "id INTEGER PRIMARY KEY, collection_id VARCHAR(36), "
                "status VARCHAR(32))"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO datasets VALUES "
                "(1, 'dataset-1', 'deleting'), (2, 'dataset-2', 'ready')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO milvus_collections VALUES "
                "(1, 'collection-1', 'deleting')"
            )
        )
        operations = Operations(MigrationContext.configure(connection))
        monkeypatch.setattr(migration, "op", operations)

        migration.upgrade()

        dataset_rows = connection.execute(
            sa.text(
                "SELECT dataset_id, deletion_owner FROM datasets ORDER BY id"
            )
        ).all()
        collection_rows = connection.execute(
            sa.text(
                "SELECT collection_id, deletion_owner "
                "FROM milvus_collections ORDER BY id"
            )
        ).all()

    assert migration.revision == "045_deletion_fence_owners"
    assert migration.down_revision == "044_sync_training_claims"
    assert dataset_rows == [
        ("dataset-1", "dataset:dataset-1"),
        ("dataset-2", None),
    ]
    assert collection_rows == [
        ("collection-1", "manual:collection-1")
    ]


def test_sync_training_snapshot_entity_is_non_nullable():
    column = ExternalSyncTrainingDB.__table__.columns.get(
        "target_config_snapshot"
    )

    assert column is not None
    assert isinstance(column.type, sa.JSON)
    assert column.nullable is False


def test_sync_training_claim_migration_backfills_snapshot_and_round_trips(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "044_add_sync_training_claim_recovery"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE external_sync_trainings ("
                "id INTEGER PRIMARY KEY)"
            )
        )
        connection.execute(
            sa.text("INSERT INTO external_sync_trainings (id) VALUES (1)")
        )
        context = MigrationContext.configure(connection)
        monkeypatch.setattr(migration, "op", Operations(context))

        migration.upgrade()

        upgraded_columns = {
            column["name"]: column
            for column in sa.inspect(connection).get_columns(
                "external_sync_trainings"
            )
        }
        snapshot = connection.execute(
            sa.text(
                "SELECT target_config_snapshot "
                "FROM external_sync_trainings WHERE id = 1"
            )
        ).scalar_one()
        assert upgraded_columns["target_config_snapshot"]["nullable"] is False
        assert snapshot in ({}, "{}")

        migration.downgrade()
        downgraded_columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns(
                "external_sync_trainings"
            )
        }

    assert migration.revision == "044_sync_training_claims"
    assert migration.down_revision == "043_add_training_run_token"
    assert "target_config_snapshot" not in downgraded_columns


def test_milvus_sync_claim_migration_backfills_only_unique_owned_current_names(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    try:
        migration = _import_migration(
            "train_factory.storage.migrations.versions."
            "046_add_milvus_sync_claims"
        )
    except ModuleNotFoundError:
        pytest.fail("missing Milvus sync claim migration")

    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE milvus_collections ("
                "id INTEGER PRIMARY KEY, collection_name VARCHAR(255), "
                "user_id VARCHAR(64))"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE external_sync_tasks ("
                "id INTEGER PRIMARY KEY, task_id VARCHAR(36), "
                "user_id VARCHAR(64), milvus_collection_name VARCHAR(255))"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO milvus_collections VALUES "
                "(1, 'exact-current', 'user-1'), "
                "(2, 'cross-tenant', 'user-1'), "
                "(3, 'ambiguous-current', 'user-1'), "
                "(4, 'tf_sync_v3_legacy_prefix_base', 'user-1'), "
                "(5, 'manual-collection', 'user-1')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO external_sync_tasks VALUES "
                "(1, 'sync-1', 'user-1', 'exact-current'), "
                "(2, 'sync-2', 'user-2', 'cross-tenant'), "
                "(3, 'sync-3', 'user-1', 'ambiguous-current'), "
                "(4, 'sync-4', 'user-1', 'ambiguous-current'), "
                "(5, 'legacy-prefix-task', 'user-1', NULL)"
            )
        )
        operations = Operations(MigrationContext.configure(connection))
        monkeypatch.setattr(migration, "op", operations)

        migration.upgrade()

        rows = connection.execute(
            sa.text(
                "SELECT collection_name, sync_task_id "
                "FROM milvus_collections ORDER BY id"
            )
        ).all()
        column = next(
            item
            for item in sa.inspect(connection).get_columns("milvus_collections")
            if item["name"] == "sync_task_id"
        )

    assert migration.revision == "046_milvus_sync_claims"
    assert migration.down_revision == "045_deletion_fence_owners"
    assert column["nullable"] is True
    assert column["type"].length == 36
    assert rows == [
        ("exact-current", "sync-1"),
        ("cross-tenant", None),
        ("ambiguous-current", None),
        ("tf_sync_v3_legacy_prefix_base", None),
        ("manual-collection", None),
    ]


def test_milvus_sync_claim_migration_uses_mysql_portable_nullable_column(
    monkeypatch,
):
    try:
        migration = _import_migration(
            "train_factory.storage.migrations.versions."
            "046_add_milvus_sync_claims"
        )
    except ModuleNotFoundError:
        pytest.fail("missing Milvus sync claim migration")

    class Operations:
        def __init__(self):
            self.added_columns = []
            self.bind = SimpleNamespace(
                dialect=SimpleNamespace(name="mysql")
            )

        def get_bind(self):
            return self.bind

        def add_column(self, table, column):
            self.added_columns.append((table, column))

    class Inspector:
        @staticmethod
        def get_table_names():
            return ["milvus_collections", "external_sync_tasks"]

        @staticmethod
        def get_columns(table):
            assert table == "milvus_collections"
            return [{"name": "collection_name"}]

    operations = Operations()
    backfills = []
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())
    monkeypatch.setattr(
        migration,
        "_backfill_unique_current_claims",
        lambda: backfills.append(True),
    )

    migration.upgrade()

    assert len(operations.added_columns) == 1
    table, column = operations.added_columns[0]
    assert table == "milvus_collections"
    assert column.name == "sync_task_id"
    assert isinstance(column.type, sa.String)
    assert column.type.length == 36
    assert column.nullable is True
    assert backfills == [True]


def test_milvus_registry_schema_persists_manual_hybrid_creation_intent():
    column = MilvusCollectionDB.__table__.columns.get("hybrid_enabled")

    assert column is not None
    assert isinstance(column.type, sa.Boolean)
    # Existing pre-migration creating rows have unknown intent and must remain
    # fail-closed rather than being guessed as non-hybrid.
    assert column.nullable is True


def test_milvus_hybrid_intent_migration_preserves_legacy_unknown(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    try:
        migration = _import_migration(
            "train_factory.storage.migrations.versions."
            "047_add_milvus_hybrid_intent"
        )
    except ModuleNotFoundError:
        pytest.fail("missing Milvus hybrid creation intent migration")

    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE milvus_collections ("
                "id INTEGER PRIMARY KEY, collection_name VARCHAR(255))"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO milvus_collections VALUES "
                "(1, 'legacy-creating')"
            )
        )
        operations = Operations(MigrationContext.configure(connection))
        monkeypatch.setattr(migration, "op", operations)

        migration.upgrade()

        value = connection.execute(
            sa.text(
                "SELECT hybrid_enabled FROM milvus_collections WHERE id = 1"
            )
        ).scalar_one()
        column = next(
            item
            for item in sa.inspect(connection).get_columns(
                "milvus_collections"
            )
            if item["name"] == "hybrid_enabled"
        )

    assert migration.revision == "047_milvus_hybrid_intent"
    assert migration.down_revision == "046_milvus_sync_claims"
    assert isinstance(column["type"], sa.Boolean)
    assert column["nullable"] is True
    assert value is None


def test_schema_reconciliation_reports_duplicates_before_adding_unique():
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "040_reconcile_entity_query_indexes"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE training_tasks (user_id VARCHAR(64), task_name VARCHAR(255))"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO training_tasks (user_id, task_name) "
                "VALUES ('user-1', 'duplicate'), ('user-1', 'duplicate')"
            )
        )

        with pytest.raises(RuntimeError, match="uq_task_user_name"):
            migration._assert_no_duplicate_rows(
                connection,
                "training_tasks",
                ("user_id", "task_name"),
                "uq_task_user_name",
            )


def test_user_first_composite_indexes_replace_redundant_single_column_indexes():
    evaluation_indexes = {
        tuple(index.columns.keys()) for index in EvaluationTaskDB.__table__.indexes
    }
    milvus_indexes = {
        tuple(index.columns.keys()) for index in MilvusCollectionDB.__table__.indexes
    }

    assert ("user_id", "created_at") in evaluation_indexes
    assert ("user_id",) not in evaluation_indexes
    assert ("user_id", "created_at") in milvus_indexes
    assert ("user_id",) not in milvus_indexes


_RUN_TOKEN_MIGRATION_CASES = (
    (
        "train_factory.storage.migrations.versions."
        "050_add_evaluation_run_token",
        "evaluation_tasks",
        "idx_eval_run_token",
    ),
    (
        "train_factory.storage.migrations.versions."
        "051_add_generation_run_token",
        "generation_tasks",
        "idx_gen_task_run_token",
    ),
)


def _use_real_alembic_operations(monkeypatch, migration, connection):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    monkeypatch.setattr(
        migration,
        "op",
        Operations(MigrationContext.configure(connection)),
    )


def _sqlite_table_schema_state(connection, table_name):
    inspector = sa.inspect(connection)
    return {
        "columns": tuple(
            sorted(
                (
                    column["name"],
                    str(column["type"]),
                    column["nullable"],
                )
                for column in inspector.get_columns(table_name)
            )
        ),
        "indexes": tuple(
            sorted(
                (
                    index["name"],
                    tuple(index.get("column_names") or ()),
                    bool(index.get("unique")),
                )
                for index in inspector.get_indexes(table_name)
            )
        ),
    }


def _create_run_token_table(
    connection,
    table_name,
    *,
    run_token_definition=None,
):
    run_token_sql = (
        f", run_token {run_token_definition}" if run_token_definition else ""
    )
    connection.execute(
        sa.text(
            f"CREATE TABLE {table_name} ("
            "task_id VARCHAR(36) NOT NULL"
            f"{run_token_sql})"
        )
    )


def test_training_process_identity_upgrade_requires_training_tasks(monkeypatch):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _use_real_alembic_operations(monkeypatch, migration, connection)

        with pytest.raises(
            RuntimeError,
            match="Required table 'training_tasks' is missing",
        ):
            migration.upgrade()

        assert sa.inspect(connection).get_table_names() == []


@pytest.mark.parametrize(
    "column_definition",
    ["VARCHAR(36)", "DOUBLE NOT NULL"],
    ids=["wrong-type", "not-null"],
)
def test_training_process_identity_upgrade_rejects_malformed_column_without_mutation(
    monkeypatch,
    column_definition,
):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE training_tasks ("
                "task_id VARCHAR(36) NOT NULL, "
                f"process_create_time {column_definition})"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_table_schema_state(connection, "training_tasks")

        with pytest.raises(RuntimeError, match="must be nullable DOUBLE"):
            migration.upgrade()

        assert _sqlite_table_schema_state(connection, "training_tasks") == before


def test_training_process_identity_upgrade_completes_partial_schema_idempotently(
    monkeypatch,
):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE training_tasks (task_id VARCHAR(36) NOT NULL)"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)

        migration.upgrade()
        migration.upgrade()

        column = next(
            column
            for column in sa.inspect(connection).get_columns("training_tasks")
            if column["name"] == "process_create_time"
        )

    assert getattr(column["type"], "__visit_name__", "").lower() == "double"
    assert column["nullable"] is True


@pytest.mark.parametrize(
    ("module_name", "column_type", "index_mapping"),
    (
        ("048_add_training_process_identity", sa.Double(), False),
        ("049_add_dataset_lineage_edge_key", mysql_types.VARCHAR(64), False),
        ("050_add_evaluation_run_token", mysql_types.VARCHAR(36), True),
        ("051_add_generation_run_token", mysql_types.VARCHAR(36), True),
    ),
)
def test_lifecycle_managed_columns_use_table_primary_key_metadata(
    module_name,
    column_type,
    index_mapping,
):
    migration = _import_migration(
        f"train_factory.storage.migrations.versions.{module_name}"
    )
    indexes = {} if index_mapping else []
    snapshot = {
        "columns": {
            "id": {"name": "id", "primary_key": True},
            migration.COLUMN_NAME: {
                "name": migration.COLUMN_NAME,
                "type": column_type,
                "nullable": True,
                "default": None,
                "computed": None,
                "identity": None,
                "primary_key": None,
            }
        },
        "indexes": indexes,
        "unique_constraints": [],
        "primary_key_columns": ("id",),
    }

    migration._validate_existing_schema(snapshot)

    snapshot["primary_key_columns"] = (migration.COLUMN_NAME,)
    with pytest.raises(RuntimeError, match="must not be a primary key"):
        migration._validate_existing_schema(snapshot)


def test_generation_publication_columns_reject_table_primary_key_membership():
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "052_generation_publication_staging"
    )
    snapshot = {
        "datasets": {
            "columns": {
                "id": {"name": "id", "primary_key": True},
                "generation_run_token": {
                    "name": "generation_run_token",
                    "type": mysql_types.VARCHAR(36),
                    "nullable": True,
                    "default": None,
                    "computed": None,
                    "identity": None,
                    "primary_key": None,
                }
            },
            "indexes": {},
            "primary_key_columns": ("generation_run_token",),
        }
    }

    with pytest.raises(RuntimeError, match="must not be a primary key"):
        migration._validate_existing_schema(snapshot)


@pytest.mark.parametrize(
    "module_name",
    (
        "048_add_training_process_identity",
        "049_add_dataset_lineage_edge_key",
        "050_add_evaluation_run_token",
        "051_add_generation_run_token",
        "052_generation_publication_staging",
    ),
)
@pytest.mark.parametrize(
    "failure_mode",
    ("method-absent-marker-missing", "method-error", "unknown-column"),
)
def test_lifecycle_primary_key_reflection_fails_closed(
    monkeypatch,
    module_name,
    failure_mode,
):
    migration = _import_migration(
        f"train_factory.storage.migrations.versions.{module_name}"
    )
    table_name = (
        "datasets"
        if module_name.startswith("052_")
        else migration.TABLE_NAME
    )

    class Inspector:
        @staticmethod
        def get_table_names():
            return [table_name]

        @staticmethod
        def get_columns(_table):
            if failure_mode == "method-absent-marker-missing":
                return [{"name": "id"}]
            return [{"name": "id", "primary_key": True}]

        @staticmethod
        def get_indexes(_table):
            return []

        @staticmethod
        def get_unique_constraints(_table):
            return []

        @staticmethod
        def get_pk_constraint(_table):
            if failure_mode == "method-error":
                raise AttributeError("private-pk-canary")
            return {"constrained_columns": ["private-unknown-column"]}

    if failure_mode == "method-absent-marker-missing":
        monkeypatch.delattr(Inspector, "get_pk_constraint")
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())
    with pytest.raises(RuntimeError) as exc_info:
        if module_name.startswith("052_"):
            migration._schema_snapshot(object(), require_all_tables=False)
        else:
            migration._schema_snapshot(object(), require_table=False)

    assert "primary key metadata is invalid" in str(exc_info.value)
    assert "private" not in repr(exc_info.value)


@pytest.mark.parametrize(
    "primary_key",
    (
        None,
        {"constrained_columns": None},
        {"constrained_columns": ["id", "id"]},
        {"constrained_columns": ["private-unknown-column"]},
    ),
)
def test_lifecycle_validator_rejects_malformed_raw_primary_key_metadata(
    primary_key,
):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "053_validate_lifecycle_schema"
    )
    errors = []

    result = migration._primary_key_columns(
        {"columns": [{"name": "id"}], "primary_key": primary_key},
        "048_add_training_process_identity",
        "training_tasks",
        errors,
    )

    assert result is None
    assert errors == [
        {
            "revision": "048_add_training_process_identity",
            "category": "primary_key_metadata",
            "object_name": "training_tasks",
            "count": 1,
        }
    ]


@pytest.mark.parametrize(
    "column_definition",
    ["VARCHAR(36)", "DOUBLE NOT NULL"],
    ids=["wrong-type", "not-null"],
)
def test_training_process_identity_downgrade_preserves_malformed_column(
    monkeypatch,
    column_definition,
):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE training_tasks ("
                "task_id VARCHAR(36) NOT NULL, "
                f"process_create_time {column_definition})"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_table_schema_state(connection, "training_tasks")

        with pytest.raises(RuntimeError, match="must be nullable DOUBLE"):
            migration.downgrade()

        assert _sqlite_table_schema_state(connection, "training_tasks") == before


@pytest.mark.parametrize(
    ("module_name", "table_name", "index_name"),
    _RUN_TOKEN_MIGRATION_CASES,
    ids=["evaluation", "generation"],
)
@pytest.mark.parametrize(
    "column_definition",
    ["VARCHAR(35)", "VARCHAR(36) NOT NULL"],
    ids=["wrong-length", "not-null"],
)
def test_run_token_upgrade_rejects_malformed_column_without_mutation(
    monkeypatch,
    module_name,
    table_name,
    index_name,
    column_definition,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_run_token_table(
            connection,
            table_name,
            run_token_definition=column_definition,
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_table_schema_state(connection, table_name)

        with pytest.raises(RuntimeError, match="must be nullable VARCHAR\\(36\\)"):
            migration.upgrade()

        assert _sqlite_table_schema_state(connection, table_name) == before
        assert index_name not in {index[0] for index in before["indexes"]}


@pytest.mark.parametrize(
    ("module_name", "table_name", "index_name"),
    _RUN_TOKEN_MIGRATION_CASES,
    ids=["evaluation", "generation"],
)
@pytest.mark.parametrize(
    ("indexed_column", "unique"),
    [("task_id", False), ("run_token", True)],
    ids=["wrong-columns", "unique"],
)
def test_run_token_upgrade_rejects_malformed_index_without_mutation(
    monkeypatch,
    module_name,
    table_name,
    index_name,
    indexed_column,
    unique,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_run_token_table(
            connection,
            table_name,
            run_token_definition="VARCHAR(36)",
        )
        unique_sql = "UNIQUE " if unique else ""
        connection.execute(
            sa.text(
                f"CREATE {unique_sql}INDEX {index_name} "
                f"ON {table_name} ({indexed_column})"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_table_schema_state(connection, table_name)

        with pytest.raises(
            RuntimeError,
            match="must be a non-unique index on \\('run_token',\\)",
        ):
            migration.upgrade()

        assert _sqlite_table_schema_state(connection, table_name) == before


@pytest.mark.parametrize(
    ("module_name", "table_name", "index_name"),
    _RUN_TOKEN_MIGRATION_CASES,
    ids=["evaluation", "generation"],
)
@pytest.mark.parametrize(
    "column_exists",
    [False, True],
    ids=["missing-column-and-index", "missing-index"],
)
def test_run_token_upgrade_completes_partial_schema_idempotently(
    monkeypatch,
    module_name,
    table_name,
    index_name,
    column_exists,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_run_token_table(
            connection,
            table_name,
            run_token_definition="VARCHAR(36)" if column_exists else None,
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)

        migration.upgrade()
        migration.upgrade()

        inspector = sa.inspect(connection)
        column = next(
            column
            for column in inspector.get_columns(table_name)
            if column["name"] == "run_token"
        )
        index = next(
            index
            for index in inspector.get_indexes(table_name)
            if index["name"] == index_name
        )

    assert getattr(column["type"], "__visit_name__", "").lower() == "varchar"
    assert column["type"].length == 36
    assert column["nullable"] is True
    assert tuple(index["column_names"]) == ("run_token",)
    assert index["unique"] in (False, 0)


@pytest.mark.parametrize(
    ("module_name", "table_name", "index_name"),
    _RUN_TOKEN_MIGRATION_CASES,
    ids=["evaluation", "generation"],
)
@pytest.mark.parametrize(
    "column_definition",
    ["VARCHAR(35)", "VARCHAR(36) NOT NULL"],
    ids=["wrong-length", "not-null"],
)
def test_run_token_downgrade_preserves_malformed_column_and_index(
    monkeypatch,
    module_name,
    table_name,
    index_name,
    column_definition,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_run_token_table(
            connection,
            table_name,
            run_token_definition=column_definition,
        )
        connection.execute(
            sa.text(
                f"CREATE INDEX {index_name} ON {table_name} (run_token)"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_table_schema_state(connection, table_name)

        with pytest.raises(RuntimeError, match="must be nullable VARCHAR\\(36\\)"):
            migration.downgrade()

        assert _sqlite_table_schema_state(connection, table_name) == before


@pytest.mark.parametrize(
    ("module_name", "table_name", "index_name"),
    _RUN_TOKEN_MIGRATION_CASES,
    ids=["evaluation", "generation"],
)
@pytest.mark.parametrize(
    ("indexed_column", "unique"),
    [("task_id", False), ("run_token", True)],
    ids=["wrong-columns", "unique"],
)
def test_run_token_downgrade_preserves_malformed_index_and_column(
    monkeypatch,
    module_name,
    table_name,
    index_name,
    indexed_column,
    unique,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_run_token_table(
            connection,
            table_name,
            run_token_definition="VARCHAR(36)",
        )
        unique_sql = "UNIQUE " if unique else ""
        connection.execute(
            sa.text(
                f"CREATE {unique_sql}INDEX {index_name} "
                f"ON {table_name} ({indexed_column})"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_table_schema_state(connection, table_name)

        with pytest.raises(
            RuntimeError,
            match="must be a non-unique index on \\('run_token',\\)",
        ):
            migration.downgrade()

        assert _sqlite_table_schema_state(connection, table_name) == before


_DEPENDENT_INDEX_CASES = (
    ("non-unique-single", False, ("target",)),
    ("unique-single", True, ("target",)),
    ("non-unique-target-first", False, ("target", "task_id")),
    ("non-unique-target-last", False, ("task_id", "target")),
    ("unique-target-first", True, ("target", "task_id")),
    ("unique-target-last", True, ("task_id", "target")),
)


def _sqlite_full_table_state(connection, table_name):
    schema = connection.execute(
        sa.text(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE tbl_name = :table_name "
            "AND type IN ('table', 'index') ORDER BY type, name"
        ),
        {"table_name": table_name},
    ).all()
    rows = connection.execute(
        sa.text(f"SELECT * FROM {table_name} ORDER BY task_id")
    ).all()
    return tuple(schema), tuple(rows)


def _create_auxiliary_index(
    connection,
    *,
    table_name,
    index_name,
    target_column,
    unique,
    columns,
):
    resolved_columns = tuple(
        target_column if column == "target" else column for column in columns
    )
    unique_sql = "UNIQUE " if unique else ""
    connection.execute(
        sa.text(
            f"CREATE {unique_sql}INDEX {index_name} ON {table_name} "
            f"({', '.join(resolved_columns)})"
        )
    )


@pytest.mark.parametrize(
    ("case_name", "unique", "columns"),
    _DEPENDENT_INDEX_CASES,
    ids=[case[0] for case in _DEPENDENT_INDEX_CASES],
)
def test_training_process_identity_downgrade_preserves_dependent_auxiliary_index(
    monkeypatch,
    case_name,
    unique,
    columns,
):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE training_tasks ("
                "task_id VARCHAR(36) NOT NULL, process_create_time DOUBLE)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO training_tasks "
                "(task_id, process_create_time) VALUES ('task-1', 1234.5)"
            )
        )
        index_name = f"aux_training_process_{case_name.replace('-', '_')}"
        _create_auxiliary_index(
            connection,
            table_name="training_tasks",
            index_name=index_name,
            target_column="process_create_time",
            unique=unique,
            columns=columns,
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_full_table_state(connection, "training_tasks")

        with pytest.raises(RuntimeError, match="unexpected index") as exc_info:
            migration.downgrade()

        assert index_name in str(exc_info.value)
        assert "process_create_time" in str(exc_info.value)
        assert _sqlite_full_table_state(connection, "training_tasks") == before


def test_training_process_identity_downgrade_removes_valid_column(monkeypatch):
    migration = _import_migration(
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity"
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE training_tasks ("
                "task_id VARCHAR(36) NOT NULL, process_create_time DOUBLE)"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)

        migration.downgrade()

        columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns("training_tasks")
        }

    assert columns == {"task_id"}


@pytest.mark.parametrize(
    ("module_name", "table_name", "index_name"),
    _RUN_TOKEN_MIGRATION_CASES,
    ids=["evaluation", "generation"],
)
@pytest.mark.parametrize(
    ("case_name", "unique", "columns"),
    _DEPENDENT_INDEX_CASES,
    ids=[case[0] for case in _DEPENDENT_INDEX_CASES],
)
def test_run_token_downgrade_preserves_expected_and_dependent_auxiliary_indexes(
    monkeypatch,
    module_name,
    table_name,
    index_name,
    case_name,
    unique,
    columns,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_run_token_table(
            connection,
            table_name,
            run_token_definition="VARCHAR(36)",
        )
        connection.execute(
            sa.text(f"CREATE INDEX {index_name} ON {table_name} (run_token)")
        )
        auxiliary_name = f"aux_{table_name}_{case_name.replace('-', '_')}"
        _create_auxiliary_index(
            connection,
            table_name=table_name,
            index_name=auxiliary_name,
            target_column="run_token",
            unique=unique,
            columns=columns,
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_full_table_state(connection, table_name)

        with pytest.raises(RuntimeError, match="unexpected index") as exc_info:
            migration.downgrade()

        assert auxiliary_name in str(exc_info.value)
        assert "run_token" in str(exc_info.value)
        assert _sqlite_full_table_state(connection, table_name) == before


@pytest.mark.parametrize(
    ("module_name", "table_name", "index_name"),
    _RUN_TOKEN_MIGRATION_CASES,
    ids=["evaluation", "generation"],
)
def test_run_token_downgrade_removes_valid_expected_objects(
    monkeypatch,
    module_name,
    table_name,
    index_name,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_run_token_table(
            connection,
            table_name,
            run_token_definition="VARCHAR(36)",
        )
        connection.execute(
            sa.text(f"CREATE INDEX {index_name} ON {table_name} (run_token)")
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)

        migration.downgrade()

        inspector = sa.inspect(connection)
        columns = {column["name"] for column in inspector.get_columns(table_name)}
        indexes = {index["name"] for index in inspector.get_indexes(table_name)}

    assert columns == {"task_id"}
    assert index_name not in indexes


_DEPENDENCY_MIGRATION_CASES = (
    (
        "train_factory.storage.migrations.versions."
        "048_add_training_process_identity",
        "training_tasks",
        "process_create_time",
        "DOUBLE",
        None,
    ),
    (
        "train_factory.storage.migrations.versions."
        "050_add_evaluation_run_token",
        "evaluation_tasks",
        "run_token",
        "VARCHAR(36)",
        "idx_eval_run_token",
    ),
    (
        "train_factory.storage.migrations.versions."
        "051_add_generation_run_token",
        "generation_tasks",
        "run_token",
        "VARCHAR(36)",
        "idx_gen_task_run_token",
    ),
)


def _create_dependency_schema(
    connection,
    *,
    table_name,
    target_column,
    target_definition,
    constraint_sql="",
    expected_index_name=None,
):
    connection.execute(
        sa.text(
            f"CREATE TABLE {table_name} ("
            "task_id VARCHAR(36) NOT NULL, "
            f"{target_column} {target_definition}{constraint_sql})"
        )
    )
    if expected_index_name is not None:
        connection.execute(
            sa.text(
                f"CREATE INDEX {expected_index_name} "
                f"ON {table_name} ({target_column})"
            )
        )


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize(
    (
        "module_name",
        "table_name",
        "target_column",
        "target_definition",
        "expected_index_name",
    ),
    _DEPENDENCY_MIGRATION_CASES,
    ids=["training", "evaluation", "generation"],
)
def test_identity_migration_preserves_unexpected_target_unique_constraint(
    monkeypatch,
    direction,
    module_name,
    table_name,
    target_column,
    target_definition,
    expected_index_name,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_dependency_schema(
            connection,
            table_name=table_name,
            target_column=target_column,
            target_definition=target_definition,
            constraint_sql=(
                f", CONSTRAINT foreign_unique UNIQUE ({target_column})"
            ),
            expected_index_name=expected_index_name,
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_full_table_state(connection, table_name)

        with pytest.raises(
            RuntimeError,
            match="unexpected unique constraint",
        ) as exc_info:
            getattr(migration, direction)()

        assert "foreign_unique" in str(exc_info.value)
        assert target_column in str(exc_info.value)
        assert _sqlite_full_table_state(connection, table_name) == before


@pytest.mark.parametrize(
    (
        "module_name",
        "table_name",
        "target_column",
        "target_definition",
        "expected_index_name",
    ),
    _DEPENDENCY_MIGRATION_CASES,
    ids=["training", "evaluation", "generation"],
)
def test_identity_upgrade_preserves_unexpected_target_unique_index(
    monkeypatch,
    module_name,
    table_name,
    target_column,
    target_definition,
    expected_index_name,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_dependency_schema(
            connection,
            table_name=table_name,
            target_column=target_column,
            target_definition=target_definition,
            expected_index_name=expected_index_name,
        )
        auxiliary_name = f"foreign_unique_{target_column}"
        connection.execute(
            sa.text(
                f"CREATE UNIQUE INDEX {auxiliary_name} "
                f"ON {table_name} ({target_column})"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_full_table_state(connection, table_name)

        with pytest.raises(RuntimeError, match="unexpected unique index") as exc_info:
            migration.upgrade()

        assert auxiliary_name in str(exc_info.value)
        assert _sqlite_full_table_state(connection, table_name) == before


@pytest.mark.parametrize(
    (
        "module_name",
        "table_name",
        "target_column",
        "target_definition",
        "expected_index_name",
    ),
    _DEPENDENCY_MIGRATION_CASES,
    ids=["training", "evaluation", "generation"],
)
def test_identity_upgrade_rejects_unknown_target_index_uniqueness(
    monkeypatch,
    module_name,
    table_name,
    target_column,
    target_definition,
    expected_index_name,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_dependency_schema(
            connection,
            table_name=table_name,
            target_column=target_column,
            target_definition=target_definition,
            expected_index_name=expected_index_name,
        )
        auxiliary_name = f"unknown_unique_{target_column}"
        connection.execute(
            sa.text(
                f"CREATE INDEX {auxiliary_name} "
                f"ON {table_name} ({target_column})"
            )
        )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        actual_inspect = sa.inspect

        class UnknownUniqueInspector:
            def __init__(self, bind):
                self.inspector = actual_inspect(bind)

            def get_table_names(self):
                return self.inspector.get_table_names()

            def get_columns(self, inspected_table):
                return self.inspector.get_columns(inspected_table)

            def get_indexes(self, inspected_table):
                indexes = self.inspector.get_indexes(inspected_table)
                for index in indexes:
                    if index["name"] == auxiliary_name:
                        index["unique"] = None
                return indexes

            def get_unique_constraints(self, inspected_table):
                return self.inspector.get_unique_constraints(inspected_table)

        monkeypatch.setattr(
            migration,
            "inspect",
            lambda bind: UnknownUniqueInspector(bind),
        )
        before = _sqlite_full_table_state(connection, table_name)

        with pytest.raises(RuntimeError, match="unknown uniqueness") as exc_info:
            migration.upgrade()

        assert auxiliary_name in str(exc_info.value)
        assert _sqlite_full_table_state(connection, table_name) == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize(
    (
        "module_name",
        "table_name",
        "target_column",
        "target_definition",
        "expected_index_name",
    ),
    _DEPENDENCY_MIGRATION_CASES,
    ids=["training", "evaluation", "generation"],
)
def test_identity_migration_preserves_computed_target_column_before_mutation(
    monkeypatch,
    direction,
    module_name,
    table_name,
    target_column,
    target_definition,
    expected_index_name,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                f"CREATE TABLE {table_name} ("
                "task_id VARCHAR(36) NOT NULL, "
                f"source_value {target_definition}, "
                f"{target_column} {target_definition} GENERATED ALWAYS AS "
                "(source_value) VIRTUAL)"
            )
        )
        connection.execute(
            sa.text(
                f"INSERT INTO {table_name} (task_id, source_value) "
                "VALUES ('task-1', 'source-1')"
            )
        )
        if direction == "downgrade" and expected_index_name is not None:
            connection.execute(
                sa.text(
                    f"CREATE INDEX {expected_index_name} "
                    f"ON {table_name} ({target_column})"
                )
            )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_full_table_state(connection, table_name)

        with pytest.raises(RuntimeError, match="computed column") as exc_info:
            getattr(migration, direction)()

        assert target_column in str(exc_info.value)
        assert _sqlite_full_table_state(connection, table_name) == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize(
    "column_shape",
    ["server-default", "primary-key"],
)
@pytest.mark.parametrize(
    (
        "module_name",
        "table_name",
        "target_column",
        "target_definition",
        "expected_index_name",
    ),
    _DEPENDENCY_MIGRATION_CASES,
    ids=["training", "evaluation", "generation"],
)
def test_identity_migration_preserves_non_plain_target_column_before_mutation(
    monkeypatch,
    direction,
    column_shape,
    module_name,
    table_name,
    target_column,
    target_definition,
    expected_index_name,
):
    migration = _import_migration(module_name)
    if column_shape == "server-default":
        shape_sql = " DEFAULT 0.0" if target_definition == "DOUBLE" else " DEFAULT 'fixed'"
        expected_error = "server default"
    else:
        shape_sql = " PRIMARY KEY"
        expected_error = "primary key"
    target_value = "1234.5" if target_definition == "DOUBLE" else "run-1"

    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                f"CREATE TABLE {table_name} ("
                "task_id VARCHAR(36) NOT NULL, "
                f"{target_column} {target_definition}{shape_sql})"
            )
        )
        connection.execute(
            sa.text(
                f"INSERT INTO {table_name} (task_id, {target_column}) "
                "VALUES ('task-1', :target_value)"
            ),
            {"target_value": target_value},
        )
        if direction == "downgrade" and expected_index_name is not None:
            connection.execute(
                sa.text(
                    f"CREATE INDEX {expected_index_name} "
                    f"ON {table_name} ({target_column})"
                )
            )
        _use_real_alembic_operations(monkeypatch, migration, connection)
        before = _sqlite_full_table_state(connection, table_name)

        with pytest.raises(RuntimeError, match=expected_error) as exc_info:
            getattr(migration, direction)()

        assert target_column in str(exc_info.value)
        assert _sqlite_full_table_state(connection, table_name) == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize(
    (
        "module_name",
        "table_name",
        "target_column",
        "target_definition",
        "expected_index_name",
    ),
    _DEPENDENCY_MIGRATION_CASES,
    ids=["training", "evaluation", "generation"],
)
def test_identity_migration_preserves_identity_target_column_before_mutation(
    monkeypatch,
    direction,
    module_name,
    table_name,
    target_column,
    target_definition,
    expected_index_name,
):
    migration = _import_migration(module_name)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_dependency_schema(
            connection,
            table_name=table_name,
            target_column=target_column,
            target_definition=target_definition,
            expected_index_name=(
                expected_index_name if direction == "downgrade" else None
            ),
        )
        actual_inspect = sa.inspect

        class IdentityInspector:
            def __init__(self, bind):
                self.inspector = actual_inspect(bind)

            def get_table_names(self):
                return self.inspector.get_table_names()

            def get_columns(self, inspected_table):
                columns = self.inspector.get_columns(inspected_table)
                for column in columns:
                    if column["name"] == target_column:
                        column["identity"] = {"always": False}
                return columns

            def get_indexes(self, inspected_table):
                return self.inspector.get_indexes(inspected_table)

            def get_unique_constraints(self, inspected_table):
                return self.inspector.get_unique_constraints(inspected_table)

        _use_real_alembic_operations(monkeypatch, migration, connection)
        monkeypatch.setattr(
            migration,
            "inspect",
            lambda bind: IdentityInspector(bind),
        )
        before = _sqlite_full_table_state(connection, table_name)

        with pytest.raises(RuntimeError, match="identity column") as exc_info:
            getattr(migration, direction)()

        assert target_column in str(exc_info.value)
        assert _sqlite_full_table_state(connection, table_name) == before
