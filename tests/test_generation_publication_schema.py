"""Schema contract for attempt-owned generation publication staging."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
import sqlalchemy as sa

from train_factory.enums.dataset_status import DatasetStatus
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.milvus_collection_entity import (
    CollectionDatasetLinkDB,
    MilvusCollectionDB,
)


MIGRATION_MODULE = (
    "train_factory.storage.migrations.versions."
    "052_generation_publication_staging"
)
VALIDATION_MIGRATION_MODULE = (
    "train_factory.storage.migrations.versions."
    "053_validate_lifecycle_schema"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_COLUMN_SPECS = {
    "datasets": ("generation_run_token",),
    "milvus_collections": (
        "generation_task_id",
        "generation_run_token",
    ),
    "collection_dataset_links": ("generation_run_token",),
}
EXPECTED_INDEX_SPECS = {
    "datasets": {
        "idx_dataset_generation_run_token": ("generation_run_token",),
    },
    "milvus_collections": {
        "idx_milvus_generation_attempt": (
            "generation_task_id",
            "generation_run_token",
        ),
    },
    "collection_dataset_links": {
        "idx_coll_link_generation_run_token": ("generation_run_token",),
    },
}


def _migration():
    try:
        return importlib.import_module(MIGRATION_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == "alembic":
            alembic_stub = ModuleType("alembic")
            alembic_stub.op = object()
            sys.modules["alembic"] = alembic_stub
            try:
                return importlib.import_module(MIGRATION_MODULE)
            finally:
                sys.modules.pop("alembic", None)
        if exc.name != MIGRATION_MODULE:
            raise
        pytest.fail("missing generation publication staging migration 052")


def _validation_migration():
    try:
        return importlib.import_module(VALIDATION_MIGRATION_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == "alembic":
            alembic_stub = ModuleType("alembic")
            alembic_stub.op = object()
            sys.modules["alembic"] = alembic_stub
            try:
                migration = importlib.import_module(VALIDATION_MIGRATION_MODULE)
                migration._load_lifecycle_migrations()
                return migration
            finally:
                sys.modules.pop("alembic", None)
        if exc.name != VALIDATION_MIGRATION_MODULE:
            raise
        pytest.fail("missing lifecycle schema validation migration 053")


def _valid_lifecycle_reflection():
    columns = {
        "training_tasks": [
            {
                "name": "id",
                "type": sa.Integer(),
                "nullable": False,
                "primary_key": True,
            },
            {
                "name": "process_create_time",
                "type": sa.DOUBLE(),
                "nullable": True,
                "default": None,
                "computed": None,
                "identity": None,
                "primary_key": False,
            },
        ],
        "dataset_lineage_edges": [
            {
                "name": "id",
                "type": sa.Integer(),
                "nullable": False,
                "primary_key": True,
            },
            {"name": "edge_id", "type": sa.VARCHAR(36), "nullable": False},
            {"name": "from_dataset_id", "type": sa.VARCHAR(36), "nullable": True},
            {"name": "to_dataset_id", "type": sa.VARCHAR(36), "nullable": False},
            {"name": "relation_type", "type": sa.VARCHAR(32), "nullable": False},
            {
                "name": "edge_key",
                "type": sa.VARCHAR(64),
                "nullable": False,
                "default": None,
                "computed": None,
                "identity": None,
                "primary_key": False,
            },
        ],
        "evaluation_tasks": [
            {
                "name": "id",
                "type": sa.Integer(),
                "nullable": False,
                "primary_key": True,
            },
            {
                "name": "run_token",
                "type": sa.VARCHAR(36),
                "nullable": True,
                "default": None,
                "computed": None,
                "identity": None,
                "primary_key": False,
            },
        ],
        "generation_tasks": [
            {
                "name": "id",
                "type": sa.Integer(),
                "nullable": False,
                "primary_key": True,
            },
            {
                "name": "run_token",
                "type": sa.VARCHAR(36),
                "nullable": True,
                "default": None,
                "computed": None,
                "identity": None,
                "primary_key": False,
            },
        ],
    }
    publication_columns, publication_indexes = _valid_reflected_schema()
    columns.update(publication_columns)
    indexes = {
        "training_tasks": [],
        "dataset_lineage_edges": [],
        "evaluation_tasks": [
            {
                "name": "idx_eval_run_token",
                "column_names": ["run_token"],
                "unique": False,
            }
        ],
        "generation_tasks": [
            {
                "name": "idx_gen_task_run_token",
                "column_names": ["run_token"],
                "unique": False,
            }
        ],
        **publication_indexes,
    }
    unique_constraints = {table_name: [] for table_name in columns}
    unique_constraints["dataset_lineage_edges"] = [
        {
            "name": "uq_lineage_edge_key",
            "column_names": ["edge_key"],
        }
    ]
    return columns, indexes, unique_constraints


class _LifecycleInspector:
    def __init__(self, columns, indexes, unique_constraints):
        self.columns = columns
        self.indexes = indexes
        self.unique_constraints = unique_constraints

    def get_table_names(self):
        return list(self.columns)

    def get_columns(self, table_name):
        return self.columns[table_name]

    def get_indexes(self, table_name):
        return self.indexes[table_name]

    def get_unique_constraints(self, table_name):
        return self.unique_constraints[table_name]

    def get_pk_constraint(self, table_name):
        assert table_name in self.columns
        return {"name": "PRIMARY", "constrained_columns": ["id"]}


class _EmptyRows:
    @staticmethod
    def mappings():
        return []


class _ReadOnlyLifecycleBind:
    def __init__(self):
        self.dialect = SimpleNamespace(name="mysql")
        self.statements = []

    def execute(self, statement, *_args, **_kwargs):
        sql = str(statement)
        self.statements.append(sql)
        assert sql.lstrip().upper().startswith("SELECT")
        return _EmptyRows()


class _PrivateOperationalError(Exception):
    pass


class _RawReflectionFailureInspector:
    def __init__(self, delegate, table_name, exception_type):
        self._delegate = delegate
        self._table_name = table_name
        self._exception_type = exception_type

    def get_table_names(self):
        return self._delegate.get_table_names()

    def get_columns(self, table_name):
        if table_name == self._table_name:
            raise self._exception_type("raw-reflection-private-canary")
        return self._delegate.get_columns(table_name)

    def get_indexes(self, table_name):
        return self._delegate.get_indexes(table_name)

    def get_unique_constraints(self, table_name):
        return self._delegate.get_unique_constraints(table_name)

    def get_pk_constraint(self, table_name):
        return self._delegate.get_pk_constraint(table_name)


def _install_lifecycle_inspector(monkeypatch, migration, inspector):
    monkeypatch.setattr(
        migration,
        "inspect",
        lambda _bind: inspector,
        raising=False,
    )
    alembic_stub = ModuleType("alembic")
    alembic_stub.op = object()
    monkeypatch.setitem(sys.modules, "alembic", alembic_stub)
    historical_migrations = migration._load_lifecycle_migrations()
    monkeypatch.delitem(sys.modules, "alembic")
    for historical in historical_migrations:
        monkeypatch.setattr(historical, "inspect", lambda _bind: inspector)


def _index_columns(table) -> dict[str, tuple[str, ...]]:
    return {
        index.name: tuple(column.name for column in index.columns)
        for index in table.indexes
    }


def _valid_reflected_schema():
    columns = {}
    indexes = {}
    for table_name, column_names in EXPECTED_COLUMN_SPECS.items():
        columns[table_name] = [
            {
                "name": "id",
                "type": sa.Integer(),
                "nullable": False,
                "primary_key": True,
            },
            *(
                {
                    "name": column_name,
                    "type": sa.VARCHAR(length=36),
                    "nullable": True,
                    "default": None,
                    "computed": None,
                    "identity": None,
                    "primary_key": False,
                }
                for column_name in column_names
            ),
        ]
        indexes[table_name] = [
            {
                "name": index_name,
                "column_names": list(index_columns),
                "unique": False,
            }
            for index_name, index_columns in EXPECTED_INDEX_SPECS[
                table_name
            ].items()
        ]
    return columns, indexes


class _ReflectedInspector:
    def __init__(self, columns, indexes):
        self.columns = columns
        self.indexes = indexes

    def get_table_names(self):
        return list(self.columns)

    def get_columns(self, table_name):
        return self.columns[table_name]

    def get_indexes(self, table_name):
        return self.indexes[table_name]

    def get_pk_constraint(self, table_name):
        assert table_name in self.columns
        return {"name": "PRIMARY", "constrained_columns": ["id"]}


class _RecordingOperations:
    def __init__(self):
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
        self.mutations = []

    def get_bind(self):
        return self.bind

    def add_column(self, *args, **kwargs):
        self.mutations.append(("add_column", args, kwargs))

    def create_index(self, *args, **kwargs):
        self.mutations.append(("create_index", args, kwargs))

    def drop_index(self, *args, **kwargs):
        self.mutations.append(("drop_index", args, kwargs))

    def drop_column(self, *args, **kwargs):
        self.mutations.append(("drop_column", args, kwargs))


class _BoundRecordingOperations(_RecordingOperations):
    def __init__(self, bind):
        super().__init__()
        self.bind = bind


def _create_complete_lifecycle_sqlite_schema(
    connection,
    *,
    edge_key_definition="VARCHAR(64) NOT NULL",
):
    statements = (
        "CREATE TABLE training_tasks ("
        "id INTEGER PRIMARY KEY, process_create_time DOUBLE NULL)",
        "CREATE TABLE dataset_lineage_edges ("
        "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
        f"edge_key {edge_key_definition}, from_dataset_id VARCHAR(36), "
        "to_dataset_id VARCHAR(36) NOT NULL, relation_type VARCHAR(32) NOT NULL, "
        "op_task_type VARCHAR(32), op_task_id VARCHAR(36), op_params JSON, "
        "CONSTRAINT uq_lineage_edge_key UNIQUE (edge_key))",
        "CREATE TABLE evaluation_tasks ("
        "id INTEGER PRIMARY KEY, run_token VARCHAR(36) NULL)",
        "CREATE INDEX idx_eval_run_token ON evaluation_tasks (run_token)",
        "CREATE TABLE generation_tasks ("
        "id INTEGER PRIMARY KEY, run_token VARCHAR(36) NULL)",
        "CREATE INDEX idx_gen_task_run_token ON generation_tasks (run_token)",
        "CREATE TABLE datasets ("
        "id INTEGER PRIMARY KEY, generation_run_token VARCHAR(36) NULL)",
        "CREATE INDEX idx_dataset_generation_run_token "
        "ON datasets (generation_run_token)",
        "CREATE TABLE milvus_collections ("
        "id INTEGER PRIMARY KEY, generation_task_id VARCHAR(36) NULL, "
        "generation_run_token VARCHAR(36) NULL)",
        "CREATE INDEX idx_milvus_generation_attempt "
        "ON milvus_collections (generation_task_id, generation_run_token)",
        "CREATE TABLE collection_dataset_links ("
        "id INTEGER PRIMARY KEY, generation_run_token VARCHAR(36) NULL)",
        "CREATE INDEX idx_coll_link_generation_run_token "
        "ON collection_dataset_links (generation_run_token)",
    )
    for statement in statements:
        connection.execute(sa.text(statement))


def _install_bound_lifecycle_operations(monkeypatch, migration, connection):
    operations = _BoundRecordingOperations(connection)
    monkeypatch.setattr(migration, "op", operations)
    for historical in migration._load_lifecycle_migrations():
        monkeypatch.setattr(historical, "op", operations)
    return operations


def _malformed_schema_cases():
    cases = []
    for table_name, column_names in EXPECTED_COLUMN_SPECS.items():
        for column_name in column_names:
            cases.extend(
                (
                    ("column_length", table_name, column_name),
                    ("column_nullable", table_name, column_name),
                )
            )
    for table_name, index_specs in EXPECTED_INDEX_SPECS.items():
        for index_name, index_columns in index_specs.items():
            cases.extend(
                (
                    ("index_columns", table_name, index_name),
                    ("index_unique", table_name, index_name),
                )
            )
            if len(index_columns) > 1:
                cases.append(("index_order", table_name, index_name))
    return cases


MALFORMED_SCHEMA_CASES = _malformed_schema_cases()


def _malformed_reflected_schema(kind, table_name, object_name):
    columns, indexes = _valid_reflected_schema()
    if kind.startswith("column_"):
        column = next(
            item
            for item in columns[table_name]
            if item["name"] == object_name
        )
        if kind == "column_length":
            column["type"] = sa.VARCHAR(length=12)
        else:
            column["nullable"] = False
    else:
        index = next(
            item
            for item in indexes[table_name]
            if item["name"] == object_name
        )
        if kind == "index_columns":
            index["column_names"] = ["id"]
        elif kind == "index_order":
            index["column_names"] = list(reversed(index["column_names"]))
        else:
            index["unique"] = True
    return columns, indexes


def _remove_valid_mutation_probe(columns, indexes, table_name, object_name):
    probe_table = "collection_dataset_links"
    probe_column = "generation_run_token"
    if table_name == probe_table and object_name in {
        probe_column,
        "idx_coll_link_generation_run_token",
    }:
        probe_table = "datasets"

    columns[probe_table] = [
        column
        for column in columns[probe_table]
        if column["name"] != probe_column
    ]
    indexes[probe_table] = []


def test_generation_publication_entities_persist_internal_attempt_ownership():
    assert DatasetStatus.STAGING.value == "staging"

    dataset_token = DatasetDB.__table__.columns.get("generation_run_token")
    assert dataset_token is not None
    assert dataset_token.type.length == 36
    assert dataset_token.nullable is True
    assert _index_columns(DatasetDB.__table__)[
        "idx_dataset_generation_run_token"
    ] == ("generation_run_token",)

    collection_task = MilvusCollectionDB.__table__.columns.get(
        "generation_task_id"
    )
    collection_token = MilvusCollectionDB.__table__.columns.get(
        "generation_run_token"
    )
    assert collection_task is not None
    assert collection_token is not None
    assert collection_task.type.length == 36
    assert collection_token.type.length == 36
    assert collection_task.nullable is True
    assert collection_token.nullable is True
    assert _index_columns(MilvusCollectionDB.__table__)[
        "idx_milvus_generation_attempt"
    ] == ("generation_task_id", "generation_run_token")

    link_token = CollectionDatasetLinkDB.__table__.columns.get(
        "generation_run_token"
    )
    assert link_token is not None
    assert link_token.type.length == 36
    assert link_token.nullable is True
    assert _index_columns(CollectionDatasetLinkDB.__table__)[
        "idx_coll_link_generation_run_token"
    ] == ("generation_run_token",)

    staging = DatasetDB(
        dataset_name="attempt-owned",
        status=DatasetStatus.STAGING.value,
        generation_run_token="internal-token",
    )
    staging.update_status(DatasetStatus.READY.value)
    assert staging.status == DatasetStatus.READY.value
    assert "generation_run_token" not in staging.to_dict()
    assert "generation_run_token" not in MilvusCollectionDB(
        collection_name="attempt-owned",
        generation_task_id="generation-task",
        generation_run_token="internal-token",
    ).to_dict()
    assert "generation_run_token" not in CollectionDatasetLinkDB(
        collection_name="attempt-owned",
        dataset_id="dataset-id",
        generation_run_token="internal-token",
    ).to_dict()


def test_init_sql_adds_only_attempt_metadata_to_existing_dataset_table():
    sql = (PROJECT_ROOT / "docker" / "init.sql").read_text(
        encoding="utf-8"
    ).lower()

    datasets = sql.split("create table if not exists datasets (", 1)[1].split(
        ") engine=", 1
    )[0]
    assert "generation_run_token varchar(36)" in datasets
    assert (
        "index idx_dataset_generation_run_token (generation_run_token)"
        in datasets
    )
    assert "create table if not exists milvus_collections" not in sql
    assert "create table if not exists collection_dataset_links" not in sql
    assert "create table if not exists generation_tasks" not in sql


def test_generation_publication_migration_is_the_single_head():
    pytest.importorskip("alembic")
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    migration = _migration()
    assert migration.revision == "052_generation_publication_staging"
    assert migration.down_revision == "051_add_generation_run_token"

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


def test_generation_publication_migration_round_trips_idempotently_on_sqlite(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE datasets ("
                "id INTEGER PRIMARY KEY, dataset_id VARCHAR(36))"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE milvus_collections ("
                "id INTEGER PRIMARY KEY, collection_id VARCHAR(36))"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE collection_dataset_links ("
                "id INTEGER PRIMARY KEY, dataset_id VARCHAR(36))"
            )
        )
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )

        migration.upgrade()
        migration.upgrade()

        inspector = sa.inspect(connection)
        dataset_columns = {
            column["name"]: column
            for column in inspector.get_columns("datasets")
        }
        collection_columns = {
            column["name"]: column
            for column in inspector.get_columns("milvus_collections")
        }
        link_columns = {
            column["name"]: column
            for column in inspector.get_columns("collection_dataset_links")
        }
        assert dataset_columns["generation_run_token"]["type"].length == 36
        assert collection_columns["generation_task_id"]["type"].length == 36
        assert collection_columns["generation_run_token"]["type"].length == 36
        assert link_columns["generation_run_token"]["type"].length == 36
        assert {
            index["name"] for index in inspector.get_indexes("datasets")
        } == {"idx_dataset_generation_run_token"}
        assert {
            index["name"]
            for index in inspector.get_indexes("milvus_collections")
        } == {"idx_milvus_generation_attempt"}
        assert {
            index["name"]
            for index in inspector.get_indexes("collection_dataset_links")
        } == {"idx_coll_link_generation_run_token"}

        migration.downgrade()
        migration.downgrade()

        inspector = sa.inspect(connection)
        assert "generation_run_token" not in {
            column["name"] for column in inspector.get_columns("datasets")
        }
        assert {
            "generation_task_id",
            "generation_run_token",
        }.isdisjoint(
            column["name"]
            for column in inspector.get_columns("milvus_collections")
        )
        assert "generation_run_token" not in {
            column["name"]
            for column in inspector.get_columns("collection_dataset_links")
        }


def test_generation_publication_upgrade_completes_valid_partial_sqlite_schema(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE datasets ("
                "id INTEGER PRIMARY KEY, "
                "generation_run_token VARCHAR(36) NULL)"
            )
        )
        connection.execute(
            sa.text(
                "CREATE INDEX idx_dataset_generation_run_token "
                "ON datasets (generation_run_token)"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE milvus_collections ("
                "id INTEGER PRIMARY KEY, "
                "generation_task_id VARCHAR(36) NULL)"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE collection_dataset_links ("
                "id INTEGER PRIMARY KEY, "
                "generation_run_token VARCHAR(36) NULL)"
            )
        )
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )

        migration.upgrade()

        inspector = sa.inspect(connection)
        assert {
            column["name"]
            for column in inspector.get_columns("milvus_collections")
        } >= {"generation_task_id", "generation_run_token"}
        assert {
            index["name"]
            for index in inspector.get_indexes("milvus_collections")
        } >= {"idx_milvus_generation_attempt"}
        assert {
            index["name"]
            for index in inspector.get_indexes("collection_dataset_links")
        } >= {"idx_coll_link_generation_run_token"}


def test_generation_publication_migration_emits_mysql_portable_ddl(monkeypatch):
    migration = _migration()

    class Inspector:
        @staticmethod
        def get_table_names():
            return [
                "datasets",
                "milvus_collections",
                "collection_dataset_links",
            ]

        @staticmethod
        def get_columns(_table):
            return [{"name": "id", "primary_key": True}]

        @staticmethod
        def get_indexes(_table):
            return []

        @staticmethod
        def get_pk_constraint(_table):
            return {"name": "PRIMARY", "constrained_columns": ["id"]}

    class Operations:
        def __init__(self):
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
            self.added_columns = []
            self.created_indexes = []

        def get_bind(self):
            return self.bind

        def add_column(self, table, column):
            self.added_columns.append((table, column))

        def create_index(self, name, table, columns, unique=False):
            self.created_indexes.append(
                (name, table, tuple(columns), unique)
            )

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    migration.upgrade()

    assert [
        (table, column.name, column.type.length, column.nullable)
        for table, column in operations.added_columns
    ] == [
        ("datasets", "generation_run_token", 36, True),
        ("milvus_collections", "generation_task_id", 36, True),
        ("milvus_collections", "generation_run_token", 36, True),
        ("collection_dataset_links", "generation_run_token", 36, True),
    ]
    assert operations.created_indexes == [
        (
            "idx_dataset_generation_run_token",
            "datasets",
            ("generation_run_token",),
            False,
        ),
        (
            "idx_milvus_generation_attempt",
            "milvus_collections",
            ("generation_task_id", "generation_run_token"),
            False,
        ),
        (
            "idx_coll_link_generation_run_token",
            "collection_dataset_links",
            ("generation_run_token",),
            False,
        ),
    ]


def test_generation_publication_upgrade_preflights_every_required_table(
    monkeypatch,
):
    migration = _migration()

    class Inspector:
        @staticmethod
        def get_table_names():
            return ["datasets", "milvus_collections"]

    class Operations:
        def __init__(self):
            self.mutations = []

        @staticmethod
        def get_bind():
            return object()

        def add_column(self, *args):
            self.mutations.append(("add_column", args))

        def create_index(self, *args, **kwargs):
            self.mutations.append(("create_index", args, kwargs))

    operations = Operations()
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: Inspector())

    with pytest.raises(RuntimeError, match="collection_dataset_links"):
        migration.upgrade()

    assert operations.mutations == []


@pytest.mark.parametrize(
    ("kind", "table_name", "object_name"),
    MALFORMED_SCHEMA_CASES,
    ids=lambda value: str(value),
)
def test_generation_publication_upgrade_rejects_malformed_existing_objects_before_mutation(
    monkeypatch,
    kind,
    table_name,
    object_name,
):
    migration = _migration()
    columns, indexes = _malformed_reflected_schema(
        kind,
        table_name,
        object_name,
    )
    _remove_valid_mutation_probe(
        columns,
        indexes,
        table_name,
        object_name,
    )
    operations = _RecordingOperations()
    inspector = _ReflectedInspector(columns, indexes)
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: inspector)

    with pytest.raises(RuntimeError, match=table_name):
        migration.upgrade()

    assert operations.mutations == []


@pytest.mark.parametrize(
    ("kind", "table_name", "object_name"),
    MALFORMED_SCHEMA_CASES,
    ids=lambda value: str(value),
)
def test_generation_publication_downgrade_preserves_malformed_existing_objects(
    monkeypatch,
    kind,
    table_name,
    object_name,
):
    migration = _migration()
    columns, indexes = _malformed_reflected_schema(
        kind,
        table_name,
        object_name,
    )
    operations = _RecordingOperations()
    inspector = _ReflectedInspector(columns, indexes)
    monkeypatch.setattr(migration, "op", operations)
    monkeypatch.setattr(migration, "inspect", lambda _bind: inspector)

    with pytest.raises(RuntimeError, match=table_name):
        migration.downgrade()

    assert operations.mutations == []


def test_lifecycle_public_preflight_is_read_only_and_data_safe(monkeypatch):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)

    plan = migration.build_preflight_plan(bind)

    assert plan["errors"] == ()
    assert plan["repair_count"] == 0
    assert "repair_digest" not in plan
    assert "lineage_plan_count" not in plan
    assert "lineage_plan_digest" not in plan
    assert bind.statements
    encoded = json.dumps(plan, sort_keys=True)
    for forbidden in (
        "row_id",
        "edge_id",
        "from_dataset_id",
        "to_dataset_id",
        "relation_type",
        "provenance",
    ):
        assert forbidden not in encoded


def test_lifecycle_preflight_accumulates_earlier_missing_and_later_malformed(
    monkeypatch,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    columns["training_tasks"] = [columns["training_tasks"][0]]
    dataset_token = next(
        column
        for column in columns["datasets"]
        if column["name"] == "generation_run_token"
    )
    dataset_token["default"] = "private-business-value"
    indexes["milvus_collections"].append(
        {
            "name": "alternate-secret-index",
            "column_names": ["generation_run_token"],
            "unique": True,
        }
    )
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)

    plan = migration.build_preflight_plan(bind)

    assert plan["repair_count"] >= 1
    assert {
        (repair["revision"], repair["object_name"])
        for repair in plan["repairs"]
    } >= {
        ("048_add_training_process_identity", "training_tasks.process_create_time")
    }
    assert {
        (error["revision"], error["category"], error["object_name"])
        for error in plan["errors"]
    } >= {
        (
            "052_generation_publication_staging",
            "column_shape",
            "datasets.generation_run_token",
        ),
        (
            "052_generation_publication_staging",
            "unexpected_unique_index",
            "milvus_collections.alternate-secret-index",
        ),
    }
    encoded = json.dumps(plan, sort_keys=True)
    assert "private-business-value" not in encoded


@pytest.mark.parametrize(
    ("case_name", "expected_category"),
    [
        ("computed-column", "column_shape"),
        ("identity-column", "column_shape"),
        ("primary-key-column", "column_shape"),
        ("unique-constraint", "unexpected_unique_constraint"),
        ("unique-index", "unexpected_unique_index"),
        ("unknown-unique-index", "unknown_index_uniqueness"),
        ("named-nonunique-index", "unexpected_index"),
        ("duplicate-expected-index", "duplicate_index_name"),
    ],
)
def test_lifecycle_preflight_validates_every_raw_publication_object(
    monkeypatch,
    case_name,
    expected_category,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    target = next(
        column
        for column in columns["datasets"]
        if column["name"] == "generation_run_token"
    )
    if case_name == "computed-column":
        target["computed"] = {"sqltext": "private_expression"}
    elif case_name == "identity-column":
        target["identity"] = {"always": False}
    elif case_name == "primary-key-column":
        target["primary_key"] = True
    elif case_name == "unique-constraint":
        constraints["datasets"].append(
            {
                "name": "uq_private_publication",
                "column_names": ["generation_run_token"],
            }
        )
    elif case_name == "duplicate-expected-index":
        indexes["datasets"].append(dict(indexes["datasets"][0]))
    else:
        indexes["datasets"].append(
            {
                "name": "private_publication_index",
                "column_names": ["generation_run_token"],
                "unique": (
                    True
                    if case_name == "unique-index"
                    else False
                    if case_name == "named-nonunique-index"
                    else None
                ),
            }
        )
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)

    plan = migration.build_preflight_plan(bind)

    assert expected_category in {
        error["category"]
        for error in plan["errors"]
        if error["revision"] == "052_generation_publication_staging"
    }
    assert "private_expression" not in json.dumps(plan, sort_keys=True)


@pytest.mark.parametrize(
    ("case_name", "expected_revision", "expected_category"),
    [
        (
            "048-duplicate-columns",
            "048_add_training_process_identity",
            "primary_key_metadata",
        ),
        (
            "050-duplicate-good-indexes",
            "050_add_evaluation_run_token",
            "duplicate_index_name",
        ),
        (
            "050-duplicate-one-bad-index",
            "050_add_evaluation_run_token",
            "duplicate_index_name",
        ),
        (
            "051-overlapping-unique-constraint",
            "051_add_generation_run_token",
            "unexpected_unique_constraint",
        ),
        (
            "050-overlapping-unique-index",
            "050_add_evaluation_run_token",
            "unexpected_unique_index",
        ),
        (
            "051-unknown-index-uniqueness",
            "051_add_generation_run_token",
            "unknown_index_uniqueness",
        ),
        (
            "050-wrong-index-name",
            "050_add_evaluation_run_token",
            "unexpected_index",
        ),
    ],
)
def test_lifecycle_preflight_validates_raw_048_050_051_invariants(
    monkeypatch,
    case_name,
    expected_revision,
    expected_category,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    if case_name == "048-duplicate-columns":
        process_column = next(
            column
            for column in columns["training_tasks"]
            if column["name"] == "process_create_time"
        )
        columns["training_tasks"].append(dict(process_column))
    elif case_name.startswith("050-duplicate"):
        duplicate = dict(indexes["evaluation_tasks"][0])
        duplicate["column_names"] = list(duplicate["column_names"])
        if case_name.endswith("one-bad-index"):
            duplicate["column_names"] = ["id"]
        indexes["evaluation_tasks"].append(duplicate)
    elif case_name == "051-overlapping-unique-constraint":
        constraints["generation_tasks"].append(
            {
                "name": "uq_private_generation_token",
                "column_names": ["run_token"],
            }
        )
    elif case_name == "050-overlapping-unique-index":
        indexes["evaluation_tasks"].append(
            {
                "name": "private_eval_unique",
                "column_names": ["run_token"],
                "unique": True,
            }
        )
    elif case_name == "051-unknown-index-uniqueness":
        indexes["generation_tasks"].append(
            {
                "name": "private_gen_unknown",
                "column_names": ["run_token"],
                "unique": None,
            }
        )
    else:
        indexes["evaluation_tasks"][0]["name"] = "private_wrong_name"

    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)

    plan = migration.build_preflight_plan(bind)

    assert expected_category in {
        error["category"]
        for error in plan["errors"]
        if error["revision"] == expected_revision
    }


@pytest.mark.parametrize(
    ("table_name", "column_name", "second_column_valid", "expected_revision"),
    [
        (
            "dataset_lineage_edges",
            "edge_key",
            True,
            "049_dataset_lineage_edge_key",
        ),
        (
            "dataset_lineage_edges",
            "edge_key",
            False,
            "049_dataset_lineage_edge_key",
        ),
        (
            "datasets",
            "generation_run_token",
            True,
            "052_generation_publication_staging",
        ),
        (
            "datasets",
            "generation_run_token",
            False,
            "052_generation_publication_staging",
        ),
    ],
    ids=["049-two-good", "049-good-bad", "052-two-good", "052-good-bad"],
)
def test_lifecycle_preflight_rejects_raw_managed_column_name_collisions(
    monkeypatch,
    table_name,
    column_name,
    second_column_valid,
    expected_revision,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    original = next(
        column
        for column in columns[table_name]
        if column["name"] == column_name
    )
    duplicate = dict(original)
    if not second_column_valid:
        duplicate["type"] = sa.Integer()
        duplicate["nullable"] = False
    columns[table_name].append(duplicate)
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)

    plan = migration.build_preflight_plan(bind)

    expected_category = (
        "primary_key_metadata"
        if expected_revision == "052_generation_publication_staging"
        else "duplicate_column_name"
    )
    assert expected_category in {
        error["category"]
        for error in plan["errors"]
        if error["revision"] == expected_revision
    }


@pytest.mark.parametrize(
    ("constraints", "lineage_indexes", "expected_category"),
    [
        (
            [
                {
                    "name": "edge_id",
                    "column_names": ["edge_id"],
                    "duplicates_index": "edge_id",
                },
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "uq_lineage_edge_key",
                },
            ],
            [
                {
                    "name": "edge_id",
                    "column_names": ["edge_id"],
                    "unique": True,
                },
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                },
            ],
            None,
        ),
        (
            [
                {
                    "name": "edge_id",
                    "column_names": ["edge_id"],
                    "duplicates_index": "uq_lineage_edge_key",
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            "lineage_uniqueness_shape",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                }
            ],
            [
                {
                    "name": None,
                    "column_names": ["edge_id"],
                    "unique": True,
                }
            ],
            None,
        ),
        (
            [
                {
                    "name": None,
                    "column_names": ["edge_id"],
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            None,
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                }
            ],
            [
                {
                    "name": "edge_id",
                    "column_names": ["edge_id"],
                    "unique": True,
                    "duplicates_constraint": "uq_lineage_edge_key",
                }
            ],
            "lineage_uniqueness_shape",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "uq_lineage_edge_key",
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            None,
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "uq_lineage_edge_key_idx",
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key_idx",
                    "column_names": ["edge_key"],
                    "unique": True,
                    "duplicates_constraint": "uq_lineage_edge_key",
                }
            ],
            None,
        ),
        (
            [],
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            None,
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "uq_lineage_edge_key_idx",
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key_idx",
                    "column_names": ["edge_id"],
                    "unique": True,
                    "duplicates_constraint": "uq_lineage_edge_key",
                }
            ],
            "lineage_uniqueness_shape",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "uq_lineage_edge_key_idx",
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key_idx",
                    "column_names": ["edge_key"],
                    "unique": False,
                    "duplicates_constraint": "uq_lineage_edge_key",
                }
            ],
            "lineage_uniqueness_shape",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                }
            ],
            [
                {
                    "name": "alternate_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            "multiple_lineage_uniqueness",
        ),
        (
            [],
            [
                {
                    "name": None,
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            "unnamed_lineage_uniqueness",
        ),
        (
            [],
            [
                {
                    "name": "unknown_edge_key",
                    "column_names": ["edge_key"],
                    "unique": None,
                }
            ],
            "unknown_lineage_uniqueness",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "uq_lineage_edge_key_idx",
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key_idx",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            None,
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "missing_edge_key_driver",
                }
            ],
            [],
            "lineage_mirror_reference",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "ambiguous_edge_key_driver",
                }
            ],
            [
                {
                    "name": "ambiguous_edge_key_driver",
                    "column_names": ["edge_key"],
                    "unique": True,
                },
                {
                    "name": "ambiguous_edge_key_driver",
                    "column_names": ["edge_key"],
                    "unique": True,
                },
            ],
            "lineage_mirror_reference",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "shared_edge_key_driver",
                },
                {
                    "name": "alternate_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "shared_edge_key_driver",
                },
            ],
            [
                {
                    "name": "shared_edge_key_driver",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            "lineage_mirror_reference",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key_idx",
                    "column_names": ["edge_key"],
                    "unique": True,
                    "duplicates_constraint": "uq_lineage_edge_key",
                }
            ],
            "lineage_mirror_reference",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            "lineage_raw_name_collision",
        ),
        (
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                },
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                },
            ],
            [],
            "lineage_raw_name_collision",
        ),
        (
            [],
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                },
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                },
            ],
            "lineage_raw_name_collision",
        ),
        (
            [
                {
                    "name": "alternate_edge_key",
                    "column_names": ["edge_key"],
                }
            ],
            [],
            "noncanonical_lineage_uniqueness",
        ),
        (
            [],
            [
                {
                    "name": "alternate_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                }
            ],
            "noncanonical_lineage_uniqueness",
        ),
        (
            [
                {
                    "name": "alternate_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "alternate_edge_key_driver",
                }
            ],
            [
                {
                    "name": "alternate_edge_key_driver",
                    "column_names": ["edge_key"],
                    "unique": True,
                    "duplicates_constraint": "alternate_edge_key",
                }
            ],
            "noncanonical_lineage_uniqueness",
        ),
        (
            [
                {
                    "name": "alternate_edge_key",
                    "column_names": ["edge_key"],
                    "duplicates_index": "uq_lineage_edge_key",
                }
            ],
            [
                {
                    "name": "uq_lineage_edge_key",
                    "column_names": ["edge_key"],
                    "unique": True,
                    "duplicates_constraint": "alternate_edge_key",
                }
            ],
            "noncanonical_lineage_uniqueness",
        ),
    ],
    ids=[
        "sqlalchemy-mysql-two-independent-unique-mirrors",
        "incoming-edge-id-constraint-claim",
        "canonical-constraint-does-not-claim-unnamed-index",
        "canonical-index-does-not-claim-unnamed-constraint",
        "incoming-edge-id-index-claim",
        "sqlalchemy-mysql-canonical-reflection",
        "canonical-pair",
        "single-sided",
        "mirror-columns-mismatch",
        "mirror-unique-mismatch",
        "canonical-plus-alternate",
        "unnamed",
        "unknown-unique",
        "constraint-authority-without-index-backlink",
        "constraint-dangling-mirror",
        "constraint-ambiguous-mirror",
        "constraint-shared-index-ambiguity",
        "index-one-way-mirror",
        "raw-name-collision",
        "same-type-constraint-name-collision",
        "same-type-index-name-collision",
        "noncanonical-single-constraint",
        "noncanonical-single-index",
        "noncanonical-reciprocal-pair",
        "canonical-index-with-noncanonical-constraint",
    ],
)
def test_lifecycle_preflight_validates_mysql_lineage_uniqueness_mirrors(
    monkeypatch,
    constraints,
    lineage_indexes,
    expected_category,
):
    migration = _validation_migration()
    columns, indexes, unique_constraints = _valid_lifecycle_reflection()
    indexes["dataset_lineage_edges"] = lineage_indexes
    unique_constraints["dataset_lineage_edges"] = constraints
    inspector = _LifecycleInspector(columns, indexes, unique_constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)

    plan = migration.build_preflight_plan(bind)
    categories = {
        error["category"]
        for error in plan["errors"]
        if error["revision"] == "049_dataset_lineage_edge_key"
    }

    if expected_category is None:
        assert categories == set()
    else:
        assert expected_category in categories


def test_lifecycle_preflight_classifies_stored_lineage_key_mismatch_safely():
    migration = _validation_migration()
    migration_049 = migration._load_lifecycle_migrations()[1]
    expected_key = migration_049._build_edge_key(
        "private-source-path",
        "private-target-path",
        "merged",
    )
    private_digest = migration._digest((("update", 73, expected_key),))
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(connection)
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type, op_task_id) VALUES "
                "(73, 'private-edge-id', 'stored-wrong-secret', "
                "'private-source-path', 'private-target-path', 'merged', "
                "'private-provenance')"
            )
        )

        plan = migration.build_preflight_plan(connection)

    mismatch = [
        repair
        for repair in plan["repairs"]
        if repair["category"] == "stored_lineage_key_mismatch"
    ]
    assert mismatch == [
        {
            "revision": "049_dataset_lineage_edge_key",
            "category": "stored_lineage_key_mismatch",
            "object_name": "dataset_lineage_edges.edge_key",
            "count": 1,
        }
    ]
    encoded = json.dumps(plan, sort_keys=True)
    assert "lineage_plan_count" not in plan
    assert "lineage_plan_digest" not in plan
    assert private_digest not in encoded
    for secret in (
        "private-edge-id",
        "stored-wrong-secret",
        "private-source-path",
        "private-target-path",
        "private-provenance",
    ):
        assert secret not in encoded


def test_lifecycle_public_plan_uses_object_level_counts_without_digests():
    migration = _validation_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(connection)
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'private-edge-a', 'wrong-private-a', 'source-a', "
                "'target-a', 'merged'), "
                "(2, 'private-edge-b', 'wrong-private-b', 'source-b', "
                "'target-b', 'merged')"
            )
        )

        plan = migration.build_preflight_plan(connection)

    mismatch = [
        item
        for item in plan["repairs"]
        if item["category"] == "stored_lineage_key_mismatch"
    ]
    assert mismatch == [
        {
            "revision": "049_dataset_lineage_edge_key",
            "category": "stored_lineage_key_mismatch",
            "object_name": "dataset_lineage_edges.edge_key",
            "count": 1,
        }
    ]
    assert plan["repair_count"] == len(plan["repairs"])
    assert "repair_digest" not in plan
    encoded = json.dumps(plan, sort_keys=True)
    assert not any(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for value in re.findall(r'"([0-9a-f]{64})"', encoded)
    )


@pytest.mark.parametrize("drift_kind", ["added_id", "changed_source"])
def test_lifecycle_private_scan_rejects_drift_from_legacy_planner(
    monkeypatch,
    drift_kind,
):
    migration = _validation_migration()
    migration_049 = migration._load_lifecycle_migrations()[1]
    stale_key = migration_049._build_edge_key(
        "stale-private-source",
        "private-target",
        "merged",
    )
    current_source = (
        "stale-private-source"
        if drift_kind == "added_id"
        else "changed-private-source"
    )
    legacy_plan = (
        []
        if drift_kind == "added_id"
        else [("update", 1, stale_key)]
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(connection)
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'private-edge', :edge_key, :source, "
                "'private-target', 'merged')"
            ),
            {"edge_key": stale_key, "source": current_source},
        )
        monkeypatch.setattr(
            migration_049,
            "_build_backfill_plan",
            lambda _bind: list(legacy_plan),
        )

        plan = migration.build_preflight_plan(connection)

    assert "lineage_data_incompatible" in {
        error["category"] for error in plan["errors"]
    }
    encoded = json.dumps(plan, sort_keys=True)
    for secret in (
        "private-edge",
        "stale-private-source",
        "changed-private-source",
        "private-target",
        stale_key,
    ):
        assert secret not in encoded


def test_lifecycle_private_plan_freezes_old_key_and_source_fields():
    migration = _validation_migration()
    migration_049 = migration._load_lifecycle_migrations()[1]
    expected_key = migration_049._build_edge_key(
        "private-source",
        "private-target",
        "merged",
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(connection)
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'private-edge', 'private-old-key', 'private-source', "
                "'private-target', 'merged')"
            )
        )
        snapshot = migration_049._schema_snapshot(
            connection,
            require_table=True,
        )

        private_plan = migration._build_private_049_plan(
            connection,
            migration_049,
            snapshot,
        )

    assert private_plan == (
        {
            "operation": "update",
            "id": 1,
            "edge_key": expected_key,
            "old_edge_key": "private-old-key",
            "from_dataset_id": "private-source",
            "to_dataset_id": "private-target",
            "relation_type": "merged",
        },
    )


def test_lifecycle_mysql_recheck_locks_the_rich_lineage_scan(monkeypatch):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)
    operations = _BoundRecordingOperations(bind)
    monkeypatch.setattr(migration, "op", operations)

    migration.upgrade()

    assert any("FOR UPDATE" in statement.upper() for statement in bind.statements)
    assert operations.mutations == []


@pytest.mark.parametrize("raced_field", ["edge_key", "from_dataset_id"])
def test_lifecycle_conditional_lineage_update_rejects_raced_rows_and_is_retryable(
    monkeypatch,
    raced_field,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _validation_migration()
    migration_049 = migration._load_lifecycle_migrations()[1]
    expected_after_retry = migration_049._build_edge_key(
        "race-private-source" if raced_field == "from_dataset_id" else "source",
        "target",
        "merged",
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(connection)
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'private-edge', 'private-old-key', "
                "'source', 'target', 'merged')"
            )
        )
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )
        raced = False

        def race_before_update(_conn, cursor, statement, _params, _context, _many):
            nonlocal raced
            if raced or not statement.lstrip().upper().startswith(
                "UPDATE DATASET_LINEAGE_EDGES"
            ):
                return
            raced = True
            if raced_field == "edge_key":
                cursor.execute(
                    "UPDATE dataset_lineage_edges "
                    "SET edge_key = 'race-private-key' WHERE id = 1"
                )
            else:
                cursor.execute(
                    "UPDATE dataset_lineage_edges "
                    "SET from_dataset_id = 'race-private-source' WHERE id = 1"
                )

        sa.event.listen(connection, "before_cursor_execute", race_before_update)
        try:
            with pytest.raises(RuntimeError, match="repair failed.*049") as exc_info:
                migration.upgrade()
        finally:
            sa.event.remove(connection, "before_cursor_execute", race_before_update)

        assert raced is True
        assert exc_info.value.__cause__ is None
        for secret in (
            "private-edge",
            "private-old-key",
            "race-private-key",
            "race-private-source",
        ):
            assert secret not in str(exc_info.value)
        if raced_field == "edge_key":
            assert connection.execute(
                sa.text(
                    "SELECT edge_key FROM dataset_lineage_edges WHERE id = 1"
                )
            ).scalar_one() == "race-private-key"
        else:
            assert connection.execute(
                sa.text(
                    "SELECT edge_key FROM dataset_lineage_edges WHERE id = 1"
                )
            ).scalar_one() == "private-old-key"

        migration.upgrade()

        assert connection.execute(
            sa.text("SELECT edge_key FROM dataset_lineage_edges WHERE id = 1")
        ).scalar_one() == expected_after_retry
        assert migration.build_preflight_plan(connection)["repair_count"] == 0


def test_lifecycle_preflight_repairs_nullable_lineage_key_even_when_data_is_current(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _validation_migration()
    migration_049 = migration._load_lifecycle_migrations()[1]
    expected_key = migration_049._build_edge_key("source", "target", "merged")
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(
            connection,
            edge_key_definition="VARCHAR(64) NULL",
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'edge-one', :edge_key, 'source', 'target', 'merged')"
            ),
            {"edge_key": expected_key},
        )

        plan = migration.build_preflight_plan(connection)

        assert plan["errors"] == ()
        assert {
            (item["category"], item["object_name"])
            for item in plan["repairs"]
        } >= {("nullable_column", "dataset_lineage_edges.edge_key")}
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )

        migration.upgrade()

        edge_key_column = next(
            column
            for column in sa.inspect(connection).get_columns(
                "dataset_lineage_edges"
            )
            if column["name"] == "edge_key"
        )
        assert edge_key_column["nullable"] is False
        assert migration.build_preflight_plan(connection)["repair_count"] == 0


def test_lifecycle_complete_schema_stored_mismatch_uses_data_only_update(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _validation_migration()
    migration_049 = migration._load_lifecycle_migrations()[1]
    expected_key = migration_049._build_edge_key("source", "target", "merged")
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(connection)
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'edge-one', 'wrong-key', 'source', 'target', 'merged')"
            )
        )
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )
        mutations = []

        def record_mutations(_conn, _cursor, statement, _params, _context, _many):
            verb = statement.lstrip().split(None, 1)[0].upper()
            if verb not in {"SELECT", "PRAGMA"}:
                mutations.append(statement)

        sa.event.listen(connection, "before_cursor_execute", record_mutations)
        try:
            migration.upgrade()
        finally:
            sa.event.remove(connection, "before_cursor_execute", record_mutations)

        assert mutations
        assert all(statement.lstrip().upper().startswith("UPDATE") for statement in mutations)
        assert connection.execute(
            sa.text("SELECT edge_key FROM dataset_lineage_edges WHERE id = 1")
        ).scalar_one() == expected_key
        assert migration.build_preflight_plan(connection)["repair_count"] == 0


def test_lifecycle_upgrade_combines_safe_missing_and_lineage_conflict_with_zero_writes(
    monkeypatch,
):
    migration = _validation_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(connection)
        connection.execute(sa.text("ALTER TABLE training_tasks DROP COLUMN process_create_time"))
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type, op_task_id) VALUES "
                "(91, 'edge-secret-a', 'wrong-key-a', 'source-secret', "
                "'target-secret', 'merged', 'provenance-secret-a'), "
                "(92, 'edge-secret-b', 'wrong-key-b', 'source-secret', "
                "'target-secret', 'merged', 'provenance-secret-b')"
            )
        )
        operations = _install_bound_lifecycle_operations(
            monkeypatch,
            migration,
            connection,
        )
        observed = []

        def reject_writes(_conn, _cursor, statement, _params, _context, _many):
            verb = statement.lstrip().split(None, 1)[0].upper()
            observed.append(verb)
            assert verb in {"SELECT", "PRAGMA"}, statement

        sa.event.listen(connection, "before_cursor_execute", reject_writes)
        try:
            with pytest.raises(RuntimeError, match="lineage_data_incompatible") as exc_info:
                migration.upgrade()
        finally:
            sa.event.remove(connection, "before_cursor_execute", reject_writes)

        assert operations.mutations == []
        assert observed
        assert set(observed) <= {"SELECT", "PRAGMA"}
        assert exc_info.value.__cause__ is None
        message = str(exc_info.value)
        for secret in (
            "91",
            "92",
            "edge-secret-a",
            "edge-secret-b",
            "source-secret",
            "target-secret",
            "provenance-secret-a",
            "provenance-secret-b",
        ):
            assert secret not in message


def test_lifecycle_upgrade_rechecks_private_lineage_fingerprint_before_mutation(
    monkeypatch,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)
    operations = _BoundRecordingOperations(bind)
    monkeypatch.setattr(migration, "op", operations)
    calls = 0

    def changing_private_plan(_bind, _module, _snapshot, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ()
        return (("update", "private-row", "private-edge-key"),)

    monkeypatch.setattr(
        migration,
        "_build_private_049_plan",
        changing_private_plan,
    )

    with pytest.raises(RuntimeError, match="plan_changed") as exc_info:
        migration.upgrade()

    assert calls == 2
    assert operations.mutations == []
    assert exc_info.value.__cause__ is None
    assert "private-row" not in str(exc_info.value)
    assert "private-edge-key" not in str(exc_info.value)


@pytest.mark.parametrize("module_index", range(5))
def test_lifecycle_preflight_sanitizes_every_historical_snapshot_exception(
    monkeypatch,
    module_index,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)
    historical = migration._load_lifecycle_migrations()[module_index]

    def fail_snapshot(*_args, **_kwargs):
        raise ValueError("snapshot-private-canary")

    monkeypatch.setattr(historical, "_schema_snapshot", fail_snapshot)

    plan = migration.build_preflight_plan(bind)

    assert {
        (error["revision"], error["category"])
        for error in plan["errors"]
    } >= {(historical.revision, "historical_snapshot_failed")}
    assert "snapshot-private-canary" not in json.dumps(plan, sort_keys=True)


@pytest.mark.parametrize("exception_type", [ValueError, _PrivateOperationalError])
@pytest.mark.parametrize(
    ("module_index", "table_name"),
    [
        (0, "training_tasks"),
        (1, "dataset_lineage_edges"),
        (2, "evaluation_tasks"),
        (3, "generation_tasks"),
        (4, "datasets"),
    ],
)
def test_lifecycle_preflight_sanitizes_every_raw_reflection_exception(
    monkeypatch,
    exception_type,
    module_index,
    table_name,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    if module_index == 2:
        indexes["generation_tasks"][0]["unique"] = True
        continued_module_index = 3
    else:
        indexes["evaluation_tasks"][0]["unique"] = True
        continued_module_index = 2
    historical_inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(
        monkeypatch,
        migration,
        historical_inspector,
    )
    raw_inspector = _RawReflectionFailureInspector(
        historical_inspector,
        table_name,
        exception_type,
    )
    monkeypatch.setattr(migration, "inspect", lambda _bind: raw_inspector)
    historical = migration._load_lifecycle_migrations()

    plan = migration.build_preflight_plan(bind)

    error_keys = {
        (error["revision"], error["category"], error["object_name"])
        for error in plan["errors"]
    }
    assert (
        historical[module_index].revision,
        "raw_snapshot_failed",
        table_name,
    ) in error_keys
    continued_revision = historical[continued_module_index].revision
    assert any(
        error["revision"] == continued_revision
        and error["category"] == "index_shape"
        for error in plan["errors"]
    )
    assert not any(
        repair["revision"] == historical[module_index].revision
        for repair in plan["repairs"]
    )
    assert "raw-reflection-private-canary" not in json.dumps(plan, sort_keys=True)

    operations = _BoundRecordingOperations(bind)
    monkeypatch.setattr(migration, "op", operations)
    with pytest.raises(RuntimeError, match="raw_snapshot_failed") as exc_info:
        migration.upgrade()
    assert exc_info.value.__cause__ is None
    assert "raw-reflection-private-canary" not in str(exc_info.value)
    assert operations.mutations == []


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_lifecycle_raw_reflection_does_not_swallow_control_flow_exceptions(
    monkeypatch,
    exception_type,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    historical_inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(
        monkeypatch,
        migration,
        historical_inspector,
    )
    raw_inspector = _RawReflectionFailureInspector(
        historical_inspector,
        "dataset_lineage_edges",
        exception_type,
    )
    monkeypatch.setattr(migration, "inspect", lambda _bind: raw_inspector)

    with pytest.raises(exception_type):
        migration.build_preflight_plan(bind)


@pytest.mark.parametrize("exception_type", [ValueError, _PrivateOperationalError])
def test_lifecycle_preflight_sanitizes_private_planner_exceptions(
    monkeypatch,
    exception_type,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)
    migration_049 = migration._load_lifecycle_migrations()[1]

    def fail_planner(_bind):
        raise exception_type("planner-private-canary")

    monkeypatch.setattr(migration_049, "_build_backfill_plan", fail_planner)

    plan = migration.build_preflight_plan(bind)

    assert "lineage_data_incompatible" in {
        error["category"] for error in plan["errors"]
    }
    assert "planner-private-canary" not in json.dumps(plan, sort_keys=True)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_lifecycle_preflight_does_not_swallow_control_flow_exceptions(
    monkeypatch,
    exception_type,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)
    migration_049 = migration._load_lifecycle_migrations()[1]

    def interrupt_planner(_bind):
        raise exception_type()

    monkeypatch.setattr(migration_049, "_build_backfill_plan", interrupt_planner)

    with pytest.raises(exception_type):
        migration.build_preflight_plan(bind)


@pytest.mark.parametrize("exception_type", [ValueError, _PrivateOperationalError])
def test_lifecycle_upgrade_sanitizes_private_recheck_exceptions(
    monkeypatch,
    exception_type,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)
    monkeypatch.setattr(migration, "op", _BoundRecordingOperations(bind))
    calls = 0

    def fail_second_call(_bind, _module, _snapshot, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ()
        raise exception_type("recheck-private-canary")

    monkeypatch.setattr(
        migration,
        "_build_private_049_plan",
        fail_second_call,
    )

    with pytest.raises(RuntimeError, match="private lineage recheck failed") as exc_info:
        migration.upgrade()

    assert calls == 2
    assert exc_info.value.__cause__ is None
    assert "recheck-private-canary" not in str(exc_info.value)


@pytest.mark.parametrize("exception_type", [ValueError, _PrivateOperationalError])
def test_lifecycle_upgrade_sanitizes_historical_repair_errors(
    monkeypatch,
    exception_type,
):
    migration = _validation_migration()
    columns, indexes, constraints = _valid_lifecycle_reflection()
    columns["training_tasks"] = [columns["training_tasks"][0]]
    inspector = _LifecycleInspector(columns, indexes, constraints)
    bind = _ReadOnlyLifecycleBind()
    _install_lifecycle_inspector(monkeypatch, migration, inspector)
    operations = _BoundRecordingOperations(bind)
    monkeypatch.setattr(migration, "op", operations)
    migration_048 = migration._load_lifecycle_migrations()[0]

    def fail_repair():
        raise exception_type("private-row private-path private-business-value")

    monkeypatch.setattr(migration_048, "upgrade", fail_repair)

    with pytest.raises(RuntimeError, match="repair failed.*048") as exc_info:
        migration.upgrade()

    assert exc_info.value.__cause__ is None
    message = str(exc_info.value)
    assert "private-row" not in message
    assert "private-path" not in message
    assert "private-business-value" not in message


def _create_partial_lifecycle_sqlite_schema(connection):
    statements = (
        "CREATE TABLE training_tasks (id INTEGER PRIMARY KEY)",
        "CREATE TABLE dataset_lineage_edges ("
        "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
        "edge_key VARCHAR(64) NULL, from_dataset_id VARCHAR(36), "
        "to_dataset_id VARCHAR(36) NOT NULL, relation_type VARCHAR(32) NOT NULL, "
        "op_task_type VARCHAR(32), op_task_id VARCHAR(36), op_params JSON)",
        "CREATE TABLE evaluation_tasks (id INTEGER PRIMARY KEY)",
        "CREATE TABLE generation_tasks ("
        "id INTEGER PRIMARY KEY, run_token VARCHAR(36) NULL)",
        "CREATE TABLE datasets ("
        "id INTEGER PRIMARY KEY, generation_run_token VARCHAR(36) NULL)",
        "CREATE INDEX idx_dataset_generation_run_token "
        "ON datasets (generation_run_token)",
        "CREATE TABLE milvus_collections ("
        "id INTEGER PRIMARY KEY, generation_task_id VARCHAR(36) NULL)",
        "CREATE TABLE collection_dataset_links (id INTEGER PRIMARY KEY)",
    )
    for statement in statements:
        connection.execute(sa.text(statement))
    connection.execute(
        sa.text(
            "INSERT INTO dataset_lineage_edges "
            "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
            "relation_type) VALUES "
            "(1, 'edge-one', 'wrong-key', 'source-one', 'target-one', 'merged')"
        )
    )


def _all_lifecycle_sqlite_state(connection):
    schema = connection.execute(
        sa.text(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'index') ORDER BY type, name"
        )
    ).all()
    lineage = connection.execute(
        sa.text("SELECT * FROM dataset_lineage_edges ORDER BY id")
    ).all()
    return tuple(schema), tuple(lineage)


def test_lifecycle_upgrade_repairs_049_first_then_is_fully_idempotent(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _validation_migration()
    migration_049 = migration._load_lifecycle_migrations()[1]
    expected_key = migration_049._build_edge_key(
        "source-one",
        "target-one",
        "merged",
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_partial_lifecycle_sqlite_schema(connection)
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )
        mutation_sql = []

        def record_mutations(_conn, _cursor, statement, _params, _context, _many):
            verb = statement.lstrip().split(None, 1)[0].upper()
            if verb not in {"SELECT", "PRAGMA"}:
                mutation_sql.append(statement)

        sa.event.listen(connection, "before_cursor_execute", record_mutations)
        try:
            migration.upgrade()
        finally:
            sa.event.remove(connection, "before_cursor_execute", record_mutations)

        assert mutation_sql
        assert mutation_sql[0].lstrip().upper().startswith(
            "UPDATE DATASET_LINEAGE_EDGES"
        )
        assert connection.execute(
            sa.text("SELECT edge_key FROM dataset_lineage_edges WHERE id = 1")
        ).scalar_one() == expected_key
        final_plan = migration.build_preflight_plan(connection)
        assert final_plan["errors"] == ()
        assert final_plan["repair_count"] == 0

        second_mutations = []

        def record_second(_conn, _cursor, statement, _params, _context, _many):
            verb = statement.lstrip().split(None, 1)[0].upper()
            if verb not in {"SELECT", "PRAGMA"}:
                second_mutations.append(statement)

        sa.event.listen(connection, "before_cursor_execute", record_second)
        try:
            migration.upgrade()
        finally:
            sa.event.remove(connection, "before_cursor_execute", record_second)

        assert second_mutations == []


def test_lifecycle_downgrade_preserves_all_managed_objects_and_data(monkeypatch):
    migration = _validation_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_complete_lifecycle_sqlite_schema(connection)
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'edge-one', 'key-one', 'source-one', 'target-one', 'merged')"
            )
        )
        operations = _install_bound_lifecycle_operations(
            monkeypatch,
            migration,
            connection,
        )
        before = _all_lifecycle_sqlite_state(connection)

        migration.downgrade()

        assert operations.mutations == []
        assert _all_lifecycle_sqlite_state(connection) == before
