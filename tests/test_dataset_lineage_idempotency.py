from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import importlib
import sys
import threading
from types import ModuleType

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, create_engine, select

from train_factory.storage.entities.dataset_lineage_entity import (
    DatasetLineageEdgeDB,
    build_dataset_lineage_edge_key,
)
from train_factory.storage.services.dataset_lineage_service import (
    DatasetLineageService,
    _is_edge_key_unique_conflict,
)


@pytest.fixture
def lineage_service(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'lineage.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    DatasetLineageEdgeDB.__table__.create(engine)
    service = DatasetLineageService()
    service.engine = engine
    return service


def _all_edges(service):
    with Session(service.engine) as session:
        return session.exec(select(DatasetLineageEdgeDB)).all()


def _import_lineage_migration():
    module_name = (
        "train_factory.storage.migrations.versions." "049_add_dataset_lineage_edge_key"
    )
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


def _create_legacy_lineage_table(connection, *, named_unique=False):
    unique_clause = (
        ", CONSTRAINT legacy_edge_identity UNIQUE (edge_key)" if named_unique else ""
    )
    connection.execute(
        sa.text(
            "CREATE TABLE dataset_lineage_edges ("
            "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
            "edge_key VARCHAR(64), from_dataset_id VARCHAR(36), "
            "to_dataset_id VARCHAR(36) NOT NULL, "
            "relation_type VARCHAR(32) NOT NULL, op_task_type VARCHAR(32), "
            "op_task_id VARCHAR(36), op_params JSON, created_at DATETIME"
            f"{unique_clause})"
        )
    )


def _insert_duplicate_legacy_edges(
    connection,
    *,
    first_task_type="generation",
    second_task_type="generation",
    first_task_id="task-1",
    second_task_id="task-1",
    first_params='{"round":1}',
    second_params='{"round":1}',
    first_edge_key=None,
    second_edge_key=None,
):
    connection.execute(
        sa.text(
            "INSERT INTO dataset_lineage_edges "
            "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
            "relation_type, op_task_type, op_task_id, op_params) VALUES "
            "(1, 'edge-1', :first_edge_key, 'source', 'target', 'merged', "
            ":first_task_type, :first_task_id, :first_params), "
            "(2, 'edge-2', :second_edge_key, 'source', 'target', 'merged', "
            ":second_task_type, :second_task_id, :second_params)"
        ),
        {
            "first_edge_key": first_edge_key,
            "second_edge_key": second_edge_key,
            "first_task_type": first_task_type,
            "second_task_type": second_task_type,
            "first_task_id": first_task_id,
            "second_task_id": second_task_id,
            "first_params": first_params,
            "second_params": second_params,
        },
    )


def _legacy_edge_rows(connection):
    return connection.execute(
        sa.text(
            "SELECT id, edge_id, edge_key, op_task_type, op_task_id, op_params "
            "FROM dataset_lineage_edges ORDER BY id"
        )
    ).all()


def test_edge_key_is_deterministic_and_preserves_nullable_source_identity():
    first = build_dataset_lineage_edge_key(None, "target", "sync_fetched")

    assert first == build_dataset_lineage_edge_key(
        None,
        "target",
        "sync_fetched",
    )
    assert first != build_dataset_lineage_edge_key(
        "root",
        "target",
        "sync_fetched",
    )
    assert first != build_dataset_lineage_edge_key(
        None,
        "target",
        "merged",
    )


def test_migration_collapses_mysql_duplicate_unique_reflection(monkeypatch):
    migration = _import_lineage_migration()

    class MySQLShapedInspector:
        def get_unique_constraints(self, _table_name):
            return [
                {
                    "name": "mysql_edge_identity",
                    "column_names": ["edge_key"],
                    "duplicates_index": "mysql_edge_identity",
                }
            ]

        def get_indexes(self, _table_name):
            return [
                {
                    "name": "mysql_edge_identity",
                    "column_names": ["edge_key"],
                    "unique": True,
                    "duplicates_constraint": "mysql_edge_identity",
                }
            ]

    monkeypatch.setattr(
        migration,
        "inspect",
        lambda _bind: MySQLShapedInspector(),
    )

    assert migration._unique_edge_key_objects(object()) == [
        ("constraint", "mysql_edge_identity")
    ]


def test_lineage_upgrade_fails_when_dependency_table_is_missing(monkeypatch):
    migration = _import_lineage_migration()

    class EmptyInspector:
        def get_table_names(self):
            return []

    monkeypatch.setattr(
        migration,
        "op",
        type("FakeOperations", (), {"get_bind": staticmethod(object)})(),
    )
    monkeypatch.setattr(
        migration,
        "inspect",
        lambda _bind: EmptyInspector(),
    )

    with pytest.raises(RuntimeError, match="dataset_lineage_edges"):
        migration.upgrade()


def test_create_edge_is_idempotent(lineage_service):
    first = lineage_service.create_edge("source", "target", "merged")
    second = lineage_service.create_edge("source", "target", "merged")

    assert first["edge_id"] == second["edge_id"]
    assert len(_all_edges(lineage_service)) == 1


def test_bulk_deduplicates_repeated_sources_and_existing_edges(lineage_service):
    created = lineage_service.create_edges_bulk(
        ["source-a", "source-a", "source-b", "source-a"],
        "target",
        "merged",
    )
    repeated = lineage_service.create_edges_bulk(
        ["source-b", "source-a", "source-b"],
        "target",
        "merged",
    )

    assert {edge["from_dataset_id"] for edge in created} == {
        "source-a",
        "source-b",
    }
    assert repeated == []
    assert len(_all_edges(lineage_service)) == 2


def test_single_and_bulk_concurrent_insert_share_one_database_edge(
    lineage_service,
):
    barrier = threading.Barrier(2)

    def create_single():
        barrier.wait()
        return lineage_service.create_edge("source", "target", "merged")

    def create_bulk():
        barrier.wait()
        return lineage_service.create_edges_bulk(
            ["source", "source"],
            "target",
            "merged",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        single_future = executor.submit(create_single)
        bulk_future = executor.submit(create_bulk)
        single = single_future.result(timeout=30)
        bulk = bulk_future.result(timeout=30)

    rows = _all_edges(lineage_service)
    assert len(rows) == 1
    assert single["edge_id"] == rows[0].edge_id
    assert bulk == [] or bulk[0]["edge_id"] == rows[0].edge_id


def test_database_unique_key_rejects_duplicate_triples(lineage_service):
    key = build_dataset_lineage_edge_key("source", "target", "merged")
    with Session(lineage_service.engine) as session:
        session.add(
            DatasetLineageEdgeDB(
                edge_key=key,
                from_dataset_id="source",
                to_dataset_id="target",
                relation_type="merged",
            )
        )
        session.commit()

    with Session(lineage_service.engine) as session:
        session.add(
            DatasetLineageEdgeDB(
                edge_key=key,
                from_dataset_id="source",
                to_dataset_id="target",
                relation_type="merged",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_unrelated_integrity_error_is_not_misreported_as_existing_edge():
    existing = DatasetLineageEdgeDB(
        from_dataset_id="source",
        to_dataset_id="target",
        relation_type="merged",
    )

    class FakeResult:
        def one_or_none(self):
            return existing

    class FakeSession:
        def begin_nested(self):
            return nullcontext()

        def add(self, _edge):
            return None

        def flush(self):
            raise IntegrityError(
                "INSERT",
                {},
                RuntimeError("CHECK constraint failed: ck_unrelated"),
            )

        def exec(self, _statement):
            return FakeResult()

    with pytest.raises(IntegrityError, match="ck_unrelated"):
        DatasetLineageService._create_edge_in_session(
            FakeSession(),
            "source",
            "target",
            "merged",
            None,
            None,
            None,
        )


def test_edge_key_conflict_classifier_is_constraint_specific():
    duplicate = IntegrityError(
        "INSERT",
        {},
        RuntimeError(
            1062,
            "Duplicate entry for key 'uq_lineage_edge_key'",
        ),
    )
    unrelated = IntegrityError(
        "INSERT",
        {},
        RuntimeError(
            1062,
            "Duplicate entry for key 'uq_unrelated_constraint'",
        ),
    )

    assert _is_edge_key_unique_conflict(duplicate) is True
    assert _is_edge_key_unique_conflict(unrelated) is False


def test_unique_conflict_uses_current_read_to_find_concurrent_winner():
    existing = DatasetLineageEdgeDB(
        from_dataset_id="source",
        to_dataset_id="target",
        relation_type="merged",
    )
    statements = []

    class FakeResult:
        def one_or_none(self):
            return existing

    class FakeSession:
        def begin_nested(self):
            return nullcontext()

        def add(self, _edge):
            return None

        def flush(self):
            raise IntegrityError(
                "INSERT",
                {},
                RuntimeError(
                    "UNIQUE constraint failed: " "dataset_lineage_edges.edge_key"
                ),
            )

        def exec(self, statement):
            statements.append(statement)
            return FakeResult()

    edge, created = DatasetLineageService._create_edge_in_session(
        FakeSession(),
        "source",
        "target",
        "merged",
        None,
        None,
        None,
    )

    assert edge is existing
    assert created is False
    assert statements[0]._for_update_arg is not None
    mysql_statement = str(statements[0].compile(dialect=mysql.dialect()))
    assert mysql_statement.endswith(" FOR UPDATE")


def test_bulk_inserts_edges_in_deterministic_key_order(
    lineage_service,
    monkeypatch,
):
    observed_sources = []

    def capture_source(
        _session,
        from_dataset_id,
        _to_dataset_id,
        _relation_type,
        _op_task_type,
        _op_task_id,
        _op_params,
    ):
        observed_sources.append(from_dataset_id)
        return DatasetLineageEdgeDB(
            from_dataset_id=from_dataset_id,
            to_dataset_id="target",
            relation_type="merged",
        ), False

    monkeypatch.setattr(
        lineage_service,
        "_create_edge_in_session",
        capture_source,
    )
    sources = ["source-b", "source-a", "source-c", "source-a"]

    assert (
        lineage_service.create_edges_bulk(
            sources,
            "target",
            "merged",
        )
        == []
    )
    assert observed_sources == sorted(
        set(sources),
        key=lambda source: build_dataset_lineage_edge_key(
            source,
            "target",
            "merged",
        ),
    )


def test_lineage_edge_key_migration_rejects_duplicates_before_mutation(monkeypatch):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                "from_dataset_id VARCHAR(36), to_dataset_id VARCHAR(36) NOT NULL, "
                "relation_type VARCHAR(32) NOT NULL, op_task_type VARCHAR(32), "
                "op_task_id VARCHAR(36), op_params JSON, created_at DATETIME)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, from_dataset_id, to_dataset_id, relation_type) "
                "VALUES "
                "(1, 'edge-1', NULL, 'target', 'sync_fetched'), "
                "(2, 'edge-2', NULL, 'target', 'sync_fetched'), "
                "(3, 'edge-3', 'root', 'target', 'sync_fetched'), "
                "(4, 'edge-4', NULL, 'target', 'merged')"
            )
        )
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )
        before_rows = connection.execute(
            sa.text(
                "SELECT id, edge_id FROM dataset_lineage_edges ORDER BY id"
            )
        ).all()
        before_columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns(
                "dataset_lineage_edges"
            )
        }

        with pytest.raises(RuntimeError, match="duplicate lineage triple") as exc_info:
            migration.upgrade()

        rows = connection.execute(
            sa.text(
                "SELECT id, edge_id FROM dataset_lineage_edges ORDER BY id"
            )
        ).all()
        columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns("dataset_lineage_edges")
        }

    assert migration.revision == "049_dataset_lineage_edge_key"
    assert migration.down_revision == "048_add_training_process_identity"
    assert rows == before_rows
    assert columns == before_columns
    assert "edge_key" not in columns
    message = str(exc_info.value)
    assert "edge-1" in message
    assert "edge-2" in message
    assert "from_dataset_id=None" in message
    assert "to_dataset_id='target'" in message
    assert "relation_type='sync_fetched'" in message


def test_lineage_migration_refuses_to_discard_conflicting_provenance():
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                "edge_key VARCHAR(64), from_dataset_id VARCHAR(36), "
                "to_dataset_id VARCHAR(36) NOT NULL, "
                "relation_type VARCHAR(32) NOT NULL, op_task_type VARCHAR(32), "
                "op_task_id VARCHAR(36), op_params JSON, created_at DATETIME)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, from_dataset_id, to_dataset_id, relation_type, "
                "op_task_type, op_task_id, op_params) VALUES "
                "(1, 'edge-1', 'source', 'target', 'merged', "
                "'generation', 'task-1', :first_params), "
                "(2, 'edge-2', 'source', 'target', 'merged', "
                "'generation', 'task-2', :second_params)"
            ),
            {
                "first_params": '{"round": 1}',
                "second_params": '{"round": 2}',
            },
        )

        with pytest.raises(RuntimeError, match="conflicting provenance"):
            migration._backfill_and_deduplicate(connection)

        rows = connection.execute(
            sa.text(
                "SELECT edge_id, edge_key, op_task_id "
                "FROM dataset_lineage_edges ORDER BY id"
            )
        ).all()

    assert [(row.edge_id, row.edge_key, row.op_task_id) for row in rows] == [
        ("edge-1", None, "task-1"),
        ("edge-2", None, "task-2"),
    ]


@pytest.mark.parametrize(
    ("field_name", "overrides", "secret_values"),
    [
        (
            "op_task_type",
            {
                "first_task_type": "private-type-alpha",
                "second_task_type": "private-type-beta",
            },
            ("private-type-alpha", "private-type-beta"),
        ),
        (
            "op_task_id",
            {
                "first_task_id": "private-id-alpha",
                "second_task_id": "private-id-beta",
            },
            ("private-id-alpha", "private-id-beta"),
        ),
        (
            "op_params",
            {
                "first_params": '{"private":"value-alpha"}',
                "second_params": '{"private":"value-beta"}',
            },
            ("value-alpha", "value-beta"),
        ),
    ],
)
def test_lineage_migration_reports_each_conflicting_provenance_field(
    field_name,
    overrides,
    secret_values,
):
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_lineage_table(connection)
        _insert_duplicate_legacy_edges(connection, **overrides)

        with pytest.raises(RuntimeError) as exc_info:
            migration._backfill_and_deduplicate(connection)

        rows = _legacy_edge_rows(connection)

    message = str(exc_info.value)
    assert "source" in message
    assert "target" in message
    assert "merged" in message
    assert field_name in message
    assert all(secret not in message for secret in secret_values)
    assert [(row.edge_id, row.edge_key) for row in rows] == [
        ("edge-1", None),
        ("edge-2", None),
    ]


def test_lineage_migration_rejects_equivalent_reordered_json_duplicates():
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_lineage_table(connection)
        _insert_duplicate_legacy_edges(
            connection,
            first_params='{"b":[2,1],"a":{"y":true,"x":null}}',
            second_params='{"a":{"x":null,"y":true},"b":[2,1]}',
        )

        with pytest.raises(RuntimeError, match="duplicate lineage triple") as exc_info:
            migration._backfill_and_deduplicate(connection)
        rows = _legacy_edge_rows(connection)

    assert [(row.id, row.edge_id, row.edge_key) for row in rows] == [
        (1, "edge-1", None),
        (2, "edge-2", None),
    ]
    assert "edge-1" in str(exc_info.value)
    assert "edge-2" in str(exc_info.value)


def test_lineage_migration_rejects_malformed_json_instead_of_matching_string():
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_lineage_table(connection)
        _insert_duplicate_legacy_edges(
            connection,
            first_params="private-bare-text",
            second_params='"private-bare-text"',
        )

        with pytest.raises(RuntimeError, match="op_params") as exc_info:
            migration._backfill_and_deduplicate(connection)

        rows = _legacy_edge_rows(connection)

    assert "private-bare-text" not in str(exc_info.value)
    assert len(rows) == 2
    assert all(row.edge_key is None for row in rows)


def test_lineage_migration_preserves_exact_high_precision_json_numbers():
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_lineage_table(connection)
        _insert_duplicate_legacy_edges(
            connection,
            first_params='{"score":9007199254740992.0}',
            second_params='{"score":9007199254740993.0}',
        )

        with pytest.raises(RuntimeError, match="op_params"):
            migration._backfill_and_deduplicate(connection)

        rows = _legacy_edge_rows(connection)

    assert len(rows) == 2
    assert all(row.edge_key is None for row in rows)


def test_lineage_migration_rejects_duplicate_json_object_keys():
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_lineage_table(connection)
        _insert_duplicate_legacy_edges(
            connection,
            first_params='{"mode":"first","mode":"second"}',
            second_params='{"mode":"first","mode":"second"}',
        )

        with pytest.raises(RuntimeError, match="op_params") as exc_info:
            migration._backfill_and_deduplicate(connection)

        rows = _legacy_edge_rows(connection)

    assert "first" not in str(exc_info.value)
    assert "second" not in str(exc_info.value)
    assert len(rows) == 2
    assert all(row.edge_key is None for row in rows)


def test_lineage_migration_rejects_unsupported_decoded_json_value():
    migration = _import_lineage_migration()

    with pytest.raises(RuntimeError, match="op_params"):
        migration._canonical_op_params({"unsupported": object()})


@pytest.mark.parametrize("raw_value", ["NaN", "Infinity", "-Infinity"])
def test_lineage_migration_rejects_raw_nonfinite_json_numbers(raw_value):
    migration = _import_lineage_migration()

    with pytest.raises(RuntimeError, match="op_params"):
        migration._canonical_op_params(raw_value)


@pytest.mark.parametrize(
    "decoded_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_lineage_migration_rejects_decoded_nonfinite_json_numbers(decoded_value):
    migration = _import_lineage_migration()

    with pytest.raises(RuntimeError, match="op_params"):
        migration._canonical_op_params({"value": decoded_value})


@pytest.mark.parametrize(
    ("first_params", "second_params"),
    [
        (None, None),
        ("null", "null"),
        ("{}", "{ }"),
        ("[]", "[ ]"),
        ('""', '""'),
    ],
)
def test_lineage_migration_rejects_equivalent_null_and_empty_json_duplicates(
    first_params,
    second_params,
):
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_lineage_table(connection)
        _insert_duplicate_legacy_edges(
            connection,
            first_params=first_params,
            second_params=second_params,
        )

        with pytest.raises(RuntimeError, match="duplicate lineage triple"):
            migration._backfill_and_deduplicate(connection)
        rows = _legacy_edge_rows(connection)

    assert [(row.id, row.edge_id, row.edge_key) for row in rows] == [
        (1, "edge-1", None),
        (2, "edge-2", None),
    ]


@pytest.mark.parametrize(
    ("first_params", "second_params"),
    [
        (None, "null"),
        ("null", "{}"),
        ("{}", "[]"),
        ("[]", '""'),
    ],
)
def test_lineage_migration_keeps_sql_null_distinct_from_json_null_and_empties(
    first_params,
    second_params,
):
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_lineage_table(connection)
        _insert_duplicate_legacy_edges(
            connection,
            first_params=first_params,
            second_params=second_params,
        )

        with pytest.raises(RuntimeError, match="op_params"):
            migration._backfill_and_deduplicate(connection)

        rows = _legacy_edge_rows(connection)

    assert len(rows) == 2
    assert all(row.edge_key is None for row in rows)


def test_lineage_upgrade_preserves_rows_and_named_unique_on_equivalent_duplicate(
    monkeypatch,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_legacy_lineage_table(connection, named_unique=True)
        _insert_duplicate_legacy_edges(
            connection,
            first_edge_key="legacy-key-alpha",
            second_edge_key="legacy-key-beta",
        )
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )

        with pytest.raises(RuntimeError, match="duplicate lineage triple") as exc_info:
            migration.upgrade()

        rows = _legacy_edge_rows(connection)
        unique_names = {
            constraint["name"]
            for constraint in sa.inspect(connection).get_unique_constraints(
                "dataset_lineage_edges"
            )
            if tuple(constraint["column_names"]) == ("edge_key",)
        }

    assert "edge-1" in str(exc_info.value)
    assert "edge-2" in str(exc_info.value)
    assert [(row.edge_id, row.edge_key) for row in rows] == [
        ("edge-1", "legacy-key-alpha"),
        ("edge-2", "legacy-key-beta"),
    ]
    assert unique_names == {"legacy_edge_identity"}


@pytest.mark.parametrize(
    "unique_kind",
    ["constraint", "index", "unnamed_constraint"],
)
def test_lineage_migration_preserves_partial_unique_backfill_on_duplicate(
    monkeypatch,
    unique_kind,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _import_lineage_migration()
    canonical_key = build_dataset_lineage_edge_key(
        "source",
        "target",
        "merged",
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        unique_clause = (
            ", CONSTRAINT partial_edge_identity UNIQUE (edge_key)"
            if unique_kind == "constraint"
            else ", UNIQUE (edge_key)"
            if unique_kind == "unnamed_constraint"
            else ""
        )
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                "edge_key VARCHAR(64), from_dataset_id VARCHAR(36), "
                "to_dataset_id VARCHAR(36) NOT NULL, "
                "relation_type VARCHAR(32) NOT NULL"
                f"{unique_clause})"
            )
        )
        if unique_kind == "index":
            connection.execute(
                sa.text(
                    "CREATE UNIQUE INDEX partial_edge_identity "
                    "ON dataset_lineage_edges (edge_key)"
                )
            )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'edge-1', NULL, 'source', 'target', 'merged'), "
                "(2, 'edge-2', :edge_key, 'source', 'target', 'merged')"
            ),
            {"edge_key": canonical_key},
        )
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )

        with pytest.raises(RuntimeError, match="duplicate lineage triple") as exc_info:
            migration.upgrade()

        rows = connection.execute(
            sa.text(
                "SELECT id, edge_id, edge_key FROM dataset_lineage_edges " "ORDER BY id"
            )
        ).all()
        inspector = sa.inspect(connection)
        unique_names = {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("dataset_lineage_edges")
            if tuple(constraint["column_names"]) == ("edge_key",)
        }
        unique_index_names = {
            index["name"]
            for index in inspector.get_indexes("dataset_lineage_edges")
            if index.get("unique")
            and tuple(index.get("column_names") or ()) == ("edge_key",)
        }
        edge_key_column = next(
            column
            for column in inspector.get_columns("dataset_lineage_edges")
            if column["name"] == "edge_key"
        )

    assert [(row.id, row.edge_id, row.edge_key) for row in rows] == [
        (1, "edge-1", None),
        (2, "edge-2", canonical_key),
    ]
    assert edge_key_column["nullable"] is True
    if unique_kind == "constraint":
        assert unique_names == {"partial_edge_identity"}
        assert unique_index_names == set()
    elif unique_kind == "unnamed_constraint":
        assert unique_names == {None}
        assert unique_index_names == set()
    else:
        assert unique_names == set()
        assert unique_index_names == {"partial_edge_identity"}
    assert "edge-1" in str(exc_info.value)
    assert "edge-2" in str(exc_info.value)


@pytest.mark.parametrize(
    "unique_kind",
    ["constraint", "index", "unnamed_constraint"],
)
def test_lineage_migration_downgrade_drops_actual_unique_name(
    monkeypatch,
    unique_kind,
):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        unique_clause = (
            ", CONSTRAINT legacy_edge_identity UNIQUE (edge_key)"
            if unique_kind == "constraint"
            else ", UNIQUE (edge_key)"
            if unique_kind == "unnamed_constraint"
            else ""
        )
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                "edge_key VARCHAR(64) NOT NULL"
                f"{unique_clause})"
            )
        )
        if unique_kind == "index":
            connection.execute(
                sa.text(
                    "CREATE UNIQUE INDEX legacy_edge_identity "
                    "ON dataset_lineage_edges (edge_key)"
                )
            )
        monkeypatch.setattr(
            migration,
            "op",
            Operations(MigrationContext.configure(connection)),
        )

        migration.downgrade()

        columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns("dataset_lineage_edges")
        }

    assert "edge_key" not in columns


_MALFORMED_LINEAGE_UNIQUENESS_CASES = (
    (
        "same-name-non-unique-index",
        "",
        "CREATE INDEX uq_lineage_edge_key "
        "ON dataset_lineage_edges (edge_key)",
        "uq_lineage_edge_key",
    ),
    (
        "same-name-wrong-column-index",
        "",
        "CREATE UNIQUE INDEX uq_lineage_edge_key "
        "ON dataset_lineage_edges (edge_id)",
        "uq_lineage_edge_key",
    ),
    (
        "composite-edge-key-index",
        "",
        "CREATE UNIQUE INDEX legacy_edge_identity "
        "ON dataset_lineage_edges (edge_key, edge_id)",
        "legacy_edge_identity",
    ),
    (
        "same-name-wrong-column-constraint",
        ", CONSTRAINT uq_lineage_edge_key UNIQUE (edge_id)",
        None,
        "uq_lineage_edge_key",
    ),
    (
        "composite-edge-key-constraint",
        ", CONSTRAINT legacy_edge_identity UNIQUE (edge_key, edge_id)",
        None,
        "legacy_edge_identity",
    ),
)


def _configure_real_lineage_operations(monkeypatch, migration, connection):
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    monkeypatch.setattr(
        migration,
        "op",
        Operations(MigrationContext.configure(connection)),
    )


def _create_lineage_schema_for_preflight(
    connection,
    *,
    edge_key_definition="VARCHAR(64)",
    constraint_sql="",
    index_sql=None,
):
    connection.execute(
        sa.text(
            "CREATE TABLE dataset_lineage_edges ("
            "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
            f"edge_key {edge_key_definition}, from_dataset_id VARCHAR(36), "
            "to_dataset_id VARCHAR(36) NOT NULL, "
            "relation_type VARCHAR(32) NOT NULL, op_task_type VARCHAR(32), "
            "op_task_id VARCHAR(36), op_params JSON, created_at DATETIME"
            f"{constraint_sql})"
        )
    )
    if index_sql is not None:
        connection.execute(sa.text(index_sql))
    connection.execute(
        sa.text(
            "INSERT INTO dataset_lineage_edges "
            "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
            "relation_type) VALUES "
            "(1, 'edge-1', NULL, 'source', 'target', 'merged')"
        )
    )


def _lineage_sqlite_state(connection):
    schema = connection.execute(
        sa.text(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE tbl_name = 'dataset_lineage_edges' "
            "AND type IN ('table', 'index') ORDER BY type, name"
        )
    ).all()
    rows = connection.execute(
        sa.text("SELECT * FROM dataset_lineage_edges ORDER BY id")
    ).all()
    return tuple(schema), tuple(rows)


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize(
    "edge_key_definition",
    ["INTEGER", "VARCHAR(63)"],
    ids=["integer", "wrong-length"],
)
def test_lineage_migration_preserves_wrong_edge_key_type_before_mutation(
    monkeypatch,
    direction,
    edge_key_definition,
):
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_lineage_schema_for_preflight(
            connection,
            edge_key_definition=edge_key_definition,
        )
        _configure_real_lineage_operations(monkeypatch, migration, connection)
        before = _lineage_sqlite_state(connection)

        with pytest.raises(RuntimeError, match="must be VARCHAR\\(64\\)"):
            getattr(migration, direction)()

        assert _lineage_sqlite_state(connection) == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize(
    ("_case_name", "constraint_sql", "index_sql", "object_name"),
    _MALFORMED_LINEAGE_UNIQUENESS_CASES,
    ids=[case[0] for case in _MALFORMED_LINEAGE_UNIQUENESS_CASES],
)
def test_lineage_migration_preserves_malformed_uniqueness_before_mutation(
    monkeypatch,
    direction,
    _case_name,
    constraint_sql,
    index_sql,
    object_name,
):
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_lineage_schema_for_preflight(
            connection,
            constraint_sql=constraint_sql,
            index_sql=index_sql,
        )
        _configure_real_lineage_operations(monkeypatch, migration, connection)
        before = _lineage_sqlite_state(connection)

        with pytest.raises(RuntimeError, match="uniqueness object") as exc_info:
            getattr(migration, direction)()

        assert object_name in str(exc_info.value)
        assert _lineage_sqlite_state(connection) == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_lineage_migration_preserves_unknown_edge_key_nullability_before_mutation(
    monkeypatch,
    direction,
):
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_lineage_schema_for_preflight(
            connection,
            constraint_sql=", CONSTRAINT legacy_edge_identity UNIQUE (edge_key)",
        )
        _configure_real_lineage_operations(monkeypatch, migration, connection)
        actual_inspect = sa.inspect

        class UnknownNullableInspector:
            def __init__(self, bind):
                self.inspector = actual_inspect(bind)

            def get_table_names(self):
                return self.inspector.get_table_names()

            def get_columns(self, table_name):
                columns = self.inspector.get_columns(table_name)
                for column in columns:
                    if column["name"] == "edge_key":
                        column["nullable"] = None
                return columns

            def get_unique_constraints(self, table_name):
                return self.inspector.get_unique_constraints(table_name)

            def get_indexes(self, table_name):
                return self.inspector.get_indexes(table_name)

        monkeypatch.setattr(
            migration,
            "inspect",
            lambda bind: UnknownNullableInspector(bind),
        )
        before = _lineage_sqlite_state(connection)

        with pytest.raises(RuntimeError, match="partial or complete schema"):
            getattr(migration, direction)()

        assert _lineage_sqlite_state(connection) == before


@pytest.mark.parametrize(
    "unique_kind",
    ["none", "constraint", "index", "unnamed_constraint"],
)
def test_lineage_migration_completes_valid_partial_schema_idempotently(
    monkeypatch,
    unique_kind,
):
    migration = _import_lineage_migration()
    first_key = build_dataset_lineage_edge_key(
        "source-a",
        "target-a",
        "merged",
    )
    second_key = build_dataset_lineage_edge_key(
        "source-b",
        "target-b",
        "sync_fetched",
    )
    constraint_sql = (
        ", CONSTRAINT partial_edge_identity UNIQUE (edge_key)"
        if unique_kind == "constraint"
        else ", UNIQUE (edge_key)"
        if unique_kind == "unnamed_constraint"
        else ""
    )
    index_sql = (
        "CREATE UNIQUE INDEX partial_edge_identity "
        "ON dataset_lineage_edges (edge_key)"
        if unique_kind == "index"
        else None
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _create_lineage_schema_for_preflight(
            connection,
            constraint_sql=constraint_sql,
            index_sql=index_sql,
        )
        connection.execute(
            sa.text(
                "UPDATE dataset_lineage_edges SET "
                "from_dataset_id = 'source-a', to_dataset_id = 'target-a' "
                "WHERE id = 1"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(2, 'edge-2', :edge_key, 'source-b', 'target-b', "
                "'sync_fetched')"
            ),
            {"edge_key": second_key},
        )
        _configure_real_lineage_operations(monkeypatch, migration, connection)

        migration.upgrade()
        migration.upgrade()

        inspector = sa.inspect(connection)
        edge_key_column = next(
            column
            for column in inspector.get_columns("dataset_lineage_edges")
            if column["name"] == "edge_key"
        )
        unique_constraints = [
            constraint
            for constraint in inspector.get_unique_constraints(
                "dataset_lineage_edges"
            )
            if tuple(constraint.get("column_names") or ()) == ("edge_key",)
        ]
        rows = connection.execute(
            sa.text(
                "SELECT edge_id, edge_key FROM dataset_lineage_edges ORDER BY id"
            )
        ).all()

    assert getattr(edge_key_column["type"], "__visit_name__", "").lower() == "varchar"
    assert edge_key_column["type"].length == 64
    assert edge_key_column["nullable"] is False
    assert [constraint["name"] for constraint in unique_constraints] == [
        migration.CONSTRAINT_NAME
    ]
    assert rows == [("edge-1", first_key), ("edge-2", second_key)]


_LINEAGE_DEPENDENT_INDEX_CASES = (
    ("non-unique-single", False, ("edge_key",)),
    ("non-unique-target-first", False, ("edge_key", "edge_id")),
    ("non-unique-target-last", False, ("edge_id", "edge_key")),
    ("unique-target-first", True, ("edge_key", "edge_id")),
    ("unique-target-last", True, ("edge_id", "edge_key")),
)


@pytest.mark.parametrize(
    ("case_name", "unique", "columns"),
    _LINEAGE_DEPENDENT_INDEX_CASES,
    ids=[case[0] for case in _LINEAGE_DEPENDENT_INDEX_CASES],
)
def test_lineage_downgrade_preserves_expected_and_dependent_auxiliary_indexes(
    monkeypatch,
    case_name,
    unique,
    columns,
):
    migration = _import_lineage_migration()
    canonical_key = build_dataset_lineage_edge_key(
        "source",
        "target",
        "merged",
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                "edge_key VARCHAR(64) NOT NULL, from_dataset_id VARCHAR(36), "
                "to_dataset_id VARCHAR(36) NOT NULL, "
                "relation_type VARCHAR(32) NOT NULL, "
                "CONSTRAINT uq_lineage_edge_key UNIQUE (edge_key))"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'edge-1', :edge_key, 'source', 'target', 'merged')"
            ),
            {"edge_key": canonical_key},
        )
        auxiliary_name = f"aux_lineage_{case_name.replace('-', '_')}"
        unique_sql = "UNIQUE " if unique else ""
        connection.execute(
            sa.text(
                f"CREATE {unique_sql}INDEX {auxiliary_name} "
                f"ON dataset_lineage_edges ({', '.join(columns)})"
            )
        )
        _configure_real_lineage_operations(monkeypatch, migration, connection)
        before = _lineage_sqlite_state(connection)

        with pytest.raises(
            RuntimeError,
            match="unexpected index|uniqueness object",
        ) as exc_info:
            migration.downgrade()

        assert auxiliary_name in str(exc_info.value)
        assert _lineage_sqlite_state(connection) == before


def test_lineage_upgrade_rejects_unknown_auxiliary_index_uniqueness(monkeypatch):
    migration = _import_lineage_migration()
    canonical_key = build_dataset_lineage_edge_key(
        "source",
        "target",
        "merged",
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                "edge_key VARCHAR(64) NOT NULL, from_dataset_id VARCHAR(36), "
                "to_dataset_id VARCHAR(36) NOT NULL, "
                "relation_type VARCHAR(32) NOT NULL, "
                "CONSTRAINT uq_lineage_edge_key UNIQUE (edge_key))"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'edge-1', :edge_key, 'source', 'target', 'merged')"
            ),
            {"edge_key": canonical_key},
        )
        auxiliary_name = "unknown_unique_edge_key"
        connection.execute(
            sa.text(
                f"CREATE INDEX {auxiliary_name} "
                "ON dataset_lineage_edges (edge_key)"
            )
        )
        _configure_real_lineage_operations(monkeypatch, migration, connection)
        actual_inspect = sa.inspect

        class UnknownUniqueInspector:
            def __init__(self, bind):
                self.inspector = actual_inspect(bind)

            def get_table_names(self):
                return self.inspector.get_table_names()

            def get_columns(self, table_name):
                return self.inspector.get_columns(table_name)

            def get_unique_constraints(self, table_name):
                return self.inspector.get_unique_constraints(table_name)

            def get_indexes(self, table_name):
                indexes = self.inspector.get_indexes(table_name)
                for index in indexes:
                    if index["name"] == auxiliary_name:
                        index["unique"] = None
                return indexes

        monkeypatch.setattr(
            migration,
            "inspect",
            lambda bind: UnknownUniqueInspector(bind),
        )
        before = _lineage_sqlite_state(connection)

        with pytest.raises(RuntimeError, match="unknown uniqueness") as exc_info:
            migration.upgrade()

        assert auxiliary_name in str(exc_info.value)
        assert _lineage_sqlite_state(connection) == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_lineage_migration_preserves_computed_edge_key_before_mutation(
    monkeypatch,
    direction,
):
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                "edge_key VARCHAR(64) GENERATED ALWAYS AS (lower(edge_id)) "
                "VIRTUAL, from_dataset_id VARCHAR(36), "
                "to_dataset_id VARCHAR(36) NOT NULL, "
                "relation_type VARCHAR(32) NOT NULL)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, from_dataset_id, to_dataset_id, relation_type) "
                "VALUES (1, 'EDGE-1', 'source', 'target', 'merged')"
            )
        )
        _configure_real_lineage_operations(monkeypatch, migration, connection)
        before = _lineage_sqlite_state(connection)

        with pytest.raises(RuntimeError, match="computed column") as exc_info:
            getattr(migration, direction)()

        assert "edge_key" in str(exc_info.value)
        assert _lineage_sqlite_state(connection) == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize(
    ("column_shape", "edge_key_definition", "expected_error"),
    [
        ("server-default", "VARCHAR(64) DEFAULT 'fixed'", "server default"),
        ("primary-key", "VARCHAR(64) PRIMARY KEY", "primary key"),
    ],
    ids=["server-default", "primary-key"],
)
def test_lineage_migration_preserves_non_plain_edge_key_before_mutation(
    monkeypatch,
    direction,
    column_shape,
    edge_key_definition,
    expected_error,
):
    migration = _import_lineage_migration()
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                f"edge_key {edge_key_definition}, "
                "from_dataset_id VARCHAR(36), "
                "to_dataset_id VARCHAR(36) NOT NULL, "
                "relation_type VARCHAR(32) NOT NULL)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'edge-1', 'initial-key', 'source', 'target', 'merged')"
            )
        )
        _configure_real_lineage_operations(monkeypatch, migration, connection)
        before = _lineage_sqlite_state(connection)

        with pytest.raises(RuntimeError, match=expected_error) as exc_info:
            getattr(migration, direction)()

        assert column_shape in str(exc_info.value).replace(" ", "-")
        assert "edge_key" in str(exc_info.value)
        assert _lineage_sqlite_state(connection) == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_lineage_migration_preserves_identity_edge_key_before_mutation(
    monkeypatch,
    direction,
):
    migration = _import_lineage_migration()
    canonical_key = build_dataset_lineage_edge_key(
        "source",
        "target",
        "merged",
    )
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dataset_lineage_edges ("
                "id INTEGER PRIMARY KEY, edge_id VARCHAR(36) NOT NULL UNIQUE, "
                "edge_key VARCHAR(64) NOT NULL, from_dataset_id VARCHAR(36), "
                "to_dataset_id VARCHAR(36) NOT NULL, "
                "relation_type VARCHAR(32) NOT NULL, "
                "CONSTRAINT uq_lineage_edge_key UNIQUE (edge_key))"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(id, edge_id, edge_key, from_dataset_id, to_dataset_id, "
                "relation_type) VALUES "
                "(1, 'edge-1', :edge_key, 'source', 'target', 'merged')"
            ),
            {"edge_key": canonical_key},
        )
        actual_inspect = sa.inspect

        class IdentityInspector:
            def __init__(self, bind):
                self.inspector = actual_inspect(bind)

            def get_table_names(self):
                return self.inspector.get_table_names()

            def get_columns(self, table_name):
                columns = self.inspector.get_columns(table_name)
                for column in columns:
                    if column["name"] == "edge_key":
                        column["identity"] = {"always": False}
                return columns

            def get_unique_constraints(self, table_name):
                return self.inspector.get_unique_constraints(table_name)

            def get_indexes(self, table_name):
                return self.inspector.get_indexes(table_name)

        _configure_real_lineage_operations(monkeypatch, migration, connection)
        monkeypatch.setattr(
            migration,
            "inspect",
            lambda bind: IdentityInspector(bind),
        )
        before = _lineage_sqlite_state(connection)

        with pytest.raises(RuntimeError, match="identity column") as exc_info:
            getattr(migration, direction)()

        assert "edge_key" in str(exc_info.value)
        assert _lineage_sqlite_state(connection) == before
