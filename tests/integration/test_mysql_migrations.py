"""Real-MySQL lifecycle migration integration tests.

The URL is supplied only by the controlled runner environment.  Each test gets
its own random database and drops only that database during teardown.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine, make_url
from pymysql.constants import CLIENT


ROOT_DIR = Path(__file__).parents[2]
ALEMBIC_INI = ROOT_DIR / "train_factory" / "storage" / "migrations" / "alembic.ini"
INIT_SQL = ROOT_DIR / "docker" / "init.sql"
HEAD = "058_add_model_artifact_membership_gate"
REVISION_049 = "049_dataset_lineage_edge_key"
REVISION_052 = "052_generation_publication_staging"
REAPPLIED_REVISIONS = (
    "053_validate_lifecycle_schema",
    "054_add_deployment_replicas",
    "055_bind_configs_and_adapters_to_replicas",
    "056_bind_sync_targets_to_replicas",
    "057_add_replica_operation_fence",
    "058_add_model_artifact_membership_gate",
)
_MUTATION = re.compile(r"^(ALTER|CREATE|DROP|UPDATE|DELETE|INSERT)\b", re.IGNORECASE)


pytestmark = pytest.mark.skipif(
    "TRAINFACTORY_TEST_MYSQL_URL" not in os.environ,
    reason="requires the isolated MySQL migration runner",
)


def _quoted_database(name: str) -> str:
    assert re.fullmatch(r"tf_migration_[0-9a-f]{32}_[0-9a-f]{16}", name)
    return f"`{name}`"


def _runner_base_url():
    run_id = os.environ.get("TRAINFACTORY_MYSQL_RUN_ID", "")
    expected_uuid = os.environ.get("TRAINFACTORY_MYSQL_SERVER_UUID", "")
    value = os.environ.get("TRAINFACTORY_TEST_MYSQL_URL", "")
    try:
        parsed = make_url(value)
        valid = (
            re.fullmatch(r"[0-9a-f]{32}", run_id) is not None
            and parsed.drivername == "mysql+pymysql"
            and parsed.username == "root"
            and bool(parsed.password)
            and parsed.host == "127.0.0.1"
            and parsed.port is not None
            and 1024 <= parsed.port <= 65535
            and parsed.port != 3306
            and parsed.database == f"tf_runner_{run_id}"
            and not parsed.query
            and re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                expected_uuid,
            ) is not None
        )
    except Exception:
        valid = False
    if not valid:
        raise RuntimeError("Isolated MySQL runner environment is invalid") from None
    return parsed, run_id, expected_uuid


def _canonical_bootstrap_sql(database: str) -> str:
    """Load the deployment baseline; Alembic 001 is intentionally not a baseline."""
    descriptor = -1
    try:
        lexical = Path(os.path.abspath(INIT_SQL))
        expected = ROOT_DIR / "docker" / "init.sql"
        metadata = lexical.lstat()
        attributes = getattr(metadata, "st_file_attributes", 0)
        if (
            lexical != expected
            or not stat.S_ISREG(metadata.st_mode)
            or lexical.is_symlink()
            or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or not 1 <= metadata.st_size <= 1024 * 1024
        ):
            raise RuntimeError("Canonical database baseline is invalid")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        payload = os.read(descriptor, metadata.st_size + 1)
        after = lexical.lstat()
        if (
            (opened.st_dev, opened.st_ino, opened.st_size)
            != (metadata.st_dev, metadata.st_ino, metadata.st_size)
            or (after.st_dev, after.st_ino, after.st_size)
            != (metadata.st_dev, metadata.st_ino, metadata.st_size)
            or len(payload) != metadata.st_size
        ):
            raise RuntimeError("Canonical database baseline is invalid")
        expected_digest = os.environ.get("TRAINFACTORY_INIT_SQL_SHA256", "")
        actual_digest = hashlib.sha256(payload).hexdigest()
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
            or not hmac.compare_digest(actual_digest, expected_digest)
        ):
            raise RuntimeError("Canonical database baseline is invalid")
        source = payload.decode("utf-8")
    except (OSError, UnicodeError):
        raise RuntimeError("Canonical database baseline is invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if source.count("USE train_factory;") != 1:
        raise RuntimeError("Canonical database baseline is invalid")
    return source.replace("USE train_factory;", f"USE {_quoted_database(database)};")


def _apply_canonical_baseline(url: str, baseline: str) -> None:
    engine = sa.create_engine(
        url,
        connect_args={"client_flag": CLIENT.MULTI_STATEMENTS},
    )
    try:
        raw = engine.raw_connection()
        try:
            cursor = raw.cursor()
            try:
                cursor.execute(baseline)
                while cursor.nextset():
                    pass
            finally:
                cursor.close()
            raw.commit()
        finally:
            raw.close()
    finally:
        engine.dispose()


@pytest.fixture
def mysql_url() -> Iterator[str]:
    base_url, run_id, expected_uuid = _runner_base_url()
    database = f"tf_migration_{run_id}_{secrets.token_hex(8)}"
    baseline = _canonical_bootstrap_sql(database)
    admin = sa.create_engine(base_url, pool_pre_ping=True)
    created = False
    try:
        with admin.begin() as connection:
            actual_uuid, actual_database = connection.execute(
                sa.text("SELECT @@server_uuid, DATABASE()")
            ).one()
            if actual_uuid != expected_uuid or actual_database != f"tf_runner_{run_id}":
                raise RuntimeError("Isolated MySQL server identity mismatch")
            connection.execute(
                sa.text(
                    f"CREATE DATABASE {_quoted_database(database)} "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            created = True
        case_url = base_url.set(database=database).render_as_string(hide_password=False)
        _apply_canonical_baseline(case_url, baseline)
        yield case_url
    finally:
        if created:
            with admin.begin() as connection:
                connection.execute(
                    sa.text(f"DROP DATABASE {_quoted_database(database)}")
                )
        admin.dispose()


def _alembic_config() -> Config:
    return Config(os.fspath(ALEMBIC_INI))


def _set_migration_url(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("TRAINFACTORY_TEST_MYSQL_URL", url)
    monkeypatch.setenv("MYSQL_URL", url)
    monkeypatch.delenv("MYSQL_URL_FILE", raising=False)


def _upgrade(monkeypatch: pytest.MonkeyPatch, url: str, revision: str) -> None:
    _set_migration_url(monkeypatch, url)
    command.upgrade(_alembic_config(), revision)


def _stamp(monkeypatch: pytest.MonkeyPatch, url: str, revision: str) -> None:
    _set_migration_url(monkeypatch, url)
    command.stamp(_alembic_config(), revision)


def _engine(url: str) -> Engine:
    return sa.create_engine(url, pool_pre_ping=True)


def _current_revision(engine: Engine) -> str:
    with engine.connect() as connection:
        return connection.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()


def _without_leading_comments(statement: str) -> str:
    value = statement.lstrip()
    while True:
        if value.startswith("/*"):
            end = value.find("*/", 2)
            if end < 0:
                return ""
            value = value[end + 2 :].lstrip()
            continue
        if value.startswith("--") or value.startswith("#"):
            newline = value.find("\n")
            if newline < 0:
                return ""
            value = value[newline + 1 :].lstrip()
            continue
        return value


@contextmanager
def _mutation_recorder() -> Iterator[list[str]]:
    mutations: list[str] = []

    def before_cursor_execute(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = _without_leading_comments(str(statement))
        if _MUTATION.match(normalized):
            mutations.append(normalized)

    sa.event.listen(Engine, "before_cursor_execute", before_cursor_execute)
    try:
        yield mutations
    finally:
        sa.event.remove(Engine, "before_cursor_execute", before_cursor_execute)


def _drop_index(connection, table: str, index: str) -> None:
    connection.execute(sa.text(f"ALTER TABLE `{table}` DROP INDEX `{index}`"))


def test_fresh_deployment_baseline_upgrades_to_lifecycle_head(mysql_url, monkeypatch):
    engine = _engine(mysql_url)
    try:
        with _mutation_recorder() as mutations:
            _upgrade(monkeypatch, mysql_url, "head")
        normalized = [" ".join(statement.split()).upper() for statement in mutations]
        assert any(
            statement == (
                "ALTER TABLE AUDIT_LOGS ADD INDEX IDX_AUDIT_CREATED_AT "
                "(CREATED_AT), ALGORITHM=INPLACE, LOCK=NONE"
            )
            for statement in normalized
        )
        inspector = sa.inspect(engine)
        audit_indexes = {
            item.get("name"): tuple(item.get("column_names") or ())
            for item in inspector.get_indexes("audit_logs")
        }
        assert audit_indexes["idx_audit_created_at"] == ("created_at",)
        assert _current_revision(engine) == HEAD
    finally:
        engine.dispose()


def test_deployed_049_repairs_only_safe_missing_object_without_replaying_048_or_backfill(
    mysql_url,
    monkeypatch,
):
    engine = _engine(mysql_url)
    try:
        with engine.begin() as connection:
            _drop_index(
                connection,
                "model_registry",
                "ix_model_registry_source_task_id",
            )
        with _mutation_recorder() as initial_mutations:
            _upgrade(monkeypatch, mysql_url, REVISION_049)
        normalized = [
            " ".join(statement.split()).upper()
            for statement in initial_mutations
        ]
        assert any(
            statement == (
                "ALTER TABLE MODEL_REGISTRY ADD INDEX IX_MODEL_REGISTRY_SOURCE_TASK_ID "
                "(SOURCE_TASK_ID), ALGORITHM=INPLACE, LOCK=NONE"
            )
            for statement in normalized
        )
        model_indexes = {
            item.get("name"): tuple(item.get("column_names") or ())
            for item in sa.inspect(engine).get_indexes("model_registry")
        }
        assert model_indexes["ix_model_registry_source_task_id"] == (
            "source_task_id",
        )
        _stamp(monkeypatch, mysql_url, REVISION_049)
        with engine.begin() as connection:
            _drop_index(connection, "dataset_lineage_edges", "uq_lineage_edge_key")
        with _mutation_recorder() as mutations:
            _upgrade(monkeypatch, mysql_url, "head")
        assert _current_revision(engine) == HEAD
        assert not any("process_create_time" in statement for statement in mutations)
        assert not any(
            statement.upper().startswith("UPDATE DATASET_LINEAGE_EDGES")
            for statement in mutations
        )
        constraints = sa.inspect(engine).get_unique_constraints("dataset_lineage_edges")
        assert any(
            item.get("name") == "uq_lineage_edge_key"
            and tuple(item.get("column_names") or ()) == ("edge_key",)
            for item in constraints
        )
    finally:
        engine.dispose()


def test_lifecycle_schema_chain_reapply_is_idempotent(mysql_url, monkeypatch):
    _upgrade(monkeypatch, mysql_url, "head")
    _stamp(monkeypatch, mysql_url, REVISION_052)
    engine = _engine(mysql_url)
    try:
        with _mutation_recorder() as mutations:
            _upgrade(monkeypatch, mysql_url, "head")
        framework_updates = [
            statement
            for statement in mutations
            if statement.upper().startswith("UPDATE ALEMBIC_VERSION")
        ]
        business_mutations = [
            statement for statement in mutations if statement not in framework_updates
        ]
        assert len(framework_updates) == len(REAPPLIED_REVISIONS)
        assert business_mutations == []
        assert _current_revision(engine) == HEAD
    finally:
        engine.dispose()


def test_053_repairs_safe_missing_objects_from_each_lifecycle_revision(
    mysql_url,
    monkeypatch,
):
    _upgrade(monkeypatch, mysql_url, REVISION_052)
    engine = _engine(mysql_url)
    try:
        with engine.begin() as connection:
            connection.execute(sa.text("ALTER TABLE training_tasks DROP COLUMN process_create_time"))
            _drop_index(connection, "dataset_lineage_edges", "uq_lineage_edge_key")
            _drop_index(connection, "evaluation_tasks", "idx_eval_run_token")
            connection.execute(sa.text("ALTER TABLE evaluation_tasks DROP COLUMN run_token"))
            _drop_index(connection, "generation_tasks", "idx_gen_task_run_token")
            connection.execute(sa.text("ALTER TABLE generation_tasks DROP COLUMN run_token"))
            _drop_index(connection, "datasets", "idx_dataset_generation_run_token")
            connection.execute(sa.text("ALTER TABLE datasets DROP COLUMN generation_run_token"))
        _upgrade(monkeypatch, mysql_url, "head")
        inspector = sa.inspect(engine)
        assert "process_create_time" in {
            column["name"] for column in inspector.get_columns("training_tasks")
        }
        assert "run_token" in {
            column["name"] for column in inspector.get_columns("evaluation_tasks")
        }
        assert "run_token" in {
            column["name"] for column in inspector.get_columns("generation_tasks")
        }
        assert "generation_run_token" in {
            column["name"] for column in inspector.get_columns("datasets")
        }
        assert _current_revision(engine) == HEAD
    finally:
        engine.dispose()


def test_malformed_same_name_object_fails_without_mutation_or_stamp(mysql_url, monkeypatch):
    _upgrade(monkeypatch, mysql_url, REVISION_052)
    engine = _engine(mysql_url)
    try:
        with engine.begin() as connection:
            _drop_index(connection, "evaluation_tasks", "idx_eval_run_token")
            connection.execute(
                sa.text(
                    "CREATE UNIQUE INDEX idx_eval_run_token "
                    "ON evaluation_tasks (run_token)"
                )
            )
        with _mutation_recorder() as mutations:
            with pytest.raises(RuntimeError) as exc_info:
                _upgrade(monkeypatch, mysql_url, "head")
        assert "050_add_evaluation_run_token:index_shape" in str(exc_info.value)
        assert mutations == []
        assert _current_revision(engine) == REVISION_052
    finally:
        engine.dispose()


def test_lineage_conflict_precedes_all_safe_repairs_on_real_mysql(mysql_url, monkeypatch):
    _upgrade(monkeypatch, mysql_url, REVISION_052)
    engine = _engine(mysql_url)
    try:
        with engine.begin() as connection:
            connection.execute(sa.text("ALTER TABLE training_tasks DROP COLUMN process_create_time"))
            _drop_index(connection, "dataset_lineage_edges", "uq_lineage_edge_key")
            statement = sa.text(
                "INSERT INTO dataset_lineage_edges "
                "(edge_id, edge_key, from_dataset_id, to_dataset_id, relation_type, "
                "op_task_type, op_task_id, op_params, created_at) VALUES "
                "(:edge_id, :edge_key, :source, :target, :relation, "
                ":task_type, :task_id, :params, CURRENT_TIMESTAMP)"
            )
            common = {
                "source": "source-dataset",
                "target": "target-dataset",
                "relation": "merged",
                "task_type": "sync",
            }
            connection.execute(
                statement,
                {
                    **common,
                    "edge_id": "11111111-1111-1111-1111-111111111111",
                    "edge_key": "a" * 64,
                    "task_id": "task-one",
                    "params": '{"source":"one"}',
                },
            )
            connection.execute(
                statement,
                {
                    **common,
                    "edge_id": "22222222-2222-2222-2222-222222222222",
                    "edge_key": "b" * 64,
                    "task_id": "task-two",
                    "params": '{"source":"two"}',
                },
            )
        with _mutation_recorder() as mutations:
            with pytest.raises(RuntimeError) as exc_info:
                _upgrade(monkeypatch, mysql_url, "head")
        assert "lineage_data_incompatible" in str(exc_info.value)
        assert mutations == []
        assert _current_revision(engine) == REVISION_052
        assert "process_create_time" not in {
            column["name"] for column in sa.inspect(engine).get_columns("training_tasks")
        }
    finally:
        engine.dispose()
