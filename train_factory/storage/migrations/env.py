"""
Alembic migration environment configuration for TrainFactory.

This module configures how Alembic runs migrations, connecting to the
database and using SQLModel's metadata for auto-generation support.
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, text
from sqlalchemy.engine import make_url
from alembic import context
import os
import re
import sys

# Add project root to path so we can import our modules
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.insert(0, project_root)

# Alembic Config object
config = context.config

# Set up Python logging from the config file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def _controlled_runner_url():
    run_id = os.environ.get("TRAINFACTORY_MYSQL_RUN_ID")
    test_url = os.environ.get("TRAINFACTORY_TEST_MYSQL_URL")
    server_uuid = os.environ.get("TRAINFACTORY_MYSQL_SERVER_UUID")
    if run_id is None and test_url is None and server_uuid is None:
        return None
    try:
        parsed = make_url(test_url or "")
        valid = (
            re.fullmatch(r"[0-9a-f]{32}", run_id or "") is not None
            and re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                server_uuid or "",
            ) is not None
            and parsed.drivername == "mysql+pymysql"
            and parsed.username == "root"
            and bool(parsed.password)
            and parsed.host == "127.0.0.1"
            and parsed.port is not None
            and 1024 <= parsed.port <= 65535
            and parsed.port != 3306
            and re.fullmatch(
                rf"tf_migration_{run_id}_[0-9a-f]{{16}}",
                parsed.database or "",
            ) is not None
            and not parsed.query
        )
    except Exception:
        valid = False
    if not valid:
        raise RuntimeError("Controlled migration URL is invalid") from None
    return test_url


_CONTROLLED_RUNNER_URL = _controlled_runner_url()

if _CONTROLLED_RUNNER_URL is None:
    # Entity metadata is needed for normal operation and autogenerate.  The
    # controlled integration runner performs upgrade/current only and avoids
    # importing the application (and its Settings/Torch side effects).
    from sqlmodel import SQLModel  # noqa: E402
    from train_factory.storage.entities import (  # noqa: E402, F401
        TrainingTaskDB,
        TrainingTaskEventDB,
        ModelRegistryDB,
        ModelVersionDB,
        DeploymentDB,
        LoadedAdapterDB,
        ModelConfigDB,
        DatasetDB,
        UserDB,
        AuditLogDB,
        EvaluationTaskDB,
        DeepEvaluationTaskDB,
        GenerationTaskDB,
        MilvusCollectionDB,
        CollectionDatasetLinkDB,
        ExternalSyncTaskDB,
        ExternalSyncBatchDB,
        ExternalApiConfigDB,
        DatasetAssetDB,
        DatasetLineageEdgeDB,
    )

    target_metadata = SQLModel.metadata
else:
    target_metadata = None

from train_factory.storage.migrations.version_table import (  # noqa: E402
    ensure_version_table_capacity,
)


def get_url():
    """Get database URL from settings or environment."""
    if _CONTROLLED_RUNNER_URL is not None:
        return _CONTROLLED_RUNNER_URL
    try:
        from train_factory.config import settings
    except ImportError:
        settings = None

    if settings is not None:
        return settings.mysql_url

    url = os.environ.get("MYSQL_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("MYSQL_URL or DATABASE_URL is required for migrations")
    return url


def _verify_controlled_connection(connection) -> None:
    if _CONTROLLED_RUNNER_URL is None:
        return
    run_id = os.environ["TRAINFACTORY_MYSQL_RUN_ID"]
    expected_uuid = os.environ["TRAINFACTORY_MYSQL_SERVER_UUID"]
    expected_database = make_url(_CONTROLLED_RUNNER_URL).database
    marker_database = f"tf_runner_{run_id}"
    try:
        row = connection.execute(
            text(
                "SELECT @@server_uuid AS server_uuid, DATABASE() AS database_name, "
                "EXISTS(SELECT 1 FROM information_schema.schemata "
                "WHERE schema_name = :marker_database) AS marker_exists"
            ),
            {"marker_database": marker_database},
        ).mappings().one()
        valid = (
            row["server_uuid"] == expected_uuid
            and row["database_name"] == expected_database
            and row["marker_exists"] in (1, True)
        )
    except Exception:
        valid = False
    if not valid:
        raise RuntimeError("Controlled migration database identity mismatch") from None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    if _CONTROLLED_RUNNER_URL is not None:
        raise RuntimeError("Controlled migrations require an online connection")
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        _verify_controlled_connection(connection)
        ensure_version_table_capacity(connection)
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()

        # SQLAlchemy 2.x + MySQL non-transactional DDL can leave the
        # alembic_version UPDATE in an implicit transaction.
        # Explicitly commit so revision stamps persist.
        if connection.in_transaction():
            connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
