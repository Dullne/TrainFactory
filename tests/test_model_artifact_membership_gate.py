from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.services.model_artifact_membership_service import (
    lock_model_artifact_membership,
)


MIGRATION_PATH = (
    Path(__file__).parents[1]
    / "train_factory"
    / "storage"
    / "migrations"
    / "versions"
    / "058_add_model_artifact_membership_gate.py"
)


def test_058_migration_is_present_and_descends_from_current_head() -> None:
    assert MIGRATION_PATH.is_file()
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    assert 'revision = "058_add_model_artifact_membership_gate"' in source
    assert 'down_revision = "057_add_replica_operation_fence"' in source


def test_058_creates_idempotent_singleton_artifact_membership_gate(
    monkeypatch,
) -> None:
    pytest.importorskip("alembic")
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = importlib.import_module(
        "train_factory.storage.migrations.versions.058_add_model_artifact_membership_gate"
    )
    assert migration.revision == "058_add_model_artifact_membership_gate"
    assert migration.down_revision == "057_add_replica_operation_fence"

    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        monkeypatch.setattr(migration, "op", operations)

        migration.upgrade()
        migration.upgrade()

        inspector = sa.inspect(connection)
        assert migration.TABLE_NAME in inspector.get_table_names()
        columns = inspector.get_columns(migration.TABLE_NAME)
        assert [(column["name"], column["nullable"]) for column in columns] == [
            ("gate_id", False)
        ]
        rows = connection.execute(
            sa.text(
                f"SELECT gate_id FROM {migration.TABLE_NAME} ORDER BY gate_id"
            )
        ).all()
        assert rows == [(migration.GATE_ID,)]

        migration.downgrade()
        assert migration.TABLE_NAME not in sa.inspect(connection).get_table_names()


def test_sqlite_gate_blocks_a_second_connection_until_owner_releases(tmp_path) -> None:
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'membership-gate.db'}",
        connect_args={"timeout": 0.05, "check_same_thread": False},
    )
    ModelArtifactMembershipGateDB.__table__.create(engine)
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()

    owner = Session(engine)
    contender = Session(engine)
    try:
        lock_model_artifact_membership(owner)
        with pytest.raises(OperationalError, match="locked"):
            lock_model_artifact_membership(contender)
        contender.rollback()
        owner.rollback()

        lock_model_artifact_membership(contender)
        contender.rollback()
    finally:
        owner.close()
        contender.close()
