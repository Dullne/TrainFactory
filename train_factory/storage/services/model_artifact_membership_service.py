"""Database serialization for registry artifact membership changes."""

from sqlalchemy import update
from sqlmodel import select

from ..entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)


MODEL_ARTIFACT_MEMBERSHIP_GATE_ID = 1


def lock_model_artifact_membership(session) -> None:
    """Take the fixed gate row as the first lock in a membership transaction."""
    # SQLite ignores ``SELECT .. FOR UPDATE``. A no-op UPDATE obtains its
    # database write lock there and a row lock on MySQL; existence is verified
    # independently so driver-specific no-op row counts are irrelevant.
    session.exec(
        update(ModelArtifactMembershipGateDB)
        .where(
            ModelArtifactMembershipGateDB.gate_id
            == MODEL_ARTIFACT_MEMBERSHIP_GATE_ID
        )
        .values(gate_id=ModelArtifactMembershipGateDB.gate_id)
    )
    gate = session.exec(
        select(ModelArtifactMembershipGateDB)
        .where(
            ModelArtifactMembershipGateDB.gate_id
            == MODEL_ARTIFACT_MEMBERSHIP_GATE_ID
        )
        .with_for_update()
    ).first()
    if gate is None:
        raise RuntimeError("model artifact membership gate is missing")
