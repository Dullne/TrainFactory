"""Singleton row used to serialize model artifact membership changes."""

from sqlmodel import Field, SQLModel


class ModelArtifactMembershipGateDB(SQLModel, table=True):
    __tablename__ = "model_artifact_membership_gate"

    gate_id: int = Field(primary_key=True)
