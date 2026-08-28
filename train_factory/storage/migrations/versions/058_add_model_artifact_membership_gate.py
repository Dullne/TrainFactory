"""Add the singleton model artifact membership serialization gate.

Revision ID: 058_add_model_artifact_membership_gate
Revises: 057_add_replica_operation_fence
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "058_add_model_artifact_membership_gate"
down_revision = "057_add_replica_operation_fence"
branch_labels = None
depends_on = None

TABLE_NAME = "model_artifact_membership_gate"
GATE_ID = 1


def _validate(bind) -> None:
    reflected = inspect(bind)
    if TABLE_NAME not in set(reflected.get_table_names()):
        raise RuntimeError(f"Required {TABLE_NAME} table is missing")
    columns = reflected.get_columns(TABLE_NAME)
    if len(columns) != 1:
        raise RuntimeError(f"{TABLE_NAME} shape is invalid")
    column = columns[0]
    if (
        column["name"] != "gate_id"
        or getattr(column["type"], "__visit_name__", "").lower()
        not in {"integer", "int"}
        or column["nullable"] is not False
    ):
        raise RuntimeError(f"{TABLE_NAME}.gate_id shape is invalid")
    primary_key = reflected.get_pk_constraint(TABLE_NAME)
    if primary_key.get("constrained_columns") != ["gate_id"]:
        raise RuntimeError(f"{TABLE_NAME}.gate_id primary key is invalid")


def upgrade() -> None:
    bind = op.get_bind()
    if TABLE_NAME not in set(inspect(bind).get_table_names()):
        op.create_table(
            TABLE_NAME,
            sa.Column("gate_id", sa.Integer(), primary_key=True, nullable=False),
        )
    _validate(bind)
    table = sa.Table(TABLE_NAME, sa.MetaData(), autoload_with=bind)
    if bind.scalar(
        sa.select(sa.func.count()).select_from(table).where(
            table.c.gate_id == GATE_ID
        )
    ) == 0:
        bind.execute(table.insert().values(gate_id=GATE_ID))


def downgrade() -> None:
    bind = op.get_bind()
    if TABLE_NAME in set(inspect(bind).get_table_names()):
        _validate(bind)
        op.drop_table(TABLE_NAME)
