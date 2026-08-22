"""Persist sync training claims for crash-safe reconciliation.

Revision ID: 044_sync_training_claims
Revises: 043_add_training_run_token
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "044_sync_training_claims"
down_revision = "043_add_training_run_token"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if "external_sync_trainings" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("external_sync_trainings")
    }
    additions = (
        (
            "claimed_sample_count",
            sa.Column(
                "claimed_sample_count",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        ),
        (
            "previous_target_status",
            sa.Column("previous_target_status", sa.String(length=32), nullable=True),
        ),
        (
            "previous_parent_status",
            sa.Column("previous_parent_status", sa.String(length=32), nullable=True),
        ),
        (
            "claim_reconciled",
            sa.Column(
                "claim_reconciled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        ),
        (
            "target_config_snapshot",
            sa.Column(
                "target_config_snapshot",
                sa.JSON(),
                nullable=True,
            ),
        ),
    )
    for name, column in additions:
        if name not in columns:
            op.add_column("external_sync_trainings", column)

    inspector = inspect(op.get_bind())
    columns = {
        column["name"]
        for column in inspector.get_columns("external_sync_trainings")
    }
    if "target_config_snapshot" in columns:
        trainings = sa.table(
            "external_sync_trainings",
            sa.column("target_config_snapshot", sa.JSON()),
        )
        op.execute(
            trainings.update()
            .where(trainings.c.target_config_snapshot.is_(None))
            .values(target_config_snapshot={})
        )
        with op.batch_alter_table("external_sync_trainings") as batch_op:
            batch_op.alter_column(
                "target_config_snapshot",
                existing_type=sa.JSON(),
                nullable=False,
                # MySQL 8.0.13+ 支持 JSON 表达式默认值：避免 NOT NULL 无默认值
                # 的死角（非实体路径 INSERT 报 1364）。
                server_default=sa.text("(JSON_OBJECT())"),
            )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if "external_sync_trainings" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("external_sync_trainings")
    }
    for name in (
        "target_config_snapshot",
        "claim_reconciled",
        "previous_parent_status",
        "previous_target_status",
        "claimed_sample_count",
    ):
        if name in columns:
            op.drop_column("external_sync_trainings", name)
