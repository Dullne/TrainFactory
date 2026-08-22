"""Add deep evaluation fields to deep_evaluation_tasks

Adds eval_type and model_config_data columns for deep evaluation tasks.

Revision ID: 012_add_deep_eval_fields
Revises: 011_add_model_config_container_fields
Create Date: 2026-02-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "012_add_deep_eval_fields"
down_revision: Union[str, None] = "011_add_model_config_container_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = inspector.get_columns(table_name)
    return any(col["name"] == column_name for col in columns)


def upgrade() -> None:
    if not table_exists("deep_evaluation_tasks"):
        return

    if not column_exists("deep_evaluation_tasks", "eval_type"):
        op.add_column(
            "deep_evaluation_tasks",
            sa.Column("eval_type", sa.String(32), nullable=False, server_default="rerank"),
        )

    if not column_exists("deep_evaluation_tasks", "model_config_data"):
        op.add_column(
            "deep_evaluation_tasks",
            sa.Column("model_config_data", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    if not table_exists("deep_evaluation_tasks"):
        return

    if column_exists("deep_evaluation_tasks", "model_config_data"):
        op.drop_column("deep_evaluation_tasks", "model_config_data")

    if column_exists("deep_evaluation_tasks", "eval_type"):
        op.drop_column("deep_evaluation_tasks", "eval_type")
