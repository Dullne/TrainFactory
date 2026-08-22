"""Add a per-attempt token to training tasks.

Revision ID: 043_add_training_run_token
Revises: 042_add_audit_created_at_index
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "043_add_training_run_token"
down_revision = "042_add_audit_created_at_index"
branch_labels = None
depends_on = None

COLUMN_NAME = "run_token"
INDEX_NAME = "idx_training_run_token"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if "training_tasks" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("training_tasks")}
    if COLUMN_NAME not in columns:
        op.add_column(
            "training_tasks",
            sa.Column(COLUMN_NAME, sa.String(length=36), nullable=True),
        )

    inspector = inspect(op.get_bind())
    if not any(
        tuple(index.get("column_names") or ()) == (COLUMN_NAME,)
        for index in inspector.get_indexes("training_tasks")
    ):
        op.create_index(
            INDEX_NAME,
            "training_tasks",
            [COLUMN_NAME],
            unique=False,
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if "training_tasks" not in inspector.get_table_names():
        return

    if any(
        index.get("name") == INDEX_NAME
        for index in inspector.get_indexes("training_tasks")
    ):
        op.drop_index(INDEX_NAME, table_name="training_tasks")

    columns = {column["name"] for column in inspector.get_columns("training_tasks")}
    if COLUMN_NAME in columns:
        op.drop_column("training_tasks", COLUMN_NAME)
