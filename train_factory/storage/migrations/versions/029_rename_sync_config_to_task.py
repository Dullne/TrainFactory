"""Rename sync config to sync task.

Renames:
  - Table: external_sync_configs → external_sync_tasks
  - Columns: config_id → task_id, config_name → task_name (main table)
  - FK columns: config_id → task_id (batches, generations, trainings)

Revision ID: 029_rename_sync_config_to_task
Revises: 028_make_sync_batch_since_time_nullable
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "029_rename_sync_config_to_task"
down_revision = "028_make_sync_batch_since_time_nullable"
branch_labels = None
depends_on = None


def _table_exists(name):
    conn = op.get_bind()
    return name in inspect(conn).get_table_names()


def _column_exists(table, column):
    conn = op.get_bind()
    cols = [c["name"] for c in inspect(conn).get_columns(table)]
    return column in cols


def upgrade():
    # 1. Rename main table (skip if already renamed)
    if _table_exists("external_sync_configs"):
        op.rename_table("external_sync_configs", "external_sync_tasks")

    # 2. Rename columns in main table (skip if already renamed)
    if _column_exists("external_sync_tasks", "config_id"):
        with op.batch_alter_table("external_sync_tasks") as batch_op:
            batch_op.alter_column("config_id", new_column_name="task_id",
                                  existing_type=sa.String(36))
            batch_op.alter_column("config_name", new_column_name="task_name",
                                  existing_type=sa.String(255))

    # 3. Rename FK column in child tables
    if _column_exists("external_sync_batches", "config_id"):
        with op.batch_alter_table("external_sync_batches") as batch_op:
            batch_op.alter_column("config_id", new_column_name="task_id",
                                  existing_type=sa.String(36))

    if _column_exists("external_sync_generations", "config_id"):
        with op.batch_alter_table("external_sync_generations") as batch_op:
            batch_op.alter_column("config_id", new_column_name="task_id",
                                  existing_type=sa.String(36))

    if _column_exists("external_sync_trainings", "config_id"):
        with op.batch_alter_table("external_sync_trainings") as batch_op:
            batch_op.alter_column("config_id", new_column_name="task_id",
                                  existing_type=sa.String(36))


def downgrade():
    # Reverse: rename task back to config
    with op.batch_alter_table("external_sync_trainings") as batch_op:
        batch_op.alter_column("task_id", new_column_name="config_id",
                              existing_type=sa.String(36))

    with op.batch_alter_table("external_sync_generations") as batch_op:
        batch_op.alter_column("task_id", new_column_name="config_id",
                              existing_type=sa.String(36))

    with op.batch_alter_table("external_sync_batches") as batch_op:
        batch_op.alter_column("task_id", new_column_name="config_id",
                              existing_type=sa.String(36))

    with op.batch_alter_table("external_sync_tasks") as batch_op:
        batch_op.alter_column("task_id", new_column_name="config_id",
                              existing_type=sa.String(36))
        batch_op.alter_column("task_name", new_column_name="config_name",
                              existing_type=sa.String(255))

    op.rename_table("external_sync_tasks", "external_sync_configs")
