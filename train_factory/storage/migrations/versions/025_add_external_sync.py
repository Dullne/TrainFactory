"""
Add external sync tables for two-level threshold auto-trigger pipeline.

Creates 4 tables:
- external_sync_configs: per-user sync configuration
- external_sync_batches: incremental data fetch batches
- external_sync_generations: generation task tracking
- external_sync_trainings: training task tracking

Revision ID: 025_add_external_sync
Revises: 024_add_collection_registry
Create Date: 2026-02-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "025_add_external_sync"
down_revision: Union[str, None] = "024_add_collection_registry"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    # ── 1. external_sync_configs ──
    if not _table_exists("external_sync_configs"):
        op.create_table(
            "external_sync_configs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("config_id", sa.String(36), nullable=False, unique=True),
            sa.Column("config_name", sa.String(255), nullable=False),
            sa.Column("user_id", sa.String(64), nullable=False),
            # 外部 API
            sa.Column("external_api_url", sa.String(1024), nullable=False),
            sa.Column("external_auth_config", sa.JSON(), nullable=False),
            # 同步参数
            sa.Column("sync_interval_seconds", sa.Integer(), nullable=False, server_default="300"),
            sa.Column("last_sync_at", sa.DateTime(), nullable=True),
            sa.Column("last_sync_boundary_ids", sa.JSON(), nullable=True),
            # Level 1
            sa.Column("generation_threshold", sa.Integer(), nullable=False, server_default="500"),
            sa.Column("generation_mode", sa.String(32), nullable=False, server_default="doc_to_training"),
            sa.Column("generation_config", sa.JSON(), nullable=True),
            # Level 2
            sa.Column("training_threshold", sa.Integer(), nullable=False, server_default="1000"),
            sa.Column("training_config", sa.JSON(), nullable=True),
            # 部署
            sa.Column("base_deployment_id", sa.String(36), nullable=True),
            # 计数器
            sa.Column("pending_record_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("pending_training_samples", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("total_record_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("total_training_samples", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("total_trainings", sa.Integer(), nullable=False, server_default="0"),
            # 当前 adapter
            sa.Column("current_adapter_name", sa.String(255), nullable=True),
            sa.Column("current_adapter_id", sa.String(36), nullable=True),
            sa.Column("current_training_id", sa.String(36), nullable=True),
            # 状态
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
            sa.Column("status", sa.String(32), nullable=False, server_default="idle"),
            sa.Column("error_message", sa.Text(), nullable=True),
            # 时间
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_sync_config_id", "external_sync_configs", ["config_id"])
        op.create_index("idx_sync_user", "external_sync_configs", ["user_id"])
        op.create_index("idx_sync_active", "external_sync_configs", ["is_active", "status"])
        op.create_unique_constraint(
            "uq_sync_user_name", "external_sync_configs", ["user_id", "config_name"]
        )

    # ── 2. external_sync_batches ──
    if not _table_exists("external_sync_batches"):
        op.create_table(
            "external_sync_batches",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("batch_id", sa.String(36), nullable=False, unique=True),
            sa.Column("config_id", sa.String(36), nullable=False),
            sa.Column("user_id", sa.String(64), nullable=False),
            sa.Column("record_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("storage_path", sa.String(1024), nullable=False),
            sa.Column("dataset_id", sa.String(36), nullable=True),
            sa.Column("since_time", sa.DateTime(), nullable=False),
            sa.Column("until_time", sa.DateTime(), nullable=True),
            sa.Column("fetched_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("status", sa.String(32), nullable=False, server_default="fetched"),
            sa.Column("generation_task_id", sa.String(36), nullable=True),
        )
        op.create_index("idx_batch_id", "external_sync_batches", ["batch_id"])
        op.create_index("idx_batch_config", "external_sync_batches", ["config_id", "status"])
        op.create_index("idx_batch_user", "external_sync_batches", ["user_id"])

    # ── 3. external_sync_generations ──
    if not _table_exists("external_sync_generations"):
        op.create_table(
            "external_sync_generations",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("config_id", sa.String(36), nullable=False),
            sa.Column("generation_task_id", sa.String(36), nullable=False, unique=True),
            sa.Column("user_id", sa.String(64), nullable=False),
            sa.Column("input_batch_ids", sa.JSON(), nullable=False),
            sa.Column("input_record_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("output_dataset_id", sa.String(36), nullable=True),
            sa.Column("output_sample_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
        )
        op.create_index("idx_syncgen_config", "external_sync_generations", ["config_id", "status"])
        op.create_index("idx_syncgen_task", "external_sync_generations", ["generation_task_id"])

    # ── 4. external_sync_trainings ──
    if not _table_exists("external_sync_trainings"):
        op.create_table(
            "external_sync_trainings",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("config_id", sa.String(36), nullable=False),
            sa.Column("training_task_id", sa.String(36), nullable=False, unique=True),
            sa.Column("user_id", sa.String(64), nullable=False),
            sa.Column("input_dataset_ids", sa.JSON(), nullable=False),
            sa.Column("total_samples", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("training_round", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("output_adapter_path", sa.String(1024), nullable=True),
            sa.Column("output_model_registry_id", sa.String(36), nullable=True),
            sa.Column("loaded_adapter_name", sa.String(255), nullable=True),
            sa.Column("loaded_adapter_id", sa.String(36), nullable=True),
            sa.Column("previous_training_task_id", sa.String(36), nullable=True),
            sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
        )
        op.create_index("idx_synctrain_config", "external_sync_trainings", ["config_id", "status"])
        op.create_index("idx_synctrain_task", "external_sync_trainings", ["training_task_id"])


def downgrade() -> None:
    for table in ("external_sync_trainings", "external_sync_generations",
                  "external_sync_batches", "external_sync_configs"):
        if _table_exists(table):
            op.drop_table(table)
