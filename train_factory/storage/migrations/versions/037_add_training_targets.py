"""Add training targets table and related columns.

Revision ID: 037_add_training_targets
Revises: 036_add_sync_milvus_collection
"""

from alembic import op
import sqlalchemy as sa

revision = "037_add_training_targets"
down_revision = "036_add_sync_milvus_collection"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "external_sync_training_targets",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("target_id", sa.String(36), unique=True, index=True, nullable=False),
        sa.Column("task_id", sa.String(36), index=True, nullable=False),
        sa.Column("target_name", sa.String(255), nullable=False),
        sa.Column("model_type", sa.String(32), nullable=False, server_default="embedding"),
        sa.Column("data_phase", sa.String(32), nullable=False, server_default="final"),
        sa.Column("training_method", sa.String(32), nullable=False, server_default="sft"),
        sa.Column("training_config", sa.JSON, nullable=True),
        sa.Column("base_model_path", sa.String(1024), nullable=False, server_default=""),
        sa.Column("base_deployment_id", sa.String(36), nullable=True),
        sa.Column("training_threshold", sa.Integer, nullable=False, server_default="1000"),
        sa.Column("pending_training_samples", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_training_samples", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_trainings", sa.Integer, nullable=False, server_default="0"),
        sa.Column("current_adapter_name", sa.String(255), nullable=True),
        sa.Column("current_adapter_id", sa.String(36), nullable=True),
        sa.Column("current_training_id", sa.String(36), nullable=True),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="idle"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("1")),
        sa.Column("sort_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_index(
        "idx_training_target_task",
        "external_sync_training_targets",
        ["task_id", "is_active"],
    )

    op.add_column(
        "external_sync_generations",
        sa.Column("qa_dataset_id", sa.String(36), nullable=True),
    )
    op.add_column(
        "external_sync_generations",
        sa.Column("qa_sample_count", sa.Integer, nullable=False, server_default="0"),
    )

    op.add_column(
        "external_sync_trainings",
        sa.Column("target_id", sa.String(36), nullable=True),
    )


def downgrade():
    op.drop_column("external_sync_trainings", "target_id")
    op.drop_column("external_sync_generations", "qa_sample_count")
    op.drop_column("external_sync_generations", "qa_dataset_id")
    op.drop_index("idx_training_target_task", table_name="external_sync_training_targets")
    op.drop_table("external_sync_training_targets")
