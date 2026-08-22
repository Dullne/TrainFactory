"""Add milvus_collection_name column to external_sync_tasks.

Revision ID: 036_add_sync_milvus_collection
Revises: 035_add_deployment_external_api_config_id
"""

from alembic import op
import sqlalchemy as sa


revision = "036_add_sync_milvus_collection"
down_revision = "035_add_deployment_external_api_config_id"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "external_sync_tasks",
        sa.Column("milvus_collection_name", sa.String(length=255), nullable=True),
    )


def downgrade():
    op.drop_column("external_sync_tasks", "milvus_collection_name")
