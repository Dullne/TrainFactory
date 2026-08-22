"""Set the database default for dataset status to registered.

Revision ID: 041_set_dataset_status_default
Revises: 040_reconcile_entity_query_indexes
"""

from alembic import op
import sqlalchemy as sa


revision = "041_set_dataset_status_default"
down_revision = "040_reconcile_entity_query_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "datasets",
        "status",
        existing_type=sa.String(length=50),
        existing_nullable=False,
        server_default="registered",
    )


def downgrade() -> None:
    op.alter_column(
        "datasets",
        "status",
        existing_type=sa.String(length=50),
        existing_nullable=False,
        # 与实体 dataset_entity 的默认 'registered' 保持一致（'uploading' 是
        # 034 改名前的旧语义，回滚后写入会静默产生旧状态）。
        server_default="registered",
    )
