"""Add authentication and audit log tables.

Revision ID: 038_add_auth_and_audit_tables
Revises: 037_add_training_targets
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "038_add_auth_and_audit_tables"
down_revision = "037_add_training_targets"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def upgrade():
    if not _table_exists("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            # uuid4 为 36 字符，与 audit_logs.user_id 保持一致
            sa.Column("user_id", sa.String(36), nullable=False),
            sa.Column("username", sa.String(64), nullable=False),
            sa.Column("email", sa.String(256), nullable=True),
            sa.Column("hashed_password", sa.String(256), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("is_admin", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)
        op.create_index("ix_users_is_active", "users", ["is_active"])
        op.create_index("ix_users_user_id", "users", ["user_id"], unique=True)
        op.create_index("ix_users_username", "users", ["username"], unique=True)

    if not _table_exists("audit_logs"):
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("log_id", sa.String(36), nullable=False),
            sa.Column("user_id", sa.String(36), nullable=False),
            sa.Column("username", sa.String(255), nullable=True),
            sa.Column("action", sa.String(50), nullable=False),
            sa.Column("resource_type", sa.String(50), nullable=False),
            sa.Column("resource_id", sa.String(36), nullable=True),
            sa.Column("method", sa.String(10), nullable=False),
            sa.Column("endpoint", sa.String(500), nullable=False),
            sa.Column("request_body", sa.JSON(), nullable=True),
            sa.Column("status_code", sa.Integer(), nullable=False),
            sa.Column("response_summary", sa.String(1000), nullable=True),
            sa.Column("ip_address", sa.String(45), nullable=True),
            sa.Column("user_agent", sa.String(500), nullable=True),
            sa.Column("extra_data", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("idx_audit_action", "audit_logs", ["action"])
        op.create_index(
            "idx_audit_resource",
            "audit_logs",
            ["resource_type", "resource_id"],
        )
        op.create_index(
            "idx_audit_user_time",
            "audit_logs",
            ["user_id", "created_at"],
        )
        op.create_index("ix_audit_logs_log_id", "audit_logs", ["log_id"])
        op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])


def downgrade():
    # These tables may have been created by SQLModel before this migration was
    # introduced. Preserve authentication and audit data rather than deleting a
    # table that upgrade() merely adopted.
    pass
