"""
Add external_api_configs table and external_api_config_id column.

Revision ID: 028_add_external_api_configs
Revises: 027_add_last_sync_boundary_ids
Create Date: 2026-02-12
"""

from alembic import op
import sqlalchemy as sa

revision = "028_add_external_api_configs"
down_revision = "027_add_last_sync_boundary_ids"
branch_labels = None
depends_on = None


def _table_exists(conn, table: str) -> bool:
    inspector = sa.inspect(conn)
    return table in inspector.get_table_names()


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    conn = op.get_bind()

    # Create external_api_configs table
    if not _table_exists(conn, "external_api_configs"):
        op.create_table(
            "external_api_configs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("config_id", sa.String(36), unique=True, nullable=False, index=True),
            sa.Column("config_name", sa.String(255), nullable=False),
            sa.Column("user_id", sa.String(64), nullable=False),
            sa.Column("api_url", sa.String(1024), nullable=False),
            sa.Column("auth_config", sa.JSON(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("status", sa.String(32), nullable=False, server_default="active"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        op.create_index("idx_ext_api_user", "external_api_configs", ["user_id"])
        op.create_index("idx_ext_api_status", "external_api_configs", ["status"])

    # Add external_api_config_id to external_sync_configs
    if _table_exists(conn, "external_sync_configs"):
        if not _column_exists(conn, "external_sync_configs", "external_api_config_id"):
            op.add_column(
                "external_sync_configs",
                sa.Column("external_api_config_id", sa.String(36), nullable=True),
            )

        # Make external_api_url and external_auth_config nullable
        # (they can be empty when using external_api_config_id reference)
        try:
            op.alter_column(
                "external_sync_configs", "external_api_url",
                existing_type=sa.String(1024), nullable=True, server_default="",
            )
        except Exception:
            pass
        try:
            op.alter_column(
                "external_sync_configs", "external_auth_config",
                existing_type=sa.JSON(), nullable=True,
            )
        except Exception:
            pass


def downgrade() -> None:
    conn = op.get_bind()

    if _table_exists(conn, "external_sync_configs"):
        if _column_exists(conn, "external_sync_configs", "external_api_config_id"):
            op.drop_column("external_sync_configs", "external_api_config_id")

    if _table_exists(conn, "external_api_configs"):
        op.drop_table("external_api_configs")
