"""Add token version to users.

Revision ID: 039_add_user_token_version
Revises: 038_add_auth_and_audit_tables
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "039_add_user_token_version"
down_revision = "038_add_auth_and_audit_tables"
branch_labels = None
depends_on = None


def _user_columns() -> set[str]:
    return {
        column["name"]
        for column in inspect(op.get_bind()).get_columns("users")
    }


def upgrade():
    if "token_version" not in _user_columns():
        op.add_column(
            "users",
            sa.Column(
                "token_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade():
    if "token_version" in _user_columns():
        op.drop_column("users", "token_version")
