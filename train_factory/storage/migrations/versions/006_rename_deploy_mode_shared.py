"""Rename deploy_mode external to shared

Updates deployments.deploy_mode values from 'external' to 'shared' and
sets the default to 'shared'.

Revision ID: 006_rename_deploy_mode_shared
Revises: 005_add_model_progress
Create Date: 2026-01-29
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '006_rename_deploy_mode_shared'
down_revision: Union[str, None] = '005_add_model_progress'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """Update deploy_mode values and default to shared."""
    if column_exists('deployments', 'deploy_mode'):
        op.execute(
            "UPDATE deployments SET deploy_mode = 'shared' "
            "WHERE deploy_mode = 'external' OR deploy_mode IS NULL"
        )
        op.alter_column('deployments', 'deploy_mode', server_default='shared')


def downgrade() -> None:
    """Revert deploy_mode values and default to external."""
    if column_exists('deployments', 'deploy_mode'):
        op.execute(
            "UPDATE deployments SET deploy_mode = 'external' "
            "WHERE deploy_mode = 'shared'"
        )
        op.alter_column('deployments', 'deploy_mode', server_default='external')
