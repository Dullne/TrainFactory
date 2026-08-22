"""
Add container_name and inference_framework fields to model_configs table.

These fields are needed to properly display container information
for local deployed model configurations.
"""

import logging

from alembic import op
import sqlalchemy as sa

logger = logging.getLogger(__name__)

# revision identifiers
revision = '011_add_model_config_container_fields'
down_revision = '010_add_deep_evaluation_tasks'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add container_name and inference_framework columns to model_configs."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Check if model_configs table exists
    if 'model_configs' not in inspector.get_table_names():
        logger.info("model_configs table does not exist, skipping migration")
        return

    # Get existing columns
    existing_columns = [col['name'] for col in inspector.get_columns('model_configs')]

    # Add container_name column if not exists
    if 'container_name' not in existing_columns:
        op.add_column(
            'model_configs',
            sa.Column('container_name', sa.String(255), nullable=True)
        )
        logger.info("Added container_name column to model_configs")
    else:
        logger.info("container_name column already exists in model_configs")

    # Add inference_framework column if not exists
    if 'inference_framework' not in existing_columns:
        op.add_column(
            'model_configs',
            sa.Column('inference_framework', sa.String(32), nullable=True)
        )
        logger.info("Added inference_framework column to model_configs")
    else:
        logger.info("inference_framework column already exists in model_configs")


def downgrade() -> None:
    """Remove container_name and inference_framework columns from model_configs."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if 'model_configs' not in inspector.get_table_names():
        return

    existing_columns = [col['name'] for col in inspector.get_columns('model_configs')]

    if 'container_name' in existing_columns:
        op.drop_column('model_configs', 'container_name')

    if 'inference_framework' in existing_columns:
        op.drop_column('model_configs', 'inference_framework')
