"""
Add eval_llm_config column for independent evaluation LLM.

Revision ID: 021_add_eval_llm_config
Revises: 020_simplify_generation_modes
Create Date: 2026-02-09
"""

from alembic import op
import sqlalchemy as sa

revision = '021_add_eval_llm_config'
down_revision = '020_simplify_generation_modes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('generation_tasks', sa.Column('eval_llm_config', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('generation_tasks', 'eval_llm_config')
