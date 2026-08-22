"""Merge migration heads

Merge 012_add_generation_tasks and 012_add_deep_eval_fields.

Revision ID: 013_merge_heads
Revises: 012_add_generation_tasks, 012_add_deep_eval_fields
Create Date: 2026-02-05

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "013_merge_heads"
down_revision: Union[str, Sequence[str], None] = (
    "012_add_generation_tasks",
    "012_add_deep_eval_fields",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
