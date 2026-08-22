"""
Add full pipeline fields for training_gen from raw docs.

Stores QA intermediate output path and dataset ID when
training_gen runs the full pipeline (raw docs → QA → training data).

Revision ID: 019_add_full_pipeline_fields
Revises: 018_add_two_phase_generation_fields
Create Date: 2026-02-06
"""

from alembic import op
import sqlalchemy as sa

revision = '019_add_full_pipeline_fields'
down_revision = '018_add_two_phase_generation_fields'
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c['name'] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    """Add qa_output_path and qa_dataset_id columns."""
    conn = op.get_bind()

    columns_to_add = [
        ('qa_output_path', sa.String(1024)),
        ('qa_dataset_id', sa.String(36)),
        ('qa_filtered_path', sa.String(1024)),
        ('qa_filtered_dataset_id', sa.String(36)),
        ('retrieval_top_k', sa.Integer()),
    ]

    for col_name, col_type in columns_to_add:
        if not _column_exists(conn, 'generation_tasks', col_name):
            op.add_column(
                'generation_tasks',
                sa.Column(col_name, col_type, nullable=True)
            )


def downgrade() -> None:
    """Remove full pipeline fields."""
    conn = op.get_bind()

    for col_name in ['retrieval_top_k', 'qa_filtered_dataset_id', 'qa_filtered_path', 'qa_dataset_id', 'qa_output_path']:
        if _column_exists(conn, 'generation_tasks', col_name):
            op.drop_column('generation_tasks', col_name)
