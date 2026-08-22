"""
Add two-phase generation fields to generation_tasks.

Phase 1 (qa_extraction): raw data -> QA pairs dataset
Phase 2 (training_gen): QA dataset + embedding -> filtered training data + Milvus ingestion

Revision ID: 018_add_two_phase_generation_fields
Revises: 017_add_generation_content_field
Create Date: 2026-02-06
"""

from alembic import op
import sqlalchemy as sa

revision = '018_add_two_phase_generation_fields'
down_revision = '017_add_generation_content_field'
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c['name'] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    """Add two-phase generation fields."""
    conn = op.get_bind()

    columns_to_add = [
        ('source_qa_dataset_id', sa.String(36)),
        ('embedding_config_id', sa.String(36)),
        ('milvus_collection', sa.String(255)),
        ('similarity_threshold', sa.Float()),
        ('filter_stats', sa.JSON()),
    ]

    for col_name, col_type in columns_to_add:
        if not _column_exists(conn, 'generation_tasks', col_name):
            default = 0.85 if col_name == 'similarity_threshold' else None
            op.add_column(
                'generation_tasks',
                sa.Column(col_name, col_type, nullable=True, server_default=str(default) if default else None)
            )

    # Add index on source_qa_dataset_id for dedup lookups
    try:
        op.create_index('idx_gen_task_qa_dataset', 'generation_tasks', ['source_qa_dataset_id'])
    except Exception:
        pass  # Index may already exist


def downgrade() -> None:
    """Remove two-phase generation fields."""
    conn = op.get_bind()

    try:
        op.drop_index('idx_gen_task_qa_dataset', table_name='generation_tasks')
    except Exception:
        pass

    for col_name in ['filter_stats', 'similarity_threshold', 'milvus_collection',
                     'embedding_config_id', 'source_qa_dataset_id']:
        if _column_exists(conn, 'generation_tasks', col_name):
            op.drop_column('generation_tasks', col_name)
