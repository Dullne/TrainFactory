"""
Add source_dataset_id to generation_tasks and datasets tables.

Unify source dataset reference for all generation modes,
and track dataset lineage in datasets table.

Revision ID: 021_add_source_dataset_id
Revises: 021_add_eval_llm_config
Create Date: 2026-02-09
"""

from alembic import op
import sqlalchemy as sa

revision = '021_add_source_dataset_id'
down_revision = '021_add_eval_llm_config'
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    columns = [c['name'] for c in inspector.get_columns(table)]
    return column in columns


def upgrade() -> None:
    conn = op.get_bind()

    # === generation_tasks 表 ===

    # 1. 添加新字段
    if not _column_exists(conn, 'generation_tasks', 'source_dataset_id'):
        op.add_column(
            'generation_tasks',
            sa.Column('source_dataset_id', sa.String(36), nullable=True)
        )
        op.create_index(
            'idx_gen_task_source_dataset',
            'generation_tasks',
            ['source_dataset_id'],
        )

    # 2. 将旧字段数据迁移到新字段（source_qa_dataset_id → source_dataset_id）
    if _column_exists(conn, 'generation_tasks', 'source_qa_dataset_id'):
        conn.execute(sa.text(
            "UPDATE generation_tasks SET source_dataset_id = source_qa_dataset_id "
            "WHERE source_qa_dataset_id IS NOT NULL AND source_dataset_id IS NULL"
        ))

    # 3. 删除旧字段
    if _column_exists(conn, 'generation_tasks', 'source_qa_dataset_id'):
        try:
            op.drop_index('idx_gen_task_qa_dataset', table_name='generation_tasks')
        except Exception:
            pass
        op.drop_column('generation_tasks', 'source_qa_dataset_id')

    # === datasets 表 ===

    if not _column_exists(conn, 'datasets', 'source_dataset_id'):
        op.add_column(
            'datasets',
            sa.Column('source_dataset_id', sa.String(36), nullable=True)
        )
        op.create_index(
            'idx_dataset_source_dataset',
            'datasets',
            ['source_dataset_id'],
        )

    # 回填：从 extra_metadata 中的 source_task_id 关联 generation_tasks 的 source_dataset_id
    # 只对 source_type='generated' 且 source_dataset_id 为空的记录操作
    conn.execute(sa.text(
        "UPDATE datasets d "
        "INNER JOIN generation_tasks g ON JSON_UNQUOTE(JSON_EXTRACT(d.extra_metadata, '$.source_task_id')) = g.task_id "
        "SET d.source_dataset_id = g.source_dataset_id "
        "WHERE d.source_type = 'generated' AND d.source_dataset_id IS NULL AND g.source_dataset_id IS NOT NULL"
    ))


def downgrade() -> None:
    conn = op.get_bind()

    # === datasets 表 ===

    if _column_exists(conn, 'datasets', 'source_dataset_id'):
        try:
            op.drop_index('idx_dataset_source_dataset', table_name='datasets')
        except Exception:
            pass
        op.drop_column('datasets', 'source_dataset_id')

    # === generation_tasks 表 ===

    if not _column_exists(conn, 'generation_tasks', 'source_qa_dataset_id'):
        op.add_column(
            'generation_tasks',
            sa.Column('source_qa_dataset_id', sa.String(36), nullable=True)
        )
        op.create_index(
            'idx_gen_task_qa_dataset',
            'generation_tasks',
            ['source_qa_dataset_id'],
        )

    if _column_exists(conn, 'generation_tasks', 'source_dataset_id'):
        conn.execute(sa.text(
            "UPDATE generation_tasks SET source_qa_dataset_id = source_dataset_id "
            "WHERE source_dataset_id IS NOT NULL AND source_qa_dataset_id IS NULL "
            "AND generation_mode = 'qa_to_training'"
        ))
        try:
            op.drop_index('idx_gen_task_source_dataset', table_name='generation_tasks')
        except Exception:
            pass
        op.drop_column('generation_tasks', 'source_dataset_id')
