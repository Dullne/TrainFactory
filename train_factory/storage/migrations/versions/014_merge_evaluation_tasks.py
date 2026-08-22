"""
Merge deep_evaluation_tasks into evaluation_tasks.

Revision ID: 014_merge_evaluation_tasks
Revises: 013_merge_heads
Create Date: 2025-02-05
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = '014_merge_evaluation_tasks'
down_revision = '013_merge_heads'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Merge deep_evaluation_tasks into evaluation_tasks."""
    conn = op.get_bind()

    # 1. Add new columns to evaluation_tasks if they don't exist
    inspector = sa.inspect(conn)
    existing_columns = [col['name'] for col in inspector.get_columns('evaluation_tasks')]

    new_columns = [
        ('eval_framework', "VARCHAR(32) DEFAULT 'mteb'"),
        ('field_mapping', 'JSON'),
        ('metrics', 'JSON'),
        ('llm_config', 'JSON'),
        ('total_samples', 'INT DEFAULT 0'),
        ('processed_samples', 'INT DEFAULT 0'),
    ]

    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            op.execute(text(f'ALTER TABLE evaluation_tasks ADD COLUMN {col_name} {col_type}'))

    # Rename report_path to results_path if needed
    if 'report_path' in existing_columns and 'results_path' not in existing_columns:
        op.execute(text('ALTER TABLE evaluation_tasks CHANGE COLUMN report_path results_path VARCHAR(1024)'))
    elif 'results_path' not in existing_columns:
        op.execute(text('ALTER TABLE evaluation_tasks ADD COLUMN results_path VARCHAR(1024)'))

    # 2. Set eval_framework = 'mteb' for existing records
    op.execute(text("UPDATE evaluation_tasks SET eval_framework = 'mteb' WHERE eval_framework IS NULL"))

    # 3. Create index on eval_framework
    try:
        op.execute(text('CREATE INDEX idx_eval_framework ON evaluation_tasks (eval_framework)'))
    except Exception:
        pass  # Index may already exist

    # 4. Check if deep_evaluation_tasks exists and migrate data
    tables = inspector.get_table_names()
    if 'deep_evaluation_tasks' in tables:
        # Get count of records to migrate
        result = conn.execute(text('SELECT COUNT(*) FROM deep_evaluation_tasks'))
        count = result.scalar()

        if count > 0:
            # Migrate data from deep_evaluation_tasks to evaluation_tasks
            # Convert single model/dataset config to list format
            op.execute(text("""
                INSERT INTO evaluation_tasks (
                    task_id, task_name, description, eval_framework, eval_type,
                    model_configs, dataset_configs, field_mapping, metrics,
                    max_samples, workers, status, progress, error_message,
                    total_samples, processed_samples, results, results_path,
                    user_id, created_at, started_at, completed_at
                )
                SELECT
                    task_id, task_name, description, 'deepeval', eval_type,
                    CASE
                        WHEN model_config_data IS NOT NULL
                        THEN JSON_ARRAY(model_config_data)
                        ELSE NULL
                    END,
                    CASE
                        WHEN dataset_id IS NOT NULL
                        THEN JSON_ARRAY(JSON_OBJECT('dataset_id', dataset_id))
                        ELSE NULL
                    END,
                    field_mapping, metrics,
                    sample_size, concurrency, status, progress, error_message,
                    total_samples, processed_samples, results_summary, results_path,
                    user_id, created_at, started_at, completed_at
                FROM deep_evaluation_tasks
            """))

        # 5. Drop deep_evaluation_tasks table
        op.execute(text('DROP TABLE deep_evaluation_tasks'))


def downgrade() -> None:
    """Recreate deep_evaluation_tasks and restore data."""
    op.get_bind()

    # 1. Recreate deep_evaluation_tasks table
    op.execute(text("""
        CREATE TABLE deep_evaluation_tasks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            task_id VARCHAR(255) UNIQUE NOT NULL,
            task_name VARCHAR(255),
            description VARCHAR(255),
            eval_type VARCHAR(32) DEFAULT 'rerank',
            dataset_id VARCHAR(36),
            sample_size INT,
            field_mapping JSON,
            model_config_data JSON,
            metrics JSON,
            concurrency INT DEFAULT 5,
            status VARCHAR(50) DEFAULT 'pending',
            progress FLOAT DEFAULT 0.0,
            total_samples INT DEFAULT 0,
            processed_samples INT DEFAULT 0,
            results_summary JSON,
            results_path VARCHAR(1024),
            error_message VARCHAR(255),
            user_id VARCHAR(64),
            created_at DATETIME,
            started_at DATETIME,
            completed_at DATETIME,
            source_type VARCHAR(32),
            split VARCHAR(32),
            llm_config JSON,
            INDEX idx_deep_eval_user_created (user_id, created_at),
            INDEX idx_deep_eval_status (status),
            INDEX idx_deep_eval_task_id (task_id)
        )
    """))

    # 2. Migrate deepeval data back
    op.execute(text("""
        INSERT INTO deep_evaluation_tasks (
            task_id, task_name, description, eval_type,
            dataset_id, sample_size, field_mapping, model_config_data,
            metrics, concurrency, status, progress,
            total_samples, processed_samples, results_summary, results_path,
            error_message, user_id, created_at, started_at, completed_at
        )
        SELECT
            task_id, task_name, description, eval_type,
            JSON_UNQUOTE(JSON_EXTRACT(dataset_configs, '$[0].dataset_id')),
            max_samples, field_mapping,
            JSON_EXTRACT(model_configs, '$[0]'),
            metrics, workers, status, progress,
            total_samples, processed_samples, results, results_path,
            error_message, user_id, created_at, started_at, completed_at
        FROM evaluation_tasks
        WHERE eval_framework = 'deepeval'
    """))

    # 3. Delete deepeval records from evaluation_tasks
    op.execute(text("DELETE FROM evaluation_tasks WHERE eval_framework = 'deepeval'"))

    # 4. Remove new columns (optional, keep for compatibility)
    # op.execute(text('ALTER TABLE evaluation_tasks DROP COLUMN eval_framework'))
