"""Add external_api_config_id column to deployments.

Revision ID: 035_add_deployment_external_api_config_id
Revises: 034_rename_uploading_to_registered
"""

from alembic import op
import sqlalchemy as sa


revision = "035_add_deployment_external_api_config_id"
down_revision = "034_rename_uploading_to_registered"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    column_names = {
        column["name"] for column in inspector.get_columns("deployments")
    }
    index_names = {
        index["name"] for index in inspector.get_indexes("deployments")
    }

    if "external_api_config_id" not in column_names:
        op.add_column(
            "deployments",
            sa.Column(
                "external_api_config_id",
                sa.String(length=36),
                nullable=True,
            ),
        )
    if "idx_deployment_external_api_config_id" not in index_names:
        op.create_index(
            "idx_deployment_external_api_config_id",
            "deployments",
            ["external_api_config_id"],
            unique=False,
        )
    if "idx_deployment_external_api_status" not in index_names:
        op.create_index(
            "idx_deployment_external_api_status",
            "deployments",
            ["external_api_config_id", "status"],
            unique=False,
        )

    dialect = bind.dialect.name
    if dialect == "mysql":
        op.execute(
            """
            UPDATE deployments
            SET external_api_config_id = NULLIF(
                TRIM(JSON_UNQUOTE(JSON_EXTRACT(config, '$.external_api_config_id'))),
                ''
            )
            WHERE external_api_config_id IS NULL
            """
        )
    else:
        # Best-effort backfill for SQLite-compatible environments used in tests.
        op.execute(
            """
            UPDATE deployments
            SET external_api_config_id = NULLIF(TRIM(json_extract(config, '$.external_api_config_id')), '')
            WHERE external_api_config_id IS NULL
            """
        )


def downgrade():
    op.drop_index("idx_deployment_external_api_status", table_name="deployments")
    op.drop_index("idx_deployment_external_api_config_id", table_name="deployments")
    op.drop_column("deployments", "external_api_config_id")
