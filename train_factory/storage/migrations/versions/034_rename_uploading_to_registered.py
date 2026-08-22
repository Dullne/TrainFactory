"""Rename dataset status 'uploading' to 'registered'.

The 'uploading' status was misnamed — there was no actual file upload
functionality. It represented 'registered but not yet ready', which
aligns with the model registry's 'registered' status. This migration
renames existing records and the new 'uploading' status will be used
for actual file uploads going forward.

Revision ID: 034_rename_uploading_to_registered
Revises: 033_add_generation_excluded
"""

from alembic import op


revision = "034_rename_uploading_to_registered"
down_revision = "033_add_generation_excluded"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("UPDATE datasets SET status = 'registered' WHERE status = 'uploading'")


def downgrade():
    op.execute("UPDATE datasets SET status = 'uploading' WHERE status = 'registered'")
