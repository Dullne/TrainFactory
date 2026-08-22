"""Add path hash columns with global uniqueness

Revision ID: 006_add_path_hashes
Revises: 006_rename_deploy_mode_shared
Create Date: 2026-01-29
"""
from typing import Sequence, Union, Any, Dict
import hashlib
import json
import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text


# revision identifiers, used by Alembic.
revision: str = '006_add_path_hashes'
down_revision: Union[str, None] = '006_rename_deploy_mode_shared'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def unique_constraint_exists(table_name: str, constraint_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    constraints = inspector.get_unique_constraints(table_name)
    return any(c.get('name') == constraint_name for c in constraints)


def _normalize_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def _hash_with_salt(path: str, salt: Union[str, None]) -> str:
    normalized = _normalize_path(path)
    if salt:
        normalized = f"{normalized}::{salt}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _fill_hashes_generic() -> None:
    conn = op.get_bind()
    rows = conn.execute(
        text("SELECT id, model_id, model_name, model_path, source_type, extra_metadata FROM model_registry")
    ).fetchall()
    path_counts = {}
    path_seen = {}
    for row in rows:
        model_path = row[3] or ""
        normalized = _normalize_path(model_path)
        path_counts[normalized] = path_counts.get(normalized, 0) + 1

    for row in rows:
        row_id = row[0]
        model_id = row[1]
        model_name = row[2]
        model_path = row[3] or ""
        source_type = row[4] or ""
        extra_metadata = row[5]
        normalized = _normalize_path(model_path)

        salt = None
        seen = path_seen.get(normalized, 0)
        if source_type == "external_bind":
            meta = _parse_metadata(extra_metadata)
            bound_uid = meta.get("bound_model_uid")
            if bound_uid:
                salt = str(bound_uid)
            elif model_name:
                salt = str(model_name)
            else:
                salt = str(model_id or row_id)
        elif path_counts.get(normalized, 0) > 1:
            if seen > 0:
                salt = str(model_id or row_id)

        path_seen[normalized] = seen + 1

        digest = _hash_with_salt(model_path, salt)
        conn.execute(
            text("UPDATE model_registry SET model_path_hash = :digest WHERE id = :id"),
            {"digest": digest, "id": row_id},
        )

    rows = conn.execute(text("SELECT id, dataset_id, storage_path FROM datasets")).fetchall()
    dataset_counts = {}
    dataset_seen = {}
    for row in rows:
        storage_path = row[2] or ""
        normalized = _normalize_path(storage_path)
        dataset_counts[normalized] = dataset_counts.get(normalized, 0) + 1
    for row in rows:
        row_id = row[0]
        dataset_id = row[1]
        storage_path = row[2] or ""
        normalized = _normalize_path(storage_path)
        seen = dataset_seen.get(normalized, 0)
        salt = None
        if dataset_counts.get(normalized, 0) > 1 and seen > 0:
            salt = str(dataset_id or row_id)
        dataset_seen[normalized] = seen + 1
        digest = _hash_with_salt(storage_path, salt)
        conn.execute(
            text("UPDATE datasets SET storage_path_hash = :digest WHERE id = :id"),
            {"digest": digest, "id": row_id},
        )


def upgrade() -> None:
    if not column_exists('model_registry', 'model_path_hash'):
        op.add_column('model_registry', sa.Column('model_path_hash', sa.String(length=64), nullable=True))
    if not column_exists('datasets', 'storage_path_hash'):
        op.add_column('datasets', sa.Column('storage_path_hash', sa.String(length=64), nullable=True))

    _fill_hashes_generic()

    # Make columns non-null after backfill (MySQL requires existing_type)
    op.alter_column(
        'model_registry',
        'model_path_hash',
        existing_type=sa.String(length=64),
        nullable=False,
    )
    op.alter_column(
        'datasets',
        'storage_path_hash',
        existing_type=sa.String(length=64),
        nullable=False,
    )

    if not unique_constraint_exists('model_registry', 'uq_model_path_hash'):
        op.create_unique_constraint('uq_model_path_hash', 'model_registry', ['model_path_hash'])
    if not unique_constraint_exists('datasets', 'uq_dataset_storage_path_hash'):
        op.create_unique_constraint('uq_dataset_storage_path_hash', 'datasets', ['storage_path_hash'])


def downgrade() -> None:
    if unique_constraint_exists('model_registry', 'uq_model_path_hash'):
        op.drop_constraint('uq_model_path_hash', 'model_registry', type_='unique')
    if unique_constraint_exists('datasets', 'uq_dataset_storage_path_hash'):
        op.drop_constraint('uq_dataset_storage_path_hash', 'datasets', type_='unique')

    if column_exists('model_registry', 'model_path_hash'):
        op.drop_column('model_registry', 'model_path_hash')
    if column_exists('datasets', 'storage_path_hash'):
        op.drop_column('datasets', 'storage_path_hash')
