"""
Dataset service for managing dataset metadata.

Provides CRUD operations and business logic for dataset management.
"""

import json
import logging
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from ...utils.path_utils import hash_path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime
from train_factory.core.time_utils import now_naive

from sqlmodel import Session, select, func
from sqlalchemy import case, or_, text, update

from ..database import get_engine
from ..entities.dataset_entity import DatasetDB
from ..entities.dataset_asset_entity import DatasetAssetDB
from ...enums.dataset_status import DatasetStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetUpdateOutcome:
    """Exactly one public result of a dataset metadata CAS update."""

    dataset: Optional[Dict[str, Any]] = None
    missing: bool = False
    ownership_mismatch: bool = False
    conflict: bool = False

    def __post_init__(self) -> None:
        result_count = sum(
            (
                self.dataset is not None,
                self.missing,
                self.ownership_mismatch,
                self.conflict,
            )
        )
        if result_count != 1:
            raise ValueError(
                "DatasetUpdateOutcome requires exactly one result"
            )


class DatasetConsumptionUnavailableError(ValueError):
    """Base error for a managed dataset that cannot accept new consumers."""


class DatasetDeletionInProgressError(DatasetConsumptionUnavailableError):
    """Raised when a new task tries to consume a deleting dataset."""


class DatasetReferenceUnavailableError(DatasetConsumptionUnavailableError):
    """Raised when a previously validated managed dataset has disappeared."""


class DatasetDeletionOwnerConflictError(ValueError):
    """Raised when another logical operation owns a dataset deletion fence."""


_MAX_DELETION_OWNER_LENGTH = 128


def require_dataset_deletion_owner(deletion_owner: Optional[str]) -> str:
    """Return a normalized non-empty deletion owner token."""
    if not isinstance(deletion_owner, str) or not deletion_owner.strip():
        raise ValueError("deletion_owner must be a non-empty string")
    normalized_owner = deletion_owner.strip()
    if len(normalized_owner) > _MAX_DELETION_OWNER_LENGTH:
        raise ValueError(
            f"deletion_owner must be at most {_MAX_DELETION_OWNER_LENGTH} characters"
        )
    return normalized_owner


def _assert_dataset_deletion_owner(
    dataset: DatasetDB,
    deletion_owner: str,
) -> None:
    if dataset.deletion_owner != deletion_owner:
        raise DatasetDeletionOwnerConflictError(
            "Dataset is owned by another deletion operation"
        )


def _normalize_s3_uri(storage_uri: Optional[str]) -> Optional[str]:
    """Return canonical ``s3://bucket/key`` URI or None."""
    if not storage_uri:
        return None
    from ..object_store import parse_s3_uri

    try:
        bucket, key = parse_s3_uri(storage_uri.strip())
    except ValueError:
        logger.warning(f"Invalid S3 URI: {storage_uri!r}")
        return None
    return f"s3://{bucket}/{key}"


def _hash_storage_ref(
    storage_backend: str,
    storage_path: Optional[str],
    storage_uri: Optional[str],
) -> Optional[str]:
    """Build a stable hash for local paths or S3 URIs."""
    if storage_backend == "s3":
        normalized_uri = _normalize_s3_uri(storage_uri)
        if not normalized_uri:
            return None
        return hashlib.sha256(normalized_uri.encode("utf-8")).hexdigest()

    normalized_path = (storage_path or "").strip() or None
    if not normalized_path:
        return None
    return hash_path(normalized_path)


def _storage_reference_identity(storage_ref: Any) -> Optional[tuple[str, str]]:
    """Return a mutation-target identity without touching the resource."""
    if not isinstance(storage_ref, str) or not storage_ref.strip():
        return None
    normalized = storage_ref.strip()
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() == "s3" and parsed.netloc:
        return "s3", f"{parsed.netloc}/{parsed.path.lstrip('/')}"
    try:
        local_path = str(Path(normalized).resolve())
    except (OSError, RuntimeError, ValueError):
        local_path = os.path.abspath(normalized)
    return "local", os.path.normcase(local_path)


def _storage_reference_identities_overlap(
    left: Optional[tuple[str, str]],
    right: Optional[tuple[str, str]],
) -> bool:
    """Return whether two identities can name overlapping mutation targets."""
    if left is None or right is None or left[0] != right[0]:
        return False
    if left[0] == "s3":
        # S3 keys are objects, not filesystem directories. Prefix matches
        # would incorrectly conflate distinct objects.
        return left[1] == right[1]
    try:
        common_path = os.path.commonpath((left[1], right[1]))
    except ValueError:
        return False
    return common_path in {left[1], right[1]}


def storage_references_overlap(left: Any, right: Any) -> bool:
    """Return whether deleting either local reference can affect the other."""
    return _storage_reference_identities_overlap(
        _storage_reference_identity(left),
        _storage_reference_identity(right),
    )


def storage_reference_sets_overlap(
    left_references: Iterable[str],
    right_references: Iterable[str],
) -> bool:
    """Return whether any references overlap under destructive semantics."""
    left_identities = {
        identity
        for reference in left_references
        if (identity := _storage_reference_identity(reference)) is not None
    }
    right_identities = {
        identity
        for reference in right_references
        if (identity := _storage_reference_identity(reference)) is not None
    }
    return any(
        _storage_reference_identities_overlap(left, right)
        for left in left_identities
        for right in right_identities
    )


def lock_datasets_for_consumption(
    session: Session,
    *,
    dataset_ids: Iterable[str] = (),
    storage_refs: Iterable[str] = (),
    require_all_dataset_ids: bool = False,
    require_all_storage_refs: bool = False,
) -> List[DatasetDB]:
    """Lock matching dataset rows and reject durable deletion intents.

    Task creation and resume call this in the same transaction as their write.
    Paths that do not belong to a registered dataset remain supported for
    authentication-disabled and legacy workflows.
    """
    normalized_ids = {
        value.strip()
        for value in dataset_ids
        if isinstance(value, str) and value.strip()
    }
    normalized_refs = {
        value.strip()
        for value in storage_refs
        if isinstance(value, str) and value.strip()
    }

    storage_hashes = set()
    canonical_uris = set()
    hashes_by_ref: Dict[str, set[str]] = {}
    uris_by_ref: Dict[str, set[str]] = {}
    for storage_ref in normalized_refs:
        ref_hashes = set()
        ref_uris = {storage_ref}
        if storage_ref.startswith("s3://"):
            canonical_uri = _normalize_s3_uri(storage_ref)
            if canonical_uri:
                canonical_uris.add(canonical_uri)
                ref_uris.add(canonical_uri)
                new_hash = _hash_storage_ref("s3", None, canonical_uri)
                if new_hash:
                    storage_hashes.add(new_hash)
                    ref_hashes.add(new_hash)
                legacy_hash = hash_path(canonical_uri)
                storage_hashes.add(legacy_hash)
                ref_hashes.add(legacy_hash)
            raw_hash = hash_path(storage_ref)
            storage_hashes.add(raw_hash)
            ref_hashes.add(raw_hash)
        else:
            storage_hash = _hash_storage_ref("local", storage_ref, None)
            if storage_hash:
                storage_hashes.add(storage_hash)
                ref_hashes.add(storage_hash)
        hashes_by_ref[storage_ref] = ref_hashes
        uris_by_ref[storage_ref] = ref_uris

    conditions = []
    if normalized_ids:
        conditions.append(DatasetDB.dataset_id.in_(normalized_ids))
    if normalized_refs:
        conditions.extend(
            (
                DatasetDB.storage_path.in_(normalized_refs),
                DatasetDB.storage_uri.in_(normalized_refs | canonical_uris),
            )
        )
    if storage_hashes:
        conditions.append(DatasetDB.storage_path_hash.in_(storage_hashes))
    if not conditions:
        return []

    datasets = list(
        session.exec(
            select(DatasetDB).where(or_(*conditions)).with_for_update()
        ).all()
    )
    if require_all_dataset_ids:
        found_ids = {dataset.dataset_id for dataset in datasets}
        missing_ids = normalized_ids - found_ids
        if missing_ids:
            raise DatasetReferenceUnavailableError(
                "Managed dataset no longer exists: " + ", ".join(sorted(missing_ids))
            )
    if require_all_storage_refs:
        missing_refs = []
        for storage_ref in sorted(normalized_refs):
            matched = any(
                dataset.storage_path == storage_ref
                or dataset.storage_uri in uris_by_ref[storage_ref]
                or dataset.storage_path_hash in hashes_by_ref[storage_ref]
                for dataset in datasets
            )
            if not matched:
                missing_refs.append(storage_ref)
        if missing_refs:
            raise DatasetReferenceUnavailableError(
                "Managed dataset reference no longer exists"
            )
    deleting_ids = sorted(
        dataset.dataset_id
        for dataset in datasets
        if dataset.status == DatasetStatus.DELETING.value
    )
    if deleting_ids:
        raise DatasetDeletionInProgressError(
            "Dataset deletion is in progress: " + ", ".join(deleting_ids)
        )
    staging_ids = sorted(
        dataset.dataset_id
        for dataset in datasets
        if dataset.status == DatasetStatus.STAGING.value
    )
    if staging_ids:
        raise DatasetConsumptionUnavailableError(
            "Dataset publication is not complete: "
            + ", ".join(staging_ids)
        )
    return datasets


class DatasetService:
    """Service for managing datasets."""

    _LIST_RESPONSE_KEYS = [
        "dataset_id",
        "dataset_name",
        "display_name",
        "description",
        "dataset_type",
        "usage",
        "model_type",
        "source_type",
        "source_path",
        "remote_repo",
        "hf_subset",
        "storage_path",
        "storage_backend",
        "storage_uri",
        "version",
        "file_format",
        "columns",
        "sample_data",
        "content_schema",
        "num_rows",
        "num_train",
        "num_eval",
        "num_test",
        "file_size",
        "source_task_type",
        "source_task_id",
        "tags",
        "extra_metadata",
        "status",
        "error_message",
        "user_id",
        "created_at",
        "updated_at",
    ]

    _LIST_SELECT_FIELDS = (
        DatasetDB.dataset_id,
        DatasetDB.dataset_name,
        DatasetDB.display_name,
        DatasetDB.description,
        DatasetDB.dataset_type,
        DatasetDB.usage,
        DatasetDB.model_type,
        DatasetDB.source_type,
        DatasetDB.source_path,
        DatasetDB.remote_repo,
        DatasetDB.hf_subset,
        DatasetDB.storage_path,
        DatasetDB.storage_backend,
        DatasetDB.storage_uri,
        DatasetDB.version,
        DatasetDB.file_format,
        DatasetDB.num_rows,
        DatasetDB.num_train,
        DatasetDB.num_eval,
        DatasetDB.num_test,
        DatasetDB.file_size,
        DatasetDB.source_task_type,
        DatasetDB.source_task_id,
        DatasetDB.tags,
        DatasetDB.status,
        DatasetDB.error_message,
        DatasetDB.user_id,
        DatasetDB.created_at,
        DatasetDB.updated_at,
    )

    def __init__(self):
        self.engine = None

    def _get_engine(self):
        """Get database engine lazily."""
        if self.engine is None:
            self.engine = get_engine()
        return self.engine

    def list_external_storage_reference_consumers(
        self,
        storage_references: Iterable[str],
        *,
        exclude_dataset_ids: Iterable[str] = (),
    ) -> List[Dict[str, str]]:
        """Find all-tenant dataset/asset rows sharing deletion targets.

        This is an internal destructive-operation preflight. It deliberately
        bypasses tenant-filtered public dataset listings.
        """
        target_identities = {
            identity
            for reference in storage_references
            if (identity := _storage_reference_identity(reference)) is not None
        }
        if not target_identities:
            return []
        excluded_ids = {
            dataset_id.strip()
            for dataset_id in exclude_dataset_ids
            if isinstance(dataset_id, str) and dataset_id.strip()
        }

        with Session(self._get_engine()) as session:
            dataset_stmt = select(
                DatasetDB.dataset_id,
                DatasetDB.storage_path,
                DatasetDB.storage_uri,
            )
            asset_stmt = select(
                DatasetAssetDB.asset_id,
                DatasetAssetDB.dataset_id,
                DatasetAssetDB.storage_uri,
            )
            if excluded_ids:
                dataset_stmt = dataset_stmt.where(
                    ~DatasetDB.dataset_id.in_(excluded_ids)
                )
                asset_stmt = asset_stmt.where(
                    ~DatasetAssetDB.dataset_id.in_(excluded_ids)
                )
            dataset_rows = session.exec(dataset_stmt).all()
            asset_rows = session.exec(asset_stmt).all()

        consumers: List[Dict[str, str]] = []
        for dataset_id, storage_path, storage_uri in dataset_rows:
            for reference in (storage_path, storage_uri):
                reference_identity = _storage_reference_identity(reference)
                if any(
                    _storage_reference_identities_overlap(
                        reference_identity,
                        target_identity,
                    )
                    for target_identity in target_identities
                ):
                    consumers.append(
                        {
                            "source": "dataset",
                            "dataset_id": str(dataset_id),
                            "reference": str(reference),
                        }
                    )
        for asset_id, dataset_id, storage_uri in asset_rows:
            storage_identity = _storage_reference_identity(storage_uri)
            if any(
                _storage_reference_identities_overlap(
                    storage_identity,
                    target_identity,
                )
                for target_identity in target_identities
            ):
                consumers.append(
                    {
                        "source": "asset",
                        "asset_id": str(asset_id),
                        "dataset_id": str(dataset_id),
                        "reference": str(storage_uri),
                    }
                )
        return sorted(
            consumers,
            key=lambda item: (
                item["source"],
                item["dataset_id"],
                item.get("asset_id", ""),
                item["reference"],
            ),
        )

    @classmethod
    def _row_to_list_dict(cls, row: Any) -> Dict[str, Any]:
        """Convert a lightweight select row into the DatasetResponse-compatible shape."""
        data = {key: None for key in cls._LIST_RESPONSE_KEYS}

        mapping = getattr(row, "_mapping", None)
        if mapping is None and isinstance(row, dict):
            mapping = row
        if mapping is None:
            raise TypeError(f"Unsupported dataset row type: {type(row)!r}")

        for key in mapping.keys():
            data[key] = mapping[key]

        for key in ("created_at", "updated_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()

        return data

    def create_dataset(
        self,
        dataset_name: str,
        storage_path: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_type: str = "custom",
        usage: str = "train",
        model_type: Optional[List[str]] = None,
        source_type: str = "uploaded",
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        source_path: Optional[str] = None,
        source_dataset_id: Optional[str] = None,
        remote_repo: Optional[str] = None,
        hf_subset: Optional[str] = None,
        file_format: str = "parquet",
        columns: Optional[List[Dict[str, Any]]] = None,
        sample_data: Optional[List[Dict[str, Any]]] = None,
        num_rows: Optional[int] = None,
        num_train: Optional[int] = None,
        num_eval: Optional[int] = None,
        num_test: Optional[int] = None,
        file_size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        # Storage refactor fields
        storage_backend: str = "local",
        storage_uri: Optional[str] = None,
        source_task_type: Optional[str] = None,
        source_task_id: Optional[str] = None,
        content_schema: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new dataset record."""
        # Auto-infer content_schema from dataset_type if not provided
        if content_schema is None:
            from .dataset_schema import infer_schema
            inferred = infer_schema(dataset_type)
            if inferred.get("fields"):
                content_schema = inferred

        # Local mode stores filesystem path; S3 mode stores URI only.
        effective_storage_path = (storage_path or "").strip() or None
        effective_storage_uri = (storage_uri or "").strip() or None
        if storage_backend == "s3":
            effective_storage_path = None
            effective_storage_uri = _normalize_s3_uri(effective_storage_uri)

        dataset = DatasetDB(
            dataset_id=dataset_id or str(uuid4()),
            dataset_name=dataset_name,
            display_name=display_name or dataset_name,
            description=description,
            dataset_type=dataset_type,
            usage=usage,
            model_type=model_type,
            source_type=source_type,
            source_path=source_path,
            source_dataset_id=source_dataset_id,
            remote_repo=remote_repo,
            hf_subset=hf_subset,
            storage_path=effective_storage_path,
            storage_path_hash=_hash_storage_ref(
                storage_backend=storage_backend,
                storage_path=effective_storage_path,
                storage_uri=effective_storage_uri,
            ),
            file_format=file_format,
            storage_backend=storage_backend,
            storage_uri=effective_storage_uri,
            content_schema=content_schema,
            source_task_type=source_task_type,
            source_task_id=source_task_id,
            columns=columns,
            sample_data=sample_data,
            num_rows=num_rows,
            num_train=num_train,
            num_eval=num_eval,
            num_test=num_test,
            file_size=file_size,
            tags=tags,
            extra_metadata=extra_metadata,
            status=status or ("ready" if num_rows else "registered"),
            user_id=user_id,
        )

        with Session(self._get_engine()) as session:
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            logger.info(f"Created dataset: {dataset.dataset_id}")
            return dataset.to_dict()

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get a publicly visible dataset by ID."""
        with Session(self._get_engine()) as session:
            statement = select(DatasetDB).where(
                DatasetDB.dataset_id == dataset_id,
                DatasetDB.status != DatasetStatus.STAGING.value,
            )
            dataset = session.exec(statement).first()
            return dataset.to_dict() if dataset else None

    def get_dataset_internal(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Read a dataset including non-public generation staging rows."""
        with Session(self._get_engine()) as session:
            dataset = session.exec(
                select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
            ).first()
            if dataset is None:
                return None
            result = dataset.to_dict()
            result["generation_run_token"] = dataset.generation_run_token
            result["deletion_owner"] = dataset.deletion_owner
            return result

    def get_dataset_by_name(
        self, dataset_name: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get dataset by name."""
        with Session(self._get_engine()) as session:
            statement = select(DatasetDB).where(
                DatasetDB.dataset_name == dataset_name,
                DatasetDB.status != DatasetStatus.STAGING.value,
            )
            if user_id:
                statement = statement.where(DatasetDB.user_id == user_id)
            dataset = session.exec(statement).first()
            return dataset.to_dict() if dataset else None

    def get_dataset_by_storage_path(
        self, storage_path: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get dataset by storage reference.

        Accepts local path or ``s3://bucket/key`` URI for backward compatibility.
        """
        if not storage_path:
            return None

        is_s3 = storage_path.startswith("s3://")
        with Session(self._get_engine()) as session:
            if is_s3:
                normalized_uri = _normalize_s3_uri(storage_path) or storage_path.strip()
                candidate_hashes = set()

                # New hash strategy for S3 storage references.
                new_hash = _hash_storage_ref(
                    storage_backend="s3",
                    storage_path=None,
                    storage_uri=normalized_uri,
                )
                if new_hash:
                    candidate_hashes.add(new_hash)

                # Legacy compatibility: some historical rows hashed S3 URI via hash_path.
                if normalized_uri:
                    candidate_hashes.add(hash_path(normalized_uri))
                raw_uri = storage_path.strip()
                if raw_uri and raw_uri != normalized_uri:
                    candidate_hashes.add(hash_path(raw_uri))

                statement = select(DatasetDB).where(
                    (DatasetDB.storage_path_hash.in_(list(candidate_hashes)))
                    | (DatasetDB.storage_uri == normalized_uri)
                    | (DatasetDB.storage_path == normalized_uri)
                )
            else:
                storage_hash = _hash_storage_ref(
                    storage_backend="local",
                    storage_path=storage_path,
                    storage_uri=None,
                )
                if not storage_hash:
                    return None
                statement = select(DatasetDB).where(
                    DatasetDB.storage_path_hash == storage_hash
                )

            if user_id:
                statement = statement.where(DatasetDB.user_id == user_id)
            statement = statement.where(
                DatasetDB.status != DatasetStatus.STAGING.value
            )
            dataset = session.exec(statement).first()
            return dataset.to_dict() if dataset else None

    def list_datasets(
        self,
        dataset_type: Optional[str] = None,
        usage: Optional[str] = None,
        model_type: Optional[str] = None,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
        source_task_type: Optional[str] = None,
        source_task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List datasets with optional filters.

        Returns:
            Tuple of (datasets, total_count)
        """
        with Session(self._get_engine()) as session:
            # Build base filter conditions
            conditions = []
            if dataset_type:
                conditions.append(DatasetDB.dataset_type == dataset_type)
            if usage:
                conditions.append(DatasetDB.usage == usage)
            if model_type:
                model_type_json = json.dumps(model_type)
                conditions.append(
                    text("JSON_CONTAINS(model_type, :model_type_json)").bindparams(
                        model_type_json=model_type_json
                    )
                )
            if source_type:
                conditions.append(DatasetDB.source_type == source_type)
            if status:
                conditions.append(DatasetDB.status == status)
            if source_task_type:
                conditions.append(DatasetDB.source_task_type == source_task_type)
            if source_task_id:
                conditions.append(DatasetDB.source_task_id == source_task_id)
            if user_id:
                conditions.append(DatasetDB.user_id == user_id)

            conditions.append(DatasetDB.status != DatasetStatus.STAGING.value)

            # Get total count
            count_stmt = select(func.count()).select_from(DatasetDB)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Get paginated data
            statement = select(*self._LIST_SELECT_FIELDS)
            for cond in conditions:
                statement = statement.where(cond)
            statement = statement.order_by(
                DatasetDB.created_at.desc(),
                DatasetDB.dataset_id.asc(),
            )
            statement = statement.offset(offset).limit(limit)

            rows = session.exec(statement).all()
            return [self._row_to_list_dict(row) for row in rows], total

    def search_datasets(
        self,
        query: str,
        dataset_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search datasets by name or description."""
        with Session(self._get_engine()) as session:
            statement = select(*self._LIST_SELECT_FIELDS).where(
                (DatasetDB.dataset_name.contains(query))
                | (DatasetDB.description.contains(query))
                | (DatasetDB.display_name.contains(query))
            ).where(DatasetDB.status != DatasetStatus.STAGING.value)

            if dataset_type:
                statement = statement.where(DatasetDB.dataset_type == dataset_type)
            if user_id:
                statement = statement.where(DatasetDB.user_id == user_id)

            statement = statement.limit(limit)
            rows = session.exec(statement).all()
            return [self._row_to_list_dict(row) for row in rows]

    def update_dataset(
        self,
        dataset_id: str,
        *,
        expected_user_id: Optional[str],
        expected_status: str,
        expected_deletion_owner: Optional[str],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        dataset_type: Optional[str] = None,
        usage: Optional[str] = None,
        model_type: Optional[List[str]] = None,
        columns: Optional[List[Dict[str, Any]]] = None,
        sample_data: Optional[List[Dict[str, Any]]] = None,
        num_rows: Optional[int] = None,
        num_train: Optional[int] = None,
        num_eval: Optional[int] = None,
        num_test: Optional[int] = None,
        file_size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> DatasetUpdateOutcome:
        """Update dataset metadata only while its authorization snapshot matches."""
        if status is not None:
            status_validator = DatasetDB(
                dataset_name="__status_validation__",
                status=expected_status,
            )
            status_validator.update_status(status)

        optional_values = {
            "display_name": display_name,
            "description": description,
            "dataset_type": dataset_type,
            "usage": usage,
            "model_type": model_type,
            "columns": columns,
            "sample_data": sample_data,
            "num_rows": num_rows,
            "num_train": num_train,
            "num_eval": num_eval,
            "num_test": num_test,
            "file_size": file_size,
            "tags": tags,
            "extra_metadata": extra_metadata,
            "status": status,
        }
        update_values = {
            name: value
            for name, value in optional_values.items()
            if value is not None
        }
        update_values["updated_at"] = now_naive()

        blocked_statuses = (
            DatasetStatus.DELETING.value,
            DatasetStatus.STAGING.value,
            DatasetStatus.UPLOADING.value,
            DatasetStatus.DOWNLOADING.value,
            DatasetStatus.PROCESSING.value,
        )
        statement = (
            update(DatasetDB)
            .where(
                DatasetDB.dataset_id == dataset_id,
                DatasetDB.user_id == expected_user_id,
                DatasetDB.status == expected_status,
                DatasetDB.deletion_owner == expected_deletion_owner,
                DatasetDB.status.notin_(blocked_statuses),
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )

        with Session(self._get_engine()) as session:
            result = session.execute(statement)
            if result.rowcount == 1:
                updated = session.exec(
                    select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
                ).one()
                snapshot = updated.to_dict()
                session.commit()
                logger.info("Updated dataset: %s", dataset_id)
                return DatasetUpdateOutcome(dataset=snapshot)

            current = session.exec(
                select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
            ).first()
            if current is None:
                return DatasetUpdateOutcome(missing=True)
            if current.user_id != expected_user_id:
                return DatasetUpdateOutcome(ownership_mismatch=True)
            return DatasetUpdateOutcome(conflict=True)

    def cache_preview_if_snapshot_matches(
        self,
        dataset_id: str,
        *,
        expected_status: str,
        expected_storage_backend: str,
        expected_storage_path: Optional[str],
        expected_storage_uri: Optional[str],
        expected_file_format: str,
        expected_version: int,
        columns: Optional[List[Dict[str, Any]]] = None,
        sample_data: Optional[List[Dict[str, Any]]] = None,
        num_rows: Optional[int] = None,
        num_train: Optional[int] = None,
        num_eval: Optional[int] = None,
        num_test: Optional[int] = None,
        file_size: Optional[int] = None,
    ) -> bool:
        """Cache preview metadata only while its source snapshot is current."""
        if expected_status not in {
            DatasetStatus.REGISTERED.value,
            DatasetStatus.READY.value,
        }:
            return False

        incoming_sample_count = len(sample_data or [])
        if incoming_sample_count == 0:
            return False

        optional_values = {
            "columns": columns,
            "sample_data": sample_data,
            "num_rows": num_rows,
            "num_train": num_train,
            "num_eval": num_eval,
            "num_test": num_test,
            "file_size": file_size,
        }
        preview_values = {
            name: value
            for name, value in optional_values.items()
            if value is not None
        }
        if expected_status == DatasetStatus.REGISTERED.value:
            preview_values["status"] = DatasetStatus.READY.value

        engine = self._get_engine()
        statement = update(DatasetDB).where(
            DatasetDB.dataset_id == dataset_id,
            DatasetDB.status == expected_status,
            DatasetDB.storage_backend == expected_storage_backend,
            DatasetDB.storage_path == expected_storage_path,
            DatasetDB.storage_uri == expected_storage_uri,
            DatasetDB.file_format == expected_file_format,
            DatasetDB.version == expected_version,
        )
        if expected_status == DatasetStatus.READY.value:
            if engine.dialect.name == "mysql":
                cached_sample_count = case(
                    (func.json_type(DatasetDB.sample_data) == "NULL", 0),
                    else_=func.coalesce(
                        func.json_length(DatasetDB.sample_data),
                        0,
                    ),
                )
            elif engine.dialect.name == "sqlite":
                cached_sample_count = func.coalesce(
                    func.json_array_length(DatasetDB.sample_data),
                    0,
                )
            else:
                logger.warning(
                    "Preview cache CAS does not support database dialect %s",
                    engine.dialect.name,
                )
                return False
            statement = statement.where(
                cached_sample_count < incoming_sample_count
            )
        statement = (
            statement.values(
                **preview_values,
                updated_at=now_naive(),
            )
            .execution_options(synchronize_session=False)
        )
        with Session(engine) as session:
            result = session.execute(statement)
            if result.rowcount == 1:
                session.commit()
                logger.info("Cached stable dataset preview: %s", dataset_id)
                return True

        if expected_status != DatasetStatus.READY.value:
            return False
        with Session(engine) as session:
            current = session.exec(
                select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
            ).first()
        return bool(
            current
            and current.status == expected_status
            and current.storage_backend == expected_storage_backend
            and current.storage_path == expected_storage_path
            and current.storage_uri == expected_storage_uri
            and current.file_format == expected_file_format
            and current.version == expected_version
            and len(current.sample_data or []) >= incoming_sample_count
        )

    def update_status(
        self, dataset_id: str, status: str, error_message: Optional[str] = None
    ) -> bool:
        """Update dataset status."""
        with Session(self._get_engine()) as session:
            statement = select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
            dataset = session.exec(statement).first()

            if not dataset:
                return False

            dataset.update_status(status, error_message)
            session.add(dataset)
            session.commit()
            logger.info(f"Updated dataset status: {dataset_id} -> {status}")
            return True

    def mark_deleting(
        self,
        dataset_id: str,
        *,
        deletion_owner: str,
        user_id: Optional[str],
    ) -> bool:
        """Lock a dataset and durably block new consumers before cleanup."""
        normalized_owner = require_dataset_deletion_owner(deletion_owner)
        with Session(self._get_engine()) as session:
            statement = (
                select(DatasetDB)
                .where(DatasetDB.dataset_id == dataset_id)
                .with_for_update()
            )
            dataset = session.exec(statement).first()
            if not dataset:
                return False
            if user_id and dataset.user_id != user_id:
                raise PermissionError("Dataset owner changed during deletion")
            if dataset.status == "deleting":
                _assert_dataset_deletion_owner(dataset, normalized_owner)
                return True
            if dataset.deletion_owner is not None:
                _assert_dataset_deletion_owner(dataset, normalized_owner)
                raise RuntimeError(
                    "Dataset has a deletion owner without a deletion fence"
                )
            dataset.update_status("deleting")
            dataset.deletion_owner = normalized_owner
            session.add(dataset)
            session.commit()
            logger.info("Marked dataset %s for deletion", dataset_id)
            return True

    def restore_from_deleting(
        self,
        dataset_id: str,
        *,
        deletion_owner: str,
        status: str,
        user_id: Optional[str],
    ) -> bool:
        """Undo an unstarted deletion after a consumer conflict."""
        normalized_owner = require_dataset_deletion_owner(deletion_owner)
        if status == "deleting":
            raise ValueError("Cannot restore dataset to deleting")
        valid_statuses = {value.value for value in DatasetStatus}
        if status not in valid_statuses:
            raise ValueError(f"Invalid restored dataset status: {status}")
        with Session(self._get_engine()) as session:
            dataset = session.exec(
                select(DatasetDB)
                .where(DatasetDB.dataset_id == dataset_id)
                .with_for_update()
            ).first()
            if not dataset or dataset.status != "deleting":
                return False
            if user_id and dataset.user_id != user_id:
                raise PermissionError("Dataset owner changed during restore")
            _assert_dataset_deletion_owner(dataset, normalized_owner)
            dataset.status = status
            dataset.deletion_owner = None
            dataset.updated_at = now_naive()
            session.add(dataset)
            session.commit()
            logger.info("Restored dataset %s status to %s", dataset_id, status)
            return True

    def delete_dataset(
        self,
        dataset_id: str,
        *,
        deletion_owner: str,
        user_id: Optional[str],
    ) -> bool:
        """Delete a dataset only while the caller owns its durable fence."""
        normalized_owner = require_dataset_deletion_owner(deletion_owner)
        with Session(self._get_engine()) as session:
            statement = (
                select(DatasetDB)
                .where(DatasetDB.dataset_id == dataset_id)
                .with_for_update()
            )
            dataset = session.exec(statement).first()

            if not dataset:
                return False
            if user_id and dataset.user_id != user_id:
                raise PermissionError("Dataset owner changed during deletion")
            if dataset.status != "deleting":
                raise RuntimeError("Dataset is not fenced for deletion")
            _assert_dataset_deletion_owner(dataset, normalized_owner)

            session.delete(dataset)
            session.commit()
            logger.info(f"Deleted dataset: {dataset_id}")
            return True

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get dataset statistics."""
        with Session(self._get_engine()) as session:
            def _grouped(column):
                stmt = (
                    select(column, func.count())
                    .where(DatasetDB.status != DatasetStatus.STAGING.value)
                    .group_by(column)
                )
                if user_id:
                    stmt = stmt.where(DatasetDB.user_id == user_id)
                # row[0] = group value, row[1] = count; skip null/empty keys
                return {row[0]: row[1] for row in session.exec(stmt).all() if row[0]}

            # total + size/rows sums in a single aggregate query
            agg_stmt = select(
                func.count(),
                func.coalesce(func.sum(DatasetDB.file_size), 0),
                func.coalesce(func.sum(DatasetDB.num_rows), 0),
            ).select_from(DatasetDB).where(
                DatasetDB.status != DatasetStatus.STAGING.value
            )
            if user_id:
                agg_stmt = agg_stmt.where(DatasetDB.user_id == user_id)
            total, total_size, total_rows = session.exec(agg_stmt).one()

            return {
                "total": total,
                "by_type": _grouped(DatasetDB.dataset_type),
                "by_source": _grouped(DatasetDB.source_type),
                "by_status": _grouped(DatasetDB.status),
                "by_usage": _grouped(DatasetDB.usage),
                "total_size_bytes": int(total_size or 0),
                "total_rows": int(total_rows or 0),
            }


# Global service instance
dataset_service = DatasetService()
