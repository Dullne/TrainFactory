"""Resolve evaluation datasets from tenant-owned API-managed records."""

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..auth.resource_provenance import (
    ResourceProvenanceError,
    require_managed_dataset_provenance,
)
from ..config.settings import get_settings
from ..storage.services.dataset_service import dataset_service
from ..storage.services.generation_task_service import generation_task_service


class UnsupportedEvaluationDatasetStorageError(ResourceProvenanceError):
    """Raised when an evaluator cannot consume a dataset storage backend."""


def _iter_sync_batches(task_id: str) -> Iterable[Dict[str, Any]]:
    from ..storage.services.external_sync_service import external_sync_service

    limit = 200
    offset = 0
    while True:
        batches, total = external_sync_service.list_batches(
            task_id=task_id,
            limit=limit,
            offset=offset,
        )
        if not batches:
            return
        yield from batches
        offset += len(batches)
        if not isinstance(total, int) or offset >= total:
            return


def _require_local_dataset_record(dataset_id: str) -> Dict[str, Any]:
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ResourceProvenanceError("Evaluation dataset ID is missing")

    dataset = dataset_service.get_dataset(dataset_id.strip())
    if not dataset:
        raise ResourceProvenanceError("Evaluation dataset does not exist")
    if dataset.get("status") != "ready":
        raise ResourceProvenanceError("Evaluation dataset is not ready")
    if dataset.get("storage_backend", "local") != "local":
        raise UnsupportedEvaluationDatasetStorageError(
            "S3 datasets are not supported by local evaluation workers"
        )

    storage_path = dataset.get("storage_path")
    if not isinstance(storage_path, str) or not storage_path.strip():
        raise ResourceProvenanceError("Evaluation dataset has no local storage path")
    return dataset


def resolve_local_evaluation_dataset_record(dataset_id: str) -> Dict[str, Any]:
    """Resolve an unscoped local dataset record for auth-disabled compatibility."""
    return _require_local_dataset_record(dataset_id)


def resolve_managed_local_dataset(
    dataset_id: str,
    *,
    user_id: str,
) -> Dict[str, Any]:
    """Resolve and prove an owned, API-managed local dataset."""
    if not isinstance(user_id, str) or not user_id.strip() or user_id == "anonymous":
        raise ResourceProvenanceError("Dataset owner is missing")

    dataset = _require_local_dataset_record(dataset_id)
    storage_path = dataset["storage_path"]
    settings = get_settings()
    from ..storage.services.external_sync_service import external_sync_service

    require_managed_dataset_provenance(
        dataset,
        storage_path,
        user_id=user_id,
        datasets_dir=Path(settings.datasets_dir),
        s3_bucket=settings.minio_bucket,
        generation_output_dir=Path(
            os.environ.get("GENERATION_OUTPUT_DIR", str(settings.datasets_dir))
        ),
        generation_task_lookup=generation_task_service.get_task,
        sync_data_dir=Path(os.environ.get("SYNC_DATA_DIR", "/app/data/sync")),
        sync_task_lookup=external_sync_service.get_task_raw,
        sync_training_lookup=external_sync_service.get_training_by_task_id,
        sync_batch_lookup=_iter_sync_batches,
    )
    return dataset


def resolve_managed_local_evaluation_dataset(
    dataset_id: str,
    *,
    user_id: str,
) -> Dict[str, Any]:
    """Backward-compatible evaluation-specific name for the shared resolver."""
    return resolve_managed_local_dataset(dataset_id, user_id=user_id)


def evaluation_dataset_allowed_roots() -> list[Path]:
    """Return filesystem roots that can hold managed evaluation datasets."""
    settings = get_settings()
    roots = [Path(settings.datasets_dir)]
    generation_root = Path(
        os.environ.get("GENERATION_OUTPUT_DIR", str(settings.datasets_dir))
    )
    if generation_root not in roots:
        roots.append(generation_root)
    sync_root = Path(os.environ.get("SYNC_DATA_DIR", "/app/data/sync"))
    if sync_root not in roots:
        roots.append(sync_root)
    return roots


def task_requires_dataset_provenance(user_id: Optional[str]) -> bool:
    """Keep worker-side enforcement tied to current authentication mode."""
    if not get_settings().auth_enabled:
        return False
    if not user_id or user_id == "anonymous":
        raise ResourceProvenanceError("Authenticated evaluation task owner is missing")
    return True
