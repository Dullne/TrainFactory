"""Proof checks for tenant-owned filesystem and object-storage resources."""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import urlsplit


class ResourceProvenanceError(ValueError):
    """Raised when a resource record cannot prove an API-managed origin."""


TaskLookup = Callable[[str], Optional[Dict[str, Any]]]
BatchLookup = Callable[[str], Iterable[Dict[str, Any]]]


def _require_nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourceProvenanceError(f"{label} is missing")
    return value.strip()


def _canonical_local_path(value: Any) -> Optional[str]:
    if not isinstance(value, (str, os.PathLike)):
        return None
    path = os.fspath(value).strip()
    if not path:
        return None
    if path.startswith(("s3://", "http://", "https://")):
        return None
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _same_local_path(left: Any, right: Any) -> bool:
    left_path = _canonical_local_path(left)
    right_path = _canonical_local_path(right)
    return bool(left_path and right_path and left_path == right_path)


def _is_exact_managed_path(
    requested_path: Any,
    managed_root: Path,
    relative_path: str,
) -> bool:
    expected_path = Path(managed_root) / relative_path
    if not _same_local_path(requested_path, expected_path):
        return False

    canonical_root = _canonical_local_path(managed_root)
    canonical_target = _canonical_local_path(expected_path)
    if not canonical_root or not canonical_target:
        return False
    try:
        relative_target = Path(canonical_target).relative_to(Path(canonical_root))
    except ValueError:
        return False
    return bool(relative_target.parts)


def _valid_resource_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    resource_id = value.strip()
    if (
        not resource_id
        or resource_id in {".", ".."}
        or "/" in resource_id
        or "\\" in resource_id
    ):
        return None
    return resource_id


def require_managed_model_provenance(
    model: Dict[str, Any],
    requested_path: str,
    *,
    user_id: str,
    models_dir: Path,
    training_output_dir: Path,
    training_task_lookup: TaskLookup,
) -> None:
    """Require a completed API download or an owned successful training output."""
    if model.get("user_id") != user_id:
        raise ResourceProvenanceError("Model owner does not match")
    if not _same_local_path(requested_path, model.get("model_path")):
        raise ResourceProvenanceError("Model path does not match its registry record")

    source_type = str(model.get("source_type") or "").strip().lower()
    if source_type == "downloaded":
        model_id = _valid_resource_id(model.get("model_id"))
        if not model_id:
            raise ResourceProvenanceError("Downloaded model ID is invalid")
        if model.get("status") != "available":
            raise ResourceProvenanceError("Downloaded model is not available")
        if not _is_exact_managed_path(requested_path, models_dir, model_id):
            raise ResourceProvenanceError("Downloaded model path is not API-managed")
        return

    if source_type != "trained":
        raise ResourceProvenanceError("Model source is not API-managed")
    if model.get("status") != "available":
        raise ResourceProvenanceError("Trained model is not available")

    source_task_id = _require_nonempty(
        model.get("source_task_id"),
        "Model source task",
    )
    task = training_task_lookup(source_task_id)
    if not task or task.get("task_id") != source_task_id:
        raise ResourceProvenanceError("Model source task does not exist")
    if task.get("user_id") != user_id:
        raise ResourceProvenanceError("Model source task owner does not match")
    if task.get("status") != "succeeded":
        raise ResourceProvenanceError("Model source task did not succeed")
    if not _same_local_path(requested_path, task.get("final_model_path")):
        raise ResourceProvenanceError("Model path is not the source task final model")

    # 纵深防御：训练产物必须位于受管输出根目录内（与 downloaded 分支的
    # _is_exact_managed_path 同等约束），防止未来训练路径允许逃逸时
    # provenance 防线失效。
    output_root = _canonical_local_path(training_output_dir)
    final_path = _canonical_local_path(Path(requested_path))
    if not output_root or not final_path:
        raise ResourceProvenanceError("Model path is not canonicalizable")
    try:
        Path(final_path).relative_to(Path(output_root))
    except ValueError:
        raise ResourceProvenanceError(
            "Model path is not inside the managed output directory"
        )


def _require_managed_s3_dataset(
    dataset: Dict[str, Any],
    requested_reference: str,
    *,
    bucket: str,
) -> None:
    if dataset.get("storage_backend") != "s3":
        raise ResourceProvenanceError("Dataset storage backend does not match")
    storage_uri = _require_nonempty(dataset.get("storage_uri"), "Dataset storage URI")
    if requested_reference.strip() != storage_uri:
        raise ResourceProvenanceError("Dataset URI does not match its record")

    dataset_id = _valid_resource_id(dataset.get("dataset_id"))
    if not dataset_id:
        raise ResourceProvenanceError("Dataset ID is invalid")
    parsed = urlsplit(storage_uri)
    if parsed.scheme != "s3" or parsed.netloc != bucket:
        raise ResourceProvenanceError("Dataset URI uses an unmanaged bucket")
    if parsed.query or parsed.fragment or "\\" in parsed.path:
        raise ResourceProvenanceError("Dataset URI is invalid")

    key_parts = parsed.path.lstrip("/").split("/")
    if (
        len(key_parts) < 3
        or key_parts[0] != "datasets"
        or key_parts[1] != dataset_id
        or any(part in {"", ".", ".."} for part in key_parts)
    ):
        raise ResourceProvenanceError("Dataset URI is not bound to its dataset ID")


_GENERATION_OUTPUT_FIELDS = (
    ("output_path", "output_dataset_id", "generated"),
    ("qa_output_path", "qa_dataset_id", "qa_extracted"),
    ("qa_filtered_path", "qa_filtered_dataset_id", "qa_filtered"),
    ("deep_eval_path", "deep_eval_dataset_id", "deep_eval"),
)

_LEGACY_SYNC_MERGED_FILENAME = re.compile(r"merged_[0-9]{8}_[0-9]{6}\.jsonl")
_SYNC_ATTEMPT_FILENAME = re.compile(r"merged_[0-9a-f]{32}\.jsonl")


def _require_generation_dataset(
    dataset: Dict[str, Any],
    requested_path: str,
    *,
    user_id: str,
    generation_output_dir: Path,
    generation_task_lookup: TaskLookup,
) -> None:
    if dataset.get("storage_backend", "local") != "local":
        raise ResourceProvenanceError("Generated dataset must use local storage")
    if dataset.get("source_task_type") != "generation":
        raise ResourceProvenanceError("Generated dataset source type is unsupported")

    source_task_id = _valid_resource_id(dataset.get("source_task_id"))
    if not source_task_id:
        raise ResourceProvenanceError("Dataset source task is invalid")
    task = generation_task_lookup(source_task_id)
    if not task or task.get("task_id") != source_task_id:
        raise ResourceProvenanceError("Dataset source task does not exist")
    if task.get("user_id") != user_id:
        raise ResourceProvenanceError("Dataset source task owner does not match")
    if task.get("status") != "completed":
        raise ResourceProvenanceError("Dataset source task did not complete")

    dataset_id = _require_nonempty(dataset.get("dataset_id"), "Dataset ID")
    for path_field, dataset_id_field, filename_prefix in _GENERATION_OUTPUT_FIELDS:
        if task.get(dataset_id_field) != dataset_id or not _same_local_path(
            requested_path,
            task.get(path_field),
        ):
            continue
        for task_tag in (source_task_id, source_task_id[:8]):
            if _is_exact_managed_path(
                requested_path,
                generation_output_dir,
                f"{filename_prefix}_{task_tag}.jsonl",
            ):
                return

    raise ResourceProvenanceError("Dataset is not a controlled source task output")


def _require_sync_dataset(
    dataset: Dict[str, Any],
    requested_path: str,
    *,
    user_id: str,
    sync_data_dir: Path,
    sync_task_lookup: TaskLookup,
    sync_training_lookup: Optional[TaskLookup],
    sync_batch_lookup: Optional[BatchLookup],
    training_task_id: Optional[str],
) -> None:
    if dataset.get("storage_backend", "local") != "local":
        raise ResourceProvenanceError("Sync dataset must use local storage")

    source_task_id = _valid_resource_id(dataset.get("source_task_id"))
    if not source_task_id or len(source_task_id) < 8:
        raise ResourceProvenanceError("Sync dataset source task ID is invalid")
    metadata = dataset.get("extra_metadata")
    if not isinstance(metadata, dict) or metadata.get("sync_task_id") != source_task_id:
        raise ResourceProvenanceError("Sync dataset metadata is not bound to its task")

    sync_task = sync_task_lookup(source_task_id)
    if not sync_task or sync_task.get("task_id") != source_task_id:
        raise ResourceProvenanceError("Sync dataset source task does not exist")
    if sync_task.get("user_id") != user_id:
        raise ResourceProvenanceError("Sync dataset source task owner does not match")

    dataset_id = _valid_resource_id(dataset.get("dataset_id"))
    safe_user_id = _valid_resource_id(user_id)
    filename = Path(requested_path).name
    if not dataset_id or not safe_user_id:
        raise ResourceProvenanceError("Sync dataset output identity is invalid")

    is_new_layout = bool(_SYNC_ATTEMPT_FILENAME.fullmatch(filename)) and (
        _is_exact_managed_path(
            requested_path,
            sync_data_dir,
            str(Path(safe_user_id) / source_task_id / "merged" / filename),
        )
    )
    is_legacy_layout = bool(_LEGACY_SYNC_MERGED_FILENAME.fullmatch(filename)) and (
        _is_exact_managed_path(
            requested_path,
            sync_data_dir,
            str(Path(safe_user_id[:16]) / source_task_id[:8] / "merged" / filename),
        )
    )
    if not (is_new_layout or is_legacy_layout):
        raise ResourceProvenanceError("Sync dataset path is not a controlled merged output")

    if training_task_id and sync_training_lookup:
        sync_training = sync_training_lookup(training_task_id)
        if sync_training is not None:
            input_dataset_ids = sync_training.get("input_dataset_ids")
            if (
                sync_training.get("training_task_id") != training_task_id
                or sync_training.get("task_id") != source_task_id
                or sync_training.get("user_id") != user_id
                or not isinstance(input_dataset_ids, list)
                or dataset_id not in input_dataset_ids
            ):
                raise ResourceProvenanceError(
                    "Sync dataset does not match its training tracking record"
                )
            return

    if sync_batch_lookup:
        for batch in sync_batch_lookup(source_task_id):
            if (
                isinstance(batch, dict)
                and batch.get("task_id") == source_task_id
                and batch.get("user_id") == user_id
                and batch.get("dataset_id") == dataset_id
            ):
                return

    raise ResourceProvenanceError("Sync dataset is not bound to a tracked batch")


def require_managed_dataset_provenance(
    dataset: Dict[str, Any],
    requested_reference: str,
    *,
    user_id: str,
    datasets_dir: Path,
    s3_bucket: str,
    generation_output_dir: Path,
    generation_task_lookup: TaskLookup,
    sync_data_dir: Optional[Path] = None,
    sync_task_lookup: Optional[TaskLookup] = None,
    sync_training_lookup: Optional[TaskLookup] = None,
    sync_batch_lookup: Optional[BatchLookup] = None,
    training_task_id: Optional[str] = None,
) -> None:
    """Require an API upload/download or a bound generated output."""
    if dataset.get("user_id") != user_id:
        raise ResourceProvenanceError("Dataset owner does not match")
    if dataset.get("status") != "ready":
        raise ResourceProvenanceError("Dataset is not ready")

    source_type = str(dataset.get("source_type") or "").strip().lower()
    if source_type == "generated":
        if not _same_local_path(requested_reference, dataset.get("storage_path")):
            raise ResourceProvenanceError("Dataset path does not match its record")
        source_task_type = str(dataset.get("source_task_type") or "").strip().lower()
        if source_task_type == "generation":
            _require_generation_dataset(
                dataset,
                requested_reference,
                user_id=user_id,
                generation_output_dir=generation_output_dir,
                generation_task_lookup=generation_task_lookup,
            )
        elif source_task_type == "sync" and sync_data_dir and sync_task_lookup:
            _require_sync_dataset(
                dataset,
                requested_reference,
                user_id=user_id,
                sync_data_dir=sync_data_dir,
                sync_task_lookup=sync_task_lookup,
                sync_training_lookup=sync_training_lookup,
                sync_batch_lookup=sync_batch_lookup,
                training_task_id=training_task_id,
            )
        else:
            raise ResourceProvenanceError("Generated dataset source type is unsupported")
        return

    if source_type not in {"uploaded", "huggingface", "modelscope"}:
        raise ResourceProvenanceError("Dataset source is not API-managed")

    if requested_reference.startswith("s3://"):
        _require_managed_s3_dataset(
            dataset,
            requested_reference,
            bucket=s3_bucket,
        )
        return

    if dataset.get("storage_backend", "local") != "local":
        raise ResourceProvenanceError("Dataset storage backend does not match")
    if not _same_local_path(requested_reference, dataset.get("storage_path")):
        raise ResourceProvenanceError("Dataset path does not match its record")
    dataset_id = _valid_resource_id(dataset.get("dataset_id"))
    if not dataset_id:
        raise ResourceProvenanceError("Dataset ID is invalid")
    if not _is_exact_managed_path(requested_reference, datasets_dir, dataset_id):
        raise ResourceProvenanceError("Dataset path is not bound to its dataset ID")
