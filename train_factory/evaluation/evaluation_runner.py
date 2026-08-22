"""
Evaluation task runner.

Executes evaluation tasks in the background using qwen3-rerank-trainer.
Supports multi-model, multi-dataset, parallel evaluation with progress tracking.
"""

import hashlib
import logging
import time
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..auth.resource_provenance import ResourceProvenanceError
from ..config.settings import get_settings
from ..storage.services.outbound_endpoint_policy import validate_user_outbound_url
from .dataset_access import (
    evaluation_dataset_allowed_roots,
    resolve_local_evaluation_dataset_record,
    resolve_managed_local_evaluation_dataset,
    task_requires_dataset_provenance,
)
from .secure_api_reranker import SecureAPIReranker


MAX_EVALUATION_MODELS = 16
MAX_EVALUATION_DATASETS = 32
MAX_EVALUATION_COMBINATIONS = 128
MAX_EVALUATION_SAMPLES = 1_000_000
MAX_EVALUATION_BATCH_SIZE = 512
MAX_EVALUATION_WORKERS = 16
MAX_EVALUATION_MODEL_WORKERS = 4

EVALUATION_IDENTITY_SCHEMA_VERSION = 2
LEGACY_EVALUATION_IDENTITY_SCHEMA_VERSION = 1
EVALUATION_IDENTITY_FIELD = "_evaluation_identity"
MTEB_RERANKING_DATASET_NAMES = frozenset(
    {
        "T2Reranking",
        "MMarcoReranking",
        "CMedQAv1-reranking",
        "CMedQAv2-reranking",
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
        "WebLINXCandidatesReranking",
        "BuiltBenchReranking",
    }
)


def _evaluation_identity(result_key: str) -> Dict[str, Any]:
    return {
        "schema_version": EVALUATION_IDENTITY_SCHEMA_VERSION,
        "result_key": result_key,
    }


def canonical_evaluation_model_name(config: Dict[str, Any]) -> str:
    """Return the stable result key for an evaluation model configuration."""
    for field_name in ("name", "model_name"):
        value = config.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Evaluation model name or model_name must be non-empty")


def canonicalize_evaluation_model_configs(
    model_configs: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Normalize and uniquely fence identities used as evaluation result keys."""
    normalized_configs = []
    seen_names = set()
    for raw_config in model_configs:
        config = dict(raw_config)
        model_name = config.get("model_name")
        if isinstance(model_name, str):
            config["model_name"] = model_name.strip() or None
        canonical_name = canonical_evaluation_model_name(config)
        if canonical_name in seen_names:
            raise ValueError(
                f"Duplicate evaluation model identity: {canonical_name}"
            )
        seen_names.add(canonical_name)
        config["name"] = canonical_name
        config[EVALUATION_IDENTITY_FIELD] = _evaluation_identity(canonical_name)
        normalized_configs.append(config)
    return normalized_configs


def canonical_evaluation_dataset_key(config: Dict[str, Any]) -> str:
    """Return a namespace-aware result key without changing the display name."""
    dataset_type = config.get("type")
    if not isinstance(dataset_type, str) or not dataset_type.strip():
        raise ValueError("Evaluation dataset type must be non-empty")
    dataset_type = dataset_type.strip().lower()

    if dataset_type == "mteb":
        local_only_fields = [
            field_name
            for field_name in ("dataset_id", "path")
            if config.get(field_name) is not None
        ]
        if local_only_fields:
            raise ValueError(
                "MTEB evaluation datasets cannot include local-only fields: "
                + ", ".join(local_only_fields)
            )
        name = config.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MTEB evaluation dataset name must be non-empty")
        name = name.strip()
        if name not in MTEB_RERANKING_DATASET_NAMES:
            raise ValueError(f"Unsupported MTEB evaluation dataset: {name}")
        return f"mteb:{name}"

    if dataset_type in {"local", "registered"}:
        dataset_id = config.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id.strip():
            return f"dataset:{dataset_id.strip()}"
        path = config.get("path")
        if isinstance(path, str) and path.strip():
            digest = hashlib.sha256(path.strip().encode("utf-8")).hexdigest()
            return f"local:sha256:{digest}"
        raise ValueError(
            "Local evaluation dataset must have a dataset_id or managed path"
        )

    raise ValueError(f"Unsupported evaluation dataset type: {dataset_type}")


def canonicalize_evaluation_dataset_configs(
    dataset_configs: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Attach stable result keys while preserving user-facing dataset labels."""
    normalized_configs = []
    seen_keys = set()
    for raw_config in dataset_configs:
        config = dict(raw_config)
        dataset_type = config.get("type")
        if isinstance(dataset_type, str):
            config["type"] = dataset_type.strip().lower()
        for field_name in ("name", "dataset_id", "path"):
            value = config.get(field_name)
            if isinstance(value, str):
                config[field_name] = value.strip()
        result_key = canonical_evaluation_dataset_key(config)
        if result_key in seen_keys:
            raise ValueError(f"Duplicate evaluation dataset identity: {result_key}")
        seen_keys.add(result_key)
        config["result_key"] = result_key
        config[EVALUATION_IDENTITY_FIELD] = _evaluation_identity(result_key)
        normalized_configs.append(config)
    return normalized_configs


def _v1_evaluation_dataset_key(config: Dict[str, Any]) -> str:
    """Reconstruct the exact dataset key written by identity schema v1."""
    dataset_type = config.get("type")
    if not isinstance(dataset_type, str) or not dataset_type.strip():
        raise ValueError("Evaluation dataset type must be non-empty")
    dataset_type = dataset_type.strip().lower()
    if dataset_type == "mteb":
        return canonical_evaluation_dataset_key(config)
    if dataset_type in {"local", "registered"}:
        dataset_id = config.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id.strip():
            return f"dataset:{dataset_id.strip()}"
        path = config.get("path")
        if isinstance(path, str) and path.strip():
            return f"local:{path.strip()}"
        raise ValueError(
            "Local evaluation dataset must have a dataset_id or managed path"
        )
    raise ValueError(f"Unsupported evaluation dataset type: {dataset_type}")


def _persisted_identity_schema_version(
    model_configs: list[Dict[str, Any]],
    dataset_configs: list[Dict[str, Any]],
) -> Optional[int]:
    """Validate identity markers and identify authoritative snapshots."""
    configs = [("model", config) for config in model_configs] + [
        ("dataset", config) for config in dataset_configs
    ]
    marker_presence = [EVALUATION_IDENTITY_FIELD in config for _, config in configs]
    if not any(marker_presence):
        return None
    if not all(marker_presence):
        raise ValueError("Stored evaluation identity snapshot is incomplete")

    versions = set()
    for kind, config in configs:
        identity = config.get(EVALUATION_IDENTITY_FIELD)
        if not isinstance(identity, dict):
            raise ValueError("Stored evaluation identity marker must be an object")
        schema_version = identity.get("schema_version")
        if isinstance(schema_version, bool) or schema_version not in {
            LEGACY_EVALUATION_IDENTITY_SCHEMA_VERSION,
            EVALUATION_IDENTITY_SCHEMA_VERSION,
        }:
            raise ValueError("Unsupported stored evaluation identity schema version")
        versions.add(schema_version)

    if len(versions) != 1:
        raise ValueError("Stored evaluation identity snapshot mixes schema versions")
    schema_version = next(iter(versions))

    for kind, config in configs:
        identity = config[EVALUATION_IDENTITY_FIELD]
        expected_key = (
            canonical_evaluation_model_name(config)
            if kind == "model"
            else (
                _v1_evaluation_dataset_key(config)
                if schema_version == LEGACY_EVALUATION_IDENTITY_SCHEMA_VERSION
                else canonical_evaluation_dataset_key(config)
            )
        )
        if identity.get("result_key") != expected_key:
            raise ValueError("Stored evaluation identity key does not match its config")
    return schema_version


def _legacy_result_alias(value: Any) -> set[Any]:
    aliases = set()
    if value is None:
        aliases.update((None, "null"))
    elif isinstance(value, str):
        aliases.add(value)
        if value.strip():
            aliases.add(value.strip())
    return aliases


def _evaluation_identity_aliases(
    raw_configs: list[Dict[str, Any]],
    canonical_configs: list[Dict[str, Any]],
    *,
    kind: str,
) -> tuple[set[str], Dict[Any, set[str]]]:
    canonical_keys = set()
    aliases: Dict[Any, set[str]] = {}
    for raw_config, canonical_config in zip(raw_configs, canonical_configs):
        if kind == "model":
            canonical_key = canonical_config["name"]
            legacy_value = (
                raw_config.get("name")
                if "name" in raw_config
                else raw_config.get("model_name", "unknown")
            )
        else:
            canonical_key = canonical_config["result_key"]
            if canonical_config["type"] == "mteb":
                legacy_values = {raw_config.get("name")}
            else:
                legacy_values = {
                    raw_config.get("name", raw_config.get("path", "local"))
                }
                raw_path = raw_config.get("path")
                if isinstance(raw_path, str) and raw_path.strip():
                    legacy_values.add(f"local:{raw_path.strip()}")

        canonical_keys.add(canonical_key)
        if kind == "model":
            legacy_values = {legacy_value}
        aliases_for_config = {canonical_key}
        for legacy_value in legacy_values:
            aliases_for_config.update(_legacy_result_alias(legacy_value))
        for alias in aliases_for_config:
            aliases.setdefault(alias, set()).add(canonical_key)
    return canonical_keys, aliases


def _authoritative_identity_key_map(
    raw_configs: list[Dict[str, Any]],
    canonical_configs: list[Dict[str, Any]],
    *,
    kind: str,
) -> Dict[Any, str]:
    """Map validated persisted marker keys to current canonical keys."""
    key_map = {}
    for raw_config, canonical_config in zip(raw_configs, canonical_configs):
        source_key = raw_config[EVALUATION_IDENTITY_FIELD]["result_key"]
        target_key = (
            canonical_config["name"]
            if kind == "model"
            else canonical_config["result_key"]
        )
        if source_key in key_map:
            raise ValueError(
                f"Stored evaluation {kind} identity key collision: {source_key}"
            )
        key_map[source_key] = target_key
    return key_map


def _migrate_evaluation_key_matrix(
    value: Any,
    *,
    model_keys: set[str],
    model_aliases: Dict[Any, set[str]],
    dataset_keys: set[str],
    dataset_aliases: Dict[Any, set[str]],
    field_name: str,
    authoritative_model_keys: Optional[Dict[Any, str]],
    authoritative_dataset_keys: Optional[Dict[Any, str]],
) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Stored evaluation {field_name} must be an object")

    def resolve_key(
        stored_key: Any,
        canonical_keys: set[str],
        aliases: Dict[Any, set[str]],
        authoritative_keys: Optional[Dict[Any, str]],
        identity_kind: str,
    ) -> str:
        if authoritative_keys is not None:
            if stored_key in authoritative_keys:
                return authoritative_keys[stored_key]
            raise ValueError(
                f"Unknown stored evaluation {identity_kind} key: {stored_key!r}"
            )

        # Unversioned snapshots are legacy data. Resolve every possible alias
        # before accepting a canonical-looking key because a display label may
        # itself look namespaced and collide with a different canonical key.
        candidates = aliases.get(stored_key, set())
        if not candidates:
            raise ValueError(
                f"Unknown stored evaluation {identity_kind} key: {stored_key!r}"
            )
        if len(candidates) != 1:
            raise ValueError(
                f"Ambiguous legacy evaluation {identity_kind} key: {stored_key!r}"
            )
        return next(iter(candidates))

    migrated: Dict[str, Any] = {}
    for stored_model_key, stored_datasets in value.items():
        model_key = resolve_key(
            stored_model_key,
            model_keys,
            model_aliases,
            authoritative_model_keys,
            "model",
        )
        if model_key in migrated:
            raise ValueError(
                f"Stored evaluation {field_name} model key collision: {model_key}"
            )
        if not isinstance(stored_datasets, dict):
            raise ValueError(
                f"Stored evaluation {field_name} entry for {model_key} must be an object"
            )

        migrated_datasets: Dict[str, Any] = {}
        for stored_dataset_key, stored_value in stored_datasets.items():
            if field_name == "results" and stored_dataset_key == "_error":
                dataset_key = "_error"
            else:
                dataset_key = resolve_key(
                    stored_dataset_key,
                    dataset_keys,
                    dataset_aliases,
                    authoritative_dataset_keys,
                    "dataset",
                )
            if dataset_key in migrated_datasets:
                raise ValueError(
                    f"Stored evaluation {field_name} dataset key collision: "
                    f"{model_key}/{dataset_key}"
                )
            migrated_datasets[dataset_key] = stored_value
        migrated[model_key] = migrated_datasets
    return migrated


def canonicalize_evaluation_resume_state(
    model_configs: Iterable[Dict[str, Any]],
    dataset_configs: Iterable[Dict[str, Any]],
    results: Any,
    model_progress: Any,
    *,
    canonical_model_configs: Optional[Iterable[Dict[str, Any]]] = None,
    canonical_dataset_configs: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Canonicalize a persisted resume bundle without partially discarding state."""
    raw_model_configs = [dict(config) for config in model_configs]
    raw_dataset_configs = [dict(config) for config in dataset_configs]
    source_schema_version = _persisted_identity_schema_version(
        raw_model_configs,
        raw_dataset_configs,
    )
    normalized_model_configs = canonicalize_evaluation_model_configs(
        canonical_model_configs
        if canonical_model_configs is not None
        else raw_model_configs
    )
    normalized_dataset_configs = canonicalize_evaluation_dataset_configs(
        canonical_dataset_configs
        if canonical_dataset_configs is not None
        else raw_dataset_configs
    )
    model_keys, model_aliases = _evaluation_identity_aliases(
        raw_model_configs,
        normalized_model_configs,
        kind="model",
    )
    dataset_keys, dataset_aliases = _evaluation_identity_aliases(
        raw_dataset_configs,
        normalized_dataset_configs,
        kind="dataset",
    )
    authoritative_model_keys = None
    authoritative_dataset_keys = None
    if source_schema_version is not None:
        authoritative_model_keys = _authoritative_identity_key_map(
            raw_model_configs,
            normalized_model_configs,
            kind="model",
        )
        authoritative_dataset_keys = _authoritative_identity_key_map(
            raw_dataset_configs,
            normalized_dataset_configs,
            kind="dataset",
        )

    migrated_results = _migrate_evaluation_key_matrix(
        results,
        model_keys=model_keys,
        model_aliases=model_aliases,
        dataset_keys=dataset_keys,
        dataset_aliases=dataset_aliases,
        field_name="results",
        authoritative_model_keys=authoritative_model_keys,
        authoritative_dataset_keys=authoritative_dataset_keys,
    )
    # Migrate progress for collision/shape validation, then rebuild it from
    # durable results so stale "completed" flags can never skip missing work.
    _migrate_evaluation_key_matrix(
        model_progress,
        model_keys=model_keys,
        model_aliases=model_aliases,
        dataset_keys=dataset_keys,
        dataset_aliases=dataset_aliases,
        field_name="model progress",
        authoritative_model_keys=authoritative_model_keys,
        authoritative_dataset_keys=authoritative_dataset_keys,
    )
    reconciled_progress = {}
    for model_config in normalized_model_configs:
        model_key = model_config["name"]
        reconciled_progress[model_key] = {}
        for dataset_config in normalized_dataset_configs:
            dataset_key = dataset_config["result_key"]
            result = migrated_results.get(model_key, {}).get(dataset_key)
            completed = isinstance(result, dict) and "error" not in result
            reconciled_progress[model_key][dataset_key] = {
                "progress": 100 if completed else 0,
                "status": "completed" if completed else "pending",
            }

    return {
        "model_configs": normalized_model_configs,
        "dataset_configs": normalized_dataset_configs,
        "results": migrated_results,
        "model_progress": reconciled_progress,
        "source_identity_schema_version": source_schema_version,
    }


def validate_evaluation_resource_config(config: Dict[str, Any]) -> None:
    """Reject evaluation work that could create unbounded fan-out."""
    model_configs = config.get("model_configs")
    dataset_configs = config.get("dataset_configs")
    if not isinstance(model_configs, list) or not (
        1 <= len(model_configs) <= MAX_EVALUATION_MODELS
    ):
        raise ValueError(
            f"Evaluation must contain 1-{MAX_EVALUATION_MODELS} model configurations"
        )
    if not isinstance(dataset_configs, list) or not (
        1 <= len(dataset_configs) <= MAX_EVALUATION_DATASETS
    ):
        raise ValueError(
            f"Evaluation must contain 1-{MAX_EVALUATION_DATASETS} dataset configurations"
        )
    if len(model_configs) * len(dataset_configs) > MAX_EVALUATION_COMBINATIONS:
        raise ValueError(
            "Evaluation model/dataset combinations exceed the configured limit"
        )

    numeric_limits = {
        "batch_size": (1, MAX_EVALUATION_BATCH_SIZE, 50),
        "workers": (1, MAX_EVALUATION_WORKERS, 8),
        "model_workers": (1, MAX_EVALUATION_MODEL_WORKERS, 2),
    }
    for name, (minimum, maximum, default) in numeric_limits.items():
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Evaluation {name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(
                f"Evaluation {name} must be between {minimum} and {maximum}"
            )

    max_samples = config.get("max_samples")
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise ValueError("Evaluation max_samples must be an integer or null")
        if not 1 <= max_samples <= MAX_EVALUATION_SAMPLES:
            raise ValueError(
                f"Evaluation max_samples must be between 1 and {MAX_EVALUATION_SAMPLES}"
            )

logger = logging.getLogger(__name__)

# Track cancelled tasks (thread-safe)
_cancelled_tasks: set = set()
_cancelled_lock = threading.Lock()
# Cache DB-based cancellation checks with TTL (task_id -> last_check_time)
_cancel_check_cache: Dict[str, float] = {}
_CANCEL_CHECK_TTL = 5.0  # seconds between DB checks


def resolve_evaluation_dataset_path(
    path: str,
    *,
    allowed_roots: Optional[Iterable[Path]] = None,
) -> str:
    """Resolve a local evaluation dataset under a managed storage root."""
    if not path or not str(path).strip():
        raise ValueError("Local evaluation dataset path cannot be empty")

    roots = list(allowed_roots or [get_settings().datasets_dir])
    try:
        target = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Invalid local evaluation dataset path: {exc}") from exc

    for root in roots:
        try:
            target.relative_to(Path(root).resolve(strict=True))
            return str(target)
        except (OSError, RuntimeError, ValueError):
            continue

    raise ValueError("Local evaluation dataset must be inside the datasets directory")


def cancel_evaluation(task_id: str):
    """Mark a task for cancellation."""
    with _cancelled_lock:
        _cancelled_tasks.add(task_id)


def is_cancelled(task_id: str) -> bool:
    """Check if a task has been cancelled (in-memory flag or database status with TTL cache)."""
    with _cancelled_lock:
        if task_id in _cancelled_tasks:
            return True

    # Fallback: check database status with TTL to avoid excessive queries
    now = time.monotonic()
    last_check = _cancel_check_cache.get(task_id, 0)
    if now - last_check < _CANCEL_CHECK_TTL:
        return False  # Recently checked, not cancelled

    _cancel_check_cache[task_id] = now
    from ..storage.services.evaluation_task_service import evaluation_task_service
    task = evaluation_task_service.get_task(task_id)
    if task and task.get("status") == "cancelled":
        with _cancelled_lock:
            _cancelled_tasks.add(task_id)
        return True
    return False


def _cleanup_cancelled(task_id: str):
    """Remove task from cancellation tracking."""
    with _cancelled_lock:
        _cancelled_tasks.discard(task_id)
    _cancel_check_cache.pop(task_id, None)


def run_evaluation_task(task_id: str, eval_config: Dict[str, Any]):
    """
    Run evaluation task in background.

    Called by FastAPI BackgroundTasks.

    Args:
        task_id: Evaluation task ID
        eval_config: Configuration dict containing model_configs, dataset_configs, etc.
            - existing_results: Optional dict of already completed evaluations (for resume)
    """
    from ..storage.services.evaluation_task_service import evaluation_task_service

    run_token: Optional[str] = None
    try:
        task = evaluation_task_service.get_task(task_id)
        if not task:
            raise ValueError(f"Evaluation task not found: {task_id}")
        if task.get("status") != "running" or is_cancelled(task_id):
            logger.info(
                "Evaluation task %s is no longer active before startup",
                task_id,
            )
            return
        run_token = task.get("run_token")
        user_id = task.get("user_id")

        persisted_model_configs = task.get("model_configs")
        raw_model_configs = (
            persisted_model_configs
            if isinstance(persisted_model_configs, list)
            else eval_config.get("model_configs")
        )
        persisted_dataset_configs = task.get("dataset_configs")
        raw_dataset_configs = (
            persisted_dataset_configs
            if isinstance(persisted_dataset_configs, list)
            else eval_config.get("dataset_configs")
        )
        bounded_config = {
            "model_configs": raw_model_configs,
            "dataset_configs": raw_dataset_configs,
            "max_samples": task.get("max_samples", eval_config.get("max_samples")),
            "batch_size": task.get(
                "batch_size", eval_config.get("batch_size", 50)
            ),
            "workers": task.get("workers", eval_config.get("workers", 8)),
            "model_workers": task.get(
                "model_workers", eval_config.get("model_workers", 2)
            ),
        }
        validate_evaluation_resource_config(bounded_config)

        model_configs = canonicalize_evaluation_model_configs(raw_model_configs)

        dataset_configs = []
        require_provenance = task_requires_dataset_provenance(user_id)
        for raw_dataset_config in raw_dataset_configs:
            dataset_config = dict(raw_dataset_config)
            dataset_type = dataset_config.get("type")
            if not isinstance(dataset_type, str) or not dataset_type.strip():
                raise ValueError("Evaluation dataset type must be non-empty")
            dataset_type = dataset_type.strip().lower()
            dataset_config["type"] = dataset_type
            if dataset_type == "mteb":
                # Defense in depth for tasks written directly to persistence.
                canonical_evaluation_dataset_key(dataset_config)
            elif dataset_type in {"local", "registered"}:
                dataset_id = dataset_config.get("dataset_id")
                dataset = None
                if require_provenance:
                    if not dataset_id:
                        raise ResourceProvenanceError(
                            "Authenticated evaluation dataset ID is missing"
                        )
                    dataset = resolve_managed_local_evaluation_dataset(
                        dataset_id,
                        user_id=user_id,
                    )
                elif dataset_id:
                    dataset = resolve_local_evaluation_dataset_record(dataset_id)

                if dataset:
                    dataset_config["dataset_id"] = dataset["dataset_id"]
                    dataset_config["name"] = (
                        dataset.get("dataset_name")
                        or dataset.get("display_name")
                        or dataset["dataset_id"]
                    )
                    dataset_config["path"] = dataset["storage_path"]
                dataset_config["path"] = resolve_evaluation_dataset_path(
                    dataset_config.get("path", ""),
                    allowed_roots=evaluation_dataset_allowed_roots(),
                )
            else:
                raise ValueError(
                    f"Unsupported evaluation dataset type: {dataset_type}"
                )
            dataset_configs.append(dataset_config)
        dataset_configs = canonicalize_evaluation_dataset_configs(dataset_configs)

        persisted_identity_schema_version = _persisted_identity_schema_version(
            raw_model_configs,
            raw_dataset_configs,
        )
        persisted_results = task.get("results")
        persisted_model_progress = task.get("model_progress")
        resume_results = (
            persisted_results
            if persisted_identity_schema_version is not None
            or persisted_results is not None
            else eval_config.get("existing_results", {})
        )
        resume_model_progress = (
            persisted_model_progress
            if persisted_identity_schema_version is not None
            or persisted_model_progress is not None
            else eval_config.get("existing_model_progress", {})
        )
        resume_state = canonicalize_evaluation_resume_state(
            raw_model_configs,
            raw_dataset_configs,
            resume_results or {},
            resume_model_progress or {},
            canonical_model_configs=model_configs,
            canonical_dataset_configs=dataset_configs,
        )
        model_configs = resume_state["model_configs"]
        dataset_configs = resume_state["dataset_configs"]
        max_samples = bounded_config["max_samples"]
        batch_size = bounded_config["batch_size"]
        workers = bounded_config["workers"]
        model_workers = bounded_config["model_workers"]
        existing_results = resume_state["results"]
        if not model_configs:
            raise ValueError("No model configurations provided")
        if not dataset_configs:
            raise ValueError("No dataset configurations provided")

        if not evaluation_task_service.update_status(
            task_id,
            "running",
            run_token=run_token,
        ):
            logger.info(
                "Evaluation task %s lost its execution attempt before startup",
                task_id,
            )
            return
        if is_cancelled(task_id):
            logger.info(
                "Evaluation task %s was cancelled during startup",
                task_id,
            )
            return

        def is_completed_result(result: Any) -> bool:
            return isinstance(result, dict) and "error" not in result

        # Helper to check if evaluation is already completed successfully
        def is_completed(model_name: str, dataset_key: str) -> bool:
            """Check if this model+dataset combination already has successful results."""
            model_results = existing_results.get(model_name, {})
            if dataset_key not in model_results:
                return False
            result = model_results[dataset_key]
            # Consider completed if result exists and has no error
            return is_completed_result(result)

        # Separate MTEB and local datasets
        mteb_datasets = [d for d in dataset_configs if d.get("type") == "mteb"]
        local_datasets = [
            d for d in dataset_configs if d.get("type") in {"local", "registered"}
        ]
        all_dataset_keys = [d["result_key"] for d in mteb_datasets] + [
            d["result_key"] for d in local_datasets
        ]

        total_datasets = len(mteb_datasets) + len(local_datasets)
        all_results: Dict[str, Dict[str, Any]] = deepcopy(existing_results)

        # Get model names for initialization
        model_names = [
            cfg.get("name", cfg.get("model_name", "unknown"))
            for cfg in model_configs
        ]

        # Versioned resume snapshots were atomically persisted by the route,
        # so retain their completed progress when the worker initializes.
        if not evaluation_task_service.init_model_progress(
            task_id,
            model_names,
            all_dataset_keys,
            preserve_completed=True,
            run_token=run_token,
        ):
            logger.info(
                "Evaluation task %s lost its execution attempt before progress init",
                task_id,
            )
            return

        def evaluate_single_model(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluate one model against all datasets."""
            model_name = model_cfg["name"]
            model_results = {}

            # Copy successful migrated results before touching the endpoint.
            # A fully completed resume must not fail merely because the model is
            # now offline, nor can that failure replace a stored success.
            for dataset_key, dataset_result in existing_results.get(
                model_name, {}
            ).items():
                if isinstance(dataset_result, dict) and "error" not in dataset_result:
                    model_results[dataset_key] = dataset_result
                    logger.info(
                        "[%s] %s: Skipped (already completed)",
                        model_name,
                        dataset_key,
                    )

            if all(is_completed(model_name, key) for key in all_dataset_keys):
                return {"model": model_name, "results": model_results}

            if is_cancelled(task_id):
                return {"model": model_name, "results": model_results}

            from qwen3_rerank_trainer.evaluation import (
                MTEBRerankEvaluator,
            )

            endpoint = validate_user_outbound_url(
                model_cfg["endpoint"],
                user_id,
            )
            served_model_name = model_cfg.get("model_name") or ""
            inference_framework = (
                model_cfg.get("inference_framework") or ""
            ).lower()

            # Determine model parameter based on inference framework
            # - vLLM: uses model_uid as model name
            # - SGLang: uses empty string (uses default loaded model)
            # - Xinference: uses model_uid as model name
            if inference_framework == "sglang":
                api_model_name = ""  # SGLang uses default model
            else:
                api_model_name = served_model_name

            # Create API reranker
            reranker = SecureAPIReranker(
                endpoint,
                model=api_model_name,
                batch_size=batch_size,
                max_concurrency=workers,
                inference_framework=inference_framework,
                user_id=user_id,
            )

            # Test connection
            if not reranker.test_connection():
                logger.error(f"API connection failed for {model_name}: {endpoint}")
                model_results["_error"] = {
                    "error": f"Connection failed: {endpoint}"
                }
                return {
                    "model": model_name,
                    "results": model_results,
                }

            evaluator = MTEBRerankEvaluator(
                reranker=reranker,
                batch_size=batch_size,
                workers=workers,
            )

            # Throttled progress callback to reduce DB writes
            def make_progress_callback(
                m_name: str,
                dataset_key: str,
                dataset_display_name: str,
            ):
                last_update = [0.0]  # mutable container for closure
                def callback(current: int, total: int):
                    if total > 0:
                        progress = current / total * 100
                        now = time.monotonic()
                        # Only write to DB at most once per 2 seconds, or at 100%
                        if now - last_update[0] >= 2.0 or progress >= 100:
                            last_update[0] = now
                            if not evaluation_task_service.update_model_progress(
                                task_id,
                                m_name,
                                dataset_key,
                                progress,
                                "running",
                                current_model=m_name[:255],
                                current_dataset=dataset_display_name[:255],
                                run_token=run_token,
                            ):
                                raise RuntimeError(
                                    "Evaluation attempt became inactive during progress"
                                )
                return callback

            # Evaluate MTEB datasets
            for ds_cfg in mteb_datasets:
                ds_name = ds_cfg["name"]
                dataset_key = ds_cfg["result_key"]
                # Check cancellation before each dataset
                if is_cancelled(task_id):
                    logger.info(f"[{model_name}] Evaluation cancelled, stopping")
                    return {"model": model_name, "results": model_results}

                # Skip if already completed
                if is_completed(model_name, dataset_key):
                    continue

                # Mark as running
                if not evaluation_task_service.update_model_progress(
                    task_id,
                    model_name,
                    dataset_key,
                    0,
                    "running",
                    current_model=model_name[:255],
                    current_dataset=ds_name[:255],
                    run_token=run_token,
                ):
                    raise RuntimeError(
                        "Evaluation attempt became inactive before dataset start"
                    )

                progress_cb = make_progress_callback(
                    model_name,
                    dataset_key,
                    ds_name,
                )
                try:
                    metrics = evaluator.evaluate(
                        task_name=ds_name,
                        max_samples=max_samples,
                        progress_callback=progress_cb,
                        show_progress=True,
                    )
                    if not evaluation_task_service.save_model_dataset_result(
                        task_id,
                        model_name,
                        dataset_key,
                        metrics,
                        run_token=run_token,
                    ):
                        raise RuntimeError(
                            "Evaluation attempt became inactive while saving result"
                        )
                    model_results[dataset_key] = metrics
                    logger.info(
                        f"[{model_name}] {ds_name}: NDCG@10={metrics.get('NDCG@10', 0):.4f}"
                    )
                except Exception as e:
                    logger.error(f"[{model_name}] {ds_name} failed: {e}")
                    model_results[dataset_key] = {"error": str(e)}
                    evaluation_task_service.update_model_progress(
                        task_id,
                        model_name,
                        dataset_key,
                        100,
                        "failed",
                        run_token=run_token,
                    )

            # Evaluate local datasets
            for ds_cfg in local_datasets:
                # Check cancellation before each dataset
                if is_cancelled(task_id):
                    logger.info(f"[{model_name}] Evaluation cancelled, stopping")
                    return {"model": model_name, "results": model_results}

                ds_name = ds_cfg.get("name", ds_cfg.get("path", "local"))
                dataset_key = ds_cfg["result_key"]
                ds_path = resolve_evaluation_dataset_path(
                    ds_cfg.get("path", ""),
                    allowed_roots=evaluation_dataset_allowed_roots(),
                )

                # Skip if already completed
                if is_completed(model_name, dataset_key):
                    continue

                # Mark as running
                if not evaluation_task_service.update_model_progress(
                    task_id,
                    model_name,
                    dataset_key,
                    0,
                    "running",
                    current_model=model_name[:255],
                    current_dataset=str(ds_name)[:255],
                    run_token=run_token,
                ):
                    raise RuntimeError(
                        "Evaluation attempt became inactive before dataset start"
                    )

                progress_cb = make_progress_callback(
                    model_name,
                    dataset_key,
                    str(ds_name),
                )
                try:
                    metrics = evaluator.evaluate_local(
                        dataset=ds_path,
                        max_samples=max_samples,
                        progress_callback=progress_cb,
                    )
                    if not evaluation_task_service.save_model_dataset_result(
                        task_id,
                        model_name,
                        dataset_key,
                        metrics,
                        run_token=run_token,
                    ):
                        raise RuntimeError(
                            "Evaluation attempt became inactive while saving result"
                        )
                    model_results[dataset_key] = metrics
                except Exception as e:
                    logger.error(f"[{model_name}] {ds_name} failed: {e}")
                    model_results[dataset_key] = {"error": str(e)}
                    evaluation_task_service.update_model_progress(
                        task_id,
                        model_name,
                        dataset_key,
                        100,
                        "failed",
                        run_token=run_token,
                    )

            return {"model": model_name, "results": model_results}

        # Multi-model parallel evaluation
        effective_workers = min(model_workers, len(model_configs))

        # Count skipped evaluations
        skipped_count = sum(
            1
            for mc in model_configs
            for dataset_key in all_dataset_keys
            if is_completed(mc["name"], dataset_key)
        )
        if skipped_count > 0:
            logger.info(f"Resuming evaluation: skipping {skipped_count} already completed evaluations")

        logger.info(
            f"Starting evaluation: {len(model_configs)} models × {total_datasets} datasets, "
            f"model_workers={effective_workers}"
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(evaluate_single_model, cfg): cfg
                for cfg in model_configs
            }
            for future in as_completed(futures):
                model_name = futures[future].get("name", "unknown")
                try:
                    result = future.result()
                    model_name = result["model"]
                    model_results = all_results.setdefault(model_name, {})
                    model_results.update(result["results"])
                    if all(
                        is_completed_result(model_results.get(dataset_key))
                        for dataset_key in all_dataset_keys
                    ):
                        model_results.pop("_error", None)
                except Exception as e:
                    logger.error(f"Model {model_name} evaluation failed: {e}")
                    all_results.setdefault(model_name, {})["_error"] = {
                        "error": str(e)
                    }
                if not is_cancelled(task_id):
                    model_results = all_results.setdefault(model_name, {})
                    for dataset_key in all_dataset_keys:
                        if not is_completed_result(model_results.get(dataset_key)):
                            evaluation_task_service.update_model_progress(
                                task_id,
                                model_name,
                                dataset_key,
                                100,
                                "failed",
                                run_token=run_token,
                            )
                # 每个模型完成后立即保存中间结果，确保 cancel+resume 时结果可用
                evaluation_task_service.save_partial_results(
                    task_id,
                    deepcopy(all_results),
                    run_token=run_token,
                )

        # Check if task was cancelled during execution
        if is_cancelled(task_id):
            logger.info(f"Evaluation task {task_id} was cancelled during execution")
            # Save partial results but don't change status (already set to "cancelled" by route)
            evaluation_task_service.save_partial_results(
                task_id,
                all_results,
                run_token=run_token,
            )
            return

        # Re-read the fenced attempt so final aggregation includes any pair
        # already committed before a later evaluator/progress exception.
        latest_task = evaluation_task_service.get_task(task_id)
        if latest_task and latest_task.get("run_token") == run_token:
            durable_results = latest_task.get("results")
            if isinstance(durable_results, dict):
                for model_name in model_names:
                    durable_model_results = durable_results.get(model_name)
                    if not isinstance(durable_model_results, dict):
                        continue
                    model_results = all_results.setdefault(model_name, {})
                    for dataset_key in all_dataset_keys:
                        if dataset_key not in durable_model_results:
                            continue
                        durable_result = durable_model_results[dataset_key]
                        if (
                            is_completed_result(durable_result)
                            or dataset_key not in model_results
                        ):
                            model_results[dataset_key] = deepcopy(durable_result)

        # A task succeeds only when the full configured model/dataset matrix
        # has a successful result. Partial success remains resumable.
        all_errors = []
        for model_name in model_names:
            model_results = all_results.get(model_name, {})
            model_error = model_results.get("_error")
            for dataset_key in all_dataset_keys:
                metrics = model_results.get(dataset_key)
                if is_completed_result(metrics):
                    continue
                if isinstance(metrics, dict) and "error" in metrics:
                    error = metrics["error"]
                elif isinstance(model_error, dict) and "error" in model_error:
                    error = model_error["error"]
                else:
                    error = "missing result"
                all_errors.append(f"{model_name}/{dataset_key}: {error}")

        # Complete task
        if not all_errors:
            completed = evaluation_task_service.complete_task(
                task_id,
                status="succeeded",
                results=all_results,
                run_token=run_token,
            )
            if not completed:
                logger.info(
                    "Discarding completion for inactive evaluation task %s",
                    task_id,
                )
                return
            logger.info(f"Evaluation task {task_id} completed successfully")
        else:
            error_msg = "Evaluation matrix incomplete:\n" + "\n".join(
                all_errors[:5]
            )
            if len(all_errors) > 5:
                error_msg += f"\n... and {len(all_errors) - 5} more errors"
            completed = evaluation_task_service.complete_task(
                task_id,
                status="failed",
                results=all_results,
                error_message=error_msg,
                run_token=run_token,
            )
            if not completed:
                logger.info(
                    "Discarding completion for inactive evaluation task %s",
                    task_id,
                )
                return
            logger.error(
                "Evaluation task %s failed: evaluation matrix incomplete",
                task_id,
            )

    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Check if already cancelled - don't overwrite cancelled status
        if is_cancelled(task_id):
            logger.info(f"Evaluation task {task_id} was cancelled (ignoring error)")
            return
        completed = evaluation_task_service.complete_task(
            task_id,
            status="failed",
            error_message=error_msg,
            run_token=run_token,
        )
        if not completed:
            logger.info(
                "Discarding failure for inactive evaluation task %s",
                task_id,
            )
    finally:
        # Cleanup: remove from cancellation tracking
        _cleanup_cancelled(task_id)
