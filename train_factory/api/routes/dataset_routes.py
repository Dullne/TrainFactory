"""
Dataset API routes.

Provides endpoints for managing datasets used in training.
"""

import asyncio
import logging
import json
import csv
import shutil
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from typing import Optional, List, Dict, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...auth.dependencies import (
    get_current_user,
    requires_tenant_provenance,
    validate_storage_path,
    verify_resource_ownership,
)
from ...config.settings import get_settings
from ...core.idempotency import check_idempotency, store_idempotency_response
from ...core.remote_download_security import (
    DownloadConcurrencyExceeded,
    DownloadMetadataUnavailable,
    DownloadPolicyViolation,
    DownloadSizeExceeded,
    DownloadStorageQuotaExceeded,
    validate_remote_repo_id,
)
from ...storage.services.dataset_service import (
    DatasetDeletionOwnerConflictError,
    dataset_service,
)
from ...storage.services.dataset_asset_service import dataset_asset_service
from ...storage.services.dataset_lineage_service import dataset_lineage_service
from ...storage.services.evaluation_task_service import evaluation_task_service
from ...storage.services.generation_task_service import generation_task_service
from ...storage.services.milvus_collection_service import milvus_collection_service
from ...storage.services.training_task_service import training_task_service
from ...storage.backends import get_storage_backend
from ...utils.path_utils import map_storage_path, unmap_storage_path

logger = logging.getLogger(__name__)

router = APIRouter()


class _DatasetDeletionAlreadyRunning(RuntimeError):
    """Raised when this process is already deleting the same dataset."""


class _DatasetDeletionGuard:
    def __init__(self, dataset_id: str):
        self._dataset_id = dataset_id
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        with _dataset_deletion_lock:
            _datasets_being_deleted.discard(self._dataset_id)
        self._released = True


_dataset_deletion_lock = threading.Lock()
_datasets_being_deleted: set[str] = set()


def _begin_dataset_deletion(dataset_id: str) -> _DatasetDeletionGuard:
    """Serialize direct deletion requests for one dataset in this process."""
    with _dataset_deletion_lock:
        if dataset_id in _datasets_being_deleted:
            raise _DatasetDeletionAlreadyRunning(
                "Dataset deletion is already running in this process"
            )
        _datasets_being_deleted.add(dataset_id)
    return _DatasetDeletionGuard(dataset_id)


SUPPORTED_EXPORT_SOURCE_FORMATS = {"jsonl", "json", "csv", "parquet", "arrow"}

# === Shared Validation ===

VALID_DATASET_TYPES = [
    # Embedding types (10 types)
    "embedding_universal", "embedding_pair", "embedding_triplet",
    "embedding_multi_neg", "embedding_dynamic_neg", "embedding_cosine",
    "embedding_margin", "embedding_margin_multi", "embedding_scored",
    "embedding_score_triplet",
    # Rerank types
    "rerank_pair", "rerank_triplet", "rerank_listwise",
    # LLM types
    "sft_instruct", "dpo_preference", "rl_reward",
    # Generation intermediate types
    "qa_pair",
    "custom",
]

VALID_USAGES = ["raw", "train", "eval", "test"]

VALID_MODEL_TYPES = {"embedding", "rerank", "llm"}

VALID_STORAGE_BACKENDS = {"local", "s3"}


def _validate_dataset_params(
    dataset_type: str,
    usage: str,
    model_type_list: Optional[List[str]] = None,
    storage_backend: Optional[str] = None,
) -> None:
    """Validate common dataset parameters. Raises HTTPException on invalid values."""
    if dataset_type not in VALID_DATASET_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset_type: {dataset_type}. Must be one of {VALID_DATASET_TYPES}",
        )
    if usage not in VALID_USAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid usage: {usage}. Must be one of {VALID_USAGES}",
        )
    if model_type_list:
        invalid = [t for t in model_type_list if t not in VALID_MODEL_TYPES]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type tags: {invalid}. Each must be one of {sorted(VALID_MODEL_TYPES)}",
            )
    if storage_backend is not None and storage_backend not in VALID_STORAGE_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid storage_backend: {storage_backend}. Must be one of {sorted(VALID_STORAGE_BACKENDS)}",
        )


def _iter_records_from_source(
    source_path: Path,
    source_format: str,
    limit: Optional[int] = None,
):
    """Yield dataset records as dicts from supported local source formats."""
    emitted = 0

    def _yield_row(row: Any):
        nonlocal emitted
        if limit is not None and emitted >= limit:
            return False
        if isinstance(row, dict):
            result = row
        else:
            result = {"value": row}
        emitted += 1
        return result

    if source_format == "jsonl":
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = _yield_row(json.loads(line))
                if row is False:
                    break
                yield row
        return

    if source_format == "json":
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                row = _yield_row(item)
                if row is False:
                    break
                yield row
        elif data is not None:
            row = _yield_row(data)
            if row is not False:
                yield row
        return

    if source_format == "csv":
        with open(source_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for item in reader:
                row = _yield_row(dict(item))
                if row is False:
                    break
                yield row
        return

    if source_format == "parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(source_path)
        for batch in parquet_file.iter_batches(batch_size=2000):
            for item in batch.to_pylist():
                row = _yield_row(item)
                if row is False:
                    return
                yield row
        return

    if source_format == "arrow":
        from datasets import Dataset

        ds = Dataset.load_from_disk(str(source_path))
        for item in ds:
            row = _yield_row(item)
            if row is False:
                break
            yield row
        return

    raise ValueError(f"Unsupported export source format: {source_format}")


def _to_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _write_export_file(
    source_path: Path,
    source_format: str,
    output_path: Path,
    export_format: Literal["jsonl", "csv"],
    limit: Optional[int] = None,
) -> int:
    """Convert source dataset into an export file and return row count."""
    count = 0

    if export_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as out:
            for row in _iter_records_from_source(source_path, source_format, limit=limit):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count

    # csv export — two streaming passes so a large/unbounded export is never
    # fully materialized in memory, while the header still covers every column.
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        # Pass 1: union of keys across all rows (first-seen order); a heterogeneous
        # dataset would otherwise lose any column absent from the first row.
        fieldnames = []
        seen = set()
        any_row = False
        for row in _iter_records_from_source(source_path, source_format, limit=limit):
            any_row = True
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        if not any_row:
            out.write("")
            return 0
        # Pass 2: stream rows to the writer.
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        count = 0
        for row in _iter_records_from_source(source_path, source_format, limit=limit):
            writer.writerow({k: _to_csv_cell(v) for k, v in row.items()})
            count += 1
    return count


def _suffix_from_format(file_format: str) -> str:
    return {
        "jsonl": ".jsonl",
        "json": ".json",
        "csv": ".csv",
        "parquet": ".parquet",
        "arrow": ".arrow",
    }.get(file_format, ".data")


# === Request/Response Models ===

class CreateDatasetRequest(BaseModel):
    """Create dataset request."""
    dataset_name: str = Field(..., description="Dataset name")
    storage_path: Optional[str] = Field(default=None, description="Local storage path")
    storage_backend: Optional[Literal["local", "s3"]] = Field(default=None, description="Storage backend (defaults to system STORAGE_BACKEND setting)")
    storage_uri: Optional[str] = Field(default=None, description="Object storage URI (s3://bucket/key)")
    dataset_type: str = Field(
        default="custom",
        description="Dataset type: embedding_pair, embedding_triplet, rerank_pair, rerank_triplet, "
                    "rerank_listwise, sft_instruct, dpo_preference, rl_reward, custom"
    )
    usage: str = Field(default="train", description="Usage: raw (原始数据), train (训练数据集), eval (验证数据集), test (测试数据集)")
    model_type: Optional[List[str]] = Field(default=None, description="适用模型标签 (可多选): embedding, rerank, llm")
    source_type: str = Field(default="uploaded", description="Source type: uploaded, huggingface, modelscope, local")
    display_name: Optional[str] = Field(default=None, description="Display name")
    description: Optional[str] = Field(default=None, description="Description")
    source_path: Optional[str] = Field(default=None, description="Original source path")
    remote_repo: Optional[str] = Field(default=None, description="Remote repository (HF/MS)")
    hf_subset: Optional[str] = Field(default=None, description="HuggingFace subset name")
    file_format: str = Field(default="parquet", description="File format: parquet, jsonl, json, csv, arrow")
    columns: Optional[List[Dict[str, Any]]] = Field(default=None, description="Column schema")
    num_rows: Optional[int] = Field(default=None, description="Number of rows")
    num_train: Optional[int] = Field(default=None, description="Training split size")
    num_eval: Optional[int] = Field(default=None, description="Evaluation split size")
    num_test: Optional[int] = Field(default=None, description="Test split size")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    tags: Optional[List[str]] = Field(default=None, description="Tags")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extra metadata")
    source_task_type: Optional[str] = Field(default=None, description="Task type that produced this dataset")
    source_task_id: Optional[str] = Field(default=None, description="Task ID that produced this dataset")
    user_id: Optional[str] = Field(default=None, description="User ID")


class UpdateDatasetRequest(BaseModel):
    """Update dataset request."""
    display_name: Optional[str] = Field(default=None, description="Display name")
    description: Optional[str] = Field(default=None, description="Description")
    dataset_type: Optional[str] = Field(default=None, description="Dataset type")
    usage: Optional[str] = Field(default=None, description="Usage: raw, train, eval, test")
    model_type: Optional[List[str]] = Field(default=None, description="适用模型标签 (可多选): embedding, rerank, llm")
    tags: Optional[List[str]] = Field(default=None, description="Tags")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extra metadata")
    status: Optional[Literal["ready", "archived"]] = Field(
        default=None,
        description="Status",
    )


class DatasetResponse(BaseModel):
    """Dataset response."""
    dataset_id: str
    dataset_name: str
    display_name: Optional[str]
    description: Optional[str]
    dataset_type: str
    usage: str = "train"
    model_type: Optional[List[str]] = None
    source_type: str
    source_path: Optional[str]
    remote_repo: Optional[str]
    hf_subset: Optional[str]
    storage_path: Optional[str]
    storage_backend: str = "local"
    storage_uri: Optional[str] = None
    version: int = 1
    file_format: str
    columns: Optional[List[Dict[str, Any]]]
    sample_data: Optional[List[Dict[str, Any]]]
    content_schema: Optional[Dict[str, Any]] = None
    num_rows: Optional[int]
    num_train: Optional[int]
    num_eval: Optional[int]
    num_test: Optional[int]
    file_size: Optional[int]
    source_task_type: Optional[str] = None
    source_task_id: Optional[str] = None
    tags: Optional[List[str]]
    extra_metadata: Optional[Dict[str, Any]]
    status: str
    error_message: Optional[str]
    user_id: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class DatasetListResponse(BaseModel):
    """Dataset list response."""
    datasets: List[DatasetResponse]
    total: int
    # Global (user-scoped, filter-independent) stats for summary cards; optional
    # so search/other callers can omit it.
    stats: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    """Statistics response."""
    total: int
    by_type: Dict[str, int]
    by_source: Dict[str, int]
    by_status: Dict[str, int]
    by_usage: Dict[str, int] = {}
    total_size_bytes: int
    total_rows: int


class DownloadDatasetRequest(BaseModel):
    """Download dataset from remote request."""
    dataset_name: str = Field(..., min_length=1, max_length=255, description="Name for the dataset")
    remote_repo: str = Field(
        ...,
        min_length=3,
        max_length=96,
        description="Remote repository (e.g., 'sentence-transformers/all-nli')",
    )
    source_type: Literal["huggingface", "modelscope"] = Field(
        default="huggingface",
        description="Source: huggingface, modelscope",
    )
    hf_subset: Optional[str] = Field(default=None, min_length=1, max_length=128, description="HuggingFace subset name")
    dataset_type: str = Field(default="custom", min_length=1, max_length=64, description="Dataset type")
    usage: Literal["raw", "train", "eval", "test"] = Field(
        default="train",
        description="Usage: raw, train, eval, test",
    )
    model_type: Optional[List[Literal["embedding", "rerank", "llm"]]] = Field(
        default=None,
        max_length=8,
        description="适用模型标签 (可多选): embedding, rerank, llm",
    )
    output_format: Literal["jsonl", "json", "parquet", "arrow"] = Field(
        default="jsonl",
        description="Output format: jsonl, json, parquet, arrow",
    )
    clean_cache: bool = Field(default=True, description="Clean HuggingFace cache after download")
    description: Optional[str] = Field(default=None, max_length=4000, description="Description")
    tags: Optional[List[str]] = Field(default=None, max_length=32, description="Tags")
    user_id: Optional[str] = Field(default=None, max_length=36, description="User ID")


# === Helper Functions ===

def _dataset_to_response(dataset: Dict[str, Any]) -> DatasetResponse:
    """Convert dataset dict to response with unmapped storage path."""
    dataset = dict(dataset)  # Copy to avoid mutating original
    storage_backend = dataset.get("storage_backend", "local")

    # Defensive fallback for legacy rows that accidentally stored S3 URI in storage_path.
    if isinstance(dataset.get("storage_path"), str) and dataset["storage_path"].startswith("s3://"):
        if not dataset.get("storage_uri"):
            dataset["storage_uri"] = dataset["storage_path"]
        dataset["storage_path"] = None

    # Convert container path back to host path for display
    if dataset.get("storage_path") and storage_backend == "local":
        dataset["storage_path"] = unmap_storage_path(dataset["storage_path"])
    return DatasetResponse(**dataset)


# === API Endpoints ===

@router.post("/datasets", response_model=DatasetResponse)
async def create_dataset(
    request: CreateDatasetRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Create a new dataset record.

    Use this after uploading dataset files to register them in the system.
    Pass Idempotency-Key header to prevent duplicate creation on retries.
    """
    effective_backend = request.storage_backend or get_settings().storage_backend
    _validate_dataset_params(
        dataset_type=request.dataset_type,
        usage=request.usage,
        model_type_list=request.model_type,
        storage_backend=effective_backend,
    )
    if requires_tenant_provenance(current_user):
        raise HTTPException(
            status_code=403,
            detail=(
                "Direct dataset storage registration is disabled while authentication is enabled; "
                "use dataset upload, download, generation, or sync"
            ),
        )

    mapped_path: Optional[str] = None
    storage_uri_for_db: Optional[str] = None
    if effective_backend == "local":
        if not request.storage_path:
            raise HTTPException(
                status_code=400,
                detail="storage_path is required when storage_backend=local",
            )
        mapped_path, host_path = map_storage_path(request.storage_path)
        # Validate storage path to prevent path traversal
        validate_storage_path(mapped_path, resource_type="dataset")
        storage_path_for_db = mapped_path
        storage_uri_for_db = None
    else:  # s3
        if not request.storage_uri:
            raise HTTPException(
                status_code=400,
                detail="storage_uri is required when storage_backend=s3",
            )
        if not request.storage_uri.startswith("s3://"):
            raise HTTPException(
                status_code=400,
                detail="storage_uri must start with s3://",
            )
        from ...storage.object_store import parse_s3_uri

        try:
            bucket, key = parse_s3_uri(request.storage_uri)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        configured_bucket = get_settings().minio_bucket
        if bucket != configured_bucket:
            raise HTTPException(
                status_code=400,
                detail=(
                    "storage_uri bucket must match configured bucket "
                    f"'{configured_bucket}', got '{bucket}'"
                ),
            )
        storage_uri_for_db = f"s3://{bucket}/{key}"
        host_path = None
        storage_path_for_db = None

    # Use authenticated user_id
    user_id = current_user["user_id"]

    # Check idempotency
    is_duplicate, cached_response = check_idempotency(
        idempotency_key, user_id, "/api/datasets"
    )
    if is_duplicate and cached_response:
        return DatasetResponse(**cached_response)

    # Enforce unique storage reference (local path or S3 URI).
    if mapped_path:
        existing = dataset_service.get_dataset_by_storage_path(mapped_path)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Dataset path already registered: {mapped_path}"
            )
    elif storage_uri_for_db:
        existing = dataset_service.get_dataset_by_storage_path(storage_uri_for_db)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Dataset URI already registered: {storage_uri_for_db}"
            )

    dataset = dataset_service.create_dataset(
        dataset_name=request.dataset_name,
        storage_path=storage_path_for_db,
        dataset_type=request.dataset_type,
        usage=request.usage,
        model_type=request.model_type,
        source_type=request.source_type,
        display_name=request.display_name,
        description=request.description,
        source_path=request.source_path,
        remote_repo=request.remote_repo,
        hf_subset=request.hf_subset,
        file_format=request.file_format,
        columns=request.columns,
        num_rows=request.num_rows,
        num_train=request.num_train,
        num_eval=request.num_eval,
        num_test=request.num_test,
        file_size=request.file_size,
        tags=request.tags,
        storage_backend=effective_backend,
        storage_uri=storage_uri_for_db,
        source_task_type=request.source_task_type,
        source_task_id=request.source_task_id,
        extra_metadata={
            **(request.extra_metadata or {}),
            **({"host_storage_path": host_path} if host_path and host_path != mapped_path else {}),
        } or None,
        user_id=user_id,
    )

    # Register the primary data asset for object-storage datasets.
    if effective_backend == "s3" and storage_uri_for_db:
        try:
            dataset_asset_service.create_asset(
                dataset_id=dataset["dataset_id"],
                storage_uri=storage_uri_for_db,
                version=dataset.get("version", 1),
                asset_type="data",
                file_format=request.file_format,
                row_count=request.num_rows,
                byte_size=request.file_size,
            )
        except Exception as e:
            logger.warning(
                "Failed to register dataset asset for %s: %s",
                dataset["dataset_id"],
                e,
            )
    response = _dataset_to_response(dataset)

    # Store for idempotency
    store_idempotency_response(
        idempotency_key, user_id, "/api/datasets",
        response.model_dump()
    )

    return response


SUPPORTED_UPLOAD_EXTENSIONS = {".jsonl", ".json", ".csv", ".parquet"}


@router.post("/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file (.jsonl, .json, .csv, .parquet)"),
    dataset_name: str = Form(..., description="Dataset name"),
    dataset_type: str = Form(default="custom", description="Dataset type"),
    usage: str = Form(default="train", description="Usage: raw, train, eval, test"),
    model_type: Optional[str] = Form(default=None, description="Model type tags, comma-separated: embedding,rerank,llm"),
    storage_backend: Optional[str] = Form(default=None, description="Storage backend: local or s3"),
    description: Optional[str] = Form(default=None, description="Description"),
    tags: Optional[str] = Form(default=None, description="Tags, comma-separated"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Upload a dataset file and register it.

    Accepts a single file (.jsonl, .json, .csv, .parquet) along with metadata.
    The file is stored according to the effective storage backend (system default
    or explicitly specified).
    """
    settings = get_settings()
    user_id = current_user["user_id"]

    # Sanitize filename: strip path components to prevent path traversal
    raw_filename = file.filename or "data.jsonl"
    filename = Path(raw_filename).name
    if not filename:
        filename = "data.jsonl"

    # Validate file extension
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Must be one of {sorted(SUPPORTED_UPLOAD_EXTENSIONS)}",
        )
    file_format = suffix.lstrip(".")

    # Parse optional list fields
    model_type_list = (
        [t.strip() for t in model_type.split(",") if t.strip()]
        if model_type
        else None
    )
    tags_list = (
        [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    )

    # Validate parameters
    effective_backend = storage_backend or settings.storage_backend
    _validate_dataset_params(
        dataset_type=dataset_type,
        usage=usage,
        model_type_list=model_type_list,
        storage_backend=effective_backend,
    )

    dataset_id = str(uuid4())

    storage_path_for_db: Optional[str] = None
    storage_uri_for_db: Optional[str] = None
    file_size = 0

    max_size = settings.max_upload_size

    async def _read_chunks_to(dest_file, upload: UploadFile) -> int:
        """Stream upload chunks to a file, enforcing size limit. Returns bytes written."""
        written = 0
        while chunk := await upload.read(64 * 1024):
            written += len(chunk)
            if written > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum upload size is {max_size // (1024 * 1024)}MB",
                )
            dest_file.write(chunk)
        return written

    local_save_dir: Optional[Path] = None
    local_stored_filename: Optional[str] = None

    if effective_backend == "s3":
        # Save to temp file, then upload to S3
        tmp_path = Path(tempfile.NamedTemporaryFile(
            prefix="upload-", suffix=suffix, delete=False,
        ).name)
        try:
            with open(tmp_path, "wb") as f:
                file_size = await _read_chunks_to(f, file)

            from ...storage.object_store import get_object_store

            store = get_object_store()
            object_key = f"datasets/{dataset_id}/{filename}"
            storage_uri_for_db = store.upload_file(str(tmp_path), object_key)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        # Save to local datasets directory
        local_save_dir = Path(settings.datasets_dir) / dataset_id
        local_save_dir.mkdir(parents=True, exist_ok=True)
        split_prefix = "train"
        if usage == "eval":
            split_prefix = "eval"
        elif usage == "test":
            split_prefix = "test"
        local_stored_filename = f"{split_prefix}{suffix}"
        file_path = local_save_dir / local_stored_filename

        try:
            with open(file_path, "wb") as f:
                file_size = await _read_chunks_to(f, file)
        except Exception:
            # Cleanup partial local upload when streaming fails (e.g. size limit).
            shutil.rmtree(local_save_dir, ignore_errors=True)
            raise
        storage_path_for_db = str(local_save_dir)

    try:
        dataset = dataset_service.create_dataset(
            dataset_name=dataset_name,
            storage_path=storage_path_for_db,
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            usage=usage,
            model_type=model_type_list,
            source_type="uploaded",
            display_name=dataset_name,
            description=description,
            file_format=file_format,
            file_size=file_size,
            tags=tags_list,
            storage_backend=effective_backend,
            storage_uri=storage_uri_for_db,
            user_id=user_id,
            status="uploading",
            extra_metadata={
                "original_filename": filename,
                **(
                    {"stored_filename": local_stored_filename}
                    if local_stored_filename
                    else {}
                ),
            },
        )
    except Exception:
        # Roll back uploaded objects/files when DB registration fails.
        backend = get_storage_backend(effective_backend)
        if effective_backend == "s3" and storage_uri_for_db:
            backend.delete(storage_uri_for_db, dataset_id=dataset_id)
        elif storage_path_for_db:
            backend.delete(storage_path_for_db, dataset_id=dataset_id)
        raise

    # Register S3 asset
    if effective_backend == "s3" and storage_uri_for_db:
        try:
            dataset_asset_service.create_asset(
                dataset_id=dataset["dataset_id"],
                storage_uri=storage_uri_for_db,
                version=dataset.get("version", 1),
                asset_type="data",
                file_format=file_format,
                byte_size=file_size,
            )
        except Exception as e:
            logger.warning(
                "Failed to register upload asset for %s: %s",
                dataset["dataset_id"],
                e,
            )

    # Upload completed successfully -> move uploading to ready.
    dataset_service.update_status(dataset["dataset_id"], "ready")
    dataset = dataset_service.get_dataset(dataset["dataset_id"])
    if not dataset:
        raise HTTPException(status_code=500, detail="Dataset registration lost after upload")

    return _dataset_to_response(dataset)


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    dataset_type: Optional[str] = None,
    usage: Optional[str] = None,
    model_type: Optional[str] = Query(default=None, description="Filter by model type tag (single value, matched via JSON_CONTAINS)"),
    source_type: Optional[str] = None,
    status: Optional[str] = None,
    source_task_type: Optional[str] = None,
    source_task_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all datasets for the current user with optional filters."""
    # Use authenticated user_id
    user_id = current_user["user_id"]

    datasets, total = dataset_service.list_datasets(
        dataset_type=dataset_type,
        usage=usage,
        model_type=model_type,
        source_type=source_type,
        status=status,
        source_task_type=source_task_type,
        source_task_id=source_task_id,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    return DatasetListResponse(
        datasets=[_dataset_to_response(d) for d in datasets],
        total=total,
        stats=dataset_service.get_stats(user_id=user_id),
    )


@router.get("/datasets/search", response_model=DatasetListResponse)
async def search_datasets(
    query: str = Query(..., description="Search query"),
    dataset_type: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Search datasets by name or description."""
    # Use authenticated user_id
    user_id = current_user["user_id"]

    datasets = dataset_service.search_datasets(
        query=query,
        dataset_type=dataset_type,
        user_id=user_id,
        limit=limit,
    )
    return DatasetListResponse(
        datasets=[_dataset_to_response(d) for d in datasets],
        total=len(datasets),
    )


@router.get("/datasets/stats", response_model=StatsResponse)
async def get_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get dataset statistics for the current user."""
    user_id = current_user["user_id"]
    return dataset_service.get_stats(user_id=user_id)


@router.get("/datasets/types")
async def get_dataset_types():
    """Get supported dataset types with descriptions."""
    return {
        # Embedding types (10 types)
        "embedding_universal": {
            "model_type": "embedding",
            "description": "Universal format (query, pos[], neg[], scores optional)",
            "columns": ["query", "pos", "neg"],
            "optional_columns": ["pos_scores", "neg_scores"],
            "example_losses": ["MultipleNegativesRankingLoss", "AnglELoss"]
        },
        "embedding_pair": {
            "model_type": "embedding",
            "description": "MNR pairs for embedding training (query, positive)",
            "columns": ["query", "positive"],
            "example_losses": ["MultipleNegativesRankingLoss", "CosineSimilarityLoss"]
        },
        "embedding_triplet": {
            "model_type": "embedding",
            "description": "MNR triplets for embedding training (query, positive, negative)",
            "columns": ["query", "positive", "negative"],
            "example_losses": ["TripletLoss", "MultipleNegativesRankingLoss"]
        },
        "embedding_multi_neg": {
            "model_type": "embedding",
            "description": "MNR with multiple negatives, consistent count (query, pos, neg[])",
            "columns": ["query", "positive", "negatives"],
            "example_losses": ["MultipleNegativesRankingLoss", "InfoNCELoss"]
        },
        "embedding_dynamic_neg": {
            "model_type": "embedding",
            "description": "Dynamic negatives with variable count (query, pos, neg[])",
            "columns": ["query", "positive", "negatives"],
            "example_losses": ["MultipleNegativesRankingLoss", "InfoNCELoss"]
        },
        "embedding_cosine": {
            "model_type": "embedding",
            "description": "Cosine similarity pairs with labels (text1, text2, label)",
            "columns": ["text1", "text2", "label"],
            "example_losses": ["CosineSimilarityLoss"]
        },
        "embedding_margin": {
            "model_type": "embedding",
            "description": "Margin triplets (anchor, pos, neg, margin)",
            "columns": ["anchor", "positive", "negative", "margin"],
            "example_losses": ["TripletMarginLoss"]
        },
        "embedding_margin_multi": {
            "model_type": "embedding",
            "description": "Margin with multiple negatives (anchor, pos[], neg[], margins[])",
            "columns": ["anchor", "positives", "negatives", "margins"],
            "example_losses": ["MarginMSELoss"]
        },
        "embedding_scored": {
            "model_type": "embedding",
            "description": "Multi-scored format (anchor, pos[], neg[], scores[])",
            "columns": ["anchor", "positives", "negatives", "scores"],
            "example_losses": ["MarginMSELoss", "AnglELoss"]
        },
        "embedding_score_triplet": {
            "model_type": "embedding",
            "description": "Score triplets (query, pos, neg, [pos_score, neg_score])",
            "columns": ["query", "positive", "negative", "label"],
            "example_losses": ["MarginMSELoss"]
        },
        # Rerank types
        "rerank_pair": {
            "model_type": "rerank",
            "description": "Query-document pairs with labels for reranking",
            "columns": ["query", "document", "label"],
            "example_losses": ["CrossEntropyLoss", "MarginMSELoss"]
        },
        "rerank_triplet": {
            "model_type": "rerank",
            "description": "Query with positive and negative passages for contrastive reranking",
            "columns": ["query", "positives", "negatives"],
            "example_losses": ["MultipleNegativesRankingLoss", "InfoNCELoss"]
        },
        "rerank_listwise": {
            "model_type": "rerank",
            "description": "Query with ranked document lists for listwise reranking",
            "columns": ["query", "documents", "labels"],
            "example_losses": ["LambdaLoss", "ListMLELoss"]
        },
        # LLM types
        "sft_instruct": {
            "model_type": "llm",
            "description": "Instruction-response pairs for supervised fine-tuning",
            "columns": ["instruction", "response"],
            "example_format": "alpaca, sharegpt"
        },
        "dpo_preference": {
            "model_type": "llm",
            "description": "Preference pairs for DPO training",
            "columns": ["prompt", "chosen", "rejected"],
            "example_losses": ["DPOLoss"]
        },
        "rl_reward": {
            "model_type": "llm",
            "description": "Samples with reward signals for RL training",
            "columns": ["prompt", "response", "reward"],
            "example_methods": ["GRPO", "DAPO", "DR_GRPO"]
        },
        # Generation intermediate types
        "qa_pair": {
            "model_type": None,
            "description": "QA pairs extracted from raw data (Phase 1), with chunk references",
            "columns": ["query", "answer", "chunk_id", "chunk_content"],
            "example_usage": "Used as input for qa_to_training"
        },
        "custom": {
            "model_type": None,
            "description": "Custom dataset format",
            "columns": "user-defined"
        }
    }


# === Download from Remote (must be before /datasets/{dataset_id} to avoid route conflict) ===

class DownloadProgressResponse(BaseModel):
    """Download progress response."""
    dataset_id: str
    dataset_name: Optional[str]
    source_type: Optional[str]
    remote_repo: Optional[str]
    storage_path: Optional[str]
    status: str
    progress: int
    error: Optional[str]
    created_at: Optional[str]


class DownloadResponse(BaseModel):
    """Download response."""
    dataset_id: str
    dataset_name: str
    status: str
    storage_path: str


class ExportDatasetRequest(BaseModel):
    """Dataset export request."""

    format: Literal["original", "jsonl", "csv"] = Field(
        default="jsonl",
        description="Export format: original/jsonl/csv",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional max rows to export",
    )
    expires_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Presigned URL expiry in seconds",
    )


class ExportDatasetResponse(BaseModel):
    """Dataset export response."""

    dataset_id: str
    source_format: str
    export_format: str
    row_count: Optional[int] = None
    storage_uri: str
    download_url: str
    proxy_download_url: str
    expires_seconds: int


@router.post("/datasets/download", response_model=DownloadResponse)
async def download_dataset(
    request: DownloadDatasetRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Download dataset from remote repository (HuggingFace/ModelScope).

    The dataset will be downloaded in the background and registered automatically.
    Use the progress endpoint to check download status.
    """
    from ...storage.services.dataset_download_service import dataset_download_service

    # Use authenticated user_id
    user_id = current_user["user_id"]

    # Validate source type
    if request.source_type not in ["huggingface", "modelscope"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source type: {request.source_type}. Use 'huggingface' or 'modelscope'."
        )

    if requires_tenant_provenance(current_user):
        try:
            validate_remote_repo_id(request.remote_repo)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await asyncio.to_thread(
            dataset_download_service.start_download,
            source_type=request.source_type,
            remote_repo=request.remote_repo,
            dataset_name=request.dataset_name,
            dataset_type=request.dataset_type,
            usage=request.usage,
            model_type=request.model_type,
            hf_subset=request.hf_subset,
            output_format=request.output_format,
            clean_cache=request.clean_cache,
            description=request.description,
            user_id=user_id,
        )

        logger.info(f"Dataset download started: dataset_id={result['dataset_id']}, repo={request.remote_repo}")

        return DownloadResponse(
            dataset_id=result["dataset_id"],
            dataset_name=result["dataset_name"],
            status=result["status"],
            storage_path=result["storage_path"],
        )

    except DownloadConcurrencyExceeded as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": "5"},
        ) from e
    except DownloadSizeExceeded as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except DownloadMetadataUnavailable as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except DownloadPolicyViolation as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except DownloadStorageQuotaExceeded as e:
        raise HTTPException(status_code=507, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Failed to start dataset download (error_type=%s)",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to start remote dataset download",
        ) from e


@router.get("/datasets/download/{dataset_id}/progress", response_model=DownloadProgressResponse)
async def get_download_progress(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get download progress for a dataset."""
    # Verify ownership of the dataset
    dataset = dataset_service.get_dataset(dataset_id)
    dataset = verify_resource_ownership(dataset, current_user, "Dataset")

    from ...storage.services.dataset_download_service import dataset_download_service

    progress = dataset_download_service.get_download_progress(dataset_id)
    if not progress:
        raise HTTPException(status_code=404, detail=f"Download task not found: {dataset_id}")

    return DownloadProgressResponse(
        dataset_id=progress.get("dataset_id", dataset_id),
        dataset_name=progress.get("dataset_name"),
        source_type=progress.get("source_type"),
        remote_repo=progress.get("remote_repo"),
        storage_path=progress.get("storage_path"),
        status=progress.get("status", "unknown"),
        progress=progress.get("progress", 0),
        error=progress.get("error"),
        created_at=progress.get("created_at"),
    )


@router.get("/datasets/downloads")
async def list_downloads(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all dataset download tasks."""
    from ...storage.services.dataset_download_service import dataset_download_service

    downloads = dataset_download_service.list_downloads(
        user_id=current_user.get("user_id")
    )
    return {"downloads": downloads, "total": len(downloads)}


@router.post("/datasets/{dataset_id}/export", response_model=ExportDatasetResponse)
async def export_dataset(
    dataset_id: str,
    request: ExportDatasetRequest,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Export dataset to readable format and return a presigned download URL."""
    from ...storage.object_store import get_object_store, uri_to_key

    dataset = dataset_service.get_dataset(dataset_id)
    dataset = verify_resource_ownership(dataset, current_user, "Dataset")

    source_format = (dataset.get("file_format") or "jsonl").lower()
    if source_format not in SUPPORTED_EXPORT_SOURCE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source format for export: {source_format}",
        )

    storage_backend = dataset.get("storage_backend", "local")
    storage_path = dataset.get("storage_path")
    storage_uri = dataset.get("storage_uri")

    # Legacy fallback: some old rows may still keep s3://... in storage_path.
    if not storage_uri and isinstance(storage_path, str) and storage_path.startswith("s3://"):
        storage_uri = storage_path
        storage_backend = "s3"

    store = get_object_store()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_user_id = current_user.get("user_id") or "anonymous"
    base_name = (dataset.get("dataset_name") or dataset_id).replace(" ", "_")

    def _proxy_url(uri: str) -> str:
        quoted_uri = quote(uri, safe="")
        return str(http_request.url_for("download_export_file", dataset_id=dataset_id)) + f"?storage_uri={quoted_uri}"

    temp_paths: List[Path] = []
    try:
        if (
            storage_backend == "s3"
            and source_format == "arrow"
            and request.format != "original"
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "S3 Arrow dataset export to jsonl/csv is not supported yet. "
                    "Please use format=original or register a local Arrow directory dataset."
                ),
            )

        # Fast path: original format on S3 can be signed directly.
        if request.format == "original" and storage_backend == "s3" and storage_uri:
            try:
                object_key = uri_to_key(storage_uri, expected_bucket=store.bucket)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            url = store.get_presigned_url(object_key, expires=request.expires_seconds)
            return ExportDatasetResponse(
                dataset_id=dataset_id,
                source_format=source_format,
                export_format=source_format,
                row_count=dataset.get("num_rows"),
                storage_uri=storage_uri,
                download_url=url,
                proxy_download_url=_proxy_url(storage_uri),
                expires_seconds=request.expires_seconds,
            )

        reference = storage_uri if storage_backend == "s3" else storage_path
        if not reference:
            raise HTTPException(
                status_code=400,
                detail="No storage path/URI configured for dataset",
            )
        try:
            export_backend = get_storage_backend(storage_backend)
            source_path, source_fmt = export_backend.resolve_export_source(
                reference, source_format
            )
            if storage_backend == "s3":
                temp_paths.append(source_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if request.format == "original":
            if source_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail="Original export is only supported for single-file datasets",
                )
            export_format = source_fmt
            out_suffix = _suffix_from_format(export_format)
            out_tmp = Path(tempfile.NamedTemporaryFile(prefix="dataset-export-", suffix=out_suffix, delete=False).name)
            temp_paths.append(out_tmp)
            shutil.copyfile(source_path, out_tmp)
            row_count = dataset.get("num_rows")
        else:
            export_format = request.format
            out_suffix = ".jsonl" if export_format == "jsonl" else ".csv"
            out_tmp = Path(tempfile.NamedTemporaryFile(prefix="dataset-export-", suffix=out_suffix, delete=False).name)
            temp_paths.append(out_tmp)
            row_count = _write_export_file(
                source_path=source_path,
                source_format=source_fmt,
                output_path=out_tmp,
                export_format=export_format,
                limit=request.limit,
            )

        export_key = (
            f"exports/{safe_user_id}/{dataset_id}/"
            f"{now}_{uuid4().hex}_{base_name}{out_suffix}"
        )
        export_uri = store.upload_file(str(out_tmp), export_key)
        download_url = store.get_presigned_url(export_key, expires=request.expires_seconds)

        return ExportDatasetResponse(
            dataset_id=dataset_id,
            source_format=source_fmt,
            export_format=export_format,
            row_count=row_count,
            storage_uri=export_uri,
            download_url=download_url,
            proxy_download_url=_proxy_url(export_uri),
            expires_seconds=request.expires_seconds,
        )
    finally:
        for p in temp_paths:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                logger.warning("Failed to cleanup temp export file: %s", p)


@router.get("/datasets/{dataset_id}/export/download", name="download_export_file")
async def download_export_file(
    dataset_id: str,
    storage_uri: str = Query(..., description="Exported object storage URI"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Proxy-download an exported dataset object through API."""
    from minio.error import S3Error

    from ...storage.object_store import get_object_store, parse_s3_uri, uri_to_key

    dataset = dataset_service.get_dataset(dataset_id)
    dataset = verify_resource_ownership(dataset, current_user, "Dataset")

    store = get_object_store()
    if not storage_uri.startswith("s3://"):
        raise HTTPException(status_code=400, detail="storage_uri must start with s3://")
    try:
        parse_s3_uri(storage_uri)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        object_key = uri_to_key(storage_uri, expected_bucket=store.bucket)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    allowed_uris = set()
    if dataset.get("storage_uri"):
        allowed_uris.add(dataset["storage_uri"])
    if isinstance(dataset.get("storage_path"), str) and dataset["storage_path"].startswith("s3://"):
        allowed_uris.add(dataset["storage_path"])

    try:
        for asset in dataset_asset_service.list_assets(dataset_id=dataset_id):
            uri = asset.get("storage_uri")
            if uri:
                allowed_uris.add(uri)
    except Exception:
        # Non-critical: exports can still be authorized by prefix below.
        pass

    safe_user_id = current_user.get("user_id") or "anonymous"
    export_prefix = f"s3://{store.bucket}/exports/{safe_user_id}/{dataset_id}/"
    if storage_uri not in allowed_uris and not storage_uri.startswith(export_prefix):
        raise HTTPException(status_code=403, detail="Not authorized to download this exported object")

    try:
        response = store.client.get_object(store.bucket, object_key)
    except S3Error as e:
        if getattr(e, "code", "") in {"NoSuchKey", "NoSuchBucket"}:
            raise HTTPException(status_code=404, detail="Export object not found")
        logger.warning("Failed to fetch export object %s: %s", storage_uri, e)
        raise HTTPException(status_code=502, detail="Failed to fetch export object")
    filename = Path(object_key).name or f"{dataset_id}.bin"
    media_type = "application/octet-stream"
    if filename.endswith(".jsonl"):
        media_type = "application/x-ndjson"
    elif filename.endswith(".csv"):
        media_type = "text/csv"
    elif filename.endswith(".json"):
        media_type = "application/json"

    def _stream():
        try:
            for chunk in response.stream(amt=64 * 1024):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    return StreamingResponse(
        _stream(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# === Dataset CRUD (dynamic routes must be after static routes) ===

@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get dataset by ID."""
    dataset = dataset_service.get_dataset(dataset_id)
    dataset = verify_resource_ownership(dataset, current_user, "Dataset")
    return _dataset_to_response(dataset)


@router.put("/datasets/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    request: UpdateDatasetRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Update dataset information."""
    # Verify ownership before update
    dataset = dataset_service.get_dataset(dataset_id)
    dataset = verify_resource_ownership(dataset, current_user, "Dataset")

    public_statuses = {"ready", "archived"}
    if request.status is not None and (
        dataset.get("status") not in public_statuses
        or request.status not in public_statuses
    ):
        raise HTTPException(
            status_code=409,
            detail="Dataset status cannot be changed from its current state",
        )

    try:
        outcome = dataset_service.update_dataset(
            dataset_id=dataset_id,
            expected_user_id=dataset.get("user_id"),
            expected_status=dataset["status"],
            expected_deletion_owner=None,
            display_name=request.display_name,
            description=request.description,
            dataset_type=request.dataset_type,
            usage=request.usage,
            model_type=request.model_type,
            tags=request.tags,
            extra_metadata=request.extra_metadata,
            status=request.status,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if outcome.dataset is not None:
        return _dataset_to_response(outcome.dataset)
    if outcome.missing:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    if outcome.ownership_mismatch:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this dataset",
        )
    raise HTTPException(
        status_code=409,
        detail="Dataset changed while the update was in progress",
    )


async def _delete_dataset_impl(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a dataset and its associated files."""
    deletion_owner = f"dataset:{dataset_id}"
    dataset = dataset_service.get_dataset(dataset_id)
    dataset = verify_resource_ownership(dataset, current_user, "Dataset")
    original_status = dataset.get("status") or "registered"
    if original_status in {"uploading", "downloading", "processing"}:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot delete dataset while background storage writes may be active"
            ),
        )
    delete_was_already_pending = original_status == "deleting"
    owner_user_id = dataset.get("user_id")

    try:
        marked = dataset_service.mark_deleting(
            dataset_id,
            deletion_owner=deletion_owner,
            user_id=owner_user_id,
        )
    except DatasetDeletionOwnerConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Dataset is owned by another deletion operation",
        ) from exc
    except Exception as exc:
        logger.error(
            "Failed to mark dataset %s for deletion (error_type=%s)",
            dataset_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Dataset deletion could not be started",
        ) from exc
    if not marked:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    try:
        assets = dataset_asset_service.list_assets(dataset_id=dataset_id)
    except Exception as exc:
        logger.error(
            "Failed to list dataset assets for %s (error_type=%s)",
            dataset_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Dataset metadata cleanup failed; record retained",
        ) from exc

    dataset_paths = sorted(
        {
            reference
            for reference in (
                dataset.get("storage_path"),
                dataset.get("storage_uri"),
                *(asset.get("storage_uri") for asset in assets),
            )
            if isinstance(reference, str) and reference.strip()
        }
    )
    try:
        active_generation_consumers = (
            generation_task_service.list_active_dataset_consumers([dataset_id])
        )
        active_training_consumers = (
            training_task_service.list_active_dataset_consumers(dataset_paths)
        )
        active_evaluation_consumers = (
            evaluation_task_service.list_active_dataset_consumers(
                [dataset_id],
                dataset_paths,
            )
        )
        milvus_links = milvus_collection_service.list_dataset_links([dataset_id])
    except Exception as exc:
        logger.error(
            "Failed to inspect dependencies for dataset %s (error_type=%s)",
            dataset_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Dataset dependency check failed; record retained",
        ) from exc

    if (
        active_generation_consumers
        or active_training_consumers
        or active_evaluation_consumers
        or milvus_links
    ):
        if not delete_was_already_pending:
            try:
                restored = dataset_service.restore_from_deleting(
                    dataset_id,
                    deletion_owner=deletion_owner,
                    status=original_status,
                    user_id=owner_user_id,
                )
            except DatasetDeletionOwnerConflictError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Dataset is owned by another deletion operation",
                ) from exc
            except Exception as exc:
                logger.error(
                    "Failed to restore dataset %s after deletion conflict "
                    "(error_type=%s)",
                    dataset_id,
                    type(exc).__name__,
                )
                raise HTTPException(
                    status_code=500,
                    detail="Dataset deletion conflict could not be rolled back",
                ) from exc
            if not restored:
                raise HTTPException(
                    status_code=500,
                    detail="Dataset deletion conflict could not be rolled back",
                )
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot delete dataset with active consumers or Milvus links. "
                f"generation={active_generation_consumers[:5]}, "
                f"training={active_training_consumers[:5]}, "
                f"evaluation={active_evaluation_consumers[:5]}, "
                f"milvus_link_count={len(milvus_links)}"
            ),
        )

    storage_backend_type = dataset.get("storage_backend", "local")
    storage_path = dataset.get("storage_path")
    storage_uri = dataset.get("storage_uri")
    try:
        backend = get_storage_backend(storage_backend_type)
        if storage_backend_type == "s3":
            storage_references = {
                reference
                for reference in (
                    storage_uri,
                    *(asset.get("storage_uri") for asset in assets),
                )
                if isinstance(reference, str) and reference.strip()
            }
            for reference in sorted(storage_references):
                if not backend.delete(reference):
                    raise RuntimeError("storage backend rejected dataset cleanup")
        elif storage_path and not backend.delete(
            storage_path,
            dataset_id=dataset.get("dataset_id"),
        ):
            raise RuntimeError("storage backend rejected dataset cleanup")
    except Exception as exc:
        logger.error(
            "Failed to delete storage for dataset %s (error_type=%s)",
            dataset_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Dataset storage cleanup failed; record retained",
        ) from exc

    try:
        dataset_asset_service.delete_assets_for_dataset(dataset_id)
    except Exception as exc:
        logger.error(
            "Failed to delete dataset assets for %s (error_type=%s)",
            dataset_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Dataset metadata cleanup failed; record retained",
        ) from exc

    try:
        dataset_lineage_service.delete_edges_for_dataset(dataset_id)
    except Exception as exc:
        logger.error(
            "Failed to delete dataset lineage for %s (error_type=%s)",
            dataset_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Dataset metadata cleanup failed; record retained",
        ) from exc

    try:
        success = dataset_service.delete_dataset(
            dataset_id,
            deletion_owner=deletion_owner,
            user_id=owner_user_id,
        )
    except DatasetDeletionOwnerConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Dataset is owned by another deletion operation",
        ) from exc
    except Exception as exc:
        logger.error(
            "Failed to delete dataset record %s (error_type=%s)",
            dataset_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Dataset database cleanup failed; record retained",
        ) from exc
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Dataset database cleanup failed; record retained",
        )
    return {"message": f"Dataset {dataset_id} deleted"}


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a dataset while blocking same-process concurrent cleanup."""
    try:
        deletion_guard = _begin_dataset_deletion(dataset_id)
    except _DatasetDeletionAlreadyRunning as exc:
        raise HTTPException(
            status_code=409,
            detail="Dataset deletion is already in progress",
        ) from exc
    try:
        return await _delete_dataset_impl(dataset_id, current_user)
    finally:
        deletion_guard.release()


# ── Lineage API ──


def _filter_visible_lineage_edges(
    edges: List[Dict[str, Any]],
    current_user_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Return edges whose dataset endpoints are publicly visible to the user."""
    endpoint_ids = {
        endpoint_id
        for edge in edges
        for endpoint_id in (
            edge.get("from_dataset_id"),
            edge.get("to_dataset_id"),
        )
        if endpoint_id is not None
    }
    visible_node_ids = set()
    for endpoint_id in endpoint_ids:
        dataset = dataset_service.get_dataset(endpoint_id)
        if not dataset:
            continue
        if current_user_id and dataset.get("user_id") != current_user_id:
            continue
        visible_node_ids.add(dataset.get("dataset_id"))

    return [
        edge
        for edge in edges
        if edge.get("to_dataset_id") in visible_node_ids
        and (
            edge.get("from_dataset_id") is None
            or edge.get("from_dataset_id") in visible_node_ids
        )
    ]


@router.get("/datasets/{dataset_id}/upstream")
async def get_dataset_upstream(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get upstream edges (who produced this dataset)."""
    dataset = dataset_service.get_dataset(dataset_id)
    verify_resource_ownership(dataset, current_user, "Dataset")
    edges = dataset_lineage_service.get_upstream(dataset_id)
    return {
        "edges": _filter_visible_lineage_edges(
            edges,
            current_user.get("user_id"),
        )
    }


@router.get("/datasets/{dataset_id}/downstream")
async def get_dataset_downstream(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get downstream edges (what this dataset produced)."""
    dataset = dataset_service.get_dataset(dataset_id)
    verify_resource_ownership(dataset, current_user, "Dataset")
    edges = dataset_lineage_service.get_downstream(dataset_id)
    return {
        "edges": _filter_visible_lineage_edges(
            edges,
            current_user.get("user_id"),
        )
    }


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Number of rows to preview"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Preview dataset contents.

    Returns the first N rows of the dataset.
    """
    dataset = dataset_service.get_dataset(dataset_id)
    dataset = verify_resource_ownership(dataset, current_user, "Dataset")

    # Return cached sample_data only when it already holds enough rows for this
    # request; a small first cache (e.g. 10) must not permanently cap a later
    # larger preview. Otherwise fall through and reload from the backend.
    cached_sample = dataset.get("sample_data")
    cached_num_rows = dataset.get("num_rows")
    cache_is_complete = (
        isinstance(cached_num_rows, int)
        and cached_num_rows >= 0
        and cached_num_rows <= len(cached_sample or [])
    )
    if cached_sample and (len(cached_sample) >= limit or cache_is_complete):
        return {
            "dataset_id": dataset_id,
            "columns": dataset.get("columns", []),
            "rows": cached_sample[:limit],
            "total_rows": dataset.get("num_rows", 0),
        }

    storage_backend_type = dataset.get("storage_backend", "local")
    storage_path = dataset.get("storage_path")
    storage_uri = dataset.get("storage_uri")
    file_format = dataset.get("file_format", "jsonl")

    reference = storage_uri if storage_backend_type == "s3" else storage_path
    if not reference:
        return {
            "dataset_id": dataset_id,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "message": "No storage path/URI configured.",
        }

    backend = get_storage_backend(storage_backend_type)
    preview_data = backend.load_preview(reference, file_format, limit)

    if "error" in preview_data:
        return {
            "dataset_id": dataset_id,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "message": preview_data["error"]
        }

    # Cache the preview data and update dataset info
    if preview_data.get("rows"):
        current_status = dataset.get("status")
        # The conditional write promotes a stable registered snapshot and may
        # expand a stable ready cache. Writers/deletion and storage replacement
        # win the race because every source-identity field is part of the CAS.
        if current_status in {"registered", "ready"}:
            dataset_service.cache_preview_if_snapshot_matches(
                dataset_id,
                expected_status=current_status,
                expected_storage_backend=storage_backend_type,
                expected_storage_path=storage_path,
                expected_storage_uri=storage_uri,
                expected_file_format=file_format,
                expected_version=dataset.get("version", 1),
                columns=preview_data.get("columns"),
                sample_data=preview_data.get("rows"),
                num_rows=preview_data.get("num_rows"),
                num_train=preview_data.get("num_train"),
                num_eval=preview_data.get("num_eval"),
                num_test=preview_data.get("num_test"),
                file_size=preview_data.get("file_size"),
            )

    total_rows = preview_data.get("num_rows")
    if total_rows is None:
        total_rows = dataset.get("num_rows")
    if total_rows is None:
        total_rows = len(preview_data.get("rows", []))

    return {
        "dataset_id": dataset_id,
        "columns": preview_data.get("columns", []),
        "rows": preview_data.get("rows", []),
        "total_rows": total_rows,
    }


@router.get("/datasets/{dataset_id}/lineage")
async def get_dataset_lineage(
    dataset_id: str,
    direction: Literal["upstream", "downstream", "both"] = Query(
        default="both", description="Lineage direction",
    ),
    depth: int = Query(default=2, ge=1, le=6, description="Max traversal depth"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get dataset lineage graph."""
    dataset = dataset_service.get_dataset(dataset_id)
    dataset = verify_resource_ownership(dataset, current_user, "Dataset")

    graph = dataset_lineage_service.get_lineage_graph(
        dataset_id=dataset["dataset_id"],
        direction=direction,
        depth=depth,
    )

    # Materialize dataset summaries for all nodes the current user can access.
    current_user_id = current_user.get("user_id")
    node_items = []
    visible_node_ids = set()
    for node_id in graph.get("nodes", []):
        ds = dataset_service.get_dataset(node_id)
        if not ds:
            continue
        if current_user_id and ds.get("user_id") != current_user_id:
            continue
        visible_node_ids.add(ds.get("dataset_id"))
        node_items.append(
            {
                "dataset_id": ds.get("dataset_id"),
                "dataset_name": ds.get("dataset_name"),
                "dataset_type": ds.get("dataset_type"),
                "usage": ds.get("usage"),
                "source_task_type": ds.get("source_task_type"),
                "source_task_id": ds.get("source_task_id"),
                "created_at": ds.get("created_at"),
            }
        )

    visible_edges = [
        edge
        for edge in graph.get("edges", [])
        if edge.get("to_dataset_id") in visible_node_ids
        and (
            edge.get("from_dataset_id") is None
            or edge.get("from_dataset_id") in visible_node_ids
        )
    ]

    return {
        "dataset_id": dataset_id,
        "direction": direction,
        "depth": depth,
        "nodes": node_items,
        "edges": visible_edges,
    }
