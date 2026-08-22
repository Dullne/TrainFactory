"""S3/MinIO object storage backend."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import StorageBackend

logger = logging.getLogger(__name__)


class S3StorageBackend(StorageBackend):
    """S3/MinIO storage backend for dataset operations."""

    def exists(self, reference: str) -> bool:
        from ..object_store import get_object_store, uri_to_key

        try:
            store = get_object_store()
            key = uri_to_key(reference, expected_bucket=store.bucket)
            return store.object_exists(key)
        except Exception:
            return False

    def delete(self, reference: str, dataset_id: Optional[str] = None) -> bool:
        """Delete an object by s3://bucket/key URI."""
        from ..object_store import get_object_store, uri_to_key

        if not reference or not reference.startswith("s3://"):
            return not reference
        try:
            store = get_object_store()
            object_key = uri_to_key(reference, expected_bucket=store.bucket)
            store.delete_object(object_key)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete object {reference}: {e}")
            return False

    def load_preview(
        self, reference: str, file_format: str, limit: int = 10
    ) -> Dict[str, Any]:
        """Load preview rows from object storage.

        Supports text line-based formats (jsonl, csv).
        """
        from ..object_store import get_object_store, uri_to_key

        if file_format not in {"jsonl", "csv"}:
            return {"error": f"S3 preview does not support format: {file_format}"}

        try:
            store = get_object_store()
            try:
                key = uri_to_key(reference, expected_bucket=store.bucket)
            except ValueError as e:
                return {"error": str(e)}

            rows: List[Dict[str, Any]] = []
            columns: List[Dict[str, Any]] = []

            if file_format == "jsonl":
                for line in store.stream_lines(key, limit=limit):
                    row = json.loads(line)
                    rows.append(row)
                    if not columns and isinstance(row, dict):
                        columns = [
                            {"name": k, "type": type(v).__name__}
                            for k, v in row.items()
                        ]
            else:  # csv
                import csv

                raw_lines = list(store.stream_lines(key, limit=limit + 1))
                if raw_lines:
                    reader = csv.DictReader(raw_lines)
                    for i, row in enumerate(reader):
                        if i >= limit:
                            break
                        rows.append(dict(row))
                    if reader.fieldnames:
                        columns = [
                            {"name": name, "type": "str"}
                            for name in reader.fieldnames
                        ]

            return {
                "columns": columns,
                "rows": rows,
                # S3 preview is streaming: do not overwrite dataset num_rows.
                "num_rows": None,
            }
        except Exception as e:
            logger.error(f"Error loading S3 preview for {reference}: {e}")
            return {"error": str(e)}

    def resolve_export_source(
        self, reference: str, declared_format: str
    ) -> Tuple[Path, str]:
        """Download S3 object to temp file and return (temp_path, format).

        The caller is responsible for cleaning up the returned temp file.
        """
        from ..object_store import get_object_store, uri_to_key

        store = get_object_store()
        key = uri_to_key(reference, expected_bucket=store.bucket)
        suffix = {
            "jsonl": ".jsonl",
            "json": ".json",
            "csv": ".csv",
            "parquet": ".parquet",
            "arrow": ".arrow",
        }.get(declared_format, ".data")

        tmp = tempfile.NamedTemporaryFile(
            prefix="dataset-src-", suffix=suffix, delete=False
        )
        tmp.close()
        try:
            store.download_file(key, tmp.name)
        except Exception:
            Path(tmp.name).unlink(missing_ok=True)
            raise
        return Path(tmp.name), declared_format
