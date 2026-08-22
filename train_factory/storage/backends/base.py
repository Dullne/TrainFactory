"""Abstract storage backend interface for dataset operations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class StorageBackend(ABC):
    """Abstract base class for dataset storage backends.

    Each backend handles storage-specific operations: preview, delete,
    export source resolution, and existence checks.
    """

    @abstractmethod
    def exists(self, reference: str) -> bool:
        """Check if a storage reference (local path or S3 URI) exists."""
        ...

    @abstractmethod
    def delete(self, reference: str, dataset_id: Optional[str] = None) -> bool:
        """Delete data at the given storage reference.

        Args:
            reference: Storage path or URI to delete.
            dataset_id: Optional dataset ID for whitelist validation (local only).
        Returns ``True`` when the reference is absent or was deleted, and
        ``False`` when policy validation or storage cleanup failed.
        """
        ...

    @abstractmethod
    def load_preview(
        self, reference: str, file_format: str, limit: int = 10
    ) -> Dict[str, Any]:
        """Load preview rows from the storage reference.

        Returns:
            Dict with keys: columns, rows, num_rows, num_train, num_eval,
            num_test, file_size. May include 'error' key on failure.
        """
        ...

    @abstractmethod
    def resolve_export_source(
        self, reference: str, declared_format: str
    ) -> Tuple[Path, str]:
        """Resolve the concrete local source file/dir for export.

        For local backend, returns the path directly.
        For S3 backend, downloads to a temp file first.

        Returns:
            (local_path, detected_format)

        Raises:
            ValueError: If the source cannot be resolved.
        """
        ...

    @staticmethod
    def detect_file_format(file_path: Path) -> str:
        """Detect file format from file extension."""
        suffix = file_path.suffix.lower()
        return {
            ".jsonl": "jsonl",
            ".json": "json",
            ".parquet": "parquet",
            ".csv": "csv",
            ".arrow": "arrow",
        }.get(suffix, "jsonl")
