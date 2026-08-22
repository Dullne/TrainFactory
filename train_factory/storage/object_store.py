"""
MinIO/S3 object storage client for TrainFactory.

Provides a unified interface for uploading, downloading, and streaming
dataset files from object storage (MinIO or S3-compatible).
"""

import io
import logging
import os
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional, Tuple

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class ObjectInfo:
    """Minimal object metadata."""

    def __init__(self, key: str, size: int, etag: Optional[str] = None):
        self.key = key
        self.size = size
        self.etag = etag

    def __repr__(self):
        return f"ObjectInfo(key={self.key!r}, size={self.size})"


class ObjectStore:
    """MinIO/S3 client wrapper for dataset file management."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create the bucket if it does not exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except S3Error as e:
            logger.error(f"Failed to ensure bucket {self.bucket}: {e}")
            raise

    # ------------------------------------------------------------------
    # Upload / Download
    # ------------------------------------------------------------------

    def upload_file(self, local_path: str, object_key: str) -> str:
        """Upload a local file and return its s3:// URI."""
        file_size = os.path.getsize(local_path)
        self.client.fput_object(self.bucket, object_key, local_path)
        logger.info(
            f"Uploaded {local_path} -> s3://{self.bucket}/{object_key} "
            f"({file_size} bytes)"
        )
        return f"s3://{self.bucket}/{object_key}"

    def download_file(self, object_key: str, local_path: str) -> str:
        """Download an object to a local path. Returns *local_path*."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, object_key, local_path)
        logger.info(f"Downloaded s3://{self.bucket}/{object_key} -> {local_path}")
        return local_path

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_lines(
        self, object_key: str, limit: Optional[int] = None
    ) -> Iterator[str]:
        """Yield lines from a text object (JSONL / CSV) without downloading
        the entire file.  Honours an optional *limit* on the number of lines.

        Uses a line buffer to correctly handle lines that span chunk boundaries.
        """
        response = None
        try:
            response = self.client.get_object(self.bucket, object_key)
            count = 0
            buf = ""
            for chunk in response.stream(amt=64 * 1024):
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    stripped = line.strip()
                    if not stripped:
                        continue
                    yield stripped
                    count += 1
                    if limit and count >= limit:
                        return
            # Flush remaining buffer (last line without trailing newline)
            if buf.strip():
                yield buf.strip()
        finally:
            if response:
                response.close()
                response.release_conn()

    def stream_bytes(self, object_key: str) -> BinaryIO:
        """Return a file-like byte stream for an object (e.g. Parquet)."""
        response = self.client.get_object(self.bucket, object_key)
        buf = io.BytesIO(response.read())
        response.close()
        response.release_conn()
        buf.seek(0)
        return buf

    # ------------------------------------------------------------------
    # Presigned URLs
    # ------------------------------------------------------------------

    def get_presigned_url(self, object_key: str, expires: int = 3600) -> str:
        """Generate a presigned download URL (default 1 hour)."""
        from datetime import timedelta

        url = self.client.presigned_get_object(
            self.bucket, object_key, expires=timedelta(seconds=expires)
        )
        return url

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def delete_object(self, object_key: str) -> None:
        """Delete an object."""
        self.client.remove_object(self.bucket, object_key)
        logger.info(f"Deleted s3://{self.bucket}/{object_key}")

    def list_objects(self, prefix: str) -> List[ObjectInfo]:
        """List all objects under *prefix*."""
        results: List[ObjectInfo] = []
        for obj in self.client.list_objects(self.bucket, prefix=prefix, recursive=True):
            results.append(ObjectInfo(key=obj.object_name, size=obj.size, etag=obj.etag))
        return results

    def object_exists(self, object_key: str) -> bool:
        """Check whether an object exists."""
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except S3Error:
            return False


# ======================================================================
# Global singleton
# ======================================================================

_store: Optional[ObjectStore] = None


def get_object_store() -> ObjectStore:
    """Return the global ObjectStore instance (lazy-initialised)."""
    global _store
    if _store is None:
        from ..config.settings import get_settings

        settings = get_settings()
        if not settings.minio_access_key or not settings.minio_secret_key:
            raise RuntimeError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required to use "
                "S3 object storage"
            )
        _store = ObjectStore(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )
    return _store


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse ``s3://bucket/key`` into ``(bucket, key)``.

    Raises:
        ValueError: If URI is empty or not a valid ``s3://bucket/key`` URI.
    """
    if not uri or not uri.startswith("s3://"):
        raise ValueError("storage URI must start with s3://")

    body = uri[5:]
    bucket, sep, key = body.partition("/")
    key = key.lstrip("/")
    if not bucket or not sep or not key:
        raise ValueError("storage URI must be in format s3://bucket/key")
    return bucket, key


def uri_to_key(uri: str, expected_bucket: Optional[str] = None) -> str:
    """Convert ``s3://bucket/key`` to ``key`` with optional bucket validation."""
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        if expected_bucket and bucket != expected_bucket:
            raise ValueError(
                f"S3 bucket mismatch: expected '{expected_bucket}', got '{bucket}'"
            )
        return key
    return uri
