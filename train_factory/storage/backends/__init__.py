"""Storage backend factory."""

from functools import lru_cache

from .base import StorageBackend


@lru_cache(maxsize=4)
def get_storage_backend(backend_type: str = "local") -> StorageBackend:
    """Get a storage backend instance by type (cached singleton per type)."""
    if backend_type == "s3":
        from .s3_backend import S3StorageBackend

        return S3StorageBackend()

    if backend_type != "local":
        raise ValueError(f"Unknown storage backend: {backend_type!r}. Must be 'local' or 's3'.")

    from .local_backend import LocalStorageBackend

    return LocalStorageBackend()


__all__ = ["StorageBackend", "get_storage_backend"]
