"""Persistent-byte admission control shared by model and dataset downloads."""

import threading
from typing import Dict, Optional, Tuple
from uuid import uuid4

from sqlalchemy import func
from sqlmodel import select

from ...core.remote_download_security import DownloadStorageQuotaExceeded
from ..database import get_session
from ..entities.dataset_entity import DatasetDB
from ..entities.model_registry_entity import ModelRegistryDB


class DownloadStorageReservation:
    """One idempotently releasable reservation for an in-flight download."""

    def __init__(
        self,
        service: "DownloadStorageQuotaService",
        reservation_id: str,
        user_key: str,
        reserved_bytes: int,
        global_limit: int,
        per_user_limit: int,
    ):
        self._service = service
        self._reservation_id = reservation_id
        self._user_key = user_key
        self._reserved_bytes = reserved_bytes
        self._global_limit = global_limit
        self._per_user_limit = per_user_limit
        self._released = False
        self._lock = threading.Lock()

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved_bytes

    def verify_actual_size(self, actual_bytes: int) -> None:
        """Resize the reservation after strict local measurement."""
        with self._lock:
            if self._released:
                raise RuntimeError("Download storage reservation is already released")
            self._service._resize(
                self._reservation_id,
                self._user_key,
                actual_bytes,
                global_limit=self._global_limit,
                per_user_limit=self._per_user_limit,
            )
            self._reserved_bytes = actual_bytes

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._service._release(self._reservation_id)


class DownloadStorageQuotaService:
    """Serialize DB usage checks with all active metadata reservations.

    NOTE: reservations and the concurrency limiter are process-local. The API
    enforces ``api_workers == 1`` (validate_api_worker_count), so a single
    container is safe; multi-replica deployments must not rely on these
    in-process counters (use the DB-backed download rows for cross-process
    coordination).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._reservations: Dict[str, Tuple[str, int]] = {}

    @property
    def active_reserved_total(self) -> int:
        with self._lock:
            return sum(size for _, size in self._reservations.values())

    def reserve(
        self,
        user_id: Optional[str],
        *,
        expected_bytes: int,
        global_limit: int,
        per_user_limit: int,
    ) -> DownloadStorageReservation:
        if expected_bytes < 0:
            raise ValueError("Expected download size cannot be negative")
        user_key = user_id or "<anonymous>"
        with self._lock:
            stored_global, stored_user = self._stored_usage(user_id)
            reserved_global = sum(
                size for _, size in self._reservations.values()
            )
            reserved_user = sum(
                size
                for reservation_user, size in self._reservations.values()
                if reservation_user == user_key
            )
            self._enforce_limits(
                stored_global + reserved_global + expected_bytes,
                stored_user + reserved_user + expected_bytes,
                global_limit=global_limit,
                per_user_limit=per_user_limit,
            )
            reservation_id = str(uuid4())
            self._reservations[reservation_id] = (user_key, expected_bytes)
        return DownloadStorageReservation(
            self,
            reservation_id,
            user_key,
            expected_bytes,
            global_limit,
            per_user_limit,
        )

    def _resize(
        self,
        reservation_id: str,
        user_key: str,
        actual_bytes: int,
        *,
        global_limit: int,
        per_user_limit: int,
    ) -> None:
        if actual_bytes < 0:
            raise ValueError("Actual download size cannot be negative")
        user_id = None if user_key == "<anonymous>" else user_key
        with self._lock:
            current = self._reservations.get(reservation_id)
            if current is None or current[0] != user_key:
                raise RuntimeError("Download storage reservation is unavailable")
            current_bytes = current[1]
            stored_global, stored_user = self._stored_usage(user_id)
            reserved_global = sum(
                size for _, size in self._reservations.values()
            )
            reserved_user = sum(
                size
                for reservation_user, size in self._reservations.values()
                if reservation_user == user_key
            )
            self._enforce_limits(
                stored_global + reserved_global - current_bytes + actual_bytes,
                stored_user + reserved_user - current_bytes + actual_bytes,
                global_limit=global_limit,
                per_user_limit=per_user_limit,
            )
            self._reservations[reservation_id] = (user_key, actual_bytes)

    @staticmethod
    def _enforce_limits(
        projected_global: int,
        projected_user: int,
        *,
        global_limit: int,
        per_user_limit: int,
    ) -> None:
        if projected_global > global_limit:
            raise DownloadStorageQuotaExceeded(
                "Remote download global storage quota exceeded"
            )
        if projected_user > per_user_limit:
            raise DownloadStorageQuotaExceeded(
                "Remote download per-user storage quota exceeded"
            )

    def _release(self, reservation_id: str) -> None:
        with self._lock:
            self._reservations.pop(reservation_id, None)

    def _stored_usage(self, user_id: Optional[str]) -> Tuple[int, int]:
        model_filters = [
            ModelRegistryDB.source_type == "downloaded",
            ModelRegistryDB.file_size.is_not(None),
        ]
        dataset_filters = [
            DatasetDB.source_type.in_(["huggingface", "modelscope"]),
            DatasetDB.file_size.is_not(None),
        ]
        with get_session() as session:
            model_global = session.exec(
                select(func.coalesce(func.sum(ModelRegistryDB.file_size), 0)).where(
                    *model_filters
                )
            ).one()
            dataset_global = session.exec(
                select(func.coalesce(func.sum(DatasetDB.file_size), 0)).where(
                    *dataset_filters
                )
            ).one()
            model_user = session.exec(
                select(func.coalesce(func.sum(ModelRegistryDB.file_size), 0)).where(
                    *model_filters,
                    ModelRegistryDB.user_id == user_id,
                )
            ).one()
            dataset_user = session.exec(
                select(func.coalesce(func.sum(DatasetDB.file_size), 0)).where(
                    *dataset_filters,
                    DatasetDB.user_id == user_id,
                )
            ).one()
        return (
            int(model_global or 0) + int(dataset_global or 0),
            int(model_user or 0) + int(dataset_user or 0),
        )


download_storage_quota_service = DownloadStorageQuotaService()
