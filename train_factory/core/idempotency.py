"""
Idempotency service for preventing duplicate operations.

Provides mechanisms to ensure POST requests are idempotent by caching
responses based on client-provided idempotency keys.
"""

import hashlib
import logging
from datetime import timedelta
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from train_factory.core.time_utils import now_naive

logger = logging.getLogger(__name__)

# In-memory cache for idempotency (simple implementation)
# For production, consider using Redis or database storage
_idempotency_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = Lock()

# Default TTL for idempotency keys (24 hours)
DEFAULT_TTL_HOURS = 24

# Bound this process-local cache so untrusted headers cannot grow it indefinitely.
MAX_IDEMPOTENCY_KEY_BYTES = 256
MAX_CACHE_ENTRIES = 10_000
MAX_CACHE_ENTRIES_PER_USER = 1_000


def _is_valid_idempotency_key(idempotency_key: Optional[str]) -> bool:
    if not isinstance(idempotency_key, str) or not idempotency_key:
        return False
    return len(idempotency_key.encode("utf-8")) <= MAX_IDEMPOTENCY_KEY_BYTES


def _cleanup_expired_entries_locked(now) -> int:
    expired_keys = [
        key for key, entry in _idempotency_cache.items() if now >= entry["expires_at"]
    ]
    for key in expired_keys:
        del _idempotency_cache[key]
    return len(expired_keys)


def _evict_oldest_entry_locked(*, user_id: Optional[str] = None) -> bool:
    candidates = (
        (key, entry)
        for key, entry in _idempotency_cache.items()
        if user_id is None or entry.get("user_id") == user_id
    )
    try:
        oldest_key, _ = min(candidates, key=lambda item: item[1]["created_at"])
    except ValueError:
        return False
    del _idempotency_cache[oldest_key]
    return True


def _generate_cache_key(idempotency_key: str, user_id: str, endpoint: str) -> str:
    """Generate a unique cache key combining idempotency key, user, and endpoint."""
    raw = f"{user_id}:{endpoint}:{idempotency_key}"
    return hashlib.sha256(raw.encode()).hexdigest()


def check_idempotency(
    idempotency_key: Optional[str],
    user_id: str,
    endpoint: str,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check if a request with this idempotency key has already been processed.

    Args:
        idempotency_key: Client-provided idempotency key (from header)
        user_id: Current user ID
        endpoint: API endpoint path

    Returns:
        (is_duplicate, cached_response)
        - (False, None) if this is a new request
        - (True, response) if this is a duplicate with cached response
    """
    if not _is_valid_idempotency_key(idempotency_key):
        return False, None

    cache_key = _generate_cache_key(idempotency_key, user_id, endpoint)

    with _cache_lock:
        now = now_naive()
        _cleanup_expired_entries_locked(now)
        entry = _idempotency_cache.get(cache_key)
        if entry is not None:
            logger.info(f"Idempotency hit for key {idempotency_key[:8]}... on {endpoint}")
            return True, entry["response"]

    return False, None


def store_idempotency_response(
    idempotency_key: Optional[str],
    user_id: str,
    endpoint: str,
    response: Dict[str, Any],
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> None:
    """
    Store the response for an idempotency key.

    Args:
        idempotency_key: Client-provided idempotency key
        user_id: Current user ID
        endpoint: API endpoint path
        response: The response to cache
        ttl_hours: Time-to-live in hours
    """
    if not _is_valid_idempotency_key(idempotency_key) or ttl_hours <= 0:
        return

    cache_key = _generate_cache_key(idempotency_key, user_id, endpoint)

    with _cache_lock:
        now = now_naive()
        _cleanup_expired_entries_locked(now)

        if cache_key not in _idempotency_cache:
            if MAX_CACHE_ENTRIES_PER_USER <= 0 or MAX_CACHE_ENTRIES <= 0:
                return
            user_entries = sum(
                entry.get("user_id") == user_id
                for entry in _idempotency_cache.values()
            )
            while user_entries >= MAX_CACHE_ENTRIES_PER_USER:
                if not _evict_oldest_entry_locked(user_id=user_id):
                    break
                user_entries -= 1
            while len(_idempotency_cache) >= MAX_CACHE_ENTRIES:
                if not _evict_oldest_entry_locked():
                    break

        _idempotency_cache[cache_key] = {
            "user_id": user_id,
            "response": response,
            "created_at": now,
            "expires_at": now + timedelta(hours=ttl_hours),
        }
        logger.debug(f"Stored idempotency response for key {idempotency_key[:8]}...")


def cleanup_expired_entries() -> int:
    """
    Remove expired idempotency entries.

    Returns:
        Number of entries removed
    """
    now = now_naive()
    removed = 0

    with _cache_lock:
        removed = _cleanup_expired_entries_locked(now)

    if removed > 0:
        logger.info(f"Cleaned up {removed} expired idempotency entries")

    return removed


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the idempotency cache."""
    with _cache_lock:
        now = now_naive()
        total = len(_idempotency_cache)
        expired = sum(1 for entry in _idempotency_cache.values() if now >= entry["expires_at"])
        return {
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
        }
