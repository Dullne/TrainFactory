from datetime import timedelta

import pytest

from train_factory.core import idempotency


@pytest.fixture(autouse=True)
def _clear_idempotency_cache():
    with idempotency._cache_lock:
        idempotency._idempotency_cache.clear()
    yield
    with idempotency._cache_lock:
        idempotency._idempotency_cache.clear()


def _store(key: str, user_id: str = "user-a") -> None:
    idempotency.store_idempotency_response(
        key,
        user_id,
        "/api/test",
        {"key": key},
    )


def _is_cached(key: str, user_id: str = "user-a") -> bool:
    return idempotency.check_idempotency(key, user_id, "/api/test")[0]


def test_oversized_keys_are_ignored_without_growing_cache(monkeypatch):
    monkeypatch.setattr(idempotency, "MAX_IDEMPOTENCY_KEY_BYTES", 8)

    _store("123456789")

    assert not _is_cached("123456789")
    assert idempotency.get_cache_stats()["total_entries"] == 0


def test_store_purges_expired_entries_for_unrelated_keys():
    _store("expired")
    with idempotency._cache_lock:
        only_entry = next(iter(idempotency._idempotency_cache.values()))
        only_entry["expires_at"] = idempotency.now_naive() - timedelta(seconds=1)

    _store("fresh", user_id="user-b")

    assert idempotency.get_cache_stats() == {
        "total_entries": 1,
        "active_entries": 1,
        "expired_entries": 0,
    }
    assert _is_cached("fresh", user_id="user-b")


def test_per_user_limit_evicts_only_that_users_oldest_entry(monkeypatch):
    monkeypatch.setattr(idempotency, "MAX_CACHE_ENTRIES_PER_USER", 2)
    monkeypatch.setattr(idempotency, "MAX_CACHE_ENTRIES", 10)

    _store("oldest")
    _store("newer")
    _store("other-user", user_id="user-b")
    _store("newest")

    assert not _is_cached("oldest")
    assert _is_cached("newer")
    assert _is_cached("newest")
    assert _is_cached("other-user", user_id="user-b")


def test_global_limit_evicts_oldest_entry(monkeypatch):
    monkeypatch.setattr(idempotency, "MAX_CACHE_ENTRIES_PER_USER", 10)
    monkeypatch.setattr(idempotency, "MAX_CACHE_ENTRIES", 3)

    _store("first", user_id="user-a")
    _store("second", user_id="user-b")
    _store("third", user_id="user-c")
    _store("fourth", user_id="user-d")

    assert not _is_cached("first", user_id="user-a")
    assert _is_cached("second", user_id="user-b")
    assert _is_cached("third", user_id="user-c")
    assert _is_cached("fourth", user_id="user-d")
    assert idempotency.get_cache_stats()["total_entries"] == 3
