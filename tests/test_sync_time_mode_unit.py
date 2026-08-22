"""Unit tests for sync timezone behavior (single source: APP_TIMEZONE)."""

from datetime import datetime, timezone

from train_factory.config.settings import get_settings
from train_factory.core import time_utils


def _reset_time_caches() -> None:
    get_settings.cache_clear()
    time_utils._build_zoneinfo.cache_clear()


def test_sync_now_naive_follows_app_timezone(monkeypatch):
    monkeypatch.setenv("APP_TIMEZONE", "Asia/Shanghai")
    _reset_time_caches()

    got = time_utils.sync_now_naive()
    expected = datetime.now(time_utils.get_app_timezone()).replace(tzinfo=None)

    assert abs((got - expected).total_seconds()) < 3


def test_sync_normalize_naive_follows_app_timezone(monkeypatch):
    aware_utc = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setenv("APP_TIMEZONE", "Asia/Shanghai")
    _reset_time_caches()
    assert time_utils.sync_normalize_naive(aware_utc) == datetime(2026, 1, 1, 8, 0, 0)


def test_sync_normalize_naive_invalid_timezone_falls_back_to_utc(monkeypatch):
    aware_utc = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setenv("APP_TIMEZONE", "Invalid/Zone")
    _reset_time_caches()
    assert time_utils.sync_normalize_naive(aware_utc) == datetime(2026, 1, 1, 0, 0, 0)
