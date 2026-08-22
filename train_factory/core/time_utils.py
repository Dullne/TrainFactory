"""
Time utility helpers with configurable application timezone.

By default APP_TIMEZONE=UTC. Returned naive datetimes follow APP_TIMEZONE.
"""

import logging
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _build_zoneinfo(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        logger.warning("Invalid APP_TIMEZONE=%s, fallback to UTC", tz_name)
        return ZoneInfo("UTC")


def get_app_timezone_name() -> str:
    settings = get_settings()
    tz_name = (settings.app_timezone or "UTC").strip()
    return tz_name or "UTC"


def get_app_timezone() -> ZoneInfo:
    return _build_zoneinfo(get_app_timezone_name())


def now_aware() -> datetime:
    """Return timezone-aware current time in APP_TIMEZONE."""
    return datetime.now(get_app_timezone())


def now_naive() -> datetime:
    """Return naive current time in APP_TIMEZONE."""
    return now_aware().replace(tzinfo=None)


def sync_now_naive() -> datetime:
    """Return naive current time for sync pipeline (follows APP_TIMEZONE)."""
    return now_naive()


def sync_normalize_naive(dt: datetime) -> datetime:
    """
    Normalize datetime to APP_TIMEZONE naive form.

    Naive input is returned as-is for backward compatibility.
    Aware input is converted to APP_TIMEZONE then converted to naive.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(get_app_timezone()).replace(tzinfo=None)
