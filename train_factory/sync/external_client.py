"""
External API client for incremental data fetching.

Handles authentication and paginated data retrieval from the
external business API using time-based incremental sync.

Auth: Supports both Bearer Token and cookie-based (bm_auth) authentication.
The auth_method field in auth_config controls which method is used:
  - "bearer" (default): sends token as Authorization: Bearer header
  - "cookie": sends token as bm_auth cookie
"""

import json
import logging
import math
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..storage.services.outbound_endpoint_policy import (
    DEFAULT_MAX_OUTBOUND_RESPONSE_BYTES,
    OutboundResponseTooLargeError,
    create_pinned_async_client,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_EXTERNAL_SYNC_PAGES = 1_000
DEFAULT_MAX_EXTERNAL_SYNC_RECORDS = 1_000_000
DEFAULT_MAX_EXTERNAL_SYNC_RESPONSE_BYTES = 100 * 1024 * 1024
MAX_EXTERNAL_SYNC_JSON_DEPTH = 64


class ExternalApiResponseError(Exception):
    """Raised when an external API returns an unsupported or invalid response."""


class ExternalApiAuthenticationError(RuntimeError):
    """Raised when an external API rejects the configured credentials."""


def _validate_positive_limit(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_non_negative_limit(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_page_payload(data: Any) -> Tuple[int, List[Dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ExternalApiResponseError(
            "External API response schema is invalid: root must be an object"
        )
    if "total" not in data:
        raise ExternalApiResponseError(
            "External API response schema is invalid: total is required"
        )
    total = data["total"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ExternalApiResponseError(
            "External API response schema is invalid: "
            "total must be a non-negative integer"
        )
    if "items" not in data:
        raise ExternalApiResponseError(
            "External API response schema is invalid: items is required"
        )
    items = data["items"]
    if not isinstance(items, list) or any(
        not isinstance(item, dict) for item in items
    ):
        raise ExternalApiResponseError(
            "External API response schema is invalid: items must be a list of objects"
        )
    return total, items


def _expected_has_more(
    total: int,
    items: List[Dict[str, Any]],
    offset: int,
) -> bool:
    page_end = offset + len(items)
    if page_end > total:
        raise ExternalApiResponseError(
            "External API pagination is inconsistent: "
            "page items extend beyond the reported total"
        )
    return page_end < total


def _bounded_json_size(value: Any, max_bytes: int) -> Optional[int]:
    remaining = max_bytes
    active_containers: set[int] = set()

    def consume(byte_count: int) -> bool:
        nonlocal remaining
        if byte_count > remaining:
            return False
        remaining -= byte_count
        return True

    def consume_string(text: str) -> bool:
        if not consume(1):
            return False
        for character in text:
            codepoint = ord(character)
            if character in {'"', "\\"} or codepoint in {8, 9, 10, 12, 13}:
                byte_count = 2
            elif codepoint < 0x20:
                byte_count = 6
            elif codepoint <= 0x7F:
                byte_count = 1
            elif codepoint <= 0x7FF:
                byte_count = 2
            elif 0xD800 <= codepoint <= 0xDFFF:
                raise ExternalApiResponseError(
                    "External API response schema is invalid: "
                    "strings must contain valid Unicode"
                )
            elif codepoint <= 0xFFFF:
                byte_count = 3
            else:
                byte_count = 4
            if not consume(byte_count):
                return False
        return consume(1)

    def consume_value(item: Any, depth: int) -> bool:
        if depth > MAX_EXTERNAL_SYNC_JSON_DEPTH:
            raise ExternalApiResponseError(
                "External API response schema is invalid: "
                f"maximum nesting depth is {MAX_EXTERNAL_SYNC_JSON_DEPTH}"
            )

        if item is None:
            return consume(4)
        if item is True:
            return consume(4)
        if item is False:
            return consume(5)
        if type(item) is int:
            try:
                rendered = str(item)
            except ValueError as exc:
                raise ExternalApiResponseError(
                    "External API response schema is invalid: integer is too large"
                ) from exc
            return consume(len(rendered))
        if type(item) is float:
            if not math.isfinite(item):
                raise ExternalApiResponseError(
                    "External API response schema is invalid: "
                    "numbers must be finite"
                )
            return consume(len(float.__repr__(item)))
        if type(item) is str:
            return consume_string(item)
        if type(item) not in {dict, list}:
            raise ExternalApiResponseError(
                "External API response schema is invalid: values must use JSON types"
            )

        marker = id(item)
        if marker in active_containers:
            raise ExternalApiResponseError(
                "External API response schema is invalid: "
                "circular containers are not allowed"
            )
        active_containers.add(marker)
        try:
            if type(item) is list:
                if not consume(1):
                    return False
                for index, child in enumerate(item):
                    if index and not consume(1):
                        return False
                    if not consume_value(child, depth + 1):
                        return False
                return consume(1)

            if not consume(1):
                return False
            for index, (key, child) in enumerate(item.items()):
                if type(key) is not str:
                    raise ExternalApiResponseError(
                        "External API response schema is invalid: "
                        "object keys must be strings"
                    )
                if index and not consume(1):
                    return False
                if not consume_string(key) or not consume(1):
                    return False
                if not consume_value(child, depth + 1):
                    return False
            return consume(1)
        finally:
            active_containers.remove(marker)

    if not consume_value(value, 0):
        return None
    return max_bytes - remaining


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _load_json_response(response_body: bytearray) -> Any:
    try:
        return json.loads(
            response_body,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ExternalApiResponseError(
            "External API returned invalid JSON"
        ) from exc


class ExternalApiClient:
    """Client for external API with cookie or Bearer Token auth."""

    def __init__(
        self,
        api_url: str,
        auth_config: Dict[str, Any],
        timeout: int = 30,
        user_id: Optional[str] = None,
        *,
        max_pages: int = DEFAULT_MAX_EXTERNAL_SYNC_PAGES,
        max_records: int = DEFAULT_MAX_EXTERNAL_SYNC_RECORDS,
        max_aggregate_response_bytes: int = (
            DEFAULT_MAX_EXTERNAL_SYNC_RESPONSE_BYTES
        ),
    ):
        self.api_url = api_url.rstrip("/")
        self.auth_config = auth_config
        self.timeout = timeout
        self.user_id = user_id
        self.max_pages = _validate_positive_limit("max_pages", max_pages)
        self.max_records = _validate_positive_limit("max_records", max_records)
        self.max_aggregate_response_bytes = _validate_positive_limit(
            "max_aggregate_response_bytes",
            max_aggregate_response_bytes,
        )
        self._page_response_limit: ContextVar[Optional[int]] = ContextVar(
            f"external_api_page_response_limit_{id(self)}",
            default=None,
        )
        self._last_response_bytes: ContextVar[Optional[int]] = ContextVar(
            f"external_api_last_response_bytes_{id(self)}",
            default=None,
        )
        self._headers: Dict[str, str] = {}
        self._cookies: Dict[str, str] = {}

        token = self.auth_config.get("token", "")
        auth_method = self.auth_config.get("auth_method", "bearer")

        if token:
            if auth_method == "cookie":
                self._cookies["bm_auth"] = token
            else:
                # Default: Bearer auth
                self._headers["Authorization"] = f"Bearer {token}"

    async def _fetch_incremental_page(
        self,
        since: Optional[datetime],
        offset: int,
        limit: int,
        max_response_bytes: int,
    ) -> Tuple[Dict[str, Any], int]:
        params: Dict[str, Any] = {
            "offset": offset,
            "limit": limit,
        }
        if since is not None:
            params["since"] = since.strftime("%Y-%m-%dT%H:%M:%S")

        url = self.api_url

        headers = dict(self._headers)
        headers["Accept-Encoding"] = "identity"
        client_kwargs: Dict[str, Any] = {
            "headers": headers,
            "timeout": self.timeout,
            "max_response_bytes": max_response_bytes,
        }
        if self._cookies:
            # Bind cookies at client-level to avoid deprecated per-request cookies usage.
            client_kwargs["cookies"] = self._cookies

        async with create_pinned_async_client(
            self.api_url,
            self.user_id,
            **client_kwargs,
        ) as client:
            async with client.stream("GET", url, params=params) as resp:
                if resp.status_code == 401:
                    raise ExternalApiAuthenticationError(
                        "Token expired or invalid, please update the token"
                    )

                resp.raise_for_status()
                content_encoding = resp.headers.get("content-encoding", "")
                if content_encoding.strip().lower() not in {"", "identity"}:
                    raise ExternalApiResponseError(
                        "External API response Content-Encoding must be identity"
                    )
                response_body = bytearray()
                response_bytes = 0
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    response_bytes += len(chunk)
                    if response_bytes > max_response_bytes:
                        raise OutboundResponseTooLargeError(max_response_bytes)
                    response_body.extend(chunk)

            data = _load_json_response(response_body)

            total, items = _validate_page_payload(data)
            has_more = _expected_has_more(total, items, offset)

            return (
                {
                    "total": total,
                    "items": items,
                    "has_more": has_more,
                },
                response_bytes,
            )

    async def fetch_incremental(
        self,
        since: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Fetch data from external API with optional time filter and pagination.

        Args:
            since: If provided, only fetch records after this time.
                   If None, fetch all records from the beginning.

        Returns: {"total": int, "items": list, "has_more": bool}
        """
        offset = _validate_non_negative_limit("offset", offset)
        limit = _validate_positive_limit("limit", limit)
        max_response_bytes = (
            self._page_response_limit.get()
            or DEFAULT_MAX_OUTBOUND_RESPONSE_BYTES
        )
        result, response_bytes = await self._fetch_incremental_page(
            since,
            offset,
            limit,
            max_response_bytes,
        )
        self._last_response_bytes.set(response_bytes)
        return result

    async def fetch_all_since(self, since: Optional[datetime] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch all records since a given time, handling pagination.

        Args:
            since: If provided, only fetch records after this time.
                   If None, fetch all records from the beginning.
        """
        limit = _validate_positive_limit("limit", limit)
        all_items: List[Dict[str, Any]] = []
        offset = 0
        pages = 0
        aggregate_response_bytes = 0

        while True:
            if pages >= self.max_pages:
                raise RuntimeError(
                    "External API pagination exceeded the maximum of "
                    f"{self.max_pages} pages"
                )
            if len(all_items) >= self.max_records:
                raise RuntimeError(
                    "External API pagination exceeded the maximum of "
                    f"{self.max_records} records"
                )

            remaining_response_bytes = (
                self.max_aggregate_response_bytes - aggregate_response_bytes
            )
            if remaining_response_bytes <= 0:
                raise RuntimeError(
                    "External API pagination exceeded the maximum aggregate "
                    "response size of "
                    f"{self.max_aggregate_response_bytes} bytes"
                )

            response_limit = min(
                DEFAULT_MAX_OUTBOUND_RESPONSE_BYTES,
                remaining_response_bytes,
            )
            response_limit_token = self._page_response_limit.set(response_limit)
            response_bytes_token = self._last_response_bytes.set(None)
            try:
                try:
                    result = await self.fetch_incremental(
                        since,
                        offset=offset,
                        limit=limit,
                    )
                except OutboundResponseTooLargeError as exc:
                    if response_limit == remaining_response_bytes:
                        raise RuntimeError(
                            "External API pagination exceeded the maximum aggregate "
                            "response size of "
                            f"{self.max_aggregate_response_bytes} bytes"
                        ) from exc
                    raise
                response_bytes = self._last_response_bytes.get()
            finally:
                self._last_response_bytes.reset(response_bytes_token)
                self._page_response_limit.reset(response_limit_token)

            total, items = _validate_page_payload(result)
            encoded_response_bytes = _bounded_json_size(
                result,
                remaining_response_bytes,
            )
            if encoded_response_bytes is None:
                raise RuntimeError(
                    "External API pagination exceeded the maximum aggregate "
                    "response size of "
                    f"{self.max_aggregate_response_bytes} bytes"
                )
            response_bytes = max(response_bytes or 0, encoded_response_bytes)

            has_more = result.get("has_more")
            if not isinstance(has_more, bool):
                raise ExternalApiResponseError(
                    "External API response schema is invalid: "
                    "has_more must be a boolean"
                )
            expected_has_more = _expected_has_more(total, items, offset)

            pages += 1
            if response_bytes > remaining_response_bytes:
                raise RuntimeError(
                    "External API pagination exceeded the maximum aggregate "
                    "response size of "
                    f"{self.max_aggregate_response_bytes} bytes"
                )
            aggregate_response_bytes += response_bytes

            if not items:
                if has_more or expected_has_more:
                    raise RuntimeError(
                        "External API pagination made no offset progress"
                    )
                break

            if has_more is not expected_has_more:
                raise ExternalApiResponseError(
                    "External API response schema is invalid: "
                    "has_more is inconsistent with total, offset, and items"
                )

            projected_records = len(all_items) + len(items)
            if projected_records > self.max_records:
                raise RuntimeError(
                    "External API pagination exceeded the maximum of "
                    f"{self.max_records} records"
                )
            all_items.extend(items)
            logger.debug(
                f"[external_client] Fetched {len(items)} items "
                f"(offset={offset}, total={result['total']})"
            )

            if not has_more:
                break

            next_offset = offset + len(items)
            if next_offset <= offset:
                raise RuntimeError("External API pagination made no offset progress")
            offset = next_offset

        if all_items:
            logger.info(
                f"[external_client] Total fetched: {len(all_items)} items since {since}"
            )

        return all_items
