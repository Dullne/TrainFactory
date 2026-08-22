"""Resource-bound pagination tests for the external sync client."""

import asyncio
import gzip
import importlib
import json
import math
import tracemalloc

import httpx
import pytest

from train_factory.sync import external_client as external_client_module
from train_factory.sync.external_client import (
    ExternalApiClient,
    ExternalApiResponseError,
    _bounded_json_size,
)


class _FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, body: bytes) -> None:
        self.content = body

    def raise_for_status(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def aiter_bytes(self, chunk_size=None):
        chunk_size = chunk_size or len(self.content) or 1
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]

    def json(self):
        return json.loads(self.content)


class _FakeAsyncClient:
    def __init__(self, page_for_params, calls) -> None:
        self._page_for_params = page_for_params
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    def stream(self, _method, _url, *, params):
        captured = dict(params)
        self._calls.append(captured)
        return _FakeResponse(self._page_for_params(captured))


class _CountingCompressedStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes, chunk_size: int = 32) -> None:
        self._chunks = [
            content[index : index + chunk_size]
            for index in range(0, len(content), chunk_size)
        ]
        self.chunks_read = 0
        self.closed = False

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    async def __aiter__(self):
        for chunk in self._chunks:
            self.chunks_read += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _page_body(items, *, total, padding=0) -> bytes:
    body = json.dumps(
        {"total": total, "items": items},
        separators=(",", ":"),
    ).encode("utf-8")
    return body + (b" " * padding)


def _make_client(monkeypatch, page_for_params, **limits):
    calls = []
    response_budgets = []

    def create_client(*_args, **kwargs):
        response_budgets.append(kwargs.get("max_response_bytes"))
        return _FakeAsyncClient(page_for_params, calls)

    monkeypatch.setattr(
        external_client_module,
        "create_pinned_async_client",
        create_client,
    )
    client = ExternalApiClient(
        "https://api.example.com/items",
        {},
        **limits,
    )
    return client, calls, response_budgets


@pytest.mark.parametrize(
    ("offset", "limit", "message"),
    [
        (-1, 1, "offset must be a non-negative integer"),
        (True, 1, "offset must be a non-negative integer"),
        (0, 0, "limit must be a positive integer"),
        (0, True, "limit must be a positive integer"),
    ],
)
def test_fetch_incremental_rejects_invalid_pagination_arguments(
    monkeypatch,
    offset,
    limit,
    message,
):
    client, calls, _ = _make_client(
        monkeypatch,
        lambda _params: _page_body([], total=0),
        max_pages=2,
        max_records=10,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(ValueError, match=message):
        asyncio.run(client.fetch_incremental(offset=offset, limit=limit))

    assert calls == []


def test_fetch_all_since_validates_limit_before_calling_override():
    calls = []

    class OverrideClient(ExternalApiClient):
        async def fetch_incremental(self, since=None, offset=0, limit=1000):
            calls.append((offset, limit))
            return {"total": 0, "items": [], "has_more": False}

    client = OverrideClient("https://api.example.com/items", {})

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        asyncio.run(client.fetch_all_since(limit=0))

    assert calls == []


def test_fetch_all_since_stops_nonempty_huge_total_at_page_limit(monkeypatch):
    def endless_page(params):
        return _page_body(
            [{"id": f"record-{params['offset']}"}],
            total=10**18,
        )

    client, calls, _ = _make_client(
        monkeypatch,
        endless_page,
        max_pages=3,
        max_records=100,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(RuntimeError, match="maximum of 3 pages"):
        asyncio.run(client.fetch_all_since(limit=1))

    assert [call["offset"] for call in calls] == [0, 1, 2]


def test_fetch_all_since_rejects_page_that_exceeds_record_limit(monkeypatch):
    pages = {
        0: _page_body([{"id": "one"}, {"id": "two"}], total=4),
        2: _page_body([{"id": "three"}, {"id": "four"}], total=4),
    }
    client, calls, _ = _make_client(
        monkeypatch,
        lambda params: pages[params["offset"]],
        max_pages=10,
        max_records=3,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(RuntimeError, match="maximum of 3 records"):
        asyncio.run(client.fetch_all_since(limit=2))

    assert [call["offset"] for call in calls] == [0, 2]


def test_fetch_all_since_counts_actual_aggregate_response_body_bytes(monkeypatch):
    first = _page_body([{"id": "one"}], total=2, padding=37)
    second = _page_body([{"id": "two"}], total=2, padding=41)
    max_bytes = len(first) + len(second.rstrip()) + 1
    pages = {0: first, 1: second}
    client, calls, response_budgets = _make_client(
        monkeypatch,
        lambda params: pages[params["offset"]],
        max_pages=10,
        max_records=10,
        max_aggregate_response_bytes=max_bytes,
    )

    with pytest.raises(RuntimeError, match="maximum aggregate response size"):
        asyncio.run(client.fetch_all_since(limit=1))

    assert [call["offset"] for call in calls] == [0, 1]
    assert response_budgets == [max_bytes, max_bytes - len(first)]


def test_fetch_all_since_accepts_exact_page_record_and_byte_bounds(monkeypatch):
    first_items = [{"id": "one"}, {"id": "two"}]
    second_items = [{"id": "three"}]
    first = _page_body(first_items, total=3)
    second = _page_body(second_items, total=3)
    first_encoded = _bounded_json_size(
        {"total": 3, "items": first_items, "has_more": True},
        10_000,
    )
    second_encoded = _bounded_json_size(
        {"total": 3, "items": second_items, "has_more": False},
        10_000,
    )
    assert first_encoded is not None
    assert second_encoded is not None
    first_accounted = max(len(first), first_encoded)
    second_accounted = max(len(second), second_encoded)
    max_bytes = first_accounted + second_accounted
    pages = {0: first, 2: second}
    client, calls, response_budgets = _make_client(
        monkeypatch,
        lambda params: pages[params["offset"]],
        max_pages=2,
        max_records=3,
        max_aggregate_response_bytes=max_bytes,
    )

    items = asyncio.run(client.fetch_all_since(limit=2))

    assert [item["id"] for item in items] == ["one", "two", "three"]
    assert [call["offset"] for call in calls] == [0, 2]
    assert response_budgets == [max_bytes, second_accounted]


def test_fetch_all_since_advances_offset_by_each_actual_page_size(monkeypatch):
    pages = {
        0: _page_body([{"id": "one"}, {"id": "two"}], total=5),
        2: _page_body([{"id": "three"}], total=5),
        3: _page_body([{"id": "four"}, {"id": "five"}], total=5),
    }
    client, calls, _ = _make_client(
        monkeypatch,
        lambda params: pages[params["offset"]],
        max_pages=3,
        max_records=5,
        max_aggregate_response_bytes=10_000,
    )

    items = asyncio.run(client.fetch_all_since(limit=2))

    assert len(items) == 5
    assert [call["offset"] for call in calls] == [0, 2, 3]


def test_fetch_all_since_stops_decoded_gzip_body_before_full_stream(monkeypatch):
    decoded = _page_body(
        [{"id": "one", "content": "x" * 1_000_000}],
        total=1,
    )
    compressed = gzip.compress(decoded)
    max_bytes = 2_000
    assert len(compressed) < max_bytes < len(decoded)
    stream = _CountingCompressedStream(compressed, chunk_size=len(compressed))
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=stream,
        )

    def create_client(*_args, **kwargs):
        return httpx.AsyncClient(
            headers=kwargs.get("headers"),
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(
        external_client_module,
        "create_pinned_async_client",
        create_client,
    )
    client = ExternalApiClient(
        "https://api.example.com/items",
        {},
        max_pages=2,
        max_records=2,
        max_aggregate_response_bytes=max_bytes,
    )

    with pytest.raises(ExternalApiResponseError) as exc_info:
        asyncio.run(client.fetch_all_since(limit=1))

    assert "Content-Encoding" in str(exc_info.value)
    assert stream.chunk_count == 1
    assert stream.chunks_read == 0
    assert stream.closed is True
    assert requests[0].headers["accept-encoding"] == "identity"


def test_fetch_all_since_preserves_fetch_incremental_override(monkeypatch):
    calls = []

    class OverrideClient(ExternalApiClient):
        async def fetch_incremental(self, since=None, offset=0, limit=1000):
            calls.append((since, offset, limit))
            if offset == 0:
                return {
                    "total": 3,
                    "items": [{"id": "one"}, {"id": "two"}],
                    "has_more": True,
                }
            return {
                "total": 3,
                "items": [{"id": "three"}],
                "has_more": False,
            }

    def unexpected_client(*_args, **_kwargs):
        raise AssertionError("fetch_incremental override was bypassed")

    monkeypatch.setattr(
        external_client_module,
        "create_pinned_async_client",
        unexpected_client,
    )
    client = OverrideClient(
        "https://api.example.com/items",
        {},
        max_pages=2,
        max_records=3,
        max_aggregate_response_bytes=10_000,
    )

    items = asyncio.run(client.fetch_all_since(limit=2))

    assert [item["id"] for item in items] == ["one", "two", "three"]
    assert [offset for _, offset, _ in calls] == [0, 2]


def test_fetch_all_since_recounts_super_delegating_override_result(monkeypatch):
    calls = []

    def create_client(*_args, **_kwargs):
        return _FakeAsyncClient(
            lambda _params: _page_body([{"id": "one"}], total=1),
            calls,
        )

    monkeypatch.setattr(
        external_client_module,
        "create_pinned_async_client",
        create_client,
    )

    class AmplifyingClient(ExternalApiClient):
        async def fetch_incremental(self, since=None, offset=0, limit=1000):
            result = await super().fetch_incremental(since, offset, limit)
            result["items"][0]["content"] = "x" * 10_000
            return result

    client = AmplifyingClient(
        "https://api.example.com/items",
        {},
        max_pages=1,
        max_records=1,
        max_aggregate_response_bytes=100,
    )

    with pytest.raises(RuntimeError, match="maximum aggregate response size"):
        asyncio.run(client.fetch_all_since(limit=1))


def test_fetch_all_since_recursively_validates_super_delegating_override(
    monkeypatch,
):
    calls = []

    def create_client(*_args, **_kwargs):
        return _FakeAsyncClient(
            lambda _params: _page_body([{"id": "one"}], total=1),
            calls,
        )

    monkeypatch.setattr(
        external_client_module,
        "create_pinned_async_client",
        create_client,
    )

    class CircularClient(ExternalApiClient):
        async def fetch_incremental(self, since=None, offset=0, limit=1000):
            result = await super().fetch_incremental(since, offset, limit)
            circular = []
            circular.append(circular)
            result["items"][0]["circular"] = circular
            return result

    client = CircularClient(
        "https://api.example.com/items",
        {},
        max_pages=1,
        max_records=1,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(ExternalApiResponseError, match="circular"):
        asyncio.run(client.fetch_all_since(limit=1))


def test_fetch_all_since_bounds_override_result_by_deterministic_json_size():
    calls = []

    class OverrideClient(ExternalApiClient):
        async def fetch_incremental(self, since=None, offset=0, limit=1000):
            calls.append(offset)
            return {
                "total": 2,
                "items": [
                    {"id": "one", "content": "x" * 600_000},
                    {"id": "two", "content": "y" * 600_000},
                ],
                "has_more": False,
            }

    client = OverrideClient(
        "https://api.example.com/items",
        {},
        max_pages=2,
        max_records=10,
        max_aggregate_response_bytes=1024 * 1024,
    )

    with pytest.raises(RuntimeError, match="maximum aggregate response size"):
        asyncio.run(client.fetch_all_since(limit=10))

    assert calls == [0]


def test_override_fallback_does_not_use_json_encoder(monkeypatch):
    class OverrideClient(ExternalApiClient):
        async def fetch_incremental(self, since=None, offset=0, limit=1000):
            return {
                "total": 1,
                "items": [{"content": "\x01" * 250_000}],
                "has_more": False,
            }

    def unexpected_encoder(*_args, **_kwargs):
        raise AssertionError("fallback must not materialize encoded JSON chunks")

    monkeypatch.setattr(
        external_client_module.json,
        "JSONEncoder",
        unexpected_encoder,
    )
    client = OverrideClient(
        "https://api.example.com/items",
        {},
        max_pages=1,
        max_records=1,
        max_aggregate_response_bytes=1024,
    )

    with pytest.raises(RuntimeError, match="maximum aggregate response size"):
        asyncio.run(client.fetch_all_since(limit=1))


def test_bounded_json_size_stops_without_large_escaped_string_copy():
    value = {"content": "\x01" * 250_000}

    tracemalloc.start()
    try:
        result = _bounded_json_size(value, 1024)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result is None
    assert peak_bytes < 1024 * 1024


def test_bounded_json_size_counts_utf8_and_escapes_exactly():
    value = {
        "bool": True,
        "float": 1.25,
        "items": [None, "quote:\" slash:\\ control:\x01", "汉🙂"],
    }
    expected = len(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )

    assert _bounded_json_size(value, expected) == expected
    assert _bounded_json_size(value, expected - 1) is None


@pytest.mark.parametrize(
    "value",
    [
        {"nested": {1: "integer-key"}},
        {"nested": ("tuple-is-not-json",)},
        {"nested": object()},
        {"nested": math.nan},
        {"nested": math.inf},
        {"nested": -math.inf},
    ],
)
def test_bounded_json_size_rejects_non_json_values(value):
    with pytest.raises(ExternalApiResponseError, match="schema is invalid"):
        _bounded_json_size(value, 10_000)


def test_bounded_json_size_rejects_circular_containers():
    circular = []
    circular.append(circular)

    with pytest.raises(ExternalApiResponseError, match="circular"):
        _bounded_json_size(circular, 10_000)


def test_bounded_json_size_rejects_excessive_nesting():
    value = "leaf"
    for _ in range(128):
        value = [value]

    with pytest.raises(ExternalApiResponseError, match="nesting depth"):
        _bounded_json_size(value, 10_000)


def test_fetch_incremental_rejects_invalid_json_with_response_error(monkeypatch):
    client, _, _ = _make_client(
        monkeypatch,
        lambda _params: b"{not-json",
        max_pages=2,
        max_records=10,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(ExternalApiResponseError, match="invalid JSON"):
        asyncio.run(client.fetch_incremental(limit=1))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_fetch_incremental_rejects_non_finite_json_constants(
    monkeypatch,
    constant,
):
    client, _, _ = _make_client(
        monkeypatch,
        lambda _params: (
            f'{{"total":1,"items":[{{"value":{constant}}}]}}'.encode("ascii")
        ),
        max_pages=2,
        max_records=10,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(ExternalApiResponseError, match="invalid JSON"):
        asyncio.run(client.fetch_incremental(limit=1))


@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        b'{"total":true,"items":[]}',
        b'{"total":-1,"items":[]}',
        b'{"total":1,"items":{}}',
        b'{"total":1,"items":["not-an-object"]}',
    ],
)
def test_fetch_incremental_rejects_invalid_response_schema(monkeypatch, body):
    client, _, _ = _make_client(
        monkeypatch,
        lambda _params: body,
        max_pages=2,
        max_records=10,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(
        ExternalApiResponseError,
        match="response schema is invalid",
    ):
        asyncio.run(client.fetch_incremental(limit=1))


def test_fetch_incremental_rejects_items_beyond_reported_total(monkeypatch):
    client, _, _ = _make_client(
        monkeypatch,
        lambda _params: _page_body([{"id": "unexpected"}], total=0),
        max_pages=2,
        max_records=10,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(ExternalApiResponseError, match="pagination is inconsistent"):
        asyncio.run(client.fetch_incremental(offset=0, limit=1))


def test_fetch_all_since_rejects_empty_page_when_total_has_more(monkeypatch):
    client, calls, _ = _make_client(
        monkeypatch,
        lambda _params: _page_body([], total=1),
        max_pages=2,
        max_records=10,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(RuntimeError, match="made no offset progress"):
        asyncio.run(client.fetch_all_since(limit=1))

    assert [call["offset"] for call in calls] == [0]


@pytest.mark.parametrize(
    ("total", "has_more"),
    [
        (1, False),
        (0, True),
    ],
)
def test_fetch_all_since_rejects_override_empty_page_with_more_signal(
    total,
    has_more,
):
    class OverrideClient(ExternalApiClient):
        async def fetch_incremental(self, since=None, offset=0, limit=1000):
            return {"total": total, "items": [], "has_more": has_more}

    client = OverrideClient(
        "https://api.example.com/items",
        {},
        max_pages=2,
        max_records=2,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(RuntimeError, match="made no offset progress"):
        asyncio.run(client.fetch_all_since(limit=1))


@pytest.mark.parametrize(
    ("total", "has_more"),
    [
        (2, False),
        (1, True),
    ],
)
def test_fetch_all_since_rejects_override_has_more_inconsistent_with_total(
    total,
    has_more,
):
    class OverrideClient(ExternalApiClient):
        async def fetch_incremental(self, since=None, offset=0, limit=1000):
            return {
                "total": total,
                "items": [{"id": "one"}],
                "has_more": has_more,
            }

    client = OverrideClient(
        "https://api.example.com/items",
        {},
        max_pages=2,
        max_records=2,
        max_aggregate_response_bytes=10_000,
    )

    with pytest.raises(ExternalApiResponseError, match="has_more is inconsistent"):
        asyncio.run(client.fetch_all_since(limit=1))


def test_fetch_incremental_raises_typed_authentication_error(monkeypatch):
    class UnauthorizedResponse(_FakeResponse):
        status_code = 401

    class UnauthorizedClient(_FakeAsyncClient):
        def stream(self, _method, _url, *, params):
            self._calls.append(dict(params))
            return UnauthorizedResponse(b"")

    monkeypatch.setattr(
        external_client_module,
        "create_pinned_async_client",
        lambda *_args, **_kwargs: UnauthorizedClient(lambda _params: b"", []),
    )
    client = ExternalApiClient("https://api.example.com/items", {})

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(client.fetch_incremental(limit=1))

    authentication_error = getattr(
        external_client_module,
        "ExternalApiAuthenticationError",
        None,
    )
    assert authentication_error is not None
    assert isinstance(exc_info.value, authentication_error)


def test_connection_response_errors_are_not_reported_as_auth_failures(monkeypatch):
    from train_factory.api.routes import external_api_config_routes

    class RejectingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def fetch_incremental(self, **_kwargs):
            raise ExternalApiResponseError("External API returned invalid JSON")

    monkeypatch.setattr(
        external_client_module,
        "ExternalApiClient",
        RejectingClient,
    )
    monkeypatch.setattr(
        external_api_config_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )

    result = asyncio.run(
        external_api_config_routes.test_connection_inline(
            external_api_config_routes.TestConnectionRequest(
                api_url="https://api.example.com/items",
                auth_config={},
            ),
            {"user_id": "user-1"},
        )
    )

    assert result["success"] is False
    assert result["status_code"] == 0
    assert "invalid JSON" in result["message"]


@pytest.mark.parametrize("stored", [False, True])
def test_connection_runtime_errors_are_not_reported_as_auth_failures(
    monkeypatch,
    stored,
):
    from train_factory.api.routes import external_api_config_routes

    class RejectingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def fetch_incremental(self, **_kwargs):
            raise RuntimeError("non-auth runtime failure")

    monkeypatch.setattr(
        external_client_module,
        "ExternalApiClient",
        RejectingClient,
    )
    monkeypatch.setattr(
        external_api_config_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )

    if stored:
        service_module = importlib.import_module(
            "train_factory.storage.services.external_api_config_service"
        )
        service = service_module.external_api_config_service
        monkeypatch.setattr(
            service,
            "get_config",
            lambda config_id: {"config_id": config_id, "user_id": "user-1"},
        )
        monkeypatch.setattr(
            service,
            "get_config_raw",
            lambda config_id: {
                "config_id": config_id,
                "api_url": "https://api.example.com/items",
                "auth_config": {},
            },
        )
        monkeypatch.setattr(service, "update_config", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            external_api_config_routes,
            "verify_resource_ownership",
            lambda resource, *_args: resource,
        )
        result = asyncio.run(
            external_api_config_routes.test_connection_by_id(
                "config-1",
                {"user_id": "user-1"},
            )
        )
    else:
        result = asyncio.run(
            external_api_config_routes.test_connection_inline(
                external_api_config_routes.TestConnectionRequest(
                    api_url="https://api.example.com/items",
                    auth_config={},
                ),
                {"user_id": "user-1"},
            )
        )

    assert result["success"] is False
    assert result["status_code"] == 0
    assert result["message"] == "non-auth runtime failure"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_pages", 0),
        ("max_records", -1),
        ("max_aggregate_response_bytes", True),
    ],
)
def test_external_client_rejects_invalid_pagination_limits(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        ExternalApiClient(
            "https://api.example.com/items",
            {},
            **{field: value},
        )
