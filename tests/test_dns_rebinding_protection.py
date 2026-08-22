"""Regression tests for DNS-rebinding-safe outbound transports."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import httpx
import pytest
import requests

from train_factory.core.ssrf import resolve_outbound_url
from train_factory.storage.services import outbound_endpoint_policy as policy


PUBLIC_IP = "93.184.216.34"
PRIVATE_IP = "127.0.0.1"


def _address(ip: str, port: int = 443) -> tuple[Any, ...]:
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))


def test_resolution_is_reused_by_requests_adapter_without_second_dns_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups = 0

    def rebinding_getaddrinfo(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        nonlocal lookups
        lookups += 1
        return [_address(PUBLIC_IP if lookups == 1 else PRIVATE_IP)]

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)
    target = resolve_outbound_url("https://api.example.test/v1")
    assert target.ip_address == PUBLIC_IP
    assert lookups == 1

    adapter = policy.PinnedHTTPAdapter(target)
    pool_calls: list[tuple[str, int, str, dict[str, Any]]] = []

    class FakePoolManager:
        def connection_from_host(self, host, port, scheme, pool_kwargs):
            pool_calls.append((host, port, scheme, pool_kwargs))
            return object()

    adapter.poolmanager = FakePoolManager()
    prepared = requests.Request("GET", target.url).prepare()
    adapter.get_connection_with_tls_context(prepared, verify=True)

    # requests<2.32 uses get_connection() from HTTPAdapter.send().
    adapter.get_connection(target.url)

    assert lookups == 1
    assert pool_calls[0][:3] == (PUBLIC_IP, 443, "https")
    assert pool_calls[0][3]["assert_hostname"] == "api.example.test"
    assert pool_calls[0][3]["server_hostname"] == "api.example.test"
    assert pool_calls[1][:3] == (PUBLIC_IP, 443, "https")


def test_httpx_transport_connects_to_pinned_ip_and_preserves_https_host_and_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups = 0

    def rebinding_getaddrinfo(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        nonlocal lookups
        lookups += 1
        return [_address(PUBLIC_IP if lookups == 1 else PRIVATE_IP)]

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)
    target = resolve_outbound_url("https://api.example.test:8443/v1")
    observed: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = request.url
        observed["host"] = request.headers["host"]
        observed["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, json={"ok": True})

    transport = policy.PinnedAsyncHTTPTransport(
        target,
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            response = await client.get(target.url)
            assert response.json() == {"ok": True}

    asyncio.run(exercise())

    assert lookups == 1
    assert observed["url"] == httpx.URL(
        f"https://{PUBLIC_IP}:8443/v1"
    )
    assert observed["host"] == "api.example.test:8443"
    assert observed["sni"] == "api.example.test"


def test_literal_ip_target_never_calls_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_lookup(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        raise AssertionError("literal IP targets must not use DNS")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_lookup)
    target = resolve_outbound_url("https://8.8.8.8:8443/v1")

    assert target.ip_address == "8.8.8.8"
    assert target.hostname == "8.8.8.8"
    assert target.port == 8443


def test_httpx_transport_rejects_a_different_origin() -> None:
    target = resolve_outbound_url("https://8.8.8.8/v1")
    transport = policy.PinnedAsyncHTTPTransport(
        target,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
    )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
            with pytest.raises(policy.OutboundOriginError):
                await client.get("https://1.1.1.1/v1")

    asyncio.run(exercise())


def test_requests_helper_forces_no_redirects_or_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = resolve_outbound_url("https://8.8.8.8/v1")
    observed: dict[str, Any] = {}

    class FakeSession:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def mount(self, prefix, adapter):
            observed["mount"] = (prefix, adapter)

        def request(self, method, url, **kwargs):
            observed["request"] = (method, url, kwargs)
            observed["trust_env"] = self.trust_env
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            response._content_consumed = True
            return response

    monkeypatch.setattr(policy, "resolve_user_outbound_url", lambda *_args: target)
    monkeypatch.setattr(policy.requests, "Session", FakeSession)

    policy.request_user_outbound(
        "GET",
        target.url,
        "user-1",
        allow_redirects=True,
        proxies={"https": "http://proxy.invalid"},
    )

    assert observed["trust_env"] is False
    assert observed["mount"][0] == "https://"
    assert isinstance(observed["mount"][1], policy.PinnedHTTPAdapter)
    assert observed["request"][2]["allow_redirects"] is False
    assert observed["request"][2]["proxies"] == {}
    assert observed["request"][2]["stream"] is True


def test_requests_helper_rejects_chunked_response_over_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = resolve_outbound_url("https://8.8.8.8/v1")

    class FakeResponse:
        headers: dict[str, str] = {}
        closed = False

        def iter_content(self, chunk_size):
            assert chunk_size > 0
            yield b"abc"
            yield b"def"

        def close(self):
            self.closed = True

    response = FakeResponse()

    class FakeSession:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def mount(self, *_args):
            return None

        def request(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr(policy, "resolve_user_outbound_url", lambda *_args: target)
    monkeypatch.setattr(policy.requests, "Session", FakeSession)

    with pytest.raises(policy.OutboundResponseTooLargeError):
        policy.request_user_outbound(
            "GET",
            target.url,
            "user-1",
            max_response_bytes=5,
        )

    assert response.closed is True


def test_httpx_transport_rejects_chunked_response_over_limit() -> None:
    target = resolve_outbound_url("https://8.8.8.8/v1")

    class ChunkedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"abc"
            yield b"def"

    transport = policy.PinnedAsyncHTTPTransport(
        target,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=ChunkedStream())
        ),
        max_response_bytes=5,
    )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
            with pytest.raises(policy.OutboundResponseTooLargeError):
                await client.get(target.url)

    asyncio.run(exercise())


def test_httpx_transport_rejects_oversized_content_length_before_reading() -> None:
    target = resolve_outbound_url("https://8.8.8.8/v1")

    class UnexpectedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise AssertionError("oversized declared body must not be read")
            yield b""

    transport = policy.PinnedAsyncHTTPTransport(
        target,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-length": "6"},
                stream=UnexpectedStream(),
            )
        ),
        max_response_bytes=5,
    )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
            with pytest.raises(policy.OutboundResponseTooLargeError):
                await client.get(target.url)

    asyncio.run(exercise())


def test_private_system_destination_allowlist_is_exact_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        policy,
        "_collect_user_policy",
        lambda _user_id: (set(), {("10.23.45.67", 10001)}),
    )

    target = policy.resolve_user_outbound_url(
        "http://10.23.45.67:10001/v1",
        "user-1",
    )
    assert target.ip_address == "10.23.45.67"
    assert target.port == 10001

    with pytest.raises(policy.SSRFError):
        policy.resolve_user_outbound_url(
            "http://10.23.45.67:10002/v1",
            "user-1",
        )
