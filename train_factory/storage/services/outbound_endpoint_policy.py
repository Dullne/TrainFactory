"""User-scoped policy for validating outbound HTTP endpoints."""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import httpx
import requests
from requests.adapters import HTTPAdapter
from sqlmodel import select

from ...config.settings import get_settings
from ...core.ssrf import (
    ResolvedOutboundURL,
    SSRFError,
    resolve_outbound_url,
    validate_outbound_url,
)
from ..database import get_session
from ..entities.deployment_entity import DeploymentDB
from ..entities.deployment_replica_entity import DeploymentReplicaDB
from ..entities.model_config_entity import ModelConfigDB

logger = logging.getLogger(__name__)


def _extract_hostname(endpoint: str) -> Optional[str]:
    candidate = (endpoint or "").strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    try:
        return (urlparse(candidate).hostname or "").lower() or None
    except (TypeError, ValueError):
        return None


def _add_hosts(values: Iterable[str], hosts: Set[str]) -> None:
    for value in values:
        hostname = _extract_hostname(value)
        if hostname:
            hosts.add(hostname)


Destination = Tuple[str, int]
OutboundTarget = Union[str, ResolvedOutboundURL]
DEFAULT_MAX_OUTBOUND_RESPONSE_BYTES = 10 * 1024 * 1024


class OutboundOriginError(SSRFError):
    """Raised when a pinned client is asked to contact another origin."""

    def __init__(self) -> None:
        super().__init__("Outbound request origin does not match pinned endpoint", 403)


class OutboundResponseTooLargeError(SSRFError):
    """Raised before an untrusted response can consume unbounded API memory."""

    def __init__(self, max_response_bytes: int) -> None:
        super().__init__(
            f"Outbound response exceeds the {max_response_bytes}-byte limit",
            502,
        )


def _validate_response_limit(max_response_bytes: int) -> int:
    if isinstance(max_response_bytes, bool):
        raise ValueError("max_response_bytes must be a positive integer")
    try:
        parsed = int(max_response_bytes)
    except (TypeError, ValueError):
        raise ValueError("max_response_bytes must be a positive integer") from None
    if parsed <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    return parsed


def _reject_oversized_content_length(
    headers: Any,
    max_response_bytes: int,
) -> None:
    raw_length = headers.get("content-length")
    if raw_length is None:
        return
    try:
        content_length = int(raw_length)
    except (TypeError, ValueError):
        return
    if content_length > max_response_bytes:
        raise OutboundResponseTooLargeError(max_response_bytes)


class _LimitedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, max_response_bytes: int) -> None:
        self._stream = stream
        self._max_response_bytes = max_response_bytes

    async def __aiter__(self):
        received = 0
        async for chunk in self._stream:
            received += len(chunk)
            if received > self._max_response_bytes:
                await self._stream.aclose()
                raise OutboundResponseTooLargeError(self._max_response_bytes)
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


def _url_origin(url: Union[str, httpx.URL]) -> Destination:
    parsed = urlparse(str(url))
    hostname = (parsed.hostname or "").encode("idna").decode("ascii").lower()
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise OutboundOriginError() from exc
    return hostname, port


def _require_target_origin(
    url: Union[str, httpx.URL],
    target: ResolvedOutboundURL,
) -> None:
    parsed = urlparse(str(url))
    if parsed.scheme != target.scheme or _url_origin(url) != (
        target.hostname,
        target.port,
    ):
        raise OutboundOriginError()


class PinnedHTTPAdapter(HTTPAdapter):
    """Requests adapter that connects to a resolved IP while preserving TLS identity."""

    def __init__(self, target: ResolvedOutboundURL) -> None:
        self.target = target
        super().__init__(max_retries=0)

    def get_connection_with_tls_context(
        self,
        request,
        verify,
        proxies=None,
        cert=None,
    ):
        _require_target_origin(request.url, self.target)
        _, pool_kwargs = self.build_connection_pool_key_attributes(
            request,
            verify,
            cert,
        )
        if self.target.scheme == "https":
            pool_kwargs["assert_hostname"] = self.target.hostname
            pool_kwargs["server_hostname"] = self.target.hostname
        return self.poolmanager.connection_from_host(
            self.target.ip_address,
            self.target.port,
            self.target.scheme,
            pool_kwargs=pool_kwargs,
        )

    def get_connection(self, url, proxies=None):
        """Pin connections for Requests versions predating the TLS-context API."""
        _require_target_origin(url, self.target)
        pool_kwargs: dict[str, Any] = {}
        if self.target.scheme == "https":
            pool_kwargs["assert_hostname"] = self.target.hostname
            pool_kwargs["server_hostname"] = self.target.hostname
        return self.poolmanager.connection_from_host(
            self.target.ip_address,
            self.target.port,
            self.target.scheme,
            pool_kwargs=pool_kwargs,
        )

    def send(self, request, **kwargs):
        _require_target_origin(request.url, self.target)
        request.headers["Host"] = self.target.host_header
        # 与 async 路径一致：禁用压缩，避免解压炸弹绕过大小上限
        request.headers["Accept-Encoding"] = "identity"
        kwargs["proxies"] = {}
        return super().send(request, **kwargs)


class PinnedAsyncHTTPTransport(httpx.AsyncBaseTransport):
    """HTTPX transport that never resolves the validated hostname again."""

    def __init__(
        self,
        target: ResolvedOutboundURL,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        max_response_bytes: int = DEFAULT_MAX_OUTBOUND_RESPONSE_BYTES,
    ) -> None:
        self.target = target
        self._transport = transport or httpx.AsyncHTTPTransport(trust_env=False)
        self._max_response_bytes = _validate_response_limit(max_response_bytes)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _require_target_origin(request.url, self.target)
        headers = httpx.Headers(request.headers)
        headers["Host"] = self.target.host_header
        # 禁用压缩：_LimitedAsyncByteStream 按传输层字节计数，若不强制
        # identity，gzip/br 可把 10MB 压缩响应解压为 GB 级内存（解压炸弹
        # 绕过字节上限）。
        headers["Accept-Encoding"] = "identity"
        extensions = dict(request.extensions)
        if self.target.scheme == "https":
            extensions["sni_hostname"] = self.target.hostname
        pinned_request = httpx.Request(
            request.method,
            request.url.copy_with(host=self.target.ip_address),
            headers=headers,
            stream=request.stream,
            extensions=extensions,
        )
        response = await self._transport.handle_async_request(pinned_request)
        try:
            # Accept-Encoding is advisory. Reject a noncompliant upstream before
            # HTTPX's decoder can expand bytes beyond the transport-level limit.
            encoding = response.headers.get("content-encoding", "identity")
            if encoding.strip().lower() != "identity":
                raise SSRFError("Encoded outbound responses are not supported", 502)
            _reject_oversized_content_length(
                response.headers,
                self._max_response_bytes,
            )
        except SSRFError:
            await response.aclose()
            raise
        response.stream = _LimitedAsyncByteStream(
            response.stream,
            self._max_response_bytes,
        )
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()


def _extract_destination(endpoint: str) -> Optional[Destination]:
    candidate = (endpoint or "").strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    try:
        parsed = urlparse(candidate)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return hostname, port
    except (TypeError, ValueError):
        return None


def _collect_user_policy(
    user_id: Optional[str],
) -> Tuple[Set[str], Set[Destination]]:
    """Collect administrator hosts and exact system-generated destinations.

    Shared and bound deployments are deliberately excluded: their endpoints are
    user-controlled, so trusting them would let a previously stored malicious
    endpoint bootstrap itself into the allowlist.
    """
    allowed_hosts: Set[str] = set()
    trusted_destinations: Set[Destination] = set()

    configured_hosts = os.getenv("DISCOVER_MODELS_ALLOWED_PRIVATE_HOSTS", "")
    _add_hosts(configured_hosts.split(","), allowed_hosts)

    settings = get_settings()
    _add_hosts(
        [settings.xinference_endpoint or "http://xinference:9997"],
        allowed_hosts,
    )
    # Deployment configs expose the system's shared service through HOST_IP.
    # Trust only this configured service's derived host/port, never a shared
    # deployment's caller-supplied endpoint or every port on the alias host.
    default_endpoint = settings.xinference_endpoint or "http://xinference:9997"
    default_host = _extract_hostname(default_endpoint) or ""
    if default_host in {"xinference", "localhost", "127.0.0.1", "172.17.0.1"} or default_host.startswith(("xf-", "vllm-", "sglang-")):
        from ...deployment.deployment_service import deployment_service

        alias_destination = _extract_destination(
            deployment_service._to_external_endpoint(default_endpoint)
        )
        if alias_destination:
            trusted_destinations.add(alias_destination)

    if user_id == "anonymous" or user_id == "":
        return allowed_hosts, trusted_destinations

    try:
        with get_session() as session:
            owner_filter = (
                DeploymentDB.user_id.is_(None)
                if user_id is None
                else DeploymentDB.user_id == user_id
            )
            statement = select(DeploymentDB).where(
                owner_filter,
                DeploymentDB.deploy_mode == "container",
            )
            deployments = session.exec(statement).all()
            running_deployment_ids = []
            replica_deployment_ids = []
            host_ip = _extract_hostname(os.getenv("HOST_IP", ""))

            for deployment in deployments:
                status = getattr(deployment, "status", "running")
                config = getattr(deployment, "config", None)
                uses_replica_lifecycle = (
                    isinstance(config, dict)
                    and type(config.get("replica_schema_version")) is int
                    and config.get("replica_schema_version") == 1
                    and status in DeploymentDB.VALID_TRANSITIONS
                )
                if uses_replica_lifecycle:
                    replica_deployment_ids.append(deployment.deployment_id)
                    if user_id is not None and status == "running":
                        running_deployment_ids.append(deployment.deployment_id)
                    continue
                if user_id is None or status != "running":
                    continue

                destination = _extract_destination(
                    deployment.xinference_endpoint
                )
                if destination:
                    trusted_destinations.add(destination)

                deployment_port = deployment.port
                if deployment_port is None and destination:
                    deployment_port = destination[1]
                if host_ip and deployment_port is not None:
                    trusted_destinations.add((host_ip, deployment_port))
                running_deployment_ids.append(deployment.deployment_id)

            if replica_deployment_ids:
                replica_statement = select(DeploymentReplicaDB.endpoint).where(
                    DeploymentReplicaDB.deployment_id.in_(replica_deployment_ids)
                )
                for endpoint in session.exec(replica_statement).all():
                    destination = _extract_destination(endpoint)
                    if destination:
                        trusted_destinations.add(destination)

            if running_deployment_ids:
                config_statement = select(ModelConfigDB.api_endpoint).where(
                    ModelConfigDB.user_id == user_id,
                    ModelConfigDB.source_type == "local_deployed",
                    ModelConfigDB.deployment_id.in_(running_deployment_ids),
                )
                for endpoint in session.exec(config_statement).all():
                    destination = _extract_destination(endpoint)
                    if destination:
                        trusted_destinations.add(destination)
    except Exception as exc:
        # A missing/unavailable database must narrow the allowlist, never broaden it.
        logger.debug("Could not load container endpoint allowlist: %s", exc)

    return allowed_hosts, trusted_destinations


def collect_user_allowed_private_hosts(user_id: Optional[str]) -> Set[str]:
    """Return private hosts explicitly trusted by administrators."""
    allowed_hosts, _ = _collect_user_policy(user_id)
    return allowed_hosts


def validate_user_outbound_url(
    url: str,
    user_id: Optional[str] = None,
) -> str:
    """Validate a user-controlled endpoint using the shared SSRF policy."""
    allowed_hosts, trusted_destinations = _collect_user_policy(user_id)
    destination = _extract_destination(url)
    allowed_destinations = set()
    if destination in trusted_destinations:
        # 精确 (host, port) 放行——不放大到 hostname 级全端口，避免
        # "声明一个部署端口即可访问该主机任意端口"的 SSRF 放大。
        allowed_destinations.add(destination)

    return validate_outbound_url(
        url,
        allowed_private_hosts=allowed_hosts,
        allowed_destinations=allowed_destinations,
        allow_private_all=False,
    )


def resolve_user_outbound_url(
    url: str,
    user_id: Optional[str] = None,
) -> ResolvedOutboundURL:
    """Apply the user policy and return a single DNS-pinned destination."""
    allowed_hosts, trusted_destinations = _collect_user_policy(user_id)
    destination = _extract_destination(url)
    allowed_destinations = set()
    if destination in trusted_destinations:
        # 精确 (host, port) 放行——不放大到 hostname 级全端口
        allowed_destinations.add(destination)
    return resolve_outbound_url(
        url,
        allowed_private_hosts=allowed_hosts,
        allowed_destinations=allowed_destinations,
        allow_private_all=False,
    )


def _coerce_target(
    target: OutboundTarget,
    user_id: Optional[str],
) -> ResolvedOutboundURL:
    if isinstance(target, ResolvedOutboundURL):
        return target
    return resolve_user_outbound_url(target, user_id)


def create_pinned_async_client(
    target: OutboundTarget,
    user_id: Optional[str] = None,
    **client_kwargs: Any,
) -> httpx.AsyncClient:
    """Create an HTTPX client bound to one validated IP and original TLS host."""
    resolved = _coerce_target(target, user_id)
    max_response_bytes = _validate_response_limit(
        client_kwargs.pop(
            "max_response_bytes",
            DEFAULT_MAX_OUTBOUND_RESPONSE_BYTES,
        )
    )
    client_kwargs.pop("transport", None)
    client_kwargs.pop("proxy", None)
    client_kwargs.pop("proxies", None)
    client_kwargs.pop("mounts", None)
    client_kwargs["transport"] = PinnedAsyncHTTPTransport(
        resolved,
        max_response_bytes=max_response_bytes,
    )
    client_kwargs["trust_env"] = False
    client_kwargs["follow_redirects"] = False
    return httpx.AsyncClient(**client_kwargs)


def request_user_outbound(
    method: str,
    url: str,
    user_id: Optional[str] = None,
    max_response_bytes: int = DEFAULT_MAX_OUTBOUND_RESPONSE_BYTES,
    **request_kwargs: Any,
) -> requests.Response:
    """Perform one DNS-pinned requests call with redirects and proxies disabled."""
    target = resolve_user_outbound_url(url, user_id)
    max_response_bytes = _validate_response_limit(max_response_bytes)
    request_kwargs["allow_redirects"] = False
    request_kwargs["proxies"] = {}
    request_kwargs["stream"] = True
    with requests.Session() as session:
        session.trust_env = False
        session.mount(f"{target.scheme}://", PinnedHTTPAdapter(target))
        response = session.request(method, target.url, **request_kwargs)
        try:
            _reject_oversized_content_length(
                response.headers,
                max_response_bytes,
            )
            content = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > max_response_bytes:
                    raise OutboundResponseTooLargeError(max_response_bytes)
            response._content = bytes(content)
            response._content_consumed = True
            return response
        finally:
            response.close()


async def async_request_user_outbound(
    method: str,
    url: str,
    user_id: Optional[str] = None,
    **request_kwargs: Any,
) -> httpx.Response:
    """Perform one DNS-pinned HTTPX call with redirects and proxies disabled."""
    client_kwargs: dict[str, Any] = {}
    for key in ("timeout", "verify", "cert", "max_response_bytes"):
        if key in request_kwargs:
            client_kwargs[key] = request_kwargs.pop(key)
    request_kwargs.pop("follow_redirects", None)
    request_kwargs.pop("proxy", None)
    request_kwargs.pop("proxies", None)
    async with create_pinned_async_client(url, user_id, **client_kwargs) as client:
        return await client.request(
            method,
            url,
            follow_redirects=False,
            **request_kwargs,
        )
