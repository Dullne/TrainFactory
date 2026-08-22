"""Shared SSRF validator for user-controlled outbound URLs.

Centralizes the private/loopback/link-local blocking that previously lived only
in ``deployment_routes._validate_discovery_endpoint`` so every endpoint that
accepts an external URL (model-config validate/create/test, deep-evaluation,
external-api-config test-connection, deployment discover-models) applies the
same protection.

This module is intentionally dependency-free (stdlib only) so it can be unit
tested without a database or running server. Callers that have a per-user
private-host allowlist (e.g. deployments the user already created) pass it via
``allowed_private_hosts``; callers without one leave it empty, in which case
private addresses are rejected unless ``allow_private_all`` is set.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse


class SSRFError(Exception):
    """Raised when a URL targets a blocked/sensitive host or is malformed.

    ``status_code`` carries the intended HTTP status (400 for malformed input,
    403 for a blocked sensitive address) so routers can map it directly.
    """

    def __init__(self, detail: str, status_code: int = 403) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class ResolvedOutboundURL:
    """A validated outbound URL bound to one already-resolved IP address.

    ``url`` and ``hostname`` retain the caller-facing authority so HTTP Host,
    TLS SNI, and certificate verification can use the original hostname.  The
    transport must connect to ``ip_address`` directly and must not resolve the
    hostname again.
    """

    url: str
    scheme: str
    hostname: str
    port: int
    ip_address: str

    @property
    def host_header(self) -> str:
        host = f"[{self.hostname}]" if ":" in self.hostname else self.hostname
        default_port = 443 if self.scheme == "https" else 80
        return host if self.port == default_port else f"{host}:{self.port}"


def _allow_private_all_from_env() -> bool:
    return os.getenv("DISCOVER_MODELS_ALLOW_PRIVATE_ALL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _normalize_url(url: str) -> Tuple[str, str, str, int]:
    candidate = (url or "").strip()
    if not candidate:
        raise SSRFError("Endpoint 不能为空", status_code=400)

    if "://" not in candidate:
        candidate = f"http://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise SSRFError("Endpoint 仅支持 http/https", status_code=400)
    if not parsed.hostname:
        raise SSRFError("Endpoint 缺少主机名", status_code=400)
    if parsed.username or parsed.password:
        raise SSRFError("Endpoint 不支持账号信息", status_code=400)
    if parsed.query or parsed.fragment:
        raise SSRFError("Endpoint 不支持 query/fragment", status_code=400)

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise SSRFError("Endpoint 端口无效", status_code=400) from exc

    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise SSRFError("Endpoint 主机名无效", status_code=400) from exc

    return candidate.rstrip("/"), parsed.scheme, hostname, port


def _check_ip_address(
    ip_obj: "ipaddress._BaseAddress",
    *,
    hostname: str,
    port: int,
    allowed_private_hosts: Set[str],
    allowed_destinations: Set[Tuple[str, int]],
    allow_private_all: bool,
) -> None:
    # IPv4-mapped IPv6（::ffff:a.b.c.d）先展开为 IPv4 再分类，防止 mapped
    # loopback/link-local（::ffff:127.0.0.1 等）绕过敏感地址封锁。
    if getattr(ip_obj, "ipv4_mapped", None) is not None:
        ip_obj = ip_obj.ipv4_mapped
    # NAT64（64:ff9b::/96）可路由到内嵌 IPv4（含环回/内网），直接拒绝
    if (
        isinstance(ip_obj, ipaddress.IPv6Address)
        and ip_obj in ipaddress.ip_network("64:ff9b::/96")
    ):
        raise SSRFError(f"禁止访问敏感地址: {hostname}", status_code=403)
    if (
        ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
        or ip_obj.is_reserved
    ):
        raise SSRFError(f"禁止访问敏感地址: {hostname}", status_code=403)
    if allow_private_all and ip_obj.is_private:
        return
    if (
        ip_obj.is_private
        and hostname not in allowed_private_hosts
        and str(ip_obj) not in allowed_private_hosts
        and (hostname, port) not in allowed_destinations
    ):
        raise SSRFError(
            "私网 endpoint 未在允许列表，请先创建部署或配置 DISCOVER_MODELS_ALLOWED_PRIVATE_HOSTS",
            status_code=403,
        )


def _resolve_addresses(
    hostname: str,
    port: int,
    *,
    allowed_private_hosts: Set[str],
    allowed_destinations: Set[Tuple[str, int]],
    allow_private_all: bool,
) -> List[str]:
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        _check_ip_address(
            literal_ip,
            hostname=hostname,
            port=port,
            allowed_private_hosts=allowed_private_hosts,
            allowed_destinations=allowed_destinations,
            allow_private_all=allow_private_all,
        )
        return [str(literal_ip)]

    addresses: List[str] = []
    try:
        infos = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror:
        return addresses

    for info in infos:
        ip_obj = ipaddress.ip_address(info[4][0])
        _check_ip_address(
            ip_obj,
            hostname=hostname,
            port=port,
            allowed_private_hosts=allowed_private_hosts,
            allowed_destinations=allowed_destinations,
            allow_private_all=allow_private_all,
        )
        normalized = str(ip_obj)
        if normalized not in addresses:
            addresses.append(normalized)
    return addresses


def _require_resolved_addresses(
    hostname: str,
    port: int,
    *,
    allowed_private_hosts: Set[str],
    allowed_destinations: Set[Tuple[str, int]],
    allow_private_all: bool,
) -> List[str]:
    """Resolve and validate at least one outbound destination address."""
    addresses = _resolve_addresses(
        hostname,
        port,
        allowed_private_hosts=allowed_private_hosts,
        allowed_destinations=allowed_destinations,
        allow_private_all=allow_private_all,
    )
    if not addresses:
        raise SSRFError(f"Endpoint 主机名无法解析: {hostname}", status_code=400)
    return addresses


def resolve_outbound_url(
    url: str,
    allowed_private_hosts: Optional[Set[str]] = None,
    allow_private_all: Optional[bool] = None,
    allowed_destinations: Optional[Set[Tuple[str, int]]] = None,
) -> ResolvedOutboundURL:
    """Validate and resolve an endpoint once for a DNS-pinned transport."""
    candidate, scheme, hostname, port = _normalize_url(url)
    if hostname in {"localhost", "localhost.localdomain"}:
        raise SSRFError("禁止访问 localhost", status_code=403)

    if allow_private_all is None:
        allow_private_all = _allow_private_all_from_env()
    addresses = _require_resolved_addresses(
        hostname,
        port,
        allowed_private_hosts=allowed_private_hosts or set(),
        allowed_destinations=allowed_destinations or set(),
        allow_private_all=allow_private_all,
    )

    return ResolvedOutboundURL(
        url=candidate,
        scheme=scheme,
        hostname=hostname,
        port=port,
        ip_address=addresses[0],
    )


def validate_outbound_url(
    url: str,
    allowed_private_hosts: Optional[Set[str]] = None,
    allow_private_all: Optional[bool] = None,
    allowed_destinations: Optional[Set[Tuple[str, int]]] = None,
) -> str:
    """Validate a user-supplied URL and block SSRF targets.

    Args:
        url: The raw URL string from the user.
        allowed_private_hosts: Explicitly allowed private hostnames/IPs (lowercased).
            Bypasses the private-address block for those hosts only.
        allow_private_all: If True, allow any private address (still blocks
            loopback/link-local/multicast/unspecified/reserved). If None, read
            from the DISCOVER_MODELS_ALLOW_PRIVATE_ALL env var.
        allowed_destinations: Exact (hostname, port) pairs allowed to bypass the
            private-address block. Unlike ``allowed_private_hosts`` this does
            NOT open every port of the host — only the exact destination.

    Returns:
        The normalized URL (scheme guaranteed, trailing slash stripped).

    Raises:
        SSRFError: if the URL is empty, malformed, non-http(s), carries
            userinfo/query/fragment, or resolves to a sensitive address.
    """
    candidate, _, hostname, port = _normalize_url(url)
    if hostname in {"localhost", "localhost.localdomain"}:
        raise SSRFError("禁止访问 localhost", status_code=403)

    if allow_private_all is None:
        allow_private_all = _allow_private_all_from_env()
    _require_resolved_addresses(
        hostname,
        port,
        allowed_private_hosts=allowed_private_hosts or set(),
        allowed_destinations=allowed_destinations or set(),
        allow_private_all=allow_private_all,
    )

    return candidate.rstrip("/")
