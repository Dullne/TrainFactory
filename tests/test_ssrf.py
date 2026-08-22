"""Tests for the shared SSRF validator (train_factory.core.ssrf).

These are pure-function tests: no network egress is performed for the blocking
cases (literal IP / localhost hostname), and the "public allow" case uses a
literal public IP so it does not depend on DNS resolution.
"""
import socket

import pytest

from train_factory.core.ssrf import (
    SSRFError,
    resolve_outbound_url,
    validate_outbound_url,
)


PUBLIC_IP = "93.184.216.34"
PRIVATE_IP = "10.23.45.67"


def _address(ip: str, port: int = 443):
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://localhost.localdomain/",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://0.0.0.0/",
    ],
)
def test_blocks_sensitive_addresses(url):
    with pytest.raises(SSRFError) as exc_info:
        validate_outbound_url(url)
    # Sensitive/loopback/private targets must be 403, not a 400 format error
    assert exc_info.value.status_code == 403


def test_allows_public_ip():
    # Literal public IP avoids any DNS dependency in CI
    result = validate_outbound_url("http://8.8.8.8/v1")
    assert result == "http://8.8.8.8/v1"


@pytest.mark.parametrize("validator", [validate_outbound_url, resolve_outbound_url])
@pytest.mark.parametrize("resolution", ["empty", "error"])
def test_domain_validation_fails_closed_when_dns_has_no_addresses(
    monkeypatch,
    validator,
    resolution,
):
    def resolve(*_args, **_kwargs):
        if resolution == "error":
            raise socket.gaierror("synthetic DNS failure")
        return []

    monkeypatch.setattr(socket, "getaddrinfo", resolve)

    with pytest.raises(SSRFError) as exc_info:
        validator("https://api.example.test/v1")

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("validator", [validate_outbound_url, resolve_outbound_url])
def test_domain_validation_allows_only_public_dns_answers(monkeypatch, validator):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [_address(PUBLIC_IP)],
    )

    result = validator("https://api.example.test/v1")

    assert getattr(result, "url", result) == "https://api.example.test/v1"
    if validator is resolve_outbound_url:
        assert result.ip_address == PUBLIC_IP


@pytest.mark.parametrize(
    "answers",
    [
        [PRIVATE_IP],
        [PUBLIC_IP, PRIVATE_IP],
    ],
)
@pytest.mark.parametrize("validator", [validate_outbound_url, resolve_outbound_url])
def test_domain_validation_rejects_private_or_mixed_dns_answers(
    monkeypatch,
    answers,
    validator,
):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [_address(ip) for ip in answers],
    )

    with pytest.raises(SSRFError) as exc_info:
        validator("https://api.example.test/v1")

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize("validator", [validate_outbound_url, resolve_outbound_url])
@pytest.mark.parametrize(
    "blocked_url",
    (
        "https://api.example.test:9443/v1",
        "https://other.example.test:8443/v1",
    ),
)
def test_private_domain_allowlist_is_bound_to_exact_host_and_port(
    monkeypatch,
    validator,
    blocked_url,
):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [_address(PRIVATE_IP, port)],
    )
    allowed_destination = {("api.example.test", 8443)}

    result = validator(
        "https://api.example.test:8443/v1",
        allowed_destinations=allowed_destination,
    )

    assert getattr(result, "url", result) == "https://api.example.test:8443/v1"
    with pytest.raises(SSRFError) as exc_info:
        validator(
            blocked_url,
            allowed_destinations=allowed_destination,
        )

    assert exc_info.value.status_code == 403


def test_auto_prepends_http_scheme():
    # No scheme -> auto http://; public host (literal IP) still allowed
    assert validate_outbound_url("8.8.8.8") == "http://8.8.8.8"


def test_rejects_non_http_scheme():
    with pytest.raises(SSRFError) as exc_info:
        validate_outbound_url("file:///etc/passwd")
    assert exc_info.value.status_code == 400


def test_rejects_userinfo_and_query_and_fragment():
    with pytest.raises(SSRFError) as exc_info:
        validate_outbound_url("http://u:p@8.8.8.8/v1")
    assert exc_info.value.status_code == 400

    with pytest.raises(SSRFError) as exc_info:
        validate_outbound_url("https://8.8.8.8/v1?x=1")
    assert exc_info.value.status_code == 400

    with pytest.raises(SSRFError) as exc_info:
        validate_outbound_url("https://8.8.8.8/v1#frag")
    assert exc_info.value.status_code == 400


def test_rejects_empty():
    with pytest.raises(SSRFError) as exc_info:
        validate_outbound_url("")
    assert exc_info.value.status_code == 400

    with pytest.raises(SSRFError):
        validate_outbound_url("   ")


def test_private_host_allowed_via_allowlist():
    # Default: private blocked
    with pytest.raises(SSRFError):
        validate_outbound_url("http://10.0.0.5/")
    # With hostname in allowlist -> allowed
    result = validate_outbound_url(
        "http://10.0.0.5/v1", allowed_private_hosts={"10.0.0.5"}
    )
    assert result == "http://10.0.0.5/v1"


def test_allow_private_all_permits_private():
    result = validate_outbound_url("http://10.0.0.5/", allow_private_all=True)
    assert result == "http://10.0.0.5"
    # But loopback/link-local are still blocked even with allow_private_all
    with pytest.raises(SSRFError):
        validate_outbound_url("http://127.0.0.1/", allow_private_all=True)
    with pytest.raises(SSRFError):
        validate_outbound_url("http://169.254.169.254/", allow_private_all=True)


def test_strips_trailing_slash():
    assert validate_outbound_url("http://8.8.8.8/v1/").rstrip("/") == "http://8.8.8.8/v1"


def test_normalize_api_endpoint_applies_user_scoped_ssrf_guard(monkeypatch):
    """model_config_service.normalize_api_endpoint must block SSRF targets."""
    from train_factory.storage.services.model_config_service import (
        normalize_api_endpoint,
    )

    # Loopback, metadata, and arbitrary private targets are blocked.
    for bad in [
        "http://127.0.0.1:11434",
        "http://169.254.169.254",
        "http://localhost:9997",
        "http://10.0.0.5:8080",
    ]:
        with pytest.raises(SSRFError):
            normalize_api_endpoint(bad, "custom")

    # public allowed and normalized to /v1 for openai-compatible provider
    # （用字面公网 IP，避免依赖 DNS 的公网解析——本地 DNS 可能被污染）
    assert normalize_api_endpoint("https://8.8.8.8", "openai") == "https://8.8.8.8/v1"
    # Administrators can explicitly allow a private inference host.
    monkeypatch.setenv("DISCOVER_MODELS_ALLOWED_PRIVATE_HOSTS", "10.0.0.5")
    assert (
        normalize_api_endpoint("http://10.0.0.5:8080", "custom", "user-1")
        == "http://10.0.0.5:8080/v1"
    )
