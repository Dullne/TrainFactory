"""Side-effect-free parsing for the externally visible application origin."""

from ipaddress import ip_address
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


def _invalid(setting_name: str, category: str) -> ValueError:
    return ValueError(f"{setting_name} has invalid {category}")


def parse_public_origin(value: object, setting_name: str) -> str:
    """Return a canonical HTTP(S) origin without exposing invalid input."""
    if not isinstance(value, str) or not value:
        raise _invalid(setting_name, "syntax")

    try:
        parsed = urlsplit(value)
        parsed.hostname
        parsed.port
    except (TypeError, ValueError):
        raise _invalid(setting_name, "syntax") from None

    if parsed.scheme.lower() not in {"http", "https"}:
        raise _invalid(setting_name, "scheme")
    if parsed.username is not None or parsed.password is not None:
        raise _invalid(setting_name, "credentials")
    if "?" in value:
        raise _invalid(setting_name, "query")
    if "#" in value:
        raise _invalid(setting_name, "fragment")
    if parsed.path not in {"", "/"}:
        raise _invalid(setting_name, "path")
    if "\\" in value:
        raise _invalid(setting_name, "syntax")

    try:
        validated = _HTTP_URL_ADAPTER.validate_python(value)
    except ValidationError:
        raise _invalid(setting_name, "syntax") from None

    if validated.host is None:
        raise _invalid(setting_name, "syntax")
    canonical_host = validated.host.lower()
    if ":" in canonical_host and not canonical_host.startswith("["):
        canonical_host = f"[{canonical_host}]"
    port = validated.port
    default_port = (validated.scheme == "http" and port == 80) or (
        validated.scheme == "https" and port == 443
    )
    port_suffix = "" if port is None or default_port else f":{port}"
    canonical = f"{validated.scheme}://{canonical_host}{port_suffix}"
    comparable = value[:-1] if value.endswith("/") else value
    if comparable != canonical:
        raise _invalid(setting_name, "canonical")
    return canonical


def is_loopback_bind_address(value: object, setting_name: str) -> bool:
    """Classify a literal bind address without accepting ambiguous host forms."""
    if not isinstance(value, str) or not value:
        raise _invalid(setting_name, "address")
    if value.lower() == "localhost":
        return True
    try:
        address = ip_address(value)
    except ValueError:
        raise _invalid(setting_name, "address") from None
    if getattr(address, "ipv4_mapped", None) is not None:
        raise _invalid(setting_name, "address")
    return address.is_loopback


def is_loopback_public_origin(origin: str) -> bool:
    """Return whether a previously parsed canonical origin names loopback."""
    hostname = urlsplit(origin).hostname
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        address = ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback and getattr(address, "ipv4_mapped", None) is None
