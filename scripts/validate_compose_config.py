"""Validate rendered Docker Compose transport policy without echoing values."""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("compose validator bootstrap is unavailable")


def _bootstrap_realpath(value: str) -> str:
    return _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(value))


_script_path = _bootstrap_realpath(__file__)
_script_dir = _bootstrap_os.path.dirname(_script_path)
_root_entry = _bootstrap_os.path.dirname(_script_dir)
_trusted_prefixes = {
    _bootstrap_realpath(sys.base_prefix),
    _bootstrap_realpath(sys.prefix),
}
_pythonpath_entries = {
    _bootstrap_realpath(entry)
    for entry in _bootstrap_os.environ.get("PYTHONPATH", "").split(
        _bootstrap_os.pathsep
    )
    if entry
}


def _is_interpreter_path(candidate: str) -> bool:
    for prefix in _trusted_prefixes:
        try:
            if _bootstrap_os.path.commonpath((candidate, prefix)) == prefix:
                return True
        except ValueError:
            continue
    return False


_trusted_sys_path: list[str] = []
_seen_paths = {_root_entry}
for _entry in sys.path:
    _candidate = _bootstrap_realpath(_entry or _bootstrap_os.getcwd())
    if _candidate in {_root_entry, _script_dir} or _candidate in _seen_paths:
        continue
    if _candidate in _pythonpath_entries and not _is_interpreter_path(_candidate):
        continue
    if not _is_interpreter_path(_candidate):
        continue
    _seen_paths.add(_candidate)
    _trusted_sys_path.append(_entry)

# Import project and standard-library modules only after untrusted search paths
# and the script directory have been removed.
sys.path[:] = [_root_entry, *_trusted_sys_path]

import argparse  # noqa: E402
from ipaddress import ip_address  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import stat  # noqa: E402
from typing import Any  # noqa: E402

ROOT_DIR = Path(_root_entry)

from train_factory.config.public_origin import (  # noqa: E402
    is_loopback_public_origin,
    parse_public_origin,
)
from train_factory.config.secret_files import (  # noqa: E402
    classify_required_secret_sources,
)

API_SERVICES = ("train-factory-api", "train-factory-api-dev")
REQUIRED_TRANSPORT_ENVIRONMENT = (
    "AUTH_ENABLED",
    "HOST_BIND_ADDRESS",
    "PUBLIC_BASE_URL",
    "AUTH_COOKIE_SECURE",
)
REQUIRED_IMAGE_LOCK_KEYS = (
    "API_BASE_IMAGE",
    "API_TEST_BASE_IMAGE",
    "WEB_NODE_BUILD_IMAGE",
    "WEB_NGINX_IMAGE",
    "MYSQL_IMAGE",
    "NODE_IMAGE",
    "PLAYWRIGHT_IMAGE",
    "VLLM_IMAGE",
    "SGLANG_IMAGE",
    "XINFERENCE_IMAGE",
    "ETCD_IMAGE",
    "MINIO_IMAGE",
    "MILVUS_IMAGE",
)
_IMAGE_REFERENCE_PATTERN = re.compile(
    r"^(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*:"
    r"[A-Za-z0-9_][A-Za-z0-9_.-]*@sha256:([0-9a-f]{64})$"
)


def parse_images_lock_text(text: str) -> dict[str, str]:
    """Parse the exact release image lock without exposing invalid values."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            raise ValueError("images lock is invalid")
        name, value = line.split("=", 1)
        if name not in REQUIRED_IMAGE_LOCK_KEYS or name in values:
            raise ValueError("images lock is invalid")
        match = _IMAGE_REFERENCE_PATTERN.fullmatch(value)
        if match is None or match.group(1) == "0" * 64:
            raise ValueError("images lock is invalid")
        values[name] = value
    if tuple(values) != REQUIRED_IMAGE_LOCK_KEYS:
        raise ValueError("images lock is invalid")
    return values


def load_images_lock(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ValueError("images lock is invalid") from None
    return parse_images_lock_text(text)


def _parse_bool(value: object) -> bool | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _deployment_bind_address(value: object):
    if not isinstance(value, str) or not value:
        raise ValueError("deployment bind address is invalid")
    try:
        address = ip_address(value)
    except ValueError:
        raise ValueError("deployment bind address is invalid") from None
    if getattr(address, "ipv4_mapped", None) is not None:
        raise ValueError("deployment bind address is invalid")
    return address


def _service_environment(service: object) -> dict[str, Any]:
    if not isinstance(service, dict):
        return {}
    environment = service.get("environment")
    return environment if isinstance(environment, dict) else {}


def _nonempty(environment: dict[str, Any], name: str) -> bool:
    return isinstance(environment.get(name), str) and environment[name] != ""


def _decode_compose_literal_path(value: object) -> str | None:
    if not isinstance(value, str) or value == "":
        return None
    decoded: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "$":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(value) or value[index + 1] != "$":
            return None
        decoded.append("$")
        index += 2
    return "".join(decoded)


def _attached_secret_targets(service: object) -> dict[str, list[str | None]]:
    if not isinstance(service, dict) or not isinstance(service.get("secrets"), list):
        return {}
    targets: dict[str, list[str | None]] = {}
    for entry in service["secrets"]:
        if isinstance(entry, str):
            targets.setdefault(entry, []).append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("source"), str):
            source = entry["source"]
            target = entry.get("target", source)
            targets.setdefault(source, []).append(
                target if isinstance(target, str) else None
            )
    return targets


def _validate_compose_secrets(
    compose: dict[str, Any],
    api_service_names: tuple[str, ...],
) -> list[str]:
    services = compose["services"]
    mysql_environment = _service_environment(services.get("mysql"))
    api_environments = {
        name: _service_environment(services.get(name)) for name in api_service_names
    }
    file_mode = any(
        _nonempty(mysql_environment, name)
        for name in ("MYSQL_ROOT_PASSWORD_FILE", "MYSQL_PASSWORD_FILE")
    ) or any(
        _nonempty(environment, name)
        for environment in api_environments.values()
        for name in (
            "MYSQL_URL_FILE",
            "JWT_SECRET_KEY_FILE",
            "DEFAULT_ADMIN_PASSWORD_FILE",
        )
    )
    errors: list[str] = []

    if not file_mode:
        for name in ("MYSQL_ROOT_PASSWORD", "MYSQL_PASSWORD"):
            if not _nonempty(mysql_environment, name):
                errors.append(f"service mysql is missing required environment {name}")
        normalized: dict[str, Any] = {
            "MYSQL_ROOT_PASSWORD": mysql_environment.get("MYSQL_ROOT_PASSWORD"),
            "MYSQL_PASSWORD": mysql_environment.get("MYSQL_PASSWORD"),
        }
        for service_name, environment in api_environments.items():
            for name in (
                "MYSQL_APP_USER",
                "MYSQL_APP_PASSWORD",
                "MYSQL_HOST",
                "MYSQL_DATABASE",
                "JWT_SECRET_KEY",
                "DEFAULT_ADMIN_PASSWORD",
            ):
                if not _nonempty(environment, name):
                    errors.append(
                        f"service {service_name} is missing required environment {name}"
                    )
            for name in (
                "MYSQL_URL_FILE",
                "JWT_SECRET_KEY_FILE",
                "DEFAULT_ADMIN_PASSWORD_FILE",
            ):
                if name in environment:
                    errors.append(
                        f"service {service_name} has conflicting environment {name}"
                    )
            if "MYSQL_URL" in environment:
                errors.append(
                    f"service {service_name} has conflicting environment MYSQL_URL"
                )
            if (
                _nonempty(mysql_environment, "MYSQL_PASSWORD")
                and _nonempty(environment, "MYSQL_APP_PASSWORD")
                and mysql_environment["MYSQL_PASSWORD"]
                != environment["MYSQL_APP_PASSWORD"]
            ):
                errors.append(
                    "service mysql environment MYSQL_PASSWORD does not match "
                    f"service {service_name} environment MYSQL_APP_PASSWORD"
                )
            current = {
                "MYSQL_URL": "derived"
                if all(
                    _nonempty(environment, name)
                    for name in (
                        "MYSQL_APP_USER",
                        "MYSQL_APP_PASSWORD",
                        "MYSQL_HOST",
                        "MYSQL_DATABASE",
                    )
                )
                else None,
                "JWT_SECRET_KEY": environment.get("JWT_SECRET_KEY"),
                "DEFAULT_ADMIN_PASSWORD": environment.get("DEFAULT_ADMIN_PASSWORD"),
            }
            if "MYSQL_URL" not in normalized:
                normalized.update(current)
            elif any(normalized.get(name) != current.get(name) for name in current):
                errors.append(
                    f"service {service_name} secret environment does not match train-factory-api"
                )
        try:
            classify_required_secret_sources(normalized)
        except ValueError:
            if not errors:
                errors.append("compose required secret sources are incomplete")
        return errors

    for name in ("MYSQL_ROOT_PASSWORD", "MYSQL_PASSWORD"):
        if name in mysql_environment:
            errors.append(f"service mysql has conflicting environment {name}")
    mysql_file_bindings = {
        "MYSQL_ROOT_PASSWORD_FILE": "/run/secrets/mysql_root_password",
        "MYSQL_PASSWORD_FILE": "/run/secrets/mysql_app_password",
    }
    for name, expected_path in mysql_file_bindings.items():
        if not _nonempty(mysql_environment, name):
            errors.append(f"service mysql is missing required environment {name}")
        elif mysql_environment[name] != expected_path:
            errors.append(f"service mysql has invalid environment {name}")

    normalized = {
        "MYSQL_ROOT_PASSWORD_FILE": mysql_environment.get("MYSQL_ROOT_PASSWORD_FILE"),
        "MYSQL_PASSWORD_FILE": mysql_environment.get("MYSQL_PASSWORD_FILE"),
    }
    for service_name, environment in api_environments.items():
        for name in (
            "MYSQL_URL",
            "MYSQL_APP_PASSWORD",
            "JWT_SECRET_KEY",
            "DEFAULT_ADMIN_PASSWORD",
        ):
            if name in environment and (
                name == "MYSQL_URL"
                or type(environment[name]) is not str
                or environment[name] != ""
            ):
                errors.append(
                    f"service {service_name} has conflicting environment {name}"
                )
        api_file_bindings = {
            "MYSQL_URL_FILE": "/run/secrets/mysql_url",
            "JWT_SECRET_KEY_FILE": "/run/secrets/jwt_secret_key",
            "DEFAULT_ADMIN_PASSWORD_FILE": "/run/secrets/default_admin_password",
        }
        binding_is_valid = True
        for name, expected_path in api_file_bindings.items():
            if not _nonempty(environment, name):
                errors.append(
                    f"service {service_name} is missing required environment {name}"
                )
                binding_is_valid = False
            elif environment[name] != expected_path:
                errors.append(f"service {service_name} has invalid environment {name}")
                binding_is_valid = False
        current = {
            "MYSQL_URL_FILE": environment.get("MYSQL_URL_FILE"),
            "JWT_SECRET_KEY_FILE": environment.get("JWT_SECRET_KEY_FILE"),
            "DEFAULT_ADMIN_PASSWORD_FILE": environment.get(
                "DEFAULT_ADMIN_PASSWORD_FILE"
            ),
        }
        if "MYSQL_URL_FILE" not in normalized:
            normalized.update(current)
        elif binding_is_valid and any(
            normalized.get(name) != current.get(name) for name in current
        ):
            errors.append(
                f"service {service_name} secret file environment does not match train-factory-api"
            )

    try:
        classify_required_secret_sources(normalized)
    except ValueError:
        if not errors:
            errors.append("compose required secret sources are incomplete")

    required_attachments = {
        "mysql": ("mysql_root_password", "mysql_app_password"),
        **{
            name: ("mysql_url", "jwt_secret_key", "default_admin_password")
            for name in api_service_names
        },
    }
    for service_name, required_sources in required_attachments.items():
        attached = _attached_secret_targets(services.get(service_name))
        for source in required_sources:
            if source not in attached:
                errors.append(
                    f"service {service_name} is missing required secret {source}"
                )
            elif attached[source] != [source]:
                errors.append(
                    f"service {service_name} has invalid secret attachment {source}"
                )

    secret_definitions = compose.get("secrets")
    if not isinstance(secret_definitions, dict):
        secret_definitions = {}
    for source in (
        "mysql_root_password",
        "mysql_app_password",
        "mysql_url",
        "jwt_secret_key",
        "default_admin_password",
    ):
        definition = secret_definitions.get(source)
        host_file = definition.get("file") if isinstance(definition, dict) else None
        decoded_host_file = _decode_compose_literal_path(host_file)
        valid = decoded_host_file is not None
        if valid:
            path = Path(decoded_host_file)
            try:
                metadata = path.lstat()
                valid = (
                    path.is_absolute()
                    and not path.is_symlink()
                    and stat.S_ISREG(metadata.st_mode)
                )
            except OSError:
                valid = False
        if not valid:
            errors.append(f"secret {source} has invalid host file")
    return errors


def validate_inspected_secret_environment(
    inspect: object,
    *,
    service_name: str,
    mode: str,
) -> list[str]:
    """Validate docker-inspect Config.Env without returning any values."""
    if mode not in {"direct", "files"}:
        return [f"service {service_name} secret mode is invalid"]
    if not isinstance(inspect, list) or len(inspect) != 1:
        return [f"service {service_name} inspect result is invalid"]
    container = inspect[0]
    config = container.get("Config") if isinstance(container, dict) else None
    entries = config.get("Env") if isinstance(config, dict) else None
    if not isinstance(entries, list) or not all(
        isinstance(entry, str) and "=" in entry for entry in entries
    ):
        return [f"service {service_name} inspect result is invalid"]
    environment = {entry.split("=", 1)[0]: entry.split("=", 1)[1] for entry in entries}
    if mode != "files":
        return []
    forbidden = (
        ("MYSQL_ROOT_PASSWORD", "MYSQL_PASSWORD")
        if service_name == "mysql"
        else (
            "MYSQL_URL",
            "MYSQL_APP_PASSWORD",
            "JWT_SECRET_KEY",
            "DEFAULT_ADMIN_PASSWORD",
        )
    )
    return [
        f"service {service_name} has conflicting environment {name}"
        for name in forbidden
        if name in environment
    ]


def _nvidia_gpu_reservation(service: dict[str, Any]) -> tuple[bool, bool]:
    """Return (shape_is_valid, has_nvidia_gpu) for a rendered service."""
    deploy = service.get("deploy")
    if deploy is None:
        return True, False
    if not isinstance(deploy, dict):
        return False, False
    resources = deploy.get("resources")
    if resources is None:
        return True, False
    if not isinstance(resources, dict):
        return False, False
    reservations = resources.get("reservations")
    if reservations is None:
        return True, False
    if not isinstance(reservations, dict):
        return False, False
    devices = reservations.get("devices")
    if devices is None or devices == []:
        return True, False
    if not isinstance(devices, list) or len(devices) != 1:
        return False, False
    device = devices[0]
    if not isinstance(device, dict) or device.get("driver") != "nvidia":
        return False, False
    count = device.get("count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or (count != -1 and count <= 0)
    ):
        return False, False
    capabilities = device.get("capabilities")
    if not isinstance(capabilities, list) or "gpu" not in capabilities:
        return False, False
    if not all(isinstance(capability, str) for capability in capabilities):
        return False, False
    return True, True


def _contains_nvidia_reference(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        return (
            "nvidia" in normalized
            or "/dev/dxg" in normalized
            or "nvidia.com/gpu" in normalized
        )
    if isinstance(value, list):
        return any(_contains_nvidia_reference(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_nvidia_reference(item) for item in value.values())
    return False


def _has_unsupported_gpu_activation(service: dict[str, Any]) -> bool:
    for name in (
        "gpus",
        "runtime",
        "devices",
        "device_requests",
        "device_cgroup_rules",
    ):
        value = service.get(name)
        if value is not None and value != "" and value != []:
            return True
    privileged = service.get("privileged")
    if privileged is not None and privileged is not False:
        return True
    return _contains_nvidia_reference(service.get("volumes"))


def _valid_required_visible_devices(value: str | None) -> bool:
    if value is None or value == "all":
        return True
    selectors = value.split(",")
    return bool(selectors) and all(
        selector.isdecimal()
        or re.fullmatch(r"GPU-[0-9A-Fa-f]+(?:-[0-9A-Fa-f]+)+", selector) is not None
        for selector in selectors
    )


def _valid_required_driver_capabilities(value: str | None) -> bool:
    if value is None:
        return True
    capabilities = value.split(",")
    return all(capabilities) and {"compute", "utility"}.issubset(capabilities)


def _validate_gpu_preflight_policy(
    service: dict[str, Any], service_name: str
) -> list[str]:
    environment = _service_environment(service)
    mode = environment.get("GPU_PREFLIGHT_MODE")
    if mode is None:
        return [
            f"service {service_name} is missing required environment GPU_PREFLIGHT_MODE"
        ]
    if not isinstance(mode, str) or mode not in {"required", "off"}:
        return [f"service {service_name} has invalid environment GPU_PREFLIGHT_MODE"]

    override = environment.get("NVIDIA_DISABLE_REQUIRE")
    if override is not None and (
        not isinstance(override, str) or override not in {"0", "1", ""}
    ):
        return [
            f"service {service_name} has invalid environment NVIDIA_DISABLE_REQUIRE"
        ]
    if _has_unsupported_gpu_activation(service):
        return [f"service {service_name} has unsupported GPU activation configuration"]

    valid_reservation, has_gpu = _nvidia_gpu_reservation(service)
    if not valid_reservation:
        return [f"service {service_name} has invalid NVIDIA GPU reservation"]
    if (mode == "required") != has_gpu:
        return [f"service {service_name} violates GPU preflight policy"]
    if override in {"0", "1"} and (mode != "required" or not has_gpu):
        return [f"service {service_name} violates GPU preflight policy"]
    if override == "" and (mode != "off" or has_gpu):
        return [f"service {service_name} violates GPU preflight policy"]
    visible_devices = environment.get("NVIDIA_VISIBLE_DEVICES")
    driver_capabilities = environment.get("NVIDIA_DRIVER_CAPABILITIES")
    if visible_devices is not None and not isinstance(visible_devices, str):
        return [f"service {service_name} violates GPU preflight policy"]
    if driver_capabilities is not None and not isinstance(driver_capabilities, str):
        return [f"service {service_name} violates GPU preflight policy"]
    if mode == "off" and (visible_devices != "void" or driver_capabilities != ""):
        return [f"service {service_name} violates GPU preflight policy"]
    if mode == "required" and (
        not _valid_required_visible_devices(visible_devices)
        or not _valid_required_driver_capabilities(driver_capabilities)
    ):
        return [f"service {service_name} violates GPU preflight policy"]
    return []


def validate_compose_config(
    compose: object,
    *,
    profile: str,
) -> list[str]:
    """Return privacy-safe policy errors for an already rendered config."""
    if profile not in {"minimal", "all"}:
        return ["compose validation profile is invalid"]
    if not isinstance(compose, dict) or not isinstance(compose.get("services"), dict):
        return ["compose JSON is missing services"]

    services = compose["services"]
    expected_bind: str | None = None
    expected_loopback: bool | None = None
    expected_secure: bool | None = None
    expected_auth_enabled: bool | None = None
    errors: list[str] = []

    required_services = ("train-factory-api", "train-factory-web")
    if profile == "all":
        required_services = (
            "train-factory-api",
            "train-factory-web",
            "train-factory-api-dev",
            "train-factory-web-dev",
        )

    for service_name in required_services:
        if service_name not in services:
            errors.append(
                f"compose profile {profile} is missing service {service_name}"
            )

    invalid_required_services: set[str] = set()
    for service_name in required_services:
        if service_name not in services:
            invalid_required_services.add(service_name)
            continue
        service = services[service_name]
        if not isinstance(service, dict):
            errors.append(f"service {service_name} definition is invalid")
            invalid_required_services.add(service_name)
            continue
        ports = service.get("ports")
        if not isinstance(ports, list) or not ports:
            errors.append(f"service {service_name} must publish at least one port")
            invalid_required_services.add(service_name)

    for service_name in API_SERVICES:
        if service_name not in services or service_name in invalid_required_services:
            continue
        environment = _service_environment(services[service_name])
        missing = [
            name
            for name in REQUIRED_TRANSPORT_ENVIRONMENT
            if name not in environment or environment[name] is None
        ]
        for name in missing:
            errors.append(
                f"service {service_name} is missing required environment {name}"
            )
        if missing:
            continue

        bind = environment["HOST_BIND_ADDRESS"]
        public_url = environment["PUBLIC_BASE_URL"]
        secure = _parse_bool(environment["AUTH_COOKIE_SECURE"])
        auth_enabled = _parse_bool(environment.get("AUTH_ENABLED", "true"))
        if auth_enabled is None:
            errors.append(
                f"service {service_name} has invalid environment AUTH_ENABLED"
            )
            continue
        if secure is None:
            errors.append(
                f"service {service_name} has invalid environment AUTH_COOKIE_SECURE"
            )
            continue
        try:
            address = _deployment_bind_address(bind)
            loopback = address.is_loopback
        except ValueError:
            errors.append(
                f"service {service_name} has invalid environment HOST_BIND_ADDRESS"
            )
            continue
        try:
            origin = parse_public_origin(public_url, "PUBLIC_BASE_URL")
        except ValueError:
            errors.append(
                f"service {service_name} has invalid environment PUBLIC_BASE_URL"
            )
            continue

        if auth_enabled:
            scheme = origin.split(":", 1)[0]
            transport_is_valid = (
                scheme == "http"
                and loopback
                and is_loopback_public_origin(origin)
                and not secure
            ) or (scheme == "https" and secure)
            if not transport_is_valid:
                errors.append(
                    f"service {service_name} violates deployment transport policy"
                )

        if expected_bind is None:
            expected_bind = bind
            expected_loopback = loopback
            expected_secure = secure
            expected_auth_enabled = auth_enabled
        elif bind != expected_bind:
            errors.append(
                f"service {service_name} environment HOST_BIND_ADDRESS does not match train-factory-api"
            )

    required_api_services = tuple(
        name
        for name in API_SERVICES
        if name in required_services
        and name not in invalid_required_services
        and isinstance(services.get(name), dict)
    )
    if required_api_services:
        for service_name in required_api_services:
            errors.extend(
                _validate_gpu_preflight_policy(services[service_name], service_name)
            )
        errors.extend(_validate_compose_secrets(compose, required_api_services))

    for service_name, service in services.items():
        if not isinstance(service, dict):
            continue
        ports = service.get("ports", [])
        if not isinstance(ports, list):
            if service_name in invalid_required_services:
                continue
            errors.append(f"service {service_name} has invalid published ports")
            continue
        for port in ports:
            if not isinstance(port, dict):
                errors.append(
                    f"service {service_name} published port definition is invalid"
                )
                continue
            target = port.get("target")
            if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
                errors.append(
                    f"service {service_name} published port has invalid target"
                )
                continue
            published = port.get("published")
            if not isinstance(published, str) or not published:
                errors.append(
                    f"service {service_name} published port has invalid published"
                )
                continue
            host_ip = port.get("host_ip")
            if not isinstance(host_ip, str) or not host_ip:
                errors.append(
                    f"service {service_name} published port has invalid host_ip"
                )
                continue
            try:
                published_address = _deployment_bind_address(host_ip)
            except ValueError:
                errors.append(
                    f"service {service_name} published port has invalid host_ip"
                )
                continue
            if port.get("protocol") != "tcp":
                errors.append(
                    f"service {service_name} published port has invalid protocol"
                )
                continue
            if expected_bind is not None and port["host_ip"] != expected_bind:
                local_web_override = (
                    service_name == "train-factory-web"
                    and (
                        published_address.is_private
                        or published_address.is_link_local
                        or published_address.is_unspecified
                    )
                    and not published_address.is_loopback
                )
                if not local_web_override:
                    message = (
                        "published port host_ip is unsafe for local Web access"
                        if service_name == "train-factory-web"
                        and published_address.is_global
                        else "published port host_ip does not match HOST_BIND_ADDRESS"
                    )
                    errors.append(f"service {service_name} {message}")
            elif (
                expected_auth_enabled
                and expected_secure is False
                and expected_loopback is False
            ):
                errors.append(
                    f"service {service_name} published port host_ip is unsafe for insecure transport"
                )

    return errors


def _load_json(path: str) -> object:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "compose validator arguments are invalid\n")


def main(argv: list[str] | None = None) -> int:
    parser = _PrivateArgumentParser()
    parser.add_argument("--compose-json")
    parser.add_argument("--profile")
    parser.add_argument("--images-lock")
    parser.add_argument("--require-digests", action="store_true")
    args = parser.parse_args(argv)

    if args.require_digests:
        if args.images_lock is None:
            print("images lock is invalid", file=sys.stderr)
            return 2
        try:
            load_images_lock(Path(args.images_lock))
        except ValueError:
            print("images lock is invalid", file=sys.stderr)
            return 2
    elif args.images_lock is not None:
        print("compose validator arguments are invalid", file=sys.stderr)
        return 2

    has_compose = args.compose_json is not None or args.profile is not None
    if not has_compose:
        if args.require_digests:
            return 0
        print("compose validator arguments are invalid", file=sys.stderr)
        return 2
    if args.compose_json is None or args.profile is None:
        print("compose validator arguments are invalid", file=sys.stderr)
        return 2

    if args.profile not in {"minimal", "all"}:
        print("compose validation profile is invalid", file=sys.stderr)
        return 2

    try:
        compose = _load_json(args.compose_json)
    except (OSError, json.JSONDecodeError):
        print("compose JSON is invalid", file=sys.stderr)
        return 2

    errors = validate_compose_config(compose, profile=args.profile)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
