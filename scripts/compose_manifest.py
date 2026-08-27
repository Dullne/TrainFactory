"""Freeze and verify ordered Compose inputs without exposing their values."""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("compose manifest bootstrap is unavailable")


def _bootstrap_realpath(value: str) -> str:
    return _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(value))


_script_path = _bootstrap_realpath(__file__)
_script_dir = _bootstrap_os.path.dirname(_script_path)
_root_entry = _bootstrap_os.path.dirname(_script_dir)
if not __package__:
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
    sys.path[:] = [_root_entry, *_trusted_sys_path]

import argparse  # noqa: E402
import hashlib  # noqa: E402
from ipaddress import ip_address  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from subprocess import CompletedProcess  # noqa: E402
from urllib.parse import unquote, urlsplit  # noqa: E402

import yaml  # noqa: E402
from yaml.nodes import MappingNode, ScalarNode, SequenceNode  # noqa: E402

if __package__:
    from train_factory.config.secret_files import resolve_secret
    from scripts.materialize_compose_secrets import _harden_path, _verify_hardened_path
else:
    from train_factory.config.secret_files import resolve_secret
    from scripts.materialize_compose_secrets import (  # type: ignore[no-redef]
        _harden_path,
        _verify_hardened_path,
    )


ROOT_DIR = Path(_root_entry)
RUNTIME_DIR = ROOT_DIR / ".runtime"
_RANDOM_PROJECT = re.compile(r"trainfactory-(?:verify|ci|rollback-verify)-[0-9a-f]{32}")
_SAFE_PROJECT = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
_ROLLBACK_VARIANTS = frozenset({"raw-old", "release-api-old-web", "compat-api-old-web"})


def _validate_rollback_variant(mode: object, value: object) -> None:
    if mode == "rollback-verify":
        if type(value) is not str or value not in _ROLLBACK_VARIANTS:
            raise ManifestError("compose manifest rollback variant is invalid")
    elif value is not None:
        raise ManifestError("compose manifest rollback variant is invalid")


_MAX_INPUT_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_MANIFEST_NAME = re.compile(
    r"(?:[a-z0-9-]+-|\.web-promotion-[0-9a-f]{40}-)?manifest\.json",
    re.ASCII,
)
_CI_SELECTION_NAME = re.compile(r"ci-release-[0-9a-f]{32}\.env", re.ASCII)
_SECRET_PATHS = {
    "MYSQL_ROOT_PASSWORD_SECRET_PATH": "mysql_root_password",
    "MYSQL_APP_PASSWORD_SECRET_PATH": "mysql_app_password",
    "MYSQL_URL_SECRET_PATH": "mysql_url",
    "JWT_SECRET_KEY_SECRET_PATH": "jwt_secret_key",
    "DEFAULT_ADMIN_PASSWORD_SECRET_PATH": "default_admin_password",
}
_SECRET_ENV_CONSTANTS = {
    "MYSQL_APP_USER": "trainfactory_app",
    "DEFAULT_ADMIN_USERNAME": "admin",
    "API_PORT": "18000",
    "WEB_PORT": "3000",
}
_REFERENCED_SECRET_ROLES = tuple(
    [f"secret:{filename}" for filename in _SECRET_PATHS.values()]
    + ["secret:admin_username", "secret-state"]
)
_PRODUCTION_REFERENCED_ROLES = tuple(
    f"secret:{filename}" for filename in _SECRET_PATHS.values()
)
_IMAGE_LOCK_KEYS = {
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
}
_IMAGE_SELECTION_KEYS = {
    "API_IMAGE",
    "WEB_IMAGE",
    "RELEASE_REVISION",
    "API_IMAGE_ID",
    "WEB_IMAGE_ID",
}
_ROLLBACK_IMAGE_SELECTION_KEYS = {
    "API_IMAGE",
    "WEB_IMAGE",
    "API_REVISION",
    "WEB_REVISION",
    "API_IMAGE_ID",
    "WEB_IMAGE_ID",
}
_DIRECT_SECRET_KEYS = {
    "MYSQL_ROOT_PASSWORD",
    "MYSQL_APP_USER",
    "MYSQL_APP_PASSWORD",
    "MYSQL_PASSWORD",
    "MYSQL_URL",
    "JWT_SECRET_KEY",
    "DEFAULT_ADMIN_USERNAME",
    "DEFAULT_ADMIN_PASSWORD",
}
_ROLLBACK_VERIFY_TRANSPORT_KEYS = {
    "HOST_BIND_ADDRESS",
    "PUBLIC_BASE_URL",
    "AUTH_COOKIE_SECURE",
    "API_PORT",
    "WEB_PORT",
}
_COMPOSE_INTERPOLATION_KEYS = frozenset(
    """
    ALLOW_MODEL_REMOTE_CODE API_BASE_IMAGE API_CPU_LIMIT API_DEV_PORT API_HOST
    API_IMAGE API_MEM_LIMIT API_PIDS_LIMIT API_PORT API_SHM_SIZE API_WORKERS
    APP_TIMEZONE AUDIT_LOG_CLEANUP_INTERVAL_HOURS AUDIT_LOG_RETENTION_DAYS
    AUTH_COOKIE_SECURE AUTH_ENABLED BACKGROUND_TASK_MAX_ACTIVE_GLOBAL
    BACKGROUND_TASK_MAX_ACTIVE_PER_USER BUILD_VERSION COMPOSE_PROJECT_NAME
    DATASETS_DIR DATASET_DOWNLOAD_MAX_BYTES DB_MAX_OVERFLOW DB_POOL_SIZE
    DEFAULT_ADMIN_EMAIL DEFAULT_ADMIN_PASSWORD DEFAULT_ADMIN_PASSWORD_SECRET_PATH
    DEFAULT_ADMIN_USERNAME DISCOVER_MODELS_ALLOWED_PRIVATE_HOSTS
    DISCOVER_MODELS_ALLOW_PRIVATE_ALL DOCKER_DYNAMIC_IP_RANGE DOCKER_NETWORK_NAME
    DOCKER_NETWORK_SUBNET DOWNLOAD_MAX_CONCURRENT_GLOBAL
    DOWNLOAD_MAX_CONCURRENT_PER_USER DOWNLOAD_MAX_WORKERS
    DOWNLOAD_METADATA_TIMEOUT_SECONDS DOWNLOAD_STORAGE_MAX_BYTES_GLOBAL
    DOWNLOAD_STORAGE_MAX_BYTES_PER_USER E2E_NODE_MODULES_VOLUME ETCD_DATA_VOLUME
    ETCD_IMAGE HF_DATASETS_CACHE HF_ENDPOINT HF_HOME HOST_BIND_ADDRESS HOST_IP
    JWT_SECRET_KEY JWT_SECRET_KEY_SECRET_PATH LOG_LEVEL MILVUS_DATA_VOLUME
    MILVUS_GRPC_PORT MILVUS_IMAGE MILVUS_PORT MINIO_ACCESS_KEY MINIO_BUCKET
    MINIO_CONSOLE_PORT MINIO_DATA_VOLUME MINIO_IMAGE MINIO_SECRET_KEY
    MODELSCOPE_CACHE MODELS_DIR MODEL_DOWNLOAD_MAX_BYTES MYSQL_APP_PASSWORD
    MYSQL_APP_PASSWORD_SECRET_PATH MYSQL_APP_USER MYSQL_DATABASE MYSQL_DATA_VOLUME
    MYSQL_IMAGE MYSQL_PASSWORD MYSQL_PASSWORD_FILE MYSQL_ROOT_PASSWORD
    MYSQL_ROOT_PASSWORD_SECRET_PATH MYSQL_URL_SECRET_PATH MYSQL_USER NODE_IMAGE
    NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES OUTPUT_DIR PATH_MAPPINGS
    PLAYWRIGHT_IMAGE PORT_RANGE_END PORT_RANGE_START PUBLIC_BASE_URL
    QWEN3_RERANK_TRAINER_PATH RATE_LIMIT_DEFAULT RATE_LIMIT_DEV_TRUSTED_PROXIES
    RATE_LIMIT_ENABLED RATE_LIMIT_LOGIN RATE_LIMIT_REGISTER
    RATE_LIMIT_TRUSTED_PROXIES SELF_REGISTRATION_ENABLED SGLANG_IMAGE
    SGLANG_TEMPLATE_VOLUME
    SOURCE_REPOSITORY STORAGE_BACKEND SWANLAB_API_KEY SYNC_BOUNDARY_MAX_BYTES
    SYNC_BOUNDARY_MAX_IDS SYNC_GENERATION_MAX_INPUT_BYTES
    SYNC_HISTORICAL_MAX_BYTES SYNC_HISTORICAL_MAX_DOCS
    SYNC_MAX_FUTURE_SKEW_SECONDS SYNC_MAX_RECORD_BYTES
    SYNC_PENDING_MAX_BATCHES_PER_TASK SYNC_PENDING_MAX_RECORDS_PER_TASK
    SYNC_STORAGE_MAX_BYTES_GLOBAL SYNC_STORAGE_MAX_BYTES_PER_USER
    TRAINING_ALLOW_CPU_FALLBACK TRAINING_CACHE TRAIN_CACHE_VOLUME VCS_DATE
    VCS_REF VERSION VLLM_IMAGE WEB_DEV_PORT WEB_DEV_PROXY_IPV4 WEB_IMAGE
    WEB_NGINX_IMAGE WEB_NODE_BUILD_IMAGE WEB_NODE_MODULES_VOLUME WEB_PORT
    WEB_PROXY_IPV4 XINFERENCE_CACHE_VOLUME XINFERENCE_HOME_VOLUME
    XINFERENCE_CONTRACT_VOLUME XINFERENCE_IMAGE XINFERENCE_PATCH_VOLUME
    XINFERENCE_PORT
    """.split()
)


class ManifestError(RuntimeError):
    """A fixed-message failure safe for console output."""


def _is_reparse(path: Path, metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _contained(path: Path, parent: Path) -> bool:
    try:
        return os.path.normcase(os.path.commonpath((path, parent))) == os.path.normcase(
            os.fspath(parent)
        )
    except ValueError:
        return False


def _validate_ancestor_chain(path: Path, *, boundary: Path) -> None:
    """Reject lexical path traversal through a symlink, junction, or file."""
    lexical = Path(os.path.abspath(path))
    root = Path(os.path.abspath(boundary))
    if not _contained(lexical, root):
        raise ManifestError("compose manifest path is invalid")
    try:
        root_metadata = root.lstat()
        if not root.is_dir() or _is_reparse(root, root_metadata):
            raise ManifestError("compose manifest path is invalid")
        relative_parts = lexical.relative_to(root).parts
        if any(":" in part or part.endswith((" ", ".")) for part in relative_parts):
            raise ManifestError("compose manifest path is invalid")
        current = root
        for part in relative_parts[:-1]:
            current = current / part
            metadata = current.lstat()
            if not current.is_dir() or _is_reparse(current, metadata):
                raise ManifestError("compose manifest path is invalid")
    except ManifestError:
        raise
    except (OSError, ValueError):
        raise ManifestError("compose manifest path is invalid") from None


def _read_stable(path: Path, *, max_bytes: int) -> tuple[bytes, tuple[int, int]]:
    descriptor = -1
    try:
        lexical = Path(os.path.abspath(path))
        metadata = lexical.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse(lexical, metadata)
            or not 1 <= metadata.st_size <= max_bytes
        ):
            raise ManifestError("compose manifest input is invalid")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        payload = os.read(descriptor, metadata.st_size + 1)
        finished = os.fstat(descriptor)
        after = lexical.lstat()
        os.lseek(descriptor, 0, os.SEEK_SET)
        confirmation = os.read(descriptor, metadata.st_size + 1)
        confirmed = os.fstat(descriptor)
        final_path = lexical.lstat()
        identities = {
            (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            for item in (metadata, opened, finished, after, confirmed, final_path)
        }
        if (
            len(identities) != 1
            or len(payload) != metadata.st_size
            or confirmation != payload
        ):
            raise ManifestError("compose manifest input is invalid")
        return payload, (metadata.st_dev, metadata.st_ino)
    except ManifestError:
        raise
    except OSError:
        raise ManifestError("compose manifest input is invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _record(
    path: Path,
    *,
    role: str,
    sensitive: bool,
) -> tuple[dict[str, object], tuple[int, int], bytes]:
    lexical = Path(os.path.abspath(path))
    payload, identity = _read_stable(lexical, max_bytes=_MAX_INPUT_BYTES)
    return (
        {
            "role": role,
            "path": os.fspath(lexical),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sensitive": sensitive,
        },
        identity,
        payload,
    )


def _validate_compose_source(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise ManifestError("compose manifest Compose source is invalid") from None
    try:
        documents = list(yaml.compose_all(text))
    except yaml.YAMLError:
        raise ManifestError("compose manifest Compose source is invalid")

    def mapping_entries(node: object) -> dict[str, object]:
        if not isinstance(node, MappingNode):
            return {}
        result: dict[str, object] = {}
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode) or not isinstance(
                key_node.value, str
            ):
                raise ManifestError("compose manifest Compose source is invalid")
            if key_node.value in result:
                raise ManifestError("compose manifest Compose source is invalid")
            result[key_node.value] = value_node
        return result

    def reject_merge_keys(node: object) -> None:
        if isinstance(node, MappingNode):
            for key_node, value_node in node.value:
                if isinstance(key_node, ScalarNode) and key_node.value == "<<":
                    raise ManifestError("compose manifest Compose source is invalid")
                reject_merge_keys(key_node)
                reject_merge_keys(value_node)
        elif isinstance(node, SequenceNode):
            for child in node.value:
                reject_merge_keys(child)

    for document in documents:
        reject_merge_keys(document)
        top = mapping_entries(document)
        if "include" in top:
            raise ManifestError("compose manifest Compose source is invalid")
        services = mapping_entries(top.get("services"))
        for definition in services.values():
            service = mapping_entries(definition)
            if {"env_file", "extends"} & set(service):
                raise ManifestError("compose manifest Compose source is invalid")
    variables: set[str] = set()
    index = 0
    while index < len(text):
        if text[index] != "$":
            index += 1
            continue
        run_end = index
        while run_end < len(text) and text[run_end] == "$":
            run_end += 1
        if (run_end - index) % 2:
            variable_start = run_end
            if variable_start < len(text) and text[variable_start] == "{":
                variable_start += 1
            match = re.match(
                r"[A-Za-z_][A-Za-z0-9_]*",
                text[variable_start:],
            )
            if match:
                variables.add(match.group(0))
        index = run_end
    if not variables <= _COMPOSE_INTERPOLATION_KEYS:
        raise ManifestError("compose manifest Compose source is invalid")


def _compose_roles(mode: str, gpu_mode: str, secret_mode: str) -> tuple[str, ...]:
    if gpu_mode not in {"raw", "compat", "cpu"}:
        raise ManifestError("compose manifest mode is invalid")
    if secret_mode not in {"files", "direct"}:
        raise ManifestError("compose manifest mode is invalid")
    if mode == "production":
        if gpu_mode == "cpu":
            raise ManifestError("compose manifest mode is invalid")
        roles = ["base", "release"]
        if gpu_mode == "compat":
            roles.append("gpu-compat")
        if secret_mode == "files":
            roles.append("secrets")
        return tuple(roles)
    if mode == "verify":
        if secret_mode != "files":
            raise ManifestError("compose manifest mode is invalid")
        roles = ["base", "release", "verify"]
        if gpu_mode == "compat":
            roles.append("gpu-compat")
        elif gpu_mode == "cpu":
            roles.append("cpu")
        if secret_mode == "files":
            roles.append("secrets")
        return tuple(roles)
    if mode == "ci":
        if gpu_mode != "cpu" or secret_mode != "files":
            raise ManifestError("compose manifest mode is invalid")
        return ("base", "release", "verify", "cpu", "secrets")
    if mode == "rollback-verify":
        if gpu_mode != "cpu" or secret_mode != "direct":
            raise ManifestError("compose manifest mode is invalid")
        return ("base", "release", "verify", "cpu")
    if mode in {"rollback-pre", "rollback-post-migration"}:
        if secret_mode != "direct" or gpu_mode == "cpu":
            raise ManifestError("compose manifest mode is invalid")
        roles = ["base", "release"]
        if gpu_mode == "compat":
            roles.append("gpu-compat")
        return tuple(roles)
    if mode == "rollback-web-only":
        if gpu_mode == "cpu":
            raise ManifestError("compose manifest mode is invalid")
        roles = ["base", "release"]
        if gpu_mode == "compat":
            roles.append("gpu-compat")
        if secret_mode == "files":
            roles.append("secrets")
        return tuple(roles)
    raise ManifestError("compose manifest mode is invalid")


_COMPOSE_NAMES = {
    "base": "docker-compose.yml",
    "release": "docker-compose.release.yml",
    "verify": "docker-compose.verify.yml",
    "gpu-compat": "docker-compose.gpu-compat.yml",
    "cpu": "docker-compose.cpu.yml",
    "secrets": "docker-compose.secrets.yml",
}


def _env_roles(mode: str, secret_mode: str) -> tuple[tuple[str, bool], ...]:
    if mode in {"verify", "ci"}:
        return (
            ("secret-paths", True),
            ("images-lock", False),
            ("image-selection", False),
        )
    if mode == "rollback-verify":
        return (
            ("base-environment", True),
            ("images-lock", False),
            ("rollback-direct", True),
            ("image-selection", False),
        )
    if mode in {"rollback-pre", "rollback-post-migration"}:
        selection_role = "rollback-pre" if mode == "rollback-pre" else "rollback-post"
        return (
            ("base-environment", True),
            ("images-lock", False),
            ("rollback-direct", True),
            (selection_role, False),
        )
    if mode in {"production", "rollback-web-only"}:
        credential_role = (
            "production-files" if secret_mode == "files" else "production-direct"
        )
        image_role = "image-selection"
        return (
            ("base-environment", True),
            (credential_role, True),
            ("images-lock", False),
            (image_role, False),
        )
    raise ManifestError("compose manifest mode is invalid")


def _validate_project(mode: str, project: str) -> None:
    if _SAFE_PROJECT.fullmatch(project) is None:
        raise ManifestError("compose manifest project is invalid")
    if mode in {"verify", "ci", "rollback-verify"}:
        if _RANDOM_PROJECT.fullmatch(project) is None:
            raise ManifestError("compose manifest project is invalid")


def _ci_run_id(project: str) -> str:
    match = re.fullmatch(r"trainfactory-ci-([0-9a-f]{32})", project, re.ASCII)
    if match is None:
        raise ManifestError("compose manifest project is invalid")
    return match.group(1)


def _matches_exact_path(candidate: Path, expected: Path, *, root: Path) -> bool:
    candidate_text = os.fspath(candidate)
    expected_text = os.fspath(expected)
    if os.path.isabs(candidate_text):
        return os.path.normcase(candidate_text) == os.path.normcase(expected_text)
    relative = os.path.relpath(expected_text, os.fspath(root))
    return os.path.normcase(candidate_text) == os.path.normcase(relative)


def _validate_ci_manifest_path(
    path: Path,
    *,
    mode: str,
    project: str,
    root: Path,
) -> None:
    if mode != "ci":
        return
    expected = Path(
        os.path.abspath(
            root / ".runtime" / f"ci-compose-{_ci_run_id(project)}-manifest.json"
        )
    )
    if not _matches_exact_path(path, expected, root=root):
        raise ManifestError("compose manifest output is invalid")


def _validate_input_path(path: Path, *, role: str, root: Path) -> None:
    lexical = Path(os.path.abspath(path))
    docker = Path(os.path.abspath(root / "docker"))
    runtime = Path(os.path.abspath(root / ".runtime"))
    if role in _COMPOSE_NAMES:
        expected = docker / _COMPOSE_NAMES[role]
        valid = os.path.normcase(lexical) == os.path.normcase(expected)
    elif role == "images-lock":
        valid = os.path.normcase(lexical) == os.path.normcase(
            docker / "images.lock.env"
        )
    elif role == "base-environment":
        valid = os.path.normcase(lexical) == os.path.normcase(root / ".env")
    else:
        valid = _contained(lexical, runtime) and lexical != runtime
    if not valid:
        raise ManifestError("compose manifest input is invalid")
    boundary = (
        root
        if role in _COMPOSE_NAMES or role in {"images-lock", "base-environment"}
        else runtime
    )
    try:
        _validate_ancestor_chain(lexical, boundary=boundary)
    except ManifestError:
        raise ManifestError("compose manifest input is invalid") from None


def _decode_single_quoted_path(value: str) -> str:
    if len(value) < 2 or value[0] != "'" or value[-1] != "'":
        raise ManifestError("compose manifest environment is invalid")
    body = value[1:-1]
    decoded: list[str] = []
    index = 0
    while index < len(body):
        if body[index] == "\\":
            run_end = index
            while run_end < len(body) and body[run_end] == "\\":
                run_end += 1
            slash_count = run_end - index
            if run_end == len(body):
                if slash_count % 2:
                    raise ManifestError("compose manifest environment is invalid")
                decoded.append("\\" * slash_count)
                index = run_end
            elif body[run_end] == "'":
                if slash_count % 2 == 0:
                    raise ManifestError("compose manifest environment is invalid")
                decoded.append("\\" * (slash_count - 1))
                decoded.append("'")
                index = run_end + 1
            else:
                decoded.append("\\" * slash_count)
                index = run_end
        elif body[index] == "'":
            raise ManifestError("compose manifest environment is invalid")
        else:
            decoded.append(body[index])
            index += 1
    result = "".join(decoded)
    if not result or any(character in result for character in ("\x00", "\r", "\n")):
        raise ManifestError("compose manifest environment is invalid")
    return result


def _parse_closed_environment(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ManifestError("compose manifest environment is invalid") from None
    result: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            raise ManifestError("compose manifest environment is invalid")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key, re.ASCII) or key in result:
            raise ManifestError("compose manifest environment is invalid")
        result[key] = value
    return result


def _parse_environment_file(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ManifestError("compose manifest environment is invalid") from None
    result: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            raise ManifestError("compose manifest environment is invalid")
        key, value = line.split("=", 1)
        if (
            not re.fullmatch(r"[A-Z][A-Z0-9_]*", key, re.ASCII)
            or key in result
            or any(character in value for character in ("\x00", "\r", "\n"))
        ):
            raise ManifestError("compose manifest environment is invalid")
        result[key] = value
    return result


def _validate_environment_schema(
    payload: bytes,
    *,
    role: str,
    project: str,
    secret_mode: str,
    mode: str,
) -> None:
    values = _parse_environment_file(payload)
    compose_keys = {key for key in values if key.startswith("COMPOSE_")}
    if role == "base-environment":
        if compose_keys - {"COMPOSE_PROJECT_NAME"}:
            raise ManifestError("compose manifest environment is invalid")
        if any(
            value != value.strip()
            or value.startswith(("'", '"'))
            or "$" in value
            or "\\" in value
            or re.search(r"\s#", value) is not None
            for value in values.values()
        ):
            raise ManifestError("compose manifest environment is invalid")
        bind_address = values.get("HOST_BIND_ADDRESS")
        if bind_address is not None:
            try:
                parsed_bind = ip_address(bind_address)
            except ValueError:
                raise ManifestError("compose manifest environment is invalid") from None
            if getattr(parsed_bind, "ipv4_mapped", None) is not None:
                raise ManifestError("compose manifest environment is invalid")
        if mode == "rollback-verify":
            base_project = values.get("COMPOSE_PROJECT_NAME", "")
            if (
                _SAFE_PROJECT.fullmatch(base_project) is None
                or _RANDOM_PROJECT.fullmatch(base_project) is not None
            ):
                raise ManifestError("compose manifest environment is invalid")
        elif values.get("COMPOSE_PROJECT_NAME") != project:
            raise ManifestError("compose manifest environment is invalid")
        if secret_mode == "files" and any(
            values.get(key, "")
            for key in {
                "MYSQL_ROOT_PASSWORD",
                "MYSQL_APP_PASSWORD",
                "MYSQL_PASSWORD",
                "MYSQL_URL",
                "JWT_SECRET_KEY",
                "DEFAULT_ADMIN_PASSWORD",
            }
        ):
            raise ManifestError("compose manifest environment is invalid")
        return
    if compose_keys:
        raise ManifestError("compose manifest environment is invalid")
    if role == "images-lock":
        if set(values) != _IMAGE_LOCK_KEYS or any(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:/+-]*@sha256:[0-9a-f]{64}",
                value,
                re.ASCII,
            )
            is None
            or ".." in value
            or "//" in value
            for value in values.values()
        ):
            raise ManifestError("compose manifest environment is invalid")
    elif role in {"image-selection", "rollback-pre", "rollback-post"}:
        observed_keys = set(values)
        if role == "image-selection" and mode == "production":
            if observed_keys not in (
                _IMAGE_SELECTION_KEYS,
                _ROLLBACK_IMAGE_SELECTION_KEYS,
            ):
                raise ManifestError("compose manifest environment is invalid")
            rollback_selection = observed_keys == _ROLLBACK_IMAGE_SELECTION_KEYS
        else:
            rollback_selection = mode.startswith("rollback")
            expected_keys = (
                _ROLLBACK_IMAGE_SELECTION_KEYS
                if rollback_selection
                else _IMAGE_SELECTION_KEYS
            )
            if observed_keys != expected_keys:
                raise ManifestError("compose manifest environment is invalid")
        if (
            any(
                re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}",
                    values[key],
                    re.ASCII,
                )
                is None
                or ".." in values[key]
                or "//" in values[key]
                for key in ("API_IMAGE", "WEB_IMAGE")
            )
            or any(
                re.fullmatch(r"[0-9a-f]{40}", values[key], re.ASCII) is None
                for key in (
                    ("API_REVISION", "WEB_REVISION")
                    if rollback_selection
                    else ("RELEASE_REVISION",)
                )
            )
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", values[key], re.ASCII) is None
                for key in ("API_IMAGE_ID", "WEB_IMAGE_ID")
            )
        ):
            raise ManifestError("compose manifest environment is invalid")
    elif role in {"production-direct", "rollback-direct"}:
        expected_keys = (
            _DIRECT_SECRET_KEYS | _ROLLBACK_VERIFY_TRANSPORT_KEYS
            if role == "rollback-direct" and mode == "rollback-verify"
            else _DIRECT_SECRET_KEYS
        )
        if set(values) != expected_keys:
            raise ManifestError("compose manifest environment is invalid")
        decoded = {
            key: _decode_single_quoted_path(value) for key, value in values.items()
        }
        _validate_direct_environment_values(decoded)
        if role == "rollback-direct" and mode == "rollback-verify":
            if {key: decoded[key] for key in _ROLLBACK_VERIFY_TRANSPORT_KEYS} != {
                "HOST_BIND_ADDRESS": "127.0.0.1",
                "PUBLIC_BASE_URL": "http://127.0.0.1:3000",
                "AUTH_COOKIE_SECURE": "false",
                "API_PORT": "18000",
                "WEB_PORT": "3000",
            }:
                raise ManifestError("compose manifest environment is invalid")
    elif role == "production-files":
        if set(values) != set(_SECRET_PATHS):
            raise ManifestError("compose manifest environment is invalid")


def _validate_direct_environment_values(values: dict[str, str]) -> None:
    admin_username = values["DEFAULT_ADMIN_USERNAME"]
    jwt_bytes = values["JWT_SECRET_KEY"].strip().encode("utf-8")
    if (
        re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", values["MYSQL_APP_USER"], re.ASCII)
        is None
        or values["MYSQL_APP_PASSWORD"] != values["MYSQL_PASSWORD"]
        or not 3 <= len(admin_username) <= 64
        or any(ord(character) < 32 for character in admin_username)
        or any(
            not 1 <= len(values[key]) <= 1024
            for key in ("MYSQL_ROOT_PASSWORD", "MYSQL_APP_PASSWORD")
        )
        or not 32 <= len(jwt_bytes) <= 1024
        or values["JWT_SECRET_KEY"].strip()
        in {
            "dev-only-secret-key-must-change-in-production",
            "dev-secret-change-me",
            "change-me-to-random-string",
        }
        or not 10 <= len(values["DEFAULT_ADMIN_PASSWORD"]) <= 128
    ):
        raise ManifestError("compose manifest environment is invalid")
    try:
        parsed = urlsplit(values["MYSQL_URL"])
        port = parsed.port
    except ValueError:
        raise ManifestError("compose manifest environment is invalid") from None
    if (
        parsed.scheme != "mysql+pymysql"
        or parsed.username != values["MYSQL_APP_USER"]
        or unquote(parsed.password or "") != values["MYSQL_APP_PASSWORD"]
        or parsed.hostname != "mysql"
        or port != 3306
        or parsed.path != "/train_factory"
        or parsed.query
        or parsed.fragment
    ):
        raise ManifestError("compose manifest environment is invalid")


def _validate_secret_path_environment(
    payload: bytes,
    *,
    bundle: Path,
    env_path: Path,
) -> tuple[tuple[str, Path], ...]:
    if not _verify_hardened_path(bundle) or not _verify_hardened_path(env_path):
        raise ManifestError("compose manifest environment is invalid")
    values = _parse_closed_environment(payload)
    if set(values) != set(_SECRET_PATHS) | set(_SECRET_ENV_CONSTANTS):
        raise ManifestError("compose manifest environment is invalid")
    if any(values[key] != value for key, value in _SECRET_ENV_CONSTANTS.items()):
        raise ManifestError("compose manifest environment is invalid")
    referenced: list[tuple[str, Path]] = []
    paths: dict[str, Path] = {}
    for key, filename in _SECRET_PATHS.items():
        candidate = Path(_decode_single_quoted_path(values[key]))
        expected = bundle / filename
        if not candidate.is_absolute() or os.path.normcase(
            os.fspath(candidate)
        ) != os.path.normcase(os.fspath(expected)):
            raise ManifestError("compose manifest environment is invalid")
        try:
            _validate_ancestor_chain(candidate, boundary=bundle)
            _read_stable(candidate, max_bytes=64 * 1024)
            if not _verify_hardened_path(candidate):
                raise ManifestError("compose manifest environment is invalid")
        except ManifestError:
            raise ManifestError("compose manifest environment is invalid") from None
        referenced.append((f"secret:{filename}", candidate))
        paths[filename] = candidate
    admin_username = bundle / "admin_username"
    state_path = bundle / "compose-secrets.state.json"
    try:
        for candidate in (admin_username, state_path):
            _validate_ancestor_chain(candidate, boundary=bundle)
            if not _verify_hardened_path(candidate):
                raise ManifestError("compose manifest environment is invalid")
        state_payload, _identity = _read_stable(state_path, max_bytes=64 * 1024)
        state = json.loads(state_payload)
    except (ManifestError, UnicodeError, json.JSONDecodeError, TypeError):
        raise ManifestError("compose manifest environment is invalid") from None
    expected_members = [
        *_SECRET_PATHS.values(),
        "admin_username",
        "compose-secrets.env",
        "compose-secrets.state.json",
    ]
    expected_state = {
        "version": 1,
        "scope": bundle.name.split("-", 1)[0],
        "run_id": bundle.name.rsplit("-", 1)[-1],
        "directory": str(bundle),
        "env_file": "compose-secrets.env",
        "state_file": "compose-secrets.state.json",
        "members": expected_members,
        "permissions_hardened": True,
        "cleanup_started": False,
    }
    if type(state) is not dict or type(state.get("directory")) is not str:
        raise ManifestError("compose manifest environment is invalid")
    if not os.path.isabs(state["directory"]):
        raise ManifestError("compose manifest environment is invalid")
    state_directory = os.path.normcase(state["directory"])
    expected_directory = os.path.normcase(expected_state["directory"])
    normalized_state = dict(state)
    normalized_state["directory"] = expected_state["directory"]
    if state_directory != expected_directory or normalized_state != expected_state:
        raise ManifestError("compose manifest environment is invalid")
    referenced.extend(
        (("secret:admin_username", admin_username), ("secret-state", state_path))
    )
    paths["admin_username"] = admin_username
    _validate_secret_values(
        paths,
        require_admin_username=True,
        generated=True,
    )
    return tuple(referenced)


def _private_text(path: Path) -> str:
    try:
        value = resolve_secret(
            direct_value=None,
            file_path=path,
            setting_name="deployment secret",
        )
    except ValueError:
        raise ManifestError("compose manifest secret source is invalid") from None
    if value is None:
        raise ManifestError("compose manifest secret source is invalid")
    return value


def _validate_secret_values(
    paths: dict[str, Path],
    *,
    require_admin_username: bool,
    generated: bool,
) -> None:
    values = {name: _private_text(path) for name, path in paths.items()}
    if generated:
        if (
            re.fullmatch(r"[0-9a-f]{64}", values["mysql_root_password"], re.ASCII)
            is None
            or re.fullmatch(r"[0-9a-f]{64}", values["mysql_app_password"], re.ASCII)
            is None
            or re.fullmatch(r"[0-9a-f]{64}", values["jwt_secret_key"], re.ASCII) is None
            or re.fullmatch(
                r"[A-Za-z0-9_-]{20,64}",
                values["default_admin_password"],
                re.ASCII,
            )
            is None
        ):
            raise ManifestError("compose manifest secret source is invalid")
    else:
        jwt_bytes = values["jwt_secret_key"].strip().encode("utf-8")
        if (
            any(
                not 1 <= len(values[key]) <= 1024
                for key in ("mysql_root_password", "mysql_app_password")
            )
            or not 32 <= len(jwt_bytes) <= 1024
            or values["jwt_secret_key"].strip()
            in {
                "dev-only-secret-key-must-change-in-production",
                "dev-secret-change-me",
                "change-me-to-random-string",
            }
            or not 10 <= len(values["default_admin_password"]) <= 128
        ):
            raise ManifestError("compose manifest secret source is invalid")
    if require_admin_username and values.get("admin_username") != "admin":
        raise ManifestError("compose manifest secret source is invalid")
    try:
        parsed = urlsplit(values["mysql_url"])
        port = parsed.port
    except ValueError:
        raise ManifestError("compose manifest secret source is invalid") from None
    if (
        parsed.scheme != "mysql+pymysql"
        or not parsed.username
        or (generated and parsed.username != "trainfactory_app")
        or unquote(parsed.password or "") != values["mysql_app_password"]
        or parsed.hostname != "mysql"
        or port != 3306
        or parsed.path != "/train_factory"
        or parsed.query
        or parsed.fragment
    ):
        raise ManifestError("compose manifest secret source is invalid")


def _validate_production_file_environment(
    payload: bytes,
    *,
    root: Path,
) -> tuple[tuple[str, Path], ...]:
    values = _parse_environment_file(payload)
    paths: dict[str, Path] = {}
    references: list[tuple[str, Path]] = []
    runtime = Path(os.path.abspath(root / ".runtime"))
    for key, filename in _SECRET_PATHS.items():
        candidate = Path(_decode_single_quoted_path(values[key]))
        if not candidate.is_absolute() or not _contained(candidate, runtime):
            raise ManifestError("compose manifest environment is invalid")
        try:
            _validate_ancestor_chain(candidate, boundary=runtime)
            _read_stable(candidate, max_bytes=64 * 1024)
            if not _verify_hardened_path(candidate):
                raise ManifestError("compose manifest environment is invalid")
        except ManifestError:
            raise ManifestError("compose manifest environment is invalid") from None
        paths[filename] = candidate
        references.append((f"secret:{filename}", candidate))
    _validate_secret_values(
        paths,
        require_admin_username=False,
        generated=False,
    )
    return tuple(references)


def _validate_role_source(
    path: Path,
    payload: bytes,
    *,
    role: str,
    mode: str,
    project: str,
    secret_mode: str,
    rollback_variant: str | None,
    root: Path,
) -> tuple[tuple[str, Path], ...]:
    lexical = Path(os.path.abspath(path))
    runtime = Path(os.path.abspath(root / ".runtime"))
    run_id = project.rsplit("-", 1)[-1]
    if role in {
        "base-environment",
        "images-lock",
        "image-selection",
        "production-files",
        "production-direct",
        "rollback-direct",
        "rollback-pre",
        "rollback-post",
    }:
        _validate_environment_schema(
            payload,
            role=role,
            project=project,
            secret_mode=secret_mode,
            mode=mode,
        )
    if role in {
        "production-files",
        "production-direct",
        "rollback-direct",
    } and not _verify_hardened_path(lexical):
        raise ManifestError("compose manifest environment is invalid")
    expected: Path | None = None
    if role == "secret-paths":
        scope = "ci" if mode == "ci" else "verify"
        bundle = runtime / f"{scope}-{run_id}"
        if lexical != bundle / "compose-secrets.env":
            raise ManifestError("compose manifest environment is invalid")
        return _validate_secret_path_environment(
            payload,
            bundle=bundle,
            env_path=lexical,
        )
    if role == "image-selection":
        names = {
            "production": "release.env",
            "verify": "release.env",
            "rollback-web-only": "rollback-web-only.env",
        }
        if mode == "ci":
            expected = runtime / f"ci-release-{_ci_run_id(project)}.env"
        elif mode == "rollback-verify":
            if rollback_variant not in _ROLLBACK_VARIANTS:
                raise ManifestError("compose manifest environment is invalid")
            expected = (
                runtime
                / f"rollback-verify-{run_id}"
                / f"rollback-verify-{rollback_variant}.env"
            )
        else:
            expected = runtime / names[mode]
    elif role == "production-files":
        expected = runtime / "production-files.env"
    elif role == "production-direct":
        expected = runtime / "production-direct.env"
    elif role == "rollback-direct":
        expected = runtime / "rollback-direct.env"
        if mode == "rollback-verify":
            expected = runtime / f"rollback-verify-{run_id}" / "rollback-direct.env"
    elif role == "rollback-pre":
        expected = runtime / "rollback-pre.env"
    elif role == "rollback-post":
        expected = runtime / "rollback-post.env"
    if (
        mode == "ci"
        and expected is not None
        and not _matches_exact_path(path, expected, root=root)
    ) or (
        mode != "ci"
        and expected is not None
        and os.path.normcase(os.fspath(lexical))
        != os.path.normcase(os.fspath(expected))
    ):
        raise ManifestError("compose manifest environment is invalid")
    if role == "production-files":
        return _validate_production_file_environment(payload, root=root)
    return ()


def _validate_cross_role_identity(
    payloads: dict[str, bytes],
    *,
    secret_mode: str,
) -> None:
    base_payload = payloads.get("base-environment")
    if base_payload is None:
        return
    base = _parse_environment_file(base_payload)
    base_user = base.get("MYSQL_APP_USER", "trainfactory_app")
    base_admin = base.get("DEFAULT_ADMIN_USERNAME", "admin")
    if (
        re.fullmatch(r"[A-Za-z0-9_]{1,64}", base_user, re.ASCII) is None
        or not 3 <= len(base_admin) <= 64
        or any(ord(character) < 32 for character in base_admin)
    ):
        raise ManifestError("compose manifest credential identity is invalid")
    if secret_mode == "direct":
        credential_payload = payloads.get("production-direct") or payloads.get(
            "rollback-direct"
        )
        if credential_payload is None:
            raise ManifestError("compose manifest credential identity is invalid")
        credential = {
            key: _decode_single_quoted_path(value)
            for key, value in _parse_environment_file(credential_payload).items()
        }
        if (
            credential.get("MYSQL_APP_USER") != base_user
            or credential.get("DEFAULT_ADMIN_USERNAME") != base_admin
        ):
            raise ManifestError("compose manifest credential identity is invalid")
    else:
        files_payload = payloads.get("production-files")
        if files_payload is None:
            raise ManifestError("compose manifest credential identity is invalid")
        paths = _parse_environment_file(files_payload)
        try:
            mysql_url_path = Path(
                _decode_single_quoted_path(paths["MYSQL_URL_SECRET_PATH"])
            )
            parsed = urlsplit(_private_text(mysql_url_path))
        except (KeyError, ValueError, ManifestError):
            raise ManifestError(
                "compose manifest credential identity is invalid"
            ) from None
        if parsed.username != base_user:
            raise ManifestError("compose manifest credential identity is invalid")


def _publish_manifest(
    output: Path,
    manifest: dict[str, object],
    *,
    root: Path,
    input_identities: set[tuple[int, int]],
    exclusive: bool = False,
) -> tuple[int, int]:
    runtime = Path(os.path.abspath(root / ".runtime"))
    target = Path(os.path.abspath(output))
    if (
        not _contained(target, runtime)
        or target == runtime
        or _MANIFEST_NAME.fullmatch(target.name) is None
    ):
        raise ManifestError("compose manifest output is invalid")
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    published_identity: tuple[int, int] | None = None
    target_intent = False
    publish_complete = False
    descriptor = -1
    try:
        _validate_ancestor_chain(target, boundary=runtime)
        target.parent.mkdir(parents=True, exist_ok=True)
        _validate_ancestor_chain(target, boundary=runtime)
        parent_metadata = target.parent.lstat()
        if not target.parent.is_dir() or _is_reparse(target.parent, parent_metadata):
            raise ManifestError("compose manifest output is invalid")
        if target.exists() or target.is_symlink():
            if exclusive:
                raise ManifestError("compose manifest output is invalid")
            metadata = target.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _is_reparse(target, metadata)
                or (metadata.st_dev, metadata.st_ino) in input_identities
            ):
                raise ManifestError("compose manifest output is invalid")
            try:
                existing = _load_manifest(target, root=root)
                _validate_manifest_structure(existing)
            except ManifestError:
                raise ManifestError("compose manifest output is invalid") from None
        temporary = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        opened_metadata = os.fstat(descriptor)
        temporary_identity = (opened_metadata.st_dev, opened_metadata.st_ino)
        payload = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        descriptor_metadata = os.fstat(descriptor)
        descriptor_identity = (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
            descriptor_metadata.st_size,
        )
        if descriptor_identity[:2] != temporary_identity:
            raise ManifestError("compose manifest publish failed")
        os.close(descriptor)
        descriptor = -1
        _harden_path(temporary)
        if not _verify_hardened_path(temporary):
            raise ManifestError("compose manifest publish failed")
        temporary_metadata = temporary.lstat()
        if (
            temporary_metadata.st_dev,
            temporary_metadata.st_ino,
            temporary_metadata.st_size,
        ) != descriptor_identity:
            raise ManifestError("compose manifest publish failed")
        published_identity = (temporary_metadata.st_dev, temporary_metadata.st_ino)
        if published_identity != temporary_identity:
            raise ManifestError("compose manifest publish failed")
        verified_temporary, verified_temporary_identity = _read_stable(
            temporary, max_bytes=_MAX_MANIFEST_BYTES
        )
        if (
            verified_temporary != payload
            or verified_temporary_identity != temporary_identity
        ):
            raise ManifestError("compose manifest publish failed")
        target_intent = True
        if exclusive:
            os.link(temporary, target)
            try:
                current_temporary = temporary.lstat()
                if (
                    current_temporary.st_dev,
                    current_temporary.st_ino,
                ) != temporary_identity:
                    raise ManifestError("compose manifest publish failed")
                temporary.unlink()
            except OSError:
                raise ManifestError("compose manifest publish failed") from None
        else:
            os.replace(temporary, target)
        temporary = None
        if not _verify_hardened_path(target):
            raise ManifestError("compose manifest publish failed")
        metadata = target.lstat()
        if (metadata.st_dev, metadata.st_ino) != published_identity:
            raise ManifestError("compose manifest publish failed")
        verified_payload, verified_identity = _read_stable(
            target, max_bytes=_MAX_MANIFEST_BYTES
        )
        if verified_payload != payload or verified_identity != published_identity:
            raise ManifestError("compose manifest publish failed")
        publish_complete = True
        return published_identity
    except ManifestError:
        raise
    except (OSError, ValueError):
        raise ManifestError("compose manifest publish failed") from None
    finally:
        cleanup_failed = False
        if descriptor >= 0:
            try:
                if temporary_identity is None:
                    recovered = os.fstat(descriptor)
                    temporary_identity = (recovered.st_dev, recovered.st_ino)
                os.close(descriptor)
            except OSError:
                cleanup_failed = True
        if temporary is not None and temporary_identity is not None:
            if not _unlink_owned_output(temporary, temporary_identity):
                cleanup_failed = True
        if (
            target_intent
            and not publish_complete
            and published_identity is not None
            and not _unlink_owned_output(target, published_identity)
        ):
            cleanup_failed = True
        if cleanup_failed:
            raise ManifestError("compose manifest publish failed") from None


def freeze_manifest(
    *,
    mode: str,
    project: str,
    env_files: Sequence[Path],
    gpu_mode: str,
    secret_mode: str,
    output: Path,
    rollback_variant: str | None = None,
    root: Path = ROOT_DIR,
    _exclusive_output: bool = False,
    _published_identity: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    root = Path(os.path.abspath(root))
    _validate_project(mode, project)
    _validate_rollback_variant(mode, rollback_variant)
    _validate_ci_manifest_path(output, mode=mode, project=project, root=root)
    if mode == "rollback-verify":
        run_id = project.rsplit("-", 1)[-1]
        expected_output = (
            root
            / ".runtime"
            / f"rollback-verify-{run_id}"
            / f"rollback-verify-{rollback_variant}-manifest.json"
        )
        if os.path.normcase(os.path.abspath(output)) != os.path.normcase(
            os.fspath(expected_output)
        ):
            raise ManifestError("compose manifest rollback variant is invalid")
    compose_roles = _compose_roles(mode, gpu_mode, secret_mode)
    env_roles = _env_roles(mode, secret_mode)
    if len(env_files) != len(env_roles):
        raise ManifestError("compose manifest environment order is invalid")
    identities: set[tuple[int, int]] = set()
    compose_records = []
    for role in compose_roles:
        path = root / "docker" / _COMPOSE_NAMES[role]
        _validate_input_path(path, role=role, root=root)
        record, identity, payload = _record(path, role=role, sensitive=False)
        _validate_compose_source(payload)
        _validate_role_source(
            path,
            payload,
            role=role,
            mode=mode,
            project=project,
            secret_mode=secret_mode,
            rollback_variant=rollback_variant,
            root=root,
        )
        if identity in identities:
            raise ManifestError("compose manifest input is invalid")
        identities.add(identity)
        compose_records.append(record)
    env_records = []
    env_payloads: dict[str, bytes] = {}
    bootstrap_path = root / "docker" / "init.sql"
    _validate_ancestor_chain(bootstrap_path, boundary=root)
    bootstrap_record, bootstrap_identity, _payload = _record(
        bootstrap_path,
        role="database-bootstrap",
        sensitive=False,
    )
    if bootstrap_identity in identities:
        raise ManifestError("compose manifest input is invalid")
    identities.add(bootstrap_identity)
    referenced_records = [bootstrap_record]
    for path, (role, sensitive) in zip(env_files, env_roles, strict=True):
        _validate_input_path(path, role=role, root=root)
        record, identity, payload = _record(path, role=role, sensitive=sensitive)
        references = _validate_role_source(
            path,
            payload,
            role=role,
            mode=mode,
            project=project,
            secret_mode=secret_mode,
            rollback_variant=rollback_variant,
            root=root,
        )
        if identity in identities:
            raise ManifestError("compose manifest input is invalid")
        identities.add(identity)
        env_records.append(record)
        env_payloads[role] = payload
        for referenced_role, referenced_path in references:
            referenced_record, referenced_identity, _payload = _record(
                referenced_path,
                role=referenced_role,
                sensitive=True,
            )
            if referenced_identity in identities:
                raise ManifestError("compose manifest input is invalid")
            identities.add(referenced_identity)
            referenced_records.append(referenced_record)
    _validate_cross_role_identity(env_payloads, secret_mode=secret_mode)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "project": project,
        "gpu_mode": gpu_mode,
        "secret_mode": secret_mode,
        "rollback_variant": rollback_variant,
        "compose_files": compose_records,
        "env_files": env_records,
        "referenced_files": referenced_records,
    }
    identity = _publish_manifest(
        output,
        manifest,
        root=root,
        input_identities=identities,
        exclusive=_exclusive_output or mode == "ci",
    )
    if _published_identity is not None:
        _published_identity.append(identity)
    return manifest


def _load_manifest(path: Path, *, root: Path) -> dict[str, object]:
    lexical = Path(os.path.abspath(path))
    runtime = Path(os.path.abspath(root / ".runtime"))
    if (
        not _contained(lexical, runtime)
        or lexical == runtime
        or _MANIFEST_NAME.fullmatch(lexical.name) is None
    ):
        raise ManifestError("compose manifest is invalid")
    try:
        _validate_ancestor_chain(lexical, boundary=runtime)
    except ManifestError:
        raise ManifestError("compose manifest is invalid") from None
    payload, _identity = _read_stable(lexical, max_bytes=_MAX_MANIFEST_BYTES)
    try:

        def reject_duplicate_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate")
                result[key] = value
            return result

        manifest = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ManifestError("compose manifest is invalid") from None
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "mode",
        "project",
        "gpu_mode",
        "secret_mode",
        "rollback_variant",
        "compose_files",
        "env_files",
        "referenced_files",
    }:
        raise ManifestError("compose manifest is invalid")
    return manifest


def _validate_manifest_structure(
    manifest: dict[str, object],
) -> tuple[tuple[str, ...], tuple[tuple[str, bool], ...]]:
    if manifest["schema_version"] != 1 or type(manifest["schema_version"]) is not int:
        raise ManifestError("compose manifest is invalid")
    if any(
        type(manifest[key]) is not str
        for key in ("mode", "project", "gpu_mode", "secret_mode")
    ):
        raise ManifestError("compose manifest is invalid")
    mode = manifest["mode"]
    project = manifest["project"]
    gpu_mode = manifest["gpu_mode"]
    secret_mode = manifest["secret_mode"]
    rollback_variant = manifest["rollback_variant"]
    try:
        _validate_rollback_variant(mode, rollback_variant)
    except ManifestError:
        raise ManifestError("compose manifest is invalid") from None
    _validate_project(mode, project)
    expected_compose = _compose_roles(mode, gpu_mode, secret_mode)
    expected_env = _env_roles(mode, secret_mode)
    compose_files = manifest["compose_files"]
    env_files = manifest["env_files"]
    referenced_files = manifest["referenced_files"]
    if (
        not isinstance(compose_files, list)
        or not isinstance(env_files, list)
        or not isinstance(referenced_files, list)
    ):
        raise ManifestError("compose manifest is invalid")
    if not all(
        isinstance(item, dict)
        for item in (*compose_files, *env_files, *referenced_files)
    ):
        raise ManifestError("compose manifest is invalid")
    if [item.get("role") for item in compose_files] != list(expected_compose):
        raise ManifestError("compose manifest is invalid")
    if [item.get("role") for item in env_files] != [role for role, _ in expected_env]:
        raise ManifestError("compose manifest is invalid")
    if [item.get("sensitive") for item in env_files] != [
        sensitive for _role, sensitive in expected_env
    ]:
        raise ManifestError("compose manifest is invalid")
    if any(item.get("sensitive") is not False for item in compose_files):
        raise ManifestError("compose manifest is invalid")
    if any(role == "secret-paths" for role, _sensitive in expected_env):
        secret_referenced_roles = _REFERENCED_SECRET_ROLES
    elif any(role == "production-files" for role, _sensitive in expected_env):
        secret_referenced_roles = _PRODUCTION_REFERENCED_ROLES
    else:
        secret_referenced_roles = ()
    expected_referenced = [
        ("database-bootstrap", False),
        *((role, True) for role in secret_referenced_roles),
    ]
    if [
        (item.get("role"), item.get("sensitive")) for item in referenced_files
    ] != expected_referenced:
        raise ManifestError("compose manifest is invalid")
    for item in (*compose_files, *env_files, *referenced_files):
        if set(item) != {"role", "path", "sha256", "sensitive"}:
            raise ManifestError("compose manifest is invalid")
        if (
            type(item["role"]) is not str
            or type(item["path"]) is not str
            or type(item["sha256"]) is not str
            or type(item["sensitive"]) is not bool
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"], re.ASCII) is None
        ):
            raise ManifestError("compose manifest is invalid")
    return expected_compose, expected_env


def _records_match(
    observed: list[dict[str, object]],
    expected: list[dict[str, object]],
) -> bool:
    if len(observed) != len(expected):
        return False
    for observed_item, expected_item in zip(observed, expected, strict=True):
        if any(
            observed_item[key] != expected_item[key]
            for key in ("role", "sha256", "sensitive")
        ) or os.path.normcase(str(observed_item["path"])) != os.path.normcase(
            str(expected_item["path"])
        ):
            return False
    return True


def verify_manifest_inputs(
    path: Path,
    *,
    root: Path = ROOT_DIR,
) -> dict[str, object]:
    try:
        manifest = _load_manifest(path, root=root)
        expected_compose, expected_env = _validate_manifest_structure(manifest)
        _validate_ci_manifest_path(
            path,
            mode=str(manifest["mode"]),
            project=str(manifest["project"]),
            root=root,
        )
        compose_files = manifest["compose_files"]
        env_files = manifest["env_files"]
        referenced_files = manifest["referenced_files"]
        bootstrap_path = root / "docker" / "init.sql"
        _validate_ancestor_chain(bootstrap_path, boundary=root)
        bootstrap_record, bootstrap_identity, _payload = _record(
            bootstrap_path,
            role="database-bootstrap",
            sensitive=False,
        )
        identities: set[tuple[int, int]] = {bootstrap_identity}
        recomputed_references: list[dict[str, object]] = [bootstrap_record]
        env_payloads: dict[str, bytes] = {}
        for item in (*compose_files, *env_files):
            role = item["role"]
            candidate = Path(item["path"])
            _validate_input_path(candidate, role=role, root=root)
            record, identity, payload = _record(
                candidate,
                role=role,
                sensitive=item["sensitive"],
            )
            if item in compose_files:
                _validate_compose_source(payload)
            references = _validate_role_source(
                candidate,
                payload,
                role=role,
                mode=manifest["mode"],
                project=manifest["project"],
                secret_mode=manifest["secret_mode"],
                rollback_variant=manifest["rollback_variant"],
                root=root,
            )
            if record != item or identity in identities:
                raise ManifestError("compose manifest input verification failed")
            identities.add(identity)
            if item in env_files:
                env_payloads[role] = payload
            for referenced_role, referenced_path in references:
                referenced_record, referenced_identity, _payload = _record(
                    referenced_path,
                    role=referenced_role,
                    sensitive=True,
                )
                if referenced_identity in identities:
                    raise ManifestError("compose manifest input verification failed")
                identities.add(referenced_identity)
                recomputed_references.append(referenced_record)
        _validate_cross_role_identity(
            env_payloads,
            secret_mode=manifest["secret_mode"],
        )
        if not _records_match(recomputed_references, referenced_files):
            raise ManifestError("compose manifest input verification failed")
        return manifest
    except ManifestError as exc:
        if str(exc) == "compose manifest input verification failed":
            raise
        raise ManifestError("compose manifest input verification failed") from None


def _selection_values(manifest: Mapping[str, object]) -> dict[str, str]:
    try:
        item = next(
            record
            for record in manifest["env_files"]  # type: ignore[union-attr]
            if record["role"] in {"image-selection", "rollback-pre", "rollback-post"}
        )
        payload, _identity = _read_stable(
            Path(item["path"]),
            max_bytes=_MAX_INPUT_BYTES,
        )
        return _parse_environment_file(payload)
    except (ManifestError, KeyError, StopIteration, TypeError):
        raise ManifestError("compose manifest image verification failed") from None


def _manifest_environment_values(
    manifest: Mapping[str, object],
    role: str,
) -> dict[str, str]:
    try:
        item = next(
            record
            for record in manifest["env_files"]  # type: ignore[union-attr]
            if record["role"] == role
        )
        payload, _identity = _read_stable(
            Path(item["path"]),
            max_bytes=_MAX_INPUT_BYTES,
        )
        return _parse_environment_file(payload)
    except (ManifestError, KeyError, StopIteration, TypeError):
        raise ManifestError("compose manifest image verification failed") from None


def verify_manifest_inputs_only(
    path: Path,
    *,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Verify frozen inputs, resolved image refs, immutable IDs, and OCI labels."""
    try:
        manifest = verify_manifest_inputs(path, root=root)
        selection = _selection_values(manifest)
        images_lock = _manifest_environment_values(manifest, "images-lock")
        if "RELEASE_REVISION" in selection:
            revisions = {
                "train-factory-api": selection["RELEASE_REVISION"],
                "train-factory-web": selection["RELEASE_REVISION"],
            }
        else:
            revisions = {
                "train-factory-api": selection["API_REVISION"],
                "train-factory-web": selection["WEB_REVISION"],
            }
        refs = {
            "train-factory-api": selection["API_IMAGE"],
            "train-factory-web": selection["WEB_IMAGE"],
        }
        image_ids = {
            "train-factory-api": selection["API_IMAGE_ID"],
            "train-factory-web": selection["WEB_IMAGE_ID"],
        }
        from scripts import compose_release

        environment = compose_release._clean_environment(base_environment)
        compose = compose_release._load_resolved_compose(
            manifest,
            root=root,
            run=run,
            environment=environment,
        )
        compose_release._validate_resolved_config(manifest, compose, root=root)
        services = compose.get("services")
        if not isinstance(services, dict):
            raise ManifestError("compose manifest image verification failed")
        if manifest["mode"] in {"verify", "ci", "rollback-verify"}:
            if set(services) != {"mysql", "train-factory-api", "train-factory-web"}:
                raise ManifestError("compose manifest image verification failed")
        mysql = services.get("mysql")
        if (
            not isinstance(mysql, dict)
            or mysql.get("image") != images_lock["MYSQL_IMAGE"]
            or mysql.get("build") not in {None, False}
        ):
            raise ManifestError("compose manifest image verification failed")
        for service, image_ref in refs.items():
            definition = services.get(service)
            if (
                not isinstance(definition, dict)
                or definition.get("image") != image_ref
                or definition.get("build") not in {None, False}
            ):
                raise ManifestError("compose manifest image verification failed")
            inspected = compose_release._run_private(
                run,
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}|{{.Os}}|{{.Architecture}}|"
                    '{{ index .Config.Labels "org.opencontainers.image.version" }}|'
                    '{{ index .Config.Labels "org.opencontainers.image.revision" }}|'
                    '{{ index .Config.Labels "org.opencontainers.image.created" }}|'
                    '{{ index .Config.Labels "org.opencontainers.image.source" }}|'
                    '{{ index .Config.Labels "io.train-factory.migration-compat" }}',
                    image_ref,
                ],
                environment,
            )
            parts = inspected.stdout.strip().split("|")
            if (
                len(parts) != 8
                or parts[0] != image_ids[service]
                or parts[1] != "linux"
                or parts[2] != "amd64"
                or not parts[3]
                or parts[4] != revisions[service]
                or not parts[5]
                or not parts[6].startswith("https://")
            ):
                raise ManifestError("compose manifest image verification failed")
            expected_compat = (
                "053_validate_lifecycle_schema"
                if (
                    manifest["mode"] == "rollback-post-migration"
                    or (
                        manifest["mode"] == "rollback-verify"
                        and manifest["rollback_variant"] == "compat-api-old-web"
                    )
                )
                and service == "train-factory-api"
                else ""
            )
            observed_compat = "" if parts[7] == "<no value>" else parts[7]
            if observed_compat != expected_compat:
                raise ManifestError("compose manifest image verification failed")
        if verify_manifest_inputs(path, root=root) != manifest:
            raise ManifestError("compose manifest image verification failed")
        return manifest
    except ManifestError:
        raise
    except Exception:
        raise ManifestError("compose manifest image verification failed") from None


def verify_running_container(
    path: Path,
    *,
    selection_path: Path,
    service: str,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> None:
    """Verify exactly one running service without reading container environment."""
    try:
        if service not in {"mysql", "train-factory-api", "train-factory-web"}:
            raise ManifestError("compose running verification failed")
        manifest = verify_manifest_inputs_only(
            path,
            root=root,
            base_environment=base_environment,
            run=run,
        )
        selection_record = next(
            item
            for item in manifest["env_files"]
            if item["role"] in {"image-selection", "rollback-pre", "rollback-post"}
        )
        if os.path.normcase(os.path.abspath(selection_path)) != os.path.normcase(
            str(selection_record["path"])
        ):
            raise ManifestError("compose running verification failed")
        selection = _selection_values(manifest)
        from scripts import compose_release

        environment = compose_release._clean_environment(base_environment)
        if service == "mysql":
            images_lock = _manifest_environment_values(manifest, "images-lock")
            expected_image = compose_release._run_private(
                run,
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    images_lock["MYSQL_IMAGE"],
                ],
                environment,
            ).stdout.strip()
            if re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image, re.ASCII) is None:
                raise ManifestError("compose running verification failed")
            expected_id = expected_image
        else:
            expected_id = selection[
                "API_IMAGE_ID" if service == "train-factory-api" else "WEB_IMAGE_ID"
            ]
        listed = compose_release._run_private(
            run,
            [
                "docker",
                "ps",
                "--no-trunc",
                "--filter",
                f"label=com.docker.compose.project={manifest['project']}",
                "--filter",
                f"label=com.docker.compose.service={service}",
                "--format",
                "{{.ID}}",
            ],
            environment,
        )
        identifiers = [
            line.strip() for line in listed.stdout.splitlines() if line.strip()
        ]
        if (
            len(identifiers) != 1
            or re.fullmatch(r"[0-9a-f]{64}", identifiers[0], re.ASCII) is None
        ):
            raise ManifestError("compose running verification failed")
        container_id = identifiers[0]
        inspected = compose_release._run_private(
            run,
            [
                "docker",
                "inspect",
                "--format",
                "{{.Id}}|{{.Image}}|{{.State.Running}}|"
                '{{ index .Config.Labels "com.docker.compose.project" }}|'
                '{{ index .Config.Labels "com.docker.compose.service" }}',
                container_id,
            ],
            environment,
        )
        parts = inspected.stdout.strip().split("|")
        if parts != [
            container_id,
            expected_id,
            "true",
            str(manifest["project"]),
            service,
        ]:
            raise ManifestError("compose running verification failed")
    except ManifestError:
        raise
    except Exception:
        raise ManifestError("compose running verification failed") from None


def _read_selection_source(path: Path, *, root: Path) -> dict[str, str]:
    runtime = Path(os.path.abspath(root / ".runtime"))
    lexical = Path(os.path.abspath(path))
    try:
        if not _contained(lexical, runtime) or lexical == runtime:
            raise ManifestError("compose image selection is invalid")
        _validate_ancestor_chain(lexical, boundary=runtime)
        payload, _identity = _read_stable(lexical, max_bytes=64 * 1024)
        values = _parse_environment_file(payload)
        if set(values) == _IMAGE_SELECTION_KEYS:
            mode = "production"
        elif set(values) == _ROLLBACK_IMAGE_SELECTION_KEYS:
            mode = "rollback-web-only"
        else:
            raise ManifestError("compose image selection is invalid")
        _validate_environment_schema(
            payload,
            role="image-selection",
            project="trainfactory",
            secret_mode="files",
            mode=mode,
        )
        return values
    except ManifestError:
        raise
    except Exception:
        raise ManifestError("compose image selection is invalid") from None


def _inspect_selected_image(
    values: Mapping[str, str],
    *,
    prefix: str,
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
    expected_compat: str = "",
) -> None:
    from scripts import compose_release

    revision_key = (
        f"{prefix}_REVISION" if f"{prefix}_REVISION" in values else "RELEASE_REVISION"
    )
    completed = compose_release._run_private(
        run,
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}|{{.Os}}|{{.Architecture}}|"
            '{{ index .Config.Labels "org.opencontainers.image.version" }}|'
            '{{ index .Config.Labels "org.opencontainers.image.revision" }}|'
            '{{ index .Config.Labels "org.opencontainers.image.created" }}|'
            '{{ index .Config.Labels "org.opencontainers.image.source" }}|'
            '{{ index .Config.Labels "io.train-factory.migration-compat" }}',
            values[f"{prefix}_IMAGE"],
        ],
        environment,
    )
    parts = completed.stdout.strip().split("|")
    observed_compat = (
        ""
        if len(parts) == 8 and parts[7] == "<no value>"
        else parts[7]
        if len(parts) == 8
        else None
    )
    if (
        len(parts) != 8
        or parts[0] != values[f"{prefix}_IMAGE_ID"]
        or parts[1:3] != ["linux", "amd64"]
        or not parts[3]
        or parts[4] != values[revision_key]
        or not parts[5]
        or not parts[6].startswith("https://")
        or observed_compat != expected_compat
    ):
        raise ManifestError("compose image selection is invalid")


def _publish_selection_environment(
    output: Path,
    values: Mapping[str, str],
    *,
    root: Path,
) -> tuple[int, int]:
    runtime = Path(os.path.abspath(root / ".runtime"))
    target = Path(os.path.abspath(output))
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    published_identity: tuple[int, int] | None = None
    link_intent = False
    publish_complete = False
    descriptor = -1
    try:
        if (
            not _contained(target, runtime)
            or target == runtime
            or (
                target.name
                not in {
                    "release.env",
                    "rollback-pre.env",
                    "rollback-post.env",
                    "rollback-web-only.env",
                    "rollback-verify.env",
                }
                and _CI_SELECTION_NAME.fullmatch(target.name) is None
            )
            or target.exists()
            or target.is_symlink()
        ):
            raise ManifestError("compose image selection output is invalid")
        if (
            not target.parent.exists()
            or not target.parent.is_dir()
            or _is_reparse(target.parent, target.parent.lstat())
        ):
            raise ManifestError("compose image selection output is invalid")
        _validate_ancestor_chain(target, boundary=runtime)
        _validate_ancestor_chain(target, boundary=runtime)
        payload = "".join(f"{key}={value}\n" for key, value in values.items()).encode()
        temporary = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        opened_metadata = os.fstat(descriptor)
        temporary_identity = (opened_metadata.st_dev, opened_metadata.st_ino)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        descriptor_metadata = os.fstat(descriptor)
        descriptor_identity = (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
            descriptor_metadata.st_size,
        )
        if descriptor_identity[:2] != temporary_identity:
            raise ManifestError("compose image selection publish failed")
        os.close(descriptor)
        descriptor = -1
        _harden_path(temporary)
        if not _verify_hardened_path(temporary):
            raise ManifestError("compose image selection publish failed")
        temporary_metadata = temporary.lstat()
        if (
            temporary_metadata.st_dev,
            temporary_metadata.st_ino,
            temporary_metadata.st_size,
        ) != descriptor_identity:
            raise ManifestError("compose image selection publish failed")
        published_identity = (temporary_metadata.st_dev, temporary_metadata.st_ino)
        if published_identity != temporary_identity:
            raise ManifestError("compose image selection publish failed")
        verified_temporary, verified_temporary_identity = _read_stable(
            temporary, max_bytes=64 * 1024
        )
        if (
            verified_temporary != payload
            or verified_temporary_identity != temporary_identity
        ):
            raise ManifestError("compose image selection publish failed")
        link_intent = True
        os.link(temporary, target)
        try:
            current_temporary = temporary.lstat()
            if (
                current_temporary.st_dev,
                current_temporary.st_ino,
            ) != temporary_identity:
                raise ManifestError("compose image selection publish failed")
            temporary.unlink()
        except OSError:
            raise ManifestError("compose image selection publish failed") from None
        temporary = None
        if not _verify_hardened_path(target):
            try:
                metadata = target.lstat()
                if (metadata.st_dev, metadata.st_ino) == published_identity:
                    target.unlink()
            except OSError:
                pass
            raise ManifestError("compose image selection publish failed")
        metadata = target.lstat()
        if (metadata.st_dev, metadata.st_ino) != published_identity:
            raise ManifestError("compose image selection publish failed")
        verified_payload, verified_identity = _read_stable(target, max_bytes=64 * 1024)
        if verified_payload != payload or verified_identity != published_identity:
            raise ManifestError("compose image selection publish failed")
        publish_complete = True
        return published_identity
    except ManifestError:
        raise
    except (OSError, ValueError):
        raise ManifestError("compose image selection publish failed") from None
    finally:
        cleanup_failed = False
        if descriptor >= 0:
            try:
                if temporary_identity is None:
                    recovered = os.fstat(descriptor)
                    temporary_identity = (recovered.st_dev, recovered.st_ino)
                os.close(descriptor)
            except OSError:
                cleanup_failed = True
        if temporary is not None and temporary_identity is not None:
            if not _unlink_owned_output(temporary, temporary_identity):
                cleanup_failed = True
        if (
            link_intent
            and not publish_complete
            and published_identity is not None
            and not _unlink_owned_output(target, published_identity)
        ):
            cleanup_failed = True
        if cleanup_failed:
            raise ManifestError("compose image selection publish failed") from None


def _unlink_owned_output(path: Path, identity: tuple[int, int]) -> bool:
    """Remove a published output only while it is still the inode we created."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if (metadata.st_dev, metadata.st_ino) != identity:
        return True
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _publish_verified_selection(
    output: Path,
    values: Mapping[str, str],
    *,
    root: Path,
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
    api_compat: str = "",
) -> tuple[int, int]:
    identity = _publish_selection_environment(output, values, root=root)
    try:
        if _read_selection_source(output, root=root) != values:
            raise ManifestError("compose image selection publish failed")
        _inspect_selected_image(
            values,
            prefix="API",
            run=run,
            environment=environment,
            expected_compat=api_compat,
        )
        _inspect_selected_image(
            values,
            prefix="WEB",
            run=run,
            environment=environment,
        )
        if _read_selection_source(output, root=root) != values:
            raise ManifestError("compose image selection publish failed")
        return identity
    except Exception:
        if not _unlink_owned_output(output, identity):
            raise ManifestError("compose image selection cleanup failed") from None
        raise ManifestError("compose image selection publish failed") from None


def build_forward_web_selection(
    *,
    production_manifest: Path,
    web_image: str,
    web_image_id: str,
    web_revision: str,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    """Build an old-API/new-Web selection without publishing runtime state."""
    try:
        manifest = verify_manifest_inputs_only(
            production_manifest,
            root=root,
            base_environment=base_environment,
            run=run,
        )
        if manifest.get("mode") != "production":
            raise ManifestError("compose forward Web selection failed")
        source = _selection_values(manifest)
        if set(source) not in (
            _IMAGE_SELECTION_KEYS,
            _ROLLBACK_IMAGE_SELECTION_KEYS,
        ):
            raise ManifestError("compose forward Web selection failed")
        api_revision = source.get("API_REVISION", source.get("RELEASE_REVISION"))
        if api_revision is None:
            raise ManifestError("compose forward Web selection failed")
        values = {
            "API_IMAGE": source["API_IMAGE"],
            "WEB_IMAGE": web_image,
            "API_REVISION": api_revision,
            "WEB_REVISION": web_revision,
            "API_IMAGE_ID": source["API_IMAGE_ID"],
            "WEB_IMAGE_ID": web_image_id,
        }
        payload = "".join(f"{key}={value}\n" for key, value in values.items()).encode()
        _validate_environment_schema(
            payload,
            role="image-selection",
            project=str(manifest["project"]),
            secret_mode=str(manifest["secret_mode"]),
            mode="production",
        )
        from scripts import compose_release

        environment = compose_release._clean_environment(base_environment)
        _inspect_selected_image(
            values,
            prefix="WEB",
            run=run,
            environment=environment,
        )
        verified_manifest = verify_manifest_inputs_only(
            production_manifest,
            root=root,
            base_environment=base_environment,
            run=run,
        )
        if (
            verified_manifest != manifest
            or _selection_values(verified_manifest) != source
        ):
            raise ManifestError("compose forward Web selection failed")
        return values
    except Exception:
        raise ManifestError("compose forward Web selection failed") from None


def mix_image_selection(
    *,
    api_source: Path,
    web_source: Path,
    output: Path,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    """Combine already-inspected API and Web selections without trusting tags."""
    runtime = Path(os.path.abspath(root / ".runtime"))
    if (
        os.path.normcase(os.path.abspath(api_source))
        != os.path.normcase(os.fspath(runtime / "release.env"))
        or os.path.normcase(os.path.abspath(web_source))
        != os.path.normcase(os.fspath(runtime / "rollback-pre.env"))
        or os.path.normcase(os.path.abspath(output))
        != os.path.normcase(os.fspath(runtime / "rollback-web-only.env"))
    ):
        raise ManifestError("compose image selection is invalid")
    api = _read_selection_source(api_source, root=root)
    web = _read_selection_source(web_source, root=root)
    if set(api) != _IMAGE_SELECTION_KEYS or set(web) != _ROLLBACK_IMAGE_SELECTION_KEYS:
        raise ManifestError("compose image selection is invalid")
    from scripts import compose_release

    environment = compose_release._clean_environment(base_environment)
    _inspect_selected_image(api, prefix="API", run=run, environment=environment)
    _inspect_selected_image(web, prefix="WEB", run=run, environment=environment)
    if (
        _read_selection_source(api_source, root=root) != api
        or _read_selection_source(web_source, root=root) != web
    ):
        raise ManifestError("compose image selection is invalid")
    values = {
        "API_IMAGE": api["API_IMAGE"],
        "WEB_IMAGE": web["WEB_IMAGE"],
        "API_REVISION": (
            api["API_REVISION"] if "API_REVISION" in api else api["RELEASE_REVISION"]
        ),
        "WEB_REVISION": (
            web["WEB_REVISION"] if "WEB_REVISION" in web else web["RELEASE_REVISION"]
        ),
        "API_IMAGE_ID": api["API_IMAGE_ID"],
        "WEB_IMAGE_ID": web["WEB_IMAGE_ID"],
    }
    _publish_verified_selection(
        output,
        values,
        root=root,
        run=run,
        environment=environment,
    )
    if (
        _read_selection_source(api_source, root=root) != api
        or _read_selection_source(web_source, root=root) != web
    ):
        raise ManifestError("compose image selection is invalid")
    return values


def create_image_selection(
    *,
    api_image: str,
    web_image: str,
    api_image_id: str,
    web_image_id: str,
    revision: str,
    output: Path,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    """Inspect two release images and publish a closed non-secret selection."""
    runtime = Path(os.path.abspath(root / ".runtime"))
    target = Path(os.path.abspath(output))
    if (
        target.parent != runtime
        or (
            target.name != "release.env"
            and _CI_SELECTION_NAME.fullmatch(target.name) is None
        )
        or (
            _CI_SELECTION_NAME.fullmatch(target.name) is not None
            and not _matches_exact_path(
                output,
                runtime / target.name,
                root=Path(os.path.abspath(root)),
            )
        )
    ):
        raise ManifestError("compose image selection is invalid")
    values = {
        "API_IMAGE": api_image,
        "WEB_IMAGE": web_image,
        "RELEASE_REVISION": revision,
        "API_IMAGE_ID": api_image_id,
        "WEB_IMAGE_ID": web_image_id,
    }
    payload = "".join(f"{key}={value}\n" for key, value in values.items()).encode()
    _validate_environment_schema(
        payload,
        role="image-selection",
        project="trainfactory",
        secret_mode="files",
        mode="production",
    )
    from scripts import compose_release

    environment = compose_release._clean_environment(base_environment)
    _inspect_selected_image(
        values,
        prefix="API",
        run=run,
        environment=environment,
    )
    _inspect_selected_image(
        values,
        prefix="WEB",
        run=run,
        environment=environment,
    )
    _publish_verified_selection(
        output,
        values,
        root=root,
        run=run,
        environment=environment,
    )
    return values


def _deployment_credentials_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, str]:
    base = _manifest_environment_values(manifest, "base-environment")
    if manifest["secret_mode"] == "direct":
        roles = {
            str(item["role"])
            for item in manifest["env_files"]  # type: ignore[union-attr]
        }
        role = (
            "production-direct" if "production-direct" in roles else "rollback-direct"
        )
        values = _manifest_environment_values(manifest, role)
        decoded = {
            key: _decode_single_quoted_path(value) for key, value in values.items()
        }
        if set(decoded) != _DIRECT_SECRET_KEYS:
            raise ManifestError("compose rollback capture failed")
        return decoded
    try:
        paths = {
            str(item["role"]).removeprefix("secret:"): Path(str(item["path"]))
            for item in manifest["referenced_files"]  # type: ignore[union-attr]
            if str(item["role"]).startswith("secret:")
        }
        root_password = _private_text(paths["mysql_root_password"])
        app_password = _private_text(paths["mysql_app_password"])
        mysql_url = _private_text(paths["mysql_url"])
        jwt_secret = _private_text(paths["jwt_secret_key"])
        admin_password = _private_text(paths["default_admin_password"])
        result = {
            "MYSQL_ROOT_PASSWORD": root_password,
            "MYSQL_APP_USER": base.get("MYSQL_APP_USER", "trainfactory_app"),
            "MYSQL_APP_PASSWORD": app_password,
            "MYSQL_PASSWORD": app_password,
            "MYSQL_URL": mysql_url,
            "JWT_SECRET_KEY": jwt_secret,
            "DEFAULT_ADMIN_USERNAME": base.get("DEFAULT_ADMIN_USERNAME", "admin"),
            "DEFAULT_ADMIN_PASSWORD": admin_password,
        }
    except (KeyError, TypeError):
        raise ManifestError("compose rollback capture failed") from None
    _validate_direct_environment_values(result)
    return result


def _read_rollback_credentials(
    path: Path,
    *,
    project: str,
    root: Path,
) -> dict[str, str]:
    try:
        _validate_input_path(path, role="rollback-direct", root=root)
        payload, _identity = _read_stable(path, max_bytes=_MAX_INPUT_BYTES)
        _validate_environment_schema(
            payload,
            role="rollback-direct",
            project=project,
            secret_mode="direct",
            mode="rollback-pre",
        )
        values = {
            key: _decode_single_quoted_path(value)
            for key, value in _parse_environment_file(payload).items()
        }
        if set(values) != _DIRECT_SECRET_KEYS:
            raise ManifestError("compose rollback capture failed")
        return values
    except ManifestError:
        raise ManifestError("compose rollback capture failed") from None


def capture_rollback(
    *,
    source: Path,
    project: str,
    api_service: str,
    web_service: str,
    rollback_direct_env: Path,
    release_sha: str,
    output_env: Path,
    output_manifest: Path,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
    token_hex: Callable[[int], str] = secrets.token_hex,
) -> dict[str, str]:
    """Capture the exact running API/Web images without reading container env."""
    runtime = Path(os.path.abspath(root / ".runtime"))
    if (
        _SAFE_PROJECT.fullmatch(project) is None
        or _RANDOM_PROJECT.fullmatch(project) is not None
        or api_service != "train-factory-api"
        or web_service != "train-factory-web"
        or re.fullmatch(r"[0-9a-f]{40}", release_sha, re.ASCII) is None
        or os.path.normcase(os.path.abspath(source))
        != os.path.normcase(os.fspath(runtime / "production-compose-manifest.json"))
        or os.path.normcase(os.path.abspath(rollback_direct_env))
        != os.path.normcase(os.fspath(runtime / "rollback-direct.env"))
        or os.path.normcase(os.path.abspath(output_env))
        != os.path.normcase(os.fspath(runtime / "rollback-pre.env"))
        or os.path.normcase(os.path.abspath(output_manifest))
        != os.path.normcase(os.fspath(runtime / "rollback-pre-compose-manifest.json"))
        or output_env.exists()
        or output_env.is_symlink()
        or output_manifest.exists()
        or output_manifest.is_symlink()
    ):
        raise ManifestError("compose rollback capture failed")
    from scripts import compose_release

    original = verify_manifest_inputs(source, root=root)
    if original["mode"] != "production" or original["project"] != project:
        raise ManifestError("compose rollback capture failed")
    source_selection = _selection_values(original)
    source_revision = source_selection.get(
        "RELEASE_REVISION", source_selection.get("WEB_REVISION")
    )
    if source_revision != release_sha:
        raise ManifestError("compose rollback capture failed")
    if _read_rollback_credentials(
        rollback_direct_env,
        project=project,
        root=root,
    ) != _deployment_credentials_from_manifest(original):
        raise ManifestError("compose rollback capture failed")
    environment = compose_release._clean_environment(base_environment)
    verified_source = verify_manifest_inputs_only(
        source,
        root=root,
        base_environment=base_environment,
        run=run,
    )
    if verified_source != original:
        raise ManifestError("compose rollback capture failed")
    token = token_hex(16)
    if re.fullmatch(r"[0-9a-f]{32}", token, re.ASCII) is None:
        raise ManifestError("compose rollback capture failed")
    captured: dict[str, tuple[str, str, str]] = {}
    captured_containers: dict[str, str] = {}
    tag_intents: list[tuple[str, str]] = []
    owned_outputs: dict[Path, tuple[int, int]] = {}
    try:
        container_format = (
            "{{.Id}}|{{.Image}}|{{.State.Running}}|"
            '{{ index .Config.Labels "com.docker.compose.project" }}|'
            '{{ index .Config.Labels "com.docker.compose.service" }}'
        )
        image_format = (
            "{{.Id}}|{{.Os}}|{{.Architecture}}|"
            '{{ index .Config.Labels "org.opencontainers.image.version" }}|'
            '{{ index .Config.Labels "org.opencontainers.image.revision" }}|'
            '{{ index .Config.Labels "org.opencontainers.image.created" }}|'
            '{{ index .Config.Labels "org.opencontainers.image.source" }}|'
            '{{ index .Config.Labels "io.train-factory.migration-compat" }}'
        )
        for service, suffix in (
            (api_service, "api"),
            (web_service, "web"),
        ):
            listed = compose_release._run_private(
                run,
                [
                    "docker",
                    "ps",
                    "--no-trunc",
                    "--filter",
                    f"label=com.docker.compose.project={project}",
                    "--filter",
                    f"label=com.docker.compose.service={service}",
                    "--format",
                    "{{.ID}}",
                ],
                environment,
            )
            identifiers = [
                line.strip() for line in listed.stdout.splitlines() if line.strip()
            ]
            if (
                len(identifiers) != 1
                or re.fullmatch(r"[0-9a-f]{64}", identifiers[0], re.ASCII) is None
            ):
                raise ManifestError("compose rollback capture failed")
            container_id = identifiers[0]
            captured_containers[service] = container_id

            def inspect_container() -> list[str]:
                inspected = compose_release._run_private(
                    run,
                    [
                        "docker",
                        "inspect",
                        "--format",
                        container_format,
                        container_id,
                    ],
                    environment,
                )
                return inspected.stdout.strip().split("|")

            container_parts = inspect_container()
            if (
                len(container_parts) != 5
                or container_parts
                != [
                    container_id,
                    container_parts[1],
                    "true",
                    project,
                    service,
                ]
                or re.fullmatch(r"sha256:[0-9a-f]{64}", container_parts[1], re.ASCII)
                is None
            ):
                raise ManifestError("compose rollback capture failed")
            image_id = container_parts[1]
            image = compose_release._run_private(
                run,
                ["docker", "image", "inspect", "--format", image_format, image_id],
                environment,
            )
            image_parts = image.stdout.strip().split("|")
            observed_compat = (
                ""
                if len(image_parts) == 8 and image_parts[7] == "<no value>"
                else image_parts[7]
                if len(image_parts) == 8
                else None
            )
            if (
                len(image_parts) != 8
                or image_parts[0] != image_id
                or image_parts[1:3] != ["linux", "amd64"]
                or not image_parts[3]
                or re.fullmatch(r"[0-9a-f]{40}", image_parts[4], re.ASCII) is None
                or not image_parts[5]
                or not image_parts[6].startswith("https://")
                or observed_compat != ""
                or inspect_container() != container_parts
            ):
                raise ManifestError("compose rollback capture failed")
            tag = f"trainfactory-rollback-{release_sha[:12]}-{token}-{suffix}:captured"
            collision = compose_release._run_private(
                run,
                [
                    "docker",
                    "image",
                    "ls",
                    "--filter",
                    f"reference={tag}",
                    "--format",
                    "{{.Repository}}:{{.Tag}}",
                ],
                environment,
            )
            if collision.stdout.strip():
                raise ManifestError("compose rollback capture failed")
            tag_intents.append((tag, image_id))
            compose_release._run_private(
                run,
                ["docker", "image", "tag", image_id, tag],
                environment,
            )
            tagged_id = compose_release._run_private(
                run,
                ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
                environment,
            ).stdout.strip()
            if tagged_id != image_id:
                raise ManifestError("compose rollback capture failed")
            captured[service] = (tag, image_id, image_parts[4])
        values = {
            "API_IMAGE": captured["train-factory-api"][0],
            "WEB_IMAGE": captured["train-factory-web"][0],
            "API_REVISION": captured["train-factory-api"][2],
            "WEB_REVISION": captured["train-factory-web"][2],
            "API_IMAGE_ID": captured["train-factory-api"][1],
            "WEB_IMAGE_ID": captured["train-factory-web"][1],
        }
        if verify_manifest_inputs(source, root=root) != original:
            raise ManifestError("compose rollback capture failed")
        owned_outputs[output_env] = _publish_selection_environment(
            output_env, values, root=root
        )
        base_record = next(
            item for item in original["env_files"] if item["role"] == "base-environment"
        )
        lock_record = next(
            item for item in original["env_files"] if item["role"] == "images-lock"
        )
        manifest_identities: list[tuple[int, int]] = []
        frozen = freeze_manifest(
            mode="rollback-pre",
            project=project,
            env_files=(
                Path(base_record["path"]),
                Path(lock_record["path"]),
                rollback_direct_env,
                output_env,
            ),
            gpu_mode=str(original["gpu_mode"]),
            secret_mode="direct",
            output=output_manifest,
            root=root,
            _exclusive_output=True,
            _published_identity=manifest_identities,
        )
        if len(manifest_identities) != 1:
            raise ManifestError("compose rollback capture failed")
        owned_outputs[output_manifest] = manifest_identities[0]
        verified_capture = verify_manifest_inputs_only(
            output_manifest,
            root=root,
            base_environment=base_environment,
            run=run,
        )
        verified_source = verify_manifest_inputs_only(
            source,
            root=root,
            base_environment=base_environment,
            run=run,
        )
        if (
            verified_capture != frozen
            or _selection_values(verified_capture) != values
            or verified_source != original
            or _deployment_credentials_from_manifest(verified_capture)
            != _deployment_credentials_from_manifest(verified_source)
            or _read_rollback_credentials(
                rollback_direct_env,
                project=project,
                root=root,
            )
            != _deployment_credentials_from_manifest(verified_source)
        ):
            raise ManifestError("compose rollback capture failed")
        for service, container_id in captured_containers.items():
            final_container = (
                compose_release._run_private(
                    run,
                    [
                        "docker",
                        "inspect",
                        "--format",
                        container_format,
                        container_id,
                    ],
                    environment,
                )
                .stdout.strip()
                .split("|")
            )
            if final_container != [
                container_id,
                captured[service][1],
                "true",
                project,
                service,
            ]:
                raise ManifestError("compose rollback capture failed")
        return values
    except Exception as error:
        cleanup_failed = False
        for tag, expected_image_id in reversed(tag_intents):
            try:
                listing = compose_release._run_private(
                    run,
                    [
                        "docker",
                        "image",
                        "ls",
                        "--filter",
                        f"reference={tag}",
                        "--format",
                        "{{.Repository}}:{{.Tag}}",
                    ],
                    environment,
                )
                names = [
                    line.strip() for line in listing.stdout.splitlines() if line.strip()
                ]
                if not names:
                    continue
                if names != [tag]:
                    cleanup_failed = True
                    continue
                current_id = compose_release._run_private(
                    run,
                    ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
                    environment,
                ).stdout.strip()
                if current_id != expected_image_id:
                    cleanup_failed = True
                    continue
                compose_release._run_private(
                    run,
                    ["docker", "image", "rm", tag],
                    environment,
                )
            except Exception:
                cleanup_failed = True
        for path, identity in owned_outputs.items():
            if not _unlink_owned_output(path, identity):
                cleanup_failed = True
        if cleanup_failed:
            raise ManifestError("compose rollback capture cleanup failed") from None
        if isinstance(error, ManifestError):
            raise
        raise ManifestError("compose rollback capture failed") from None


def replace_image_selection(
    *,
    source: Path,
    image_env: Path,
    mode: str,
    output: Path,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Create a new manifest by replacing only its final image-selection role."""
    runtime = Path(os.path.abspath(root / ".runtime"))
    contracts = {
        "rollback-post-migration": {
            "source_mode": "rollback-pre",
            "source": runtime / "rollback-pre-compose-manifest.json",
            "image": runtime / "rollback-post.env",
            "output": runtime / "rollback-post-compose-manifest.json",
            "role": "rollback-post",
            "api_compat": "053_validate_lifecycle_schema",
        },
        "rollback-web-only": {
            "source_mode": "production",
            "source": runtime / "production-compose-manifest.json",
            "image": runtime / "rollback-web-only.env",
            "output": runtime / "rollback-web-only-compose-manifest.json",
            "role": "image-selection",
            "api_compat": "",
        },
    }
    contract = contracts.get(mode)
    if (
        contract is None
        or any(
            os.path.normcase(os.path.abspath(actual))
            != os.path.normcase(os.fspath(contract[key]))
            for actual, key in (
                (source, "source"),
                (image_env, "image"),
                (output, "output"),
            )
        )
        or output.exists()
        or output.is_symlink()
    ):
        raise ManifestError("compose image selection replacement failed")
    owned_output: tuple[int, int] | None = None
    try:
        original = verify_manifest_inputs(source, root=root)
        verified_original = verify_manifest_inputs_only(
            source,
            root=root,
            base_environment=base_environment,
            run=run,
        )
        if verified_original != original:
            raise ManifestError("compose image selection replacement failed")
        if original["mode"] != contract["source_mode"]:
            raise ManifestError("compose image selection replacement failed")
        replacement = _read_selection_source(image_env, root=root)
        if set(replacement) != _ROLLBACK_IMAGE_SELECTION_KEYS:
            raise ManifestError("compose image selection replacement failed")
        from scripts import compose_release

        environment = compose_release._clean_environment(base_environment)
        _inspect_selected_image(
            replacement,
            prefix="API",
            run=run,
            environment=environment,
            expected_compat=str(contract["api_compat"]),
        )
        _inspect_selected_image(
            replacement,
            prefix="WEB",
            run=run,
            environment=environment,
        )
        if _read_selection_source(image_env, root=root) != replacement:
            raise ManifestError("compose image selection replacement failed")
        original_selection = _selection_values(original)
        if mode == "rollback-post-migration":
            if replacement["API_REVISION"] != original_selection["API_REVISION"] or any(
                replacement[key] != original_selection[key]
                for key in ("WEB_IMAGE", "WEB_REVISION", "WEB_IMAGE_ID")
            ):
                raise ManifestError("compose image selection replacement failed")
        else:
            original_api_revision = original_selection.get(
                "API_REVISION", original_selection.get("RELEASE_REVISION")
            )
        if mode == "rollback-web-only" and any(
            replacement[key] != expected
            for key, expected in (
                ("API_IMAGE", original_selection["API_IMAGE"]),
                ("API_REVISION", original_api_revision),
                ("API_IMAGE_ID", original_selection["API_IMAGE_ID"]),
            )
        ):
            raise ManifestError("compose image selection replacement failed")
        role = str(contract["role"])
        _validate_input_path(image_env, role=role, root=root)
        record, replacement_identity, payload = _record(
            image_env,
            role=role,
            sensitive=False,
        )
        if _parse_environment_file(payload) != replacement:
            raise ManifestError("compose image selection replacement failed")
        _validate_role_source(
            image_env,
            payload,
            role=role,
            mode=mode,
            project=str(original["project"]),
            secret_mode=str(original["secret_mode"]),
            rollback_variant=None,
            root=root,
        )
        replaced = dict(original)
        replaced["mode"] = mode
        replaced["env_files"] = [
            *[dict(item) for item in original["env_files"][:-1]],
            record,
        ]
        _validate_manifest_structure(replaced)
        identities = {replacement_identity}
        for path in (
            source,
            *(
                Path(item["path"])
                for item in (
                    *original["compose_files"],
                    *original["env_files"],
                    *original["referenced_files"],
                )
            ),
        ):
            metadata = Path(path).lstat()
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in identities:
                raise ManifestError("compose image selection replacement failed")
            identities.add(identity)
        if (
            verify_manifest_inputs_only(
                source,
                root=root,
                base_environment=base_environment,
                run=run,
            )
            != original
        ):
            raise ManifestError("compose image selection replacement failed")
        owned_output = _publish_manifest(
            output,
            replaced,
            root=root,
            input_identities=identities,
            exclusive=True,
        )
        verified_output = verify_manifest_inputs_only(
            output,
            root=root,
            base_environment=base_environment,
            run=run,
        )
        if (
            verified_output != replaced
            or _selection_values(verified_output) != replacement
        ):
            raise ManifestError("compose image selection replacement failed")
        return replaced
    except ManifestError:
        if owned_output is not None and not _unlink_owned_output(output, owned_output):
            raise ManifestError(
                "compose image selection replacement cleanup failed"
            ) from None
        raise
    except Exception:
        if owned_output is not None and not _unlink_owned_output(output, owned_output):
            raise ManifestError(
                "compose image selection replacement cleanup failed"
            ) from None
        raise ManifestError("compose image selection replacement failed") from None


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ManifestError("compose manifest arguments are invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _PrivateArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_PrivateArgumentParser,
    )
    freeze = subparsers.add_parser("freeze", allow_abbrev=False)
    freeze.add_argument("--mode", required=True)
    freeze.add_argument("--project", required=True)
    freeze.add_argument("--env-file", action="append", type=Path, required=True)
    freeze.add_argument("--gpu-mode", choices=("raw", "compat", "cpu"), required=True)
    freeze.add_argument("--secret-mode", choices=("files", "direct"), required=True)
    freeze.add_argument(
        "--rollback-variant",
        choices=tuple(sorted(_ROLLBACK_VARIANTS)),
    )
    freeze.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--inputs-only", action="store_true")
    running = subparsers.add_parser("verify-running", allow_abbrev=False)
    running.add_argument("--manifest", type=Path, required=True)
    running.add_argument("--image-selection", type=Path, required=True)
    running.add_argument(
        "--service",
        choices=("mysql", "train-factory-api", "train-factory-web"),
        required=True,
    )
    mix = subparsers.add_parser("mix-image-selection", allow_abbrev=False)
    mix.add_argument("--api-from", type=Path, required=True)
    mix.add_argument("--web-from", type=Path, required=True)
    mix.add_argument("--output", type=Path, required=True)
    capture = subparsers.add_parser("capture-rollback", allow_abbrev=False)
    capture.add_argument("--source", type=Path, required=True)
    capture.add_argument("--project", required=True)
    capture.add_argument("--api-service", required=True)
    capture.add_argument("--web-service", required=True)
    capture.add_argument("--rollback-direct-env", type=Path, required=True)
    capture.add_argument("--release-sha", required=True)
    capture.add_argument("--output-env", type=Path, required=True)
    capture.add_argument("--output-manifest", type=Path, required=True)
    replace = subparsers.add_parser("replace-image-selection", allow_abbrev=False)
    replace.add_argument("--source", type=Path, required=True)
    replace.add_argument("--image-env", type=Path, required=True)
    replace.add_argument(
        "--mode",
        choices=("rollback-post-migration", "rollback-web-only"),
        required=True,
    )
    replace.add_argument("--output", type=Path, required=True)
    selection = subparsers.add_parser("image-selection", allow_abbrev=False)
    selection.add_argument("--api-image", required=True)
    selection.add_argument("--web-image", required=True)
    selection.add_argument("--api-image-id", required=True)
    selection.add_argument("--web-image-id", required=True)
    selection.add_argument("--revision", required=True)
    selection.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "freeze":
            freeze_manifest(
                mode=arguments.mode,
                project=arguments.project,
                env_files=arguments.env_file,
                gpu_mode=arguments.gpu_mode,
                secret_mode=arguments.secret_mode,
                rollback_variant=arguments.rollback_variant,
                output=arguments.output,
            )
        elif arguments.command == "verify" and arguments.inputs_only:
            verify_manifest_inputs_only(arguments.manifest)
        elif arguments.command == "verify-running":
            verify_running_container(
                arguments.manifest,
                selection_path=arguments.image_selection,
                service=arguments.service,
            )
        elif arguments.command == "mix-image-selection":
            mix_image_selection(
                api_source=arguments.api_from,
                web_source=arguments.web_from,
                output=arguments.output,
            )
        elif arguments.command == "capture-rollback":
            capture_rollback(
                source=arguments.source,
                project=arguments.project,
                api_service=arguments.api_service,
                web_service=arguments.web_service,
                rollback_direct_env=arguments.rollback_direct_env,
                release_sha=arguments.release_sha,
                output_env=arguments.output_env,
                output_manifest=arguments.output_manifest,
            )
        elif arguments.command == "replace-image-selection":
            replace_image_selection(
                source=arguments.source,
                image_env=arguments.image_env,
                mode=arguments.mode,
                output=arguments.output,
            )
        elif arguments.command == "image-selection":
            create_image_selection(
                api_image=arguments.api_image,
                web_image=arguments.web_image,
                api_image_id=arguments.api_image_id,
                web_image_id=arguments.web_image_id,
                revision=arguments.revision,
                output=arguments.output,
            )
        else:
            raise ManifestError("compose manifest verification mode is required")
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        print("compose manifest command failed", file=sys.stderr)
        return 1
    except ManifestError:
        print("compose manifest command failed", file=sys.stderr)
        return 1
    except Exception:
        print("compose manifest command failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
