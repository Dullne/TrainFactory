"""Execute only the frozen, ordered Compose command surface."""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("release Compose bootstrap is unavailable")


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
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from decimal import Decimal, InvalidOperation  # noqa: E402
from pathlib import Path  # noqa: E402
from subprocess import CompletedProcess  # noqa: E402

if __package__:
    from scripts.materialize_compose_secrets import _verify_hardened_path
else:
    from scripts.materialize_compose_secrets import (  # type: ignore[no-redef]
        _verify_hardened_path,
    )

if __package__:
    from scripts.compose_manifest import (
        ROOT_DIR,
        ManifestError,
        _read_stable,
        _selection_values,
        _validate_ancestor_chain,
        verify_manifest_inputs,
    )
else:
    from scripts.compose_manifest import (  # type: ignore[no-redef]
        ROOT_DIR,
        ManifestError,
        _read_stable,
        _selection_values,
        _validate_ancestor_chain,
        verify_manifest_inputs,
    )


class ReleaseComposeError(RuntimeError):
    """A fixed-message wrapper failure safe for console output."""


_ENV_ALLOWLIST = {
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "DOCKER_CONFIG",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
}
_SERVICES = {"mysql", "train-factory-api", "train-factory-web"}
_SHARED_PRODUCTION_SELECTION_KEYS = frozenset(
    {
        "API_IMAGE",
        "WEB_IMAGE",
        "RELEASE_REVISION",
        "API_IMAGE_ID",
        "WEB_IMAGE_ID",
    }
)
_SPLIT_PRODUCTION_SELECTION_KEYS = frozenset(
    {
        "API_IMAGE",
        "WEB_IMAGE",
        "API_REVISION",
        "WEB_REVISION",
        "API_IMAGE_ID",
        "WEB_IMAGE_ID",
    }
)
_SPLIT_PRODUCTION_WEB_TAIL = (
    "up",
    "-d",
    "--no-deps",
    "--force-recreate",
    "--wait",
    "--wait-timeout",
    "600",
    "train-factory-web",
)
_SERVICE_KEYS = {
    "mysql": frozenset(
        "command container_name entrypoint environment healthcheck image networks "
        "restart secrets volumes".split()
    ),
    "train-factory-api": frozenset(
        "command container_name depends_on deploy entrypoint environment image "
        "networks pid ports restart secrets shm_size volumes".split()
    ),
    "train-factory-web": frozenset(
        "command container_name depends_on entrypoint environment healthcheck image "
        "networks ports restart".split()
    ),
}
_MYSQL_ENVIRONMENT_KEYS = frozenset(
    "MYSQL_CHARSET MYSQL_COLLATION MYSQL_DATABASE MYSQL_PASSWORD "
    "MYSQL_PASSWORD_FILE MYSQL_ROOT_PASSWORD MYSQL_ROOT_PASSWORD_FILE MYSQL_USER".split()
)
_WEB_ENVIRONMENT_KEYS = frozenset("API_HOST API_PORT".split())
_API_ENVIRONMENT_KEYS = frozenset(
    """
    ALLOW_MODEL_REMOTE_CODE API_HOST API_PORT API_WORKERS APP_TIMEZONE
    AUDIT_LOG_CLEANUP_INTERVAL_HOURS AUDIT_LOG_RETENTION_DAYS AUTH_COOKIE_SECURE
    AUTH_ENABLED BACKGROUND_TASK_MAX_ACTIVE_GLOBAL BACKGROUND_TASK_MAX_ACTIVE_PER_USER
    COMPOSE_PROJECT_NAME DATASETS_DIR DATASET_DOWNLOAD_MAX_BYTES DB_MAX_OVERFLOW
    DB_POOL_SIZE DEFAULT_ADMIN_EMAIL DEFAULT_ADMIN_PASSWORD
    DEFAULT_ADMIN_PASSWORD_FILE DEFAULT_ADMIN_USERNAME DISCOVER_MODELS_ALLOWED_PRIVATE_HOSTS
    DISCOVER_MODELS_ALLOW_PRIVATE_ALL DOCKER_NETWORK_NAME DOWNLOAD_MAX_CONCURRENT_GLOBAL
    DOWNLOAD_MAX_CONCURRENT_PER_USER DOWNLOAD_MAX_WORKERS
    DOWNLOAD_METADATA_TIMEOUT_SECONDS DOWNLOAD_STORAGE_MAX_BYTES_GLOBAL
    DOWNLOAD_STORAGE_MAX_BYTES_PER_USER GPU_PREFLIGHT_MODE HF_DATASETS_CACHE HF_ENDPOINT
    HF_HOME HOST_BIND_ADDRESS HOST_IP JWT_SECRET_KEY JWT_SECRET_KEY_FILE LOG_LEVEL
    MILVUS_HOST MILVUS_PORT MINIO_ACCESS_KEY MINIO_BUCKET MINIO_ENDPOINT
    MINIO_SECRET_KEY MODELSCOPE_CACHE MODELS_DIR MODEL_DOWNLOAD_MAX_BYTES
    MYSQL_APP_PASSWORD MYSQL_APP_USER MYSQL_DATABASE MYSQL_HOST MYSQL_URL_FILE
    NVIDIA_DISABLE_REQUIRE NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES OUTPUT_DIR
    PATH_MAPPINGS PORT_RANGE_END PORT_RANGE_START PUBLIC_BASE_URL RATE_LIMIT_DEFAULT
    RATE_LIMIT_ENABLED RATE_LIMIT_LOGIN RATE_LIMIT_REGISTER RATE_LIMIT_TRUSTED_PROXIES
    SELF_REGISTRATION_ENABLED SGLANG_IMAGE STORAGE_BACKEND SWANLAB_API_KEY
    SYNC_BOUNDARY_MAX_BYTES SYNC_BOUNDARY_MAX_IDS SYNC_GENERATION_MAX_INPUT_BYTES
    SYNC_HISTORICAL_MAX_BYTES SYNC_HISTORICAL_MAX_DOCS SYNC_MAX_FUTURE_SKEW_SECONDS
    SYNC_MAX_RECORD_BYTES SYNC_PENDING_MAX_BATCHES_PER_TASK
    SYNC_PENDING_MAX_RECORDS_PER_TASK SYNC_STORAGE_MAX_BYTES_GLOBAL
    SYNC_STORAGE_MAX_BYTES_PER_USER TRAINING_ALLOW_CPU_FALLBACK TRAINING_CACHE
    VLLM_IMAGE XINFERENCE_IMAGE
    """.split()
)
_API_ENVIRONMENT_DEFAULT_VALUES = dict(
    line.split("=", 1)
    for line in """
ALLOW_MODEL_REMOTE_CODE=false
API_HOST=0.0.0.0
API_PORT=18000
API_WORKERS=1
APP_TIMEZONE=UTC
AUDIT_LOG_CLEANUP_INTERVAL_HOURS=24
AUDIT_LOG_RETENTION_DAYS=90
AUTH_COOKIE_SECURE=false
AUTH_ENABLED=true
BACKGROUND_TASK_MAX_ACTIVE_GLOBAL=8
BACKGROUND_TASK_MAX_ACTIVE_PER_USER=2
COMPOSE_PROJECT_NAME=trainfactory
DATASETS_DIR=/app/data/datasets
DATASET_DOWNLOAD_MAX_BYTES=10737418240
DB_MAX_OVERFLOW=10
DB_POOL_SIZE=5
DEFAULT_ADMIN_EMAIL=
DEFAULT_ADMIN_USERNAME=admin
DISCOVER_MODELS_ALLOWED_PRIVATE_HOSTS=
DISCOVER_MODELS_ALLOW_PRIVATE_ALL=false
DOCKER_NETWORK_NAME=trainfactory_network
DOWNLOAD_MAX_CONCURRENT_GLOBAL=4
DOWNLOAD_MAX_CONCURRENT_PER_USER=2
DOWNLOAD_MAX_WORKERS=4
DOWNLOAD_METADATA_TIMEOUT_SECONDS=15
DOWNLOAD_STORAGE_MAX_BYTES_GLOBAL=214748364800
DOWNLOAD_STORAGE_MAX_BYTES_PER_USER=64424509440
GPU_PREFLIGHT_MODE=required
HF_DATASETS_CACHE=/app/cache/hf_datasets
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/app/cache/hf_home
HOST_BIND_ADDRESS=127.0.0.1
HOST_IP=172.17.0.1
LOG_LEVEL=INFO
MILVUS_HOST=milvus
MILVUS_PORT=19530
MINIO_ACCESS_KEY=
MINIO_BUCKET=trainfactory
MINIO_ENDPOINT=minio:9000
MINIO_SECRET_KEY=
MODELSCOPE_CACHE=/app/cache/modelscope
MODELS_DIR=/app/models
MODEL_DOWNLOAD_MAX_BYTES=32212254720
MYSQL_APP_USER=trainfactory_app
MYSQL_DATABASE=train_factory
MYSQL_HOST=mysql
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_VISIBLE_DEVICES=all
OUTPUT_DIR=/app/output
PATH_MAPPINGS=
PORT_RANGE_END=10100
PORT_RANGE_START=9997
PUBLIC_BASE_URL=http://localhost:3000
RATE_LIMIT_DEFAULT=100/minute
RATE_LIMIT_ENABLED=true
RATE_LIMIT_LOGIN=5/minute
RATE_LIMIT_REGISTER=3/hour
RATE_LIMIT_TRUSTED_PROXIES=172.18.0.4/32
SELF_REGISTRATION_ENABLED=false
SGLANG_IMAGE=lmsysorg/sglang:v0.5.16@sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0
STORAGE_BACKEND=local
SWANLAB_API_KEY=
SYNC_BOUNDARY_MAX_BYTES=4194304
SYNC_BOUNDARY_MAX_IDS=10000
SYNC_GENERATION_MAX_INPUT_BYTES=1073741824
SYNC_HISTORICAL_MAX_BYTES=67108864
SYNC_HISTORICAL_MAX_DOCS=100000
SYNC_MAX_FUTURE_SKEW_SECONDS=300
SYNC_MAX_RECORD_BYTES=8388608
SYNC_PENDING_MAX_BATCHES_PER_TASK=1000
SYNC_PENDING_MAX_RECORDS_PER_TASK=1000000
SYNC_STORAGE_MAX_BYTES_GLOBAL=21474836480
SYNC_STORAGE_MAX_BYTES_PER_USER=5368709120
TRAINING_ALLOW_CPU_FALLBACK=false
TRAINING_CACHE=/app/cache
VLLM_IMAGE=vllm/vllm-openai:v0.11.0@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c
XINFERENCE_IMAGE=xprobe/xinference:v1.13.0@sha256:b5df50f3d04e5f7290cf0c6765b5a2b06ab29657d721cfaa13389a7c1d666291
""".strip().splitlines()
)
_HEALTHCHECK_HASHES = {
    "mysql": "3a8e6220e9e064cbbb5d4798114c8c4b6dae601c9a1fb646d4d00682df279c81",
    "train-factory-web": "82a1c0ce480c669c97dc5e1c777ce54628a59aff0104781ebee952b83de49490",
}


def _normalized_compose_size(value: str) -> str:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kmgt]?)(?:i?b)?", value, re.I)
    if match is None:
        raise ValueError("size")
    multipliers = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    try:
        result = Decimal(match.group(1)) * multipliers[match.group(2).lower()]
    except InvalidOperation:
        raise ValueError("size") from None
    if result != result.to_integral_value() or result <= 0:
        raise ValueError("size")
    return str(int(result))


def _validate_up_tail(tail: Sequence[str]) -> tuple[tuple[str, ...], int]:
    if not tail or tail[0] != "up":
        raise ReleaseComposeError("release Compose command is invalid")
    result = tuple(tail)
    detached = False
    wait = False
    no_dependencies = False
    force_recreate = False
    wait_timeout: int | None = None
    services: list[str] = []
    index = 1
    while index < len(result):
        argument = result[index]
        if argument in {"-d", "--detach"}:
            if detached:
                raise ReleaseComposeError("release Compose command is invalid")
            detached = True
        elif argument == "--wait":
            if wait:
                raise ReleaseComposeError("release Compose command is invalid")
            wait = True
        elif argument == "--no-deps":
            if no_dependencies:
                raise ReleaseComposeError("release Compose command is invalid")
            no_dependencies = True
        elif argument == "--force-recreate":
            if force_recreate:
                raise ReleaseComposeError("release Compose command is invalid")
            force_recreate = True
        elif argument == "--wait-timeout":
            if wait_timeout is not None:
                raise ReleaseComposeError("release Compose command is invalid")
            index += 1
            if index >= len(result) or not result[index].isdigit():
                raise ReleaseComposeError("release Compose command is invalid")
            wait_timeout = int(result[index])
            if not 1 <= wait_timeout <= 3600:
                raise ReleaseComposeError("release Compose command is invalid")
        elif argument.startswith("-") or argument not in _SERVICES:
            raise ReleaseComposeError("release Compose command is invalid")
        elif argument in services:
            raise ReleaseComposeError("release Compose command is invalid")
        else:
            services.append(argument)
        index += 1
    if not detached or not wait or not services:
        raise ReleaseComposeError("release Compose command is invalid")
    return result, (wait_timeout or 600) + 60


def _validate_staged_replacement(
    mode: str,
    tail: Sequence[str],
    *,
    selection_keys: frozenset[str] | None = None,
) -> None:
    if mode == "rollback-web-only":
        if tuple(tail) != _SPLIT_PRODUCTION_WEB_TAIL or (
            selection_keys != _SPLIT_PRODUCTION_SELECTION_KEYS
        ):
            raise ReleaseComposeError("release Compose command is invalid")
        return
    if mode == "production" and selection_keys is not None:
        if selection_keys == _SPLIT_PRODUCTION_SELECTION_KEYS:
            if tuple(tail) != _SPLIT_PRODUCTION_WEB_TAIL:
                raise ReleaseComposeError("release Compose command is invalid")
            return
        if selection_keys != _SHARED_PRODUCTION_SELECTION_KEYS:
            raise ReleaseComposeError("release Compose command is invalid")
    staged_modes = {
        "production",
        "rollback-pre",
        "rollback-post-migration",
    }
    services = tuple(argument for argument in tail if argument in _SERVICES)
    flags = {
        argument for argument in tail if argument in {"--no-deps", "--force-recreate"}
    }
    staged_service = len(services) == 1 and services[0] in {
        "train-factory-api",
        "train-factory-web",
    }
    if (mode in staged_modes and staged_service) != (
        flags == {"--no-deps", "--force-recreate"}
    ):
        raise ReleaseComposeError("release Compose command is invalid")
    if flags and (mode not in staged_modes or not staged_service):
        raise ReleaseComposeError("release Compose command is invalid")


def _clean_environment(source: Mapping[str, str]) -> dict[str, str]:
    clean: dict[str, str] = {}
    canonical_source: dict[str, str] = {}
    for key, value in source.items():
        canonical = key.upper()
        if canonical in canonical_source and canonical_source[canonical] != value:
            raise ReleaseComposeError("release Compose environment is invalid")
        canonical_source[canonical] = value
        if canonical not in _ENV_ALLOWLIST:
            continue
        if canonical in clean and clean[canonical] != value:
            raise ReleaseComposeError("release Compose environment is invalid")
        clean[canonical] = value
    if "DOCKER_CONTEXT" in clean and "DOCKER_CONFIG" not in clean:
        home = canonical_source.get("USERPROFILE") or canonical_source.get("HOME")
        try:
            if not home:
                raise ValueError("home")
            candidate = (Path(home) / ".docker").resolve(strict=True)
            if (
                not candidate.is_absolute()
                or not candidate.is_dir()
                or candidate.is_symlink()
            ):
                raise ValueError("config")
        except (OSError, ValueError):
            raise ReleaseComposeError(
                "release Compose environment is invalid"
            ) from None
        clean["DOCKER_CONFIG"] = os.fspath(candidate)
    clean["COMPOSE_DISABLE_ENV_FILE"] = "1"
    return clean


def execute_manifest(
    manifest_path: Path,
    tail: Sequence[str],
    *,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
    mutation_guard: Callable[[], None] | None = None,
) -> None:
    """Run an approved command against a verified immutable input manifest."""
    try:
        approved_tail, command_timeout = _validate_up_tail(tail)
        manifest = verify_manifest_inputs(manifest_path, root=root)
        selection = _selection_values(manifest)
        selection_keys = frozenset(selection)
        _validate_staged_replacement(
            str(manifest["mode"]),
            approved_tail,
            selection_keys=selection_keys,
        )
        environment = _clean_environment(base_environment)
        compose = _load_resolved_compose(
            manifest,
            root=root,
            run=run,
            environment=environment,
        )
        _validate_resolved_config(manifest, compose, root=root)
        if manifest["mode"] in {"verify", "ci", "rollback-verify"}:
            _require_empty_project_resources(
                run,
                environment,
                str(manifest["project"]),
            )
            expected_volumes = _expected_volume_names(
                compose,
                str(manifest["project"]),
            )
            existing = _parse_resource_listing(
                _run_private(
                    run,
                    ["docker", "volume", "ls", "--format", "{{.Name}}"],
                    environment,
                ).stdout
            )
            if expected_volumes.intersection(existing):
                raise ReleaseComposeError(
                    "release Compose project resources are not empty"
                )
            expected_networks = _expected_network_names(
                compose,
                str(manifest["project"]),
            )
            existing_networks = _parse_resource_listing(
                _run_private(
                    run,
                    ["docker", "network", "ls", "--format", "{{.Name}}"],
                    environment,
                ).stdout
            )
            if expected_networks.intersection(existing_networks):
                raise ReleaseComposeError(
                    "release Compose project resources are not empty"
                )
            existing_containers = _parse_resource_listing(
                _run_private(
                    run,
                    ["docker", "container", "ls", "-a", "--format", "{{.Names}}"],
                    environment,
                ).stdout
            )
            project_prefixes = (
                str(manifest["project"]) + "-",
                str(manifest["project"]) + "_",
            )
            if any(name.startswith(project_prefixes) for name in existing_containers):
                raise ReleaseComposeError(
                    "release Compose project resources are not empty"
                )
        from scripts.compose_manifest import verify_manifest_inputs_only

        verified_manifest = verify_manifest_inputs_only(
            manifest_path,
            root=root,
            base_environment=base_environment,
            run=run,
        )
        if verified_manifest != manifest:
            raise ReleaseComposeError("release Compose command failed")
        execution_compose = copy.deepcopy(compose)
        execution_services = execution_compose["services"]
        execution_services["train-factory-api"]["image"] = selection["API_IMAGE_ID"]
        execution_services["train-factory-web"]["image"] = selection["WEB_IMAGE_ID"]
        resolved_payload = (
            json.dumps(
                execution_compose,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        argv = [
            "docker",
            "compose",
            "--project-directory",
            os.fspath(root / "docker"),
            "-f",
            "-",
        ]
        argv.extend(("-p", str(manifest["project"]), *approved_tail))
        if mutation_guard is not None:
            mutation_guard()
        completed = run(
            argv,
            shell=False,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=resolved_payload,
            timeout=command_timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise ReleaseComposeError("release Compose command failed")
    except ReleaseComposeError:
        raise
    except (ManifestError, OSError, subprocess.SubprocessError, TypeError, ValueError):
        raise ReleaseComposeError("release Compose command failed") from None


_VERIFY_VOLUME_KEYS = {
    "mysql_data",
    "train_cache",
    "verify_data",
    "verify_models",
    "verify_output",
    "etcd_data",
    "minio_data",
    "milvus_data",
    "web_node_modules",
    "e2e_node_modules",
    "xinference_cache",
    "xinference_home",
}


def _run_private(
    run: Callable[..., CompletedProcess[str]],
    argv: Sequence[str],
    environment: Mapping[str, str],
    *,
    timeout: float = 600,
    timeout_message: str = "release Compose command failed",
) -> CompletedProcess[str]:
    try:
        completed = run(
            list(argv),
            shell=False,
            env=dict(environment),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ReleaseComposeError(timeout_message) from None
    except (OSError, subprocess.SubprocessError):
        raise ReleaseComposeError("release Compose command failed") from None
    if not isinstance(completed, CompletedProcess) or completed.returncode != 0:
        raise ReleaseComposeError("release Compose command failed")
    return completed


def _parse_resource_listing(payload: str) -> tuple[str, ...]:
    names = tuple(line.strip() for line in payload.splitlines() if line.strip())
    if (
        len(names) != len(set(names))
        or any(any(character.isspace() for character in name) for name in names)
        or any(any(ord(character) < 32 for character in name) for name in names)
    ):
        raise ReleaseComposeError("release Compose project resources are invalid")
    return names


def _list_project_resources(
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
    project: str,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for resource, output_format in (
        ("container", "{{.ID}}"),
        ("network", "{{.Name}}"),
        ("volume", "{{.Name}}"),
    ):
        argv = ["docker", resource, "ls"]
        if resource == "container":
            argv.append("-a")
        argv.extend(
            (
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                output_format,
            )
        )
        completed = _run_private(run, argv, environment)
        result[resource] = _parse_resource_listing(completed.stdout)
    return result


def _require_empty_project_resources(
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
    project: str,
) -> None:
    resources = _list_project_resources(run, environment, project)
    if any(resources.values()):
        raise ReleaseComposeError("release Compose project resources are not empty")


def _compose_prefix(manifest: Mapping[str, object], *, root: Path) -> list[str]:
    argv = [
        "docker",
        "compose",
        "--project-directory",
        os.fspath(root / "docker"),
    ]
    for item in manifest["env_files"]:  # type: ignore[union-attr]
        argv.extend(("--env-file", str(item["path"])))
    for item in manifest["compose_files"]:  # type: ignore[union-attr]
        argv.extend(("-f", str(item["path"])))
    argv.extend(("-p", str(manifest["project"])))
    return argv


def _load_resolved_compose(
    manifest: Mapping[str, object],
    *,
    root: Path,
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
) -> dict[str, object]:
    completed = _run_private(
        run,
        [*_compose_prefix(manifest, root=root), "config", "--format", "json"],
        environment,
    )
    try:
        result = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        raise ReleaseComposeError(
            "release Compose resolved config is invalid"
        ) from None
    if not isinstance(result, dict):
        raise ReleaseComposeError("release Compose resolved config is invalid")
    return result


def _expected_volume_names(
    compose: Mapping[str, object],
    project: str,
) -> set[str]:
    try:
        services = compose["services"]
        volumes = compose["volumes"]
        if not isinstance(services, dict) or not isinstance(volumes, dict):
            raise ValueError("shape")
        if not all(isinstance(service, dict) for service in services.values()):
            raise ValueError("services")
        used_keys: set[str] = set()
        for service in services.values():
            mounts = service.get("volumes", [])
            if not isinstance(mounts, list):
                raise ValueError("mounts")
            for mount in mounts:
                if not isinstance(mount, dict):
                    raise ValueError("mount")
                if mount.get("type") == "volume":
                    source = mount.get("source")
                    if not isinstance(source, str) or source not in volumes:
                        raise ValueError("source")
                    used_keys.add(source)
        expected_names: set[str] = set()
        for key in used_keys:
            definition = volumes[key]
            expected = f"{project}_{key}"
            if (
                not isinstance(definition, dict)
                or definition.get("external")
                or definition.get("name") != expected
            ):
                raise ValueError("definition")
            expected_names.add(expected)
    except (KeyError, TypeError, ValueError):
        raise ReleaseComposeError(
            "release Compose resolved config is invalid"
        ) from None
    if not expected_names:
        raise ReleaseComposeError("release Compose resolved config is invalid")
    return expected_names


def _expected_network_names(
    compose: Mapping[str, object],
    project: str,
) -> set[str]:
    try:
        networks = compose["networks"]
        if not isinstance(networks, dict) or set(networks) != {"default"}:
            raise ValueError("networks")
        definition = networks["default"]
        expected = f"{project}_default"
        if (
            not isinstance(definition, dict)
            or set(definition) - {"name", "labels", "ipam"}
            or definition.get("external")
            or definition.get("name") != expected
            or definition.get("ipam") not in (None, {})
        ):
            raise ValueError("network")
    except (KeyError, TypeError, ValueError):
        raise ReleaseComposeError(
            "release Compose resolved config is invalid"
        ) from None
    return {expected}


def _validate_resolved_config(
    manifest: Mapping[str, object],
    compose: Mapping[str, object],
    *,
    root: Path,
) -> None:
    """Reject resolved runtime surfaces outside the frozen deployment policy."""
    try:
        allowed_top_level = {"name", "networks", "services", "volumes"}
        if manifest["secret_mode"] == "files":
            allowed_top_level.add("secrets")
        if set(compose) - allowed_top_level or compose.get("name") not in {
            None,
            manifest["project"],
        }:
            raise ValueError("top-level schema")

        def normalized_mounts(
            service: Mapping[str, object],
        ) -> set[tuple[str, str, str, bool]]:
            mounts = service.get("volumes", [])
            if not isinstance(mounts, list):
                raise ValueError("mounts")
            observed: set[tuple[str, str, str, bool]] = set()
            targets: set[str] = set()
            for mount in mounts:
                if not isinstance(mount, dict):
                    raise ValueError("mount")
                mount_type = mount.get("type")
                source = mount.get("source")
                target = mount.get("target")
                if not all(
                    isinstance(value, str) for value in (mount_type, source, target)
                ):
                    raise ValueError("mount fields")
                if target in targets:
                    raise ValueError("duplicate mount")
                targets.add(target)
                if mount_type == "volume":
                    if (
                        set(mount) != {"type", "source", "target", "volume"}
                        or mount.get("volume") != {}
                    ):
                        raise ValueError("volume mount")
                    read_only = False
                elif mount_type == "bind":
                    allowed = {"type", "source", "target", "bind", "read_only"}
                    if (
                        set(mount) - allowed
                        or mount.get("bind") != {"create_host_path": True}
                        or mount.get("read_only") not in {None, True}
                    ):
                        raise ValueError("bind mount")
                    source = os.path.normcase(os.path.abspath(source))
                    read_only = mount.get("read_only") is True
                else:
                    raise ValueError("mount type")
                observed.add((mount_type, source, target, read_only))
            if len(observed) != len(mounts):
                raise ValueError("duplicate mount")
            return observed

        services = compose["services"]
        if not isinstance(services, dict) or set(services) != _SERVICES:
            raise ValueError("services")
        environment_keys = {
            "mysql": _MYSQL_ENVIRONMENT_KEYS,
            "train-factory-api": _API_ENVIRONMENT_KEYS,
            "train-factory-web": _WEB_ENVIRONMENT_KEYS,
        }
        file_api_keys = {
            "MYSQL_URL_FILE",
            "JWT_SECRET_KEY_FILE",
            "DEFAULT_ADMIN_PASSWORD_FILE",
        }
        direct_api_keys = {
            "MYSQL_APP_PASSWORD",
            "JWT_SECRET_KEY",
            "DEFAULT_ADMIN_PASSWORD",
        }
        expected_environment_keys = {
            "mysql": (
                {
                    "MYSQL_CHARSET",
                    "MYSQL_COLLATION",
                    "MYSQL_DATABASE",
                    "MYSQL_PASSWORD_FILE",
                    "MYSQL_ROOT_PASSWORD_FILE",
                    "MYSQL_USER",
                }
                if manifest["secret_mode"] == "files"
                else {
                    "MYSQL_CHARSET",
                    "MYSQL_COLLATION",
                    "MYSQL_DATABASE",
                    "MYSQL_PASSWORD",
                    "MYSQL_ROOT_PASSWORD",
                    "MYSQL_USER",
                }
            ),
            "train-factory-api": (
                set(_API_ENVIRONMENT_KEYS) - file_api_keys - direct_api_keys
            ),
            "train-factory-web": set(_WEB_ENVIRONMENT_KEYS),
        }
        if manifest["secret_mode"] == "files":
            expected_environment_keys["train-factory-api"].update(
                file_api_keys | direct_api_keys
            )
        else:
            expected_environment_keys["train-factory-api"].update(direct_api_keys)
        expected_commands = {
            "mysql": [
                "--character-set-server=utf8mb4",
                "--collation-server=utf8mb4_unicode_ci",
            ],
            "train-factory-api": None,
            "train-factory-web": None,
        }
        expected_dependencies = {
            "train-factory-api": {
                "mysql": {"condition": "service_healthy", "required": True}
            },
            "train-factory-web": {
                "train-factory-api": {
                    "condition": "service_started",
                    "required": True,
                }
            },
        }
        random_mode = manifest["mode"] in {"verify", "ci", "rollback-verify"}
        from scripts.compose_manifest import (
            _decode_single_quoted_path,
            _manifest_environment_values,
        )

        control_values = {
            "ALLOW_MODEL_REMOTE_CODE": "false",
            "API_CPU_LIMIT": "0",
            "API_MEM_LIMIT": "0",
            "API_PIDS_LIMIT": "0",
            "API_PORT": "18000",
            "API_SHM_SIZE": "1g",
            "AUTH_COOKIE_SECURE": "false",
            "AUTH_ENABLED": "true",
            "DEFAULT_ADMIN_USERNAME": "admin",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HOST_BIND_ADDRESS": "127.0.0.1",
            "LOG_LEVEL": "INFO",
            "MYSQL_APP_USER": "trainfactory_app",
            "PUBLIC_BASE_URL": "http://localhost:3000",
            "SELF_REGISTRATION_ENABLED": "false",
            "WEB_PORT": "3000",
        }
        ordered_roles = [
            str(item["role"])
            for item in manifest.get("env_files", [])
            if isinstance(item, dict) and isinstance(item.get("role"), str)
        ]
        roles = set(ordered_roles)
        for role in ordered_roles:
            values = _manifest_environment_values(manifest, role)
            if role in {"production-direct", "rollback-direct"}:
                values = {
                    key: _decode_single_quoted_path(value)
                    for key, value in values.items()
                }
            control_values.update(values)
        direct_values: dict[str, str] = {}
        direct_role = next(
            (
                role
                for role in ("rollback-direct", "production-direct")
                if role in roles
            ),
            None,
        )
        if direct_role is not None:
            direct_values = {
                key: _decode_single_quoted_path(value)
                for key, value in _manifest_environment_values(
                    manifest, direct_role
                ).items()
            }
            control_values.update(
                {
                    key: direct_values[key]
                    for key in (
                        "API_PORT",
                        "AUTH_COOKIE_SECURE",
                        "HOST_BIND_ADDRESS",
                        "PUBLIC_BASE_URL",
                        "WEB_PORT",
                    )
                    if key in direct_values
                }
            )
        expected_network_environment = (
            f"{manifest['project']}_default"
            if random_mode
            else control_values.get("DOCKER_NETWORK_NAME", "trainfactory_network")
        )
        expected_environment = {
            "mysql": {
                "MYSQL_CHARSET": "utf8mb4",
                "MYSQL_COLLATION": "utf8mb4_unicode_ci",
                "MYSQL_DATABASE": "train_factory",
                "MYSQL_USER": control_values["MYSQL_APP_USER"],
            },
            "train-factory-api": dict(_API_ENVIRONMENT_DEFAULT_VALUES),
            "train-factory-web": {
                "API_HOST": "train-factory-api",
                "API_PORT": control_values["API_PORT"],
            },
        }
        expected_api = expected_environment["train-factory-api"]
        expected_api.update(
            {
                key: value
                for key, value in control_values.items()
                if key in _API_ENVIRONMENT_KEYS
            }
        )
        expected_api.update(
            {
                "API_HOST": "0.0.0.0",
                "COMPOSE_PROJECT_NAME": str(manifest["project"]),
                "DOCKER_NETWORK_NAME": expected_network_environment,
                "GPU_PREFLIGHT_MODE": (
                    "off" if manifest["gpu_mode"] == "cpu" else "required"
                ),
                "MYSQL_DATABASE": "train_factory",
                "MYSQL_HOST": "mysql",
            }
        )
        if manifest["gpu_mode"] == "cpu":
            expected_api.update(
                {
                    "NVIDIA_DRIVER_CAPABILITIES": "",
                    "NVIDIA_VISIBLE_DEVICES": "void",
                }
            )
        if manifest["secret_mode"] == "files":
            expected_environment["mysql"].update(
                {
                    "MYSQL_PASSWORD_FILE": "/run/secrets/mysql_app_password",
                    "MYSQL_ROOT_PASSWORD_FILE": "/run/secrets/mysql_root_password",
                }
            )
            expected_api.update(
                {
                    "DEFAULT_ADMIN_PASSWORD": "",
                    "DEFAULT_ADMIN_PASSWORD_FILE": "/run/secrets/default_admin_password",
                    "JWT_SECRET_KEY": "",
                    "JWT_SECRET_KEY_FILE": "/run/secrets/jwt_secret_key",
                    "MYSQL_APP_PASSWORD": "",
                    "MYSQL_URL_FILE": "/run/secrets/mysql_url",
                }
            )
        elif direct_values:
            expected_environment["mysql"].update(
                {
                    "MYSQL_PASSWORD": direct_values["MYSQL_PASSWORD"],
                    "MYSQL_ROOT_PASSWORD": direct_values["MYSQL_ROOT_PASSWORD"],
                }
            )
            expected_api.update(
                {
                    "DEFAULT_ADMIN_PASSWORD": direct_values["DEFAULT_ADMIN_PASSWORD"],
                    "JWT_SECRET_KEY": direct_values["JWT_SECRET_KEY"],
                    "MYSQL_APP_PASSWORD": direct_values["MYSQL_APP_PASSWORD"],
                }
            )
        else:
            mysql_environment = services["mysql"].get("environment", {})
            api_environment = services["train-factory-api"].get("environment", {})
            if not isinstance(mysql_environment, dict) or not isinstance(
                api_environment, dict
            ):
                raise ValueError("service environment")
            expected_environment["mysql"].update(
                {
                    key: mysql_environment.get(key)
                    for key in ("MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD")
                }
            )
            expected_api.update(
                {key: api_environment.get(key) for key in direct_api_keys}
            )
        expected_api["NVIDIA_DISABLE_REQUIRE"] = {
            "raw": "0",
            "compat": "1",
            "cpu": "",
        }[str(manifest["gpu_mode"])]
        expected_environment = {
            service_name: {
                key: str(values[key]) for key in expected_environment_keys[service_name]
            }
            for service_name, values in expected_environment.items()
        }
        for service_name, service in services.items():
            expected_keys = set(_SERVICE_KEYS[service_name])
            if random_mode:
                expected_keys.discard("container_name")
                expected_keys.discard("pid")
            if manifest["secret_mode"] != "files":
                expected_keys.discard("secrets")
            if not isinstance(service, dict) or set(service) != expected_keys:
                raise ValueError("service schema")
            environment_values = service.get("environment", {})
            if (
                not isinstance(environment_values, dict)
                or not set(environment_values) <= environment_keys[service_name]
                or set(environment_values) != expected_environment_keys[service_name]
                or environment_values != expected_environment[service_name]
            ):
                raise ValueError("service environment")
            if (
                "command" in service
                and service["command"] != expected_commands[service_name]
            ):
                raise ValueError("service command")
            if "entrypoint" in service and service["entrypoint"] is not None:
                raise ValueError("service entrypoint")
            if service_name in expected_dependencies and "depends_on" in service:
                if service["depends_on"] != expected_dependencies[service_name]:
                    raise ValueError("service dependency")
            if "healthcheck" in service:
                healthcheck = service["healthcheck"]
                payload = json.dumps(
                    healthcheck, sort_keys=True, separators=(",", ":")
                ).encode()
                if hashlib.sha256(payload).hexdigest() != _HEALTHCHECK_HASHES.get(
                    service_name
                ):
                    raise ValueError("service healthcheck")
            deploy = service.get("deploy")
            if deploy is not None:
                if not isinstance(deploy, dict) or set(deploy) - {
                    "placement",
                    "resources",
                }:
                    raise ValueError("service deploy")
                if deploy.get("placement") not in (None, (), {}):
                    raise ValueError("service placement")
                resources = deploy.get("resources")
                if resources is not None:
                    if not isinstance(resources, dict) or set(resources) - {
                        "limits",
                        "reservations",
                    }:
                        raise ValueError("service resources")
                    if not isinstance(resources.get("limits", {}), dict):
                        raise ValueError("service limits")
                    reservations = resources.get("reservations", {})
                    if not isinstance(reservations, dict) or set(reservations) - {
                        "devices"
                    }:
                        raise ValueError("service reservations")

        api_service = services["train-factory-api"]
        if api_service.get("shm_size") != _normalized_compose_size(
            control_values["API_SHM_SIZE"]
        ):
            raise ValueError("service shm")
        expected_limits: dict[str, object] = {}
        if control_values["API_CPU_LIMIT"] not in {"", "0"}:
            expected_limits["cpus"] = float(control_values["API_CPU_LIMIT"])
        if control_values["API_MEM_LIMIT"] not in {"", "0"}:
            expected_limits["memory"] = _normalized_compose_size(
                control_values["API_MEM_LIMIT"]
            )
        if control_values["API_PIDS_LIMIT"] not in {"", "0"}:
            expected_limits["pids"] = int(control_values["API_PIDS_LIMIT"])
        expected_devices = (
            []
            if manifest["gpu_mode"] == "cpu"
            else [
                {
                    "capabilities": ["gpu"],
                    "count": -1,
                    "driver": "nvidia",
                }
            ]
        )
        api_resources = api_service.get("deploy", {}).get("resources", {})
        if (
            not isinstance(api_resources, dict)
            or api_resources.get("limits", {}) != expected_limits
            or api_resources.get("reservations", {}).get("devices", [])
            != expected_devices
        ):
            raise ValueError("service resources")

        expected_secret_names = {
            "mysql_root_password",
            "mysql_app_password",
            "mysql_url",
            "jwt_secret_key",
            "default_admin_password",
        }
        if manifest["secret_mode"] == "files":
            referenced = {
                str(item["role"]).removeprefix("secret:"): str(item["path"])
                for item in manifest["referenced_files"]
                if str(item["role"]).startswith("secret:")
                and str(item["role"]).removeprefix("secret:") in expected_secret_names
            }
            secrets = compose.get("secrets")
            if (
                set(referenced) != expected_secret_names
                or not isinstance(secrets, dict)
                or set(secrets) != expected_secret_names
            ):
                raise ValueError("secret definitions")
            for name, definition in secrets.items():
                if not isinstance(definition, dict):
                    raise ValueError("secret definition")
                secret_file = definition.get("file")
                if (
                    set(definition) - {"file", "name"}
                    or not isinstance(secret_file, str)
                    or not os.path.isabs(secret_file)
                    or os.path.normcase(secret_file)
                    != os.path.normcase(referenced[name])
                    or definition.get("name")
                    not in {None, f"{manifest['project']}_{name}"}
                ):
                    raise ValueError("secret definition")
            expected_service_secrets = {
                "mysql": [
                    {"source": "mysql_root_password", "target": "mysql_root_password"},
                    {"source": "mysql_app_password", "target": "mysql_app_password"},
                ],
                "train-factory-api": [
                    {"source": "mysql_url", "target": "mysql_url"},
                    {"source": "jwt_secret_key", "target": "jwt_secret_key"},
                    {
                        "source": "default_admin_password",
                        "target": "default_admin_password",
                    },
                ],
                "train-factory-web": [],
            }
            if any(
                services[name].get("secrets", []) != expected
                for name, expected in expected_service_secrets.items()
            ):
                raise ValueError("service secrets")
        elif compose.get("secrets") not in (None, (), {}):
            raise ValueError("unexpected secrets")
        forbidden = {
            "ipc",
            "network_mode",
            "privileged",
            "runtime",
            "gpus",
            "devices",
            "device_requests",
            "device_cgroup_rules",
            "cap_add",
            "security_opt",
            "read_only",
        }
        for service in services.values():
            if not isinstance(service, dict):
                raise ValueError("service")
            if service.get("build") not in {None, False}:
                raise ValueError("build")
            if any(service.get(key) not in {None, False, "", ()} for key in forbidden):
                raise ValueError("surface")
        mode = manifest["mode"]
        if mode in {"verify", "ci", "rollback-verify"}:
            if any(
                service.get("pid") not in {None, ""}
                or service.get("container_name") not in {None, ""}
                or service.get("restart") not in {None, "no"}
                or (
                    isinstance(service.get("deploy"), dict)
                    and service["deploy"].get("restart_policy") not in (None, {})
                )
                for service in services.values()
            ):
                raise ValueError("random surface")
            expected_volume_keys = {
                "mysql_data",
                "train_cache",
                "verify_data",
                "verify_models",
                "verify_output",
            }
            volumes = compose.get("volumes")
            if not isinstance(volumes, dict) or set(volumes) != expected_volume_keys:
                raise ValueError("volumes")
            if any(
                not isinstance(definition, dict) or set(definition) - {"name", "labels"}
                for definition in volumes.values()
            ):
                raise ValueError("volume definitions")
            if len(_expected_volume_names(compose, str(manifest["project"]))) != 5:
                raise ValueError("volume names")
            _expected_network_names(compose, str(manifest["project"]))
            expected_mounts = {
                "mysql": {
                    ("volume", "mysql_data", "/var/lib/mysql", False),
                    (
                        "bind",
                        os.path.normcase(os.path.abspath(root / "docker" / "init.sql")),
                        "/docker-entrypoint-initdb.d/init.sql",
                        True,
                    ),
                },
                "train-factory-api": {
                    ("volume", "verify_data", "/app/data", False),
                    ("volume", "verify_models", "/app/models", False),
                    ("volume", "verify_output", "/app/output", False),
                    ("volume", "train_cache", "/app/cache", False),
                },
                "train-factory-web": set(),
            }
            for service_name, expected in expected_mounts.items():
                if normalized_mounts(services[service_name]) != expected:
                    raise ValueError("mount policy")
            for service_name in ("train-factory-api", "train-factory-web"):
                ports = services[service_name].get("ports")
                if not isinstance(ports, list) or len(ports) != 1:
                    raise ValueError("ports")
                port = ports[0]
                if (
                    not isinstance(port, dict)
                    or set(port)
                    != {"host_ip", "mode", "protocol", "published", "target"}
                    or port.get("host_ip") != "127.0.0.1"
                    or str(port.get("published")) != "0"
                    or port.get("target")
                    != (18000 if service_name == "train-factory-api" else 80)
                    or port.get("protocol", "tcp") != "tcp"
                    or port.get("mode", "ingress") != "ingress"
                ):
                    raise ValueError("port")
            if services["mysql"].get("ports") not in (None, (), ""):
                mysql_ports = services["mysql"].get("ports")
                if mysql_ports != []:
                    raise ValueError("mysql ports")
            from scripts.validate_compose_config import _validate_gpu_preflight_policy

            gpu_errors = _validate_gpu_preflight_policy(
                services["train-factory-api"],
                "train-factory-api",
            )
            if gpu_errors:
                raise ValueError("gpu")
        else:
            from scripts.compose_manifest import _manifest_environment_values

            base_values = _manifest_environment_values(manifest, "base-environment")
            api_port_text = base_values.get("API_PORT", "18000")
            web_port_text = base_values.get("WEB_PORT", "3000")
            bind_address = base_values.get("HOST_BIND_ADDRESS", "127.0.0.1")
            from scripts.validate_compose_config import _deployment_bind_address

            try:
                _deployment_bind_address(bind_address)
            except ValueError:
                raise ValueError("production bind address") from None
            if (
                not api_port_text.isdigit()
                or not web_port_text.isdigit()
                or not 1 <= int(api_port_text) <= 65535
                or not 1 <= int(web_port_text) <= 65535
                or not bind_address
            ):
                raise ValueError("production ports")
            expected_published = {
                "train-factory-api": api_port_text,
                "train-factory-web": web_port_text,
            }
            expected_targets = {
                "train-factory-api": int(api_port_text),
                "train-factory-web": 80,
            }
            expected_volume_names = {
                "mysql_data": base_values.get(
                    "MYSQL_DATA_VOLUME", "trainfactory_mysql_data"
                ),
                "train_cache": base_values.get(
                    "TRAIN_CACHE_VOLUME", "trainfactory_train_cache"
                ),
            }
            expected_network_name = base_values.get(
                "DOCKER_NETWORK_NAME", "trainfactory_network"
            )
            expected_ipam = {
                "config": [
                    {
                        "subnet": base_values.get(
                            "DOCKER_NETWORK_SUBNET", "172.18.0.0/16"
                        ),
                        "ip_range": base_values.get(
                            "DOCKER_DYNAMIC_IP_RANGE", "172.18.128.0/17"
                        ),
                    }
                ]
            }
            volumes = compose.get("volumes")
            if not isinstance(volumes, dict) or set(volumes) != {
                "mysql_data",
                "train_cache",
            }:
                raise ValueError("production volumes")
            if any(
                not isinstance(definition, dict)
                or set(definition) != {"name"}
                or not isinstance(definition.get("name"), str)
                or not definition["name"]
                or definition["name"] != expected_volume_names[key]
                for key, definition in volumes.items()
            ):
                raise ValueError("production volume definitions")
            networks = compose.get("networks")
            if not isinstance(networks, dict) or set(networks) != {"default"}:
                raise ValueError("production networks")
            network = networks["default"]
            if (
                not isinstance(network, dict)
                or set(network) - {"name", "labels", "ipam"}
                or not isinstance(network.get("name"), str)
                or network["name"] != expected_network_name
                or network.get("ipam") != expected_ipam
            ):
                raise ValueError("production network definition")
            expected_network_attachments = {
                "mysql": {"default": None},
                "train-factory-api": {"default": None},
                "train-factory-web": {
                    "default": {
                        "ipv4_address": base_values.get("WEB_PROXY_IPV4", "172.18.0.4")
                    }
                },
            }
            if any(
                services[name].get("networks") != expected
                for name, expected in expected_network_attachments.items()
            ):
                raise ValueError("production network attachments")
            expected_container_names = {
                "mysql": f"{manifest['project']}-mysql",
                "train-factory-api": f"{manifest['project']}-api",
                "train-factory-web": f"{manifest['project']}-web",
            }
            for service_name, service in services.items():
                if (
                    service.get("container_name")
                    != expected_container_names[service_name]
                ):
                    raise ValueError("container name")
                if service.get("restart") != "unless-stopped":
                    raise ValueError("restart")
                if service_name == "train-factory-api":
                    if service.get("pid") != "host":
                        raise ValueError("pid")
                elif service.get("pid") not in {None, ""}:
                    raise ValueError("pid")
            expected_production_mounts = {
                "mysql": {
                    ("volume", "mysql_data", "/var/lib/mysql", False),
                    (
                        "bind",
                        os.path.normcase(os.path.abspath(root / "docker" / "init.sql")),
                        "/docker-entrypoint-initdb.d/init.sql",
                        True,
                    ),
                },
                "train-factory-api": {
                    (
                        "bind",
                        os.path.normcase(os.path.abspath(root / "data")),
                        "/app/data",
                        False,
                    ),
                    (
                        "bind",
                        os.path.normcase(os.path.abspath(root / "models")),
                        "/app/models",
                        False,
                    ),
                    (
                        "bind",
                        os.path.normcase(os.path.abspath(root / "output")),
                        "/app/output",
                        False,
                    ),
                    ("volume", "train_cache", "/app/cache", False),
                },
                "train-factory-web": set(),
            }
            for service_name, expected in expected_production_mounts.items():
                if normalized_mounts(services[service_name]) != expected:
                    raise ValueError("production mounts")
            if services["mysql"].get("ports") not in (None, (), "", []):
                raise ValueError("mysql ports")
            for service_name in ("train-factory-api", "train-factory-web"):
                ports = services[service_name].get("ports")
                if not isinstance(ports, list) or len(ports) != 1:
                    raise ValueError("ports")
                port = ports[0]
                if (
                    not isinstance(port, dict)
                    or set(port)
                    != {"host_ip", "mode", "protocol", "published", "target"}
                    or port.get("host_ip") != bind_address
                    or str(port.get("published", ""))
                    != expected_published[service_name]
                    or port.get("target") != expected_targets[service_name]
                    or port.get("protocol") != "tcp"
                    or port.get("mode") != "ingress"
                ):
                    raise ValueError("production port")
            from scripts.validate_compose_config import _validate_gpu_preflight_policy

            if _validate_gpu_preflight_policy(
                services["train-factory-api"],
                "train-factory-api",
            ):
                raise ValueError("gpu")
        published_endpoints: set[tuple[str, str, str]] = set()
        for service in services.values():
            ports = service.get("ports", [])
            if not isinstance(ports, list):
                raise ValueError("ports")
            for port in ports:
                if not isinstance(port, dict):
                    raise ValueError("port")
                endpoint = (
                    str(port.get("host_ip", "")),
                    str(port.get("published", "")),
                    str(port.get("protocol", "")),
                )
                if endpoint[1] != "0" and endpoint in published_endpoints:
                    raise ValueError("duplicate port")
                if endpoint[1] != "0":
                    published_endpoints.add(endpoint)
        from scripts.validate_compose_config import validate_compose_config

        if validate_compose_config(compose, profile="minimal"):
            raise ValueError("compose policy")
    except (KeyError, ManifestError, TypeError, ValueError):
        raise ReleaseComposeError(
            "release Compose resolved config is invalid"
        ) from None


def execute_safe_down_volumes(
    manifest_path: Path,
    *,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> None:
    """Remove volumes only for a frozen random verification project."""
    try:
        manifest = verify_manifest_inputs(manifest_path, root=root)
        if manifest["mode"] not in {"verify", "ci", "rollback-verify"}:
            raise ReleaseComposeError("release Compose cleanup is not permitted")
        project = str(manifest["project"])
        environment = _clean_environment(base_environment)
        configured = _run_private(
            run,
            [*_compose_prefix(manifest, root=root), "config", "--format", "json"],
            environment,
        )
        try:
            compose = json.loads(configured.stdout)
            services = compose["services"]
            volumes = compose["volumes"]
            if not isinstance(services, dict) or not isinstance(volumes, dict):
                raise ValueError("shape")
            if not all(isinstance(service, dict) for service in services.values()):
                raise ValueError("service")
            used_keys = set()
            for service in services.values():
                mounts = service.get("volumes", [])
                if not isinstance(mounts, list):
                    raise ValueError("mounts")
                for mount in mounts:
                    if not isinstance(mount, dict):
                        raise ValueError("mount")
                    if mount.get("type") == "volume":
                        source = mount.get("source")
                        if not isinstance(source, str):
                            raise ValueError("source")
                        used_keys.add(source)
            expected_names = set()
            for key in used_keys:
                definition = volumes[key]
                if not isinstance(definition, dict) or definition.get("external"):
                    raise ValueError("volume")
                expected_name = f"{project}_{key}"
                if definition.get("name") != expected_name:
                    raise ValueError("name")
                expected_names.add(expected_name)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise ReleaseComposeError(
                "release Compose cleanup is not permitted"
            ) from None
        if not expected_names:
            raise ReleaseComposeError("release Compose cleanup is not permitted")
        expected_networks = _expected_network_names(compose, project)
        list_argv = [
            "docker",
            "volume",
            "ls",
            "--format",
            "{{.Name}}",
        ]
        listed = _run_private(run, list_argv, environment)
        names = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        if len(names) != len(set(names)):
            raise ReleaseComposeError("release Compose cleanup is not permitted")
        present = expected_names.intersection(names)
        if any(
            name.startswith(project + "_") and name not in expected_names
            for name in names
        ):
            raise ReleaseComposeError("release Compose cleanup is not permitted")
        labeled_resources = _list_project_resources(run, environment, project)
        labeled_volumes = set(labeled_resources["volume"])
        if labeled_volumes != present:
            raise ReleaseComposeError("release Compose cleanup is not permitted")
        network_list_argv = [
            "docker",
            "network",
            "ls",
            "--format",
            "{{.Name}}",
        ]
        network_names = set(
            _parse_resource_listing(
                _run_private(run, network_list_argv, environment).stdout
            )
        )
        present_networks = expected_networks.intersection(network_names)
        if set(labeled_resources["network"]) != present_networks:
            raise ReleaseComposeError("release Compose cleanup is not permitted")
        for name in sorted(present):
            inspected = _run_private(
                run,
                [
                    "docker",
                    "volume",
                    "inspect",
                    "--format",
                    "{{json .Labels}}",
                    name,
                ],
                environment,
            )
            try:
                labels = json.loads(inspected.stdout)
                volume_key = name.removeprefix(project + "_")
            except (json.JSONDecodeError, TypeError):
                raise ReleaseComposeError(
                    "release Compose cleanup is not permitted"
                ) from None
            if (
                not isinstance(labels, dict)
                or labels.get("com.docker.compose.project") != project
                or labels.get("com.docker.compose.volume") != volume_key
            ):
                raise ReleaseComposeError("release Compose cleanup is not permitted")
        for name in sorted(present_networks):
            inspected = _run_private(
                run,
                [
                    "docker",
                    "network",
                    "inspect",
                    "--format",
                    "{{json .Labels}}",
                    name,
                ],
                environment,
            )
            try:
                labels = json.loads(inspected.stdout)
            except (json.JSONDecodeError, TypeError):
                raise ReleaseComposeError(
                    "release Compose cleanup is not permitted"
                ) from None
            if (
                not isinstance(labels, dict)
                or labels.get("com.docker.compose.project") != project
                or labels.get("com.docker.compose.network") != "default"
            ):
                raise ReleaseComposeError("release Compose cleanup is not permitted")
        _run_private(
            run,
            [*_compose_prefix(manifest, root=root), "down", "--volumes"],
            environment,
        )
        remaining = _run_private(run, list_argv, environment)
        remaining_names = {
            line.strip() for line in remaining.stdout.splitlines() if line.strip()
        }
        if expected_names.intersection(remaining_names):
            raise ReleaseComposeError("release Compose cleanup failed")
        remaining_networks = set(
            _parse_resource_listing(
                _run_private(run, network_list_argv, environment).stdout
            )
        )
        if expected_networks.intersection(remaining_networks):
            raise ReleaseComposeError("release Compose cleanup failed")
        _require_empty_project_resources(run, environment, project)
    except ReleaseComposeError:
        raise
    except (ManifestError, TypeError, ValueError):
        raise ReleaseComposeError("release Compose cleanup failed") from None


def _load_stop_state(path: Path, *, root: Path) -> dict[str, object]:
    runtime = Path(os.path.abspath(root / ".runtime"))
    lexical = Path(os.path.abspath(path))
    try:
        if (
            os.path.commonpath((runtime, lexical)) != os.fspath(runtime)
            or lexical == runtime
            or lexical.is_symlink()
        ):
            raise ValueError("path")
        _validate_ancestor_chain(lexical, boundary=runtime)
        if not _verify_hardened_path(lexical):
            raise ValueError("state")
        payload, _identity = _read_stable(lexical, max_bytes=64 * 1024)

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate")
                result[key] = value
            return result

        state = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, ManifestError):
        raise ReleaseComposeError("release Compose stop state is invalid") from None
    if not isinstance(state, dict) or set(state) != {
        "version",
        "project",
        "service",
        "container_id",
    }:
        raise ReleaseComposeError("release Compose stop state is invalid")
    if (
        state["version"] != 1
        or type(state["version"]) is not int
        or type(state["project"]) is not str
        or state["service"] not in {"train-factory-api", "train-factory-web"}
        or type(state["container_id"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", state["container_id"], re.ASCII) is None
    ):
        raise ReleaseComposeError("release Compose stop state is invalid")
    return state


def execute_graceful_stop_no_kill(
    manifest_path: Path,
    *,
    service: str,
    state_path: Path,
    wait_seconds: int,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Send one SIGTERM and poll without ever escalating to SIGKILL."""
    try:
        manifest = verify_manifest_inputs(manifest_path, root=root)
        state = _load_stop_state(state_path, root=root)
        if (
            service != state["service"]
            or manifest["project"] != state["project"]
            or not 1 <= wait_seconds <= 3600
        ):
            raise ReleaseComposeError("release Compose stop state is invalid")
        environment = _clean_environment(base_environment)
        selected = _run_private(
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
        ids = [line.strip() for line in selected.stdout.splitlines() if line.strip()]
        if ids != [state["container_id"]]:
            raise ReleaseComposeError("release Compose stop state is invalid")
        container_id = str(state["container_id"])

        def running_status(
            *,
            timeout: float = 600,
            timeout_message: str = "release Compose command failed",
        ) -> bool:
            inspect_format = (
                "{{.Id}}|{{.State.Running}}|{{ index .Config.Labels "
                '"com.docker.compose.project" }}|{{ index .Config.Labels '
                '"com.docker.compose.service" }}'
            )
            inspected = _run_private(
                run,
                ["docker", "inspect", "--format", inspect_format, container_id],
                environment,
                timeout=timeout,
                timeout_message=timeout_message,
            )
            parts = inspected.stdout.strip().split("|")
            if len(parts) != 4 or parts[1] not in {"true", "false"}:
                raise ReleaseComposeError(
                    "release Compose stop state is invalid"
                ) from None
            if (
                parts[0] != container_id
                or parts[2] != manifest["project"]
                or parts[3] != service
            ):
                raise ReleaseComposeError("release Compose stop state is invalid")
            return parts[1] == "true"

        if not running_status():
            raise ReleaseComposeError("release Compose stop state is invalid")
        _run_private(
            run,
            ["docker", "kill", "--signal", "SIGTERM", container_id],
            environment,
        )
        deadline = monotonic() + wait_seconds
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ReleaseComposeError("release Compose graceful stop timed out")
            if not running_status(
                timeout=max(0.05, min(5.0, remaining)),
                timeout_message="release Compose graceful stop timed out",
            ):
                return
            sleep(min(1.0, remaining))
    except ReleaseComposeError:
        raise
    except (ManifestError, TypeError, ValueError):
        raise ReleaseComposeError("release Compose graceful stop failed") from None


class _PrivateParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ReleaseComposeError("release Compose arguments are invalid")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _PrivateParser(allow_abbrev=False)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--safe-down-volumes", action="store_true")
    parser.add_argument("--graceful-stop-no-kill")
    parser.add_argument("--stop-state", type=Path)
    parser.add_argument("--wait-seconds", type=int)
    parser.add_argument("tail", nargs=argparse.REMAINDER)
    try:
        arguments = parser.parse_args(argv)
        tail = arguments.tail
        if tail[:1] == ["--"]:
            tail = tail[1:]
        special_count = int(arguments.safe_down_volumes) + int(
            arguments.graceful_stop_no_kill is not None
        )
        if special_count > 1:
            raise ReleaseComposeError("release Compose arguments are invalid")
        if arguments.safe_down_volumes:
            if (
                tail
                or arguments.stop_state is not None
                or arguments.wait_seconds is not None
            ):
                raise ReleaseComposeError("release Compose arguments are invalid")
            execute_safe_down_volumes(arguments.manifest)
        elif arguments.graceful_stop_no_kill is not None:
            if tail or arguments.stop_state is None or arguments.wait_seconds is None:
                raise ReleaseComposeError("release Compose arguments are invalid")
            execute_graceful_stop_no_kill(
                arguments.manifest,
                service=arguments.graceful_stop_no_kill,
                state_path=arguments.stop_state,
                wait_seconds=arguments.wait_seconds,
            )
        else:
            if arguments.stop_state is not None or arguments.wait_seconds is not None:
                raise ReleaseComposeError("release Compose arguments are invalid")
            execute_manifest(arguments.manifest, tail)
    except Exception:
        print("release Compose command failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
