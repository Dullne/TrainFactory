"""Fail-closed verification for a frozen Train Factory deployment."""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("deployment verifier bootstrap is unavailable")


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
import ast  # noqa: E402
import http.cookiejar  # noqa: E402
import ipaddress  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
import urllib.parse  # noqa: E402
from collections.abc import Callable, Mapping  # noqa: E402
from pathlib import Path  # noqa: E402
from subprocess import CompletedProcess  # noqa: E402

from scripts.materialize_compose_secrets import _verify_hardened_path  # noqa: E402
from scripts.compose_manifest import (  # noqa: E402
    ROOT_DIR,
    ManifestError,
    _read_stable,
    _selection_values,
    _validate_ancestor_chain,
)


class VerificationError(RuntimeError):
    """A fixed-message deployment verification failure."""


_SHARED_REVISION_SELECTION_KEYS = frozenset(
    {
        "API_IMAGE",
        "WEB_IMAGE",
        "RELEASE_REVISION",
        "API_IMAGE_ID",
        "WEB_IMAGE_ID",
    }
)
_SPLIT_REVISION_SELECTION_KEYS = frozenset(
    {
        "API_IMAGE",
        "WEB_IMAGE",
        "API_REVISION",
        "WEB_REVISION",
        "API_IMAGE_ID",
        "WEB_IMAGE_ID",
    }
)


def _service_revisions(selection: Mapping[str, str]) -> tuple[str, str, bool]:
    keys = frozenset(selection)
    if keys == _SHARED_REVISION_SELECTION_KEYS:
        revision = selection["RELEASE_REVISION"]
        return revision, revision, False
    if keys == _SPLIT_REVISION_SELECTION_KEYS:
        return selection["API_REVISION"], selection["WEB_REVISION"], True
    raise VerificationError("deployment verification failed")


_DB_HEAD_SCRIPT = """import json
import os
import stat
from urllib.parse import quote
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

direct = os.environ.get("MYSQL_URL")
file_path = os.environ.get("MYSQL_URL_FILE")
if direct is not None and file_path is not None:
    raise RuntimeError("database URL source is invalid")
if direct is None and file_path is None:
    components = {
        name: os.environ.get(name)
        for name in (
            "MYSQL_APP_USER",
            "MYSQL_APP_PASSWORD",
            "MYSQL_HOST",
            "MYSQL_DATABASE",
        )
    }
    if any(value is None or value == "" for value in components.values()):
        raise RuntimeError("database URL source is invalid")
    url = (
        "mysql+pymysql://"
        + quote(components["MYSQL_APP_USER"], safe="")
        + ":"
        + quote(components["MYSQL_APP_PASSWORD"], safe="")
        + "@"
        + quote(components["MYSQL_HOST"], safe="[].:")
        + ":3306/"
        + quote(components["MYSQL_DATABASE"], safe="")
    )
elif direct is not None:
    url = direct
else:
    before = os.lstat(file_path)
    if not stat.S_ISREG(before.st_mode) or before.st_size > 65536:
        raise RuntimeError("database URL file is invalid")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(file_path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
            or opened.st_size > 65536
        ):
            raise RuntimeError("database URL file changed")
        chunks = []
        remaining = 65537
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(file_path)
    content = b"".join(chunks)
    if (
        not stat.S_ISREG(after_open.st_mode)
        or not stat.S_ISREG(after_path.st_mode)
        or stat.S_ISLNK(after_path.st_mode)
        or (after_open.st_dev, after_open.st_ino) != (before.st_dev, before.st_ino)
        or (after_path.st_dev, after_path.st_ino) != (before.st_dev, before.st_ino)
        or after_open.st_size != before.st_size
        or after_path.st_size != before.st_size
        or len(content) != before.st_size
        or len(content) > 65536
    ):
        raise RuntimeError("database URL file changed")
    url = content.decode("utf-8")
    if url.endswith("\\r\\n"):
        url = url[:-2]
    elif url.endswith("\\n"):
        url = url[:-1]
if not url or "\\x00" in url:
    raise RuntimeError("database URL is invalid")
engine = create_engine(url, poolclass=NullPool)
connection = engine.connect()
try:
    rows = connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
finally:
    connection.close()
    engine.dispose()
print(json.dumps(rows, separators=(",", ":")))
"""
_GPU_PROBE_SCRIPT = r"""import contextlib
import importlib
import io
import json
import platform
import sys
import warnings
from pathlib import Path


def result(mode, ok, compiled=None, driver=None, count=0, executed=False, reason=None):
    return {
        "mode": mode,
        "ok": ok,
        "compiled_cuda": compiled,
        "driver_version": driver,
        "device_count": count,
        "probe_executed": executed,
        "reason_code": reason,
    }


def numeric(value, parts):
    if not isinstance(value, str):
        return None
    values = value.split(".")
    if len(values) != parts or any(
        not item or len(item) > 6 or not item.isascii() or not item.isdecimal()
        for item in values
    ):
        return None
    return tuple(int(item) for item in values)


def failure(reason, compiled=None, driver=None, count=0, executed=False):
    return result("required", False, compiled, driver, count, executed, reason)


def required_probe():
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return failure("torch_import_failed")
    try:
        compiled = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        return failure("compiled_cuda_invalid")
    cuda = numeric(compiled, 2)
    if cuda is None:
        return failure("compiled_cuda_invalid")
    try:
        nvml = importlib.import_module("pynvml")
    except Exception:
        return failure("nvml_import_failed", compiled)
    initialized = False
    driver = None
    count = 0
    executed = False
    answer = None
    try:
        try:
            nvml.nvmlInit()
            initialized = True
        except Exception:
            return failure("nvml_init_failed", compiled)
        try:
            driver = nvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("ascii")
        except Exception:
            return failure("driver_query_failed", compiled)
        system_name = platform.system()
        if system_name == "Windows":
            platform_kind = "windows"
        elif system_name == "Linux":
            try:
                osrelease = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
            except Exception:
                return failure("platform_unsupported", compiled)
            platform_kind = "wsl" if "microsoft" in osrelease.lower() else "linux"
        else:
            return failure("platform_unsupported", compiled)
        parsed_driver = numeric(driver, 3 if platform_kind == "linux" else 2)
        if parsed_driver is None:
            return failure("driver_version_invalid", compiled)
        if cuda[0] == 12:
            minimum = (
                (525, 60, 13) if platform_kind == "linux" else (528, 33)
            )
        elif cuda[0] == 13:
            if platform_kind != "linux":
                return failure("platform_unsupported", compiled, driver)
            minimum = (580, 65, 6)
        else:
            return failure("cuda_unsupported", compiled, driver)
        if parsed_driver < minimum:
            return failure("driver_incompatible", compiled, driver)
        try:
            available = torch.cuda.is_available()
        except Exception:
            return failure("cuda_availability_failed", compiled, driver)
        if available is not True:
            return failure("cuda_unavailable", compiled, driver)
        try:
            count = torch.cuda.device_count()
        except Exception:
            return failure("device_count_failed", compiled, driver)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return failure("device_count_invalid", compiled, driver)
        if count == 0:
            return failure("no_devices", compiled, driver)
        try:
            tensor = torch.ones(1, device="cuda:0")
            executed = True
        except Exception:
            return failure("tensor_allocation_failed", compiled, driver, count, True)
        try:
            tensor.add(1)
        except Exception:
            return failure("tensor_operation_failed", compiled, driver, count, True)
        try:
            torch.cuda.synchronize(0)
        except Exception:
            return failure("synchronize_failed", compiled, driver, count, True)
        answer = result("required", True, compiled, driver, count, True, None)
    finally:
        if initialized:
            try:
                nvml.nvmlShutdown()
            except Exception:
                answer = failure("nvml_shutdown_failed", compiled, driver, count, executed)
    return answer


arguments = sys.argv[1:]
if arguments == ["off"]:
    answer = result("off", True)
elif arguments == ["required"]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            answer = required_probe()
else:
    answer = result("required", False, reason="arguments_invalid")
print(json.dumps(answer, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if answer["ok"] else 1)
"""
_USERNAME_MAX_BYTES = 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        return None


class _PrivateHttpClient:
    def __init__(self, *, allowed_hosts: set[str] | None = None) -> None:
        self._bearer: str | None = None
        self._allowed_hosts = allowed_hosts or {"127.0.0.1", "::1"}
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
            _NoRedirect(),
        )

    def set_bearer(self, token: str) -> None:
        if (
            not token
            or len(token.encode("utf-8")) > 65536
            or any(character in token for character in "\x00\r\n")
        ):
            raise VerificationError("deployment verification failed")
        self._bearer = token

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
    ) -> tuple[int, bytes]:
        parsed = urllib.parse.urlsplit(url)
        try:
            address = ipaddress.ip_address(parsed.hostname or "")
        except ValueError:
            raise VerificationError("deployment verification failed") from None
        if (
            parsed.scheme != "http"
            or address.compressed not in self._allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.port is None
        ):
            raise VerificationError("deployment verification failed")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._bearer is not None:
            headers["Authorization"] = f"Bearer {self._bearer}"
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=15) as response:
                payload = response.read(65537)
                if len(payload) > 65536:
                    raise VerificationError("deployment verification failed")
                return int(response.status), payload
        except (OSError, urllib.error.URLError, ValueError):
            raise VerificationError("deployment verification failed") from None


def _parse_json(payload: bytes, *, expected_keys: set[str]) -> dict[str, object]:
    def closed_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite number")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise VerificationError("deployment verification failed") from None
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise VerificationError("deployment verification failed")
    return value


def _private_file_snapshot(
    path: Path, *, root: Path, max_bytes: int
) -> tuple[str, tuple[int, int, int, int]]:
    runtime = Path(os.path.abspath(root / ".runtime"))
    lexical = Path(os.path.abspath(path))
    try:
        if os.path.normcase(os.path.commonpath((lexical, runtime))) != os.path.normcase(
            os.fspath(runtime)
        ):
            raise ValueError("boundary")
        _validate_ancestor_chain(lexical, boundary=runtime)
        metadata = lexical.lstat()
        if not stat.S_ISREG(metadata.st_mode) or lexical.is_symlink():
            raise ValueError("file")
        if not _verify_hardened_path(lexical):
            raise ValueError("acl")
        payload, identity = _read_stable(lexical, max_bytes=max_bytes)
        if identity != (metadata.st_dev, metadata.st_ino):
            raise ValueError("identity")
        value = payload.decode("utf-8")
        if value.endswith("\r\n"):
            value = value[:-2]
        elif value.endswith("\n"):
            value = value[:-1]
        if not value or "\x00" in value or "\r" in value or "\n" in value:
            raise ValueError("value")
        final_metadata = lexical.lstat()
        final_identity = (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
        )
        if final_identity[:2] != identity:
            raise ValueError("identity")
        if not _verify_hardened_path(lexical):
            raise ValueError("acl")
        return value, final_identity
    except (ManifestError, OSError, UnicodeError, ValueError):
        raise VerificationError("deployment verification failed") from None


def _private_file(path: Path, *, root: Path, max_bytes: int) -> str:
    return _private_file_snapshot(path, root=root, max_bytes=max_bytes)[0]


def _repository_head(root: Path) -> str:
    try:
        versions = Path(
            os.path.abspath(
                root / "train_factory" / "storage" / "migrations" / "versions"
            )
        )
        if not versions.is_dir() or versions.is_symlink():
            raise ValueError("migration directory")
        graph: dict[str, tuple[str, ...]] = {}
        for source in sorted(versions.glob("*.py")):
            if source.is_symlink() or not source.is_file():
                raise ValueError("migration source")
            tree = ast.parse(source.read_bytes(), filename=os.fspath(source))
            assignments: dict[str, object] = {}
            for node in tree.body:
                name: str | None = None
                value_node: ast.expr | None = None
                if isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    name, value_node = node.target.id, node.value
                elif (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    name, value_node = node.targets[0].id, node.value
                if name in {"revision", "down_revision"} and value_node is not None:
                    if name in assignments:
                        raise ValueError("duplicate migration metadata")
                    assignments[name] = ast.literal_eval(value_node)
            revision = assignments.get("revision")
            down_revision = assignments.get("down_revision")
            if (
                not isinstance(revision, str)
                or re.fullmatch(r"[0-9a-z_]{1,128}", revision, re.ASCII) is None
                or revision in graph
            ):
                raise ValueError("migration revision")
            if down_revision is None:
                parents: tuple[str, ...] = ()
            elif isinstance(down_revision, str):
                parents = (down_revision,)
            elif isinstance(down_revision, tuple) and all(
                isinstance(parent, str) for parent in down_revision
            ):
                parents = down_revision
            else:
                raise ValueError("migration parent")
            graph[revision] = parents
        if not graph or any(
            parent not in graph for parents in graph.values() for parent in parents
        ):
            raise ValueError("migration graph")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(revision: str) -> None:
            if revision in visiting:
                raise ValueError("migration cycle")
            if revision in visited:
                return
            visiting.add(revision)
            for parent in graph[revision]:
                visit(parent)
            visiting.remove(revision)
            visited.add(revision)

        for revision in graph:
            visit(revision)
        parents = {parent for value in graph.values() for parent in value}
        heads = set(graph) - parents
        if len(heads) != 1:
            raise ValueError("migration heads")
        return next(iter(heads))
    except Exception:
        raise VerificationError("deployment verification failed") from None


def _effective_runtime_endpoint(manifest: Mapping[str, object]) -> tuple[int, str]:
    from scripts import compose_manifest as manifest_module

    value = "18000"
    bind_address = "127.0.0.1"
    records = manifest.get("env_files")
    if not isinstance(records, list):
        raise VerificationError("deployment verification failed")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("role"), str):
            raise VerificationError("deployment verification failed")
        role = record["role"]
        if role in {
            "base-environment",
            "rollback-direct",
            "production-direct",
        }:
            values = manifest_module._manifest_environment_values(manifest, role)
            if "API_PORT" in values:
                value = values["API_PORT"]
            if "HOST_BIND_ADDRESS" in values:
                bind_address = values["HOST_BIND_ADDRESS"]
    try:
        parsed_bind = ipaddress.ip_address(bind_address)
        if isinstance(parsed_bind, ipaddress.IPv6Address) and parsed_bind.ipv4_mapped:
            raise ValueError("mapped address")
        normalized_bind = parsed_bind.compressed
    except ValueError:
        raise VerificationError("deployment verification failed") from None
    if not value.isdigit() or not 1 <= int(value) <= 65535:
        raise VerificationError("deployment verification failed")
    return int(value), normalized_bind


def _port(
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
    container_id: str,
    container_port: int,
    expected_host: str = "127.0.0.1",
) -> str:
    from scripts.compose_release import _run_private

    completed = _run_private(
        run,
        ["docker", "port", container_id, f"{container_port}/tcp"],
        environment,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise VerificationError("deployment verification failed")
    match = re.fullmatch(r"(?:([^\[\]:]+)|\[([^\]]+)\]):([1-9][0-9]{0,4})", lines[0])
    if match is None or int(match.group(3)) > 65535:
        raise VerificationError("deployment verification failed")
    observed_text = match.group(1) or match.group(2)
    try:
        observed_address = ipaddress.ip_address(observed_text)
        expected_address = ipaddress.ip_address(expected_host)
        if any(
            isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped
            for address in (observed_address, expected_address)
        ):
            raise ValueError("mapped address")
        observed = observed_address.compressed
        expected = expected_address.compressed
    except ValueError:
        raise VerificationError("deployment verification failed") from None
    if observed != expected:
        raise VerificationError("deployment verification failed")
    probe = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(observed, observed)
    host = f"[{probe}]" if ":" in probe else probe
    return f"http://{host}:{match.group(3)}"


def _expected_image_ids(
    manifest: Mapping[str, object],
    selection: Mapping[str, str],
    *,
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
) -> dict[str, str]:
    from scripts import compose_manifest as manifest_module
    from scripts.compose_release import _run_private

    images_lock = manifest_module._manifest_environment_values(manifest, "images-lock")
    mysql_ref = images_lock.get("MYSQL_IMAGE")
    if not mysql_ref:
        raise VerificationError("deployment verification failed")
    inspected = _run_private(
        run,
        ["docker", "image", "inspect", "--format", "{{.Id}}", mysql_ref],
        environment,
    )
    mysql_id = inspected.stdout.strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", mysql_id, re.ASCII) is None:
        raise VerificationError("deployment verification failed")
    return {
        "mysql": mysql_id,
        "train-factory-api": selection["API_IMAGE_ID"],
        "train-factory-web": selection["WEB_IMAGE_ID"],
    }


def _database_head(
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
    api_container_id: str,
) -> str:
    from scripts.compose_release import _run_private

    completed = _run_private(
        run,
        [
            "docker",
            "exec",
            api_container_id,
            "python",
            "-I",
            "-c",
            _DB_HEAD_SCRIPT,
        ],
        environment,
    )
    try:
        heads = json.loads(
            completed.stdout,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite number")
            ),
        )
    except (json.JSONDecodeError, ValueError):
        raise VerificationError("deployment verification failed") from None
    if (
        not isinstance(heads, list)
        or len(heads) != 1
        or not isinstance(heads[0], str)
        or re.fullmatch(r"[0-9a-z_]{1,128}", heads[0], re.ASCII) is None
    ):
        raise VerificationError("deployment verification failed")
    return heads[0]


def _local_engine_endpoint(value: str) -> str:
    if re.fullmatch(r"npipe:////\./pipe/[A-Za-z0-9_.-]{1,128}", value, re.ASCII):
        return value.lower()
    if value.startswith("unix://"):
        path = value.removeprefix("unix://")
        if (
            not path.startswith("/")
            or len(path) > 4096
            or any(character in value for character in "\x00\r\n?#")
            or ".." in Path(path).parts
        ):
            raise VerificationError("deployment verification failed")
        return "unix://" + path
    raise VerificationError("deployment verification failed")


def _require_local_engine(
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
) -> str:
    from scripts.compose_release import _run_private

    declared_host = environment.get("DOCKER_HOST")
    normalized_declared = (
        _local_engine_endpoint(declared_host) if declared_host is not None else None
    )
    completed = _run_private(
        run,
        [
            "docker",
            "context",
            "inspect",
            "--format",
            '{{ (index .Endpoints "docker").Host }}',
        ],
        environment,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise VerificationError("deployment verification failed")
    observed = _local_engine_endpoint(lines[0])
    if normalized_declared is not None and observed != normalized_declared:
        raise VerificationError("deployment verification failed")
    return observed


def _pin_local_engine_environment(
    environment: Mapping[str, str], endpoint: str
) -> dict[str, str]:
    pinned = dict(environment)
    for key in ("DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        pinned.pop(key, None)
    pinned["DOCKER_HOST"] = endpoint
    return pinned


def _snapshot(
    run: Callable[..., CompletedProcess[str]],
    environment: Mapping[str, str],
    *,
    project: str,
    service: str,
    expected_image_id: str,
    container_id: str | None = None,
) -> tuple[str, str, int, str]:
    from scripts.compose_release import _run_private

    if container_id is None:
        listed = _run_private(
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
            raise VerificationError("deployment verification failed")
        container_id = identifiers[0]
    inspected = _run_private(
        run,
        [
            "docker",
            "inspect",
            "--format",
            "{{.Id}}|{{.Image}}|{{.State.Running}}|{{.State.Health.Status}}|"
            "{{.RestartCount}}|"
            '{{ index .Config.Labels "com.docker.compose.project" }}|'
            '{{ index .Config.Labels "com.docker.compose.service" }}',
            container_id,
        ],
        environment,
    )
    parts = inspected.stdout.strip().split("|")
    try:
        restart_count = int(parts[4])
    except (IndexError, ValueError):
        raise VerificationError("deployment verification failed") from None
    if (
        len(parts) != 7
        or parts[0] != container_id
        or re.fullmatch(r"sha256:[0-9a-f]{64}", parts[1], re.ASCII) is None
        or parts[1] != expected_image_id
        or parts[2] != "true"
        or parts[3] != "healthy"
        or restart_count < 0
        or parts[5:] != [project, service]
    ):
        raise VerificationError("deployment verification failed")
    return container_id, parts[1], restart_count, parts[3]


def verify_deployment(
    *,
    project: str,
    compose_manifest: Path,
    release_env: Path,
    expected_alembic: str,
    require_gpu: bool,
    expected_revision: str | None = None,
    expected_api_revision: str | None = None,
    expected_web_revision: str | None = None,
    scope: str = "full",
    username_file: Path | None = None,
    password_file: Path | None = None,
    rollback_mode: str | None = None,
    rollback_target: str | None = None,
    rollback_env: Path | None = None,
    root: Path = ROOT_DIR,
    base_environment: Mapping[str, str] = os.environ,
    run: Callable[..., CompletedProcess[str]] = subprocess.run,
    request: Callable[[str, str, bytes | None], tuple[int, bytes]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Verify the frozen deployment without exposing secrets or container env."""
    try:
        shared_revision_expectation = (
            expected_revision is not None
            and expected_api_revision is None
            and expected_web_revision is None
        )
        split_revision_expectation = (
            expected_revision is None
            and expected_api_revision is not None
            and expected_web_revision is not None
        )
        supplied_revisions = tuple(
            value
            for value in (
                expected_revision,
                expected_api_revision,
                expected_web_revision,
            )
            if value is not None
        )
        if (
            scope not in {"full", "api-only"}
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", project, re.ASCII) is None
            or not (shared_revision_expectation or split_revision_expectation)
            or any(
                re.fullmatch(r"[0-9a-f]{40}", revision, re.ASCII) is None
                for revision in supplied_revisions
            )
            or re.fullmatch(r"[0-9a-z_]{1,128}", expected_alembic, re.ASCII) is None
            or (username_file is None) != (password_file is None)
            or (scope != "full" and username_file is not None)
            or rollback_mode not in {None, "compat-api-old-web", "release-api-old-web"}
            or rollback_target not in {None, "isolated", "production"}
            or (rollback_mode is None) != (rollback_env is None)
            or (rollback_mode is None) != (rollback_target is None)
            or (
                rollback_mode is not None and (scope != "full" or username_file is None)
            )
            or (rollback_mode is not None and not shared_revision_expectation)
        ):
            raise VerificationError("deployment verification failed")
        if (
            expected_alembic != "053_validate_lifecycle_schema"
            or _repository_head(root) != expected_alembic
        ):
            raise VerificationError("deployment verification failed")
        from scripts import compose_manifest as manifest_module
        from scripts.compose_release import _clean_environment, _run_private

        environment = _clean_environment(base_environment)
        engine_endpoint = _require_local_engine(run, environment)
        environment = _pin_local_engine_environment(environment, engine_endpoint)
        manifest = manifest_module.verify_manifest_inputs_only(
            compose_manifest,
            root=root,
            base_environment=environment,
            run=run,
        )
        if manifest.get("project") != project:
            raise VerificationError("deployment verification failed")
        runtime = Path(os.path.abspath(root / ".runtime"))
        canonical_release = runtime / "release.env"
        selection_record = manifest["env_files"][-1]
        if not isinstance(selection_record, dict):
            raise VerificationError("deployment verification failed")
        selection = _selection_values(manifest)
        api_revision, web_revision, split_revision_selection = _service_revisions(
            selection
        )
        active_selection_path = release_env
        approved_compat_path: Path | None = None
        approved_compat: dict[str, str] | None = None
        if rollback_mode is None:
            manifest_mode = str(manifest.get("mode"))
            if manifest_mode == "ci":
                match = re.fullmatch(
                    r"trainfactory-ci-([0-9a-f]{32})",
                    project,
                    re.ASCII,
                )
                if match is None:
                    raise VerificationError("deployment verification failed")
                run_id = match.group(1)
                expected_release_path = runtime / f"ci-release-{run_id}.env"
                expected_manifest_path = runtime / f"ci-compose-{run_id}-manifest.json"
            else:
                expected_release_path = {
                    "production": canonical_release,
                    "verify": canonical_release,
                }.get(manifest_mode)
                expected_manifest_path = (
                    runtime / "production-compose-manifest.json"
                    if manifest_mode == "production"
                    else None
                )
            if (
                expected_release_path is None
                or (
                    expected_manifest_path is not None
                    and not manifest_module._matches_exact_path(
                        compose_manifest,
                        expected_manifest_path,
                        root=root,
                    )
                )
                or not manifest_module._matches_exact_path(
                    release_env,
                    expected_release_path,
                    root=root,
                )
                or selection_record.get("role") != "image-selection"
                or not manifest_module._matches_exact_path(
                    Path(str(selection_record.get("path", ""))),
                    expected_release_path,
                    root=root,
                )
            ):
                raise VerificationError("deployment verification failed")
            if manifest_mode == "production":
                if split_revision_selection:
                    if (
                        not split_revision_expectation
                        or api_revision != expected_api_revision
                        or web_revision != expected_web_revision
                    ):
                        raise VerificationError("deployment verification failed")
                elif (
                    not shared_revision_expectation or api_revision != expected_revision
                ):
                    raise VerificationError("deployment verification failed")
            elif (
                split_revision_selection
                or not shared_revision_expectation
                or api_revision != expected_revision
            ):
                raise VerificationError("deployment verification failed")
        else:
            from scripts.compose_manifest import (
                _inspect_selected_image,
                _read_selection_source,
            )

            assert rollback_env is not None
            if os.path.normcase(os.path.abspath(release_env)) != os.path.normcase(
                os.fspath(canonical_release)
            ):
                raise VerificationError("deployment verification failed")
            if rollback_target == "isolated":
                run_id = project.rsplit("-", 1)[-1]
                expected_mode = "rollback-verify"
                expected_path = (
                    runtime
                    / f"rollback-verify-{run_id}"
                    / f"rollback-verify-{rollback_mode}.env"
                )
                expected_role = "image-selection"
                expected_variant = rollback_mode
                expected_manifest_path = (
                    runtime
                    / f"rollback-verify-{run_id}"
                    / f"rollback-verify-{rollback_mode}-manifest.json"
                )
            else:
                contracts = {
                    "release-api-old-web": (
                        "rollback-web-only",
                        runtime / "rollback-web-only.env",
                        runtime / "rollback-web-only-compose-manifest.json",
                        "image-selection",
                    ),
                    "compat-api-old-web": (
                        "rollback-post-migration",
                        runtime / "rollback-post.env",
                        runtime / "rollback-post-compose-manifest.json",
                        "rollback-post",
                    ),
                }
                (
                    expected_mode,
                    expected_path,
                    expected_manifest_path,
                    expected_role,
                ) = contracts[rollback_mode]
                expected_variant = None
            if (
                manifest.get("mode") != expected_mode
                or manifest.get("rollback_variant") != expected_variant
                or os.path.normcase(os.path.abspath(compose_manifest))
                != os.path.normcase(os.fspath(expected_manifest_path))
                or os.path.normcase(os.path.abspath(rollback_env))
                != os.path.normcase(os.fspath(expected_path))
                or selection_record.get("role") != expected_role
                or os.path.normcase(str(selection_record.get("path", "")))
                != os.path.normcase(os.fspath(expected_path))
                or set(selection)
                != {
                    "API_IMAGE",
                    "WEB_IMAGE",
                    "API_REVISION",
                    "WEB_REVISION",
                    "API_IMAGE_ID",
                    "WEB_IMAGE_ID",
                }
            ):
                raise VerificationError("deployment verification failed")
            current_release = _read_selection_source(release_env, root=root)
            captured_path = runtime / "rollback-pre.env"
            captured = _read_selection_source(captured_path, root=root)
            if (
                "RELEASE_REVISION" not in current_release
                or current_release["RELEASE_REVISION"] != expected_revision
                or "API_REVISION" not in captured
            ):
                raise VerificationError("deployment verification failed")
            environment_for_images = environment
            _inspect_selected_image(
                current_release,
                prefix="API",
                run=run,
                environment=environment_for_images,
            )
            _inspect_selected_image(
                current_release,
                prefix="WEB",
                run=run,
                environment=environment_for_images,
            )
            _inspect_selected_image(
                captured,
                prefix="API",
                run=run,
                environment=environment_for_images,
            )
            _inspect_selected_image(
                captured,
                prefix="WEB",
                run=run,
                environment=environment_for_images,
            )
            if any(
                selection[key] != captured[key]
                for key in ("WEB_IMAGE", "WEB_IMAGE_ID", "WEB_REVISION")
            ):
                raise VerificationError("deployment verification failed")
            if rollback_mode == "release-api-old-web":
                if any(
                    selection[key] != value
                    for key, value in (
                        ("API_IMAGE", current_release["API_IMAGE"]),
                        ("API_IMAGE_ID", current_release["API_IMAGE_ID"]),
                        ("API_REVISION", current_release["RELEASE_REVISION"]),
                    )
                ):
                    raise VerificationError("deployment verification failed")
            else:
                approved_compat_path = runtime / "rollback-post.env"
                approved_compat = _read_selection_source(
                    approved_compat_path,
                    root=root,
                )
                _inspect_selected_image(
                    approved_compat,
                    prefix="API",
                    run=run,
                    environment=environment_for_images,
                    expected_compat="053_validate_lifecycle_schema",
                )
                _inspect_selected_image(
                    approved_compat,
                    prefix="WEB",
                    run=run,
                    environment=environment_for_images,
                )
                if (
                    selection["API_REVISION"] != captured["API_REVISION"]
                    or selection != approved_compat
                ):
                    raise VerificationError("deployment verification failed")
            active_selection_path = rollback_env
        if require_gpu != (manifest.get("gpu_mode") != "cpu"):
            raise VerificationError("deployment verification failed")
        services = ["mysql", "train-factory-api"]
        if scope == "full":
            services.append("train-factory-web")
        expected_image_ids = _expected_image_ids(
            manifest,
            selection,
            run=run,
            environment=environment,
        )
        service_summary: dict[str, object] = {}
        snapshots: dict[str, tuple[str, str, int, str]] = {}
        for service in services:
            manifest_module.verify_running_container(
                compose_manifest,
                selection_path=active_selection_path,
                service=service,
                root=root,
                base_environment=environment,
                run=run,
            )
            snapshots[service] = _snapshot(
                run,
                environment,
                project=project,
                service=service,
                expected_image_id=expected_image_ids[service],
            )
        api_id = snapshots["train-factory-api"][0]
        api_container_port, bind_address = _effective_runtime_endpoint(manifest)
        api_origin = _port(
            run,
            environment,
            api_id,
            api_container_port,
            expected_host=bind_address,
        )
        http_client: object | None = None
        if request is None:
            probe_address = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(
                bind_address, bind_address
            )
            http_client = _PrivateHttpClient(allowed_hosts={probe_address})
            http_request = http_client.request
        elif callable(request):
            http_request = request
        elif callable(getattr(request, "request", None)):
            http_client = request
            http_request = request.request
        else:
            raise VerificationError("deployment verification failed")
        status, payload = http_request("GET", api_origin + "/health", None)
        health = _parse_json(payload, expected_keys={"status", "version"})
        if (
            status != 200
            or health["status"] != "healthy"
            or not isinstance(health["version"], str)
            or not health["version"]
        ):
            raise VerificationError("deployment verification failed")
        status, payload = http_request("GET", api_origin + "/api/auth/config", None)
        auth_config = _parse_json(
            payload,
            expected_keys={
                "self_registration_enabled",
                "direct_storage_registration_enabled",
            },
        )
        if status != 200 or any(
            type(item) is not bool for item in auth_config.values()
        ):
            raise VerificationError("deployment verification failed")
        web_origin = None
        if scope == "full":
            web_origin = _port(
                run,
                environment,
                snapshots["train-factory-web"][0],
                80,
                expected_host=bind_address,
            )
            status, payload = http_request("GET", web_origin + "/health", None)
            if status != 200 or payload != b"healthy\n":
                raise VerificationError("deployment verification failed")
            status, payload = http_request("GET", web_origin + "/api/auth/config", None)
            proxied = _parse_json(
                payload,
                expected_keys={
                    "self_registration_enabled",
                    "direct_storage_registration_enabled",
                },
            )
            if status != 200 or proxied != auth_config:
                raise VerificationError("deployment verification failed")

        initial_database_head = _database_head(run, environment, api_id)
        if initial_database_head != expected_alembic:
            raise VerificationError("deployment verification failed")
        gpu_mode = "required" if require_gpu else "off"
        gpu = _run_private(
            run,
            [
                "docker",
                "exec",
                api_id,
                "python",
                "-I",
                "-c",
                _GPU_PROBE_SCRIPT,
                gpu_mode,
            ],
            environment,
        )
        gpu_result = _parse_json(
            gpu.stdout.encode(),
            expected_keys={
                "mode",
                "ok",
                "compiled_cuda",
                "driver_version",
                "device_count",
                "probe_executed",
                "reason_code",
            },
        )
        if require_gpu:
            if (
                gpu_result["mode"] != "required"
                or gpu_result["ok"] is not True
                or type(gpu_result["device_count"]) is not int
                or gpu_result["device_count"] <= 0
                or gpu_result["probe_executed"] is not True
                or gpu_result["reason_code"] is not None
                or not isinstance(gpu_result["compiled_cuda"], str)
                or re.fullmatch(
                    r"[0-9]{1,6}\.[0-9]{1,6}",
                    gpu_result["compiled_cuda"],
                    re.ASCII,
                )
                is None
                or not isinstance(gpu_result["driver_version"], str)
                or re.fullmatch(
                    r"[0-9]{1,6}\.[0-9]{1,6}(?:\.[0-9]{1,6})?",
                    gpu_result["driver_version"],
                    re.ASCII,
                )
                is None
            ):
                raise VerificationError("deployment verification failed")
        elif gpu_result != {
            "mode": "off",
            "ok": True,
            "compiled_cuda": None,
            "driver_version": None,
            "device_count": 0,
            "probe_executed": False,
            "reason_code": None,
        }:
            raise VerificationError("deployment verification failed")

        auth_verified = False
        credential_state: (
            tuple[
                str,
                tuple[int, int, int, int],
                str,
                tuple[int, int, int, int],
            ]
            | None
        ) = None
        if username_file is not None and password_file is not None:
            if web_origin is None:
                raise VerificationError("deployment verification failed")
            username, username_identity = _private_file_snapshot(
                username_file, root=root, max_bytes=_USERNAME_MAX_BYTES
            )
            password, password_identity = _private_file_snapshot(
                password_file, root=root, max_bytes=65536
            )
            if not 3 <= len(username) <= 64 or not 10 <= len(password) <= 128:
                raise VerificationError("deployment verification failed")
            login_body = json.dumps(
                {"username": username, "password": password},
                separators=(",", ":"),
            ).encode()
            status, payload = http_request(
                "POST", web_origin + "/api/auth/login", login_body
            )
            login = _parse_json(payload, expected_keys={"access_token", "token_type"})
            if (
                status != 200
                or not isinstance(login["access_token"], str)
                or not login["access_token"]
                or login["token_type"] != "bearer"
            ):
                raise VerificationError("deployment verification failed")
            if http_client is None or not callable(
                getattr(http_client, "set_bearer", None)
            ):
                raise VerificationError("deployment verification failed")
            http_client.set_bearer(login["access_token"])
            status, payload = http_request("GET", web_origin + "/api/auth/me", None)
            current = _parse_json(
                payload,
                expected_keys={
                    "user_id",
                    "username",
                    "email",
                    "is_active",
                    "is_admin",
                    "created_at",
                    "updated_at",
                },
            )
            if (
                status != 200
                or not isinstance(current["user_id"], str)
                or not current["user_id"]
                or current["username"] != username
                or current["is_active"] is not True
                or current["is_admin"] is not True
                or not (current["email"] is None or isinstance(current["email"], str))
                or not (
                    current["created_at"] is None
                    or isinstance(current["created_at"], str)
                )
                or not (
                    current["updated_at"] is None
                    or isinstance(current["updated_at"], str)
                )
            ):
                raise VerificationError("deployment verification failed")
            auth_verified = True
            credential_state = (
                username,
                username_identity,
                password,
                password_identity,
            )
        sleep(1.0)
        final_manifest = manifest_module.verify_manifest_inputs_only(
            compose_manifest,
            root=root,
            base_environment=environment,
            run=run,
        )
        if final_manifest != manifest or _selection_values(final_manifest) != selection:
            raise VerificationError("deployment verification failed")
        if _repository_head(root) != expected_alembic:
            raise VerificationError("deployment verification failed")
        if rollback_mode is not None:
            if (
                _read_selection_source(release_env, root=root) != current_release
                or _read_selection_source(captured_path, root=root) != captured
                or (
                    approved_compat_path is not None
                    and _read_selection_source(approved_compat_path, root=root)
                    != approved_compat
                )
            ):
                raise VerificationError("deployment verification failed")
            _inspect_selected_image(
                current_release,
                prefix="API",
                run=run,
                environment=environment,
            )
            if approved_compat is not None:
                _inspect_selected_image(
                    approved_compat,
                    prefix="API",
                    run=run,
                    environment=environment,
                    expected_compat="053_validate_lifecycle_schema",
                )
                _inspect_selected_image(
                    approved_compat,
                    prefix="WEB",
                    run=run,
                    environment=environment,
                )
            _inspect_selected_image(
                captured,
                prefix="API",
                run=run,
                environment=environment,
            )
            _inspect_selected_image(
                captured,
                prefix="WEB",
                run=run,
                environment=environment,
            )
            _inspect_selected_image(
                current_release,
                prefix="WEB",
                run=run,
                environment=environment,
            )
        for service, initial in snapshots.items():
            manifest_module.verify_running_container(
                compose_manifest,
                selection_path=active_selection_path,
                service=service,
                root=root,
                base_environment=environment,
                run=run,
            )
            final = _snapshot(
                run,
                environment,
                project=project,
                service=service,
                expected_image_id=expected_image_ids[service],
            )
            if final != initial:
                raise VerificationError("deployment verification failed")
            service_summary[service] = {
                "container": final[0][:12],
                "image_id": final[1][7:19],
                "revision": (
                    None
                    if service == "mysql"
                    else api_revision
                    if service == "train-factory-api"
                    else web_revision
                ),
                "health": True,
            }
        final_database_head = _database_head(run, environment, api_id)
        if (
            final_database_head != initial_database_head
            or final_database_head != expected_alembic
        ):
            raise VerificationError("deployment verification failed")
        if credential_state is not None:
            assert username_file is not None and password_file is not None
            username, username_identity, password, password_identity = credential_state
            if _private_file_snapshot(
                username_file, root=root, max_bytes=_USERNAME_MAX_BYTES
            ) != (username, username_identity) or _private_file_snapshot(
                password_file, root=root, max_bytes=65536
            ) != (password, password_identity):
                raise VerificationError("deployment verification failed")
        return {
            "project": project,
            "scope": scope,
            "services": service_summary,
            "api_revision": api_revision,
            "web_revision": web_revision,
            "revision": api_revision if api_revision == web_revision else None,
            "db_head": final_database_head,
            "gpu_reason_code": gpu_result["reason_code"],
            "auth_verified": auth_verified,
            "rollback_mode": rollback_mode,
            "rollback_target": rollback_target,
        }
    except VerificationError:
        raise
    except Exception:
        raise VerificationError("deployment verification failed") from None


class _PrivateParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise VerificationError("deployment verification failed")


def _parser() -> argparse.ArgumentParser:
    parser = _PrivateParser(allow_abbrev=False)
    parser.add_argument("--project", required=True)
    parser.add_argument("--compose-manifest", type=Path, required=True)
    parser.add_argument("--release-env", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-api-revision")
    parser.add_argument("--expected-web-revision")
    parser.add_argument("--expected-alembic", required=True)
    gpu = parser.add_mutually_exclusive_group(required=True)
    gpu.add_argument("--require-gpu", action="store_true")
    gpu.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--scope", choices=("full", "api-only"), default="full")
    parser.add_argument("--username-file", type=Path)
    parser.add_argument("--password-file", type=Path)
    parser.add_argument(
        "--rollback-mode",
        choices=("compat-api-old-web", "release-api-old-web"),
    )
    parser.add_argument(
        "--rollback-target",
        choices=("isolated", "production"),
    )
    parser.add_argument("--rollback-env", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        summary = verify_deployment(
            project=arguments.project,
            compose_manifest=arguments.compose_manifest,
            release_env=arguments.release_env,
            expected_revision=arguments.expected_revision,
            expected_api_revision=arguments.expected_api_revision,
            expected_web_revision=arguments.expected_web_revision,
            expected_alembic=arguments.expected_alembic,
            require_gpu=arguments.require_gpu,
            scope=arguments.scope,
            username_file=arguments.username_file,
            password_file=arguments.password_file,
            rollback_mode=arguments.rollback_mode,
            rollback_target=arguments.rollback_target,
            rollback_env=arguments.rollback_env,
        )
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0
    except SystemExit as error:
        if error.code == 0:
            return 0
        print("deployment verification failed", file=sys.stderr)
        return 1
    except VerificationError:
        print("deployment verification failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
