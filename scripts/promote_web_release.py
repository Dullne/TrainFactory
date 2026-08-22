"""Promote one immutable Web image while preserving the running production API."""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("web promotion bootstrap is unavailable")


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
import base64  # noqa: E402
import ctypes  # noqa: E402
import errno  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

from scripts import compose_manifest, compose_release, verify_deployment  # noqa: E402
from scripts.materialize_compose_secrets import (  # noqa: E402
    _harden_path,
    _verify_hardened_path,
)


ROOT_DIR = Path(_root_entry)
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}", re.ASCII)
_REVISION = re.compile(r"[0-9a-f]{40}", re.ASCII)
_IMAGE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}", re.ASCII)
_TAIL = (
    "up",
    "-d",
    "--no-deps",
    "--force-recreate",
    "--wait",
    "--wait-timeout",
    "600",
    "train-factory-web",
)
_PRODUCTION_HTTP_VERIFIER = r"""
import http.client
import json
import os
import secrets

_HTML_LIMIT = 512 * 1024
_JSON_LIMIT = 64 * 1024


def _fail():
    raise ValueError


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            _fail()
        result[key] = value
    return result


def _json(payload):
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: _fail(),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        _fail()
    if type(value) is not dict:
        _fail()
    return value


def _request(host, port, method, path, *, body=None, headers=None, limit):
    connection = http.client.HTTPConnection(host, port, timeout=20)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read(limit + 1)
        if len(payload) > limit:
            _fail()
        return response.status, tuple(response.getheaders()), payload
    finally:
        connection.close()


def _web_request(method, path, *, body=None, headers=None, limit):
    return _request(
        "train-factory-web",
        80,
        method,
        path,
        body=body,
        headers=headers,
        limit=limit,
    )


def _api_request(method, path, *, body=None, headers=None, limit):
    return _request(
        "127.0.0.1",
        _API_PORT,
        method,
        path,
        body=body,
        headers=headers,
        limit=limit,
    )


def _content_type(headers):
    values = [value for name, value in headers if name.lower() == "content-type"]
    if len(values) != 1:
        _fail()
    return values[0].split(";", 1)[0].strip().lower()


def _html(path):
    status, headers, payload = _web_request("GET", path, limit=_HTML_LIMIT)
    if (
        status != 200
        or _content_type(headers) != "text/html"
        or b'<div id="root"></div>' not in payload
        or b"<title>TrainFactory - AI Training Platform</title>" not in payload
    ):
        _fail()
    return payload


def _main():
    global _API_PORT
    port = os.environ.get("API_PORT", "18000")
    if not port.isdecimal() or not 1 <= int(port) <= 65535:
        _fail()
    _API_PORT = int(port)

    login_page = _html("/login")
    health_status, health_headers, health_payload = _api_request(
        "GET", "/health", limit=_JSON_LIMIT
    )
    health = _json(health_payload)
    if (
        health_status != 200
        or _content_type(health_headers) != "application/json"
        or set(health) != {"status", "version"}
        or health["status"] != "healthy"
        or type(health["version"]) is not str
        or not 1 <= len(health["version"]) <= 64
    ):
        _fail()

    api_config_status, api_config_headers, api_config_payload = _api_request(
        "GET", "/api/auth/config", limit=_JSON_LIMIT
    )
    web_config_status, web_config_headers, web_config_payload = _web_request(
        "GET", "/api/auth/config", limit=_JSON_LIMIT
    )
    api_config = _json(api_config_payload)
    web_config = _json(web_config_payload)
    if (
        api_config_status != 200
        or web_config_status != 200
        or _content_type(api_config_headers) != "application/json"
        or _content_type(web_config_headers) != "application/json"
        or api_config != web_config
        or set(api_config)
        != {"self_registration_enabled", "direct_storage_registration_enabled"}
        or type(api_config["self_registration_enabled"]) is not bool
        or api_config["direct_storage_registration_enabled"] is not False
    ):
        _fail()

    rejection_headers = {
        "Accept": "application/json",
        "Cookie": "access_token=web-promotion-invalid-" + secrets.token_hex(32),
    }
    api_me_status, api_me_headers, api_me_payload = _api_request(
        "GET",
        "/api/auth/me",
        headers=rejection_headers,
        limit=_JSON_LIMIT,
    )
    web_me_status, web_me_headers, web_me_payload = _web_request(
        "GET",
        "/api/auth/me",
        headers=rejection_headers,
        limit=_JSON_LIMIT,
    )
    api_me = _json(api_me_payload)
    web_me = _json(web_me_payload)
    api_authenticate = [
        value for name, value in api_me_headers if name.lower() == "www-authenticate"
    ]
    web_authenticate = [
        value for name, value in web_me_headers if name.lower() == "www-authenticate"
    ]
    if (
        api_me_status != 401
        or web_me_status != 401
        or _content_type(api_me_headers) != "application/json"
        or _content_type(web_me_headers) != "application/json"
        or api_me != {"detail": "Invalid or expired token"}
        or web_me != api_me
        or api_authenticate != ["Bearer"]
        or web_authenticate != ["Bearer"]
        or any(name.lower() == "set-cookie" for name, _value in api_me_headers)
        or any(name.lower() == "set-cookie" for name, _value in web_me_headers)
    ):
        _fail()

    training_page = _html("/training")
    if training_page != login_page:
        _fail()
    print(
        '{"success":true,"code":"OK","login_page":true,'
        '"api_health":true,"api_proxy":true,"auth_rejection":true,'
        '"training_route":true}'
    )


try:
    _main()
except Exception:
    raise SystemExit(1) from None
""".strip()
_PROJECT_CONTAINERS = frozenset(
    {
        "trainfactory-api",
        "trainfactory-web",
        "trainfactory-mysql",
        "trainfactory-milvus",
        "trainfactory-minio",
        "trainfactory-etcd",
    }
)
_CONTAINER_SERVICES = {
    "trainfactory-api": "train-factory-api",
    "trainfactory-web": "train-factory-web",
    "trainfactory-mysql": "mysql",
    "trainfactory-milvus": "milvus",
    "trainfactory-minio": "minio",
    "trainfactory-etcd": "etcd",
}
_FIXED_CONTAINERS = tuple(sorted(_PROJECT_CONTAINERS - {"trainfactory-web"}))
_REQUIRED_FIXED_CONTAINERS = frozenset({"trainfactory-api", "trainfactory-mysql"})
_FIXED_VOLUME_ROLES = {
    "trainfactory_etcd_data": "etcd_data",
    "trainfactory_milvus_data": "milvus_data",
    "trainfactory_minio_data": "minio_data",
    "trainfactory_runtime_mysql_data": "mysql_data",
    "trainfactory_runtime_train_cache": "train_cache",
}
_FIXED_VOLUMES = frozenset(_FIXED_VOLUME_ROLES)
_REQUIRED_FIXED_VOLUMES = frozenset(
    {"trainfactory_runtime_mysql_data", "trainfactory_runtime_train_cache"}
)
_SHARED_SELECTION_KEYS = (
    "API_IMAGE",
    "WEB_IMAGE",
    "RELEASE_REVISION",
    "API_IMAGE_ID",
    "WEB_IMAGE_ID",
)
_SPLIT_SELECTION_KEYS = (
    "API_IMAGE",
    "WEB_IMAGE",
    "API_REVISION",
    "WEB_REVISION",
    "API_IMAGE_ID",
    "WEB_IMAGE_ID",
)
_SAFE_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", re.ASCII)
_CLAIM_NAME = re.compile(
    r"\.web-promotion-([0-9a-f]{32})-(selection|manifest|journal)\.claim",
    re.ASCII,
)
_RETIRED_NAME = re.compile(r"\..+\.[0-9a-f]{32}\.delete", re.ASCII)
_TEMPORARY_NAME = re.compile(
    r"\.(?:release\.env|production-compose-manifest\.json|"
    r"web-promotion-active\.json|\.web-promotion-[0-9a-f]{40}-manifest\.json)"
    r"\.[0-9a-f]{32}\.tmp",
    re.ASCII,
)
_PREPARED_NAME = re.compile(r"\.web-promotion-([0-9a-f]{40})-manifest\.json", re.ASCII)
_JOURNAL_RETIRING_NAME = re.compile(
    r"\.web-promotion-([0-9a-f]{32})-journal\.retiring", re.ASCII
)
_MAX_RETIRED_FILES = 4096
_MAX_RETIRED_BYTES = 512 * 1024 * 1024
_MAX_JOURNAL_BYTES = 512 * 1024
_PROMOTION_RETIRED_FILE_RESERVE = 16
_PROMOTION_RETIRED_BYTE_RESERVE = 8 * 1024 * 1024
_DELETE_NOT_REMOVED = "not-removed"
_DELETE_REMOVED_DURABLE = "removed-durable"
_DELETE_REMOVED_UNSYNCED = "removed-unsynced"

if os.name == "nt":
    import msvcrt  # noqa: E402
else:
    import fcntl  # noqa: E402


class WebPromotionError(RuntimeError):
    """Fixed-message production promotion failure."""


class _Parser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise WebPromotionError("web promotion arguments are invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--web-image", required=True)
    parser.add_argument("--web-image-id", required=True)
    parser.add_argument("--web-revision", required=True)
    return parser


def _regular_file(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError:
        raise WebPromotionError("web promotion state is invalid") from None
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise WebPromotionError("web promotion state is invalid")
    return metadata


def _stable_bytes(path: Path, *, max_bytes: int) -> tuple[bytes, tuple[int, int]]:
    _regular_file(path)
    try:
        compose_manifest._validate_ancestor_chain(path, boundary=path.parent)
        return compose_manifest._read_stable(path, max_bytes=max_bytes)
    except Exception:
        raise WebPromotionError("web promotion state is invalid") from None


def _runtime_owned_by_current_user(metadata: os.stat_result) -> bool:
    if os.name == "nt":
        return True
    geteuid = getattr(os, "geteuid", None)
    return callable(geteuid) and metadata.st_uid == geteuid()


def _private_runtime_identity(root: Path, runtime: Path) -> tuple[int, int]:
    expected = Path(os.path.abspath(root / ".runtime"))
    try:
        if runtime != expected:
            raise ValueError
        compose_manifest._validate_ancestor_chain(runtime, boundary=root)
        metadata = runtime.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or not _runtime_owned_by_current_user(metadata)
            or not _verify_hardened_path(runtime)
        ):
            raise ValueError
        return metadata.st_dev, metadata.st_ino
    except Exception:
        raise WebPromotionError(
            "web promotion runtime directory is not private"
        ) from None


def _verify_private_runtime(runtime: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = runtime.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
            or not _runtime_owned_by_current_user(metadata)
            or not _verify_hardened_path(runtime)
        ):
            raise ValueError
    except Exception:
        raise WebPromotionError(
            "web promotion runtime directory is not private"
        ) from None


def _reject_crash_temporaries(runtime: Path) -> None:
    try:
        entries = tuple(runtime.iterdir())
    except OSError:
        raise WebPromotionError(
            "web promotion temporary state requires inspection"
        ) from None
    if any(_TEMPORARY_NAME.fullmatch(path.name) is not None for path in entries):
        raise WebPromotionError("web promotion temporary state requires inspection")


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        create_file = ctypes.windll.kernel32.CreateFileW
        create_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        create_file.restype = ctypes.c_void_p
        flush = ctypes.windll.kernel32.FlushFileBuffers
        flush.argtypes = (ctypes.c_void_p,)
        flush.restype = ctypes.c_int
        close = ctypes.windll.kernel32.CloseHandle
        close.argtypes = (ctypes.c_void_p,)
        close.restype = ctypes.c_int
        handle = create_file(
            os.fspath(path),
            0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise OSError
        try:
            if not flush(handle):
                raise OSError
        finally:
            close(handle)
        return
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _claim_path(runtime: Path, nonce: str, role: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", nonce, re.ASCII) is None or role not in {
        "selection",
        "manifest",
        "journal",
    }:
        raise WebPromotionError("web promotion recovery state is invalid")
    return runtime / f".web-promotion-{nonce}-{role}.claim"


def _move_noreplace(source: Path, target: Path) -> None:
    if source.parent != target.parent:
        raise OSError
    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform.startswith("linux"):
        try:
            renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
        except AttributeError:
            renameat2 = None
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), os.fspath(target))
            return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace rename is unavailable",
        os.fspath(target),
    )


def _move_noreplace_owned(
    source: Path,
    target: Path,
    identity: tuple[int, int],
) -> None:
    moved = False
    try:
        metadata = source.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise OSError
        _move_noreplace(source, target)
        moved = True
        metadata = target.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise OSError
    except (OSError, WebPromotionError):
        if moved:
            try:
                if not (source.exists() or source.is_symlink()):
                    _move_noreplace(target, source)
                    _sync_directory(source.parent)
            except (OSError, WebPromotionError):
                pass
        raise OSError from None


def _retired_usage(parent: Path) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    try:
        entries = tuple(parent.iterdir())
    except OSError:
        raise WebPromotionError("web promotion retired state is invalid") from None
    for path in entries:
        if _RETIRED_NAME.fullmatch(path.name) is None:
            continue
        try:
            metadata = path.lstat()
        except OSError:
            raise WebPromotionError("web promotion retired state is invalid") from None
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise WebPromotionError("web promotion retired state is invalid")
        count += 1
        total_bytes += metadata.st_size
        if count > _MAX_RETIRED_FILES or total_bytes > _MAX_RETIRED_BYTES:
            raise WebPromotionError("web promotion retired state limit exceeded")
    return count, total_bytes


def _ensure_retired_capacity(
    parent: Path,
    *,
    files: int,
    total_bytes: int,
) -> None:
    if files < 0 or total_bytes < 0:
        raise WebPromotionError("web promotion retired state is invalid")
    retired_files, retired_bytes = _retired_usage(parent)
    if (
        retired_files > _MAX_RETIRED_FILES - files
        or retired_bytes > _MAX_RETIRED_BYTES - total_bytes
    ):
        raise WebPromotionError("web promotion retired state limit exceeded")


def _journal_retiring_path(claim_path: Path) -> Path:
    match = _CLAIM_NAME.fullmatch(claim_path.name)
    if match is None or match.group(2) != "journal":
        raise WebPromotionError("web promotion recovery state is invalid")
    return claim_path.parent / f".web-promotion-{match.group(1)}-journal.retiring"


def _descriptor_payload(descriptor: int, *, max_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise OSError


def _delete_isolated_owned(
    path: Path,
    identity: tuple[int, int],
    *,
    expected_size: int,
    expected_payload: bytes | None = None,
    mutation_guard: Callable[[], None] | None = None,
    _before_bound_delete: Callable[[Path], None] | None = None,
) -> str:
    guard = mutation_guard or (lambda: None)
    before_delete = _before_bound_delete or (lambda _path: None)
    descriptor = -1

    if os.name == "nt":
        create_file = ctypes.windll.kernel32.CreateFileW
        create_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        create_file.restype = ctypes.c_void_p
        close_handle = ctypes.windll.kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        handle = create_file(
            os.fspath(path),
            0x80000000 | 0x00010000,
            0x00000001,
            None,
            3,
            0x00200000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            return _DELETE_NOT_REMOVED
        try:
            descriptor = msvcrt.open_osfhandle(
                int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
            )
        except (OSError, OverflowError):
            close_handle(handle)
            return _DELETE_NOT_REMOVED
        try:
            opened = os.fstat(descriptor)
            current = path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or (current.st_dev, current.st_ino) != identity
                or opened.st_size != expected_size
                or current.st_size != expected_size
                or expected_payload is not None
                and _descriptor_payload(
                    descriptor, max_bytes=max(1, len(expected_payload))
                )
                != expected_payload
            ):
                return _DELETE_NOT_REMOVED
            guard()
            before_delete(path)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or opened.st_size != expected_size
            ):
                return _DELETE_NOT_REMOVED

            class FileDispositionInfo(ctypes.Structure):
                _fields_ = (("delete_file", ctypes.c_ubyte),)

            disposition = FileDispositionInfo(1)
            set_information = ctypes.windll.kernel32.SetFileInformationByHandle
            set_information.argtypes = (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            )
            set_information.restype = ctypes.c_int
            if not set_information(
                ctypes.c_void_p(msvcrt.get_osfhandle(descriptor)),
                4,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                return _DELETE_NOT_REMOVED
            try:
                os.close(descriptor)
            except OSError:
                descriptor = -1
                return _DELETE_REMOVED_UNSYNCED
            descriptor = -1
            try:
                _sync_directory(path.parent)
            except OSError:
                return _DELETE_REMOVED_UNSYNCED
            return _DELETE_REMOVED_DURABLE
        except (OSError, WebPromotionError):
            return _DELETE_NOT_REMOVED
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    parent_descriptor = -1
    removed = False
    try:
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        parent_opened = os.fstat(parent_descriptor)
        parent_current = path.parent.lstat()
        if (
            not stat.S_ISDIR(parent_opened.st_mode)
            or not stat.S_ISDIR(parent_current.st_mode)
            or stat.S_ISLNK(parent_current.st_mode)
            or (parent_opened.st_dev, parent_opened.st_ino)
            != (parent_current.st_dev, parent_current.st_ino)
        ):
            return _DELETE_NOT_REMOVED
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino) != identity
            or (current.st_dev, current.st_ino) != identity
            or opened.st_size != expected_size
            or current.st_size != expected_size
            or expected_payload is not None
            and _descriptor_payload(descriptor, max_bytes=max(1, len(expected_payload)))
            != expected_payload
        ):
            return _DELETE_NOT_REMOVED
        guard()
        before_delete(path)
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
            or current.st_size != expected_size
        ):
            return _DELETE_NOT_REMOVED
        links_before = opened.st_nlink
        os.unlink(path.name, dir_fd=parent_descriptor)
        removed = True
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
        ) != identity or after.st_nlink != links_before - 1:
            return _DELETE_REMOVED_UNSYNCED
        try:
            os.fsync(parent_descriptor)
        except OSError:
            return _DELETE_REMOVED_UNSYNCED
        return _DELETE_REMOVED_DURABLE
    except (OSError, WebPromotionError):
        return _DELETE_REMOVED_UNSYNCED if removed else _DELETE_NOT_REMOVED
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_descriptor >= 0:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


def _unlink_owned(
    path: Path,
    identity: tuple[int, int],
    *,
    mutation_guard: Callable[[], None] | None = None,
    expected_payload: bytes | None = None,
    _before_bound_delete: Callable[[Path], None] | None = None,
) -> bool:
    quarantine = path.parent / f".{path.name}.{secrets.token_hex(16)}.delete"
    guard = mutation_guard or (lambda: None)
    moved = False
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            return False
        _ensure_retired_capacity(path.parent, files=1, total_bytes=metadata.st_size)
        guard()
        _move_noreplace_owned(path, quarantine, identity)
        moved = True
        _sync_directory(path.parent)
        claimed = quarantine.lstat()
        if (
            not stat.S_ISREG(claimed.st_mode)
            or stat.S_ISLNK(claimed.st_mode)
            or (claimed.st_dev, claimed.st_ino) != identity
            or path.exists()
            or path.is_symlink()
        ):
            return False
        retired = quarantine.lstat()
        if (
            not stat.S_ISREG(retired.st_mode)
            or stat.S_ISLNK(retired.st_mode)
            or (retired.st_dev, retired.st_ino) != identity
            or retired.st_size != metadata.st_size
            or path.exists()
            or path.is_symlink()
        ):
            return False
        _retired_usage(path.parent)
        outcome = _delete_isolated_owned(
            quarantine,
            identity,
            expected_size=metadata.st_size,
            expected_payload=expected_payload,
            mutation_guard=guard,
            _before_bound_delete=_before_bound_delete,
        )
        if outcome != _DELETE_NOT_REMOVED:
            moved = False
        return (
            outcome == _DELETE_REMOVED_DURABLE
            and not (path.exists() or path.is_symlink())
            and not (quarantine.exists() or quarantine.is_symlink())
        )
    except (OSError, WebPromotionError):
        return False
    finally:
        if moved:
            try:
                guard()
                _move_noreplace(quarantine, path)
                _sync_directory(path.parent)
            except (OSError, WebPromotionError):
                pass


def _unlink_owned_payload(
    path: Path,
    identity: tuple[int, int],
    payload: bytes,
    *,
    max_bytes: int,
    mutation_guard: Callable[[], None] | None = None,
) -> bool:
    try:
        observed_payload, observed_identity = _stable_bytes(path, max_bytes=max_bytes)
    except WebPromotionError:
        return False
    return (
        observed_identity == identity
        and observed_payload == payload
        and _unlink_owned(
            path,
            identity,
            mutation_guard=mutation_guard,
            expected_payload=payload,
        )
    )


def _verify_process_lock(path: Path, descriptor: int) -> None:
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
    except OSError:
        raise WebPromotionError("web promotion lock is invalid") from None
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or opened.st_nlink != 1
        or current.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise WebPromotionError("web promotion lock is invalid")


def _acquire_process_lock(path: Path) -> int:
    descriptor = -1
    locked = False
    try:
        parent = path.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
            raise WebPromotionError("web promotion lock is invalid")
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOINHERIT", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags, 0o600)
        os.set_inheritable(descriptor, False)
        _verify_process_lock(path, descriptor)
        opened = os.fstat(descriptor)
        if opened.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise WebPromotionError("web promotion is already active") from None
        locked = True
        _verify_process_lock(path, descriptor)
        if not _harden_path(path):
            raise WebPromotionError("web promotion lock is invalid")
        _verify_process_lock(path, descriptor)
        _sync_directory(path.parent)
        return descriptor
    except WebPromotionError:
        raise
    except OSError:
        raise WebPromotionError("web promotion lock is invalid") from None
    finally:
        if descriptor >= 0 and (not locked or sys.exc_info()[0] is not None):
            if locked:
                _release_process_lock(descriptor)
            else:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _release_process_lock(descriptor: int) -> None:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _atomic_replace(
    path: Path,
    payload: bytes,
    *,
    expected_identity: tuple[int, int] | None = None,
    expected_payload: bytes | None = None,
    require_absent: bool = False,
    claim_path: Path | None = None,
    retain_claim: bool = False,
    mutation_guard: Callable[[], None] | None = None,
    _before_publish: Callable[[tuple[int, int]], None] | None = None,
    _published_identity: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    conditional = expected_identity is not None or expected_payload is not None
    if (
        require_absent
        and (conditional or claim_path is not None)
        or not require_absent
        and ((expected_identity is None) != (expected_payload is None))
        or conditional
        and claim_path is None
        or retain_claim
        and not conditional
        or claim_path is not None
        and claim_path.parent != path.parent
        or _published_identity is not None
        and _published_identity
    ):
        raise WebPromotionError("web promotion state update failed")
    parent = path.parent
    try:
        parent_metadata = parent.lstat()
        if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(
            parent_metadata.st_mode
        ):
            raise OSError
    except OSError:
        raise WebPromotionError("web promotion state update failed") from None
    temporary = parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    identity: tuple[int, int] | None = None
    claim_identity: tuple[int, int] | None = None
    claim_active = False
    published = False
    complete = False
    guard = mutation_guard or (lambda: None)
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
        if (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino) != identity:
            raise OSError
        os.close(descriptor)
        descriptor = -1
        if not _harden_path(temporary):
            raise OSError
        verified_payload, verified_identity = _stable_bytes(
            temporary, max_bytes=max(1, len(payload))
        )
        if verified_payload != payload or verified_identity != identity:
            raise OSError
        _sync_directory(parent)
        if _before_publish is not None:
            _before_publish(identity)
            verified_payload, verified_identity = _stable_bytes(
                temporary, max_bytes=max(1, len(payload))
            )
            if verified_payload != payload or verified_identity != identity:
                raise OSError
        if require_absent:
            guard()
            _move_noreplace_owned(temporary, path, identity)
            published = True
            if _published_identity is not None:
                _published_identity.append(identity)
            _sync_directory(parent)
        elif conditional:
            assert expected_identity is not None
            assert expected_payload is not None
            assert claim_path is not None
            guard()
            _move_noreplace_owned(path, claim_path, expected_identity)
            claim_active = True
            claimed = claim_path.lstat()
            claim_identity = (claimed.st_dev, claimed.st_ino)
            _sync_directory(parent)
            current_payload, current_identity = _stable_bytes(
                claim_path, max_bytes=max(1, len(expected_payload))
            )
            if (
                current_identity != expected_identity
                or current_payload != expected_payload
                or current_identity != claim_identity
            ):
                raise OSError
            guard()
            _move_noreplace_owned(temporary, path, identity)
            published = True
            if _published_identity is not None:
                _published_identity.append(identity)
            _sync_directory(parent)
        else:
            guard()
            os.replace(temporary, path)
            published = True
            if _published_identity is not None:
                _published_identity.append(identity)
            _sync_directory(parent)
        final_payload, final_identity = _stable_bytes(
            path, max_bytes=max(1, len(payload))
        )
        if final_payload != payload or final_identity != identity:
            raise OSError
        if claim_active and not retain_claim:
            if claim_identity is None or not _unlink_owned(
                claim_path, claim_identity, mutation_guard=guard
            ):
                raise OSError
            claim_active = False
        complete = True
        return final_identity
    except (OSError, WebPromotionError):
        raise WebPromotionError("web promotion state update failed") from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if identity is not None and temporary.exists():
            _unlink_owned(temporary, identity)
        if (
            not complete
            and claim_active
            and not published
            and claim_identity is not None
        ):
            try:
                guard()
                _move_noreplace_owned(claim_path, path, claim_identity)
                _sync_directory(parent)
            except (OSError, WebPromotionError):
                pass


def _commit_journal(
    *,
    canonical_path: Path,
    claim_path: Path,
    expected_identity: tuple[int, int],
    payload: bytes,
    verify_state: Callable[[], None],
    error_message: str,
) -> None:
    retiring_path = _journal_retiring_path(claim_path)
    retired_path = retiring_path.parent / (
        f".{retiring_path.name}.{secrets.token_hex(16)}.delete"
    )
    active_path: Path | None = None

    def verify_journal_transition(expected_path: Path) -> None:
        observed = {
            path: _optional_stable_bytes(path, max_bytes=max(1, len(payload)))
            for path in (canonical_path, claim_path, retiring_path)
        }
        if observed[expected_path] != (payload, expected_identity) or any(
            value is not None
            for path, value in observed.items()
            if path != expected_path
        ):
            raise WebPromotionError(error_message)

    try:
        present = [
            path
            for path in (canonical_path, claim_path, retiring_path)
            if path.exists() or path.is_symlink()
        ]
        if len(present) != 1:
            raise WebPromotionError(error_message)
        active_path = present[0]
        observed, observed_identity = _stable_bytes(
            active_path, max_bytes=max(1, len(payload))
        )
        if observed != payload or observed_identity != expected_identity:
            raise WebPromotionError(error_message)
        verify_journal_transition(active_path)
        if active_path == canonical_path:
            verify_state()
            verify_journal_transition(canonical_path)
            _move_noreplace_owned(canonical_path, claim_path, expected_identity)
            active_path = claim_path
            _sync_directory(canonical_path.parent)
            observed, observed_identity = _stable_bytes(
                active_path, max_bytes=max(1, len(payload))
            )
            if observed != payload or observed_identity != expected_identity:
                raise WebPromotionError(error_message)
            verify_journal_transition(claim_path)
        if active_path == claim_path:
            verify_state()
            verify_journal_transition(claim_path)
            _move_noreplace_owned(claim_path, retiring_path, expected_identity)
            active_path = retiring_path
            _sync_directory(canonical_path.parent)
            observed, observed_identity = _stable_bytes(
                active_path, max_bytes=max(1, len(payload))
            )
            if observed != payload or observed_identity != expected_identity:
                raise WebPromotionError(error_message)
            verify_journal_transition(retiring_path)
        if active_path != retiring_path:
            raise WebPromotionError(error_message)
        verify_state()
        observed, observed_identity = _stable_bytes(
            retiring_path, max_bytes=max(1, len(payload))
        )
        if observed != payload or observed_identity != expected_identity:
            raise WebPromotionError(error_message)
        _ensure_retired_capacity(
            retiring_path.parent,
            files=1,
            total_bytes=len(payload),
        )
        verify_state()
        verify_journal_transition(retiring_path)
        _move_noreplace_owned(retiring_path, retired_path, expected_identity)
        active_path = retired_path
        _sync_directory(canonical_path.parent)
        observed, observed_identity = _stable_bytes(
            retired_path, max_bytes=max(1, len(payload))
        )
        if observed != payload or observed_identity != expected_identity:
            raise WebPromotionError(error_message)
        verify_state()
        if any(
            _optional_stable_bytes(path, max_bytes=max(1, len(payload))) is not None
            for path in (canonical_path, claim_path, retiring_path)
        ):
            raise WebPromotionError(error_message)
        observed, observed_identity = _stable_bytes(
            retired_path, max_bytes=max(1, len(payload))
        )
        if observed != payload or observed_identity != expected_identity:
            raise WebPromotionError(error_message)
        _retired_usage(retired_path.parent)
        delete_outcome = _delete_isolated_owned(
            retired_path,
            expected_identity,
            expected_size=len(payload),
            expected_payload=payload,
            mutation_guard=verify_state,
        )
        if delete_outcome != _DELETE_NOT_REMOVED:
            active_path = None
        if delete_outcome == _DELETE_NOT_REMOVED:
            raise WebPromotionError(error_message)
        if delete_outcome != _DELETE_REMOVED_DURABLE:
            raise WebPromotionError("web promotion terminal cleanup state is invalid")
        try:
            terminal_entries_present = any(
                _optional_stable_bytes(path, max_bytes=max(1, len(payload))) is not None
                for path in (
                    canonical_path,
                    claim_path,
                    retiring_path,
                    retired_path,
                )
            )
        except WebPromotionError:
            raise WebPromotionError(
                "web promotion terminal cleanup state is invalid"
            ) from None
        if terminal_entries_present:
            raise WebPromotionError("web promotion terminal cleanup state is invalid")
    except Exception as exc:
        if isinstance(exc, WebPromotionError) and str(exc) in {
            "web promotion lock is invalid",
            "web promotion terminal cleanup state is invalid",
        }:
            raise
        if active_path == retired_path:
            try:
                if not (retiring_path.exists() or retiring_path.is_symlink()):
                    _move_noreplace_owned(
                        retired_path,
                        retiring_path,
                        expected_identity,
                    )
                    _sync_directory(canonical_path.parent)
            except (OSError, WebPromotionError):
                pass
        raise WebPromotionError(error_message) from None


def _selection_payload(values: Mapping[str, str]) -> bytes:
    if tuple(values) != _SPLIT_SELECTION_KEYS:
        raise WebPromotionError("web promotion selection is invalid")
    return "".join(f"{key}={values[key]}\n" for key in _SPLIT_SELECTION_KEYS).encode(
        "utf-8"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _forward_manifest_payload(
    manifest: Mapping[str, object],
    *,
    selection_path: Path,
    selection_payload: bytes,
) -> bytes:
    try:
        value = json.loads(json.dumps(manifest))
        records = value["env_files"]
        matches = [
            item
            for item in records
            if isinstance(item, dict) and item.get("role") == "image-selection"
        ]
        if len(matches) != 1:
            raise ValueError
        record = matches[0]
        if os.path.normcase(os.path.abspath(record["path"])) != os.path.normcase(
            os.fspath(selection_path)
        ):
            raise ValueError
        record["sha256"] = _sha256(selection_payload)
        return _canonical_json(value)
    except (KeyError, TypeError, ValueError):
        raise WebPromotionError("web promotion manifest is invalid") from None


def _valid_journal_progress(
    *,
    phase: object,
    new_selection_identity: object,
    prepared_identity: object,
    new_manifest_identity: object,
    previous_journal_identity: object,
    previous_journal_sha256: object,
) -> bool:
    present = (
        new_selection_identity is not None,
        prepared_identity is not None,
        new_manifest_identity is not None,
    )
    if present not in {
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    }:
        return False
    lineage_present = (
        previous_journal_identity is not None,
        previous_journal_sha256 is not None,
    )
    if lineage_present not in {(False, False), (True, True)}:
        return False
    if phase == "applying":
        return lineage_present == ((True, True) if any(present) else (False, False))
    return phase == "committing" and all(present) and lineage_present == (True, True)


def _journal_payload(
    *,
    old_selection: bytes,
    old_manifest: bytes,
    new_selection: bytes,
    new_manifest: bytes,
    old_web_id: str,
    target_web_id: str,
    web_revision: str,
    api_image_id: str,
    api_container_id: str,
    fixed_resources: Mapping[str, object],
    nonce: str,
    old_selection_identity: tuple[int, int],
    old_manifest_identity: tuple[int, int],
    phase: str = "applying",
    prepared_identity: tuple[int, int] | None = None,
    new_selection_identity: tuple[int, int] | None = None,
    new_manifest_identity: tuple[int, int] | None = None,
    previous_journal_identity: tuple[int, int] | None = None,
    previous_journal_sha256: str | None = None,
) -> bytes:
    identities = (
        old_selection_identity,
        old_manifest_identity,
        prepared_identity,
        new_selection_identity,
        new_manifest_identity,
        previous_journal_identity,
    )
    if (
        _IMAGE_ID.fullmatch(old_web_id) is None
        or _IMAGE_ID.fullmatch(target_web_id) is None
        or _IMAGE_ID.fullmatch(api_image_id) is None
        or _REVISION.fullmatch(web_revision) is None
        or _SAFE_IDENTITY.fullmatch(api_container_id) is None
        or re.fullmatch(r"[0-9a-f]{32}", nonce, re.ASCII) is None
        or phase not in {"applying", "committing"}
        or old_selection_identity is None
        or old_manifest_identity is None
        or any(
            identity is not None
            and (
                type(identity) is not tuple
                or len(identity) != 2
                or any(type(item) is not int or item < 0 for item in identity)
            )
            for identity in identities
        )
        or (previous_journal_identity is None) != (previous_journal_sha256 is None)
        or previous_journal_sha256 is not None
        and (
            type(previous_journal_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", previous_journal_sha256, re.ASCII) is None
        )
        or not _valid_journal_progress(
            phase=phase,
            new_selection_identity=new_selection_identity,
            prepared_identity=prepared_identity,
            new_manifest_identity=new_manifest_identity,
            previous_journal_identity=previous_journal_identity,
            previous_journal_sha256=previous_journal_sha256,
        )
    ):
        raise WebPromotionError("web promotion recovery state is invalid")
    value = {
        "version": 5,
        "phase": phase,
        "nonce": nonce,
        "old_selection": base64.b64encode(old_selection).decode("ascii"),
        "old_manifest": base64.b64encode(old_manifest).decode("ascii"),
        "new_selection": base64.b64encode(new_selection).decode("ascii"),
        "old_selection_sha256": _sha256(old_selection),
        "old_manifest_sha256": _sha256(old_manifest),
        "new_selection_sha256": _sha256(new_selection),
        "new_manifest_sha256": _sha256(new_manifest),
        "old_web_id": old_web_id,
        "target_web_id": target_web_id,
        "target_web_revision": web_revision,
        "api_image_id": api_image_id,
        "api_container_id": api_container_id,
        "fixed_resources": fixed_resources,
        "old_selection_identity": list(old_selection_identity),
        "old_manifest_identity": list(old_manifest_identity),
        "prepared_identity": (
            list(prepared_identity) if prepared_identity is not None else None
        ),
        "new_selection_identity": (
            list(new_selection_identity) if new_selection_identity is not None else None
        ),
        "new_manifest_identity": (
            list(new_manifest_identity) if new_manifest_identity is not None else None
        ),
        "previous_journal_identity": (
            list(previous_journal_identity)
            if previous_journal_identity is not None
            else None
        ),
        "previous_journal_sha256": previous_journal_sha256,
    }
    payload = _canonical_json(value)
    if len(payload) > _MAX_JOURNAL_BYTES:
        raise WebPromotionError("web promotion recovery state is invalid")
    return payload


def _read_journal(
    path: Path,
) -> tuple[dict[str, object], tuple[int, int], bytes]:
    payload, identity = _stable_bytes(path, max_bytes=_MAX_JOURNAL_BYTES)
    try:
        value = _strict_json(payload.decode("utf-8"))
        expected = {
            "version",
            "phase",
            "nonce",
            "old_selection",
            "old_manifest",
            "new_selection",
            "old_selection_sha256",
            "old_manifest_sha256",
            "new_selection_sha256",
            "new_manifest_sha256",
            "old_web_id",
            "target_web_id",
            "target_web_revision",
            "api_image_id",
            "api_container_id",
            "fixed_resources",
            "old_selection_identity",
            "old_manifest_identity",
            "prepared_identity",
            "new_selection_identity",
            "new_manifest_identity",
            "previous_journal_identity",
            "previous_journal_sha256",
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or type(value["version"]) is not int
            or value["version"] != 5
            or value["phase"] not in {"applying", "committing"}
            or type(value["nonce"]) is not str
            or re.fullmatch(r"[0-9a-f]{32}", value["nonce"], re.ASCII) is None
            or type(value["old_web_id"]) is not str
            or _IMAGE_ID.fullmatch(value["old_web_id"]) is None
            or type(value["target_web_id"]) is not str
            or _IMAGE_ID.fullmatch(value["target_web_id"]) is None
            or type(value["target_web_revision"]) is not str
            or _REVISION.fullmatch(value["target_web_revision"]) is None
            or type(value["api_image_id"]) is not str
            or _IMAGE_ID.fullmatch(value["api_image_id"]) is None
            or type(value["api_container_id"]) is not str
            or _SAFE_IDENTITY.fullmatch(value["api_container_id"]) is None
            or type(value["fixed_resources"]) is not dict
        ):
            raise ValueError
        for key, optional in (
            ("old_selection_identity", False),
            ("old_manifest_identity", False),
            ("prepared_identity", True),
            ("new_selection_identity", True),
            ("new_manifest_identity", True),
            ("previous_journal_identity", True),
        ):
            encoded_identity = value[key]
            if optional and encoded_identity is None:
                continue
            if (
                type(encoded_identity) is not list
                or len(encoded_identity) != 2
                or any(type(item) is not int or item < 0 for item in encoded_identity)
            ):
                raise ValueError
        previous_identity = value["previous_journal_identity"]
        previous_sha256 = value["previous_journal_sha256"]
        if (previous_identity is None) != (previous_sha256 is None) or (
            previous_sha256 is not None
            and (
                type(previous_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", previous_sha256, re.ASCII) is None
            )
        ):
            raise ValueError
        if not _valid_journal_progress(
            phase=value["phase"],
            new_selection_identity=value["new_selection_identity"],
            prepared_identity=value["prepared_identity"],
            new_manifest_identity=value["new_manifest_identity"],
            previous_journal_identity=previous_identity,
            previous_journal_sha256=previous_sha256,
        ):
            raise ValueError
        decoded: dict[str, bytes] = {}
        limits = {
            "old_selection": 64 * 1024,
            "new_selection": 64 * 1024,
            "old_manifest": 512 * 1024,
        }
        for key, limit in limits.items():
            if type(value[key]) is not str:
                raise ValueError
            decoded[key] = base64.b64decode(value[key], validate=True)
            if not decoded[key] or len(decoded[key]) > limit:
                raise ValueError
        for key in (
            "old_selection_sha256",
            "old_manifest_sha256",
            "new_selection_sha256",
            "new_manifest_sha256",
        ):
            if (
                type(value[key]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value[key], re.ASCII) is None
            ):
                raise ValueError
        if any(
            value[digest] != _sha256(decoded[source])
            for source, digest in (
                ("old_selection", "old_selection_sha256"),
                ("old_manifest", "old_manifest_sha256"),
                ("new_selection", "new_selection_sha256"),
            )
        ):
            raise ValueError
        state = dict(value)
        state.update(decoded)
        for key in (
            "old_selection_identity",
            "old_manifest_identity",
            "prepared_identity",
            "new_selection_identity",
            "new_manifest_identity",
            "previous_journal_identity",
        ):
            if value[key] is not None:
                state[key] = tuple(value[key])
        return state, identity, payload
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
        WebPromotionError,
    ):
        raise WebPromotionError("web promotion recovery state is invalid") from None


def _is_journal_predecessor(
    current: Mapping[str, object],
    predecessor: Mapping[str, object],
    *,
    predecessor_identity: tuple[int, int],
    predecessor_payload: bytes,
) -> bool:
    if current.get("previous_journal_identity") != predecessor_identity or current.get(
        "previous_journal_sha256"
    ) != _sha256(predecessor_payload):
        return False
    progress = {
        "prepared_identity",
        "new_selection_identity",
        "new_manifest_identity",
    }
    lineage = {"previous_journal_identity", "previous_journal_sha256"}
    if any(
        current.get(key) != predecessor.get(key)
        for key in set(current) | set(predecessor)
        if key not in progress | lineage | {"phase"}
    ):
        return False
    ordered_progress = (
        "new_selection_identity",
        "prepared_identity",
        "new_manifest_identity",
    )
    before = tuple(predecessor.get(key) for key in ordered_progress)
    after = tuple(current.get(key) for key in ordered_progress)
    if any(
        item is not None and item != after[index] for index, item in enumerate(before)
    ):
        return False
    predecessor_phase = predecessor.get("phase")
    current_phase = current.get("phase")
    if predecessor_phase == current_phase == "applying":
        return sum(item is not None for item in after) == (
            sum(item is not None for item in before) + 1
        )
    return (
        predecessor_phase == "applying"
        and current_phase == "committing"
        and all(item is not None for item in before)
        and after == before
    )


def _claim_inventory(runtime: Path) -> dict[tuple[str, str], Path]:
    claims: dict[tuple[str, str], Path] = {}
    try:
        entries = tuple(runtime.iterdir())
    except OSError:
        raise WebPromotionError("web promotion recovery claim is invalid") from None
    for path in entries:
        if not (
            path.name.startswith(".web-promotion-") and path.name.endswith(".claim")
        ):
            continue
        match = _CLAIM_NAME.fullmatch(path.name)
        if match is None:
            raise WebPromotionError("web promotion recovery claim is invalid")
        key = (match.group(1), match.group(2))
        try:
            metadata = path.lstat()
        except OSError:
            raise WebPromotionError("web promotion recovery claim is invalid") from None
        if (
            key in claims
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        ):
            raise WebPromotionError("web promotion recovery claim is invalid")
        claims[key] = path
    return claims


def _retiring_journal_inventory(runtime: Path) -> dict[str, Path]:
    journals: dict[str, Path] = {}
    try:
        entries = tuple(runtime.iterdir())
    except OSError:
        raise WebPromotionError("web promotion recovery state is invalid") from None
    for path in entries:
        if not (
            path.name.startswith(".web-promotion-")
            and path.name.endswith("-journal.retiring")
        ):
            continue
        match = _JOURNAL_RETIRING_NAME.fullmatch(path.name)
        if match is None:
            raise WebPromotionError("web promotion recovery state is invalid")
        nonce = match.group(1)
        try:
            metadata = path.lstat()
        except OSError:
            raise WebPromotionError("web promotion recovery state is invalid") from None
        if (
            nonce in journals
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        ):
            raise WebPromotionError("web promotion recovery state is invalid")
        journals[nonce] = path
    return journals


def _prepared_manifest_inventory(runtime: Path) -> dict[str, Path]:
    prepared: dict[str, Path] = {}
    try:
        entries = tuple(runtime.iterdir())
    except OSError:
        raise WebPromotionError("web promotion recovery state is invalid") from None
    for path in entries:
        if not (
            path.name.startswith(".web-promotion-")
            and path.name.endswith("-manifest.json")
        ):
            continue
        match = _PREPARED_NAME.fullmatch(path.name)
        if match is None:
            raise WebPromotionError("web promotion recovery state is invalid")
        revision = match.group(1)
        try:
            metadata = path.lstat()
        except OSError:
            raise WebPromotionError("web promotion recovery state is invalid") from None
        if (
            revision in prepared
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        ):
            raise WebPromotionError("web promotion recovery state is invalid")
        prepared[revision] = path
    return prepared


def _verify_cleanup_inventory(
    runtime: Path,
    *,
    nonce: str,
    prepared_path: Path,
    error_message: str,
) -> None:
    try:
        if (
            prepared_path.parent != runtime
            or _PREPARED_NAME.fullmatch(prepared_path.name) is None
        ):
            raise WebPromotionError(error_message)
        claims = _claim_inventory(runtime)
        retiring = _retiring_journal_inventory(runtime)
        prepared = _prepared_manifest_inventory(runtime)
    except WebPromotionError:
        raise WebPromotionError(error_message) from None
    if set(claims) - {(nonce, "journal")} or set(retiring) - {nonce} or prepared:
        raise WebPromotionError(error_message)


def _recovery_journal_path(runtime: Path, journal_path: Path) -> Path | None:
    claims = _claim_inventory(runtime)
    retiring = _retiring_journal_inventory(runtime)
    try:
        journal_path.lstat()
    except FileNotFoundError:
        pass
    except OSError:
        raise WebPromotionError("web promotion recovery state is invalid") from None
    else:
        if retiring:
            raise WebPromotionError("web promotion recovery state is invalid")
        return journal_path
    journal_claims = [
        path for (_nonce, role), path in claims.items() if role == "journal"
    ]
    if journal_claims and retiring:
        raise WebPromotionError("web promotion recovery state is invalid")
    if journal_claims:
        if len(journal_claims) != 1:
            raise WebPromotionError("web promotion recovery claim is invalid")
        state, _identity, _payload = _read_journal(journal_claims[0])
        nonce = str(state["nonce"])
        if any(claim_nonce != nonce for claim_nonce, _role in claims):
            raise WebPromotionError("web promotion recovery claim is invalid")
        return journal_claims[0]
    if retiring:
        if len(retiring) != 1:
            raise WebPromotionError("web promotion recovery state is invalid")
        nonce, path = next(iter(retiring.items()))
        state, _identity, _payload = _read_journal(path)
        if state["nonce"] != nonce or any(
            claim_nonce != nonce for claim_nonce, _role in claims
        ):
            raise WebPromotionError("web promotion recovery state is invalid")
        return path
    if not claims:
        return None
    raise WebPromotionError("web promotion recovery claim is invalid")


def _optional_stable_bytes(
    path: Path, *, max_bytes: int
) -> tuple[bytes, tuple[int, int]] | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise WebPromotionError("web promotion state is invalid") from None
    return _stable_bytes(path, max_bytes=max_bytes)


def _inspect_recovery_target(
    path: Path,
    *,
    claim_path: Path,
    old_payload: bytes,
    transaction_payload: bytes,
    old_identity: tuple[int, int],
    transaction_identity: tuple[int, int] | None,
) -> tuple[
    str,
    tuple[bytes, tuple[int, int]] | None,
    tuple[bytes, tuple[int, int]] | None,
]:
    limit = max(1, len(old_payload), len(transaction_payload))
    target = _optional_stable_bytes(path, max_bytes=limit)
    claim = _optional_stable_bytes(claim_path, max_bytes=limit)
    target_kind = None
    if target is not None:
        if target[0] == old_payload and target[1] == old_identity:
            target_kind = "old"
        elif (
            target[0] == transaction_payload
            and transaction_identity is not None
            and target[1] == transaction_identity
        ):
            target_kind = "transaction"
        else:
            raise WebPromotionError("web promotion recovery target drifted")
    claim_kind = None
    if claim is not None:
        if claim[0] == old_payload and claim[1] == old_identity:
            claim_kind = "old"
        elif (
            claim[0] == transaction_payload
            and transaction_identity is not None
            and claim[1] == transaction_identity
        ):
            claim_kind = "transaction"
        else:
            raise WebPromotionError("web promotion recovery claim is invalid")
    if target_kind is None and claim_kind is None:
        raise WebPromotionError("web promotion recovery target drifted")
    if claim_kind is None:
        return ("done" if target_kind == "old" else "replace"), target, claim
    if claim_kind == "old":
        if target_kind is None:
            return "restore-claim", target, claim
        if target_kind == "transaction":
            return "restore-over-transaction", target, claim
        if target is not None and target[1] == claim[1]:
            return "drop-claim", target, claim
        raise WebPromotionError("web promotion recovery claim is invalid")
    if target_kind is None:
        return "publish-old", target, claim
    if target_kind == "old":
        return "drop-claim", target, claim
    if target is not None and target[1] == claim[1]:
        return "publish-old", target, claim
    raise WebPromotionError("web promotion recovery claim is invalid")


def _apply_recovery_target(
    path: Path,
    *,
    claim_path: Path,
    old_payload: bytes,
    transaction_payload: bytes,
    plan: tuple[
        str,
        tuple[bytes, tuple[int, int]] | None,
        tuple[bytes, tuple[int, int]] | None,
    ],
    mutation_guard: Callable[[], None] | None = None,
) -> None:
    action, target, claim = plan
    guard = mutation_guard or (lambda: None)
    if action == "done":
        return
    if action == "replace":
        assert target is not None
        _atomic_replace(
            path,
            old_payload,
            expected_identity=target[1],
            expected_payload=transaction_payload,
            claim_path=claim_path,
            mutation_guard=guard,
        )
        return
    assert claim is not None
    if action == "drop-claim":
        if not _unlink_owned_payload(
            claim_path,
            claim[1],
            claim[0],
            max_bytes=max(1, len(claim[0])),
            mutation_guard=guard,
        ):
            raise WebPromotionError("web promotion recovery cleanup failed")
        return
    if action in {"restore-over-transaction", "publish-old"} and target is not None:
        if not _unlink_owned_payload(
            path,
            target[1],
            target[0],
            max_bytes=max(1, len(target[0])),
            mutation_guard=guard,
        ):
            raise WebPromotionError("web promotion recovery target drifted")
    if action in {"restore-claim", "restore-over-transaction"}:
        try:
            observed, observed_identity = _stable_bytes(
                claim_path, max_bytes=max(1, len(old_payload))
            )
            if observed != old_payload or observed_identity != claim[1]:
                raise OSError
            guard()
            _move_noreplace_owned(claim_path, path, claim[1])
            _sync_directory(path.parent)
            restored, restored_identity = _stable_bytes(
                path, max_bytes=max(1, len(old_payload))
            )
            if restored != old_payload or restored_identity != claim[1]:
                raise OSError
        except (OSError, WebPromotionError):
            raise WebPromotionError("web promotion recovery target drifted") from None
        return
    if action == "publish-old":
        _atomic_replace(
            path,
            old_payload,
            require_absent=True,
            mutation_guard=guard,
        )
        if not _unlink_owned_payload(
            claim_path,
            claim[1],
            transaction_payload,
            max_bytes=max(1, len(transaction_payload)),
            mutation_guard=guard,
        ):
            raise WebPromotionError("web promotion recovery cleanup failed")
        return
    raise WebPromotionError("web promotion recovery state is invalid")


def _clean_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = compose_release._clean_environment(source)
    for name in (
        "DEBUG",
        "PWDEBUG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        environment.pop(name, None)
    return environment


def _pinned_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = _clean_environment(source)
    try:
        endpoint = verify_deployment._require_local_engine(subprocess.run, environment)
        pinned = verify_deployment._pin_local_engine_environment(environment, endpoint)
    except Exception:
        raise WebPromotionError("web promotion Docker engine is invalid") from None
    pinned.pop("DOCKER_CONFIG", None)
    return pinned


def _run_private(
    arguments: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
    timeout: int = 60,
) -> str:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=root,
            env=dict(environment),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise WebPromotionError("web promotion probe failed") from None
    if completed.returncode != 0 or completed.stderr:
        raise WebPromotionError("web promotion probe failed")
    return completed.stdout


def _strict_json(text: str) -> object:
    def pairs(items):
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise WebPromotionError("web promotion probe failed") from None


def _container_projection(
    name: str, *, environment: Mapping[str, str]
) -> dict[str, object]:
    output = compose_release._run_private(
        subprocess.run,
        ["docker", "inspect", name],
        environment,
        timeout=30,
        timeout_message="web promotion probe failed",
    )
    value = _strict_json(output.stdout)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise WebPromotionError("web promotion probe failed")
    item = value[0]
    try:
        health = item["State"]["Health"]["Status"]
        return {
            "id": item["Id"],
            "image": item["Image"],
            "status": item["State"]["Status"],
            "health": health,
            "restart_count": item["RestartCount"],
            "project": item["Config"]["Labels"]["com.docker.compose.project"],
            "service": item["Config"]["Labels"]["com.docker.compose.service"],
        }
    except (KeyError, TypeError):
        raise WebPromotionError("web promotion probe failed") from None


def _bounded_resource_text(
    value: object,
    *,
    max_length: int,
    allow_empty: bool = False,
) -> str:
    if (
        type(value) is not str
        or len(value) > max_length
        or not allow_empty
        and not value
        or any(ord(character) < 32 for character in value)
    ):
        raise WebPromotionError("web promotion resource boundary is invalid")
    return value


def _valid_restart_count(value: object) -> bool:
    return type(value) is int and 0 <= value <= 2**63 - 1


def _snapshot_fixed_resources(
    *,
    environment: Mapping[str, str],
    allowed_web_images: frozenset[str] | None = None,
) -> dict[str, object]:
    listed = compose_release._run_private(
        subprocess.run,
        [
            "docker",
            "ps",
            "-a",
            "--no-trunc",
            "--filter",
            "label=com.docker.compose.project=trainfactory",
            "--format",
            "{{.Names}}",
        ],
        environment,
        timeout=30,
        timeout_message="web promotion probe failed",
    )
    names = frozenset(
        line.strip() for line in listed.stdout.splitlines() if line.strip()
    )
    allowed_fixed_names = frozenset(_FIXED_CONTAINERS)
    if not _REQUIRED_FIXED_CONTAINERS.issubset(
        names
    ) or not names <= allowed_fixed_names | {"trainfactory-web"}:
        raise WebPromotionError("web promotion resource boundary is invalid")
    fixed_names = names & allowed_fixed_names
    containers = {
        name: _container_projection(name, environment=environment)
        for name in sorted(fixed_names)
    }
    if any(
        not isinstance(value["id"], str)
        or _SAFE_IDENTITY.fullmatch(value["id"]) is None
        or not isinstance(value["image"], str)
        or _IMAGE_ID.fullmatch(value["image"]) is None
        or value["status"] != "running"
        or value["health"] != "healthy"
        or not _valid_restart_count(value["restart_count"])
        or value["project"] != "trainfactory"
        or value["service"] != _CONTAINER_SERVICES[name]
        for name, value in containers.items()
    ):
        raise WebPromotionError("web promotion resource boundary is invalid")
    web = None
    if "trainfactory-web" in names:
        web = _container_projection("trainfactory-web", environment=environment)
        if (
            web["project"] != "trainfactory"
            or web["service"] != _CONTAINER_SERVICES["trainfactory-web"]
            or not isinstance(web["id"], str)
            or _SAFE_IDENTITY.fullmatch(web["id"]) is None
            or not isinstance(web["image"], str)
            or _IMAGE_ID.fullmatch(web["image"]) is None
            or not _valid_restart_count(web["restart_count"])
            or allowed_web_images is not None
            and web["image"] not in allowed_web_images
        ):
            raise WebPromotionError("web promotion resource boundary is invalid")
    network_output = compose_release._run_private(
        subprocess.run,
        ["docker", "network", "inspect", "trainfactory_network"],
        environment,
        timeout=30,
        timeout_message="web promotion probe failed",
    )
    network_value = _strict_json(network_output.stdout)
    volume_output = compose_release._run_private(
        subprocess.run,
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            "label=com.docker.compose.project=trainfactory",
            "--format",
            "{{.Name}}",
        ],
        environment,
        timeout=30,
        timeout_message="web promotion probe failed",
    )
    volume_names = sorted(
        line.strip() for line in volume_output.stdout.splitlines() if line.strip()
    )
    observed_volume_names = frozenset(volume_names)
    if (
        not _REQUIRED_FIXED_VOLUMES.issubset(observed_volume_names)
        or not observed_volume_names <= _FIXED_VOLUMES
        or len(volume_names) != len(observed_volume_names)
    ):
        raise WebPromotionError("web promotion resource boundary is invalid")
    inspected_volumes = compose_release._run_private(
        subprocess.run,
        ["docker", "volume", "inspect", *volume_names],
        environment,
        timeout=30,
        timeout_message="web promotion probe failed",
    )
    volume_value = _strict_json(inspected_volumes.stdout)
    if (
        not isinstance(network_value, list)
        or len(network_value) != 1
        or not isinstance(network_value[0], dict)
        or not isinstance(volume_value, list)
        or len(volume_value) != len(observed_volume_names)
        or any(not isinstance(item, dict) for item in volume_value)
        or {item.get("Name") for item in volume_value} != observed_volume_names
    ):
        raise WebPromotionError("web promotion resource boundary is invalid")
    network_source = network_value[0]
    endpoints = network_source.get("Containers", {})
    labels = network_source.get("Labels")
    if (
        not isinstance(endpoints, dict)
        or not isinstance(labels, dict)
        or labels.get("com.docker.compose.project") != "trainfactory"
        or labels.get("com.docker.compose.network") != "default"
    ):
        raise WebPromotionError("web promotion resource boundary is invalid")
    fixed_ids = {value["id"] for value in containers.values()}
    mutable_endpoint_ids = set(endpoints) - fixed_ids
    expected_mutable_endpoint_ids = {web["id"]} if web is not None else set()
    if (
        network_source.get("Name") != "trainfactory_network"
        or not fixed_ids.issubset(endpoints)
        or mutable_endpoint_ids != expected_mutable_endpoint_ids
        or any(not isinstance(value, dict) for value in endpoints.values())
    ):
        raise WebPromotionError("web promotion resource boundary is invalid")
    if web is not None:
        mutable_endpoint = endpoints[web["id"]]
        if mutable_endpoint.get("Name") not in {
            "trainfactory-web",
            "/trainfactory-web",
        }:
            raise WebPromotionError("web promotion resource boundary is invalid")
    try:
        network = {
            "id": _bounded_resource_text(network_source.get("Id"), max_length=128),
            "name": "trainfactory_network",
            "driver": _bounded_resource_text(
                network_source.get("Driver"), max_length=64
            ),
            "scope": _bounded_resource_text(network_source.get("Scope"), max_length=64),
            "internal": network_source["Internal"],
            "attachable": network_source["Attachable"],
            "ingress": network_source["Ingress"],
            "enable_ipv6": network_source["EnableIPv6"],
            "fixed_endpoints": {
                identifier: {
                    "name": _bounded_resource_text(
                        endpoints[identifier].get("Name"), max_length=128
                    ),
                    "endpoint_id": _bounded_resource_text(
                        endpoints[identifier].get("EndpointID"), max_length=128
                    ),
                    "mac_address": _bounded_resource_text(
                        endpoints[identifier].get("MacAddress"), max_length=64
                    ),
                    "ipv4_address": _bounded_resource_text(
                        endpoints[identifier].get("IPv4Address"), max_length=128
                    ),
                    "ipv6_address": _bounded_resource_text(
                        endpoints[identifier].get("IPv6Address"),
                        max_length=128,
                        allow_empty=True,
                    ),
                }
                for identifier in sorted(fixed_ids)
            },
        }
        if any(
            type(network[key]) is not bool
            for key in ("internal", "attachable", "ingress", "enable_ipv6")
        ):
            raise WebPromotionError("web promotion resource boundary is invalid")
        ordered_volumes = []
        for item in sorted(volume_value, key=lambda value: value["Name"]):
            name = item["Name"]
            volume_labels = item.get("Labels")
            if (
                not isinstance(volume_labels, dict)
                or volume_labels.get("com.docker.compose.project") != "trainfactory"
                or volume_labels.get("com.docker.compose.volume")
                != _FIXED_VOLUME_ROLES[name]
            ):
                raise WebPromotionError("web promotion resource boundary is invalid")
            driver = _bounded_resource_text(item.get("Driver"), max_length=64)
            scope = _bounded_resource_text(item.get("Scope"), max_length=64)
            if driver != "local" or scope != "local":
                raise WebPromotionError("web promotion resource boundary is invalid")
            ordered_volumes.append(
                {
                    "name": name,
                    "driver": driver,
                    "scope": scope,
                    "mountpoint": _bounded_resource_text(
                        item.get("Mountpoint"), max_length=4096
                    ),
                    "created_at": _bounded_resource_text(
                        item.get("CreatedAt"), max_length=128
                    ),
                    "project": "trainfactory",
                    "volume": _FIXED_VOLUME_ROLES[name],
                }
            )
    except (KeyError, TypeError):
        raise WebPromotionError("web promotion resource boundary is invalid") from None
    return {
        "containers": containers,
        "network": network,
        "volumes": ordered_volumes,
    }


def _verify_fixed_resources(
    snapshot: Mapping[str, object], *, environment: Mapping[str, str]
) -> None:
    if _snapshot_fixed_resources(environment=environment) != snapshot:
        raise WebPromotionError("web promotion fixed resources drifted")


def _verify_runtime(
    *,
    manifest_path: Path,
    selection_path: Path,
    expected_api_id: str,
    expected_web_id: str,
    expected_api_container_id: str,
    root: Path,
    environment: Mapping[str, str],
) -> None:
    compose_manifest.verify_running_container(
        manifest_path,
        selection_path=selection_path,
        service="train-factory-api",
        root=root,
        base_environment=environment,
    )
    compose_manifest.verify_running_container(
        manifest_path,
        selection_path=selection_path,
        service="train-factory-web",
        root=root,
        base_environment=environment,
    )
    api = _container_projection("trainfactory-api", environment=environment)
    web = _container_projection("trainfactory-web", environment=environment)
    if (
        api["id"] != expected_api_container_id
        or api["image"] != expected_api_id
        or web["image"] != expected_web_id
        or api["status"] != "running"
        or web["status"] != "running"
        or api["health"] != "healthy"
        or web["health"] != "healthy"
        or not _valid_restart_count(api["restart_count"])
        or not _valid_restart_count(web["restart_count"])
    ):
        raise WebPromotionError("web promotion runtime identity is invalid")


def _run_browser_verifier(
    *, root: Path, environment: Mapping[str, str]
) -> dict[str, object]:
    output = _run_private(
        (
            "docker",
            "exec",
            "trainfactory-api",
            "python",
            "-I",
            "-c",
            _PRODUCTION_HTTP_VERIFIER,
        ),
        root=root,
        environment=environment,
        timeout=90,
    )
    value = _strict_json(output.strip())
    expected_keys = {
        "success",
        "code",
        "login_page",
        "api_health",
        "api_proxy",
        "auth_rejection",
        "training_route",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value["success"] is not True
        or value["code"] != "OK"
        or value["login_page"] is not True
        or value["api_health"] is not True
        or value["api_proxy"] is not True
        or value["auth_rejection"] is not True
        or value["training_route"] is not True
    ):
        raise WebPromotionError("web promotion verification failed")
    return {
        "login_page_reachable": True,
        "api_health_reachable": True,
        "api_proxy_reachable": True,
        "auth_rejection_contract_verified": True,
        "training_route_reachable": True,
    }


def _selection_record(manifest: Mapping[str, object], selection_path: Path) -> None:
    try:
        records = manifest["env_files"]
        selection = next(
            item
            for item in records
            if isinstance(item, dict) and item.get("role") == "image-selection"
        )
        if os.path.normcase(os.path.abspath(selection["path"])) != os.path.normcase(
            os.fspath(selection_path)
        ):
            raise ValueError
    except (KeyError, StopIteration, TypeError, ValueError):
        raise WebPromotionError("web promotion manifest is invalid") from None


def _execute_web(
    manifest_path: Path,
    *,
    root: Path,
    environment: Mapping[str, str],
    mutation_guard: Callable[[], None],
) -> None:
    mutation_guard()
    compose_release.execute_manifest(
        manifest_path,
        _TAIL,
        root=root,
        base_environment=environment,
        mutation_guard=mutation_guard,
    )


def _selection_material(payload: bytes) -> dict[str, str]:
    try:
        values = compose_manifest._parse_closed_environment(payload)
        if tuple(values) not in {_SHARED_SELECTION_KEYS, _SPLIT_SELECTION_KEYS}:
            raise ValueError
        compose_manifest._validate_environment_schema(
            payload,
            role="image-selection",
            project="trainfactory",
            secret_mode="files",
            mode="production",
        )
        return values
    except Exception:
        raise WebPromotionError("web promotion recovery state is invalid") from None


def _manifest_material(
    payload: bytes,
    *,
    selection_payload: bytes,
    selection_path: Path,
    manifest_path: Path,
    root: Path,
) -> dict[str, object]:
    expected_manifest_path = root / ".runtime" / "production-compose-manifest.json"
    if os.path.normcase(os.path.abspath(manifest_path)) != os.path.normcase(
        os.fspath(expected_manifest_path)
    ):
        raise WebPromotionError("web promotion recovery state is invalid")
    try:
        value = _strict_json(payload.decode("utf-8"))
        if type(value) is not dict:
            raise ValueError
        compose_manifest._validate_manifest_structure(value)
        if value["mode"] != "production" or value["project"] != "trainfactory":
            raise ValueError
        compose_manifest._validate_ci_manifest_path(
            manifest_path,
            mode="production",
            project="trainfactory",
            root=root,
        )
        compose_files = value["compose_files"]
        env_files = value["env_files"]
        referenced_files = value["referenced_files"]
        bootstrap = root / "docker" / "init.sql"
        compose_manifest._validate_ancestor_chain(bootstrap, boundary=root)
        bootstrap_record, bootstrap_identity, _bootstrap_payload = (
            compose_manifest._record(
                bootstrap, role="database-bootstrap", sensitive=False
            )
        )
        identities = {bootstrap_identity}
        references: list[dict[str, object]] = [bootstrap_record]
        env_payloads: dict[str, bytes] = {}
        image_records = 0
        for item in (*compose_files, *env_files):
            role = item["role"]
            candidate = Path(item["path"])
            compose_manifest._validate_input_path(candidate, role=role, root=root)
            if role == "image-selection":
                image_records += 1
                if os.path.normcase(os.path.abspath(candidate)) != os.path.normcase(
                    os.fspath(selection_path)
                ):
                    raise ValueError
                current_payload = selection_payload
                record = {
                    "role": role,
                    "path": os.fspath(Path(os.path.abspath(candidate))),
                    "sha256": _sha256(current_payload),
                    "sensitive": item["sensitive"],
                }
                identity = None
            else:
                record, identity, current_payload = compose_manifest._record(
                    candidate, role=role, sensitive=item["sensitive"]
                )
            if record != item or identity is not None and identity in identities:
                raise ValueError
            if identity is not None:
                identities.add(identity)
            if item in compose_files:
                compose_manifest._validate_compose_source(current_payload)
            discovered = compose_manifest._validate_role_source(
                candidate,
                current_payload,
                role=role,
                mode=value["mode"],
                project=value["project"],
                secret_mode=value["secret_mode"],
                rollback_variant=value["rollback_variant"],
                root=root,
            )
            if item in env_files:
                env_payloads[role] = current_payload
            for referenced_role, referenced_path in discovered:
                referenced_record, referenced_identity, _referenced_payload = (
                    compose_manifest._record(
                        referenced_path, role=referenced_role, sensitive=True
                    )
                )
                if referenced_identity in identities:
                    raise ValueError
                identities.add(referenced_identity)
                references.append(referenced_record)
        if image_records != 1 or not compose_manifest._records_match(
            references, referenced_files
        ):
            raise ValueError
        compose_manifest._validate_cross_role_identity(
            env_payloads, secret_mode=value["secret_mode"]
        )
        return value
    except Exception:
        raise WebPromotionError("web promotion recovery state is invalid") from None


def _validate_recovery_material(
    state: Mapping[str, object],
    *,
    root: Path,
    selection_path: Path,
    manifest_path: Path,
    environment: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str], bytes]:
    try:
        old_selection = state["old_selection"]
        new_selection = state["new_selection"]
        old_manifest_payload = state["old_manifest"]
        if not all(
            isinstance(item, bytes)
            for item in (old_selection, new_selection, old_manifest_payload)
        ):
            raise ValueError
        old_values = _selection_material(old_selection)
        new_values = _selection_material(new_selection)
        if tuple(new_values) != _SPLIT_SELECTION_KEYS:
            raise ValueError
        old_manifest = _manifest_material(
            old_manifest_payload,
            selection_payload=old_selection,
            selection_path=selection_path,
            manifest_path=manifest_path,
            root=root,
        )
        expected_new_manifest = _forward_manifest_payload(
            old_manifest,
            selection_path=selection_path,
            selection_payload=new_selection,
        )
        old_api_revision = old_values.get(
            "API_REVISION", old_values.get("RELEASE_REVISION")
        )
        if (
            _sha256(expected_new_manifest) != state["new_manifest_sha256"]
            or _sha256(old_selection) != state["old_selection_sha256"]
            or _sha256(old_manifest_payload) != state["old_manifest_sha256"]
            or _sha256(new_selection) != state["new_selection_sha256"]
            or old_values["API_IMAGE"] != new_values["API_IMAGE"]
            or old_values["API_IMAGE_ID"] != new_values["API_IMAGE_ID"]
            or old_api_revision != new_values["API_REVISION"]
            or old_values["WEB_IMAGE_ID"] != state["old_web_id"]
            or new_values["WEB_IMAGE_ID"] != state["target_web_id"]
            or new_values["WEB_REVISION"] != state["target_web_revision"]
            or old_values["API_IMAGE_ID"] != state["api_image_id"]
        ):
            raise ValueError
        nonce = str(state["nonce"])
        old_selection_identity = state["old_selection_identity"]
        old_manifest_identity = state["old_manifest_identity"]
        if not isinstance(old_selection_identity, tuple) or not isinstance(
            old_manifest_identity, tuple
        ):
            raise ValueError
        _inspect_recovery_target(
            selection_path,
            claim_path=_claim_path(root / ".runtime", nonce, "selection"),
            old_payload=old_selection,
            transaction_payload=new_selection,
            old_identity=old_selection_identity,
            transaction_identity=state["new_selection_identity"],
        )
        _inspect_recovery_target(
            manifest_path,
            claim_path=_claim_path(root / ".runtime", nonce, "manifest"),
            old_payload=old_manifest_payload,
            transaction_payload=expected_new_manifest,
            old_identity=old_manifest_identity,
            transaction_identity=state["new_manifest_identity"],
        )
        fixed = _snapshot_fixed_resources(
            environment=environment,
            allowed_web_images=frozenset(
                {str(state["old_web_id"]), str(state["target_web_id"])}
            ),
        )
        if fixed != state["fixed_resources"]:
            raise WebPromotionError("web promotion fixed resources drifted")
        api = _container_projection("trainfactory-api", environment=environment)
        if (
            api["id"] != state["api_container_id"]
            or api["image"] != state["api_image_id"]
        ):
            raise WebPromotionError("web promotion runtime identity is invalid")
        return old_values, new_values, expected_new_manifest
    except WebPromotionError:
        raise
    except Exception:
        raise WebPromotionError("web promotion recovery state is invalid") from None


def _known_target(
    path: Path,
    *,
    old_payload: bytes,
    transaction_payload: bytes,
) -> tuple[str, tuple[int, int]]:
    current_payload, current_identity = _stable_bytes(
        path, max_bytes=max(len(old_payload), len(transaction_payload))
    )
    if current_payload == old_payload:
        return "old", current_identity
    if current_payload == transaction_payload:
        return "transaction", current_identity
    raise WebPromotionError("web promotion recovery target drifted")


def _restore_transaction_target(
    path: Path,
    *,
    claim_path: Path,
    old_payload: bytes,
    transaction_payload: bytes,
    old_identity: tuple[int, int],
    owned_identity: tuple[int, int] | None,
    mutation_guard: Callable[[], None] | None = None,
) -> None:
    plan = _inspect_recovery_target(
        path,
        claim_path=claim_path,
        old_payload=old_payload,
        transaction_payload=transaction_payload,
        old_identity=old_identity,
        transaction_identity=owned_identity,
    )
    claim = plan[2]
    if (
        claim is not None
        and claim != (old_payload, old_identity)
        or plan[0] in {"replace", "publish-old"}
    ):
        raise WebPromotionError("web promotion rollback target drifted")
    _apply_recovery_target(
        path,
        claim_path=claim_path,
        old_payload=old_payload,
        transaction_payload=transaction_payload,
        plan=plan,
        mutation_guard=mutation_guard,
    )


def _recover(
    *,
    root: Path,
    journal_path: Path,
    selection_path: Path,
    manifest_path: Path,
    environment: Mapping[str, str],
    mutation_guard: Callable[[], None] | None = None,
) -> None:
    guard = mutation_guard or (lambda: None)
    state, journal_identity, journal_payload = _read_journal(journal_path)
    old_values, new_values, new_manifest = _validate_recovery_material(
        state,
        root=root,
        selection_path=selection_path,
        manifest_path=manifest_path,
        environment=environment,
    )
    old_selection = state["old_selection"]
    old_manifest = state["old_manifest"]
    new_selection = state["new_selection"]
    assert isinstance(old_selection, bytes)
    assert isinstance(old_manifest, bytes)
    assert isinstance(new_selection, bytes)
    runtime = root / ".runtime"
    nonce = str(state["nonce"])
    selection_claim = _claim_path(runtime, nonce, "selection")
    manifest_claim = _claim_path(runtime, nonce, "manifest")
    journal_claim = _claim_path(runtime, nonce, "journal")
    journal_retiring = _journal_retiring_path(journal_claim)
    canonical_journal = runtime / "web-promotion-active.json"
    claims = _claim_inventory(runtime)
    if any(claim_nonce != nonce for claim_nonce, _role in claims):
        raise WebPromotionError("web promotion recovery claim is invalid")
    if journal_path not in {canonical_journal, journal_claim, journal_retiring}:
        raise WebPromotionError("web promotion recovery state is invalid")
    duplicate_journal_claim: tuple[bytes, tuple[int, int]] | None = None
    if journal_path == journal_claim:
        if _optional_stable_bytes(
            canonical_journal, max_bytes=512 * 1024
        ) is not None or _retiring_journal_inventory(runtime):
            raise WebPromotionError("web promotion recovery claim is invalid")
    elif journal_path == journal_retiring:
        if (
            _optional_stable_bytes(canonical_journal, max_bytes=512 * 1024) is not None
            or _optional_stable_bytes(journal_claim, max_bytes=512 * 1024) is not None
            or _retiring_journal_inventory(runtime) != {nonce: journal_retiring}
        ):
            raise WebPromotionError("web promotion recovery state is invalid")
    elif journal_claim in claims.values():
        predecessor, duplicate_identity, duplicate_payload = _read_journal(
            journal_claim
        )
        same_inode_duplicate = (
            duplicate_payload == journal_payload
            and duplicate_identity == journal_identity
        )
        if not same_inode_duplicate and not _is_journal_predecessor(
            state,
            predecessor,
            predecessor_identity=duplicate_identity,
            predecessor_payload=duplicate_payload,
        ):
            raise WebPromotionError("web promotion recovery claim is invalid")
        duplicate_journal_claim = duplicate_payload, duplicate_identity
    old_selection_identity = state["old_selection_identity"]
    old_manifest_identity = state["old_manifest_identity"]
    if not isinstance(old_selection_identity, tuple) or not isinstance(
        old_manifest_identity, tuple
    ):
        raise WebPromotionError("web promotion recovery state is invalid")
    prepared_path = (
        runtime / f'.web-promotion-{state["target_web_revision"]}-manifest.json'
    )
    prepared = _optional_stable_bytes(prepared_path, max_bytes=512 * 1024)
    try:
        prepared_inventory = _prepared_manifest_inventory(runtime)
    except WebPromotionError:
        raise WebPromotionError("web promotion recovery cleanup failed") from None
    if set(prepared_inventory) - {str(state["target_web_revision"])}:
        raise WebPromotionError("web promotion recovery cleanup failed")
    if prepared is not None:
        prepared_identity = state["prepared_identity"]
        if (
            not isinstance(prepared_identity, tuple)
            or prepared[0] != new_manifest
            or prepared[1] != prepared_identity
        ):
            raise WebPromotionError("web promotion recovery cleanup failed")
    selection_plan = _inspect_recovery_target(
        selection_path,
        claim_path=selection_claim,
        old_payload=old_selection,
        transaction_payload=new_selection,
        old_identity=old_selection_identity,
        transaction_identity=state["new_selection_identity"],
    )
    manifest_plan = _inspect_recovery_target(
        manifest_path,
        claim_path=manifest_claim,
        old_payload=old_manifest,
        transaction_payload=new_manifest,
        old_identity=old_manifest_identity,
        transaction_identity=state["new_manifest_identity"],
    )
    observed_journal, observed_journal_identity = _stable_bytes(
        journal_path, max_bytes=512 * 1024
    )
    if (
        observed_journal_identity != journal_identity
        or observed_journal != journal_payload
    ):
        raise WebPromotionError("web promotion recovery state is invalid")

    def verify_recovery_journal_authority() -> None:
        guard()
        current_payload, current_identity = _stable_bytes(
            journal_path, max_bytes=512 * 1024
        )
        if current_identity != journal_identity or current_payload != journal_payload:
            raise WebPromotionError("web promotion recovery state is invalid")
        if journal_path == canonical_journal:
            current_claim = _optional_stable_bytes(journal_claim, max_bytes=512 * 1024)
            if current_claim != duplicate_journal_claim or _retiring_journal_inventory(
                runtime
            ):
                raise WebPromotionError("web promotion recovery state is invalid")
        elif journal_path == journal_claim:
            if _optional_stable_bytes(
                canonical_journal, max_bytes=512 * 1024
            ) is not None or _retiring_journal_inventory(runtime):
                raise WebPromotionError("web promotion recovery state is invalid")
        elif (
            _optional_stable_bytes(canonical_journal, max_bytes=512 * 1024) is not None
            or _optional_stable_bytes(journal_claim, max_bytes=512 * 1024) is not None
            or _retiring_journal_inventory(runtime) != {nonce: journal_retiring}
        ):
            raise WebPromotionError("web promotion recovery state is invalid")

    def verify_duplicate_journal_cleanup_authority() -> None:
        guard()
        current_payload, current_identity = _stable_bytes(
            journal_path, max_bytes=512 * 1024
        )
        current_claim = _optional_stable_bytes(journal_claim, max_bytes=512 * 1024)
        if (
            journal_path != canonical_journal
            or current_identity != journal_identity
            or current_payload != journal_payload
            or current_claim not in (None, duplicate_journal_claim)
            or _retiring_journal_inventory(runtime)
        ):
            raise WebPromotionError("web promotion recovery state is invalid")

    if state["phase"] == "committing":
        new_selection_identity = state["new_selection_identity"]
        new_manifest_identity = state["new_manifest_identity"]
        prepared_identity = state["prepared_identity"]
        if not all(
            isinstance(identity, tuple)
            for identity in (
                new_selection_identity,
                new_manifest_identity,
                prepared_identity,
            )
        ):
            raise WebPromotionError("web promotion recovery state is invalid")
        selection_target, selection_old_claim = selection_plan[1:]
        manifest_target, manifest_old_claim = manifest_plan[1:]
        if (
            selection_target != (new_selection, new_selection_identity)
            or manifest_target != (new_manifest, new_manifest_identity)
            or selection_old_claim is not None
            and selection_old_claim != (old_selection, old_selection_identity)
            or manifest_old_claim is not None
            and manifest_old_claim != (old_manifest, old_manifest_identity)
        ):
            raise WebPromotionError("web promotion commit state drifted")

        def verify_committed_state() -> None:
            guard()
            current_selection, current_selection_identity = _stable_bytes(
                selection_path, max_bytes=max(1, len(new_selection))
            )
            current_manifest, current_manifest_identity = _stable_bytes(
                manifest_path, max_bytes=max(1, len(new_manifest))
            )
            if (
                current_selection != new_selection
                or current_selection_identity != new_selection_identity
                or current_manifest != new_manifest
                or current_manifest_identity != new_manifest_identity
            ):
                raise WebPromotionError("web promotion commit state drifted")
            _verify_fixed_resources(state["fixed_resources"], environment=environment)

        def verify_committing_authority() -> None:
            verify_recovery_journal_authority()
            verify_committed_state()

        verify_committing_authority()
        manifest = compose_manifest.verify_manifest_inputs_only(
            manifest_path, root=root, base_environment=environment
        )
        if compose_manifest._selection_values(manifest) != new_values:
            raise WebPromotionError("web promotion recovery verification failed")
        _execute_web(
            manifest_path,
            root=root,
            environment=environment,
            mutation_guard=verify_committing_authority,
        )
        _verify_fixed_resources(state["fixed_resources"], environment=environment)
        _verify_runtime(
            manifest_path=manifest_path,
            selection_path=selection_path,
            expected_api_id=str(state["api_image_id"]),
            expected_web_id=str(state["target_web_id"]),
            expected_api_container_id=str(state["api_container_id"]),
            root=root,
            environment=environment,
        )
        _run_browser_verifier(root=root, environment=environment)
        verify_committing_authority()

        for path, identity, payload, limit in (
            (
                selection_claim,
                old_selection_identity,
                old_selection,
                64 * 1024,
            ),
            (
                manifest_claim,
                old_manifest_identity,
                old_manifest,
                512 * 1024,
            ),
            (prepared_path, prepared_identity, new_manifest, 512 * 1024),
        ):
            if path.exists() or path.is_symlink():
                if not _unlink_owned_payload(
                    path,
                    identity,
                    payload,
                    max_bytes=limit,
                    mutation_guard=verify_committing_authority,
                ):
                    raise WebPromotionError("web promotion recovery cleanup failed")
        if duplicate_journal_claim is not None:
            duplicate_payload, duplicate_identity = duplicate_journal_claim
            if not _unlink_owned_payload(
                journal_claim,
                duplicate_identity,
                duplicate_payload,
                max_bytes=512 * 1024,
                mutation_guard=verify_duplicate_journal_cleanup_authority,
            ):
                raise WebPromotionError("web promotion recovery cleanup failed")

        def verify_committed_cleanup_state() -> None:
            verify_committed_state()
            _verify_cleanup_inventory(
                runtime,
                nonce=nonce,
                prepared_path=prepared_path,
                error_message="web promotion recovery cleanup failed",
            )

        verify_committed_cleanup_state()
        _commit_journal(
            canonical_path=canonical_journal,
            claim_path=journal_claim,
            expected_identity=journal_identity,
            payload=journal_payload,
            verify_state=verify_committed_cleanup_state,
            error_message="web promotion recovery cleanup failed",
        )
        return

    for plan, old_payload, old_identity in (
        (selection_plan, old_selection, old_selection_identity),
        (manifest_plan, old_manifest, old_manifest_identity),
    ):
        claim = plan[2]
        if (
            claim is not None
            and claim != (old_payload, old_identity)
            or plan[0] in {"replace", "publish-old"}
        ):
            raise WebPromotionError("web promotion recovery claim is invalid")
    verify_recovery_journal_authority()
    _apply_recovery_target(
        selection_path,
        claim_path=selection_claim,
        old_payload=old_selection,
        transaction_payload=new_selection,
        plan=selection_plan,
        mutation_guard=verify_recovery_journal_authority,
    )
    verify_recovery_journal_authority()
    _apply_recovery_target(
        manifest_path,
        claim_path=manifest_claim,
        old_payload=old_manifest,
        transaction_payload=new_manifest,
        plan=manifest_plan,
        mutation_guard=verify_recovery_journal_authority,
    )

    def verify_recovered_targets() -> None:
        current_selection, current_selection_identity = _stable_bytes(
            selection_path, max_bytes=max(1, len(old_selection))
        )
        current_manifest, current_manifest_identity = _stable_bytes(
            manifest_path, max_bytes=max(1, len(old_manifest))
        )
        if (
            current_selection != old_selection
            or current_selection_identity != old_selection_identity
            or current_manifest != old_manifest
            or current_manifest_identity != old_manifest_identity
        ):
            raise WebPromotionError("web promotion recovery target drifted")

    def verify_applying_authority() -> None:
        verify_recovery_journal_authority()
        verify_recovered_targets()
        _verify_fixed_resources(state["fixed_resources"], environment=environment)

    verify_applying_authority()
    manifest = compose_manifest.verify_manifest_inputs_only(
        manifest_path, root=root, base_environment=environment
    )
    verify_applying_authority()
    values = compose_manifest._selection_values(manifest)
    if values != old_values:
        raise WebPromotionError("web promotion recovery verification failed")
    _execute_web(
        manifest_path,
        root=root,
        environment=environment,
        mutation_guard=verify_applying_authority,
    )
    verify_applying_authority()
    _verify_fixed_resources(state["fixed_resources"], environment=environment)
    _verify_runtime(
        manifest_path=manifest_path,
        selection_path=selection_path,
        expected_api_id=str(state["api_image_id"]),
        expected_web_id=str(state["old_web_id"]),
        expected_api_container_id=str(state["api_container_id"]),
        root=root,
        environment=environment,
    )
    recovered_selection, recovered_selection_identity = _stable_bytes(
        selection_path, max_bytes=max(1, len(old_selection))
    )
    recovered_manifest, recovered_manifest_identity = _stable_bytes(
        manifest_path, max_bytes=max(1, len(old_manifest))
    )
    if (
        recovered_selection != old_selection
        or recovered_selection_identity != old_selection_identity
        or recovered_manifest != old_manifest
        or recovered_manifest_identity != old_manifest_identity
    ):
        raise WebPromotionError("web promotion recovery target drifted")

    def verify_recovered_state() -> None:
        guard()
        verify_recovered_targets()
        _verify_fixed_resources(state["fixed_resources"], environment=environment)

    cleanup_ok = True
    verify_recovery_journal_authority()
    if prepared_path.exists() or prepared_path.is_symlink():
        prepared_identity = state["prepared_identity"]
        if not isinstance(prepared_identity, tuple) or not _unlink_owned_payload(
            prepared_path,
            prepared_identity,
            new_manifest,
            max_bytes=512 * 1024,
            mutation_guard=verify_applying_authority,
        ):
            cleanup_ok = False
    if cleanup_ok:
        try:
            remaining_claims = _claim_inventory(runtime)
        except WebPromotionError:
            cleanup_ok = False
        else:
            allowed = {journal_claim} if journal_path == journal_claim else set()
            if duplicate_journal_claim is not None:
                allowed.add(journal_claim)
            if set(remaining_claims.values()) - allowed:
                cleanup_ok = False
    if cleanup_ok and duplicate_journal_claim is not None:
        verify_recovery_journal_authority()
        duplicate_payload, duplicate_identity = duplicate_journal_claim
        if not _unlink_owned_payload(
            journal_claim,
            duplicate_identity,
            duplicate_payload,
            max_bytes=512 * 1024,
            mutation_guard=verify_duplicate_journal_cleanup_authority,
        ):
            cleanup_ok = False
    if cleanup_ok:
        try:

            def verify_recovered_cleanup_state() -> None:
                verify_recovered_state()
                _verify_cleanup_inventory(
                    runtime,
                    nonce=nonce,
                    prepared_path=prepared_path,
                    error_message="web promotion recovery cleanup failed",
                )

            verify_recovered_cleanup_state()
            _commit_journal(
                canonical_path=canonical_journal,
                claim_path=journal_claim,
                expected_identity=journal_identity,
                payload=journal_payload,
                verify_state=verify_recovered_cleanup_state,
                error_message="web promotion recovery cleanup failed",
            )
        except WebPromotionError as exc:
            if str(exc) in {
                "web promotion lock is invalid",
                "web promotion terminal cleanup state is invalid",
            }:
                raise
            cleanup_ok = False
    if not cleanup_ok:
        raise WebPromotionError("web promotion recovery cleanup failed")


def promote_web_release(
    *,
    root: Path,
    web_image: str,
    web_image_id: str,
    web_revision: str,
    base_environment: Mapping[str, str] = os.environ,
) -> dict[str, object]:
    root = Path(os.path.abspath(root))
    if (
        _IMAGE_REF.fullmatch(web_image) is None
        or ".." in web_image
        or "//" in web_image
        or _IMAGE_ID.fullmatch(web_image_id) is None
        or _REVISION.fullmatch(web_revision) is None
    ):
        raise WebPromotionError("web promotion arguments are invalid")
    runtime = root / ".runtime"
    selection_path = runtime / "release.env"
    manifest_path = runtime / "production-compose-manifest.json"
    journal_path = runtime / "web-promotion-active.json"
    lock_path = runtime / "web-promotion.lock"
    prepared_path = runtime / f".web-promotion-{web_revision}-manifest.json"
    environment = _pinned_environment(base_environment)
    runtime_identity = _private_runtime_identity(root, runtime)
    lock_descriptor = -1

    def verify_lock() -> None:
        _verify_private_runtime(runtime, runtime_identity)
        if lock_descriptor < 0:
            raise WebPromotionError("web promotion lock is invalid")
        _verify_process_lock(lock_path, lock_descriptor)

    try:
        lock_descriptor = _acquire_process_lock(lock_path)
        verify_lock()
        _reject_crash_temporaries(runtime)
        _ensure_retired_capacity(
            runtime,
            files=_PROMOTION_RETIRED_FILE_RESERVE,
            total_bytes=_PROMOTION_RETIRED_BYTE_RESERVE,
        )
        recovery_journal = _recovery_journal_path(runtime, journal_path)
        if recovery_journal is not None:
            try:
                _recover(
                    root=root,
                    journal_path=recovery_journal,
                    selection_path=selection_path,
                    manifest_path=manifest_path,
                    environment=environment,
                    mutation_guard=verify_lock,
                )
            except WebPromotionError as exc:
                if str(exc) == "web promotion lock is invalid":
                    raise
                raise WebPromotionError(
                    "web promotion startup recovery failed"
                ) from None
            except Exception:
                raise WebPromotionError(
                    "web promotion startup recovery failed"
                ) from None
        if _claim_inventory(runtime):
            raise WebPromotionError("web promotion recovery claim is invalid")
        _verify_process_lock(lock_path, lock_descriptor)
        _ensure_retired_capacity(
            runtime,
            files=_PROMOTION_RETIRED_FILE_RESERVE,
            total_bytes=_PROMOTION_RETIRED_BYTE_RESERVE,
        )
        if _prepared_manifest_inventory(runtime):
            raise WebPromotionError("web promotion prepared state already exists")
        old_selection, old_selection_identity = _stable_bytes(
            selection_path, max_bytes=64 * 1024
        )
        old_manifest_payload, old_manifest_identity = _stable_bytes(
            manifest_path, max_bytes=512 * 1024
        )
        old_manifest = compose_manifest.verify_manifest_inputs_only(
            manifest_path, root=root, base_environment=environment
        )
        if (
            old_manifest.get("mode") != "production"
            or old_manifest.get("project") != "trainfactory"
        ):
            raise WebPromotionError("web promotion manifest is invalid")
        _selection_record(old_manifest, selection_path)
        old_values = compose_manifest._selection_values(old_manifest)
        expected_api_id = old_values["API_IMAGE_ID"]
        old_web_id = old_values["WEB_IMAGE_ID"]
        api = _container_projection("trainfactory-api", environment=environment)
        expected_api_container_id = str(api["id"])
        _verify_runtime(
            manifest_path=manifest_path,
            selection_path=selection_path,
            expected_api_id=expected_api_id,
            expected_web_id=old_web_id,
            expected_api_container_id=expected_api_container_id,
            root=root,
            environment=environment,
        )
        fixed = _snapshot_fixed_resources(environment=environment)
        new_values = compose_manifest.build_forward_web_selection(
            production_manifest=manifest_path,
            web_image=web_image,
            web_image_id=web_image_id,
            web_revision=web_revision,
            root=root,
            base_environment=environment,
        )
        new_selection = _selection_payload(new_values)
        final_selection, final_selection_identity = _stable_bytes(
            selection_path, max_bytes=64 * 1024
        )
        final_manifest, final_manifest_identity = _stable_bytes(
            manifest_path, max_bytes=512 * 1024
        )
        if (
            final_selection != old_selection
            or final_selection_identity != old_selection_identity
            or final_manifest != old_manifest_payload
            or final_manifest_identity != old_manifest_identity
        ):
            raise WebPromotionError("web promotion input drifted")
        new_manifest_payload = _forward_manifest_payload(
            old_manifest,
            selection_path=selection_path,
            selection_payload=new_selection,
        )
        _verify_fixed_resources(fixed, environment=environment)
        _verify_runtime(
            manifest_path=manifest_path,
            selection_path=selection_path,
            expected_api_id=expected_api_id,
            expected_web_id=old_web_id,
            expected_api_container_id=expected_api_container_id,
            root=root,
            environment=environment,
        )
        nonce = secrets.token_hex(16)
        selection_claim = _claim_path(runtime, nonce, "selection")
        manifest_claim = _claim_path(runtime, nonce, "manifest")
        journal_claim = _claim_path(runtime, nonce, "journal")

        def journal_payload(
            *,
            recorded_phase: str = "applying",
            recorded_prepared_identity: tuple[int, int] | None = None,
            recorded_selection_identity: tuple[int, int] | None = None,
            recorded_manifest_identity: tuple[int, int] | None = None,
            previous_identity: tuple[int, int] | None = None,
            previous_payload: bytes | None = None,
        ) -> bytes:
            return _journal_payload(
                old_selection=old_selection,
                old_manifest=old_manifest_payload,
                new_selection=new_selection,
                new_manifest=new_manifest_payload,
                old_web_id=old_web_id,
                target_web_id=web_image_id,
                web_revision=web_revision,
                api_image_id=expected_api_id,
                api_container_id=expected_api_container_id,
                fixed_resources=fixed,
                nonce=nonce,
                old_selection_identity=old_selection_identity,
                old_manifest_identity=old_manifest_identity,
                phase=recorded_phase,
                prepared_identity=recorded_prepared_identity,
                new_selection_identity=recorded_selection_identity,
                new_manifest_identity=recorded_manifest_identity,
                previous_journal_identity=previous_identity,
                previous_journal_sha256=(
                    _sha256(previous_payload) if previous_payload is not None else None
                ),
            )

        current_journal_payload = journal_payload()
        _verify_process_lock(lock_path, lock_descriptor)
        journal_identity = _atomic_replace(
            journal_path,
            current_journal_payload,
            require_absent=True,
            mutation_guard=verify_lock,
        )
        prepared_identity: tuple[int, int] | None = None
        owned_selection_identity: tuple[int, int] | None = None
        owned_manifest_identity: tuple[int, int] | None = None
        published_selection_identities: list[tuple[int, int]] = []
        published_manifest_identities: list[tuple[int, int]] = []
        predecessor_journal_identity: tuple[int, int] | None = None
        predecessor_journal_payload: bytes | None = None
        web_execution_attempted = False
        commit_decided = False

        def publish_journal(
            updated_payload: bytes,
            *,
            mutation_guard: Callable[[], None] | None = None,
        ) -> None:
            nonlocal current_journal_payload
            nonlocal journal_identity
            nonlocal predecessor_journal_identity
            nonlocal predecessor_journal_payload
            previous_identity = journal_identity
            previous_payload = current_journal_payload
            published: list[tuple[int, int]] = []

            def verify_publish_boundary() -> None:
                verify_lock()
                canonical = _optional_stable_bytes(journal_path, max_bytes=512 * 1024)
                claim = _optional_stable_bytes(journal_claim, max_bytes=512 * 1024)
                previous = (previous_payload, previous_identity)
                if not (
                    canonical == previous
                    and claim is None
                    or canonical is None
                    and claim == previous
                ):
                    raise WebPromotionError("web promotion recovery state is invalid")
                if mutation_guard is not None:
                    mutation_guard()

            verify_publish_boundary()
            try:
                updated_identity = _atomic_replace(
                    journal_path,
                    updated_payload,
                    expected_identity=previous_identity,
                    expected_payload=previous_payload,
                    claim_path=journal_claim,
                    retain_claim=True,
                    mutation_guard=verify_publish_boundary,
                    _published_identity=published,
                )
            except Exception:
                if len(published) == 1:
                    predecessor_journal_identity = previous_identity
                    predecessor_journal_payload = previous_payload
                    journal_identity = published[0]
                    current_journal_payload = updated_payload
                raise
            predecessor_journal_identity = previous_identity
            predecessor_journal_payload = previous_payload
            journal_identity = updated_identity
            current_journal_payload = updated_payload
            cleanup_journal_predecessor()

        def verify_current_journal_authority() -> None:
            verify_lock()
            canonical = _optional_stable_bytes(journal_path, max_bytes=512 * 1024)
            claim = _optional_stable_bytes(journal_claim, max_bytes=512 * 1024)
            expected_claim = (
                None
                if predecessor_journal_identity is None
                or predecessor_journal_payload is None
                else (predecessor_journal_payload, predecessor_journal_identity)
            )
            if canonical != (current_journal_payload, journal_identity) or (
                claim != expected_claim
            ):
                raise WebPromotionError("web promotion recovery state is invalid")

        def cleanup_journal_predecessor() -> None:
            nonlocal predecessor_journal_identity
            nonlocal predecessor_journal_payload
            if journal_claim.exists() or journal_claim.is_symlink():
                verify_current_journal_authority()
                if (
                    predecessor_journal_identity is None
                    or predecessor_journal_payload is None
                ):
                    raise WebPromotionError("web promotion cleanup failed")

                expected_predecessor = (
                    predecessor_journal_payload,
                    predecessor_journal_identity,
                )

                def verify_cleanup_window() -> None:
                    verify_lock()
                    canonical = _optional_stable_bytes(
                        journal_path, max_bytes=512 * 1024
                    )
                    claim = _optional_stable_bytes(journal_claim, max_bytes=512 * 1024)
                    if canonical != (current_journal_payload, journal_identity) or (
                        claim is not None and claim != expected_predecessor
                    ):
                        raise WebPromotionError(
                            "web promotion recovery state is invalid"
                        )

                if not _unlink_owned_payload(
                    journal_claim,
                    predecessor_journal_identity,
                    predecessor_journal_payload,
                    max_bytes=512 * 1024,
                    mutation_guard=verify_cleanup_window,
                ):
                    raise WebPromotionError("web promotion cleanup failed")
                predecessor_journal_identity = None
                predecessor_journal_payload = None

        def verify_active_journal_progress(
            *,
            expected_prepared_identity: tuple[int, int] | None,
            expected_selection_identity: tuple[int, int] | None,
            expected_manifest_identity: tuple[int, int] | None,
            expected_phase: str = "applying",
        ) -> None:
            verify_current_journal_authority()
            observed, observed_identity, observed_payload = _read_journal(journal_path)
            if (
                observed_identity != journal_identity
                or observed_payload != current_journal_payload
                or observed["phase"] != expected_phase
                or observed["prepared_identity"] != expected_prepared_identity
                or observed["new_selection_identity"] != expected_selection_identity
                or observed["new_manifest_identity"] != expected_manifest_identity
            ):
                raise WebPromotionError("web promotion recovery state is invalid")

        try:

            def record_selection_candidate(identity: tuple[int, int]) -> None:
                nonlocal owned_selection_identity
                owned_selection_identity = identity
                updated_payload = journal_payload(
                    recorded_selection_identity=identity,
                    previous_identity=journal_identity,
                    previous_payload=current_journal_payload,
                )
                publish_journal(updated_payload)

            _verify_process_lock(lock_path, lock_descriptor)
            owned_selection_identity = _atomic_replace(
                selection_path,
                new_selection,
                expected_identity=old_selection_identity,
                expected_payload=old_selection,
                claim_path=selection_claim,
                retain_claim=True,
                mutation_guard=lambda: verify_active_journal_progress(
                    expected_prepared_identity=None,
                    expected_selection_identity=owned_selection_identity,
                    expected_manifest_identity=None,
                ),
                _before_publish=record_selection_candidate,
                _published_identity=published_selection_identities,
            )
            verify_active_journal_progress(
                expected_prepared_identity=None,
                expected_selection_identity=owned_selection_identity,
                expected_manifest_identity=None,
            )

            def record_prepared_candidate(identity: tuple[int, int]) -> None:
                nonlocal prepared_identity
                prepared_identity = identity
                updated_payload = journal_payload(
                    recorded_prepared_identity=identity,
                    recorded_selection_identity=owned_selection_identity,
                    previous_identity=journal_identity,
                    previous_payload=current_journal_payload,
                )
                publish_journal(updated_payload)

            prepared_identity = _atomic_replace(
                prepared_path,
                new_manifest_payload,
                require_absent=True,
                mutation_guard=lambda: verify_active_journal_progress(
                    expected_prepared_identity=prepared_identity,
                    expected_selection_identity=owned_selection_identity,
                    expected_manifest_identity=None,
                ),
                _before_publish=record_prepared_candidate,
            )
            prepared_payload, verified_prepared_identity = _stable_bytes(
                prepared_path, max_bytes=512 * 1024
            )
            if (
                verified_prepared_identity != prepared_identity
                or prepared_payload != new_manifest_payload
            ):
                raise WebPromotionError("web promotion prepared state is invalid")
            verify_active_journal_progress(
                expected_prepared_identity=prepared_identity,
                expected_selection_identity=owned_selection_identity,
                expected_manifest_identity=None,
            )
            compose_manifest.verify_manifest_inputs_only(
                prepared_path, root=root, base_environment=environment
            )

            def record_manifest_candidate(identity: tuple[int, int]) -> None:
                nonlocal owned_manifest_identity
                owned_manifest_identity = identity
                updated_payload = journal_payload(
                    recorded_prepared_identity=prepared_identity,
                    recorded_selection_identity=owned_selection_identity,
                    recorded_manifest_identity=identity,
                    previous_identity=journal_identity,
                    previous_payload=current_journal_payload,
                )
                publish_journal(updated_payload)

            _verify_process_lock(lock_path, lock_descriptor)
            owned_manifest_identity = _atomic_replace(
                manifest_path,
                prepared_payload,
                expected_identity=old_manifest_identity,
                expected_payload=old_manifest_payload,
                claim_path=manifest_claim,
                retain_claim=True,
                mutation_guard=lambda: verify_active_journal_progress(
                    expected_prepared_identity=prepared_identity,
                    expected_selection_identity=owned_selection_identity,
                    expected_manifest_identity=owned_manifest_identity,
                ),
                _before_publish=record_manifest_candidate,
                _published_identity=published_manifest_identities,
            )
            verify_active_journal_progress(
                expected_prepared_identity=prepared_identity,
                expected_selection_identity=owned_selection_identity,
                expected_manifest_identity=owned_manifest_identity,
            )
            compose_manifest.verify_manifest_inputs_only(
                manifest_path, root=root, base_environment=environment
            )
            if owned_selection_identity is None or owned_manifest_identity is None:
                raise WebPromotionError("web promotion state update failed")

            def verify_new_canonical_state() -> None:
                verify_lock()
                current_selection, current_selection_identity = _stable_bytes(
                    selection_path, max_bytes=max(1, len(new_selection))
                )
                current_manifest, current_manifest_identity = _stable_bytes(
                    manifest_path, max_bytes=max(1, len(new_manifest_payload))
                )
                if (
                    current_selection != new_selection
                    or current_selection_identity != owned_selection_identity
                    or current_manifest != new_manifest_payload
                    or current_manifest_identity != owned_manifest_identity
                ):
                    raise WebPromotionError("web promotion commit state drifted")

            def verify_new_committed_state() -> None:
                verify_new_canonical_state()
                _verify_fixed_resources(fixed, environment=environment)

            def verify_new_application_authority() -> None:
                verify_active_journal_progress(
                    expected_prepared_identity=prepared_identity,
                    expected_selection_identity=owned_selection_identity,
                    expected_manifest_identity=owned_manifest_identity,
                )
                verify_new_committed_state()

            verify_new_application_authority()
            web_execution_attempted = True
            _execute_web(
                manifest_path,
                root=root,
                environment=environment,
                mutation_guard=verify_new_application_authority,
            )
            _verify_fixed_resources(fixed, environment=environment)
            _verify_runtime(
                manifest_path=manifest_path,
                selection_path=selection_path,
                expected_api_id=expected_api_id,
                expected_web_id=web_image_id,
                expected_api_container_id=expected_api_container_id,
                root=root,
                environment=environment,
            )
            verify_new_application_authority()
            browser = _run_browser_verifier(root=root, environment=environment)
            compose_manifest.verify_manifest_inputs_only(
                manifest_path, root=root, base_environment=environment
            )
            _verify_fixed_resources(fixed, environment=environment)
            _verify_runtime(
                manifest_path=manifest_path,
                selection_path=selection_path,
                expected_api_id=expected_api_id,
                expected_web_id=web_image_id,
                expected_api_container_id=expected_api_container_id,
                root=root,
                environment=environment,
            )
            verify_new_application_authority()

            def verify_commit_transition_state() -> None:
                verify_new_committed_state()
                current_prepared, current_prepared_identity = _stable_bytes(
                    prepared_path, max_bytes=512 * 1024
                )
                if (
                    current_prepared != new_manifest_payload
                    or current_prepared_identity != prepared_identity
                ):
                    raise WebPromotionError("web promotion prepared state is invalid")

            verify_commit_transition_state()
            committing_payload = journal_payload(
                recorded_phase="committing",
                recorded_prepared_identity=prepared_identity,
                recorded_selection_identity=owned_selection_identity,
                recorded_manifest_identity=owned_manifest_identity,
                previous_identity=journal_identity,
                previous_payload=current_journal_payload,
            )
            try:
                publish_journal(
                    committing_payload,
                    mutation_guard=verify_commit_transition_state,
                )
            except Exception:
                commit_decided = current_journal_payload == committing_payload
                raise
            commit_decided = True

            def verify_new_commit_authority() -> None:
                verify_active_journal_progress(
                    expected_prepared_identity=prepared_identity,
                    expected_selection_identity=owned_selection_identity,
                    expected_manifest_identity=owned_manifest_identity,
                    expected_phase="committing",
                )
                verify_new_committed_state()

            verify_new_commit_authority()
            if not _unlink_owned_payload(
                selection_claim,
                old_selection_identity,
                old_selection,
                max_bytes=64 * 1024,
                mutation_guard=verify_new_commit_authority,
            ):
                raise WebPromotionError("web promotion cleanup failed")
            if not _unlink_owned_payload(
                manifest_claim,
                old_manifest_identity,
                old_manifest_payload,
                max_bytes=512 * 1024,
                mutation_guard=verify_new_commit_authority,
            ):
                raise WebPromotionError("web promotion cleanup failed")
            if not _unlink_owned_payload(
                prepared_path,
                prepared_identity,
                new_manifest_payload,
                max_bytes=512 * 1024,
                mutation_guard=verify_new_commit_authority,
            ):
                raise WebPromotionError("web promotion cleanup failed")
            cleanup_journal_predecessor()

            def verify_new_cleanup_state() -> None:
                verify_new_committed_state()
                _verify_cleanup_inventory(
                    runtime,
                    nonce=nonce,
                    prepared_path=prepared_path,
                    error_message="web promotion cleanup failed",
                )

            verify_new_cleanup_state()
            _commit_journal(
                canonical_path=journal_path,
                claim_path=journal_claim,
                expected_identity=journal_identity,
                payload=current_journal_payload,
                verify_state=verify_new_cleanup_state,
                error_message="web promotion cleanup failed",
            )
        except Exception as exc:
            if isinstance(exc, WebPromotionError) and str(exc) == (
                "web promotion terminal cleanup state is invalid"
            ):
                raise
            if commit_decided:
                raise WebPromotionError(
                    "web promotion committed; cleanup is incomplete"
                ) from None
            try:
                if (
                    owned_selection_identity is None
                    and len(published_selection_identities) == 1
                ):
                    owned_selection_identity = published_selection_identities[0]
                if (
                    owned_manifest_identity is None
                    and len(published_manifest_identities) == 1
                ):
                    owned_manifest_identity = published_manifest_identities[0]

                def verify_rollback_authority() -> None:
                    verify_lock()
                    _verify_fixed_resources(fixed, environment=environment)
                    canonical = _optional_stable_bytes(
                        journal_path, max_bytes=512 * 1024
                    )
                    claim = _optional_stable_bytes(journal_claim, max_bytes=512 * 1024)
                    expected_claim = (
                        None
                        if predecessor_journal_identity is None
                        or predecessor_journal_payload is None
                        else (
                            predecessor_journal_payload,
                            predecessor_journal_identity,
                        )
                    )
                    if canonical != (current_journal_payload, journal_identity) or (
                        claim != expected_claim
                        and not (expected_claim is not None and claim is None)
                    ):
                        raise WebPromotionError(
                            "web promotion recovery state is invalid"
                        )

                verify_rollback_authority()
                _restore_transaction_target(
                    selection_path,
                    claim_path=selection_claim,
                    old_payload=old_selection,
                    transaction_payload=new_selection,
                    old_identity=old_selection_identity,
                    owned_identity=owned_selection_identity,
                    mutation_guard=verify_rollback_authority,
                )
                verify_rollback_authority()
                _restore_transaction_target(
                    manifest_path,
                    claim_path=manifest_claim,
                    old_payload=old_manifest_payload,
                    transaction_payload=new_manifest_payload,
                    old_identity=old_manifest_identity,
                    owned_identity=owned_manifest_identity,
                    mutation_guard=verify_rollback_authority,
                )
                compose_manifest.verify_manifest_inputs_only(
                    manifest_path, root=root, base_environment=environment
                )

                def verify_old_commit_state() -> None:
                    verify_lock()
                    current_selection, current_selection_identity = _stable_bytes(
                        selection_path, max_bytes=max(1, len(old_selection))
                    )
                    current_manifest, current_manifest_identity = _stable_bytes(
                        manifest_path, max_bytes=max(1, len(old_manifest_payload))
                    )
                    if (
                        current_selection != old_selection
                        or current_selection_identity != old_selection_identity
                        or current_manifest != old_manifest_payload
                        or current_manifest_identity != old_manifest_identity
                    ):
                        raise WebPromotionError("web promotion rollback target drifted")
                    _verify_fixed_resources(fixed, environment=environment)

                def verify_old_rollback_authority() -> None:
                    verify_rollback_authority()
                    verify_old_commit_state()

                verify_old_rollback_authority()
                if web_execution_attempted:
                    _execute_web(
                        manifest_path,
                        root=root,
                        environment=environment,
                        mutation_guard=verify_old_rollback_authority,
                    )
                _verify_fixed_resources(fixed, environment=environment)
                _verify_runtime(
                    manifest_path=manifest_path,
                    selection_path=selection_path,
                    expected_api_id=expected_api_id,
                    expected_web_id=old_web_id,
                    expected_api_container_id=expected_api_container_id,
                    root=root,
                    environment=environment,
                )
                restored_selection, restored_selection_identity = _stable_bytes(
                    selection_path, max_bytes=max(1, len(old_selection))
                )
                restored_manifest, restored_manifest_identity = _stable_bytes(
                    manifest_path, max_bytes=max(1, len(old_manifest_payload))
                )
                if (
                    restored_selection != old_selection
                    or restored_selection_identity != old_selection_identity
                    or restored_manifest != old_manifest_payload
                    or restored_manifest_identity != old_manifest_identity
                ):
                    raise WebPromotionError("web promotion rollback target drifted")
                verify_old_rollback_authority()
                if prepared_path.exists() or prepared_path.is_symlink():
                    if prepared_identity is None or not _unlink_owned_payload(
                        prepared_path,
                        prepared_identity,
                        new_manifest_payload,
                        max_bytes=512 * 1024,
                        mutation_guard=verify_old_rollback_authority,
                    ):
                        raise WebPromotionError("web promotion cleanup failed")
                cleanup_journal_predecessor()

                def verify_old_cleanup_state() -> None:
                    verify_old_commit_state()
                    _verify_cleanup_inventory(
                        runtime,
                        nonce=nonce,
                        prepared_path=prepared_path,
                        error_message="web promotion cleanup failed",
                    )

                verify_old_cleanup_state()
                _commit_journal(
                    canonical_path=journal_path,
                    claim_path=journal_claim,
                    expected_identity=journal_identity,
                    payload=current_journal_payload,
                    verify_state=verify_old_cleanup_state,
                    error_message="web promotion cleanup failed",
                )
            except Exception as exc:
                if isinstance(exc, WebPromotionError) and str(exc) == (
                    "web promotion terminal cleanup state is invalid"
                ):
                    raise
                raise WebPromotionError(
                    "web promotion failed; automatic rollback also failed"
                ) from None
            raise WebPromotionError("web promotion failed; old web restored") from None
        return {
            "release": "web-only",
            "success": True,
            "api_container_unchanged": True,
            "api_image_id": expected_api_id,
            "web_image_id": web_image_id,
            "api_revision": new_values["API_REVISION"],
            "web_revision": new_values["WEB_REVISION"],
            "fixed_resources_unchanged": True,
            **browser,
        }
    except WebPromotionError:
        raise
    except Exception:
        raise WebPromotionError("web promotion failed") from None
    finally:
        if lock_descriptor >= 0:
            _release_process_lock(lock_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        result = promote_web_release(
            root=ROOT_DIR,
            web_image=arguments.web_image,
            web_image_id=arguments.web_image_id,
            web_revision=arguments.web_revision,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except WebPromotionError as exc:
        internal_message = str(exc)
        operator_messages = {
            "web promotion committed; cleanup is incomplete": (
                "web promotion committed; recovery is required before retry"
            ),
            "web promotion failed; automatic rollback also failed": (
                "web promotion state is uncertain; recovery is required before retry"
            ),
            "web promotion failed; old web restored": (
                "web promotion rejected; old web was restored"
            ),
            "web promotion temporary state requires inspection": (
                "web promotion blocked; temporary state requires inspection"
            ),
            "web promotion startup recovery failed": (
                "web promotion recovery failed; manual inspection is required before retry"
            ),
            "web promotion terminal cleanup state is invalid": (
                "web promotion terminal state requires manual inspection before retry"
            ),
            "web promotion lock is invalid": (
                "web promotion terminal state requires manual inspection before retry"
            ),
        }
        operator_message = operator_messages.get(internal_message)
        print(operator_message or "web promotion failed", file=sys.stderr)
        return 1
    except Exception:
        print("web promotion failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
