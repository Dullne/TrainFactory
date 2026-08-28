"""Build, verify, and clean one isolated CPU Compose smoke deployment."""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("compose smoke bootstrap is unavailable")


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
sys.path[:] = [_root_entry, *_trusted_sys_path]

import argparse  # noqa: E402
from datetime import datetime  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from urllib.parse import quote, quote_plus, urlsplit  # noqa: E402

from scripts.check_pytest_partitions import _run_process_tree  # noqa: E402


ROOT_DIR = Path(_root_entry)
RUN_ID = re.compile(r"^[0-9a-f]{32}$", re.ASCII)
REVISION = re.compile(r"^[0-9a-f]{40}$", re.ASCII)
DIGEST_IMAGE = re.compile(r"^[^\s=@]+(?::[^\s=@]+)?@sha256:[0-9a-f]{64}$", re.ASCII)
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$", re.ASCII)
LOCK_KEYS = {
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
ALEMBIC_HEAD = "058_add_model_artifact_membership_gate"
BUILD_TIMEOUT = 1800
COMMAND_TIMEOUT = 900
PROBE_TIMEOUT = 30
IMAGE_VERSION = "0.1.0-ci-smoke"
STAGING_MEMBERS = frozenset(
    {
        "mysql_root_password",
        "mysql_app_password",
        "mysql_url",
        "jwt_secret_key",
        "default_admin_password",
        "admin_username",
        "compose-secrets.env",
        "compose-secrets.state.json",
    }
)


class SmokeError(RuntimeError):
    """Raised with a fixed private failure boundary."""


class _Parser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise SmokeError("compose smoke failed")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--images-lock", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--vcs-date", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--no-gpu", action="store_true", required=True)
    parser.add_argument("--github-actions-mask", action="store_true")
    return parser


def _base_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "PROGRAMDATA",
        "XDG_RUNTIME_DIR",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "DOCKER_CONFIG",
        "PROGRAMFILES",
        "PROGRAMW6432",
        "PROGRAMFILES(X86)",
        "GITHUB_ACTIONS",
    }
    return {
        name: value
        for name, value in os.environ.items()
        if name.upper() in allowed
    }


def _is_reparse(path: Path, metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & 0x400
    )


def _validate_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _is_reparse(path, metadata)
            or os.path.normcase(os.path.realpath(path))
            != os.path.normcase(os.fspath(path))
        ):
            raise SmokeError("compose smoke failed")
    except OSError:
        raise SmokeError("compose smoke failed") from None


def _read_iid(path: Path) -> tuple[str, tuple[int, int]]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse(path, metadata)
            or metadata.st_size > 128
        ):
            raise SmokeError("compose smoke failed")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            payload = os.read(descriptor, 129)
            final = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = path.lstat()
        identities = {
            (item.st_dev, item.st_ino, item.st_size)
            for item in (metadata, opened, final, current)
        }
        image_id = payload.decode("ascii").removesuffix("\n")
        if (
            len(identities) != 1
            or len(payload) > 128
            or payload not in {image_id.encode("ascii"), image_id.encode("ascii") + b"\n"}
            or IMAGE_ID.fullmatch(image_id) is None
        ):
            raise SmokeError("compose smoke failed")
        return image_id, (metadata.st_dev, metadata.st_ino)
    except (OSError, UnicodeError):
        raise SmokeError("compose smoke failed") from None


def _completed(
    run: Callable[..., subprocess.CompletedProcess[str]],
    arguments: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = run(
            list(arguments),
            cwd=root,
            env=dict(environment),
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise SmokeError("compose smoke failed") from None
    if completed.returncode != 0 or completed.stderr:
        raise SmokeError("compose smoke failed")
    return completed


def _local_engine(
    run: Callable[..., subprocess.CompletedProcess[str]],
    *,
    root: Path,
) -> dict[str, str]:
    from scripts.verify_deployment import _local_engine_endpoint

    environment = _base_environment()
    declared = environment.get("DOCKER_HOST")
    try:
        normalized_declared = (
            _local_engine_endpoint(declared) if declared is not None else None
        )
    except RuntimeError:
        raise SmokeError("compose smoke failed") from None
    completed = _completed(
        run,
        ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
        root=root,
        environment=environment,
        timeout=PROBE_TIMEOUT,
    )
    lines = completed.stdout.splitlines()
    if len(lines) != 1 or not lines[0] or len(lines[0]) > 512:
        raise SmokeError("compose smoke failed")
    try:
        endpoint = _local_engine_endpoint(lines[0])
    except RuntimeError:
        raise SmokeError("compose smoke failed") from None
    if normalized_declared is not None and endpoint != normalized_declared:
        raise SmokeError("compose smoke failed")
    environment["DOCKER_HOST"] = endpoint
    for name in ("DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(name, None)
    return environment


def _read_stable_regular(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[bytes, tuple[int, int]]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse(path, metadata)
            or metadata.st_size > max_bytes
            or os.path.normcase(os.path.realpath(path))
            != os.path.normcase(os.fspath(path))
        ):
            raise SmokeError("compose smoke failed")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            payload = os.read(descriptor, max_bytes + 1)
            os.lseek(descriptor, 0, os.SEEK_SET)
            repeated_payload = os.read(descriptor, max_bytes + 1)
            final = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = path.lstat()
        snapshots = (metadata, opened, final, current)
        if (
            len(payload) > max_bytes
            or repeated_payload != payload
            or len(
                {
                    (
                        item.st_dev,
                        item.st_ino,
                        item.st_size,
                        item.st_mtime_ns,
                        item.st_ctime_ns,
                    )
                    for item in snapshots
                }
            )
            != 1
        ):
            raise SmokeError("compose smoke failed")
        return payload, (metadata.st_dev, metadata.st_ino)
    except OSError:
        raise SmokeError("compose smoke failed") from None


def _read_lock(path: Path) -> tuple[dict[str, str], bytes, tuple[int, int]]:
    payload, identity = _read_stable_regular(path, max_bytes=64 * 1024)
    try:
        text = payload.decode("utf-8")
    except UnicodeError:
        raise SmokeError("compose smoke failed") from None
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.count("=") != 1:
            raise SmokeError("compose smoke failed")
        key, value = line.split("=", 1)
        if key in values or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise SmokeError("compose smoke failed")
        if DIGEST_IMAGE.fullmatch(value) is None:
            raise SmokeError("compose smoke failed")
        values[key] = value
    if set(values) != LOCK_KEYS:
        raise SmokeError("compose smoke failed")
    return values, payload, identity


def _verify_lock(
    path: Path,
    *,
    payload: bytes,
    identity: tuple[int, int],
) -> None:
    current_payload, current_identity = _read_stable_regular(
        path,
        max_bytes=64 * 1024,
    )
    if current_payload != payload or current_identity != identity:
        raise SmokeError("compose smoke failed")


def _stable_hardened_secret(path: Path) -> tuple[bytes, tuple[int, int]]:
    from scripts.materialize_compose_secrets import _verify_hardened_path

    if not _verify_hardened_path(path):
        raise SmokeError("compose smoke failed")
    payload, identity = _read_stable_regular(path, max_bytes=4096)
    if not _verify_hardened_path(path):
        raise SmokeError("compose smoke failed")
    final_payload, final_identity = _read_stable_regular(path, max_bytes=4096)
    if (
        final_payload != payload
        or final_identity != identity
        or not _verify_hardened_path(path)
    ):
        raise SmokeError("compose smoke failed")
    return payload, identity


def _secret_mask_values(
    bundle: Path,
) -> tuple[tuple[str, ...], tuple[tuple[Path, bytes, tuple[int, int]], ...]]:
    values: list[str] = []
    snapshots: list[tuple[Path, bytes, tuple[int, int]]] = []
    for name in (
        "mysql_root_password",
        "mysql_app_password",
        "mysql_url",
        "jwt_secret_key",
        "default_admin_password",
    ):
        path = bundle / name
        payload, identity = _stable_hardened_secret(path)
        snapshots.append((path, payload, identity))
        try:
            value = payload.decode("utf-8")
        except UnicodeError:
            raise SmokeError("compose smoke failed") from None
        if value.endswith("\r\n"):
            value = value[:-2]
        elif value.endswith("\n"):
            value = value[:-1]
        if (
            not value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise SmokeError("compose smoke failed")
        values.append(value)
    masked: list[str] = []
    for value in values:
        for candidate in (value, quote(value, safe=""), quote_plus(value, safe="")):
            if candidate and candidate not in masked:
                masked.append(candidate)
    return tuple(masked), tuple(snapshots)


def _verify_secret_snapshots(
    snapshots: tuple[tuple[Path, bytes, tuple[int, int]], ...],
) -> None:
    for path, payload, identity in snapshots:
        final_payload, final_identity = _stable_hardened_secret(path)
        if final_payload != payload or final_identity != identity:
            raise SmokeError("compose smoke failed")


def _github_mask_sink(value: str) -> None:
    print(f"::add-mask::{value.replace('%', '%25')}", flush=True)


def _strict_json(text: str) -> object:
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise SmokeError("compose smoke failed") from None


def _inspect_image(
    tag: str,
    *,
    revision: str,
    vcs_date: str,
    source: str,
    run: Callable[..., subprocess.CompletedProcess[str]],
    root: Path,
    environment: Mapping[str, str],
) -> str:
    completed = _completed(
        run,
        ["docker", "image", "inspect", "--format", "{{json .}}", tag],
        root=root,
        environment=environment,
        timeout=PROBE_TIMEOUT,
    )
    value = _strict_json(completed.stdout)
    if type(value) is not dict:
        raise SmokeError("compose smoke failed")
    image_id = value.get("Id")
    tags = value.get("RepoTags")
    config = value.get("Config")
    labels = config.get("Labels") if type(config) is dict else None
    expected_labels = {
        "org.opencontainers.image.version": IMAGE_VERSION,
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.created": vcs_date,
        "org.opencontainers.image.source": source,
    }
    if (
        type(image_id) is not str
        or IMAGE_ID.fullmatch(image_id) is None
        or type(tags) is not list
        or tag not in tags
        or value.get("Os") != "linux"
        or value.get("Architecture") != "amd64"
        or type(labels) is not dict
        or any(labels.get(key) != expected for key, expected in expected_labels.items())
    ):
        raise SmokeError("compose smoke failed")
    return image_id


def _record_file(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if _is_reparse(path, metadata) or not stat.S_ISREG(metadata.st_mode):
        raise SmokeError("compose smoke failed")
    return metadata.st_dev, metadata.st_ino


def _record_directory(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if _is_reparse(path, metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise SmokeError("compose smoke failed")
    return metadata.st_dev, metadata.st_ino


def _cleanup_owned_staging_windows(path: Path, identity: tuple[int, int]) -> bool:
    import ctypes
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(UnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", ctypes.c_void_p),
            ("SecurityQualityOfService", ctypes.c_void_p),
        ]

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOL)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    ]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    ntdll.NtCreateFile.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        ctypes.c_void_p,
        wintypes.ULONG,
    ]
    ntdll.NtCreateFile.restype = ctypes.c_long

    def handle_identity(handle) -> tuple[tuple[int, int], int] | None:
        information = ByHandleFileInformation()
        if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
            return None
        file_index = (information.nFileIndexHigh << 32) | information.nFileIndexLow
        return (
            (information.dwVolumeSerialNumber, file_index),
            information.dwFileAttributes,
        )

    def mark_delete(handle) -> bool:
        disposition = FileDispositionInfo(True)
        return bool(
            kernel32.SetFileInformationByHandle(
                handle,
                4,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            )
        )

    directory = kernel32.CreateFileW(
        os.fspath(path),
        0x00010000 | 0x00000080 | 0x00100000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if directory in {None, invalid_handle}:
        return not path.exists() and not path.is_symlink()
    try:
        directory_information = handle_identity(directory)
        if (
            directory_information is None
            or directory_information[0] != identity
            or directory_information[1] & 0x400
            or not directory_information[1] & 0x10
        ):
            return False
        try:
            names = [entry.name for entry in path.iterdir()]
        except OSError:
            return False
        if len(names) != len(set(names)) or any(
            name not in STAGING_MEMBERS for name in names
        ):
            return False
        for name in names:
            buffer = ctypes.create_unicode_buffer(name)
            name_length = len(name.encode("utf-16-le"))
            unicode_name = UnicodeString(
                name_length,
                name_length + 2,
                ctypes.cast(buffer, wintypes.LPWSTR),
            )
            attributes = ObjectAttributes(
                ctypes.sizeof(ObjectAttributes),
                directory,
                ctypes.pointer(unicode_name),
                0x40,
                None,
                None,
            )
            status_block = IoStatusBlock()
            member = wintypes.HANDLE()
            status = ntdll.NtCreateFile(
                ctypes.byref(member),
                0x00010000 | 0x00000080 | 0x00100000,
                ctypes.byref(attributes),
                ctypes.byref(status_block),
                None,
                0,
                0x00000001 | 0x00000002 | 0x00000004,
                1,
                0x00000020 | 0x00000040 | 0x00200000,
                None,
                0,
            )
            if status < 0 or not member:
                return False
            try:
                member_information = handle_identity(member)
                if (
                    member_information is None
                    or member_information[1] & (0x10 | 0x400)
                    or not mark_delete(member)
                ):
                    return False
            finally:
                kernel32.CloseHandle(member)
        try:
            current_path = path.lstat()
            path_is_owned = (current_path.st_dev, current_path.st_ino) == identity
        except FileNotFoundError:
            path_is_owned = False
        except OSError:
            return False
        if not mark_delete(directory):
            return False
        return path_is_owned
    finally:
        kernel32.CloseHandle(directory)


def _rename_noreplace_posix(
    parent: int,
    source_name: str,
    target_name: str,
) -> bool:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    return (
        renameat2(
            parent,
            os.fsencode(source_name),
            parent,
            os.fsencode(target_name),
            1,
        )
        == 0
    )


def _cleanup_owned_staging_posix(path: Path, identity: tuple[int, int]) -> bool:
    parent = -1
    directory = -1
    quarantine_name = f".{path.name}.{secrets.token_hex(16)}.cleanup"
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent = os.open(path.parent, flags)
        directory = os.open(path.name, flags, dir_fd=parent)
    except FileNotFoundError:
        if parent >= 0:
            os.close(parent)
        return True
    except OSError:
        if parent >= 0:
            os.close(parent)
        return False
    try:
        metadata = os.fstat(directory)
        if (metadata.st_dev, metadata.st_ino) != identity or not stat.S_ISDIR(
            metadata.st_mode
        ):
            return False
        if not _rename_noreplace_posix(parent, path.name, quarantine_name):
            return False
        try:
            quarantine = os.stat(
                quarantine_name,
                dir_fd=parent,
                follow_symlinks=False,
            )
        except OSError:
            _rename_noreplace_posix(parent, quarantine_name, path.name)
            return False
        if (
            (quarantine.st_dev, quarantine.st_ino) != identity
            or not stat.S_ISDIR(quarantine.st_mode)
        ):
            _rename_noreplace_posix(parent, quarantine_name, path.name)
            return False
        names = os.listdir(directory)
        if len(names) != len(set(names)) or any(
            name not in STAGING_MEMBERS for name in names
        ):
            return False
        for name in names:
            member = -1
            try:
                before = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if not stat.S_ISREG(before.st_mode):
                    return False
                member = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory,
                )
                current = os.fstat(member)
                if (
                    (before.st_dev, before.st_ino, before.st_size)
                    != (current.st_dev, current.st_ino, current.st_size)
                    or not stat.S_ISREG(current.st_mode)
                ):
                    return False
                os.unlink(name, dir_fd=directory)
            finally:
                if member >= 0:
                    os.close(member)
        final_directory = os.fstat(directory)
        final_quarantine = os.stat(
            quarantine_name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            (final_directory.st_dev, final_directory.st_ino) != identity
            or (final_quarantine.st_dev, final_quarantine.st_ino) != identity
            or not stat.S_ISDIR(final_quarantine.st_mode)
        ):
            return False
        os.rmdir(quarantine_name, dir_fd=parent)
        try:
            os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return False
    except OSError:
        return False
    finally:
        if directory >= 0:
            os.close(directory)
        if parent >= 0:
            os.close(parent)


def _cleanup_owned_staging(path: Path, identity: tuple[int, int]) -> bool:
    if os.name == "nt":
        return _cleanup_owned_staging_windows(path, identity)
    return _cleanup_owned_staging_posix(path, identity)


def _recover_selection_identity(
    path: Path,
    *,
    expected: Mapping[str, str],
    root: Path,
) -> tuple[int, int] | None:
    try:
        from scripts.compose_manifest import _parse_environment_file, _read_stable

        payload, identity = _read_stable(path, max_bytes=64 * 1024)
        if _parse_environment_file(payload) != dict(expected):
            return None
        final_payload, final_identity = _read_stable(path, max_bytes=64 * 1024)
        if final_payload != payload or final_identity != identity:
            return None
        return identity
    except Exception:
        return None


def _recover_manifest_identity(
    path: Path,
    *,
    project: str,
    secret_env: Path,
    lock_path: Path,
    selection: Path,
    root: Path,
) -> tuple[int, int] | None:
    try:
        from scripts.compose_manifest import (
            _matches_exact_path,
            _read_stable,
            verify_manifest_inputs,
        )

        payload, identity = _read_stable(path, max_bytes=256 * 1024)
        initial = _strict_json(payload.decode("utf-8"))
        if type(initial) is not dict:
            return None
        manifest = verify_manifest_inputs(path, root=root)
        records = manifest.get("env_files")
        expected = (
            ("secret-paths", secret_env),
            ("images-lock", lock_path),
            ("image-selection", selection),
        )
        if (
            manifest.get("mode") != "ci"
            or manifest != initial
            or manifest.get("project") != project
            or manifest.get("gpu_mode") != "cpu"
            or manifest.get("secret_mode") != "files"
            or type(records) is not list
            or len(records) != len(expected)
        ):
            return None
        for record, (role, expected_path) in zip(records, expected, strict=True):
            if (
                type(record) is not dict
                or record.get("role") != role
                or not _matches_exact_path(
                    Path(str(record.get("path", ""))),
                    expected_path,
                    root=root,
                )
            ):
                return None
        final_payload, final_identity = _read_stable(path, max_bytes=256 * 1024)
        if final_payload != payload or final_identity != identity:
            return None
        return identity
    except Exception:
        return None


def _unlink_owned(path: Path, identity: tuple[int, int] | None) -> bool:
    if identity is None:
        return not path.exists() and not path.is_symlink()
    try:
        metadata = path.lstat()
        if (metadata.st_dev, metadata.st_ino) != identity:
            return False
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _require_empty_project(
    project: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    root: Path,
    environment: Mapping[str, str],
) -> None:
    commands = (
        ["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"],
        ["docker", "volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
        ["docker", "network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
    )
    for command in commands:
        completed = _completed(
            run,
            command,
            root=root,
            environment=environment,
            timeout=PROBE_TIMEOUT,
        )
        if completed.stdout:
            raise SmokeError("compose smoke failed")
    exact_names = (
        *(f"{project}-{service}-1" for service in ("mysql", "train-factory-api", "train-factory-web")),
        *(f"{project}_{volume}" for volume in ("mysql_data", "train_cache", "verify_data", "verify_models", "verify_output")),
        f"{project}_default",
    )
    exact_commands = (
        *(
            ["docker", "container", "ls", "-aq", "--filter", f"name=^/{name}$"]
            for name in exact_names[:3]
        ),
        *(
            ["docker", "volume", "ls", "-q", "--filter", f"name=^{name}$"]
            for name in exact_names[3:8]
        ),
        ["docker", "network", "ls", "-q", "--filter", f"name=^{exact_names[8]}$"],
    )
    for command in exact_commands:
        if _completed(
            run,
            command,
            root=root,
            environment=environment,
            timeout=PROBE_TIMEOUT,
        ).stdout:
            raise SmokeError("compose smoke failed")


def _tag_absent(
    tag: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    root: Path,
    environment: Mapping[str, str],
) -> None:
    completed = _completed(
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
        root=root,
        environment=environment,
        timeout=PROBE_TIMEOUT,
    )
    if completed.stdout:
        raise SmokeError("compose smoke failed")


def _remove_owned_image(
    tag: str,
    expected_id: str | None,
    *,
    revision: str,
    vcs_date: str,
    source: str,
    run: Callable[..., subprocess.CompletedProcess[str]],
    root: Path,
    environment: Mapping[str, str],
) -> bool:
    try:
        listing = _completed(
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
            root=root,
            environment=environment,
            timeout=PROBE_TIMEOUT,
        )
        if listing.stdout == "":
            return True
        if expected_id is None:
            return False
        if listing.stdout != f"{tag}\n":
            return False
        current_id = _inspect_image(
            tag,
            revision=revision,
            vcs_date=vcs_date,
            source=source,
            run=run,
            root=root,
            environment=environment,
        )
        if expected_id is not None and current_id != expected_id:
            return False
        containers = _completed(
            run,
            ["docker", "ps", "-aq", "--filter", f"ancestor={current_id}"],
            root=root,
            environment=environment,
            timeout=PROBE_TIMEOUT,
        )
        if containers.stdout:
            return False
        _completed(
            run,
            ["docker", "image", "rm", tag],
            root=root,
            environment=environment,
            timeout=PROBE_TIMEOUT,
        )
        return (
            _completed(
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
                root=root,
                environment=environment,
                timeout=PROBE_TIMEOUT,
            ).stdout
            == ""
        )
    except SmokeError:
        return False


def _validate_arguments(
    arguments,
    root: Path,
    *,
    runtime_ownership: list[tuple[Path, tuple[int, int]]] | None = None,
) -> tuple[Path, str, str, str, Path]:
    if arguments.images_lock != "docker/images.lock.env":
        raise SmokeError("compose smoke failed")
    if arguments.work_root != ".runtime" or not arguments.no_gpu:
        raise SmokeError("compose smoke failed")
    if (os.environ.get("GITHUB_ACTIONS") == "true") != arguments.github_actions_mask:
        raise SmokeError("compose smoke failed")
    if REVISION.fullmatch(arguments.revision) is None:
        raise SmokeError("compose smoke failed")
    try:
        timestamp = datetime.fromisoformat(arguments.vcs_date.replace("Z", "+00:00"))
    except ValueError:
        raise SmokeError("compose smoke failed") from None
    if timestamp.tzinfo is None:
        raise SmokeError("compose smoke failed")
    parsed = urlsplit(arguments.source_repository)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SmokeError("compose smoke failed")
    lock = root / "docker" / "images.lock.env"
    root = Path(os.path.abspath(root))
    _validate_directory(root)
    runtime = root / ".runtime"
    try:
        if not runtime.exists() and not runtime.is_symlink():
            runtime.mkdir(mode=0o700)
            if runtime_ownership is not None:
                runtime_ownership.append((runtime, _record_directory(runtime)))
            os.chmod(runtime, 0o700)
    except OSError:
        raise SmokeError("compose smoke failed") from None
    _validate_directory(runtime)
    return lock, arguments.revision, arguments.vcs_date, arguments.source_repository, runtime


def run_smoke(
    arguments,
    *,
    root: Path,
    run: Callable[..., subprocess.CompletedProcess[str]],
    token_hex: Callable[[int], str],
    mask_sink: Callable[[str], None],
    descriptor_ownership: list[int],
    runtime_ownership: list[tuple[Path, tuple[int, int]]] | None = None,
) -> str:
    lock_path, revision, vcs_date, source, runtime = _validate_arguments(
        arguments,
        root,
        runtime_ownership=runtime_ownership,
    )
    lock, lock_payload, lock_identity = _read_lock(lock_path)
    run_id = token_hex(16)
    if type(run_id) is not str or RUN_ID.fullmatch(run_id) is None:
        raise SmokeError("compose smoke failed")
    project = f"trainfactory-ci-{run_id}"
    bundle = runtime / f"ci-{run_id}"
    secret_env = bundle / "compose-secrets.env"
    secret_state = bundle / "compose-secrets.state.json"
    staging = runtime / f".ci-staging-{run_id}"
    selection = runtime / f"ci-release-{run_id}.env"
    manifest = runtime / f"ci-compose-{run_id}-manifest.json"
    api_iid = runtime / f"ci-api-{run_id}.iid"
    web_iid = runtime / f"ci-web-{run_id}.iid"
    api_tag = f"trainfactory-api-smoke:{run_id}"
    web_tag = f"trainfactory-web-smoke:{run_id}"
    for path in (bundle, staging, selection, manifest, api_iid, web_iid):
        if path.exists() or path.is_symlink():
            raise SmokeError("compose smoke failed")

    environment = _local_engine(run, root=root)
    _require_empty_project(
        project,
        run=run,
        root=root,
        environment=environment,
    )
    for tag in (api_tag, web_tag):
        _tag_absent(tag, run=run, root=root, environment=environment)

    def script(name: str) -> str:
        return os.fspath(root / "scripts" / name)

    tag_intents: list[str] = []
    image_ids: dict[str, str] = {}
    iid_identities: dict[Path, tuple[int, int]] = {}
    selection_identity: tuple[int, int] | None = None
    manifest_identity: tuple[int, int] | None = None
    manifest_payload: bytes | None = None
    secret_snapshots: tuple[tuple[Path, bytes, tuple[int, int]], ...] = ()
    selection_intent = False
    manifest_intent = False
    materializer_intent = False
    staging_identity: tuple[int, int] | None = None
    staging_guard = -1
    up_intent = False
    cleanup_failed = False
    try:
        tag_intents.append(api_tag)
        api_build = [
                "docker",
                "build",
                "--quiet",
                "--platform",
                "linux/amd64",
                "-f",
                "docker/Dockerfile.test",
                "--target",
                "api-smoke",
                "--build-arg",
                f"API_TEST_BASE_IMAGE={lock['API_TEST_BASE_IMAGE']}",
                "--build-arg",
                f"BUILD_VERSION={IMAGE_VERSION}",
                "--build-arg",
                f"VCS_REF={revision}",
                "--build-arg",
                f"VCS_DATE={vcs_date}",
                "--build-arg",
                f"SOURCE_REPOSITORY={source}",
                "--iidfile",
                os.fspath(api_iid),
                "-t",
                api_tag,
                ".",
        ]
        try:
            _completed(
                run,
                api_build,
                root=root,
                environment=environment,
                timeout=BUILD_TIMEOUT,
            )
        except SmokeError:
            if api_iid.exists() or api_iid.is_symlink():
                image_ids[api_tag], iid_identities[api_iid] = _read_iid(api_iid)
            raise
        image_ids[api_tag], iid_identities[api_iid] = _read_iid(api_iid)
        if image_ids[api_tag] != _inspect_image(
            api_tag,
            revision=revision,
            vcs_date=vcs_date,
            source=source,
            run=run,
            root=root,
            environment=environment,
        ):
            raise SmokeError("compose smoke failed")

        tag_intents.append(web_tag)
        web_build = [
                "docker",
                "build",
                "--quiet",
                "--platform",
                "linux/amd64",
                "-f",
                "web/Dockerfile",
                "--build-arg",
                f"WEB_NODE_BUILD_IMAGE={lock['WEB_NODE_BUILD_IMAGE']}",
                "--build-arg",
                f"WEB_NGINX_IMAGE={lock['WEB_NGINX_IMAGE']}",
                "--build-arg",
                f"BUILD_VERSION={IMAGE_VERSION}",
                "--build-arg",
                f"VCS_REF={revision}",
                "--build-arg",
                f"VCS_DATE={vcs_date}",
                "--build-arg",
                f"SOURCE_REPOSITORY={source}",
                "--iidfile",
                os.fspath(web_iid),
                "-t",
                web_tag,
                "web",
        ]
        try:
            _completed(
                run,
                web_build,
                root=root,
                environment=environment,
                timeout=BUILD_TIMEOUT,
            )
        except SmokeError:
            if web_iid.exists() or web_iid.is_symlink():
                image_ids[web_tag], iid_identities[web_iid] = _read_iid(web_iid)
            raise
        image_ids[web_tag], iid_identities[web_iid] = _read_iid(web_iid)
        if image_ids[web_tag] != _inspect_image(
            web_tag,
            revision=revision,
            vcs_date=vcs_date,
            source=source,
            run=run,
            root=root,
            environment=environment,
        ):
            raise SmokeError("compose smoke failed")

        _verify_lock(lock_path, payload=lock_payload, identity=lock_identity)

        try:
            staging.mkdir(mode=0o700)
            os.chmod(staging, 0o700)
            if os.name == "nt":
                staging_identity = _record_directory(staging)
            else:
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                staging_guard = os.open(staging, flags)
                os.set_inheritable(staging_guard, False)
                descriptor_ownership.append(staging_guard)
                guarded = os.fstat(staging_guard)
                current = staging.lstat()
                if (
                    not stat.S_ISDIR(guarded.st_mode)
                    or _is_reparse(staging, current)
                    or not stat.S_ISDIR(current.st_mode)
                    or (guarded.st_dev, guarded.st_ino)
                    != (current.st_dev, current.st_ino)
                ):
                    raise OSError("staging identity mismatch")
                staging_identity = guarded.st_dev, guarded.st_ino
        except OSError:
            if staging_guard >= 0:
                try:
                    os.close(staging_guard)
                except OSError:
                    pass
                else:
                    descriptor_ownership.remove(staging_guard)
                staging_guard = -1
            raise SmokeError("compose smoke failed") from None
        materializer_intent = True
        _completed(
            run,
            [
                sys.executable,
                "-I",
                script("materialize_compose_secrets.py"),
                "create",
                "--scope",
                "ci",
                "--run-id",
                run_id,
                "--output-root",
                os.fspath(runtime),
                "--env-out",
                os.fspath(secret_env),
                "--state-out",
                os.fspath(secret_state),
                "--staging-dir",
                os.fspath(staging),
            ],
            root=root,
            environment=environment,
            timeout=COMMAND_TIMEOUT,
        )
        _verify_lock(lock_path, payload=lock_payload, identity=lock_identity)
        _record_file(secret_state)
        if arguments.github_actions_mask:
            try:
                mask_values, secret_snapshots = _secret_mask_values(bundle)
                for value in mask_values:
                    mask_sink(value)
                _verify_secret_snapshots(secret_snapshots)
            except Exception:
                raise SmokeError("compose smoke failed") from None
        selection_values = {
            "API_IMAGE": api_tag,
            "WEB_IMAGE": web_tag,
            "RELEASE_REVISION": revision,
            "API_IMAGE_ID": image_ids[api_tag],
            "WEB_IMAGE_ID": image_ids[web_tag],
        }
        selection_intent = True
        _completed(
            run,
            [
                sys.executable,
                "-I",
                script("compose_manifest.py"),
                "image-selection",
                "--api-image",
                api_tag,
                "--web-image",
                web_tag,
                "--api-image-id",
                image_ids[api_tag],
                "--web-image-id",
                image_ids[web_tag],
                "--revision",
                revision,
                "--output",
                os.fspath(selection),
            ],
            root=root,
            environment=environment,
            timeout=COMMAND_TIMEOUT,
        )
        selection_identity = _record_file(selection)
        manifest_intent = True
        _completed(
            run,
            [
                sys.executable,
                "-I",
                script("compose_manifest.py"),
                "freeze",
                "--mode",
                "ci",
                "--project",
                project,
                "--env-file",
                os.fspath(secret_env),
                "--env-file",
                os.fspath(lock_path),
                "--env-file",
                os.fspath(selection),
                "--gpu-mode",
                "cpu",
                "--secret-mode",
                "files",
                "--output",
                os.fspath(manifest),
            ],
            root=root,
            environment=environment,
            timeout=COMMAND_TIMEOUT,
        )
        manifest_payload, manifest_identity = _read_stable_regular(
            manifest,
            max_bytes=1024 * 1024,
        )
        _verify_lock(lock_path, payload=lock_payload, identity=lock_identity)
        if secret_snapshots:
            _verify_secret_snapshots(secret_snapshots)
        up_intent = True
        _completed(
            run,
            [
                sys.executable,
                "-I",
                script("compose_release.py"),
                "--manifest",
                os.fspath(manifest),
                "--",
                "up",
                "-d",
                "--wait",
                "--wait-timeout",
                "600",
                "mysql",
                "train-factory-api",
                "train-factory-web",
            ],
            root=root,
            environment=environment,
            timeout=COMMAND_TIMEOUT,
        )
        if secret_snapshots:
            _verify_secret_snapshots(secret_snapshots)
        _completed(
            run,
            [
                sys.executable,
                "-I",
                script("verify_deployment.py"),
                "--project",
                project,
                "--compose-manifest",
                os.fspath(manifest),
                "--release-env",
                os.fspath(selection),
                "--expected-revision",
                revision,
                "--expected-alembic",
                ALEMBIC_HEAD,
                "--no-gpu",
                "--scope",
                "full",
                "--username-file",
                os.fspath(bundle / "admin_username"),
                "--password-file",
                os.fspath(bundle / "default_admin_password"),
            ],
            root=root,
            environment=environment,
            timeout=COMMAND_TIMEOUT,
        )
        if secret_snapshots:
            _verify_secret_snapshots(secret_snapshots)
        _verify_lock(lock_path, payload=lock_payload, identity=lock_identity)
    finally:
        if selection_intent and selection_identity is None and (
            selection.exists() or selection.is_symlink()
        ):
            selection_identity = _recover_selection_identity(
                selection,
                expected=selection_values,
                root=root,
            )
            if selection_identity is None:
                cleanup_failed = True
        if manifest_intent and manifest_identity is None and (
            manifest.exists() or manifest.is_symlink()
        ):
            manifest_identity = _recover_manifest_identity(
                manifest,
                project=project,
                secret_env=secret_env,
                lock_path=lock_path,
                selection=selection,
                root=root,
            )
            if manifest_identity is None:
                cleanup_failed = True
            else:
                recovered_payload, recovered_identity = _read_stable_regular(
                    manifest,
                    max_bytes=1024 * 1024,
                )
                if recovered_identity != manifest_identity:
                    manifest_identity = None
                    cleanup_failed = True
                else:
                    manifest_payload = recovered_payload
        resources_empty = True
        if up_intent:
            manifest_is_current = False
            if manifest_identity is not None and manifest_payload is not None:
                try:
                    current_payload, current_identity = _read_stable_regular(
                        manifest,
                        max_bytes=1024 * 1024,
                    )
                    manifest_is_current = (
                        current_payload == manifest_payload
                        and current_identity == manifest_identity
                    )
                except SmokeError:
                    manifest_is_current = False
            if manifest_is_current:
                try:
                    _completed(
                        run,
                        [
                            sys.executable,
                            "-I",
                            script("compose_release.py"),
                            "--manifest",
                            os.fspath(manifest),
                            "--safe-down-volumes",
                        ],
                        root=root,
                        environment=environment,
                        timeout=COMMAND_TIMEOUT,
                    )
                except SmokeError:
                    cleanup_failed = True
            else:
                cleanup_failed = True
            try:
                _require_empty_project(
                    project,
                    run=run,
                    root=root,
                    environment=environment,
                )
            except SmokeError:
                resources_empty = False
                cleanup_failed = True
        if resources_empty:
            if staging_identity is not None:
                try:
                    staging_cleaned = _cleanup_owned_staging(
                        staging,
                        staging_identity,
                    )
                except (OSError, RuntimeError, ValueError):
                    cleanup_failed = True
                else:
                    if not staging_cleaned:
                        cleanup_failed = True
            if materializer_intent and (
                secret_state.exists() or secret_state.is_symlink()
            ):
                try:
                    _record_file(secret_state)
                    _completed(
                        run,
                        [
                            sys.executable,
                            "-I",
                            script("materialize_compose_secrets.py"),
                            "cleanup",
                            "--state",
                            os.fspath(secret_state),
                        ],
                        root=root,
                        environment=environment,
                        timeout=COMMAND_TIMEOUT,
                    )
                    if bundle.exists() or bundle.is_symlink():
                        cleanup_failed = True
                except SmokeError:
                    cleanup_failed = True
            elif bundle.exists() or bundle.is_symlink():
                cleanup_failed = True
            if not _unlink_owned(manifest, manifest_identity):
                cleanup_failed = True
            if not _unlink_owned(selection, selection_identity):
                cleanup_failed = True
            for iid_path, iid_identity in iid_identities.items():
                if not _unlink_owned(iid_path, iid_identity):
                    cleanup_failed = True
        if staging_guard >= 0:
            try:
                os.close(staging_guard)
            except OSError:
                cleanup_failed = True
            else:
                descriptor_ownership.remove(staging_guard)
            staging_guard = -1
        for tag in reversed(tag_intents):
            if not _remove_owned_image(
                tag,
                image_ids.get(tag),
                revision=revision,
                vcs_date=vcs_date,
                source=source,
                run=run,
                root=root,
                environment=environment,
            ):
                cleanup_failed = True
        if cleanup_failed:
            raise SmokeError("compose smoke failed")
    return project


def _cleanup_created_runtime(
    ownership: list[tuple[Path, tuple[int, int]]],
) -> bool:
    if not ownership:
        return True
    if len(ownership) != 1:
        return False
    path, identity = ownership[0]
    try:
        metadata = path.lstat()
        if (
            (metadata.st_dev, metadata.st_ino) != identity
            or _is_reparse(path, metadata)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            return False
        if any(path.iterdir()):
            return False
        path.rmdir()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _close_owned_descriptors(descriptors: list[int]) -> bool:
    cleanup_succeeded = True
    while descriptors:
        descriptor = descriptors.pop()
        try:
            os.close(descriptor)
        except OSError:
            cleanup_succeeded = False
    return cleanup_succeeded


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = ROOT_DIR,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_process_tree,
    token_hex: Callable[[int], str] = secrets.token_hex,
    mask_sink: Callable[[str], None] = _github_mask_sink,
) -> int:
    descriptor_ownership: list[int] = []
    runtime_ownership: list[tuple[Path, tuple[int, int]]] = []
    cleanup_failed = False
    try:
        arguments = _parser().parse_args(argv)
        project = run_smoke(
            arguments,
            root=Path(root),
            run=run,
            token_hex=token_hex,
            mask_sink=mask_sink,
            descriptor_ownership=descriptor_ownership,
            runtime_ownership=runtime_ownership,
        )
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    except (OSError, RuntimeError, SmokeError, subprocess.SubprocessError, ValueError):
        print("compose smoke failed", file=sys.stderr)
        return 1
    finally:
        if not _close_owned_descriptors(descriptor_ownership):
            cleanup_failed = True
        if not _cleanup_created_runtime(runtime_ownership):
            cleanup_failed = True
    if cleanup_failed:
        print("compose smoke failed", file=sys.stderr)
        return 1
    print(f"compose smoke passed project={project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
