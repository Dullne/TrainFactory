"""Validate one JUnit report before artifact upload without echoing secrets."""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("test report validator bootstrap is unavailable")


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
from dataclasses import dataclass  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import stat  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from urllib.parse import unquote, unquote_plus  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

from scripts.materialize_compose_secrets import (  # noqa: E402
    _harden_path,
    _verify_hardened_path,
)


MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_SECRET_BYTES = 65_536
_SAFE_REPORT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.xml$")
_CONNECTION_URL = re.compile(
    r"(?i)\b(?:mysql(?:\+pymysql)?|postgresql(?:\+[a-z0-9_]+)?)://"
    r"[^\s<>'\"]+"
)
_PASSWORD_VALUE = re.compile(
    r"(?i)(?<![A-Za-z0-9])['\"]?(?:[A-Z0-9]+_)*PASSWORD['\"]?"
    r"\s*(?:=|:)\s*['\"]?"
    r"(?!(?:\*+|<redacted>|&lt;redacted&gt;|%3credacted%3e|redacted|null)"
    r"(?:['\"]?[\s,}><]|$))"
    r"[^\s<>'\",}]{4,}"
)
_JWT_VALUE = re.compile(
    r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
_JWT_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])['\"]?(?:[A-Z0-9]+_)*JWT_SECRET(?:_KEY)?['\"]?"
    r"\s*(?:=|:)\s*['\"]?"
    r"(?!(?:\*+|<redacted>|&lt;redacted&gt;|%3credacted%3e|redacted|null)"
    r"(?:['\"]?[\s,}><]|$))"
    r"[^\s<>'\",}]{4,}"
)


class ReportError(RuntimeError):
    """Raised when a report or secret-file boundary is invalid."""


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    metadata: os.stat_result
    payload: bytes
    private: bool


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
        stat.S_IFMT(left.st_mode),
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
        stat.S_IFMT(right.st_mode),
    )


def _is_reparse(path: Path, metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _validate_ancestor_chain(path: Path) -> None:
    if not path.is_absolute():
        raise ReportError("test report validation failed")
    anchor = Path(path.anchor)
    current = anchor
    try:
        for part in path.parts[1:-1]:
            current = current / part
            metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or _is_reparse(current, metadata):
                raise ReportError("test report validation failed")
    except (OSError, ValueError):
        raise ReportError("test report validation failed") from None


def _validate_canonical_existing_path(path: Path) -> None:
    _validate_ancestor_chain(path)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ReportError("test report validation failed") from None
    if os.path.normcase(os.fspath(resolved)) != os.path.normcase(os.fspath(path)):
        raise ReportError("test report validation failed")


def _stable_read(
    path: Path,
    *,
    max_bytes: int,
    private: bool,
    verify_hardened: Callable[[Path], bool],
) -> _FileSnapshot:
    _validate_canonical_existing_path(path)
    before = path.lstat()
    if _is_reparse(path, before) or not stat.S_ISREG(before.st_mode):
        raise ReportError("test report validation failed")
    if before.st_size > max_bytes:
        raise ReportError("test report validation failed")
    if private and not verify_hardened(path):
        raise ReportError("test report validation failed")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if not _same_identity(before, opened):
            raise ReportError("test report validation failed")
        payload = handle.read(max_bytes + 1)
        after_read = os.fstat(handle.fileno())
    after = path.lstat()
    if len(payload) > max_bytes or not _same_identity(opened, after_read):
        raise ReportError("test report validation failed")
    if not _same_identity(after_read, after):
        raise ReportError("test report validation failed")
    if private and not verify_hardened(path):
        raise ReportError("test report validation failed")
    _validate_canonical_existing_path(path)
    final = path.lstat()
    if not _same_identity(after, final):
        raise ReportError("test report validation failed")
    return _FileSnapshot(path=path, metadata=after, payload=payload, private=private)


def _revalidate_snapshot(
    snapshot: _FileSnapshot,
    *,
    verify_hardened: Callable[[Path], bool],
) -> None:
    current = _stable_read(
        snapshot.path,
        max_bytes=len(snapshot.payload),
        private=snapshot.private,
        verify_hardened=verify_hardened,
    )
    if (
        not _same_identity(snapshot.metadata, current.metadata)
        or snapshot.payload != current.payload
    ):
        raise ReportError("test report validation failed")


def _canonical_report(report_root: Path, report: Path) -> Path:
    if not report_root.is_absolute() or not report.is_absolute():
        raise ReportError("test report validation failed")
    _validate_ancestor_chain(report_root / "boundary")
    root_before = report_root.lstat()
    if _is_reparse(report_root, root_before) or not stat.S_ISDIR(root_before.st_mode):
        raise ReportError("test report validation failed")
    canonical_root = report_root.resolve(strict=True)
    canonical_report = report.resolve(strict=True)
    if os.path.normcase(os.fspath(canonical_root)) != os.path.normcase(
        os.fspath(report_root)
    ):
        raise ReportError("test report validation failed")
    try:
        contained = os.path.commonpath((canonical_root, canonical_report))
    except ValueError:
        raise ReportError("test report validation failed") from None
    if os.path.normcase(os.fspath(contained)) != os.path.normcase(
        os.fspath(canonical_root)
    ):
        raise ReportError("test report validation failed")
    if canonical_report.parent != canonical_root:
        raise ReportError("test report validation failed")
    if canonical_report != report:
        raise ReportError("test report validation failed")
    if _SAFE_REPORT_NAME.fullmatch(report.name) is None:
        raise ReportError("test report validation failed")
    return report


def _decode_secret(payload: bytes) -> str:
    try:
        value = payload.decode("utf-8")
    except UnicodeError:
        raise ReportError("test report validation failed") from None
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    if not value or "\x00" in value or "\r" in value or "\n" in value:
        raise ReportError("test report validation failed")
    return value


def _decoded_views(text: str) -> tuple[tuple[str, ...], bool]:
    views = {text}
    frontier = {text}
    for _ in range(2):
        decoded_views = {
            decoded
            for value in frontier
            for decoded in (unquote(value), unquote_plus(value))
            if decoded != value and decoded not in views
        }
        if not decoded_views:
            frontier = set()
            break
        views.update(decoded_views)
        if len(views) > 8:
            raise ReportError("test report validation failed")
        frontier = decoded_views
    exceeded = bool(frontier) and any(
        decoded != value
        for value in frontier
        for decoded in (unquote(value), unquote_plus(value))
    )
    return tuple(views), exceeded


def _scan_texts(texts: Sequence[str], secrets: Sequence[str]) -> set[str]:
    rules: set[str] = set()
    decode_exceeded = False
    for text in texts:
        views, exceeded = _decoded_views(text)
        decode_exceeded = decode_exceeded or exceeded
        for view in views:
            if any(value in view for value in secrets):
                rules.add("secret_value")
            if _CONNECTION_URL.search(view):
                rules.add("connection_url")
            if _PASSWORD_VALUE.search(view):
                rules.add("password_value")
            if _JWT_VALUE.search(view) or _JWT_ASSIGNMENT.search(view):
                rules.add("jwt_value")
    if decode_exceeded and not rules:
        raise ReportError("test report validation failed")
    return rules


def _xml_values(root: ET.Element) -> tuple[str, ...]:
    values: list[str] = []
    for element in root.iter():
        values.extend(element.attrib.values())
        if element.text:
            values.append(element.text)
        if element.tail:
            values.append(element.tail)
    values.append("".join(root.itertext()))
    return tuple(values)


def _validate_report_snapshot(
    *,
    report_root: Path,
    report: Path,
    secret_files: Sequence[Path] = (),
    verify_hardened: Callable[[Path], bool] = _verify_hardened_path,
) -> tuple[list[str], _FileSnapshot]:
    report = _canonical_report(report_root, report)
    secret_snapshots: list[_FileSnapshot] = []
    secrets = []
    for secret_file in secret_files:
        if not secret_file.is_absolute():
            raise ReportError("test report validation failed")
        secret_snapshot = _stable_read(
            secret_file,
            max_bytes=MAX_SECRET_BYTES,
            private=True,
            verify_hardened=verify_hardened,
        )
        secret_snapshots.append(secret_snapshot)
        secrets.append(_decode_secret(secret_snapshot.payload))

    report_snapshot = _stable_read(
        report,
        max_bytes=MAX_REPORT_BYTES,
        private=False,
        verify_hardened=verify_hardened,
    )
    try:
        text = report_snapshot.payload.decode("utf-8")
    except UnicodeError:
        raise ReportError("test report validation failed") from None
    rules = _scan_texts((text,), secrets)
    if rules:
        for snapshot in secret_snapshots:
            _revalidate_snapshot(snapshot, verify_hardened=verify_hardened)
        _revalidate_snapshot(report_snapshot, verify_hardened=verify_hardened)
        return sorted(rules), report_snapshot
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ReportError("test report validation failed")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        raise ReportError("test report validation failed") from None
    if root.tag not in {"testsuite", "testsuites"}:
        raise ReportError("test report validation failed")
    rules = _scan_texts(_xml_values(root), secrets)
    for snapshot in secret_snapshots:
        _revalidate_snapshot(snapshot, verify_hardened=verify_hardened)
    _revalidate_snapshot(report_snapshot, verify_hardened=verify_hardened)
    return sorted(rules), report_snapshot


def validate_report(
    *,
    report_root: Path,
    report: Path,
    secret_files: Sequence[Path] = (),
    verify_hardened: Callable[[Path], bool] = _verify_hardened_path,
) -> list[str]:
    rules, _snapshot = _validate_report_snapshot(
        report_root=report_root,
        report=report,
        secret_files=secret_files,
        verify_hardened=verify_hardened,
    )
    return rules


def _unlink_owned(path: Path, identity: tuple[int, int]) -> bool:
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


def _validated_output_path(report_root: Path, output: Path, report: Path) -> Path:
    if not output.is_absolute() or output == report:
        raise ReportError("test report validation failed")
    if _SAFE_REPORT_NAME.fullmatch(output.name) is None:
        raise ReportError("test report validation failed")
    if os.path.normcase(os.fspath(output.parent)) != os.path.normcase(
        os.fspath(report_root)
    ):
        raise ReportError("test report validation failed")
    _validate_ancestor_chain(output)
    if output.exists() or output.is_symlink():
        raise ReportError("test report validation failed")
    return output


def _publish_validated_report(
    *,
    report_root: Path,
    report: Path,
    output: Path,
    payload: bytes,
    verify_hardened: Callable[[Path], bool],
    harden: Callable[[Path], object],
) -> tuple[int, int]:
    target = _validated_output_path(report_root, output, report)
    temporary = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
    descriptor = -1
    link_intent = False
    complete = False
    temporary_identity: tuple[int, int] | None = None
    owned_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        opened_metadata = os.fstat(descriptor)
        temporary_identity = (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        )
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
            raise ReportError("test report validation failed")
        os.close(descriptor)
        descriptor = -1
        harden(temporary)
        temporary_snapshot = _stable_read(
            temporary,
            max_bytes=len(payload),
            private=True,
            verify_hardened=verify_hardened,
        )
        if (
            temporary_snapshot.metadata.st_dev,
            temporary_snapshot.metadata.st_ino,
            temporary_snapshot.metadata.st_size,
        ) != descriptor_identity or temporary_snapshot.payload != payload:
            raise ReportError("test report validation failed")
        owned_identity = (
            temporary_snapshot.metadata.st_dev,
            temporary_snapshot.metadata.st_ino,
        )
        link_intent = True
        os.link(temporary, target)
        if not _unlink_owned(temporary, temporary_identity):
            raise ReportError("test report validation failed")
        published = _stable_read(
            target,
            max_bytes=len(payload),
            private=True,
            verify_hardened=verify_hardened,
        )
        if (
            published.metadata.st_dev,
            published.metadata.st_ino,
        ) != owned_identity or published.payload != payload:
            raise ReportError("test report validation failed")
        complete = True
        return owned_identity
    except ReportError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise ReportError("test report validation failed") from None
    finally:
        cleanup_failed = False
        if descriptor >= 0:
            if temporary_identity is None:
                try:
                    recovery_metadata = os.fstat(descriptor)
                    temporary_identity = (
                        recovery_metadata.st_dev,
                        recovery_metadata.st_ino,
                    )
                except OSError:
                    cleanup_failed = True
            try:
                os.close(descriptor)
            except OSError:
                cleanup_failed = True
        if temporary_identity is not None and not _unlink_owned(
            temporary,
            temporary_identity,
        ):
            cleanup_failed = True
        if link_intent and not complete and owned_identity is not None:
            if not _unlink_owned(target, owned_identity):
                cleanup_failed = True
        if cleanup_failed:
            raise ReportError("test report validation failed") from None


def validate_and_publish(
    *,
    report_root: Path,
    report: Path,
    validated_output: Path,
    secret_files: Sequence[Path] = (),
    verify_hardened: Callable[[Path], bool] = _verify_hardened_path,
    harden: Callable[[Path], object] = _harden_path,
) -> list[str]:
    rules, report_snapshot = _validate_report_snapshot(
        report_root=report_root,
        report=report,
        secret_files=secret_files,
        verify_hardened=verify_hardened,
    )
    if not rules:
        _publish_validated_report(
            report_root=report_root,
            report=report,
            output=validated_output,
            payload=report_snapshot.payload,
            verify_hardened=verify_hardened,
            harden=harden,
        )
    return rules


class _FixedArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ReportError("test report validation failed")


def _parser() -> argparse.ArgumentParser:
    parser = _FixedArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--validated-output", required=True)
    parser.add_argument("--secret-file", action="append", default=[])
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    verify_hardened: Callable[[Path], bool] = _verify_hardened_path,
    harden: Callable[[Path], object] = _harden_path,
) -> int:
    try:
        arguments = _parser().parse_args(argv)
        report = Path(arguments.report)
        rules = validate_and_publish(
            report_root=Path(arguments.report_root),
            report=report,
            validated_output=Path(arguments.validated_output),
            secret_files=tuple(Path(path) for path in arguments.secret_file),
            verify_hardened=verify_hardened,
            harden=harden,
        )
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    except (OSError, ReportError, RuntimeError, UnicodeError, ValueError):
        print("test report validation failed", file=sys.stderr)
        return 2
    if rules:
        for rule in rules:
            print(f"{report.name}:{rule}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
