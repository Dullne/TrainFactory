"""Scan only Git-tracked text and report locations without matched values."""

from __future__ import annotations

import argparse
import codecs
import os
import re
import stat
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 5 * 1024 * 1024
BINARY_PROBE_BYTES = 65_536
SKIPPED_SUFFIXES = {
    ".db",
    ".dump",
    ".sqlite",
    ".sqlite3",
}
RULES = (
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    ("aws_access_key", re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])")),
    ("github_token", re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{36,}")),
)


class ScanError(Exception):
    pass


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule_id: str


def safe_display_path(path: str) -> str:
    if ":" in path or any(unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"} for character in path):
        return "<unsafe-path>"
    return path.replace("\\", "/")


def _tracked_paths(repository: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ScanError
    try:
        return [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]
    except UnicodeDecodeError:
        raise ScanError from None


def _is_skipped_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    return (
        normalized == ".runtime"
        or normalized.startswith(".runtime/")
        or Path(normalized).suffix.lower() in SKIPPED_SUFFIXES
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _read_tracked_text(candidate: Path) -> str | None:
    try:
        before = candidate.lstat()
    except OSError:
        raise ScanError from None
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        return None

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _file_identity(opened) != _file_identity(before)
                or opened.st_size != before.st_size
            ):
                raise ScanError
            oversized = opened.st_size > MAX_TEXT_BYTES
            chunks = []
            bytes_read = 0
            contains_nul = False
            invalid_utf8 = False
            decoder = codecs.getincrementaldecoder("utf-8")("strict") if oversized else None
            while True:
                chunk = os.read(descriptor, BINARY_PROBE_BYTES)
                if not chunk:
                    break
                bytes_read += len(chunk)
                contains_nul = contains_nul or b"\0" in chunk
                if oversized:
                    if not invalid_utf8:
                        try:
                            decoder.decode(chunk, final=False)
                        except UnicodeDecodeError:
                            invalid_utf8 = True
                else:
                    chunks.append(chunk)
                    if bytes_read > MAX_TEXT_BYTES:
                        break
            if oversized and not invalid_utf8:
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError:
                    invalid_utf8 = True
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = candidate.lstat()
    except ScanError:
        raise
    except OSError:
        raise ScanError from None

    content = b"".join(chunks)
    if (
        not stat.S_ISREG(after_open.st_mode)
        or not stat.S_ISREG(after_path.st_mode)
        or stat.S_ISLNK(after_path.st_mode)
        or _file_identity(after_open) != _file_identity(before)
        or _file_identity(after_path) != _file_identity(before)
        or after_open.st_size != before.st_size
        or after_path.st_size != before.st_size
        or bytes_read != before.st_size
    ):
        raise ScanError
    if oversized:
        if contains_nul or invalid_utf8:
            return None
        raise ScanError
    if len(content) > MAX_TEXT_BYTES or len(content) != before.st_size:
        raise ScanError
    if b"\0" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_repository(repository_root: str | Path) -> list[Finding]:
    repository = Path(repository_root).resolve(strict=True)
    findings: list[Finding] = []
    for relative_path in _tracked_paths(repository):
        if _is_skipped_path(relative_path):
            continue
        candidate = repository / relative_path
        text = _read_tracked_text(candidate)
        if text is None:
            continue
        display_path = safe_display_path(relative_path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule_id, pattern in RULES:
                if pattern.search(line):
                    findings.append(Finding(display_path, line_number, rule_id))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(ROOT_DIR))
    args = parser.parse_args(argv)
    try:
        findings = scan_repository(args.repo)
    except (OSError, ScanError):
        print("tracked secret scan failed", file=sys.stderr)
        return 2
    for finding in findings:
        print(f"{finding.path}:{finding.line}:{finding.rule_id}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
