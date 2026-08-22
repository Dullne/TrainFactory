"""Snapshot fixed providers from a digest-pinned API base image."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from scripts.check_base_image_providers import (
        EXPECTED_PROVIDER_NAMES,
        _parse_expected,
    )
except ModuleNotFoundError:  # Direct script execution.
    from check_base_image_providers import EXPECTED_PROVIDER_NAMES, _parse_expected


IMAGE_PATTERN = re.compile(r"^(?!-)[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}$")
SNAPSHOT_PROGRAM = """import importlib.metadata
import re
names = set(%r)
found = {}
for distribution in importlib.metadata.distributions():
    raw_name = distribution.metadata.get("Name")
    if not raw_name:
        continue
    name = re.sub(r"[-_.]+", "-", raw_name).lower()
    if name not in names:
        continue
    if name in found:
        raise SystemExit("duplicate provider distribution")
    found[name] = distribution.version
if set(found) != names:
    raise SystemExit("provider distribution missing")
for name in sorted(found):
    print(f"{name}=={found[name]}")
""" % (
    EXPECTED_PROVIDER_NAMES,
)


class _PrivateParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("snapshot arguments are invalid")


def snapshot_command(image: str) -> list[str]:
    if not IMAGE_PATTERN.fullmatch(image):
        raise ValueError("base image must use a complete digest")
    return [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python",
        image,
        "-c",
        SNAPSHOT_PROGRAM,
    ]


def _normalized_snapshot(text: str) -> str:
    parsed = _parse_expected("\n".join(sorted(text.splitlines())) + "\n")
    return "".join(f"{name}=={parsed[name]}\n" for name in sorted(parsed))


def compare_snapshot(text: str, path: Path, *, write: bool) -> bool:
    normalized = _normalized_snapshot(text)
    if write:
        temporary = path.with_name(f".{path.name}.tmp")
        if temporary.exists():
            raise ValueError("provider snapshot update failed")
        try:
            temporary.write_text(normalized, encoding="utf-8", newline="\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return True
    return path.read_bytes() == normalized.encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = _PrivateParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("requirements/base-image-provided.txt"),
    )
    parser.add_argument("--write", action="store_true")
    try:
        args = parser.parse_args(argv)
        completed = subprocess.run(
            snapshot_command(args.image),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=300,
        )
        if completed.returncode != 0:
            raise ValueError("provider snapshot command failed")
        matches = compare_snapshot(completed.stdout, args.snapshot, write=args.write)
    except (OSError, UnicodeError, ValueError, subprocess.TimeoutExpired):
        print("base provider snapshot failed", file=sys.stderr)
        return 2 if "args" not in locals() else 1
    if not matches:
        print("base provider snapshot differs", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
