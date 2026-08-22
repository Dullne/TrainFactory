"""Verify the exact package providers supplied by the API base image."""

from __future__ import annotations

import importlib.metadata
import re
import sys
from collections.abc import Mapping
from pathlib import Path


EXPECTED_PROVIDER_NAMES = (
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
    "setuptools",
    "torch",
    "torchaudio",
    "torchelastic",
    "torchvision",
    "triton",
    "wheel",
)


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _parse_expected(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines != sorted(lines):
        raise ValueError("provider snapshot is invalid")
    parsed: dict[str, str] = {}
    for line in lines:
        if line.count("==") != 1:
            raise ValueError("provider snapshot is invalid")
        name, version = line.split("==", 1)
        canonical = _canonical_name(name)
        if not name or not version or canonical in parsed:
            raise ValueError("provider snapshot is invalid")
        parsed[canonical] = version
    if tuple(sorted(parsed)) != tuple(sorted(EXPECTED_PROVIDER_NAMES)):
        raise ValueError("provider snapshot is invalid")
    return parsed


def installed_provider_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _canonical_name(raw_name)
        if name not in EXPECTED_PROVIDER_NAMES:
            continue
        if name in installed:
            raise ValueError("duplicate provider distribution")
        installed[name] = distribution.version
    return installed


def validate_installed_providers(
    expected_text: str,
    installed: Mapping[str, str],
) -> list[str]:
    expected = _parse_expected(expected_text)
    canonical_installed = {_canonical_name(name): version for name, version in installed.items()}
    errors: list[str] = []
    for name in sorted(expected):
        if name not in canonical_installed:
            errors.append(f"provider {name} is missing")
        elif canonical_installed[name] != expected[name]:
            errors.append(f"provider {name} version mismatch")
    return errors


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("base image provider verification failed", file=sys.stderr)
        return 2
    try:
        snapshot = Path(arguments[0]).read_text(encoding="utf-8")
        errors = validate_installed_providers(snapshot, installed_provider_versions())
    except (OSError, UnicodeError, ValueError):
        print("base image provider verification failed", file=sys.stderr)
        return 1
    if errors:
        print("base image provider verification failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
