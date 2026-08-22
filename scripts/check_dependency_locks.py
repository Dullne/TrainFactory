"""Regenerate and validate project-level Linux dependency locks."""

from __future__ import annotations

import sys
import os as _bootstrap_os


def _bootstrap_realpath(value: str) -> str:
    return _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(value))


_script_path = _bootstrap_realpath(__file__)
_script_dir = _bootstrap_os.path.dirname(_script_path)
_root_entry = _bootstrap_os.path.dirname(_script_dir)
if __name__ == "__main__":
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
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import tomllib  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from scripts.check_base_image_providers import EXPECTED_PROVIDER_NAMES  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_UV_VERSION = "0.11.19"
ALLOWED_ENVIRONMENT = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


@dataclass(frozen=True)
class LockMetadata:
    root: Path
    data: dict[str, Any]


class _PrivateParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("dependency lock arguments are invalid")


def _fixed_metadata() -> dict[str, Any]:
    return {
        "tool": {
            "uv_version": "0.11.19",
            "python_version": "3.11",
            "python_platform": "x86_64-unknown-linux-gnu",
            "default_index": "https://pypi.org/simple",
            "cpu_index": "https://download.pytorch.org/whl/cpu",
            "torch_backend": "cpu",
            "index_strategy": "first-index",
            "resolution": "highest",
            "prerelease": "if-necessary",
            "keyring_provider": "disabled",
            "exclude_newer": "2026-08-16T00:00:00Z",
            "only_binary": ":all:",
            "no_sources": True,
            "no_header": True,
        },
        "runtime": {
            "inputs": ["pyproject.toml"],
            "overrides": "requirements/runtime-overrides.txt",
            "providers": "requirements/base-image-provided.txt",
            "output": "requirements/runtime.lock",
        },
        "test_cpu": {
            "inputs": [
                "pyproject.toml",
                "requirements/test-build-requirements.in",
            ],
            "extra": ["dev"],
            "overrides": "requirements/test-cpu-overrides.txt",
            "output": "requirements/test-cpu.lock",
        },
    }


def load_lock_metadata(root: str | os.PathLike[str]) -> LockMetadata:
    repository = Path(root).resolve(strict=True)
    path = repository / "requirements" / "lock-metadata.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ValueError("dependency lock metadata is invalid") from None
    if data != _fixed_metadata():
        raise ValueError("dependency lock metadata is invalid")
    return LockMetadata(repository, data)


def _provider_names(metadata: LockMetadata) -> list[str]:
    path = metadata.root / metadata.data["runtime"]["providers"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError("dependency provider inputs are invalid") from None
    names = [line.split("==", 1)[0] for line in lines if line.count("==") == 1]
    if names != sorted(EXPECTED_PROVIDER_NAMES):
        raise ValueError("dependency provider inputs are invalid")
    return names


def build_compile_command(
    metadata: LockMetadata,
    *,
    kind: str,
    output_file: str | os.PathLike[str],
) -> list[str]:
    if kind not in {"runtime", "test_cpu"}:
        raise ValueError("dependency lock kind is invalid")
    tool = metadata.data["tool"]
    section = metadata.data[kind]
    command = ["uv", "--no-config", "pip", "compile", *section["inputs"]]
    if kind == "test_cpu":
        for extra in section["extra"]:
            command.extend(("--extra", extra))
    command.extend(
        (
            "--python-version",
            tool["python_version"],
            "--python-platform",
            tool["python_platform"],
            "--default-index",
            tool["default_index"],
            "--index-strategy",
            tool["index_strategy"],
            "--resolution",
            tool["resolution"],
            "--prerelease",
            tool["prerelease"],
            "--keyring-provider",
            tool["keyring_provider"],
            "--no-sources",
            "--overrides",
            section["overrides"],
            "--exclude-newer",
            tool["exclude_newer"],
            "--only-binary",
            tool["only_binary"],
            "--no-header",
            "--generate-hashes",
        )
    )
    if kind == "runtime":
        for name in _provider_names(metadata):
            command.extend(("--no-emit-package", name))
    else:
        command.extend(("--torch-backend", tool["torch_backend"]))
    command.extend(("--output-file", str(output_file)))
    return command


def sanitized_subprocess_environment(
    source: dict[str, str] | os._Environ[str],
    *,
    cache_directory: Path,
) -> dict[str, str]:
    casefolded = {key.upper(): value for key, value in source.items()}
    result = {name: casefolded[name] for name in ALLOWED_ENVIRONMENT if name in casefolded and casefolded[name]}
    result["UV_CACHE_DIR"] = str(cache_directory)
    safe_home = cache_directory / "home"
    result["HOME"] = str(safe_home)
    result["USERPROFILE"] = str(safe_home)
    return result


_REQUIREMENT_LINE = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9._-]*)==" r"(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)(?: \\)?$"
)
_HASH_LINE = re.compile(r"^    --hash=sha256:[0-9a-f]{64}(?: \\)?$")


def _lock_error(label: str, category: str) -> list[str]:
    return [f"{label} lock has invalid {category} entry"]


def validate_lock_text(
    text: str,
    *,
    kind: str,
    provider_names: set[str],
) -> list[str]:
    if kind not in {"runtime", "test_cpu"}:
        raise ValueError("dependency lock kind is invalid")
    label = "runtime" if kind == "runtime" else "test_cpu"
    lowered = text.lower()
    if re.search(r"https?://[^\s/@:]+:[^\s/@]+@", text):
        return _lock_error(label, "credential")
    if re.search(r"(?im)^\s*(?:-e|--editable)(?:\s|=)|file://", text):
        return _lock_error(label, "editable")
    if re.search(r"(?im)(?:^|\s|@)(?:[a-z]:[\\/]|\\\\|/[^\s])", text):
        return _lock_error(label, "absolute")
    if re.search(r"(?m)^--", text):
        return _lock_error(label, "option")
    if " @ " in text:
        return _lock_error(label, "direct")
    if ";" in text and any(token in lowered for token in ("win32", "windows", "os_name", "sys_platform")):
        return _lock_error(label, "windows")

    normalized_providers = {name.lower().replace("_", "-") for name in provider_names}
    parsed: dict[str, str] = {}
    current_name: str | None = None
    current_hashes = 0

    def finish_block() -> list[str] | None:
        if current_name is not None and current_hashes == 0:
            return _lock_error(label, "unhashed")
        return None

    for line in text.splitlines():
        if not line or line.startswith("#") or re.fullmatch(r"    #.*", line):
            continue
        if line[0].isspace():
            if "--hash=" in line and not _HASH_LINE.fullmatch(line):
                return _lock_error(label, "hash")
            if current_name is None or not _HASH_LINE.fullmatch(line):
                return _lock_error(label, "requirement")
            current_hashes += 1
            continue

        unfinished = finish_block()
        if unfinished:
            return unfinished
        match = _REQUIREMENT_LINE.fullmatch(line)
        if match is None:
            return _lock_error(label, "requirement")
        raw_name = match.group("name")
        name = raw_name.lower().replace("_", "-").replace(".", "-")
        if raw_name != name or name in parsed:
            return _lock_error(label, "requirement")
        version = match.group("version")
        if kind == "runtime" and name in normalized_providers:
            return ["runtime lock has invalid provider entry"]
        if kind == "test_cpu":
            forbidden_provider = name in normalized_providers and name not in {
                "setuptools",
                "wheel",
                "torch",
            }
            if forbidden_provider or "+cu" in version.lower():
                return ["test_cpu lock has invalid cpu provider entry"]
            if name == "torch" and version != "2.6.0+cpu":
                return ["test_cpu lock has invalid cpu provider entry"]
        parsed[name] = version
        current_name = name
        current_hashes = 0

    unfinished = finish_block()
    if unfinished:
        return unfinished
    if kind == "test_cpu":
        required = {
            "torch": "2.6.0+cpu",
            "setuptools": "75.8.0",
            "wheel": "0.45.1",
        }
        if any(parsed.get(name) != version for name, version in required.items()):
            return ["test_cpu lock has invalid required entry"]
    return []


def _copy_generation_inputs(source: Path, destination: Path, metadata: LockMetadata) -> None:
    relative_paths = {
        "requirements/lock-metadata.toml",
        metadata.data["runtime"]["overrides"],
        metadata.data["runtime"]["providers"],
        metadata.data["test_cpu"]["overrides"],
        *metadata.data["runtime"]["inputs"],
        *metadata.data["test_cpu"]["inputs"],
    }
    for relative in sorted(relative_paths):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)


def _uv_version(environment: dict[str, str]) -> str:
    try:
        completed = subprocess.run(
            ["uv", "--version"],
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("dependency lock generator is unavailable") from None
    if completed.returncode != 0:
        raise ValueError("dependency lock generator is unavailable")
    match = re.fullmatch(r"uv ([0-9]+\.[0-9]+\.[0-9]+)(?: .*)?\s*", completed.stdout)
    if match is None or match.group(1) != EXPECTED_UV_VERSION:
        raise ValueError("dependency lock generator version mismatch")
    return match.group(1)


def generate_lock(
    root: Path,
    metadata: LockMetadata,
    *,
    kind: str,
    output_file: Path,
    environment: dict[str, str],
) -> str:
    try:
        completed = subprocess.run(
            build_compile_command(metadata, kind=kind, output_file=output_file),
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("dependency lock generation failed") from None
    if completed.returncode != 0:
        raise ValueError("dependency lock generation failed")
    try:
        return output_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ValueError("dependency lock generation failed") from None


def generate_lock_pair(root: Path, metadata: LockMetadata) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="trainfactory-lock-") as temporary_name:
        temporary = Path(temporary_name)
        _copy_generation_inputs(root, temporary, metadata)
        temporary_metadata = load_lock_metadata(temporary)
        cache = temporary / ".uv-cache"
        environment = sanitized_subprocess_environment(os.environ, cache_directory=cache)
        _uv_version(environment)
        generated: dict[str, str] = {}
        for kind in ("runtime", "test_cpu"):
            output = temporary / temporary_metadata.data[kind]["output"]
            output.parent.mkdir(parents=True, exist_ok=True)
            generated[kind] = generate_lock(
                temporary,
                temporary_metadata,
                kind=kind,
                output_file=output,
                environment=environment,
            )
        return generated


def _write_bytes_atomically(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _replace_lock_pair(root: Path, generated: dict[str, str]) -> None:
    paths = {
        "runtime": root / "requirements" / "runtime.lock",
        "test_cpu": root / "requirements" / "test-cpu.lock",
    }
    originals = {kind: path.read_bytes() if path.exists() else None for kind, path in paths.items()}
    replaced: list[str] = []
    try:
        for kind in ("runtime", "test_cpu"):
            _write_bytes_atomically(paths[kind], generated[kind].encode("utf-8"))
            replaced.append(kind)
    except OSError:
        try:
            for kind in reversed(replaced):
                original = originals[kind]
                if original is None:
                    paths[kind].unlink(missing_ok=True)
                else:
                    _write_bytes_atomically(paths[kind], original)
        except OSError:
            raise ValueError("dependency lock rollback failed") from None
        raise ValueError("dependency lock update failed") from None


def update_locks_transactionally(root: Path, *, metadata: LockMetadata) -> None:
    generated = generate_lock_pair(root, metadata)
    providers = set(EXPECTED_PROVIDER_NAMES)
    for kind in ("runtime", "test_cpu"):
        if validate_lock_text(generated[kind], kind=kind, provider_names=providers):
            raise ValueError("dependency lock validation failed")
    _replace_lock_pair(root, generated)


def check_locks(root: Path, *, write: bool) -> None:
    metadata = load_lock_metadata(root)
    generated = generate_lock_pair(root, metadata)
    providers = set(EXPECTED_PROVIDER_NAMES)
    for kind in ("runtime", "test_cpu"):
        errors = validate_lock_text(generated[kind], kind=kind, provider_names=providers)
        if errors:
            raise ValueError("dependency lock validation failed")
    if write:
        _replace_lock_pair(root, generated)
        return
    for kind in ("runtime", "test_cpu"):
        target = root / metadata.data[kind]["output"]
        if target.read_bytes() != generated[kind].encode("utf-8"):
            raise ValueError("dependency lock differs")


def main(argv: list[str] | None = None) -> int:
    parser = _PrivateParser()
    parser.add_argument("--write", action="store_true")
    try:
        args = parser.parse_args(argv)
        check_locks(ROOT_DIR, write=args.write)
    except (OSError, UnicodeError, ValueError):
        print("dependency lock check failed", file=sys.stderr)
        return 2 if "args" not in locals() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
