"""Build release images with repository-derived OCI identity."""

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

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import tomllib  # noqa: E402
import unicodedata  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import urlsplit  # noqa: E402

ROOT_DIR = Path(_root_entry)
REQUIRED_IMAGE_LOCK_KEYS = (
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
)
CI_BUILD_ARG_KEYS = REQUIRED_IMAGE_LOCK_KEYS[:4]
_IMAGE_REFERENCE_PATTERN = re.compile(
    r"^(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*:"
    r"[A-Za-z0-9_][A-Za-z0-9_.-]*@sha256:([0-9a-f]{64})$"
)
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:([0-9a-f]{64})$")
_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9A-Za-z-]+)+$")
_DNS_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_REPOSITORY_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class ReleaseError(RuntimeError):
    """A privacy-safe release validation failure."""


@dataclass(frozen=True)
class RepositoryIdentity:
    revision: str
    vcs_date: str
    source_repository: str


def parse_images_lock_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            raise ReleaseError("images lock is invalid")
        name, value = line.split("=", 1)
        if name not in REQUIRED_IMAGE_LOCK_KEYS or name in values:
            raise ReleaseError("images lock is invalid")
        match = _IMAGE_REFERENCE_PATTERN.fullmatch(value)
        if match is None or match.group(1) == "0" * 64:
            raise ReleaseError("images lock is invalid")
        values[name] = value
    if tuple(values) != REQUIRED_IMAGE_LOCK_KEYS:
        raise ReleaseError("images lock is invalid")
    return values


def load_images_lock(path: Path) -> dict[str, str]:
    try:
        return parse_images_lock_text(path.read_text(encoding="utf-8"))
    except ReleaseError:
        raise
    except (OSError, UnicodeError):
        raise ReleaseError("images lock is invalid") from None


def ci_build_arguments(root: Path | None = None) -> tuple[str, ...]:
    workspace = ROOT_DIR if root is None else root
    images = load_images_lock(workspace / "docker" / "images.lock.env")
    return tuple(f"{name}={images[name]}" for name in CI_BUILD_ARG_KEYS)


def _has_unsafe_characters(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        for character in value
    )


def _normalize_repository_path(path: str) -> str:
    normalized = path.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    segments = normalized.split("/")
    if len(segments) < 2 or any(
        not segment
        or segment in {".", ".."}
        or _REPOSITORY_SEGMENT_PATTERN.fullmatch(segment) is None
        for segment in segments
    ):
        raise ReleaseError("source repository is invalid")
    return "/".join(segments)


def _valid_dns_name(host: str) -> bool:
    labels = host.split(".")
    return bool(labels) and all(
        _DNS_LABEL_PATTERN.fullmatch(label) is not None for label in labels
    )


def normalize_source_repository(value: str) -> str:
    if (
        not isinstance(value, str)
        or value == ""
        or value != value.strip()
        or _has_unsafe_characters(value)
        or "\\" in value
        or "?" in value
        or "#" in value
    ):
        raise ReleaseError("source repository is invalid")

    if value.startswith("git@"):
        if value.count(":") != 1:
            raise ReleaseError("source repository is invalid")
        host_part, path = value.split(":", 1)
        host = host_part.removeprefix("git@").lower()
        if not _valid_dns_name(host) or path.startswith("/"):
            raise ReleaseError("source repository is invalid")
        repository_path = _normalize_repository_path(path)
        return f"https://{host}/{repository_path}"

    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ReleaseError("source repository is invalid") from None
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or parsed.query
        or parsed.fragment
    ):
        raise ReleaseError("source repository is invalid")
    host = parsed.hostname.lower()
    authority = f"{host}:{port}" if port is not None else host
    if not _valid_dns_name(host) or not parsed.path.startswith("/"):
        raise ReleaseError("source repository is invalid")
    repository_path = _normalize_repository_path(parsed.path[1:])
    return f"{parsed.scheme}://{authority}/{repository_path}"


def command_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("GIT_")
    }


def _run_command(argv: list[str], *, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=command_environment(),
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=3600,
        )
    except (OSError, subprocess.SubprocessError):
        raise ReleaseError("release command failed") from None
    if completed.returncode != 0:
        raise ReleaseError("release command failed")
    return completed.stdout.strip()


def _read_project_version(root: Path) -> str:
    try:
        project = tomllib.loads(
            (root / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        version = project["version"]
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError):
        raise ReleaseError("project version is invalid") from None
    if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
        raise ReleaseError("project version is invalid")
    return version


def _valid_rfc3339(value: str) -> bool:
    if "T" not in value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _read_repository_identity(root: Path) -> RepositoryIdentity:
    revision = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    vcs_date = _run_command(
        ["git", "show", "-s", "--format=%cI", "HEAD"], cwd=root
    )
    remote = _run_command(["git", "remote", "get-url", "origin"], cwd=root)
    if _REVISION_PATTERN.fullmatch(revision) is None or not _valid_rfc3339(vcs_date):
        raise ReleaseError("repository identity is invalid")
    return RepositoryIdentity(
        revision=revision,
        vcs_date=vcs_date,
        source_repository=normalize_source_repository(remote),
    )


def _require_clean_repository(root: Path) -> None:
    status = _run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root
    )
    if status:
        raise ReleaseError("release build requires a clean repository")


def _expected_labels(version: str, identity: RepositoryIdentity) -> dict[str, str]:
    return {
        "org.opencontainers.image.version": version,
        "org.opencontainers.image.revision": identity.revision,
        "org.opencontainers.image.created": identity.vcs_date,
        "org.opencontainers.image.source": identity.source_repository,
    }


def _inspect_image(root: Path, image: str, labels: dict[str, str]) -> str:
    raw = _run_command(["docker", "image", "inspect", image], cwd=root)
    try:
        payload = json.loads(raw)
        if not isinstance(payload, list) or len(payload) != 1:
            raise TypeError
        inspected = payload[0]
        image_id = inspected["Id"]
        actual_labels = inspected["Config"]["Labels"]
        os_name = inspected["Os"]
        architecture = inspected["Architecture"]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
        raise ReleaseError("release image inspection failed") from None
    match = _IMAGE_ID_PATTERN.fullmatch(image_id) if isinstance(image_id, str) else None
    if (
        match is None
        or match.group(1) == "0" * 64
        or not isinstance(actual_labels, dict)
        or any(actual_labels.get(name) != value for name, value in labels.items())
        or os_name != "linux"
        or architecture != "amd64"
    ):
        raise ReleaseError("release image inspection failed")
    return image_id


def write_release_env(path: Path, content: str) -> None:
    if "\r" in content or "\0" in content:
        raise ReleaseError("release environment publish failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, UnicodeError):
        raise ReleaseError("release environment publish failed") from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def check_release(root: Path = ROOT_DIR) -> None:
    load_images_lock(root / "docker" / "images.lock.env")
    _read_project_version(root)
    _read_repository_identity(root)


def build_release(root: Path = ROOT_DIR) -> None:
    images = load_images_lock(root / "docker" / "images.lock.env")
    version = _read_project_version(root)
    identity = _read_repository_identity(root)
    _require_clean_repository(root)
    if _run_command(["git", "rev-parse", "HEAD"], cwd=root) != identity.revision:
        raise ReleaseError("repository identity changed during release build")

    tag = f"{version}-{identity.revision[:8]}"
    api_image = f"trainfactory-api:{tag}"
    web_image = f"trainfactory-web:{tag}"
    common_args = [
        "--build-arg",
        f"BUILD_VERSION={version}",
        "--build-arg",
        f"VCS_REF={identity.revision}",
        "--build-arg",
        f"VCS_DATE={identity.vcs_date}",
        "--build-arg",
        f"SOURCE_REPOSITORY={identity.source_repository}",
    ]
    _run_command(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--file",
            str(root / "docker" / "Dockerfile"),
            "--build-arg",
            f"API_BASE_IMAGE={images['API_BASE_IMAGE']}",
            *common_args,
            "--tag",
            api_image,
            str(root),
        ],
        cwd=root,
    )
    _run_command(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--file",
            str(root / "web" / "Dockerfile"),
            "--build-arg",
            f"WEB_NODE_BUILD_IMAGE={images['WEB_NODE_BUILD_IMAGE']}",
            "--build-arg",
            f"WEB_NGINX_IMAGE={images['WEB_NGINX_IMAGE']}",
            *common_args,
            "--tag",
            web_image,
            str(root / "web"),
        ],
        cwd=root,
    )

    labels = _expected_labels(version, identity)
    api_id = _inspect_image(root, api_image, labels)
    web_id = _inspect_image(root, web_image, labels)

    final_identity = _read_repository_identity(root)
    _require_clean_repository(root)
    if final_identity != identity:
        raise ReleaseError("repository identity changed during release build")

    write_release_env(
        root / ".runtime" / "release.env",
        (
            f"API_IMAGE={api_image}\n"
            f"WEB_IMAGE={web_image}\n"
            f"RELEASE_REVISION={identity.revision}\n"
            f"API_IMAGE_ID={api_id}\n"
            f"WEB_IMAGE_ID={web_id}\n"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments not in (["--check"], ["--build"], ["--ci-build-args"]):
        print("release arguments are invalid", file=sys.stderr)
        return 2
    try:
        if arguments == ["--check"]:
            check_release()
        elif arguments == ["--build"]:
            build_release()
        else:
            print("\n".join(ci_build_arguments()))
    except ReleaseError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("release operation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
