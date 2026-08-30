"""Prepare verified public release images without registry credentials."""

from __future__ import annotations

import os as _bootstrap_os
import sys


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
    _seen_paths = {_root_entry, _script_dir}
    for _entry in sys.path:
        _candidate = _bootstrap_realpath(_entry or _bootstrap_os.getcwd())
        if _candidate in _seen_paths:
            continue
        if _candidate in _pythonpath_entries and not _is_interpreter_path(_candidate):
            continue
        if not _is_interpreter_path(_candidate):
            continue
        _seen_paths.add(_candidate)
        _trusted_sys_path.append(_entry)
    sys.path[:] = [_root_entry, *_trusted_sys_path]

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import urlencode  # noqa: E402


ROOT_DIR = Path(_root_entry)
PUBLIC_REPOSITORY = "Dullne/TrainFactory"
PUBLIC_SOURCE_REPOSITORY = "https://github.com/Dullne/TrainFactory"
PUBLIC_CLONE_URL = f"{PUBLIC_SOURCE_REPOSITORY}.git"
PUBLIC_BRANCH = "master"
PUBLIC_BRANCH_REF = "refs/heads/master"
PUBLIC_REMOTE_REF = "refs/remotes/origin/master"
GITHUB_API_ROOT = f"https://api.github.com/repos/{PUBLIC_REPOSITORY}"
GHCR_REPOSITORIES = {
    "api": "ghcr.io/dullne/trainfactory-api",
    "web": "ghcr.io/dullne/trainfactory-web",
}
EXPECTED_CI_JOBS = frozenset(
    {
        "lock-and-static",
        "backend-full",
        "host-policy",
        "mysql-migrations",
        "frontend-quality",
        "frontend-e2e",
        "compose-smoke",
        "release-input-validation",
    }
)
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9A-Za-z-]+)+$")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_RELEASE_ENV_BYTES = 4096
_PROHIBITED_LOCAL_CONFIG_PREFIXES = (
    "alias.",
    "credential.",
    "diff.",
    "filter.",
    "http.",
    "include.",
    "includeif.",
    "merge.",
    "protocol.",
    "url.",
)
_PROHIBITED_LOCAL_CONFIG_KEYS = {
    "core.attributesfile",
    "core.excludesfile",
    "core.fsmonitor",
    "core.hookspath",
    "core.pager",
    "core.sshcommand",
    "core.sparsecheckout",
    "core.sparsecheckoutcone",
    "index.sparse",
    "interactive.difffilter",
}


class PreparationError(RuntimeError):
    """A privacy-safe public release preparation failure."""


@dataclass(frozen=True)
class LocalRelease:
    version: str
    revision: str
    vcs_date: str
    api_image: str
    web_image: str
    api_id: str
    web_id: str


def _load_build_release_module():
    module_name = "trainfactory_build_release_for_public_release"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = ROOT_DIR / "scripts" / "build_release.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PreparationError("public release preparation failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise PreparationError("public release preparation failed") from None
    return module


try:
    build_release = _load_build_release_module()
except PreparationError:
    build_release = None


def _safe_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
        }
    )
    return environment


def _trusted_tool_path(name: str, cwd: Path) -> Path:
    candidate_value = shutil.which(name)
    if candidate_value is None:
        raise PreparationError("public release tool is invalid")
    try:
        candidate = Path(candidate_value).resolve(strict=True)
        workspace = Path(cwd).resolve(strict=True)
        ambient_directory = Path.cwd().resolve(strict=True)
        metadata = candidate.lstat()
    except OSError:
        raise PreparationError("public release tool is invalid") from None
    if (
        not candidate.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or candidate.is_symlink()
        or _is_reparse_point(metadata)
        or _is_path_inside(candidate, ROOT_DIR.resolve())
        or _is_path_inside(candidate, workspace)
        or _is_path_inside(candidate, ambient_directory)
    ):
        raise PreparationError("public release tool is invalid")
    return candidate


def _run_process(
    argv: list[str],
    *,
    cwd: Path,
    allowed_returncodes: tuple[int, ...] = (0,),
    timeout: int = 120,
    text: bool = True,
) -> subprocess.CompletedProcess:
    arguments = list(argv)
    if arguments and arguments[0] in {"git", "docker"}:
        arguments[0] = str(_trusted_tool_path(arguments[0], cwd))
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            env=_safe_environment(),
            shell=False,
            capture_output=True,
            text=text,
            encoding="utf-8" if text else None,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        raise PreparationError("public release checkout is invalid") from None
    if completed.returncode not in allowed_returncodes:
        raise PreparationError("public release checkout is invalid")
    return completed


def _run_git_text(argv: list[str], *, cwd: Path) -> str:
    completed = _run_process(["git", *argv], cwd=cwd)
    return completed.stdout.strip()


def _resolve_git_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError:
        raise PreparationError("public release checkout is invalid") from None


def _is_path_inside(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _read_local_config_keys(root: Path) -> set[str]:
    completed = _run_process(
        ["git", "config", "--local", "--name-only", "--null", "--list"],
        cwd=root,
        text=False,
    )
    try:
        return {
            item.decode("utf-8").lower()
            for item in completed.stdout.split(b"\0")
            if item
        }
    except UnicodeDecodeError:
        raise PreparationError("public release checkout is invalid") from None


def _read_local_config_values(root: Path, key: str) -> tuple[str, ...]:
    completed = _run_process(
        ["git", "config", "--local", "--null", "--get-all", key],
        cwd=root,
        allowed_returncodes=(0, 1),
        text=False,
    )
    if completed.returncode == 1:
        if completed.stdout:
            raise PreparationError("public release checkout is invalid")
        return ()
    if not completed.stdout or not completed.stdout.endswith(b"\0"):
        raise PreparationError("public release checkout is invalid")
    try:
        return tuple(
            item.decode("utf-8") for item in completed.stdout[:-1].split(b"\0")
        )
    except UnicodeDecodeError:
        raise PreparationError("public release checkout is invalid") from None


def _require_safe_local_config(root: Path) -> None:
    keys = _read_local_config_keys(root)
    if any(
        key.startswith(_PROHIBITED_LOCAL_CONFIG_PREFIXES)
        or key in _PROHIBITED_LOCAL_CONFIG_KEYS
        or key.endswith(".pushurl")
        or key.endswith(".uploadpack")
        or key.endswith(".receivepack")
        for key in keys
    ):
        raise PreparationError("public release checkout is invalid")

    if _read_local_config_values(root, "remote.origin.url") != (
        PUBLIC_CLONE_URL,
    ):
        raise PreparationError("public release checkout is invalid")
    if _read_local_config_values(root, "remote.origin.pushurl"):
        raise PreparationError("public release checkout is invalid")


def _require_safe_index(root: Path) -> None:
    completed = _run_process(
        ["git", "ls-files", "--stage", "-z", "--"],
        cwd=root,
        text=False,
    )
    indexed_paths: list[bytes] = []
    for entry in completed.stdout.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, _path = entry.split(b"\t", 1)
            mode, object_id, stage = metadata.split(b" ", 2)
        except ValueError:
            raise PreparationError("public release checkout is invalid") from None
        if (
            mode not in {b"100644", b"100755"}
            or stage != b"0"
            or len(object_id) not in {40, 64}
            or object_id == b"0" * len(object_id)
            or re.fullmatch(rb"[0-9a-f]+", object_id) is None
        ):
            raise PreparationError("public release checkout is invalid")
        indexed_paths.append(_path)

    flags = _run_process(
        ["git", "ls-files", "-v", "-z", "--"],
        cwd=root,
        text=False,
    )
    flagged_paths: list[bytes] = []
    for entry in flags.stdout.split(b"\0"):
        if not entry:
            continue
        if not entry.startswith(b"H "):
            raise PreparationError("public release checkout is invalid")
        flagged_paths.append(entry[2:])
    if flagged_paths != indexed_paths:
        raise PreparationError("public release checkout is invalid")


def _decode_tracked_path(raw_path: bytes) -> str:
    try:
        path = raw_path.decode("utf-8")
    except UnicodeDecodeError:
        raise PreparationError("public release checkout is invalid") from None
    segments = path.split("/")
    reserved_names = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(
            not segment
            or segment in {".", ".."}
            or ":" in segment
            or segment.endswith((" ", "."))
            or segment.split(".", 1)[0].casefold() in reserved_names
            or any(ord(character) < 32 or ord(character) == 127 for character in segment)
            for segment in segments
        )
    ):
        raise PreparationError("public release checkout is invalid")
    return path


def _tracked_worktree_paths(root: Path) -> set[str]:
    completed = _run_process(
        ["git", "ls-files", "-z", "--"],
        cwd=root,
        text=False,
    )
    paths = {
        _decode_tracked_path(raw_path)
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    }
    folded = {path.casefold() for path in paths}
    if len(paths) != len(folded):
        raise PreparationError("public release checkout is invalid")
    return paths


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & attribute)


def _require_only_tracked_worktree_files(
    root: Path,
    *,
    allowed_untracked: frozenset[str] = frozenset(),
) -> None:
    tracked = _tracked_worktree_paths(root)
    allowed = set(allowed_untracked)
    if any(_decode_tracked_path(path.encode("utf-8")) != path for path in allowed):
        raise PreparationError("public release checkout is invalid")
    expected_files = tracked | allowed
    expected_directories = {
        "/".join(path.split("/")[:index])
        for path in expected_files
        for index in range(1, len(path.split("/")))
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    stack: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    try:
        while stack:
            directory, relative_parts = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not relative_parts and entry.name == ".git":
                        metadata = entry.stat(follow_symlinks=False)
                        if (
                            not stat.S_ISDIR(metadata.st_mode)
                            or entry.is_symlink()
                            or _is_reparse_point(metadata)
                        ):
                            raise PreparationError(
                                "public release checkout is invalid"
                            )
                        continue
                    relative = "/".join((*relative_parts, entry.name))
                    _decode_tracked_path(relative.encode("utf-8"))
                    metadata = entry.stat(follow_symlinks=False)
                    if entry.is_symlink() or _is_reparse_point(metadata):
                        raise PreparationError("public release checkout is invalid")
                    if stat.S_ISDIR(metadata.st_mode):
                        actual_directories.add(relative)
                        stack.append((Path(entry.path), (*relative_parts, entry.name)))
                    elif stat.S_ISREG(metadata.st_mode):
                        actual_files.add(relative)
                    else:
                        raise PreparationError("public release checkout is invalid")
    except PreparationError:
        raise
    except (OSError, UnicodeError):
        raise PreparationError("public release checkout is invalid") from None
    if actual_files != expected_files or actual_directories != expected_directories:
        raise PreparationError("public release checkout is invalid")


def _require_clean_checkout(root: Path) -> None:
    completed = _run_process(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        text=False,
    )
    if completed.stdout:
        raise PreparationError("public release checkout is invalid")


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "TrainFactory-public-release-preparation",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise PreparationError("public CI evidence is not successful")
            raw = response.read(_MAX_JSON_BYTES + 1)
    except PreparationError:
        raise
    except (OSError, urllib.error.URLError, ValueError):
        raise PreparationError("public CI evidence is not successful") from None
    if len(raw) > _MAX_JSON_BYTES:
        raise PreparationError("public CI evidence is not successful")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise PreparationError("public CI evidence is not successful") from None
    if not isinstance(payload, dict):
        raise PreparationError("public CI evidence is not successful")
    return payload


def _fetch_public_master_revision() -> str:
    try:
        payload = _fetch_json(f"{GITHUB_API_ROOT}/git/ref/heads/{PUBLIC_BRANCH}")
    except PreparationError:
        raise PreparationError("public release checkout is invalid") from None
    try:
        reference = payload["ref"]
        target = payload["object"]
        target_type = target["type"]
        revision = target["sha"]
    except (KeyError, TypeError):
        raise PreparationError("public release checkout is invalid") from None
    if (
        reference != PUBLIC_BRANCH_REF
        or target_type != "commit"
        or not isinstance(revision, str)
        or _REVISION_PATTERN.fullmatch(revision) is None
    ):
        raise PreparationError("public release checkout is invalid")
    return revision


def _require_public_checkout(
    root: Path,
    identity,
) -> None:
    try:
        workspace = root.resolve(strict=True)
    except OSError:
        raise PreparationError("public release checkout is invalid") from None
    if not workspace.is_dir():
        raise PreparationError("public release checkout is invalid")

    top_level = _resolve_git_path(
        workspace,
        _run_git_text(["rev-parse", "--show-toplevel"], cwd=workspace),
    )
    git_dir = _resolve_git_path(
        workspace,
        _run_git_text(["rev-parse", "--git-dir"], cwd=workspace),
    )
    common_dir = _resolve_git_path(
        workspace,
        _run_git_text(["rev-parse", "--git-common-dir"], cwd=workspace),
    )
    if (
        top_level != workspace
        or not _is_path_inside(git_dir, workspace)
        or not _is_path_inside(common_dir, workspace)
        or _run_git_text(["rev-parse", "--is-bare-repository"], cwd=workspace)
        != "false"
        or _run_git_text(["rev-parse", "--is-shallow-repository"], cwd=workspace)
        != "false"
        or (common_dir / "objects" / "info" / "alternates").exists()
        or (common_dir / "info" / "grafts").exists()
        or (common_dir / "info" / "attributes").exists()
        or _run_git_text(
            ["for-each-ref", "--format=%(refname)", "refs/replace/"],
            cwd=workspace,
        )
    ):
        raise PreparationError("public release checkout is invalid")

    _require_safe_local_config(workspace)
    if _run_git_text(["remote"], cwd=workspace).splitlines() != ["origin"]:
        raise PreparationError("public release checkout is invalid")
    if _run_git_text(["symbolic-ref", "-q", "HEAD"], cwd=workspace) != PUBLIC_BRANCH_REF:
        raise PreparationError("public release checkout is invalid")

    head = _run_git_text(["rev-parse", "HEAD"], cwd=workspace)
    local_master = _run_git_text(["rev-parse", PUBLIC_BRANCH_REF], cwd=workspace)
    remote_master = _run_git_text(["rev-parse", PUBLIC_REMOTE_REF], cwd=workspace)
    _require_safe_index(workspace)
    _require_clean_checkout(workspace)
    if (
        _REVISION_PATTERN.fullmatch(head) is None
        or head != identity.revision
        or head != local_master
        or head != remote_master
        or head != _fetch_public_master_revision()
    ):
        raise PreparationError("public release checkout is invalid")


def validate_ci_evidence(
    runs_payload: dict,
    jobs_payload: dict,
    revision: str,
) -> int:
    error = PreparationError("public CI evidence is not successful")
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise error
    try:
        runs = runs_payload["workflow_runs"]
        total_runs = runs_payload["total_count"]
        jobs = jobs_payload["jobs"]
        total_jobs = jobs_payload["total_count"]
    except (KeyError, TypeError):
        raise error from None
    if (
        type(total_runs) is not int
        or total_runs != 1
        or not isinstance(runs, list)
        or len(runs) != 1
        or type(total_jobs) is not int
        or total_jobs != len(EXPECTED_CI_JOBS)
        or not isinstance(jobs, list)
        or len(jobs) != len(EXPECTED_CI_JOBS)
    ):
        raise error

    run = runs[0]
    try:
        run_id = run["id"]
        repository = run["repository"]["full_name"]
        head_repository = run["head_repository"]["full_name"]
        run_attempt = run["run_attempt"]
    except (KeyError, TypeError):
        raise error from None
    if (
        type(run_id) is not int
        or run_id <= 0
        or type(run_attempt) is not int
        or run_attempt <= 0
        or run.get("name") != "CI"
        or run.get("path") != ".github/workflows/ci.yml"
        or run.get("event") != "push"
        or run.get("head_branch") != PUBLIC_BRANCH
        or run.get("head_sha") != revision
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or repository != PUBLIC_REPOSITORY
        or head_repository != PUBLIC_REPOSITORY
    ):
        raise error

    job_names: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise error
        name = job.get("name")
        if (
            not isinstance(name, str)
            or job.get("run_id") != run_id
            or job.get("run_attempt") != run_attempt
            or job.get("head_sha") != revision
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
        ):
            raise error
        job_names.append(name)
    if len(set(job_names)) != len(job_names) or set(job_names) != EXPECTED_CI_JOBS:
        raise error
    return run_id


def fetch_ci_evidence(revision: str) -> int:
    query = urlencode(
        {
            "branch": PUBLIC_BRANCH,
            "event": "push",
            "head_sha": revision,
            "per_page": "100",
        }
    )
    runs = _fetch_json(
        f"{GITHUB_API_ROOT}/actions/workflows/ci.yml/runs?{query}"
    )
    try:
        workflow_runs = runs["workflow_runs"]
        run_id = workflow_runs[0]["id"]
        run_attempt = workflow_runs[0]["run_attempt"]
    except (KeyError, IndexError, TypeError):
        raise PreparationError("public CI evidence is not successful") from None
    if (
        type(run_id) is not int
        or run_id <= 0
        or type(run_attempt) is not int
        or run_attempt <= 0
    ):
        raise PreparationError("public CI evidence is not successful")
    jobs = _fetch_json(
        f"{GITHUB_API_ROOT}/actions/runs/{run_id}/attempts/"
        f"{run_attempt}/jobs?per_page=100"
    )
    return validate_ci_evidence(runs, jobs, revision)


def _scan_tracked_secrets(root: Path) -> None:
    completed = _run_process(
        [
            sys.executable,
            "-I",
            str(root / "scripts" / "scan_tracked_secrets.py"),
            "--repo",
            str(root),
        ],
        cwd=root,
        allowed_returncodes=(0, 1, 2),
        timeout=300,
    )
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise PreparationError("tracked secret scan failed")


def _parse_release_env(root: Path, identity) -> LocalRelease:
    path = root / ".runtime" / "release.env"
    try:
        raw = path.read_bytes()
    except OSError:
        raise PreparationError("public release build failed") from None
    if (
        len(raw) > _MAX_RELEASE_ENV_BYTES
        or b"\r" in raw
        or b"\0" in raw
        or not raw.endswith(b"\n")
    ):
        raise PreparationError("public release build failed")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise PreparationError("public release build failed") from None
    expected_keys = (
        "API_IMAGE",
        "WEB_IMAGE",
        "RELEASE_REVISION",
        "API_IMAGE_ID",
        "WEB_IMAGE_ID",
    )
    values: dict[str, str] = {}
    for line, expected_key in zip(lines, expected_keys, strict=False):
        if "=" not in line:
            raise PreparationError("public release build failed")
        key, value = line.split("=", 1)
        if key != expected_key or not value:
            raise PreparationError("public release build failed")
        values[key] = value
    if len(lines) != len(expected_keys) or tuple(values) != expected_keys:
        raise PreparationError("public release build failed")

    try:
        version = build_release._read_project_version(root)
    except build_release.ReleaseError:
        raise PreparationError("public release build failed") from None
    tag = f"{version}-{identity.revision[:8]}"
    api_id = values["API_IMAGE_ID"]
    web_id = values["WEB_IMAGE_ID"]
    if (
        values["API_IMAGE"] != f"trainfactory-api:{tag}"
        or values["WEB_IMAGE"] != f"trainfactory-web:{tag}"
        or values["RELEASE_REVISION"] != identity.revision
        or _IMAGE_ID_PATTERN.fullmatch(api_id) is None
        or _IMAGE_ID_PATTERN.fullmatch(web_id) is None
        or api_id == "sha256:" + "0" * 64
        or web_id == "sha256:" + "0" * 64
    ):
        raise PreparationError("public release build failed")
    return LocalRelease(
        version=version,
        revision=identity.revision,
        vcs_date=identity.vcs_date,
        api_image=values["API_IMAGE"],
        web_image=values["WEB_IMAGE"],
        api_id=api_id,
        web_id=web_id,
    )


def build_with_clean_docker_config(root: Path) -> LocalRelease:
    saved_environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper().startswith(("DOCKER_", "BUILDX_", "BUILDKIT_"))
    }
    affected_names = {
        name
        for name in os.environ
        if name.upper().startswith(("DOCKER_", "BUILDX_", "BUILDKIT_"))
    }
    original_run_command = build_release._run_command

    def trusted_release_command(argv: list[str], *, cwd: Path) -> str:
        arguments = list(argv)
        if not arguments or arguments[0] not in {"git", "docker"}:
            raise build_release.ReleaseError("release command failed")
        arguments[0] = str(_trusted_tool_path(arguments[0], cwd))
        return original_run_command(arguments, cwd=cwd)

    try:
        with tempfile.TemporaryDirectory(prefix="trainfactory-public-build-") as directory:
            config_root = Path(directory)
            config_root.chmod(0o700)
            config_path = config_root / "config.json"
            config_path.write_text('{"auths":{}}\n', encoding="utf-8", newline="\n")
            config_path.chmod(0o600)
            for name in affected_names:
                os.environ.pop(name, None)
            os.environ["DOCKER_CONFIG"] = str(config_root)
            build_release._run_command = trusted_release_command
            try:
                build_release.build_release(root)
                identity = build_release._read_repository_identity(root)
                return _parse_release_env(root, identity)
            except PreparationError:
                raise
            except build_release.ReleaseError:
                raise PreparationError("public release build failed") from None
            except (OSError, UnicodeError):
                raise PreparationError("public release build failed") from None
    finally:
        build_release._run_command = original_run_command
        for name in list(os.environ):
            if name.upper().startswith(("DOCKER_", "BUILDX_", "BUILDKIT_")):
                os.environ.pop(name, None)
        os.environ.update(saved_environment)


def create_publish_plan(release: LocalRelease, *, ci_run_id: int) -> dict:
    expected_local_tag = f"{release.version}-{release.revision[:8]}"
    if (
        _REVISION_PATTERN.fullmatch(release.revision) is None
        or release.revision == "0" * 40
        or not isinstance(release.version, str)
        or _VERSION_PATTERN.fullmatch(release.version) is None
        or not isinstance(release.vcs_date, str)
        or not build_release._valid_rfc3339(release.vcs_date)
        or release.api_image != f"trainfactory-api:{expected_local_tag}"
        or release.web_image != f"trainfactory-web:{expected_local_tag}"
        or _IMAGE_ID_PATTERN.fullmatch(release.api_id) is None
        or _IMAGE_ID_PATTERN.fullmatch(release.web_id) is None
        or release.api_id == "sha256:" + "0" * 64
        or release.web_id == "sha256:" + "0" * 64
        or type(ci_run_id) is not int
        or ci_run_id <= 0
    ):
        raise PreparationError("public release plan is invalid")
    expected_labels = {
        "org.opencontainers.image.version": release.version,
        "org.opencontainers.image.revision": release.revision,
        "org.opencontainers.image.created": release.vcs_date,
        "org.opencontainers.image.source": PUBLIC_SOURCE_REPOSITORY,
    }

    def image_plan(kind: str, local_id: str) -> dict:
        return {
            "repository": GHCR_REPOSITORIES[kind],
            "tags": [
                f"sha-{release.revision}",
                f"{release.version}-{release.revision[:8]}",
            ],
            "local_image_id": local_id,
            "platform": "linux/amd64",
            "expected_labels": dict(expected_labels),
        }

    return {
        "schema_version": 1,
        "source_repository": PUBLIC_SOURCE_REPOSITORY,
        "release_revision": release.revision,
        "version": release.version,
        "ci_run_id": ci_run_id,
        "ci_run_url": (
            f"{PUBLIC_SOURCE_REPOSITORY}/actions/runs/{ci_run_id}"
        ),
        "images": {
            "api": image_plan("api", release.api_id),
            "web": image_plan("web", release.web_id),
        },
    }


def _write_publish_plan(path: Path, plan: dict) -> None:
    try:
        content = json.dumps(
            plan,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        ) + "\n"
    except (TypeError, ValueError):
        raise PreparationError("public release plan is invalid") from None
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
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
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError:
        raise PreparationError("public release plan publish failed") from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _safe_runtime_directory(root: Path) -> Path:
    try:
        workspace = root.resolve(strict=True)
        if not workspace.is_dir():
            raise OSError
        runtime = workspace / ".runtime"
        runtime.mkdir(mode=0o700, exist_ok=True)
        metadata = runtime.lstat()
        resolved_runtime = runtime.resolve(strict=True)
    except OSError:
        raise PreparationError("public release runtime is invalid") from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or runtime.is_symlink()
        or _is_reparse_point(metadata)
        or resolved_runtime.parent != workspace
    ):
        raise PreparationError("public release runtime is invalid")
    return runtime


def _invalidate_publish_plan(runtime: Path) -> None:
    plan_path = runtime / "ghcr-publish-plan.json"
    try:
        metadata = plan_path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise PreparationError("public release plan invalidation failed") from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or plan_path.is_symlink()
        or _is_reparse_point(metadata)
    ):
        raise PreparationError("public release plan invalidation failed")
    try:
        plan_path.unlink()
    except OSError:
        raise PreparationError("public release plan invalidation failed") from None


@contextmanager
def _preparation_lock(root: Path):
    runtime = _safe_runtime_directory(Path(root))
    lock_path = runtime / "ghcr-prepare.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    lock_identity: tuple[int, int] | None = None
    try:
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except FileExistsError:
            raise PreparationError(
                "public release preparation is already running"
            ) from None
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or _is_reparse_point(metadata):
            raise PreparationError("public release preparation lock failed")
        lock_identity = metadata.st_dev, metadata.st_ino
        os.fsync(descriptor)
        yield runtime
    except PreparationError:
        raise
    except OSError:
        raise PreparationError("public release preparation lock failed") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if lock_identity is not None:
            try:
                current = lock_path.lstat()
                if (
                    stat.S_ISREG(current.st_mode)
                    and not lock_path.is_symlink()
                    and not _is_reparse_point(current)
                    and (current.st_dev, current.st_ino) == lock_identity
                ):
                    lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _read_repository_identity_safe(root: Path):
    if build_release is None:
        raise PreparationError("public release preparation failed")
    try:
        revision = _run_git_text(["rev-parse", "HEAD"], cwd=root)
        vcs_date = _run_git_text(
            ["show", "-s", "--format=%cI", "HEAD"],
            cwd=root,
        )
        remote_urls = _read_local_config_values(root, "remote.origin.url")
        if len(remote_urls) != 1:
            raise PreparationError("public release source is invalid")
        source_repository = build_release.normalize_source_repository(remote_urls[0])
    except (PreparationError, build_release.ReleaseError):
        raise PreparationError("public release source is invalid") from None
    if (
        _REVISION_PATTERN.fullmatch(revision) is None
        or revision == "0" * 40
        or not build_release._valid_rfc3339(vcs_date)
    ):
        raise PreparationError("public release source is invalid")
    return build_release.RepositoryIdentity(
        revision=revision,
        vcs_date=vcs_date,
        source_repository=source_repository,
    )


def _read_public_identity(root: Path):
    identity = _read_repository_identity_safe(root)
    if identity.source_repository != PUBLIC_SOURCE_REPOSITORY:
        raise PreparationError("public release source is invalid")
    return identity


@contextmanager
def _independent_public_clone(revision: str):
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise PreparationError("public release checkout is invalid")
    try:
        with tempfile.TemporaryDirectory(
            prefix="trainfactory-public-source-"
        ) as directory:
            temporary_root = Path(directory)
            template_root = temporary_root / "empty-template"
            build_root = temporary_root / "source"
            template_root.mkdir()
            _run_process(
                [
                    "git",
                    "clone",
                    "--no-local",
                    "--no-hardlinks",
                    "--no-tags",
                    "--single-branch",
                    "--branch",
                    PUBLIC_BRANCH,
                    "--origin",
                    "origin",
                    "--config",
                    "core.autocrlf=false",
                    f"--template={template_root}",
                    PUBLIC_CLONE_URL,
                    str(build_root),
                ],
                cwd=temporary_root,
                timeout=3600,
            )
            identity = _read_public_identity(build_root)
            if identity.revision != revision:
                raise PreparationError("public release checkout is invalid")
            _require_public_checkout(build_root, identity)
            _run_process(
                ["git", "fsck", "--full", "--strict", "--no-reflogs"],
                cwd=build_root,
                timeout=3600,
            )
            _require_only_tracked_worktree_files(build_root)
            yield build_root
    except PreparationError:
        raise
    except OSError:
        raise PreparationError("public release checkout is invalid") from None


def _prepare_locked(workspace: Path, runtime: Path) -> dict:
    if build_release is None:
        raise PreparationError("public release preparation failed")
    identity = _read_public_identity(workspace)
    _require_public_checkout(workspace, identity)
    ci_run_id = fetch_ci_evidence(identity.revision)
    _scan_tracked_secrets(workspace)

    with _independent_public_clone(identity.revision) as build_root:
        _scan_tracked_secrets(build_root)
        release = build_with_clean_docker_config(build_root)
        build_identity = _read_public_identity(build_root)
        if build_identity != identity or release.revision != identity.revision:
            raise PreparationError("public release checkout is invalid")
        _require_public_checkout(build_root, build_identity)
        _require_only_tracked_worktree_files(
            build_root,
            allowed_untracked=frozenset({".runtime/release.env"}),
        )
        _scan_tracked_secrets(build_root)

    final_identity = _read_public_identity(workspace)
    if final_identity != identity:
        raise PreparationError("public release checkout is invalid")
    _require_public_checkout(workspace, final_identity)
    if fetch_ci_evidence(final_identity.revision) != ci_run_id:
        raise PreparationError("public CI evidence is not successful")
    _scan_tracked_secrets(workspace)

    plan = create_publish_plan(release, ci_run_id=ci_run_id)
    _write_publish_plan(runtime / "ghcr-publish-plan.json", plan)
    return plan


def prepare_public_release(root: Path = ROOT_DIR) -> dict:
    workspace = Path(root)
    with _preparation_lock(workspace) as runtime:
        _invalidate_publish_plan(runtime)
        return _prepare_locked(workspace, runtime)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments != ["--prepare"]:
        print("public release arguments are invalid", file=sys.stderr)
        return 2
    try:
        prepare_public_release()
    except PreparationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("public release preparation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
