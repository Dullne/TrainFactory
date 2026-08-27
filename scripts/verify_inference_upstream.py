import argparse
import hashlib
import json
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GIT_COMMAND_TIMEOUT_SECONDS = 600


class ProvenanceError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ProvenanceError(
            "unable to load inference compatibility contract"
        ) from None
    if not isinstance(contract, dict) or contract.get("schema_version") != 2:
        raise ProvenanceError("unsupported inference compatibility contract")
    return contract


def _git_bytes(
    arguments: list[str],
    *,
    hooks_path: Path,
    failure_message: str,
) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-c", f"core.hooksPath={hooks_path}", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        raise ProvenanceError("Git executable is unavailable") from None
    except subprocess.TimeoutExpired:
        raise ProvenanceError("Git command timed out") from None
    except OSError:
        raise ProvenanceError("unable to execute Git command") from None
    if completed.returncode != 0:
        raise ProvenanceError(failure_message)
    return completed.stdout


def _git_text(
    arguments: list[str],
    *,
    hooks_path: Path,
    failure_message: str,
) -> str:
    try:
        return (
            _git_bytes(
                arguments,
                hooks_path=hooks_path,
                failure_message=failure_message,
            )
            .decode("utf-8")
            .rstrip("\r\n")
        )
    except UnicodeDecodeError:
        raise ProvenanceError(failure_message) from None


def _ensure_bare_repository(
    *,
    framework: str,
    repository_url: str,
    cache_root: Path,
    hooks_path: Path,
) -> Path:
    repository = cache_root / f"{framework}.git"
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ProvenanceError(
            f"{framework}: unable to create bare repository cache"
        ) from None

    try:
        repository_metadata = repository.lstat()
    except FileNotFoundError:
        repository_metadata = None
    except OSError:
        raise ProvenanceError(f"{framework}: invalid bare repository cache") from None

    if repository_metadata is None:
        _git_text(
            ["init", "--bare", str(repository)],
            hooks_path=hooks_path,
            failure_message=f"{framework}: unable to initialize bare repository cache",
        )
        _git_text(
            ["--git-dir", str(repository), "remote", "add", "origin", repository_url],
            hooks_path=hooks_path,
            failure_message=f"{framework}: unable to configure repository URL",
        )
    else:
        try:
            expected_repository = cache_root.resolve(strict=True) / repository.name
            resolved_repository = repository.resolve(strict=True)
        except (OSError, RuntimeError):
            raise ProvenanceError(
                f"{framework}: invalid bare repository cache"
            ) from None
        if (
            stat.S_ISLNK(repository_metadata.st_mode)
            or getattr(repository_metadata, "st_reparse_tag", 0) != 0
            or resolved_repository != expected_repository
        ):
            raise ProvenanceError(f"{framework}: invalid bare repository cache")
        is_bare = _git_text(
            ["--git-dir", str(repository), "rev-parse", "--is-bare-repository"],
            hooks_path=hooks_path,
            failure_message=f"{framework}: invalid bare repository cache",
        )
        if is_bare != "true":
            raise ProvenanceError(f"{framework}: invalid bare repository cache")

    actual_url = _git_text(
        ["--git-dir", str(repository), "remote", "get-url", "origin"],
        hooks_path=hooks_path,
        failure_message=f"{framework}: repository URL mismatch",
    )
    if actual_url != repository_url:
        raise ProvenanceError(f"{framework}: repository URL mismatch")
    return repository


def _fetch_declared_tag(
    *,
    framework: str,
    repository: Path,
    tag: str,
    hooks_path: Path,
) -> str:
    audit_ref = f"refs/trainfactory/provenance/{framework}/tag"
    _git_text(
        [
            "--git-dir",
            str(repository),
            "fetch",
            "--force",
            "--no-tags",
            "origin",
            f"refs/tags/{tag}:{audit_ref}",
        ],
        hooks_path=hooks_path,
        failure_message=f"{framework}: unable to fetch declared tag",
    )
    return audit_ref


def _verify_source(
    *,
    framework: str,
    source_name: str,
    source_contract: dict[str, Any],
    repository: Path,
    commit: str,
    workspace_root: Path,
    hooks_path: Path,
) -> None:
    upstream_path = source_contract["upstream_path"]
    object_spec = f"{commit}:{upstream_path}"
    actual_blob = _git_text(
        ["--git-dir", str(repository), "rev-parse", object_spec],
        hooks_path=hooks_path,
        failure_message=f"{framework}/{source_name}: source path is missing",
    )
    if actual_blob != source_contract["git_blob_sha1"]:
        raise ProvenanceError(f"{framework}/{source_name}: Git blob mismatch")

    source_bytes = _git_bytes(
        ["--git-dir", str(repository), "cat-file", "blob", object_spec],
        hooks_path=hooks_path,
        failure_message=f"{framework}/{source_name}: source path is missing",
    )
    if len(source_bytes) != source_contract["size"]:
        raise ProvenanceError(f"{framework}/{source_name}: source size mismatch")
    if hashlib.sha256(source_bytes).hexdigest() != source_contract["sha256"]:
        raise ProvenanceError(f"{framework}/{source_name}: source SHA-256 mismatch")

    vendored_path = source_contract.get("vendored_path")
    if vendored_path is None:
        return
    resolved_workspace = workspace_root.resolve()
    resolved_vendored = (resolved_workspace / vendored_path).resolve()
    if not resolved_vendored.is_relative_to(resolved_workspace):
        raise ProvenanceError(
            f"{framework}/{source_name}: vendored path escapes workspace root"
        )
    try:
        vendored_bytes = resolved_vendored.read_bytes()
    except OSError:
        raise ProvenanceError(
            f"{framework}/{source_name}: vendored source mismatch"
        ) from None
    if vendored_bytes != source_bytes:
        raise ProvenanceError(f"{framework}/{source_name}: vendored source mismatch")


def verify_framework(
    *,
    framework: str,
    framework_contract: dict[str, Any],
    cache_root: Path,
    workspace_root: Path,
    hooks_path: Path,
) -> None:
    if not isinstance(framework_contract, dict):
        raise ProvenanceError("invalid inference compatibility contract")
    repository_contract = framework_contract.get("repository")
    sources = framework_contract.get("sources")
    if (
        not isinstance(repository_contract, dict)
        or not isinstance(sources, dict)
        or any(
            not isinstance(source_contract, dict)
            for source_contract in sources.values()
        )
    ):
        raise ProvenanceError("invalid inference compatibility contract")
    repository = _ensure_bare_repository(
        framework=framework,
        repository_url=repository_contract["url"],
        cache_root=cache_root,
        hooks_path=hooks_path,
    )
    audit_ref = _fetch_declared_tag(
        framework=framework,
        repository=repository,
        tag=repository_contract["tag"],
        hooks_path=hooks_path,
    )

    object_type = _git_text(
        ["--git-dir", str(repository), "cat-file", "-t", audit_ref],
        hooks_path=hooks_path,
        failure_message=f"{framework}: unable to inspect declared tag",
    )
    tag_kind = {"commit": "lightweight", "tag": "annotated"}.get(object_type)
    if tag_kind != repository_contract["tag_kind"]:
        raise ProvenanceError(f"{framework}: tag kind mismatch")

    commit = _git_text(
        ["--git-dir", str(repository), "rev-parse", f"{audit_ref}^{{commit}}"],
        hooks_path=hooks_path,
        failure_message=f"{framework}: unable to peel declared tag",
    )
    if commit != repository_contract["commit"]:
        raise ProvenanceError(f"{framework}: tag commit mismatch")

    for source_name, source_contract in sources.items():
        _verify_source(
            framework=framework,
            source_name=source_name,
            source_contract=source_contract,
            repository=repository,
            commit=commit,
            workspace_root=workspace_root,
            hooks_path=hooks_path,
        )


def verify_contract(
    *,
    contract: dict[str, Any],
    contract_path: Path,
    cache_root: Path,
    workspace_root: Path,
) -> None:
    del contract_path
    if contract.get("schema_version") != 2:
        raise ProvenanceError("unsupported inference compatibility contract")
    try:
        with tempfile.TemporaryDirectory(
            prefix="trainfactory-git-hooks-",
        ) as hooks_directory:
            hooks_path = Path(hooks_directory)
            for framework in ("xinference", "sglang"):
                verify_framework(
                    framework=framework,
                    framework_contract=contract[framework],
                    cache_root=cache_root,
                    workspace_root=workspace_root,
                    hooks_path=hooks_path,
                )
    except OSError:
        raise ProvenanceError("unable to create safe Git hooks directory") from None
    except (AttributeError, KeyError, TypeError):
        raise ProvenanceError("invalid inference compatibility contract") from None


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ProvenanceError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Verify inference source provenance with Git objects.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except SystemExit as error:
        return 0 if error.code == 0 else 1
    except ProvenanceError as error:
        print(f"upstream provenance verification failed: {error}", file=sys.stderr)
        return 1

    try:
        contract = load_contract(arguments.contract)
        verify_contract(
            contract=contract,
            contract_path=arguments.contract,
            cache_root=arguments.cache_root,
            workspace_root=arguments.workspace_root,
        )
    except ProvenanceError as error:
        print(f"upstream provenance verification failed: {error}", file=sys.stderr)
        return 1

    for framework in ("xinference", "sglang"):
        repository = contract[framework]["repository"]
        source_count = len(contract[framework]["sources"])
        print(
            f"{framework}: tag={repository['tag']} "
            f"commit={repository['commit']} sources={source_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
