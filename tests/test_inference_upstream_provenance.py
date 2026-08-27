import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT_DIR = Path(__file__).parents[1]
VERIFIER_PATH = ROOT_DIR / "scripts" / "verify_inference_upstream.py"

pytestmark = pytest.mark.host_tools


def _load_verifier():
    assert VERIFIER_PATH.is_file(), "upstream provenance verifier is missing"
    spec = importlib.util.spec_from_file_location(
        "verify_inference_upstream_under_test",
        VERIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("GIT_DEFAULT_HASH", None)
    environment.update(
        {
            "APP_TIMEZONE": "Asia/Shanghai",
            "DEBUG": "false",
            "GIT_AUTHOR_EMAIL": "provenance@example.invalid",
            "GIT_AUTHOR_NAME": "Provenance Test",
            "GIT_COMMITTER_EMAIL": "provenance@example.invalid",
            "GIT_COMMITTER_NAME": "Provenance Test",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    return environment


def _git(*arguments: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        env=_git_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.rstrip("\r\n")


def _source_record(upstream_path: str, content: bytes, blob: str) -> dict[str, object]:
    return {
        "upstream_path": upstream_path,
        "git_blob_sha1": blob,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


@pytest.fixture
def provenance_case(tmp_path: Path) -> SimpleNamespace:
    work_repository = tmp_path / "work-repository"
    bare_remote = tmp_path / "upstream.git"
    workspace_root = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    contract_path = tmp_path / "contract.json"
    work_repository.mkdir()
    workspace_root.mkdir()

    _git("init", "--bare", "--object-format=sha1", str(bare_remote), cwd=tmp_path)
    _git("init", "--object-format=sha1", cwd=work_repository)
    _git("config", "core.autocrlf", "false", cwd=work_repository)

    vendored_bytes = b"value = 'upstream'\nsecond_line = True\n"
    non_vendored_bytes = b"{% set mode = 'rerank' %}\n{{ mode }}\n"
    vendored_upstream_path = "src/vendored.py"
    non_vendored_upstream_path = "templates/rerank.jinja"
    vendored_source = work_repository / vendored_upstream_path
    non_vendored_source = work_repository / non_vendored_upstream_path
    vendored_source.parent.mkdir(parents=True)
    non_vendored_source.parent.mkdir(parents=True)
    vendored_source.write_bytes(vendored_bytes)
    non_vendored_source.write_bytes(non_vendored_bytes)

    _git(
        "add",
        "--",
        vendored_upstream_path,
        non_vendored_upstream_path,
        cwd=work_repository,
    )
    _git("commit", "-m", "Add provenance fixtures", cwd=work_repository)
    commit = _git("rev-parse", "HEAD", cwd=work_repository)
    vendored_blob = _git(
        "rev-parse", f"{commit}:{vendored_upstream_path}", cwd=work_repository
    )
    non_vendored_blob = _git(
        "rev-parse",
        f"{commit}:{non_vendored_upstream_path}",
        cwd=work_repository,
    )
    for object_id in (commit, vendored_blob, non_vendored_blob):
        assert len(object_id) == 40
        assert set(object_id) <= set("0123456789abcdef")

    lightweight_tag = "lightweight-v1"
    annotated_tag = "annotated-v1"
    _git("tag", lightweight_tag, cwd=work_repository)
    _git(
        "tag",
        "-a",
        annotated_tag,
        "-m",
        "Annotated provenance tag",
        cwd=work_repository,
    )
    remote_url = str(bare_remote.resolve())
    _git("remote", "add", "origin", remote_url, cwd=work_repository)
    _git(
        "push",
        "origin",
        f"refs/tags/{lightweight_tag}",
        f"refs/tags/{annotated_tag}",
        cwd=work_repository,
    )

    vendored_path = "vendor/vendored.py"
    workspace_vendored = workspace_root / vendored_path
    workspace_vendored.parent.mkdir(parents=True)
    workspace_vendored.write_bytes(vendored_bytes)

    vendored_record = _source_record(
        vendored_upstream_path, vendored_bytes, vendored_blob
    )
    vendored_record["vendored_path"] = vendored_path
    contract = {
        "schema_version": 2,
        "xinference": {
            "repository": {
                "url": remote_url,
                "tag": lightweight_tag,
                "tag_kind": "lightweight",
                "commit": commit,
            },
            "sources": {"vendored": vendored_record},
        },
        "sglang": {
            "repository": {
                "url": remote_url,
                "tag": annotated_tag,
                "tag_kind": "annotated",
                "commit": commit,
            },
            "sources": {
                "non_vendored": _source_record(
                    non_vendored_upstream_path,
                    non_vendored_bytes,
                    non_vendored_blob,
                )
            },
        },
    }
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return SimpleNamespace(
        contract=contract,
        contract_path=contract_path,
        cache_root=cache_root,
        workspace_root=workspace_root,
        workspace_vendored=workspace_vendored,
        vendored_bytes=vendored_bytes,
    )


def _verify(
    module, case: SimpleNamespace, contract: dict[str, object] | None = None
) -> None:
    module.verify_contract(
        contract=case.contract if contract is None else contract,
        contract_path=case.contract_path,
        cache_root=case.cache_root,
        workspace_root=case.workspace_root,
    )


def _changed_contract(case: SimpleNamespace) -> dict[str, object]:
    return copy.deepcopy(case.contract)


def _install_reference_transaction_hook(hook_path: Path, marker: Path) -> None:
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path = marker.resolve().as_posix()
    if marker.drive:
        marker_path = f"/{marker.drive[0].lower()}{marker_path[2:]}"
    marker_path = marker_path.replace("'", "'\"'\"'")
    hook_path.write_bytes(f"#!/bin/sh\nprintf invoked > '{marker_path}'\n".encode())
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC)


def _cli_contract() -> dict[str, object]:
    return {
        "schema_version": 2,
        "xinference": {
            "repository": {
                "url": "https://example.invalid/xinference.git",
                "tag": "xinference-v1",
                "tag_kind": "lightweight",
                "commit": "1" * 40,
            },
            "sources": {"embedding": {}},
        },
        "sglang": {
            "repository": {
                "url": "https://example.invalid/sglang.git",
                "tag": "sglang-v1",
                "tag_kind": "annotated",
                "commit": "2" * 40,
            },
            "sources": {"template": {}, "handler": {}},
        },
    }


def _expected_cli_success() -> str:
    return (
        f"xinference: tag=xinference-v1 commit={'1' * 40} sources=1\n"
        f"sglang: tag=sglang-v1 commit={'2' * 40} sources=2\n"
    )


@pytest.mark.parametrize(
    "vendored_path",
    [
        "docker/xinference-patches/sentence_transformers_core.py",
        "docker/xinference-patches/rerank_sentence_transformers_core.py",
    ],
)
def test_vendored_xinference_snapshots_are_forced_to_lf(vendored_path):
    assert _git("check-attr", "eol", "--", vendored_path, cwd=ROOT_DIR) == (
        f"{vendored_path}: eol: lf"
    )


def test_verify_contract_accepts_exact_tags_blobs_and_vendored_bytes(provenance_case):
    module = _load_verifier()

    assert _verify(module, provenance_case) is None


def test_verify_contract_rejects_reused_cache_with_repository_url_mismatch(
    provenance_case,
):
    module = _load_verifier()
    _verify(module, provenance_case)
    _git(
        "--git-dir",
        str(provenance_case.cache_root / "xinference.git"),
        "remote",
        "set-url",
        "origin",
        str(provenance_case.cache_root / "different.git"),
        cwd=provenance_case.cache_root,
    )

    with pytest.raises(module.ProvenanceError, match="repository URL mismatch"):
        _verify(module, provenance_case)


@pytest.mark.parametrize("hook_source", ["repository", "configured"])
def test_verify_contract_disables_hooks_from_reused_cache(
    provenance_case,
    tmp_path,
    hook_source,
):
    module = _load_verifier()
    _verify(module, provenance_case)
    repository = provenance_case.cache_root / "xinference.git"
    _git(
        "--git-dir",
        str(repository),
        "update-ref",
        "-d",
        "refs/trainfactory/provenance/xinference/tag",
        cwd=provenance_case.cache_root,
    )
    marker = repository / "untrusted-hook-ran"
    if hook_source == "repository":
        hooks_root = repository / "hooks"
    else:
        hooks_root = tmp_path / "untrusted-hooks"
        _git(
            "--git-dir",
            str(repository),
            "config",
            "core.hooksPath",
            str(hooks_root),
            cwd=provenance_case.cache_root,
        )
    _install_reference_transaction_hook(
        hooks_root / "reference-transaction",
        marker,
    )

    _verify(module, provenance_case)

    assert not marker.exists()


def test_verify_contract_rejects_symlinked_reused_cache(provenance_case):
    module = _load_verifier()
    _verify(module, provenance_case)
    repository = provenance_case.cache_root / "xinference.git"
    real_repository = provenance_case.cache_root / "xinference-real.git"
    repository.rename(real_repository)
    try:
        repository.symlink_to(real_repository, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        real_repository.rename(repository)
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(module.ProvenanceError, match="invalid bare repository cache"):
        _verify(module, provenance_case)


def test_verify_contract_rejects_tag_kind_mismatch(provenance_case):
    module = _load_verifier()
    contract = _changed_contract(provenance_case)
    contract["xinference"]["repository"]["tag_kind"] = "annotated"

    with pytest.raises(module.ProvenanceError, match="tag kind mismatch"):
        _verify(module, provenance_case, contract)


def test_verify_contract_rejects_tag_commit_mismatch(provenance_case):
    module = _load_verifier()
    contract = _changed_contract(provenance_case)
    contract["xinference"]["repository"]["commit"] = "0" * 40

    with pytest.raises(module.ProvenanceError, match="tag commit mismatch"):
        _verify(module, provenance_case, contract)


def test_verify_contract_rejects_missing_source_path(provenance_case):
    module = _load_verifier()
    contract = _changed_contract(provenance_case)
    contract["xinference"]["sources"]["vendored"]["upstream_path"] = "src/missing.py"

    with pytest.raises(module.ProvenanceError, match="source path is missing"):
        _verify(module, provenance_case, contract)


def test_verify_contract_rejects_git_blob_mismatch(provenance_case):
    module = _load_verifier()
    contract = _changed_contract(provenance_case)
    contract["xinference"]["sources"]["vendored"]["git_blob_sha1"] = "0" * 40

    with pytest.raises(module.ProvenanceError, match="Git blob mismatch"):
        _verify(module, provenance_case, contract)


def test_verify_contract_rejects_source_size_mismatch(provenance_case):
    module = _load_verifier()
    contract = _changed_contract(provenance_case)
    contract["xinference"]["sources"]["vendored"]["size"] += 1

    with pytest.raises(module.ProvenanceError, match="source size mismatch"):
        _verify(module, provenance_case, contract)


def test_verify_contract_rejects_source_sha256_mismatch(provenance_case):
    module = _load_verifier()
    contract = _changed_contract(provenance_case)
    contract["xinference"]["sources"]["vendored"]["sha256"] = "0" * 64

    with pytest.raises(module.ProvenanceError, match="source SHA-256 mismatch"):
        _verify(module, provenance_case, contract)


def test_verify_contract_rejects_vendored_source_mismatch(provenance_case):
    module = _load_verifier()
    provenance_case.workspace_vendored.write_bytes(b"value = 'drifted'\n")

    with pytest.raises(module.ProvenanceError, match="vendored source mismatch"):
        _verify(module, provenance_case)


def test_verify_contract_rejects_vendored_path_escape(provenance_case, tmp_path):
    module = _load_verifier()
    escaped_source = tmp_path / "escaped.py"
    escaped_source.write_bytes(provenance_case.vendored_bytes)
    contract = _changed_contract(provenance_case)
    contract["xinference"]["sources"]["vendored"]["vendored_path"] = "../escaped.py"

    with pytest.raises(
        module.ProvenanceError, match="vendored path escapes workspace root"
    ):
        _verify(module, provenance_case, contract)


def test_verify_contract_reads_raw_blobs_when_bare_cache_has_autocrlf_true(
    provenance_case,
):
    module = _load_verifier()
    _verify(module, provenance_case)
    for framework in ("xinference", "sglang"):
        bare_cache = provenance_case.cache_root / f"{framework}.git"
        _git(
            "--git-dir",
            str(bare_cache),
            "config",
            "core.autocrlf",
            "true",
            cwd=provenance_case.cache_root,
        )
        assert (
            _git(
                "--git-dir",
                str(bare_cache),
                "config",
                "--get",
                "core.autocrlf",
                cwd=provenance_case.cache_root,
            )
            == "true"
        )

    assert _verify(module, provenance_case) is None


def test_load_contract_rejects_schema_v1(tmp_path):
    module = _load_verifier()
    contract_path = tmp_path / "schema-v1.json"
    contract_path.write_text('{"schema_version": 1}\n', encoding="utf-8")

    with pytest.raises(
        module.ProvenanceError,
        match="unsupported inference compatibility contract",
    ):
        module.load_contract(contract_path)


def test_cli_success_uses_repository_root_as_default_workspace(
    monkeypatch, capsys, tmp_path
):
    module = _load_verifier()
    contract = _cli_contract()
    contract_path = tmp_path / "contract.json"
    cache_root = tmp_path / "cache"
    repository_root = tmp_path / "repository-root"
    captured_call = {}
    monkeypatch.setattr(module, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(module, "load_contract", lambda path: contract)
    monkeypatch.setattr(
        module,
        "verify_contract",
        lambda **kwargs: captured_call.update(kwargs),
    )

    exit_code = module.main(
        ["--contract", str(contract_path), "--cache-root", str(cache_root)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == _expected_cli_success()
    assert captured.err == ""
    assert captured_call == {
        "contract": contract,
        "contract_path": contract_path,
        "cache_root": cache_root,
        "workspace_root": repository_root,
    }


def test_cli_success_honors_explicit_workspace_override(monkeypatch, capsys, tmp_path):
    module = _load_verifier()
    contract = _cli_contract()
    contract_path = tmp_path / "contract.json"
    cache_root = tmp_path / "cache"
    workspace_root = tmp_path / "explicit-workspace"
    captured_call = {}
    monkeypatch.setattr(module, "load_contract", lambda path: contract)
    monkeypatch.setattr(
        module,
        "verify_contract",
        lambda **kwargs: captured_call.update(kwargs),
    )

    exit_code = module.main(
        [
            "--contract",
            str(contract_path),
            "--cache-root",
            str(cache_root),
            "--workspace-root",
            str(workspace_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == _expected_cli_success()
    assert captured.err == ""
    assert captured_call["workspace_root"] == workspace_root


def test_cli_reports_provenance_error_to_stderr_and_returns_one(
    monkeypatch, capsys, tmp_path
):
    module = _load_verifier()
    monkeypatch.setattr(module, "load_contract", lambda path: _cli_contract())

    def fail_verification(**_kwargs):
        raise module.ProvenanceError("tag commit mismatch")

    monkeypatch.setattr(module, "verify_contract", fail_verification)

    exit_code = module.main(
        [
            "--contract",
            str(tmp_path / "contract.json"),
            "--cache-root",
            str(tmp_path / "cache"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "upstream provenance verification failed: tag commit mismatch\n"
    )


def test_cli_requires_contract_and_cache_root(capsys):
    module = _load_verifier()

    exit_code = module.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "upstream provenance verification failed: "
        "the following arguments are required: --contract, --cache-root\n"
    )


def test_cli_reports_malformed_schema_v2_as_concise_failure(capsys, provenance_case):
    module = _load_verifier()
    contract = _changed_contract(provenance_case)
    contract["xinference"]["sources"] = None
    provenance_case.contract_path.write_text(
        json.dumps(contract) + "\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--contract",
            str(provenance_case.contract_path),
            "--cache-root",
            str(provenance_case.cache_root),
            "--workspace-root",
            str(provenance_case.workspace_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "upstream provenance verification failed: "
        "invalid inference compatibility contract\n"
    )


def test_cli_reports_invalid_utf8_contract_as_concise_failure(capsys, tmp_path):
    module = _load_verifier()
    contract_path = tmp_path / "invalid-utf8.json"
    contract_path.write_bytes(b"\xff")

    exit_code = module.main(
        [
            "--contract",
            str(contract_path),
            "--cache-root",
            str(tmp_path / "cache"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "upstream provenance verification failed: "
        "unable to load inference compatibility contract\n"
    )


@pytest.mark.parametrize(
    ("raised_error", "expected_message"),
    [
        (
            subprocess.TimeoutExpired(cmd=["git", "fetch"], timeout=600),
            "Git command timed out",
        ),
        (PermissionError("denied"), "unable to execute Git command"),
    ],
)
def test_verify_contract_converts_git_execution_failures(
    monkeypatch,
    provenance_case,
    raised_error,
    expected_message,
):
    module = _load_verifier()
    calls = []

    def fail_git(*args, **kwargs):
        calls.append((args, kwargs))
        raise raised_error

    monkeypatch.setattr(module.subprocess, "run", fail_git)

    with pytest.raises(module.ProvenanceError, match=f"^{expected_message}$"):
        _verify(module, provenance_case)

    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 600


def test_verify_contract_reports_unavailable_git(monkeypatch, provenance_case):
    module = _load_verifier()
    calls = []

    def unavailable_git(*args, **kwargs):
        calls.append((args, kwargs))
        raise FileNotFoundError

    monkeypatch.setattr(module.subprocess, "run", unavailable_git)

    with pytest.raises(module.ProvenanceError, match="^Git executable is unavailable$"):
        _verify(module, provenance_case)

    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 600
