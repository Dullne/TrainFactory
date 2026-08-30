import importlib.util
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).parents[1]
PREPARE_RELEASE = ROOT_DIR / "scripts" / "prepare_public_release.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_public_release_under_test",
        PREPARE_RELEASE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _successful_run(module, revision):
    return {
        "id": 123456,
        "name": "CI",
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_branch": "master",
        "head_sha": revision,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "repository": {"full_name": module.PUBLIC_REPOSITORY},
        "head_repository": {"full_name": module.PUBLIC_REPOSITORY},
    }


def _successful_jobs(module, revision):
    return {
        "total_count": len(module.EXPECTED_CI_JOBS),
        "jobs": [
            {
                "name": name,
                "run_id": 123456,
                "run_attempt": 1,
                "head_sha": revision,
                "status": "completed",
                "conclusion": "success",
            }
            for name in sorted(module.EXPECTED_CI_JOBS)
        ],
    }


def test_ci_evidence_requires_one_exact_successful_run_and_all_jobs():
    module = _load_module()
    revision = "1" * 40

    run_id = module.validate_ci_evidence(
        {"total_count": 1, "workflow_runs": [_successful_run(module, revision)]},
        _successful_jobs(module, revision),
        revision,
    )

    assert run_id == 123456


@pytest.mark.parametrize(
    ("mutation", "target"),
    (
        ("run-failed", "run"),
        ("wrong-branch", "run"),
        ("wrong-repository", "run"),
        ("job-failed", "jobs"),
        ("job-skipped", "jobs"),
        ("missing-job", "jobs"),
        ("extra-run", "runs"),
    ),
)
def test_ci_evidence_fails_closed_for_ambiguous_or_incomplete_results(
    mutation,
    target,
):
    module = _load_module()
    revision = "2" * 40
    run = _successful_run(module, revision)
    jobs = _successful_jobs(module, revision)
    runs = {"total_count": 1, "workflow_runs": [run]}
    if mutation == "run-failed":
        run["conclusion"] = "failure"
    elif mutation == "wrong-branch":
        run["head_branch"] = "release"
    elif mutation == "wrong-repository":
        run["head_repository"] = {"full_name": "private/example"}
    elif mutation == "job-failed":
        jobs["jobs"][0]["conclusion"] = "failure"
    elif mutation == "job-skipped":
        jobs["jobs"][0]["conclusion"] = "skipped"
    elif mutation == "missing-job":
        jobs["jobs"].pop()
        jobs["total_count"] -= 1
    else:
        runs["workflow_runs"].append(dict(run, id=654321))
        runs["total_count"] = 2

    with pytest.raises(module.PreparationError) as exc_info:
        module.validate_ci_evidence(runs, jobs, revision)

    assert str(exc_info.value) == "public CI evidence is not successful"
    assert target not in str(exc_info.value)


def test_publish_plan_has_only_fixed_ghcr_images_and_immutable_tags():
    module = _load_module()
    revision = "3" * 40
    release = module.LocalRelease(
        version="0.1.0",
        revision=revision,
        vcs_date="2026-08-30T00:00:00Z",
        api_image="trainfactory-api:0.1.0-33333333",
        web_image="trainfactory-web:0.1.0-33333333",
        api_id="sha256:" + "a" * 64,
        web_id="sha256:" + "b" * 64,
    )

    plan = module.create_publish_plan(release, ci_run_id=123456)

    assert plan["schema_version"] == 1
    assert plan["source_repository"] == module.PUBLIC_SOURCE_REPOSITORY
    assert plan["release_revision"] == revision
    assert set(plan["images"]) == {"api", "web"}
    assert plan["images"]["api"]["repository"] == (
        "ghcr.io/dullne/trainfactory-api"
    )
    assert plan["images"]["web"]["repository"] == (
        "ghcr.io/dullne/trainfactory-web"
    )
    for image in plan["images"].values():
        assert image["tags"] == [
            f"sha-{revision}",
            f"0.1.0-{revision[:8]}",
        ]
        assert "latest" not in image["tags"]
        assert image["local_image_id"].startswith("sha256:")
        assert "local_image" not in image


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("version", "0.1.0 latest"),
        ("api_image", "private.invalid/forged:latest"),
        ("web_image", "trainfactory-web:latest"),
        ("api_id", "sha256:" + "0" * 64),
        ("vcs_date", "not-a-date"),
    ),
)
def test_publish_plan_rejects_forged_local_release_fields(field, value):
    module = _load_module()
    revision = "3" * 40
    values = {
        "version": "0.1.0",
        "revision": revision,
        "vcs_date": "2026-08-30T00:00:00Z",
        "api_image": "trainfactory-api:0.1.0-33333333",
        "web_image": "trainfactory-web:0.1.0-33333333",
        "api_id": "sha256:" + "a" * 64,
        "web_id": "sha256:" + "b" * 64,
    }
    values[field] = value

    with pytest.raises(module.PreparationError) as exc_info:
        module.create_publish_plan(module.LocalRelease(**values), ci_run_id=123456)

    assert str(exc_info.value) == "public release plan is invalid"


def test_prepare_rejects_non_public_source_before_ci_build_or_docker(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    calls = []
    private_identity = module.build_release.RepositoryIdentity(
        revision="4" * 40,
        vcs_date="2026-08-30T00:00:00Z",
        source_repository="https://private.invalid/internal/example",
    )
    monkeypatch.setattr(
        module,
        "_read_repository_identity_safe",
        lambda root: private_identity,
    )
    monkeypatch.setattr(
        module,
        "fetch_ci_evidence",
        lambda revision: calls.append("ci"),
    )
    monkeypatch.setattr(
        module,
        "build_with_clean_docker_config",
        lambda root: calls.append("build"),
    )

    with pytest.raises(module.PreparationError) as exc_info:
        module.prepare_public_release(tmp_path)

    assert str(exc_info.value) == "public release source is invalid"
    assert calls == []


def test_public_identity_never_uses_build_release_git_runner(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    revision = "4" * 40
    vcs_date = "2026-08-30T00:00:00Z"

    monkeypatch.setattr(
        module.build_release,
        "_read_repository_identity",
        lambda root: pytest.fail("unsafe build_release Git runner was used"),
    )

    def fake_git(argv, *, cwd):
        if argv == ["rev-parse", "HEAD"]:
            return revision
        if argv == ["show", "-s", "--format=%cI", "HEAD"]:
            return vcs_date
        pytest.fail("unexpected Git identity command")

    monkeypatch.setattr(module, "_run_git_text", fake_git)
    monkeypatch.setattr(
        module,
        "_read_local_config_values",
        lambda root, key: (module.PUBLIC_CLONE_URL,),
    )

    identity = module._read_public_identity(tmp_path)

    assert identity.revision == revision
    assert identity.vcs_date == vcs_date
    assert identity.source_repository == module.PUBLIC_SOURCE_REPOSITORY


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong-job-run",
        "wrong-job-head",
        "wrong-job-attempt",
        "duplicate-job",
        "wrong-workflow",
        "wrong-event",
        "in-progress",
    ),
)
def test_ci_evidence_rejects_cross_run_or_non_final_jobs(mutation):
    module = _load_module()
    revision = "5" * 40
    run = _successful_run(module, revision)
    jobs = _successful_jobs(module, revision)
    if mutation == "wrong-job-run":
        jobs["jobs"][0]["run_id"] = 999999
    elif mutation == "wrong-job-head":
        jobs["jobs"][0]["head_sha"] = "6" * 40
    elif mutation == "wrong-job-attempt":
        jobs["jobs"][0]["run_attempt"] = 2
    elif mutation == "duplicate-job":
        jobs["jobs"][0]["name"] = jobs["jobs"][1]["name"]
    elif mutation == "wrong-workflow":
        run["path"] = ".github/workflows/other.yml"
    elif mutation == "wrong-event":
        run["event"] = "workflow_dispatch"
    else:
        jobs["jobs"][0]["status"] = "in_progress"
        jobs["jobs"][0]["conclusion"] = None

    with pytest.raises(module.PreparationError) as exc_info:
        module.validate_ci_evidence(
            {"total_count": 1, "workflow_runs": [run]},
            jobs,
            revision,
        )

    assert str(exc_info.value) == "public CI evidence is not successful"


def test_ci_requests_are_fixed_anonymous_and_bounded(monkeypatch):
    module = _load_module()
    revision = "7" * 40
    requests = []
    runs = {
        "total_count": 1,
        "workflow_runs": [_successful_run(module, revision)],
    }
    jobs = _successful_jobs(module, revision)

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            encoded = json.dumps(self.payload).encode("utf-8")
            assert limit == module._MAX_JSON_BYTES + 1
            return encoded

    def fake_urlopen(request, *, timeout):
        requests.append(request)
        assert timeout == 30
        if "/actions/workflows/ci.yml/runs?" in request.full_url:
            return Response(runs)
        return Response(jobs)

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    assert module.fetch_ci_evidence(revision) == 123456
    assert [request.full_url for request in requests] == [
        (
            f"{module.GITHUB_API_ROOT}/actions/workflows/ci.yml/runs?"
            f"branch=master&event=push&head_sha={revision}&per_page=100"
        ),
        (
            f"{module.GITHUB_API_ROOT}/actions/runs/123456/attempts/1/jobs?"
            "per_page=100"
        ),
    ]
    for request in requests:
        headers = {name.lower(): value for name, value in request.header_items()}
        assert "authorization" not in headers
        assert headers["accept"] == "application/vnd.github+json"


def test_ci_job_policy_is_bound_to_workflow_and_release_validation_command():
    module = _load_module()
    workflow = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    jobs_section = workflow.split("\njobs:\n", 1)[1]
    workflow_jobs = set(
        re.findall(r"(?m)^  ([a-z][a-z0-9-]+):\s*$", jobs_section)
    )

    assert workflow_jobs == module.EXPECTED_CI_JOBS
    assert (
        "tests/test_dependency_locks.py tests/test_build_release.py "
        "tests/test_prepare_public_release.py"
    ) in workflow


def test_ci_failure_stops_before_scan_build_or_plan(tmp_path, monkeypatch):
    module = _load_module()
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    stale_plan = runtime / "ghcr-publish-plan.json"
    stale_plan.write_text('{"stale":true}\n', encoding="utf-8")
    revision = "8" * 40
    identity = module.build_release.RepositoryIdentity(
        revision=revision,
        vcs_date="2026-08-30T00:00:00Z",
        source_repository=module.PUBLIC_SOURCE_REPOSITORY,
    )
    calls = []
    monkeypatch.setattr(module, "_read_public_identity", lambda root: identity)
    monkeypatch.setattr(
        module,
        "_require_public_checkout",
        lambda root, actual: calls.append("checkout"),
    )

    def fail_ci(actual_revision):
        calls.append("ci")
        raise module.PreparationError("public CI evidence is not successful")

    monkeypatch.setattr(module, "fetch_ci_evidence", fail_ci)
    monkeypatch.setattr(
        module,
        "_scan_tracked_secrets",
        lambda root: calls.append("scan"),
    )
    monkeypatch.setattr(
        module,
        "build_with_clean_docker_config",
        lambda root: calls.append("build"),
    )
    monkeypatch.setattr(
        module,
        "_write_publish_plan",
        lambda path, plan: calls.append("write"),
    )

    with pytest.raises(module.PreparationError) as exc_info:
        module.prepare_public_release(tmp_path)

    assert str(exc_info.value) == "public CI evidence is not successful"
    assert calls == ["checkout", "ci"]
    assert not stale_plan.exists()


def test_preparation_lock_rejects_concurrent_attempt_and_cleans_up(tmp_path):
    module = _load_module()

    with module._preparation_lock(tmp_path) as runtime:
        assert runtime == tmp_path / ".runtime"
        assert (runtime / "ghcr-prepare.lock").is_file()
        with pytest.raises(module.PreparationError) as exc_info:
            with module._preparation_lock(tmp_path):
                pytest.fail("concurrent lock must not be acquired")

    assert str(exc_info.value) == "public release preparation is already running"
    assert not (tmp_path / ".runtime" / "ghcr-prepare.lock").exists()


def test_success_builds_only_from_independent_public_clone(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    operator_root = tmp_path / "operator-checkout"
    build_root = tmp_path / "independent-public-clone"
    operator_root.mkdir()
    build_root.mkdir()
    revision = "8" * 40
    identity = module.build_release.RepositoryIdentity(
        revision=revision,
        vcs_date="2026-08-30T00:00:00Z",
        source_repository=module.PUBLIC_SOURCE_REPOSITORY,
    )
    release = module.LocalRelease(
        version="0.1.0",
        revision=revision,
        vcs_date=identity.vcs_date,
        api_image="trainfactory-api:0.1.0-88888888",
        web_image="trainfactory-web:0.1.0-88888888",
        api_id="sha256:" + "a" * 64,
        web_id="sha256:" + "b" * 64,
    )
    calls = []

    monkeypatch.setattr(
        module,
        "_read_public_identity",
        lambda root: calls.append(("identity", Path(root))) or identity,
    )
    monkeypatch.setattr(
        module,
        "_require_public_checkout",
        lambda root, actual: calls.append(("checkout", Path(root))),
    )
    monkeypatch.setattr(
        module,
        "fetch_ci_evidence",
        lambda actual: calls.append(("ci", actual)) or 123456,
    )
    monkeypatch.setattr(
        module,
        "_scan_tracked_secrets",
        lambda root: calls.append(("scan", Path(root))),
    )

    @contextmanager
    def fake_clone(actual_revision):
        calls.append(("clone", actual_revision))
        yield build_root

    monkeypatch.setattr(module, "_independent_public_clone", fake_clone)
    monkeypatch.setattr(
        module,
        "build_with_clean_docker_config",
        lambda root: calls.append(("build", Path(root))) or release,
    )
    monkeypatch.setattr(
        module,
        "_require_only_tracked_worktree_files",
        lambda root, **kwargs: calls.append(("tree", Path(root))),
    )
    written = {}
    monkeypatch.setattr(
        module,
        "_write_publish_plan",
        lambda path, plan: written.update(path=Path(path), plan=plan),
    )

    plan = module.prepare_public_release(operator_root)

    assert plan == written["plan"]
    assert written["path"] == operator_root / ".runtime" / "ghcr-publish-plan.json"
    assert ("clone", revision) in calls
    assert ("build", build_root) in calls
    assert ("build", operator_root) not in calls
    assert calls.index(("ci", revision)) < calls.index(("build", build_root))
    assert calls.count(("ci", revision)) == 2


def test_build_uses_empty_temporary_docker_config_and_restores_environment(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    revision = "9" * 40
    identity = module.build_release.RepositoryIdentity(
        revision=revision,
        vcs_date="2026-08-30T00:00:00Z",
        source_repository=module.PUBLIC_SOURCE_REPOSITORY,
    )
    release = module.LocalRelease(
        version="0.1.0",
        revision=revision,
        vcs_date=identity.vcs_date,
        api_image="trainfactory-api:0.1.0-99999999",
        web_image="trainfactory-web:0.1.0-99999999",
        api_id="sha256:" + "a" * 64,
        web_id="sha256:" + "b" * 64,
    )
    old_config = str(tmp_path / "existing-docker-config")
    old_host = "tcp://private-canary.invalid:2376"
    old_buildx = str(tmp_path / "existing-buildx-config")
    monkeypatch.setenv("DOCKER_CONFIG", old_config)
    monkeypatch.setenv("DOCKER_HOST", old_host)
    monkeypatch.setenv("BUILDX_CONFIG", old_buildx)
    observed = {}

    def fake_build(root):
        config_root = Path(os.environ["DOCKER_CONFIG"])
        observed["root"] = config_root
        observed["config"] = (config_root / "config.json").read_text(
            encoding="utf-8"
        )
        observed["docker_host"] = os.environ.get("DOCKER_HOST")
        observed["buildx_config"] = os.environ.get("BUILDX_CONFIG")

    monkeypatch.setattr(module.build_release, "build_release", fake_build)
    monkeypatch.setattr(
        module.build_release,
        "_read_repository_identity",
        lambda root: identity,
    )
    monkeypatch.setattr(module, "_parse_release_env", lambda root, actual: release)

    assert module.build_with_clean_docker_config(tmp_path) == release
    assert observed["config"] == '{"auths":{}}\n'
    assert observed["docker_host"] is None
    assert observed["buildx_config"] is None
    assert not observed["root"].exists()
    assert os.environ["DOCKER_CONFIG"] == old_config
    assert os.environ["DOCKER_HOST"] == old_host
    assert os.environ["BUILDX_CONFIG"] == old_buildx


def test_build_failure_restores_docker_environment_without_echo(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    old_config = str(tmp_path / "private-canary")
    monkeypatch.setenv("DOCKER_CONFIG", old_config)

    def fail_build(root):
        raise module.build_release.ReleaseError("private-canary")

    monkeypatch.setattr(module.build_release, "build_release", fail_build)

    with pytest.raises(module.PreparationError) as exc_info:
        module.build_with_clean_docker_config(tmp_path)

    assert str(exc_info.value) == "public release build failed"
    assert "private-canary" not in str(exc_info.value)
    assert os.environ["DOCKER_CONFIG"] == old_config


def test_release_env_parser_accepts_only_exact_verified_fields(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    revision = "a" * 40
    identity = module.build_release.RepositoryIdentity(
        revision=revision,
        vcs_date="2026-08-30T00:00:00Z",
        source_repository=module.PUBLIC_SOURCE_REPOSITORY,
    )
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    release_env = runtime / "release.env"
    release_env.write_text(
        (
            "API_IMAGE=trainfactory-api:0.1.0-aaaaaaaa\n"
            "WEB_IMAGE=trainfactory-web:0.1.0-aaaaaaaa\n"
            f"RELEASE_REVISION={revision}\n"
            f"API_IMAGE_ID=sha256:{'b' * 64}\n"
            f"WEB_IMAGE_ID=sha256:{'c' * 64}\n"
        ),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(
        module.build_release,
        "_read_project_version",
        lambda root: "0.1.0",
    )

    parsed = module._parse_release_env(tmp_path, identity)
    assert parsed.revision == revision
    assert parsed.api_id == "sha256:" + "b" * 64

    release_env.write_text(
        release_env.read_text(encoding="utf-8") + "EXTRA=forbidden\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(module.PreparationError) as exc_info:
        module._parse_release_env(tmp_path, identity)
    assert str(exc_info.value) == "public release build failed"


@pytest.mark.parametrize("mode", (b"120000", b"160000"))
def test_index_gate_rejects_symlinks_and_gitlinks(tmp_path, monkeypatch, mode):
    module = _load_module()
    payload = mode + b" " + b"d" * 40 + b" 0\tunsafe\0"
    monkeypatch.setattr(
        module,
        "_run_process",
        lambda *args, **kwargs: SimpleNamespace(stdout=payload),
    )

    with pytest.raises(module.PreparationError) as exc_info:
        module._require_safe_index(tmp_path)

    assert str(exc_info.value) == "public release checkout is invalid"


def test_index_gate_rejects_skip_worktree_and_assume_unchanged(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    stage_payload = b"100644 " + b"d" * 40 + b" 0\tsafe.txt\0"

    def fake_run(argv, **kwargs):
        if "--stage" in argv:
            return SimpleNamespace(stdout=stage_payload)
        return SimpleNamespace(stdout=b"S safe.txt\0")

    monkeypatch.setattr(module, "_run_process", fake_run)

    with pytest.raises(module.PreparationError) as exc_info:
        module._require_safe_index(tmp_path)

    assert str(exc_info.value) == "public release checkout is invalid"


def test_materialized_build_tree_rejects_ignored_or_untracked_files(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    (tmp_path / ".git").mkdir()
    (tmp_path / "safe.txt").write_text("safe\n", encoding="utf-8")
    hidden = tmp_path / "ignored-private.txt"
    hidden.write_text("private\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_tracked_worktree_paths",
        lambda root: {"safe.txt"},
    )

    with pytest.raises(module.PreparationError) as exc_info:
        module._require_only_tracked_worktree_files(tmp_path)

    assert str(exc_info.value) == "public release checkout is invalid"
    hidden.unlink()
    module._require_only_tracked_worktree_files(tmp_path)


def test_local_config_gate_rejects_url_rewrites_before_remote_lookup(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_read_local_config_keys",
        lambda root: {"url.https://private.invalid/.insteadof"},
    )
    monkeypatch.setattr(
        module,
        "_run_git_text",
        lambda *args, **kwargs: pytest.fail("remote lookup must not run"),
    )

    with pytest.raises(module.PreparationError) as exc_info:
        module._require_safe_local_config(tmp_path)

    assert str(exc_info.value) == "public release checkout is invalid"


def test_tool_resolution_rejects_checkout_local_executable(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    operator_root = tmp_path / "operator-checkout"
    build_root = tmp_path / "independent-build-source"
    operator_root.mkdir()
    build_root.mkdir()
    fake_docker = operator_root / "docker.exe"
    fake_docker.write_bytes(b"forged")
    monkeypatch.setattr(module, "ROOT_DIR", operator_root)
    monkeypatch.setattr(module.shutil, "which", lambda name: str(fake_docker))

    with pytest.raises(module.PreparationError) as exc_info:
        module._trusted_tool_path("docker", build_root)

    assert str(exc_info.value) == "public release tool is invalid"


def test_cli_rejects_unknown_arguments_without_echo(capsys):
    module = _load_module()

    assert module.main(["--token=private-canary"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "public release arguments are invalid\n"
    assert "private-canary" not in captured.err
