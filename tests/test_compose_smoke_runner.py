import errno
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).parents[1]
RUN_ID = "1" * 32
REVISION = "a" * 40
API_ID = "sha256:" + "b" * 64
WEB_ID = "sha256:" + "c" * 64


def _module():
    from scripts import run_compose_smoke

    return run_compose_smoke


def _argument(arguments, name):
    return arguments[arguments.index(name) + 1]


class FakeRun:
    def __init__(
        self,
        root,
        *,
        fail_stage=None,
        stale_resource=False,
        exact_name_collision=False,
        engine_endpoint="npipe:////./pipe/docker_engine",
    ):
        self.root = root
        self.fail_stage = fail_stage
        self.stale_resource = stale_resource
        self.exact_name_collision = exact_name_collision
        self.engine_endpoint = engine_endpoint
        self.calls = []
        self.built_tags = set()

    def _stage(self, arguments):
        if arguments[:2] == ["docker", "build"]:
            return "build-api" if "docker/Dockerfile.test" in arguments else "build-web"
        if arguments[:3] == [sys.executable, "-I", os.fspath(self.root / "scripts" / "materialize_compose_secrets.py")]:
            return "materialize-cleanup" if "cleanup" in arguments else "materialize-create"
        if arguments[:3] == [sys.executable, "-I", os.fspath(self.root / "scripts" / "compose_manifest.py")]:
            return "freeze" if "freeze" in arguments else "image-selection"
        if arguments[:3] == [sys.executable, "-I", os.fspath(self.root / "scripts" / "compose_release.py")]:
            return "safe-down" if "--safe-down-volumes" in arguments else "up"
        if arguments[:3] == [sys.executable, "-I", os.fspath(self.root / "scripts" / "verify_deployment.py")]:
            return "verify"
        if arguments[:3] == ["docker", "image", "rm"]:
            return "image-rm"
        return "probe"

    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        stage = self._stage(arguments)
        self.calls.append((stage, arguments, kwargs))
        if stage == self.fail_stage:
            return subprocess.CompletedProcess(
                arguments,
                1,
                stdout="",
                stderr="PRIVATE-SMOKE-BACKEND-CANARY",
            )
        if arguments[:3] == ["docker", "context", "inspect"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=self.engine_endpoint + "\n",
                stderr="",
            )
        if arguments[:3] in (
            ["docker", "ps", "-aq"],
            ["docker", "volume", "ls"],
            ["docker", "network", "ls"],
        ):
            output = "stale-resource\n" if self.stale_resource and "label=" in " ".join(arguments) else ""
            return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")
        if arguments[:4] in (
            ["docker", "container", "ls", "-aq"],
            ["docker", "volume", "ls", "-q"],
            ["docker", "network", "ls", "-q"],
        ):
            output = "wrong-label-collision\n" if self.exact_name_collision else ""
            return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")
        if arguments[:3] == ["docker", "image", "ls"]:
            reference = _argument(arguments, "--filter").removeprefix("reference=")
            output = f"{reference}\n" if reference in self.built_tags else ""
            return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")
        if arguments[:3] == ["docker", "image", "inspect"]:
            image = arguments[-1]
            image_id = API_ID if "api-smoke" in image else WEB_ID
            payload = {
                "Id": image_id,
                "RepoTags": [image, "inherited:tag"],
                "Os": "linux",
                "Architecture": "amd64",
                "Config": {
                    "Labels": {
                        "maintainer": "NGINX Docker Maintainers <docker-maint@nginx.com>",
                        "org.opencontainers.image.version": "0.1.0-ci-smoke",
                        "org.opencontainers.image.revision": REVISION,
                        "org.opencontainers.image.created": "2026-08-17T00:00:00Z",
                        "org.opencontainers.image.source": "https://example.invalid/train-factory",
                    }
                },
            }
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=json.dumps(payload) + "\n",
                stderr="",
            )
        if stage == "materialize-create":
            env_out = Path(_argument(arguments, "--env-out"))
            state_out = Path(_argument(arguments, "--state-out"))
            env_out.parent.mkdir(parents=True)
            env_out.write_text("PATHS-ONLY\n", encoding="utf-8")
            state_out.write_text("{}\n", encoding="utf-8")
            secret_values = {
                "mysql_root_password": "root-private-canary",
                "mysql_app_password": "app private+canary",
                "mysql_url": "mysql+pymysql://user:app%20private%2Bcanary@mysql/db",
                "jwt_secret_key": "jwt-private-canary",
                "default_admin_password": "admin-private-canary",
                "admin_username": "admin",
            }
            for name, value in secret_values.items():
                (env_out.parent / name).write_bytes(value.encode("utf-8"))
        elif stage == "image-selection":
            Path(_argument(arguments, "--output")).write_text(
                "SELECTION\n",
                encoding="utf-8",
            )
        elif stage == "freeze":
            Path(_argument(arguments, "--output")).write_text(
                "{}\n",
                encoding="utf-8",
            )
        elif stage == "materialize-cleanup":
            state = Path(_argument(arguments, "--state"))
            if state.parent.exists():
                shutil.rmtree(state.parent)
        elif stage in {"build-api", "build-web"}:
            tag = _argument(arguments, "-t")
            self.built_tags.add(tag)
            Path(_argument(arguments, "--iidfile")).write_bytes(
                (API_ID if "api-smoke" in tag else WEB_ID).encode("ascii")
            )
        elif stage == "image-rm":
            self.built_tags.discard(arguments[-1])
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")


def _root(tmp_path, *, create_runtime=True):
    root = tmp_path / "repo"
    (root / "docker").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "web").mkdir()
    if create_runtime:
        (root / ".runtime").mkdir()
    (root / "docker" / "images.lock.env").write_text(
        (ROOT_DIR / "docker" / "images.lock.env").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return root


def _argv(*, github_actions_mask=False):
    arguments = [
        "--images-lock",
        "docker/images.lock.env",
        "--revision",
        REVISION,
        "--vcs-date",
        "2026-08-17T00:00:00Z",
        "--source-repository",
        "https://example.invalid/train-factory",
        "--work-root",
        ".runtime",
        "--no-gpu",
    ]
    if github_actions_mask:
        arguments.append("--github-actions-mask")
    return arguments


def test_compose_smoke_runs_exact_transaction_and_cleans_owned_artifacts(
    tmp_path,
    capsys,
):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root)
    requested_bytes = []

    result = module.main(
        _argv(),
        root=root,
        run=fake,
        token_hex=lambda size: requested_bytes.append(size) or RUN_ID,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert captured.out == f"compose smoke passed project=trainfactory-ci-{RUN_ID}\n"
    assert requested_bytes == [16]
    stages = [stage for stage, _arguments, _kwargs in fake.calls]
    assert stages.index("build-api") < stages.index("build-web")
    assert stages.index("materialize-create") < stages.index("image-selection")
    assert stages.index("image-selection") < stages.index("freeze") < stages.index("up")
    assert stages.index("up") < stages.index("verify") < stages.index("safe-down")
    assert stages.index("safe-down") < stages.index("materialize-cleanup")
    assert stages.count("image-rm") == 2
    api_build = next(arguments for stage, arguments, _kwargs in fake.calls if stage == "build-api")
    web_build = next(arguments for stage, arguments, _kwargs in fake.calls if stage == "build-web")
    assert "API_TEST_BASE_IMAGE=" in " ".join(api_build)
    assert "WEB_NODE_BUILD_IMAGE=" in " ".join(web_build)
    assert "WEB_NGINX_IMAGE=" in " ".join(web_build)
    assert f"VCS_REF={REVISION}" in api_build
    assert f"VCS_REF={REVISION}" in web_build
    assert "--quiet" in api_build and "--quiet" in web_build
    assert api_build[api_build.index("--platform") + 1] == "linux/amd64"
    assert web_build[web_build.index("--platform") + 1] == "linux/amd64"
    assert "--iidfile" in api_build and "--iidfile" in web_build
    up = next(arguments for stage, arguments, _kwargs in fake.calls if stage == "up")
    assert up[-9:] == [
        "--",
        "up",
        "-d",
        "--wait",
        "--wait-timeout",
        "600",
        "mysql",
        "train-factory-api",
        "train-factory-web",
    ]
    verify = next(arguments for stage, arguments, _kwargs in fake.calls if stage == "verify")
    assert "--scope" in verify and _argument(verify, "--scope") == "full"
    assert "--no-gpu" in verify
    assert "--username-file" in verify and "--password-file" in verify
    assert not (root / ".runtime" / f"ci-release-{RUN_ID}.env").exists()
    assert not (root / ".runtime" / f"ci-compose-{RUN_ID}-manifest.json").exists()
    assert not (root / ".runtime" / f"ci-{RUN_ID}").exists()
    assert not list((root / ".runtime").glob("*.iid"))


def test_clean_checkout_runtime_is_created_and_validated(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path, create_runtime=False)
    fake = FakeRun(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 0
    assert not (root / ".runtime").exists()
    assert capsys.readouterr().err == ""


def test_created_runtime_is_removed_after_early_engine_failure(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path, create_runtime=False)
    fake = FakeRun(root, fail_stage="probe")

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert not (root / ".runtime").exists()
    assert capsys.readouterr().err == "compose smoke failed\n"


class _ConcurrentRuntimeContent(FakeRun):
    def __call__(self, arguments, **kwargs):
        completed = super().__call__(arguments, **kwargs)
        if self._stage(list(arguments)) == "image-rm" and not self.built_tags:
            (self.root / ".runtime" / "foreign").write_text(
                "FOREIGN-CANARY", encoding="utf-8"
            )
        return completed


def test_created_runtime_with_concurrent_content_is_preserved(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path, create_runtime=False)
    fake = _ConcurrentRuntimeContent(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    marker = root / ".runtime" / "foreign"
    assert result == 1
    assert marker.read_text(encoding="utf-8") == "FOREIGN-CANARY"
    assert capsys.readouterr().err == "compose smoke failed\n"


@pytest.mark.parametrize("kind", ("file", "reparse"))
def test_runtime_non_directory_or_reparse_is_rejected_before_docker(
    tmp_path,
    monkeypatch,
    capsys,
    kind,
):
    module = _module()
    root = _root(tmp_path, create_runtime=False)
    runtime = root / ".runtime"
    if kind == "file":
        runtime.write_text("not-a-directory", encoding="utf-8")
    else:
        runtime.mkdir()
        original = module._is_reparse
        monkeypatch.setattr(
            module,
            "_is_reparse",
            lambda path, metadata: path == runtime or original(path, metadata),
        )
    fake = FakeRun(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert fake.calls == []
    assert capsys.readouterr().err == "compose smoke failed\n"


def test_child_environment_pins_engine_and_keeps_windows_plugin_roots(
    tmp_path,
    monkeypatch,
):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root)
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.setenv("ProgramW6432", r"C:\Program Files")
    monkeypatch.setenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
    monkeypatch.setenv("DOCKER_CONTEXT", "mutable-context")
    monkeypatch.setenv("PYTHONPATH", "PRIVATE-PYTHONPATH-CANARY")

    assert module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    ) == 0

    for _stage, arguments, kwargs in fake.calls[1:]:
        environment = kwargs["env"]
        assert environment["DOCKER_HOST"] == "npipe:////./pipe/docker_engine"
        assert "DOCKER_CONTEXT" not in environment
        assert "PYTHONPATH" not in environment
        folded = {key.upper(): value for key, value in environment.items()}
        assert folded["PROGRAMFILES"] == r"C:\Program Files"
        assert folded["PROGRAMW6432"] == r"C:\Program Files"
        assert folded["PROGRAMFILES(X86)"] == r"C:\Program Files (x86)"


def test_github_masking_occurs_immediately_after_materialization(
    tmp_path,
    monkeypatch,
    capsys,
):
    from scripts import materialize_compose_secrets

    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        materialize_compose_secrets,
        "_verify_hardened_path",
        lambda _path: True,
    )
    masks = []

    def mask(value):
        assert fake.calls[-1][0] == "materialize-create"
        masks.append(value)

    result = module.main(
        _argv(github_actions_mask=True),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
        mask_sink=mask,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "app private+canary" in masks
    assert "app%20private%2Bcanary" in masks
    assert "app+private%2Bcanary" in masks
    assert "admin" not in masks
    assert all(value not in captured.out for value in masks)
    assert all(value not in captured.err for value in masks)


def test_github_mask_flag_requires_trusted_actions_environment(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    result = module.main(
        _argv(github_actions_mask=True),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
        mask_sink=lambda _value: pytest.fail("mask sink must not run"),
    )

    assert result == 1
    assert fake.calls == []
    assert capsys.readouterr().err == "compose smoke failed\n"


def test_github_actions_requires_mask_flag_before_docker(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    result = module.main(
        _argv(),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
        mask_sink=lambda _value: pytest.fail("mask sink must not run"),
    )

    assert result == 1
    assert fake.calls == []
    assert capsys.readouterr().err == "compose smoke failed\n"


def test_github_mask_sink_flushes_each_workflow_command(monkeypatch):
    module = _module()

    class BufferedObserver:
        def __init__(self):
            self.pending = ""
            self.delivered = ""
            self.flushes = 0

        def write(self, value):
            self.pending += value
            return len(value)

        def flush(self):
            self.flushes += 1
            self.delivered += self.pending
            self.pending = ""

    observer = BufferedObserver()
    monkeypatch.setattr(sys, "stdout", observer)

    module._github_mask_sink("private%value")

    assert observer.flushes == 1
    assert observer.pending == ""
    assert observer.delivered == "::add-mask::private%25value\n"


def test_secret_drift_during_masking_is_rejected_before_next_child(
    tmp_path,
    monkeypatch,
    capsys,
):
    from scripts import materialize_compose_secrets

    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        materialize_compose_secrets,
        "_verify_hardened_path",
        lambda _path: True,
    )
    changed = False

    def mask(_value):
        nonlocal changed
        if not changed:
            changed = True
            secret = root / ".runtime" / f"ci-{RUN_ID}" / "default_admin_password"
            secret.write_text("UNMASKED-REPLACEMENT-CANARY", encoding="utf-8")

    result = module.main(
        _argv(github_actions_mask=True),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
        mask_sink=mask,
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert result == 1
    assert changed is True
    assert "image-selection" not in stages
    assert "freeze" not in stages
    assert "up" not in stages
    assert "UNMASKED-REPLACEMENT-CANARY" not in capsys.readouterr().err


def test_secret_snapshot_rechecks_acl_after_final_read(
    tmp_path,
    monkeypatch,
):
    from scripts import materialize_compose_secrets

    module = _module()
    secret = tmp_path / "secret"
    secret.write_text("PRIVATE-CANARY", encoding="utf-8")
    checks = iter((True, True, False))
    monkeypatch.setattr(
        materialize_compose_secrets,
        "_verify_hardened_path",
        lambda _path: next(checks),
    )

    with pytest.raises(module.SmokeError, match="^compose smoke failed$"):
        module._stable_hardened_secret(secret)


class _SecretDriftAfterSelection(FakeRun):
    def __call__(self, arguments, **kwargs):
        completed = super().__call__(arguments, **kwargs)
        if self._stage(list(arguments)) == "image-selection":
            secret = (
                self.root
                / ".runtime"
                / f"ci-{RUN_ID}"
                / "default_admin_password"
            )
            secret.write_text("UNMASKED-AFTER-SELECTION-CANARY", encoding="utf-8")
        return completed


def test_secret_drift_after_selection_is_rejected_before_up(
    tmp_path,
    monkeypatch,
    capsys,
):
    from scripts import materialize_compose_secrets

    module = _module()
    root = _root(tmp_path)
    fake = _SecretDriftAfterSelection(root)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        materialize_compose_secrets,
        "_verify_hardened_path",
        lambda _path: True,
    )

    result = module.main(
        _argv(github_actions_mask=True),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
        mask_sink=lambda _value: None,
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert result == 1
    assert "freeze" in stages
    assert "up" not in stages
    assert "UNMASKED-AFTER-SELECTION-CANARY" not in capsys.readouterr().err


class _LockDriftAfterSelection(FakeRun):
    def __call__(self, arguments, **kwargs):
        completed = super().__call__(arguments, **kwargs)
        if self._stage(list(arguments)) == "image-selection":
            lock = self.root / "docker" / "images.lock.env"
            payload = lock.read_bytes()
            lock.write_bytes(
                payload.replace(b"python:3.11-slim", b"python:3.11-SLIM", 1)
            )
        return completed


def test_lock_drift_after_selection_is_rejected_before_up(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = _LockDriftAfterSelection(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert result == 1
    assert "freeze" in stages
    assert "up" not in stages
    assert capsys.readouterr().err == "compose smoke failed\n"


class _SecretDriftAfterStage(FakeRun):
    def __init__(self, root, *, stage, name):
        super().__init__(root)
        self.drift_stage = stage
        self.name = name

    def __call__(self, arguments, **kwargs):
        completed = super().__call__(arguments, **kwargs)
        if self._stage(list(arguments)) == self.drift_stage:
            secret = self.root / ".runtime" / f"ci-{RUN_ID}" / self.name
            secret.write_text("UNMASKED-LATE-DRIFT-CANARY", encoding="utf-8")
        return completed


@pytest.mark.parametrize(
    ("drift_stage", "name", "forbidden_stage"),
    (
        ("up", "default_admin_password", "verify"),
        ("verify", "jwt_secret_key", None),
    ),
)
def test_late_secret_drift_is_rejected_before_next_transaction_stage(
    tmp_path,
    monkeypatch,
    capsys,
    drift_stage,
    name,
    forbidden_stage,
):
    from scripts import materialize_compose_secrets

    module = _module()
    root = _root(tmp_path)
    fake = _SecretDriftAfterStage(root, stage=drift_stage, name=name)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        materialize_compose_secrets,
        "_verify_hardened_path",
        lambda _path: True,
    )

    result = module.main(
        _argv(github_actions_mask=True),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
        mask_sink=lambda _value: None,
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert result == 1
    assert drift_stage in stages
    if forbidden_stage is not None:
        assert forbidden_stage not in stages
    else:
        assert "safe-down" in stages
    assert "UNMASKED-LATE-DRIFT-CANARY" not in capsys.readouterr().err


def test_exact_name_collision_is_rejected_before_build(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root, exact_name_collision=True)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert all(stage != "build-api" for stage, _args, _kwargs in fake.calls)
    assert capsys.readouterr().err == "compose smoke failed\n"


@pytest.mark.parametrize(
    "endpoint",
    (
        "unix://relative",
        "unix:///tmp/../docker.sock",
        "unix:///tmp/docker.sock?query",
        "npipe:////./pipe/docker_engine/extra",
        "tcp://127.0.0.1:2375",
    ),
)
def test_invalid_engine_endpoint_is_rejected_before_build(tmp_path, capsys, endpoint):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root, engine_endpoint=endpoint)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert all(stage != "build-api" for stage, _args, _kwargs in fake.calls)
    assert capsys.readouterr().err == "compose smoke failed\n"


def test_declared_engine_must_equal_observed_endpoint(tmp_path, monkeypatch, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root)
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert all(stage != "build-api" for stage, _args, _kwargs in fake.calls)
    assert capsys.readouterr().err == "compose smoke failed\n"


class _BuildSideEffectFailure(FakeRun):
    def __init__(self, root, *, owned):
        super().__init__(root)
        self.owned = owned

    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        if self._stage(arguments) == "build-api":
            self.calls.append(("build-api", arguments, kwargs))
            tag = _argument(arguments, "-t")
            self.built_tags.add(tag)
            if self.owned:
                Path(_argument(arguments, "--iidfile")).write_bytes(
                    API_ID.encode("ascii")
                )
            return subprocess.CompletedProcess(arguments, 1, "", "PRIVATE-CANARY")
        return super().__call__(arguments, **kwargs)


@pytest.mark.parametrize("owned", (True, False))
def test_failed_build_removes_only_iid_proven_tag(tmp_path, capsys, owned):
    module = _module()
    root = _root(tmp_path)
    fake = _BuildSideEffectFailure(root, owned=owned)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    rm_calls = [args for stage, args, _kwargs in fake.calls if stage == "image-rm"]
    assert bool(rm_calls) is owned
    assert capsys.readouterr().err == "compose smoke failed\n"


def test_web_build_failure_stops_before_materializer_and_cleans_api(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root, fail_stage="build-web")

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert result == 1
    assert "materialize-create" not in stages
    assert stages.count("image-rm") == 1
    assert capsys.readouterr().err == "compose smoke failed\n"


class _ImageInspectFailure(FakeRun):
    def __init__(self, root, *, target, wrong_id):
        super().__init__(root)
        self.target = target
        self.wrong_id = wrong_id

    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        if arguments[:3] == ["docker", "image", "inspect"] and self.target in arguments[-1]:
            self.calls.append(("probe", arguments, kwargs))
            if not self.wrong_id:
                return subprocess.CompletedProcess(arguments, 1, "", "PRIVATE-CANARY")
            completed = super().__call__(arguments, **kwargs)
            payload = json.loads(completed.stdout)
            payload["Id"] = "sha256:" + "d" * 64
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps(payload) + "\n", ""
            )
        return super().__call__(arguments, **kwargs)


@pytest.mark.parametrize("target", ("api-smoke", "web-smoke"))
@pytest.mark.parametrize("wrong_id", (False, True), ids=("inspect-error", "wrong-id"))
def test_image_inspect_failure_stops_before_materializer_and_cleans_owned_images(
    tmp_path,
    capsys,
    target,
    wrong_id,
):
    module = _module()
    root = _root(tmp_path)
    fake = _ImageInspectFailure(root, target=target, wrong_id=wrong_id)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    removable_prior_images = 0 if target == "api-smoke" else 1
    assert result == 1
    assert "materialize-create" not in stages
    assert stages.count("image-rm") == removable_prior_images
    assert capsys.readouterr().err == "compose smoke failed\n"


class _MaterializerSideEffectFailure(FakeRun):
    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        if self._stage(arguments) == "materialize-create":
            super().__call__(arguments, **kwargs)
            return subprocess.CompletedProcess(arguments, 1, "", "PRIVATE-CANARY")
        return super().__call__(arguments, **kwargs)


def test_materializer_write_then_failure_still_cleans_bundle(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = _MaterializerSideEffectFailure(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert not (root / ".runtime" / f"ci-{RUN_ID}").exists()
    assert "materialize-cleanup" in [stage for stage, _args, _kwargs in fake.calls]
    assert capsys.readouterr().err == "compose smoke failed\n"


class _MaterializerStagingFailure(FakeRun):
    def __init__(self, root, *, replace_foreign=False):
        super().__init__(root)
        self.replace_foreign = replace_foreign

    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        if self._stage(arguments) == "materialize-create":
            self.calls.append(("materialize-create", arguments, kwargs))
            staging = Path(_argument(arguments, "--staging-dir"))
            (staging / "default_admin_password").write_text(
                "PRIVATE-STAGING-CANARY", encoding="utf-8"
            )
            if self.replace_foreign:
                shutil.rmtree(staging)
                staging.mkdir()
                (staging / "foreign").write_text("FOREIGN-CANARY", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 1, "", "PRIVATE-CANARY")
        return super().__call__(arguments, **kwargs)


def test_materializer_partial_staging_is_owned_cleanup(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = _MaterializerStagingFailure(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert not (root / ".runtime" / f".ci-staging-{RUN_ID}").exists()
    assert "PRIVATE" not in capsys.readouterr().err


def test_materializer_foreign_staging_replacement_is_preserved(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = _MaterializerStagingFailure(root, replace_foreign=True)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    staging = root / ".runtime" / f".ci-staging-{RUN_ID}"
    assert result == 1
    assert (staging / "foreign").read_text(encoding="utf-8") == "FOREIGN-CANARY"
    assert "PRIVATE" not in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="POSIX guard descriptor contract")
def test_staging_cleanup_exception_closes_guard_and_continues_image_cleanup(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    root = _root(tmp_path)
    staging = root / ".runtime" / f".ci-staging-{RUN_ID}"
    fake = FakeRun(root)
    original_open = module.os.open
    guard_descriptors = []

    def record_guard(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if Path(path) == staging:
            guard_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(module.os, "open", record_guard)
    monkeypatch.setattr(
        module,
        "_cleanup_owned_staging",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup-private-canary")
        ),
    )

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert len(guard_descriptors) == 1
    with pytest.raises(OSError) as exc_info:
        os.fstat(guard_descriptors[0])
    assert exc_info.value.errno == errno.EBADF
    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert stages.count("image-rm") == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "compose smoke failed\n"
    assert "cleanup-private-canary" not in captured.err


@pytest.mark.skipif(os.name != "nt", reason="Windows directory handle contract")
def test_staging_cleanup_handle_stays_bound_across_directory_swap(
    tmp_path,
    monkeypatch,
):
    module = _module()
    root = _root(tmp_path)
    staging = root / ".runtime" / f".ci-staging-{RUN_ID}"
    staging.mkdir()
    (staging / "default_admin_password").write_text(
        "OWNED-PRIVATE-CANARY", encoding="utf-8"
    )
    identity = module._record_directory(staging)
    backup = staging.with_name(staging.name + "-backup")
    original_iterdir = Path.iterdir
    swapped = False

    def swap_then_iterdir(path):
        nonlocal swapped
        if path == staging and not swapped:
            swapped = True
            path.rename(backup)
            path.mkdir()
            (path / "default_admin_password").write_text(
                "FOREIGN-CANARY", encoding="utf-8"
            )
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", swap_then_iterdir)

    assert module._cleanup_owned_staging(staging, identity) is False
    assert not backup.exists()
    assert (staging / "default_admin_password").read_text(encoding="utf-8") == (
        "FOREIGN-CANARY"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX dir_fd contract")
def test_posix_staging_cleanup_stays_bound_across_directory_swap(
    tmp_path,
    monkeypatch,
):
    module = _module()
    root = _root(tmp_path)
    staging = root / ".runtime" / f".ci-staging-{RUN_ID}"
    staging.mkdir()
    (staging / "default_admin_password").write_text(
        "OWNED-PRIVATE-CANARY", encoding="utf-8"
    )
    identity = module._record_directory(staging)
    original_listdir = os.listdir
    swapped = False

    def swap_then_listdir(path):
        nonlocal swapped
        if isinstance(path, int) and not swapped:
            swapped = True
            staging.mkdir()
            (staging / "default_admin_password").write_text(
                "FOREIGN-CANARY", encoding="utf-8"
            )
        return original_listdir(path)

    monkeypatch.setattr(os, "listdir", swap_then_listdir)

    assert module._cleanup_owned_staging_posix(staging, identity) is False
    assert (staging / "default_admin_password").read_text(encoding="utf-8") == (
        "FOREIGN-CANARY"
    )
    assert not list(staging.parent.glob(f".{staging.name}.*.cleanup"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX renameat2 contract")
def test_posix_staging_cleanup_does_not_overwrite_quarantine_collision(
    tmp_path,
    monkeypatch,
):
    module = _module()
    root = _root(tmp_path)
    staging = root / ".runtime" / f".ci-staging-{RUN_ID}"
    staging.mkdir()
    secret = staging / "default_admin_password"
    secret.write_text("OWNED-PRIVATE-CANARY", encoding="utf-8")
    identity = module._record_directory(staging)
    monkeypatch.setattr(module.secrets, "token_hex", lambda _size: "a" * 32)
    quarantine = staging.with_name(f".{staging.name}.{'a' * 32}.cleanup")
    quarantine.mkdir()
    (quarantine / "foreign").write_text("FOREIGN-CANARY", encoding="utf-8")

    assert module._cleanup_owned_staging_posix(staging, identity) is False
    assert secret.read_text(encoding="utf-8") == "OWNED-PRIVATE-CANARY"
    assert (quarantine / "foreign").read_text(encoding="utf-8") == (
        "FOREIGN-CANARY"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX renameat2 contract")
def test_posix_staging_cleanup_preserves_mismatched_quarantine_identity(
    tmp_path,
    monkeypatch,
):
    module = _module()
    root = _root(tmp_path)
    staging = root / ".runtime" / f".ci-staging-{RUN_ID}"
    staging.mkdir()
    (staging / "default_admin_password").write_text(
        "OWNED-PRIVATE-CANARY", encoding="utf-8"
    )
    identity = module._record_directory(staging)
    monkeypatch.setattr(module.secrets, "token_hex", lambda _size: "b" * 32)
    original_rename = module._rename_noreplace_posix
    owned_backup = staging.with_name(staging.name + "-owned-backup")

    def replace_after_rename(parent, source_name, target_name):
        result = original_rename(parent, source_name, target_name)
        if result and target_name.endswith(".cleanup"):
            os.rename(target_name, owned_backup.name, src_dir_fd=parent, dst_dir_fd=parent)
            os.mkdir(target_name, dir_fd=parent)
            (staging.parent / target_name / "foreign").write_text(
                "FOREIGN-CANARY", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(module, "_rename_noreplace_posix", replace_after_rename)

    assert module._cleanup_owned_staging_posix(staging, identity) is False
    assert (owned_backup / "default_admin_password").read_text(encoding="utf-8") == (
        "OWNED-PRIVATE-CANARY"
    )
    assert (staging / "foreign").read_text(encoding="utf-8") == "FOREIGN-CANARY"


@pytest.mark.skipif(os.name == "nt", reason="POSIX renameat2 contract")
def test_posix_staging_cleanup_restores_canonical_path_after_stat_failure(
    tmp_path,
    monkeypatch,
):
    module = _module()
    root = _root(tmp_path)
    staging = root / ".runtime" / f".ci-staging-{RUN_ID}"
    staging.mkdir()
    secret = staging / "default_admin_password"
    secret.write_text("OWNED-PRIVATE-CANARY", encoding="utf-8")
    identity = module._record_directory(staging)
    monkeypatch.setattr(module.secrets, "token_hex", lambda _size: "c" * 32)
    original_stat = module.os.stat
    failed = False

    def fail_quarantine_stat(path, *args, **kwargs):
        nonlocal failed
        if not failed and str(path).endswith(".cleanup"):
            failed = True
            raise PermissionError("quarantine stat denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "stat", fail_quarantine_stat)

    assert module._cleanup_owned_staging_posix(staging, identity) is False
    assert secret.read_text(encoding="utf-8") == "OWNED-PRIVATE-CANARY"
    assert not list(staging.parent.glob(f".{staging.name}.*.cleanup"))


class _LockDriftAfterBuild(FakeRun):
    def __call__(self, arguments, **kwargs):
        completed = super().__call__(arguments, **kwargs)
        if self._stage(list(arguments)) == "build-web":
            lock = self.root / "docker" / "images.lock.env"
            payload = lock.read_bytes()
            lock.write_bytes(payload.replace(b"python:3.11-slim", b"python:3.11-SLIM", 1))
        return completed


def test_lock_drift_after_build_is_rejected_before_freeze_or_up(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = _LockDriftAfterBuild(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert "freeze" not in stages
    assert "up" not in stages
    assert capsys.readouterr().err == "compose smoke failed\n"


class _ArtifactSideEffectFailure(FakeRun):
    def __init__(self, root, *, stage, timeout):
        super().__init__(root)
        self.stage = stage
        self.timeout = timeout

    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        stage = self._stage(arguments)
        if stage == self.stage:
            super().__call__(arguments, **kwargs)
            if stage == "image-selection":
                Path(_argument(arguments, "--output")).write_text(
                    "API_IMAGE=" + _argument(arguments, "--api-image") + "\n"
                    "WEB_IMAGE=" + _argument(arguments, "--web-image") + "\n"
                    f"RELEASE_REVISION={REVISION}\n"
                    f"API_IMAGE_ID={API_ID}\n"
                    f"WEB_IMAGE_ID={WEB_ID}\n",
                    encoding="utf-8",
                )
            elif stage == "freeze":
                Path(_argument(arguments, "--output")).write_text(
                    json.dumps(_fake_ci_manifest(self.root), separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
            if self.timeout:
                raise subprocess.TimeoutExpired(arguments, 30)
            return subprocess.CompletedProcess(arguments, 1, "", "PRIVATE-CANARY")
        return super().__call__(arguments, **kwargs)


def _fake_ci_manifest(root):
    return {
        "mode": "ci",
        "project": f"trainfactory-ci-{RUN_ID}",
        "gpu_mode": "cpu",
        "secret_mode": "files",
        "env_files": [
            {
                "role": "secret-paths",
                "path": os.fspath(
                    root / ".runtime" / f"ci-{RUN_ID}" / "compose-secrets.env"
                ),
            },
            {
                "role": "images-lock",
                "path": os.fspath(root / "docker" / "images.lock.env"),
            },
            {
                "role": "image-selection",
                "path": os.fspath(root / ".runtime" / f"ci-release-{RUN_ID}.env"),
            },
        ],
    }


@pytest.mark.parametrize("stage", ("image-selection", "freeze"))
@pytest.mark.parametrize("timeout", (False, True), ids=("nonzero", "timeout"))
def test_published_artifact_then_child_failure_is_identity_cleaned(
    tmp_path,
    monkeypatch,
    capsys,
    stage,
    timeout,
):
    from scripts import compose_manifest

    module = _module()
    root = _root(tmp_path)
    fake = _ArtifactSideEffectFailure(root, stage=stage, timeout=timeout)

    def verified_manifest(path, **_kwargs):
        return _fake_ci_manifest(root)

    monkeypatch.setattr(compose_manifest, "verify_manifest_inputs", verified_manifest)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert not (root / ".runtime" / f"ci-release-{RUN_ID}.env").exists()
    assert not (root / ".runtime" / f"ci-compose-{RUN_ID}-manifest.json").exists()
    stages = [name for name, _args, _kwargs in fake.calls]
    assert "safe-down" not in stages
    assert capsys.readouterr().err == "compose smoke failed\n"


def test_selection_recovery_preserves_replacement_between_verify_and_identity(
    tmp_path,
    monkeypatch,
):
    from scripts import compose_manifest

    module = _module()
    root = _root(tmp_path)
    path = root / ".runtime" / f"ci-release-{RUN_ID}.env"
    expected = {
        "API_IMAGE": f"trainfactory-api-smoke:{RUN_ID}",
        "WEB_IMAGE": f"trainfactory-web-smoke:{RUN_ID}",
        "RELEASE_REVISION": REVISION,
        "API_IMAGE_ID": API_ID,
        "WEB_IMAGE_ID": WEB_ID,
    }
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in expected.items()),
        encoding="utf-8",
    )
    original = compose_manifest._read_stable
    reads = 0

    def replace_after_first_read(candidate, **kwargs):
        nonlocal reads
        result = original(candidate, **kwargs)
        reads += 1
        if reads == 1:
            candidate.unlink()
            candidate.write_text("FOREIGN-CANARY", encoding="utf-8")
        return result

    monkeypatch.setattr(compose_manifest, "_read_stable", replace_after_first_read)

    assert module._recover_selection_identity(
        path, expected=expected, root=root
    ) is None
    assert path.read_text(encoding="utf-8") == "FOREIGN-CANARY"


def test_manifest_recovery_rejects_verify_swap_and_restore(tmp_path, monkeypatch):
    from scripts import compose_manifest

    module = _module()
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / f"ci-release-{RUN_ID}.env"
    secret_env = runtime / f"ci-{RUN_ID}" / "compose-secrets.env"
    lock = root / "docker" / "images.lock.env"
    path = runtime / f"ci-compose-{RUN_ID}-manifest.json"
    expected = {
        "mode": "ci",
        "project": f"trainfactory-ci-{RUN_ID}",
        "gpu_mode": "cpu",
        "secret_mode": "files",
        "env_files": [
            {"role": "secret-paths", "path": os.fspath(secret_env)},
            {"role": "images-lock", "path": os.fspath(lock)},
            {"role": "image-selection", "path": os.fspath(selection)},
        ],
    }
    original_payload = json.dumps(expected, separators=(",", ":")) + "\n"
    path.write_text(original_payload, encoding="utf-8")

    def swap_and_restore(_path, **_kwargs):
        changed = {**expected, "foreign": True}
        path.write_text(json.dumps(changed), encoding="utf-8")
        path.write_text(original_payload, encoding="utf-8")
        return changed

    monkeypatch.setattr(compose_manifest, "verify_manifest_inputs", swap_and_restore)

    assert module._recover_manifest_identity(
        path,
        project=f"trainfactory-ci-{RUN_ID}",
        secret_env=secret_env,
        lock_path=lock,
        selection=selection,
        root=root,
    ) is None
    assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_stale_project_resources_are_rejected_before_any_build(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root, stale_resource=True)

    result = module.main(
        _argv(),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "compose smoke failed\n"
    assert all(stage != "build-api" for stage, _arguments, _kwargs in fake.calls)


@pytest.mark.parametrize("failure", ("freeze", "up", "verify"))
def test_compose_smoke_failure_still_attempts_all_cleanup(
    tmp_path,
    capsys,
    failure,
):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root, fail_stage=failure)

    result = module.main(
        _argv(),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "compose smoke failed\n"
    assert "PRIVATE" not in captured.err
    stages = [stage for stage, _arguments, _kwargs in fake.calls]
    if failure in {"up", "verify"}:
        assert "safe-down" in stages
    assert "materialize-cleanup" in stages
    assert stages.count("image-rm") == 2


def test_cleanup_failure_makes_successful_smoke_nonzero_and_continues_cleanup(
    tmp_path,
    capsys,
):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root, fail_stage="safe-down")

    result = module.main(
        _argv(),
        root=root,
        run=fake,
        token_hex=lambda _size: RUN_ID,
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.err == "compose smoke failed\n"
    stages = [stage for stage, _arguments, _kwargs in fake.calls]
    assert "materialize-cleanup" in stages
    assert stages.count("image-rm") == 2


@pytest.mark.parametrize("failure", ("materialize-cleanup", "image-rm"))
def test_late_cleanup_failure_is_private_and_other_cleanup_continues(
    tmp_path,
    capsys,
    failure,
):
    module = _module()
    root = _root(tmp_path)
    fake = FakeRun(root, fail_stage=failure)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert result == 1
    assert "materialize-cleanup" in stages
    assert stages.count("image-rm") == 2
    assert not (root / ".runtime" / f"ci-release-{RUN_ID}.env").exists()
    assert not (root / ".runtime" / f"ci-compose-{RUN_ID}-manifest.json").exists()
    assert capsys.readouterr().err == "compose smoke failed\n"


class _NoOpMaterializerCleanup(FakeRun):
    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        if self._stage(arguments) == "materialize-cleanup":
            self.calls.append(("materialize-cleanup", arguments, kwargs))
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return super().__call__(arguments, **kwargs)


def test_materializer_cleanup_success_requires_bundle_absent(tmp_path, capsys):
    module = _module()
    root = _root(tmp_path)
    fake = _NoOpMaterializerCleanup(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    assert result == 1
    assert (root / ".runtime" / f"ci-{RUN_ID}").is_dir()
    assert stages.count("image-rm") == 2
    assert capsys.readouterr().err == "compose smoke failed\n"


class _ResidualAfterFailedDown(FakeRun):
    def __init__(self, root):
        super().__init__(root, fail_stage="safe-down")
        self.down_attempted = False

    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        if self._stage(arguments) == "safe-down":
            self.down_attempted = True
        if (
            self.down_attempted
            and arguments[:3] == ["docker", "ps", "-aq"]
            and any("label=com.docker.compose.project=" in item for item in arguments)
        ):
            self.calls.append(("probe", arguments, kwargs))
            return subprocess.CompletedProcess(arguments, 0, "owned-container\n", "")
        return super().__call__(arguments, **kwargs)


def test_failed_down_with_residual_resources_preserves_recovery_artifacts(
    tmp_path,
    capsys,
):
    module = _module()
    root = _root(tmp_path)
    fake = _ResidualAfterFailedDown(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    assert result == 1
    assert (root / ".runtime" / f"ci-{RUN_ID}").is_dir()
    assert (root / ".runtime" / f"ci-release-{RUN_ID}.env").is_file()
    assert (root / ".runtime" / f"ci-compose-{RUN_ID}-manifest.json").is_file()
    assert capsys.readouterr().err == "compose smoke failed\n"


class _UpSideEffectManifestReplacement(FakeRun):
    def __init__(self, root):
        super().__init__(root)
        self.up_attempted = False

    def __call__(self, arguments, **kwargs):
        arguments = list(arguments)
        stage = self._stage(arguments)
        if stage == "up":
            super().__call__(arguments, **kwargs)
            manifest = Path(_argument(arguments, "--manifest"))
            manifest.unlink()
            manifest.write_text("FOREIGN-CANARY", encoding="utf-8")
            self.up_attempted = True
            return subprocess.CompletedProcess(arguments, 1, "", "PRIVATE-CANARY")
        if (
            self.up_attempted
            and arguments[:3] == ["docker", "ps", "-aq"]
            and any("label=com.docker.compose.project=" in item for item in arguments)
        ):
            self.calls.append(("probe", arguments, kwargs))
            return subprocess.CompletedProcess(arguments, 0, "owned-container\n", "")
        return super().__call__(arguments, **kwargs)


def test_up_side_effect_with_replaced_manifest_preserves_recovery_without_down(
    tmp_path,
    capsys,
):
    module = _module()
    root = _root(tmp_path)
    fake = _UpSideEffectManifestReplacement(root)

    result = module.main(
        _argv(), root=root, run=fake, token_hex=lambda _size: RUN_ID
    )

    stages = [stage for stage, _args, _kwargs in fake.calls]
    manifest = root / ".runtime" / f"ci-compose-{RUN_ID}-manifest.json"
    assert result == 1
    assert "safe-down" not in stages
    assert manifest.read_text(encoding="utf-8") == "FOREIGN-CANARY"
    assert (root / ".runtime" / f"ci-release-{RUN_ID}.env").is_file()
    assert (root / ".runtime" / f"ci-{RUN_ID}").is_dir()
    assert capsys.readouterr().err == "compose smoke failed\n"


def test_compose_smoke_cli_is_isolated_and_private(tmp_path):
    canary = "COMPOSE-SMOKE-SITECUSTOMIZE-CANARY"
    (tmp_path / "sitecustomize.py").write_text(f"print({canary!r})\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            os.fspath(ROOT_DIR / "scripts" / "run_compose_smoke.py"),
            "--help",
        ],
        cwd=ROOT_DIR,
        env=environment,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    assert canary not in completed.stdout
    assert canary not in completed.stderr


def test_compose_smoke_process_timeout_kills_owned_descendants(tmp_path):
    module = _module()
    marker = tmp_path / "delayed-side-effect"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib,sys,time\n"
        "time.sleep(0.8)\n"
        "pathlib.Path(sys.argv[1]).write_text('late', encoding='utf-8')\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess,sys,time\n"
        "subprocess.Popen([sys.executable, '-I', sys.argv[1], sys.argv[2]])\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )

    with pytest.raises(module.SmokeError, match="^compose smoke failed$"):
        module._completed(
            module._run_process_tree,
            [sys.executable, "-I", os.fspath(parent), os.fspath(child), os.fspath(marker)],
            root=tmp_path,
            environment={"PATH": os.environ.get("PATH", "")},
            timeout=0.2,
        )

    time.sleep(1.0)
    assert not marker.exists()
