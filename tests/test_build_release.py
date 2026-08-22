import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parents[1]
BUILD_RELEASE = ROOT_DIR / "scripts" / "build_release.py"
IMAGES_LOCK = ROOT_DIR / "docker" / "images.lock.env"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_release_under_test", BUILD_RELEASE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.host_tools
def test_check_accepts_dirty_implementation_tree_without_building_or_writing():
    release_env = ROOT_DIR / ".runtime" / "release.env"
    before = release_env.read_bytes() if release_env.exists() else None

    completed = subprocess.run(
        [sys.executable, str(BUILD_RELEASE), "--check"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert (release_env.read_bytes() if release_env.exists() else None) == before


@pytest.mark.parametrize(
    ("remote", "expected"),
    (
        (
            "git@gitee.com:jiajiwang/train-factory.git",
            "https://gitee.com/jiajiwang/train-factory",
        ),
        (
            "https://github.com/example/train-factory.git/",
            "https://github.com/example/train-factory",
        ),
        (
            "http://example.invalid/org/repo",
            "http://example.invalid/org/repo",
        ),
    ),
)
def test_source_repository_normalization(remote, expected):
    module = _load_module()

    assert module.normalize_source_repository(remote) == expected


@pytest.mark.parametrize(
    "remote",
    (
        "https://user:private-canary@example.invalid/org/repo",
        "https://example.invalid/org/repo?private-canary",
        "https://example.invalid/org/repo#private-canary",
        "file:///private-canary/repo",
        "C:\\private-canary\\repo",
        "\\\\server\\private-canary\\repo",
        "/private-canary/repo",
        "git@example.invalid:org\\private-canary",
        "git@example.invalid:/private-canary",
        "git@example.invalid:private-canary",
        "https://example.invalid/org/private-canary\nignored",
        "https://example.invalid/org/../private-canary",
        "git@example.invalid:org/../private-canary",
        "https://a..b/org/private-canary",
        "git@a..b:org/private-canary",
        "https://-example.invalid/org/private-canary",
        "https://example-.invalid/org/private-canary",
    ),
)
def test_source_repository_normalization_rejects_unsafe_values_without_echo(remote):
    module = _load_module()

    with pytest.raises(module.ReleaseError) as exc_info:
        module.normalize_source_repository(remote)

    assert str(exc_info.value) == "source repository is invalid"
    assert "private-canary" not in str(exc_info.value)


def _fake_root(tmp_path):
    root = tmp_path / "workspace with spaces"
    (root / "docker").mkdir(parents=True)
    (root / "web").mkdir()
    (root / "scripts").mkdir()
    (root / ".runtime").mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "train-factory"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (root / "docker" / "images.lock.env").write_bytes(IMAGES_LOCK.read_bytes())
    (root / "docker" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / "web" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    return root


def _inspect_payload(image_id, labels, *, os_name="linux", architecture="amd64"):
    return json.dumps(
        [
            {
                "Id": image_id,
                "Os": os_name,
                "Architecture": architecture,
                "Config": {"Labels": labels},
            }
        ]
    )


def test_build_uses_fixed_argv_identity_and_writes_verified_release_env(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    root = _fake_root(tmp_path)
    revision = "0123456789abcdef0123456789abcdef01234567"
    vcs_date = "2026-08-17T06:07:08+08:00"
    source = "https://gitee.com/jiajiwang/train-factory"
    version = "0.1.0"
    tag = f"{version}-{revision[:8]}"
    api_image = f"trainfactory-api:{tag}"
    web_image = f"trainfactory-web:{tag}"
    labels = {
        "org.opencontainers.image.version": version,
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.created": vcs_date,
        "org.opencontainers.image.source": source,
    }
    api_id = "sha256:" + "a" * 64
    web_id = "sha256:" + "b" * 64
    calls = []

    def fake_run(argv, *, cwd):
        calls.append((list(argv), Path(cwd)))
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return revision
        if argv[:3] == ["git", "show", "-s"]:
            return vcs_date
        if argv[:4] == ["git", "remote", "get-url", "origin"]:
            return "git@gitee.com:jiajiwang/train-factory.git"
        if argv[:2] == ["git", "status"]:
            return ""
        if argv[:2] == ["docker", "build"]:
            return ""
        if argv[:3] == ["docker", "image", "inspect"]:
            return _inspect_payload(
                api_id if argv[-1] == api_image else web_id,
                labels,
            )
        raise AssertionError("unexpected command category")

    monkeypatch.setattr(module, "_run_command", fake_run)

    module.build_release(root)

    build_calls = [argv for argv, _ in calls if argv[:2] == ["docker", "build"]]
    assert len(build_calls) == 2
    for argv in build_calls:
        assert argv[argv.index("--platform") + 1] == "linux/amd64"
        assert f"BUILD_VERSION={version}" in argv
        assert f"VCS_REF={revision}" in argv
        assert f"VCS_DATE={vcs_date}" in argv
        assert f"SOURCE_REPOSITORY={source}" in argv
    assert any(
        argument.startswith("API_BASE_IMAGE=") for argument in build_calls[0]
    )
    assert any(
        argument.startswith("WEB_NODE_BUILD_IMAGE=") for argument in build_calls[1]
    )
    assert any(
        argument.startswith("WEB_NGINX_IMAGE=") for argument in build_calls[1]
    )
    assert str(root) in build_calls[0]
    assert str(root / "web") in build_calls[1]

    assert (root / ".runtime" / "release.env").read_text(encoding="utf-8") == (
        f"API_IMAGE={api_image}\n"
        f"WEB_IMAGE={web_image}\n"
        f"RELEASE_REVISION={revision}\n"
        f"API_IMAGE_ID={api_id}\n"
        f"WEB_IMAGE_ID={web_id}\n"
    )


def test_build_rejects_dirty_tree_before_docker_without_echo(tmp_path, monkeypatch):
    module = _load_module()
    root = _fake_root(tmp_path)
    calls = []

    def fake_run(argv, *, cwd):
        calls.append(list(argv))
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return "1" * 40
        if argv[:3] == ["git", "show", "-s"]:
            return "2026-08-17T00:00:00Z"
        if argv[:4] == ["git", "remote", "get-url", "origin"]:
            return "https://example.invalid/org/repo"
        if argv[:2] == ["git", "status"]:
            return "?? private-canary"
        raise AssertionError("docker must not run for a dirty tree")

    monkeypatch.setattr(module, "_run_command", fake_run)

    with pytest.raises(module.ReleaseError) as exc_info:
        module.build_release(root)

    assert str(exc_info.value) == "release build requires a clean repository"
    assert "private-canary" not in str(exc_info.value)
    assert not any(argv[0] == "docker" for argv in calls)
    assert not (root / ".runtime" / "release.env").exists()


@pytest.mark.parametrize(
    "mutation",
    ("multi-object", "wrong-label", "missing-label", "bad-id", "wrong-platform"),
)
def test_build_inspect_failure_preserves_previous_release_env(
    tmp_path,
    monkeypatch,
    mutation,
):
    module = _load_module()
    root = _fake_root(tmp_path)
    output = root / ".runtime" / "release.env"
    output.write_bytes(b"PREVIOUS=unchanged\n")
    revision = "2" * 40
    labels = {
        "org.opencontainers.image.version": "0.1.0",
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.created": "2026-08-17T00:00:00Z",
        "org.opencontainers.image.source": "https://example.invalid/org/repo",
    }
    inspect_count = 0

    def fake_run(argv, *, cwd):
        nonlocal inspect_count
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return revision
        if argv[:3] == ["git", "show", "-s"]:
            return "2026-08-17T00:00:00Z"
        if argv[:4] == ["git", "remote", "get-url", "origin"]:
            return "https://example.invalid/org/repo"
        if argv[:2] == ["git", "status"] or argv[:2] == ["docker", "build"]:
            return ""
        if argv[:3] == ["docker", "image", "inspect"]:
            inspect_count += 1
            payload = json.loads(_inspect_payload("sha256:" + "a" * 64, labels))
            if inspect_count == 2:
                if mutation == "multi-object":
                    payload.append(payload[0])
                elif mutation == "wrong-label":
                    payload[0]["Config"]["Labels"][
                        "org.opencontainers.image.revision"
                    ] = "3" * 40
                elif mutation == "missing-label":
                    del payload[0]["Config"]["Labels"][
                        "org.opencontainers.image.source"
                    ]
                elif mutation == "bad-id":
                    payload[0]["Id"] = "sha256:short"
                else:
                    payload[0]["Architecture"] = "arm64"
            return json.dumps(payload)
        raise AssertionError("unexpected command category")

    monkeypatch.setattr(module, "_run_command", fake_run)

    with pytest.raises(module.ReleaseError) as exc_info:
        module.build_release(root)

    assert str(exc_info.value) == "release image inspection failed"
    assert output.read_bytes() == b"PREVIOUS=unchanged\n"
    assert not list(output.parent.glob(".release.env.*.tmp"))


def test_build_rechecks_clean_identity_before_atomic_publish(tmp_path, monkeypatch):
    module = _load_module()
    root = _fake_root(tmp_path)
    output = root / ".runtime" / "release.env"
    output.write_bytes(b"PREVIOUS=unchanged\n")
    first_revision = "4" * 40
    second_revision = "5" * 40
    revision_reads = 0
    inspect_reads = 0

    def fake_run(argv, *, cwd):
        nonlocal revision_reads, inspect_reads
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            revision_reads += 1
            return first_revision if revision_reads <= 2 else second_revision
        if argv[:3] == ["git", "show", "-s"]:
            return "2026-08-17T00:00:00Z"
        if argv[:4] == ["git", "remote", "get-url", "origin"]:
            return "https://example.invalid/org/repo"
        if argv[:2] == ["git", "status"] or argv[:2] == ["docker", "build"]:
            return ""
        if argv[:3] == ["docker", "image", "inspect"]:
            inspect_reads += 1
            labels = {
                "org.opencontainers.image.version": "0.1.0",
                "org.opencontainers.image.revision": first_revision,
                "org.opencontainers.image.created": "2026-08-17T00:00:00Z",
                "org.opencontainers.image.source": "https://example.invalid/org/repo",
            }
            return _inspect_payload("sha256:" + "a" * 64, labels)
        raise AssertionError("unexpected command category")

    monkeypatch.setattr(module, "_run_command", fake_run)

    with pytest.raises(module.ReleaseError) as exc_info:
        module.build_release(root)

    assert str(exc_info.value) == "repository identity changed during release build"
    assert output.read_bytes() == b"PREVIOUS=unchanged\n"
    assert inspect_reads == 2


def test_atomic_publish_failure_preserves_previous_bytes(tmp_path, monkeypatch):
    module = _load_module()
    output = tmp_path / "release.env"
    output.write_bytes(b"PREVIOUS=unchanged\r\n")

    def fail_replace(source, destination):
        raise OSError("private-canary")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(module.ReleaseError) as exc_info:
        module.write_release_env(output, "API_IMAGE=safe\n")

    assert str(exc_info.value) == "release environment publish failed"
    assert output.read_bytes() == b"PREVIOUS=unchanged\r\n"
    assert not list(tmp_path.glob(".release.env.*.tmp"))


def test_subprocess_environment_drops_git_repository_overrides(monkeypatch):
    module = _load_module()
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
    ):
        monkeypatch.setenv(name, "private-canary")

    environment = module.command_environment()

    assert not any(name.startswith("GIT_CONFIG") for name in environment)
    assert "GIT_DIR" not in environment
    assert "GIT_WORK_TREE" not in environment
    assert "GIT_INDEX_FILE" not in environment
    assert "private-canary" not in environment.values()


def test_cli_unknown_argument_does_not_echo_token():
    completed = subprocess.run(
        [sys.executable, str(BUILD_RELEASE), "--private-canary"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "release arguments are invalid\n"
    assert "private-canary" not in completed.stderr


def test_isolated_ci_build_args_cli_exports_only_locked_build_inputs(tmp_path):
    module = _load_module()
    expected = module.load_images_lock(IMAGES_LOCK)
    (tmp_path / "sitecustomize.py").write_text(
        'print("private-preload-canary")\n', encoding="utf-8"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-I", str(BUILD_RELEASE), "--ci-build-args"],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "private-preload-canary" not in completed.stdout
    assert completed.stdout.splitlines() == [
        f"{name}={expected[name]}"
        for name in (
            "API_BASE_IMAGE",
            "API_TEST_BASE_IMAGE",
            "WEB_NODE_BUILD_IMAGE",
            "WEB_NGINX_IMAGE",
        )
    ]


def test_api_and_web_dockerfiles_require_and_write_release_identity():
    api = (ROOT_DIR / "docker" / "Dockerfile").read_text(encoding="utf-8")
    web = (ROOT_DIR / "web" / "Dockerfile").read_text(encoding="utf-8")

    for dockerfile in (api, web):
        for name in ("BUILD_VERSION", "VCS_REF", "VCS_DATE", "SOURCE_REPOSITORY"):
            assert f"ARG {name}\n" in dockerfile
            assert f'test -n "${{{name}}}"' in dockerfile
        for label in ("version", "revision", "created", "source"):
            assert f"org.opencontainers.image.{label}" in dockerfile
    assert "ARG WEB_NODE_BUILD_IMAGE\n" in web
    assert "ARG WEB_NGINX_IMAGE\n" in web
    assert "ARG WEB_NODE_BUILD_IMAGE=" not in web
    assert "ARG WEB_NGINX_IMAGE=" not in web


def test_inference_runtime_defaults_use_verified_releases():
    expected = (
        "vllm/vllm-openai:v0.11.0@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c",
        "xprobe/xinference:v1.13.0@sha256:b5df50f3d04e5f7290cf0c6765b5a2b06ab29657d721cfaa13389a7c1d666291",
        "lmsysorg/sglang:v0.5.16@sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0",
    )
    deployer = (
        ROOT_DIR / "train_factory" / "deployment" / "docker_deployer.py"
    ).read_text(encoding="utf-8")

    for image in expected:
        assert image in deployer
    assert "nightly-dev-20260127-53992403" not in deployer
    assert "docker.1ms.run/lmsysorg/sglang" not in deployer


def test_operator_guidance_does_not_reference_removed_sglang_release():
    sources = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "README_CN.md",
        ROOT_DIR / "docs" / "production-web-release.md",
    ]

    for path in sources:
        content = path.read_text(encoding="utf-8")
        assert "nightly-dev-20260127-53992403" not in content
        assert "docker.1ms.run/lmsysorg/sglang" not in content
