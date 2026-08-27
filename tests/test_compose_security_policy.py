import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parents[1]
COMPOSE_FILE = ROOT_DIR / "docker" / "docker-compose.yml"
EXAMPLE_ENV_FILE = ROOT_DIR / ".env.example"
CPU_OVERLAY = ROOT_DIR / "docker" / "docker-compose.cpu.yml"
GPU_COMPAT_OVERLAY = ROOT_DIR / "docker" / "docker-compose.gpu-compat.yml"
SECRETS_OVERLAY = ROOT_DIR / "docker" / "docker-compose.secrets.yml"
DEPLOYMENT_OVERLAY = ROOT_DIR / "docker" / "docker-compose.deployment.yml"
RELEASE_OVERLAY = ROOT_DIR / "docker" / "docker-compose.release.yml"
VERIFY_OVERLAY = ROOT_DIR / "docker" / "docker-compose.verify.yml"
VALIDATOR = ROOT_DIR / "scripts" / "validate_compose_config.py"
IMAGES_LOCK = ROOT_DIR / "docker" / "images.lock.env"
EXPECTED_IMAGE_LOCK = {
    "API_BASE_IMAGE": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56",
    "API_TEST_BASE_IMAGE": "python:3.11-slim@sha256:a630a63cdb314e2d138a2fca3e375e319e8568346ffafac5b980f888630ac4f1",
    "WEB_NODE_BUILD_IMAGE": "node:22-slim@sha256:d649c27dae7ba0137b3cef5dd75baa422c08dc3d9e3fc0c23dfb172dc3cc6436",
    "WEB_NGINX_IMAGE": "nginx:alpine@sha256:4a73073bd557c65b759505da037898b61f1be6cbcc3c2c3aeac22d2a470c1752",
    "MYSQL_IMAGE": "mysql:8.0@sha256:7dcddc01f13bab2f15cde676d44d01f61fc9f99fe7785e86196dfc07d358ae2b",
    "NODE_IMAGE": "node:20-alpine@sha256:fb4cd12c85ee03686f6af5362a0b0d56d50c58a04632e6c0fb8363f609372293",
    "PLAYWRIGHT_IMAGE": "mcr.microsoft.com/playwright:v1.57.0-jammy@sha256:6aca677c27a967caf7673d108ac67ffaf8fed134f27e17b27a05464ca0ace831",
    "VLLM_IMAGE": "vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52",
    "SGLANG_IMAGE": "lmsysorg/sglang:v0.5.17@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155",
    "XINFERENCE_IMAGE": "xprobe/xinference:v3.1.0@sha256:ec41459d15cc1c18842370c267e9c9a12a0001245dea9fe3b939e4075dc18178",
    "ETCD_IMAGE": "quay.io/coreos/etcd:v3.5.18@sha256:d0a641d5fbcc89678c931a61b7de7b8a1cf097149f135c9c73bc81d076a1494b",
    "MINIO_IMAGE": "minio/minio:RELEASE.2024-11-07T00-52-20Z@sha256:ac591851803a79aee64bc37f66d77c56b0a4b6e12d9e5356380f4105510f2332",
    "MILVUS_IMAGE": "milvusdb/milvus:v2.5.4@sha256:081e5471cb53756bd86bf2e2bfbedadec9e87cb181583c6f19410b23a711be00",
}
PUBLISHED_SERVICES = {
    "train-factory-web",
    "train-factory-api",
    "train-factory-web-dev",
    "train-factory-api-dev",
    "xinference",
    "minio",
    "milvus",
}


def _lock_text(values=EXPECTED_IMAGE_LOCK):
    return "".join(f"{name}={value}\n" for name, value in values.items())


def _assert_no_privileged_container_mounts(compose):
    serialized = json.dumps(compose["services"]).replace("\\\\", "/").lower()
    assert "/var/run/docker.sock" not in serialized
    assert "/usr/bin/docker" not in serialized
    assert "//./pipe/docker_engine" not in serialized


def test_images_lock_is_complete_and_validator_accepts_digest_pins():
    assert IMAGES_LOCK.exists()
    assert IMAGES_LOCK.read_text(encoding="utf-8") == _lock_text()

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--images-lock",
            str(IMAGES_LOCK),
            "--require-digests",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


@pytest.mark.parametrize(
    "invalid_text",
    (
        "private-canary=value\n",
        _lock_text() + "PRIVATE_CANARY=image:tag@sha256:" + "1" * 64 + "\n",
        _lock_text()
        + next(iter(EXPECTED_IMAGE_LOCK))
        + "=image:tag@sha256:"
        + "1" * 64
        + "\n",
        _lock_text({**EXPECTED_IMAGE_LOCK, "API_BASE_IMAGE": ""}),
        _lock_text({**EXPECTED_IMAGE_LOCK, "API_BASE_IMAGE": "image:tag"}),
        _lock_text(
            {**EXPECTED_IMAGE_LOCK, "API_BASE_IMAGE": "image:tag@sha256:" + "1" * 63}
        ),
        _lock_text(
            {**EXPECTED_IMAGE_LOCK, "API_BASE_IMAGE": "image:tag@sha256:" + "0" * 64}
        ),
        _lock_text(
            {
                **EXPECTED_IMAGE_LOCK,
                "API_BASE_IMAGE": "user:password@example/image:tag@sha256:" + "1" * 64,
            }
        ),
    ),
)
def test_images_lock_validator_rejects_malformed_input_without_echo(
    tmp_path,
    invalid_text,
):
    lock_path = tmp_path / "images.lock.env"
    lock_path.write_text(invalid_text, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--images-lock",
            str(lock_path),
            "--require-digests",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "images lock is invalid\n"
    assert "private-canary" not in completed.stderr


def test_compose_image_defaults_are_digest_locked():
    compose_source = COMPOSE_FILE.read_text(encoding="utf-8")

    for name, image in EXPECTED_IMAGE_LOCK.items():
        if name == "API_TEST_BASE_IMAGE":
            continue
        assert f"${{{name}:-{image}}}" in compose_source
    assert "docker.1ms.run/lmsysorg/sglang" not in compose_source


@pytest.mark.host_tools
def test_example_environment_resolves_all_release_images_to_digest_pins():
    compose = _render_compose(profiles=("dev", "inference", "vector", "test", "pull"))

    assert (
        compose["services"]["train-factory-api"]["build"]["args"]["API_BASE_IMAGE"]
        == EXPECTED_IMAGE_LOCK["API_BASE_IMAGE"]
    )
    assert (
        compose["services"]["train-factory-web"]["build"]["args"][
            "WEB_NODE_BUILD_IMAGE"
        ]
        == EXPECTED_IMAGE_LOCK["WEB_NODE_BUILD_IMAGE"]
    )
    assert (
        compose["services"]["train-factory-web"]["build"]["args"]["WEB_NGINX_IMAGE"]
        == EXPECTED_IMAGE_LOCK["WEB_NGINX_IMAGE"]
    )
    service_images = {
        "mysql": "MYSQL_IMAGE",
        "train-factory-web-dev": "NODE_IMAGE",
        "train-factory-e2e": "PLAYWRIGHT_IMAGE",
        "vllm-pull": "VLLM_IMAGE",
        "sglang-pull": "SGLANG_IMAGE",
        "xinference": "XINFERENCE_IMAGE",
        "etcd": "ETCD_IMAGE",
        "minio": "MINIO_IMAGE",
        "milvus": "MILVUS_IMAGE",
    }
    for service_name, variable in service_images.items():
        assert (
            compose["services"][service_name]["image"] == EXPECTED_IMAGE_LOCK[variable]
        )


@pytest.mark.parametrize(
    ("overlays", "profiles"),
    (
        ((RELEASE_OVERLAY,), ()),
        ((DEPLOYMENT_OVERLAY, RELEASE_OVERLAY), ("deployment",)),
    ),
)
@pytest.mark.host_tools
def test_release_overlay_uses_verified_images_and_clears_privileged_mounts(
    overlays,
    profiles,
):
    api_image = "trainfactory-api:0.1.0-01234567"
    web_image = "trainfactory-web:0.1.0-01234567"
    compose = _render_compose(
        overlays=overlays,
        environment_overrides={"API_IMAGE": api_image, "WEB_IMAGE": web_image},
        profiles=profiles,
    )

    api = compose["services"]["train-factory-api"]
    web = compose["services"]["train-factory-web"]
    assert api.get("build") is None
    assert web.get("build") is None
    assert api["image"] == api_image
    assert web["image"] == web_image
    assert [
        (
            volume["type"],
            Path(volume["source"]).resolve()
            if volume["type"] == "bind"
            else volume["source"],
            volume["target"],
        )
        for volume in api["volumes"]
    ] == [
        ("bind", (ROOT_DIR / "data").resolve(), "/app/data"),
        ("bind", (ROOT_DIR / "models").resolve(), "/app/models"),
        ("bind", (ROOT_DIR / "output").resolve(), "/app/output"),
        ("volume", "train_cache", "/app/cache"),
    ]
    _assert_no_privileged_container_mounts(compose)


@pytest.mark.parametrize("pipe_name", ("docker_engine", "Docker_Engine"))
def test_release_privileged_mount_check_detects_windows_named_pipe(pipe_name):
    compose = {
        "services": {
            "train-factory-api": {
                "volumes": [
                    {
                        "type": "npipe",
                        "source": rf"\\.\pipe\{pipe_name}",
                        "target": rf"\\.\pipe\{pipe_name}",
                    }
                ]
            }
        }
    }

    with pytest.raises(AssertionError):
        _assert_no_privileged_container_mounts(compose)


def test_privileged_deployment_overlay_is_explicit_opt_in_profile():
    import yaml

    deployment = yaml.safe_load(DEPLOYMENT_OVERLAY.read_text(encoding="utf-8"))

    assert deployment["services"]["train-factory-api"]["profiles"] == ["deployment"]


@pytest.mark.host_tools
def test_release_overlay_is_required_last_and_requires_both_images():
    overrides = {
        "API_IMAGE": "trainfactory-api:0.1.0-01234567",
        "WEB_IMAGE": "trainfactory-web:0.1.0-01234567",
    }
    without_release = _render_compose()
    wrong_order = _render_compose(
        overlays=(RELEASE_OVERLAY, DEPLOYMENT_OVERLAY),
        environment_overrides=overrides,
        profiles=("deployment",),
    )
    assert without_release["services"]["train-factory-api"].get("build") is not None
    assert "/var/run/docker.sock" in json.dumps(
        wrong_order["services"]["train-factory-api"]
    )

    for missing_name in ("API_IMAGE", "WEB_IMAGE"):
        environment = _compose_environment(overrides=overrides)
        environment.pop(missing_name)
        command = [
            "docker",
            "compose",
            "--env-file",
            str(EXAMPLE_ENV_FILE),
            "-f",
            str(COMPOSE_FILE),
            "-f",
            str(RELEASE_OVERLAY),
            "config",
            "--format",
            "json",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert completed.returncode != 0
        assert "private-canary" not in completed.stdout + completed.stderr


def _validate_compose_config(compose, *, profile):
    from scripts.validate_compose_config import validate_compose_config

    return validate_compose_config(compose, profile=profile)


def _validate_inspected_environment(inspect, *, service_name, mode):
    from scripts.validate_compose_config import validate_inspected_secret_environment

    return validate_inspected_secret_environment(
        inspect,
        service_name=service_name,
        mode=mode,
    )


def test_example_environment_documents_optional_web_only_lan_bind_without_overriding_fallback():
    lines = EXAMPLE_ENV_FILE.read_text(encoding="utf-8").splitlines()

    assert "# WEB_HOST_BIND_ADDRESS=0.0.0.0" in lines
    assert not any(line.startswith("WEB_HOST_BIND_ADDRESS=") for line in lines)


def _compose_environment(bind="127.0.0.1", overrides=None):
    environment = os.environ.copy()
    example_names = {
        line.split("=", 1)[0]
        for line in EXAMPLE_ENV_FILE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    for name in tuple(environment):
        if (
            name in example_names
            or name.startswith("COMPOSE_")
            or name
            in {
                "API_IMAGE",
                "WEB_IMAGE",
                "GPU_PREFLIGHT_MODE",
                "NVIDIA_DISABLE_REQUIRE",
            }
        ):
            environment.pop(name)
    environment.update(
        {
            "COMPOSE_PROJECT_NAME": "trainfactory-policy-test",
            "DEBUG": "false",
            "HOST_BIND_ADDRESS": bind,
            "PUBLIC_BASE_URL": "http://localhost:3000",
            "AUTH_COOKIE_SECURE": "false",
            "JWT_SECRET_KEY": "test-only-secret-that-is-not-published",
            "DEFAULT_ADMIN_PASSWORD": "test-only-admin-password",
            "MYSQL_ROOT_PASSWORD": "test-root-password-placeholder",
            "MYSQL_APP_PASSWORD": "test-app-password-placeholder",
        }
    )
    if overrides:
        environment.update(overrides)
    return environment


def _render_compose(
    bind="127.0.0.1",
    overlays=(),
    environment_overrides=None,
    profiles=("dev", "inference", "vector"),
):
    command = [
        "docker",
        "compose",
        "--env-file",
        str(EXAMPLE_ENV_FILE),
        "-f",
        str(COMPOSE_FILE),
    ]
    for overlay in overlays:
        command.extend(("-f", str(overlay)))
    for profile in profiles:
        command.extend(("--profile", profile))
    command.extend(("config", "--format", "json"))
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=_compose_environment(bind, environment_overrides),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _run_entrypoint(tmp_path, environment, assertion):
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("POSIX shell is unavailable")
    child_environment = {
        name: os.environ[name]
        for name in (
            "HOME",
            "LANG",
            "LC_ALL",
            "LOCALAPPDATA",
            "PATHEXT",
            "SystemRoot",
            "TEMP",
            "TMP",
            "TMPDIR",
            "USERPROFILE",
            "WINDIR",
        )
        if name in os.environ
    }
    child_environment["PATH"] = (
        os.fspath(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    child_environment["MSYS2_ENV_CONV_EXCL"] = "*"
    child_environment["PYTHONNOUSERSITE"] = "1"
    child_environment["PYTHONPATH"] = os.fspath(ROOT_DIR)
    child_environment.update(environment)
    child_environment["GPU_PREFLIGHT_MODE"] = "off"
    return subprocess.run(
        [
            shell,
            os.fspath(ROOT_DIR / "docker" / "entrypoint.sh"),
            shell,
            "-c",
            assertion,
        ],
        cwd=tmp_path,
        env=child_environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@pytest.mark.parametrize(
    ("direct_name", "file_name"),
    (
        ("MYSQL_URL", "MYSQL_URL_FILE"),
        ("JWT_SECRET_KEY", "JWT_SECRET_KEY_FILE"),
        ("DEFAULT_ADMIN_PASSWORD", "DEFAULT_ADMIN_PASSWORD_FILE"),
    ),
)
def test_entrypoint_unsets_only_empty_alias_paired_with_file_source(
    tmp_path,
    direct_name,
    file_name,
):
    file_path = f"/run/secrets/{file_name.lower()}-canary"
    completed = _run_entrypoint(
        tmp_path,
        {direct_name: "", file_name: file_path},
        f'test "${{{direct_name}+x}}" != x '
        f'&& test "${{{file_name}}}" = "{file_path}"',
    )

    assert completed.returncode == 0


@pytest.mark.parametrize(
    ("direct_name", "file_name"),
    (
        ("MYSQL_URL", "MYSQL_URL_FILE"),
        ("JWT_SECRET_KEY", "JWT_SECRET_KEY_FILE"),
        ("DEFAULT_ADMIN_PASSWORD", "DEFAULT_ADMIN_PASSWORD_FILE"),
    ),
)
def test_entrypoint_preserves_nonempty_alias_conflicts(
    tmp_path,
    direct_name,
    file_name,
):
    completed = _run_entrypoint(
        tmp_path,
        {direct_name: "private-canary", file_name: "/run/secrets/private"},
        "exit 99",
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == "secret source environment is invalid\n"


@pytest.mark.parametrize(
    "file_name",
    ("MYSQL_URL_FILE", "JWT_SECRET_KEY_FILE", "DEFAULT_ADMIN_PASSWORD_FILE"),
)
def test_entrypoint_rejects_empty_file_source_before_gpu(tmp_path, file_name):
    completed = _run_entrypoint(
        tmp_path,
        {file_name: ""},
        "exit 99",
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == "secret source environment is invalid\n"


@pytest.mark.parametrize(
    "direct_name",
    ("MYSQL_URL", "JWT_SECRET_KEY", "DEFAULT_ADMIN_PASSWORD"),
)
def test_entrypoint_preserves_empty_alias_without_file_source(
    tmp_path,
    direct_name,
):
    environment = {direct_name: ""}
    completed = _run_entrypoint(
        tmp_path,
        environment,
        f'test "${{{direct_name}+x}}" = x && test -z "${{{direct_name}}}"',
    )

    assert completed.returncode == 0


@pytest.mark.parametrize(
    ("direct_name", "unrelated_file_name"),
    (
        ("MYSQL_URL", "JWT_SECRET_KEY_FILE"),
        ("JWT_SECRET_KEY", "DEFAULT_ADMIN_PASSWORD_FILE"),
        ("DEFAULT_ADMIN_PASSWORD", "MYSQL_URL_FILE"),
    ),
)
def test_entrypoint_does_not_unset_empty_alias_for_unrelated_file_source(
    tmp_path,
    direct_name,
    unrelated_file_name,
):
    environment = {
        direct_name: "",
        unrelated_file_name: "/run/secrets/private",
    }
    completed = _run_entrypoint(
        tmp_path,
        environment,
        f'test "${{{direct_name}+x}}" = x && test -z "${{{direct_name}}}"',
    )

    assert completed.returncode == 0


def test_entrypoint_unsets_all_empty_aliases_without_changing_file_sources(tmp_path):
    environment = {
        "MYSQL_URL": "",
        "MYSQL_URL_FILE": "/run/secrets/mysql-url-canary",
        "JWT_SECRET_KEY": "",
        "JWT_SECRET_KEY_FILE": "/run/secrets/jwt-canary",
        "DEFAULT_ADMIN_PASSWORD": "",
        "DEFAULT_ADMIN_PASSWORD_FILE": "/run/secrets/admin-canary",
    }

    completed = _run_entrypoint(
        tmp_path,
        environment,
        'test "${MYSQL_URL+x}" != x '
        '&& test "${JWT_SECRET_KEY+x}" != x '
        '&& test "${DEFAULT_ADMIN_PASSWORD+x}" != x '
        '&& test "$MYSQL_URL_FILE" = /run/secrets/mysql-url-canary '
        '&& test "$JWT_SECRET_KEY_FILE" = /run/secrets/jwt-canary '
        '&& test "$DEFAULT_ADMIN_PASSWORD_FILE" = /run/secrets/admin-canary',
    )

    assert completed.returncode == 0


@pytest.mark.parametrize(
    ("conflicting_name", "conflicting_value"),
    tuple(
        (name, value)
        for name in (
            "MYSQL_URL",
            "JWT_SECRET_KEY",
            "DEFAULT_ADMIN_PASSWORD",
        )
        for value in ("   ", "private-conflict-canary")
    ),
)
def test_entrypoint_rejects_one_conflict_across_all_file_pairs_without_exec(
    tmp_path,
    conflicting_name,
    conflicting_value,
):
    marker = tmp_path / "exec-canary"
    environment = {
        "MYSQL_URL": "",
        "MYSQL_URL_FILE": "/run/secrets/mysql_url",
        "JWT_SECRET_KEY": "",
        "JWT_SECRET_KEY_FILE": "/run/secrets/jwt_secret_key",
        "DEFAULT_ADMIN_PASSWORD": "",
        "DEFAULT_ADMIN_PASSWORD_FILE": "/run/secrets/default_admin_password",
    }
    environment[conflicting_name] = conflicting_value

    completed = _run_entrypoint(
        tmp_path,
        environment,
        "printf executed > exec-canary",
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == "secret source environment is invalid\n"
    assert not marker.exists()
    assert conflicting_value not in completed.stdout
    assert conflicting_value not in completed.stderr


def test_entrypoint_harness_ignores_ambient_python_compose_docker_and_secret_values(
    tmp_path,
    monkeypatch,
):
    canary = "ambient-entrypoint-canary"
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "NVIDIA_DISABLE_REQUIRE",
        "MYSQL_URL",
        "MYSQL_APP_USER",
        "MYSQL_APP_PASSWORD",
        "MYSQL_HOST",
        "MYSQL_DATABASE",
        "JWT_SECRET_KEY",
        "DEFAULT_ADMIN_PASSWORD",
        "COMPOSE_FILE",
        "DOCKER_HOST",
    ):
        monkeypatch.setenv(name, canary)

    completed = _run_entrypoint(
        tmp_path,
        {},
        'test "${PYTHONHOME+x}" != x '
        '&& test "${NVIDIA_DISABLE_REQUIRE+x}" != x '
        '&& test "${MYSQL_URL+x}" != x '
        '&& test "${MYSQL_APP_USER+x}" != x '
        '&& test "${JWT_SECRET_KEY+x}" != x '
        '&& test "${COMPOSE_FILE+x}" != x '
        '&& test "${DOCKER_HOST+x}" != x '
        f'&& case "$PYTHONPATH" in *{canary}*) exit 1;; esac',
    )

    assert completed.returncode == 0
    assert canary not in completed.stdout
    assert canary not in completed.stderr


@pytest.mark.host_tools
def test_real_file_secret_compose_environment_passes_entrypoint_boundary(tmp_path):
    secret_paths = {
        name: "/run/secrets/private"
        for name in (
            "MYSQL_ROOT_PASSWORD_SECRET_PATH",
            "MYSQL_APP_PASSWORD_SECRET_PATH",
            "MYSQL_URL_SECRET_PATH",
            "JWT_SECRET_KEY_SECRET_PATH",
            "DEFAULT_ADMIN_PASSWORD_SECRET_PATH",
        )
    }
    secret_paths.update(
        {
            "API_IMAGE": "local/api@sha256:" + "a" * 64,
            "MYSQL_ROOT_PASSWORD": "",
            "MYSQL_APP_PASSWORD": "",
            "MYSQL_PASSWORD": "",
            "MYSQL_URL": "",
            "JWT_SECRET_KEY": "",
            "DEFAULT_ADMIN_PASSWORD": "",
            "WEB_IMAGE": "local/web@sha256:" + "b" * 64,
        }
    )
    compose = _render_compose(
        overlays=(RELEASE_OVERLAY, VERIFY_OVERLAY, CPU_OVERLAY, SECRETS_OVERLAY),
        environment_overrides=secret_paths,
        profiles=(),
    )
    api_environment = compose["services"]["train-factory-api"]["environment"]
    assert "MYSQL_URL" not in api_environment
    assert api_environment["MYSQL_APP_PASSWORD"] == ""
    assert api_environment["JWT_SECRET_KEY"] == ""
    assert api_environment["DEFAULT_ADMIN_PASSWORD"] == ""
    assert {
        name: api_environment[name]
        for name in (
            "MYSQL_URL_FILE",
            "JWT_SECRET_KEY_FILE",
            "DEFAULT_ADMIN_PASSWORD_FILE",
        )
    } == {
        "MYSQL_URL_FILE": "/run/secrets/mysql_url",
        "JWT_SECRET_KEY_FILE": "/run/secrets/jwt_secret_key",
        "DEFAULT_ADMIN_PASSWORD_FILE": "/run/secrets/default_admin_password",
    }

    completed = _run_entrypoint(
        tmp_path,
        api_environment,
        'test "${MYSQL_APP_PASSWORD+x}" = x '
        '&& test -z "$MYSQL_APP_PASSWORD" '
        '&& test "${MYSQL_URL+x}" != x '
        '&& test "${JWT_SECRET_KEY+x}" != x '
        '&& test "${DEFAULT_ADMIN_PASSWORD+x}" != x '
        '&& test "$MYSQL_URL_FILE" = /run/secrets/mysql_url '
        '&& test "$JWT_SECRET_KEY_FILE" = /run/secrets/jwt_secret_key '
        '&& test "$DEFAULT_ADMIN_PASSWORD_FILE" = /run/secrets/default_admin_password',
    )

    assert completed.returncode == 0


@pytest.mark.parametrize("bind", ("127.0.0.1", "::1"))
@pytest.mark.parametrize("overlays", ((), (CPU_OVERLAY,), (GPU_COMPAT_OVERLAY,)))
@pytest.mark.host_tools
def test_real_compose_config_preserves_explicit_host_ip(bind, overlays):
    compose = _render_compose(bind, overlays)

    actual_published = set()
    for service_name, service in compose["services"].items():
        for port in service.get("ports", []):
            actual_published.add(service_name)
            assert port["host_ip"] == bind

    assert actual_published == PUBLISHED_SERVICES


@pytest.mark.host_tools
def test_real_compose_config_accepts_explicit_web_only_lan_bind():
    compose = _render_compose(
        "127.0.0.1",
        environment_overrides={"WEB_HOST_BIND_ADDRESS": "0.0.0.0"},
    )

    assert compose["services"]["train-factory-web"]["ports"][0]["host_ip"] == "0.0.0.0"
    for service_name, service in compose["services"].items():
        if service_name == "train-factory-web":
            continue
        for port in service.get("ports", []):
            assert port["host_ip"] == "127.0.0.1"

    assert _validate_compose_config(compose, profile="all") == []


@pytest.mark.parametrize("bind", ("127.0.0.1", "::1"))
@pytest.mark.host_tools
def test_cpu_and_gpu_overlays_preserve_resolved_port_contract(bind):
    def port_contract(compose):
        return {
            service_name: [
                (
                    port["target"],
                    port["published"],
                    port["host_ip"],
                    port["protocol"],
                )
                for port in service.get("ports", [])
            ]
            for service_name, service in compose["services"].items()
            if service.get("ports")
        }

    base = port_contract(_render_compose(bind))
    assert port_contract(_render_compose(bind, (CPU_OVERLAY,))) == base
    assert port_contract(_render_compose(bind, (GPU_COMPAT_OVERLAY,))) == base


@pytest.mark.parametrize("bind", ("127.0.0.1", "::1"))
@pytest.mark.host_tools
def test_minimal_real_compose_config_preserves_explicit_host_ip(bind):
    command = [
        "docker",
        "compose",
        "--env-file",
        str(EXAMPLE_ENV_FILE),
        "-f",
        str(COMPOSE_FILE),
        "config",
        "--format",
        "json",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=_compose_environment(bind),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    compose = json.loads(completed.stdout)

    assert set(compose["services"]) == {
        "mysql",
        "train-factory-api",
        "train-factory-web",
    }
    for service in compose["services"].values():
        for port in service.get("ports", []):
            assert port["host_ip"] == bind
    assert _validate_compose_config(compose, profile="minimal") == []


@pytest.mark.host_tools
def test_real_compose_config_passes_api_transport_environment_to_prod_and_dev():
    compose = _render_compose()

    for service_name in ("train-factory-api", "train-factory-api-dev"):
        environment = compose["services"][service_name]["environment"]
        assert environment["HOST_BIND_ADDRESS"] == "127.0.0.1"
        assert environment["PUBLIC_BASE_URL"] == "http://localhost:3000"
        assert environment["AUTH_COOKIE_SECURE"] == "false"

    assert _validate_compose_config(compose, profile="all") == []


@pytest.mark.host_tools
def test_real_compose_config_ignores_host_project_and_secret_environment(monkeypatch):
    canary = "host-environment-canary-must-not-resolve"
    monkeypatch.setenv("JWT_SECRET_KEY", canary)
    monkeypatch.setenv("DEFAULT_ADMIN_PASSWORD", canary)
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", canary)

    compose = _render_compose()

    assert canary not in json.dumps(compose)


def _secret_path_environment(tmp_path):
    environment = {}
    for index, name in enumerate(
        (
            "MYSQL_ROOT_PASSWORD_SECRET_PATH",
            "MYSQL_APP_PASSWORD_SECRET_PATH",
            "MYSQL_URL_SECRET_PATH",
            "JWT_SECRET_KEY_SECRET_PATH",
            "DEFAULT_ADMIN_PASSWORD_SECRET_PATH",
        )
    ):
        secret_file = tmp_path / f"secret-{index}"
        secret_file.write_text(f"private-file-value-{index}", encoding="utf-8")
        environment[name] = str(secret_file.resolve())
    environment["MYSQL_APP_PASSWORD"] = "alias-render-private-canary"
    return environment


@pytest.mark.parametrize("runtime_overlay", (None, CPU_OVERLAY, GPU_COMPAT_OVERLAY))
@pytest.mark.host_tools
def test_real_secret_overlay_removes_direct_values_and_passes_policy(
    tmp_path,
    runtime_overlay,
):
    overlays = (SECRETS_OVERLAY,)
    if runtime_overlay is not None:
        overlays += (runtime_overlay,)
    compose = _render_compose(
        overlays=overlays,
        environment_overrides=_secret_path_environment(tmp_path),
    )

    mysql_environment = compose["services"]["mysql"]["environment"]
    assert "MYSQL_ROOT_PASSWORD" not in mysql_environment
    assert "MYSQL_PASSWORD" not in mysql_environment
    assert mysql_environment["MYSQL_ROOT_PASSWORD_FILE"] == (
        "/run/secrets/mysql_root_password"
    )
    assert mysql_environment["MYSQL_PASSWORD_FILE"] == (
        "/run/secrets/mysql_app_password"
    )
    for service_name in ("train-factory-api", "train-factory-api-dev"):
        environment = compose["services"][service_name]["environment"]
        for name in (
            "MYSQL_URL",
            "MYSQL_APP_PASSWORD",
            "JWT_SECRET_KEY",
            "DEFAULT_ADMIN_PASSWORD",
        ):
            assert name not in environment
        assert environment["MYSQL_URL_FILE"] == "/run/secrets/mysql_url"
        assert environment["JWT_SECRET_KEY_FILE"] == "/run/secrets/jwt_secret_key"
        assert environment["DEFAULT_ADMIN_PASSWORD_FILE"] == (
            "/run/secrets/default_admin_password"
        )
    assert "alias-render-private-canary" not in json.dumps(compose)
    assert _validate_compose_config(compose, profile="all") == []


def test_standard_public_origin_submodule_import_has_no_settings_side_effects(
    tmp_path,
):
    environment = _compose_environment()
    environment.update(
        {
            "TEST_SIDE_EFFECT_ROOT": str(tmp_path),
            "TRAINING_CACHE": str(tmp_path / "cache"),
            "MODELS_DIR": str(tmp_path / "models"),
            "DATASETS_DIR": str(tmp_path / "datasets"),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "LOCAL_CACHE_DIR": str(tmp_path / "local-cache"),
            "JWT_SECRET_KEY": "test-only-secret-that-is-not-published",
        }
    )
    environment.pop("HF_HOME", None)
    code = """
import os
import sys
from pathlib import Path

root = Path(os.environ["TEST_SIDE_EFFECT_ROOT"])
import train_factory.config.public_origin

assert "train_factory.config.settings" not in sys.modules
assert "HF_HOME" not in os.environ
assert not any((root / name).exists() for name in (
    "cache", "models", "datasets", "output", "local-cache"
))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_config_exports_are_lazy_and_settings_singleton_remains_compatible(tmp_path):
    environment = _compose_environment()
    environment.update(
        {
            "TEST_SIDE_EFFECT_ROOT": str(tmp_path),
            "TRAINING_CACHE": str(tmp_path / "cache"),
            "MODELS_DIR": str(tmp_path / "models"),
            "DATASETS_DIR": str(tmp_path / "datasets"),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "LOCAL_CACHE_DIR": str(tmp_path / "local-cache"),
            "JWT_SECRET_KEY": "test-only-secret-that-is-not-published",
        }
    )
    environment.pop("HF_HOME", None)
    code = """
import os
from pathlib import Path

root = Path(os.environ["TEST_SIDE_EFFECT_ROOT"])
import train_factory.config as config

assert config.__all__ == ["Settings", "settings", "get_settings"]
assert {"Settings", "settings", "get_settings"} <= set(dir(config))
assert "HF_HOME" not in os.environ
assert not any((root / name).exists() for name in (
    "cache", "models", "datasets", "output", "local-cache"
))

from train_factory.config import Settings, get_settings
assert Settings.__name__ == "Settings"
assert callable(get_settings)
assert "HF_HOME" not in os.environ
assert not any((root / name).exists() for name in (
    "cache", "models", "datasets", "output", "local-cache"
))

settings_module = __import__(
    "train_factory.config.settings", fromlist=["Settings"]
)
assert settings_module.Settings is Settings
from train_factory.config import settings
assert type(settings) is Settings
assert settings is get_settings()
assert config.settings is settings
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _minimal_config(*, bind="127.0.0.1", url="http://localhost:3000", secure="false"):
    api_environment = {
        "AUTH_ENABLED": "true",
        "HOST_BIND_ADDRESS": bind,
        "PUBLIC_BASE_URL": url,
        "AUTH_COOKIE_SECURE": secure,
        "MYSQL_APP_USER": "trainfactory_app",
        "MYSQL_APP_PASSWORD": "direct-app-password",
        "MYSQL_HOST": "mysql",
        "MYSQL_DATABASE": "train_factory",
        "JWT_SECRET_KEY": "direct-jwt-secret",
        "DEFAULT_ADMIN_PASSWORD": "direct-admin-password",
        "GPU_PREFLIGHT_MODE": "required",
    }
    return {
        "services": {
            "train-factory-api": {
                "environment": api_environment,
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": -1,
                                    "capabilities": ["gpu"],
                                }
                            ]
                        }
                    }
                },
                "ports": [
                    {
                        "target": 18000,
                        "published": "18000",
                        "host_ip": bind,
                        "protocol": "tcp",
                    }
                ],
            },
            "train-factory-web": {
                "ports": [
                    {
                        "target": 80,
                        "published": "3000",
                        "host_ip": bind,
                        "protocol": "tcp",
                    }
                ]
            },
            "mysql": {
                "environment": {
                    "MYSQL_ROOT_PASSWORD": "direct-root-password",
                    "MYSQL_PASSWORD": "direct-app-password",
                    "MYSQL_USER": "trainfactory_app",
                    "MYSQL_DATABASE": "train_factory",
                }
            },
        }
    }


@pytest.mark.host_tools
def test_gpu_preflight_compose_modes_are_literal_and_closed():
    base = _render_compose(
        environment_overrides={
            "GPU_PREFLIGHT_MODE": "off",
            "NVIDIA_DISABLE_REQUIRE": "private-host-canary",
        }
    )
    cpu = _render_compose(
        overlays=(GPU_COMPAT_OVERLAY, CPU_OVERLAY),
        environment_overrides={
            "GPU_PREFLIGHT_MODE": "required",
            "NVIDIA_DISABLE_REQUIRE": "private-host-canary",
        },
    )
    compat = _render_compose(
        overlays=(GPU_COMPAT_OVERLAY,),
        environment_overrides={"GPU_PREFLIGHT_MODE": "off"},
    )

    assert (
        base["services"]["train-factory-api"]["environment"]["GPU_PREFLIGHT_MODE"]
        == "required"
    )
    assert (
        base["services"]["train-factory-api-dev"]["environment"]["GPU_PREFLIGHT_MODE"]
        == "required"
    )
    assert (
        base["services"]["train-factory-api"]["environment"]["NVIDIA_DISABLE_REQUIRE"]
        == "0"
    )
    assert (
        base["services"]["train-factory-api-dev"]["environment"][
            "NVIDIA_DISABLE_REQUIRE"
        ]
        == "0"
    )

    cpu_api = cpu["services"]["train-factory-api"]
    assert cpu_api["environment"]["GPU_PREFLIGHT_MODE"] == "off"
    assert cpu_api["environment"]["NVIDIA_DISABLE_REQUIRE"] == ""
    assert (
        cpu_api.get("deploy", {})
        .get("resources", {})
        .get("reservations", {})
        .get("devices", [])
        == []
    )

    compat_api = compat["services"]["train-factory-api"]
    assert compat_api["environment"]["GPU_PREFLIGHT_MODE"] == "required"
    assert compat_api["environment"]["NVIDIA_DISABLE_REQUIRE"] == "1"
    assert compat_api["deploy"]["resources"]["reservations"]["devices"]


@pytest.mark.parametrize(
    ("mode", "override", "devices", "expected_error"),
    (
        (None, None, True, "is missing required environment GPU_PREFLIGHT_MODE"),
        ("warn", None, True, "has invalid environment GPU_PREFLIGHT_MODE"),
        ("private-canary", None, True, "has invalid environment GPU_PREFLIGHT_MODE"),
        ("off", None, True, "violates GPU preflight policy"),
        ("required", None, False, "violates GPU preflight policy"),
        ("off", "1", False, "violates GPU preflight policy"),
        ("off", "0", False, "violates GPU preflight policy"),
        ("required", "1", False, "violates GPU preflight policy"),
        (
            "required",
            "private-canary",
            True,
            "has invalid environment NVIDIA_DISABLE_REQUIRE",
        ),
        ("required", "", True, "violates GPU preflight policy"),
        ("required", 1, True, "has invalid environment NVIDIA_DISABLE_REQUIRE"),
    ),
)
def test_validator_rejects_gpu_preflight_policy_mismatches_without_echo(
    mode,
    override,
    devices,
    expected_error,
):
    compose = _minimal_config()
    api = compose["services"]["train-factory-api"]
    if mode is None:
        api["environment"].pop("GPU_PREFLIGHT_MODE")
    else:
        api["environment"]["GPU_PREFLIGHT_MODE"] = mode
    if override is not None:
        api["environment"]["NVIDIA_DISABLE_REQUIRE"] = override
    if not devices:
        api["deploy"]["resources"]["reservations"]["devices"] = []

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [f"service train-factory-api {expected_error}"]
    assert "private-canary" not in "\n".join(errors)


def test_validator_accepts_base_cpu_and_compat_gpu_preflight_shapes():
    base = _minimal_config()
    base["services"]["train-factory-api"]["environment"]["NVIDIA_DISABLE_REQUIRE"] = "0"
    assert _validate_compose_config(base, profile="minimal") == []

    compat = _minimal_config()
    compat["services"]["train-factory-api"]["environment"]["NVIDIA_DISABLE_REQUIRE"] = (
        "1"
    )
    assert _validate_compose_config(compat, profile="minimal") == []

    cpu = _minimal_config()
    cpu_api = cpu["services"]["train-factory-api"]
    cpu_api["environment"]["GPU_PREFLIGHT_MODE"] = "off"
    cpu_api["environment"]["NVIDIA_DISABLE_REQUIRE"] = ""
    cpu_api["environment"]["NVIDIA_VISIBLE_DEVICES"] = "void"
    cpu_api["environment"]["NVIDIA_DRIVER_CAPABILITIES"] = ""
    cpu_api["deploy"]["resources"]["reservations"]["devices"] = []
    assert _validate_compose_config(cpu, profile="minimal") == []


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("gpus", "all"),
        ("runtime", "nvidia"),
        ("devices", ["/dev/nvidia0:/dev/nvidia0"]),
        ("devices", [{"source": "/dev/nvidiactl", "target": "/dev/nvidiactl"}]),
        ("devices", ["/dev/dxg:/dev/dxg"]),
        ("device_requests", [{"driver": "nvidia", "count": -1}]),
        ("device_cgroup_rules", ["c 195:* rmw"]),
        ("runtime", "private-runtime-canary"),
        ("privileged", True),
    ),
)
def test_validator_rejects_alternate_gpu_activation_surfaces(field_name, field_value):
    compose = _minimal_config()
    api = compose["services"]["train-factory-api"]
    api[field_name] = field_value

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        "service train-factory-api has unsupported GPU activation configuration"
    ]


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("NVIDIA_VISIBLE_DEVICES", "all"),
        ("NVIDIA_VISIBLE_DEVICES", "private-canary"),
        ("NVIDIA_DRIVER_CAPABILITIES", "compute,utility"),
        ("NVIDIA_DRIVER_CAPABILITIES", "private-canary"),
    ),
)
def test_validator_rejects_cpu_mode_with_gpu_environment_activation_without_echo(
    name, value
):
    compose = _minimal_config()
    api = compose["services"]["train-factory-api"]
    api["environment"].update(
        {
            "GPU_PREFLIGHT_MODE": "off",
            "NVIDIA_DISABLE_REQUIRE": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "NVIDIA_DRIVER_CAPABILITIES": "",
            name: value,
        }
    )
    api["deploy"]["resources"]["reservations"]["devices"] = []

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == ["service train-factory-api violates GPU preflight policy"]
    assert "private-canary" not in "\n".join(errors)


@pytest.mark.parametrize(
    "missing_name",
    ("NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"),
)
def test_validator_requires_explicit_cpu_gpu_suppression(missing_name):
    compose = _minimal_config()
    api = compose["services"]["train-factory-api"]
    api["environment"].update(
        {
            "GPU_PREFLIGHT_MODE": "off",
            "NVIDIA_DISABLE_REQUIRE": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "NVIDIA_DRIVER_CAPABILITIES": "",
        }
    )
    api["environment"].pop(missing_name)
    api["deploy"]["resources"]["reservations"]["devices"] = []

    assert _validate_compose_config(compose, profile="minimal") == [
        "service train-factory-api violates GPU preflight policy"
    ]


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("NVIDIA_VISIBLE_DEVICES", "void"),
        ("NVIDIA_VISIBLE_DEVICES", ""),
        ("NVIDIA_DRIVER_CAPABILITIES", ""),
        ("NVIDIA_DRIVER_CAPABILITIES", "compute"),
    ),
)
def test_validator_rejects_required_mode_that_hides_required_gpu(name, value):
    compose = _minimal_config()
    compose["services"]["train-factory-api"]["environment"][name] = value

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == ["service train-factory-api violates GPU preflight policy"]


@pytest.mark.parametrize(
    ("visible", "capabilities"),
    (
        ("all", "compute,utility"),
        ("0", "utility,compute"),
        ("0,1", "compute,video,utility"),
        ("GPU-01234567-89ab-cdef-0123-456789abcdef", "utility,compute"),
    ),
)
def test_validator_accepts_required_gpu_selectors_and_capability_sets(
    visible, capabilities
):
    compose = _minimal_config()
    compose["services"]["train-factory-api"]["environment"].update(
        {
            "NVIDIA_VISIBLE_DEVICES": visible,
            "NVIDIA_DRIVER_CAPABILITIES": capabilities,
        }
    )

    assert _validate_compose_config(compose, profile="minimal") == []


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("GPU_PREFLIGHT_MODE", []),
        ("NVIDIA_DISABLE_REQUIRE", []),
        ("NVIDIA_VISIBLE_DEVICES", []),
        ("NVIDIA_DRIVER_CAPABILITIES", {}),
    ),
)
def test_validator_fails_closed_for_malformed_gpu_environment_types(name, value):
    compose = _minimal_config()
    compose["services"]["train-factory-api"]["environment"][name] = value

    errors = _validate_compose_config(compose, profile="minimal")

    expected_name = name
    if name in {"NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"}:
        assert errors == ["service train-factory-api violates GPU preflight policy"]
    else:
        assert errors == [
            f"service train-factory-api has invalid environment {expected_name}"
        ]


@pytest.mark.host_tools
def test_real_cpu_overlay_validator_rejects_runtime_nvidia_escape(tmp_path):
    escape_overlay = tmp_path / "gpu-escape.yml"
    escape_overlay.write_text(
        "services:\n"
        "  train-factory-api:\n"
        "    runtime: nvidia\n"
        "    environment:\n"
        "      NVIDIA_VISIBLE_DEVICES: private-canary\n",
        encoding="utf-8",
        newline="\n",
    )
    compose = _render_compose(overlays=(CPU_OVERLAY, escape_overlay))

    errors = _validate_compose_config(compose, profile="all")

    assert errors == [
        "service train-factory-api has unsupported GPU activation configuration"
    ]
    assert "private-canary" not in "\n".join(errors)


@pytest.mark.parametrize(
    "devices",
    (
        [{}],
        [{"driver": "tpu", "count": -1, "capabilities": ["gpu"]}],
        [{"driver": "nvidia", "count": 0, "capabilities": ["gpu"]}],
        [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}],
        [{"driver": "nvidia", "count": -1, "capabilities": ["compute"]}],
        "private-canary",
    ),
)
def test_validator_rejects_malformed_gpu_reservations_without_echo(devices):
    compose = _minimal_config()
    compose["services"]["train-factory-api"]["deploy"]["resources"]["reservations"][
        "devices"
    ] = devices

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == ["service train-factory-api has invalid NVIDIA GPU reservation"]
    assert "private-canary" not in "\n".join(errors)


@pytest.mark.host_tools
def test_cpu_then_compat_overlay_is_rejected_but_compat_then_cpu_is_valid():
    invalid = _render_compose(overlays=(CPU_OVERLAY, GPU_COMPAT_OVERLAY))
    valid = _render_compose(overlays=(GPU_COMPAT_OVERLAY, CPU_OVERLAY))

    assert _validate_compose_config(invalid, profile="all") == [
        "service train-factory-api violates GPU preflight policy"
    ]
    assert _validate_compose_config(valid, profile="all") == []


@pytest.mark.host_tools
def test_gpu_compat_overlay_cannot_be_rendered_without_the_base_compose():
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(EXAMPLE_ENV_FILE),
            "-f",
            str(GPU_COMPAT_OVERLAY),
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT_DIR,
        env=_compose_environment(
            overrides={"NVIDIA_DISABLE_REQUIRE": "private-canary"}
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode != 0
    assert "private-canary" not in completed.stdout + completed.stderr


@pytest.mark.parametrize("invalid_bind", (None, "", "0.0.0.0", "::"))
def test_validator_rejects_missing_or_public_host_ip_with_insecure_cookie(
    invalid_bind,
):
    compose = _minimal_config()
    for service in compose["services"].values():
        if service.get("ports"):
            service["ports"][0]["host_ip"] = invalid_bind
    compose["services"]["train-factory-api"]["environment"]["HOST_BIND_ADDRESS"] = (
        invalid_bind or ""
    )

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors
    assert any("host_ip" in error for error in errors)


@pytest.mark.parametrize(
    "invalid_bind",
    ("LOCALHOST", "localhost", "deployment-host.invalid", "::ffff:127.0.0.1"),
)
def test_deployment_validator_requires_literal_nonmapped_bind_address(invalid_bind):
    compose = _minimal_config(
        bind=invalid_bind,
        url="https://deployment.example.com",
        secure="true",
    )

    errors = _validate_compose_config(compose, profile="minimal")

    assert any("HOST_BIND_ADDRESS" in error for error in errors)


def test_validator_requires_api_transport_environment_without_echoing_values():
    private_value = "private-public-origin-value.invalid"
    compose = _minimal_config(url=f"http://{private_value}")
    del compose["services"]["train-factory-api"]["environment"]["AUTH_COOKIE_SECURE"]

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        "service train-factory-api is missing required environment AUTH_COOKIE_SECURE"
    ]
    assert private_value not in "\n".join(errors)


def test_validator_requires_auth_enabled_and_minimal_web_service():
    compose = _minimal_config()
    del compose["services"]["train-factory-api"]["environment"]["AUTH_ENABLED"]
    del compose["services"]["train-factory-web"]

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        "compose profile minimal is missing service train-factory-web",
        "service train-factory-api is missing required environment AUTH_ENABLED",
    ]


@pytest.mark.parametrize(
    ("service_value", "expected_error"),
    (
        ("not-a-service-mapping", "definition is invalid"),
        ({}, "must publish at least one port"),
    ),
)
def test_validator_rejects_invalid_required_web_service(
    service_value,
    expected_error,
):
    compose = _minimal_config()
    compose["services"]["train-factory-web"] = service_value

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [f"service train-factory-web {expected_error}"]


@pytest.mark.parametrize("missing_field", ("target", "published", "protocol"))
def test_validator_requires_complete_published_port_shape(missing_field):
    compose = _minimal_config()
    del compose["services"]["train-factory-web"]["ports"][0][missing_field]

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        f"service train-factory-web published port has invalid {missing_field}"
    ]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("target", "80"),
        ("published", 3000),
        ("host_ip", 123),
        ("protocol", "udp"),
    ),
)
def test_validator_rejects_invalid_published_port_types(field_name, invalid_value):
    compose = _minimal_config()
    compose["services"]["train-factory-web"]["ports"][0][field_name] = invalid_value

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        f"service train-factory-web published port has invalid {field_name}"
    ]


def test_validator_rejects_public_http_even_with_secure_cookie():
    compose = _minimal_config(bind="0.0.0.0", url="http://host:3000", secure="true")

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == ["service train-factory-api violates deployment transport policy"]


def test_validator_rejects_loopback_bind_with_public_http_origin():
    compose = _minimal_config(url="http://public.example:3000")

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == ["service train-factory-api violates deployment transport policy"]


def test_validator_accepts_public_https_with_secure_cookie():
    compose = _minimal_config(
        bind="0.0.0.0",
        url="https://train.example",
        secure="true",
    )

    assert _validate_compose_config(compose, profile="minimal") == []


def test_validator_auth_disabled_skips_cookie_coupling_but_still_parses_url():
    compose = _minimal_config(bind="0.0.0.0", url="http://host:3000", secure="false")
    compose["services"]["train-factory-api"]["environment"]["AUTH_ENABLED"] = "false"
    assert _validate_compose_config(compose, profile="minimal") == []

    compose["services"]["train-factory-api"]["environment"]["PUBLIC_BASE_URL"] = (
        "https://user:password@private-canary.invalid"
    )
    errors = _validate_compose_config(compose, profile="minimal")
    assert errors == [
        "service train-factory-api has invalid environment PUBLIC_BASE_URL"
    ]
    assert "private-canary" not in "\n".join(errors)


def test_validator_cli_reports_names_without_environment_values():
    private_value = "private-query-value"
    compose = _minimal_config(url=f"https://example.test?{private_value}")
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--compose-json",
            "-",
            "--profile",
            "minimal",
        ],
        cwd=ROOT_DIR,
        input=json.dumps(compose),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode != 0
    assert "train-factory-api" in completed.stderr
    assert "PUBLIC_BASE_URL" in completed.stderr
    assert private_value not in completed.stderr
    assert private_value not in completed.stdout


@pytest.mark.parametrize(
    "invalid_url",
    ("http://bad host/private-canary", "http://host\\private-canary"),
)
def test_validator_cli_rejects_invalid_origin_syntax_without_echo(invalid_url):
    compose = _minimal_config(url=invalid_url)
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--compose-json",
            "-",
            "--profile",
            "minimal",
        ],
        cwd=ROOT_DIR,
        input=json.dumps(compose),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 1
    assert "PUBLIC_BASE_URL" in completed.stderr
    assert "private-canary" not in completed.stderr
    assert "private-canary" not in completed.stdout


def test_validator_rejects_port_host_ip_that_disagrees_with_api_bind():
    compose = copy.deepcopy(_minimal_config())
    compose["services"]["train-factory-web"]["ports"][0]["host_ip"] = "::1"

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        "service train-factory-web published port host_ip does not match HOST_BIND_ADDRESS"
    ]


def test_validator_accepts_web_only_unspecified_bind_while_api_stays_loopback():
    compose = copy.deepcopy(_minimal_config())
    compose["services"]["train-factory-web"]["ports"][0]["host_ip"] = "0.0.0.0"

    assert _validate_compose_config(compose, profile="minimal") == []


def test_validator_rejects_web_only_bind_to_a_global_address_for_http_login():
    compose = copy.deepcopy(_minimal_config())
    compose["services"]["train-factory-web"]["ports"][0]["host_ip"] = "8.8.8.8"

    assert _validate_compose_config(compose, profile="minimal") == [
        "service train-factory-web published port host_ip is unsafe for local Web access"
    ]


@pytest.mark.parametrize("invalid_profile", ("", "minmal", "production"))
def test_validator_rejects_unknown_profile_without_inspecting_config(invalid_profile):
    private_value = "invalid-profile-private-canary"
    compose = _minimal_config(url=f"http://{private_value}")

    errors = _validate_compose_config(compose, profile=invalid_profile)

    assert errors == ["compose validation profile is invalid"]
    assert private_value not in "\n".join(errors)


def test_validator_cli_rejects_unknown_profile_before_reading_environment_values():
    private_value = "invalid-profile-cli-private-canary"
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--compose-json",
            "-",
            "--profile",
            "production",
        ],
        cwd=ROOT_DIR,
        input=json.dumps(_minimal_config(url=f"http://{private_value}")),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stderr == "compose validation profile is invalid\n"
    assert "production" not in completed.stderr
    assert private_value not in completed.stderr
    assert private_value not in completed.stdout


def test_validator_rejects_missing_direct_secret_source_without_echoing_values():
    compose = _minimal_config()
    private_value = compose["services"]["mysql"]["environment"].pop(
        "MYSQL_ROOT_PASSWORD"
    )

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        "service mysql is missing required environment MYSQL_ROOT_PASSWORD"
    ]
    assert private_value not in "\n".join(errors)


def _file_secret_config(tmp_path):
    compose = _minimal_config()
    api_environment = compose["services"]["train-factory-api"]["environment"]
    mysql_environment = compose["services"]["mysql"]["environment"]
    for name in ("MYSQL_APP_PASSWORD", "JWT_SECRET_KEY", "DEFAULT_ADMIN_PASSWORD"):
        api_environment.pop(name)
    mysql_environment.pop("MYSQL_ROOT_PASSWORD")
    mysql_environment.pop("MYSQL_PASSWORD")
    api_environment.update(
        {
            "MYSQL_URL_FILE": "/run/secrets/mysql_url",
            "JWT_SECRET_KEY_FILE": "/run/secrets/jwt_secret_key",
            "DEFAULT_ADMIN_PASSWORD_FILE": "/run/secrets/default_admin_password",
        }
    )
    mysql_environment.update(
        {
            "MYSQL_ROOT_PASSWORD_FILE": "/run/secrets/mysql_root_password",
            "MYSQL_PASSWORD_FILE": "/run/secrets/mysql_app_password",
        }
    )
    source_names = (
        "mysql_root_password",
        "mysql_app_password",
        "mysql_url",
        "jwt_secret_key",
        "default_admin_password",
    )
    compose["secrets"] = {}
    for name in source_names:
        path = tmp_path / name
        path.write_text("file-private-value", encoding="utf-8")
        compose["secrets"][name] = {"file": str(path.resolve())}
    compose["services"]["mysql"]["secrets"] = [
        {"source": "mysql_root_password", "target": "mysql_root_password"},
        {"source": "mysql_app_password", "target": "mysql_app_password"},
    ]
    compose["services"]["train-factory-api"]["secrets"] = [
        {"source": "mysql_url", "target": "mysql_url"},
        {"source": "jwt_secret_key", "target": "jwt_secret_key"},
        {
            "source": "default_admin_password",
            "target": "default_admin_password",
        },
    ]
    return compose


def test_validator_accepts_complete_file_secret_contract(tmp_path):
    assert (
        _validate_compose_config(
            _file_secret_config(tmp_path),
            profile="minimal",
        )
        == []
    )


def test_validator_rejects_relative_or_non_regular_secret_host_path(tmp_path):
    compose = _file_secret_config(tmp_path)
    compose["secrets"]["mysql_url"]["file"] = "relative-private-path"

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == ["secret mysql_url has invalid host file"]
    assert "relative-private-path" not in "\n".join(errors)


@pytest.mark.parametrize("directory_name", ("single$literal", "double$$literal"))
def test_validator_decodes_compose_literal_dollars_in_secret_host_path(
    tmp_path,
    directory_name,
):
    compose = _file_secret_config(tmp_path)
    special_directory = tmp_path / directory_name
    special_directory.mkdir()
    secret_file = special_directory / "mysql_url"
    secret_file.write_text("file-private-value", encoding="utf-8")
    compose["secrets"]["mysql_url"]["file"] = str(secret_file.resolve()).replace(
        "$", "$$"
    )

    assert _validate_compose_config(compose, profile="minimal") == []


def test_validator_rejects_unpaired_compose_dollar_in_secret_host_path(tmp_path):
    compose = _file_secret_config(tmp_path)
    special_directory = tmp_path / "unpaired$literal"
    special_directory.mkdir()
    secret_file = special_directory / "mysql_url"
    secret_file.write_text("file-private-value", encoding="utf-8")
    compose["secrets"]["mysql_url"]["file"] = str(secret_file.resolve())

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == ["secret mysql_url has invalid host file"]
    assert "unpaired" not in "\n".join(errors)


def test_validator_rejects_file_mode_with_secret_alias_residue(tmp_path):
    compose = _file_secret_config(tmp_path)
    private_value = "residual-alias-private-canary"
    compose["services"]["train-factory-api"]["environment"]["MYSQL_APP_PASSWORD"] = (
        private_value
    )

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        "service train-factory-api has conflicting environment MYSQL_APP_PASSWORD"
    ]
    assert private_value not in "\n".join(errors)


@pytest.mark.parametrize(
    ("name", "value"),
    tuple(
        (name, value)
        for name in (
            "MYSQL_APP_PASSWORD",
            "JWT_SECRET_KEY",
            "DEFAULT_ADMIN_PASSWORD",
        )
        for value in (None, 0, False, "private-canary")
    ),
)
def test_validator_rejects_nonempty_or_nonstring_file_mode_aliases(
    tmp_path,
    name,
    value,
):
    compose = _file_secret_config(tmp_path)
    compose["services"]["train-factory-api"]["environment"][name] = value

    assert _validate_compose_config(compose, profile="minimal") == [
        f"service train-factory-api has conflicting environment {name}"
    ]


def test_validator_rejects_file_mode_without_service_secret_attachment(tmp_path):
    compose = _file_secret_config(tmp_path)
    compose["services"]["train-factory-api"]["secrets"] = []

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        "service train-factory-api is missing required secret mysql_url",
        "service train-factory-api is missing required secret jwt_secret_key",
        "service train-factory-api is missing required secret default_admin_password",
    ]


@pytest.mark.parametrize(
    ("service_name", "environment_name", "expected_path"),
    (
        ("mysql", "MYSQL_ROOT_PASSWORD_FILE", "/run/secrets/mysql_root_password"),
        ("mysql", "MYSQL_PASSWORD_FILE", "/run/secrets/mysql_app_password"),
        ("train-factory-api", "MYSQL_URL_FILE", "/run/secrets/mysql_url"),
        ("train-factory-api", "JWT_SECRET_KEY_FILE", "/run/secrets/jwt_secret_key"),
        (
            "train-factory-api",
            "DEFAULT_ADMIN_PASSWORD_FILE",
            "/run/secrets/default_admin_password",
        ),
    ),
)
def test_validator_requires_exact_container_secret_file_paths(
    tmp_path,
    service_name,
    environment_name,
    expected_path,
):
    compose = _file_secret_config(tmp_path)
    compose["services"][service_name]["environment"][environment_name] = (
        f"{expected_path}-wrong"
    )

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        f"service {service_name} has invalid environment {environment_name}"
    ]
    assert "-wrong" not in "\n".join(errors)


@pytest.mark.parametrize(
    ("service_name", "source_name"),
    (
        ("mysql", "mysql_root_password"),
        ("mysql", "mysql_app_password"),
        ("train-factory-api", "mysql_url"),
        ("train-factory-api", "jwt_secret_key"),
        ("train-factory-api", "default_admin_password"),
    ),
)
def test_validator_requires_attachment_target_to_match_source(
    tmp_path,
    service_name,
    source_name,
):
    compose = _file_secret_config(tmp_path)
    attachments = compose["services"][service_name]["secrets"]
    attachment = next(item for item in attachments if item["source"] == source_name)
    attachment["target"] = "wrong-private-target"

    errors = _validate_compose_config(compose, profile="minimal")

    assert errors == [
        f"service {service_name} has invalid secret attachment {source_name}"
    ]
    assert "wrong-private-target" not in "\n".join(errors)


def test_validator_accepts_string_secret_attachment_with_default_target(tmp_path):
    compose = _file_secret_config(tmp_path)
    compose["services"]["train-factory-api"]["secrets"] = [
        "mysql_url",
        "jwt_secret_key",
        "default_admin_password",
    ]

    assert _validate_compose_config(compose, profile="minimal") == []


def test_validator_checks_dev_api_exact_file_path_binding(tmp_path):
    compose = _file_secret_config(tmp_path)
    compose["services"]["train-factory-api-dev"] = copy.deepcopy(
        compose["services"]["train-factory-api"]
    )
    compose["services"]["train-factory-web-dev"] = copy.deepcopy(
        compose["services"]["train-factory-web"]
    )
    compose["services"]["train-factory-api-dev"]["environment"]["MYSQL_URL_FILE"] = (
        "/run/secrets/wrong-private-target"
    )

    errors = _validate_compose_config(compose, profile="all")

    assert errors == [
        "service train-factory-api-dev has invalid environment MYSQL_URL_FILE"
    ]


def test_mocked_docker_inspect_rejects_file_mode_alias_residue_without_echo():
    private_value = "inspect-alias-private-canary"
    inspect = [
        {
            "Config": {
                "Env": [
                    "MYSQL_URL_FILE=/run/secrets/mysql_url",
                    "JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret_key",
                    "DEFAULT_ADMIN_PASSWORD_FILE=/run/secrets/default_admin_password",
                    f"MYSQL_APP_PASSWORD={private_value}",
                ]
            }
        }
    ]

    errors = _validate_inspected_environment(
        inspect,
        service_name="train-factory-api",
        mode="files",
    )

    assert errors == [
        "service train-factory-api has conflicting environment MYSQL_APP_PASSWORD"
    ]
    assert private_value not in "\n".join(errors)


def test_api_and_web_healthchecks_use_tools_guaranteed_by_locked_images():
    import yaml

    dockerfile = (ROOT_DIR / "docker" / "Dockerfile").read_text(encoding="utf-8")
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    web_healthcheck = compose["services"]["train-factory-web"]["healthcheck"]["test"]

    assert "urllib.request" in dockerfile
    assert "curl" not in dockerfile
    assert web_healthcheck == [
        "CMD",
        "wget",
        "-q",
        "-T",
        "3",
        "-O",
        "/dev/null",
        "http://127.0.0.1/health",
    ]
    assert "curl" not in web_healthcheck


def test_api_and_web_builds_supply_non_release_identity_sentinels():
    import yaml

    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    expected = {
        "BUILD_VERSION": "${BUILD_VERSION:-0.1.0-dev}",
        "VCS_REF": "${VCS_REF:-0000000000000000000000000000000000000000}",
        "VCS_DATE": "${VCS_DATE:-1970-01-01T00:00:00Z}",
        "SOURCE_REPOSITORY": "${SOURCE_REPOSITORY:-https://example.invalid/train-factory}",
    }

    for service_name in ("train-factory-api", "train-factory-api-dev"):
        build = compose["services"][service_name]["build"]
        assert build["dockerfile"] == "docker/Dockerfile"
        assert {key: build["args"][key] for key in expected} == expected

    web_build = compose["services"]["train-factory-web"]["build"]
    assert web_build["dockerfile"] == "Dockerfile"
    assert {key: web_build["args"][key] for key in expected} == expected


def test_validator_cli_prefers_repository_code_over_pythonpath_shadow(tmp_path):
    shadow_root = tmp_path / "shadow"
    shadow_config = shadow_root / "train_factory" / "config"
    shadow_config.mkdir(parents=True)
    (shadow_root / "train_factory" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (shadow_config / "__init__.py").write_text("", encoding="utf-8")
    (shadow_config / "public_origin.py").write_text(
        """
def is_loopback_bind_address(value, setting_name):
    return True

def is_loopback_public_origin(origin):
    return True

def parse_public_origin(value, setting_name):
    return "http://localhost"
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(shadow_root), str(ROOT_DIR)))
    private_value = "pythonpath-shadow-private-canary"
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--compose-json",
            "-",
            "--profile",
            "minimal",
        ],
        cwd=ROOT_DIR,
        env=environment,
        input=json.dumps(_minimal_config(url=f"http://bad host/{private_value}")),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stderr == (
        "service train-factory-api has invalid environment PUBLIC_BASE_URL\n"
    )
    assert private_value not in completed.stdout
    assert private_value not in completed.stderr


@pytest.mark.parametrize("shadowed_module", ("argparse", "json", "pathlib"))
def test_validator_cli_rejects_pythonpath_standard_library_shadows(
    tmp_path,
    shadowed_module,
):
    shadow_root = tmp_path / "stdlib-shadow"
    shadow_root.mkdir()
    (shadow_root / f"{shadowed_module}.py").write_text(
        "raise SystemExit(73)\n",
        encoding="utf-8",
    )
    bootstrap_root = tmp_path / "bootstrap"
    bootstrap_root.mkdir()
    (bootstrap_root / "sitecustomize.py").write_text(
        "\n".join(
            (
                "import sys",
                f"sys.path.insert(0, {str(shadow_root)!r})",
                f"sys.modules.pop({shadowed_module!r}, None)",
            )
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(bootstrap_root), str(ROOT_DIR)))
    private_value = "stdlib-shadow-private-canary"
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--compose-json",
            "-",
            "--profile",
            "minimal",
        ],
        cwd=ROOT_DIR,
        env=environment,
        input=json.dumps(_minimal_config(url=f"http://bad host/{private_value}")),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stderr == (
        "service train-factory-api has invalid environment PUBLIC_BASE_URL\n"
    )
    assert private_value not in completed.stdout
    assert private_value not in completed.stderr


def test_ci_workflow_has_exact_jobs_pinned_actions_and_tool_versions():
    workflow = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    jobs = set(
        re.findall(
            r"^  ([a-z][a-z0-9-]+):\n    runs-on: ubuntu-24\.04$",
            workflow,
            flags=re.MULTILINE,
        )
    )
    assert jobs == {
        "lock-and-static",
        "backend-full",
        "host-policy",
        "mysql-migrations",
        "frontend-quality",
        "frontend-e2e",
        "compose-smoke",
        "release-input-validation",
    }
    assert "permissions:\n  contents: read\n" in workflow
    assert workflow.count("permissions:\n      contents: read\n") == len(jobs)
    for job in jobs:
        body = workflow.split(f"  {job}:\n", 1)[1]
        assert body.index("permissions:\n      contents: read\n") < body.index(
            "steps:\n"
        )
    allowed_actions = {
        "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
        "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
        "actions/setup-node@395ad3262231945c25e8478fd5baf05154b1d79f",
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9",
        "docker/setup-buildx-action@8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
        "docker/setup-compose-action@2fe291b7677a45ee1269ec56a42604c143505e7e",
    }
    uses = set(re.findall(r"uses: ([^\s#]+)", workflow))
    assert uses == allowed_actions
    assert workflow.count("persist-credentials: false") == workflow.count(
        "actions/checkout@"
    )
    assert 'python-version: "3.11.13"' in workflow
    assert 'node-version: "22.20.0"' in workflow
    assert 'version: "0.11.19"' in workflow
    assert (
        "7035608168e106375b36d0c818d537a889c51a8625fe7f8f7cad5e62b947c368" in workflow
    )
    assert "enable-cache: false" in workflow
    assert 'version: "v2.29.2"' in workflow
    assert 'version: "v0.16.2"' in workflow
    assert "docker compose version --short" in workflow
    assert "docker buildx version" in workflow
    assert "2.29.2" in workflow and "v0.16.2" in workflow
    assert "2.43.0" in workflow
    lock_job = workflow.split("  lock-and-static:\n", 1)[1].split(
        "\n  backend-full:\n", 1
    )[0]
    lock_check = "python -I scripts/check_dependency_locks.py"
    locked_install = (
        "uv pip install --system --require-hashes -r requirements/test-cpu.lock"
    )
    compose_validation = "python -I scripts/validate_compose_config.py"
    assert lock_check in lock_job
    assert locked_install in lock_job
    assert "uv pip install --system --no-deps --no-build-isolation ." in lock_job
    assert compose_validation in lock_job
    assert lock_job.index(lock_check) < lock_job.index(locked_install)
    assert lock_job.index(locked_install) < lock_job.index(compose_validation)


def test_ci_workflow_uses_private_validated_reports_and_controlled_runners_only():
    workflow = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "services:" not in workflow
    backend = workflow.split("  backend-full:\n", 1)[1].split("\n  host-policy:\n", 1)[
        0
    ]
    assert "/var/run/docker.sock" not in backend
    assert "--privileged" not in backend
    assert '-m "not host_tools"' in backend
    assert "--ignore=tests/integration/test_mysql_migrations.py" in backend
    assert "python -I scripts/check_pytest_partitions.py" in workflow
    assert "python -I scripts/run_mysql_migration_tests.py" in workflow
    assert "python -I scripts/run_compose_smoke.py" in workflow
    assert workflow.count("--github-actions-mask") >= 2
    uploads = workflow.split(
        "uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
    )[1:]
    assert uploads
    for upload in uploads:
        step = upload.split("\n      - ", 1)[0]
        assert "retention-days: 1" in step
        assert "validated-" in step
        assert ".xml" in step
        assert ".runtime" not in step
        assert ".env" not in step
    assert "python -I scripts/validate_test_report.py" in workflow
    assert "-e TRAINFACTORY_JUNIT_CANARY=$" not in workflow
    assert '-e "TRAINFACTORY_JUNIT_CANARY=' not in workflow
    assert "ci-smoke.yml" not in workflow


def test_ci_report_jobs_validate_on_failure_and_upload_only_validated_artifacts():
    workflow = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    job_bounds = (
        ("backend-full", "host-policy", "backend-tests", "backend-validator"),
        ("host-policy", "mysql-migrations", "host-tests", "host-validator"),
        (
            "mysql-migrations",
            "frontend-quality",
            "mysql-tests",
            "mysql-validator",
        ),
    )
    for job, next_job, test_id, validator_id in job_bounds:
        body = workflow.split(f"  {job}:\n", 1)[1].split(f"\n  {next_job}:\n", 1)[0]
        assert f"id: {test_id}" in body
        assert "continue-on-error: true" in body
        assert f"id: {validator_id}" in body
        assert "if: always()" in body
        assert '--secret-file "$REPORT_SECRET"' in body
        assert "::add-mask::%s\\n" in body
        assert 'install -m 600 /dev/null "$REPORT_SECRET"' in body
        assert ("if: always() && " f"steps.{validator_id}.outcome == 'success'") in body
        assert f"steps.{test_id}.outcome" in body
        assert f"steps.{validator_id}.outcome" in body
        upload = body.split("uses: actions/upload-artifact@", 1)[1]
        assert "validated-" in upload.split("\n      - ", 1)[0]
        validator = body.split(f"id: {validator_id}", 1)[1].split("\n      - ", 1)[0]
        assert 'REPORT_SECRET="$REPORT_ROOT/' in validator
    assert 'install -m 600 /dev/null "$REPORT_ROOT/backend.xml"' in workflow
    assert 'install -m 600 /dev/null "$REPORT_ROOT/host-policy.xml"' in workflow


def test_ci_build_and_backend_commands_use_isolated_python_and_exact_buildx_gate():
    workflow = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    backend = workflow.split("  backend-full:\n", 1)[1].split("\n  host-policy:\n", 1)[
        0
    ]
    release = workflow.split("  release-input-validation:\n", 1)[1]
    setup_python = "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405"
    assert setup_python in backend
    assert 'python-version: "3.11.13"' in backend
    assert "python -I -m pytest" in backend
    assert "python -m pytest" not in backend
    assert "python -c" not in workflow
    assert "python -I -c" in workflow
    buildx_gate = (
        "test \"$(docker buildx version | awk '{print $2}')\" = " '"$BUILDX_VERSION"'
    )
    for body in (backend, release):
        assert buildx_gate in body
        assert body.index(buildx_gate) < body.index("docker build")
        assert "python -I scripts/build_release.py --ci-build-args" in body
        assert "mapfile -t IMAGE_ARGS" in body
    lock_text = (ROOT_DIR / "docker" / "images.lock.env").read_text(encoding="utf-8")
    locked_digests = [
        line.split("@sha256:", 1)[1]
        for line in lock_text.splitlines()
        if "@sha256:" in line
    ]
    assert locked_digests
    assert not any(digest in workflow for digest in locked_digests)
    assert "API_TEST_BASE_IMAGE:" not in workflow
    assert "$(sed -n" not in release
    for index, name in enumerate(
        (
            "API_BASE_IMAGE",
            "API_TEST_BASE_IMAGE",
            "WEB_NODE_BUILD_IMAGE",
            "WEB_NGINX_IMAGE",
        )
    ):
        assert f'"${{IMAGE_ARGS[{index}]}}" == {name}=*' in backend
        assert f'"${{IMAGE_ARGS[{index}]}}" == {name}=*' in release
