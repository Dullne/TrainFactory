import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE = ROOT_DIR / "docker" / "docker-compose.yml"
RELEASE = ROOT_DIR / "docker" / "docker-compose.release.yml"
VERIFY = ROOT_DIR / "docker" / "docker-compose.verify.yml"
CPU = ROOT_DIR / "docker" / "docker-compose.cpu.yml"
GPU_COMPAT = ROOT_DIR / "docker" / "docker-compose.gpu-compat.yml"
SECRETS = ROOT_DIR / "docker" / "docker-compose.secrets.yml"
EXAMPLE_ENV = ROOT_DIR / ".env.example"
PROJECT = "trainfactory-verify-" + "ab" * 16
MYSQL_HEALTHCHECK = {
    "test": [
        "CMD-SHELL",
        'password_file="$${MYSQL_PASSWORD_FILE:-}"; if [ -n "$$password_file" ]; then\n'
        '  MYSQL_PWD="$$(cat -- "$$password_file")";\n'
        "else\n"
        '  MYSQL_PWD="$${MYSQL_PASSWORD:-}";\n'
        'fi; export MYSQL_PWD; mysql --protocol=TCP -h 127.0.0.1 -u"$${MYSQL_USER}" '
        '"$${MYSQL_DATABASE}" --batch --skip-column-names --execute="SELECT 1" '
        ">/dev/null 2>&1",
    ],
    "timeout": "5s",
    "interval": "10s",
    "retries": 10,
    "start_period": "30s",
}
WEB_HEALTHCHECK = {
    "test": [
        "CMD",
        "wget",
        "-q",
        "-T",
        "3",
        "-O",
        "/dev/null",
        "http://127.0.0.1/health",
    ],
    "timeout": "10s",
    "interval": "30s",
    "retries": 3,
}


def test_manifest_name_accepts_only_exact_web_promotion_prepared_name():
    from scripts import compose_manifest

    revision = "a" * 40
    assert (
        compose_manifest._MANIFEST_NAME.fullmatch(
            f".web-promotion-{revision}-manifest.json"
        )
        is not None
    )
    for rejected in (
        ".manifest.json",
        ".web-promotion-manifest.json",
        f".web-promotion-{revision[:-1]}-manifest.json",
        f".web-promotion-{revision}x-manifest.json",
    ):
        assert compose_manifest._MANIFEST_NAME.fullmatch(rejected) is None


@pytest.fixture(autouse=True)
def _avoid_retesting_platform_acl_implementation(monkeypatch):
    """Task2 owns platform ACL mechanics; these tests exercise manifest policy."""
    from scripts import compose_manifest
    from scripts import compose_release

    monkeypatch.setattr(compose_manifest, "_verify_hardened_path", lambda _path: True)
    monkeypatch.setattr(compose_manifest, "_harden_path", lambda _path: None)
    monkeypatch.setattr(compose_release, "_verify_hardened_path", lambda _path: True)


def _verify_environment():
    environment = os.environ.copy()
    example_names = {
        line.split("=", 1)[0]
        for line in EXAMPLE_ENV.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    for name in tuple(environment):
        if (
            name in example_names
            or name.startswith("COMPOSE_")
            or name.endswith("_FILE")
            or name in {"API_IMAGE", "WEB_IMAGE"}
        ):
            environment.pop(name)
    environment.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": PROJECT,
            "API_IMAGE": f"trainfactory-api@sha256:{'a' * 64}",
            "WEB_IMAGE": f"trainfactory-web@sha256:{'b' * 64}",
            "HOST_BIND_ADDRESS": "127.0.0.1",
        }
    )
    return environment


def _render_verify(*extra_overlays, environment_overrides=None):
    environment = _verify_environment()
    if environment_overrides:
        environment.update(environment_overrides)
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            os.fspath(ROOT_DIR / "docker"),
            "--env-file",
            os.fspath(EXAMPLE_ENV),
            "-p",
            PROJECT,
            "-f",
            os.fspath(COMPOSE),
            "-f",
            os.fspath(RELEASE),
            "-f",
            os.fspath(VERIFY),
            *[
                item
                for overlay in extra_overlays
                for item in ("-f", os.fspath(overlay))
            ],
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, "verify compose render failed"
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def _render_production(env_file=EXAMPLE_ENV, *, gpu_mode="compat"):
    assert gpu_mode in {"raw", "compat"}
    environment = _verify_environment()
    environment.update(
        {
            "COMPOSE_PROJECT_NAME": "trainfactory",
            "MYSQL_ROOT_PASSWORD": "root-test-value",
            "MYSQL_APP_PASSWORD": "app-test-value",
            "JWT_SECRET_KEY": "j" * 32,
            "DEFAULT_ADMIN_PASSWORD": "admin-test-value",
        }
    )
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            os.fspath(ROOT_DIR / "docker"),
            "--env-file",
            os.fspath(env_file),
            "-p",
            "trainfactory",
            "-f",
            os.fspath(COMPOSE),
            "-f",
            os.fspath(RELEASE),
            *(("-f", os.fspath(GPU_COMPAT)) if gpu_mode == "compat" else ()),
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, "production compose render failed"
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def _canonical_verify_environment(service):
    from scripts.compose_release import _API_ENVIRONMENT_DEFAULT_VALUES

    if service == "mysql":
        return {
            "MYSQL_CHARSET": "utf8mb4",
            "MYSQL_COLLATION": "utf8mb4_unicode_ci",
            "MYSQL_DATABASE": "train_factory",
            "MYSQL_PASSWORD": "",
            "MYSQL_ROOT_PASSWORD": "",
            "MYSQL_USER": "trainfactory_app",
        }
    if service == "train-factory-web":
        return {"API_HOST": "train-factory-api", "API_PORT": "18000"}
    if service != "train-factory-api":
        raise AssertionError("unknown canonical service")
    environment = copy.deepcopy(_API_ENVIRONMENT_DEFAULT_VALUES)
    environment.update(
        {
            "COMPOSE_PROJECT_NAME": PROJECT,
            "DOCKER_NETWORK_NAME": f"{PROJECT}_default",
            "MYSQL_APP_PASSWORD": "",
            "JWT_SECRET_KEY": "",
            "DEFAULT_ADMIN_PASSWORD": "",
        }
    )
    return environment


def _production_validation_manifest(*, gpu_mode="compat"):
    return {
        "mode": "production",
        "project": "trainfactory",
        "gpu_mode": gpu_mode,
        "secret_mode": "direct",
        "env_files": [{"role": "base-environment", "path": os.fspath(EXAMPLE_ENV)}],
        "referenced_files": [],
    }


@pytest.mark.host_tools
def test_verify_overlay_isolates_minimal_service_identity_and_network():
    compose = _render_verify()

    assert set(compose["services"]) == {
        "mysql",
        "train-factory-api",
        "train-factory-web",
    }
    for service in compose["services"].values():
        assert "container_name" not in service
        assert service.get("restart") == "no"
        assert service.get("networks") == {"default": None}
    assert "pid" not in compose["services"]["train-factory-api"]
    assert compose["networks"]["default"]["name"] == f"{PROJECT}_default"
    assert not compose["networks"]["default"].get("ipam", {}).get("config")


@pytest.mark.host_tools
def test_verify_overlay_uses_only_project_named_storage_and_canonical_init_bind():
    compose = _render_verify()
    api = compose["services"]["train-factory-api"]
    mysql = compose["services"]["mysql"]

    api_targets = {
        item["target"]: (item["type"], item["source"]) for item in api["volumes"]
    }
    assert api_targets == {
        "/app/data": ("volume", "verify_data"),
        "/app/models": ("volume", "verify_models"),
        "/app/output": ("volume", "verify_output"),
        "/app/cache": ("volume", "train_cache"),
    }
    mysql_targets = {item["target"]: item for item in mysql["volumes"]}
    assert mysql_targets["/var/lib/mysql"] == {
        "type": "volume",
        "source": "mysql_data",
        "target": "/var/lib/mysql",
        "volume": {},
    }
    init_mount = mysql_targets["/docker-entrypoint-initdb.d/init.sql"]
    assert init_mount["type"] == "bind"
    assert (
        Path(init_mount["source"]).resolve()
        == (ROOT_DIR / "docker" / "init.sql").resolve()
    )
    assert init_mount["read_only"] is True
    assert all(
        definition["name"].startswith(PROJECT + "_")
        for definition in compose["volumes"].values()
    )
    assert set(compose["volumes"]) == {
        "mysql_data",
        "train_cache",
        "verify_data",
        "verify_models",
        "verify_output",
    }
    for _kind, source in api_targets.values():
        assert compose["volumes"][source]["name"] == f"{PROJECT}_{source}"
    assert compose["volumes"]["mysql_data"]["name"] == f"{PROJECT}_mysql_data"


@pytest.mark.host_tools
def test_verify_overlay_has_only_random_loopback_ports_and_no_privileged_mounts():
    compose = _render_verify()
    published = set()
    serialized = json.dumps(compose).replace("\\", "/").lower()
    for service in compose["services"].values():
        for port in service.get("ports", ()):
            assert port["host_ip"] == "127.0.0.1"
            published.add(str(port["published"]))
    assert published == {"0"}
    assert not published.intersection({"3000", "18000", "3306"})
    for forbidden in (
        "../data",
        "../models",
        "../output",
        "/var/run/docker.sock",
        "/usr/bin/docker",
        "//./pipe/docker_engine",
        "trainfactory_mysql_data",
        "trainfactory_train_cache",
        "trainfactory_network",
    ):
        assert forbidden not in serialized


@pytest.mark.host_tools
def test_cpu_overlay_changes_only_gpu_policy_after_verify_isolation():
    verify = _render_verify()
    cpu = _render_verify(CPU)

    def storage_and_network(compose):
        return {
            "volumes": compose["volumes"],
            "networks": compose["networks"],
            "service_volumes": {
                name: service.get("volumes", [])
                for name, service in compose["services"].items()
            },
            "service_networks": {
                name: service.get("networks", {})
                for name, service in compose["services"].items()
            },
        }

    assert storage_and_network(cpu) == storage_and_network(verify)
    api = cpu["services"]["train-factory-api"]
    assert "build" not in api
    assert api["environment"]["GPU_PREFLIGHT_MODE"] == "off"
    assert api["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert api["environment"]["NVIDIA_DRIVER_CAPABILITIES"] == ""
    assert (
        not api.get("deploy", {})
        .get("resources", {})
        .get("reservations", {})
        .get("devices")
    )


@pytest.mark.host_tools
def test_verify_render_ignores_ambient_secret_and_compose_canaries(monkeypatch):
    canary = "private-compose-host-canary"
    for name in (
        "JWT_SECRET_KEY",
        "MYSQL_ROOT_PASSWORD",
        "MYSQL_APP_PASSWORD_FILE",
        "COMPOSE_FILE",
        "COMPOSE_FUTURE_CANARY",
        "API_IMAGE_FILE",
    ):
        monkeypatch.setenv(name, canary)

    rendered = json.dumps(_render_verify())

    assert canary not in rendered


@pytest.mark.host_tools
def test_release_policy_accepts_real_canonical_production_render():
    from scripts import compose_release

    compose = _render_production()

    compose_release._validate_resolved_config(
        _production_validation_manifest(),
        compose,
        root=ROOT_DIR,
    )


@pytest.mark.host_tools
def test_release_policy_accepts_canonical_raw_gpu_render():
    from scripts import compose_release

    compose = _render_production(gpu_mode="raw")
    manifest = _production_validation_manifest(gpu_mode="raw")

    compose_release._validate_resolved_config(manifest, compose, root=ROOT_DIR)


def _file_secret_validation_manifest():
    secret_path = os.fspath(ROOT_DIR / "docker" / "init.sql")
    return {
        "mode": "verify",
        "project": PROJECT,
        "gpu_mode": "compat",
        "secret_mode": "files",
        "env_files": [{"role": "base-environment", "path": os.fspath(EXAMPLE_ENV)}],
        "referenced_files": [
            {"role": f"secret:{name}", "path": secret_path}
            for name in (
                "mysql_root_password",
                "mysql_app_password",
                "mysql_url",
                "jwt_secret_key",
                "default_admin_password",
            )
        ],
    }


@pytest.mark.host_tools
def test_release_policy_accepts_real_canonical_file_secret_render():
    from scripts import compose_release

    secret_paths = {
        key: os.fspath(ROOT_DIR / "docker" / "init.sql")
        for key in (
            "MYSQL_ROOT_PASSWORD_SECRET_PATH",
            "MYSQL_APP_PASSWORD_SECRET_PATH",
            "MYSQL_URL_SECRET_PATH",
            "JWT_SECRET_KEY_SECRET_PATH",
            "DEFAULT_ADMIN_PASSWORD_SECRET_PATH",
        )
    }
    secret_paths.update(
        {
            "MYSQL_ROOT_PASSWORD": "",
            "MYSQL_APP_PASSWORD": "",
            "MYSQL_PASSWORD": "",
            "MYSQL_URL": "",
            "JWT_SECRET_KEY": "",
            "DEFAULT_ADMIN_PASSWORD": "",
        }
    )
    compose = _render_verify(
        GPU_COMPAT,
        SECRETS,
        environment_overrides=secret_paths,
    )
    api_environment = compose["services"]["train-factory-api"]["environment"]
    mysql_environment = compose["services"]["mysql"]["environment"]

    assert {
        key: api_environment[key]
        for key in (
            "MYSQL_APP_PASSWORD",
            "JWT_SECRET_KEY",
            "DEFAULT_ADMIN_PASSWORD",
        )
    } == {
        "MYSQL_APP_PASSWORD": "",
        "JWT_SECRET_KEY": "",
        "DEFAULT_ADMIN_PASSWORD": "",
    }
    assert "MYSQL_URL" not in api_environment
    assert "MYSQL_PASSWORD" not in mysql_environment
    assert "MYSQL_ROOT_PASSWORD" not in mysql_environment

    compose_release._validate_resolved_config(
        _file_secret_validation_manifest(),
        compose,
        root=ROOT_DIR,
    )


@pytest.mark.host_tools
def test_release_policy_rejects_file_secret_direct_value_drift():
    from scripts import compose_release

    secret_paths = {
        key: os.fspath(ROOT_DIR / "docker" / "init.sql")
        for key in (
            "MYSQL_ROOT_PASSWORD_SECRET_PATH",
            "MYSQL_APP_PASSWORD_SECRET_PATH",
            "MYSQL_URL_SECRET_PATH",
            "JWT_SECRET_KEY_SECRET_PATH",
            "DEFAULT_ADMIN_PASSWORD_SECRET_PATH",
        )
    }
    secret_paths.update(
        {
            "MYSQL_ROOT_PASSWORD": "",
            "MYSQL_APP_PASSWORD": "",
            "MYSQL_PASSWORD": "",
            "MYSQL_URL": "",
            "JWT_SECRET_KEY": "",
            "DEFAULT_ADMIN_PASSWORD": "",
        }
    )
    compose = _render_verify(
        GPU_COMPAT,
        SECRETS,
        environment_overrides=secret_paths,
    )
    mutations = (
        ("train-factory-api", "MYSQL_APP_PASSWORD", "private-canary"),
        ("train-factory-api", "JWT_SECRET_KEY", "private-canary"),
        ("train-factory-api", "DEFAULT_ADMIN_PASSWORD", "private-canary"),
        ("train-factory-api", "MYSQL_URL", ""),
        ("mysql", "MYSQL_PASSWORD", ""),
        ("mysql", "MYSQL_ROOT_PASSWORD", ""),
    )
    for service, key, value in mutations:
        mutated = copy.deepcopy(compose)
        mutated["services"][service]["environment"][key] = value
        with pytest.raises(compose_release.ReleaseComposeError):
            compose_release._validate_resolved_config(
                _file_secret_validation_manifest(),
                mutated,
                root=ROOT_DIR,
            )


@pytest.mark.host_tools
def test_release_policy_accepts_custom_ports_and_rejects_duplicate_endpoint(tmp_path):
    from scripts import compose_release

    base = tmp_path / "production.env"
    base.write_text(
        "HOST_BIND_ADDRESS=::1\n"
        "PUBLIC_BASE_URL=http://[::1]:3100\n"
        "API_PORT=19000\n"
        "WEB_PORT=3100\n"
        "XINFERENCE_PATCH_VOLUME=/workspace/train-factory/docker/xinference-patches:/opt/trainfactory/xinference-patches:ro\n"
        "XINFERENCE_CONTRACT_VOLUME=/workspace/train-factory/docker/inference-contracts:/opt/trainfactory/inference-contracts:ro\n"
        "SGLANG_TEMPLATE_VOLUME=/workspace/train-factory/docker/sglang-templates:/opt/trainfactory/sglang-templates:ro\n",
        encoding="utf-8",
    )
    compose = _render_production(base)
    api = compose["services"]["train-factory-api"]
    web = compose["services"]["train-factory-web"]
    api["environment"]["HOST_BIND_ADDRESS"] = "::1"
    api["environment"]["PUBLIC_BASE_URL"] = "http://[::1]:3100"
    api["environment"]["API_PORT"] = "19000"
    web["environment"]["API_PORT"] = "19000"
    api["ports"][0].update({"host_ip": "::1", "target": 19000, "published": "19000"})
    web["ports"][0].update({"host_ip": "::1", "published": "3100"})
    base = tmp_path / "production.env"
    base.write_text(
        "HOST_BIND_ADDRESS=::1\n"
        "PUBLIC_BASE_URL=http://[::1]:3100\n"
        "API_PORT=19000\n"
        "WEB_PORT=3100\n",
        encoding="utf-8",
    )
    manifest = {
        **_production_validation_manifest(),
        "env_files": [{"role": "base-environment", "path": os.fspath(base)}],
    }

    compose_release._validate_resolved_config(manifest, compose, root=ROOT_DIR)

    base.write_text(
        "HOST_BIND_ADDRESS=::1\n"
        "PUBLIC_BASE_URL=http://[::1]:3100\n"
        "API_PORT=19000\n"
        "WEB_PORT=19000\n"
        "XINFERENCE_PATCH_VOLUME=/workspace/train-factory/docker/xinference-patches:/opt/trainfactory/xinference-patches:ro\n"
        "XINFERENCE_CONTRACT_VOLUME=/workspace/train-factory/docker/inference-contracts:/opt/trainfactory/inference-contracts:ro\n"
        "SGLANG_TEMPLATE_VOLUME=/workspace/train-factory/docker/sglang-templates:/opt/trainfactory/sglang-templates:ro\n",
        encoding="utf-8",
    )
    web["ports"][0]["published"] = "19000"
    with pytest.raises(compose_release.ReleaseComposeError):
        compose_release._validate_resolved_config(manifest, compose, root=ROOT_DIR)


@pytest.mark.host_tools
def test_release_policy_accepts_frozen_nondefault_runtime_values(tmp_path):
    from scripts import compose_release

    overrides = {
        "ALLOW_MODEL_REMOTE_CODE": "true",
        "SELF_REGISTRATION_ENABLED": "true",
        "HF_ENDPOINT": "https://models.example.invalid",
        "LOG_LEVEL": "WARNING",
        "API_SHM_SIZE": "2g",
        "API_MEM_LIMIT": "1g",
        "API_CPU_LIMIT": "2",
        "API_PIDS_LIMIT": "100",
        "WEB_HOST_BIND_ADDRESS": "0.0.0.0",
        "NVIDIA_VISIBLE_DEVICES": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "XINFERENCE_PATCH_VOLUME": "/srv/trainfactory/xinference-patches:/opt/trainfactory/xinference-patches:ro",
        "XINFERENCE_CONTRACT_VOLUME": "/srv/trainfactory/inference-contracts:/opt/trainfactory/inference-contracts:ro",
        "SGLANG_TEMPLATE_VOLUME": "/srv/trainfactory/sglang-templates:/opt/trainfactory/sglang-templates:ro",
    }
    lines = EXAMPLE_ENV.read_text(encoding="utf-8").splitlines()
    present = set()
    rendered_lines = []
    for line in lines:
        key = line.split("=", 1)[0] if "=" in line and not line.startswith("#") else ""
        if key in overrides:
            rendered_lines.append(f"{key}={overrides[key]}")
            present.add(key)
        else:
            rendered_lines.append(line)
    rendered_lines.extend(
        f"{key}={value}" for key, value in overrides.items() if key not in present
    )
    base = tmp_path / "production.env"
    base.write_text("\n".join(rendered_lines) + "\n", encoding="utf-8")
    compose = _render_production(base)
    manifest = {
        **_production_validation_manifest(),
        "env_files": [{"role": "base-environment", "path": os.fspath(base)}],
    }

    compose_release._validate_resolved_config(manifest, compose, root=ROOT_DIR)


@pytest.mark.host_tools
@pytest.mark.parametrize(
    "variable",
    (
        "XINFERENCE_PATCH_VOLUME",
        "XINFERENCE_CONTRACT_VOLUME",
        "SGLANG_TEMPLATE_VOLUME",
    ),
)
def test_release_policy_rejects_inference_payload_volume_drift(variable):
    from scripts import compose_release

    compose = _render_production()
    compose["services"]["train-factory-api"]["environment"][variable] = (
        "/different/source:/opt/trainfactory/changed:rw"
    )

    with pytest.raises(compose_release.ReleaseComposeError):
        compose_release._validate_resolved_config(
            _production_validation_manifest(),
            compose,
            root=ROOT_DIR,
        )


@pytest.mark.host_tools
def test_release_policy_rejects_web_bind_address_drift(tmp_path):
    from scripts import compose_release

    base = tmp_path / "production.env"
    base.write_text(
        EXAMPLE_ENV.read_text(encoding="utf-8")
        + "\nWEB_HOST_BIND_ADDRESS=0.0.0.0\n",
        encoding="utf-8",
    )
    compose = _render_production(base)
    compose["services"]["train-factory-web"]["ports"][0]["host_ip"] = "127.0.0.1"
    manifest = {
        **_production_validation_manifest(),
        "env_files": [{"role": "base-environment", "path": os.fspath(base)}],
    }

    with pytest.raises(compose_release.ReleaseComposeError):
        compose_release._validate_resolved_config(manifest, compose, root=ROOT_DIR)


def _manifest_module():
    from scripts import compose_manifest

    return compose_manifest


def _fake_manifest_root(tmp_path):
    root = tmp_path / "repo"
    docker = root / "docker"
    runtime = root / ".runtime"
    secret_run = runtime / ("verify-" + "ab" * 16)
    docker.mkdir(parents=True)
    secret_run.mkdir(parents=True)
    for name in (
        "docker-compose.yml",
        "docker-compose.release.yml",
        "docker-compose.verify.yml",
        "docker-compose.gpu-compat.yml",
        "docker-compose.cpu.yml",
        "docker-compose.secrets.yml",
    ):
        (docker / name).write_text(f"# {name}\n", encoding="utf-8")
    (docker / "init.sql").write_text(
        "USE train_factory;\nCREATE TABLE marker (id INT);\n",
        encoding="utf-8",
    )
    image_keys = (
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
    (docker / "images.lock.env").write_text(
        "".join(
            f"{key}=example/{key.lower()}:fixed@sha256:{index:064x}\n"
            for index, key in enumerate(image_keys, start=1)
        ),
        encoding="utf-8",
    )
    secret_names = {
        "MYSQL_ROOT_PASSWORD_SECRET_PATH": "mysql_root_password",
        "MYSQL_APP_PASSWORD_SECRET_PATH": "mysql_app_password",
        "MYSQL_URL_SECRET_PATH": "mysql_url",
        "JWT_SECRET_KEY_SECRET_PATH": "jwt_secret_key",
        "DEFAULT_ADMIN_PASSWORD_SECRET_PATH": "default_admin_password",
    }
    secret_values = {
        "mysql_root_password": "a" * 64,
        "mysql_app_password": "b" * 64,
        "mysql_url": (
            "mysql+pymysql://trainfactory_app:" + "b" * 64 + "@mysql:3306/train_factory"
        ),
        "jwt_secret_key": "c" * 64,
        "default_admin_password": "admin_token_1234567890",
        "admin_username": "admin",
    }
    for filename, value in secret_values.items():
        (secret_run / filename).write_text(value, encoding="utf-8")
    secret_env = secret_run / "compose-secrets.env"
    secret_env.write_text(
        "".join(
            f"{name}='{secret_run / filename}'\n"
            for name, filename in secret_names.items()
        )
        + "MYSQL_APP_USER=trainfactory_app\n"
        + "DEFAULT_ADMIN_USERNAME=admin\n"
        + "API_PORT=18000\n"
        + "WEB_PORT=3000\n",
        encoding="utf-8",
    )
    state = {
        "version": 1,
        "scope": "verify",
        "run_id": "ab" * 16,
        "directory": str(secret_run),
        "env_file": "compose-secrets.env",
        "state_file": "compose-secrets.state.json",
        "members": [
            *secret_names.values(),
            "admin_username",
            "compose-secrets.env",
            "compose-secrets.state.json",
        ],
        "permissions_hardened": True,
        "cleanup_started": False,
    }
    (secret_run / "compose-secrets.state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )
    release_env = runtime / "release.env"
    release_env.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:fixed\n"
        f"RELEASE_REVISION={'a' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )
    return root, secret_env, release_env


def _fake_production_manifest_inputs(root, release_env):
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n"
        "DEBUG=false\n",
        encoding="utf-8",
    )
    direct = root / ".runtime" / "production-direct.env"
    direct.write_text(
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n",
        encoding="utf-8",
    )
    return (
        root / ".env",
        direct,
        root / "docker" / "images.lock.env",
        release_env,
    )


def _minimal_resolved_verify_config(project=PROJECT, *, root=None, gpu_mode="raw"):
    volume_keys = (
        "mysql_data",
        "train_cache",
        "verify_data",
        "verify_models",
        "verify_output",
    )
    api_environment = {
        key: value
        for key, value in _canonical_verify_environment("train-factory-api").items()
        if key
        not in {
            "MYSQL_APP_PASSWORD",
            "JWT_SECRET_KEY",
            "DEFAULT_ADMIN_PASSWORD",
        }
    }
    if root is not None:
        lock_values = dict(
            line.split("=", 1)
            for line in (root / "docker" / "images.lock.env")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and "=" in line
        )
        api_environment.update(
            {
                key: lock_values[f"{key.removesuffix('_IMAGE')}_IMAGE"]
                for key in ("VLLM_IMAGE", "SGLANG_IMAGE", "XINFERENCE_IMAGE")
            }
        )
    api_environment.update(
        {
            "MYSQL_APP_PASSWORD": "",
            "JWT_SECRET_KEY": "",
            "DEFAULT_ADMIN_PASSWORD": "",
            "HOST_BIND_ADDRESS": "127.0.0.1",
            "PUBLIC_BASE_URL": "http://localhost:3000",
            "AUTH_COOKIE_SECURE": "false",
            "AUTH_ENABLED": "true",
            "MYSQL_URL_FILE": "/run/secrets/mysql_url",
            "JWT_SECRET_KEY_FILE": "/run/secrets/jwt_secret_key",
            "DEFAULT_ADMIN_PASSWORD_FILE": "/run/secrets/default_admin_password",
            "GPU_PREFLIGHT_MODE": "off" if gpu_mode == "cpu" else "required",
            "NVIDIA_VISIBLE_DEVICES": "void" if gpu_mode == "cpu" else "all",
            "NVIDIA_DRIVER_CAPABILITIES": (
                "" if gpu_mode == "cpu" else "compute,utility"
            ),
        }
    )
    if gpu_mode == "raw":
        api_environment["NVIDIA_DISABLE_REQUIRE"] = "0"
    elif gpu_mode == "compat":
        api_environment["NVIDIA_DISABLE_REQUIRE"] = "1"
    elif gpu_mode == "cpu":
        api_environment["NVIDIA_DISABLE_REQUIRE"] = ""
    return {
        "name": project,
        "services": {
            "mysql": {
                "command": [
                    "--character-set-server=utf8mb4",
                    "--collation-server=utf8mb4_unicode_ci",
                ],
                "entrypoint": None,
                "image": "example/mysql_image:fixed@sha256:" + f"{5:064x}",
                "restart": "no",
                "networks": {"default": None},
                "healthcheck": copy.deepcopy(MYSQL_HEALTHCHECK),
                "environment": {
                    **{
                        key: value
                        for key, value in _canonical_verify_environment("mysql").items()
                        if key not in {"MYSQL_ROOT_PASSWORD", "MYSQL_PASSWORD"}
                    },
                    "MYSQL_ROOT_PASSWORD_FILE": "/run/secrets/mysql_root_password",
                    "MYSQL_PASSWORD_FILE": "/run/secrets/mysql_app_password",
                },
                "secrets": [
                    {"source": "mysql_root_password", "target": "mysql_root_password"},
                    {"source": "mysql_app_password", "target": "mysql_app_password"},
                ],
                "volumes": [
                    {
                        "type": "volume",
                        "source": "mysql_data",
                        "target": "/var/lib/mysql",
                        "volume": {},
                    },
                    {
                        "type": "bind",
                        "source": os.fspath(root / "docker" / "init.sql")
                        if root
                        else "docker/init.sql",
                        "target": "/docker-entrypoint-initdb.d/init.sql",
                        "read_only": True,
                        "bind": {"create_host_path": True},
                    },
                ],
            },
            "train-factory-api": {
                "command": None,
                "entrypoint": None,
                "depends_on": {
                    "mysql": {"condition": "service_healthy", "required": True}
                },
                "image": "local/api:fixed",
                "restart": "no",
                "networks": {"default": None},
                "shm_size": "1073741824",
                "environment": api_environment,
                "deploy": (
                    {
                        "placement": {},
                        "resources": {
                            "limits": {},
                            "reservations": (
                                {}
                                if gpu_mode == "cpu"
                                else {
                                    "devices": [
                                        {
                                            "driver": "nvidia",
                                            "count": -1,
                                            "capabilities": ["gpu"],
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ),
                "secrets": [
                    {"source": "mysql_url", "target": "mysql_url"},
                    {"source": "jwt_secret_key", "target": "jwt_secret_key"},
                    {
                        "source": "default_admin_password",
                        "target": "default_admin_password",
                    },
                ],
                "volumes": [
                    {
                        "type": "volume",
                        "source": "verify_data",
                        "target": "/app/data",
                        "volume": {},
                    },
                    {
                        "type": "volume",
                        "source": "verify_models",
                        "target": "/app/models",
                        "volume": {},
                    },
                    {
                        "type": "volume",
                        "source": "verify_output",
                        "target": "/app/output",
                        "volume": {},
                    },
                    {
                        "type": "volume",
                        "source": "train_cache",
                        "target": "/app/cache",
                        "volume": {},
                    },
                ],
                "ports": [
                    {
                        "host_ip": "127.0.0.1",
                        "published": "0",
                        "target": 18000,
                        "protocol": "tcp",
                        "mode": "ingress",
                    }
                ],
            },
            "train-factory-web": {
                "command": None,
                "entrypoint": None,
                "depends_on": {
                    "train-factory-api": {
                        "condition": "service_started",
                        "required": True,
                    }
                },
                "image": "local/web:fixed",
                "restart": "no",
                "networks": {"default": None},
                "environment": {"API_HOST": "train-factory-api", "API_PORT": "18000"},
                "healthcheck": copy.deepcopy(WEB_HEALTHCHECK),
                "ports": [
                    {
                        "host_ip": "127.0.0.1",
                        "published": "0",
                        "target": 80,
                        "protocol": "tcp",
                        "mode": "ingress",
                    }
                ],
            },
        },
        "volumes": {key: {"name": f"{project}_{key}"} for key in volume_keys},
        "networks": {"default": {"name": f"{project}_default"}},
        "secrets": {
            name: {
                "file": os.fspath(root / ".runtime" / ("verify-" + "ab" * 16) / name)
                if root
                else os.fspath(ROOT_DIR / "docker" / "init.sql")
            }
            for name in (
                "mysql_root_password",
                "mysql_app_password",
                "mysql_url",
                "jwt_secret_key",
                "default_admin_password",
            )
        },
    }


def test_release_policy_rejects_secret_path_parent_alias(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest = manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    compose = _minimal_resolved_verify_config(root=root)
    secret_file = Path(compose["secrets"]["mysql_url"]["file"])
    compose["secrets"]["mysql_url"]["file"] = os.fspath(
        secret_file.parent / "missing" / ".." / secret_file.name
    )

    with pytest.raises(release_module.ReleaseComposeError):
        release_module._validate_resolved_config(manifest, compose, root=root)


def test_manifest_freeze_records_closed_order_roles_and_full_hashes(tmp_path, capsys):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    output = root / ".runtime" / "verify-compose-manifest.json"
    project = "trainfactory-verify-" + "ab" * 16

    manifest = module.freeze_manifest(
        mode="verify",
        project=project,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="compat",
        secret_mode="files",
        output=output,
        root=root,
    )

    assert [item["role"] for item in manifest["compose_files"]] == [
        "base",
        "release",
        "verify",
        "gpu-compat",
        "secrets",
    ]
    assert [item["role"] for item in manifest["env_files"]] == [
        "secret-paths",
        "images-lock",
        "image-selection",
    ]
    assert [item["sensitive"] for item in manifest["env_files"]] == [
        True,
        False,
        False,
    ]
    assert all(
        len(item["sha256"]) == 64
        for item in (*manifest["compose_files"], *manifest["env_files"])
    )
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert capsys.readouterr().out == ""


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter semantics")
def test_manifest_freeze_accepts_state_directory_with_equivalent_drive_case(
    tmp_path,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    state_path = secret_env.parent / "compose-secrets.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    directory = state["directory"]
    state["directory"] = directory[0].swapcase() + directory[1:]
    assert state["directory"] != directory
    state_path.write_text(json.dumps(state), encoding="utf-8")

    manifest = module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=root / ".runtime" / "verify-compose-manifest.json",
        root=root,
    )

    assert manifest["mode"] == "verify"


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter semantics")
def test_manifest_verify_accepts_bootstrap_record_with_equivalent_drive_case(
    tmp_path,
):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    output = root / ".runtime" / "production-compose-manifest.json"
    manifest = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=_fake_production_manifest_inputs(root, release_env),
        gpu_mode="compat",
        secret_mode="direct",
        output=output,
        root=root,
    )
    bootstrap = manifest["referenced_files"][0]
    original = bootstrap["path"]
    bootstrap["path"] = original[0].swapcase() + original[1:]
    assert bootstrap["path"] != original
    output.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    assert module.verify_manifest_inputs(output, root=root) == manifest


def test_manifest_freeze_rejects_state_directory_for_different_path(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    state_path = secret_env.parent / "compose-secrets.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["directory"] = str(secret_env.parent.parent / "different-run")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(module.ManifestError) as exc_info:
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "verify-compose-manifest.json",
            root=root,
        )

    assert str(exc_info.value) == "compose manifest environment is invalid"


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_manifest_freeze_rejects_relative_state_directory(tmp_path, monkeypatch):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    state_path = secret_env.parent / "compose-secrets.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    monkeypatch.chdir(root)
    state["directory"] = os.path.relpath(secret_env.parent, root)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(module.ManifestError) as exc_info:
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "verify-compose-manifest.json",
            root=root,
        )

    assert str(exc_info.value) == "compose manifest environment is invalid"


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_manifest_freeze_rejects_noncanonical_parent_segments_in_state_directory(
    tmp_path,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    state_path = secret_env.parent / "compose-secrets.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["directory"] = str(secret_env.parent / "nonexistent-child" / "..")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(module.ManifestError) as exc_info:
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "verify-compose-manifest.json",
            root=root,
        )

    assert str(exc_info.value) == "compose manifest environment is invalid"


def test_manifest_verification_rejects_any_input_drift_without_disclosing_values(
    tmp_path,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    output = root / ".runtime" / "verify-compose-manifest.json"
    module.freeze_manifest(
        mode="verify",
        project="trainfactory-verify-" + "ab" * 16,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=output,
        root=root,
    )
    secret_env.write_text("PRIVATE_DRIFT_CANARY=changed\n", encoding="utf-8")

    with pytest.raises(module.ManifestError) as exc_info:
        module.verify_manifest_inputs(output, root=root)

    assert str(exc_info.value) == "compose manifest input verification failed"
    assert "PRIVATE_DRIFT_CANARY" not in repr(exc_info.value)


def test_inputs_only_verifies_resolved_images_ids_and_oci_without_containers(
    tmp_path,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        if args[:3] == ["docker", "image", "inspect"]:
            image = args[-1]
            image_id = "b" * 64 if image == "local/api:fixed" else "c" * 64
            return subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        raise AssertionError("container state must not be read")

    module.verify_manifest_inputs_only(
        manifest_path,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    assert sum(call[:3] == ["docker", "image", "inspect"] for call in calls) == 2
    assert not any(
        call[:2] in (["docker", "ps"], ["docker", "container"]) for call in calls
    )


def test_inputs_only_rejects_image_id_or_revision_mismatch(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )

    def run(args, **kwargs):
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{'f' * 64}|windows|amd64|0.1.0|{'e' * 40}|"
            "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
            "",
        )

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs_only(
            manifest_path,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )


@pytest.mark.parametrize("compat_service", ("api", "web"))
def test_inputs_only_rejects_unexpected_migration_compat_image_label(
    tmp_path,
    compat_service,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )

    def run(args, **kwargs):
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        image = args[-1]
        image_id = "b" * 64 if image == "local/api:fixed" else "c" * 64
        has_compat = (image == "local/api:fixed") == (compat_service == "api")
        compat = "053_validate_lifecycle_schema" if has_compat else ""
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
            "2026-08-17T00:00:00Z|https://example.invalid/train-factory|"
            f"{compat}\n",
            "",
        )

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs_only(
            manifest_path,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "pid",
        "privileged",
        "socket",
        "public-port",
        "devices",
        "mysql-port",
        "wrong-target",
        "wrong-protocol",
        "volume-driver-options",
        "network-driver-options",
        "volumes-from",
        "command",
        "entrypoint",
        "extra-hosts",
        "dns",
        "userns",
        "uts",
        "cgroup",
        "user",
        "sysctls",
        "health-disable",
        "profiles",
        "deploy-replicas",
        "logging",
        "labels",
        "environment",
        "top-config",
        "top-secret",
        "port-mode",
        "port-extra",
        "remove-health",
    ),
)
def test_inputs_only_rejects_dangerous_resolved_runtime_surface(
    tmp_path,
    mutation,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    compose = _minimal_resolved_verify_config(root=root)
    api = compose["services"]["train-factory-api"]
    if mutation == "pid":
        api["pid"] = "host"
    elif mutation == "privileged":
        api["privileged"] = True
    elif mutation == "socket":
        api["volumes"].append(
            {
                "type": "bind",
                "source": "/var/run/docker.sock",
                "target": "/var/run/docker.sock",
            }
        )
    elif mutation == "public-port":
        api["ports"][0]["host_ip"] = "0.0.0.0"
    elif mutation == "devices":
        api["devices"] = ["/dev/nvidia0:/dev/nvidia0"]
    elif mutation == "mysql-port":
        compose["services"]["mysql"]["ports"] = [
            {
                "host_ip": "0.0.0.0",
                "published": "3306",
                "target": 3306,
                "protocol": "tcp",
            }
        ]
    elif mutation == "wrong-target":
        api["ports"][0]["target"] = 22
    elif mutation == "wrong-protocol":
        api["ports"][0]["protocol"] = "udp"
    elif mutation == "volume-driver-options":
        compose["volumes"]["verify_data"]["driver_opts"] = {
            "type": "none",
            "o": "bind",
            "device": "/",
        }
    elif mutation == "network-driver-options":
        compose["networks"]["default"]["driver_opts"] = {"device": "/"}
    elif mutation == "volumes-from":
        api["volumes_from"] = ["train-factory-web"]
    elif mutation == "command":
        api["command"] = ["private-command"]
    elif mutation == "entrypoint":
        api["entrypoint"] = ["private-entrypoint"]
    elif mutation == "extra-hosts":
        api["extra_hosts"] = ["mysql=127.0.0.1"]
    elif mutation == "dns":
        api["dns"] = ["203.0.113.1"]
    elif mutation == "userns":
        api["userns_mode"] = "host"
    elif mutation == "uts":
        api["uts"] = "host"
    elif mutation == "cgroup":
        api["cgroup"] = "host"
    elif mutation == "user":
        api["user"] = "0:0"
    elif mutation == "sysctls":
        api["sysctls"] = {"net.ipv4.ip_forward": "1"}
    elif mutation == "health-disable":
        compose["services"]["train-factory-web"]["healthcheck"] = {"disable": True}
    elif mutation == "profiles":
        api["profiles"] = ["disabled"]
    elif mutation == "deploy-replicas":
        api["deploy"]["replicas"] = 0
    elif mutation == "logging":
        api["logging"] = {
            "driver": "syslog",
            "options": {"syslog-address": "tcp://private"},
        }
    elif mutation == "labels":
        api["labels"] = {"private-owner": "other"}
    elif mutation == "environment":
        api["environment"]["LD_PRELOAD"] = "/app/data/private.so"
    elif mutation == "top-config":
        compose["configs"] = {"private": {"file": "/private"}}
        api["configs"] = [{"source": "private", "target": "/private"}]
    elif mutation == "top-secret":
        compose["secrets"]["private"] = {"file": "/private"}
        api["secrets"].append({"source": "private", "target": "private"})
    elif mutation == "port-mode":
        api["ports"][0]["mode"] = "host"
    elif mutation == "port-extra":
        api["ports"][0]["app_protocol"] = "private"
    else:
        compose["services"]["mysql"].pop("healthcheck")

    def run(args, **kwargs):
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(compose), "")
        image = args[-1]
        image_id = "b" * 64 if image == "local/api:fixed" else "c" * 64
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
            "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
            "",
        )

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs_only(
            manifest_path,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "socket-bind",
        "volume-driver-options",
        "network-driver-options",
        "restart",
        "wrong-target",
        "volume-name",
        "network-name",
        "network-ipam",
        "bind-propagation",
        "duplicate-mount",
        "volume-nocopy",
        "web-api-host",
        "web-api-port",
        "api-mysql-host",
        "api-port-environment",
        "api-project-environment",
        "api-network-environment",
        "api-auth-disabled",
        "api-remote-code",
        "api-self-registration",
        "api-hf-endpoint",
        "api-log-level",
        "api-shm-size",
        "api-memory-limit",
        "api-device-options",
        "api-private-hosts-all",
        "api-rate-limit-disabled",
        "api-trusted-proxies",
        "api-models-dir",
        "api-datasets-dir",
        "api-external-minio",
        "api-host-ip",
    ),
)
@pytest.mark.host_tools
def test_release_policy_rejects_production_runtime_surface_drift(mutation):
    from scripts import compose_release

    compose = _render_production()
    api = compose["services"]["train-factory-api"]
    if mutation == "socket-bind":
        api["volumes"].append(
            {
                "type": "bind",
                "source": "/var/run/docker.sock",
                "target": "/var/run/docker.sock",
            }
        )
    elif mutation == "volume-driver-options":
        compose["volumes"]["train_cache"]["driver_opts"] = {
            "type": "none",
            "o": "bind",
            "device": "/",
        }
    elif mutation == "network-driver-options":
        compose["networks"]["default"]["driver_opts"] = {"device": "/"}
    elif mutation == "restart":
        api["restart"] = "always"
    elif mutation == "wrong-target":
        api["ports"][0]["target"] = 22
    elif mutation == "volume-name":
        compose["volumes"]["mysql_data"]["name"] = "attacker_mysql_data"
    elif mutation == "network-name":
        compose["networks"]["default"]["name"] = "attacker_network"
    elif mutation == "network-ipam":
        compose["networks"]["default"]["ipam"] = {}
    elif mutation == "bind-propagation":
        api["volumes"][0]["bind"]["propagation"] = "rshared"
    elif mutation == "duplicate-mount":
        api["volumes"].append(dict(api["volumes"][0]))
    elif mutation == "volume-nocopy":
        api["volumes"][-1]["volume"]["nocopy"] = True
    elif mutation == "web-api-host":
        compose["services"]["train-factory-web"]["environment"]["API_HOST"] = (
            "attacker.invalid"
        )
    elif mutation == "web-api-port":
        compose["services"]["train-factory-web"]["environment"]["API_PORT"] = "443"
    elif mutation == "api-mysql-host":
        api["environment"]["MYSQL_HOST"] = "db.attacker.invalid"
    elif mutation == "api-port-environment":
        api["environment"]["API_PORT"] = "1"
    elif mutation == "api-project-environment":
        api["environment"]["COMPOSE_PROJECT_NAME"] = "other"
    elif mutation == "api-network-environment":
        api["environment"]["DOCKER_NETWORK_NAME"] = "other-network"
    elif mutation == "api-auth-disabled":
        api["environment"]["AUTH_ENABLED"] = "false"
    elif mutation == "api-remote-code":
        api["environment"]["ALLOW_MODEL_REMOTE_CODE"] = "true"
    elif mutation == "api-self-registration":
        api["environment"]["SELF_REGISTRATION_ENABLED"] = "true"
    elif mutation == "api-hf-endpoint":
        api["environment"]["HF_ENDPOINT"] = "https://attacker.invalid"
    elif mutation == "api-log-level":
        api["environment"]["LOG_LEVEL"] = "DEBUG"
    elif mutation == "api-shm-size":
        api["shm_size"] = "1"
    elif mutation == "api-memory-limit":
        api["deploy"]["resources"]["limits"] = {"memory": "1048576"}
    elif mutation == "api-device-options":
        api["deploy"]["resources"]["reservations"]["devices"][0]["options"] = {
            "canary": "true"
        }
    elif mutation == "api-private-hosts-all":
        api["environment"]["DISCOVER_MODELS_ALLOW_PRIVATE_ALL"] = "true"
    elif mutation == "api-rate-limit-disabled":
        api["environment"]["RATE_LIMIT_ENABLED"] = "false"
    elif mutation == "api-trusted-proxies":
        api["environment"]["RATE_LIMIT_TRUSTED_PROXIES"] = "0.0.0.0/0"
    elif mutation == "api-models-dir":
        api["environment"]["MODELS_DIR"] = "/run/secrets"
    elif mutation == "api-datasets-dir":
        api["environment"]["DATASETS_DIR"] = "/run/secrets"
    elif mutation == "api-external-minio":
        api["environment"]["STORAGE_BACKEND"] = "minio"
        api["environment"]["MINIO_ENDPOINT"] = "https://attacker.invalid"
    else:
        api["environment"]["HOST_IP"] = "203.0.113.10"

    with pytest.raises(compose_release.ReleaseComposeError):
        compose_release._validate_resolved_config(
            _production_validation_manifest(),
            compose,
            root=ROOT_DIR,
        )


def test_verify_running_requires_one_exact_labeled_container_and_image(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    container_id = "d" * 64
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        if args[:3] == ["docker", "image", "inspect"]:
            image = args[-1]
            image_id = "b" * 64 if image == "local/api:fixed" else "c" * 64
            return subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        if args[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(args, 0, container_id + "\n", "")
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args,
                0,
                f"{container_id}|sha256:{'b' * 64}|true|{PROJECT}|train-factory-api\n",
                "",
            )
        raise AssertionError("unexpected Docker command")

    module.verify_running_container(
        manifest_path,
        selection_path=release_env,
        service="train-factory-api",
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    ps_call = next(call for call in calls if call[:2] == ["docker", "ps"])
    assert "--no-trunc" in ps_call
    assert f"label=com.docker.compose.project={PROJECT}" in ps_call
    assert "label=com.docker.compose.service=train-factory-api" in ps_call


@pytest.mark.parametrize(
    "values",
    (
        {
            "API_IMAGE": "local/api:fixed",
            "WEB_IMAGE": "local/web:fixed",
            "RELEASE_REVISION": "a" * 40,
            "API_IMAGE_ID": "sha256:" + "b" * 64,
            "WEB_IMAGE_ID": "sha256:" + "c" * 64,
        },
        {
            "API_IMAGE": "local/api:fixed",
            "WEB_IMAGE": "local/web:fixed",
            "API_REVISION": "a" * 40,
            "WEB_REVISION": "d" * 40,
            "API_IMAGE_ID": "sha256:" + "b" * 64,
            "WEB_IMAGE_ID": "sha256:" + "c" * 64,
        },
    ),
)
def test_production_image_selection_accepts_each_closed_revision_schema(values):
    module = _manifest_module()
    payload = "".join(f"{key}={value}\n" for key, value in values.items()).encode()

    module._validate_environment_schema(
        payload,
        role="image-selection",
        project="trainfactory",
        secret_mode="direct",
        mode="production",
    )


@pytest.mark.parametrize(
    "revision_lines",
    (
        (f"RELEASE_REVISION={'a' * 40}", f"API_REVISION={'b' * 40}"),
        (f"API_REVISION={'a' * 40}",),
        (
            f"API_REVISION={'a' * 40}",
            f"WEB_REVISION={'b' * 40}",
            "EXTRA_REVISION=blocked",
        ),
    ),
)
def test_production_image_selection_rejects_mixed_incomplete_or_extra_schema(
    revision_lines,
):
    module = _manifest_module()
    payload = (
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:fixed\n"
        + "\n".join(revision_lines)
        + "\n"
        + f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        + f"WEB_IMAGE_ID=sha256:{'c' * 64}\n"
    ).encode()

    with pytest.raises(
        module.ManifestError,
        match="^compose manifest environment is invalid$",
    ):
        module._validate_environment_schema(
            payload,
            role="image-selection",
            project="trainfactory",
            secret_mode="direct",
            mode="production",
        )


def test_production_manifest_freezes_per_service_release_revisions(tmp_path):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    release_env.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:fixed\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )
    output = root / ".runtime" / "production-compose-manifest.json"

    manifest = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=_fake_production_manifest_inputs(root, release_env),
        gpu_mode="compat",
        secret_mode="direct",
        output=output,
        root=root,
    )

    assert manifest["mode"] == "production"
    assert module._selection_values(manifest) == {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "local/web:fixed",
        "API_REVISION": "a" * 40,
        "WEB_REVISION": "d" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "c" * 64,
    }


@pytest.mark.parametrize(
    ("source_revision_lines", "expected_api_revision"),
    (
        ((f"RELEASE_REVISION={'a' * 40}",), "a" * 40),
        (
            (f"API_REVISION={'d' * 40}", f"WEB_REVISION={'e' * 40}"),
            "d" * 40,
        ),
    ),
)
def test_forward_web_builder_preserves_verified_production_api_and_inspects_web(
    tmp_path,
    monkeypatch,
    source_revision_lines,
    expected_api_revision,
):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    release_env.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:old\n"
        + "\n".join(source_revision_lines)
        + "\n"
        + f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        + f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )
    source_payload = release_env.read_bytes()
    source = root / ".runtime" / "production-compose-manifest.json"
    module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=_fake_production_manifest_inputs(root, release_env),
        gpu_mode="compat",
        secret_mode="direct",
        output=source,
        root=root,
    )
    initial_runtime_entries = set((root / ".runtime").iterdir())
    verified_paths = []

    def verify_inputs(path, **kwargs):
        verified_paths.append(path)
        return module.verify_manifest_inputs(path, root=root)

    monkeypatch.setattr(module, "verify_manifest_inputs_only", verify_inputs)
    inspect_calls = []

    def run(args, **kwargs):
        inspect_calls.append(list(args))
        assert args[-1] == "local/web:new"
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{'9' * 64}|linux|amd64|0.2.0|{'f' * 40}|"
            "2026-08-19T00:00:00Z|https://example.invalid/train-factory|\n",
            "",
        )

    values = module.build_forward_web_selection(
        production_manifest=source,
        web_image="local/web:new",
        web_image_id="sha256:" + "9" * 64,
        web_revision="f" * 40,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    assert values == {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "local/web:new",
        "API_REVISION": expected_api_revision,
        "WEB_REVISION": "f" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "9" * 64,
    }
    assert verified_paths == [source, source]
    assert len(inspect_calls) == 1
    assert release_env.read_bytes() == source_payload
    assert set((root / ".runtime").iterdir()) == initial_runtime_entries


@pytest.mark.parametrize(
    "inspect_output",
    (
        "sha256:{id}|windows|amd64|0.2.0|{revision}|created|https://source|",
        "sha256:{id}|linux|arm64|0.2.0|{revision}|created|https://source|",
        "sha256:{wrong}|linux|amd64|0.2.0|{revision}|created|https://source|",
        "sha256:{id}|linux|amd64|0.2.0|{wrong_revision}|created|https://source|",
        "sha256:{id}|linux|amd64||{revision}|created|https://source|",
        "sha256:{id}|linux|amd64|0.2.0|{revision}||https://source|",
        "sha256:{id}|linux|amd64|0.2.0|{revision}|created|http://source|",
    ),
)
def test_forward_web_builder_rejects_invalid_web_immutable_identity(
    tmp_path,
    monkeypatch,
    inspect_output,
):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    source = root / ".runtime" / "production-compose-manifest.json"
    manifest = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=_fake_production_manifest_inputs(root, release_env),
        gpu_mode="compat",
        secret_mode="direct",
        output=source,
        root=root,
    )
    monkeypatch.setattr(
        module,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: manifest,
    )
    output = inspect_output.format(
        id="9" * 64,
        wrong="8" * 64,
        revision="f" * 40,
        wrong_revision="e" * 40,
    )

    with pytest.raises(
        module.ManifestError,
        match="^compose forward Web selection failed$",
    ):
        module.build_forward_web_selection(
            production_manifest=source,
            web_image="local/web:new",
            web_image_id="sha256:" + "9" * 64,
            web_revision="f" * 40,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda args, **_kwargs: subprocess.CompletedProcess(
                args, 0, output + "\n", ""
            ),
        )


def test_forward_web_builder_rejects_source_manifest_drift(tmp_path, monkeypatch):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    source = root / ".runtime" / "production-compose-manifest.json"
    manifest = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=_fake_production_manifest_inputs(root, release_env),
        gpu_mode="compat",
        secret_mode="direct",
        output=source,
        root=root,
    )
    drifted = dict(manifest)
    drifted["project"] = "other"
    verified = iter((manifest, drifted))
    monkeypatch.setattr(
        module,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: next(verified),
    )

    with pytest.raises(
        module.ManifestError,
        match="^compose forward Web selection failed$",
    ):
        module.build_forward_web_selection(
            production_manifest=source,
            web_image="local/web:new",
            web_image_id="sha256:" + "9" * 64,
            web_revision="f" * 40,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda args, **_kwargs: subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{'9' * 64}|linux|amd64|0.2.0|{'f' * 40}|"
                "2026-08-19T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            ),
        )


def test_forward_web_builder_rejects_nonproduction_manifest(tmp_path, monkeypatch):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest = {
        "mode": "rollback-pre",
        "project": "trainfactory",
        "secret_mode": "direct",
        "env_files": [{"role": "image-selection", "path": os.fspath(release_env)}],
    }
    monkeypatch.setattr(
        module,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: manifest,
    )

    with pytest.raises(
        module.ManifestError,
        match="^compose forward Web selection failed$",
    ):
        module.build_forward_web_selection(
            production_manifest=root / ".runtime" / "rollback-pre-manifest.json",
            web_image="local/web:new",
            web_image_id="sha256:" + "9" * 64,
            web_revision="f" * 40,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda args, **_kwargs: subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{'9' * 64}|linux|amd64|0.2.0|{'f' * 40}|"
                "2026-08-19T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            ),
        )


def test_mix_image_selection_inspects_each_source_and_writes_per_service_revision(
    tmp_path,
):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    captured = root / ".runtime" / "rollback-pre.env"
    captured.write_text(
        "API_IMAGE=old-api:fixed\n"
        "WEB_IMAGE=old-web:fixed\n"
        f"API_REVISION={'d' * 40}\n"
        f"WEB_REVISION={'e' * 40}\n"
        f"API_IMAGE_ID=sha256:{'f' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'9' * 64}\n",
        encoding="utf-8",
    )
    output = root / ".runtime" / "rollback-web-only.env"
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        image = args[-1]
        if image == "local/api:fixed":
            image_id, revision = "b" * 64, "a" * 40
        elif image == "old-web:fixed":
            image_id, revision = "9" * 64, "e" * 40
        else:
            raise AssertionError("unselected image must not be inspected")
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{image_id}|linux|amd64|0.1.0|{revision}|"
            "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
            "",
        )

    values = module.mix_image_selection(
        api_source=release_env,
        web_source=captured,
        output=output,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    assert values == {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "old-web:fixed",
        "API_REVISION": "a" * 40,
        "WEB_REVISION": "e" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "9" * 64,
    }
    assert output.read_text(encoding="utf-8").splitlines() == [
        f"{key}={value}" for key, value in values.items()
    ]
    assert len(calls) == 4


def test_image_selection_inspects_expected_ids_and_writes_closed_environment(tmp_path):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    output = root / ".runtime" / f"ci-release-{'1' * 32}.env"

    def run(args, **kwargs):
        image = args[-1]
        image_id = "b" * 64 if image == "local/api:ci" else "c" * 64
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
            "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
            "",
        )

    values = module.create_image_selection(
        api_image="local/api:ci",
        web_image="local/web:ci",
        api_image_id="sha256:" + "b" * 64,
        web_image_id="sha256:" + "c" * 64,
        revision="a" * 40,
        output=output,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    assert set(values) == {
        "API_IMAGE",
        "WEB_IMAGE",
        "RELEASE_REVISION",
        "API_IMAGE_ID",
        "WEB_IMAGE_ID",
    }
    assert output.read_text(encoding="utf-8").splitlines() == [
        f"{key}={value}" for key, value in values.items()
    ]


def _ci_manifest_inputs(root, secret_env, release_env, run_id):
    source_bundle = secret_env.parent
    ci_bundle = root / ".runtime" / f"ci-{run_id}"
    source_bundle.rename(ci_bundle)
    ci_secret_env = ci_bundle / "compose-secrets.env"
    state_path = ci_bundle / "compose-secrets.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "scope": "ci",
            "run_id": run_id,
            "directory": os.fspath(ci_bundle),
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    ci_secret_env.write_text(
        ci_secret_env.read_text(encoding="utf-8").replace(
            os.fspath(source_bundle), os.fspath(ci_bundle)
        ),
        encoding="utf-8",
    )
    ci_selection = root / ".runtime" / f"ci-release-{run_id}.env"
    release_env.rename(ci_selection)
    return [
        ci_secret_env,
        root / "docker" / "images.lock.env",
        ci_selection,
    ]


def test_ci_manifest_binds_project_selection_and_manifest_run_id(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    run_id = "1" * 32
    env_files = _ci_manifest_inputs(root, secret_env, release_env, run_id)
    manifest_path = root / ".runtime" / f"ci-compose-{run_id}-manifest.json"

    manifest = module.freeze_manifest(
        mode="ci",
        project=f"trainfactory-ci-{run_id}",
        env_files=env_files,
        gpu_mode="cpu",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )

    assert module.verify_manifest_inputs(manifest_path, root=root) == manifest


@pytest.mark.parametrize(
    ("selection_name", "manifest_name"),
    [
        ("ci-release.env", f"ci-compose-{'1' * 32}-manifest.json"),
        (f"ci-release-{'2' * 32}.env", f"ci-compose-{'1' * 32}-manifest.json"),
        (f"ci-release-{'1' * 32}.env", "ci-compose-manifest.json"),
        (f"ci-release-{'1' * 32}.env", f"ci-compose-{'2' * 32}-manifest.json"),
    ],
)
def test_ci_manifest_rejects_unsynchronized_artifact_names(
    tmp_path,
    selection_name,
    manifest_name,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    run_id = "1" * 32
    env_files = _ci_manifest_inputs(root, secret_env, release_env, run_id)
    selection = root / ".runtime" / selection_name
    env_files[-1].rename(selection)
    env_files[-1] = selection

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="ci",
            project=f"trainfactory-ci-{run_id}",
            env_files=env_files,
            gpu_mode="cpu",
            secret_mode="files",
            output=root / ".runtime" / manifest_name,
            root=root,
        )


def test_ci_manifest_verify_rejects_copy_at_wrong_run_bound_path(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    run_id = "1" * 32
    env_files = _ci_manifest_inputs(root, secret_env, release_env, run_id)
    manifest_path = root / ".runtime" / f"ci-compose-{run_id}-manifest.json"
    module.freeze_manifest(
        mode="ci",
        project=f"trainfactory-ci-{run_id}",
        env_files=env_files,
        gpu_mode="cpu",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    copied = root / ".runtime" / f"ci-compose-{'2' * 32}-manifest.json"
    copied.write_bytes(manifest_path.read_bytes())

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs(copied, root=root)


def test_ci_manifest_freeze_is_exclusive_and_preserves_concurrent_target(
    tmp_path,
    monkeypatch,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    run_id = "1" * 32
    env_files = _ci_manifest_inputs(root, secret_env, release_env, run_id)
    output = root / ".runtime" / f"ci-compose-{run_id}-manifest.json"
    original_link = module.os.link

    def concurrent_target(source, destination):
        Path(destination).write_text("FOREIGN-MANIFEST-CANARY", encoding="utf-8")
        return original_link(source, destination)

    monkeypatch.setattr(module.os, "link", concurrent_target)

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="ci",
            project=f"trainfactory-ci-{run_id}",
            env_files=env_files,
            gpu_mode="cpu",
            secret_mode="files",
            output=output,
            root=root,
        )

    assert output.read_text(encoding="utf-8") == "FOREIGN-MANIFEST-CANARY"


def test_ci_paths_reject_parent_alias_components(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    run_id = "1" * 32
    env_files = _ci_manifest_inputs(root, secret_env, release_env, run_id)
    runtime = root / ".runtime"
    selection_alias = runtime / "child" / ".." / f"ci-release-{run_id}.env"
    env_files[-1] = selection_alias

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="ci",
            project=f"trainfactory-ci-{run_id}",
            env_files=env_files,
            gpu_mode="cpu",
            secret_mode="files",
            output=runtime / f"ci-compose-{run_id}-manifest.json",
            root=root,
        )


def test_ci_image_selection_rejects_parent_alias_output(tmp_path):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    runtime = root / ".runtime"
    (runtime / "child").mkdir()
    output = runtime / "child" / ".." / f"ci-release-{'1' * 32}.env"

    with pytest.raises(module.ManifestError):
        module.create_image_selection(
            api_image="local/api:ci",
            web_image="local/web:ci",
            api_image_id="sha256:" + "b" * 64,
            web_image_id="sha256:" + "c" * 64,
            revision="a" * 40,
            output=output,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda *_args, **_kwargs: pytest.fail("inspect must not run"),
        )


@pytest.mark.host_tools
def test_capture_rollback_uses_exact_labels_and_records_immutable_images(
    tmp_path, monkeypatch
):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    project = "trainfactory"
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n"
        "DEBUG=false\n",
        encoding="utf-8",
    )
    direct_payload = (
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n"
    )
    production_direct = root / ".runtime" / "production-direct.env"
    production_direct.write_text(direct_payload, encoding="utf-8")
    release_env.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:fixed\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )
    source = root / ".runtime" / "production-compose-manifest.json"
    module.freeze_manifest(
        mode="production",
        project=project,
        env_files=(
            root / ".env",
            production_direct,
            root / "docker" / "images.lock.env",
            release_env,
        ),
        gpu_mode="compat",
        secret_mode="direct",
        output=source,
        root=root,
    )
    rollback_direct = root / ".runtime" / "rollback-direct.env"
    rollback_direct.write_text(direct_payload, encoding="utf-8")
    output = root / ".runtime" / "rollback-pre.env"
    output_manifest = root / ".runtime" / "rollback-pre-compose-manifest.json"
    container_ids = {
        "train-factory-api": "1" * 64,
        "train-factory-web": "2" * 64,
    }
    image_ids = {
        "train-factory-api": "sha256:" + "3" * 64,
        "train-factory-web": "sha256:" + "4" * 64,
    }
    revisions = {
        "train-factory-api": "5" * 40,
        "train-factory-web": "6" * 40,
    }
    calls = []
    tags = {}
    fail_next_remove = [False]
    remove_attempts = []
    source_compose = _render_production(root / ".env")
    source_compose["services"]["mysql"]["image"] = (
        "example/mysql_image:fixed@sha256:" + f"{5:064x}"
    )
    source_compose["services"]["train-factory-api"]["image"] = "local/api:fixed"
    source_compose["services"]["train-factory-web"]["image"] = "local/web:fixed"
    source_compose["services"]["mysql"]["environment"].update(
        {
            "MYSQL_ROOT_PASSWORD": "private-root-123456",
            "MYSQL_PASSWORD": "private-app-123456",
        }
    )
    source_compose["services"]["train-factory-api"]["environment"].update(
        {
            "MYSQL_APP_PASSWORD": "private-app-123456",
            "JWT_SECRET_KEY": "private-jwt-123456-private-jwt-123456",
            "DEFAULT_ADMIN_PASSWORD": "private-admin-123456",
            "VLLM_IMAGE": "example/vllm_image:fixed@sha256:" + f"{8:064x}",
            "SGLANG_IMAGE": "example/sglang_image:fixed@sha256:" + f"{9:064x}",
            "XINFERENCE_IMAGE": "example/xinference_image:fixed@sha256:" + f"{10:064x}",
        }
    )
    for service in source_compose["services"].values():
        for mount in service.get("volumes", []):
            if mount["type"] != "bind":
                continue
            target = mount["target"]
            if target == "/docker-entrypoint-initdb.d/init.sql":
                mount["source"] = os.fspath(root / "docker" / "init.sql")
            else:
                mount["source"] = os.fspath(root / target.removeprefix("/app/"))

    def run(args, **kwargs):
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            normalized_args = {os.path.normcase(os.path.abspath(item)) for item in args}
            if os.path.normcase(os.path.abspath(output)) not in normalized_args:
                return subprocess.CompletedProcess(
                    args, 0, json.dumps(source_compose), ""
                )
            rendered = copy.deepcopy(source_compose)
            rendered["services"]["train-factory-api"]["image"] = next(
                tag for tag in tags if tag.endswith("-api:captured")
            )
            rendered["services"]["train-factory-web"]["image"] = next(
                tag for tag in tags if tag.endswith("-web:captured")
            )
            return subprocess.CompletedProcess(args, 0, json.dumps(rendered), "")
        if args[:2] == ["docker", "ps"]:
            service_filter = next(item for item in args if "service=" in item)
            service = service_filter.rsplit("=", 1)[1]
            return subprocess.CompletedProcess(
                args, 0, container_ids[service] + "\n", ""
            )
        if args[:2] == ["docker", "inspect"]:
            container_id = args[-1]
            service = next(
                name
                for name, identifier in container_ids.items()
                if identifier == container_id
            )
            return subprocess.CompletedProcess(
                args,
                0,
                f"{container_id}|{image_ids[service]}|true|{project}|{service}\n",
                "",
            )
        if args[:3] == ["docker", "image", "inspect"]:
            if args[-1] in {"local/api:fixed", "local/web:fixed"}:
                is_api = args[-1] == "local/api:fixed"
                image_id = "b" * 64 if is_api else "c" * 64
                revision = "a" * 40 if is_api else "d" * 40
                return subprocess.CompletedProcess(
                    args,
                    0,
                    f"sha256:{image_id}|linux|amd64|0.1.0|{revision}|"
                    "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                    "",
                )
            image_id = tags.get(args[-1], args[-1])
            if args[-1].startswith("trainfactory-rollback-") and args[-1] not in tags:
                return subprocess.CompletedProcess(args, 1, "", "not found")
            if args[4] == "{{.Id}}":
                return subprocess.CompletedProcess(args, 0, image_id + "\n", "")
            service = next(
                name for name, value in image_ids.items() if value == image_id
            )
            return subprocess.CompletedProcess(
                args,
                0,
                f"{image_id}|linux|amd64|0.1.0|{revisions[service]}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        if args[:3] == ["docker", "image", "tag"]:
            tags[args[-1]] = args[-2]
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ["docker", "image", "rm"]:
            remove_attempts.append(args[-1])
            if fail_next_remove[0]:
                fail_next_remove[0] = False
                return subprocess.CompletedProcess(args, 1, "", "remove failed")
            tags.pop(args[-1], None)
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ["docker", "image", "ls"]:
            tag = next(
                item.removeprefix("reference=")
                for item in args
                if item.startswith("reference=")
            )
            return subprocess.CompletedProcess(
                args,
                0,
                (tag + "\n") if tag in tags else "",
                "",
            )
        raise AssertionError("unexpected Docker command")

    values = module.capture_rollback(
        source=source,
        project=project,
        api_service="train-factory-api",
        web_service="train-factory-web",
        rollback_direct_env=rollback_direct,
        release_sha="d" * 40,
        output_env=output,
        output_manifest=output_manifest,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
        token_hex=lambda _size: "7" * 32,
    )

    assert values == {
        "API_IMAGE": "trainfactory-rollback-"
        + "d" * 12
        + "-"
        + "7" * 32
        + "-api:captured",
        "WEB_IMAGE": "trainfactory-rollback-"
        + "d" * 12
        + "-"
        + "7" * 32
        + "-web:captured",
        "API_REVISION": "5" * 40,
        "WEB_REVISION": "6" * 40,
        "API_IMAGE_ID": image_ids["train-factory-api"],
        "WEB_IMAGE_ID": image_ids["train-factory-web"],
    }
    assert output.read_text(encoding="utf-8").splitlines() == [
        f"{key}={value}" for key, value in values.items()
    ]
    assert sum(call[:3] == ["docker", "image", "tag"] for call in calls) == 2
    inspect_templates = [call[4] for call in calls if call[:2] == ["docker", "inspect"]]
    assert all(
        ".Config.Env" not in template and "{{json .}}" not in template
        for template in inspect_templates
    )
    captured_manifest = module.verify_manifest_inputs(output_manifest, root=root)
    assert captured_manifest["mode"] == "rollback-pre"
    assert [item["role"] for item in captured_manifest["env_files"]] == [
        "base-environment",
        "images-lock",
        "rollback-direct",
        "rollback-pre",
    ]

    output.unlink()
    output_manifest.unlink()
    tags.clear()
    original_read_credentials = module._read_rollback_credentials
    credential_reads = 0

    def replace_after_first_read(*args, **kwargs):
        nonlocal credential_reads
        credential_reads += 1
        result = original_read_credentials(*args, **kwargs)
        if credential_reads > 1:
            result = dict(result)
            result["JWT_SECRET_KEY"] = "different-private-jwt-secret-value"
        return result

    monkeypatch.setattr(
        module,
        "_read_rollback_credentials",
        replace_after_first_read,
    )
    fail_next_remove[0] = True
    with pytest.raises(module.ManifestError) as exc_info:
        module.capture_rollback(
            source=source,
            project=project,
            api_service="train-factory-api",
            web_service="train-factory-web",
            rollback_direct_env=rollback_direct,
            release_sha="d" * 40,
            output_env=output,
            output_manifest=output_manifest,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
            token_hex=lambda _size: "8" * 32,
        )
    assert str(exc_info.value) == "compose rollback capture cleanup failed"
    assert len(remove_attempts) == 2
    assert len(tags) == 1
    assert not output.exists()
    assert not output_manifest.exists()


def test_capture_rejects_credential_discontinuity_before_tagging(tmp_path, monkeypatch):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n",
        encoding="utf-8",
    )

    def direct(app_password):
        return (
            "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
            "MYSQL_APP_USER='trainfactory_app'\n"
            f"MYSQL_APP_PASSWORD='{app_password}'\n"
            f"MYSQL_PASSWORD='{app_password}'\n"
            "MYSQL_URL='mysql+pymysql://trainfactory_app:"
            f"{app_password}@mysql:3306/train_factory'\n"
            "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
            "DEFAULT_ADMIN_USERNAME='admin'\n"
            "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n"
        )

    production_direct = root / ".runtime" / "production-direct.env"
    production_direct.write_text(direct("source-app-123456"), encoding="utf-8")
    source = root / ".runtime" / "production-compose-manifest.json"
    original = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=(
            root / ".env",
            production_direct,
            root / "docker" / "images.lock.env",
            release_env,
        ),
        gpu_mode="raw",
        secret_mode="direct",
        output=source,
        root=root,
    )
    rollback_direct = root / ".runtime" / "rollback-direct.env"
    rollback_direct.write_text(direct("other-app-123456"), encoding="utf-8")
    monkeypatch.setattr(
        module, "verify_manifest_inputs_only", lambda *_a, **_k: original
    )
    calls = []

    with pytest.raises(module.ManifestError):
        module.capture_rollback(
            source=source,
            project="trainfactory",
            api_service="train-factory-api",
            web_service="train-factory-web",
            rollback_direct_env=rollback_direct,
            release_sha="a" * 40,
            output_env=root / ".runtime" / "rollback-pre.env",
            output_manifest=root / ".runtime" / "rollback-pre-compose-manifest.json",
            root=root,
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


def test_capture_rejects_file_secret_discontinuity_before_tagging(
    tmp_path, monkeypatch
):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n",
        encoding="utf-8",
    )
    values = {
        "mysql_root_password": "private-root-123456",
        "mysql_app_password": "private-app-123456",
        "mysql_url": (
            "mysql+pymysql://trainfactory_app:private-app-123456"
            "@mysql:3306/train_factory"
        ),
        "jwt_secret_key": "private-jwt-123456-private-jwt-123456",
        "default_admin_password": "private-admin-123456",
    }
    paths = {}
    for name, value in values.items():
        path = root / ".runtime" / name
        path.write_text(value + "\n", encoding="utf-8")
        paths[name] = path
    production_files = root / ".runtime" / "production-files.env"
    production_files.write_text(
        "".join(
            f"{key}='{paths[filename]}'\n"
            for key, filename in {
                "MYSQL_ROOT_PASSWORD_SECRET_PATH": "mysql_root_password",
                "MYSQL_APP_PASSWORD_SECRET_PATH": "mysql_app_password",
                "MYSQL_URL_SECRET_PATH": "mysql_url",
                "JWT_SECRET_KEY_SECRET_PATH": "jwt_secret_key",
                "DEFAULT_ADMIN_PASSWORD_SECRET_PATH": "default_admin_password",
            }.items()
        ),
        encoding="utf-8",
    )
    source = root / ".runtime" / "production-compose-manifest.json"
    original = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=(
            root / ".env",
            production_files,
            root / "docker" / "images.lock.env",
            release_env,
        ),
        gpu_mode="raw",
        secret_mode="files",
        output=source,
        root=root,
    )
    rollback_direct = root / ".runtime" / "rollback-direct.env"
    matching_direct = (
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n"
    )
    rollback_direct.write_text(
        matching_direct,
        encoding="utf-8",
    )
    assert module._read_rollback_credentials(
        rollback_direct,
        project="trainfactory",
        root=root,
    ) == module._deployment_credentials_from_manifest(original)
    rollback_direct.write_text(
        matching_direct.replace(
            "private-jwt-123456-private-jwt-123456",
            "different-private-jwt-secret-value",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module, "verify_manifest_inputs_only", lambda *_a, **_k: original
    )
    calls = []

    with pytest.raises(module.ManifestError):
        module.capture_rollback(
            source=source,
            project="trainfactory",
            api_service="train-factory-api",
            web_service="train-factory-web",
            rollback_direct_env=rollback_direct,
            release_sha="a" * 40,
            output_env=root / ".runtime" / "rollback-pre.env",
            output_manifest=root / ".runtime" / "rollback-pre-compose-manifest.json",
            root=root,
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


def test_capture_cleans_tag_created_before_tag_command_timeout(tmp_path, monkeypatch):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n",
        encoding="utf-8",
    )
    direct_payload = (
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n"
    )
    production_direct = root / ".runtime" / "production-direct.env"
    rollback_direct = root / ".runtime" / "rollback-direct.env"
    production_direct.write_text(direct_payload, encoding="utf-8")
    rollback_direct.write_text(direct_payload, encoding="utf-8")
    source = root / ".runtime" / "production-compose-manifest.json"
    original = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=(
            root / ".env",
            production_direct,
            root / "docker" / "images.lock.env",
            release_env,
        ),
        gpu_mode="raw",
        secret_mode="direct",
        output=source,
        root=root,
    )
    monkeypatch.setattr(
        module, "verify_manifest_inputs_only", lambda *_a, **_k: original
    )
    container_id = "1" * 64
    image_id = "sha256:" + "3" * 64
    tags = {}
    calls = []
    foreign_tag = False

    def run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(args, 0, container_id + "\n", "")
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args,
                0,
                f"{container_id}|{image_id}|true|trainfactory|train-factory-api\n",
                "",
            )
        if args[:3] == ["docker", "image", "ls"]:
            tag = next(
                item.removeprefix("reference=")
                for item in args
                if item.startswith("reference=")
            )
            return subprocess.CompletedProcess(
                args, 0, (tag + "\n") if tag in tags else "", ""
            )
        if args[:3] == ["docker", "image", "inspect"]:
            if args[-1] in tags:
                return subprocess.CompletedProcess(args, 0, tags[args[-1]] + "\n", "")
            return subprocess.CompletedProcess(
                args,
                0,
                f"{image_id}|linux|amd64|0.1.0|{'5' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        if args[:3] == ["docker", "image", "tag"]:
            tags[args[-1]] = "sha256:" + "f" * 64 if foreign_tag else args[-2]
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if args[:3] == ["docker", "image", "rm"]:
            tags.pop(args[-1])
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError("unexpected Docker command")

    output_env = root / ".runtime" / "rollback-pre.env"
    output_manifest = root / ".runtime" / "rollback-pre-compose-manifest.json"
    with pytest.raises(module.ManifestError):
        module.capture_rollback(
            source=source,
            project="trainfactory",
            api_service="train-factory-api",
            web_service="train-factory-web",
            rollback_direct_env=rollback_direct,
            release_sha="a" * 40,
            output_env=output_env,
            output_manifest=output_manifest,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
            token_hex=lambda _size: "7" * 32,
        )

    assert tags == {}
    assert sum(call[:3] == ["docker", "image", "rm"] for call in calls) == 1
    assert not output_env.exists()
    assert not output_manifest.exists()

    tags.clear()
    calls.clear()
    foreign_tag = True
    with pytest.raises(module.ManifestError) as exc_info:
        module.capture_rollback(
            source=source,
            project="trainfactory",
            api_service="train-factory-api",
            web_service="train-factory-web",
            rollback_direct_env=rollback_direct,
            release_sha="a" * 40,
            output_env=output_env,
            output_manifest=output_manifest,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
            token_hex=lambda _size: "8" * 32,
        )
    assert str(exc_info.value) == "compose rollback capture cleanup failed"
    assert len(tags) == 1
    assert not any(call[:3] == ["docker", "image", "rm"] for call in calls)


def test_mix_image_selection_rejects_sources_with_reversed_selection_schemas(
    tmp_path,
):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    captured = root / ".runtime" / "rollback-pre.env"
    release_env.write_text(
        "API_IMAGE=old-api:fixed\n"
        "WEB_IMAGE=old-web:fixed\n"
        f"API_REVISION={'d' * 40}\n"
        f"WEB_REVISION={'e' * 40}\n"
        f"API_IMAGE_ID=sha256:{'f' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'9' * 64}\n",
        encoding="utf-8",
    )
    captured.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:fixed\n"
        f"RELEASE_REVISION={'a' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ManifestError):
        module.mix_image_selection(
            api_source=release_env,
            web_source=captured,
            output=root / ".runtime" / "rollback-web-only.env",
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda *_args, **_kwargs: pytest.fail(
                "images must not be inspected for reversed selection schemas"
            ),
        )


def test_manifest_rejects_cross_role_hardlinks_and_symlinks(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    images_lock = root / "docker" / "images.lock.env"
    release_env.unlink()
    try:
        os.link(secret_env, release_env)
    except OSError:
        pytest.skip("hardlinks are unavailable")

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project="trainfactory-verify-" + "ab" * 16,
            env_files=(secret_env, images_lock, release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "manifest.json",
            root=root,
        )


def test_manifest_rejects_runtime_input_through_symlinked_ancestor(tmp_path):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    actual = root / ".runtime" / "actual"
    actual.mkdir()
    secret_env = actual / "verify.env"
    secret_env.write_text("SECRET_FILE=/run/private\n", encoding="utf-8")
    link = root / ".runtime" / "linked"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(
                link / "verify.env",
                root / "docker" / "images.lock.env",
                release_env,
            ),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "manifest.json",
            root=root,
        )


def test_manifest_rejects_output_through_symlinked_ancestor(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    actual = root / ".runtime" / "actual"
    (actual / "nested").mkdir(parents=True)
    link = root / ".runtime" / "linked"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=link / "nested" / "manifest.json",
            root=root,
        )


def test_stable_read_rejects_same_size_in_place_mutation(tmp_path, monkeypatch):
    module = _manifest_module()
    path = tmp_path / "input.env"
    path.write_bytes(b"A" * 32)
    original_read = module.os.read

    def mutate_after_read(descriptor, count):
        payload = original_read(descriptor, count)
        path.write_bytes(b"B" * 32)
        return payload

    monkeypatch.setattr(module.os, "read", mutate_after_read)

    with pytest.raises(module.ManifestError):
        module._read_stable(path, max_bytes=1024)


@pytest.mark.parametrize(
    "payload",
    (
        b'include: ["private.yml"]\nservices: {}\n',
        b'services:\n  mysql:\n    "env_file": private.env\n',
        b"services: {mysql: {env_file: private.env}}\n",
        b"services: {mysql: {extends: private-service}}\n",
        b"x-private: &private {env_file: private.env}\n"
        b"services: {mysql: {<<: *private, image: mysql}}\n",
        b"services: {mysql: {environment: {LOG_LEVEL: ${HTTPS_PROXY}}}}\n",
        b"services: {mysql: {environment: {LOG_LEVEL: $HTTPS_PROXY}}}\n",
        b"services: {mysql: {environment: {LOG_LEVEL: $$${EVIL_GATE_CANARY}}}}\n",
        b"services: {mysql: {environment: {LOG_LEVEL: $$$EVIL_GATE_CANARY}}}\n",
    ),
)
def test_compose_source_rejects_hidden_or_ambient_inputs(payload):
    module = _manifest_module()

    with pytest.raises(module.ManifestError):
        module._validate_compose_source(payload)


def test_compose_source_accepts_escaped_container_shell_variable():
    module = _manifest_module()

    module._validate_compose_source(
        b"services: {mysql: {command: ['echo', '$${MYSQL_PASSWORD_FILE}']}}\n"
    )
    module._validate_compose_source(b"services: {mysql: {command: ['echo', '$$$$']}}\n")


def test_compose_source_accepts_all_tracked_base_interpolation_variables():
    module = _manifest_module()

    module._validate_compose_source(
        (ROOT_DIR / "docker" / "docker-compose.yml").read_bytes()
    )


def test_compose_source_rejects_ambient_nvidia_compat_interpolation():
    module = _manifest_module()

    with pytest.raises(module.ManifestError):
        module._validate_compose_source(
            b"services: {train-factory-api: {environment: "
            b"{NVIDIA_DISABLE_REQUIRE: '${NVIDIA_DISABLE_REQUIRE:-1}'}}}\n"
        )


def test_manifest_record_preserves_lexical_path_case(monkeypatch):
    module = _manifest_module()
    monkeypatch.setattr(module.os.path, "abspath", lambda _path: r"E:\Repo\Input.env")
    monkeypatch.setattr(
        module.os.path,
        "normcase",
        lambda _value: pytest.fail("record creation must not rewrite path case"),
    )
    monkeypatch.setattr(
        module, "_read_stable", lambda *_args, **_kwargs: (b"value", (1, 2))
    )

    record, _identity, _payload = module._record(
        Path("ignored"),
        role="images-lock",
        sensitive=False,
    )

    assert record["path"] == r"E:\Repo\Input.env"


def test_manifest_output_cannot_overwrite_input_or_unowned_runtime_file(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    original = secret_env.read_bytes()

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=secret_env,
            root=root,
        )
    assert secret_env.read_bytes() == original

    arbitrary = root / ".runtime" / "cleanup-state.json"
    arbitrary.write_text("private-state-canary", encoding="utf-8")
    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=arbitrary,
            root=root,
        )
    assert arbitrary.read_text(encoding="utf-8") == "private-state-canary"

    disguised = root / ".runtime" / "disguised-manifest.json"
    disguised.write_text('{"schema_version":1}', encoding="utf-8")
    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=disguised,
            root=root,
        )
    assert disguised.read_text(encoding="utf-8") == '{"schema_version":1}'


def test_selection_publish_is_exclusive_and_preserves_concurrent_output(
    tmp_path, monkeypatch
):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    target = root / ".runtime" / "rollback-post.env"
    values = {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "local/web:fixed",
        "API_REVISION": "a" * 40,
        "WEB_REVISION": "a" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "c" * 64,
    }
    original_link = module.os.link

    def create_concurrent_target(source, destination):
        Path(destination).write_text("CONCURRENT-CANARY", encoding="utf-8")
        return original_link(source, destination)

    monkeypatch.setattr(module.os, "link", create_concurrent_target)

    with pytest.raises(module.ManifestError):
        module._publish_selection_environment(target, values, root=root)

    assert target.read_text(encoding="utf-8") == "CONCURRENT-CANARY"


def test_selection_publish_does_not_delete_post_link_replacement(tmp_path, monkeypatch):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    target = root / ".runtime" / "rollback-post.env"
    values = {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "local/web:fixed",
        "API_REVISION": "a" * 40,
        "WEB_REVISION": "a" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "c" * 64,
    }
    original_link = module.os.link

    def replace_linked_target(source, destination):
        original_link(source, destination)
        Path(destination).unlink()
        Path(destination).write_text("REPLACEMENT-CANARY", encoding="utf-8")

    monkeypatch.setattr(module.os, "link", replace_linked_target)

    with pytest.raises(module.ManifestError):
        module._publish_selection_environment(target, values, root=root)

    assert target.read_text(encoding="utf-8") == "REPLACEMENT-CANARY"


def test_selection_publish_rolls_back_link_when_temp_unlink_fails(
    tmp_path, monkeypatch
):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    target = root / ".runtime" / "rollback-post.env"
    values = {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "local/web:fixed",
        "API_REVISION": "a" * 40,
        "WEB_REVISION": "a" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "c" * 64,
    }
    original_unlink = Path.unlink
    failed = False

    def fail_first_temp_unlink(path, *args, **kwargs):
        nonlocal failed
        if not failed and path.name.endswith(".tmp"):
            failed = True
            raise OSError("private unlink canary")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_temp_unlink)

    with pytest.raises(module.ManifestError):
        module._publish_selection_environment(target, values, root=root)

    assert not target.exists()


def test_selection_publish_rejects_post_link_temp_replacement_without_deleting_it(
    tmp_path,
    monkeypatch,
):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    target = root / ".runtime" / "rollback-post.env"
    values = {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "local/web:fixed",
        "API_REVISION": "a" * 40,
        "WEB_REVISION": "a" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "c" * 64,
    }
    original_link = module.os.link
    replacement = None

    def replace_temp_after_link(source, destination):
        nonlocal replacement
        original_link(source, destination)
        replacement = Path(source)
        replacement.unlink()
        replacement.write_text("FOREIGN-TEMP-CANARY", encoding="utf-8")

    monkeypatch.setattr(module.os, "link", replace_temp_after_link)

    with pytest.raises(module.ManifestError):
        module._publish_selection_environment(target, values, root=root)

    assert replacement is not None
    assert replacement.read_text(encoding="utf-8") == "FOREIGN-TEMP-CANARY"
    assert not target.exists()


def test_selection_publish_temp_collision_preserves_foreign_file(
    tmp_path,
    monkeypatch,
):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    target = root / ".runtime" / "rollback-post.env"
    temporary = target.parent / f".{target.name}.{'1' * 16}.tmp"
    temporary.write_text("FOREIGN-TEMP-CANARY", encoding="utf-8")
    monkeypatch.setattr(module.secrets, "token_hex", lambda _size: "1" * 16)
    values = {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "local/web:fixed",
        "API_REVISION": "a" * 40,
        "WEB_REVISION": "a" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "c" * 64,
    }

    with pytest.raises(module.ManifestError):
        module._publish_selection_environment(target, values, root=root)

    assert temporary.read_text(encoding="utf-8") == "FOREIGN-TEMP-CANARY"


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
    ),
)
def test_manifest_loader_rejects_duplicate_keys_and_nonfinite_json(tmp_path, payload):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    output = root / ".runtime" / "malformed-manifest.json"
    output.write_text(payload, encoding="utf-8")

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs(output, root=root)


@pytest.mark.parametrize("value", ([], {}, True))
def test_manifest_structure_rejects_nonstring_rollback_variant(value):
    module = _manifest_module()
    manifest = {
        "schema_version": 1,
        "mode": "rollback-verify",
        "project": "trainfactory-rollback-verify-" + "ab" * 16,
        "gpu_mode": "cpu",
        "secret_mode": "direct",
        "rollback_variant": value,
        "compose_files": [],
        "env_files": [],
        "referenced_files": [],
    }

    with pytest.raises(module.ManifestError, match="^compose manifest is invalid$"):
        module._validate_manifest_structure(manifest)


@pytest.mark.parametrize(
    "invalid_bind",
    ("LOCALHOST", "localhost", "deployment-host.invalid", "::ffff:127.0.0.1"),
)
def test_manifest_base_environment_requires_literal_nonmapped_bind_address(
    invalid_bind,
):
    module = _manifest_module()
    payload = (
        "COMPOSE_PROJECT_NAME=trainfactory\n" f"HOST_BIND_ADDRESS={invalid_bind}\n"
    ).encode()

    with pytest.raises(module.ManifestError):
        module._validate_environment_schema(
            payload,
            role="base-environment",
            project="trainfactory",
            secret_mode="direct",
            mode="production",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", 2), ("sensitive", False)),
)
def test_manifest_verification_rejects_schema_or_sensitivity_tampering(
    tmp_path,
    field,
    value,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    output = root / ".runtime" / "manifest.json"
    manifest = module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=output,
        root=root,
    )
    if field == "schema_version":
        manifest[field] = value
    else:
        manifest["env_files"][0][field] = value
    output.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs(output, root=root)


def test_rollback_manifest_uses_closed_environment_role_order(tmp_path):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n"
        "DEBUG=false\n",
        encoding="utf-8",
    )
    direct = root / ".runtime" / "rollback-direct.env"
    selection = root / ".runtime" / "rollback-pre.env"
    direct.write_text(
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n",
        encoding="utf-8",
    )
    selection.write_text(
        "API_IMAGE=api:fixed\n"
        "WEB_IMAGE=web:fixed\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )

    manifest = module.freeze_manifest(
        mode="rollback-pre",
        project="trainfactory",
        env_files=(
            root / ".env",
            root / "docker" / "images.lock.env",
            direct,
            selection,
        ),
        gpu_mode="raw",
        secret_mode="direct",
        output=root / ".runtime" / "rollback-manifest.json",
        root=root,
    )

    assert [item["role"] for item in manifest["env_files"]] == [
        "base-environment",
        "images-lock",
        "rollback-direct",
        "rollback-pre",
    ]
    assert [item["sensitive"] for item in manifest["env_files"]] == [
        True,
        False,
        True,
        False,
    ]


def test_replace_image_selection_changes_only_last_role_for_rollback_post(
    tmp_path, monkeypatch
):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n"
        "DEBUG=false\n",
        encoding="utf-8",
    )
    direct = root / ".runtime" / "rollback-direct.env"
    direct.write_text(
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n",
        encoding="utf-8",
    )
    captured = root / ".runtime" / "rollback-pre.env"
    captured.write_text(
        "API_IMAGE=old-api:fixed\n"
        "WEB_IMAGE=old-web:fixed\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )
    source = root / ".runtime" / "rollback-pre-compose-manifest.json"
    original = module.freeze_manifest(
        mode="rollback-pre",
        project="trainfactory",
        env_files=(
            root / ".env",
            root / "docker" / "images.lock.env",
            direct,
            captured,
        ),
        gpu_mode="raw",
        secret_mode="direct",
        output=source,
        root=root,
    )
    replacement = root / ".runtime" / "rollback-post.env"
    replacement.write_text(
        "API_IMAGE=compat-api:fixed\n"
        "WEB_IMAGE=old-web:fixed\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'e' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )

    def run(args, **kwargs):
        image = args[-1]
        if image == "compat-api:fixed":
            image_id, revision, compat = (
                "e" * 64,
                "a" * 40,
                "053_validate_lifecycle_schema",
            )
        else:
            image_id, revision, compat = "c" * 64, "d" * 40, ""
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{image_id}|linux|amd64|0.1.0|{revision}|"
            "2026-08-17T00:00:00Z|https://example.invalid/train-factory|"
            f"{compat}\n",
            "",
        )

    output = root / ".runtime" / "rollback-post-compose-manifest.json"
    verified_paths = []

    def verify_inputs(path, **_kwargs):
        verified_paths.append(Path(path))
        return module.verify_manifest_inputs(path, root=root)

    monkeypatch.setattr(
        module,
        "verify_manifest_inputs_only",
        verify_inputs,
    )
    replaced = module.replace_image_selection(
        source=source,
        image_env=replacement,
        mode="rollback-post-migration",
        output=output,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    assert replaced["mode"] == "rollback-post-migration"
    for key in (
        "project",
        "gpu_mode",
        "secret_mode",
        "compose_files",
        "referenced_files",
    ):
        assert replaced[key] == original[key]
    assert replaced["env_files"][:-1] == original["env_files"][:-1]
    assert replaced["env_files"][-1]["role"] == "rollback-post"
    assert replaced["env_files"][-1]["path"] == os.path.abspath(replacement)
    assert module.verify_manifest_inputs(output, root=root) == replaced
    assert verified_paths == [source, source, output]

    output.unlink()
    verify_count = 0

    def fail_final_verification(path, **_kwargs):
        nonlocal verify_count
        verify_count += 1
        if verify_count == 3:
            raise module.ManifestError("compose manifest image verification failed")
        return module.verify_manifest_inputs(path, root=root)

    monkeypatch.setattr(
        module,
        "verify_manifest_inputs_only",
        fail_final_verification,
    )
    with pytest.raises(module.ManifestError):
        module.replace_image_selection(
            source=source,
            image_env=replacement,
            mode="rollback-post-migration",
            output=output,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )
    assert not output.exists()

    output.write_text("CONCURRENT-CANARY", encoding="utf-8")
    with pytest.raises(module.ManifestError):
        module.replace_image_selection(
            source=source,
            image_env=replacement,
            mode="rollback-post-migration",
            output=output,
            root=root,
            run=run,
        )
    assert output.read_text(encoding="utf-8") == "CONCURRENT-CANARY"


def test_replace_web_only_selection_preserves_a_split_api_revision(
    tmp_path, monkeypatch
):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    release_env.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:new\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )
    source = root / ".runtime" / "production-compose-manifest.json"
    original = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=_fake_production_manifest_inputs(root, release_env),
        gpu_mode="compat",
        secret_mode="direct",
        output=source,
        root=root,
    )
    replacement = root / ".runtime" / "rollback-web-only.env"
    replacement_values = {
        "API_IMAGE": "local/api:fixed",
        "WEB_IMAGE": "old-web:fixed",
        "API_REVISION": "a" * 40,
        "WEB_REVISION": "e" * 40,
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "f" * 64,
    }
    replacement.write_text(
        "".join(f"{key}={value}\n" for key, value in replacement_values.items()),
        encoding="utf-8",
    )

    def run(args, **_kwargs):
        image = args[-1]
        if image == "local/api:fixed":
            image_id, revision = "b" * 64, "a" * 40
        elif image == "old-web:fixed":
            image_id, revision = "f" * 64, "e" * 40
        else:
            raise AssertionError("unselected image must not be inspected")
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{image_id}|linux|amd64|0.1.0|{revision}|"
            "2026-08-19T00:00:00Z|https://example.invalid/train-factory|\n",
            "",
        )

    monkeypatch.setattr(
        module,
        "verify_manifest_inputs_only",
        lambda path, **_kwargs: module.verify_manifest_inputs(path, root=root),
    )
    output = root / ".runtime" / "rollback-web-only-compose-manifest.json"

    replaced = module.replace_image_selection(
        source=source,
        image_env=replacement,
        mode="rollback-web-only",
        output=output,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    assert replaced["mode"] == "rollback-web-only"
    assert replaced["env_files"][:-1] == original["env_files"][:-1]
    assert module._selection_values(replaced) == replacement_values
    assert module.verify_manifest_inputs(output, root=root) == replaced


@pytest.mark.parametrize(
    "variant",
    ("raw-old", "release-api-old-web", "compat-api-old-web"),
)
def test_rollback_verify_manifest_uses_base_lock_direct_selection_order(
    tmp_path, monkeypatch, variant
):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n"
        "DEBUG=false\n",
        encoding="utf-8",
    )
    bundle = root / ".runtime" / ("rollback-verify-" + "ab" * 16)
    bundle.mkdir()
    direct = bundle / "rollback-direct.env"
    selection = bundle / f"rollback-verify-{variant}.env"
    direct.write_text(
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n"
        "HOST_BIND_ADDRESS='127.0.0.1'\n"
        "PUBLIC_BASE_URL='http://127.0.0.1:3000'\n"
        "AUTH_COOKIE_SECURE='false'\n"
        "API_PORT='18000'\n"
        "WEB_PORT='3000'\n",
        encoding="utf-8",
    )
    selection.write_text(
        "API_IMAGE=api:fixed\n"
        "WEB_IMAGE=web:fixed\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )

    output = bundle / f"rollback-verify-{variant}-manifest.json"
    wrong_selection = bundle / "rollback-verify.env"
    wrong_selection.write_bytes(selection.read_bytes())
    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="rollback-verify",
            project="trainfactory-rollback-verify-" + "ab" * 16,
            env_files=(
                root / ".env",
                root / "docker" / "images.lock.env",
                direct,
                wrong_selection,
            ),
            gpu_mode="cpu",
            secret_mode="direct",
            rollback_variant=variant,
            output=output,
            root=root,
        )
    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="rollback-verify",
            project="trainfactory-rollback-verify-" + "ab" * 16,
            env_files=(
                root / ".env",
                root / "docker" / "images.lock.env",
                direct,
                selection,
            ),
            gpu_mode="cpu",
            secret_mode="direct",
            rollback_variant=variant,
            output=bundle / "rollback-verify-manifest.json",
            root=root,
        )
    manifest = module.freeze_manifest(
        mode="rollback-verify",
        project="trainfactory-rollback-verify-" + "ab" * 16,
        env_files=(
            root / ".env",
            root / "docker" / "images.lock.env",
            direct,
            selection,
        ),
        gpu_mode="cpu",
        secret_mode="direct",
        rollback_variant=variant,
        output=output,
        root=root,
    )

    assert [item["role"] for item in manifest["env_files"]] == [
        "base-environment",
        "images-lock",
        "rollback-direct",
        "image-selection",
    ]
    assert manifest["rollback_variant"] == variant

    release_module = _release_module()
    images_lock = module._manifest_environment_values(manifest, "images-lock")
    monkeypatch.setattr(
        release_module,
        "_load_resolved_compose",
        lambda *_args, **_kwargs: {
            "services": {
                "mysql": {"image": images_lock["MYSQL_IMAGE"], "build": None},
                "train-factory-api": {"image": "api:fixed", "build": None},
                "train-factory-web": {"image": "web:fixed", "build": None},
            }
        },
    )
    monkeypatch.setattr(
        release_module,
        "_validate_resolved_config",
        lambda *_args, **_kwargs: None,
    )

    def inspect(args, **_kwargs):
        image = args[-1]
        image_id = "b" * 64 if image == "api:fixed" else "c" * 64
        revision = "a" * 40 if image == "api:fixed" else "d" * 40
        compat = (
            "053_validate_lifecycle_schema"
            if variant == "compat-api-old-web" and image == "api:fixed"
            else ""
        )
        return subprocess.CompletedProcess(
            args,
            0,
            f"sha256:{image_id}|linux|amd64|0.1.0|{revision}|"
            "2026-08-17T00:00:00Z|https://example.invalid/train-factory|"
            f"{compat}\n",
            "",
        )

    assert (
        module.verify_manifest_inputs_only(
            output,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=inspect,
        )["rollback_variant"]
        == variant
    )

    def inspect_with_wrong_api_compat(args, **_kwargs):
        completed = inspect(args)
        if args[-1] != "api:fixed":
            return completed
        fields = completed.stdout.rstrip("\n").split("|")
        fields[-1] = (
            "" if variant == "compat-api-old-web" else "053_validate_lifecycle_schema"
        )
        return subprocess.CompletedProcess(args, 0, "|".join(fields) + "\n", "")

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs_only(
            output,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=inspect_with_wrong_api_compat,
        )

    def inspect_with_web_compat(args, **_kwargs):
        completed = inspect(args)
        if args[-1] != "web:fixed":
            return completed
        fields = completed.stdout.rstrip("\n").split("|")
        fields[-1] = "053_validate_lifecycle_schema"
        return subprocess.CompletedProcess(args, 0, "|".join(fields) + "\n", "")

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs_only(
            output,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=inspect_with_web_compat,
        )

    for malformed in ([], {}, True):
        tampered = dict(manifest)
        tampered["rollback_variant"] = malformed
        output.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(
            module.ManifestError,
            match="^compose manifest input verification failed$",
        ):
            module.verify_manifest_inputs(output, root=root)
    output.write_text(json.dumps(manifest), encoding="utf-8")


def test_rollback_verify_manifest_requires_explicit_closed_variant(tmp_path):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n",
        encoding="utf-8",
    )
    bundle = root / ".runtime" / ("rollback-verify-" + "ab" * 16)
    bundle.mkdir()
    direct = bundle / "rollback-direct.env"
    direct.write_text(
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n"
        "HOST_BIND_ADDRESS='127.0.0.1'\n"
        "PUBLIC_BASE_URL='http://127.0.0.1:3000'\n"
        "AUTH_COOKIE_SECURE='false'\n"
        "API_PORT='18000'\n"
        "WEB_PORT='3000'\n",
        encoding="utf-8",
    )
    selection = bundle / "rollback-verify.env"
    selection.write_text(
        "API_IMAGE=api:fixed\n"
        "WEB_IMAGE=web:fixed\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="rollback-verify",
            project="trainfactory-rollback-verify-" + "ab" * 16,
            env_files=(
                root / ".env",
                root / "docker" / "images.lock.env",
                direct,
                selection,
            ),
            gpu_mode="cpu",
            secret_mode="direct",
            output=root / ".runtime" / "rollback-verify-manifest.json",
            root=root,
        )


def test_rollback_selection_rejects_single_shared_revision_schema(tmp_path):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n",
        encoding="utf-8",
    )
    direct = root / ".runtime" / "rollback-direct.env"
    direct.write_text(
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='trainfactory_app'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://trainfactory_app:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n",
        encoding="utf-8",
    )
    selection = root / ".runtime" / "rollback-pre.env"
    selection.write_text(
        "API_IMAGE=api:fixed\n"
        "WEB_IMAGE=web:fixed\n"
        f"RELEASE_REVISION={'a' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="rollback-pre",
            project="trainfactory",
            env_files=(
                root / ".env",
                root / "docker" / "images.lock.env",
                direct,
                selection,
            ),
            gpu_mode="raw",
            secret_mode="direct",
            output=root / ".runtime" / "rollback-manifest.json",
            root=root,
        )


def test_manifest_loader_rejects_symlinked_ancestor(tmp_path):
    module = _manifest_module()
    root, _secret_env, _release_env = _fake_manifest_root(tmp_path)
    actual = root / ".runtime" / "actual"
    actual.mkdir()
    manifest_path = actual / "manifest.json"
    manifest_path.write_text('{"schema_version":1}', encoding="utf-8")
    link = root / ".runtime" / "linked"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(module.ManifestError):
        module.verify_manifest_inputs(link / "manifest.json", root=root)


@pytest.mark.parametrize(
    ("mode", "gpu_mode", "secret_mode"),
    (
        ("production", "cpu", "files"),
        ("verify", "raw", "direct"),
        ("rollback-web-only", "cpu", "files"),
    ),
)
def test_manifest_rejects_mode_combinations_outside_closed_policy(
    tmp_path,
    mode,
    gpu_mode,
    secret_mode,
):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n"
        "DEBUG=false\n",
        encoding="utf-8",
    )
    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode=mode,
            project=PROJECT if mode == "verify" else "trainfactory",
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode=gpu_mode,
            secret_mode=secret_mode,
            output=root / ".runtime" / "manifest.json",
            root=root,
        )


def test_verify_manifest_rejects_secret_bundle_from_another_run(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    other_run = root / ".runtime" / ("verify-" + "cd" * 16)
    other_run.mkdir()
    copied = other_run / "compose-secrets.env"
    copied.write_bytes(secret_env.read_bytes())

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(copied, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "manifest.json",
            root=root,
        )


def test_verify_manifest_rejects_direct_value_or_wrong_secret_path(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    original = secret_env.read_text(encoding="utf-8")
    secret_env.write_text(
        original + "MYSQL_URL=mysql+pymysql://private-canary\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "manifest.json",
            root=root,
        )


def test_wrapper_rejects_referenced_secret_drift_before_docker_call(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    (secret_env.parent / "mysql_url").write_text(
        "private-drift-canary\n",
        encoding="utf-8",
    )
    calls = []

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_manifest(
            manifest_path,
            ("up", "-d", "--wait", "mysql"),
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


def test_wrapper_rejects_database_bootstrap_drift_before_docker_call(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    (root / "docker" / "init.sql").write_text(
        "USE train_factory;\nPRIVATE_DRIFT_CANARY\n",
        encoding="utf-8",
    )
    calls = []

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_manifest(
            manifest_path,
            ("up", "-d", "--wait", "mysql"),
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


def test_manifest_rejects_unhardened_materialized_bundle(tmp_path, monkeypatch):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    bad_path = secret_env.parent / "mysql_url"
    monkeypatch.setattr(
        module,
        "_verify_hardened_path",
        lambda path: path != bad_path,
    )

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "manifest.json",
            root=root,
        )


def test_manifest_rejects_compose_control_key_in_frozen_environment(tmp_path):
    module = _manifest_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    release_env.write_text(
        release_env.read_text(encoding="utf-8") + "COMPOSE_REMOVE_ORPHANS=1\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ManifestError) as exc_info:
        module.freeze_manifest(
            mode="verify",
            project=PROJECT,
            env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "manifest.json",
            root=root,
        )

    assert "COMPOSE_REMOVE_ORPHANS" not in repr(exc_info.value)


@pytest.mark.parametrize(
    "value",
    (
        "'DEBUG'",
        '"https://models.example.invalid"',
        "INFO # host-side comment",
        "$HTTPS_PROXY",
        "${HTTPS_PROXY}",
        " INFO",
        "INFO ",
        r"INFO\DEBUG",
    ),
)
def test_base_environment_rejects_noncanonical_dotenv_values(value):
    module = _manifest_module()

    with pytest.raises(module.ManifestError):
        module._validate_environment_schema(
            f"COMPOSE_PROJECT_NAME=trainfactory\nLOG_LEVEL={value}\n".encode(),
            role="base-environment",
            project="trainfactory",
            secret_mode="direct",
            mode="production",
        )


def test_production_direct_rejects_user_and_admin_identity_drift(tmp_path):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=base_user\n"
        "DEFAULT_ADMIN_USERNAME=base_admin\n",
        encoding="utf-8",
    )
    direct = root / ".runtime" / "production-direct.env"
    direct.write_text(
        "MYSQL_ROOT_PASSWORD='private-root-123456'\n"
        "MYSQL_APP_USER='other_user'\n"
        "MYSQL_APP_PASSWORD='private-app-123456'\n"
        "MYSQL_PASSWORD='private-app-123456'\n"
        "MYSQL_URL='mysql+pymysql://other_user:private-app-123456@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='private-jwt-123456-private-jwt-123456'\n"
        "DEFAULT_ADMIN_USERNAME='other_admin'\n"
        "DEFAULT_ADMIN_PASSWORD='private-admin-123456'\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="production",
            project="trainfactory",
            env_files=(
                root / ".env",
                direct,
                root / "docker" / "images.lock.env",
                release_env,
            ),
            gpu_mode="raw",
            secret_mode="direct",
            output=root / ".runtime" / "production-manifest.json",
            root=root,
        )


def test_production_direct_accepts_application_credential_boundaries(tmp_path):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=custom_user\n"
        "DEFAULT_ADMIN_USERNAME=custom-admin\n",
        encoding="utf-8",
    )
    direct = root / ".runtime" / "production-direct.env"
    direct.write_text(
        "MYSQL_ROOT_PASSWORD='r'\n"
        "MYSQL_APP_USER='custom_user'\n"
        "MYSQL_APP_PASSWORD='p'\n"
        "MYSQL_PASSWORD='p'\n"
        "MYSQL_URL='mysql+pymysql://custom_user:p@mysql:3306/train_factory'\n"
        "JWT_SECRET_KEY='密密密密密密密密密密密密密密密密'\n"
        "DEFAULT_ADMIN_USERNAME='custom-admin'\n"
        "DEFAULT_ADMIN_PASSWORD='1234567890'\n",
        encoding="utf-8",
    )

    manifest = module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=(
            root / ".env",
            direct,
            root / "docker" / "images.lock.env",
            release_env,
        ),
        gpu_mode="raw",
        secret_mode="direct",
        output=root / ".runtime" / "production-manifest.json",
        root=root,
    )

    assert manifest["mode"] == "production"


def test_production_files_rejects_url_user_different_from_base(tmp_path):
    module = _manifest_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=base_user\n"
        "DEFAULT_ADMIN_USERNAME=base_admin\n",
        encoding="utf-8",
    )
    secret_dir = root / ".runtime" / "production-secrets"
    secret_dir.mkdir()
    values = {
        "mysql_root_password": "root-password-1234",
        "mysql_app_password": "app-password-12345",
        "mysql_url": "mysql+pymysql://other_user:app-password-12345@mysql:3306/train_factory",
        "jwt_secret_key": "jwt-secret-1234567890-jwt-secret-1234567890",
        "default_admin_password": "admin-password-1234",
    }
    for name, value in values.items():
        (secret_dir / name).write_text(value, encoding="utf-8")
    files = root / ".runtime" / "production-files.env"
    files.write_text(
        "".join(
            f"{key}='{secret_dir / filename}'\n"
            for key, filename in {
                "MYSQL_ROOT_PASSWORD_SECRET_PATH": "mysql_root_password",
                "MYSQL_APP_PASSWORD_SECRET_PATH": "mysql_app_password",
                "MYSQL_URL_SECRET_PATH": "mysql_url",
                "JWT_SECRET_KEY_SECRET_PATH": "jwt_secret_key",
                "DEFAULT_ADMIN_PASSWORD_SECRET_PATH": "default_admin_password",
            }.items()
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.ManifestError):
        module.freeze_manifest(
            mode="production",
            project="trainfactory",
            env_files=(
                root / ".env",
                files,
                root / "docker" / "images.lock.env",
                release_env,
            ),
            gpu_mode="raw",
            secret_mode="files",
            output=root / ".runtime" / "production-manifest.json",
            root=root,
        )


def _release_module():
    from scripts import compose_release

    return compose_release


@pytest.mark.parametrize(
    "script_name",
    ("compose_manifest.py", "compose_release.py"),
)
def test_compose_cli_direct_script_bootstrap_is_runnable(script_name):
    completed = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / script_name), "--help"],
        cwd=ROOT_DIR,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        },
    )

    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize(
    "script_name",
    (
        "compose_manifest.py",
        "compose_release.py",
        "materialize_compose_secrets.py",
        "run_mysql_migration_tests.py",
        "verify_deployment.py",
    ),
)
def test_security_cli_isolated_mode_ignores_ambient_sitecustomize(
    tmp_path, script_name
):
    canary = "AMBIENT-SITECUSTOMIZE-CANARY"
    (tmp_path / "sitecustomize.py").write_text(
        f"print({canary!r})\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            os.fspath(ROOT_DIR / "scripts" / script_name),
            "--help",
        ],
        cwd=ROOT_DIR,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "PYTHONPATH": os.fspath(tmp_path),
        },
    )

    assert completed.returncode == 0
    assert canary not in completed.stdout
    assert canary not in completed.stderr
    assert "Traceback" not in completed.stderr


def test_freeze_cli_routes_explicit_rollback_variant(monkeypatch):
    module = _manifest_module()
    observed = []
    monkeypatch.setattr(
        module,
        "freeze_manifest",
        lambda **kwargs: observed.append(kwargs) or {},
    )

    result = module.main(
        [
            "freeze",
            "--mode",
            "rollback-verify",
            "--project",
            "trainfactory-rollback-verify-" + "ab" * 16,
            "--env-file",
            "base.env",
            "--gpu-mode",
            "cpu",
            "--secret-mode",
            "direct",
            "--rollback-variant",
            "compat-api-old-web",
            "--output",
            "rollback-verify-compat-api-old-web-manifest.json",
        ]
    )

    assert result == 0
    assert observed[0]["rollback_variant"] == "compat-api-old-web"


def test_freeze_cli_rejects_missing_rollback_variant_with_fixed_error(capsys):
    module = _manifest_module()

    result = module.main(
        [
            "freeze",
            "--mode",
            "rollback-verify",
            "--project",
            "trainfactory-rollback-verify-" + "ab" * 16,
            "--env-file",
            "PRIVATE-PATH-CANARY",
            "--gpu-mode",
            "cpu",
            "--secret-mode",
            "direct",
            "--output",
            "PRIVATE-OUTPUT-CANARY",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "compose manifest command failed\n"
    assert "PRIVATE" not in captured.err


@pytest.mark.parametrize("slash_count", (0, 2, 4, 6))
def test_single_quoted_decoder_rejects_even_escape_run_before_apostrophe(
    slash_count,
):
    module = _manifest_module()
    payload = "'prefix" + ("\\" * slash_count) + "'suffix'"

    with pytest.raises(module.ManifestError):
        module._decode_single_quoted_path(payload)


@pytest.mark.parametrize(
    "tail",
    (
        ("-f", "private.yml", "up"),
        ("up", "-d", "--build", "mysql"),
        ("up", "-d", "--pull=always", "mysql"),
        ("up", "-d", "--scale", "mysql=2", "mysql"),
        ("up", "-d", "--remove-orphans", "mysql"),
        ("run", "-v", "private:/host", "train-factory-api"),
        ("run", "-e", "PRIVATE_CANARY=value", "train-factory-api"),
        ("down", "--volumes"),
        ("down", "-v"),
        ("down", "--rmi=all"),
        ("rm", "-v", "mysql"),
    ),
)
def test_release_wrapper_rejects_tail_injection_before_any_docker_call(
    tmp_path,
    tail,
):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project="trainfactory-verify-" + "ab" * 16,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_manifest(
            manifest_path,
            tail,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


def test_release_wrapper_accepts_exact_staged_replacement_flags():
    release_module = _release_module()
    tail = (
        "up",
        "-d",
        "--no-deps",
        "--force-recreate",
        "--wait",
        "--wait-timeout",
        "600",
        "train-factory-api",
    )

    approved, timeout = release_module._validate_up_tail(tail)
    release_module._validate_staged_replacement("production", approved)

    assert approved == tail
    assert timeout == 660


_SHARED_PRODUCTION_SELECTION_KEYS = frozenset(
    {
        "API_IMAGE",
        "WEB_IMAGE",
        "RELEASE_REVISION",
        "API_IMAGE_ID",
        "WEB_IMAGE_ID",
    }
)
_SPLIT_PRODUCTION_SELECTION_KEYS = frozenset(
    {
        "API_IMAGE",
        "WEB_IMAGE",
        "API_REVISION",
        "WEB_REVISION",
        "API_IMAGE_ID",
        "WEB_IMAGE_ID",
    }
)
_EXACT_WEB_REPLACEMENT_TAIL = (
    "up",
    "-d",
    "--no-deps",
    "--force-recreate",
    "--wait",
    "--wait-timeout",
    "600",
    "train-factory-web",
)


def test_split_production_accepts_only_exact_web_replacement_tail():
    release_module = _release_module()

    approved, timeout = release_module._validate_up_tail(_EXACT_WEB_REPLACEMENT_TAIL)
    release_module._validate_staged_replacement(
        "production",
        approved,
        selection_keys=_SPLIT_PRODUCTION_SELECTION_KEYS,
    )

    assert approved == _EXACT_WEB_REPLACEMENT_TAIL
    assert timeout == 660


def test_web_only_rollback_accepts_only_exact_web_replacement_tail():
    release_module = _release_module()

    approved, timeout = release_module._validate_up_tail(_EXACT_WEB_REPLACEMENT_TAIL)
    release_module._validate_staged_replacement(
        "rollback-web-only",
        approved,
        selection_keys=_SPLIT_PRODUCTION_SELECTION_KEYS,
    )

    assert approved == _EXACT_WEB_REPLACEMENT_TAIL
    assert timeout == 660


@pytest.mark.parametrize(
    "selection_keys",
    (None, _SHARED_PRODUCTION_SELECTION_KEYS),
)
def test_web_only_rollback_requires_split_selection_schema(selection_keys):
    release_module = _release_module()
    approved, _timeout = release_module._validate_up_tail(_EXACT_WEB_REPLACEMENT_TAIL)

    with pytest.raises(
        release_module.ReleaseComposeError,
        match="^release Compose command is invalid$",
    ):
        release_module._validate_staged_replacement(
            "rollback-web-only",
            approved,
            selection_keys=selection_keys,
        )


@pytest.mark.parametrize(
    "tail",
    (
        (
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "600",
            "mysql",
            "train-factory-api",
            "train-factory-web",
        ),
        (
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            "600",
            "train-factory-api",
        ),
    ),
)
def test_web_only_rollback_rejects_nonexact_activation_before_run(
    tmp_path, monkeypatch, tail
):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    source = root / ".runtime" / "production-compose-manifest.json"
    manifest_module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=_fake_production_manifest_inputs(root, release_env),
        gpu_mode="compat",
        secret_mode="direct",
        output=source,
        root=root,
    )
    replacement = root / ".runtime" / "rollback-web-only.env"
    replacement.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:old\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        manifest_module,
        "verify_manifest_inputs_only",
        lambda path, **_kwargs: manifest_module.verify_manifest_inputs(path, root=root),
    )
    manifest_path = root / ".runtime" / "rollback-web-only-compose-manifest.json"
    manifest_module.replace_image_selection(
        source=source,
        image_env=replacement,
        mode="rollback-web-only",
        output=manifest_path,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=lambda args, **_kwargs: subprocess.CompletedProcess(
            args,
            0,
            (
                f"sha256:{'b' * 64}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n"
                if args[-1] == "local/api:fixed"
                else f"sha256:{'c' * 64}|linux|amd64|0.1.0|{'d' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n"
            ),
            "",
        ),
    )
    calls = []

    with pytest.raises(
        release_module.ReleaseComposeError,
        match="^release Compose command is invalid$",
    ):
        release_module.execute_manifest(
            manifest_path,
            tail,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


@pytest.mark.parametrize(
    "tail",
    (
        (
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "600",
            "mysql",
            "train-factory-api",
            "train-factory-web",
        ),
        (
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            "600",
            "train-factory-api",
        ),
        ("up", "-d", "--wait", "train-factory-web"),
    ),
)
def test_split_production_rejects_nonexact_web_activation_before_run(
    tmp_path,
    tail,
):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    release_env.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:new\n"
        f"API_REVISION={'a' * 40}\n"
        f"WEB_REVISION={'d' * 40}\n"
        f"API_IMAGE_ID=sha256:{'b' * 64}\n"
        f"WEB_IMAGE_ID=sha256:{'c' * 64}\n",
        encoding="utf-8",
    )
    manifest_path = root / ".runtime" / "production-compose-manifest.json"
    manifest_module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=_fake_production_manifest_inputs(root, release_env),
        gpu_mode="compat",
        secret_mode="direct",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 1, "", "")

    with pytest.raises(
        release_module.ReleaseComposeError,
        match="^release Compose command is invalid$",
    ):
        release_module.execute_manifest(
            manifest_path,
            tail,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )

    assert calls == []


@pytest.mark.parametrize(
    "tail",
    (
        (
            "up",
            "-d",
            "--wait",
            "mysql",
            "train-factory-api",
            "train-factory-web",
        ),
        (
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            "600",
            "train-factory-api",
        ),
        _EXACT_WEB_REPLACEMENT_TAIL,
    ),
)
def test_shared_revision_production_keeps_existing_activation_surface(tail):
    release_module = _release_module()

    approved, _timeout = release_module._validate_up_tail(tail)
    release_module._validate_staged_replacement(
        "production",
        approved,
        selection_keys=_SHARED_PRODUCTION_SELECTION_KEYS,
    )


@pytest.mark.parametrize(
    "tail",
    (
        ("up", "-d", "--wait", "train-factory-api"),
        (
            "up",
            "-d",
            "--no-deps",
            "--wait",
            "train-factory-api",
        ),
        (
            "up",
            "-d",
            "--force-recreate",
            "--wait",
            "train-factory-web",
        ),
    ),
)
def test_production_single_service_replacement_requires_dependency_isolation(tail):
    release_module = _release_module()

    approved, _timeout = release_module._validate_up_tail(tail)

    with pytest.raises(
        release_module.ReleaseComposeError,
        match="^release Compose command is invalid$",
    ):
        release_module._validate_staged_replacement("production", approved)


def test_release_wrapper_uses_frozen_order_and_sanitized_environment(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    project = "trainfactory-verify-" + "ab" * 16
    manifest_module.freeze_manifest(
        mode="verify",
        project=project,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="compat",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []
    canary = "private-wrapper-canary"

    def run(args, **kwargs):
        calls.append((list(args), kwargs))
        if args[:2] == ["docker", "container"] or args[:2] == ["docker", "network"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["docker", "volume"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["docker", "network"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["docker", "container"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    _minimal_resolved_verify_config(
                        project,
                        root=root,
                        gpu_mode="compat",
                    )
                ),
                "",
            )
        if args[:3] == ["docker", "image", "inspect"]:
            image_id = "b" * 64 if args[-1] == "local/api:fixed" else "c" * 64
            return subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    release_module.execute_manifest(
        manifest_path,
        ("up", "-d", "--wait", "mysql", "train-factory-api", "train-factory-web"),
        root=root,
        base_environment={
            "PATH": os.environ.get("PATH", ""),
            "COMPOSE_FUTURE_CANARY": canary,
            "API_IMAGE": canary,
            "MYSQL_URL_FILE": canary,
            "JWT_SECRET_KEY": canary,
            "DOCKER_HOST": "npipe:////./pipe/docker_engine",
            "docker_context": "release-context",
            "DOCKER_TLS_VERIFY": "1",
            "docker_cert_path": "C:/safe-cert",
            "DOCKER_CONFIG": "C:/safe-docker-config",
            "http_proxy": "http://127.0.0.1:8123",
            "HTTPS_PROXY": "http://127.0.0.1:8124",
            "no_proxy": "127.0.0.1,localhost",
        },
        run=run,
    )

    assert len(calls) == 11
    assert calls[0][0][-3:] == ["config", "--format", "json"]
    assert [call[0][1:3] for call in calls[1:4]] == [
        ["container", "ls"],
        ["network", "ls"],
        ["volume", "ls"],
    ]
    argv, kwargs = calls[-1]
    assert argv[:4] == [
        "docker",
        "compose",
        "--project-directory",
        os.fspath(root / "docker"),
    ]
    assert argv[4:] == [
        "-f",
        "-",
        "-p",
        project,
        "up",
        "-d",
        "--wait",
        "mysql",
        "train-factory-api",
        "train-factory-web",
    ]
    expected_resolved = _minimal_resolved_verify_config(
        project,
        root=root,
        gpu_mode="compat",
    )
    expected_resolved["services"]["train-factory-api"]["image"] = "sha256:" + "b" * 64
    expected_resolved["services"]["train-factory-web"]["image"] = "sha256:" + "c" * 64
    assert kwargs["input"] == (
        json.dumps(expected_resolved, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["env"]["COMPOSE_DISABLE_ENV_FILE"] == "1"
    assert kwargs["env"]["DOCKER_HOST"] == "npipe:////./pipe/docker_engine"
    assert kwargs["env"]["DOCKER_CONTEXT"] == "release-context"
    assert kwargs["env"]["DOCKER_TLS_VERIFY"] == "1"
    assert kwargs["env"]["DOCKER_CERT_PATH"] == "C:/safe-cert"
    assert kwargs["env"]["DOCKER_CONFIG"] == "C:/safe-docker-config"
    assert kwargs["env"]["HTTP_PROXY"] == "http://127.0.0.1:8123"
    assert kwargs["env"]["HTTPS_PROXY"] == "http://127.0.0.1:8124"
    assert kwargs["env"]["NO_PROXY"] == "127.0.0.1,localhost"
    assert canary not in "\n".join(kwargs["env"].values())


def test_release_wrapper_rejects_existing_exact_configured_volume_before_up(
    tmp_path,
):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        if args[:3] == ["docker", "volume", "ls"] and "--filter" not in args:
            return subprocess.CompletedProcess(
                args,
                0,
                f"{PROJECT}_mysql_data\n",
                "",
            )
        if args[:3] == ["docker", "image", "inspect"]:
            image_id = "b" * 64 if args[-1] == "local/api:fixed" else "c" * 64
            return subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_manifest(
            manifest_path,
            ("up", "-d", "--wait", "mysql"),
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )

    assert not any("up" in call for call in calls)


@pytest.mark.parametrize(
    ("resource", "occupied_name"),
    (
        ("network", f"{PROJECT}_default"),
        ("container", f"{PROJECT}-mysql-1"),
    ),
)
def test_release_wrapper_rejects_wrong_label_exact_runtime_name_before_up(
    tmp_path,
    resource,
    occupied_name,
):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        if args[:3] == ["docker", resource, "ls"] and "--filter" not in args:
            return subprocess.CompletedProcess(args, 0, occupied_name + "\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_manifest(
            manifest_path,
            ("up", "-d", "--wait", "mysql"),
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )

    assert not any("up" in call for call in calls)


def test_release_wrapper_rejects_existing_random_project_resource_before_up(
    tmp_path,
):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[:3] == ["docker", "network", "ls"]:
            return subprocess.CompletedProcess(args, 0, "stale-network\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_manifest(
            manifest_path,
            ("up", "-d", "--wait", "mysql"),
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )

    assert not any("up" in call for call in calls)


def test_release_wrapper_rejects_case_colliding_transport_environment(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_manifest(
            manifest_path,
            ("up", "-d", "--wait", "mysql"),
            root=root,
            base_environment={
                "HTTP_PROXY": "http://127.0.0.1:1",
                "http_proxy": "http://127.0.0.1:2",
            },
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


def test_release_wrapper_process_timeout_exceeds_compose_wait_timeout(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append((list(args), kwargs))
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        if args[:3] == ["docker", "image", "inspect"]:
            image_id = "b" * 64 if args[-1] == "local/api:fixed" else "c" * 64
            return subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    release_module.execute_manifest(
        manifest_path,
        ("up", "-d", "--wait", "--wait-timeout", "601", "mysql"),
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    up_call = next((argv, kwargs) for argv, kwargs in calls if "up" in argv)
    assert up_call[1]["timeout"] > 601
    assert up_call[1]["timeout"] <= 661


def test_release_wrapper_rechecks_mutation_authority_immediately_before_up(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        if args[:3] == ["docker", "image", "inspect"]:
            image_id = "b" * 64 if args[-1] == "local/api:fixed" else "c" * 64
            return subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    class AuthorityLost(RuntimeError):
        pass

    guard_calls = []

    def reject_mutation():
        guard_calls.append(len(calls))
        raise AuthorityLost

    with pytest.raises(AuthorityLost):
        release_module.execute_manifest(
            manifest_path,
            ("up", "-d", "--wait", "mysql"),
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
            mutation_guard=reject_mutation,
        )

    assert guard_calls == [len(calls)]
    assert calls
    assert not any("up" in call for call in calls)


@pytest.mark.parametrize(
    ("mode", "selection_schema", "tail"),
    (
        (
            "production",
            "shared",
            ("up", "-d", "--wait", "mysql", "train-factory-api", "train-factory-web"),
        ),
        ("production", "split", _EXACT_WEB_REPLACEMENT_TAIL),
        (
            "verify",
            "shared",
            ("up", "-d", "--wait", "mysql", "train-factory-api", "train-factory-web"),
        ),
        (
            "ci",
            "shared",
            ("up", "-d", "--wait", "mysql", "train-factory-api", "train-factory-web"),
        ),
        (
            "rollback-verify",
            "split",
            ("up", "-d", "--wait", "mysql", "train-factory-api", "train-factory-web"),
        ),
        (
            "rollback-pre",
            "split",
            ("up", "-d", "--wait", "mysql", "train-factory-api", "train-factory-web"),
        ),
        (
            "rollback-post-migration",
            "split",
            ("up", "-d", "--wait", "mysql", "train-factory-api", "train-factory-web"),
        ),
        ("rollback-web-only", "split", _EXACT_WEB_REPLACEMENT_TAIL),
    ),
)
def test_release_wrapper_pins_selected_image_ids_in_every_manifest_mode(
    tmp_path, monkeypatch, mode, selection_schema, tail
):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root = tmp_path / "repo"
    (root / "docker").mkdir(parents=True)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest = {"mode": mode, "project": "trainfactory-test"}
    selection = {
        "API_IMAGE": "local/api:tag",
        "WEB_IMAGE": "local/web:tag",
        "API_IMAGE_ID": "sha256:" + "b" * 64,
        "WEB_IMAGE_ID": "sha256:" + "c" * 64,
    }
    if selection_schema == "shared":
        selection["RELEASE_REVISION"] = "a" * 40
    else:
        selection["API_REVISION"] = "a" * 40
        selection["WEB_REVISION"] = "d" * 40
    resolved = {
        "services": {
            "mysql": {"image": "mysql:fixed"},
            "train-factory-api": {"image": selection["API_IMAGE"]},
            "train-factory-web": {"image": selection["WEB_IMAGE"]},
        }
    }
    original = copy.deepcopy(resolved)
    calls = []

    monkeypatch.setattr(
        release_module, "verify_manifest_inputs", lambda *_args, **_kwargs: manifest
    )
    monkeypatch.setattr(release_module, "_selection_values", lambda _value: selection)
    monkeypatch.setattr(
        release_module,
        "_load_resolved_compose",
        lambda *_args, **_kwargs: resolved,
    )
    monkeypatch.setattr(
        release_module, "_validate_resolved_config", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        release_module,
        "_require_empty_project_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        release_module, "_expected_volume_names", lambda *_args, **_kwargs: frozenset()
    )
    monkeypatch.setattr(
        release_module, "_expected_network_names", lambda *_args, **_kwargs: frozenset()
    )
    monkeypatch.setattr(
        manifest_module,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: manifest,
    )

    def run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    release_module.execute_manifest(
        manifest_path,
        tail,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    up_calls = [(argv, kwargs) for argv, kwargs in calls if "up" in argv]
    assert len(up_calls) == 1
    executed = json.loads(up_calls[0][1]["input"])
    assert executed["services"]["mysql"] == original["services"]["mysql"]
    assert (
        executed["services"]["train-factory-api"]["image"] == selection["API_IMAGE_ID"]
    )
    assert (
        executed["services"]["train-factory-web"]["image"] == selection["WEB_IMAGE_ID"]
    )
    assert resolved == original


def test_release_wrapper_executes_resolved_config_with_selected_images_pinned_by_id(
    tmp_path,
):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest = manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    resolved = _minimal_resolved_verify_config(root=root)
    source_paths = {
        str(item["path"])
        for item in (*manifest["compose_files"], *manifest["env_files"])
    }
    up_calls = []

    def run(args, **kwargs):
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(resolved), "")
        if args[:3] == ["docker", "image", "inspect"]:
            image_id = "b" * 64 if args[-1] == "local/api:fixed" else "c" * 64
            return subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{image_id}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        if "up" in args:
            for source in source_paths:
                Path(source).write_text("services: {foreign: {privileged: true}}\n")
            up_calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    release_module.execute_manifest(
        manifest_path,
        ("up", "-d", "--wait", "mysql"),
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    assert len(up_calls) == 1
    argv, kwargs = up_calls[0]
    assert "--env-file" not in argv
    assert all(source not in argv for source in source_paths)
    assert argv[argv.index("-f") + 1] == "-"
    executed = json.loads(kwargs["input"])
    expected = copy.deepcopy(resolved)
    expected["services"]["train-factory-api"]["image"] = "sha256:" + "b" * 64
    expected["services"]["train-factory-web"]["image"] = "sha256:" + "c" * 64
    assert executed == expected
    assert resolved["services"]["train-factory-api"]["image"] == "local/api:fixed"
    assert resolved["services"]["train-factory-web"]["image"] == "local/web:fixed"


def test_release_wrapper_rejects_retagged_selection_before_up(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(_minimal_resolved_verify_config(root=root)),
                "",
            )
        if args[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                args,
                0,
                f"sha256:{'f' * 64}|linux|amd64|0.1.0|{'a' * 40}|"
                "2026-08-17T00:00:00Z|https://example.invalid/train-factory|\n",
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_manifest(
            manifest_path,
            ("up", "-d", "--wait", "mysql"),
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )

    assert not any("up" in call for call in calls)


def test_clean_environment_derives_explicit_config_for_named_context(tmp_path):
    release_module = _release_module()
    docker_config = tmp_path / ".docker"
    docker_config.mkdir()

    clean = release_module._clean_environment(
        {
            "PATH": os.environ.get("PATH", ""),
            "DOCKER_CONTEXT": "desktop-linux",
            "USERPROFILE": os.fspath(tmp_path),
        }
    )

    assert clean["DOCKER_CONTEXT"] == "desktop-linux"
    assert clean["DOCKER_CONFIG"] == os.fspath(docker_config.resolve())


def test_safe_down_volumes_rejects_production_manifest_before_docker(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, _secret_env, release_env = _fake_manifest_root(tmp_path)
    (root / ".env").write_text(
        "COMPOSE_PROJECT_NAME=trainfactory\n"
        "MYSQL_APP_USER=trainfactory_app\n"
        "DEFAULT_ADMIN_USERNAME=admin\n"
        "DEBUG=false\n",
        encoding="utf-8",
    )
    production_files = root / ".runtime" / "production-files.env"
    production_secret_dir = root / ".runtime" / "production-secrets"
    production_secret_dir.mkdir()
    production_values = {
        "mysql_root_password": "d" * 64,
        "mysql_app_password": "e" * 64,
        "mysql_url": (
            "mysql+pymysql://trainfactory_app:" + "e" * 64 + "@mysql:3306/train_factory"
        ),
        "jwt_secret_key": "f" * 64,
        "default_admin_password": "production_admin_123456",
    }
    for filename, value in production_values.items():
        (production_secret_dir / filename).write_text(value, encoding="utf-8")
    production_files.write_text(
        "".join(
            f"{key}='{production_secret_dir / filename}'\n"
            for key, filename in {
                "MYSQL_ROOT_PASSWORD_SECRET_PATH": "mysql_root_password",
                "MYSQL_APP_PASSWORD_SECRET_PATH": "mysql_app_password",
                "MYSQL_URL_SECRET_PATH": "mysql_url",
                "JWT_SECRET_KEY_SECRET_PATH": "jwt_secret_key",
                "DEFAULT_ADMIN_PASSWORD_SECRET_PATH": "default_admin_password",
            }.items()
        ),
        encoding="utf-8",
    )
    manifest_path = root / ".runtime" / "production-manifest.json"
    manifest_module.freeze_manifest(
        mode="production",
        project="trainfactory",
        env_files=(
            root / ".env",
            production_files,
            root / "docker" / "images.lock.env",
            release_env,
        ),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_safe_down_volumes(
            manifest_path,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


def test_safe_down_volumes_inspects_each_label_before_compose_down(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []
    volume_list_calls = 0

    def run(args, **kwargs):
        nonlocal volume_list_calls
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            compose = {
                "services": {
                    "mysql": {
                        "volumes": [
                            {"type": "volume", "source": "train_cache"},
                        ]
                    },
                    "train-factory-api": {
                        "volumes": [
                            {"type": "volume", "source": "verify_data"},
                        ]
                    },
                },
                "volumes": {
                    "train_cache": {"name": f"{PROJECT}_train_cache"},
                    "verify_data": {"name": f"{PROJECT}_verify_data"},
                },
                "networks": {"default": {"name": f"{PROJECT}_default"}},
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(compose), "")
        if args[:3] == ["docker", "volume", "ls"] and "--filter" not in args:
            volume_list_calls += 1
            output = (
                f"{PROJECT}_verify_data\n{PROJECT}_train_cache\n"
                if volume_list_calls == 1
                else ""
            )
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[:3] == ["docker", "volume", "ls"] and "--filter" in args:
            output = (
                f"{PROJECT}_verify_data\n{PROJECT}_train_cache\n"
                if volume_list_calls == 1
                else ""
            )
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[:3] in (["docker", "container", "ls"], ["docker", "network", "ls"]):
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ["docker", "volume", "inspect"]:
            name = args[-1]
            key = name.removeprefix(PROJECT + "_")
            labels = {
                "com.docker.compose.project": PROJECT,
                "com.docker.compose.volume": key,
                "com.docker.compose.version": "2.29.2",
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(labels), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    release_module.execute_safe_down_volumes(
        manifest_path,
        root=root,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
    )

    assert calls[0][-3:] == ["config", "--format", "json"]
    assert calls[1] == [
        "docker",
        "volume",
        "ls",
        "--format",
        "{{.Name}}",
    ]
    assert "--filter" in calls[2]
    inspect_calls = [
        call for call in calls if call[:3] == ["docker", "volume", "inspect"]
    ]
    assert [call[-1] for call in inspect_calls] == [
        f"{PROJECT}_train_cache",
        f"{PROJECT}_verify_data",
    ]
    assert any(call[-2:] == ["down", "--volumes"] for call in calls)
    assert calls[-1][:3] == ["docker", "volume", "ls"]


def test_safe_down_rejects_arbitrary_same_project_volume_before_down(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[-3:] == ["config", "--format", "json"]:
            compose = {
                "services": {
                    "mysql": {"volumes": [{"type": "volume", "source": "mysql_data"}]}
                },
                "volumes": {"mysql_data": {"name": f"{PROJECT}_mysql_data"}},
                "networks": {"default": {"name": f"{PROJECT}_default"}},
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(compose), "")
        if args[:3] == ["docker", "volume", "ls"] and "--filter" in args:
            return subprocess.CompletedProcess(args, 0, "arbitrary-canary\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_safe_down_volumes(
            manifest_path,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
        )

    assert not any(call[-2:] == ["down", "--volumes"] for call in calls)


def test_graceful_stop_sends_one_sigterm_and_never_kills_on_timeout(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    container_id = "d" * 64
    state_path = root / ".runtime" / "stop-state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project": PROJECT,
                "service": "train-factory-api",
                "container_id": container_id,
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(args, 0, container_id + "\n", "")
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args,
                0,
                f"{container_id}|true|{PROJECT}|train-factory-api\n",
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(release_module.ReleaseComposeError):
        release_module.execute_graceful_stop_no_kill(
            manifest_path,
            service="train-factory-api",
            state_path=state_path,
            wait_seconds=2,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
            sleep=lambda _seconds: None,
            monotonic=iter((0.0, 0.0, 1.0, 2.0)).__next__,
        )

    assert calls.count(["docker", "kill", "--signal", "SIGTERM", container_id]) == 1
    assert not any("{{json .}}" in argument for call in calls for argument in call)
    assert not any(
        "SIGKILL" in call or "stop" in call or "rm" in call for call in calls
    )


def test_graceful_stop_bounds_each_post_signal_poll_by_remaining_deadline(tmp_path):
    manifest_module = _manifest_module()
    release_module = _release_module()
    root, secret_env, release_env = _fake_manifest_root(tmp_path)
    manifest_path = root / ".runtime" / "manifest.json"
    manifest_module.freeze_manifest(
        mode="verify",
        project=PROJECT,
        env_files=(secret_env, root / "docker" / "images.lock.env", release_env),
        gpu_mode="raw",
        secret_mode="files",
        output=manifest_path,
        root=root,
    )
    container_id = "e" * 64
    state_path = root / ".runtime" / "stop-state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project": PROJECT,
                "service": "train-factory-api",
                "container_id": container_id,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    inspect_count = 0

    def run(args, **kwargs):
        nonlocal inspect_count
        calls.append((list(args), kwargs))
        if args[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(args, 0, container_id + "\n", "")
        if args[:2] == ["docker", "inspect"]:
            inspect_count += 1
            if inspect_count == 1:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    f"{container_id}|true|{PROJECT}|train-factory-api\n",
                    "",
                )
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(
        release_module.ReleaseComposeError,
        match="graceful stop timed out",
    ):
        release_module.execute_graceful_stop_no_kill(
            manifest_path,
            service="train-factory-api",
            state_path=state_path,
            wait_seconds=1,
            root=root,
            base_environment={"PATH": os.environ.get("PATH", "")},
            run=run,
            sleep=lambda _seconds: None,
            monotonic=iter((10.0, 10.25)).__next__,
        )

    post_signal_inspect = [
        kwargs for argv, kwargs in calls if argv[:2] == ["docker", "inspect"]
    ][-1]
    assert 0 < post_signal_inspect["timeout"] <= 0.75
    assert [argv for argv, _kwargs in calls].count(
        ["docker", "kill", "--signal", "SIGTERM", container_id]
    ) == 1
    assert not any(
        "SIGKILL" in argv or "stop" in argv or "rm" in argv for argv, _kwargs in calls
    )
