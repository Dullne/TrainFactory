import json
import http.server
import os
import platform
import subprocess
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest


PROJECT = "trainfactory-verify-" + "ab" * 16
REVISION = "a" * 40
API_ID = "sha256:" + "b" * 64
WEB_ID = "sha256:" + "c" * 64
MYSQL_ID = "sha256:" + "d" * 64


def _selection(path: Path) -> None:
    path.write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:fixed\n"
        f"RELEASE_REVISION={REVISION}\n"
        f"API_IMAGE_ID={API_ID}\n"
        f"WEB_IMAGE_ID={WEB_ID}\n",
        encoding="utf-8",
    )


def _manifest(selection: Path, *, gpu_mode="raw"):
    images_lock = selection.parent / "images.lock.env"
    images_lock.write_text("MYSQL_IMAGE=mysql:fixed\n", encoding="utf-8")
    return {
        "project": PROJECT,
        "mode": "verify",
        "gpu_mode": gpu_mode,
        "secret_mode": "files",
        "env_files": [
            {"role": "images-lock", "path": os.fspath(images_lock.resolve())},
            {"role": "image-selection", "path": os.fspath(selection.resolve())},
        ],
    }


def test_verifier_full_scope_checks_identity_health_restart_proxy_schema_and_gpu(
    tmp_path, monkeypatch, capsys
):
    from scripts import compose_manifest
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    selection = runtime / "release.env"
    manifest_path = tmp_path / "manifest.json"
    _selection(selection)
    manifest = _manifest(selection)
    verified_services = []
    monkeypatch.setattr(
        compose_manifest,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        compose_manifest,
        "verify_running_container",
        lambda _path, *, service, **_kwargs: verified_services.append(service),
    )
    monkeypatch.setattr(
        verify_deployment,
        "_repository_head",
        lambda _root: "053_validate_lifecycle_schema",
    )
    monkeypatch.setattr(
        verify_deployment,
        "_require_local_engine",
        lambda *_a: "unix:///var/run/docker.sock",
    )
    container_ids = {
        "mysql": "1" * 64,
        "train-factory-api": "2" * 64,
        "train-factory-web": "3" * 64,
    }
    image_ids = {
        "mysql": MYSQL_ID,
        "train-factory-api": API_ID,
        "train-factory-web": WEB_ID,
    }
    calls = []

    def run(args, **kwargs):
        calls.append((list(args), kwargs))
        if args[:2] == ["docker", "ps"]:
            service = next(
                item.rsplit("=", 1)[1] for item in args if "service=" in item
            )
            return subprocess.CompletedProcess(
                args, 0, container_ids[service] + "\n", ""
            )
        if args[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(args, 0, MYSQL_ID + "\n", "")
        if args[:2] == ["docker", "inspect"]:
            container_id = args[-1]
            service = next(
                key for key, value in container_ids.items() if value == container_id
            )
            return subprocess.CompletedProcess(
                args,
                0,
                f"{container_id}|{image_ids[service]}|true|healthy|0|{PROJECT}|{service}\n",
                "",
            )
        if args[:2] == ["docker", "port"]:
            port = "49180" if args[-1] == "18000/tcp" else "49080"
            return subprocess.CompletedProcess(args, 0, f"127.0.0.1:{port}\n", "")
        if args[:2] == ["docker", "exec"]:
            if "SELECT version_num FROM alembic_version" in " ".join(args):
                return subprocess.CompletedProcess(
                    args, 0, '["053_validate_lifecycle_schema"]\n', ""
                )
            return subprocess.CompletedProcess(
                args,
                0,
                '{"mode":"required","ok":true,"compiled_cuda":"12.4",'
                '"driver_version":"550.54.15","device_count":1,'
                '"probe_executed":true,"reason_code":null}\n',
                "",
            )
        raise AssertionError("unexpected command")

    def request(method, url, body=None):
        assert method == "GET"
        assert body is None
        if url == "http://127.0.0.1:49180/health":
            return 200, b'{"status":"healthy","version":"0.1.0"}'
        if url == "http://127.0.0.1:49180/api/auth/config":
            return 200, (
                b'{"self_registration_enabled":false,'
                b'"direct_storage_registration_enabled":false}'
            )
        if url == "http://127.0.0.1:49080/health":
            return 200, b"healthy\n"
        if url == "http://127.0.0.1:49080/api/auth/config":
            return 200, (
                b'{"self_registration_enabled":false,'
                b'"direct_storage_registration_enabled":false}'
            )
        raise AssertionError("unexpected HTTP request")

    class Session:
        bearer = None

        def set_bearer(self, token):
            self.bearer = token

        def request(self, method, url, body=None):
            if url.endswith("/api/auth/me"):
                assert self.bearer == "PRIVATE-TOKEN-CANARY"
            return request(method, url, body)

    session = Session()

    verify_deployment.verify_deployment(
        project=PROJECT,
        compose_manifest=manifest_path,
        release_env=selection,
        expected_revision=REVISION,
        expected_alembic="053_validate_lifecycle_schema",
        require_gpu=True,
        scope="full",
        root=tmp_path,
        base_environment={"PATH": os.environ.get("PATH", "")},
        run=run,
        request=session,
        sleep=lambda _seconds: None,
    )

    assert verified_services == [
        "mysql",
        "train-factory-api",
        "train-factory-web",
        "mysql",
        "train-factory-api",
        "train-factory-web",
    ]
    assert sum(call[0][:2] == ["docker", "inspect"] for call in calls) == 6
    assert all(
        ".Config.Env" not in item and "{{json .}}" not in item
        for argv, _kwargs in calls
        for item in argv
    )
    database_checks = [
        argv
        for argv, _kwargs in calls
        if "SELECT version_num FROM alembic_version" in " ".join(argv)
    ]
    assert len(database_checks) == 2
    assert database_checks[0] == database_checks[1]
    for database_check in database_checks:
        database_command = " ".join(database_check).upper()
        assert "-M ALEMBIC" not in database_command
        assert " CURRENT" not in database_command and " UPGRADE" not in database_command
        assert all(
            keyword not in database_command
            for keyword in (
                "CREATE ",
                "ALTER ",
                "UPDATE ",
                "INSERT ",
                "DELETE ",
                "DROP ",
            )
        )
    gpu_checks = [
        argv
        for argv, _kwargs in calls
        if argv[:2] == ["docker", "exec"]
        and "SELECT version_num FROM alembic_version" not in " ".join(argv)
    ]
    assert gpu_checks == [
        [
            "docker",
            "exec",
            container_ids["train-factory-api"],
            "python",
            "-I",
            "-c",
            verify_deployment._GPU_PROBE_SCRIPT,
            "required",
        ]
    ]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("mutate_credential", [None, "auth", "sleep"])
def test_verifier_full_no_gpu_uses_body_only_paired_credentials(
    tmp_path, monkeypatch, mutate_credential
):
    from scripts import compose_manifest
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    selection = runtime / "release.env"
    _selection(selection)
    manifest = _manifest(selection, gpu_mode="cpu")
    monkeypatch.setattr(
        compose_manifest,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        compose_manifest, "verify_running_container", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        compose_manifest, "_inspect_selected_image", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        verify_deployment,
        "_repository_head",
        lambda _root: "053_validate_lifecycle_schema",
    )
    monkeypatch.setattr(
        verify_deployment,
        "_require_local_engine",
        lambda *_a: "unix:///var/run/docker.sock",
    )
    monkeypatch.setattr(verify_deployment, "_verify_hardened_path", lambda _path: True)
    username = runtime / "username"
    password = runtime / "password"
    username.write_text("admin\n", encoding="utf-8")
    password.write_text("PRIVATE-PASSWORD-CANARY\n", encoding="utf-8")
    calls = []

    def run(args, **kwargs):
        calls.append((list(args), kwargs))
        if args[:2] == ["docker", "ps"]:
            service = next(
                item.rsplit("=", 1)[1] for item in args if "service=" in item
            )
            identifier = {
                "mysql": "1" * 64,
                "train-factory-api": "2" * 64,
                "train-factory-web": "3" * 64,
            }[service]
            return subprocess.CompletedProcess(args, 0, identifier + "\n", "")
        if args[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(args, 0, MYSQL_ID + "\n", "")
        if args[:2] == ["docker", "inspect"]:
            service = {
                "1" * 64: "mysql",
                "2" * 64: "train-factory-api",
                "3" * 64: "train-factory-web",
            }[args[-1]]
            image_id = {
                "mysql": MYSQL_ID,
                "train-factory-api": API_ID,
                "train-factory-web": WEB_ID,
            }[service]
            return subprocess.CompletedProcess(
                args,
                0,
                f"{args[-1]}|{image_id}|true|healthy|0|{PROJECT}|{service}\n",
                "",
            )
        if args[:2] == ["docker", "port"]:
            port = "49180" if args[-1] == "18000/tcp" else "49080"
            return subprocess.CompletedProcess(args, 0, f"127.0.0.1:{port}\n", "")
        if args[:2] == ["docker", "exec"]:
            if "SELECT version_num FROM alembic_version" in " ".join(args):
                return subprocess.CompletedProcess(
                    args, 0, '["053_validate_lifecycle_schema"]\n', ""
                )
            return subprocess.CompletedProcess(
                args,
                0,
                '{"mode":"off","ok":true,"compiled_cuda":null,'
                '"driver_version":null,"device_count":0,'
                '"probe_executed":false,"reason_code":null}\n',
                "",
            )
        raise AssertionError("unexpected command")

    def request(method, url, body=None):
        if url == "http://127.0.0.1:49180/health":
            return 200, b'{"status":"healthy","version":"0.1.0"}'
        if url == "http://127.0.0.1:49180/api/auth/config":
            return 200, (
                b'{"self_registration_enabled":false,'
                b'"direct_storage_registration_enabled":false}'
            )
        if url == "http://127.0.0.1:49080/health":
            return 200, b"healthy\n"
        if url == "http://127.0.0.1:49080/api/auth/config":
            return 200, (
                b'{"self_registration_enabled":false,'
                b'"direct_storage_registration_enabled":false}'
            )
        if url.endswith("/api/auth/login"):
            assert method == "POST"
            assert json.loads(body) == {
                "username": "admin",
                "password": "PRIVATE-PASSWORD-CANARY",
            }
            return 200, b'{"access_token":"PRIVATE-TOKEN-CANARY","token_type":"bearer"}'
        if url.endswith("/api/auth/me"):
            assert method == "GET"
            return 200, (
                b'{"user_id":"user-1","username":"admin","email":null,'
                b'"is_active":true,"is_admin":true,"created_at":null,'
                b'"updated_at":null}'
            )
        raise AssertionError("unexpected HTTP request")

    class CredentialSession:
        bearer = None

        def set_bearer(self, token):
            self.bearer = token
            if mutate_credential == "auth":
                password.write_text("DIFFERENT-PRIVATE-PASSWORD\n", encoding="utf-8")

        def request(self, method, url, body=None):
            if url.endswith("/api/auth/me"):
                assert self.bearer == "PRIVATE-TOKEN-CANARY"
            return request(method, url, body)

    credential_session = CredentialSession()

    def sleep(_seconds):
        if mutate_credential == "sleep":
            password.write_text("DIFFERENT-PRIVATE-PASSWORD\n", encoding="utf-8")

    arguments = {
        "project": PROJECT,
        "compose_manifest": tmp_path / "manifest.json",
        "release_env": selection,
        "expected_revision": REVISION,
        "expected_alembic": "053_validate_lifecycle_schema",
        "require_gpu": False,
        "scope": "full",
        "username_file": username,
        "password_file": password,
        "root": tmp_path,
        "run": run,
        "request": credential_session,
        "sleep": sleep,
    }
    if mutate_credential is not None:
        with pytest.raises(verify_deployment.VerificationError):
            verify_deployment.verify_deployment(**arguments)
        return
    verify_deployment.verify_deployment(**arguments)

    flattened = repr(calls)
    assert "PRIVATE-PASSWORD-CANARY" not in flattened


def test_verifier_rejects_unpaired_credentials_and_fixed_error_cli(tmp_path, capsys):
    from scripts import verify_deployment

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment.verify_deployment(
            project=PROJECT,
            compose_manifest=tmp_path / "manifest.json",
            release_env=tmp_path / "release.env",
            expected_revision=REVISION,
            expected_alembic="053_validate_lifecycle_schema",
            require_gpu=False,
            username_file=tmp_path / "PRIVATE-USERNAME-CANARY",
            run=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
        )

    assert verify_deployment.main(["--unknown", "PRIVATE-CLI-CANARY"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "deployment verification failed\n"
    assert "PRIVATE" not in captured.err


def test_verifier_rejects_remote_engine_before_manifest_or_http(tmp_path, monkeypatch):
    from scripts import compose_manifest
    from scripts import verify_deployment

    monkeypatch.setattr(
        verify_deployment,
        "_repository_head",
        lambda _root: "053_validate_lifecycle_schema",
    )
    manifest_calls = []
    http_calls = []
    engine_calls = []
    monkeypatch.setattr(
        compose_manifest,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: manifest_calls.append(True),
    )

    def run(args, **_kwargs):
        engine_calls.append(list(args))
        assert args[:3] == ["docker", "context", "inspect"]
        return subprocess.CompletedProcess(args, 0, "tcp://remote.invalid:2376\n", "")

    docker_config = tmp_path / ".docker"
    docker_config.mkdir()
    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment.verify_deployment(
            project=PROJECT,
            compose_manifest=tmp_path / "manifest.json",
            release_env=tmp_path / ".runtime" / "release.env",
            expected_revision=REVISION,
            expected_alembic="053_validate_lifecycle_schema",
            require_gpu=False,
            root=tmp_path,
            base_environment={
                "PATH": os.environ.get("PATH", ""),
                "DOCKER_CONTEXT": "remote-canary",
                "DOCKER_CONFIG": os.fspath(docker_config),
            },
            run=run,
            request=lambda *_args: http_calls.append(True),
        )

    assert manifest_calls == []
    assert http_calls == []
    assert len(engine_calls) == 1


def test_verifier_pins_local_engine_transport_after_context_gate(tmp_path, monkeypatch):
    from scripts import compose_manifest
    from scripts import verify_deployment

    monkeypatch.setattr(
        verify_deployment,
        "_repository_head",
        lambda _root: "053_validate_lifecycle_schema",
    )
    docker_config = tmp_path / ".docker"
    docker_config.mkdir()
    certificates = tmp_path / "certificates"
    certificates.mkdir()
    source_environment = {
        "PATH": os.environ.get("PATH", ""),
        "DOCKER_CONTEXT": "mutable-local-context",
        "DOCKER_CONFIG": os.fspath(docker_config),
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": os.fspath(certificates),
    }

    def gate(_run, _environment):
        source_environment["DOCKER_CONTEXT"] = "remote-after-gate"
        return "npipe:////./pipe/dockerDesktopLinuxEngine"

    monkeypatch.setattr(verify_deployment, "_require_local_engine", gate)
    observed_environments = []

    def verify_inputs(*_args, base_environment, **_kwargs):
        observed_environments.append(dict(base_environment))
        raise compose_manifest.ManifestError("fixed")

    monkeypatch.setattr(compose_manifest, "verify_manifest_inputs_only", verify_inputs)

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment.verify_deployment(
            project=PROJECT,
            compose_manifest=tmp_path / "manifest.json",
            release_env=tmp_path / ".runtime" / "release.env",
            expected_revision=REVISION,
            expected_alembic="053_validate_lifecycle_schema",
            require_gpu=False,
            root=tmp_path,
            base_environment=source_environment,
            run=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
        )

    assert len(observed_environments) == 1
    pinned = observed_environments[0]
    assert pinned["DOCKER_HOST"] == "npipe:////./pipe/dockerDesktopLinuxEngine"
    assert "DOCKER_CONTEXT" not in pinned
    assert "DOCKER_TLS_VERIFY" not in pinned
    assert "DOCKER_CERT_PATH" not in pinned


def test_verifier_canonical_isolated_cli_help_is_available():
    completed = subprocess.run(
        [sys.executable, "-I", "scripts/verify_deployment.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    assert "--compose-manifest" in completed.stdout
    assert completed.stderr == ""


def test_verifier_cli_success_summary_is_closed_and_redacted(monkeypatch, capsys):
    from scripts import verify_deployment

    summary = {
        "project": PROJECT,
        "scope": "api-only",
        "services": {
            "mysql": {
                "container": "1" * 12,
                "image_id": "d" * 12,
                "revision": None,
                "health": True,
            },
            "train-factory-api": {
                "container": "2" * 12,
                "image_id": "b" * 12,
                "revision": REVISION,
                "health": True,
            },
        },
        "api_revision": REVISION,
        "web_revision": REVISION,
        "revision": REVISION,
        "db_head": "053_validate_lifecycle_schema",
        "gpu_reason_code": None,
        "auth_verified": False,
        "rollback_mode": None,
        "rollback_target": None,
    }
    monkeypatch.setattr(
        verify_deployment, "verify_deployment", lambda **_kwargs: summary
    )

    assert (
        verify_deployment.main(
            [
                "--project",
                PROJECT,
                "--compose-manifest",
                "PRIVATE-MANIFEST-PATH-CANARY",
                "--release-env",
                "PRIVATE-RELEASE-PATH-CANARY",
                "--expected-revision",
                REVISION,
                "--expected-alembic",
                "053_validate_lifecycle_schema",
                "--no-gpu",
                "--scope",
                "api-only",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == summary
    assert "PRIVATE" not in captured.out
    assert captured.err == ""


def test_verifier_cli_accepts_explicit_mixed_production_revisions(monkeypatch, capsys):
    from scripts import verify_deployment

    api_revision = "d" * 40
    web_revision = "e" * 40
    observed = []
    summary = {
        "project": "trainfactory",
        "scope": "api-only",
        "services": {},
        "api_revision": api_revision,
        "web_revision": web_revision,
        "revision": None,
        "db_head": "053_validate_lifecycle_schema",
        "gpu_reason_code": None,
        "auth_verified": False,
        "rollback_mode": None,
        "rollback_target": None,
    }

    def verify(**kwargs):
        observed.append(kwargs)
        return summary

    monkeypatch.setattr(verify_deployment, "verify_deployment", verify)

    assert (
        verify_deployment.main(
            [
                "--project",
                "trainfactory",
                "--compose-manifest",
                ".runtime/production-compose-manifest.json",
                "--release-env",
                ".runtime/release.env",
                "--expected-api-revision",
                api_revision,
                "--expected-web-revision",
                web_revision,
                "--expected-alembic",
                "053_validate_lifecycle_schema",
                "--no-gpu",
                "--scope",
                "api-only",
            ]
        )
        == 0
    )

    assert len(observed) == 1
    assert observed[0]["expected_revision"] is None
    assert observed[0]["expected_api_revision"] == api_revision
    assert observed[0]["expected_web_revision"] == web_revision
    captured = capsys.readouterr()
    assert json.loads(captured.out) == summary
    assert captured.err == ""


def _api_only_fixture(
    tmp_path,
    monkeypatch,
    *,
    final_duplicate=False,
    wrong_image=False,
    restart_growth=False,
    health_degrade=False,
    alembic_output='["053_validate_lifecycle_schema"]\n',
    gpu_output=(
        '{"mode":"off","ok":true,"compiled_cuda":null,'
        '"driver_version":null,"device_count":0,'
        '"probe_executed":false,"reason_code":null}\n'
    ),
    database_outputs=None,
    selection_name="release.env",
    manifest_name=None,
    manifest_mode="verify",
    gpu_mode="cpu",
):
    from scripts import compose_manifest
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    selection = runtime / selection_name
    _selection(selection)
    manifest = _manifest(selection, gpu_mode=gpu_mode)
    manifest["mode"] = manifest_mode
    fixture_project = (
        f"trainfactory-ci-{'1' * 32}" if manifest_mode == "ci" else PROJECT
    )
    manifest["project"] = fixture_project
    verified = []
    verify_count = 0

    def verify_inputs(*_args, **_kwargs):
        nonlocal verify_count
        verify_count += 1
        return manifest

    monkeypatch.setattr(compose_manifest, "verify_manifest_inputs_only", verify_inputs)
    monkeypatch.setattr(
        compose_manifest,
        "verify_running_container",
        lambda _path, *, service, **_kwargs: verified.append(service),
    )
    monkeypatch.setattr(
        verify_deployment,
        "_repository_head",
        lambda _root: "053_validate_lifecycle_schema",
    )
    monkeypatch.setattr(
        verify_deployment,
        "_require_local_engine",
        lambda *_a: "unix:///var/run/docker.sock",
    )
    calls = []
    ps_counts = {"mysql": 0, "train-factory-api": 0}
    inspect_counts = {"mysql": 0, "train-factory-api": 0}
    database_query_count = 0
    identifiers = {"mysql": "1" * 64, "train-factory-api": "2" * 64}

    def run(args, **kwargs):
        nonlocal database_query_count
        calls.append(list(args))
        assert "train-factory-web" not in args
        if args[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(args, 0, MYSQL_ID + "\n", "")
        if args[:2] == ["docker", "ps"]:
            service = next(
                item.rsplit("=", 1)[1] for item in args if "service=" in item
            )
            ps_counts[service] += 1
            payload = identifiers[service] + "\n"
            if (
                final_duplicate
                and service == "train-factory-api"
                and ps_counts[service] == 2
            ):
                payload += "4" * 64 + "\n"
            return subprocess.CompletedProcess(args, 0, payload, "")
        if args[:2] == ["docker", "inspect"]:
            service = (
                "mysql" if args[-1] == identifiers["mysql"] else "train-factory-api"
            )
            inspect_counts[service] += 1
            image = MYSQL_ID if service == "mysql" else API_ID
            if wrong_image and service == "train-factory-api":
                image = "sha256:" + "e" * 64
            restart_count = int(
                restart_growth
                and service == "train-factory-api"
                and inspect_counts[service] == 2
            )
            health = (
                "unhealthy"
                if health_degrade
                and service == "train-factory-api"
                and inspect_counts[service] == 2
                else "healthy"
            )
            return subprocess.CompletedProcess(
                args,
                0,
                f"{args[-1]}|{image}|true|{health}|{restart_count}|"
                f"{fixture_project}|{service}\n",
                "",
            )
        if args[:2] == ["docker", "port"]:
            return subprocess.CompletedProcess(args, 0, "127.0.0.1:49180\n", "")
        if args[:2] == ["docker", "exec"]:
            if "SELECT version_num FROM alembic_version" in " ".join(args):
                outputs = database_outputs or (alembic_output,)
                output = outputs[min(database_query_count, len(outputs) - 1)]
                database_query_count += 1
                return subprocess.CompletedProcess(args, 0, output, "")
            return subprocess.CompletedProcess(
                args,
                0,
                gpu_output,
                "",
            )
        raise AssertionError("unexpected command")

    requested = []

    def request(method, url, body=None):
        requested.append((method, url, body))
        assert "49080" not in url
        if url.endswith("/health"):
            return 200, b'{"status":"healthy","version":"0.1.0"}'
        if url.endswith("/api/auth/config"):
            return 200, (
                b'{"self_registration_enabled":false,'
                b'"direct_storage_registration_enabled":false}'
            )
        raise AssertionError("unexpected HTTP request")

    kwargs = {
        "project": fixture_project,
        "compose_manifest": runtime
        / (
            manifest_name
            or (
                f"ci-compose-{'1' * 32}-manifest.json"
                if manifest_mode == "ci"
                else "manifest.json"
            )
        ),
        "release_env": selection,
        "expected_revision": REVISION,
        "expected_alembic": "053_validate_lifecycle_schema",
        "require_gpu": gpu_mode != "cpu",
        "scope": "api-only",
        "root": tmp_path,
        "run": run,
        "request": request,
        "sleep": lambda _seconds: None,
    }
    return verify_deployment, kwargs, calls, requested, verified, verify_inputs


def test_verifier_api_only_never_touches_web(tmp_path, monkeypatch):
    module, kwargs, calls, requested, verified, _verify = _api_only_fixture(
        tmp_path, monkeypatch
    )

    summary = module.verify_deployment(**kwargs)

    assert summary["scope"] == "api-only"
    assert set(summary["services"]) == {"mysql", "train-factory-api"}
    assert verified == ["mysql", "train-factory-api", "mysql", "train-factory-api"]
    assert all("train-factory-web" not in argv for argv in calls)
    assert all("49080" not in url for _method, url, _body in requested)


def test_verifier_accepts_canonical_relative_production_paths(tmp_path, monkeypatch):
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        manifest_name="production-compose-manifest.json",
        manifest_mode="production",
    )
    monkeypatch.chdir(tmp_path)
    kwargs["compose_manifest"] = Path(".runtime/production-compose-manifest.json")
    kwargs["release_env"] = Path(".runtime/release.env")

    summary = module.verify_deployment(**kwargs)

    assert summary["scope"] == "api-only"
    assert summary["api_revision"] == REVISION
    assert summary["web_revision"] == REVISION
    assert summary["revision"] == REVISION


def test_verifier_accepts_canonical_mixed_revision_production_selection(
    tmp_path, monkeypatch
):
    api_revision = "d" * 40
    web_revision = "e" * 40
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        manifest_name="production-compose-manifest.json",
        manifest_mode="production",
    )
    kwargs["release_env"].write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:fixed\n"
        f"API_REVISION={api_revision}\n"
        f"WEB_REVISION={web_revision}\n"
        f"API_IMAGE_ID={API_ID}\n"
        f"WEB_IMAGE_ID={WEB_ID}\n",
        encoding="utf-8",
    )
    del kwargs["expected_revision"]
    kwargs["expected_api_revision"] = api_revision
    kwargs["expected_web_revision"] = web_revision

    summary = module.verify_deployment(**kwargs)

    assert summary["api_revision"] == api_revision
    assert summary["web_revision"] == web_revision
    assert summary["revision"] is None
    assert summary["services"]["train-factory-api"]["revision"] == api_revision


def test_verifier_rejects_single_expected_revision_for_mixed_production(
    tmp_path, monkeypatch
):
    api_revision = "d" * 40
    web_revision = "e" * 40
    module, kwargs, calls, requested, verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        manifest_name="production-compose-manifest.json",
        manifest_mode="production",
    )
    kwargs["release_env"].write_text(
        "API_IMAGE=local/api:fixed\n"
        "WEB_IMAGE=local/web:fixed\n"
        f"API_REVISION={api_revision}\n"
        f"WEB_REVISION={web_revision}\n"
        f"API_IMAGE_ID={API_ID}\n"
        f"WEB_IMAGE_ID={WEB_ID}\n",
        encoding="utf-8",
    )
    kwargs["expected_revision"] = api_revision

    with pytest.raises(
        module.VerificationError, match="^deployment verification failed$"
    ):
        module.verify_deployment(**kwargs)

    assert calls == []
    assert requested == []
    assert verified == []


def test_verifier_rejects_noncanonical_production_manifest_path(tmp_path, monkeypatch):
    module, kwargs, calls, requested, verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        manifest_name="not-the-canonical-production-manifest.json",
        manifest_mode="production",
    )

    with pytest.raises(
        module.VerificationError, match="^deployment verification failed$"
    ):
        module.verify_deployment(**kwargs)

    assert calls == []
    assert requested == []
    assert verified == []


def test_verifier_accepts_canonical_ci_release_selection(tmp_path, monkeypatch):
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        selection_name=f"ci-release-{'1' * 32}.env",
        manifest_mode="ci",
    )

    summary = module.verify_deployment(**kwargs)

    assert summary["revision"] == REVISION


@pytest.mark.parametrize(
    ("selection_name", "manifest_name"),
    [
        ("ci-release.env", f"ci-compose-{'1' * 32}-manifest.json"),
        (f"ci-release-{'2' * 32}.env", f"ci-compose-{'1' * 32}-manifest.json"),
        (f"ci-release-{'1' * 32}.env", "ci-compose-manifest.json"),
        (f"ci-release-{'1' * 32}.env", f"ci-compose-{'2' * 32}-manifest.json"),
    ],
)
def test_verifier_rejects_ci_artifact_run_id_mismatch(
    tmp_path,
    monkeypatch,
    selection_name,
    manifest_name,
):
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        selection_name=selection_name,
        manifest_name=manifest_name,
        manifest_mode="ci",
    )

    with pytest.raises(
        module.VerificationError, match="^deployment verification failed$"
    ):
        module.verify_deployment(**kwargs)


def test_verifier_rejects_ci_parent_alias_paths(tmp_path, monkeypatch):
    run_id = "1" * 32
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        selection_name=f"ci-release-{run_id}.env",
        manifest_mode="ci",
    )
    runtime = tmp_path / ".runtime"
    (runtime / "child").mkdir()
    kwargs["release_env"] = runtime / "child" / ".." / f"ci-release-{run_id}.env"

    with pytest.raises(
        module.VerificationError, match="^deployment verification failed$"
    ):
        module.verify_deployment(**kwargs)


@pytest.mark.parametrize(
    "failure", ["duplicate", "wrong-image", "restart-growth", "health-degrade"]
)
def test_verifier_rejects_restart_window_identity_drift(tmp_path, monkeypatch, failure):
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        final_duplicate=failure == "duplicate",
        wrong_image=failure == "wrong-image",
        restart_growth=failure == "restart-growth",
        health_degrade=failure == "health-degrade",
    )

    with pytest.raises(
        module.VerificationError, match="^deployment verification failed$"
    ):
        module.verify_deployment(**kwargs)


def test_verifier_rejects_manifest_drift_after_probes(tmp_path, monkeypatch):
    from scripts import compose_manifest

    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path, monkeypatch
    )
    manifest = compose_manifest.verify_manifest_inputs_only(None)
    count = 0

    def drift(*_args, **_kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return manifest
        return {**manifest, "project": PROJECT + "-drift"}

    monkeypatch.setattr(compose_manifest, "verify_manifest_inputs_only", drift)

    with pytest.raises(
        module.VerificationError, match="^deployment verification failed$"
    ):
        module.verify_deployment(**kwargs)


def test_verifier_rejects_database_head_drift_during_final_window(
    tmp_path, monkeypatch
):
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        database_outputs=(
            '["053_validate_lifecycle_schema"]\n',
            '["052_generation_publication_staging"]\n',
        ),
    )

    with pytest.raises(module.VerificationError):
        module.verify_deployment(**kwargs)


@pytest.mark.parametrize(
    ("kind", "output"),
    [
        (
            "alembic",
            '["053_validate_lifecycle_schema","052_old"]\n',
        ),
        ("alembic", '["053_validate_lifecycle_schema_evil"]\n'),
        (
            "gpu",
            '{"mode":"off","ok":true,"compiled_cuda":null,'
            '"driver_version":null,"device_count":1,'
            '"probe_executed":false,"reason_code":null}\n',
        ),
        (
            "gpu",
            '{"mode":"off","ok":true,"compiled_cuda":null,'
            '"driver_version":null,"device_count":0,'
            '"probe_executed":false,"reason_code":null,"extra":true}\n',
        ),
    ],
)
def test_verifier_rejects_bad_alembic_and_gpu_shapes(
    tmp_path, monkeypatch, kind, output
):
    options = {"alembic_output" if kind == "alembic" else "gpu_output": output}
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path, monkeypatch, **options
    )

    with pytest.raises(module.VerificationError):
        module.verify_deployment(**kwargs)


@pytest.mark.parametrize(
    "gpu_output",
    [
        '{"mode":"required","ok":true,"compiled_cuda":"",'
        '"driver_version":"550.54.15","device_count":1,'
        '"probe_executed":true,"reason_code":null}\n',
        '{"mode":"required","ok":true,"compiled_cuda":"12.x",'
        '"driver_version":"550.54.15","device_count":1,'
        '"probe_executed":true,"reason_code":null}\n',
        '{"mode":"required","ok":true,"compiled_cuda":"12.4",'
        '"driver_version":"","device_count":1,'
        '"probe_executed":true,"reason_code":null}\n',
        '{"mode":"required","ok":true,"compiled_cuda":"12.4",'
        '"driver_version":"driver","device_count":1,'
        '"probe_executed":true,"reason_code":null}\n',
    ],
)
def test_verifier_rejects_required_gpu_without_numeric_versions(
    tmp_path, monkeypatch, gpu_output
):
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        gpu_mode="raw",
        gpu_output=gpu_output,
    )

    with pytest.raises(module.VerificationError):
        module.verify_deployment(**kwargs)


def test_verifier_accepts_required_gpu_wsl_numeric_driver_version(
    tmp_path, monkeypatch
):
    module, kwargs, _calls, _requested, _verified, _verify = _api_only_fixture(
        tmp_path,
        monkeypatch,
        gpu_mode="raw",
        gpu_output=(
            '{"mode":"required","ok":true,"compiled_cuda":"12.4",'
            '"driver_version":"546.92","device_count":1,'
            '"probe_executed":true,"reason_code":null}\n'
        ),
    )

    assert module.verify_deployment(**kwargs)["gpu_reason_code"] is None


@pytest.mark.parametrize(
    "payload",
    [
        b'{"status":"unhealthy","status":"healthy","version":"0.1.0"}',
        b'{"status":"healthy","version":NaN}',
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite(payload):
    from scripts import verify_deployment

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment._parse_json(payload, expected_keys={"status", "version"})


@pytest.mark.parametrize(
    ("published", "expected_host", "expected"),
    [
        ("127.0.0.1:49180\n", "127.0.0.1", "http://127.0.0.1:49180"),
        ("127.23.45.67:49180\n", "127.23.45.67", "http://127.23.45.67:49180"),
        ("0.0.0.0:49180\n", "0.0.0.0", "http://127.0.0.1:49180"),
        ("[::1]:49180\n", "::1", "http://[::1]:49180"),
        ("[::]:49180\n", "::", "http://[::1]:49180"),
        ("192.168.1.10:49180\n", "192.168.1.10", "http://192.168.1.10:49180"),
        ("[2001:db8::10]:49180\n", "2001:db8::10", "http://[2001:db8::10]:49180"),
    ],
)
def test_port_normalizes_approved_bindings_to_local_loopback(
    published, expected_host, expected
):
    from scripts import verify_deployment

    result = verify_deployment._port(
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, published, ""),
        {},
        "1" * 64,
        18000,
        expected_host=expected_host,
    )
    assert result == expected


@pytest.mark.parametrize(
    "published",
    [
        "203.0.113.10:49180\n",
        "[::ffff:127.0.0.1]:49180\n",
        "127.0.0.1:1\n127.0.0.1:2\n",
        "PRIVATE-CANARY",
    ],
)
def test_port_rejects_unapproved_or_ambiguous_output(published):
    from scripts import verify_deployment

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment._port(
            lambda args, **_kwargs: subprocess.CompletedProcess(
                args, 0, published, "PRIVATE-BACKEND-CANARY"
            ),
            {},
            "1" * 64,
            18000,
        )


def test_port_rejects_observed_address_that_differs_from_frozen_binding():
    from scripts import verify_deployment

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment._port(
            lambda args, **_kwargs: subprocess.CompletedProcess(
                args, 0, "192.168.1.11:49180\n", ""
            ),
            {},
            "1" * 64,
            18000,
            expected_host="192.168.1.10",
        )


def test_repository_head_is_derived_without_importing_migration_modules():
    from scripts import verify_deployment

    assert verify_deployment._repository_head(verify_deployment.ROOT_DIR) == (
        "058_add_model_artifact_membership_gate"
    )


def test_database_head_probe_executes_only_read_only_select(monkeypatch, capsys):
    from scripts import verify_deployment

    events = []

    class Result:
        def scalars(self):
            return self

        def all(self):
            return ["053_validate_lifecycle_schema"]

    class Connection:
        def execute(self, statement):
            events.append(("execute", statement))
            return Result()

        def close(self):
            events.append(("close",))

    class Engine:
        def connect(self):
            events.append(("connect",))
            return Connection()

        def dispose(self):
            events.append(("dispose",))

    sqlalchemy = types.ModuleType("sqlalchemy")
    sqlalchemy.text = lambda statement: statement

    def create_engine(url, **options):
        events.append(("create_engine", url, options))
        return Engine()

    sqlalchemy.create_engine = create_engine
    sqlalchemy_pool = types.ModuleType("sqlalchemy.pool")
    sqlalchemy_pool.NullPool = object()
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
    monkeypatch.setitem(sys.modules, "sqlalchemy.pool", sqlalchemy_pool)
    monkeypatch.setenv("MYSQL_URL", "mysql+pymysql://PRIVATE-DB-CANARY")
    monkeypatch.delenv("MYSQL_URL_FILE", raising=False)

    exec(verify_deployment._DB_HEAD_SCRIPT, {})

    assert capsys.readouterr().out == '["053_validate_lifecycle_schema"]\n'
    assert ("execute", "SELECT version_num FROM alembic_version") in events
    assert all(event[0] not in {"commit", "mkdir"} for event in events)
    assert "train_factory." not in verify_deployment._DB_HEAD_SCRIPT
    assert all(
        keyword not in verify_deployment._DB_HEAD_SCRIPT.upper()
        for keyword in ("CREATE ", "ALTER ", "UPDATE ", "INSERT ", "DELETE ", "DROP ")
    )


def test_database_head_probe_derives_direct_url_from_entrypoint_components(
    monkeypatch, capsys
):
    from scripts import verify_deployment

    observed = []

    class Result:
        def scalars(self):
            return self

        def all(self):
            return ["053_validate_lifecycle_schema"]

    class Connection:
        def execute(self, statement):
            return Result()

        def close(self):
            pass

    class Engine:
        def connect(self):
            return Connection()

        def dispose(self):
            pass

    sqlalchemy = types.ModuleType("sqlalchemy")
    sqlalchemy.text = lambda statement: statement

    def create_engine(url, **options):
        observed.append((url, options))
        return Engine()

    sqlalchemy.create_engine = create_engine
    sqlalchemy_pool = types.ModuleType("sqlalchemy.pool")
    sqlalchemy_pool.NullPool = object()
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
    monkeypatch.setitem(sys.modules, "sqlalchemy.pool", sqlalchemy_pool)
    monkeypatch.delenv("MYSQL_URL", raising=False)
    monkeypatch.delenv("MYSQL_URL_FILE", raising=False)
    monkeypatch.setenv("MYSQL_APP_USER", "trainfactory_app")
    monkeypatch.setenv("MYSQL_APP_PASSWORD", "private:/@ canary")
    monkeypatch.setenv("MYSQL_HOST", "mysql")
    monkeypatch.setenv("MYSQL_DATABASE", "train_factory")

    exec(verify_deployment._DB_HEAD_SCRIPT, {})

    assert capsys.readouterr().out == '["053_validate_lifecycle_schema"]\n'
    assert observed[0][0] == (
        "mysql+pymysql://trainfactory_app:private%3A%2F%40%20canary"
        "@mysql:3306/train_factory"
    )


def test_database_head_probe_rejects_incomplete_entrypoint_components(monkeypatch):
    from scripts import verify_deployment

    monkeypatch.delenv("MYSQL_URL", raising=False)
    monkeypatch.delenv("MYSQL_URL_FILE", raising=False)
    monkeypatch.setenv("MYSQL_APP_USER", "trainfactory_app")
    monkeypatch.setenv("MYSQL_APP_PASSWORD", "private-app-password")
    monkeypatch.setenv("MYSQL_HOST", "mysql")
    monkeypatch.delenv("MYSQL_DATABASE", raising=False)

    with pytest.raises(RuntimeError, match="^database URL source is invalid$"):
        exec(verify_deployment._DB_HEAD_SCRIPT, {})


def test_old_image_compatible_gpu_probe_is_self_contained_off_mode(monkeypatch, capsys):
    from scripts import verify_deployment

    monkeypatch.setattr(sys, "argv", ["gpu-probe", "off"])
    monkeypatch.setitem(sys.modules, "train_factory.runtime.gpu_preflight", None)

    with pytest.raises(SystemExit, match="^0$"):
        exec(verify_deployment._GPU_PROBE_SCRIPT, {})

    assert json.loads(capsys.readouterr().out) == {
        "mode": "off",
        "ok": True,
        "compiled_cuda": None,
        "driver_version": None,
        "device_count": 0,
        "probe_executed": False,
        "reason_code": None,
    }
    assert "train_factory." not in verify_deployment._GPU_PROBE_SCRIPT


def test_old_image_compatible_gpu_probe_is_self_contained_required_mode(
    monkeypatch, capsys
):
    from scripts import verify_deployment

    events = []

    class Tensor:
        def add(self, value):
            events.append(("tensor", value))

    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda="12.4")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        synchronize=lambda device: events.append(("synchronize", device)),
    )
    torch.ones = lambda count, *, device: (
        events.append(("allocate", count, device)) or Tensor()
    )
    pynvml = types.ModuleType("pynvml")
    pynvml.nvmlInit = lambda: events.append(("nvml-init",))
    pynvml.nvmlSystemGetDriverVersion = lambda: b"550.54.15"
    pynvml.nvmlShutdown = lambda: events.append(("nvml-shutdown",))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setitem(sys.modules, "train_factory.runtime.gpu_preflight", None)
    monkeypatch.setitem(sys.modules, "train_factory.config.secret_files", None)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "read_text", lambda _self, **_kwargs: "6.8.0-linux")
    monkeypatch.setattr(sys, "argv", ["gpu-probe", "required"])

    with pytest.raises(SystemExit, match="^0$"):
        exec(verify_deployment._GPU_PROBE_SCRIPT, {})

    assert json.loads(capsys.readouterr().out) == {
        "mode": "required",
        "ok": True,
        "compiled_cuda": "12.4",
        "driver_version": "550.54.15",
        "device_count": 1,
        "probe_executed": True,
        "reason_code": None,
    }
    assert events == [
        ("nvml-init",),
        ("allocate", 1, "cuda:0"),
        ("tensor", 1),
        ("synchronize", 0),
        ("nvml-shutdown",),
    ]


@pytest.mark.parametrize(
    ("compiled_cuda", "driver_version", "expected_ok", "expected_reason"),
    [
        ("12.4", "546.92", True, None),
        ("12.4", "528.32", False, "driver_incompatible"),
        ("13.0", "580.65", False, "platform_unsupported"),
    ],
)
def test_old_image_gpu_probe_matches_wsl_compatibility_matrix(
    monkeypatch,
    capsys,
    compiled_cuda,
    driver_version,
    expected_ok,
    expected_reason,
):
    from scripts import verify_deployment

    class Tensor:
        def add(self, _value):
            return None

    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda=compiled_cuda)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        synchronize=lambda _device: None,
    )
    torch.ones = lambda _count, *, device: Tensor()
    pynvml = types.ModuleType("pynvml")
    pynvml.nvmlInit = lambda: None
    pynvml.nvmlSystemGetDriverVersion = lambda: driver_version
    pynvml.nvmlShutdown = lambda: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _self, **_kwargs: "6.6.87.2-microsoft-standard-WSL2",
    )
    monkeypatch.setattr(sys, "argv", ["gpu-probe", "required"])

    with pytest.raises(SystemExit) as exit_info:
        exec(verify_deployment._GPU_PROBE_SCRIPT, {})

    payload = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == (0 if expected_ok else 1)
    assert payload["ok"] is expected_ok
    assert payload["reason_code"] == expected_reason


def test_database_head_probe_reads_stable_file_source_without_project_imports(
    tmp_path, monkeypatch, capsys
):
    from scripts import verify_deployment

    observed = []

    class Result:
        def scalars(self):
            return self

        def all(self):
            return ["053_validate_lifecycle_schema"]

    class Connection:
        def execute(self, statement):
            observed.append(("execute", statement))
            return Result()

        def close(self):
            observed.append(("close",))

    class Engine:
        def connect(self):
            return Connection()

        def dispose(self):
            observed.append(("dispose",))

    sqlalchemy = types.ModuleType("sqlalchemy")
    sqlalchemy.text = lambda statement: statement
    sqlalchemy.create_engine = lambda url, **options: (
        observed.append(("create_engine", url, options)) or Engine()
    )
    sqlalchemy_pool = types.ModuleType("sqlalchemy.pool")
    sqlalchemy_pool.NullPool = object()
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
    monkeypatch.setitem(sys.modules, "sqlalchemy.pool", sqlalchemy_pool)
    monkeypatch.setitem(sys.modules, "train_factory.config.secret_files", None)
    secret = tmp_path / "mysql-url"
    secret.write_bytes(b"mysql+pymysql://PRIVATE-FILE-CANARY\r\n")
    monkeypatch.delenv("MYSQL_URL", raising=False)
    monkeypatch.setenv("MYSQL_URL_FILE", os.fspath(secret))

    exec(verify_deployment._DB_HEAD_SCRIPT, {})

    assert capsys.readouterr().out == '["053_validate_lifecycle_schema"]\n'
    assert observed[0][0] == "create_engine"
    assert observed[0][1] == "mysql+pymysql://PRIVATE-FILE-CANARY"
    assert ("execute", "SELECT version_num FROM alembic_version") in observed


def test_private_http_client_uses_no_proxy_no_redirect_and_bearer_for_secure_cookie(
    monkeypatch,
):
    from scripts import verify_deployment

    observed = {"authorization": None, "cookie": None, "outside": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            assert self.path == "/api/auth/login"
            self.send_response(200)
            self.send_header(
                "Set-Cookie",
                "access_token=SECURE-TOKEN-CANARY; Secure; HttpOnly; Path=/",
            )
            self.end_headers()
            self.wfile.write(
                b'{"access_token":"SECURE-TOKEN-CANARY","token_type":"bearer"}'
            )

        def do_GET(self):  # noqa: N802
            if self.path == "/api/auth/me":
                observed["authorization"] = self.headers.get("Authorization")
                observed["cookie"] = self.headers.get("Cookie")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
                return
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/outside")
                self.end_headers()
                return
            if self.path == "/outside":
                observed["outside"] += 1
                self.send_response(200)
                self.end_headers()
                return
            if self.path == "/error":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"PRIVATE-HTTP-ERROR-CANARY")
                return
            raise AssertionError(self.path)

        def log_message(self, *_args):
            return

    monkeypatch.setenv("HTTP_PROXY", "http://PRIVATE-PROXY-CANARY.invalid:8080")
    original_build_opener = urllib.request.build_opener
    configured_handlers = []

    def build_opener(*handlers):
        configured_handlers.extend(handlers)
        return original_build_opener(*handlers)

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        client = verify_deployment._PrivateHttpClient()
        assert any(
            isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {}
            for handler in configured_handlers
        )
        status, payload = client.request("POST", origin + "/api/auth/login", b"{}")
        assert status == 200
        token = json.loads(payload)["access_token"]
        client.set_bearer(token)
        assert client.request("GET", origin + "/api/auth/me") == (200, b"{}")
        assert observed == {
            "authorization": "Bearer SECURE-TOKEN-CANARY",
            "cookie": None,
            "outside": 0,
        }
        for path in ("/redirect", "/error"):
            with pytest.raises(
                verify_deployment.VerificationError,
                match="^deployment verification failed$",
            ):
                client.request("GET", origin + path)
        assert observed["outside"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("payload", "max_bytes"),
    [
        (b"x" * 11, 10),
        (b"\xff", 10),
        (b"", 10),
        (b"value\n\n", 10),
        (b"value\x00", 10),
    ],
)
def test_private_credential_file_rejects_oversize_and_invalid_utf8(
    tmp_path, monkeypatch, payload, max_bytes
):
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    path = runtime / "credential"
    path.write_bytes(payload)
    monkeypatch.setattr(verify_deployment, "_verify_hardened_path", lambda _path: True)

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment._private_file_snapshot(
            path, root=tmp_path, max_bytes=max_bytes
        )


def test_private_credential_file_rechecks_acl_after_stable_read(tmp_path, monkeypatch):
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    path = runtime / "credential"
    path.write_text("private-value", encoding="utf-8")
    checks = iter((True, False))
    monkeypatch.setattr(
        verify_deployment,
        "_verify_hardened_path",
        lambda _path: next(checks),
    )

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment._private_file_snapshot(path, root=tmp_path, max_bytes=64)


def test_private_username_file_supports_64_unicode_characters_with_trailing_lf(
    tmp_path, monkeypatch
):
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    path = runtime / "username"
    username = "\U0001f642" * 64
    path.write_text(username + "\n", encoding="utf-8")
    monkeypatch.setattr(verify_deployment, "_verify_hardened_path", lambda _path: True)

    assert (
        verify_deployment._private_file_snapshot(
            path,
            root=tmp_path,
            max_bytes=verify_deployment._USERNAME_MAX_BYTES,
        )[0]
        == username
    )


def test_private_credential_file_rejects_symlink(tmp_path, monkeypatch):
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    target = runtime / "target"
    target.write_text("private-value", encoding="utf-8")
    link = runtime / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    monkeypatch.setattr(verify_deployment, "_verify_hardened_path", lambda _path: True)

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment._private_file_snapshot(link, root=tmp_path, max_bytes=64)


@pytest.mark.skipif(os.name != "nt", reason="Windows path case semantics")
def test_private_credential_file_accepts_same_windows_path_with_drive_case_change(
    tmp_path, monkeypatch
):
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    path = runtime / "credential"
    path.write_text("private-value", encoding="utf-8")
    alternate = Path(str(path)[0].swapcase() + str(path)[1:])
    monkeypatch.setattr(verify_deployment, "_verify_hardened_path", lambda _path: True)

    assert (
        verify_deployment._private_file_snapshot(
            alternate, root=tmp_path, max_bytes=64
        )[0]
        == "private-value"
    )


@pytest.mark.parametrize(
    (
        "rollback_mode",
        "rollback_target",
        "api_id",
        "tamper",
    ),
    [
        (
            "release-api-old-web",
            "isolated",
            API_ID,
            None,
        ),
        (
            "compat-api-old-web",
            "isolated",
            "sha256:" + "e" * 64,
            None,
        ),
        (
            "release-api-old-web",
            "production",
            API_ID,
            None,
        ),
        (
            "compat-api-old-web",
            "production",
            "sha256:" + "e" * 64,
            None,
        ),
        (
            "release-api-old-web",
            "isolated",
            API_ID,
            "arbitrary-web",
        ),
        (
            "compat-api-old-web",
            "isolated",
            "sha256:" + "e" * 64,
            "new-api-revision",
        ),
        (
            "compat-api-old-web",
            "isolated",
            "sha256:" + "6" * 64,
            "wrong-compat-image",
        ),
        (
            "compat-api-old-web",
            "isolated",
            "sha256:" + "e" * 64,
            "approved-compat-web",
        ),
        (
            "release-api-old-web",
            "isolated",
            API_ID,
            "production-manifest",
        ),
        (
            "release-api-old-web",
            "production",
            API_ID,
            "isolated-manifest",
        ),
        (
            "release-api-old-web",
            "isolated",
            API_ID,
            "wrong-manifest-path",
        ),
        (
            "release-api-old-web",
            "production",
            API_ID,
            "wrong-manifest-path",
        ),
    ],
)
def test_verifier_accepts_only_approved_rollback_identity_modes(
    tmp_path,
    monkeypatch,
    rollback_mode,
    rollback_target,
    api_id,
    tamper,
):
    from scripts import compose_manifest
    from scripts import verify_deployment

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    active_project = (
        "trainfactory-rollback-verify-" + "ab" * 16
        if rollback_target == "isolated"
        else "trainfactory"
    )
    release = runtime / "release.env"
    _selection(release)
    captured = runtime / "rollback-pre.env"
    captured.write_text(
        "API_IMAGE=local/api:old\nWEB_IMAGE=local/web:old\n"
        f"API_REVISION={'d' * 40}\nWEB_REVISION={'f' * 40}\n"
        f"API_IMAGE_ID={'sha256:' + '8' * 64}\n"
        f"WEB_IMAGE_ID={'sha256:' + '9' * 64}\n",
        encoding="utf-8",
    )
    if rollback_target == "isolated":
        bundle = runtime / ("rollback-verify-" + "ab" * 16)
        bundle.mkdir()
        candidate = bundle / f"rollback-verify-{rollback_mode}.env"
    elif rollback_mode == "release-api-old-web":
        candidate = runtime / "rollback-web-only.env"
    else:
        candidate = runtime / "rollback-post.env"
    candidate_payload = (
        "API_IMAGE="
        + (
            "local/api:fixed"
            if rollback_mode == "release-api-old-web"
            else "local/api:compat"
        )
        + "\nWEB_IMAGE=local/web:old\n"
        + f"API_REVISION={REVISION if rollback_mode == 'release-api-old-web' or tamper == 'new-api-revision' else 'd' * 40}\n"
        + f"WEB_REVISION={'f' * 40}\n"
        + f"API_IMAGE_ID={api_id}\n"
        + f"WEB_IMAGE_ID={'sha256:' + ('7' if tamper == 'arbitrary-web' else '9') * 64}\n"
    )
    candidate.write_text(candidate_payload, encoding="utf-8")
    if rollback_target == "isolated" and rollback_mode == "compat-api-old-web":
        approved_compat = runtime / "rollback-post.env"
        approved_payload = candidate_payload.replace(
            f"API_IMAGE_ID={api_id}",
            f"API_IMAGE_ID={'sha256:' + 'e' * 64}",
        )
        if tamper == "approved-compat-web":
            approved_payload = approved_payload.replace(
                f"WEB_IMAGE_ID={'sha256:' + '9' * 64}",
                f"WEB_IMAGE_ID={'sha256:' + '7' * 64}",
            )
        approved_compat.write_text(approved_payload, encoding="utf-8")
    manifest = _manifest(candidate, gpu_mode="cpu")
    manifest["mode"] = (
        "rollback-verify"
        if rollback_target == "isolated"
        else (
            "rollback-web-only"
            if rollback_mode == "release-api-old-web"
            else "rollback-post-migration"
        )
    )
    manifest["project"] = active_project
    manifest["rollback_variant"] = (
        rollback_mode if rollback_target == "isolated" else None
    )
    manifest["env_files"][-1]["role"] = (
        "rollback-post"
        if rollback_target == "production" and rollback_mode == "compat-api-old-web"
        else "image-selection"
    )
    if tamper == "production-manifest":
        manifest["mode"] = "rollback-web-only"
        manifest["rollback_variant"] = None
    elif tamper == "isolated-manifest":
        manifest["mode"] = "rollback-verify"
        manifest["rollback_variant"] = rollback_mode
    monkeypatch.setattr(
        compose_manifest,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        compose_manifest, "verify_running_container", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        compose_manifest, "_inspect_selected_image", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        verify_deployment,
        "_repository_head",
        lambda _root: "053_validate_lifecycle_schema",
    )
    monkeypatch.setattr(
        verify_deployment,
        "_require_local_engine",
        lambda *_a: "unix:///var/run/docker.sock",
    )
    monkeypatch.setattr(verify_deployment, "_verify_hardened_path", lambda _path: True)
    username = runtime / "username"
    password = runtime / "password"
    username.write_text("admin\n", encoding="utf-8")
    password.write_text("PRIVATE-PASSWORD-CANARY\n", encoding="utf-8")
    ids = {
        "mysql": "1" * 64,
        "train-factory-api": "2" * 64,
        "train-factory-web": "3" * 64,
    }
    images = {
        "mysql": MYSQL_ID,
        "train-factory-api": api_id,
        "train-factory-web": "sha256:"
        + ("7" if tamper == "arbitrary-web" else "9") * 64,
    }

    def run(args, **_kwargs):
        if args[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(args, 0, MYSQL_ID + "\n", "")
        if args[:2] == ["docker", "ps"]:
            service = next(
                item.rsplit("=", 1)[1] for item in args if "service=" in item
            )
            return subprocess.CompletedProcess(args, 0, ids[service] + "\n", "")
        if args[:2] == ["docker", "inspect"]:
            service = next(key for key, value in ids.items() if value == args[-1])
            return subprocess.CompletedProcess(
                args,
                0,
                f"{args[-1]}|{images[service]}|true|healthy|0|{active_project}|{service}\n",
                "",
            )
        if args[:2] == ["docker", "port"]:
            port = "49180" if args[-1].endswith("18000/tcp") else "49080"
            return subprocess.CompletedProcess(args, 0, f"127.0.0.1:{port}\n", "")
        if args[:2] == ["docker", "exec"]:
            if "SELECT version_num FROM alembic_version" in " ".join(args):
                return subprocess.CompletedProcess(
                    args, 0, '["053_validate_lifecycle_schema"]\n', ""
                )
            return subprocess.CompletedProcess(
                args,
                0,
                '{"mode":"off","ok":true,"compiled_cuda":null,'
                '"driver_version":null,"device_count":0,'
                '"probe_executed":false,"reason_code":null}\n',
                "",
            )
        raise AssertionError("unexpected command")

    def request(method, url, body=None):
        if url.endswith("/health") and "49080" not in url:
            return 200, b'{"status":"healthy","version":"0.1.0"}'
        if url.endswith("/health"):
            return 200, b"healthy\n"
        if url.endswith("/api/auth/config"):
            return 200, (
                b'{"self_registration_enabled":false,'
                b'"direct_storage_registration_enabled":false}'
            )
        if url.endswith("/api/auth/login"):
            return 200, b'{"access_token":"PRIVATE-TOKEN","token_type":"bearer"}'
        if url.endswith("/api/auth/me"):
            return 200, (
                b'{"user_id":"user-1","username":"admin","email":null,'
                b'"is_active":true,"is_admin":true,"created_at":null,'
                b'"updated_at":null}'
            )
        raise AssertionError((method, url, body))

    class Session:
        bearer = None

        def set_bearer(self, token):
            self.bearer = token

        def request(self, method, url, body=None):
            if url.endswith("/api/auth/me"):
                assert self.bearer == "PRIVATE-TOKEN"
            return request(method, url, body)

    session = Session()

    manifest_path = (
        bundle / f"rollback-verify-{rollback_mode}-manifest.json"
        if rollback_target == "isolated"
        else runtime
        / (
            "rollback-web-only-compose-manifest.json"
            if rollback_mode == "release-api-old-web"
            else "rollback-post-compose-manifest.json"
        )
    )
    if tamper == "wrong-manifest-path":
        manifest_path = runtime / "candidate-manifest.json"
    arguments = {
        "project": active_project,
        "compose_manifest": manifest_path,
        "release_env": release,
        "rollback_env": candidate,
        "rollback_mode": rollback_mode,
        "rollback_target": rollback_target,
        "expected_revision": REVISION,
        "expected_alembic": "053_validate_lifecycle_schema",
        "require_gpu": False,
        "scope": "full",
        "username_file": username,
        "password_file": password,
        "root": tmp_path,
        "run": run,
        "request": session,
        "sleep": lambda _seconds: None,
    }
    if tamper is not None:
        with pytest.raises(verify_deployment.VerificationError):
            verify_deployment.verify_deployment(**arguments)
        return
    summary = verify_deployment.verify_deployment(**arguments)

    assert summary["rollback_mode"] == rollback_mode
    assert summary["rollback_target"] == rollback_target
    assert summary["api_revision"] == (
        REVISION if rollback_mode == "release-api-old-web" else "d" * 40
    )
    assert summary["web_revision"] == "f" * 40
    assert summary["revision"] is None


def test_verifier_rejects_raw_old_variant_and_incomplete_rollback_target(tmp_path):
    from scripts import verify_deployment

    calls = []
    common = {
        "project": "trainfactory-rollback-verify-" + "ab" * 16,
        "compose_manifest": tmp_path / "manifest.json",
        "release_env": tmp_path / "release.env",
        "expected_revision": REVISION,
        "expected_alembic": "053_validate_lifecycle_schema",
        "require_gpu": False,
        "scope": "full",
        "username_file": tmp_path / "username",
        "password_file": tmp_path / "password",
        "rollback_env": tmp_path / "rollback.env",
        "root": tmp_path,
        "run": lambda *args, **kwargs: calls.append((args, kwargs)),
    }

    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment.verify_deployment(
            **common,
            rollback_mode="raw-old",
            rollback_target="isolated",
        )
    with pytest.raises(verify_deployment.VerificationError):
        verify_deployment.verify_deployment(
            **common,
            rollback_mode="release-api-old-web",
            rollback_target=None,
        )

    assert calls == []
