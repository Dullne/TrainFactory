import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).parents[1]


def _module():
    from scripts import run_mysql_migration_tests

    return run_mysql_migration_tests


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_runner_identity_is_128_bit_and_never_targets_default_project():
    module = _module()

    identity = module.new_runner_identity(token_hex=lambda size: "ab" * size)

    assert identity.run_id == "ab" * 16
    assert identity.project == f"trainfactory-mysql-{'ab' * 16}"
    assert identity.project != "trainfactory"
    assert identity.container.startswith(identity.project)
    assert identity.network.startswith(identity.project)
    assert identity.volume.startswith(identity.project)


def test_runner_loads_only_digest_pinned_mysql_image(tmp_path):
    module = _module()
    lock = tmp_path / "images.lock.env"
    digest = "a" * 64
    lock.write_text(
        f"MYSQL_IMAGE=mysql:8.0@sha256:{digest}\n"
        f"API_BASE_IMAGE=example/api:fixed@sha256:{'b' * 64}\n",
        encoding="utf-8",
    )

    assert module.load_mysql_image(lock) == f"mysql:8.0@sha256:{digest}"


@pytest.mark.parametrize(
    "value",
    ("mysql:8.0", "mysql:8.0@sha256:short", "private-canary"),
)
def test_runner_rejects_invalid_mysql_image_without_echo(tmp_path, value):
    module = _module()
    lock = tmp_path / "images.lock.env"
    lock.write_text(f"MYSQL_IMAGE={value}\n", encoding="utf-8")

    with pytest.raises(module.RunnerError) as exc_info:
        module.load_mysql_image(lock)

    assert str(exc_info.value) == "MySQL image lock is invalid"
    assert "private-canary" not in repr(exc_info.value)


def test_pytest_environment_drops_host_database_compose_and_secret_values():
    module = _module()
    canary = "private-host-canary"
    base = {
        "PATH": os.environ.get("PATH", ""),
        "MYSQL_URL": canary,
        "DATABASE_URL": canary,
        "MYSQL_ROOT_PASSWORD": canary,
        "MYSQL_APP_PASSWORD": canary,
        "JWT_SECRET_KEY": canary,
        "COMPOSE_PROJECT_NAME": canary,
        "COMPOSE_FUTURE_CANARY": canary,
    }

    environment = module.clean_pytest_environment(
        base,
        test_url="mysql+pymysql://root:test@127.0.0.1:3307/mysql",
    )

    assert environment["TRAINFACTORY_TEST_MYSQL_URL"].startswith(
        "mysql+pymysql://root:"
    )
    assert environment["PATH"] == base["PATH"]
    assert canary not in "\n".join(environment.values())
    assert not any(name.startswith("COMPOSE_") for name in environment)
    assert "MYSQL_URL" not in environment
    assert "DATABASE_URL" not in environment
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_pytest_environment_rejects_ambient_plugin_path_and_telemetry_injection():
    module = _module()
    canary = "private-ambient-canary"
    base = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PYTHONPATH": canary,
        "PYTEST_ADDOPTS": f"--ignore={canary}",
        "PYTEST_PLUGINS": canary,
        "COVERAGE_PROCESS_START": canary,
        "DD_TRACE_AGENT_URL": canary,
        "RANDOM_FUTURE_HOST_INPUT": canary,
    }

    environment = module.clean_pytest_environment(
        base,
        test_url="mysql+pymysql://root:test@127.0.0.1:3307/mysql",
    )

    assert environment["PATH"] == base["PATH"]
    assert environment["DEBUG"] == "false"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert canary not in "\n".join(environment.values())


def test_project_preflight_checks_container_network_and_volume_labels():
    module = _module()
    calls = []

    def run_cli(args):
        calls.append(args)
        return _completed(args)

    module.assert_project_unused(run_cli, "trainfactory-mysql-" + "ab" * 16)

    assert [args[1:3] for args in calls] == [
        ["ps", "-aq"],
        ["network", "ls"],
        ["volume", "ls"],
    ]
    assert all("com.docker.compose.project=" in " ".join(args) for args in calls)


def test_local_docker_contract_ignores_all_ambient_docker_overrides():
    module = _module()
    canary = "private-remote-daemon-canary"
    environment = module.local_docker_environment(
        {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "DOCKER_HOST": f"ssh://{canary}",
            "DOCKER_CONTEXT": canary,
            "DOCKER_CONFIG": canary,
            "DOCKER_TLS_VERIFY": "1",
            "DOCKER_CERT_PATH": canary,
        }
    )
    command = module.local_docker_command(["docker", "version"])

    assert not any(name.startswith("DOCKER_") for name in environment)
    assert canary not in "\n".join(environment.values())
    assert command[:3] == ["docker", "--host", module.LOCAL_DOCKER_HOST]
    assert command[-1] == "version"


def test_local_daemon_probe_rejects_remote_or_invalid_identity_before_mutation():
    module = _module()
    calls = []

    def run_cli(args):
        calls.append(args)
        if args[1] == "version":
            return _completed(args, stdout="linux/amd64\n")
        return _completed(args, stdout="private-remote-daemon-canary\n")

    with pytest.raises(module.RunnerError) as exc_info:
        module.assert_local_docker_daemon(run_cli)

    assert str(exc_info.value) == "local Docker daemon identity is invalid"
    assert len(calls) == 2
    assert not any("run" in args or "create" in args for args in calls)
    assert "private-remote-daemon-canary" not in repr(exc_info.value)


@pytest.mark.parametrize("collision_index", range(3))
def test_project_preflight_fails_closed_on_any_label_collision(collision_index):
    module = _module()
    calls = []

    def run_cli(args):
        calls.append(args)
        return _completed(args, stdout="owned-id\n" if len(calls) - 1 == collision_index else "")

    with pytest.raises(module.RunnerError) as exc_info:
        module.assert_project_unused(
            run_cli,
            "trainfactory-mysql-" + "ab" * 16,
        )

    assert str(exc_info.value) == "isolated MySQL project already exists"


def test_cleanup_refuses_wrong_project_label_before_delete():
    module = _module()
    calls = []

    def run_cli(args):
        calls.append(args)
        return _completed(args, stdout="trainfactory\n")

    with pytest.raises(module.RunnerError) as exc_info:
        module.remove_owned_resource(
            run_cli,
            kind="container",
            name="private-resource-canary",
            project="trainfactory-mysql-" + "ab" * 16,
        )

    assert str(exc_info.value) == "isolated MySQL cleanup ownership check failed"
    assert len(calls) == 1
    assert calls[0][1] == "inspect"


@pytest.mark.parametrize("kind", ("container", "network", "volume"))
def test_cleanup_inspects_exact_label_then_removes_only_owned_resource(kind):
    module = _module()
    project = "trainfactory-mysql-" + "ab" * 16
    calls = []

    def run_cli(args):
        calls.append(args)
        return _completed(args, stdout=project + "\n")

    module.remove_owned_resource(
        run_cli,
        kind=kind,
        name=f"{project}-{kind}",
        project=project,
    )

    assert len(calls) == 2
    assert calls[0][-1] == f"{project}-{kind}"
    assert calls[1][-1] == f"{project}-{kind}"
    assert "trainfactory" not in calls[1][:-1]


def test_junit_summary_is_aggregate_only(tmp_path):
    module = _module()
    report = tmp_path / "junit.xml"
    report.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites tests="7" failures="1" errors="1" skipped="2" warnings="3">'
        '<testsuite name="private-canary" tests="7" failures="1" errors="1" skipped="2" />'
        "</testsuites>",
        encoding="utf-8",
    )

    summary = module.read_junit_summary(report)

    assert summary.passed == 3
    assert summary.skipped == 2
    assert summary.warnings == 3
    assert summary.failures == 2
    assert module.format_summary(summary) == (
        "passed=3 skipped=2 warnings=3 failures=2"
    )
    assert "private-canary" not in repr(summary)


def test_junit_summary_counts_warning_from_controlled_pytest_summary(tmp_path):
    module = _module()
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuites tests="1" failures="0" errors="0" skipped="0" />',
        encoding="utf-8",
    )

    summary = module.read_junit_summary(
        report,
        pytest_output="1 passed, 2 warnings in 0.42s",
    )

    assert summary.warnings == 2


def test_junit_summary_rejects_symlink_and_oversized_report(tmp_path):
    module = _module()
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuites tests="1" failures="0" errors="0" skipped="0" />',
        encoding="utf-8",
    )
    link = tmp_path / "report-link.xml"
    try:
        link.symlink_to(report)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(module.RunnerError) as exc_info:
        module.read_junit_summary(link)
    assert str(exc_info.value) == "MySQL migration test report is invalid"


def test_junit_summary_rejects_zero_passed_or_any_skipped_as_incomplete(tmp_path):
    module = _module()
    for xml in (
        '<testsuites tests="0" failures="0" errors="0" skipped="0" />',
        '<testsuites tests="1" failures="0" errors="0" skipped="1" />',
    ):
        report = tmp_path / ("empty.xml" if 'tests="0"' in xml else "skip.xml")
        report.write_text(xml, encoding="utf-8")
        summary = module.read_junit_summary(report)
        with pytest.raises(module.RunnerError) as exc_info:
            module.require_complete_summary(summary)
        assert str(exc_info.value) == "MySQL migration test report is incomplete"
    with pytest.raises(module.RunnerError):
        module.require_complete_summary(module.TestSummary(5, 0, 0, 0))

    with pytest.raises(module.RunnerError) as exc_info:
        module.read_junit_summary(report, max_bytes=8)
    assert str(exc_info.value) == "MySQL migration test report is invalid"


def test_runner_invokes_only_the_explicit_integration_file():
    module = _module()
    config = ROOT_DIR / ".runtime" / "run" / "pytest" / "pytest.ini"
    command = module.pytest_command(
        ROOT_DIR / ".runtime" / "report.xml",
        config,
    )

    assert command[:4] == [
        os.fspath(Path(os.sys.executable)),
        "-I",
        "-m",
        "pytest",
    ]
    def normalize(value):
        return os.path.normcase(os.path.abspath(value))
    expected_test = normalize(
        ROOT_DIR / "tests" / "integration" / "test_mysql_migrations.py"
    )
    assert any(normalize(argument) == expected_test for argument in command)
    assert not any(argument == "tests" for argument in command)
    assert any(argument.startswith("--junitxml=") for argument in command)
    confcut = next(argument for argument in command if argument.startswith("--confcutdir="))
    assert normalize(confcut.split("=", 1)[1]) == normalize(
        ROOT_DIR / "tests" / "integration"
    )
    assert "--noconftest" in command
    assert "--import-mode=importlib" in command
    assert normalize(command[command.index("-c") + 1]) == normalize(config)
    rootdir = next(argument for argument in command if argument.startswith("--rootdir="))
    assert normalize(rootdir.split("=", 1)[1]) == normalize(ROOT_DIR)


def test_controlled_pytest_config_disables_repository_configuration(tmp_path):
    module = _module()
    config = tmp_path / "pytest.ini"

    module.write_controlled_pytest_config(config)

    assert config.read_text(encoding="utf-8") == "[pytest]\n"
    assert module._verify_hardened_path(config)


def test_pytest_subprocess_is_pinned_to_repository_root(monkeypatch):
    module = _module()
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured.update(kwargs)
        return _completed(args)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    environment = {"PATH": os.environ.get("PATH", ""), "DEBUG": "false"}

    module._run_pytest([os.sys.executable, "-m", "pytest"], environment)

    assert captured["cwd"] == ROOT_DIR
    assert captured["shell"] is False
    assert captured["env"] == environment
    assert captured["timeout"] == 1800


def test_mysql_container_command_uses_files_random_port_and_exact_labels(tmp_path):
    module = _module()
    identity = module.new_runner_identity(token_hex=lambda size: "ab" * size)
    root_secret = tmp_path / "root.secret"
    app_secret = tmp_path / "app.secret"
    image = f"mysql:8.0@sha256:{'a' * 64}"

    command = module.mysql_container_command(
        identity,
        image=image,
        root_secret=root_secret,
        app_secret=app_secret,
    )

    joined = " ".join(command)
    assert command[:3] == ["docker", "run", "-d"]
    assert "127.0.0.1::3306" in command
    assert f"com.docker.compose.project={identity.project}" in command
    assert "MYSQL_ROOT_PASSWORD_FILE=/run/secrets/mysql_root_password" in command
    assert "MYSQL_PASSWORD_FILE=/run/secrets/mysql_app_password" in command
    assert f"MYSQL_DATABASE=tf_runner_{identity.run_id}" in command
    health_argument = next(item for item in command if item.startswith("--health-cmd="))
    assert "/run/secrets/mysql_app_password" in health_argument
    assert "MYSQL_PWD=" in health_argument
    assert "SELECT 1" in health_argument
    assert "private-root" not in health_argument
    assert os.fspath(root_secret.resolve()) in joined
    assert os.fspath(app_secret.resolve()) in joined
    assert image == command[-1]
    assert "private-root" not in joined


def test_server_uuid_parser_is_canonical_and_private_error_is_fixed():
    module = _module()
    expected = "12345678-1234-1234-1234-123456789abc"
    assert module.parse_server_uuid(expected + "\n") == expected
    with pytest.raises(module.RunnerError) as exc_info:
        module.parse_server_uuid("private-server-canary")
    assert str(exc_info.value) == "isolated MySQL server identity is invalid"
    assert "private-server-canary" not in repr(exc_info.value)


@pytest.mark.parametrize(
    ("output", "expected"),
    (("127.0.0.1:49153\n", 49153), ("0.0.0.0:49153\n", None), ("49153\n", None)),
)
def test_port_parser_accepts_only_random_loopback_binding(output, expected):
    module = _module()
    if expected is None:
        with pytest.raises(module.RunnerError) as exc_info:
            module.parse_loopback_port(output)
        assert str(exc_info.value) == "isolated MySQL port binding is invalid"
    else:
        assert module.parse_loopback_port(output) == expected


def test_test_file_must_be_the_single_canonical_integration_file(tmp_path):
    module = _module()
    expected = ROOT_DIR / "tests" / "integration" / "test_mysql_migrations.py"

    assert module.validate_test_file(expected) == expected.resolve()

    with pytest.raises(module.RunnerError) as exc_info:
        module.validate_test_file(tmp_path / "private-canary.py")
    assert str(exc_info.value) == "MySQL migration test file is not allowed"
    assert "private-canary" not in repr(exc_info.value)


def test_runner_cleanup_attempts_only_created_resources_and_aggregates_failures():
    module = _module()
    identity = module.new_runner_identity(token_hex=lambda size: "ab" * size)
    removed = []

    def remove(kind, name, project):
        removed.append((kind, name, project))
        if kind == "network":
            raise module.RunnerError("private-cleanup-canary")

    with pytest.raises(module.RunnerError) as exc_info:
        module.cleanup_created_resources(
            identity,
            created=("container", "network", "volume"),
            remove=remove,
        )

    assert [item[0] for item in removed] == ["container", "network", "volume"]
    assert all(item[2] == identity.project for item in removed)
    assert str(exc_info.value) == "isolated MySQL cleanup failed"
    assert "private-cleanup-canary" not in repr(exc_info.value)


def test_full_runner_masks_credentials_before_first_docker_mutation(
    tmp_path,
    monkeypatch,
):
    module = _module()
    report = tmp_path / "mysql.xml"
    docker_calls = []
    child_observation = {}
    secrets = iter(("private-root-canary", "private-app-canary"))
    project = "trainfactory-mysql-" + "ab" * 16
    masks = []
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def run_cli(args):
        docker_calls.append(args)
        joined = " ".join(args)
        assert "private-root-canary" not in joined
        assert "private-app-canary" not in joined
        if args[1] == "version":
            return _completed(args, stdout="linux/amd64\n")
        if args[1] == "info":
            return _completed(args, stdout="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa\n")
        if args[:3] in (["docker", "ps", "-aq"], ["docker", "network", "ls"], ["docker", "volume", "ls"]):
            return _completed(args)
        if args[:3] == ["docker", "network", "create"]:
            assert "private-root-canary" in masks
            assert "private-app-canary" in masks
            assert "mysql+pymysql://root:private-root-canary@" in masks
            assert "mysql+pymysql://trainfactory_app:private-app-canary@" in masks
            return _completed(args, stdout="network-id\n")
        if args[:3] == ["docker", "volume", "create"]:
            return _completed(args, stdout="volume-id\n")
        if args[:3] == ["docker", "run", "-d"]:
            return _completed(args, stdout="container-id\n")
        if args[:3] == ["docker", "port", f"{project}-mysql"]:
            return _completed(args, stdout="127.0.0.1:49153\n")
        if args[:3] == ["docker", "inspect", "--format"] and "Health.Status" in args[3]:
            return _completed(args, stdout="healthy\n")
        if args[:3] == ["docker", "exec", f"{project}-mysql"]:
            return _completed(args, stdout="12345678-1234-1234-1234-123456789abc\n")
        if "inspect" in args:
            return _completed(args, stdout=project + "\n")
        return _completed(args)

    def run_child(args, env):
        child_observation["args"] = list(args)
        child_observation["env"] = dict(env)
        report.write_text(
            '<testsuites tests="6" failures="0" errors="0" skipped="0" />',
            encoding="utf-8",
        )
        return _completed(args, stdout="6 passed in 1.0s")

    summary = module.run_isolated_mysql(
        images_lock=tmp_path / "images.lock.env",
        test_file=ROOT_DIR / "tests" / "integration" / "test_mysql_migrations.py",
        work_root=tmp_path / "work",
        junit_out=report,
        run_cli=run_cli,
        run_child=run_child,
        token_hex=lambda size: "ab" * size,
        token_urlsafe=lambda _size: next(secrets),
        sleep=lambda _seconds: None,
        image_override=f"mysql:8.0@sha256:{'a' * 64}",
        github_actions_mask=True,
        mask_sink=masks.append,
    )

    assert module.format_summary(summary) == "passed=6 skipped=0 warnings=0 failures=0"
    assert "TRAINFACTORY_TEST_MYSQL_URL" in child_observation["env"]
    assert "private-root-canary" in child_observation["env"]["TRAINFACTORY_TEST_MYSQL_URL"]
    assert child_observation["env"]["TRAINFACTORY_TEST_MYSQL_URL"].endswith(
        f"/tf_runner_{'ab' * 16}"
    )
    assert child_observation["env"]["TRAINFACTORY_MYSQL_SERVER_UUID"] == (
        "12345678-1234-1234-1234-123456789abc"
    )
    assert "private-root-canary" not in " ".join(child_observation["args"])
    remove_calls = [args for args in docker_calls if args[1:3] in (["rm", "-f"], ["network", "rm"], ["volume", "rm"])]
    assert [args[-1] for args in remove_calls] == [
        f"{project}-mysql",
        f"{project}-network",
        f"{project}-data",
    ]
    assert not (tmp_path / "work" / project).exists()


def test_mysql_runner_github_environment_requires_mask_flag_before_docker(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    report = runtime / "report.xml"
    calls = []
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(module, "validate_images_lock_file", lambda _path: calls.append("lock"))
    monkeypatch.setattr(
        module,
        "run_isolated_mysql",
        lambda **_kwargs: calls.append("runner") or pytest.fail("runner must not start"),
    )

    result = module.main(
        [
            "--images-lock",
            "ignored",
            "--test-file",
            "ignored",
            "--work-root",
            os.fspath(runtime),
            "--junit-out",
            os.fspath(report),
        ]
    )

    assert result == 1
    assert calls == []
    assert capsys.readouterr().err == "MySQL migration runner arguments are invalid\n"


def test_mysql_github_mask_sink_flushes_each_command(monkeypatch):
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


def test_mysql_runner_rejects_secret_drift_after_mask_before_docker(
    tmp_path,
    monkeypatch,
):
    module = _module()
    project = "trainfactory-mysql-" + "ab" * 16
    docker_calls = []
    generated = iter(("private-root-canary", "private-app-canary"))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def run_cli(args):
        docker_calls.append(list(args))
        if args[1] == "version":
            return _completed(args, stdout="linux/amd64\n")
        if args[1] == "info":
            return _completed(args, stdout="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa\n")
        return _completed(args)

    masks = []

    def mask_sink(value):
        masks.append(value)
        if len(masks) == 1:
            (tmp_path / "work" / project / "mysql-app.secret").write_bytes(
                b"unmasked-replacement"
            )

    with pytest.raises(module.RunnerError) as exc_info:
        module.run_isolated_mysql(
            images_lock=tmp_path / "unused.lock",
            test_file=ROOT_DIR / "tests" / "integration" / "test_mysql_migrations.py",
            work_root=tmp_path / "work",
            junit_out=tmp_path / "mysql.xml",
            run_cli=run_cli,
            run_child=lambda _args, _env: pytest.fail("pytest must not start"),
            token_hex=lambda size: "ab" * size,
            token_urlsafe=lambda _size: next(generated),
            sleep=lambda _seconds: None,
            image_override=f"mysql:8.0@sha256:{'a' * 64}",
            github_actions_mask=True,
            mask_sink=mask_sink,
        )

    assert str(exc_info.value) == "isolated MySQL secret validation failed"
    assert not any(args[1:3] == ["network", "create"] for args in docker_calls)
    assert not (tmp_path / "work" / project).exists()


def test_mysql_secret_snapshot_rechecks_acl_after_final_read(tmp_path, monkeypatch):
    module = _module()
    secret = tmp_path / "secret"
    secret.write_bytes(b"private-value")
    checks = iter((True, True, False))
    monkeypatch.setattr(module, "_verify_hardened_path", lambda _path: next(checks))

    with pytest.raises(module.RunnerError) as exc_info:
        module._stable_hardened_secret(secret)

    assert str(exc_info.value) == "isolated MySQL secret validation failed"


def test_start_failure_cleans_only_network_and_volume_and_hides_docker_error(tmp_path):
    module = _module()
    docker_calls = []
    project = "trainfactory-mysql-" + "ab" * 16

    def run_cli(args):
        docker_calls.append(args)
        if args[1] == "version":
            return _completed(args, stdout="linux/amd64\n")
        if args[1] == "info":
            return _completed(args, stdout="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa\n")
        if args[:3] in (["docker", "ps", "-aq"], ["docker", "network", "ls"], ["docker", "volume", "ls"]):
            return _completed(args)
        if args[:3] in (["docker", "network", "create"], ["docker", "volume", "create"]):
            return _completed(args)
        if args[:3] == ["docker", "run", "-d"]:
            return _completed(args, returncode=1, stderr="private-docker-canary")
        if "inspect" in args:
            return _completed(args, stdout=project + "\n")
        return _completed(args)

    with pytest.raises(module.RunnerError) as exc_info:
        module.run_isolated_mysql(
            images_lock=tmp_path / "unused.lock",
            test_file=ROOT_DIR / "tests" / "integration" / "test_mysql_migrations.py",
            work_root=tmp_path / "work",
            junit_out=tmp_path / "mysql.xml",
            run_cli=run_cli,
            run_child=lambda _args, _env: pytest.fail("pytest must not start"),
            token_hex=lambda size: "ab" * size,
            token_urlsafe=lambda _size: "private-secret-canary",
            sleep=lambda _seconds: None,
            image_override=f"mysql:8.0@sha256:{'a' * 64}",
        )

    assert str(exc_info.value) == "isolated MySQL container start failed"
    assert "private-docker-canary" not in repr(exc_info.value)
    remove_calls = [args for args in docker_calls if args[1:3] in (["rm", "-f"], ["network", "rm"], ["volume", "rm"])]
    assert [args[-1] for args in remove_calls] == [
        f"{project}-network",
        f"{project}-data",
    ]


def test_cli_paths_are_contained_regular_and_never_reparse_arbitrary_roots(tmp_path):
    module = _module()
    allowed = tmp_path / ".runtime"
    allowed.mkdir()

    work_root, report = module.validate_runtime_paths(
        allowed / "mysql-work",
        allowed / "mysql.xml",
        allowed_root=allowed,
    )
    assert work_root.parent == allowed.resolve()
    assert report.parent == allowed.resolve()

    with pytest.raises(module.RunnerError) as exc_info:
        module.validate_runtime_paths(tmp_path, tmp_path / "outside.xml", allowed_root=allowed)
    assert str(exc_info.value) == "MySQL migration runtime path is invalid"

    link = allowed / "link"
    try:
        link.symlink_to(tmp_path / "target", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(module.RunnerError):
        module.validate_runtime_paths(link, allowed / "safe.xml", allowed_root=allowed)

    absent = tmp_path / "clean-checkout" / ".runtime"
    absent.parent.mkdir()
    work_root, report = module.validate_runtime_paths(
        absent,
        absent / "fresh.xml",
        allowed_root=absent,
    )
    assert absent.is_dir()
    assert work_root == absent.resolve()
    assert report.parent == absent.resolve()

    blocking_file = allowed / "private-canary"
    blocking_file.write_text("not-a-directory", encoding="utf-8")
    with pytest.raises(module.RunnerError) as exc_info:
        module.validate_runtime_paths(
            blocking_file / "work",
            allowed / "safe-report.xml",
            allowed_root=allowed,
        )
    assert str(exc_info.value) == "MySQL migration runtime path is invalid"
    assert "private-canary" not in repr(exc_info.value)

    with pytest.raises(module.RunnerError):
        module.validate_runtime_paths(
            allowed,
            allowed / "report.xml:private-stream",
            allowed_root=allowed,
        )


def test_github_report_path_is_limited_to_exact_runner_report_root(tmp_path):
    module = _module()
    allowed = tmp_path / "repo" / ".runtime"
    allowed.mkdir(parents=True)
    report_root = tmp_path / "runner-temp" / "trainfactory-reports"
    report_root.mkdir(parents=True)

    work_root, report = module.validate_runtime_paths(
        allowed,
        report_root / "mysql-migrations.xml",
        allowed_root=allowed,
        allowed_report_root=report_root,
    )

    assert work_root == allowed.resolve()
    assert report == report_root.resolve() / "mysql-migrations.xml"
    for invalid in (
        tmp_path / "runner-temp" / "outside.xml",
        report_root / "nested" / "report.xml",
        report_root / "child" / ".." / "alias.xml",
    ):
        with pytest.raises(module.RunnerError) as exc_info:
            module.validate_runtime_paths(
                allowed,
                invalid,
                allowed_root=allowed,
                allowed_report_root=report_root,
            )
        assert str(exc_info.value) == "MySQL migration runtime path is invalid"


def test_main_accepts_only_github_runner_report_root(tmp_path, monkeypatch, capsys):
    module = _module()
    report_root = tmp_path / "runner-temp" / "trainfactory-reports"
    report_root.mkdir(parents=True)
    report = report_root / "mysql-migrations.xml"
    observed = {}
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_TEMP", os.fspath(tmp_path / "runner-temp"))
    monkeypatch.setattr(module, "validate_images_lock_file", lambda _path: tmp_path / "lock")

    monkeypatch.setattr(
        module,
        "run_isolated_mysql",
        lambda **kwargs: observed.update(kwargs)
        or module.TestSummary(passed=6, skipped=0, warnings=0, failures=0),
    )

    result = module.main(
        [
            "--github-actions-mask",
            "--images-lock",
            "docker/images.lock.env",
            "--test-file",
            "tests/integration/test_mysql_migrations.py",
            "--work-root",
            ".runtime",
            "--junit-out",
            os.fspath(report),
        ]
    )

    assert result == 0
    assert observed["work_root"] == (ROOT_DIR / ".runtime").resolve()
    assert observed["junit_out"] == report
    assert observed["github_actions_mask"] is True
    assert capsys.readouterr().out == "passed=6 skipped=0 warnings=0 failures=0\n"


def test_secret_setup_partial_failure_rolls_back_only_its_new_directory(tmp_path, monkeypatch):
    module = _module()
    identity = module.new_runner_identity(token_hex=lambda size: "ab" * size)
    original = module._write_private_file
    calls = 0

    def fail_second(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("private-write-canary")
        original(path, value)

    monkeypatch.setattr(module, "_write_private_file", fail_second)
    with pytest.raises(module.RunnerError) as exc_info:
        module._create_secret_directory(
            tmp_path,
            identity,
            root_password="private-root-canary",
            app_password="private-app-canary",
        )
    assert str(exc_info.value) == "isolated MySQL secret setup failed"
    assert not (tmp_path / identity.project).exists()
    assert "private" not in repr(exc_info.value)


def test_secret_setup_write_then_hardening_failure_leaves_no_private_file(
    tmp_path, monkeypatch
):
    module = _module()
    identity = module.new_runner_identity(token_hex=lambda size: "ab" * size)

    def write_then_fail(path, value):
        path.write_text(value, encoding="utf-8")
        raise RuntimeError("private-acl-canary")

    monkeypatch.setattr(module, "_write_private_file", write_then_fail)
    with pytest.raises(module.RunnerError) as exc_info:
        module._create_secret_directory(
            tmp_path,
            identity,
            root_password="private-root-canary",
            app_password="private-app-canary",
        )

    assert str(exc_info.value) == "isolated MySQL secret setup failed"
    assert not (tmp_path / identity.project).exists()
    assert "private" not in repr(exc_info.value)


def test_secret_setup_collision_preserves_preexisting_empty_directory(tmp_path):
    module = _module()
    identity = module.new_runner_identity(token_hex=lambda size: "ab" * size)
    collision = tmp_path / identity.project
    collision.mkdir()

    with pytest.raises(module.RunnerError):
        module._create_secret_directory(
            tmp_path,
            identity,
            root_password="private-root-canary",
            app_password="private-app-canary",
        )

    assert collision.is_dir()
    assert list(collision.iterdir()) == []


def test_residue_cleanup_finds_resource_created_despite_failed_cli_result():
    module = _module()
    identity = module.new_runner_identity(token_hex=lambda size: "ab" * size)
    calls = []

    def run_cli(args):
        calls.append(args)
        if args[:3] == ["docker", "ps", "-a"]:
            return _completed(args, stdout=identity.container + "\n")
        if args[:3] == ["docker", "network", "ls"]:
            return _completed(args)
        if args[:3] == ["docker", "volume", "ls"]:
            return _completed(args)
        if "inspect" in args:
            return _completed(args, stdout=identity.project + "\n")
        return _completed(args)

    module.cleanup_project_residue(run_cli, identity)

    assert any(args[1:3] == ["rm", "-f"] and args[-1] == identity.container for args in calls)
    assert not any(args[1:3] in (["network", "rm"], ["volume", "rm"]) for args in calls)


def test_main_converts_unexpected_exception_to_fixed_private_error(tmp_path, monkeypatch, capsys):
    module = _module()
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    report = runtime / "report.xml"
    monkeypatch.setattr(module, "validate_images_lock_file", lambda _path: tmp_path / "lock")
    monkeypatch.setattr(
        module,
        "validate_runtime_paths",
        lambda _work, _report: (runtime, report),
    )

    def fail_unexpectedly(**_kwargs):
        raise ValueError("private-unexpected-canary")

    monkeypatch.setattr(module, "run_isolated_mysql", fail_unexpectedly)
    result = module.main(
        [
            "--images-lock",
            "ignored",
            "--test-file",
            "ignored",
            "--work-root",
            "ignored",
            "--junit-out",
            "ignored",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "MySQL migration runner failed\n"
    assert "private-unexpected-canary" not in captured.err


def test_mysql_runner_isolated_direct_cli_help_is_available():
    completed = subprocess.run(
        [sys.executable, "-I", "scripts/run_mysql_migration_tests.py", "--help"],
        cwd=ROOT_DIR,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    assert "--images-lock" in completed.stdout
    assert completed.stderr == ""


@pytest.mark.parametrize(
    "failure_mode",
    ("setup-acl", "timeout", "oversize", "bad-xml", "stdout-secret"),
)
def test_full_runner_discards_unverified_private_report_on_all_child_failures(
    tmp_path, failure_mode, monkeypatch
):
    module = _module()
    report = tmp_path / "mysql.xml"
    project = "trainfactory-mysql-" + "ab" * 16
    secrets = iter(("private-root-canary", "private-app-canary"))
    if failure_mode == "setup-acl":
        original_write = module._write_private_file
        write_calls = 0

        def write_then_fail_report(path, value):
            nonlocal write_calls
            write_calls += 1
            if write_calls == 3:
                path.write_text(value + "private-report-canary", encoding="utf-8")
                raise RuntimeError("private-acl-canary")
            original_write(path, value)

        monkeypatch.setattr(module, "_write_private_file", write_then_fail_report)

    def run_cli(args):
        if args[1] == "version":
            return _completed(args, stdout="linux/amd64\n")
        if args[1] == "info":
            return _completed(args, stdout="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa\n")
        if args[:3] in (["docker", "ps", "-aq"], ["docker", "network", "ls"], ["docker", "volume", "ls"]):
            return _completed(args)
        if args[:3] == ["docker", "port", f"{project}-mysql"]:
            return _completed(args, stdout="127.0.0.1:49153\n")
        if args[:3] == ["docker", "inspect", "--format"] and "Health.Status" in args[3]:
            return _completed(args, stdout="healthy\n")
        if args[:3] == ["docker", "exec", f"{project}-mysql"]:
            return _completed(args, stdout="12345678-1234-1234-1234-123456789abc\n")
        if "inspect" in args:
            return _completed(args, stdout=project + "\n")
        return _completed(args)

    def run_child(args, _env):
        if failure_mode == "setup-acl":
            pytest.fail("pytest must not execute after report ACL failure")
        if failure_mode == "timeout":
            report.write_text("private-root-canary", encoding="utf-8")
            raise TimeoutError("private-timeout-canary")
        if failure_mode == "oversize":
            report.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
            return _completed(args)
        if failure_mode == "bad-xml":
            report.write_text("<malformed", encoding="utf-8")
            return _completed(args)
        report.write_text(
            '<testsuites tests="6" failures="0" errors="0" skipped="0" />',
            encoding="utf-8",
        )
        return _completed(args, stdout="private-root-canary")

    with pytest.raises(module.RunnerError) as exc_info:
        module.run_isolated_mysql(
            images_lock=tmp_path / "unused.lock",
            test_file=ROOT_DIR / "tests" / "integration" / "test_mysql_migrations.py",
            work_root=tmp_path / "work",
            junit_out=report,
            run_cli=run_cli,
            run_child=run_child,
            token_hex=lambda size: "ab" * size,
            token_urlsafe=lambda _size: next(secrets),
            sleep=lambda _seconds: None,
            image_override=f"mysql:8.0@sha256:{'a' * 64}",
        )

    assert not report.exists()
    assert "private-root-canary" not in repr(exc_info.value)
    assert "private-timeout-canary" not in repr(exc_info.value)
