"""Run the isolated MySQL migration integration suite.

The orchestration entry point is deliberately assembled from small, testable
pieces.  In particular, cleanup always verifies Compose ownership before it
removes a Docker resource.
"""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("MySQL runner bootstrap is unavailable")


def _bootstrap_realpath(value: str) -> str:
    return _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(value))


_script_path = _bootstrap_realpath(__file__)
_script_dir = _bootstrap_os.path.dirname(_script_path)
_root_entry = _bootstrap_os.path.dirname(_script_dir)
if not __package__:
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
    _seen_paths = {_root_entry}
    for _entry in sys.path:
        _candidate = _bootstrap_realpath(_entry or _bootstrap_os.getcwd())
        if _candidate in {_root_entry, _script_dir} or _candidate in _seen_paths:
            continue
        if _candidate in _pythonpath_entries and not _is_interpreter_path(_candidate):
            continue
        if not _is_interpreter_path(_candidate):
            continue
        _seen_paths.add(_candidate)
        _trusted_sys_path.append(_entry)
    sys.path[:] = [_root_entry, *_trusted_sys_path]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import shutil  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from subprocess import CompletedProcess  # noqa: E402
from urllib.parse import quote, quote_plus  # noqa: E402

from scripts.materialize_compose_secrets import (  # noqa: E402
    _harden_path,
    _verify_hardened_path,
    _write_private_file,
)


ROOT_DIR = Path(_root_entry)
INTEGRATION_TEST = ROOT_DIR / "tests" / "integration" / "test_mysql_migrations.py"
APPROVED_INTEGRATION_TESTS = (
    INTEGRATION_TEST,
    ROOT_DIR / "tests" / "integration" / "test_mysql_sync_lock_order.py",
)
EXPECTED_INTEGRATION_TEST_COUNT = 15
PROJECT_LABEL = "com.docker.compose.project"
LOCAL_DOCKER_HOST = (
    "npipe:////./pipe/docker_engine"
    if os.name == "nt"
    else "unix:///var/run/docker.sock"
)
_IMAGE_PATTERN = re.compile(
    r"[^\s@=]+:[^\s@=]+@sha256:([0-9a-f]{64})",
    re.ASCII,
)
_CHILD_ENVIRONMENT_ALLOWLIST = {
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
}


class RunnerError(RuntimeError):
    """A fixed-message orchestration failure safe for console output."""


@dataclass(frozen=True)
class RunnerIdentity:
    run_id: str
    project: str
    container: str
    network: str
    volume: str


@dataclass(frozen=True)
class TestSummary:
    passed: int
    skipped: int
    warnings: int
    failures: int


@dataclass(frozen=True)
class _SecretSnapshot:
    path: Path
    payload: bytes
    identity: tuple[int, int, int, int, int]


DockerRunner = Callable[[Sequence[str]], CompletedProcess[str]]


def local_docker_environment(base: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in base.items()
        if name.upper() in _CHILD_ENVIRONMENT_ALLOWLIST
    }


def local_docker_command(args: Sequence[str]) -> list[str]:
    if not args or args[0] != "docker":
        raise RunnerError("Docker command is invalid")
    return ["docker", "--host", LOCAL_DOCKER_HOST, *args[1:]]


def assert_local_docker_daemon(run_cli: DockerRunner) -> str:
    version = _docker_result(
        run_cli,
        ("docker", "version", "--format", "{{.Server.Os}}/{{.Server.Arch}}"),
        message="local Docker daemon probe failed",
    )
    identity = _docker_result(
        run_cli,
        ("docker", "info", "--format", "{{.ID}}"),
        message="local Docker daemon probe failed",
    )
    daemon_id = identity.stdout.strip()
    if version.stdout.strip() != "linux/amd64" or re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        daemon_id,
    ) is None:
        raise RunnerError("local Docker daemon identity is invalid")
    return daemon_id


def new_runner_identity(*, token_hex: Callable[[int], str]) -> RunnerIdentity:
    run_id = token_hex(16)
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise RunnerError("isolated MySQL identity generation failed")
    project = f"trainfactory-mysql-{run_id}"
    return RunnerIdentity(
        run_id=run_id,
        project=project,
        container=f"{project}-mysql",
        network=f"{project}-network",
        volume=f"{project}-data",
    )


def load_mysql_image(path: Path) -> str:
    """Read only the pinned MySQL image from the release image lock."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        matches = [line.split("=", 1)[1] for line in lines if line.startswith("MYSQL_IMAGE=")]
    except (OSError, UnicodeError, IndexError):
        raise RunnerError("MySQL image lock is invalid") from None
    if len(matches) != 1:
        raise RunnerError("MySQL image lock is invalid")
    value = matches[0]
    match = _IMAGE_PATTERN.fullmatch(value)
    if match is None or match.group(1) == "0" * 64:
        raise RunnerError("MySQL image lock is invalid")
    return value


def validate_images_lock_file(path: Path) -> Path:
    expected = Path(os.path.abspath(ROOT_DIR / "docker" / "images.lock.env"))
    lexical = Path(os.path.abspath(path))
    try:
        metadata = lexical.lstat()
    except OSError:
        raise RunnerError("MySQL image lock path is invalid") from None
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        os.path.normcase(lexical) != os.path.normcase(expected)
        or not stat.S_ISREG(metadata.st_mode)
        or lexical.is_symlink()
        or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or not 1 <= metadata.st_size <= 65536
    ):
        raise RunnerError("MySQL image lock path is invalid")
    return lexical


def canonical_init_sql_digest(path: Path = ROOT_DIR / "docker" / "init.sql") -> str:
    lexical = Path(os.path.abspath(path))
    expected = Path(os.path.abspath(ROOT_DIR / "docker" / "init.sql"))
    descriptor = -1
    try:
        metadata = lexical.lstat()
        attributes = getattr(metadata, "st_file_attributes", 0)
        if (
            os.path.normcase(lexical) != os.path.normcase(expected)
            or not stat.S_ISREG(metadata.st_mode)
            or lexical.is_symlink()
            or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or not 1 <= metadata.st_size <= 1024 * 1024
        ):
            raise RunnerError("canonical database baseline is invalid")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        payload = os.read(descriptor, metadata.st_size + 1)
        finished = os.fstat(descriptor)
        after = lexical.lstat()
        identities = {
            (metadata.st_dev, metadata.st_ino, metadata.st_size),
            (opened.st_dev, opened.st_ino, opened.st_size),
            (finished.st_dev, finished.st_ino, finished.st_size),
            (after.st_dev, after.st_ino, after.st_size),
        }
        if len(identities) != 1 or len(payload) != metadata.st_size:
            raise RunnerError("canonical database baseline is invalid")
    except RunnerError:
        raise
    except OSError:
        raise RunnerError("canonical database baseline is invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def mysql_container_command(
    identity: RunnerIdentity,
    *,
    image: str,
    root_secret: Path,
    app_secret: Path,
) -> list[str]:
    """Build the secret-value-free Docker argv for the disposable server."""
    marker_database = f"tf_runner_{identity.run_id}"
    return [
        "docker",
        "run",
        "-d",
        "--name",
        identity.container,
        "--network",
        identity.network,
        "--label",
        f"{PROJECT_LABEL}={identity.project}",
        "-p",
        "127.0.0.1::3306",
        "--health-cmd=MYSQL_PWD=\"$(cat /run/secrets/mysql_app_password)\" "
        "mysql --protocol=TCP -h 127.0.0.1 -u trainfactory_test "
        f"-D {marker_database} -Nse 'SELECT 1'",
        "--health-interval=2s",
        "--health-timeout=3s",
        "--health-retries=60",
        "--health-start-period=10s",
        "--mount",
        f"type=volume,source={identity.volume},target=/var/lib/mysql",
        "--mount",
        "type=bind,source="
        f"{root_secret.resolve()},target=/run/secrets/mysql_root_password,readonly",
        "--mount",
        "type=bind,source="
        f"{app_secret.resolve()},target=/run/secrets/mysql_app_password,readonly",
        "-e",
        "MYSQL_ROOT_PASSWORD_FILE=/run/secrets/mysql_root_password",
        "-e",
        "MYSQL_PASSWORD_FILE=/run/secrets/mysql_app_password",
        "-e",
        "MYSQL_USER=trainfactory_test",
        "-e",
        f"MYSQL_DATABASE={marker_database}",
        image,
    ]


def parse_loopback_port(output: str) -> int:
    match = re.fullmatch(r"127\.0\.0\.1:([0-9]{1,5})\s*", output)
    if match is None:
        raise RunnerError("isolated MySQL port binding is invalid")
    port = int(match.group(1))
    if not 1 <= port <= 65535:
        raise RunnerError("isolated MySQL port binding is invalid")
    return port


def parse_server_uuid(output: str) -> str:
    value = output.strip()
    if re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        value,
    ) is None:
        raise RunnerError("isolated MySQL server identity is invalid")
    return value


def mysql_server_uuid_command(identity: RunnerIdentity) -> tuple[str, ...]:
    marker_database = f"tf_runner_{identity.run_id}"
    return (
        "docker",
        "exec",
        identity.container,
        "sh",
        "-c",
        "MYSQL_PWD=\"$(cat /run/secrets/mysql_root_password)\" "
        "mysql --protocol=TCP -h 127.0.0.1 -uroot "
        f"-D {marker_database} -Nse 'SELECT @@server_uuid'",
    )


def validate_test_file(path: Path) -> Path:
    try:
        lexical = Path(os.path.abspath(path))
        expected = Path(os.path.abspath(INTEGRATION_TEST))
        metadata = lexical.lstat()
    except OSError:
        raise RunnerError("MySQL migration test file is not allowed") from None
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        os.path.normcase(lexical) != os.path.normcase(expected)
        or not stat.S_ISREG(metadata.st_mode)
        or lexical.is_symlink()
        or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise RunnerError("MySQL migration test file is not allowed")
    return lexical


def clean_pytest_environment(
    base: Mapping[str, str],
    *,
    test_url: str,
    jwt_secret_key: str,
    runtime_root: Path | None = None,
    run_id: str | None = None,
    server_uuid: str | None = None,
    init_sql_sha256: str | None = None,
) -> dict[str, str]:
    """Return a child environment without host DB, secret, or Compose inputs."""
    environment = {
        name: value
        for name, value in base.items()
        if name.upper() in _CHILD_ENVIRONMENT_ALLOWLIST
    }
    environment["DEBUG"] = "false"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["TRAINFACTORY_TEST_MYSQL_URL"] = test_url
    environment["JWT_SECRET_KEY"] = jwt_secret_key
    if run_id is not None:
        environment["TRAINFACTORY_MYSQL_RUN_ID"] = run_id
    if server_uuid is not None:
        environment["TRAINFACTORY_MYSQL_SERVER_UUID"] = server_uuid
    if init_sql_sha256 is not None:
        environment["TRAINFACTORY_INIT_SQL_SHA256"] = init_sql_sha256
    if runtime_root is not None:
        runtime_paths = {
            "BASE_DIR": runtime_root / "app",
            "TRAINING_CACHE": runtime_root / "cache",
            "MODELS_DIR": runtime_root / "models",
            "DATASETS_DIR": runtime_root / "datasets",
            "OUTPUT_DIR": runtime_root / "output",
            "LOCAL_CACHE_DIR": runtime_root / "cache-datasets",
        }
        environment.update(
            {name: os.fspath(path) for name, path in runtime_paths.items()}
        )
    return environment


def _new_child_jwt_secret() -> str:
    try:
        entropy = secrets.token_bytes(32)
    except Exception:
        raise RunnerError("isolated MySQL child secret generation failed") from None
    if not isinstance(entropy, bytes) or len(entropy) < 32:
        raise RunnerError("isolated MySQL child secret generation failed")
    return entropy.hex()


def _docker_result(
    run_cli: DockerRunner, args: Sequence[str], *, message: str
) -> CompletedProcess[str]:
    try:
        result = run_cli(list(args))
    except Exception:
        raise RunnerError(message) from None
    if result.returncode != 0:
        raise RunnerError(message)
    return result


def assert_project_unused(run_cli: DockerRunner, project: str) -> None:
    label_filter = f"label={PROJECT_LABEL}={project}"
    commands = (
        ("docker", "ps", "-aq", "--filter", label_filter),
        ("docker", "network", "ls", "-q", "--filter", label_filter),
        ("docker", "volume", "ls", "-q", "--filter", label_filter),
    )
    for command in commands:
        result = _docker_result(
            run_cli,
            command,
            message="isolated MySQL project preflight failed",
        )
        if result.stdout.strip():
            raise RunnerError("isolated MySQL project already exists")


def remove_owned_resource(
    run_cli: DockerRunner, *, kind: str, name: str, project: str
) -> None:
    inspect_by_kind = {
        "container": ("docker", "inspect"),
        "network": ("docker", "network", "inspect"),
        "volume": ("docker", "volume", "inspect"),
    }
    remove_by_kind = {
        "container": ("docker", "rm", "-f"),
        "network": ("docker", "network", "rm"),
        "volume": ("docker", "volume", "rm"),
    }
    if kind not in inspect_by_kind:
        raise RunnerError("isolated MySQL cleanup resource type is invalid")
    inspect = (*inspect_by_kind[kind], "--format", f'{{{{ index .Labels "{PROJECT_LABEL}" }}}}', name)
    if kind == "container":
        inspect = (
            "docker",
            "inspect",
            "--format",
            f'{{{{ index .Config.Labels "{PROJECT_LABEL}" }}}}',
            name,
        )
    result = _docker_result(
        run_cli,
        inspect,
        message="isolated MySQL cleanup ownership check failed",
    )
    if result.stdout.strip() != project:
        raise RunnerError("isolated MySQL cleanup ownership check failed")
    _docker_result(
        run_cli,
        (*remove_by_kind[kind], name),
        message="isolated MySQL cleanup failed",
    )


def cleanup_created_resources(
    identity: RunnerIdentity,
    *,
    created: Sequence[str],
    remove: Callable[[str, str, str], None],
) -> None:
    names = {
        "container": identity.container,
        "network": identity.network,
        "volume": identity.volume,
    }
    failed = False
    for kind in ("container", "network", "volume"):
        if kind not in created:
            continue
        try:
            remove(kind, names[kind], identity.project)
        except Exception:
            failed = True
    if failed:
        raise RunnerError("isolated MySQL cleanup failed")


def cleanup_project_residue(run_cli: DockerRunner, identity: RunnerIdentity) -> None:
    """Reconcile label-owned residue after a Docker command outcome is uncertain."""
    specs = (
        (
            "container",
            identity.container,
            (
                "docker",
                "ps",
                "-a",
                "--filter",
                f"label={PROJECT_LABEL}={identity.project}",
                "--format",
                "{{.Names}}",
            ),
        ),
        (
            "network",
            identity.network,
            (
                "docker",
                "network",
                "ls",
                "--filter",
                f"label={PROJECT_LABEL}={identity.project}",
                "--format",
                "{{.Name}}",
            ),
        ),
        (
            "volume",
            identity.volume,
            (
                "docker",
                "volume",
                "ls",
                "--filter",
                f"label={PROJECT_LABEL}={identity.project}",
                "--format",
                "{{.Name}}",
            ),
        ),
    )
    failed = False
    for kind, expected_name, command in specs:
        try:
            result = _docker_result(
                run_cli,
                command,
                message="isolated MySQL cleanup reconciliation failed",
            )
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if any(name != expected_name for name in names) or len(names) > 1:
                raise RunnerError("isolated MySQL cleanup ownership check failed")
            if names:
                remove_owned_resource(
                    run_cli,
                    kind=kind,
                    name=expected_name,
                    project=identity.project,
                )
        except RunnerError:
            failed = True
    if failed:
        raise RunnerError("isolated MySQL cleanup failed")


def _read_report_payload(path: Path, *, max_bytes: int) -> bytes:
    descriptor = -1
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > max_bytes
            or before.st_size < 1
        ):
            raise RunnerError("MySQL migration test report is invalid")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise RunnerError("MySQL migration test report is invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        finished = os.fstat(descriptor)
        after = path.lstat()
        identities = {
            (before.st_dev, before.st_ino, before.st_size),
            (opened.st_dev, opened.st_ino, opened.st_size),
            (finished.st_dev, finished.st_ino, finished.st_size),
            (after.st_dev, after.st_ino, after.st_size),
        }
        if len(identities) != 1 or len(payload) != before.st_size:
            raise RunnerError("MySQL migration test report is invalid")
        return payload
    except RunnerError:
        raise
    except OSError:
        raise RunnerError("MySQL migration test report is invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _discard_unverified_report(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise RunnerError("MySQL migration report cleanup failed") from None
    try:
        if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
            raise RunnerError("MySQL migration report cleanup failed")
        path.unlink()
    except RunnerError:
        raise
    except OSError:
        raise RunnerError("MySQL migration report cleanup failed") from None


def _summary_from_payload(payload: bytes, *, pytest_output: str) -> TestSummary:
    try:
        root = ET.fromstring(payload.decode("utf-8"))
        if root.tag == "testsuite" or (
            root.tag == "testsuites" and "tests" in root.attrib
        ):
            suites = [root]
        else:
            suites = list(root.findall("testsuite"))
        if not suites:
            raise RunnerError("MySQL migration test report is invalid")
        tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
        failures = sum(
            int(suite.attrib.get("failures", "0"))
            + int(suite.attrib.get("errors", "0"))
            for suite in suites
        )
        skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
        warnings = sum(int(suite.attrib.get("warnings", "0")) for suite in suites)
        warning_matches = re.findall(r"(?<![A-Za-z0-9])(\d+) warnings?\b", pytest_output)
        if warning_matches:
            warnings = int(warning_matches[-1])
    except RunnerError:
        raise
    except (ET.ParseError, OSError, TypeError, UnicodeError, ValueError):
        raise RunnerError("MySQL migration test report is invalid") from None
    passed = tests - failures - skipped
    if min(passed, skipped, warnings, failures) < 0:
        raise RunnerError("MySQL migration test report is invalid")
    return TestSummary(
        passed=passed,
        skipped=skipped,
        warnings=warnings,
        failures=failures,
    )


def read_junit_summary(
    path: Path,
    *,
    pytest_output: str = "",
    max_bytes: int = 5 * 1024 * 1024,
) -> TestSummary:
    return _summary_from_payload(
        _read_report_payload(path, max_bytes=max_bytes),
        pytest_output=pytest_output,
    )


def format_summary(summary: TestSummary) -> str:
    return (
        f"passed={summary.passed} skipped={summary.skipped} "
        f"warnings={summary.warnings} failures={summary.failures}"
    )


def require_complete_summary(summary: TestSummary) -> None:
    if (
        summary.passed != EXPECTED_INTEGRATION_TEST_COUNT
        or summary.skipped != 0
        or summary.failures != 0
    ):
        raise RunnerError("MySQL migration test report is incomplete")


def write_controlled_pytest_config(path: Path) -> None:
    """Create the sole pytest configuration accepted by the isolated child."""
    try:
        _write_private_file(path, "[pytest]\n")
    except Exception:
        try:
            if path.exists() and not path.is_symlink():
                path.unlink()
        except OSError:
            pass
        raise RunnerError("MySQL migration pytest configuration failed") from None


def pytest_command(report_path: Path, config_path: Path) -> list[str]:
    return [
        os.fspath(Path(sys.executable)),
        "-I",
        "-m",
        "pytest",
        "-c",
        os.fspath(config_path),
        "-q",
        *(os.fspath(test_path) for test_path in APPROVED_INTEGRATION_TESTS),
        "--noconftest",
        "--import-mode=importlib",
        f"--rootdir={ROOT_DIR}",
        f"--junitxml={report_path}",
        f"--confcutdir={INTEGRATION_TEST.parent}",
        "--tb=no",
    ]


def _create_secret_directory(
    work_root: Path,
    identity: RunnerIdentity,
    *,
    root_password: str,
    app_password: str,
) -> tuple[Path, Path, Path]:
    run_directory = work_root / identity.project
    created_files: list[Path] = []
    created_directory = False
    try:
        if not work_root.exists():
            work_root.mkdir(parents=True)
            _harden_path(work_root)
        if work_root.is_symlink() or not work_root.is_dir():
            raise RunnerError("isolated MySQL secret setup failed")
        run_directory.mkdir(mode=0o700)
        created_directory = True
        _harden_path(run_directory)
        root_secret = run_directory / "mysql-root.secret"
        app_secret = run_directory / "mysql-app.secret"
        created_files.append(root_secret)
        _write_private_file(root_secret, root_password)
        created_files.append(app_secret)
        _write_private_file(app_secret, app_password)
        if not all(
            _verify_hardened_path(path)
            for path in (run_directory, root_secret, app_secret)
        ):
            raise RunnerError("isolated MySQL secret permissions are invalid")
    except Exception as exc:
        for path in reversed(created_files):
            try:
                if path.exists() and not path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        try:
            if (
                created_directory
                and run_directory.exists()
                and not run_directory.is_symlink()
            ):
                run_directory.rmdir()
        except OSError:
            pass
        if isinstance(exc, RunnerError):
            raise
        raise RunnerError("isolated MySQL secret setup failed") from None
    return run_directory, root_secret, app_secret


def _stable_hardened_secret(path: Path) -> _SecretSnapshot:
    try:
        if not _verify_hardened_path(path):
            raise RunnerError("isolated MySQL secret validation failed")
        first = _read_report_payload(path, max_bytes=512)
        first_metadata = path.lstat()
        if not _verify_hardened_path(path):
            raise RunnerError("isolated MySQL secret validation failed")
        second = _read_report_payload(path, max_bytes=512)
        second_metadata = path.lstat()
        if not _verify_hardened_path(path):
            raise RunnerError("isolated MySQL secret validation failed")
        first_identity = (
            first_metadata.st_dev,
            first_metadata.st_ino,
            first_metadata.st_size,
            first_metadata.st_mtime_ns,
            first_metadata.st_ctime_ns,
        )
        second_identity = (
            second_metadata.st_dev,
            second_metadata.st_ino,
            second_metadata.st_size,
            second_metadata.st_mtime_ns,
            second_metadata.st_ctime_ns,
        )
        if first != second or first_identity != second_identity:
            raise RunnerError("isolated MySQL secret validation failed")
        return _SecretSnapshot(path=path, payload=first, identity=first_identity)
    except RunnerError as exc:
        if str(exc) == "isolated MySQL secret validation failed":
            raise
        raise RunnerError("isolated MySQL secret validation failed") from None
    except (OSError, ValueError):
        raise RunnerError("isolated MySQL secret validation failed") from None


def _verify_secret_snapshots(snapshots: Sequence[_SecretSnapshot]) -> None:
    for expected in snapshots:
        if _stable_hardened_secret(expected.path) != expected:
            raise RunnerError("isolated MySQL secret validation failed")


def _github_mask_values(root_password: str, app_password: str) -> tuple[str, ...]:
    values: list[str] = []
    for value in (root_password, app_password):
        values.extend((value, quote(value, safe=""), quote_plus(value, safe="")))
    values.extend(
        (
            f"mysql+pymysql://root:{quote_plus(root_password)}@",
            f"mysql+pymysql://trainfactory_app:{quote_plus(app_password)}@",
        )
    )
    return tuple(dict.fromkeys(values))


def _github_mask_sink(value: str) -> None:
    print(f"::add-mask::{value.replace('%', '%25')}", flush=True)


def _remove_secret_directory(run_directory: Path) -> None:
    try:
        expected = {"mysql-root.secret", "mysql-app.secret"}
        members = {member.name for member in run_directory.iterdir()}
        if members != expected:
            raise RunnerError("isolated MySQL secret cleanup failed")
        for name in sorted(expected):
            path = run_directory / name
            if path.is_symlink() or not path.is_file():
                raise RunnerError("isolated MySQL secret cleanup failed")
        for name in sorted(expected):
            (run_directory / name).unlink()
        run_directory.rmdir()
    except RunnerError:
        raise
    except OSError:
        raise RunnerError("isolated MySQL secret cleanup failed") from None


def _wait_until_healthy(
    run_cli: DockerRunner,
    container: str,
    *,
    sleep: Callable[[float], None],
    attempts: int = 120,
) -> None:
    for _attempt in range(attempts):
        result = _docker_result(
            run_cli,
            (
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                container,
            ),
            message="isolated MySQL health inspection failed",
        )
        status = result.stdout.strip()
        if status == "healthy":
            return
        if status not in {"starting", ""}:
            raise RunnerError("isolated MySQL container became unhealthy")
        sleep(1.0)
    raise RunnerError("isolated MySQL health check timed out")


def run_isolated_mysql(
    *,
    images_lock: Path,
    test_file: Path,
    work_root: Path,
    junit_out: Path,
    run_cli: DockerRunner,
    run_child: Callable[[Sequence[str], Mapping[str, str]], CompletedProcess[str]],
    token_hex: Callable[[int], str],
    token_urlsafe: Callable[[int], str],
    sleep: Callable[[float], None],
    image_override: str | None = None,
    github_actions_mask: bool = False,
    mask_sink: Callable[[str], None] = _github_mask_sink,
) -> TestSummary:
    """Run the sole migration suite against a label-owned disposable server."""
    if (os.environ.get("GITHUB_ACTIONS") == "true") != github_actions_mask:
        raise RunnerError("MySQL migration runner arguments are invalid")
    validate_test_file(test_file)
    identity = new_runner_identity(token_hex=token_hex)
    image = image_override if image_override is not None else load_mysql_image(images_lock)
    if _IMAGE_PATTERN.fullmatch(image) is None:
        raise RunnerError("MySQL image lock is invalid")
    init_sql_sha256 = canonical_init_sql_digest()
    assert_local_docker_daemon(run_cli)
    assert_project_unused(run_cli, identity.project)

    root_password = token_urlsafe(32)
    app_password = token_urlsafe(32)
    jwt_secret_key = _new_child_jwt_secret()
    run_directory: Path | None = None
    pytest_runtime: Path | None = None
    created: list[str] = []
    primary_error: RunnerError | None = None
    summary: TestSummary | None = None
    secret_snapshots: tuple[_SecretSnapshot, ...] = ()
    try:
        run_directory, root_secret, app_secret = _create_secret_directory(
            work_root,
            identity,
            root_password=root_password,
            app_password=app_password,
        )
        secret_snapshots = (
            _stable_hardened_secret(root_secret),
            _stable_hardened_secret(app_secret),
        )
        if secret_snapshots[0].payload != root_password.encode("utf-8") or (
            secret_snapshots[1].payload != app_password.encode("utf-8")
        ):
            raise RunnerError("isolated MySQL secret validation failed")
        if github_actions_mask:
            try:
                for private_value in _github_mask_values(root_password, app_password):
                    mask_sink(private_value)
            except Exception:
                raise RunnerError("MySQL migration mask registration failed") from None
            _verify_secret_snapshots(secret_snapshots)
        pytest_runtime = run_directory / "pytest"
        pytest_runtime.mkdir(mode=0o700)
        _harden_path(pytest_runtime)
        pytest_config = pytest_runtime / "pytest.ini"
        write_controlled_pytest_config(pytest_config)
        _verify_secret_snapshots(secret_snapshots)
        _docker_result(
            run_cli,
            (
                "docker",
                "network",
                "create",
                "--label",
                f"{PROJECT_LABEL}={identity.project}",
                identity.network,
            ),
            message="isolated MySQL network creation failed",
        )
        created.append("network")
        _docker_result(
            run_cli,
            (
                "docker",
                "volume",
                "create",
                "--label",
                f"{PROJECT_LABEL}={identity.project}",
                identity.volume,
            ),
            message="isolated MySQL volume creation failed",
        )
        created.append("volume")
        _verify_secret_snapshots(secret_snapshots)
        _docker_result(
            run_cli,
            mysql_container_command(
                identity,
                image=image,
                root_secret=root_secret,
                app_secret=app_secret,
            ),
            message="isolated MySQL container start failed",
        )
        created.append("container")
        port_result = _docker_result(
            run_cli,
            ("docker", "port", identity.container, "3306/tcp"),
            message="isolated MySQL port inspection failed",
        )
        port = parse_loopback_port(port_result.stdout)
        _wait_until_healthy(run_cli, identity.container, sleep=sleep)
        uuid_result = _docker_result(
            run_cli,
            mysql_server_uuid_command(identity),
            message="isolated MySQL server identity inspection failed",
        )
        server_uuid = parse_server_uuid(uuid_result.stdout)
        marker_database = f"tf_runner_{identity.run_id}"
        test_url = (
            "mysql+pymysql://root:"
            f"{quote_plus(root_password)}@127.0.0.1:{port}/{marker_database}"
        )
        environment = clean_pytest_environment(
            os.environ,
            test_url=test_url,
            jwt_secret_key=jwt_secret_key,
            runtime_root=pytest_runtime,
            run_id=identity.run_id,
            server_uuid=server_uuid,
            init_sql_sha256=init_sql_sha256,
        )
        command = pytest_command(junit_out, pytest_config)
        try:
            _write_private_file(junit_out, "")
        except Exception:
            _discard_unverified_report(junit_out)
            raise RunnerError("MySQL migration report setup failed") from None
        _verify_secret_snapshots(secret_snapshots)
        try:
            completed = run_child(command, environment)
        except Exception:
            _discard_unverified_report(junit_out)
            raise RunnerError("MySQL migration pytest execution failed") from None
        _verify_secret_snapshots(secret_snapshots)
        pytest_output = f"{completed.stdout}\n{completed.stderr}"
        try:
            report_payload = _read_report_payload(
                junit_out,
                max_bytes=5 * 1024 * 1024,
            )
            if not _verify_hardened_path(junit_out):
                raise RunnerError("MySQL migration test report is invalid")
        except RunnerError:
            _discard_unverified_report(junit_out)
            raise
        for private_value in (
            root_password.encode("utf-8"),
            app_password.encode("utf-8"),
            test_url.encode("utf-8"),
            jwt_secret_key.encode("utf-8"),
        ):
            if private_value and (
                private_value in report_payload
                or private_value.decode("utf-8") in pytest_output
            ):
                _discard_unverified_report(junit_out)
                raise RunnerError("MySQL migration report contains private data")
        try:
            summary = _summary_from_payload(report_payload, pytest_output=pytest_output)
        except RunnerError:
            _discard_unverified_report(junit_out)
            raise
        if completed.returncode != 0:
            raise RunnerError("MySQL migration tests failed")
        require_complete_summary(summary)
    except RunnerError as exc:
        primary_error = exc
    finally:
        cleanup_error = False
        try:
            cleanup_created_resources(
                identity,
                created=created,
                remove=lambda kind, name, project: remove_owned_resource(
                    run_cli,
                    kind=kind,
                    name=name,
                    project=project,
                ),
            )
        except RunnerError:
            cleanup_error = True
        try:
            cleanup_project_residue(run_cli, identity)
        except RunnerError:
            cleanup_error = True
        if run_directory is not None:
            if pytest_runtime is not None and pytest_runtime.exists():
                try:
                    if (
                        pytest_runtime.is_symlink()
                        or pytest_runtime.resolve().parent != run_directory.resolve()
                    ):
                        raise RunnerError("isolated MySQL pytest cleanup failed")
                    shutil.rmtree(pytest_runtime)
                except (OSError, RunnerError):
                    cleanup_error = True
            try:
                _remove_secret_directory(run_directory)
            except RunnerError:
                cleanup_error = True
        if cleanup_error:
            primary_error = RunnerError("isolated MySQL cleanup failed")
    if primary_error is not None:
        raise primary_error
    if summary is None:
        raise RunnerError("MySQL migration tests failed")
    return summary


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        raise RunnerError("MySQL migration runner arguments are invalid")


def _is_reparse_path(path: Path) -> bool:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def validate_runtime_paths(
    work_root: Path,
    junit_out: Path,
    *,
    allowed_root: Path = ROOT_DIR / ".runtime",
    allowed_report_root: Path | None = None,
) -> tuple[Path, Path]:
    """Confine mutable runner paths to the repository's ignored runtime root."""
    try:
        if not allowed_root.exists() and not allowed_root.is_symlink():
            raw_root = Path(os.path.abspath(allowed_root))
            if raw_root.name != ".runtime":
                raise RunnerError("MySQL migration runtime path is invalid")
            parent = raw_root.parent.resolve(strict=True)
            if not parent.is_dir() or _is_reparse_path(parent):
                raise RunnerError("MySQL migration runtime path is invalid")
            raw_root.mkdir(mode=0o700)
        canonical_root = allowed_root.resolve(strict=True)
        if not canonical_root.is_dir() or _is_reparse_path(canonical_root):
            raise RunnerError("MySQL migration runtime path is invalid")
        canonical_report_root = canonical_root
        if allowed_report_root is not None:
            raw_report_root = Path(allowed_report_root)
            absolute_report_root = Path(os.path.abspath(raw_report_root))
            if not raw_report_root.is_absolute() or os.path.normcase(
                os.fspath(raw_report_root)
            ) != os.path.normcase(os.fspath(absolute_report_root)):
                raise RunnerError("MySQL migration runtime path is invalid")
            current = absolute_report_root
            while True:
                if current.exists() or current.is_symlink():
                    if _is_reparse_path(current):
                        raise RunnerError("MySQL migration runtime path is invalid")
                if current == current.parent:
                    break
                current = current.parent
            canonical_report_root = absolute_report_root.resolve(strict=True)
            if (
                not canonical_report_root.is_dir()
                or os.path.normcase(os.fspath(canonical_report_root))
                != os.path.normcase(os.fspath(absolute_report_root))
            ):
                raise RunnerError("MySQL migration runtime path is invalid")
        results: list[Path] = []
        for candidate_index, candidate in enumerate((work_root, junit_out)):
            candidate_root = (
                canonical_root if candidate_index == 0 else canonical_report_root
            )
            absolute = Path(os.path.abspath(candidate))
            if (
                allowed_report_root is not None
                and candidate_index == 1
                and os.path.normcase(os.fspath(candidate))
                != os.path.normcase(os.fspath(absolute))
            ):
                raise RunnerError("MySQL migration runtime path is invalid")
            if os.path.commonpath((candidate_root, absolute)) != os.fspath(
                candidate_root
            ):
                raise RunnerError("MySQL migration runtime path is invalid")
            relative = absolute.relative_to(candidate_root)
            if allowed_report_root is not None and candidate_index == 1 and len(
                relative.parts
            ) != 1:
                raise RunnerError("MySQL migration runtime path is invalid")
            current = candidate_root
            for part_index, part in enumerate(relative.parts):
                if ":" in part or part.endswith((" ", ".")):
                    raise RunnerError("MySQL migration runtime path is invalid")
                current = current / part
                if current.exists() or current.is_symlink():
                    if _is_reparse_path(current):
                        raise RunnerError("MySQL migration runtime path is invalid")
                    is_final = part_index == len(relative.parts) - 1
                    if candidate_index == 1 and is_final:
                        raise RunnerError("MySQL migration runtime path is invalid")
                    if not current.is_dir():
                        raise RunnerError("MySQL migration runtime path is invalid")
            resolved = absolute.resolve(strict=False)
            if os.path.commonpath((candidate_root, resolved)) != os.fspath(
                candidate_root
            ):
                raise RunnerError("MySQL migration runtime path is invalid")
            results.append(resolved)
        if results[1] == canonical_report_root or results[1].exists():
            raise RunnerError("MySQL migration runtime path is invalid")
    except RunnerError:
        raise
    except (OSError, ValueError):
        raise RunnerError("MySQL migration runtime path is invalid") from None
    return results[0], results[1]


def _run_docker(args: Sequence[str]) -> CompletedProcess[str]:
    try:
        return subprocess.run(
            local_docker_command(args),
            shell=False,
            env=local_docker_environment(os.environ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise RunnerError("Docker command execution failed") from None


def _run_pytest(
    args: Sequence[str], environment: Mapping[str, str]
) -> CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            shell=False,
            cwd=ROOT_DIR,
            env=dict(environment),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise RunnerError("MySQL migration pytest execution failed") from None


def _parser() -> argparse.ArgumentParser:
    parser = _PrivateArgumentParser(allow_abbrev=False)
    parser.add_argument("--images-lock", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--junit-out", required=True, type=Path)
    parser.add_argument("--github-actions-mask", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if (os.environ.get("GITHUB_ACTIONS") == "true") != (
            arguments.github_actions_mask
        ):
            raise RunnerError("MySQL migration runner arguments are invalid")
        images_lock = validate_images_lock_file(arguments.images_lock)
        if arguments.github_actions_mask:
            runner_temp = os.environ.get("RUNNER_TEMP")
            if not runner_temp:
                raise RunnerError("MySQL migration runner arguments are invalid")
            work_root, junit_out = validate_runtime_paths(
                arguments.work_root,
                arguments.junit_out,
                allowed_report_root=Path(runner_temp) / "trainfactory-reports",
            )
        else:
            work_root, junit_out = validate_runtime_paths(
                arguments.work_root,
                arguments.junit_out,
            )
        junit_out.parent.mkdir(parents=True, exist_ok=True)
        summary = run_isolated_mysql(
            images_lock=images_lock,
            test_file=arguments.test_file,
            work_root=work_root,
            junit_out=junit_out,
            run_cli=_run_docker,
            run_child=_run_pytest,
            token_hex=secrets.token_hex,
            token_urlsafe=secrets.token_urlsafe,
            sleep=time.sleep,
            github_actions_mask=arguments.github_actions_mask,
        )
    except RunnerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("MySQL migration runner failed", file=sys.stderr)
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
