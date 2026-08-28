import io
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).parents[1]
PYNVML_REDIRECT_WARNING = (
    "The pynvml package is deprecated. Please install nvidia-ml-py instead. "
    "If you did not install pynvml directly, please report this to the maintainers "
    "of the package that installed pynvml for you."
)


def _module():
    from scripts import check_pytest_partitions

    return check_pytest_partitions


def _completed(stdout, *, returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_host_tools_marker_is_registered():
    project = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["pytest"]["ini_options"]["markers"] == [
        "host_tools: requires host Git or Docker tooling and never runs in the CPU test image",
    ]


@pytest.mark.parametrize(
    ("message", "expected_returncode"),
    (
        (PYNVML_REDIRECT_WARNING, 0),
        (f"{PYNVML_REDIRECT_WARNING} adjacent-canary", 2),
    ),
    ids=("locked-pynvml-redirect", "adjacent-warning"),
)
def test_pytest_warning_policy_only_allows_locked_pynvml_redirect(
    tmp_path,
    message,
    expected_returncode,
):
    test_file = tmp_path / "test_warning_policy.py"
    test_file.write_text(
        "import warnings\n"
        f"warnings.warn_explicit({message!r}, FutureWarning, "
        "'torch/cuda/__init__.py', 61, module='torch.cuda')\n"
        "def test_placeholder():\n"
        "    pass\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-c",
            os.fspath(ROOT_DIR / "pyproject.toml"),
            os.fspath(test_file),
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

    assert completed.returncode == expected_returncode


def test_partition_checker_collects_four_disjoint_exhaustive_sets():
    module = _module()
    responses = iter(
        (
            _completed("tests/test_a.py::test_backend\ntests/test_b.py::test_host\n"
                       "tests/integration/test_mysql_migrations.py::test_mysql\n"
                       "tests/integration/test_mysql_sync_lock_order.py::test_sync\n"
                       "4 tests collected in 0.01s\n"),
            _completed("tests/test_a.py::test_backend\n1 test collected\n"),
            _completed("tests/test_b.py::test_host\n1/4 tests collected (3 deselected) in 0.01s\n"),
            _completed(
                "tests/integration/test_mysql_migrations.py::test_mysql\n"
                "tests/integration/test_mysql_sync_lock_order.py::test_sync\n"
                "2 tests collected\n"
            ),
        )
    )
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return next(responses)

    summary = module.check_partitions(root=ROOT_DIR, python=sys.executable, run=run)

    assert summary == {"all": 4, "backend": 1, "host_tools": 1, "mysql_migrations": 2}
    assert len(calls) == 4
    for arguments, kwargs in calls:
        assert arguments[:5] == [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "--collect-only",
        ]
        assert "--strict-config" in arguments
        assert "--strict-markers" in arguments
        assert arguments[5:7] == ["-p", "no:cacheprovider"]
        assert kwargs["cwd"] == ROOT_DIR
        assert kwargs["shell"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == module.COLLECT_TIMEOUT_SECONDS
        assert kwargs["env"]["DEBUG"] == "false"
        assert kwargs["env"]["GPU_PREFLIGHT_MODE"] == "off"
        assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert "PYTHONPATH" not in kwargs["env"]
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        assert "PYTEST_PLUGINS" not in kwargs["env"]

    all_argv, backend_argv, host_argv, mysql_argv = [call[0] for call in calls]
    assert "-m" not in all_argv[5:]
    assert backend_argv[-5:] == [
        "-m",
        "not host_tools",
        "--ignore=tests/integration/test_mysql_migrations.py",
        "--ignore=tests/integration/test_mysql_sync_lock_order.py",
        "tests",
    ]
    assert host_argv[-3:] == ["-m", "host_tools", "tests"]
    assert mysql_argv[-2:] == [
        "tests/integration/test_mysql_migrations.py",
        "tests/integration/test_mysql_sync_lock_order.py",
    ]


@pytest.mark.parametrize(
    "responses",
    (
        (
            _completed("tests/test_a.py::test_a\ntests/test_b.py::test_b\n2 tests collected\n"),
            _completed("tests/test_a.py::test_a\n1 test collected\n"),
            _completed("0 tests collected\n"),
            _completed("0 tests collected\n"),
        ),
        (
            _completed("tests/test_a.py::test_a\n1 test collected\n"),
            _completed("tests/test_a.py::test_a\n1 test collected\n"),
            _completed("tests/test_a.py::test_a\n1 test collected\n"),
            _completed("0 tests collected\n"),
        ),
    ),
    ids=("missing", "overlap"),
)
def test_partition_checker_rejects_missing_or_overlapping_nodes(responses):
    module = _module()
    queue = iter(responses)

    with pytest.raises(module.PartitionError, match="^pytest partition validation failed$"):
        module.check_partitions(
            root=ROOT_DIR,
            python=sys.executable,
            run=lambda *_args, **_kwargs: next(queue),
        )


@pytest.mark.parametrize(
    "responses",
    (
        (
            _completed(
                "tests/test_a.py::test_a\n"
                "tests/integration/test_mysql_migrations.py::test_mysql\n"
                "2 tests collected\n"
            ),
            _completed("tests/test_a.py::test_a\n1 test collected\n"),
            _completed("0 tests collected\n"),
            _completed(
                "tests/integration/test_mysql_migrations.py::test_mysql\n"
                "1 test collected\n"
            ),
        ),
        (
            _completed(
                "tests/test_a.py::test_a\ntests/test_b.py::test_b\n"
                "2 tests collected\n"
            ),
            _completed("tests/test_a.py::test_a\n1 test collected\n"),
            _completed("tests/test_b.py::test_b\n1 test collected\n"),
            _completed("0 tests collected\n"),
        ),
    ),
    ids=("host-tools-empty", "mysql-empty"),
)
def test_partition_checker_rejects_empty_required_partitions(responses):
    module = _module()
    queue = iter(responses)

    with pytest.raises(module.PartitionError, match="^pytest partition validation failed$"):
        module.check_partitions(
            root=ROOT_DIR,
            python=sys.executable,
            run=lambda *_args, **_kwargs: next(queue),
        )


@pytest.mark.parametrize(
    "bad_result",
    (
        _completed("", returncode=4, stderr="private-backend-canary"),
        _completed("tests/test_a.py::test_a\n", stderr="private-warning-canary"),
        _completed("tests/test_a.py::test_a\nWARNING collection warning\n1 test collected\n"),
        _completed("not-a-node\n1 test collected\n"),
        _completed(
            "tests/test_a.py::test_a\ntests/test_a.py::test_a\n2 tests collected\n"
        ),
        _completed("no tests collected in 0.01s\n", returncode=5),
    ),
)
def test_partition_checker_fails_closed_without_echoing_collection_details(
    bad_result,
    capsys,
):
    module = _module()
    queue = iter((bad_result, bad_result, bad_result, bad_result))

    result = module.main(
        [],
        root=ROOT_DIR,
        python=sys.executable,
        run=lambda *_args, **_kwargs: next(queue),
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "pytest partition validation failed\n"
    assert "private" not in captured.err


def test_partition_checker_isolated_cli_ignores_ambient_sitecustomize(tmp_path):
    canary = "PYTEST-PARTITION-SITECUSTOMIZE-CANARY"
    (tmp_path / "sitecustomize.py").write_text(
        f"print({canary!r})\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            os.fspath(ROOT_DIR / "scripts" / "check_pytest_partitions.py"),
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
    assert "Traceback" not in completed.stderr


def test_partition_checker_cli_rejects_unknown_argument_without_echo(capsys):
    module = _module()
    canary = "PRIVATE-PARTITION-ARGUMENT-CANARY"

    result = module.main(["--unknown", canary])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "pytest partition validation failed\n"
    assert canary not in captured.err


def test_collect_timeout_kills_descendant_process_before_side_effect(tmp_path):
    module = _module()
    marker = tmp_path / "descendant-side-effect"
    child = (
        "import pathlib,time; time.sleep(1.0); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    )

    completed = module._run_process_tree(
        [sys.executable, "-c", parent],
        cwd=ROOT_DIR,
        env=os.environ.copy(),
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=0.2,
        check=False,
    )

    time.sleep(1.2)
    assert completed.returncode == module.COLLECT_TIMEOUT_RETURN_CODE
    assert not marker.exists()


def test_collect_timeout_kills_descendant_after_parent_exits(tmp_path):
    module = _module()
    marker = tmp_path / "orphan-descendant-side-effect"
    child = (
        "import pathlib,time; time.sleep(4.0); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped')"
    )
    parent = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}])"
    )

    started_at = time.monotonic()
    completed = module._run_process_tree(
        [sys.executable, "-c", parent],
        cwd=ROOT_DIR,
        env=os.environ.copy(),
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=2.0,
        check=False,
    )

    elapsed = time.monotonic() - started_at
    assert completed.returncode == 0
    assert elapsed < 2.0
    time.sleep(4.2)
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX temporary output contract")
def test_posix_process_launch_failure_closes_temporary_outputs(monkeypatch):
    module = _module()
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("launch failed")),
    )
    created = []

    def temporary_file(**_kwargs):
        output = io.StringIO()
        created.append(output)
        return output

    monkeypatch.setattr(module.tempfile, "TemporaryFile", temporary_file)

    with pytest.raises(OSError, match="launch failed"):
        module._run_process_tree(
            [sys.executable, "-c", "pass"],
            timeout=1.0,
            capture_output=True,
            check=False,
        )

    assert len(created) == 2
    assert all(output.closed for output in created)


@pytest.mark.skipif(os.name == "nt", reason="POSIX temporary output contract")
def test_posix_output_read_failure_closes_temporary_outputs(monkeypatch):
    module = _module()

    class FailingRead(io.StringIO):
        def read(self, *_args, **_kwargs):
            raise UnicodeError("decode failed")

    outputs = [FailingRead(), io.StringIO()]
    pending_outputs = iter(outputs)

    class Process:
        pid = 12345
        returncode = 0

        @staticmethod
        def wait(timeout):
            assert timeout in (1.0, 30)
            return 0

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(
        module.tempfile,
        "TemporaryFile",
        lambda **_kwargs: next(pending_outputs),
    )
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(module.os, "killpg", lambda *_args: None)
    with pytest.raises(UnicodeError, match="decode failed"):
        module._run_process_tree(
            [sys.executable, "-c", "pass"],
            timeout=1.0,
            capture_output=True,
            check=False,
        )

    assert all(output.closed for output in outputs)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group contract")
def test_posix_process_group_cleanup_failure_is_nonzero(monkeypatch):
    module = _module()
    outputs = [io.StringIO(), io.StringIO()]
    pending_outputs = iter(outputs)

    class Process:
        pid = 12345
        returncode = 0

        @staticmethod
        def wait(timeout):
            assert timeout in (1.0, 30)
            return 0

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(
        module.tempfile,
        "TemporaryFile",
        lambda **_kwargs: next(pending_outputs),
    )
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("denied")),
    )

    completed = module._run_process_tree(
        [sys.executable, "-c", "pass"],
        timeout=1.0,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == module.PROCESS_CLEANUP_RETURN_CODE
    assert all(output.closed for output in outputs)


def test_partition_checker_has_total_deadline_across_all_collections():
    module = _module()
    response = _completed("tests/test_a.py::test_a\n1 test collected\n")
    calls = []
    clock_values = iter((0.0, 0.0, module.TOTAL_TIMEOUT_SECONDS + 1.0))

    with pytest.raises(module.PartitionError, match="^pytest partition validation failed$"):
        module.check_partitions(
            root=ROOT_DIR,
            python=sys.executable,
            run=lambda *_args, **_kwargs: calls.append(1) or response,
            clock=lambda: next(clock_values),
        )

    assert calls == [1]
