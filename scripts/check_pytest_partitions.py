"""Verify that pytest execution partitions are disjoint and exhaustive."""

from __future__ import annotations

import sys
import os as _bootstrap_os

if getattr(getattr(_bootstrap_os, "__spec__", None), "origin", None) != "frozen":
    raise RuntimeError("pytest partition bootstrap is unavailable")


def _bootstrap_realpath(value: str) -> str:
    return _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(value))


_script_path = _bootstrap_realpath(__file__)
_script_dir = _bootstrap_os.path.dirname(_script_path)
_root_entry = _bootstrap_os.path.dirname(_script_dir)
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
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from typing import NoReturn  # noqa: E402


ROOT_DIR = Path(_root_entry)
MYSQL_TEST = "tests/integration/test_mysql_migrations.py"
MYSQL_TESTS = (
    MYSQL_TEST,
    "tests/integration/test_mysql_sync_lock_order.py",
)
COLLECT_TIMEOUT_SECONDS = 300
TOTAL_TIMEOUT_SECONDS = 900
COLLECT_TIMEOUT_RETURN_CODE = 124
PROCESS_CLEANUP_RETURN_CODE = 125
_NODE_ID = re.compile(r"^tests/[A-Za-z0-9_./\\-]+\.py::[^\x00-\x1f\x7f]+$")
_SUMMARY = re.compile(
    r"^(?:(?P<selected>\d+)/)?(?P<total>\d+) tests? collected"
    r"(?: \(\d+ deselected\))?(?: in [0-9.]+s)?$"
)


class PartitionError(RuntimeError):
    """Raised when collection cannot prove the required partition."""


class _WindowsJob:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "job creation failed")
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle,
            9,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "job configuration failed")
        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._handle = handle
        self._wintypes = wintypes

    def assign_current_process(self) -> None:
        if not self._kernel32.AssignProcessToJobObject(
            self._handle,
            self._kernel32.GetCurrentProcess(),
        ):
            raise OSError(self._ctypes.get_last_error(), "job assignment failed")

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _run_owned_windows_job(arguments: Sequence[str]) -> NoReturn:
    job = _WindowsJob()
    exit_code = 125
    try:
        job.assign_current_process()
        child = subprocess.Popen(arguments)
        exit_code = child.wait()
    except (OSError, subprocess.SubprocessError):
        exit_code = 125
    finally:
        # Closing our kill-on-close job while we are still a member kills this
        # launcher with status zero before it can return the child's status.
        # Process exit closes the handle and reaps descendants after preserving
        # the intended status. Normalize Windows DWORD codes for os._exit's int.
        signed_code = ((exit_code + 2**31) % 2**32) - 2**31
        os._exit(signed_code)


def _terminate_process_tree(process: subprocess.Popen[str]) -> bool:
    cleanup_succeeded = True
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            cleanup_succeeded = False
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            cleanup_succeeded = False
    return cleanup_succeeded


def _run_process_tree(
    arguments: Sequence[str],
    *,
    timeout: float,
    capture_output: bool,
    check: bool,
    **kwargs,
) -> subprocess.CompletedProcess[str]:
    popen_kwargs = dict(kwargs)
    output_files = None
    if capture_output and os.name != "nt":
        encoding = popen_kwargs.get("encoding") or "utf-8"
        errors = popen_kwargs.get("errors") or "strict"
        stdout_file = tempfile.TemporaryFile(
            mode="w+", encoding=encoding, errors=errors
        )
        try:
            stderr_file = tempfile.TemporaryFile(
                mode="w+", encoding=encoding, errors=errors
            )
        except BaseException:
            stdout_file.close()
            raise
        output_files = (stdout_file, stderr_file)
        popen_kwargs.update(stdout=output_files[0], stderr=output_files[1])
    elif capture_output:
        popen_kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    process_arguments = list(arguments)
    if os.name == "nt":
        process_arguments = [
            sys.executable,
            "-I",
            __file__,
            "--_owned-job-launch",
            *process_arguments,
        ]
    try:
        process = subprocess.Popen(process_arguments, **popen_kwargs)
    except BaseException:
        if output_files is not None:
            for output_file in output_files:
                output_file.close()
        raise
    if os.name != "nt":
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            cleanup_succeeded = _terminate_process_tree(process)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
        if output_files is None:
            stdout, stderr = None, None
        else:
            stdout_file, stderr_file = output_files
            try:
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout, stderr = stdout_file.read(), stderr_file.read()
            finally:
                stdout_file.close()
                stderr_file.close()
        completed = subprocess.CompletedProcess(
            arguments,
            (
                COLLECT_TIMEOUT_RETURN_CODE
                if timed_out
                else process.returncode
                if cleanup_succeeded
                else PROCESS_CLEANUP_RETURN_CODE
            ),
            stdout,
            stderr,
        )
        if check:
            completed.check_returncode()
        return completed
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            stdout, stderr = "", ""
        return subprocess.CompletedProcess(
            arguments,
            COLLECT_TIMEOUT_RETURN_CODE,
            stdout,
            stderr,
        )
    completed = subprocess.CompletedProcess(
        arguments,
        process.returncode,
        stdout,
        stderr,
    )
    if check:
        completed.check_returncode()
    return completed


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PYTEST_") or name.startswith("PYTHON"):
            environment.pop(name)
    environment.update(
        {
            "DEBUG": "false",
            "GPU_PREFLIGHT_MODE": "off",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    return environment


def _parse_collection(completed: subprocess.CompletedProcess[str]) -> set[str]:
    if completed.returncode != 0 or completed.stderr:
        raise PartitionError("pytest partition validation failed")
    lines = [line for line in completed.stdout.splitlines() if line]
    if not lines:
        raise PartitionError("pytest partition validation failed")
    summary_match = _SUMMARY.fullmatch(lines[-1])
    if summary_match is None:
        raise PartitionError("pytest partition validation failed")
    nodes = lines[:-1]
    if any(_NODE_ID.fullmatch(node) is None for node in nodes):
        raise PartitionError("pytest partition validation failed")
    if len(nodes) != len(set(nodes)):
        raise PartitionError("pytest partition validation failed")
    selected = summary_match.group("selected") or summary_match.group("total")
    if int(selected) != len(nodes):
        raise PartitionError("pytest partition validation failed")
    return set(nodes)


def _collect(
    arguments: Sequence[str],
    *,
    root: Path,
    python: str,
    run: Callable[..., subprocess.CompletedProcess[str]],
    timeout: float,
) -> set[str]:
    completed = run(
        [
            python,
            "-I",
            "-m",
            "pytest",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            "-q",
            "--strict-config",
            "--strict-markers",
            *arguments,
        ],
        cwd=root,
        env=_clean_environment(),
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return _parse_collection(completed)


def check_partitions(
    *,
    root: Path = ROOT_DIR,
    python: str = sys.executable,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_process_tree,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, int]:
    deadline = clock() + TOTAL_TIMEOUT_SECONDS

    def collect(arguments: Sequence[str]) -> set[str]:
        remaining = deadline - clock()
        if remaining <= 0:
            raise PartitionError("pytest partition validation failed")
        return _collect(
            arguments,
            root=root,
            python=python,
            run=run,
            timeout=min(float(COLLECT_TIMEOUT_SECONDS), remaining),
        )

    all_nodes = collect(("tests",))
    backend = collect(
        (
            "-m",
            "not host_tools",
            *(f"--ignore={test_path}" for test_path in MYSQL_TESTS),
            "tests",
        ),
    )
    host_tools = collect(("-m", "host_tools", "tests"))
    mysql_migrations = collect(MYSQL_TESTS)
    partitions = (backend, host_tools, mysql_migrations)
    if any(not partition for partition in partitions):
        raise PartitionError("pytest partition validation failed")
    if any(
        left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    ):
        raise PartitionError("pytest partition validation failed")
    if set().union(*partitions) != all_nodes:
        raise PartitionError("pytest partition validation failed")
    return {
        "all": len(all_nodes),
        "backend": len(backend),
        "host_tools": len(host_tools),
        "mysql_migrations": len(mysql_migrations),
    }


def _parser() -> argparse.ArgumentParser:
    class FixedArgumentParser(argparse.ArgumentParser):
        def error(self, _message: str) -> None:
            raise PartitionError("pytest partition validation failed")

    return FixedArgumentParser(description=__doc__, allow_abbrev=False)


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = ROOT_DIR,
    python: str = sys.executable,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_process_tree,
) -> int:
    try:
        _parser().parse_args(argv)
        summary = check_partitions(root=root, python=python, run=run)
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        print("pytest partition validation failed", file=sys.stderr)
        return 1
    except (PartitionError, OSError, subprocess.SubprocessError):
        print("pytest partition validation failed", file=sys.stderr)
        return 1
    print(
        " ".join(
            f"{name}={summary[name]}"
            for name in ("all", "backend", "host_tools", "mysql_migrations")
        )
    )
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_owned-job-launch"]:
        raise SystemExit(_run_owned_windows_job(sys.argv[2:]))
    raise SystemExit(main())
