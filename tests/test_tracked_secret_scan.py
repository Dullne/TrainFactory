import importlib
import os
import subprocess
from pathlib import Path

import pytest


def _scanner_module():
    return importlib.import_module("scripts.scan_tracked_secrets")


def _git(repo, *arguments):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _initialize_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    return repo


@pytest.mark.host_tools
def test_scanner_reports_only_location_and_rule_for_tracked_secret(tmp_path, capsys):
    module = _scanner_module()
    repo = _initialize_repository(tmp_path)
    canary = "tracked-private-canary-value"
    secret_line = "-----BEGIN " + "PRIVATE KEY-----" + canary
    (repo / "tracked.txt").write_text(secret_line, encoding="utf-8")
    _git(repo, "add", "tracked.txt")

    result = module.main(["--repo", str(repo)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == "tracked.txt:1:private_key\n"
    assert captured.err == ""
    assert canary not in captured.out
    assert canary not in captured.err


@pytest.mark.host_tools
def test_scanner_ignores_untracked_runtime_and_binary_content(tmp_path, capsys):
    module = _scanner_module()
    repo = _initialize_repository(tmp_path)
    marker = "-----BEGIN " + "PRIVATE KEY-----"
    (repo / "safe.txt").write_text("safe", encoding="utf-8")
    (repo / "untracked.txt").write_text(marker, encoding="utf-8")
    (repo / "binary.bin").write_bytes(b"\x00" + marker.encode("utf-8"))
    runtime = repo / ".runtime"
    runtime.mkdir()
    (runtime / "tracked-by-mistake.txt").write_text(marker, encoding="utf-8")
    _git(repo, "add", "safe.txt", "binary.bin", ".runtime/tracked-by-mistake.txt")

    result = module.main(["--repo", str(repo)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.host_tools
def test_scanner_does_not_skip_tracked_backup_text(tmp_path, capsys):
    module = _scanner_module()
    repo = _initialize_repository(tmp_path)
    marker = "-----BEGIN " + "PRIVATE KEY-----"
    (repo / "tracked.bak").write_text(marker, encoding="utf-8")
    _git(repo, "add", "tracked.bak")

    result = module.main(["--repo", str(repo)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == "tracked.bak:1:private_key\n"
    assert captured.err == ""


@pytest.mark.host_tools
def test_scanner_fails_closed_for_oversized_tracked_text(tmp_path, capsys):
    module = _scanner_module()
    repo = _initialize_repository(tmp_path)
    (repo / "oversized.txt").write_bytes(b"x" * (module.MAX_TEXT_BYTES + 1))
    _git(repo, "add", "oversized.txt")

    result = module.main(["--repo", str(repo)])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "tracked secret scan failed\n"


@pytest.mark.host_tools
def test_scanner_skips_oversized_tracked_binary_from_checked_descriptor(
    tmp_path,
    capsys,
):
    module = _scanner_module()
    repo = _initialize_repository(tmp_path)
    (repo / "oversized.bin").write_bytes(b"\x00" + b"x" * module.MAX_TEXT_BYTES)
    _git(repo, "add", "oversized.bin")

    result = module.main(["--repo", str(repo)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.host_tools
def test_scanner_streams_oversized_binary_until_nul_on_checked_descriptor(
    tmp_path,
    capsys,
):
    module = _scanner_module()
    repo = _initialize_repository(tmp_path)
    (repo / "late-nul.bin").write_bytes(b"x" * module.MAX_TEXT_BYTES + b"\x00")
    _git(repo, "add", "late-nul.bin")

    result = module.main(["--repo", str(repo)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.host_tools
def test_scanner_skips_oversized_invalid_utf8_binary_from_checked_descriptor(
    tmp_path,
    capsys,
):
    module = _scanner_module()
    repo = _initialize_repository(tmp_path)
    (repo / "invalid-utf8.bin").write_bytes(b"\xff" * (module.MAX_TEXT_BYTES + 1))
    _git(repo, "add", "invalid-utf8.bin")

    result = module.main(["--repo", str(repo)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.host_tools
def test_scanner_fails_closed_when_tracked_file_identity_changes(
    tmp_path,
    monkeypatch,
):
    module = _scanner_module()
    repo = _initialize_repository(tmp_path)
    candidate = repo / "tracked.txt"
    candidate.write_text("safe", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    original_lstat = Path.lstat
    calls = 0

    def swapped_lstat(path):
        nonlocal calls
        result = original_lstat(path)
        if path == candidate:
            calls += 1
            if calls > 1:
                values = list(result)
                values[1] += 1
                return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", swapped_lstat)

    with pytest.raises(module.ScanError):
        module.scan_repository(repo)


@pytest.mark.parametrize(
    ("path", "expected"),
    (
        ("normal/path.txt", "normal/path.txt"),
        ("line\nbreak.txt", "<unsafe-path>"),
        ("colon:name.txt", "<unsafe-path>"),
        ("escape\x1bname.txt", "<unsafe-path>"),
        ("control\u0085name.txt", "<unsafe-path>"),
        ("format\u00adname.txt", "<unsafe-path>"),
        ("separator\u2028name.txt", "<unsafe-path>"),
        ("paragraph\u2029name.txt", "<unsafe-path>"),
        ("bidi\u202ename.txt", "<unsafe-path>"),
        ("isolate\u2066name.txt", "<unsafe-path>"),
        ("mark\u200fname.txt", "<unsafe-path>"),
        ("arabic\u061cname.txt", "<unsafe-path>"),
    ),
)
def test_scanner_sanitizes_control_and_delimiter_characters_in_paths(path, expected):
    module = _scanner_module()

    assert module.safe_display_path(path) == expected


@pytest.mark.host_tools
def test_scanner_git_failure_uses_fixed_error_without_repository_path(tmp_path, capsys):
    module = _scanner_module()
    private_path = tmp_path / "private-path-canary"

    result = module.main(["--repo", str(private_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "tracked secret scan failed\n"
    assert str(private_path) not in captured.err
