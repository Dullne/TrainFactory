import os
import stat
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_factory.api.routes import generation_routes


def _checkpoint_destination(root: Path) -> tuple[Path, Path]:
    attempt_dir = root / "generation_secure-copy" / "opaque-attempt"
    return attempt_dir, attempt_dir / "checkpoint.jsonl"


def _reparse_stat(path: Path) -> SimpleNamespace:
    current = os.stat(path)
    return SimpleNamespace(
        st_mode=current.st_mode,
        st_ino=current.st_ino,
        st_dev=current.st_dev,
        st_nlink=current.st_nlink,
        st_file_attributes=(
            getattr(current, "st_file_attributes", 0)
            | getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ),
    )


def _create_directory_link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks unavailable: {symlink_error}")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"directory reparse points unavailable: {result.stderr}")


def _remove_directory_link(link: Path) -> None:
    if not os.path.lexists(link):
        return
    try:
        link.unlink()
    except (IsADirectoryError, PermissionError):
        os.rmdir(link)


def test_restart_checkpoint_copy_rejects_source_outside_managed_root(
    tmp_path,
    monkeypatch,
):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    outside_source = tmp_path / "outside-secret.jsonl"
    outside_source.write_text("outside-secret\n", encoding="utf-8")
    attempt_dir, destination = _checkpoint_destination(managed_root)
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )

    with pytest.raises(ValueError, match="managed generation storage"):
        generation_routes._copy_restart_checkpoint(
            str(outside_source),
            destination,
            expected_directory=attempt_dir,
        )

    assert not destination.exists()


def test_restart_checkpoint_copy_rejects_check_to_open_hardlink_swap(
    tmp_path,
    monkeypatch,
):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    source = managed_root / "legacy-checkpoint.jsonl"
    source.write_text("safe-checkpoint\n", encoding="utf-8")
    outside_secret = tmp_path / "outside-secret.jsonl"
    outside_secret.write_text("swapped-secret\n", encoding="utf-8")
    attempt_dir, destination = _checkpoint_destination(managed_root)
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )

    original_is_file = Path.is_file
    swapped = False

    def swap_after_regular_file_check(path: Path) -> bool:
        nonlocal swapped
        result = original_is_file(path)
        if path == source and result and not swapped:
            source.unlink()
            os.link(outside_secret, source)
            swapped = True
        return result

    monkeypatch.setattr(Path, "is_file", swap_after_regular_file_check)

    with pytest.raises(ValueError, match="changed during restart copy"):
        generation_routes._copy_restart_checkpoint(
            str(source),
            destination,
            expected_directory=attempt_dir,
        )

    assert swapped is True
    assert not destination.exists()
    assert outside_secret.read_text(encoding="utf-8") == "swapped-secret\n"


def test_restart_checkpoint_copy_rejects_source_modified_during_fd_copy(
    tmp_path,
    monkeypatch,
):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    source = managed_root / "legacy-checkpoint.jsonl"
    source.write_bytes(b"safe-checkpoint\n")
    attempt_dir, destination = _checkpoint_destination(managed_root)
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )
    original_read = os.read
    modified = False

    def append_after_first_source_read(fd: int, size: int) -> bytes:
        nonlocal modified
        data = original_read(fd, size)
        if data and not modified:
            with source.open("ab") as source_file:
                source_file.write(b"mutated-after-read\n")
                source_file.flush()
                os.fsync(source_file.fileno())
            modified = True
        return data

    monkeypatch.setattr(os, "read", append_after_first_source_read)

    with pytest.raises(ValueError, match="changed during restart copy"):
        generation_routes._copy_restart_checkpoint(
            str(source),
            destination,
            expected_directory=attempt_dir,
        )

    assert modified is True
    assert not destination.exists()
    assert list(attempt_dir.glob(".*.tmp")) == []


def test_restart_checkpoint_copy_rejects_source_leaf_symlink(
    tmp_path,
    monkeypatch,
):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    target = managed_root / "real-checkpoint.jsonl"
    target.write_text("safe-checkpoint\n", encoding="utf-8")
    source = managed_root / "linked-checkpoint.jsonl"
    try:
        os.symlink(target, source)
    except OSError:
        source.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        original_lstat = os.lstat

        def fake_leaf_symlink(path, *args, **kwargs):
            if Path(path) == source:
                current = original_lstat(path, *args, **kwargs)
                return SimpleNamespace(
                    st_mode=stat.S_IFLNK | stat.S_IMODE(current.st_mode),
                    st_ino=current.st_ino,
                    st_dev=current.st_dev,
                    st_nlink=current.st_nlink,
                    st_file_attributes=getattr(
                        current,
                        "st_file_attributes",
                        0,
                    ),
                )
            return original_lstat(path, *args, **kwargs)

        monkeypatch.setattr(os, "lstat", fake_leaf_symlink)
    attempt_dir, destination = _checkpoint_destination(managed_root)
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )

    with pytest.raises(ValueError, match="symlink|reparse"):
        generation_routes._copy_restart_checkpoint(
            str(source),
            destination,
            expected_directory=attempt_dir,
        )

    assert not destination.exists()


def test_restart_checkpoint_copy_rejects_source_parent_reparse_component(
    tmp_path,
    monkeypatch,
):
    managed_root = tmp_path / "managed"
    parent = managed_root / "reparse-parent"
    parent.mkdir(parents=True)
    source = parent / "checkpoint.jsonl"
    source.write_text("safe-checkpoint\n", encoding="utf-8")
    attempt_dir, destination = _checkpoint_destination(managed_root)
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )
    original_lstat = os.lstat

    def fake_parent_reparse(path, *args, **kwargs):
        if Path(path) == parent:
            return _reparse_stat(parent)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", fake_parent_reparse)

    with pytest.raises(ValueError, match="symlink|reparse"):
        generation_routes._copy_restart_checkpoint(
            str(source),
            destination,
            expected_directory=attempt_dir,
        )

    assert not destination.exists()


def test_restart_checkpoint_copy_rejects_destination_parent_swap(
    tmp_path,
    monkeypatch,
):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    source = managed_root / "legacy-checkpoint.jsonl"
    source.write_text("safe-checkpoint\n", encoding="utf-8")
    attempt_dir, destination = _checkpoint_destination(managed_root)
    outside_directory = tmp_path / "outside-destination"
    outside_directory.mkdir()
    parked_directory = tmp_path / "parked-attempt"
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )
    original_mkdir = Path.mkdir
    swapped = False

    def swap_after_destination_creation(path: Path, *args, **kwargs):
        nonlocal swapped
        result = original_mkdir(path, *args, **kwargs)
        if path == attempt_dir and not swapped:
            path.rename(parked_directory)
            _create_directory_link(path, outside_directory)
            swapped = True
        return result

    monkeypatch.setattr(Path, "mkdir", swap_after_destination_creation)

    try:
        with pytest.raises(ValueError, match="changed|symlink|reparse"):
            generation_routes._copy_restart_checkpoint(
                str(source),
                destination,
                expected_directory=attempt_dir,
            )
    finally:
        _remove_directory_link(attempt_dir)

    assert swapped is True
    assert not (outside_directory / destination.name).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows directory handle semantics")
@pytest.mark.parametrize("rename_level", ("attempt", "task-parent"))
def test_restart_checkpoint_copy_holds_windows_parent_anchor_through_replace(
    tmp_path,
    monkeypatch,
    rename_level,
):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    source = managed_root / "legacy-checkpoint.jsonl"
    source.write_text("safe-checkpoint\n", encoding="utf-8")
    attempt_dir, destination = _checkpoint_destination(managed_root)
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )
    replace_entered = threading.Event()
    release_replace = threading.Event()
    original_replace = os.replace

    def pause_before_replace(source_path, destination_path, *args, **kwargs):
        if Path(destination_path) == destination:
            replace_entered.set()
            assert release_replace.wait(5)
        return original_replace(source_path, destination_path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", pause_before_replace)
    results = []
    failures = []

    def copy_checkpoint():
        try:
            results.append(
                generation_routes._copy_restart_checkpoint(
                    str(source),
                    destination,
                    expected_directory=attempt_dir,
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=copy_checkpoint)
    worker.start()
    assert replace_entered.wait(5)
    rename_target = (
        attempt_dir if rename_level == "attempt" else attempt_dir.parent
    )
    parked = tmp_path / f"parked-{rename_level}"
    rename_blocked = False
    try:
        rename_target.rename(parked)
    except PermissionError:
        rename_blocked = True
    else:
        parked.rename(rename_target)
    finally:
        release_replace.set()
        worker.join(5)

    assert worker.is_alive() is False
    assert failures == []
    assert results == [str(destination)]
    assert rename_blocked is True


def test_restart_checkpoint_copy_fails_closed_without_secure_directory_anchor(
    tmp_path,
    monkeypatch,
):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    source = managed_root / "legacy-checkpoint.jsonl"
    source.write_text("safe-checkpoint\n", encoding="utf-8")
    attempt_dir, destination = _checkpoint_destination(managed_root)
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )
    monkeypatch.setattr(
        generation_routes,
        "_supports_checkpoint_directory_fd",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        generation_routes,
        "_supports_windows_checkpoint_anchor",
        lambda: False,
        raising=False,
    )

    with pytest.raises(ValueError, match="secure directory anchoring unavailable"):
        generation_routes._copy_restart_checkpoint(
            str(source),
            destination,
            expected_directory=attempt_dir,
        )

    assert not destination.exists()


@pytest.mark.parametrize("source_layout", ("legacy", "old-attempt"))
def test_restart_checkpoint_copy_accepts_managed_legacy_and_attempt_sources(
    tmp_path,
    monkeypatch,
    source_layout,
):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    if source_layout == "legacy":
        source = managed_root / "legacy-checkpoint.jsonl"
    else:
        source = (
            managed_root
            / "generation_secure-copy"
            / "old-opaque-attempt"
            / "checkpoint.jsonl"
        )
        source.parent.mkdir(parents=True)
    source.write_text(f"{source_layout}-checkpoint\n", encoding="utf-8")
    attempt_dir, destination = _checkpoint_destination(managed_root)
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(managed_root),
    )

    copied = generation_routes._copy_restart_checkpoint(
        str(source),
        destination,
        expected_directory=attempt_dir,
    )

    assert copied == str(destination)
    assert destination.read_bytes() == source.read_bytes()
