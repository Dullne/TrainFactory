from __future__ import annotations

import ast
import http.client
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import promote_web_release as module


OLD_API_ID = "sha256:" + "1" * 64
OLD_WEB_ID = "sha256:" + "2" * 64
NEW_WEB_ID = "sha256:" + "3" * 64
OLD_REVISION = "a" * 40
NEW_REVISION = "b" * 40


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    runtime = root / ".runtime"
    docker = root / "docker"
    runtime.mkdir(parents=True)
    docker.mkdir()
    (runtime / "release.env").write_text(
        "API_IMAGE=local/api:old\n"
        "WEB_IMAGE=local/web:old\n"
        f"RELEASE_REVISION={OLD_REVISION}\n"
        f"API_IMAGE_ID={OLD_API_ID}\n"
        f"WEB_IMAGE_ID={OLD_WEB_ID}\n",
        encoding="utf-8",
    )
    (runtime / "production-compose-manifest.json").write_text(
        '{"old":true}\n', encoding="utf-8"
    )
    return root


def _old_manifest(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "production",
        "project": "trainfactory",
        "gpu_mode": "compat",
        "secret_mode": "direct",
        "compose_files": [],
        "env_files": [
            {
                "role": "image-selection",
                "path": os.fspath(root / ".runtime" / "release.env"),
            }
        ],
        "referenced_files": [],
        "rollback_variant": None,
    }


def _install_fakes(monkeypatch, root: Path, *, browser_error: bool = False):
    calls: list[tuple[str, object]] = []
    old_manifest = _old_manifest(root)
    new_manifest = {**old_manifest, "prepared": True}
    new_manifest_payload = (
        json.dumps(new_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    old_values = {
        "API_IMAGE": "local/api:old",
        "WEB_IMAGE": "local/web:old",
        "RELEASE_REVISION": OLD_REVISION,
        "API_IMAGE_ID": OLD_API_ID,
        "WEB_IMAGE_ID": OLD_WEB_ID,
    }
    new_values = {
        "API_IMAGE": "local/api:old",
        "WEB_IMAGE": "local/web:new",
        "API_REVISION": OLD_REVISION,
        "WEB_REVISION": NEW_REVISION,
        "API_IMAGE_ID": OLD_API_ID,
        "WEB_IMAGE_ID": NEW_WEB_ID,
    }

    def verify_manifest(path, **_kwargs):
        payload = Path(path).read_text(encoding="utf-8")
        return new_manifest if "prepared" in payload else old_manifest

    def verify_inputs_only(path, **_kwargs):
        calls.append(("verify", Path(path).name))
        return verify_manifest(path)

    def selection_values(manifest):
        return new_values if manifest.get("prepared") else old_values

    def create_forward(**kwargs):
        calls.append(("selection", kwargs["web_image"]))
        return dict(new_values)

    def freeze_manifest(*, output, _published_identity=None, **_kwargs):
        Path(output).write_bytes(new_manifest_payload)
        metadata = Path(output).lstat()
        if _published_identity is not None:
            _published_identity.append((metadata.st_dev, metadata.st_ino))
        calls.append(("freeze", Path(output).name))
        return new_manifest

    def execute_manifest(_manifest, tail, **_kwargs):
        calls.append(("compose", tuple(tail)))

    def verify_runtime(*, expected_web_id, **_kwargs):
        calls.append(("runtime", expected_web_id))

    def browser(*_args, **_kwargs):
        calls.append(("browser", True))
        if browser_error:
            raise module.WebPromotionError("web promotion verification failed")
        return {
            "login_page_reachable": True,
            "api_health_reachable": True,
            "api_proxy_reachable": True,
            "auth_rejection_contract_verified": True,
            "training_route_reachable": True,
        }

    monkeypatch.setattr(
        module.compose_manifest, "verify_manifest_inputs", verify_manifest
    )
    monkeypatch.setattr(
        module.compose_manifest, "verify_manifest_inputs_only", verify_inputs_only
    )
    monkeypatch.setattr(module.compose_manifest, "_selection_values", selection_values)
    monkeypatch.setattr(
        module.compose_manifest,
        "build_forward_web_selection",
        create_forward,
    )
    monkeypatch.setattr(module.compose_manifest, "freeze_manifest", freeze_manifest)
    monkeypatch.setattr(
        module,
        "_forward_manifest_payload",
        lambda *_args, **_kwargs: new_manifest_payload,
    )
    monkeypatch.setattr(
        module,
        "_validate_recovery_material",
        lambda *_args, **_kwargs: (
            dict(old_values),
            dict(new_values),
            new_manifest_payload,
        ),
    )
    monkeypatch.setattr(module.compose_release, "execute_manifest", execute_manifest)
    monkeypatch.setattr(module, "_pinned_environment", lambda source: dict(source))
    monkeypatch.setattr(module, "_verify_hardened_path", lambda _path: True)
    monkeypatch.setattr(
        module,
        "_container_projection",
        lambda name, **_kwargs: {
            "id": "api-container" if name == "trainfactory-api" else "web-container",
            "image": OLD_API_ID if name == "trainfactory-api" else OLD_WEB_ID,
            "status": "running",
            "health": "healthy",
            "restart_count": 0,
            "project": "trainfactory",
            "service": name.removeprefix("trainfactory-"),
        },
    )
    monkeypatch.setattr(
        module, "_snapshot_fixed_resources", lambda **_kwargs: {"fixed": 1}
    )
    monkeypatch.setattr(
        module,
        "_verify_fixed_resources",
        lambda snapshot, **_kwargs: calls.append(("fixed", snapshot)),
    )
    monkeypatch.setattr(module, "_verify_runtime", verify_runtime)
    monkeypatch.setattr(module, "_run_browser_verifier", browser)
    return calls, old_values, new_values


def _fake_journal_payload(
    root: Path,
    *,
    old_selection: bytes,
    old_manifest: bytes,
    new_values: dict[str, str],
    phase: str = "applying",
    prepared_identity: tuple[int, int] | None = None,
    new_selection_identity: tuple[int, int] | None = None,
    new_manifest_identity: tuple[int, int] | None = None,
    previous_journal_identity: tuple[int, int] | None = None,
    previous_journal_sha256: str | None = None,
    old_selection_identity: tuple[int, int] | None = None,
    old_manifest_identity: tuple[int, int] | None = None,
    fixed_resources: dict[str, object] | None = None,
) -> bytes:
    selection_metadata = (root / ".runtime" / "release.env").lstat()
    manifest_metadata = (root / ".runtime" / "production-compose-manifest.json").lstat()
    new_manifest = {**_old_manifest(root), "prepared": True}
    new_manifest_payload = (
        json.dumps(new_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if new_manifest_identity is not None and prepared_identity is None:
        prepared_identity = (0, 0)
    if prepared_identity is not None and new_selection_identity is None:
        new_selection_identity = (0, 0)
    if new_selection_identity is not None and previous_journal_identity is None:
        previous_journal_identity = (0, 0)
        previous_journal_sha256 = "0" * 64
    extra = {}
    if new_selection_identity is not None:
        extra["new_selection_identity"] = new_selection_identity
    if new_manifest_identity is not None:
        extra["new_manifest_identity"] = new_manifest_identity
    if previous_journal_identity is not None:
        extra["previous_journal_identity"] = previous_journal_identity
    if previous_journal_sha256 is not None:
        extra["previous_journal_sha256"] = previous_journal_sha256
    return module._journal_payload(
        old_selection=old_selection,
        old_manifest=old_manifest,
        new_selection=module._selection_payload(new_values),
        new_manifest=new_manifest_payload,
        old_web_id=OLD_WEB_ID,
        target_web_id=NEW_WEB_ID,
        web_revision=NEW_REVISION,
        api_image_id=OLD_API_ID,
        api_container_id="api-container",
        fixed_resources=({"fixed": 1} if fixed_resources is None else fixed_resources),
        nonce="c" * 32,
        old_selection_identity=(
            old_selection_identity
            if old_selection_identity is not None
            else (selection_metadata.st_dev, selection_metadata.st_ino)
        ),
        old_manifest_identity=(
            old_manifest_identity
            if old_manifest_identity is not None
            else (manifest_metadata.st_dev, manifest_metadata.st_ino)
        ),
        phase=phase,
        prepared_identity=prepared_identity,
        **extra,
    )


def test_local_engine_pin_removes_all_mutable_docker_routing(monkeypatch):
    observed = []

    def gate(_run, environment):
        observed.append(dict(environment))
        return "npipe:////./pipe/dockerDesktopLinuxEngine"

    monkeypatch.setattr(module.verify_deployment, "_require_local_engine", gate)
    source = {
        "PATH": os.environ.get("PATH", ""),
        "DOCKER_HOST": "npipe:////./pipe/dockerDesktopLinuxEngine",
        "DOCKER_CONTEXT": "mutable-context",
        "DOCKER_CONFIG": "C:/attacker-config",
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": "C:/attacker-certs",
    }

    pinned = module._pinned_environment(source)

    assert len(observed) == 1
    for name, value in source.items():
        assert observed[0][name] == value
    assert pinned["DOCKER_HOST"] == "npipe:////./pipe/dockerDesktopLinuxEngine"
    for name in (
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    ):
        assert name not in pinned


def test_remote_engine_is_rejected_before_manifest_or_browser(tmp_path, monkeypatch):
    root = _root(tmp_path)
    calls = []
    monkeypatch.setattr(
        module,
        "_pinned_environment",
        lambda _source: (_ for _ in ()).throw(
            module.WebPromotionError("web promotion Docker engine is invalid")
        ),
    )
    monkeypatch.setattr(
        module.compose_manifest,
        "verify_manifest_inputs_only",
        lambda *_args, **_kwargs: calls.append("manifest"),
    )
    monkeypatch.setattr(
        module, "_run_browser_verifier", lambda **_kwargs: calls.append("browser")
    )

    with pytest.raises(
        module.WebPromotionError, match="^web promotion Docker engine is invalid$"
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={
                "PATH": os.environ.get("PATH", ""),
                "DOCKER_HOST": "ssh://remote.invalid",
            },
        )

    assert calls == []
    assert not (root / ".runtime" / "web-promotion.lock").exists()


def test_unlocked_persistent_lock_file_allows_a_new_promotion(tmp_path, monkeypatch):
    root = _root(tmp_path)
    lock_path = root / ".runtime" / "web-promotion.lock"
    descriptor = module._acquire_process_lock(lock_path)
    module._release_process_lock(descriptor)
    calls, _old, _new = _install_fakes(monkeypatch, root)

    result = module.promote_web_release(
        root=root,
        web_image="local/web:new",
        web_image_id=NEW_WEB_ID,
        web_revision=NEW_REVISION,
        base_environment={"PATH": os.environ.get("PATH", "")},
    )

    assert result["success"] is True
    assert lock_path.is_file()
    assert any(name == "compose" for name, _value in calls)


def test_busy_process_lock_rejects_before_any_state_mutation(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection_before = (runtime / "release.env").read_bytes()
    manifest_before = (runtime / "production-compose-manifest.json").read_bytes()
    calls, _old, _new = _install_fakes(monkeypatch, root)
    descriptor = module._acquire_process_lock(runtime / "web-promotion.lock")
    try:
        with pytest.raises(
            module.WebPromotionError, match="^web promotion is already active$"
        ):
            module.promote_web_release(
                root=root,
                web_image="local/web:new",
                web_image_id=NEW_WEB_ID,
                web_revision=NEW_REVISION,
                base_environment={"PATH": os.environ.get("PATH", "")},
            )
    finally:
        module._release_process_lock(descriptor)

    assert (runtime / "release.env").read_bytes() == selection_before
    assert (
        runtime / "production-compose-manifest.json"
    ).read_bytes() == manifest_before
    assert calls == []


def test_process_lock_rejects_hardlink_before_writing_or_hardening(tmp_path):
    root = _root(tmp_path)
    outside = tmp_path / "outside-lock-target"
    outside.write_bytes(b"")
    lock_path = root / ".runtime" / "web-promotion.lock"
    os.link(outside, lock_path)

    with pytest.raises(
        module.WebPromotionError, match="^web promotion lock is invalid$"
    ):
        module._acquire_process_lock(lock_path)

    assert outside.read_bytes() == b""
    assert outside.stat().st_nlink == 2


def test_run_private_uses_the_supplied_root_as_cwd(tmp_path, monkeypatch):
    root = tmp_path / "alternate-root"
    observed = []

    def run(*_args, **kwargs):
        observed.append(kwargs["cwd"])
        return subprocess.CompletedProcess([], 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)

    assert module._run_private(("private-tool",), root=root, environment={}) == "ok\n"
    assert observed == [root]


def test_unlink_owned_preserves_a_replacement_racing_the_claim(tmp_path, monkeypatch):
    target = tmp_path / "owned.json"
    displaced = tmp_path / "displaced-owned.json"
    replacement = tmp_path / "foreign.json"
    target.write_bytes(b"owned\n")
    metadata = target.lstat()
    original_move = module._move_noreplace
    foreign = b"foreign\n"
    injected = False

    def replace_before_claim(source, destination):
        nonlocal injected
        if Path(source) == target and not injected:
            injected = True
            os.replace(target, displaced)
            replacement.write_bytes(foreign)
            os.replace(replacement, target)
        return original_move(source, destination)

    monkeypatch.setattr(module, "_move_noreplace", replace_before_claim)

    assert module._unlink_owned(target, (metadata.st_dev, metadata.st_ino)) is False
    assert injected is True
    assert target.read_bytes() == foreign
    assert displaced.read_bytes() == b"owned\n"


def test_unlink_owned_never_deletes_a_replacement_after_quarantine_check(
    tmp_path, monkeypatch
):
    target = tmp_path / "owned.json"
    displaced = tmp_path / "displaced-owned.json"
    replacement = tmp_path / "foreign.json"
    target.write_bytes(b"owned\n")
    metadata = target.lstat()
    original_lstat = Path.lstat
    foreign = b"foreign\n"
    injected = False

    def replace_after_check(path, *args, **kwargs):
        nonlocal injected
        observed = original_lstat(path, *args, **kwargs)
        candidate = Path(path)
        if candidate.name.endswith(".delete") and not injected:
            injected = True
            os.replace(candidate, displaced)
            replacement.write_bytes(foreign)
            os.replace(replacement, candidate)
        return observed

    monkeypatch.setattr(Path, "lstat", replace_after_check)

    assert module._unlink_owned(target, (metadata.st_dev, metadata.st_ino)) is False
    assert injected is True
    assert target.read_bytes() == foreign
    assert displaced.read_bytes() == b"owned\n"


def test_unlink_owned_rejects_a_foreign_reoccupation_of_the_source_path(
    tmp_path, monkeypatch
):
    target = tmp_path / "owned.json"
    foreign_source = tmp_path / "foreign.json"
    target.write_bytes(b"owned\n")
    metadata = target.lstat()
    original_move = module._move_noreplace
    injected = False

    def reoccupy_source_after_move(source, destination):
        nonlocal injected
        original_move(source, destination)
        if Path(source) == target and Path(destination).name.endswith(".delete"):
            injected = True
            foreign_source.write_bytes(b"foreign\n")
            os.replace(foreign_source, target)

    monkeypatch.setattr(module, "_move_noreplace", reoccupy_source_after_move)

    assert module._unlink_owned(target, (metadata.st_dev, metadata.st_ino)) is False
    assert injected is True
    assert target.read_bytes() == b"foreign\n"
    (retired,) = tuple(tmp_path.glob("*.delete"))
    assert retired.read_bytes() == b"owned\n"


def test_unlink_owned_rolls_back_when_retired_count_changes_during_move(
    tmp_path, monkeypatch
):
    target = tmp_path / "owned.json"
    target.write_bytes(b"owned\n")
    metadata = target.lstat()
    original_move = module._move_noreplace
    injected = False

    def insert_retired_file_after_move(source, destination):
        nonlocal injected
        original_move(source, destination)
        if Path(destination).name.endswith(".delete") and not injected:
            injected = True
            (tmp_path / f'.foreign.{"f" * 32}.delete').write_bytes(b"foreign\n")

    monkeypatch.setattr(module, "_MAX_RETIRED_FILES", 1)
    monkeypatch.setattr(module, "_move_noreplace", insert_retired_file_after_move)

    assert not module._unlink_owned(target, (metadata.st_dev, metadata.st_ino))
    assert injected is True
    assert target.read_bytes() == b"owned\n"
    assert len(tuple(tmp_path.glob("*.delete"))) == 1


def test_unlink_owned_rolls_back_when_source_grows_during_move(tmp_path, monkeypatch):
    target = tmp_path / "owned.json"
    target.write_bytes(b"owned\n")
    metadata = target.lstat()
    original_move = module._move_noreplace
    grown = b"grown-after-capacity-check\n"

    def grow_after_move(source, destination):
        original_move(source, destination)
        if Path(destination).name.endswith(".delete"):
            Path(destination).write_bytes(grown)

    monkeypatch.setattr(module, "_move_noreplace", grow_after_move)

    assert not module._unlink_owned(target, (metadata.st_dev, metadata.st_ino))
    assert target.read_bytes() == grown
    assert tuple(tmp_path.glob("*.delete")) == ()


def test_unlink_owned_removes_the_exact_single_link_without_permanent_retirement(
    tmp_path,
):
    target = tmp_path / "owned.json"
    payload = b"private-cleanup-canary\n"
    target.write_bytes(payload)
    metadata = target.lstat()
    identity = (metadata.st_dev, metadata.st_ino)

    assert module._unlink_owned(target, identity) is True

    assert not target.exists()
    assert tuple(tmp_path.glob("*.delete")) == ()


@pytest.mark.skipif(os.name != "nt", reason="Windows provides handle-bound deletion")
def test_unlink_owned_blocks_replacement_during_handle_bound_delete(tmp_path):
    target = tmp_path / "owned.json"
    displaced = tmp_path / "displaced-owned.json"
    replacement = tmp_path / "foreign.json"
    target.write_bytes(b"owned\n")
    metadata = target.lstat()
    attempted = False

    def attempt_replacement(quarantine):
        nonlocal attempted
        attempted = True
        replacement.write_bytes(b"foreign\n")
        with pytest.raises(OSError):
            os.replace(quarantine, displaced)

    assert module._unlink_owned(
        target,
        (metadata.st_dev, metadata.st_ino),
        _before_bound_delete=attempt_replacement,
    )

    assert attempted is True
    assert not target.exists()
    assert not displaced.exists()
    assert replacement.read_bytes() == b"foreign\n"
    assert tuple(tmp_path.glob("*.delete")) == ()


def test_unlink_owned_removes_only_the_named_hardlink(tmp_path):
    target = tmp_path / "owned.json"
    retained = tmp_path / "retained.json"
    payload = b"private-hardlink-canary\n"
    target.write_bytes(payload)
    os.link(target, retained)
    metadata = target.lstat()
    assert metadata.st_nlink == 2

    assert module._unlink_owned(target, (metadata.st_dev, metadata.st_ino)) is True

    assert not target.exists()
    assert retained.read_bytes() == payload
    assert tuple(tmp_path.glob("*.delete")) == ()


def test_unlink_owned_refuses_when_retired_capacity_is_exhausted(tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    target.write_bytes(b"target\n")
    target_metadata = target.lstat()
    retired = tmp_path / f'.foreign.{"f" * 32}.delete'
    retired.write_bytes(b"retired\n")
    monkeypatch.setattr(module, "_MAX_RETIRED_FILES", 1)

    assert not module._unlink_owned(
        target, (target_metadata.st_dev, target_metadata.st_ino)
    )

    assert target.read_bytes() == b"target\n"
    assert retired.read_bytes() == b"retired\n"
    assert len(tuple(tmp_path.glob("*.delete"))) == 1


def test_move_noreplace_fails_closed_without_an_atomic_platform_primitive(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.write_bytes(b"owned\n")
    link_calls = []

    with pytest.raises(OSError):
        with monkeypatch.context() as patch:
            patch.setattr(module.os, "name", "posix")
            patch.setattr(module.sys, "platform", "darwin")
            patch.setattr(
                module.os,
                "link",
                lambda *_args, **_kwargs: link_calls.append(True),
            )
            module._move_noreplace(source, target)

    assert link_calls == []
    assert source.read_bytes() == b"owned\n"
    assert not target.exists()


def test_atomic_replace_syncs_parent_directory_after_publish(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_bytes(b"old\n")
    metadata = target.lstat()
    parent_metadata = tmp_path.lstat()
    parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
    observed = []
    original_fsync = module.os.fsync

    def record_bound_directory_sync(descriptor):
        opened = module.os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) == parent_identity:
            observed.append(tmp_path)
            return
        original_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", record_bound_directory_sync)
    monkeypatch.setattr(
        module, "_sync_directory", lambda path: observed.append(Path(path))
    )

    identity = module._atomic_replace(
        target,
        b"new\n",
        expected_identity=(metadata.st_dev, metadata.st_ino),
        expected_payload=b"old\n",
        claim_path=tmp_path / "state.claim",
    )

    assert target.read_bytes() == b"new\n"
    assert identity == (target.lstat().st_dev, target.lstat().st_ino)
    assert observed == [tmp_path] * 5


def test_atomic_replace_records_candidate_before_canonical_mutation(tmp_path):
    target = tmp_path / "state.json"
    claim = tmp_path / "state.claim"
    target.write_bytes(b"old\n")
    old_metadata = target.lstat()
    old_identity = (old_metadata.st_dev, old_metadata.st_ino)
    recorded = []

    def record_candidate(identity):
        assert target.read_bytes() == b"old\n"
        assert not claim.exists()
        recorded.append(identity)

    identity = module._atomic_replace(
        target,
        b"new\n",
        expected_identity=old_identity,
        expected_payload=b"old\n",
        claim_path=claim,
        _before_publish=record_candidate,
    )

    assert recorded == [identity]
    assert target.read_bytes() == b"new\n"


def test_atomic_replace_can_retain_original_inode_for_transaction_rollback(tmp_path):
    target = tmp_path / "state.json"
    claim = tmp_path / "state.claim"
    target.write_bytes(b"old\n")
    old_metadata = target.lstat()
    old_identity = (old_metadata.st_dev, old_metadata.st_ino)

    module._atomic_replace(
        target,
        b"new\n",
        expected_identity=old_identity,
        expected_payload=b"old\n",
        claim_path=claim,
        retain_claim=True,
    )

    assert target.read_bytes() == b"new\n"
    assert claim.read_bytes() == b"old\n"
    assert (claim.lstat().st_dev, claim.lstat().st_ino) == old_identity


def test_atomic_replace_never_publishes_a_foreign_temporary_replacement(
    tmp_path, monkeypatch
):
    target = tmp_path / "state.json"
    claim = tmp_path / "state.claim"
    displaced_candidate = tmp_path / "displaced-candidate.json"
    foreign_source = tmp_path / "foreign.json"
    target.write_bytes(b"old\n")
    metadata = target.lstat()
    original_move = module._move_noreplace
    injected = False

    def replace_candidate_before_publish(source, destination):
        nonlocal injected
        if (
            Path(destination) == target
            and Path(source).suffix == ".tmp"
            and not injected
        ):
            injected = True
            os.replace(source, displaced_candidate)
            foreign_source.write_bytes(b"foreign\n")
            os.replace(foreign_source, source)
        return original_move(source, destination)

    monkeypatch.setattr(module, "_move_noreplace", replace_candidate_before_publish)

    with pytest.raises(
        module.WebPromotionError, match="^web promotion state update failed$"
    ):
        module._atomic_replace(
            target,
            b"new\n",
            expected_identity=(metadata.st_dev, metadata.st_ino),
            expected_payload=b"old\n",
            claim_path=claim,
        )

    assert injected is True
    assert target.read_bytes() == b"old\n"
    assert displaced_candidate.read_bytes() == b"new\n"
    (foreign_temporary,) = tuple(tmp_path.glob("*.tmp"))
    assert foreign_temporary.read_bytes() == b"foreign\n"
    assert not claim.exists()


def test_atomic_publish_does_not_leave_a_payload_bearing_tombstone(tmp_path):
    target = tmp_path / "state.json"

    identity = module._atomic_replace(target, b"new\n", require_absent=True)

    assert target.read_bytes() == b"new\n"
    assert identity == (target.lstat().st_dev, target.lstat().st_ino)
    assert tuple(tmp_path.glob("*.delete")) == ()


def test_commit_journal_final_retirement_failure_keeps_recovery_evidence(
    tmp_path, monkeypatch
):
    journal = tmp_path / "web-promotion-active.json"
    claim = tmp_path / f'.web-promotion-{"c" * 32}-journal.claim'
    retiring = module._journal_retiring_path(claim)
    payload = b'{"journal":true}\n'
    journal.write_bytes(payload)
    metadata = journal.lstat()
    original_move = module._move_noreplace
    rejected = False

    def reject_final_retirement(source, target):
        nonlocal rejected
        if Path(source) == retiring and Path(target).name.endswith(".delete"):
            rejected = True
            raise OSError("simulated retirement failure")
        return original_move(source, target)

    monkeypatch.setattr(module, "_move_noreplace", reject_final_retirement)

    with pytest.raises(
        module.WebPromotionError, match="^web promotion cleanup failed$"
    ):
        module._commit_journal(
            canonical_path=journal,
            claim_path=claim,
            expected_identity=(metadata.st_dev, metadata.st_ino),
            payload=payload,
            verify_state=lambda: None,
            error_message="web promotion cleanup failed",
        )

    assert rejected is True
    assert not journal.exists()
    assert not claim.exists()
    assert retiring.read_bytes() == payload


def test_commit_journal_uses_the_shared_bound_delete_primitive(tmp_path, monkeypatch):
    journal = tmp_path / "web-promotion-active.json"
    claim = tmp_path / f'.web-promotion-{"c" * 32}-journal.claim'
    payload = b'{"journal":true}\n'
    journal.write_bytes(payload)
    metadata = journal.lstat()
    original_delete = module._delete_isolated_owned
    deleted = []

    def record_delete(path, identity, **kwargs):
        deleted.append((Path(path), identity, kwargs.get("expected_payload")))
        return original_delete(path, identity, **kwargs)

    monkeypatch.setattr(module, "_delete_isolated_owned", record_delete)

    module._commit_journal(
        canonical_path=journal,
        claim_path=claim,
        expected_identity=(metadata.st_dev, metadata.st_ino),
        payload=payload,
        verify_state=lambda: None,
        error_message="web promotion cleanup failed",
    )

    assert len(deleted) == 1
    assert deleted[0][0].name.endswith(".delete")
    assert deleted[0][2] == payload


def test_commit_journal_does_not_probe_external_state_after_durable_delete(tmp_path):
    journal = tmp_path / "web-promotion-active.json"
    claim = tmp_path / f'.web-promotion-{"c" * 32}-journal.claim'
    retiring = module._journal_retiring_path(claim)
    payload = b'{"journal":true}\n'
    journal.write_bytes(payload)
    metadata = journal.lstat()
    checks = 0

    def verify_while_recovery_evidence_exists():
        nonlocal checks
        checks += 1
        recovery_entries = (journal, claim, retiring, *tmp_path.glob("*.delete"))
        if not any(path.exists() for path in recovery_entries):
            raise module.WebPromotionError("external state changed after commit")

    module._commit_journal(
        canonical_path=journal,
        claim_path=claim,
        expected_identity=(metadata.st_dev, metadata.st_ino),
        payload=payload,
        verify_state=verify_while_recovery_evidence_exists,
        error_message="web promotion cleanup failed",
    )

    assert checks > 0
    assert tuple(tmp_path.glob("*journal*")) == ()
    assert tuple(tmp_path.glob("*.delete")) == ()


def test_commit_journal_preserves_reoccupation_after_bound_delete(
    tmp_path, monkeypatch
):
    journal = tmp_path / "web-promotion-active.json"
    claim = tmp_path / f'.web-promotion-{"c" * 32}-journal.claim'
    payload = b'{"journal":true}\n'
    foreign = b'{"foreign":true}\n'
    journal.write_bytes(payload)
    metadata = journal.lstat()
    original_delete = module._delete_isolated_owned
    reoccupied = []

    def delete_then_reoccupy(path, identity, **kwargs):
        outcome = original_delete(path, identity, **kwargs)
        if outcome == module._DELETE_REMOVED_DURABLE:
            Path(path).write_bytes(foreign)
            reoccupied.append(Path(path))
        return outcome

    monkeypatch.setattr(module, "_delete_isolated_owned", delete_then_reoccupy)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion terminal cleanup state is invalid$",
    ):
        module._commit_journal(
            canonical_path=journal,
            claim_path=claim,
            expected_identity=(metadata.st_dev, metadata.st_ino),
            payload=payload,
            verify_state=lambda: None,
            error_message="web promotion cleanup failed",
        )

    assert len(reoccupied) == 1
    assert reoccupied[0].read_bytes() == foreign
    assert not journal.exists()
    assert not claim.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX unlink outcome semantics")
def test_posix_bound_delete_reports_unsynced_after_post_unlink_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "owned.json"
    target.write_bytes(b"owned\n")
    metadata = target.lstat()
    original_fstat = module.os.fstat

    def fail_after_unlink(descriptor):
        if not target.exists():
            raise OSError("simulated post-unlink failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(module.os, "fstat", fail_after_unlink)

    outcome = module._delete_isolated_owned(
        target,
        (metadata.st_dev, metadata.st_ino),
        expected_size=metadata.st_size,
        expected_payload=b"owned\n",
    )

    assert outcome == module._DELETE_REMOVED_UNSYNCED
    assert not target.exists()


def test_commit_journal_reoccupation_fails_and_preserves_recovery_evidence(
    tmp_path, monkeypatch
):
    journal = tmp_path / "web-promotion-active.json"
    claim = tmp_path / f'.web-promotion-{"c" * 32}-journal.claim'
    payload = b'{"journal":true}\n'
    foreign = b'{"foreign":true}\n'
    journal.write_bytes(payload)
    metadata = journal.lstat()
    retiring = module._journal_retiring_path(claim)
    original_move = module._move_noreplace
    injected = False

    def occupy_canonical_during_retiring_transition(source, target):
        nonlocal injected
        if Path(source) == claim and Path(target) == retiring and not injected:
            injected = True
            replacement = tmp_path / "foreign-journal.json"
            replacement.write_bytes(foreign)
            os.replace(replacement, journal)
        return original_move(source, target)

    monkeypatch.setattr(
        module, "_move_noreplace", occupy_canonical_during_retiring_transition
    )

    with pytest.raises(
        module.WebPromotionError, match="^web promotion cleanup failed$"
    ):
        module._commit_journal(
            canonical_path=journal,
            claim_path=claim,
            expected_identity=(metadata.st_dev, metadata.st_ino),
            payload=payload,
            verify_state=lambda: None,
            error_message="web promotion cleanup failed",
        )

    assert injected is True
    assert journal.read_bytes() == foreign
    assert not claim.exists()
    assert retiring.read_bytes() == payload


def test_commit_journal_restores_a_foreign_source_replacement(tmp_path, monkeypatch):
    journal = tmp_path / "web-promotion-active.json"
    claim = tmp_path / f'.web-promotion-{"c" * 32}-journal.claim'
    retiring = module._journal_retiring_path(claim)
    displaced = tmp_path / "displaced-journal.json"
    replacement = tmp_path / "foreign-journal.json"
    payload = b'{"journal":true}\n'
    foreign = b'{"foreign":true}\n'
    journal.write_bytes(payload)
    metadata = journal.lstat()
    original_move = module._move_noreplace
    injected = False

    def replace_source_before_move(source, target):
        nonlocal injected
        if Path(source) == journal and Path(target) == claim and not injected:
            injected = True
            os.replace(journal, displaced)
            replacement.write_bytes(foreign)
            os.replace(replacement, journal)
        return original_move(source, target)

    monkeypatch.setattr(module, "_move_noreplace", replace_source_before_move)

    with pytest.raises(
        module.WebPromotionError, match="^web promotion cleanup failed$"
    ):
        module._commit_journal(
            canonical_path=journal,
            claim_path=claim,
            expected_identity=(metadata.st_dev, metadata.st_ino),
            payload=payload,
            verify_state=lambda: None,
            error_message="web promotion cleanup failed",
        )

    assert injected is True
    assert journal.read_bytes() == foreign
    assert displaced.read_bytes() == payload
    assert not claim.exists()
    assert not retiring.exists()


def test_commit_journal_state_drift_keeps_a_discoverable_recovery_journal(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    claim = module._claim_path(runtime, "c" * 32, "journal")
    payload = _fake_journal_payload(
        root,
        old_selection=(runtime / "release.env").read_bytes(),
        old_manifest=(runtime / "production-compose-manifest.json").read_bytes(),
        new_values=new_values,
    )
    journal.write_bytes(payload)
    metadata = journal.lstat()
    checks = 0

    def drift_after_journal_is_isolated():
        nonlocal checks
        checks += 1
        if checks >= 4:
            raise module.WebPromotionError("web promotion commit state drifted")

    with pytest.raises(
        module.WebPromotionError, match="^web promotion cleanup failed$"
    ):
        module._commit_journal(
            canonical_path=journal,
            claim_path=claim,
            expected_identity=(metadata.st_dev, metadata.st_ino),
            payload=payload,
            verify_state=drift_after_journal_is_isolated,
            error_message="web promotion cleanup failed",
        )

    recovery = module._recovery_journal_path(runtime, journal)
    assert recovery is not None
    assert recovery.read_bytes() == payload


def test_execute_web_checks_the_lock_at_the_side_effect_boundary(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.compose_release,
        "execute_manifest",
        lambda *_args, **_kwargs: calls.append(True),
    )

    def reject_invalid_lock():
        raise module.WebPromotionError("web promotion lock is invalid")

    with pytest.raises(
        module.WebPromotionError, match="^web promotion lock is invalid$"
    ):
        module._execute_web(
            tmp_path / "manifest.json",
            root=tmp_path,
            environment={},
            mutation_guard=reject_invalid_lock,
        )

    assert calls == []


def test_production_http_verifier_is_fixed_and_accepts_only_typed_summary(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    valid = {
        "success": True,
        "code": "OK",
        "login_page": True,
        "api_health": True,
        "api_proxy": True,
        "auth_rejection": True,
        "training_route": True,
    }
    observed = []

    def run_private(arguments, **kwargs):
        observed.append((tuple(arguments), kwargs))
        return json.dumps(valid, separators=(",", ":")) + "\n"

    monkeypatch.setattr(module, "_run_private", run_private)

    assert module._run_browser_verifier(root=root, environment={}) == {
        "login_page_reachable": True,
        "api_health_reachable": True,
        "api_proxy_reachable": True,
        "auth_rejection_contract_verified": True,
        "training_route_reachable": True,
    }
    ((arguments, kwargs),) = observed
    assert arguments[:6] == (
        "docker",
        "exec",
        "trainfactory-api",
        "python",
        "-I",
        "-c",
    )
    assert len(arguments) == 7
    verifier_source = arguments[-1]
    assert "http.client" in verifier_source
    request_calls = {
        (
            node.func.id,
            node.args[0].value,
            node.args[1].value,
        )
        for node in ast.walk(ast.parse(verifier_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_api_request", "_web_request"}
        and len(node.args) >= 2
        and all(isinstance(argument, ast.Constant) for argument in node.args[:2])
    }
    assert request_calls == {
        ("_web_request", "GET", "/api/auth/config"),
        ("_api_request", "GET", "/health"),
        ("_api_request", "GET", "/api/auth/config"),
        ("_api_request", "GET", "/api/auth/me"),
        ("_web_request", "GET", "/api/auth/me"),
    }
    html_paths = {
        node.args[0].value
        for node in ast.walk(ast.parse(verifier_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_html"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
    }
    assert html_paths == {"/login", "/training"}
    assert '_web_request("GET", path' in verifier_source
    assert "playwright" not in verifier_source.lower()
    assert "DEFAULT_ADMIN" not in verifier_source
    assert "/api/auth/login" not in verifier_source
    assert verifier_source.count("access_token") == 1
    assert "SimpleCookie" not in verifier_source
    assert "node" not in arguments
    assert os.fspath(root) not in verifier_source
    assert kwargs["root"] == root
    assert kwargs["timeout"] == 90

    invalid = dict(valid)
    invalid["auth_rejection"] = 1
    monkeypatch.setattr(
        module,
        "_run_private",
        lambda *_args, **_kwargs: json.dumps(invalid, separators=(",", ":")) + "\n",
    )
    with pytest.raises(
        module.WebPromotionError, match="^web promotion verification failed$"
    ):
        module._run_browser_verifier(root=root, environment={})


def test_production_http_verifier_uses_only_an_unprivileged_rejection_probe(
    monkeypatch, capsys
):
    html = (
        b"<html><head><title>TrainFactory - AI Training Platform</title></head>"
        b'<body><div id="root"></div></body></html>'
    )
    calls = []

    class Response:
        def __init__(self, status, headers, payload):
            self.status = status
            self._headers = headers
            self._payload = payload

        def getheaders(self):
            return list(self._headers)

        def read(self, _limit):
            return self._payload

    class Connection:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.response = None

        def request(self, method, path, body=None, headers=None):
            headers = dict(headers or {})
            calls.append((self.host, self.port, method, path, body, headers))
            if self.host == "train-factory-web" and path in {"/login", "/training"}:
                assert self.port == 80 and method == "GET" and body is None
                assert "Cookie" not in headers
                if path in {"/login", "/training"}:
                    self.response = Response(
                        200, [("Content-Type", "text/html; charset=utf-8")], html
                    )
                return
            if path == "/health":
                assert self.host == "127.0.0.1" and self.port == 18000
                self.response = Response(
                    200,
                    [("Content-Type", "application/json")],
                    b'{"status":"healthy","version":"0.1.0"}',
                )
            elif path == "/api/auth/config":
                self.response = Response(
                    200,
                    [("Content-Type", "application/json")],
                    json.dumps(
                        {
                            "self_registration_enabled": False,
                            "direct_storage_registration_enabled": False,
                        }
                    ).encode(),
                )
            elif path == "/api/auth/me":
                assert method == "GET" and body is None
                assert headers["Cookie"].startswith(
                    "access_token=web-promotion-invalid-"
                )
                self.response = Response(
                    401,
                    [
                        ("Content-Type", "application/json"),
                        ("WWW-Authenticate", "Bearer"),
                    ],
                    b'{"detail":"Invalid or expired token"}',
                )
            else:
                raise AssertionError((self.host, method, path))

        def getresponse(self):
            assert self.response is not None
            return self.response

        def close(self):
            return None

    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    monkeypatch.setenv("DEFAULT_ADMIN_USERNAME", "stale-bootstrap-admin")
    monkeypatch.setenv("DEFAULT_ADMIN_PASSWORD", "stale-bootstrap-password")
    monkeypatch.setenv("API_PORT", "18000")

    exec(module._PRODUCTION_HTTP_VERIFIER, {})

    assert json.loads(capsys.readouterr().out) == {
        "success": True,
        "code": "OK",
        "login_page": True,
        "api_health": True,
        "api_proxy": True,
        "auth_rejection": True,
        "training_route": True,
    }
    assert [
        (host, method, path) for host, _port, method, path, _body, _headers in calls
    ] == [
        ("train-factory-web", "GET", "/login"),
        ("127.0.0.1", "GET", "/health"),
        ("127.0.0.1", "GET", "/api/auth/config"),
        ("train-factory-web", "GET", "/api/auth/config"),
        ("127.0.0.1", "GET", "/api/auth/me"),
        ("train-factory-web", "GET", "/api/auth/me"),
        ("train-factory-web", "GET", "/training"),
    ]


def test_success_persists_mixed_selection_and_canonical_manifest(tmp_path, monkeypatch):
    root = _root(tmp_path)
    calls, _old, new = _install_fakes(monkeypatch, root)

    result = module.promote_web_release(
        root=root,
        web_image="local/web:new",
        web_image_id=NEW_WEB_ID,
        web_revision=NEW_REVISION,
        base_environment={"PATH": os.environ.get("PATH", "")},
    )

    assert result["success"] is True
    assert result["api_revision"] == OLD_REVISION
    assert result["web_revision"] == NEW_REVISION
    assert (root / ".runtime" / "release.env").read_text(encoding="utf-8") == "".join(
        f"{key}={value}\n" for key, value in new.items()
    )
    manifest = json.loads(
        (root / ".runtime" / "production-compose-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["prepared"] is True
    compose_calls = [value for name, value in calls if name == "compose"]
    assert compose_calls == [
        (
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            "600",
            "train-factory-web",
        )
    ]
    assert calls.index(
        ("verify", f".web-promotion-{NEW_REVISION}-manifest.json")
    ) < calls.index(("compose", compose_calls[0]))
    assert not (root / ".runtime" / "web-promotion-active.json").exists()


def test_browser_failure_restores_exact_old_state_and_forces_old_web(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    old_selection = (root / ".runtime" / "release.env").read_bytes()
    old_manifest = (root / ".runtime" / "production-compose-manifest.json").read_bytes()
    calls, _old, _new = _install_fakes(monkeypatch, root, browser_error=True)

    with pytest.raises(
        module.WebPromotionError, match="^web promotion failed; old web restored$"
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert (root / ".runtime" / "release.env").read_bytes() == old_selection
    assert (
        root / ".runtime" / "production-compose-manifest.json"
    ).read_bytes() == old_manifest
    compose_calls = [value for name, value in calls if name == "compose"]
    assert compose_calls == [
        (
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            "600",
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
            "train-factory-web",
        ),
    ]
    assert ("runtime", OLD_WEB_ID) in calls


def test_live_rollback_rechecks_journal_after_restoring_old_files(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, _new = _install_fakes(monkeypatch, root, browser_error=True)
    original_restore = module._restore_transaction_target
    restored = 0
    foreign = b'{"foreign":true}\n'

    def restore_then_replace_journal(*args, **kwargs):
        nonlocal restored
        result = original_restore(*args, **kwargs)
        restored += 1
        if restored == 2:
            replacement = runtime / "foreign-journal.json"
            replacement.write_bytes(foreign)
            os.replace(replacement, runtime / "web-promotion-active.json")
        return result

    monkeypatch.setattr(
        module, "_restore_transaction_target", restore_then_replace_journal
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert restored == 2
    assert (runtime / "web-promotion-active.json").read_bytes() == foreign
    assert len([item for item in calls if item[0] == "compose"]) == 1


def test_final_journal_unlink_failure_preserves_committed_state_for_recovery(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    original_move = module._move_noreplace
    rejected = False

    def reject_committing_journal_retirement(source, target):
        nonlocal rejected
        if Path(source).name.endswith("-journal.retiring") and Path(
            target
        ).name.endswith(".delete"):
            try:
                phase = json.loads(Path(source).read_bytes())["phase"]
            except (OSError, KeyError, ValueError):
                phase = None
            if phase == "committing":
                rejected = True
                raise OSError("simulated retirement failure")
        return original_move(source, target)

    monkeypatch.setattr(module, "_move_noreplace", reject_committing_journal_retirement)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion committed; cleanup is incomplete$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert rejected is True
    assert (runtime / "release.env").read_bytes() == module._selection_payload(
        new_values
    )
    assert (runtime / "production-compose-manifest.json").read_bytes() == (
        module._forward_manifest_payload(None)
    )
    assert len([item for item in calls if item[0] == "compose"]) == 1
    canonical_journal = runtime / "web-promotion-active.json"
    recovery_journal = module._recovery_journal_path(runtime, canonical_journal)
    assert recovery_journal is not None
    assert recovery_journal.name.endswith("-journal.retiring")
    assert json.loads(recovery_journal.read_bytes())["phase"] == "committing"


def test_committing_retiring_journal_is_recovered_after_process_crash(
    tmp_path, monkeypatch
):
    class SimulatedCrash(BaseException):
        pass

    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    original_move = module._move_noreplace
    crashed = False

    def crash_after_retiring_journal_is_published(source, target):
        nonlocal crashed
        original_move(source, target)
        if (
            not crashed
            and Path(source).name.endswith("-journal.claim")
            and Path(target).name.endswith("-journal.retiring")
        ):
            try:
                phase = json.loads(Path(target).read_bytes())["phase"]
            except (OSError, KeyError, ValueError):
                phase = None
            if phase == "committing":
                crashed = True
                raise SimulatedCrash

    monkeypatch.setattr(
        module, "_move_noreplace", crash_after_retiring_journal_is_published
    )

    with pytest.raises(SimulatedCrash):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    canonical_journal = runtime / "web-promotion-active.json"
    recovery_journal = module._recovery_journal_path(runtime, canonical_journal)
    assert crashed is True
    assert recovery_journal is not None
    assert recovery_journal.name.endswith("-journal.retiring")
    assert json.loads(recovery_journal.read_bytes())["phase"] == "committing"

    monkeypatch.setattr(module, "_move_noreplace", original_move)
    module._recover(
        root=root,
        journal_path=recovery_journal,
        selection_path=runtime / "release.env",
        manifest_path=runtime / "production-compose-manifest.json",
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert (runtime / "release.env").read_bytes() == module._selection_payload(
        new_values
    )
    assert (runtime / "production-compose-manifest.json").read_bytes() == (
        module._forward_manifest_payload(None)
    )
    assert module._recovery_journal_path(runtime, canonical_journal) is None
    assert len([item for item in calls if item[0] == "compose"]) == 2


def test_journal_predecessor_cleanup_failure_rolls_back_before_web(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    old_selection = (runtime / "release.env").read_bytes()
    old_manifest = (runtime / "production-compose-manifest.json").read_bytes()
    calls, _old, _new = _install_fakes(monkeypatch, root)
    original_unlink = module._unlink_owned
    rejected = False

    def reject_first_journal_cleanup(path, identity, **kwargs):
        nonlocal rejected
        if path.name.endswith("-journal.claim") and not rejected:
            rejected = True
            return False
        return original_unlink(path, identity, **kwargs)

    monkeypatch.setattr(module, "_unlink_owned", reject_first_journal_cleanup)

    with pytest.raises(
        module.WebPromotionError, match="^web promotion failed; old web restored$"
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert rejected is True
    assert (runtime / "release.env").read_bytes() == old_selection
    assert (runtime / "production-compose-manifest.json").read_bytes() == old_manifest
    assert [item for item in calls if item[0] == "compose"] == []


def test_journal_replacement_is_preserved_and_prevents_success(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    original_browser = module._run_browser_verifier
    foreign = b'{"foreign":true}\n'

    def replace_journal(**kwargs):
        result = original_browser(**kwargs)
        replacement = runtime / "foreign-journal.json"
        replacement.write_bytes(foreign)
        os.replace(replacement, runtime / "web-promotion-active.json")
        return result

    monkeypatch.setattr(module, "_run_browser_verifier", replace_journal)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert (runtime / "release.env").read_bytes() == module._selection_payload(
        new_values
    )
    assert (runtime / "production-compose-manifest.json").read_bytes() == (
        module._forward_manifest_payload(None)
    )
    assert (runtime / "web-promotion-active.json").read_bytes() == foreign
    assert len([item for item in calls if item[0] == "compose"]) == 1


def test_prepared_output_replacement_is_preserved_and_prevents_success(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    old_selection = (runtime / "release.env").read_bytes()
    old_manifest = (runtime / "production-compose-manifest.json").read_bytes()
    calls, _old, _new = _install_fakes(monkeypatch, root)
    original_browser = module._run_browser_verifier
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    foreign = b'{"foreign-prepared":true}\n'

    def replace_prepared(**kwargs):
        result = original_browser(**kwargs)
        replacement = runtime / "foreign-prepared.json"
        replacement.write_bytes(foreign)
        os.replace(replacement, prepared)
        return result

    monkeypatch.setattr(module, "_run_browser_verifier", replace_prepared)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert (runtime / "release.env").read_bytes() == old_selection
    assert (runtime / "production-compose-manifest.json").read_bytes() == old_manifest
    assert prepared.read_bytes() == foreign
    assert len([item for item in calls if item[0] == "compose"]) == 2


def test_selection_swap_after_final_check_is_not_overwritten(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    manifest_before = manifest.read_bytes()
    calls, _old, _new_values = _install_fakes(monkeypatch, root)
    original_move = module._move_noreplace
    foreign = b"EXTERNAL_SELECTION=true\n"
    injected = False

    def replace_after_check(path, claim):
        nonlocal injected
        if Path(path) == selection and not injected:
            injected = True
            replacement = runtime / "foreign-selection.env"
            replacement.write_bytes(foreign)
            os.replace(replacement, selection)
        return original_move(path, claim)

    monkeypatch.setattr(module, "_move_noreplace", replace_after_check)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert injected is True
    assert selection.read_bytes() == foreign
    assert manifest.read_bytes() == manifest_before
    assert (runtime / "web-promotion-active.json").is_file()
    assert not any(name == "compose" for name, _value in calls)


def test_publish_identity_survives_same_payload_replacement_before_final_read(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    journal = runtime / "web-promotion-active.json"
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    new_selection = module._selection_payload(new_values)
    original_atomic = module._atomic_replace
    original_stable = module._stable_bytes
    published_lists = []
    injected = False

    def capture_published_identity(path, payload, **kwargs):
        if Path(path) == selection and payload == new_selection:
            published_lists.append(kwargs.get("_published_identity"))
        return original_atomic(path, payload, **kwargs)

    def replace_after_publish(path, *, max_bytes):
        nonlocal injected
        candidate = Path(path)
        selection_claims = tuple(runtime.glob("*-selection.claim"))
        if (
            not injected
            and candidate == selection
            and selection_claims
            and selection.read_bytes() == new_selection
        ):
            injected = True
            replacement = runtime / "same-payload-foreign.env"
            replacement.write_bytes(new_selection)
            os.replace(replacement, selection)
        return original_stable(path, max_bytes=max_bytes)

    monkeypatch.setattr(module, "_atomic_replace", capture_published_identity)
    monkeypatch.setattr(module, "_stable_bytes", replace_after_publish)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert injected is True
    assert len(published_lists) == 1
    assert isinstance(published_lists[0], list)
    assert len(published_lists[0]) == 1
    assert selection.read_bytes() == new_selection
    assert journal.is_file()
    assert not any(name == "compose" for name, _value in calls)


def test_canonical_drift_during_journal_commit_cannot_report_success(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    journal = runtime / "web-promotion-active.json"
    _install_fakes(monkeypatch, root)
    original_unlink = module._unlink_owned_payload
    foreign = b"EXTERNAL_AT_COMMIT=true\n"
    injected = False

    def replace_selection_before_journal_cleanup(path, *args, **kwargs):
        nonlocal injected
        candidate = Path(path)
        if not injected and (
            candidate == journal or candidate.name.endswith("-journal.claim")
        ):
            injected = True
            replacement = runtime / "foreign-selection.env"
            replacement.write_bytes(foreign)
            os.replace(replacement, selection)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        module, "_unlink_owned_payload", replace_selection_before_journal_cleanup
    )

    with pytest.raises(module.WebPromotionError):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert injected is True
    assert selection.read_bytes() == foreign
    assert journal.is_file()


def test_cleaned_selection_claim_reoccupation_cannot_report_success(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    journal = runtime / "web-promotion-active.json"
    _install_fakes(monkeypatch, root)
    original_unlink = module._unlink_owned_payload
    injected = False

    def reoccupy_after_success(path, *args, **kwargs):
        nonlocal injected
        removed = original_unlink(path, *args, **kwargs)
        candidate = Path(path)
        if removed and candidate.name.endswith("-selection.claim") and not injected:
            injected = True
            candidate.write_bytes(b"foreign-claim\n")
        return removed

    monkeypatch.setattr(module, "_unlink_owned_payload", reoccupy_after_success)

    with pytest.raises(module.WebPromotionError):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert injected is True
    assert journal.is_file()
    (foreign_claim,) = tuple(runtime.glob("*-selection.claim"))
    assert foreign_claim.read_bytes() == b"foreign-claim\n"


def test_active_journal_persists_all_owned_identities_before_web_execution(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    _calls, _old, _new = _install_fakes(monkeypatch, root)
    original_replace = module._atomic_replace
    journal = runtime / "web-promotion-active.json"
    journal_payloads = []

    def count_journal_publish(path, payload, **kwargs):
        if Path(path) == journal:
            journal_payloads.append(json.loads(payload))
        return original_replace(path, payload, **kwargs)

    monkeypatch.setattr(module, "_atomic_replace", count_journal_publish)

    result = module.promote_web_release(
        root=root,
        web_image="local/web:new",
        web_image_id=NEW_WEB_ID,
        web_revision=NEW_REVISION,
        base_environment={"PATH": os.environ.get("PATH", "")},
    )

    assert result["success"] is True
    assert len(journal_payloads) == 5
    assert journal_payloads[0]["phase"] == "applying"
    assert journal_payloads[0]["prepared_identity"] is None
    assert journal_payloads[0]["new_selection_identity"] is None
    assert journal_payloads[0]["new_manifest_identity"] is None
    assert journal_payloads[1]["phase"] == "applying"
    assert journal_payloads[1]["prepared_identity"] is None
    assert len(journal_payloads[1]["new_selection_identity"]) == 2
    assert journal_payloads[1]["new_manifest_identity"] is None
    assert len(journal_payloads[1]["previous_journal_identity"]) == 2
    assert len(journal_payloads[1]["previous_journal_sha256"]) == 64
    assert journal_payloads[2]["phase"] == "applying"
    assert len(journal_payloads[2]["prepared_identity"]) == 2
    assert len(journal_payloads[2]["new_selection_identity"]) == 2
    assert journal_payloads[2]["new_manifest_identity"] is None
    assert len(journal_payloads[2]["previous_journal_identity"]) == 2
    assert len(journal_payloads[2]["previous_journal_sha256"]) == 64
    assert journal_payloads[3]["phase"] == "applying"
    assert len(journal_payloads[3]["new_manifest_identity"]) == 2
    assert journal_payloads[4]["phase"] == "committing"
    assert (
        journal_payloads[4]["new_manifest_identity"]
        == journal_payloads[3]["new_manifest_identity"]
    )
    assert len(journal_payloads[4]["previous_journal_identity"]) == 2
    assert len(journal_payloads[4]["previous_journal_sha256"]) == 64
    assert not journal.exists()


@pytest.mark.parametrize(
    ("stage", "identity_key"),
    (
        ("selection", "new_selection_identity"),
        ("prepared", "prepared_identity"),
        ("manifest", "new_manifest_identity"),
    ),
)
def test_crash_after_candidate_publish_is_automatically_recoverable(
    tmp_path, monkeypatch, stage, identity_key
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    targets = {"selection": selection, "prepared": prepared, "manifest": manifest}
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    old_selection_identity = (selection.lstat().st_dev, selection.lstat().st_ino)
    old_manifest_identity = (manifest.lstat().st_dev, manifest.lstat().st_ino)
    _install_fakes(monkeypatch, root)
    original_atomic = module._atomic_replace

    class SimulatedPowerLoss(BaseException):
        pass

    def crash_after_publish(path, payload, **kwargs):
        identity = original_atomic(path, payload, **kwargs)
        if Path(path) == targets[stage]:
            raise SimulatedPowerLoss
        return identity

    monkeypatch.setattr(module, "_atomic_replace", crash_after_publish)
    with pytest.raises(SimulatedPowerLoss):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )
    monkeypatch.setattr(module, "_atomic_replace", original_atomic)

    journal = runtime / "web-promotion-active.json"
    state, _journal_identity, _journal_payload = module._read_journal(journal)
    target = targets[stage]
    assert state[identity_key] == (target.lstat().st_dev, target.lstat().st_ino)

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert selection.read_bytes() == old_selection
    assert (selection.lstat().st_dev, selection.lstat().st_ino) == (
        old_selection_identity
    )
    assert manifest.read_bytes() == old_manifest
    assert (manifest.lstat().st_dev, manifest.lstat().st_ino) == old_manifest_identity
    assert not prepared.exists()
    assert not journal.exists()
    assert module._claim_inventory(runtime) == {}


def test_recovery_can_resume_after_selection_was_restored_before_second_crash(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    old_selection_identity = (selection.lstat().st_dev, selection.lstat().st_ino)
    old_manifest_identity = (manifest.lstat().st_dev, manifest.lstat().st_ino)
    _install_fakes(monkeypatch, root)
    original_atomic = module._atomic_replace

    class SimulatedPowerLoss(BaseException):
        pass

    def crash_after_manifest_publish(path, payload, **kwargs):
        identity = original_atomic(path, payload, **kwargs)
        if Path(path) == manifest:
            raise SimulatedPowerLoss
        return identity

    monkeypatch.setattr(module, "_atomic_replace", crash_after_manifest_publish)
    with pytest.raises(SimulatedPowerLoss):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )
    monkeypatch.setattr(module, "_atomic_replace", original_atomic)
    journal = runtime / "web-promotion-active.json"
    original_apply = module._apply_recovery_target
    interrupted = False

    def interrupt_after_selection(*args, **kwargs):
        nonlocal interrupted
        result = original_apply(*args, **kwargs)
        if Path(args[0]) == selection and not interrupted:
            interrupted = True
            raise SimulatedPowerLoss
        return result

    monkeypatch.setattr(module, "_apply_recovery_target", interrupt_after_selection)
    with pytest.raises(SimulatedPowerLoss):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )
    assert (selection.lstat().st_dev, selection.lstat().st_ino) == (
        old_selection_identity
    )
    assert manifest.read_bytes() != old_manifest
    monkeypatch.setattr(module, "_apply_recovery_target", original_apply)

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert selection.read_bytes() == old_selection
    assert (selection.lstat().st_dev, selection.lstat().st_ino) == (
        old_selection_identity
    )
    assert manifest.read_bytes() == old_manifest
    assert (manifest.lstat().st_dev, manifest.lstat().st_ino) == old_manifest_identity
    assert not journal.exists()
    assert module._claim_inventory(runtime) == {}


def test_applying_recovery_accepts_noop_payloads_with_original_identities(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    selection_identity = (selection.lstat().st_dev, selection.lstat().st_ino)
    manifest_identity = (manifest.lstat().st_dev, manifest.lstat().st_ino)
    _calls, old_values, _new_values = _install_fakes(monkeypatch, root)
    monkeypatch.setattr(
        module,
        "_validate_recovery_material",
        lambda *_args, **_kwargs: (dict(old_values), dict(old_values), old_manifest),
    )
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        module._journal_payload(
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_selection=old_selection,
            new_manifest=old_manifest,
            old_web_id=OLD_WEB_ID,
            target_web_id=OLD_WEB_ID,
            web_revision=OLD_REVISION,
            api_image_id=OLD_API_ID,
            api_container_id="api-container",
            fixed_resources={"fixed": 1},
            nonce="c" * 32,
            old_selection_identity=selection_identity,
            old_manifest_identity=manifest_identity,
        )
    )

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert (selection.lstat().st_dev, selection.lstat().st_ino) == selection_identity
    assert (manifest.lstat().st_dev, manifest.lstat().st_ino) == manifest_identity
    assert not journal.exists()


def test_committing_recovery_accepts_exact_applying_predecessor(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    original_unlink = module._unlink_owned_payload

    class SimulatedPowerLoss(BaseException):
        pass

    def interrupt_before_predecessor_cleanup(path, *args, **kwargs):
        candidate = Path(path)
        journal = runtime / "web-promotion-active.json"
        if candidate.name.endswith("-journal.claim") and journal.is_file():
            current = json.loads(journal.read_bytes())
            predecessor = json.loads(candidate.read_bytes())
            if current["phase"] == "committing" and predecessor["phase"] == "applying":
                raise SimulatedPowerLoss
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        module, "_unlink_owned_payload", interrupt_before_predecessor_cleanup
    )
    with pytest.raises(SimulatedPowerLoss):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    monkeypatch.setattr(module, "_unlink_owned_payload", original_unlink)
    journal = runtime / "web-promotion-active.json"
    claims = module._claim_inventory(runtime)
    journal_claim = next(
        path for (_nonce, role), path in claims.items() if role == "journal"
    )
    current, _current_identity, _current_payload = module._read_journal(journal)
    predecessor, predecessor_identity, predecessor_payload = module._read_journal(
        journal_claim
    )
    assert current["phase"] == "committing"
    assert predecessor["phase"] == "applying"
    assert module._is_journal_predecessor(
        current,
        predecessor,
        predecessor_identity=predecessor_identity,
        predecessor_payload=predecessor_payload,
    )

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert selection.read_bytes() == module._selection_payload(new_values)
    assert manifest.read_bytes() == module._forward_manifest_payload(None)
    assert not journal.exists()
    assert module._claim_inventory(runtime) == {}
    assert len([item for item in calls if item[0] == "compose"]) == 2


def test_committing_recovery_resumes_after_first_old_claim_was_retired(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    original_unlink = module._unlink_owned_payload

    class SimulatedPowerLoss(BaseException):
        pass

    def interrupt_after_selection_claim(path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if Path(path).name.endswith("-selection.claim"):
            raise SimulatedPowerLoss
        return result

    monkeypatch.setattr(
        module, "_unlink_owned_payload", interrupt_after_selection_claim
    )
    with pytest.raises(SimulatedPowerLoss):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )
    monkeypatch.setattr(module, "_unlink_owned_payload", original_unlink)
    journal = runtime / "web-promotion-active.json"
    state, _journal_identity, _journal_payload = module._read_journal(journal)
    assert state["phase"] == "committing"
    selection_identity = (selection.lstat().st_dev, selection.lstat().st_ino)
    manifest_identity = (manifest.lstat().st_dev, manifest.lstat().st_ino)

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert selection.read_bytes() == module._selection_payload(new_values)
    assert (selection.lstat().st_dev, selection.lstat().st_ino) == selection_identity
    assert manifest.read_bytes() == module._forward_manifest_payload(None)
    assert (manifest.lstat().st_dev, manifest.lstat().st_ino) == manifest_identity
    assert not journal.exists()
    assert module._claim_inventory(runtime) == {}
    assert len([item for item in calls if item[0] == "compose"]) == 2


def test_rollback_does_not_overwrite_external_canonical_replacement(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    calls, _old, _new = _install_fakes(monkeypatch, root)
    foreign = b"EXTERNAL_DURING_VERIFY=true\n"

    def replace_then_fail(**_kwargs):
        replacement = runtime / "foreign-selection.env"
        replacement.write_bytes(foreign)
        os.replace(replacement, selection)
        raise module.WebPromotionError("web promotion verification failed")

    monkeypatch.setattr(module, "_run_browser_verifier", replace_then_fail)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert selection.read_bytes() == foreign
    assert (runtime / "web-promotion-active.json").is_file()
    assert len([item for item in calls if item[0] == "compose"]) == 1


def test_prepared_publish_window_journal_replacement_is_preserved(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    old_manifest = (runtime / "production-compose-manifest.json").read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    original_atomic = module._atomic_replace
    foreign = b'{"foreign":true}\n'

    def replace_journal_during_prepared_publish(path, payload, **kwargs):
        if Path(path) == prepared:
            before_publish = kwargs["_before_publish"]

            def replace_after_journal(identity):
                before_publish(identity)
                replacement = runtime / "foreign-journal.json"
                replacement.write_bytes(foreign)
                os.replace(replacement, runtime / "web-promotion-active.json")

            kwargs["_before_publish"] = replace_after_journal
        return original_atomic(path, payload, **kwargs)

    monkeypatch.setattr(
        module, "_atomic_replace", replace_journal_during_prepared_publish
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert (runtime / "release.env").read_bytes() == module._selection_payload(
        new_values
    )
    assert (runtime / "production-compose-manifest.json").read_bytes() == old_manifest
    assert (runtime / "web-promotion-active.json").read_bytes() == foreign


def test_failed_rollback_preserves_recovery_journal(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _install_fakes(monkeypatch, root, browser_error=True)
    original_move = module._move_noreplace

    def reject_manifest_restore(source, target):
        if Path(source).name.endswith("-manifest.claim") and Path(target) == manifest:
            raise module.WebPromotionError("web promotion state update failed")
        return original_move(source, target)

    monkeypatch.setattr(module, "_move_noreplace", reject_manifest_restore)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert selection.read_bytes() == old_selection
    assert not manifest.exists()
    manifest_claims = tuple(runtime.glob("*-manifest.claim"))
    assert len(manifest_claims) == 1
    assert manifest_claims[0].read_bytes() == old_manifest
    assert (runtime / "web-promotion-active.json").is_file()


def test_recovery_preserves_journal_when_prepared_identity_is_unknown(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    prepared.write_bytes(b'{"unowned":true}\n')
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery cleanup failed$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert prepared.read_bytes() == b'{"unowned":true}\n'
    assert journal.is_file()


def test_recovery_does_not_overwrite_drifted_canonical_target(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    foreign = b"EXTERNAL_RECOVERY_TARGET=true\n"
    replacement = runtime / "external-recovery.env"
    replacement.write_bytes(foreign)
    os.replace(replacement, selection)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery target drifted$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert selection.read_bytes() == foreign
    assert manifest.read_bytes() == old_manifest
    assert journal.is_file()
    assert not any(name == "compose" for name, _value in calls)


def test_recovery_restores_canonical_missing_after_owned_claim_move(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    claim = module._claim_path(runtime, "c" * 32, "selection")
    module._move_noreplace(selection, claim)

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert selection.read_bytes() == old_selection
    assert manifest.read_bytes() == old_manifest
    assert not claim.exists()
    assert not journal.exists()


def test_recovery_restores_known_target_and_owned_old_claim_pair(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    old_selection_metadata = selection.lstat()
    old_manifest_metadata = manifest.lstat()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    claim = module._claim_path(runtime, "c" * 32, "selection")
    module._move_noreplace(selection, claim)
    new_selection_identity = module._atomic_replace(
        selection,
        module._selection_payload(new_values),
        require_absent=True,
    )
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
            old_selection_identity=(
                old_selection_metadata.st_dev,
                old_selection_metadata.st_ino,
            ),
            old_manifest_identity=(
                old_manifest_metadata.st_dev,
                old_manifest_metadata.st_ino,
            ),
            new_selection_identity=new_selection_identity,
        )
    )

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert selection.read_bytes() == old_selection
    assert manifest.read_bytes() == old_manifest
    assert not claim.exists()
    assert not journal.exists()


def test_recovery_commit_drift_preserves_foreign_state_and_journal(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    original_move = module._move_noreplace
    foreign = b"EXTERNAL_DURING_RECOVERY_COMMIT=true\n"
    injected = False

    def replace_selection_during_journal_retiring_transition(source, target):
        nonlocal injected
        if (
            not injected
            and Path(source).name.endswith("-journal.claim")
            and Path(target).name.endswith("-journal.retiring")
        ):
            injected = True
            replacement = runtime / "foreign-recovery.env"
            replacement.write_bytes(foreign)
            os.replace(replacement, selection)
        return original_move(source, target)

    monkeypatch.setattr(
        module, "_move_noreplace", replace_selection_during_journal_retiring_transition
    )

    with pytest.raises(module.WebPromotionError):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert injected is True
    assert selection.read_bytes() == foreign
    recovery_journal = module._recovery_journal_path(runtime, journal)
    assert recovery_journal is not None
    assert recovery_journal.name.endswith("-journal.retiring")


def test_recovery_stops_when_process_lock_identity_changes(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    old_selection_metadata = selection.lstat()
    old_manifest_metadata = manifest.lstat()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    new_selection = module._selection_payload(new_values)
    new_manifest = module._forward_manifest_payload(None)
    journal = runtime / "web-promotion-active.json"
    selection_claim = module._claim_path(runtime, "c" * 32, "selection")
    manifest_claim = module._claim_path(runtime, "c" * 32, "manifest")
    new_selection_identity = module._atomic_replace(
        selection,
        new_selection,
        expected_identity=(
            old_selection_metadata.st_dev,
            old_selection_metadata.st_ino,
        ),
        expected_payload=old_selection,
        claim_path=selection_claim,
        retain_claim=True,
    )
    new_manifest_identity = module._atomic_replace(
        manifest,
        new_manifest,
        expected_identity=(
            old_manifest_metadata.st_dev,
            old_manifest_metadata.st_ino,
        ),
        expected_payload=old_manifest,
        claim_path=manifest_claim,
        retain_claim=True,
    )
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
            old_selection_identity=(
                old_selection_metadata.st_dev,
                old_selection_metadata.st_ino,
            ),
            old_manifest_identity=(
                old_manifest_metadata.st_dev,
                old_manifest_metadata.st_ino,
            ),
            new_selection_identity=new_selection_identity,
            new_manifest_identity=new_manifest_identity,
        )
    )
    original_apply = module._apply_recovery_target
    original_verify_lock = module._verify_process_lock
    lock_invalid = False

    def apply_then_invalidate(*args, **kwargs):
        nonlocal lock_invalid
        result = original_apply(*args, **kwargs)
        lock_invalid = True
        return result

    def reject_invalid_lock(path, descriptor):
        if lock_invalid:
            raise module.WebPromotionError("web promotion lock is invalid")
        return original_verify_lock(path, descriptor)

    monkeypatch.setattr(module, "_apply_recovery_target", apply_then_invalidate)
    monkeypatch.setattr(module, "_verify_process_lock", reject_invalid_lock)

    with pytest.raises(
        module.WebPromotionError, match="^web promotion lock is invalid$"
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert lock_invalid is True
    assert journal.is_file()
    assert manifest.read_bytes() == new_manifest


def test_recovery_rechecks_the_lock_after_manifest_validation(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    original_verify = module.compose_manifest.verify_manifest_inputs_only
    lock_valid = True

    def verify_then_invalidate(*args, **kwargs):
        nonlocal lock_valid
        result = original_verify(*args, **kwargs)
        lock_valid = False
        return result

    def guard():
        if not lock_valid:
            raise module.WebPromotionError("web promotion lock is invalid")

    monkeypatch.setattr(
        module.compose_manifest,
        "verify_manifest_inputs_only",
        verify_then_invalidate,
    )

    with pytest.raises(
        module.WebPromotionError, match="^web promotion lock is invalid$"
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
            mutation_guard=guard,
        )

    assert journal.is_file()
    assert not any(name == "compose" for name, _value in calls)


def test_recovery_rejects_alternate_journal_created_after_manifest_validation(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    journal_claim = module._claim_path(runtime, "c" * 32, "journal")
    original_verify = module.compose_manifest.verify_manifest_inputs_only
    injected = False

    def verify_then_create_alternate(*args, **kwargs):
        nonlocal injected
        result = original_verify(*args, **kwargs)
        if not injected:
            injected = True
            journal_claim.write_bytes(b'{"foreign":true}\n')
        return result

    monkeypatch.setattr(
        module.compose_manifest,
        "verify_manifest_inputs_only",
        verify_then_create_alternate,
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery state is invalid$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert injected is True
    assert journal_claim.read_bytes() == b'{"foreign":true}\n'
    assert not any(name == "compose" for name, _value in calls)


def test_applying_recovery_rechecks_old_inode_before_compose(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    original_verify = module.compose_manifest.verify_manifest_inputs_only
    replaced = False

    def verify_then_replace_manifest(*args, **kwargs):
        nonlocal replaced
        result = original_verify(*args, **kwargs)
        if not replaced:
            replaced = True
            replacement = runtime / "replacement-manifest.json"
            replacement.write_bytes(old_manifest)
            os.replace(replacement, manifest)
        return result

    monkeypatch.setattr(
        module.compose_manifest,
        "verify_manifest_inputs_only",
        verify_then_replace_manifest,
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery target drifted$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert replaced is True
    assert manifest.read_bytes() == old_manifest
    assert not any(name == "compose" for name, _value in calls)


def test_missing_journal_is_recovered_from_exact_nonce_claim(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    claim = module._claim_path(runtime, "c" * 32, "journal")
    module._move_noreplace(journal, claim)

    result = module.promote_web_release(
        root=root,
        web_image="local/web:new",
        web_image_id=NEW_WEB_ID,
        web_revision=NEW_REVISION,
        base_environment={"PATH": os.environ.get("PATH", "")},
    )

    assert result["success"] is True
    assert not claim.exists()


def test_recovery_cleans_same_inode_duplicate_journal_claim(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    claim = module._claim_path(runtime, "c" * 32, "journal")
    os.link(journal, claim)
    assert (journal.lstat().st_dev, journal.lstat().st_ino) == (
        claim.lstat().st_dev,
        claim.lstat().st_ino,
    )

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert selection.read_bytes() == old_selection
    assert manifest.read_bytes() == old_manifest
    assert not journal.exists()
    assert not claim.exists()


def test_recovery_restores_manifest_missing_after_owned_claim_move(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    claim = module._claim_path(runtime, "c" * 32, "manifest")
    module._move_noreplace(manifest, claim)

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert selection.read_bytes() == old_selection
    assert manifest.read_bytes() == old_manifest
    assert not journal.exists()
    assert not claim.exists()


def test_foreign_nonce_claim_collision_cannot_mutate_canonical_state(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    claim = module._claim_path(runtime, "c" * 32, "selection")
    foreign = b"FOREIGN_CLAIM=true\n"
    claim.write_bytes(foreign)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery claim is invalid$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert selection.read_bytes() == old_selection
    assert manifest.read_bytes() == old_manifest
    assert journal.is_file()
    assert claim.read_bytes() == foreign
    assert not any(name == "compose" for name, _value in calls)


def test_same_payload_foreign_claim_identity_cannot_mutate_canonical_state(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    claim = module._claim_path(runtime, "c" * 32, "selection")
    claim.write_bytes(old_selection)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery claim is invalid$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert selection.read_bytes() == old_selection
    assert manifest.read_bytes() == old_manifest
    assert journal.is_file()
    assert claim.read_bytes() == old_selection
    assert not any(name == "compose" for name, _value in calls)


def test_unrecorded_transaction_claim_is_preserved_and_rejected(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    claim = module._claim_path(runtime, "c" * 32, "selection")
    new_selection = module._selection_payload(new_values)
    claim.write_bytes(new_selection)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery claim is invalid$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert selection.read_bytes() == old_selection
    assert manifest.read_bytes() == old_manifest
    assert claim.read_bytes() == new_selection
    assert journal.is_file()
    assert not any(name == "compose" for name, _value in calls)


def test_unrecorded_transaction_target_is_preserved_and_rejected(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    calls, _old, new_values = _install_fakes(monkeypatch, root)
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    new_selection = module._selection_payload(new_values)
    replacement = runtime / "foreign-selection.env"
    replacement.write_bytes(new_selection)
    os.replace(replacement, selection)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery target drifted$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert selection.read_bytes() == new_selection
    assert manifest.read_bytes() == old_manifest
    assert journal.is_file()
    assert not any(name == "compose" for name, _value in calls)


def test_conditional_atomic_replace_failure_keeps_canonical_target(
    tmp_path, monkeypatch
):
    target = tmp_path / "state.json"
    target.write_bytes(b"old\n")
    metadata = target.lstat()
    claim = tmp_path / "state.claim"
    original_move = module._move_noreplace

    def fail_publish(source, destination, *args, **kwargs):
        if Path(destination) == target and Path(source).name.endswith(".tmp"):
            assert not target.exists()
            assert claim.read_bytes() == b"old\n"
            raise OSError("injected publish failure")
        return original_move(source, destination, *args, **kwargs)

    monkeypatch.setattr(module, "_move_noreplace", fail_publish)

    with pytest.raises(
        module.WebPromotionError, match="^web promotion state update failed$"
    ):
        module._atomic_replace(
            target,
            b"new\n",
            expected_identity=(metadata.st_dev, metadata.st_ino),
            expected_payload=b"old\n",
            claim_path=claim,
        )

    assert target.read_bytes() == b"old\n"
    assert not claim.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_recovery_claims_exact_prepared_payload_without_recorded_identity(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    prepared.write_bytes(module._forward_manifest_payload(None))
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery cleanup failed$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert prepared.is_file()
    assert journal.is_file()


def test_recovery_removes_exact_prepared_with_recorded_identity(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    prepared.write_bytes(module._forward_manifest_payload(None))
    metadata = prepared.lstat()
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
            prepared_identity=(metadata.st_dev, metadata.st_ino),
        )
    )

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert not prepared.exists()
    assert not journal.exists()


def test_recovery_preserves_same_payload_prepared_with_foreign_identity(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    payload = module._forward_manifest_payload(None)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    prepared.write_bytes(payload)
    metadata = prepared.lstat()
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
            prepared_identity=(metadata.st_dev, metadata.st_ino),
        )
    )
    replacement = runtime / "foreign-prepared.json"
    replacement.write_bytes(payload)
    os.replace(replacement, prepared)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery cleanup failed$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert prepared.read_bytes() == payload
    assert journal.is_file()


def test_recovery_rejects_updated_journal_with_unrecorded_predecessor_claim(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    prepared.write_bytes(module._forward_manifest_payload(None))
    prepared_metadata = prepared.lstat()
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
        )
    )
    claim = module._claim_path(runtime, "c" * 32, "journal")
    module._move_noreplace(journal, claim)
    module._atomic_replace(
        journal,
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
            prepared_identity=(prepared_metadata.st_dev, prepared_metadata.st_ino),
        ),
        require_absent=True,
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery claim is invalid$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert prepared.is_file()
    assert journal.is_file()
    assert claim.is_file()


def test_recovery_accepts_updated_journal_with_recorded_predecessor_claim(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    prepared.write_bytes(module._forward_manifest_payload(None))
    prepared_metadata = prepared.lstat()
    journal = runtime / "web-promotion-active.json"
    predecessor_payload = _fake_journal_payload(
        root,
        old_selection=old_selection,
        old_manifest=old_manifest,
        new_values=new_values,
        new_selection_identity=(0, 0),
    )
    journal.write_bytes(predecessor_payload)
    predecessor_metadata = journal.lstat()
    predecessor_identity = (
        predecessor_metadata.st_dev,
        predecessor_metadata.st_ino,
    )
    claim = module._claim_path(runtime, "c" * 32, "journal")
    module._move_noreplace(journal, claim)
    module._atomic_replace(
        journal,
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
            prepared_identity=(prepared_metadata.st_dev, prepared_metadata.st_ino),
            new_selection_identity=(0, 0),
            previous_journal_identity=predecessor_identity,
            previous_journal_sha256=module._sha256(predecessor_payload),
        ),
        require_absent=True,
    )

    module._recover(
        root=root,
        journal_path=journal,
        selection_path=selection,
        manifest_path=manifest,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert not prepared.exists()
    assert not journal.exists()
    assert not claim.exists()


def test_recovery_rejects_recorded_predecessor_replaced_by_same_payload(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    _calls, _old, new_values = _install_fakes(monkeypatch, root)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    prepared.write_bytes(module._forward_manifest_payload(None))
    prepared_metadata = prepared.lstat()
    journal = runtime / "web-promotion-active.json"
    predecessor_payload = _fake_journal_payload(
        root,
        old_selection=old_selection,
        old_manifest=old_manifest,
        new_values=new_values,
        new_selection_identity=(0, 0),
    )
    journal.write_bytes(predecessor_payload)
    predecessor_metadata = journal.lstat()
    predecessor_identity = (
        predecessor_metadata.st_dev,
        predecessor_metadata.st_ino,
    )
    claim = module._claim_path(runtime, "c" * 32, "journal")
    module._move_noreplace(journal, claim)
    module._atomic_replace(
        journal,
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new_values,
            prepared_identity=(prepared_metadata.st_dev, prepared_metadata.st_ino),
            new_selection_identity=(0, 0),
            previous_journal_identity=predecessor_identity,
            previous_journal_sha256=module._sha256(predecessor_payload),
        ),
        require_absent=True,
    )
    replacement = runtime / "foreign-predecessor.json"
    replacement.write_bytes(predecessor_payload)
    os.replace(replacement, claim)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery claim is invalid$",
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert journal.is_file()
    assert claim.read_bytes() == predecessor_payload
    assert prepared.is_file()


def test_valid_stale_journal_rejects_unknown_canonical_state(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    old_selection = selection.read_bytes()
    old_manifest = manifest.read_bytes()
    selection_metadata = selection.lstat()
    manifest_metadata = manifest.lstat()
    new_values = {
        "API_IMAGE": "local/api:old",
        "WEB_IMAGE": "local/web:new",
        "API_REVISION": OLD_REVISION,
        "WEB_REVISION": NEW_REVISION,
        "API_IMAGE_ID": OLD_API_ID,
        "WEB_IMAGE_ID": NEW_WEB_ID,
    }
    new_selection = module._selection_payload(new_values)
    new_manifest = b'{"prepared":true}\n'
    state = {
        "old_selection": old_selection,
        "new_selection": new_selection,
        "old_manifest": old_manifest,
        "old_selection_sha256": module._sha256(old_selection),
        "old_manifest_sha256": module._sha256(old_manifest),
        "new_selection_sha256": module._sha256(new_selection),
        "new_manifest_sha256": module._sha256(new_manifest),
        "old_web_id": OLD_WEB_ID,
        "target_web_id": NEW_WEB_ID,
        "target_web_revision": NEW_REVISION,
        "api_image_id": OLD_API_ID,
        "api_container_id": "api-container",
        "fixed_resources": {"fixed": 1},
        "nonce": "c" * 32,
        "old_selection_identity": (
            selection_metadata.st_dev,
            selection_metadata.st_ino,
        ),
        "old_manifest_identity": (
            manifest_metadata.st_dev,
            manifest_metadata.st_ino,
        ),
        "new_selection_identity": None,
        "new_manifest_identity": None,
    }
    selection.write_text("THIRD_RELEASE=true\n", encoding="utf-8")
    manifest.write_text('{"third":true}\n', encoding="utf-8")
    monkeypatch.setattr(module, "_manifest_material", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        module,
        "_forward_manifest_payload",
        lambda *_args, **_kwargs: new_manifest,
    )
    monkeypatch.setattr(
        module, "_snapshot_fixed_resources", lambda **_kwargs: {"fixed": 1}
    )
    monkeypatch.setattr(
        module,
        "_container_projection",
        lambda *_args, **_kwargs: {"id": "api-container", "image": OLD_API_ID},
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion recovery target drifted$",
    ):
        module._validate_recovery_material(
            state,
            root=root,
            selection_path=selection,
            manifest_path=manifest,
            environment={},
        )


def test_foreign_recovery_journal_cannot_change_canonical_bytes(tmp_path):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    selection_before = selection.read_bytes()
    manifest_before = manifest.read_bytes()
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(
        _fake_journal_payload(
            root,
            old_selection=selection_before,
            old_manifest=b"{}\n",
            new_values={
                "API_IMAGE": "local/api:old",
                "WEB_IMAGE": "local/web:new",
                "API_REVISION": OLD_REVISION,
                "WEB_REVISION": NEW_REVISION,
                "API_IMAGE_ID": OLD_API_ID,
                "WEB_IMAGE_ID": NEW_WEB_ID,
            },
        )
    )

    with pytest.raises(Exception):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert selection.read_bytes() == selection_before
    assert manifest.read_bytes() == manifest_before
    assert journal.exists()


def test_malformed_journal_uses_fixed_recovery_error_without_mutation(tmp_path):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection_before = (runtime / "release.env").read_bytes()
    manifest_before = (runtime / "production-compose-manifest.json").read_bytes()
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(b'{"broken":')

    with pytest.raises(
        module.WebPromotionError, match="^web promotion recovery state is invalid$"
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=runtime / "release.env",
            manifest_path=runtime / "production-compose-manifest.json",
            environment={},
        )

    assert (runtime / "release.env").read_bytes() == selection_before
    assert (
        runtime / "production-compose-manifest.json"
    ).read_bytes() == manifest_before


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("new_selection_sha256", "0" * 64),
        ("prepared_identity", [True, 1]),
        ("new_selection_identity", [True, 1]),
        ("new_manifest_identity", [1, 2]),
        ("previous_journal_identity", [1, 2]),
        ("previous_journal_sha256", "z" * 64),
    ),
)
def test_journal_rejects_invalid_digest_identity_and_lineage(tmp_path, field, value):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    payload = _fake_journal_payload(
        root,
        old_selection=(runtime / "release.env").read_bytes(),
        old_manifest=(runtime / "production-compose-manifest.json").read_bytes(),
        new_values={
            "API_IMAGE": "local/api:old",
            "WEB_IMAGE": "local/web:new",
            "API_REVISION": OLD_REVISION,
            "WEB_REVISION": NEW_REVISION,
            "API_IMAGE_ID": OLD_API_ID,
            "WEB_IMAGE_ID": NEW_WEB_ID,
        },
    )
    value_object = json.loads(payload)
    value_object[field] = value
    journal = runtime / "web-promotion-active.json"
    journal.write_text(
        json.dumps(value_object, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        module.WebPromotionError, match="^web promotion recovery state is invalid$"
    ):
        module._read_journal(journal)


def test_journal_publisher_rejects_payload_above_the_recovery_read_limit(tmp_path):
    root = _root(tmp_path)
    runtime = root / ".runtime"

    with pytest.raises(
        module.WebPromotionError, match="^web promotion recovery state is invalid$"
    ):
        _fake_journal_payload(
            root,
            old_selection=(runtime / "release.env").read_bytes(),
            old_manifest=(runtime / "production-compose-manifest.json").read_bytes(),
            new_values={
                "API_IMAGE": "local/api:old",
                "WEB_IMAGE": "local/web:new",
                "API_REVISION": OLD_REVISION,
                "WEB_REVISION": NEW_REVISION,
                "API_IMAGE_ID": OLD_API_ID,
                "WEB_IMAGE_ID": NEW_WEB_ID,
            },
            fixed_resources={"padding": "x" * (512 * 1024)},
        )


def test_journal_publisher_rejects_non_pair_prepared_identity(tmp_path):
    root = _root(tmp_path)
    runtime = root / ".runtime"

    with pytest.raises(
        module.WebPromotionError, match="^web promotion recovery state is invalid$"
    ):
        _fake_journal_payload(
            root,
            old_selection=(runtime / "release.env").read_bytes(),
            old_manifest=(runtime / "production-compose-manifest.json").read_bytes(),
            new_values={
                "API_IMAGE": "local/api:old",
                "WEB_IMAGE": "local/web:new",
                "API_REVISION": OLD_REVISION,
                "WEB_REVISION": NEW_REVISION,
                "API_IMAGE_ID": OLD_API_ID,
                "WEB_IMAGE_ID": NEW_WEB_ID,
            },
            prepared_identity=(1,),
        )


def test_recovery_fixed_resource_failure_happens_before_canonical_mutation(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    selection_before = selection.read_bytes()
    manifest_before = manifest.read_bytes()
    journal = runtime / "web-promotion-active.json"
    journal.write_text("placeholder\n", encoding="utf-8")
    journal_metadata = journal.lstat()
    monkeypatch.setattr(
        module,
        "_read_journal",
        lambda _path: (
            {},
            (journal_metadata.st_dev, journal_metadata.st_ino),
            b"placeholder\n",
        ),
    )
    monkeypatch.setattr(
        module,
        "_validate_recovery_material",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module.WebPromotionError("web promotion fixed resources drifted")
        ),
    )

    with pytest.raises(
        module.WebPromotionError, match="^web promotion fixed resources drifted$"
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=selection,
            manifest_path=manifest,
            environment={},
        )

    assert selection.read_bytes() == selection_before
    assert manifest.read_bytes() == manifest_before
    assert journal.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter semantics")
def test_stored_manifest_material_accepts_bootstrap_drive_case(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection_path = runtime / "release.env"
    manifest_path = runtime / "production-compose-manifest.json"
    bootstrap_path = root / "docker" / "init.sql"
    bootstrap_path.write_text("SELECT 1;\n", encoding="utf-8")
    selection_payload = selection_path.read_bytes()
    bootstrap_record = module.compose_manifest._record(
        bootstrap_path, role="database-bootstrap", sensitive=False
    )[0]
    original = bootstrap_record["path"]
    bootstrap_record["path"] = original[0].swapcase() + original[1:]
    manifest_value = {
        "schema_version": 1,
        "mode": "production",
        "project": "trainfactory",
        "gpu_mode": "required",
        "secret_mode": "direct",
        "rollback_variant": None,
        "compose_files": [],
        "env_files": [
            {
                "role": "image-selection",
                "path": os.fspath(selection_path),
                "sha256": module._sha256(selection_payload),
                "sensitive": False,
            }
        ],
        "referenced_files": [bootstrap_record],
    }
    monkeypatch.setattr(
        module.compose_manifest, "_validate_manifest_structure", lambda _value: None
    )
    monkeypatch.setattr(
        module.compose_manifest,
        "_validate_ci_manifest_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module.compose_manifest, "_validate_input_path", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module.compose_manifest,
        "_validate_role_source",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        module.compose_manifest,
        "_validate_cross_role_identity",
        lambda *_args, **_kwargs: None,
    )

    assert (
        module._manifest_material(
            module._canonical_json(manifest_value),
            selection_payload=selection_payload,
            selection_path=selection_path,
            manifest_path=manifest_path,
            root=root,
        )
        == manifest_value
    )


@pytest.mark.parametrize("drifted", ("compose", "bootstrap", "selection-record"))
def test_stored_manifest_material_rejects_selection_or_input_drift(
    tmp_path, monkeypatch, drifted
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection_path = runtime / "release.env"
    manifest_path = runtime / "production-compose-manifest.json"
    compose_path = root / "docker" / "dummy-compose.yml"
    bootstrap_path = root / "docker" / "init.sql"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    bootstrap_path.write_text("SELECT 1;\n", encoding="utf-8")
    selection_payload = selection_path.read_bytes()
    compose_record = module.compose_manifest._record(
        compose_path, role="base", sensitive=False
    )[0]
    bootstrap_record = module.compose_manifest._record(
        bootstrap_path, role="database-bootstrap", sensitive=False
    )[0]
    selection_record = {
        "role": "image-selection",
        "path": os.fspath(selection_path),
        "sha256": module._sha256(selection_payload),
        "sensitive": False,
    }
    manifest_value = {
        "schema_version": 1,
        "mode": "production",
        "project": "trainfactory",
        "gpu_mode": "required",
        "secret_mode": "files",
        "rollback_variant": None,
        "compose_files": [compose_record],
        "env_files": [selection_record],
        "referenced_files": [bootstrap_record],
    }
    monkeypatch.setattr(
        module.compose_manifest, "_validate_manifest_structure", lambda _value: None
    )
    monkeypatch.setattr(
        module.compose_manifest,
        "_validate_ci_manifest_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module.compose_manifest, "_validate_input_path", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module.compose_manifest, "_validate_compose_source", lambda _payload: None
    )
    monkeypatch.setattr(
        module.compose_manifest,
        "_validate_role_source",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        module.compose_manifest,
        "_validate_cross_role_identity",
        lambda *_args, **_kwargs: None,
    )

    payload = module._canonical_json(manifest_value)
    assert (
        module._manifest_material(
            payload,
            selection_payload=selection_payload,
            selection_path=selection_path,
            manifest_path=manifest_path,
            root=root,
        )
        == manifest_value
    )

    if drifted == "compose":
        compose_path.write_text("services: {changed: {}}\n", encoding="utf-8")
    elif drifted == "bootstrap":
        bootstrap_path.write_text("SELECT 2;\n", encoding="utf-8")
    else:
        selection_record["sha256"] = "0" * 64

    with pytest.raises(
        module.WebPromotionError, match="^web promotion recovery state is invalid$"
    ):
        module._manifest_material(
            module._canonical_json(manifest_value),
            selection_payload=selection_payload,
            selection_path=selection_path,
            manifest_path=manifest_path,
            root=root,
        )


def test_prepared_publish_without_write_ahead_hook_cannot_report_success(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    old_selection = (runtime / "release.env").read_bytes()
    old_manifest = (runtime / "production-compose-manifest.json").read_bytes()
    calls, _old, _new = _install_fakes(monkeypatch, root)
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    original_atomic = module._atomic_replace

    def omit_prepared_hook(path, payload, **kwargs):
        if Path(path) == prepared:
            kwargs.pop("_before_publish")
        return original_atomic(path, payload, **kwargs)

    monkeypatch.setattr(module, "_atomic_replace", omit_prepared_hook)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; old web restored$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert (runtime / "release.env").read_bytes() == old_selection
    assert (runtime / "production-compose-manifest.json").read_bytes() == old_manifest
    assert not any(item[0] == "compose" for item in calls)


def _resource_probe_fakes(
    monkeypatch,
    *,
    volume_names,
    fixed_container_names=None,
    foreign_endpoint=False,
    foreign_web_alias=False,
    web_present=True,
    network_labels_valid=True,
    volume_labels_valid=True,
    untrusted_padding=0,
    fixed_restart_count=0,
    web_restart_count=0,
):
    fixed_container_names = frozenset(
        module._FIXED_CONTAINERS
        if fixed_container_names is None
        else fixed_container_names
    )
    fixed_ids = {name: f"id-{name}" for name in fixed_container_names}
    web_id = "id-trainfactory-web"

    def projection(name, **_kwargs):
        if name == "trainfactory-web" and not web_present:
            raise AssertionError("fixed-resource snapshot probed the mutable Web")
        service = {
            "trainfactory-api": "train-factory-api",
            "trainfactory-web": "train-factory-web",
        }.get(name, name.removeprefix("trainfactory-"))
        return {
            "id": web_id if name == "trainfactory-web" else fixed_ids[name],
            "image": OLD_WEB_ID if name == "trainfactory-web" else OLD_API_ID,
            "status": "running",
            "health": "healthy",
            "restart_count": (
                web_restart_count if name == "trainfactory-web" else fixed_restart_count
            ),
            "project": "trainfactory",
            "service": service,
        }

    endpoints = {
        identifier: {
            "Name": name,
            "EndpointID": f"endpoint-{identifier}",
            "MacAddress": "02:42:ac:12:00:02",
            "IPv4Address": "172.18.0.2/16",
            "IPv6Address": "",
        }
        for name, identifier in fixed_ids.items()
    }
    if web_present:
        endpoints[web_id] = {
            "Name": "trainfactory-web",
            "EndpointID": "endpoint-web",
            "MacAddress": "02:42:ac:12:00:03",
            "IPv4Address": "172.18.0.3/16",
            "IPv6Address": "",
        }
    if foreign_endpoint:
        endpoints["foreign-id"] = {
            "Name": "trainfactory-web" if foreign_web_alias else "foreign"
        }
    network_labels = {
        "com.docker.compose.project": "trainfactory",
        "com.docker.compose.network": "default",
    }
    if not network_labels_valid:
        network_labels["com.docker.compose.project"] = "foreign"
    network = [
        {
            "Id": "network-id",
            "Name": "trainfactory_network",
            "Driver": "bridge",
            "Scope": "local",
            "Internal": False,
            "Attachable": False,
            "Ingress": False,
            "EnableIPv6": False,
            "Labels": network_labels,
            "Containers": endpoints,
            "Untrusted": "x" * untrusted_padding,
        }
    ]
    volume_roles = {
        "trainfactory_etcd_data": "etcd_data",
        "trainfactory_milvus_data": "milvus_data",
        "trainfactory_minio_data": "minio_data",
        "trainfactory_runtime_mysql_data": "mysql_data",
        "trainfactory_runtime_train_cache": "train_cache",
    }
    volumes = []
    for index, name in enumerate(volume_names):
        volume_role = volume_roles.get(name, "foreign")
        if index == 0 and not volume_labels_valid:
            volume_role = "foreign"
        volumes.append(
            {
                "Name": name,
                "Driver": "local",
                "Scope": "local",
                "Mountpoint": f"/var/lib/docker/volumes/{name}/_data",
                "CreatedAt": "2026-08-19T00:00:00Z",
                "Labels": {
                    "com.docker.compose.project": "trainfactory",
                    "com.docker.compose.volume": volume_role,
                },
                "Untrusted": "x" * untrusted_padding,
            }
        )
    project_names = sorted(
        fixed_container_names | ({"trainfactory-web"} if web_present else set())
    )

    def private(_run, arguments, *_args, **_kwargs):
        if arguments[1] == "ps":
            stdout = "\n".join(project_names) + "\n"
        elif arguments[1:3] == ["network", "inspect"]:
            stdout = json.dumps(network)
        elif arguments[1:3] == ["volume", "ls"]:
            stdout = "\n".join(volume_names) + "\n"
        elif arguments[1:3] == ["volume", "inspect"]:
            stdout = json.dumps(volumes)
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(module, "_container_projection", projection)
    monkeypatch.setattr(module.compose_release, "_run_private", private)


def test_fixed_resource_snapshot_rejects_five_wrong_volume_names(monkeypatch):
    wrong = sorted(
        (module._FIXED_VOLUMES - {"trainfactory_etcd_data"}) | {"foreign_data"}
    )
    _resource_probe_fakes(monkeypatch, volume_names=wrong)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion resource boundary is invalid$",
    ):
        module._snapshot_fixed_resources(environment={})


def test_fixed_resource_snapshot_rejects_foreign_network_endpoint(monkeypatch):
    _resource_probe_fakes(
        monkeypatch,
        volume_names=sorted(module._FIXED_VOLUMES),
        foreign_endpoint=True,
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion resource boundary is invalid$",
    ):
        module._snapshot_fixed_resources(environment={})


def test_fixed_resource_snapshot_rejects_foreign_network_provenance(monkeypatch):
    _resource_probe_fakes(
        monkeypatch,
        volume_names=sorted(module._FIXED_VOLUMES),
        network_labels_valid=False,
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion resource boundary is invalid$",
    ):
        module._snapshot_fixed_resources(environment={})


def test_fixed_resource_snapshot_rejects_foreign_volume_provenance(monkeypatch):
    _resource_probe_fakes(
        monkeypatch,
        volume_names=sorted(module._FIXED_VOLUMES),
        volume_labels_valid=False,
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion resource boundary is invalid$",
    ):
        module._snapshot_fixed_resources(environment={})


def test_fixed_resource_snapshot_discards_unbounded_inspect_metadata(monkeypatch):
    _resource_probe_fakes(
        monkeypatch,
        volume_names=sorted(module._FIXED_VOLUMES),
        untrusted_padding=512 * 1024,
    )

    snapshot = module._snapshot_fixed_resources(environment={})

    assert len(json.dumps(snapshot)) < 64 * 1024


def test_fixed_resource_snapshot_rejects_web_alias_when_project_web_is_absent(
    monkeypatch,
):
    _resource_probe_fakes(
        monkeypatch,
        volume_names=sorted(module._FIXED_VOLUMES),
        foreign_endpoint=True,
        foreign_web_alias=True,
        web_present=False,
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion resource boundary is invalid$",
    ):
        module._snapshot_fixed_resources(environment={})


def test_fixed_resource_snapshot_allows_web_absent_during_crash_recovery(
    monkeypatch,
):
    _resource_probe_fakes(
        monkeypatch,
        volume_names=sorted(module._FIXED_VOLUMES),
        web_present=False,
    )

    snapshot = module._snapshot_fixed_resources(environment={})

    assert set(snapshot["containers"]) == set(module._FIXED_CONTAINERS)
    assert set(snapshot["network"]["fixed_endpoints"]) == {
        f"id-{name}" for name in module._FIXED_CONTAINERS
    }


def test_fixed_resource_snapshot_preserves_historical_restart_counts(monkeypatch):
    _resource_probe_fakes(
        monkeypatch,
        volume_names=sorted(module._FIXED_VOLUMES),
        fixed_restart_count=3,
        web_restart_count=2,
    )

    snapshot = module._snapshot_fixed_resources(environment={})

    assert {value["restart_count"] for value in snapshot["containers"].values()} == {3}


def test_runtime_verifier_accepts_healthy_historical_restart_counts(monkeypatch):
    monkeypatch.setattr(
        module.compose_manifest,
        "verify_running_container",
        lambda *_args, **_kwargs: None,
    )

    def projection(name, **_kwargs):
        return {
            "id": "api-container" if name == "trainfactory-api" else "web-container",
            "image": OLD_API_ID if name == "trainfactory-api" else OLD_WEB_ID,
            "status": "running",
            "health": "healthy",
            "restart_count": 3 if name == "trainfactory-api" else 2,
        }

    monkeypatch.setattr(module, "_container_projection", projection)

    module._verify_runtime(
        manifest_path=Path("manifest.json"),
        selection_path=Path("selection.env"),
        expected_api_id=OLD_API_ID,
        expected_web_id=OLD_WEB_ID,
        expected_api_container_id="api-container",
        root=Path.cwd(),
        environment={},
    )


def test_fixed_resource_snapshot_allows_the_minimal_production_stack(monkeypatch):
    core_containers = {"trainfactory-api", "trainfactory-mysql"}
    core_volumes = {
        "trainfactory_runtime_mysql_data",
        "trainfactory_runtime_train_cache",
    }
    _resource_probe_fakes(
        monkeypatch,
        volume_names=sorted(core_volumes),
        fixed_container_names=core_containers,
    )

    snapshot = module._snapshot_fixed_resources(environment={})

    assert set(snapshot["containers"]) == core_containers
    assert {item["name"] for item in snapshot["volumes"]} == core_volumes
    assert set(snapshot["network"]["fixed_endpoints"]) == {
        "id-trainfactory-api",
        "id-trainfactory-mysql",
    }


def test_fixed_resource_drift_is_rejected_before_journal_mutation(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    old_selection = (runtime / "release.env").read_bytes()
    old_manifest = (runtime / "production-compose-manifest.json").read_bytes()
    calls, _old, _new = _install_fakes(monkeypatch, root)
    monkeypatch.setattr(
        module,
        "_verify_fixed_resources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module.WebPromotionError("web promotion fixed resources drifted")
        ),
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion fixed resources drifted$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert (runtime / "release.env").read_bytes() == old_selection
    assert (runtime / "production-compose-manifest.json").read_bytes() == old_manifest
    assert not (runtime / "web-promotion-active.json").exists()
    assert not any(name in {"freeze", "compose"} for name, _value in calls)


def test_fixed_resource_drift_at_final_compose_guard_blocks_forward_and_rollback(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, _new = _install_fakes(monkeypatch, root)
    fixed_valid = True

    def verify_fixed(_snapshot, **_kwargs):
        if not fixed_valid:
            raise module.WebPromotionError("web promotion fixed resources drifted")

    def drift_before_final_guard(_manifest, tail, *, mutation_guard, **_kwargs):
        nonlocal fixed_valid
        fixed_valid = False
        mutation_guard()
        calls.append(("compose", tuple(tail)))

    monkeypatch.setattr(module, "_verify_fixed_resources", verify_fixed)
    monkeypatch.setattr(
        module.compose_release, "execute_manifest", drift_before_final_guard
    )

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion failed; automatic rollback also failed$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert not any(name == "compose" for name, _value in calls)
    assert (runtime / "web-promotion-active.json").is_file()


def test_retired_capacity_is_reserved_before_journal_mutation(tmp_path, monkeypatch):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    old_selection = (runtime / "release.env").read_bytes()
    old_manifest = (runtime / "production-compose-manifest.json").read_bytes()
    calls, _old, _new = _install_fakes(monkeypatch, root)
    monkeypatch.setattr(module, "_MAX_RETIRED_FILES", 0)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion retired state limit exceeded$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert (runtime / "release.env").read_bytes() == old_selection
    assert (runtime / "production-compose-manifest.json").read_bytes() == old_manifest
    assert not (runtime / "web-promotion-active.json").exists()
    assert not any(name in {"selection", "compose"} for name, _value in calls)


def test_other_revision_prepared_manifest_blocks_before_state_mutation(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, _new = _install_fakes(monkeypatch, root)
    other = runtime / f'.web-promotion-{"e" * 40}-manifest.json'
    other.write_bytes(b"foreign-prepared\n")

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion prepared state already exists$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert other.read_bytes() == b"foreign-prepared\n"
    assert not (runtime / "web-promotion-active.json").exists()
    assert not any(name in {"selection", "freeze", "compose"} for name, _ in calls)


def test_cleanup_inventory_rejects_a_prepared_manifest_for_any_revision(tmp_path):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    other = runtime / f'.web-promotion-{"e" * 40}-manifest.json'
    other.write_bytes(b"foreign-prepared\n")

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion cleanup failed$",
    ):
        module._verify_cleanup_inventory(
            runtime,
            nonce="c" * 32,
            prepared_path=runtime / f".web-promotion-{NEW_REVISION}-manifest.json",
            error_message="web promotion cleanup failed",
        )

    assert other.read_bytes() == b"foreign-prepared\n"


@pytest.mark.parametrize(
    "target_name",
    (
        "release.env",
        "production-compose-manifest.json",
        "web-promotion-active.json",
        f".web-promotion-{NEW_REVISION}-manifest.json",
    ),
)
def test_crash_temporary_blocks_a_new_promotion_before_state_mutation(
    tmp_path, monkeypatch, target_name
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, _new = _install_fakes(monkeypatch, root)
    temporary = runtime / f'.{target_name}.{"f" * 32}.tmp'
    temporary.write_bytes(b"crash-residue\n")

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion temporary state requires inspection$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert temporary.read_bytes() == b"crash-residue\n"
    assert not (runtime / "web-promotion-active.json").exists()
    assert not any(name in {"selection", "freeze", "compose"} for name, _ in calls)


def test_unrelated_temporary_name_does_not_block_promotion(tmp_path, monkeypatch):
    root = _root(tmp_path)
    unrelated = root / ".runtime" / "operator-notes.tmp"
    unrelated.write_bytes(b"keep\n")
    _install_fakes(monkeypatch, root)

    result = module.promote_web_release(
        root=root,
        web_image="local/web:new",
        web_image_id=NEW_WEB_ID,
        web_revision=NEW_REVISION,
        base_environment={"PATH": os.environ.get("PATH", "")},
    )

    assert result["success"] is True
    assert unrelated.read_bytes() == b"keep\n"


def test_non_private_runtime_is_rejected_before_lock_or_state_mutation(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, _new = _install_fakes(monkeypatch, root)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda _path: False)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion runtime directory is not private$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert not (runtime / "web-promotion.lock").exists()
    assert not any(name in {"selection", "freeze", "compose"} for name, _ in calls)


def test_runtime_owned_by_another_identity_is_rejected_before_lock(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    calls, _old, _new = _install_fakes(monkeypatch, root)
    monkeypatch.setattr(module, "_runtime_owned_by_current_user", lambda _meta: False)

    with pytest.raises(
        module.WebPromotionError,
        match="^web promotion runtime directory is not private$",
    ):
        module.promote_web_release(
            root=root,
            web_image="local/web:new",
            web_image_id=NEW_WEB_ID,
            web_revision=NEW_REVISION,
            base_environment={"PATH": os.environ.get("PATH", "")},
        )

    assert not (runtime / "web-promotion.lock").exists()
    assert not any(name in {"selection", "freeze", "compose"} for name, _ in calls)


def test_stale_prepared_journal_is_recovered_before_new_promotion(
    tmp_path, monkeypatch
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    stale_lock = module._acquire_process_lock(runtime / "web-promotion.lock")
    module._release_process_lock(stale_lock)
    old_selection = (runtime / "release.env").read_bytes()
    old_manifest = (runtime / "production-compose-manifest.json").read_bytes()
    old_selection_metadata = (runtime / "release.env").lstat()
    old_manifest_metadata = (runtime / "production-compose-manifest.json").lstat()
    calls, _old, new = _install_fakes(monkeypatch, root)
    selection = runtime / "release.env"
    manifest = runtime / "production-compose-manifest.json"
    selection_claim = module._claim_path(runtime, "c" * 32, "selection")
    manifest_claim = module._claim_path(runtime, "c" * 32, "manifest")
    new_selection_identity = module._atomic_replace(
        selection,
        module._selection_payload(new),
        expected_identity=(
            old_selection_metadata.st_dev,
            old_selection_metadata.st_ino,
        ),
        expected_payload=old_selection,
        claim_path=selection_claim,
        retain_claim=True,
    )
    prepared = runtime / f".web-promotion-{NEW_REVISION}-manifest.json"
    prepared_identity = module._atomic_replace(
        prepared,
        module._forward_manifest_payload(None),
        require_absent=True,
    )
    new_manifest_identity = module._atomic_replace(
        manifest,
        module._forward_manifest_payload(None),
        expected_identity=(
            old_manifest_metadata.st_dev,
            old_manifest_metadata.st_ino,
        ),
        expected_payload=old_manifest,
        claim_path=manifest_claim,
        retain_claim=True,
    )
    (runtime / "web-promotion-active.json").write_bytes(
        _fake_journal_payload(
            root,
            old_selection=old_selection,
            old_manifest=old_manifest,
            new_values=new,
            old_selection_identity=(
                old_selection_metadata.st_dev,
                old_selection_metadata.st_ino,
            ),
            old_manifest_identity=(
                old_manifest_metadata.st_dev,
                old_manifest_metadata.st_ino,
            ),
            prepared_identity=prepared_identity,
            new_selection_identity=new_selection_identity,
            new_manifest_identity=new_manifest_identity,
        )
    )

    result = module.promote_web_release(
        root=root,
        web_image="local/web:new",
        web_image_id=NEW_WEB_ID,
        web_revision=NEW_REVISION,
        base_environment={"PATH": os.environ.get("PATH", "")},
    )

    assert result["success"] is True
    recovery_compose = next(
        index for index, item in enumerate(calls) if item[0] == "compose"
    )
    recovered_runtime = calls.index(("runtime", OLD_WEB_ID))
    new_selection = next(
        index for index, item in enumerate(calls) if item[0] == "selection"
    )
    assert recovery_compose < new_selection
    assert recovered_runtime < new_selection
    assert (runtime / "web-promotion.lock").is_file()


def test_committed_journal_is_rejected_without_mutation(tmp_path):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    selection_before = (runtime / "release.env").read_bytes()
    manifest_before = (runtime / "production-compose-manifest.json").read_bytes()
    journal = runtime / "web-promotion-active.json"
    payload = _fake_journal_payload(
        root,
        old_selection=selection_before,
        old_manifest=manifest_before,
        new_values={
            "API_IMAGE": "local/api:old",
            "WEB_IMAGE": "local/web:new",
            "API_REVISION": OLD_REVISION,
            "WEB_REVISION": NEW_REVISION,
            "API_IMAGE_ID": OLD_API_ID,
            "WEB_IMAGE_ID": NEW_WEB_ID,
        },
    )
    value = json.loads(payload)
    assert value["version"] == 5
    assert value["phase"] == "applying"
    value["phase"] = "committed"
    journal.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        module.WebPromotionError, match="^web promotion recovery state is invalid$"
    ):
        module._recover(
            root=root,
            journal_path=journal,
            selection_path=runtime / "release.env",
            manifest_path=runtime / "production-compose-manifest.json",
            environment={"PATH": os.environ.get("PATH", "")},
        )

    assert (runtime / "release.env").read_bytes() == selection_before
    assert (
        runtime / "production-compose-manifest.json"
    ).read_bytes() == manifest_before
    assert journal.exists()


def test_cli_uses_fixed_error_and_never_echoes_credentials(
    tmp_path, monkeypatch, capsys
):
    root = _root(tmp_path)
    secret = "private-password-canary"
    monkeypatch.setattr(module, "ROOT_DIR", root)
    monkeypatch.setattr(
        module,
        "promote_web_release",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    result = module.main(
        [
            "--web-image",
            "local/web:new",
            "--web-image-id",
            NEW_WEB_ID,
            "--web-revision",
            NEW_REVISION,
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "web promotion failed\n"
    assert secret not in captured.err


def test_cli_maps_a_real_startup_recovery_failure_to_manual_inspection(
    tmp_path, monkeypatch, capsys
):
    root = _root(tmp_path)
    runtime = root / ".runtime"
    journal = runtime / "web-promotion-active.json"
    journal.write_bytes(b'{"recovery":true}\n')
    _install_fakes(monkeypatch, root)
    monkeypatch.setattr(module, "ROOT_DIR", root)
    monkeypatch.setattr(
        module,
        "_recover",
        lambda **_kwargs: (_ for _ in ()).throw(
            module.WebPromotionError("web promotion fixed resources drifted")
        ),
    )

    result = module.main(
        [
            "--web-image",
            "local/web:new",
            "--web-image-id",
            NEW_WEB_ID,
            "--web-revision",
            NEW_REVISION,
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == (
        "web promotion recovery failed; manual inspection is required before retry\n"
    )
    assert journal.read_bytes() == b'{"recovery":true}\n'


@pytest.mark.parametrize(
    ("internal", "operator_message"),
    (
        (
            "web promotion committed; cleanup is incomplete",
            "web promotion committed; recovery is required before retry\n",
        ),
        (
            "web promotion failed; automatic rollback also failed",
            "web promotion state is uncertain; recovery is required before retry\n",
        ),
        (
            "web promotion failed; old web restored",
            "web promotion rejected; old web was restored\n",
        ),
        (
            "web promotion temporary state requires inspection",
            "web promotion blocked; temporary state requires inspection\n",
        ),
        (
            "web promotion startup recovery failed",
            "web promotion recovery failed; manual inspection is required before retry\n",
        ),
        (
            "web promotion terminal cleanup state is invalid",
            "web promotion terminal state requires manual inspection before retry\n",
        ),
        (
            "web promotion lock is invalid",
            "web promotion terminal state requires manual inspection before retry\n",
        ),
    ),
)
def test_cli_reports_fixed_actionable_recovery_outcome(
    tmp_path, monkeypatch, capsys, internal, operator_message
):
    root = _root(tmp_path)
    monkeypatch.setattr(module, "ROOT_DIR", root)
    monkeypatch.setattr(
        module,
        "promote_web_release",
        lambda **_kwargs: (_ for _ in ()).throw(module.WebPromotionError(internal)),
    )

    result = module.main(
        [
            "--web-image",
            "local/web:new",
            "--web-image-id",
            NEW_WEB_ID,
            "--web-revision",
            NEW_REVISION,
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == operator_message


def test_isolated_cli_bootstrap_ignores_sitecustomize(tmp_path):
    poison = tmp_path / "poison"
    poison.mkdir()
    marker = tmp_path / "marker"
    (poison / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(poison)

    completed = subprocess.run(
        [
            os.fspath(Path(os.sys.executable)),
            "-I",
            os.fspath(module.ROOT_DIR / "scripts" / "promote_web_release.py"),
            "--help",
        ],
        cwd=module.ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert not marker.exists()
