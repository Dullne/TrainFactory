import importlib
import os
import stat
import subprocess
import sys
import json
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parents[1]
REQUIRED_SECRET_NAMES = (
    "MYSQL_ROOT_PASSWORD",
    "MYSQL_PASSWORD",
    "MYSQL_URL",
    "JWT_SECRET_KEY",
    "DEFAULT_ADMIN_PASSWORD",
)


def resolve_secret(**kwargs):
    module = importlib.import_module("train_factory.config.secret_files")
    return module.resolve_secret(**kwargs)


def classify_required_secret_sources(environment):
    module = importlib.import_module("train_factory.config.secret_files")
    return module.classify_required_secret_sources(environment)


def test_resolve_secret_accepts_direct_value_without_transforming_it():
    value = " direct-secret-with-spaces "

    assert (
        resolve_secret(
            direct_value=value,
            file_path=None,
            setting_name="JWT_SECRET_KEY",
        )
        == value
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    (
        (b"file-secret", "file-secret"),
        (b"file-secret\n", "file-secret"),
        (b"file-secret\r\n", "file-secret"),
        (b"file-secret\n\n", "file-secret\n"),
        (b"file-secret\r\n\r\n", "file-secret\r\n"),
    ),
)
def test_resolve_secret_reads_file_and_removes_only_one_trailing_newline(
    tmp_path,
    content,
    expected,
):
    secret_file = tmp_path / "secret"
    secret_file.write_bytes(content)

    assert (
        resolve_secret(
            direct_value=None,
            file_path=secret_file,
            setting_name="JWT_SECRET_KEY",
        )
        == expected
    )


def test_resolve_secret_returns_none_when_both_sources_are_absent():
    assert (
        resolve_secret(
            direct_value=None,
            file_path=None,
            setting_name="JWT_SECRET_KEY",
        )
        is None
    )


def test_resolve_secret_rejects_conflicting_sources_without_echo(tmp_path):
    secret_file = tmp_path / "private-secret-path"
    secret_file.write_text("file-private-canary", encoding="utf-8")
    direct_canary = "direct-private-canary"

    with pytest.raises(ValueError) as exc_info:
        resolve_secret(
            direct_value=direct_canary,
            file_path=secret_file,
            setting_name="JWT_SECRET_KEY",
        )

    assert str(exc_info.value) == "JWT_SECRET_KEY has conflicting sources"
    assert direct_canary not in repr(exc_info.value)
    assert str(secret_file) not in repr(exc_info.value)


@pytest.mark.parametrize(
    ("kind", "expected_category"),
    (
        ("empty", "empty value"),
        ("directory", "invalid file type"),
        ("symlink", "invalid file type"),
        ("oversize", "file too large"),
        ("invalid_utf8", "invalid UTF-8"),
        ("nul", "NUL byte"),
    ),
)
def test_resolve_secret_rejects_invalid_files_without_echoing_path_or_content(
    tmp_path,
    monkeypatch,
    kind,
    expected_category,
):
    secret_file = tmp_path / "private-path-canary"
    max_bytes = 65_536
    if kind == "empty":
        secret_file.write_bytes(b"")
    elif kind == "directory":
        secret_file.mkdir()
    elif kind == "symlink":
        secret_file.write_bytes(b"unread-private-canary")
        original_lstat = Path.lstat

        def fake_lstat(path):
            result = original_lstat(path)
            values = list(result)
            values[0] = stat.S_IFLNK | 0o600
            return os.stat_result(values)

        monkeypatch.setattr(Path, "lstat", fake_lstat)
    elif kind == "oversize":
        max_bytes = 4
        secret_file.write_bytes(b"oversize-private-canary")
    elif kind == "invalid_utf8":
        secret_file.write_bytes(b"\xff\xfe")
    elif kind == "nul":
        secret_file.write_bytes(b"nul\x00private-canary")

    with pytest.raises(ValueError) as exc_info:
        resolve_secret(
            direct_value=None,
            file_path=secret_file,
            setting_name="JWT_SECRET_KEY",
            max_bytes=max_bytes,
        )

    assert str(exc_info.value) == f"JWT_SECRET_KEY has {expected_category}"
    assert str(secret_file) not in repr(exc_info.value)
    assert "private-canary" not in repr(exc_info.value)


def test_resolve_secret_rejects_empty_and_nul_direct_values_without_echo():
    for value, category in (("", "empty value"), ("private\x00canary", "NUL byte")):
        with pytest.raises(ValueError) as exc_info:
            resolve_secret(
                direct_value=value,
                file_path=None,
                setting_name="MYSQL_URL",
            )

        assert str(exc_info.value) == f"MYSQL_URL has {category}"
        if value:
            assert value not in repr(exc_info.value)


@pytest.mark.parametrize("swap_kind", ("opened_identity", "path_identity"))
def test_resolve_secret_fails_closed_when_file_identity_changes(
    tmp_path,
    monkeypatch,
    swap_kind,
):
    secret_file = tmp_path / "identity-private-path"
    secret_file.write_text("identity-private-canary", encoding="utf-8")
    if swap_kind == "opened_identity":
        original_fstat = os.fstat

        def swapped_fstat(descriptor):
            result = original_fstat(descriptor)
            values = list(result)
            values[1] += 1
            return os.stat_result(values)

        monkeypatch.setattr(os, "fstat", swapped_fstat)
    else:
        original_lstat = Path.lstat
        calls = 0

        def swapped_lstat(path):
            nonlocal calls
            result = original_lstat(path)
            if path == secret_file:
                calls += 1
                if calls > 1:
                    values = list(result)
                    values[1] += 1
                    return os.stat_result(values)
            return result

        monkeypatch.setattr(Path, "lstat", swapped_lstat)

    with pytest.raises(ValueError) as exc_info:
        resolve_secret(
            direct_value=None,
            file_path=secret_file,
            setting_name="JWT_SECRET_KEY",
        )

    assert str(exc_info.value) == "JWT_SECRET_KEY has changed file"
    assert str(secret_file) not in repr(exc_info.value)


def _direct_environment():
    return {name: f"private-value-{index}" for index, name in enumerate(REQUIRED_SECRET_NAMES)}


def _file_environment(tmp_path):
    return {f"{name}_FILE": str(tmp_path / f"secret-{index}") for index, name in enumerate(REQUIRED_SECRET_NAMES)}


def test_classify_required_secret_sources_returns_names_not_values(tmp_path):
    direct = _direct_environment()
    files = _file_environment(tmp_path)

    direct_result = classify_required_secret_sources(direct)
    file_result = classify_required_secret_sources(files)

    assert direct_result == {
        "mode": "direct",
        "sources": {name: name for name in REQUIRED_SECRET_NAMES},
    }
    assert file_result == {
        "mode": "files",
        "sources": {name: f"{name}_FILE" for name in REQUIRED_SECRET_NAMES},
    }
    rendered = repr((direct_result, file_result))
    assert not any(value in rendered for value in direct.values())
    assert not any(value in rendered for value in files.values())


@pytest.mark.parametrize("failure", ("missing", "mixed", "conflict", "admin_only"))
def test_classify_required_secret_sources_fails_closed_without_values(tmp_path, failure):
    direct = _direct_environment()
    files = _file_environment(tmp_path)
    if failure == "missing":
        environment = direct
        private_value = environment.pop(REQUIRED_SECRET_NAMES[0])
        category = "incomplete"
    elif failure == "mixed":
        environment = direct
        name = REQUIRED_SECRET_NAMES[0]
        environment.pop(name)
        environment[f"{name}_FILE"] = files[f"{name}_FILE"]
        private_value = next(iter(environment.values()))
        category = "mixed sources"
    elif failure == "conflict":
        environment = direct
        name = REQUIRED_SECRET_NAMES[0]
        environment[f"{name}_FILE"] = files[f"{name}_FILE"]
        private_value = environment[name]
        category = "conflicting sources"
    else:
        private_value = "admin-only-private-canary"
        environment = {"DEFAULT_ADMIN_PASSWORD": private_value}
        category = "incomplete"

    with pytest.raises(ValueError) as exc_info:
        classify_required_secret_sources(environment)

    assert str(exc_info.value) == f"required secrets have {category}"
    assert private_value not in repr(exc_info.value)


def test_importing_secret_helper_has_no_settings_or_filesystem_side_effects(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "JWT_SECRET_KEY": "import-private-canary",
            "TRAINING_CACHE": str(tmp_path / "cache"),
            "MODELS_DIR": str(tmp_path / "models"),
            "DATASETS_DIR": str(tmp_path / "datasets"),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "LOCAL_CACHE_DIR": str(tmp_path / "local-cache"),
        }
    )
    environment.pop("HF_HOME", None)
    code = """
import os
import sys
from pathlib import Path

root = Path(os.environ["TRAINING_CACHE"]).parent
import train_factory.config.secret_files

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


def _settings_values(tmp_path):
    return {
        "_env_file": None,
        "debug": False,
        "auth_enabled": True,
        "host_bind_address": "127.0.0.1",
        "public_base_url": "http://localhost:3000",
        "auth_cookie_secure": False,
        "training_cache": tmp_path / "cache",
        "models_dir": tmp_path / "models",
        "datasets_dir": tmp_path / "datasets",
        "output_dir": tmp_path / "output",
        "local_cache_dir": tmp_path / "local-cache",
    }


def _clear_secret_environment(monkeypatch):
    for name in ("JWT_SECRET_KEY", "DEFAULT_ADMIN_PASSWORD", "MYSQL_URL"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"{name}_FILE", raising=False)


def test_settings_resolves_supported_file_secrets_before_security_checks(
    tmp_path,
    monkeypatch,
):
    from train_factory.config.settings import Settings

    _clear_secret_environment(monkeypatch)
    secrets = {
        "jwt_secret_key": "jwt-file-private-canary-at-least-32-bytes",
        "default_admin_password": "admin-file-private-canary",
        "mysql_url": "mysql+pymysql://app:mysql-file-private-canary@mysql/db",
    }
    values = _settings_values(tmp_path)
    for name, secret in secrets.items():
        secret_file = tmp_path / f"{name}-private-path"
        secret_file.write_text(f"{secret}\n", encoding="utf-8")
        values[f"{name}_file"] = secret_file

    settings = Settings(**values)

    assert settings.jwt_secret_key == secrets["jwt_secret_key"]
    assert settings.default_admin_password == secrets["default_admin_password"]
    assert settings.mysql_url == secrets["mysql_url"]


@pytest.mark.parametrize(
    ("name", "direct_value"),
    (
        ("jwt_secret_key", ""),
        ("default_admin_password", ""),
        ("mysql_url", ""),
    ),
)
def test_settings_rejects_direct_and_file_sources_even_when_direct_is_empty(
    tmp_path,
    name,
    direct_value,
):
    from pydantic import ValidationError

    from train_factory.config.settings import Settings

    secret_file = tmp_path / "settings-private-path-canary"
    secret_file.write_text("file-private-canary", encoding="utf-8")
    values = _settings_values(tmp_path)
    values.update(
        {
            "jwt_secret_key": "direct-jwt-private-canary-at-least-32-bytes",
            name: direct_value,
            f"{name}_file": secret_file,
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        Settings(**values)

    rendered = repr(exc_info.value)
    assert "conflicting sources" in rendered
    assert "file-private-canary" not in rendered
    assert str(secret_file) not in rendered
    assert not any(
        (tmp_path / directory).exists() for directory in ("cache", "models", "datasets", "output", "local-cache")
    )


def test_settings_file_failure_has_no_directory_or_environment_side_effects(
    tmp_path,
    monkeypatch,
):
    from pydantic import ValidationError

    from train_factory.config.settings import Settings

    _clear_secret_environment(monkeypatch)
    monkeypatch.delenv("HF_HOME", raising=False)
    missing_path = tmp_path / "missing-private-path-canary"
    values = _settings_values(tmp_path)
    values["jwt_secret_key_file"] = missing_path

    with pytest.raises(ValidationError) as exc_info:
        Settings(**values)

    rendered = repr(exc_info.value)
    assert "unavailable file" in rendered
    assert str(missing_path) not in rendered
    assert "HF_HOME" not in os.environ
    assert not any(
        (tmp_path / directory).exists() for directory in ("cache", "models", "datasets", "output", "local-cache")
    )


def test_settings_repr_and_serialization_exclude_secret_values_and_sources(
    tmp_path,
    monkeypatch,
):
    from train_factory.config.settings import Settings

    _clear_secret_environment(monkeypatch)
    secrets = {
        "jwt_secret_key": "jwt-dump-private-canary-at-least-32-bytes",
        "default_admin_password": "admin-dump-private-canary",
        "mysql_url": "mysql+pymysql://app:mysql-dump-private-canary@mysql/db",
    }
    values = _settings_values(tmp_path)
    paths = []
    for name, secret in secrets.items():
        path = tmp_path / f"{name}-dump-private-path"
        path.write_text(secret, encoding="utf-8")
        values[f"{name}_file"] = path
        paths.append(path)

    settings = Settings(**values)
    rendered = repr(settings)
    dumped = repr(settings.model_dump())

    for secret in secrets.values():
        assert secret not in rendered
        assert secret not in dumped
    for path in paths:
        assert str(path) not in rendered
        assert str(path) not in dumped
    assert "jwt_secret_key_file" not in settings.model_dump()
    assert "default_admin_password_file" not in settings.model_dump()
    assert "mysql_url_file" not in settings.model_dump()


def _materializer_module():
    return importlib.import_module("scripts.materialize_compose_secrets")


def _materializer_paths(workspace, scope, run_id):
    run_directory = workspace / ".runtime" / f"{scope}-{run_id}"
    return (
        workspace / ".runtime",
        run_directory / "compose-secrets.env",
        run_directory / "compose-secrets.state.json",
    )


def test_materializer_atomically_creates_consistent_private_bundle(tmp_path, monkeypatch):
    module = _materializer_module()
    run_id = "a" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "verify", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)

    metadata = module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="verify",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
    )

    run_directory = state_out.parent
    assert metadata["scope"] == "verify"
    assert metadata["run_id"] == run_id
    assert metadata["permissions_hardened"] is True
    assert not any(path.name.startswith(".tmp-") for path in output_root.iterdir())
    root_password = (run_directory / "mysql_root_password").read_text(encoding="utf-8")
    app_password = (run_directory / "mysql_app_password").read_text(encoding="utf-8")
    mysql_url = (run_directory / "mysql_url").read_text(encoding="utf-8")
    assert root_password != app_password
    assert mysql_url == (
        "mysql+pymysql://trainfactory_app:"
        f"{__import__('urllib.parse', fromlist=['quote']).quote(app_password, safe='')}"
        "@mysql:3306/train_factory"
    )
    env_text = env_out.read_text(encoding="utf-8")
    state_text = state_out.read_text(encoding="utf-8")
    for private_value in (
        root_password,
        app_password,
        mysql_url,
        (run_directory / "jwt_secret_key").read_text(encoding="utf-8"),
        (run_directory / "default_admin_password").read_text(encoding="utf-8"),
    ):
        assert private_value not in env_text
        assert private_value not in state_text
        assert private_value not in repr(metadata)
    assert json.loads(state_text)["directory"] == str(run_directory.resolve())
    mysql_path = str(run_directory.resolve() / "mysql_url").replace("'", "\\'")
    assert f"MYSQL_URL_SECRET_PATH='{mysql_path}'" in env_text


def test_materializer_accepts_only_exact_owned_ci_staging_directory(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    run_id = "9" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)
    output_root.mkdir()
    staging = output_root / f".ci-staging-{run_id}"
    staging.mkdir()
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)

    metadata = module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="ci",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
        staging_dir=staging,
    )

    assert metadata["run_id"] == run_id
    assert not staging.exists()
    assert state_out.is_file()


def test_materializer_rejects_noncanonical_staging_without_deleting_it(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    run_id = "8" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)
    output_root.mkdir()
    staging = output_root / f".ci-staging-{run_id}-foreign"
    staging.mkdir()
    marker = staging / "foreign"
    marker.write_text("FOREIGN-CANARY", encoding="utf-8")
    monkeypatch.setattr(module, "_harden_path", lambda path: True)

    with pytest.raises(ValueError, match="materialization path is invalid"):
        module.create_secret_bundle(
            workspace_root=tmp_path,
            scope="ci",
            run_id=run_id,
            output_root=output_root,
            env_out=env_out,
            state_out=state_out,
            staging_dir=staging,
        )

    assert marker.read_text(encoding="utf-8") == "FOREIGN-CANARY"


def test_materializer_rejects_existing_run_without_overwrite(tmp_path, monkeypatch):
    module = _materializer_module()
    run_id = "b" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)
    state_out.parent.mkdir(parents=True)
    marker = state_out.parent / "existing-private-marker"
    marker.write_text("existing-private-value", encoding="utf-8")
    monkeypatch.setattr(module, "_harden_path", lambda path: True)

    with pytest.raises(ValueError, match="materialization target is invalid"):
        module.create_secret_bundle(
            workspace_root=tmp_path,
            scope="ci",
            run_id=run_id,
            output_root=output_root,
            env_out=env_out,
            state_out=state_out,
        )

    assert marker.read_text(encoding="utf-8") == "existing-private-value"


def test_materializer_rejects_output_outside_exact_runtime_run(tmp_path, monkeypatch):
    module = _materializer_module()
    run_id = "c" * 32
    output_root, _, state_out = _materializer_paths(tmp_path, "verify", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)

    with pytest.raises(ValueError, match="materialization path is invalid"):
        module.create_secret_bundle(
            workspace_root=tmp_path,
            scope="verify",
            run_id=run_id,
            output_root=output_root,
            env_out=tmp_path / "outside-private-path",
            state_out=state_out,
        )

    assert not (output_root / f"verify-{run_id}").exists()


def test_materializer_cleanup_preflights_all_members_before_deleting(tmp_path, monkeypatch):
    module = _materializer_module()
    run_id = "d" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "verify", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)
    module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="verify",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
    )
    extra = state_out.parent / "unexpected-private-member"
    extra.write_text("unexpected-private-value", encoding="utf-8")

    with pytest.raises(ValueError, match="cleanup state is invalid"):
        module.cleanup_secret_bundle(workspace_root=tmp_path, state_path=state_out)

    assert extra.exists()
    assert env_out.exists()
    assert state_out.exists()


def test_materializer_cleanup_removes_only_validated_run_directory(tmp_path, monkeypatch):
    module = _materializer_module()
    run_id = "e" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)
    module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="ci",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
    )

    metadata = module.cleanup_secret_bundle(
        workspace_root=tmp_path,
        state_path=state_out,
    )

    assert metadata == {"scope": "ci", "run_id": run_id, "removed": True}
    assert not state_out.parent.exists()
    assert output_root.exists()


@pytest.mark.parametrize(
    ("scope", "run_id"),
    (("release", "f" * 32), ("verify", "F" * 32), ("ci", "f" * 31)),
)
def test_materializer_rejects_noncanonical_scope_and_run_id(
    tmp_path,
    monkeypatch,
    scope,
    run_id,
):
    module = _materializer_module()
    output_root, env_out, state_out = _materializer_paths(tmp_path, scope, run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)

    with pytest.raises(ValueError, match="materialization request is invalid"):
        module.create_secret_bundle(
            workspace_root=tmp_path,
            scope=scope,
            run_id=run_id,
            output_root=output_root,
            env_out=env_out,
            state_out=state_out,
        )


def test_materializer_create_failure_removes_only_its_temporary_directory(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    run_id = "1" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "verify", run_id)
    output_root.mkdir()
    unrelated = output_root / "unrelated-private-marker"
    unrelated.write_text("unrelated-private-value", encoding="utf-8")
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    writes = 0
    original_write = module._write_private_file

    def fail_during_write(path, content):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("injected failure")
        return original_write(path, content)

    monkeypatch.setattr(module, "_write_private_file", fail_during_write)

    with pytest.raises(ValueError, match="secret materialization failed"):
        module.create_secret_bundle(
            workspace_root=tmp_path,
            scope="verify",
            run_id=run_id,
            output_root=output_root,
            env_out=env_out,
            state_out=state_out,
        )

    assert unrelated.read_text(encoding="utf-8") == "unrelated-private-value"
    assert not any(path.name.startswith(".tmp-") for path in output_root.iterdir())
    assert not state_out.parent.exists()


def test_materializer_permission_failure_removes_partially_written_secret(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    run_id = "4" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)

    def fail_secret_hardening(path):
        if path.name == "mysql_app_password":
            raise ValueError("secret permission hardening failed")
        return True

    monkeypatch.setattr(module, "_harden_path", fail_secret_hardening)

    with pytest.raises(ValueError, match="secret permission hardening failed"):
        module.create_secret_bundle(
            workspace_root=tmp_path,
            scope="ci",
            run_id=run_id,
            output_root=output_root,
            env_out=env_out,
            state_out=state_out,
        )

    assert not any(path.name.startswith(".tmp-") for path in output_root.iterdir())
    assert not state_out.parent.exists()


def test_materializer_windows_acl_uses_argument_array_and_checks_result(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    calls = []

    class Completed:
        returncode = 0
        stdout = "test-domain\\acl-test-user\n"

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)) or Completed())

    assert module._harden_path(tmp_path) is True
    assert calls[0][0] == ["whoami"]
    args, kwargs = calls[1]
    assert isinstance(args, list)
    assert args[0] == "icacls"
    assert "/inheritance:r" in args
    assert "/grant:r" in args
    assert "test-domain\\acl-test-user:(F)" in args
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True


def test_materializer_cli_metadata_excludes_paths_usernames_and_values(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _materializer_module()
    run_id = "2" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)

    result = module.main(
        [
            "create",
            "--scope",
            "ci",
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--env-out",
            str(env_out),
            "--state-out",
            str(state_out),
        ],
        workspace_root=tmp_path,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert str(tmp_path) not in captured.out
    assert os.environ.get("USERNAME", "missing-user") not in captured.out
    for name in (
        "mysql_root_password",
        "mysql_app_password",
        "mysql_url",
        "jwt_secret_key",
        "default_admin_password",
    ):
        value = (state_out.parent / name).read_text(encoding="utf-8")
        assert value not in captured.out


@pytest.mark.parametrize("symlink_target", ("state", "member"))
def test_materializer_cleanup_rejects_symlink_before_any_deletion(
    tmp_path,
    monkeypatch,
    symlink_target,
):
    module = _materializer_module()
    run_id = "3" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "verify", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)
    module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="verify",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
    )
    target = state_out if symlink_target == "state" else state_out.parent / "mysql_url"
    original_lstat = Path.lstat

    def fake_lstat(path):
        result = original_lstat(path)
        if path == target:
            values = list(result)
            values[0] = stat.S_IFLNK | 0o600
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(ValueError, match="cleanup state is invalid"):
        module.cleanup_secret_bundle(workspace_root=tmp_path, state_path=state_out)

    assert state_out.exists()
    assert env_out.exists()
    assert (state_out.parent / "mysql_root_password").exists()


def test_materializer_windows_acl_verification_rejects_everyone_rule(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    current_sid = "S-1-5-21-1000"
    snapshot = {
        "protected": True,
        "current_sid": current_sid,
        "rules": [
            {
                "sid": current_sid,
                "type": "Allow",
                "rights": 2032127,
                "inherited": False,
            }
        ],
    }

    class Completed:
        returncode = 0

        @property
        def stdout(self):
            return json.dumps(snapshot)

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Completed())
    assert module._verify_hardened_path(tmp_path) is True

    snapshot["rules"].append(
        {
            "sid": "S-1-1-0",
            "type": "Allow",
            "rights": 1,
            "inherited": False,
        }
    )
    assert module._verify_hardened_path(tmp_path) is False


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL semantics")
def test_materializer_windows_hardening_removes_preexisting_explicit_rules(tmp_path):
    module = _materializer_module()
    module._harden_path(tmp_path)
    secret_file = tmp_path / "private-value"
    secret_file.write_text("private-value", encoding="utf-8")

    module._harden_path(secret_file)

    assert module._verify_hardened_path(tmp_path) is True
    assert module._verify_hardened_path(secret_file) is True


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL semantics")
@pytest.mark.parametrize("kind", ("file", "directory"))
def test_materializer_windows_hardening_preserves_owner_and_group(tmp_path, kind):
    module = _materializer_module()
    target = tmp_path / kind
    if kind == "directory":
        target.mkdir()
    else:
        target.write_text("permission-probe", encoding="utf-8")
    identity_command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "& { param([string]$TargetPath) $ErrorActionPreference = 'Stop'; "
        "Get-Acl -LiteralPath $TargetPath | Select-Object Owner, Group | "
        "ConvertTo-Json -Compress }",
        str(target),
    ]
    before = subprocess.run(identity_command, capture_output=True, text=True, check=True)

    assert module._harden_path(target) is True

    assert module._verify_hardened_path(target) is True
    after = subprocess.run(identity_command, capture_output=True, text=True, check=True)
    assert json.loads(after.stdout) == json.loads(before.stdout)


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL semantics")
def test_materializer_windows_hardening_script_returns_failure_for_missing_path(tmp_path):
    module = _materializer_module()
    target = tmp_path / "missing"

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            module.WINDOWS_HARDEN_SCRIPT,
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not target.exists()


def test_materializer_reverifies_permissions_before_and_after_publish(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    run_id = "5" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "verify", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    verified = []

    def record(path):
        verified.append(path)
        return True

    monkeypatch.setattr(module, "_verify_hardened_path", record)
    module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="verify",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
    )

    assert state_out.parent in verified
    assert state_out in verified
    assert state_out.parent / "mysql_url" in verified


def test_materializer_post_publish_verification_failure_rolls_back_final_directory(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    run_id = "0" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "verify", run_id)
    final_directory = state_out.parent
    monkeypatch.setattr(module, "_harden_path", lambda path: True)

    def fail_after_publish(path):
        return path != final_directory and path.parent != final_directory

    monkeypatch.setattr(module, "_verify_hardened_path", fail_after_publish)

    with pytest.raises(ValueError, match="secret permission verification failed"):
        module.create_secret_bundle(
            workspace_root=tmp_path,
            scope="verify",
            run_id=run_id,
            output_root=output_root,
            env_out=env_out,
            state_out=state_out,
        )

    assert not final_directory.exists()
    assert not any(path.name.startswith(".tmp-") for path in output_root.iterdir())


def test_materializer_cleanup_acl_tamper_causes_zero_deletions(tmp_path, monkeypatch):
    module = _materializer_module()
    run_id = "9" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)
    module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="ci",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
    )
    expected_members = {path.name for path in state_out.parent.iterdir()}
    tampered = state_out.parent / "mysql_url"
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: path != tampered)

    with pytest.raises(ValueError, match="cleanup state is invalid"):
        module.cleanup_secret_bundle(workspace_root=tmp_path, state_path=state_out)

    assert {path.name for path in state_out.parent.iterdir()} == expected_members


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("scope", []),
        ("run_id", {}),
        ("directory", []),
        ("state_file", {}),
        ("env_file", []),
        ("members", {}),
        ("cleanup_started", []),
    ),
)
def test_materializer_cleanup_rejects_arbitrary_state_types_with_fixed_error(
    tmp_path,
    monkeypatch,
    field,
    invalid_value,
):
    module = _materializer_module()
    run_id = "6" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)
    module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="ci",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
    )
    state = json.loads(state_out.read_text(encoding="utf-8"))
    state[field] = invalid_value
    state_out.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        module.cleanup_secret_bundle(workspace_root=tmp_path, state_path=state_out)

    assert str(exc_info.value) == "cleanup state is invalid"
    assert exc_info.value.__cause__ is None


def test_materializer_cleanup_is_retryable_after_unlink_failure(
    tmp_path,
    monkeypatch,
):
    module = _materializer_module()
    run_id = "7" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "verify", run_id)
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)
    module.create_secret_bundle(
        workspace_root=tmp_path,
        scope="verify",
        run_id=run_id,
        output_root=output_root,
        env_out=env_out,
        state_out=state_out,
    )
    original_unlink = Path.unlink
    failed = False

    def fail_once(path, *args, **kwargs):
        nonlocal failed
        if path.name == "mysql_app_password" and not failed:
            failed = True
            raise OSError("injected unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(ValueError, match="secret cleanup failed"):
        module.cleanup_secret_bundle(workspace_root=tmp_path, state_path=state_out)

    assert state_out.exists()
    result = module.cleanup_secret_bundle(workspace_root=tmp_path, state_path=state_out)
    assert result["removed"] is True
    assert not state_out.parent.exists()


def test_materializer_special_character_random_password_is_encoded_and_not_emitted(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _materializer_module()
    run_id = "8" * 32
    output_root, env_out, state_out = _materializer_paths(tmp_path, "ci", run_id)
    canary = "random-private-@:/%-canary"
    generated = iter(("root-private", canary, "jwt-private"))
    monkeypatch.setattr(module.secrets, "token_hex", lambda size: next(generated))
    monkeypatch.setattr(module.secrets, "token_urlsafe", lambda size: "admin-private")
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)

    result = module.main(
        [
            "create",
            "--scope",
            "ci",
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--env-out",
            str(env_out),
            "--state-out",
            str(state_out),
        ],
        workspace_root=tmp_path,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert (state_out.parent / "mysql_url").read_text(encoding="utf-8") == (
        "mysql+pymysql://trainfactory_app:random-private-%40%3A%2F%25-canary" "@mysql:3306/train_factory"
    )
    assert canary not in captured.out
    assert canary not in captured.err
    assert canary not in env_out.read_text(encoding="utf-8")
    assert canary not in state_out.read_text(encoding="utf-8")


@pytest.mark.host_tools
def test_materializer_dotenv_quotes_special_absolute_paths_for_real_compose(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _materializer_module()
    workspace = tmp_path / "workspace space # single$ double$$ ${VAR} ' quote"
    workspace.mkdir()
    run_id = "c" * 32
    output_root, env_out, state_out = _materializer_paths(
        workspace,
        "verify",
        run_id,
    )
    generated_secrets = iter(("dotenv-root-secret", "dotenv-app-secret", "dotenv-jwt-secret"))
    monkeypatch.setattr(module.secrets, "token_hex", lambda size: next(generated_secrets))
    monkeypatch.setattr(module.secrets, "token_urlsafe", lambda size: "dotenv-admin-secret")
    monkeypatch.setattr(module, "_harden_path", lambda path: True)
    monkeypatch.setattr(module, "_verify_hardened_path", lambda path: True)

    result = module.main(
        [
            "create",
            "--scope",
            "verify",
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--env-out",
            str(env_out),
            "--state-out",
            str(state_out),
        ],
        workspace_root=workspace,
    )
    materializer_output = capsys.readouterr()
    assert result == 0

    example_env = ROOT_DIR / ".env.example"
    environment = os.environ.copy()
    example_names = {
        line.split("=", 1)[0]
        for line in example_env.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    for name in tuple(environment):
        if name in example_names or name.startswith("COMPOSE_"):
            environment.pop(name)
    environment.update(
        {
            "DEBUG": "false",
            "JWT_SECRET_KEY": "compose-render-placeholder",
            "DEFAULT_ADMIN_PASSWORD": "compose-render-placeholder",
            "MYSQL_ROOT_PASSWORD": "compose-render-placeholder",
            "MYSQL_APP_PASSWORD": "compose-render-placeholder",
            "VAR": "dotenv-substitution-canary",
        }
    )
    compose_command = [
        "docker",
        "compose",
        "--env-file",
        str(example_env),
        "--env-file",
        str(env_out),
        "-f",
        str(ROOT_DIR / "docker" / "docker-compose.yml"),
        "-f",
        str(ROOT_DIR / "docker" / "docker-compose.secrets.yml"),
        "config",
    ]
    completed_environment = subprocess.run(
        [*compose_command, "--environment"],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed_environment.returncode == 0
    compose_environment = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in completed_environment.stdout.splitlines()
        if "=" in line
    }
    completed = subprocess.run(
        [*compose_command, "--format", "json"],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0
    compose = json.loads(completed.stdout)
    validator = importlib.import_module("scripts.validate_compose_config")
    for secret_name in (
        "mysql_root_password",
        "mysql_app_password",
        "mysql_url",
        "jwt_secret_key",
        "default_admin_password",
    ):
        expected_path = str((state_out.parent / secret_name).resolve())
        assert (
            compose_environment[
                next(variable for variable, filename in module.PATH_VARIABLES.items() if filename == secret_name)
            ]
            == expected_path
        )
        assert validator._decode_compose_literal_path(compose["secrets"][secret_name]["file"]) == expected_path
        assert Path(expected_path).is_file()

    assert (
        validator.validate_compose_config(
            compose,
            profile="minimal",
        )
        == []
    )

    rendered_outputs = (
        materializer_output.out,
        materializer_output.err,
        completed.stdout,
        completed.stderr,
        completed_environment.stdout,
        completed_environment.stderr,
        env_out.read_text(encoding="utf-8"),
        state_out.read_text(encoding="utf-8"),
    )
    for secret in (
        "dotenv-root-secret",
        "dotenv-app-secret",
        "dotenv-jwt-secret",
        "dotenv-admin-secret",
    ):
        assert all(secret not in output for output in rendered_outputs)
