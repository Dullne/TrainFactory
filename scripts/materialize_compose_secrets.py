"""Create and remove short-lived Compose secret bundles without printing values."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SCOPES = {"verify", "ci"}
SECRET_FILES = (
    "mysql_root_password",
    "mysql_app_password",
    "mysql_url",
    "jwt_secret_key",
    "default_admin_password",
    "admin_username",
)
PATH_VARIABLES = {
    "MYSQL_ROOT_PASSWORD_SECRET_PATH": "mysql_root_password",
    "MYSQL_APP_PASSWORD_SECRET_PATH": "mysql_app_password",
    "MYSQL_URL_SECRET_PATH": "mysql_url",
    "JWT_SECRET_KEY_SECRET_PATH": "jwt_secret_key",
    "DEFAULT_ADMIN_PASSWORD_SECRET_PATH": "default_admin_password",
}
WINDOWS_FULL_CONTROL = 2_032_127
WINDOWS_HARDEN_SCRIPT = r"""& {
param([string]$TargetPath)
$acl = Get-Acl -LiteralPath $TargetPath
$acl.SetAccessRuleProtection($true, $false)
foreach ($existing in @($acl.Access)) {
  [void]$acl.RemoveAccessRuleAll($existing)
}
$currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
  $currentSid,
  [System.Security.AccessControl.FileSystemRights]::FullControl,
  [System.Security.AccessControl.AccessControlType]::Allow
)
[void]$acl.AddAccessRule($rule)
Set-Acl -LiteralPath $TargetPath -AclObject $acl
}"""
WINDOWS_ACL_SCRIPT = r"""& {
param([string]$TargetPath)
$acl = Get-Acl -LiteralPath $TargetPath
$currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$rules = @(
  $acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]) |
    ForEach-Object {
      [pscustomobject]@{
        sid = $_.IdentityReference.Value
        type = $_.AccessControlType.ToString()
        rights = [int]$_.FileSystemRights
        inherited = [bool]$_.IsInherited
      }
    }
)
[pscustomobject]@{
  protected = [bool]$acl.AreAccessRulesProtected
  current_sid = $currentSid
  rules = $rules
} | ConvertTo-Json -Compress -Depth 4
}"""


def _fixed_error(message: str) -> ValueError:
    return ValueError(message)


def _dotenv_literal(value: str) -> str:
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise _fixed_error("materialization path is invalid")
    return "'" + value.replace("'", "\\'") + "'"


def _as_workspace_path(path: str | os.PathLike[str], workspace: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return candidate.resolve(strict=False)


def _lstat_or_none(path: Path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _windows_identity() -> str:
    completed = subprocess.run(
        ["whoami"],
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    identity = completed.stdout.strip() if completed.returncode == 0 else ""
    if (
        not identity
        or len(identity) > 256
        or ":" in identity
        or any(ord(character) < 32 for character in identity)
    ):
        raise _fixed_error("secret permission hardening failed")
    return identity


def _harden_path(path: Path) -> bool:
    path.chmod(0o700 if path.is_dir() else 0o600)
    if os.name != "nt":
        return True
    identity = _windows_identity()
    completed = subprocess.run(
        [
            "icacls",
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"{identity}:(F)",
        ],
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise _fixed_error("secret permission hardening failed")
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_HARDEN_SCRIPT,
            str(path),
        ],
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise _fixed_error("secret permission hardening failed")
    return True


def _verify_hardened_path(path: Path) -> bool:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return False
    if os.name != "nt":
        return metadata.st_mode & 0o077 == 0
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                WINDOWS_ACL_SCRIPT,
                str(path),
            ],
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            return False
        snapshot = json.loads(completed.stdout)
        current_sid = snapshot.get("current_sid")
        rules = snapshot.get("rules")
        if (
            snapshot.get("protected") is not True
            or not isinstance(current_sid, str)
            or not current_sid.startswith("S-")
            or not isinstance(rules, list)
            or not rules
        ):
            return False
        return all(
            isinstance(rule, dict)
            and rule.get("sid") == current_sid
            and rule.get("type") == "Allow"
            and rule.get("inherited") is False
            and isinstance(rule.get("rights"), int)
            and rule["rights"] & WINDOWS_FULL_CONTROL == WINDOWS_FULL_CONTROL
            for rule in rules
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _verify_bundle(directory: Path, names: list[str] | tuple[str, ...]) -> bool:
    return _verify_hardened_path(directory) and all(
        _verify_hardened_path(directory / name) for name in names
    )


def _write_private_file(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        try:
            payload = content.encode("utf-8")
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _harden_path(path)
    except (OSError, ValueError):
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _cleanup_temporary_directory(directory: Path, created_names: list[str]) -> bool:
    succeeded = True
    for name in reversed(created_names):
        try:
            (directory / name).unlink()
        except OSError:
            succeeded = False
    try:
        directory.rmdir()
    except OSError:
        succeeded = False
    return succeeded and not directory.exists()


def _validate_create_paths(
    *,
    workspace_root: Path,
    scope: str,
    run_id: str,
    output_root: Path,
    env_out: Path,
    state_out: Path,
) -> tuple[Path, Path, Path, Path]:
    if scope not in SCOPES or not RUN_ID_PATTERN.fullmatch(run_id):
        raise _fixed_error("materialization request is invalid")
    workspace = workspace_root.resolve(strict=True)
    expected_output_root = workspace / ".runtime"
    resolved_output_root = _as_workspace_path(output_root, workspace)
    if resolved_output_root != expected_output_root:
        raise _fixed_error("materialization path is invalid")
    final_directory = resolved_output_root / f"{scope}-{run_id}"
    resolved_env = _as_workspace_path(env_out, workspace)
    resolved_state = _as_workspace_path(state_out, workspace)
    if (
        resolved_env.parent != final_directory
        or resolved_state.parent != final_directory
        or resolved_env == resolved_state
    ):
        raise _fixed_error("materialization path is invalid")
    if _lstat_or_none(final_directory) is not None:
        raise _fixed_error("materialization target is invalid")
    root_metadata = _lstat_or_none(resolved_output_root)
    if root_metadata is not None and (
        not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode)
    ):
        raise _fixed_error("materialization path is invalid")
    return resolved_output_root, final_directory, resolved_env, resolved_state


def create_secret_bundle(
    *,
    workspace_root: str | os.PathLike[str],
    scope: str,
    run_id: str,
    output_root: str | os.PathLike[str],
    env_out: str | os.PathLike[str],
    state_out: str | os.PathLike[str],
    staging_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_root)
    runtime, final_directory, resolved_env, resolved_state = _validate_create_paths(
        workspace_root=workspace,
        scope=scope,
        run_id=run_id,
        output_root=Path(output_root),
        env_out=Path(env_out),
        state_out=Path(state_out),
    )
    runtime.mkdir(mode=0o700, exist_ok=True)
    metadata = runtime.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise _fixed_error("materialization path is invalid")

    if staging_dir is None:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".tmp-{scope}-{run_id}-", dir=runtime)
        )
    else:
        temporary = Path(staging_dir)
        expected_staging = runtime / f".{scope}-staging-{run_id}"
        try:
            staging_metadata = temporary.lstat()
            if (
                os.path.normcase(os.fspath(temporary))
                != os.path.normcase(os.fspath(expected_staging))
                or not stat.S_ISDIR(staging_metadata.st_mode)
                or stat.S_ISLNK(staging_metadata.st_mode)
                or any(temporary.iterdir())
            ):
                raise OSError
        except OSError:
            raise _fixed_error("materialization path is invalid") from None
    created_names: list[str] = []
    rollback_directory = temporary
    try:
        _harden_path(temporary)
        root_password = secrets.token_hex(32)
        app_password = secrets.token_hex(32)
        values = {
            "mysql_root_password": root_password,
            "mysql_app_password": app_password,
            "mysql_url": (
                "mysql+pymysql://trainfactory_app:"
                f"{quote(app_password, safe='')}@mysql:3306/train_factory"
            ),
            "jwt_secret_key": secrets.token_hex(32),
            "default_admin_password": secrets.token_urlsafe(18),
            "admin_username": "admin",
        }
        for name in SECRET_FILES:
            _write_private_file(temporary / name, values[name])
            created_names.append(name)

        env_lines = [
            f"{variable}={_dotenv_literal(str(final_directory / filename))}"
            for variable, filename in PATH_VARIABLES.items()
        ]
        env_lines.extend(
            (
                "MYSQL_APP_USER=trainfactory_app",
                "DEFAULT_ADMIN_USERNAME=admin",
                "API_PORT=18000",
                "WEB_PORT=3000",
            )
        )
        _write_private_file(temporary / resolved_env.name, "\n".join(env_lines) + "\n")
        created_names.append(resolved_env.name)
        if not _verify_bundle(temporary, created_names):
            raise _fixed_error("secret permission verification failed")

        state_members = [*SECRET_FILES, resolved_env.name, resolved_state.name]
        state = {
            "version": 1,
            "scope": scope,
            "run_id": run_id,
            "directory": str(final_directory),
            "env_file": resolved_env.name,
            "state_file": resolved_state.name,
            "members": state_members,
            "permissions_hardened": True,
            "cleanup_started": False,
        }
        _write_private_file(
            temporary / resolved_state.name,
            json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
        )
        created_names.append(resolved_state.name)
        if not _verify_bundle(temporary, created_names):
            raise _fixed_error("secret permission verification failed")

        if _lstat_or_none(final_directory) is not None:
            raise _fixed_error("materialization target is invalid")
        temporary.rename(final_directory)
        rollback_directory = final_directory
        if not _verify_bundle(final_directory, created_names):
            raise _fixed_error("secret permission verification failed")
    except (OSError, ValueError) as error:
        if not _cleanup_temporary_directory(rollback_directory, created_names):
            raise _fixed_error("secret materialization rollback failed") from None
        if isinstance(error, ValueError):
            raise error
        raise _fixed_error("secret materialization failed") from None

    return {
        "scope": scope,
        "run_id": run_id,
        "files": sorted(created_names),
        "permissions_hardened": True,
    }


def _read_cleanup_state(state_path: Path) -> dict[str, Any]:
    try:
        metadata = state_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise OSError
        if metadata.st_size > 65_536:
            raise OSError
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _fixed_error("cleanup state is invalid") from None
    if not isinstance(state, dict):
        raise _fixed_error("cleanup state is invalid")
    return state


def _write_cleanup_journal(state_file: Path, state: dict[str, Any]) -> None:
    temporary = state_file.parent / f".{state_file.name}.cleanup-tmp"
    if _lstat_or_none(temporary) is not None:
        raise _fixed_error("secret cleanup failed")
    try:
        _write_private_file(
            temporary,
            json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
        )
        if not _verify_hardened_path(temporary):
            raise _fixed_error("secret cleanup failed")
        os.replace(temporary, state_file)
        if not _verify_hardened_path(state_file):
            raise _fixed_error("secret cleanup failed")
    except (OSError, ValueError):
        try:
            if _lstat_or_none(temporary) is not None:
                temporary.unlink()
        except OSError:
            raise _fixed_error("secret cleanup rollback failed") from None
        raise _fixed_error("secret cleanup failed") from None


def cleanup_secret_bundle(
    *,
    workspace_root: str | os.PathLike[str],
    state_path: str | os.PathLike[str],
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve(strict=True)
    runtime = workspace / ".runtime"
    supplied_state = Path(state_path)
    if not supplied_state.is_absolute():
        supplied_state = workspace / supplied_state
    try:
        supplied_metadata = supplied_state.lstat()
        runtime_metadata = runtime.lstat()
    except OSError:
        raise _fixed_error("cleanup state is invalid") from None
    if (
        not stat.S_ISREG(supplied_metadata.st_mode)
        or stat.S_ISLNK(supplied_metadata.st_mode)
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or stat.S_ISLNK(runtime_metadata.st_mode)
    ):
        raise _fixed_error("cleanup state is invalid")
    state_file = supplied_state.resolve(strict=True)
    if state_file.parent.parent != runtime:
        raise _fixed_error("cleanup state is invalid")
    state = _read_cleanup_state(state_file)
    scope = state.get("scope")
    run_id = state.get("run_id")
    directory_value = state.get("directory")
    state_name = state.get("state_file")
    env_name = state.get("env_file")
    members = state.get("members")
    cleanup_started = state.get("cleanup_started")
    expected_state_keys = {
        "version",
        "scope",
        "run_id",
        "directory",
        "env_file",
        "state_file",
        "members",
        "permissions_hardened",
        "cleanup_started",
    }
    if (
        set(state) != expected_state_keys
        or not isinstance(scope, str)
        or scope not in SCOPES
        or not isinstance(run_id, str)
        or not RUN_ID_PATTERN.fullmatch(run_id)
        or not isinstance(directory_value, str)
        or not isinstance(state_name, str)
        or not isinstance(env_name, str)
        or not isinstance(cleanup_started, bool)
        or state.get("version") != 1
        or isinstance(state.get("version"), bool)
        or state.get("permissions_hardened") is not True
    ):
        raise _fixed_error("cleanup state is invalid")
    directory = runtime / f"{scope}-{run_id}"
    if (
        state_file.parent != directory
        or directory_value != str(directory)
        or state_name != state_file.name
        or Path(env_name).name != env_name
        or Path(state_name).name != state_name
    ):
        raise _fixed_error("cleanup state is invalid")
    if (
        not isinstance(members, list)
        or not all(
            isinstance(name, str) and Path(name).name == name for name in members
        )
        or len(members) != len(set(members))
    ):
        raise _fixed_error("cleanup state is invalid")
    expected = set(SECRET_FILES) | {env_name, state_file.name}
    if set(members) != expected:
        raise _fixed_error("cleanup state is invalid")
    try:
        directory_metadata = directory.lstat()
        actual_names = {entry.name for entry in directory.iterdir()}
    except OSError:
        raise _fixed_error("cleanup state is invalid") from None
    if (
        not stat.S_ISDIR(directory_metadata.st_mode)
        or stat.S_ISLNK(directory_metadata.st_mode)
        or state_file.name not in actual_names
        or (
            actual_names != expected
            if not cleanup_started
            else not actual_names <= expected
        )
        or not _verify_hardened_path(directory)
    ):
        raise _fixed_error("cleanup state is invalid")
    for name in sorted(actual_names):
        member = directory / name
        try:
            member_metadata = member.lstat()
        except OSError:
            raise _fixed_error("cleanup state is invalid") from None
        if (
            not stat.S_ISREG(member_metadata.st_mode)
            or stat.S_ISLNK(member_metadata.st_mode)
            or not _verify_hardened_path(member)
        ):
            raise _fixed_error("cleanup state is invalid")

    if not cleanup_started:
        state["cleanup_started"] = True
        _write_cleanup_journal(state_file, state)

    try:
        for name in sorted(expected - {state_file.name}):
            member = directory / name
            if _lstat_or_none(member) is not None:
                member.unlink()
        state_file.unlink()
        directory.rmdir()
    except OSError:
        if directory.exists() and not state_file.exists():
            try:
                _write_private_file(
                    state_file,
                    json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
                )
                if not _verify_hardened_path(state_file):
                    raise OSError
            except (OSError, ValueError):
                raise _fixed_error("secret cleanup rollback failed") from None
        raise _fixed_error("secret cleanup failed") from None
    if directory.exists():
        raise _fixed_error("secret cleanup failed")
    return {"scope": scope, "run_id": run_id, "removed": True}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--scope", required=True, choices=sorted(SCOPES))
    create.add_argument("--run-id", required=True)
    create.add_argument("--output-root", required=True)
    create.add_argument("--env-out", required=True)
    create.add_argument("--state-out", required=True)
    create.add_argument("--staging-dir")
    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--state", required=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    workspace_root: str | os.PathLike[str] = ROOT_DIR,
) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "create":
            result = create_secret_bundle(
                workspace_root=workspace_root,
                scope=args.scope,
                run_id=args.run_id,
                output_root=args.output_root,
                env_out=args.env_out,
                state_out=args.state_out,
                staging_dir=args.staging_dir,
            )
        else:
            result = cleanup_secret_bundle(
                workspace_root=workspace_root,
                state_path=args.state,
            )
    except ValueError:
        print("secret materialization operation failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
