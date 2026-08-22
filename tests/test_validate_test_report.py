import html
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).parents[1]


def _module():
    from scripts import validate_test_report

    return validate_test_report


def _write_report(path, body=""):
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuites tests="1" failures="0" errors="0">'
        '<testsuite name="backend" tests="1" failures="0" errors="0">'
        f'<testcase classname="tests.test_safe" name="test_safe">{body}</testcase>'
        "</testsuite></testsuites>\n",
        encoding="utf-8",
    )


def _private(_path):
    return True


def _arguments(report_root, report, *secret_files):
    validated_output = report_root / f"validated-{report.name}"
    arguments = [
        "--report-root",
        str(report_root),
        "--report",
        str(report),
        "--validated-output",
        str(validated_output),
    ]
    for secret_file in secret_files:
        arguments.extend(("--secret-file", str(secret_file)))
    return arguments, validated_output


def test_clean_junit_report_passes_without_output(tmp_path, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)

    arguments, validated_output = _arguments(report_root, report)
    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""
    assert validated_output.read_bytes() == report.read_bytes()


def test_secret_file_value_in_junit_is_rejected_without_echo(tmp_path, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    canary = "private-junit-canary-98d6f3c1"
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text(canary + "\n", encoding="utf-8")
    report = report_root / "backend.xml"
    _write_report(report, f"<system-out>{canary}</system-out>")

    arguments, validated_output = _arguments(report_root, report, secret_file)
    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "backend.xml:secret_value\n"
    assert canary not in captured.err
    assert not validated_output.exists()


@pytest.mark.parametrize(
    ("payload", "rule_id"),
    (
        (
            "mysql+pymysql://trainfactory_app:private@127.0.0.1:3306/train_factory",
            "connection_url",
        ),
        ("password=private-report-password", "password_value"),
        (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJwcml2YXRlIn0.signature123456",
            "jwt_value",
        ),
    ),
    ids=("connection-url", "password-assignment", "jwt-token"),
)
def test_sensitive_report_patterns_are_rejected_with_rule_only(
    tmp_path,
    capsys,
    payload,
    rule_id,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "host-policy.xml"
    _write_report(report, f"<system-err>{payload}</system-err>")

    arguments, _validated_output = _arguments(report_root, report)
    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == f"host-policy.xml:{rule_id}\n"
    assert payload not in captured.err


@pytest.mark.parametrize(
    "payload",
    (
        "MYSQL_ROOT_PASSWORD=private-report-value",
        "MYSQL_PASSWORD:private-report-value",
        "DEFAULT_ADMIN_PASSWORD='private-report-value'",
        '{"MYSQL_ROOT_PASSWORD":"private-report-value"}',
    ),
    ids=("root-env", "mysql-env", "admin-env", "json-env"),
)
def test_environment_password_values_are_rejected(tmp_path, payload):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report, f"<system-err>{html.escape(payload)}</system-err>")

    assert module.validate_report(
        report_root=report_root,
        report=report,
        verify_hardened=_private,
    ) == ["password_value"]


@pytest.mark.parametrize(
    "payload",
    (
        "MYSQL_ROOT_PASSWORD=****",
        "MYSQL_PASSWORD:<redacted>",
        "DEFAULT_ADMIN_PASSWORD=redacted",
        "PASSWORD=null",
        '{"MYSQL_ROOT_PASSWORD":""}',
    ),
    ids=("stars", "tag", "word", "null", "empty-json"),
)
def test_redacted_environment_password_values_are_allowed(tmp_path, payload):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report, f"<system-out>{html.escape(payload)}</system-out>")

    assert module.validate_report(
        report_root=report_root,
        report=report,
        verify_hardened=_private,
    ) == []


@pytest.mark.parametrize(
    "payload",
    (
        "JWT_SECRET_KEY=0123456789abcdef0123456789abcdef",
        "APP_JWT_SECRET:private-jwt-report-value",
        '{"JWT_SECRET_KEY":"private-jwt-report-value"}',
    ),
    ids=("jwt-env", "prefixed-jwt", "json-jwt"),
)
def test_jwt_secret_assignments_are_rejected(tmp_path, payload):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report, f"<system-err>{html.escape(payload)}</system-err>")

    assert module.validate_report(
        report_root=report_root,
        report=report,
        verify_hardened=_private,
    ) == ["jwt_value"]


@pytest.mark.parametrize(
    "payload",
    (
        "JWT_SECRET_KEY=****",
        "JWT_SECRET_KEY=<redacted>",
        "JWT_SECRET_KEY=null",
        '{"JWT_SECRET_KEY":""}',
    ),
    ids=("stars", "tag", "null", "empty-json"),
)
def test_redacted_jwt_secret_assignments_are_allowed(tmp_path, payload):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report, f"<system-out>{html.escape(payload)}</system-out>")

    assert module.validate_report(
        report_root=report_root,
        report=report,
        verify_hardened=_private,
    ) == []


@pytest.mark.parametrize("kind", ("outside", "directory", "symlink", "oversized"))
def test_report_path_policy_fails_closed(tmp_path, capsys, kind):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    if kind == "outside":
        report = tmp_path / "outside.xml"
        _write_report(report)
    elif kind == "directory":
        report = report_root / "directory.xml"
        report.mkdir()
    elif kind == "symlink":
        target = report_root / "target.xml"
        _write_report(target)
        report.unlink()
        try:
            report.symlink_to(target)
        except OSError:
            pytest.skip("symlinks are unavailable")
    elif kind == "oversized":
        report.write_bytes(b"x" * (module.MAX_REPORT_BYTES + 1))

    arguments, _validated_output = _arguments(report_root, report)
    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "test report validation failed\n"


@pytest.mark.parametrize("kind", ("missing", "directory", "symlink", "oversized", "acl"))
def test_secret_file_policy_fails_closed(tmp_path, capsys, kind):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private-secret-value\n", encoding="utf-8")
    verify = _private
    if kind == "missing":
        secret_file.unlink()
    elif kind == "directory":
        secret_file.unlink()
        secret_file.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target-secret.txt"
        target.write_text("private-secret-value\n", encoding="utf-8")
        secret_file.unlink()
        try:
            secret_file.symlink_to(target)
        except OSError:
            pytest.skip("symlinks are unavailable")
    elif kind == "oversized":
        secret_file.write_bytes(b"x" * (module.MAX_SECRET_BYTES + 1))
    elif kind == "acl":
        def verify(_path):
            return False

    arguments, _validated_output = _arguments(report_root, report, secret_file)
    result = module.main(
        arguments,
        verify_hardened=verify,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "test report validation failed\n"


def test_secret_acl_is_rechecked_after_stable_read(tmp_path, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private-secret-value\n", encoding="utf-8")
    checks = iter((True, False))

    arguments, _validated_output = _arguments(report_root, report, secret_file)
    result = module.main(
        arguments,
        verify_hardened=lambda _path: next(checks),
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"


def test_report_identity_change_during_read_is_rejected(tmp_path, monkeypatch, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    original_lstat = Path.lstat
    calls = 0

    def changed_lstat(path):
        nonlocal calls
        value = original_lstat(path)
        if path == report:
            calls += 1
            if calls > 1:
                fields = list(value)
                fields[stat.ST_MTIME] += 1
                return os.stat_result(fields)
        return value

    monkeypatch.setattr(Path, "lstat", changed_lstat)

    arguments, _validated_output = _arguments(report_root, report)
    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"


def test_malformed_xml_and_unknown_argument_use_fixed_errors(tmp_path, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    report.write_text("<private-canary", encoding="utf-8")

    arguments, _validated_output = _arguments(report_root, report)
    assert module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    ) == 2
    malformed = capsys.readouterr()
    assert malformed.err == "test report validation failed\n"
    assert module.main(["--unknown", "private-argument-canary"]) == 2
    unknown = capsys.readouterr()
    assert unknown.err == "test report validation failed\n"
    assert "private" not in unknown.err


def test_report_validator_isolated_cli_ignores_ambient_sitecustomize(tmp_path):
    canary = "REPORT-VALIDATOR-SITECUSTOMIZE-CANARY"
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
            os.fspath(ROOT_DIR / "scripts" / "validate_test_report.py"),
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


def test_raw_secret_scan_precedes_xml_parsing(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    secret = "private-before-xml-canary"
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text(secret + "\n", encoding="utf-8")
    report = report_root / "malformed.xml"
    report.write_text(f"<broken>{secret}", encoding="utf-8")

    assert module.validate_report(
        report_root=report_root,
        report=report,
        secret_files=(secret_file,),
        verify_hardened=_private,
    ) == ["secret_value"]


@pytest.mark.parametrize(
    "body",
    (
        '<failure message="private&amp;canary"/>',
        "<system-out>private&amp;canary</system-out>",
        "<system-out>safe</system-out>private&amp;canary",
    ),
    ids=("attribute", "text", "tail"),
)
def test_xml_decoded_secret_values_are_rejected(tmp_path, body):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private&canary\n", encoding="utf-8")
    report = report_root / "backend.xml"
    _write_report(report, body)

    assert module.validate_report(
        report_root=report_root,
        report=report,
        secret_files=(secret_file,),
        verify_hardened=_private,
    ) == ["secret_value"]


@pytest.mark.parametrize(
    "encoded",
    ("private%2Bcanary", "private%252Bcanary"),
    ids=("single", "double"),
)
def test_percent_encoded_secret_values_are_rejected_with_bounded_decoding(
    tmp_path,
    encoded,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private+canary\n", encoding="utf-8")
    report = report_root / "backend.xml"
    _write_report(report, f"<system-out>{encoded}</system-out>")

    assert module.validate_report(
        report_root=report_root,
        report=report,
        secret_files=(secret_file,),
        verify_hardened=_private,
    ) == ["secret_value"]


def test_percent_decoding_beyond_bound_fails_closed(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report, "<system-out>private%25252Bcanary</system-out>")

    with pytest.raises(module.ReportError, match="^test report validation failed$"):
        module.validate_report(
            report_root=report_root,
            report=report,
            verify_hardened=_private,
        )


def test_secret_split_across_xml_text_nodes_is_rejected(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("privatecanary\n", encoding="utf-8")
    report = report_root / "backend.xml"
    _write_report(report, "<system-out>private<marker/>canary</system-out>")

    assert module.validate_report(
        report_root=report_root,
        report=report,
        secret_files=(secret_file,),
        verify_hardened=_private,
    ) == ["secret_value"]


def test_form_encoded_secret_value_is_rejected(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private canary\n", encoding="utf-8")
    report = report_root / "backend.xml"
    _write_report(report, "<system-out>private+canary</system-out>")

    assert module.validate_report(
        report_root=report_root,
        report=report,
        secret_files=(secret_file,),
        verify_hardened=_private,
    ) == ["secret_value"]


def test_secret_file_noncanonical_parent_traversal_is_rejected(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    (tmp_path / "child").mkdir()
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private-secret-value\n", encoding="utf-8")
    noncanonical = tmp_path / "child" / ".." / secret_file.name

    with pytest.raises(module.ReportError, match="^test report validation failed$"):
        module.validate_report(
            report_root=report_root,
            report=report,
            secret_files=(noncanonical,),
            verify_hardened=_private,
        )


def test_secret_ancestor_is_rechecked_after_scan(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    secret_parent = tmp_path / "secret-parent"
    secret_parent.mkdir()
    secret_file = secret_parent / "job-secret.txt"
    secret_file.write_text("private-secret-value\n", encoding="utf-8")
    moved_parent = tmp_path / "moved-secret-parent"
    checks = 0

    def verify(_path):
        nonlocal checks
        checks += 1
        if checks == 3:
            secret_parent.rename(moved_parent)
            try:
                secret_parent.symlink_to(moved_parent, target_is_directory=True)
            except OSError:
                moved_parent.rename(secret_parent)
                pytest.skip("directory symlinks are unavailable")
        return True

    with pytest.raises(module.ReportError, match="^test report validation failed$"):
        module.validate_report(
            report_root=report_root,
            report=report,
            secret_files=(secret_file,),
            verify_hardened=verify,
        )


def test_validated_output_is_exclusive_and_never_overwrites_existing_file(
    tmp_path,
    capsys,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    arguments, validated_output = _arguments(report_root, report)
    validated_output.write_text("CONCURRENT-CANARY", encoding="utf-8")

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"
    assert validated_output.read_text(encoding="utf-8") == "CONCURRENT-CANARY"


def test_source_replacement_after_validation_cannot_change_published_bytes(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    safe_payload = report.read_bytes()
    arguments, validated_output = _arguments(report_root, report)
    original_link = os.link
    canary = "PRIVATE-UPLOAD-WINDOW-CANARY"

    def replace_source_after_link(source, target):
        original_link(source, target)
        replacement = report_root / "replacement.xml"
        _write_report(replacement, f"<system-out>{canary}</system-out>")
        os.replace(replacement, report)

    monkeypatch.setattr(os, "link", replace_source_after_link)

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert validated_output.read_bytes() == safe_payload
    assert canary.encode() not in validated_output.read_bytes()


def test_validated_output_replacement_is_not_accepted_or_deleted(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    arguments, validated_output = _arguments(report_root, report)
    original_link = os.link
    canary = b"CONCURRENT-REPLACEMENT-CANARY"

    def replace_target_after_link(source, target):
        original_link(source, target)
        replacement = report_root / "external.xml"
        replacement.write_bytes(canary)
        os.replace(replacement, target)

    monkeypatch.setattr(os, "link", replace_target_after_link)

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"
    assert validated_output.read_bytes() == canary


def test_validated_temp_collision_is_not_deleted(tmp_path, monkeypatch, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    arguments, _validated_output = _arguments(report_root, report)
    monkeypatch.setattr(module.secrets, "token_hex", lambda _size: "fixed")
    foreign_temp = report_root / ".validated-backend.xml.fixed.tmp"
    foreign_temp.write_text("FOREIGN-TEMP-CANARY", encoding="utf-8")

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"
    assert foreign_temp.read_text(encoding="utf-8") == "FOREIGN-TEMP-CANARY"


def test_validated_temp_replacement_is_not_deleted(tmp_path, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    arguments, validated_output = _arguments(report_root, report)
    replaced_temp = None

    def replace_temp(path):
        nonlocal replaced_temp
        replacement = report_root / "foreign-temp"
        replacement.write_text("FOREIGN-REPLACEMENT-CANARY", encoding="utf-8")
        os.replace(replacement, path)
        replaced_temp = path

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=replace_temp,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"
    assert replaced_temp is not None
    assert replaced_temp.read_text(encoding="utf-8") == "FOREIGN-REPLACEMENT-CANARY"
    assert not validated_output.exists()


def test_link_side_effect_then_error_cleans_only_owned_target(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    arguments, validated_output = _arguments(report_root, report)
    original_link = os.link

    def link_then_fail(source, target):
        original_link(source, target)
        raise OSError("link-result-canary")

    monkeypatch.setattr(os, "link", link_then_fail)

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"
    assert not validated_output.exists()


def test_hardening_runtime_error_is_fixed_and_cleans_owned_temp(tmp_path, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    arguments, validated_output = _arguments(report_root, report)

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: (_ for _ in ()).throw(RuntimeError("private-canary")),
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"
    assert not validated_output.exists()
    assert list(report_root.glob("*.tmp")) == []


def test_link_callback_temp_replacement_is_preserved_and_target_is_cleaned(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    arguments, validated_output = _arguments(report_root, report)
    original_link = os.link
    foreign_temp = None

    def link_then_replace_temp(source, target):
        nonlocal foreign_temp
        original_link(source, target)
        replacement = report_root / "foreign-after-link"
        replacement.write_text("FOREIGN-AFTER-LINK-CANARY", encoding="utf-8")
        os.replace(replacement, source)
        foreign_temp = Path(source)

    monkeypatch.setattr(os, "link", link_then_replace_temp)

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"
    assert foreign_temp is not None
    assert foreign_temp.read_text(encoding="utf-8") == "FOREIGN-AFTER-LINK-CANARY"
    assert not validated_output.exists()


@pytest.mark.parametrize(
    "failure",
    ("initial-fstat", "write", "fsync", "post-write-fstat"),
)
def test_publish_io_failure_cleans_opened_owned_temp(
    tmp_path,
    monkeypatch,
    capsys,
    failure,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    arguments, validated_output = _arguments(report_root, report)
    original_write = os.write
    original_fsync = os.fsync
    original_fstat = os.fstat
    original_open = os.open
    write_seen = False
    fstat_calls = 0
    publish_descriptor = None

    def controlled_open(path, flags, mode=0o777):
        nonlocal publish_descriptor
        descriptor = original_open(path, flags, mode)
        if flags & os.O_WRONLY:
            publish_descriptor = descriptor
        return descriptor

    def controlled_write(descriptor, payload):
        nonlocal write_seen
        write_seen = True
        if failure == "write":
            raise OSError("write-canary")
        return original_write(descriptor, payload)

    def controlled_fsync(descriptor):
        if failure == "fsync":
            raise OSError("fsync-canary")
        return original_fsync(descriptor)

    def controlled_fstat(descriptor):
        nonlocal fstat_calls
        if descriptor != publish_descriptor:
            return original_fstat(descriptor)
        fstat_calls += 1
        if failure == "initial-fstat" and fstat_calls == 1:
            raise OSError("initial-fstat-canary")
        if failure == "post-write-fstat" and write_seen:
            raise OSError("fstat-canary")
        return original_fstat(descriptor)

    monkeypatch.setattr(os, "open", controlled_open)
    monkeypatch.setattr(os, "write", controlled_write)
    monkeypatch.setattr(os, "fsync", controlled_fsync)
    monkeypatch.setattr(os, "fstat", controlled_fstat)

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "test report validation failed\n"
    assert not validated_output.exists()
    assert list(report_root.glob("*.tmp")) == []


def test_stable_read_rejects_replacement_during_final_acl_check(tmp_path):
    module = _module()
    private_file = tmp_path / "private.txt"
    private_file.write_bytes(b"ORIGINAL-PRIVATE")
    checks = 0

    def verify(_path):
        nonlocal checks
        checks += 1
        if checks == 2:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"FOREIGN-PRIVATE!")
            os.replace(replacement, private_file)
        return True

    with pytest.raises(module.ReportError, match="^test report validation failed$"):
        module._stable_read(
            private_file,
            max_bytes=module.MAX_SECRET_BYTES,
            private=True,
            verify_hardened=verify,
        )


def test_validator_test_junit_contains_no_sensitive_parameter_ids(tmp_path, capsys):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "validator-tests.xml"
    selected_nodes = (
        "tests/test_validate_test_report.py::"
        "test_sensitive_report_patterns_are_rejected_with_rule_only",
        "tests/test_validate_test_report.py::"
        "test_environment_password_values_are_rejected",
        "tests/test_validate_test_report.py::test_jwt_secret_assignments_are_rejected",
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
            f"--junitxml={report}",
            *selected_nodes,
        ],
        cwd=ROOT_DIR,
        env=environment,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0
    arguments, validated_output = _arguments(report_root, report)

    result = module.main(
        arguments,
        verify_hardened=_private,
        harden=lambda _path: True,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert validated_output.read_bytes() == report.read_bytes()


def test_secret_file_reparse_ancestor_is_rejected(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    real_parent = tmp_path / "real-secrets"
    real_parent.mkdir()
    (real_parent / "job-secret.txt").write_text(
        "private-secret-value\n",
        encoding="utf-8",
    )
    alias = tmp_path / "secret-alias"
    try:
        alias.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(module.ReportError, match="^test report validation failed$"):
        module.validate_report(
            report_root=report_root,
            report=report,
            secret_files=(alias / "job-secret.txt",),
            verify_hardened=_private,
        )


def test_final_secret_acl_callback_cannot_replace_scanned_report(
    tmp_path,
):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private-final-canary\n", encoding="utf-8")
    checks = 0

    def verify(_path):
        nonlocal checks
        checks += 1
        if checks == 3:
            replacement = report_root / "replacement.xml"
            _write_report(
                replacement,
                "<system-out>private-final-canary</system-out>",
            )
            os.replace(replacement, report)
        return True

    with pytest.raises(module.ReportError, match="^test report validation failed$"):
        module.validate_report(
            report_root=report_root,
            report=report,
            secret_files=(secret_file,),
            verify_hardened=verify,
        )


def test_final_secret_reread_rejects_same_identity_same_metadata_rewrite(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report)
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private-alpha\n", encoding="utf-8")
    original = secret_file.stat()
    checks = 0

    def verify(_path):
        nonlocal checks
        checks += 1
        if checks == 3:
            secret_file.write_text("private-bravo\n", encoding="utf-8")
            os.utime(
                secret_file,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )
        return True

    with pytest.raises(module.ReportError, match="^test report validation failed$"):
        module.validate_report(
            report_root=report_root,
            report=report,
            secret_files=(secret_file,),
            verify_hardened=verify,
        )


def test_final_report_reread_rejects_same_identity_same_metadata_rewrite(tmp_path):
    module = _module()
    report_root = tmp_path / "trainfactory-reports"
    report_root.mkdir()
    report = report_root / "backend.xml"
    _write_report(report, "<system-out>safe-alpha</system-out>")
    original_payload = report.read_bytes()
    original = report.stat()
    secret_file = tmp_path / "job-secret.txt"
    secret_file.write_text("private-report-value\n", encoding="utf-8")
    checks = 0

    def verify(_path):
        nonlocal checks
        checks += 1
        if checks == 3:
            changed = original_payload.replace(b"safe-alpha", b"safe-bravo")
            assert len(changed) == len(original_payload)
            report.write_bytes(changed)
            os.utime(report, ns=(original.st_atime_ns, original.st_mtime_ns))
        return True

    with pytest.raises(module.ReportError, match="^test report validation failed$"):
        module.validate_report(
            report_root=report_root,
            report=report,
            secret_files=(secret_file,),
            verify_hardened=verify,
        )
