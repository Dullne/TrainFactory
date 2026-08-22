import os
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest


ROOT_DIR = Path(__file__).parents[1]
ENTRYPOINT = ROOT_DIR / "docker" / "entrypoint.sh"
DOCKERFILE = ROOT_DIR / "docker" / "Dockerfile"


def _run_entrypoint(tmp_path, environment):
    result_file = tmp_path / "observed-environment"
    observer = tmp_path / "observe.sh"
    observer.write_text(
        "#!/bin/sh\n"
        'if [ "${MYSQL_URL+x}" = x ]; then\n'
        '  printf \'set\\n%s\' "$MYSQL_URL" > "$RESULT_FILE"\n'
        "else\n"
        "  printf 'unset' > \"$RESULT_FILE\"\n"
        "fi\n",
        encoding="utf-8",
        newline="\n",
    )
    run_environment = os.environ.copy()
    for name in (
        "MYSQL_URL",
        "MYSQL_URL_FILE",
        "MYSQL_APP_USER",
        "MYSQL_APP_PASSWORD",
        "MYSQL_HOST",
        "MYSQL_DATABASE",
        "GPU_PREFLIGHT_MODE",
        "NVIDIA_DISABLE_REQUIRE",
    ):
        run_environment.pop(name, None)
    run_environment.update(environment)
    run_environment.setdefault("GPU_PREFLIGHT_MODE", "off")
    run_environment["RESULT_FILE"] = result_file.as_posix()
    completed = subprocess.run(
        ["sh", "docker/entrypoint.sh", "sh", observer.as_posix()],
        cwd=ROOT_DIR,
        env=run_environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    observed = result_file.read_text(encoding="utf-8") if result_file.exists() else None
    return completed, observed


def _assert_public_off_preflight(completed):
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload == {
        "compiled_cuda": None,
        "device_count": 0,
        "driver_version": None,
        "mode": "off",
        "ok": True,
        "probe_executed": False,
        "reason_code": None,
    }


def test_entrypoint_derives_url_with_strict_component_encoding_without_output(tmp_path):
    password = "entrypoint-private-@:/%-canary"
    environment = {
        "MYSQL_APP_USER": "app user",
        "MYSQL_APP_PASSWORD": password,
        "MYSQL_HOST": "mysql",
        "MYSQL_DATABASE": "train factory",
    }

    completed, observed = _run_entrypoint(tmp_path, environment)

    assert completed.returncode == 0
    _assert_public_off_preflight(completed)
    assert password not in completed.stderr
    expected = (
        "set\nmysql+pymysql://"
        f"{quote(environment['MYSQL_APP_USER'], safe='')}:"
        f"{quote(password, safe='')}@mysql:3306/"
        f"{quote(environment['MYSQL_DATABASE'], safe='')}"
    )
    assert observed == expected


def test_entrypoint_does_not_derive_when_file_source_is_present(tmp_path):
    password = "file-mode-alias-private-canary-@:/%"
    completed, observed = _run_entrypoint(
        tmp_path,
        {
            "MYSQL_URL_FILE": "/run/secrets/mysql_url",
            "MYSQL_APP_USER": "trainfactory_app",
            "MYSQL_APP_PASSWORD": password,
            "MYSQL_HOST": "mysql",
            "MYSQL_DATABASE": "train_factory",
        },
    )

    assert completed.returncode == 0
    assert observed == "unset"
    _assert_public_off_preflight(completed)
    assert password not in completed.stdout
    assert password not in completed.stderr


def test_entrypoint_preserves_explicit_url_without_printing_it(tmp_path):
    url = "mysql+pymysql://app:explicit-private-canary@mysql/train_factory"

    completed, observed = _run_entrypoint(tmp_path, {"MYSQL_URL": url})

    assert completed.returncode == 0
    assert observed == f"set\n{url}"
    _assert_public_off_preflight(completed)
    assert url not in completed.stdout
    assert url not in completed.stderr


def test_entrypoint_missing_derivation_input_fails_without_secret_output(tmp_path):
    password = "missing-input-private-canary"
    completed, observed = _run_entrypoint(
        tmp_path,
        {
            "MYSQL_APP_USER": "trainfactory_app",
            "MYSQL_APP_PASSWORD": password,
            "MYSQL_HOST": "mysql",
        },
    )

    assert completed.returncode != 0
    assert observed is None
    assert completed.stdout == ""
    assert password not in completed.stdout
    assert password not in completed.stderr


def test_entrypoint_allows_gpu_one_shot_without_database_inputs(tmp_path):
    completed, observed = _run_entrypoint(tmp_path, {})

    assert completed.returncode == 0
    assert observed == "unset"
    _assert_public_off_preflight(completed)


def test_entrypoint_runs_database_resolution_before_gpu_probe_then_exec():
    source = ENTRYPOINT.read_text(encoding="utf-8")

    derivation = source.index('MYSQL_URL="$(python -c')
    preflight = source.index("python -m train_factory.runtime.gpu_preflight")
    final_exec = source.index('exec "$@"')
    assert derivation < preflight < final_exec
    assert '${GPU_PREFLIGHT_MODE:-required}' in source


def test_entrypoint_preflight_failure_stops_before_exec_without_echo(tmp_path):
    canary = "private-mode-canary"
    completed, observed = _run_entrypoint(
        tmp_path,
        {"GPU_PREFLIGHT_MODE": canary},
    )

    assert completed.returncode == 2
    assert observed is None
    assert completed.stdout == ""
    assert completed.stderr == "gpu preflight arguments are invalid\n"
    assert canary not in completed.stderr


@pytest.mark.parametrize("override", ("1", "true", "TRUE", "private-canary"))
def test_compat_override_cannot_be_combined_with_disabled_preflight(
    tmp_path, override
):
    completed, observed = _run_entrypoint(
        tmp_path,
        {"NVIDIA_DISABLE_REQUIRE": override, "GPU_PREFLIGHT_MODE": "off"},
    )

    assert completed.returncode != 0
    assert observed is None
    assert completed.stdout == ""
    assert completed.stderr == "GPU compatibility policy is invalid\n"
    assert "private-canary" not in completed.stderr


def test_cpu_empty_override_with_disabled_preflight_executes(tmp_path):
    completed, observed = _run_entrypoint(
        tmp_path,
        {"NVIDIA_DISABLE_REQUIRE": "", "GPU_PREFLIGHT_MODE": "off"},
    )

    assert completed.returncode == 0
    assert observed == "unset"
    _assert_public_off_preflight(completed)


@pytest.mark.parametrize("override", ("1", "true", "private-canary"))
def test_compat_override_with_required_mode_reaches_the_gpu_probe(
    tmp_path, override
):
    completed, observed = _run_entrypoint(
        tmp_path,
        {"NVIDIA_DISABLE_REQUIRE": override, "GPU_PREFLIGHT_MODE": "required"},
    )

    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "required"
    assert "GPU compatibility policy is invalid" not in completed.stdout
    if completed.returncode == 0:
        assert observed == "unset"
    else:
        assert completed.returncode == 1
        assert observed is None
    assert "private-canary" not in completed.stdout


def test_production_image_installs_exec_form_entrypoint_and_lf_contract():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    attributes = (ROOT_DIR / ".gitattributes").read_text(encoding="utf-8")

    assert "COPY --chmod=0555 docker/entrypoint.sh /app/docker/entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/app/docker/entrypoint.sh"]' in dockerfile
    assert "GPU_PREFLIGHT_MODE=required" in dockerfile
    assert 'CMD ["python", "-m", "train_factory.api.server"]' in dockerfile
    assert "*.sh text eol=lf" in attributes
