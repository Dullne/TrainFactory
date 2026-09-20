"""Exercise the sourced Nginx upload hook; no application or network is needed."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
HOOK = ROOT_DIR / "web" / "docker-entrypoint.d" / "19-upload-limit.envsh"


def _upload_environment(value):
    environment = os.environ.copy()
    environment.pop("MAX_UPLOAD_SIZE", None)
    environment.pop("WEB_MAX_BODY_SIZE", None)
    if value is not None:
        environment["MAX_UPLOAD_SIZE"] = value
    return subprocess.run(
        ["sh", "-c", '. "$1" && printf "%s" "$WEB_MAX_BODY_SIZE"', "upload-test", str(HOOK)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("file_limit", "expected_body_limit"),
    [
        (None, "525336576"),
        ("2097152", "3145728"),
        ("0002097152", "3145728"),
        ("1", "1048577"),
        ("9223372036853727231", "9223372036854775807"),
    ],
)
def test_upload_body_budget_includes_one_mib_for_multipart(file_limit, expected_body_limit):
    result = _upload_environment(file_limit)
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected_body_limit


@pytest.mark.parametrize(
    "file_limit",
    ["", "0", "000", "-1", "1.5", "500m", " 1024", "1024\n", "9223372036853727232", "9" * 40],
)
def test_invalid_upload_budget_fails_closed_without_rendering_a_limit(file_limit):
    result = _upload_environment(file_limit)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "MAX_UPLOAD_SIZE must be a positive integer" in result.stderr


@pytest.mark.parametrize("file_limit", [0, -1, 9223372036853727232])
def test_api_rejects_upload_budgets_the_proxy_cannot_enforce(file_limit):
    from train_factory.config.settings import Settings

    with pytest.raises(ValueError, match="max_upload_size"):
        Settings(_env_file=None, max_upload_size=file_limit)
