"""Opt-in acceptance against an existing Nginx image, isolated from real services."""

import os
from pathlib import Path
import re
import subprocess
import time

import pytest


pytestmark = pytest.mark.host_tools
ROOT = Path(__file__).resolve().parents[1]


def _docker(*args):
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=30, check=False
    )


@pytest.mark.parametrize("configuration", ["workspace", "image"])
def test_nginx_upload_budget_is_rendered_and_scoped_to_uploads(tmp_path, configuration):
    image = os.environ.get("TRAINFACTORY_UPLOAD_TEST_IMAGE")
    if not image:
        pytest.skip("Set TRAINFACTORY_UPLOAD_TEST_IMAGE to a locally available Web image")
    inspected = _docker("image", "inspect", "--format", "{{.Id}}", image)
    assert inspected.returncode == 0, "The upload test image must already exist locally"
    image_id = inspected.stdout.strip()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)

    upstream = tmp_path / "upstream.conf"
    upstream.write_text(
        'server { listen 8081; client_max_body_size 8m; '
        'location / { return 200 "upstream-ok"; } }\n', encoding="utf-8"
    )
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    boundary = "trainfactory-upload-test"
    multipart = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        'filename="sample.jsonl"\r\nContent-Type: application/octet-stream\r\n\r\n'
    ).encode() + b"x" * (2 * 1024 * 1024) + f"\r\n--{boundary}--\r\n".encode()
    (payloads / "allowed.body").write_bytes(multipart)
    (payloads / "oversized.body").write_bytes(b"x" * (4 * 1024 * 1024))

    arguments = ["run", "-d", "--rm", "--network", "none"]
    mounts = [
        (upstream, "/etc/nginx/conf.d/upload-test-upstream.conf"),
        (payloads, "/tmp/upload-test-payloads"),
    ]
    if configuration == "workspace":
        mounts.extend([
            (ROOT / "web/nginx.conf", "/etc/nginx/templates/default.conf.template"),
            (ROOT / "web/docker-entrypoint.d/19-upload-limit.envsh",
             "/docker-entrypoint.d/19-upload-limit.envsh"),
        ])
    for source, target in mounts:
        arguments.extend(["--mount", f"type=bind,source={source},target={target},readonly"])
    arguments.extend([
        "-e", "API_HOST=127.0.0.1", "-e", "API_PORT=8081",
        "-e", "MAX_UPLOAD_SIZE=2097152", image_id,
    ])
    started = _docker(*arguments)
    assert started.returncode == 0, started.stderr
    container_id = started.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", container_id)
    try:
        for _ in range(40):
            ready = _docker("exec", container_id, "wget", "-q", "-O-", "http://127.0.0.1/health")
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            pytest.fail("Isolated upload proxy did not become ready: " + _docker("logs", container_id).stdout)

        def post(path, payload):
            return _docker(
                "exec", container_id, "wget", "-S", "-O-",
                "--header", f"Content-Type: multipart/form-data; boundary={boundary}",
                "--post-file", f"/tmp/upload-test-payloads/{payload}",
                f"http://127.0.0.1{path}",
            )

        allowed = post("/api/datasets/upload", "allowed.body")
        assert allowed.returncode == 0, allowed.stderr
        assert allowed.stdout == "upstream-ok"
        for path, payload in (
            ("/api/datasets/upload", "oversized.body"),
            ("/api/ordinary-request", "allowed.body"),
        ):
            denied = post(path, payload)
            assert denied.returncode != 0
            assert "413" in denied.stderr, denied.stderr
    finally:
        # Only the exact disposable container created by this test is removed.
        removed = _docker("rm", "-f", container_id)
        assert removed.returncode == 0, removed.stderr
