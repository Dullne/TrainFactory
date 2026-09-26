"""Web readiness must reflect API readiness in the production compose stack."""

import os
from pathlib import Path
import re
import subprocess
import time

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _docker(*args):
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=30, check=False
    )


def _request(container_id, path):
    response = _docker(
        "exec",
        container_id,
        "curl",
        "--http1.0",
        "--include",
        "--silent",
        "--show-error",
        f"http://127.0.0.1{path}",
    )
    assert response.returncode == 0, response.stderr
    parts = re.split(r"\r?\n\r?\n", response.stdout, maxsplit=1)
    assert len(parts) == 2, repr(response.stdout)
    headers, body = parts
    return headers, body


def test_web_health_proxies_api_and_waits_for_api_readiness():
    nginx = (ROOT / "web" / "nginx.conf").read_text(encoding="utf-8")
    compose = yaml.safe_load(
        (ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    api = services["train-factory-api"]
    web = services["train-factory-web"]

    assert "location = /health" in nginx
    assert "proxy_pass http://${API_HOST}:${API_PORT}/health;" in nginx
    assert "location = /live" in nginx
    assert "return 200 \"healthy\\n\";" in nginx

    api_healthcheck = api["healthcheck"]
    assert "/health" in " ".join(str(item) for item in api_healthcheck["test"])
    assert web["depends_on"]["train-factory-api"]["condition"] == "service_healthy"
    assert "/health" in " ".join(str(item) for item in web["healthcheck"]["test"])


@pytest.mark.host_tools
@pytest.mark.parametrize(
    ("api_port", "health_status", "health_body"),
    [
        (8081, "200", '{"status":"healthy","version":"0.1.0"}'),
        (8082, "503", '{"status":"unhealthy","version":"0.1.0"}'),
        (8083, "502", None),
    ],
    ids=["api-healthy", "api-unhealthy", "api-unreachable"],
)
def test_rendered_nginx_health_tracks_api_while_live_stays_healthy(
    tmp_path, api_port, health_status, health_body
):
    image = os.environ.get("TRAINFACTORY_HEALTH_TEST_IMAGE")
    if not image:
        pytest.skip(
            "Set TRAINFACTORY_HEALTH_TEST_IMAGE to a locally available Nginx image"
        )
    inspected = _docker("image", "inspect", "--format", "{{.Id}}", image)
    assert inspected.returncode == 0, "The health test image must already exist locally"
    image_id = inspected.stdout.strip()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)

    upstream = tmp_path / "health-test-upstreams.conf"
    upstream.write_text(
        "server { listen 8081; location = /health { "
        "default_type application/json; return 200 "
        "'{\"status\":\"healthy\",\"version\":\"0.1.0\"}'; } }\n"
        "server { listen 8082; location = /health { "
        "default_type application/json; return 503 "
        "'{\"status\":\"unhealthy\",\"version\":\"0.1.0\"}'; } }\n",
        encoding="utf-8",
    )
    arguments = ["run", "-d", "--rm", "--network", "none"]
    for source, target in (
        (ROOT / "web" / "nginx.conf", "/etc/nginx/templates/default.conf.template"),
        (upstream, "/etc/nginx/conf.d/health-test-upstreams.conf"),
    ):
        arguments.extend(
            ["--mount", f"type=bind,source={source},target={target},readonly"]
        )
    arguments.extend(
        [
            "-e",
            "API_HOST=127.0.0.1",
            "-e",
            f"API_PORT={api_port}",
            "-e",
            "WEB_MAX_BODY_SIZE=1048576",
            image_id,
        ]
    )
    started = _docker(*arguments)
    assert started.returncode == 0, started.stderr
    container_id = started.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", container_id)
    try:
        for _ in range(40):
            live = _docker(
                "exec", container_id, "wget", "-q", "-O-", "http://127.0.0.1/live"
            )
            if live.returncode == 0:
                break
            time.sleep(0.25)
        else:
            pytest.fail(
                "Isolated Nginx did not become live: "
                + _docker("logs", container_id).stdout
            )

        health_headers, health_response_body = _request(container_id, "/health")
        assert re.search(rf"HTTP/1\.1 {health_status}\b", health_headers), health_headers
        if health_body is not None:
            assert health_response_body == health_body

        live_headers, live_body = _request(container_id, "/live")
        assert re.search(r"HTTP/1\.1 200\b", live_headers), live_headers
        assert live_body == "healthy\n"
    finally:
        # Only the exact disposable container created by this test is removed.
        removed = _docker("rm", "-f", container_id)
        assert removed.returncode == 0, removed.stderr
