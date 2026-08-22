import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from starlette.requests import Request

from train_factory.api import server
from train_factory import cli
from train_factory.storage.services.dataset_service import (
    DatasetDeletionInProgressError,
)


def test_run_server_rejects_multiple_api_workers(monkeypatch):
    uvicorn_run = Mock()
    monkeypatch.setattr("uvicorn.run", uvicorn_run)

    with pytest.raises(ValueError, match="exactly one API worker"):
        server.run_server(workers=2)

    uvicorn_run.assert_not_called()


def test_serve_cli_preserves_invalid_zero_worker_count(monkeypatch):
    run_server = Mock(side_effect=ValueError("exactly one API worker"))
    monkeypatch.setattr(server, "run_server", run_server)

    result = cli.cmd_serve(
        SimpleNamespace(host=None, port=None, workers=0)
    )

    assert result == 1
    run_server.assert_called_once_with(
        host=server.settings.api_host,
        port=server.settings.api_port,
        workers=0,
    )


def test_integrity_errors_are_mapped_to_sanitized_conflicts():
    app = server.create_app()

    @app.get("/test/integrity-conflict")
    async def integrity_conflict():
        raise IntegrityError(
            "INSERT INTO users (secret) VALUES ('private-value')",
            {},
            Exception("private database detail"),
        )

    response = TestClient(app, raise_server_exceptions=False).get(
        "/test/integrity-conflict"
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Resource conflicts with existing data"}
    assert "private" not in response.text
    assert "INSERT" not in response.text


def test_dataset_deletion_race_is_mapped_to_conflict():
    app = server.create_app()

    @app.get("/test/dataset-deletion-conflict")
    async def dataset_deletion_conflict():
        raise DatasetDeletionInProgressError(
            "Dataset deletion is in progress: private-dataset-id"
        )

    response = TestClient(app, raise_server_exceptions=False).get(
        "/test/dataset-deletion-conflict"
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Dataset is unavailable for task execution"
    }
    assert "private-dataset-id" not in response.text


@pytest.mark.parametrize("debug", (False, True))
def test_global_exception_log_omits_exception_message(
    monkeypatch,
    caplog,
    debug,
):
    marker = "authorization-secret-marker-must-not-be-logged"
    monkeypatch.setattr(server.settings, "debug", debug)
    app = server.create_app()
    handler = app.exception_handlers[Exception]
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/api/generation/tasks",
            "raw_path": b"/api/generation/tasks",
            "query_string": b"",
            "headers": [],
            "client": ("198.51.100.10", 43210),
            "server": ("testserver", 443),
        }
    )

    async def invoke_handler():
        try:
            raise RuntimeError(marker)
        except RuntimeError as exc:
            return await handler(request, exc)

    with caplog.at_level(logging.ERROR, logger=server.logger.name):
        response = asyncio.run(invoke_handler())

    assert marker not in caplog.text
    assert "RuntimeError" in caplog.text
    assert "POST /api/generation/tasks" in caplog.text
    assert response.status_code == 500


def test_health_check_queries_database_before_reporting_healthy(monkeypatch):
    executed_statements = []
    session = Mock()
    session.exec.side_effect = lambda statement: executed_statements.append(str(statement))
    session_context = Mock()
    session_context.__enter__ = Mock(return_value=session)
    session_context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(server, "get_session", lambda: session_context, raising=False)

    response = TestClient(server.create_app(), raise_server_exceptions=False).get(
        "/health"
    )

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "0.1.0"}
    assert executed_statements == ["SELECT 1"]


@pytest.mark.parametrize("failure_stage", ["connect", "query"])
def test_health_check_returns_sanitized_503_when_database_is_unavailable(
    monkeypatch,
    failure_stage,
):
    database_error = RuntimeError(
        "mysql://admin:secret@database:3306/train_factory"
    )
    session = Mock()
    session_context = Mock()
    session_context.__enter__ = Mock(return_value=session)
    session_context.__exit__ = Mock(return_value=False)
    if failure_stage == "connect":
        session_context.__enter__.side_effect = database_error
    else:
        session.exec.side_effect = database_error
    monkeypatch.setattr(server, "get_session", lambda: session_context)

    response = TestClient(server.create_app(), raise_server_exceptions=False).get(
        "/health"
    )

    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy", "version": "0.1.0"}
    assert "secret" not in response.text
    assert "mysql" not in response.text
