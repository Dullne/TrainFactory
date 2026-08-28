import asyncio
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy.pool import StaticPool
from starlette.requests import Request
from sqlmodel import Session, create_engine, select

from train_factory.api import server
from train_factory.api.routes import auth_routes
from train_factory.auth import dependencies
from train_factory.core.time_utils import now_naive
from train_factory.storage.entities.audit_log_entity import AuditLogDB
from train_factory.storage.services.audit_log_service import audit_log_service


ROOT_DIR = Path(__file__).parents[1]


def _request(*, client_host="198.51.100.10", headers=()):
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/auth/login",
            "raw_path": b"/api/auth/login",
            "query_string": b"",
            "headers": [
                (name.lower().encode("ascii"), value.encode("ascii"))
                for name, value in headers
            ],
            "client": (client_host, 43210),
            "server": ("testserver", 80),
        }
    )


def test_audit_identity_prefers_valid_cookie_over_junk_authorization(monkeypatch):
    resolved_tokens = []

    async def resolve_current_user(token):
        resolved_tokens.append(token)
        return {
            "user_id": "database-user-id",
            "username": "current-name",
            "is_admin": False,
            "is_active": True,
        }

    monkeypatch.setattr(dependencies, "resolve_current_user", resolve_current_user)
    monkeypatch.setattr(server.settings, "auth_enabled", True)
    request = _request(
        headers=(
            ("Cookie", "access_token=valid-cookie"),
            ("Authorization", "Bearer junk-header"),
        )
    )

    identity = asyncio.run(server._resolve_audit_identity(request))

    assert identity == {
        "user_id": "database-user-id",
        "username": "current-name",
    }
    assert resolved_tokens == ["valid-cookie"]


@pytest.mark.parametrize(
    "database_user",
    [
        pytest.param(
            {
                "user_id": "user-1",
                "username": "alice",
                "is_admin": False,
                "is_active": False,
                "token_version": 7,
            },
            id="disabled",
        ),
        pytest.param(
            {
                "user_id": "user-1",
                "username": "alice",
                "is_admin": False,
                "is_active": True,
                "token_version": 8,
            },
            id="revoked-version",
        ),
    ],
)
def test_audit_identity_does_not_trust_disabled_or_revoked_token(
    monkeypatch,
    database_user,
):
    monkeypatch.setattr(server.settings, "auth_enabled", True)
    monkeypatch.setattr(
        dependencies,
        "decode_token",
        lambda token: {"sub": "user-1", "username": "alice", "ver": 7},
    )
    monkeypatch.setattr(
        dependencies,
        "user_service",
        SimpleNamespace(get_user=lambda user_id: database_user),
    )
    request = _request(headers=(("Authorization", "Bearer signed-token"),))

    identity = asyncio.run(server._resolve_audit_identity(request))

    assert identity == {"user_id": "anonymous", "username": None}


def test_invalid_audit_credentials_never_block_request_processing(monkeypatch):
    async def reject_token(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    monkeypatch.setattr(dependencies, "resolve_current_user", reject_token)
    monkeypatch.setattr(server.settings, "auth_enabled", True)
    request = _request(headers=(("Authorization", "Bearer invalid"),))

    assert asyncio.run(server._resolve_audit_identity(request)) == {
        "user_id": "anonymous",
        "username": None,
    }


def test_audit_uses_identity_valid_at_request_entry(monkeypatch):
    token_is_active = True
    recorded_logs = []

    async def resolve_current_user(token):
        if not token_is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return {
            "user_id": "user-1",
            "username": "alice",
            "is_admin": False,
            "is_active": True,
        }

    async def revoke_current_token():
        nonlocal token_is_active
        token_is_active = False
        return {"message": "revoked"}

    monkeypatch.setattr(dependencies, "resolve_current_user", resolve_current_user)
    monkeypatch.setattr(server.settings, "auth_enabled", True)
    monkeypatch.setattr(
        audit_log_service,
        "log",
        lambda **fields: recorded_logs.append(fields),
    )
    app = server.create_app()
    app.post("/api/test/revoke")(revoke_current_token)

    client = TestClient(app, raise_server_exceptions=False)
    client.cookies.set("access_token", "valid-at-entry")
    response = client.post("/api/test/revoke")

    assert response.status_code == 200
    assert recorded_logs[-1]["user_id"] == "user-1"
    assert recorded_logs[-1]["username"] == "alice"


@pytest.mark.parametrize(
    ("endpoint", "service_method", "extra_settings"),
    [
        pytest.param("login", "authenticate", {}, id="login"),
        pytest.param(
            "register",
            "register",
            {"self_registration_enabled": True},
            id="register",
        ),
    ],
)
def test_successful_authentication_audit_uses_new_identity(
    monkeypatch,
    endpoint,
    service_method,
    extra_settings,
):
    recorded_logs = []
    monkeypatch.setattr(server.settings, "auth_enabled", True)
    monkeypatch.setattr(auth_routes.settings, "rate_limit_enabled", False)
    for name, value in extra_settings.items():
        monkeypatch.setattr(auth_routes.settings, name, value)
    monkeypatch.setattr(
        auth_routes,
        "decode_token",
        lambda token: {
            "sub": "new-session-user",
            "username": "alice",
        },
        raising=False,
    )
    monkeypatch.setattr(
        auth_routes.user_service,
        service_method,
        lambda **kwargs: {
            "access_token": "new-session-token",
            "token_type": "bearer",
        },
    )
    monkeypatch.setattr(
        audit_log_service,
        "log",
        lambda **fields: recorded_logs.append(fields),
    )
    auth_routes.limiter.reset()
    app = server.create_app()

    payload = {"username": "alice", "password": "LongPassword-2026"}
    response = TestClient(app, raise_server_exceptions=False).post(
        f"/api/auth/{endpoint}",
        json=payload,
    )

    assert response.status_code == 200
    assert recorded_logs[-1]["user_id"] == "new-session-user"
    assert recorded_logs[-1]["username"] == "alice"


@pytest.mark.parametrize(
    ("client_host", "forwarded_for", "expected_ip"),
    [
        pytest.param(
            "172.18.0.4",
            "203.0.113.25",
            "203.0.113.25",
            id="trusted-proxy",
        ),
        pytest.param(
            "198.51.100.10",
            "203.0.113.25",
            "198.51.100.10",
            id="direct-client-spoof",
        ),
    ],
)
def test_audit_uses_spoof_resistant_client_ip(
    monkeypatch,
    client_host,
    forwarded_for,
    expected_ip,
):
    recorded_logs = []
    monkeypatch.setattr(server.settings, "auth_enabled", False)
    monkeypatch.setattr(
        auth_routes.settings,
        "rate_limit_trusted_proxies",
        "172.18.0.4/32",
    )
    monkeypatch.setattr(
        audit_log_service,
        "log",
        lambda **fields: recorded_logs.append(fields),
    )
    app = server.create_app()

    @app.post("/api/test/audit-ip")
    async def audited_write():
        return {"ok": True}

    async def make_request():
        transport = httpx.ASGITransport(
            app=app,
            client=(client_host, 43210),
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/test/audit-ip",
                headers={"X-Forwarded-For": forwarded_for},
            )

    response = asyncio.run(make_request())

    assert response.status_code == 200
    assert recorded_logs[-1]["ip_address"] == expected_ip


def test_rate_limit_key_uses_first_valid_xff_ip_from_trusted_proxy(monkeypatch):
    monkeypatch.setattr(
        auth_routes.settings,
        "rate_limit_trusted_proxies",
        "172.18.0.4/32",
    )
    request = _request(
        client_host="172.18.0.4",
        headers=(("X-Forwarded-For", "unknown, 203.0.113.25, 172.18.0.4"),),
    )

    assert auth_routes.get_rate_limit_key(request) == "203.0.113.25"


def test_rate_limit_key_ignores_spoofed_xff_from_direct_client(monkeypatch):
    monkeypatch.setattr(
        auth_routes.settings,
        "rate_limit_trusted_proxies",
        "172.18.0.4/32",
    )
    request = _request(
        client_host="198.51.100.10",
        headers=(("X-Forwarded-For", "203.0.113.25"),),
    )

    assert auth_routes.get_rate_limit_key(request) == "198.51.100.10"


def test_disabling_rate_limit_exempts_auth_route_decorators(monkeypatch):
    authenticate_calls = []

    def reject_login(username, password):
        authenticate_calls.append(username)
        raise ValueError("invalid")

    auth_routes.limiter.reset()
    monkeypatch.setattr(auth_routes.settings, "rate_limit_enabled", False)
    monkeypatch.setattr(auth_routes.user_service, "authenticate", reject_login)

    app = FastAPI()
    app.state.limiter = auth_routes.limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(auth_routes.router, prefix="/api/auth")

    with TestClient(app, raise_server_exceptions=False) as client:
        responses = [
            client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "LongPassword-2026"},
            )
            for _ in range(7)
        ]

    assert [response.status_code for response in responses] == [401] * 7
    assert authenticate_calls == ["alice"] * 7


def test_disabled_registration_always_returns_403_without_rate_limiting(monkeypatch):
    auth_routes.limiter.reset()
    monkeypatch.setattr(auth_routes.limiter, "enabled", True)
    monkeypatch.setattr(auth_routes.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(
        auth_routes.settings,
        "self_registration_enabled",
        False,
    )

    app = FastAPI()
    app.state.limiter = auth_routes.limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(auth_routes.router, prefix="/api/auth")

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            responses = [
                client.post(
                    "/api/auth/register",
                    json={
                        "username": "alice",
                        "password": "LongPassword-2026",
                    },
                )
                for _ in range(101)
            ]
    finally:
        auth_routes.limiter.reset()

    assert [response.status_code for response in responses] == [403] * 101


def test_proxy_rate_limit_configuration_is_explicit_and_spoof_resistant():
    compose = (ROOT_DIR / "docker" / "docker-compose.yml").read_text("utf-8")
    env_example = (ROOT_DIR / ".env.example").read_text("utf-8")
    nginx = (ROOT_DIR / "web" / "nginx.conf").read_text("utf-8")
    vite = (ROOT_DIR / "web" / "vite.config.ts").read_text("utf-8")

    assert "RATE_LIMIT_TRUSTED_PROXIES" in compose
    assert "RATE_LIMIT_DEV_TRUSTED_PROXIES" in compose
    assert "RATE_LIMIT_TRUSTED_PROXIES=" in env_example
    assert "RATE_LIMIT_DEV_TRUSTED_PROXIES=" in env_example
    assert "0.0.0.0/0" not in compose
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in nginx
    assert "$proxy_add_x_forwarded_for" not in nginx
    assert "proxyReq.setHeader('x-forwarded-for'" in vite
    assert "ip_range: ${DOCKER_DYNAMIC_IP_RANGE:-172.18.128.0/17}" in compose
    assert "aux_addresses:" not in compose


def test_default_rate_limit_is_attached_and_forwarded_to_containers():
    compose_text = (ROOT_DIR / "docker" / "docker-compose.yml").read_text("utf-8")
    compose = yaml.safe_load(compose_text)
    env_example = (ROOT_DIR / ".env.example").read_text("utf-8")

    assert auth_routes.limiter._default_limits
    assert "RATE_LIMIT_DEFAULT" in compose_text
    assert "RATE_LIMIT_DEFAULT=" in env_example
    for service_name in ("train-factory-api", "train-factory-api-dev"):
        environment = compose["services"][service_name]["environment"]
        assert environment["RATE_LIMIT_LOGIN"] == "${RATE_LIMIT_LOGIN:-5/minute}"
        assert environment["RATE_LIMIT_REGISTER"] == "${RATE_LIMIT_REGISTER:-3/hour}"
    assert "RATE_LIMIT_LOGIN=" in env_example
    assert "RATE_LIMIT_REGISTER=" in env_example

    app = server.create_app()
    assert any(
        middleware.cls is SlowAPIMiddleware
        for middleware in app.user_middleware
    )


def test_default_rate_limit_is_enforced_for_undecorated_routes(monkeypatch):
    @contextmanager
    def healthy_session():
        yield SimpleNamespace(exec=lambda statement: None)

    monkeypatch.setattr(server, "get_session", healthy_session)
    auth_routes.limiter.reset()
    try:
        client = TestClient(server.create_app(), raise_server_exceptions=False)
        responses = [client.get("/health") for _ in range(101)]
    finally:
        auth_routes.limiter.reset()

    assert [response.status_code for response in responses[:100]] == [200] * 100
    assert responses[100].status_code == 429


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/api/auth/config", False),
        ("GET", "/api/auth/me", False),
        ("GET", "/api/auth/users", True),
        ("POST", "/api/auth/login", True),
        ("GET", "/api/train", False),
    ],
)
def test_audit_filter_skips_high_frequency_public_auth_reads(method, path, expected):
    assert server._should_audit_request(method, path) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/deep-evaluation/tasks/deep-1", "deep_evaluation_task"),
        ("/api/generation/tasks/generation-1", "generation_task"),
        ("/api/evaluations/tasks/evaluation-1", "evaluation_task"),
        ("/api/sync/api-configs/config-1", "external_api_config"),
        ("/api/sync/tasks/sync-1", "sync_task"),
        ("/api/sync/configs", "sync_task"),
        ("/api/train/training-1", "training_task"),
        ("/api/models/model-1", "model"),
        ("/api/deployments/deployment-1", "deployment"),
        ("/api/datasets/dataset-1", "dataset"),
        ("/api/configs/config-1", "model_config"),
        ("/api/auth/users/user-1", "user"),
        ("/api/unmapped/resource", "unknown"),
    ],
)
def test_audit_resource_mapping_uses_specific_api_prefixes(path, expected):
    assert server._get_resource_type_from_path(path) == expected


def test_startup_audit_cleanup_uses_configured_retention(monkeypatch):
    cleanup_calls = []
    monkeypatch.setattr(server.settings, "audit_log_retention_days", 37, raising=False)
    monkeypatch.setattr(
        audit_log_service,
        "cleanup_old_logs",
        lambda retention_days: cleanup_calls.append(retention_days) or 4,
    )

    assert server.cleanup_old_audit_logs() == 4
    assert cleanup_calls == [37]


def test_audit_retention_bulk_deletes_in_bounded_batches(monkeypatch):
    import importlib

    audit_service_module = importlib.import_module(
        "train_factory.storage.services.audit_log_service"
    )

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    AuditLogDB.__table__.create(engine)
    cutoff_base = now_naive()
    with Session(engine) as session:
        for index in range(5):
            session.add(
                AuditLogDB(
                    user_id="user-1",
                    action="read",
                    resource_type="test",
                    method="GET",
                    endpoint=f"/old/{index}",
                    status_code=200,
                    created_at=cutoff_base - timedelta(days=120),
                )
            )
        for index in range(2):
            session.add(
                AuditLogDB(
                    user_id="user-1",
                    action="read",
                    resource_type="test",
                    method="GET",
                    endpoint=f"/new/{index}",
                    status_code=200,
                    created_at=cutoff_base,
                )
            )
        session.commit()

    @contextmanager
    def isolated_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(audit_service_module, "get_session", isolated_session)

    assert audit_log_service.cleanup_old_logs(90, batch_size=2) == 5
    with Session(engine) as session:
        remaining = session.exec(select(AuditLogDB)).all()
    assert [entry.endpoint for entry in remaining] == ["/new/0", "/new/1"]


def test_lifespan_cancels_audit_cleanup_when_sync_start_fails(monkeypatch):
    from train_factory.api.routes import sync_routes
    from train_factory.sync.sync_manager import sync_manager

    for name in (
        "init_db",
        "cleanup_old_audit_logs",
        "cleanup_orphan_tasks",
        "cleanup_orphan_sync_trainings",
        "cleanup_orphan_evaluation_tasks",
        "cleanup_orphan_deep_evaluation_tasks",
        "cleanup_orphan_generation_tasks",
        "cleanup_orphan_datasets",
        "cleanup_orphan_containers",
        "cleanup_orphan_sync_tasks",
    ):
        monkeypatch.setattr(server, name, lambda: None)
    monkeypatch.setattr(server, "bootstrap_default_admin", lambda settings: None)
    monkeypatch.setattr(server.settings, "storage_backend", "local")
    monkeypatch.setattr(
        server,
        "_resume_deferred_deployment_starts",
        lambda: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        server,
        "_drain_deferred_deployment_starts",
        lambda: asyncio.sleep(0),
    )

    async def cleanup_loop():
        await asyncio.Event().wait()

    async def fail_start():
        raise RuntimeError("sync startup failed")

    created_tasks = []
    original_create_task = asyncio.create_task

    def capture_task(coroutine):
        task = original_create_task(coroutine)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(server, "_audit_log_cleanup_loop", cleanup_loop)
    monkeypatch.setattr(
        sync_routes,
        "resume_pending_sync_deletions",
        lambda: asyncio.sleep(0, result=(0, 0)),
    )
    monkeypatch.setattr(sync_manager, "start", fail_start)
    monkeypatch.setattr(server.asyncio, "create_task", capture_task)

    async def exercise():
        with pytest.raises(RuntimeError, match="sync startup failed"):
            async with server.lifespan(FastAPI()):
                pass
        await asyncio.sleep(0)
        assert created_tasks
        assert created_tasks[0].done()

    asyncio.run(exercise())
