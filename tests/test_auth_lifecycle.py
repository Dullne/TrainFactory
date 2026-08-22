import ast
import asyncio
import inspect
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import bcrypt
from fastapi import FastAPI, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from train_factory.api.routes import auth_routes
from train_factory.auth import dependencies, jwt_handler
from train_factory.auth import schemas as auth_schemas
from train_factory.auth.dependencies import get_current_user
from train_factory.auth.jwt_handler import TokenError
from train_factory.auth import user_service as user_service_module
from train_factory.storage.entities.user_entity import UserDB


def _run(awaitable):
    return asyncio.run(awaitable)


def _assert_unauthorized(awaitable):
    with pytest.raises(HTTPException) as exc_info:
        _run(awaitable)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or expired token"
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


def _valid_payload(**overrides):
    payload = {"sub": "token-user-id", "username": "token-name", "ver": 4}
    payload.update(overrides)
    return payload


def _database_user(**overrides):
    user = {
        "user_id": "database-user-id",
        "username": "current-database-name",
        "email": "current@example.com",
        "is_admin": True,
        "is_active": True,
        "token_version": 4,
    }
    user.update(overrides)
    return user


def _install_resolution_fakes(monkeypatch, payload, user):
    looked_up = []

    def get_user(user_id):
        looked_up.append(user_id)
        return user

    monkeypatch.setattr(
        dependencies,
        "decode_token",
        lambda token: payload,
        raising=False,
    )
    monkeypatch.setattr(
        dependencies,
        "user_service",
        SimpleNamespace(get_user=get_user),
        raising=False,
    )
    return looked_up


def test_create_access_token_includes_integer_version_claim(monkeypatch):
    fixed_now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    encoded = {}

    def encode(payload, secret, algorithm):
        encoded.update(payload=payload, secret=secret, algorithm=algorithm)
        return "encoded-token"

    monkeypatch.setattr(
        jwt_handler,
        "get_settings",
        lambda: SimpleNamespace(
            jwt_access_token_expire_minutes=30,
            jwt_secret_key="test-secret",
            jwt_algorithm="HS256",
        ),
    )
    monkeypatch.setattr(jwt_handler, "now_aware", lambda: fixed_now)
    monkeypatch.setattr(jwt_handler.jwt, "encode", encode)

    token = jwt_handler.create_access_token("user-1", "alice", 7)

    assert token == "encoded-token"
    assert encoded == {
        "payload": {
            "sub": "user-1",
            "username": "alice",
            "ver": 7,
            "exp": fixed_now + timedelta(minutes=30),
            "iat": fixed_now,
        },
        "secret": "test-secret",
        "algorithm": "HS256",
    }


def test_create_access_token_requires_token_version_argument():
    parameter = inspect.signature(jwt_handler.create_access_token).parameters[
        "token_version"
    ]

    assert parameter.default is inspect.Parameter.empty


def test_all_production_token_creation_calls_pass_a_version_explicitly():
    source_root = Path(__file__).parents[1] / "train_factory"
    calls = []

    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id == "create_access_token":
                calls.append((source_path, node))

    assert len(calls) >= 2
    for source_path, call in calls:
        has_version_keyword = any(
            keyword.arg == "token_version" for keyword in call.keywords
        )
        assert len(call.args) >= 3 or has_version_keyword, source_path


def test_register_signs_token_with_created_users_version(monkeypatch):
    service = user_service_module.UserService()
    created_user = _database_user(
        user_id="registered-user",
        username="alice",
        token_version=6,
    )
    signed = []

    monkeypatch.setattr(service, "create_user", lambda **kwargs: created_user)

    def create_access_token(user_id, username, token_version):
        signed.append((user_id, username, token_version))
        return "registration-token"

    monkeypatch.setattr(
        user_service_module,
        "create_access_token",
        create_access_token,
    )

    result = service.register("alice", "StrongPassword-2026")

    assert signed == [("registered-user", "alice", 6)]
    assert result["access_token"] == "registration-token"
    assert result["user"] is created_user


def test_authenticate_signs_token_with_database_entity_version(monkeypatch):
    service = user_service_module.UserService()
    database_user = SimpleNamespace(
        user_id="authenticated-user",
        username="alice",
        hashed_password="stored-hash",
        is_active=True,
        token_version=9,
    )
    signed = []

    class Result:
        def first(self):
            return database_user

    class Session:
        def exec(self, statement):
            return Result()

    @contextmanager
    def get_session():
        yield Session()

    monkeypatch.setattr(user_service_module, "get_session", get_session)
    monkeypatch.setattr(user_service_module, "verify_password", lambda raw, hashed: True)

    def create_access_token(user_id, username, token_version):
        signed.append((user_id, username, token_version))
        return "login-token"

    monkeypatch.setattr(
        user_service_module,
        "create_access_token",
        create_access_token,
    )

    result = service.authenticate("alice", "StrongPassword-2026")

    assert signed == [("authenticated-user", "alice", 9)]
    assert result == {"access_token": "login-token", "token_type": "bearer"}


def test_authenticate_runs_password_verification_for_missing_user(monkeypatch):
    service = user_service_module.UserService()
    verification_calls = []

    class Result:
        def first(self):
            return None

    class Session:
        def exec(self, statement):
            return Result()

    @contextmanager
    def get_session():
        yield Session()

    def verify_password(raw_password, hashed_password):
        verification_calls.append((raw_password, hashed_password))
        return False

    monkeypatch.setattr(user_service_module, "get_session", get_session)
    monkeypatch.setattr(user_service_module, "verify_password", verify_password)

    with pytest.raises(ValueError, match="Invalid username or password"):
        service.authenticate("missing-user", "LongPassword-2026")

    assert len(verification_calls) == 1
    assert verification_calls[0][0] == "LongPassword-2026"
    assert verification_calls[0][1].startswith("$argon2id$")


def test_resolve_current_user_returns_only_current_database_identity(monkeypatch):
    looked_up = _install_resolution_fakes(
        monkeypatch,
        _valid_payload(username="stale-token-name"),
        _database_user(),
    )

    result = _run(dependencies.resolve_current_user("encoded-token"))

    assert looked_up == ["token-user-id"]
    assert result == {
        "user_id": "database-user-id",
        "username": "current-database-name",
        "is_admin": True,
        "is_active": True,
    }


def test_resolve_current_user_rejects_missing_database_user(monkeypatch):
    looked_up = _install_resolution_fakes(monkeypatch, _valid_payload(), None)

    _assert_unauthorized(dependencies.resolve_current_user("encoded-token"))

    assert looked_up == ["token-user-id"]


def test_resolve_current_user_rejects_disabled_user(monkeypatch):
    _install_resolution_fakes(
        monkeypatch,
        _valid_payload(),
        _database_user(is_active=False),
    )

    _assert_unauthorized(dependencies.resolve_current_user("encoded-token"))


def test_resolve_current_user_rejects_stale_token_version(monkeypatch):
    _install_resolution_fakes(
        monkeypatch,
        _valid_payload(ver=3),
        _database_user(token_version=4),
    )

    _assert_unauthorized(dependencies.resolve_current_user("encoded-token"))


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"username": "alice", "ver": 4}, id="missing-sub"),
        pytest.param({"sub": "user-1", "ver": 4}, id="missing-username"),
        pytest.param({"sub": 123, "username": "alice", "ver": 4}, id="integer-sub"),
        pytest.param({"sub": "", "username": "alice", "ver": 4}, id="empty-sub"),
        pytest.param({"sub": "user-1", "username": 123, "ver": 4}, id="integer-username"),
        pytest.param({"sub": "user-1", "username": "", "ver": 4}, id="empty-username"),
        pytest.param({"sub": "user-1", "username": "alice"}, id="missing-version"),
        pytest.param(_valid_payload(ver="4"), id="string-version"),
        pytest.param(_valid_payload(ver=True), id="boolean-version"),
    ],
)
def test_resolve_current_user_rejects_invalid_claims_without_database_lookup(
    monkeypatch,
    payload,
):
    looked_up = _install_resolution_fakes(monkeypatch, payload, _database_user())

    _assert_unauthorized(dependencies.resolve_current_user("encoded-token"))

    assert looked_up == []


@pytest.mark.parametrize(
    "database_version",
    [
        pytest.param("4", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="missing"),
    ],
)
def test_resolve_current_user_rejects_non_integer_database_version(
    monkeypatch,
    database_version,
):
    _install_resolution_fakes(
        monkeypatch,
        _valid_payload(),
        _database_user(token_version=database_version),
    )

    _assert_unauthorized(dependencies.resolve_current_user("encoded-token"))


def test_resolve_current_user_maps_decode_token_error_to_generic_401(monkeypatch):
    def decode_token(token):
        raise TokenError("internal decoder detail")

    monkeypatch.setattr(
        dependencies,
        "decode_token",
        decode_token,
        raising=False,
    )
    monkeypatch.setattr(
        dependencies,
        "user_service",
        SimpleNamespace(get_user=lambda user_id: pytest.fail("database was queried")),
        raising=False,
    )

    _assert_unauthorized(dependencies.resolve_current_user("encoded-token"))


def test_resolve_current_user_does_not_hide_database_failures(monkeypatch):
    def fail_database(user_id):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        dependencies,
        "decode_token",
        lambda token: _valid_payload(),
        raising=False,
    )
    monkeypatch.setattr(
        dependencies,
        "user_service",
        SimpleNamespace(get_user=fail_database),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        _run(dependencies.resolve_current_user("encoded-token"))


def test_required_and_optional_dependencies_delegate_cookie_then_bearer(monkeypatch):
    resolved_tokens = []

    async def resolve_current_user(token):
        resolved_tokens.append(token)
        return {
            "user_id": f"resolved:{token}",
            "username": "database-name",
            "is_admin": False,
            "is_active": True,
        }

    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
    )
    monkeypatch.setattr(
        dependencies,
        "resolve_current_user",
        resolve_current_user,
        raising=False,
    )
    bearer = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="bearer-token",
    )

    required = _run(
        dependencies.get_current_user(
            credentials=bearer,
            access_token="cookie-token",
        )
    )
    optional = _run(
        dependencies.get_current_user_optional(
            credentials=bearer,
            access_token=None,
        )
    )

    assert required["user_id"] == "resolved:cookie-token"
    assert optional["user_id"] == "resolved:bearer-token"
    assert resolved_tokens == ["cookie-token", "bearer-token"]


def test_auth_disabled_dependencies_return_legacy_identities_without_lookup(monkeypatch):
    async def fail_resolution(token):
        pytest.fail("token resolution must not run while auth is disabled")

    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=False),
    )
    monkeypatch.setattr(
        dependencies,
        "resolve_current_user",
        fail_resolution,
        raising=False,
    )
    monkeypatch.setattr(
        dependencies,
        "user_service",
        SimpleNamespace(get_user=lambda user_id: pytest.fail("database was queried")),
        raising=False,
    )

    required = _run(
        dependencies.get_current_user(
            credentials=None,
            access_token="ignored-cookie-token",
        )
    )
    optional = _run(
        dependencies.get_current_user_optional(
            credentials=None,
            access_token="ignored-cookie-token",
        )
    )

    assert required == {"user_id": None, "username": "anonymous"}
    # 统一匿名形态：get_current_user_optional 在 auth 关闭时也返回 user_id=None
    assert optional == {"user_id": None, "username": "anonymous"}


def test_enabled_dependencies_handle_missing_token_without_database_lookup(monkeypatch):
    async def fail_resolution(token):
        pytest.fail("resolution must not run without a token")

    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
    )
    monkeypatch.setattr(
        dependencies,
        "resolve_current_user",
        fail_resolution,
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        _run(dependencies.get_current_user(credentials=None, access_token=None))

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}
    assert _run(
        dependencies.get_current_user_optional(
            credentials=None,
            access_token=None,
        )
    ) is None


def test_require_admin_rejects_non_admin():
    current_user = {
        "user_id": "user-1",
        "username": "alice",
        "is_admin": False,
        "is_active": True,
    }

    with pytest.raises(HTTPException) as exc_info:
        _run(dependencies.require_admin(current_user))

    assert exc_info.value.status_code == 403
    assert "admin" in exc_info.value.detail.lower()


def test_require_admin_returns_admin_identity_unchanged():
    current_user = {
        "user_id": "admin-1",
        "username": "admin",
        "is_admin": True,
        "is_active": True,
    }

    assert _run(dependencies.require_admin(current_user)) is current_user


def _build_auth_client(current_user=None):
    app = FastAPI()
    app.state.limiter = auth_routes.limiter
    app.include_router(auth_routes.router, prefix="/api/auth")
    if current_user is not None:
        app.dependency_overrides[get_current_user] = lambda: current_user
    return TestClient(app, raise_server_exceptions=False)


def _public_user(**overrides):
    user = {
        "user_id": "user-1",
        "username": "alice",
        "email": "alice@example.com",
        "is_active": True,
        "is_admin": False,
        "token_version": 3,
        "hashed_password": "must-not-leak",
        "created_at": "2026-08-08T12:00:00",
        "updated_at": "2026-08-08T12:30:00",
    }
    user.update(overrides)
    return user


def test_register_schema_requires_ten_character_password_and_valid_email():
    with pytest.raises(ValidationError):
        auth_schemas.RegisterRequest(
            username="alice",
            password="123456789",
            email="alice@example.com",
        )

    with pytest.raises(ValidationError):
        auth_schemas.RegisterRequest(
            username="alice",
            password="StrongPass-2026",
            email="not-an-email",
        )

    request = auth_schemas.RegisterRequest(
        username="alice",
        password="1234567890",
        email="alice@example.com",
    )
    assert request.email == "alice@example.com"


def test_task4_schemas_expose_only_public_user_fields_and_validate_passwords():
    assert set(auth_schemas.AuthConfigResponse.model_fields) == {
        "self_registration_enabled",
        "direct_storage_registration_enabled",
    }
    assert set(auth_schemas.UserResponse.model_fields) == {
        "user_id",
        "username",
        "email",
        "is_active",
        "is_admin",
        "created_at",
        "updated_at",
    }

    for schema_name, payload in (
        (
            "ChangePasswordRequest",
            {"old_password": "old-password", "new_password": "123456789"},
        ),
        (
            "AdminUserCreateRequest",
            {"username": "alice", "password": "123456789"},
        ),
        ("AdminPasswordResetRequest", {"new_password": "123456789"}),
    ):
        schema = getattr(auth_schemas, schema_name)
        with pytest.raises(ValidationError):
            schema(**payload)

    with pytest.raises(ValidationError):
        auth_schemas.AccountFlagsUpdateRequest()


@pytest.mark.parametrize(
    "payload",
    (
        {"username": "u" * 65, "password": "ValidPassword-2026"},
        {"username": "alice", "password": "p" * 129},
    ),
)
def test_login_rejects_oversized_credentials_before_authentication(
    monkeypatch,
    payload,
):
    authenticate = Mock()
    monkeypatch.setattr(auth_routes.user_service, "authenticate", authenticate)

    response = _build_auth_client().post("/api/auth/login", json=payload)

    assert response.status_code == 422
    authenticate.assert_not_called()


def test_change_password_rejects_oversized_current_password_before_service(
    monkeypatch,
):
    change_password = Mock()
    monkeypatch.setattr(
        auth_routes.user_service,
        "change_password",
        change_password,
        raising=False,
    )

    response = _build_auth_client(
        {
            "user_id": "user-1",
            "username": "alice",
            "is_admin": False,
            "is_active": True,
        }
    ).post(
        "/api/auth/change-password",
        json={
            "old_password": "p" * 129,
            "new_password": "NewPassword-2026",
        },
    )

    assert response.status_code == 422
    change_password.assert_not_called()


@pytest.mark.parametrize(
    ("auth_enabled", "direct_storage_registration_enabled"),
    ((True, False), (False, True)),
)
def test_public_auth_config_exposes_direct_storage_registration_capability(
    monkeypatch,
    auth_enabled,
    direct_storage_registration_enabled,
):
    monkeypatch.setattr(
        auth_routes.settings,
        "self_registration_enabled",
        True,
    )
    monkeypatch.setattr(auth_routes.settings, "auth_enabled", auth_enabled)

    response = _build_auth_client().get("/api/auth/config")

    assert response.status_code == 200
    assert response.json() == {
        "self_registration_enabled": True,
        "direct_storage_registration_enabled": (
            direct_storage_registration_enabled
        ),
    }


def test_register_disabled_returns_403_without_calling_service(monkeypatch):
    register = Mock(
        return_value={"access_token": "token", "token_type": "bearer"}
    )
    monkeypatch.setattr(auth_routes.user_service, "register", register)
    monkeypatch.setattr(
        auth_routes.settings,
        "self_registration_enabled",
        False,
    )
    monkeypatch.setattr(
        auth_routes.settings,
        "jwt_access_token_expire_minutes",
        0,
    )

    response = _build_auth_client().post(
        "/api/auth/register",
        json={
            "username": "new-user",
            "password": "LongPassword-2026",
            "email": "new@example.com",
        },
    )

    assert response.status_code == 403
    register.assert_not_called()


def test_enabled_registration_sets_cookie_expiry_from_jwt_setting(monkeypatch):
    register = Mock(
        return_value={"access_token": "registration-token", "token_type": "bearer"}
    )
    monkeypatch.setattr(auth_routes.user_service, "register", register)
    monkeypatch.setattr(
        auth_routes.settings,
        "self_registration_enabled",
        True,
    )
    monkeypatch.setattr(
        auth_routes.settings,
        "jwt_access_token_expire_minutes",
        17,
    )
    monkeypatch.setattr(auth_routes.settings, "auth_cookie_secure", True)

    response = _build_auth_client().post(
        "/api/auth/register",
        json={
            "username": "new-user",
            "password": "LongPassword-2026",
            "email": "new@example.com",
        },
    )

    assert response.status_code == 200
    register.assert_called_once_with(
        username="new-user",
        password="LongPassword-2026",
        email="new@example.com",
    )
    cookie = response.headers["set-cookie"]
    assert "Max-Age=1020" in cookie
    assert "HttpOnly" in cookie
    assert "Path=/" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" in cookie


def test_clear_auth_cookie_matches_set_cookie_attributes(monkeypatch):
    monkeypatch.setattr(auth_routes.settings, "auth_cookie_secure", True)
    response = Response()

    auth_routes._clear_auth_cookie(response)

    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{auth_routes.COOKIE_NAME}=")
    assert "Max-Age=0" in cookie
    assert "HttpOnly" in cookie
    assert "Path=/" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" in cookie


def test_non_admin_cannot_list_users():
    response = _build_auth_client(
        {
            "user_id": "user-1",
            "username": "alice",
            "is_admin": False,
            "is_active": True,
        }
    ).get("/api/auth/admin/users")

    assert response.status_code == 403


def test_admin_list_users_uses_stable_envelope_and_filters_secrets(monkeypatch):
    list_users = Mock(return_value=([_public_user()], 1))
    monkeypatch.setattr(auth_routes.user_service, "list_users", list_users, raising=False)

    response = _build_auth_client(
        {
            "user_id": "admin-1",
            "username": "admin",
            "is_admin": True,
            "is_active": True,
        }
    ).get("/api/auth/admin/users?limit=20&offset=0")

    assert response.status_code == 200
    assert response.json() == {
        "users": [
            {
                "user_id": "user-1",
                "username": "alice",
                "email": "alice@example.com",
                "is_active": True,
                "is_admin": False,
                "created_at": "2026-08-08T12:00:00",
                "updated_at": "2026-08-08T12:30:00",
            }
        ],
        "total": 1,
        "limit": 20,
        "offset": 0,
    }
    list_users.assert_called_once_with(limit=20, offset=0)


def test_change_password_increments_version_and_hashes_new_password(monkeypatch):
    stored = SimpleNamespace(hashed_password="old-hash", token_version=3)
    service = user_service_module.UserService()
    monkeypatch.setattr(user_service_module, "hash_password", lambda value: f"hash:{value}")

    result = service._change_password_in_session(
        stored,
        old_password="OldPassword-2026",
        new_password="NewPassword-2026",
        password_matches=True,
    )

    assert result is True
    assert stored.hashed_password == "hash:NewPassword-2026"
    assert stored.token_version == 4


def test_successful_password_change_clears_cookie_and_requires_login(monkeypatch):
    change_password = Mock(return_value=True)
    monkeypatch.setattr(
        auth_routes.user_service,
        "change_password",
        change_password,
        raising=False,
    )
    monkeypatch.setattr(auth_routes.settings, "auth_cookie_secure", False)

    client = _build_auth_client(
        {
            "user_id": "user-1",
            "username": "alice",
            "is_admin": False,
            "is_active": True,
        }
    )
    client.cookies.set(auth_routes.COOKIE_NAME, "stale-token")
    response = client.post(
        "/api/auth/change-password",
        json={
            "old_password": "OldPassword-2026",
            "new_password": "NewPassword-2026",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Password changed successfully. Please log in again."
    }
    assert "Max-Age=0" in response.headers["set-cookie"]
    change_password.assert_called_once_with(
        user_id="user-1",
        old_password="OldPassword-2026",
        new_password="NewPassword-2026",
    )


@pytest.fixture()
def task4_user_service_db(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    UserDB.__table__.create(engine)

    @contextmanager
    def get_test_session():
        with Session(engine) as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(user_service_module, "get_session", get_test_session)
    yield user_service_module.UserService(), get_test_session
    engine.dispose()


def _admin_identity(user_id="admin-1"):
    return {
        "user_id": user_id,
        "username": "admin",
        "is_admin": True,
        "is_active": True,
    }


def test_all_new_password_schemas_require_ten_characters_and_emailstr():
    with pytest.raises(ValidationError):
        auth_schemas.ChangePasswordRequest(
            old_password="old-password",
            new_password="123456789",
        )
    with pytest.raises(ValidationError):
        auth_schemas.AdminUserCreateRequest(
            username="alice",
            password="123456789",
        )
    with pytest.raises(ValidationError):
        auth_schemas.AdminPasswordResetRequest(new_password="123456789")
    with pytest.raises(ValidationError):
        auth_schemas.AdminUserCreateRequest(
            username="alice",
            password="StrongPass-2026",
            email="invalid-email",
        )

    request = auth_schemas.AdminUserCreateRequest(
        username="alice",
        password="1234567890",
        email="alice@example.com",
    )
    assert request.is_admin is False
    assert request.email == "alice@example.com"


def test_cookie_expiry_rejects_nonpositive_or_noninteger_minutes(monkeypatch):
    for invalid_minutes in (0, -1, 1.5, True):
        monkeypatch.setattr(
            auth_routes.settings,
            "jwt_access_token_expire_minutes",
            invalid_minutes,
        )
        with pytest.raises(ValueError, match="positive integer"):
            auth_routes._set_auth_cookie(Response(), "token")


def test_logout_reuses_clear_cookie_helper(monkeypatch):
    response = Response()
    clear_cookie = Mock()
    revoke_sessions = Mock(return_value=True)
    monkeypatch.setattr(
        auth_routes,
        "_clear_auth_cookie",
        clear_cookie,
        raising=False,
    )
    monkeypatch.setattr(
        auth_routes.user_service,
        "revoke_sessions",
        revoke_sessions,
        raising=False,
    )

    result = _run(
        auth_routes.logout(
            response,
            current_user={"user_id": "user-1", "username": "alice"},
        )
    )

    assert result == {"message": "Logged out successfully"}
    revoke_sessions.assert_called_once_with("user-1")
    clear_cookie.assert_called_once_with(response)


def test_create_user_rejects_short_password_when_called_directly(
    task4_user_service_db,
):
    service, get_test_session = task4_user_service_db

    with pytest.raises(ValueError, match="at least 10"):
        service.create_user("alice", "123456789")

    with get_test_session() as session:
        assert session.exec(select(UserDB)).all() == []


def test_create_user_converts_integrity_error_without_sql_details(monkeypatch):
    class EmptyResult:
        def first(self):
            return None

    class ConflictingSession:
        def exec(self, _statement):
            return EmptyResult()

        def add(self, _user):
            pass

        def commit(self):
            raise IntegrityError(
                "INSERT INTO users ... secret sql",
                {"password": "must-not-leak"},
                Exception("duplicate key"),
            )

        def rollback(self):
            pass

    @contextmanager
    def get_conflicting_session():
        yield ConflictingSession()

    monkeypatch.setattr(user_service_module, "get_session", get_conflicting_session)
    monkeypatch.setattr(user_service_module, "hash_password", lambda _password: "hash")

    with pytest.raises(ValueError) as exc_info:
        user_service_module.UserService().create_user(
            "alice",
            "StrongAlice-2026",
            email="alice@example.com",
        )

    assert str(exc_info.value) == "Username or email already exists"
    assert "INSERT" not in str(exc_info.value)
    assert "password" not in str(exc_info.value)


def test_list_users_is_stably_sorted_and_paginated(task4_user_service_db):
    service, get_test_session = task4_user_service_db
    first = service.create_user("first", "StrongFirst-2026")
    second = service.create_user("second", "StrongSecond-2026")
    third = service.create_user("third", "StrongThird-2026")
    same_time = datetime(2026, 8, 8, 12, 0)
    with get_test_session() as session:
        users = session.exec(select(UserDB)).all()
        for user in users:
            user.created_at = same_time
            session.add(user)
        session.commit()

    users, total = service.list_users(limit=2, offset=1)

    assert total == 3
    assert [user["user_id"] for user in users] == [
        second["user_id"],
        first["user_id"],
    ]
    assert all("hashed_password" not in user for user in users)
    assert third["user_id"] not in [user["user_id"] for user in users]


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(0, 0), (101, 0), (20, -1), (True, 0)],
)
def test_list_users_validates_pagination(task4_user_service_db, limit, offset):
    service, _ = task4_user_service_db

    with pytest.raises(ValueError, match="pagination"):
        service.list_users(limit=limit, offset=offset)


def test_account_flags_reject_self_disable_and_last_admin_changes(
    task4_user_service_db,
):
    service, _ = task4_user_service_db
    admin = service.create_user(
        "admin",
        "StrongAdmin-2026",
        is_admin=True,
    )

    with pytest.raises(ValueError, match="self-disable"):
        service.set_account_flags(
            admin["user_id"],
            is_active=False,
            acting_user_id=admin["user_id"],
        )
    with pytest.raises(ValueError, match="last active administrator"):
        service.set_account_flags(
            admin["user_id"],
            is_active=False,
            acting_user_id="different-admin",
        )
    with pytest.raises(ValueError, match="last active administrator"):
        service.set_account_flags(
            admin["user_id"],
            is_admin=False,
            acting_user_id="different-admin",
        )

    unchanged = service.get_user(admin["user_id"])
    assert unchanged["is_active"] is True
    assert unchanged["is_admin"] is True


def test_other_active_admin_allows_self_demotion_but_not_self_disable(
    task4_user_service_db,
):
    service, _ = task4_user_service_db
    first = service.create_user("first-admin", "StrongFirst-2026", is_admin=True)
    service.create_user("second-admin", "StrongSecond-2026", is_admin=True)

    with pytest.raises(ValueError, match="self-disable"):
        service.set_account_flags(
            first["user_id"],
            is_active=False,
            acting_user_id=first["user_id"],
        )

    demoted = service.set_account_flags(
        first["user_id"],
        is_admin=False,
        acting_user_id=first["user_id"],
    )

    assert demoted["is_active"] is True
    assert demoted["is_admin"] is False
    assert demoted["token_version"] == first["token_version"]


def test_disabling_increments_version_once_and_reactivation_preserves_it(
    task4_user_service_db,
):
    service, _ = task4_user_service_db
    created = service.create_user("alice", "StrongAlice-2026")

    disabled = service.set_account_flags(
        created["user_id"],
        is_active=False,
        acting_user_id="admin-1",
    )
    disabled_again = service.set_account_flags(
        created["user_id"],
        is_active=False,
        acting_user_id="admin-1",
    )
    reactivated = service.set_account_flags(
        created["user_id"],
        is_active=True,
        acting_user_id="admin-1",
    )

    assert disabled["token_version"] == created["token_version"] + 1
    assert disabled_again["token_version"] == disabled["token_version"]
    assert reactivated["token_version"] == disabled["token_version"]


def test_bootstrap_flag_update_bypasses_administrator_invariants(
    task4_user_service_db,
):
    service, _ = task4_user_service_db
    admin = service.create_user("admin", "StrongAdmin-2026", is_admin=True)

    updated = service.set_account_flags(
        admin["user_id"],
        is_active=False,
        is_admin=False,
        acting_user_id=admin["user_id"],
        bootstrap=True,
    )

    assert updated["is_active"] is False
    assert updated["is_admin"] is False


def test_reset_password_validates_missing_and_increments_version(
    task4_user_service_db,
):
    service, _ = task4_user_service_db
    created = service.create_user("alice", "OldPassword-2026")

    with pytest.raises(ValueError, match="at least 10"):
        service.reset_password(created["user_id"], "123456789")
    with pytest.raises(ValueError, match="User not found"):
        service.reset_password("missing-user", "NewPassword-2026")

    updated = service.reset_password(
        created["user_id"],
        "NewPassword-2026",
    )

    assert updated["token_version"] == created["token_version"] + 1
    assert service.verify_user_password(created["user_id"], "NewPassword-2026")
    assert not service.verify_user_password(created["user_id"], "OldPassword-2026")


def test_revoke_sessions_increments_token_version(task4_user_service_db):
    service, _ = task4_user_service_db
    created = service.create_user("logout-user", "LogoutPassword-2026")

    assert service.revoke_sessions(created["user_id"]) is True

    revoked = service.get_user(created["user_id"])
    assert revoked["token_version"] == created["token_version"] + 1

    with pytest.raises(ValueError, match="User not found"):
        service.revoke_sessions("missing-user")


def test_change_password_wrong_old_password_does_not_mutate_then_succeeds(
    task4_user_service_db,
):
    service, _ = task4_user_service_db
    created = service.create_user("alice", "OldPassword-2026")

    with pytest.raises(ValueError, match="Current password is incorrect"):
        service.change_password(
            created["user_id"],
            "wrong-password",
            "NewPassword-2026",
        )
    unchanged = service.get_user(created["user_id"])
    assert unchanged["token_version"] == created["token_version"]
    assert service.verify_user_password(created["user_id"], "OldPassword-2026")

    assert service.change_password(
        created["user_id"],
        "OldPassword-2026",
        "NewPassword-2026",
    ) is True
    changed = service.get_user(created["user_id"])
    assert changed["token_version"] == created["token_version"] + 1
    assert service.verify_user_password(created["user_id"], "NewPassword-2026")


def test_admin_create_user_returns_public_record(monkeypatch):
    create_user = Mock(return_value=_public_user(is_admin=True))
    monkeypatch.setattr(auth_routes.user_service, "create_user", create_user)

    response = _build_auth_client(_admin_identity()).post(
        "/api/auth/admin/users",
        json={
            "username": "alice",
            "password": "StrongAlice-2026",
            "email": "alice@example.com",
            "is_admin": True,
        },
    )

    assert response.status_code == 201
    assert response.json()["is_admin"] is True
    assert "token_version" not in response.json()
    assert "hashed_password" not in response.json()
    create_user.assert_called_once_with(
        username="alice",
        password="StrongAlice-2026",
        email="alice@example.com",
        is_admin=True,
    )


@pytest.mark.parametrize(
    "service_error",
    [
        "Username 'alice' already exists",
        "Email 'alice@example.com' already registered",
        "(pymysql.err.IntegrityError) secret SQL",
    ],
)
def test_admin_create_duplicate_maps_to_sanitized_409(monkeypatch, service_error):
    monkeypatch.setattr(
        auth_routes.user_service,
        "create_user",
        Mock(side_effect=ValueError(service_error)),
    )

    response = _build_auth_client(_admin_identity()).post(
        "/api/auth/admin/users",
        json={"username": "alice", "password": "StrongAlice-2026"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Username or email already exists"}
    assert "SQL" not in response.text


@pytest.mark.parametrize(
    "service_error",
    [
        "self-disable is not allowed",
        "last active administrator cannot be disabled",
        "last active administrator cannot be demoted",
    ],
)
def test_admin_patch_conflicts_map_to_sanitized_409(monkeypatch, service_error):
    set_flags = Mock(side_effect=ValueError(service_error))
    monkeypatch.setattr(auth_routes.user_service, "set_account_flags", set_flags)

    response = _build_auth_client(_admin_identity()).patch(
        "/api/auth/admin/users/admin-1",
        json={"is_active": False},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Account update conflicts with safety rules"}


def test_admin_patch_passes_actor_and_rejects_empty_body(monkeypatch):
    set_flags = Mock(return_value=_public_user(is_active=False))
    monkeypatch.setattr(auth_routes.user_service, "set_account_flags", set_flags)
    client = _build_auth_client(_admin_identity("acting-admin"))

    response = client.patch(
        "/api/auth/admin/users/user-1",
        json={"is_active": False},
    )

    assert response.status_code == 200
    assert response.json()["is_active"] is False
    set_flags.assert_called_once_with(
        "user-1",
        is_active=False,
        is_admin=None,
        acting_user_id="acting-admin",
    )
    assert client.patch("/api/auth/admin/users/user-1", json={}).status_code == 422


def test_admin_account_routes_map_missing_user_to_404(monkeypatch):
    client = _build_auth_client(_admin_identity())
    monkeypatch.setattr(
        auth_routes.user_service,
        "set_account_flags",
        Mock(side_effect=ValueError("User not found")),
    )
    patch_response = client.patch(
        "/api/auth/admin/users/missing-user",
        json={"is_admin": False},
    )

    monkeypatch.setattr(
        auth_routes.user_service,
        "reset_password",
        Mock(side_effect=ValueError("User not found")),
        raising=False,
    )
    reset_response = client.post(
        "/api/auth/admin/users/missing-user/reset-password",
        json={"new_password": "NewPassword-2026"},
    )

    assert patch_response.status_code == 404
    assert reset_response.status_code == 404
    assert patch_response.json() == {"detail": "User not found"}
    assert reset_response.json() == {"detail": "User not found"}


def test_admin_reset_password_returns_stable_message(monkeypatch):
    reset_password = Mock(return_value=_public_user(token_version=4))
    monkeypatch.setattr(
        auth_routes.user_service,
        "reset_password",
        reset_password,
        raising=False,
    )

    response = _build_auth_client(_admin_identity()).post(
        "/api/auth/admin/users/user-1/reset-password",
        json={"new_password": "NewPassword-2026"},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Password reset successfully"}
    reset_password.assert_called_once_with("user-1", "NewPassword-2026")


def test_admin_user_routes_require_admin_and_expose_no_delete():
    protected_routes = [
        route
        for route in auth_routes.router.routes
        if route.path.startswith("/admin/users")
    ]

    assert protected_routes
    assert all(
        any(dependency.call is dependencies.require_admin for dependency in route.dependant.dependencies)
        for route in protected_routes
    )

    response = _build_auth_client(_admin_identity()).delete(
        "/api/auth/admin/users/user-1"
    )
    assert response.status_code == 405


def test_register_validates_cookie_configuration_before_service(
    monkeypatch,
):
    register = Mock(
        return_value={"access_token": "registration-token", "token_type": "bearer"}
    )
    monkeypatch.setattr(auth_routes.user_service, "register", register)
    monkeypatch.setattr(auth_routes.settings, "self_registration_enabled", True)
    monkeypatch.setattr(
        auth_routes.settings,
        "jwt_access_token_expire_minutes",
        0,
    )

    endpoint = inspect.unwrap(auth_routes.register)
    with pytest.raises(ValueError, match="positive integer"):
        _run(
            endpoint(
                request=SimpleNamespace(),
                response=Response(),
                body=auth_schemas.RegisterRequest(
                    username="new-user",
                    password="LongPassword-2026",
                    email="new@example.com",
                ),
            )
        )

    register.assert_not_called()


def test_login_validates_cookie_configuration_before_service(
    monkeypatch,
):
    authenticate = Mock(
        return_value={"access_token": "login-token", "token_type": "bearer"}
    )
    monkeypatch.setattr(auth_routes.user_service, "authenticate", authenticate)
    monkeypatch.setattr(
        auth_routes.settings,
        "jwt_access_token_expire_minutes",
        0,
    )

    endpoint = inspect.unwrap(auth_routes.login)
    with pytest.raises(ValueError, match="positive integer"):
        _run(
            endpoint(
                request=SimpleNamespace(),
                response=Response(),
                body=auth_schemas.LoginRequest(
                    username="alice",
                    password="LongPassword-2026",
                ),
            )
        )

    authenticate.assert_not_called()


def test_authenticate_upgrades_legacy_bcrypt_without_changing_token_version(
    task4_user_service_db,
    monkeypatch,
):
    service, get_test_session = task4_user_service_db
    password = "LegacyPassword-2026"
    created = service.create_user("legacy-user", "TemporaryPassword-2026")
    with get_test_session() as session:
        stored = session.exec(
            select(UserDB).where(UserDB.user_id == created["user_id"])
        ).one()
        stored.hashed_password = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")
        stored.token_version = 7
        session.add(stored)
        session.commit()

    signed_versions = []
    monkeypatch.setattr(
        user_service_module,
        "create_access_token",
        lambda user_id, username, token_version: signed_versions.append(
            token_version
        )
        or "legacy-login-token",
    )

    result = service.authenticate("legacy-user", password)

    assert result["access_token"] == "legacy-login-token"
    assert signed_versions == [7]
    with get_test_session() as session:
        upgraded = session.exec(
            select(UserDB).where(UserDB.user_id == created["user_id"])
        ).one()
        assert upgraded.hashed_password.startswith("$argon2id$")
        assert upgraded.token_version == 7


def test_verify_user_password_upgrades_outdated_argon_without_version_change(
    task4_user_service_db,
):
    from argon2 import PasswordHasher

    service, get_test_session = task4_user_service_db
    password = "ArgonPassword-2026"
    created = service.create_user("argon-user", "TemporaryPassword-2026")
    outdated_hash = PasswordHasher(
        time_cost=1,
        memory_cost=8192,
        parallelism=1,
    ).hash(password)
    with get_test_session() as session:
        stored = session.exec(
            select(UserDB).where(UserDB.user_id == created["user_id"])
        ).one()
        stored.hashed_password = outdated_hash
        stored.token_version = 11
        session.add(stored)
        session.commit()

    assert service.verify_user_password(created["user_id"], password) is True

    with get_test_session() as session:
        upgraded = session.exec(
            select(UserDB).where(UserDB.user_id == created["user_id"])
        ).one()
        assert upgraded.hashed_password.startswith("$argon2id$")
        assert upgraded.hashed_password != outdated_hash
        assert upgraded.token_version == 11


def test_invalid_registration_ttl_leaves_users_table_empty(
    task4_user_service_db,
    monkeypatch,
):
    service, get_test_session = task4_user_service_db
    monkeypatch.setattr(auth_routes, "user_service", service)
    monkeypatch.setattr(auth_routes.settings, "self_registration_enabled", True)
    monkeypatch.setattr(
        auth_routes.settings,
        "jwt_access_token_expire_minutes",
        0,
    )

    endpoint = inspect.unwrap(auth_routes.register)
    with pytest.raises(ValueError, match="positive integer"):
        _run(
            endpoint(
                request=SimpleNamespace(),
                response=Response(),
                body=auth_schemas.RegisterRequest(
                    username="new-user",
                    password="LongPassword-2026",
                    email="new@example.com",
                ),
            )
        )

    with get_test_session() as session:
        assert session.exec(select(UserDB)).all() == []


def test_account_flag_removal_locks_active_admins_before_target_in_id_order(
    monkeypatch,
):
    target = UserDB(
        id=1,
        user_id="admin-1",
        username="first-admin",
        hashed_password="stored-hash",
        is_active=True,
        is_admin=True,
    )
    other_admin = UserDB(
        id=2,
        user_id="admin-2",
        username="second-admin",
        hashed_password="stored-hash",
        is_active=True,
        is_admin=True,
    )
    executed_statements = []

    class Result:
        def first(self):
            return target

        def all(self):
            return [target, other_admin]

    class RecordingSession:
        def exec(self, statement):
            executed_statements.append(statement)
            return Result()

        def add(self, _user):
            pass

        def commit(self):
            pass

        def refresh(self, _user):
            pass

    @contextmanager
    def get_recording_session():
        yield RecordingSession()

    monkeypatch.setattr(
        user_service_module,
        "get_session",
        get_recording_session,
    )

    updated = user_service_module.UserService().set_account_flags(
        target.user_id,
        is_admin=False,
        acting_user_id=target.user_id,
    )

    assert updated["is_admin"] is False
    assert len(executed_statements) == 1
    first_statement = executed_statements[0]
    first_sql = str(first_statement).lower()
    assert "is_active is true" in first_sql
    assert "is_admin is true" in first_sql
    assert "order by users.id" in first_sql
    assert first_statement._for_update_arg is not None
