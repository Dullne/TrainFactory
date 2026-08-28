import asyncio
import importlib
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from train_factory.storage.entities.user_entity import UserDB


def _settings(password, *, auth_enabled=True):
    return SimpleNamespace(
        auth_enabled=auth_enabled,
        default_admin_username="admin",
        default_admin_password=password,
        default_admin_email="admin@example.com",
    )


def _bootstrap_default_admin():
    module = importlib.import_module("train_factory.auth.bootstrap")
    return module.bootstrap_default_admin


def test_bootstrap_disabled_skips_user_service_entirely():
    service = Mock()

    result = _bootstrap_default_admin()(
        _settings(None, auth_enabled=False),
        service,
    )

    assert result == "disabled"
    assert service.mock_calls == []


def test_bootstrap_creates_first_admin():
    service = Mock()
    service.has_active_admin.return_value = False
    service.get_user_by_username.return_value = None

    result = _bootstrap_default_admin()(_settings("StrongAdmin-2026"), service)

    assert result == "created"
    assert service.mock_calls == [
        call.has_active_admin(),
        call.get_user_by_username("admin"),
        call.create_user(
            username="admin",
            password="StrongAdmin-2026",
            email="admin@example.com",
            is_admin=True,
        ),
    ]


@pytest.mark.parametrize("configured_password", [None, "", "DifferentAdmin-2026"])
def test_bootstrap_existing_active_admin_is_idempotent(configured_password):
    service = Mock()
    service.has_active_admin.return_value = True

    result = _bootstrap_default_admin()(
        _settings(configured_password),
        service,
    )

    assert result == "existing"
    assert service.mock_calls == [call.has_active_admin()]


@pytest.mark.parametrize("configured_password", [None, ""])
def test_bootstrap_requires_nonempty_password_without_active_admin(
    configured_password,
):
    service = Mock()
    service.has_active_admin.return_value = False

    with pytest.raises(RuntimeError, match="DEFAULT_ADMIN_PASSWORD"):
        _bootstrap_default_admin()(_settings(configured_password), service)

    assert service.mock_calls == [call.has_active_admin()]


def test_bootstrap_rejects_password_shorter_than_ten_characters():
    service = Mock()
    service.has_active_admin.return_value = False

    with pytest.raises(RuntimeError, match="at least 10"):
        _bootstrap_default_admin()(_settings("123456789"), service)

    assert service.mock_calls == [call.has_active_admin()]


def test_bootstrap_rejects_username_collision_without_mutation():
    service = Mock()
    service.has_active_admin.return_value = False
    service.get_user_by_username.return_value = {
        "user_id": "user-1",
        "username": "admin",
        "is_active": False,
        "is_admin": False,
    }
    service.verify_user_password.return_value = False

    # 用户名被非 admin 用户占用：降级为告警跳过（不阻断启动），返回
    # "skipped-collision"，且不修改该用户。
    result = _bootstrap_default_admin()(_settings("StrongAdmin-2026"), service)
    assert result == "skipped-collision"

    assert service.mock_calls == [
        call.has_active_admin(),
        call.get_user_by_username("admin"),
        call.verify_user_password("user-1", "StrongAdmin-2026"),
    ]


def test_bootstrap_matching_collision_promotes_and_reactivates_user():
    service = Mock()
    service.has_active_admin.return_value = False
    service.get_user_by_username.return_value = {
        "user_id": "user-1",
        "username": "admin",
        "is_active": False,
        "is_admin": False,
    }
    service.verify_user_password.return_value = True

    result = _bootstrap_default_admin()(_settings("StrongAdmin-2026"), service)

    assert result == "promoted"
    assert service.mock_calls == [
        call.has_active_admin(),
        call.get_user_by_username("admin"),
        call.verify_user_password("user-1", "StrongAdmin-2026"),
        call.set_account_flags(
            "user-1",
            is_active=True,
            is_admin=True,
            acting_user_id=None,
            bootstrap=True,
        ),
    ]


@pytest.fixture()
def user_service_db(monkeypatch):
    user_service_module = importlib.import_module(
        "train_factory.auth.user_service"
    )
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
    yield (
        user_service_module.UserService(),
        get_test_session,
        user_service_module,
    )
    engine.dispose()


def test_create_user_hashes_password_and_returns_public_record(user_service_db):
    service, get_test_session, _ = user_service_db

    created = service.create_user(
        "alice",
        "StrongAlice-2026",
        email="alice@example.com",
        is_admin=True,
    )

    assert created["username"] == "alice"
    assert created["email"] == "alice@example.com"
    assert created["is_admin"] is True
    assert "hashed_password" not in created
    assert "access_token" not in created
    with get_test_session() as session:
        stored = session.exec(
            select(UserDB).where(UserDB.user_id == created["user_id"])
        ).one()
        assert stored.hashed_password != "StrongAlice-2026"
        assert service.verify_user_password(
            created["user_id"], "StrongAlice-2026"
        )


def test_create_user_enforces_username_and_email_uniqueness(user_service_db):
    service, _, _ = user_service_db
    service.create_user(
        "alice",
        "StrongAlice-2026",
        email="alice@example.com",
    )

    with pytest.raises(ValueError, match="Username 'alice' already exists"):
        service.create_user("alice", "AnotherAlice-2026")
    with pytest.raises(ValueError, match="Email 'alice@example.com' already registered"):
        service.create_user(
            "alice-2",
            "AnotherAlice-2026",
            email="alice@example.com",
        )


def test_register_reuses_create_user_and_preserves_token_response(monkeypatch):
    user_service_module = importlib.import_module(
        "train_factory.auth.user_service"
    )
    service = user_service_module.UserService()
    create_user = Mock(
        return_value={
            "user_id": "user-1",
            "username": "alice",
            "email": None,
            "is_active": True,
            "is_admin": False,
            "token_version": 0,
        }
    )
    monkeypatch.setattr(service, "create_user", create_user)
    monkeypatch.setattr(
        user_service_module,
        "create_access_token",
        lambda user_id, username, *args, **kwargs: f"token:{user_id}:{username}",
    )

    result = service.register("alice", "StrongAlice-2026")

    create_user.assert_called_once_with(
        username="alice",
        password="StrongAlice-2026",
        email=None,
        is_admin=False,
    )
    assert result == {
        "access_token": "token:user-1:alice",
        "token_type": "bearer",
        "user": create_user.return_value,
    }


def test_has_active_admin_only_counts_active_admins(user_service_db):
    service, get_test_session, _ = user_service_db
    service.create_user("regular", "StrongRegular-2026")
    inactive_admin = service.create_user(
        "admin",
        "StrongAdmin-2026",
        is_admin=True,
    )
    with get_test_session() as session:
        stored = session.exec(
            select(UserDB).where(UserDB.user_id == inactive_admin["user_id"])
        ).one()
        stored.is_active = False
        session.add(stored)
        session.commit()

    assert service.has_active_admin() is False

    service.set_account_flags(
        inactive_admin["user_id"],
        is_active=True,
        acting_user_id=None,
        bootstrap=True,
    )
    assert service.has_active_admin() is True


def test_verify_user_password_is_false_for_missing_or_wrong_password(
    user_service_db,
):
    service, _, _ = user_service_db
    created = service.create_user("alice", "StrongAlice-2026")

    assert service.verify_user_password("missing-user", "StrongAlice-2026") is False
    assert service.verify_user_password(created["user_id"], "wrong-password") is False
    assert service.verify_user_password(created["user_id"], "StrongAlice-2026") is True


def test_set_account_flags_only_changes_supplied_flags_and_updates_timestamp(
    user_service_db,
):
    service, get_test_session, _ = user_service_db
    created = service.create_user("alice", "StrongAlice-2026")
    old_timestamp = datetime(2000, 1, 1)
    with get_test_session() as session:
        stored = session.exec(
            select(UserDB).where(UserDB.user_id == created["user_id"])
        ).one()
        stored.updated_at = old_timestamp
        session.add(stored)
        session.commit()

    updated = service.set_account_flags(
        created["user_id"],
        is_admin=True,
        acting_user_id=None,
        bootstrap=True,
    )

    assert updated["is_active"] is True
    assert updated["is_admin"] is True
    assert datetime.fromisoformat(updated["updated_at"]) > old_timestamp
    with pytest.raises(ValueError, match="User not found"):
        service.set_account_flags("missing-user", is_active=False)


def _patch_lifespan_boundaries(monkeypatch, events):
    server = importlib.import_module("train_factory.api.server")
    sync_routes_module = importlib.import_module(
        "train_factory.api.routes.sync_routes"
    )
    sync_manager_module = importlib.import_module(
        "train_factory.sync.sync_manager"
    )

    class Worker:
        async def start(self):
            events.append("worker:start")

        async def stop(self):
            events.append("worker:stop")

    cleanup_names = (
        "cleanup_orphan_tasks",
        "cleanup_orphan_sync_trainings",
        "cleanup_orphan_evaluation_tasks",
        "cleanup_orphan_deep_evaluation_tasks",
        "cleanup_orphan_generation_tasks",
        "cleanup_orphan_datasets",
        "cleanup_orphan_containers",
    )
    for name in cleanup_names:
        monkeypatch.setattr(
            server,
            name,
            lambda name=name: events.append(name),
        )
    monkeypatch.setattr(
        server,
        "cleanup_orphan_sync_tasks",
        lambda: events.append("cleanup_orphan_sync_tasks"),
    )

    async def resume_pending_sync_deletions():
        events.append("resume_pending_sync_deletions")
        return 0, 0

    monkeypatch.setattr(
        sync_routes_module,
        "resume_pending_sync_deletions",
        resume_pending_sync_deletions,
    )
    monkeypatch.setattr(server.settings, "storage_backend", "local")
    monkeypatch.setattr(sync_manager_module, "sync_manager", Worker())
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
    startup_names = (
        *cleanup_names,
        "resume_pending_sync_deletions",
        "cleanup_orphan_sync_tasks",
    )
    return server, startup_names


def test_lifespan_initializes_database_and_admin_before_cleanup_and_worker(
    monkeypatch,
):
    events = []
    server, cleanup_names = _patch_lifespan_boundaries(monkeypatch, events)
    monkeypatch.setattr(server, "init_db", lambda: events.append("init_db"))
    monkeypatch.setattr(
        server,
        "bootstrap_default_admin",
        lambda current_settings: events.append("bootstrap") or "existing",
        raising=False,
    )

    async def run_lifespan():
        async with server.lifespan(SimpleNamespace()):
            events.append("yield")

    asyncio.run(run_lifespan())

    assert events == [
        "init_db",
        "bootstrap",
        *cleanup_names,
        "worker:start",
        "yield",
        "worker:stop",
    ]


@pytest.mark.parametrize("failure_stage", ["init_db", "bootstrap"])
def test_lifespan_does_not_swallow_database_or_bootstrap_errors(
    monkeypatch,
    failure_stage,
):
    events = []
    server, _ = _patch_lifespan_boundaries(monkeypatch, events)

    def init_db():
        events.append("init_db")
        if failure_stage == "init_db":
            raise RuntimeError("database failed")

    def bootstrap(current_settings):
        events.append("bootstrap")
        if failure_stage == "bootstrap":
            raise RuntimeError("bootstrap failed")
        return "existing"

    monkeypatch.setattr(server, "init_db", init_db)
    monkeypatch.setattr(
        server,
        "bootstrap_default_admin",
        bootstrap,
        raising=False,
    )

    async def run_lifespan():
        async with server.lifespan(SimpleNamespace()):
            events.append("yield")

    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(run_lifespan())

    expected = ["init_db"]
    if failure_stage == "bootstrap":
        expected.append("bootstrap")
    assert events == expected
