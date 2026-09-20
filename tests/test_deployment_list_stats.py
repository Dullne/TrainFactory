"""Deployment aggregates must describe the authorized query, not one page."""

import importlib
import threading
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from train_factory.api.routes import deployment_routes
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB


service_module = importlib.import_module("train_factory.deployment.deployment_service")


@pytest.fixture
def service(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)

    @contextmanager
    def session_scope():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", session_scope)
    with Session(engine) as session:
        for index, (owner, model, status) in enumerate([
            ("alice", "model-a", "running"),
            ("alice", "model-a", "running"),
            ("alice", "model-a", "stopped"),
            ("alice", "model-b", "failed"),
            ("bob", "model-a", "running"),
            (None, "model-a", "running"),
        ]):
            session.add(DeploymentDB(
                deployment_id=f"deployment-{index}", deployment_name=f"deployment-{index}",
                user_id=owner, model_id=model, status=status,
                xinference_endpoint="http://example.invalid:9997", deploy_mode="container",
            ))
        session.commit()
    yield service_module.deployment_service
    engine.dispose()


@pytest.mark.parametrize("filters,expected", [
    ({"user_id": "alice"}, {"total": 4, "by_status": {"running": 2, "stopped": 1, "failed": 1}}),
    ({"user_id": "alice", "model_id": "model-a"}, {"total": 3, "by_status": {"running": 2, "stopped": 1}}),
    ({"user_id": "alice", "status": "running"}, {"total": 2, "by_status": {"running": 2}}),
    ({"user_id": "alice", "model_id": "model-b", "status": "running"}, {"total": 0, "by_status": {}}),
    ({"user_id": "nobody"}, {"total": 0, "by_status": {}}),
])
def test_aggregate_respects_owner_and_filters_but_not_pagination(service, filters, expected):
    items, total, stats = service.list_deployments_with_stats(**filters, limit=1, offset=1)
    assert total == expected["total"]
    assert stats == expected
    assert len(items) == (1 if total > 1 else 0)
    assert all(item["user_id"] == filters["user_id"] for item in items)
    # Existing callers still receive the original pair.
    old_items, old_total = service.list_deployments(**filters, limit=1, offset=1)
    assert old_items == items
    assert old_total == total


def test_list_response_includes_aggregate_and_offloads_blocking_queries(service, monkeypatch):
    app = FastAPI()
    app.include_router(deployment_routes.router)
    app.dependency_overrides[deployment_routes.get_current_user] = lambda: {"user_id": "alice"}
    monkeypatch.setattr(deployment_routes.docker_deployer, "get_all_container_gpu_usage", lambda: {})
    event_loop_threads = []
    query_threads = []

    @app.middleware("http")
    async def record_loop_thread(request, call_next):
        event_loop_threads.append(threading.get_ident())
        return await call_next(request)

    original = service.list_deployments_with_stats

    def record_query_thread(**kwargs):
        query_threads.append(threading.get_ident())
        return original(**kwargs)

    monkeypatch.setattr(service, "list_deployments_with_stats", record_query_thread)
    with TestClient(app) as client:
        response = client.get("/deployments?model_id=model-a&limit=1&offset=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert len(payload["deployments"]) == 1
    assert payload["stats"] == {"total": 3, "by_status": {"running": 2, "stopped": 1}}
    assert query_threads and set(query_threads).isdisjoint(event_loop_threads)


def test_sync_aggregates_post_sync_status_without_syncing_all_pages(service, monkeypatch):
    app = FastAPI()
    app.include_router(deployment_routes.router)
    app.dependency_overrides[deployment_routes.get_current_user] = lambda: {"user_id": "alice"}
    monkeypatch.setattr(deployment_routes.docker_deployer, "get_all_container_gpu_usage", lambda: {})
    monkeypatch.setattr(deployment_routes, "_validate_deployment_endpoint", lambda _: None)
    synced = []

    def sync_one(deployment_id):
        synced.append(deployment_id)
        with service_module.get_session() as session:
            from sqlmodel import select
            deployment = session.exec(select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)).one()
            deployment.status = "failed"
            session.add(deployment)
            session.commit()

    monkeypatch.setattr(service, "sync_status", sync_one)
    monkeypatch.setattr(service, "sync_all_running", lambda: pytest.fail("must not sync all pages for statistics"))
    with TestClient(app) as client:
        response = client.get("/deployments?model_id=model-a&status=running&limit=1&sync=true")

    assert response.status_code == 200
    assert len(synced) == 1
    assert response.json()["stats"] == {"total": 1, "by_status": {"running": 1}}
    assert response.json()["total"] == 1
