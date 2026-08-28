from __future__ import annotations

import importlib
import socket
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlmodel import Session

from train_factory.core.ssrf import SSRFError
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.model_config_entity import ModelConfigDB


adapter_service_module = importlib.import_module(
    "train_factory.deployment.adapter_service"
)
docker_deployer_module = importlib.import_module(
    "train_factory.deployment.docker_deployer"
)
evaluation_runner_module = importlib.import_module(
    "train_factory.evaluation.evaluation_runner"
)
outbound_policy_module = importlib.import_module(
    "train_factory.storage.services.outbound_endpoint_policy"
)
ssrf_module = importlib.import_module("train_factory.core.ssrf")


USER_ID = "user-1"
OTHER_USER_ID = "user-2"
R0_ENDPOINT = "http://trainfactory-vllm-model-deadbeef:11000"
R1_ENDPOINT = "http://trainfactory-vllm-model-deadbeef-r1:11001"


class _SuccessfulResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def json(self):
        return {"data": [{"id": "served-model"}]}

    def iter_content(self, chunk_size: int):  # noqa: ARG002
        yield b"{}"

    def close(self) -> None:
        return None


class _SuccessfulSession:
    def __init__(self) -> None:
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def mount(self, prefix, adapter) -> None:  # noqa: ARG002
        return None

    def request(self, method, url, **kwargs):  # noqa: ARG002
        return _SuccessfulResponse()


@pytest.fixture
def policy_engine(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'replica-policy.db'}")
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)
    ModelConfigDB.__table__.create(engine)

    @contextmanager
    def get_session():
        with Session(engine) as session:
            yield session

    def getaddrinfo(host, port, **kwargs):  # noqa: ARG001
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("172.20.0.10", port),
            )
        ]

    monkeypatch.setattr(outbound_policy_module, "get_session", get_session)
    monkeypatch.setattr(
        outbound_policy_module,
        "get_settings",
        lambda: SimpleNamespace(xinference_endpoint="http://xinference:9997"),
    )
    monkeypatch.setattr(ssrf_module.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(
        outbound_policy_module.requests,
        "Session",
        _SuccessfulSession,
    )
    monkeypatch.delenv("DISCOVER_MODELS_ALLOWED_PRIVATE_HOSTS", raising=False)
    monkeypatch.delenv("HOST_IP", raising=False)
    return engine


def _persist_group(
    engine,
    *,
    status: str,
    user_id: str | None = USER_ID,
    deploy_mode: str = "container",
    replica_marker: bool = True,
) -> DeploymentDB:
    deployment = DeploymentDB(
        deployment_id="deployment-1",
        model_id="model-1",
        model_uid="served-model",
        deployment_name="replica-policy",
        xinference_endpoint=R0_ENDPOINT,
        deploy_mode=deploy_mode,
        container_name="trainfactory-vllm-model-deadbeef",
        port=11000,
        gpu_id=0,
        inference_framework="vllm",
        config={"replica_schema_version": 1} if replica_marker else {},
        status=status,
        user_id=user_id,
    )
    children = [
        DeploymentReplicaDB(
            replica_id=f"replica-{index}",
            deployment_id=deployment.deployment_id,
            replica_index=index,
            container_name=endpoint.removeprefix("http://").split(":", 1)[0],
            endpoint=endpoint,
            port=11000 + index,
            gpu_ids=[index],
            status=status,
        )
        for index, endpoint in enumerate((R0_ENDPOINT, R1_ENDPOINT))
    ]
    with Session(engine) as session:
        session.add(deployment)
        session.add_all(children)
        session.commit()
        session.refresh(deployment)
        return deployment


def test_starting_group_readiness_accepts_every_canonical_child(policy_engine) -> None:
    _persist_group(policy_engine, status="starting")
    deployer = docker_deployer_module.DockerDeployer()

    for endpoint in (R0_ENDPOINT, R1_ENDPOINT):
        assert deployer.wait_for_service(
            endpoint,
            timeout=1,
            poll_interval=0,
            user_id=USER_ID,
            framework="vllm",
        )


def test_ownerless_starting_group_readiness_accepts_canonical_child(
    policy_engine,
) -> None:
    _persist_group(policy_engine, status="starting", user_id=None)
    deployer = docker_deployer_module.DockerDeployer()

    assert deployer.wait_for_service(
        R1_ENDPOINT,
        timeout=1,
        poll_interval=0,
        user_id=None,
        framework="vllm",
    )


def test_running_group_evaluation_accepts_canonical_secondary(policy_engine) -> None:
    _persist_group(policy_engine, status="running")

    assert evaluation_runner_module.validate_user_outbound_url(
        R1_ENDPOINT,
        USER_ID,
    ) == R1_ENDPOINT


def test_running_group_adapter_client_reaches_canonical_secondary(policy_engine) -> None:
    deployment = _persist_group(policy_engine, status="running")
    client = adapter_service_module.AdapterService()._get_inference_client(
        deployment,
        endpoint=R1_ENDPOINT,
    )

    assert client.health_check() is True


def test_stopped_group_trusts_only_its_canonical_destination(policy_engine) -> None:
    _persist_group(policy_engine, status="stopped")

    assert outbound_policy_module.validate_user_outbound_url(
        R0_ENDPOINT,
        USER_ID,
    ) == R0_ENDPOINT
    with pytest.raises(SSRFError):
        outbound_policy_module.validate_user_outbound_url(
            "http://trainfactory-vllm-model-deadbeef:11999",
            USER_ID,
        )


def test_replica_destination_is_not_trusted_for_another_user(policy_engine) -> None:
    _persist_group(policy_engine, status="running")

    with pytest.raises(SSRFError):
        outbound_policy_module.validate_user_outbound_url(
            R1_ENDPOINT,
            OTHER_USER_ID,
        )


def test_shared_deployment_cannot_bootstrap_replica_destination(policy_engine) -> None:
    _persist_group(policy_engine, status="running", deploy_mode="shared")

    with pytest.raises(SSRFError):
        outbound_policy_module.validate_user_outbound_url(
            R1_ENDPOINT,
            USER_ID,
        )


def test_unmarked_container_cannot_bootstrap_replica_destination(policy_engine) -> None:
    _persist_group(policy_engine, status="running", replica_marker=False)

    with pytest.raises(SSRFError):
        outbound_policy_module.validate_user_outbound_url(
            R1_ENDPOINT,
            USER_ID,
        )
