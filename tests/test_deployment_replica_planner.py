from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from train_factory.deployment.docker_deployer import DockerDeployer
from train_factory.deployment.launch_config import parse_launch_config
from train_factory.deployment.replica_planner import plan_deployment_replicas


class FakePortAllocator:
    def __init__(self, ports: list[int], *, fail_after: int | None = None) -> None:
        self._ports = iter(ports)
        self.fail_after = fail_after
        self.allocations: list[tuple[int, str]] = []
        self.releases: list[tuple[int, str]] = []

    def find_available_port(self, *, owner_token: str) -> int:
        if self.fail_after is not None and len(self.allocations) >= self.fail_after:
            raise RuntimeError("port allocation failed")
        port = next(self._ports)
        self.allocations.append((port, owner_token))
        return port

    def release_port(self, port: int, *, owner_token: str) -> bool:
        self.releases.append((port, owner_token))
        return True


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_required_gpus_equal_tensor_pipeline_and_data_parallel(framework: str) -> None:
    allocator = FakePortAllocator([11000, 11001])
    config = parse_launch_config(
        {
            "framework": framework,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 1,
            "gpu_pool": list(range(8)),
        }
    )

    plan = plan_deployment_replicas(
        base_container_name=f"trainfactory-{framework}-model-deadbeef",
        replica_count=2,
        launch_config=config,
        gpu_inventory=list(range(8)),
        port_allocator=allocator,
        owner_token="create-owned",
    )

    assert [replica.gpu_ids for replica in plan.replicas] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ]


def test_expert_parallel_does_not_multiply_gpu_requirement() -> None:
    config = parse_launch_config(
        {
            "framework": "sglang",
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "expert_parallel_size": 4,
            "gpu_pool": [3, 1, 7, 5],
        }
    )

    plan = plan_deployment_replicas(
        base_container_name="trainfactory-sglang-model-deadbeef",
        replica_count=1,
        launch_config=config,
        gpu_inventory=[1, 3, 5, 7],
        port_allocator=FakePortAllocator([11000]),
        owner_token="create-owned",
    )

    assert plan.required_gpus_per_replica == 4
    assert plan.replicas[0].gpu_ids == (3, 1, 7, 5)


def test_overrides_take_precedence_then_auto_split_preserves_remaining_order() -> None:
    config = parse_launch_config(
        {
            "framework": "vllm",
            "tensor_parallel_size": 2,
            "gpu_pool": [5, 3, 1, 7, 9, 11],
            "replica_gpu_overrides": [
                {"replica_index": 1, "gpu_ids": [7, 3]},
            ],
        }
    )

    plan = plan_deployment_replicas(
        base_container_name="trainfactory-vllm-model-deadbeef",
        replica_count=3,
        launch_config=config,
        gpu_inventory=[1, 3, 5, 7, 9, 11],
        port_allocator=FakePortAllocator([11000, 11001, 11002]),
        owner_token="create-owned",
    )

    assert [replica.gpu_ids for replica in plan.replicas] == [
        (5, 1),
        (7, 3),
        (9, 11),
    ]


@pytest.mark.parametrize(
    ("payload", "inventory", "message"),
    [
        (
            {"framework": "vllm", "tensor_parallel_size": 2, "gpu_pool": [0]},
            [0],
            "insufficient GPUs",
        ),
        (
            {"framework": "vllm", "gpu_pool": [0, 4]},
            [0, 1],
            "GPU inventory",
        ),
        (
            {
                "framework": "vllm",
                "gpu_pool": [0, 1],
                "replica_gpu_overrides": [
                    {"replica_index": 0, "gpu_ids": [0, 1]},
                ],
            },
            [0, 1],
            "exactly 1 GPUs",
        ),
    ],
)
def test_invalid_gpu_plans_fail_before_port_reservation(
    payload: dict,
    inventory: list[int],
    message: str,
) -> None:
    allocator = FakePortAllocator([11000])

    with pytest.raises(ValueError, match=message):
        plan_deployment_replicas(
            base_container_name="trainfactory-vllm-model-deadbeef",
            replica_count=2 if message == "insufficient GPUs" else 1,
            launch_config=parse_launch_config(payload),
            gpu_inventory=inventory,
            port_allocator=allocator,
            owner_token="create-owned",
        )

    assert allocator.allocations == []


def test_duplicate_gpu_use_is_rejected_unless_reuse_is_explicit() -> None:
    base = {
        "framework": "vllm",
        "gpu_pool": [0, 1],
        "replica_gpu_overrides": [
            {"replica_index": 0, "gpu_ids": [0]},
            {"replica_index": 1, "gpu_ids": [0]},
        ],
    }
    allocator = FakePortAllocator([11000, 11001])
    with pytest.raises(ValueError, match="GPU reuse"):
        plan_deployment_replicas(
            base_container_name="trainfactory-vllm-model-deadbeef",
            replica_count=2,
            launch_config=parse_launch_config(base),
            gpu_inventory=[0, 1],
            port_allocator=allocator,
            owner_token="create-owned",
        )
    assert allocator.allocations == []

    reused = plan_deployment_replicas(
        base_container_name="trainfactory-vllm-model-deadbeef",
        replica_count=2,
        launch_config=parse_launch_config({**base, "allow_gpu_reuse": True}),
        gpu_inventory=[0, 1],
        port_allocator=allocator,
        owner_token="create-owned",
    )
    assert [replica.gpu_ids for replica in reused.replicas] == [(0,), (0,)]


def test_port_reservations_are_transactional_and_released_in_reverse_order() -> None:
    allocator = FakePortAllocator([11000, 11001], fail_after=2)
    config = parse_launch_config({"framework": "vllm", "gpu_pool": [0, 1, 2]})

    with pytest.raises(RuntimeError, match="port allocation failed"):
        plan_deployment_replicas(
            base_container_name="trainfactory-vllm-model-deadbeef",
            replica_count=3,
            launch_config=config,
            gpu_inventory=[0, 1, 2],
            port_allocator=allocator,
            owner_token="create-owned",
        )

    assert allocator.releases == [
        (11001, "create-owned"),
        (11000, "create-owned"),
    ]


def test_plan_has_stable_names_endpoints_and_unique_ports() -> None:
    plan = plan_deployment_replicas(
        base_container_name="trainfactory-vllm-model-deadbeef",
        replica_count=3,
        launch_config=parse_launch_config({"framework": "vllm", "gpu_pool": [2, 4, 6]}),
        gpu_inventory=[2, 4, 6],
        port_allocator=FakePortAllocator([11002, 11000, 11001]),
        owner_token="create-owned",
    )

    assert [replica.replica_index for replica in plan.replicas] == [0, 1, 2]
    assert [replica.container_name for replica in plan.replicas] == [
        "trainfactory-vllm-model-deadbeef",
        "trainfactory-vllm-model-deadbeef-r1",
        "trainfactory-vllm-model-deadbeef-r2",
    ]
    assert [replica.port for replica in plan.replicas] == [11002, 11000, 11001]
    assert [replica.endpoint for replica in plan.replicas] == [
        "http://trainfactory-vllm-model-deadbeef:11002",
        "http://trainfactory-vllm-model-deadbeef-r1:11000",
        "http://trainfactory-vllm-model-deadbeef-r2:11001",
    ]


def test_docker_deployer_reserves_parent_child_and_external_config_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        [
            [SimpleNamespace(port=10001)],
            [SimpleNamespace(port=10002)],
            [
                SimpleNamespace(api_endpoint="http://127.0.0.1:10003/v1"),
                SimpleNamespace(api_endpoint="http://[::1]:10004/v1"),
            ],
        ]
    )

    class FakeSession:
        def exec(self, _statement):
            return SimpleNamespace(all=lambda: next(results))

    @contextmanager
    def fake_get_session():
        yield FakeSession()

    monkeypatch.setattr(
        "train_factory.storage.database.get_session",
        fake_get_session,
    )

    assert DockerDeployer()._get_db_reserved_ports() == {
        10001,
        10002,
        10003,
        10004,
    }


@pytest.mark.parametrize("failed_query", [0, 1, 2])
def test_docker_deployer_fails_closed_when_any_reserved_port_query_fails(
    monkeypatch: pytest.MonkeyPatch,
    failed_query: int,
) -> None:
    results = [
        [SimpleNamespace(port=10001)],
        [SimpleNamespace(port=10002)],
        [SimpleNamespace(api_endpoint="http://127.0.0.1:10003/v1")],
    ]

    class FakeSession:
        def __init__(self) -> None:
            self.query_index = 0

        def exec(self, _statement):
            query_index = self.query_index
            self.query_index += 1
            if query_index == failed_query:
                raise RuntimeError(f"query {query_index} failed")
            return SimpleNamespace(all=lambda: results[query_index])

    @contextmanager
    def fake_get_session():
        yield FakeSession()

    monkeypatch.setattr(
        "train_factory.storage.database.get_session",
        fake_get_session,
    )

    with pytest.raises(RuntimeError, match="Failed to query DB for reserved ports"):
        DockerDeployer()._get_db_reserved_ports()
