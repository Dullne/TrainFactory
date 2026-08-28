"""Pure resource planning for independent inference replicas.

The planner validates the complete GPU topology before reserving ports.  It
does not write the database and never invokes Docker container mutations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .launch_config import LaunchConfig


_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class PortAllocator(Protocol):
    """Narrow reservation surface required by the pure planner."""

    def find_available_port(self, *, owner_token: str) -> int: ...

    def release_port(self, port: int, *, owner_token: str) -> bool: ...


@dataclass(frozen=True)
class ReplicaPlan:
    replica_index: int
    container_name: str
    gpu_ids: tuple[int, ...]
    port: int
    endpoint: str


@dataclass(frozen=True)
class DeploymentPlan:
    replicas: tuple[ReplicaPlan, ...]
    required_gpus_per_replica: int
    reservation_owner: str


def _strict_gpu_inventory(gpu_inventory: Sequence[int]) -> tuple[int, ...]:
    inventory = tuple(gpu_inventory)
    if any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in inventory):
        raise ValueError("GPU inventory must contain non-negative integers")
    if len(set(inventory)) != len(inventory):
        raise ValueError("GPU inventory must not contain duplicates")
    return inventory


def _plan_gpu_assignments(
    *,
    launch_config: LaunchConfig,
    replica_count: int,
    gpu_inventory: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    if type(replica_count) is not int or not 1 <= replica_count <= 8:
        raise ValueError("replica count must be an integer between 1 and 8")

    inventory = _strict_gpu_inventory(gpu_inventory)
    pool = tuple(launch_config.gpu_pool) or inventory
    inventory_set = set(inventory)
    if any(gpu_id not in inventory_set for gpu_id in pool):
        raise ValueError("GPU pool contains an ID outside the trusted GPU inventory")

    required = (
        launch_config.tensor_parallel_size
        * launch_config.pipeline_parallel_size
        * launch_config.data_parallel_size
    )
    overrides = {
        override.replica_index: tuple(override.gpu_ids)
        for override in launch_config.replica_gpu_overrides
    }
    if any(index >= replica_count for index in overrides):
        raise ValueError(
            "replica override index is outside the requested replica count"
        )

    pool_set = set(pool)
    for gpu_ids in overrides.values():
        if len(gpu_ids) != required:
            raise ValueError(
                f"each replica override must contain exactly {required} GPUs"
            )
        if any(gpu_id not in pool_set for gpu_id in gpu_ids):
            raise ValueError("replica override contains a GPU outside the GPU pool")

    overridden_ids = [gpu_id for gpu_ids in overrides.values() for gpu_id in gpu_ids]
    if not launch_config.allow_gpu_reuse and len(set(overridden_ids)) != len(
        overridden_ids
    ):
        raise ValueError("GPU reuse across replicas requires allow_gpu_reuse=true")

    reserved_by_overrides = set(overridden_ids)
    remaining = [gpu_id for gpu_id in pool if gpu_id not in reserved_by_overrides]
    automatic_count = replica_count - len(overrides)
    if len(remaining) < automatic_count * required:
        raise ValueError("insufficient GPUs for the requested replica topology")

    assignments: list[tuple[int, ...]] = []
    cursor = 0
    for replica_index in range(replica_count):
        override = overrides.get(replica_index)
        if override is not None:
            assignments.append(override)
            continue
        assignment = tuple(remaining[cursor : cursor + required])
        cursor += required
        assignments.append(assignment)
    return tuple(assignments)


def plan_deployment_replicas(
    *,
    base_container_name: str,
    replica_count: int,
    launch_config: LaunchConfig,
    gpu_inventory: Sequence[int],
    port_allocator: PortAllocator,
    owner_token: str,
) -> DeploymentPlan:
    """Return an immutable topology after transactionally reserving all ports."""

    if not isinstance(base_container_name, str) or not _CONTAINER_NAME.fullmatch(
        base_container_name
    ):
        raise ValueError("base container name is invalid")
    if not isinstance(owner_token, str) or not owner_token:
        raise ValueError("port reservation owner token must be non-empty")

    assignments = _plan_gpu_assignments(
        launch_config=launch_config,
        replica_count=replica_count,
        gpu_inventory=gpu_inventory,
    )
    required = (
        launch_config.tensor_parallel_size
        * launch_config.pipeline_parallel_size
        * launch_config.data_parallel_size
    )

    reserved_ports: list[int] = []
    try:
        for _replica_index in range(replica_count):
            port = port_allocator.find_available_port(owner_token=owner_token)
            if type(port) is not int or not 1 <= port <= 65535:
                raise RuntimeError("port allocator returned an invalid port")
            if port in reserved_ports:
                raise RuntimeError("port allocator returned a duplicate port")
            reserved_ports.append(port)
    except Exception:
        for port in reversed(reserved_ports):
            port_allocator.release_port(port, owner_token=owner_token)
        raise

    replicas = []
    for replica_index, (gpu_ids, port) in enumerate(
        zip(assignments, reserved_ports, strict=True)
    ):
        container_name = (
            base_container_name
            if replica_index == 0
            else f"{base_container_name}-r{replica_index}"
        )
        if not _CONTAINER_NAME.fullmatch(container_name):
            for reserved_port in reversed(reserved_ports):
                port_allocator.release_port(
                    reserved_port,
                    owner_token=owner_token,
                )
            raise ValueError("replica container name is invalid")
        replicas.append(
            ReplicaPlan(
                replica_index=replica_index,
                container_name=container_name,
                gpu_ids=gpu_ids,
                port=port,
                endpoint=f"http://{container_name}:{port}",
            )
        )

    return DeploymentPlan(
        replicas=tuple(replicas),
        required_gpus_per_replica=required,
        reservation_owner=owner_token,
    )


__all__ = [
    "DeploymentPlan",
    "PortAllocator",
    "ReplicaPlan",
    "plan_deployment_replicas",
]
