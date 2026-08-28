from __future__ import annotations

import json

import pytest

from train_factory.deployment.docker_deployer import DockerDeployer


@pytest.mark.parametrize(
    "inspect_result",
    [
        (False, "docker daemon unavailable"),
        (
            True,
            json.dumps(
                {
                    "Id": "a" * 64,
                    "Name": "/trainfactory-xf-orphan-deadbeef",
                    "Config": {"Labels": {}},
                }
            ),
        ),
        (
            True,
            json.dumps(
                {
                    "Id": "b" * 64,
                    "Name": "/trainfactory-xf-orphan-deadbeef",
                    "Config": {
                        "Labels": {
                            "com.trainfactory.managed": "deployment-replica",
                            "com.trainfactory.project": "other-project",
                            "com.trainfactory.deployment-id": "replacement",
                            "com.trainfactory.replica-id": "replacement-r0",
                        }
                    },
                }
            ),
        ),
    ],
    ids=("inspect-failure", "unlabelled-legacy", "same-name-replacement"),
)
def test_startup_orphan_cleanup_preserves_unverified_legacy_name(
    monkeypatch: pytest.MonkeyPatch,
    inspect_result: tuple[bool, str],
) -> None:
    deployer = DockerDeployer()
    candidate = "trainfactory-xf-orphan-deadbeef"
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "trainfactory")
    monkeypatch.setattr(
        deployer,
        "list_xinference_containers",
        lambda: [candidate],
    )
    monkeypatch.setattr(
        deployer,
        "_run_command",
        lambda *_args, **_kwargs: inspect_result,
    )
    monkeypatch.setattr(
        deployer,
        "remove_container",
        lambda _name: pytest.fail("unverified names must never be removed"),
    )
    monkeypatch.setattr(
        deployer,
        "remove_container_identity",
        lambda _container_id: pytest.fail(
            "unverified identities must never be removed"
        ),
    )

    assert deployer.cleanup_orphan_containers(set()) == 0


def test_startup_orphan_cleanup_removes_verified_identity_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    candidate = "trainfactory-xf-orphan-deadbeef"
    container_id = "c" * 64
    removed: list[str] = []
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "trainfactory")
    monkeypatch.setattr(
        deployer,
        "list_xinference_containers",
        lambda: [candidate],
    )
    monkeypatch.setattr(
        deployer,
        "_run_command",
        lambda *_args, **_kwargs: (
            True,
            json.dumps(
                {
                    "Id": container_id,
                    "Name": f"/{candidate}",
                    "Config": {
                        "Labels": {
                                "com.trainfactory.managed": "deployment-replica",
                            "com.trainfactory.project": "trainfactory",
                            "com.trainfactory.deployment-id": "orphan",
                            "com.trainfactory.replica-id": "orphan-r0",
                        }
                    },
                }
            ),
        ),
    )
    monkeypatch.setattr(
        deployer,
        "remove_container",
        lambda _name: pytest.fail("name-based removal is not allowed"),
    )
    monkeypatch.setattr(
        deployer,
        "remove_container_identity",
        lambda observed: removed.append(observed) or True,
    )

    assert deployer.cleanup_orphan_containers(set()) == 1
    assert removed == [container_id]
