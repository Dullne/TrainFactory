import asyncio
import json

import pytest
from sqlmodel import Session, create_engine

from train_factory.api.routes import dataset_routes
from train_factory.storage.entities.dataset_lineage_entity import (
    DatasetLineageEdgeDB,
)


CURRENT_USER = {"user_id": "user-1", "username": "alice", "is_admin": False}


def test_dataset_lineage_filters_edges_to_foreign_nodes(monkeypatch):
    datasets = {
        "owned": {
            "dataset_id": "owned",
            "dataset_name": "Owned",
            "user_id": "user-1",
        },
        "foreign": {
            "dataset_id": "foreign",
            "dataset_name": "Foreign",
            "user_id": "user-2",
        },
    }
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: datasets.get(dataset_id),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_lineage_service,
        "get_lineage_graph",
        lambda **_kwargs: {
            "nodes": ["owned", "foreign"],
            "edges": [
                {
                    "edge_id": "foreign-edge",
                    "from_dataset_id": "foreign",
                    "to_dataset_id": "owned",
                    "op_params": {"secret": "foreign-metadata"},
                },
                {
                    "edge_id": "root-edge",
                    "from_dataset_id": None,
                    "to_dataset_id": "owned",
                    "op_params": None,
                },
            ],
        },
    )

    response = asyncio.run(
        dataset_routes.get_dataset_lineage(
            "owned",
            direction="both",
            depth=2,
            current_user=CURRENT_USER,
        )
    )

    assert [node["dataset_id"] for node in response["nodes"]] == ["owned"]
    assert response["edges"] == [
        {
            "edge_id": "root-edge",
            "from_dataset_id": None,
            "to_dataset_id": "owned",
            "op_params": None,
        }
    ]


@pytest.mark.parametrize("public_surface", ("upstream", "downstream", "lineage"))
def test_public_generation_lineage_never_exposes_raw_attempt_token(
    tmp_path,
    monkeypatch,
    public_surface,
):
    engine = create_engine(f"sqlite:///{tmp_path / 'lineage-token.db'}")
    DatasetLineageEdgeDB.__table__.create(engine)
    raw_run_token = "91919191-9191-4191-8191-919191919191"
    with Session(engine) as session:
        session.add(
            DatasetLineageEdgeDB(
                from_dataset_id="source-dataset",
                to_dataset_id="generated-dataset",
                relation_type="training_generated",
                op_task_type="generation",
                op_task_id="generation-task",
                op_params={
                    "generation_run_token": raw_run_token,
                    "generation_attempt_key": "opaque-attempt-key",
                    "threshold": 0.8,
                },
            )
        )
        session.commit()

    monkeypatch.setattr(dataset_routes.dataset_lineage_service, "engine", engine)
    datasets = {
        dataset_id: {
            "dataset_id": dataset_id,
            "dataset_name": dataset_id,
            "user_id": CURRENT_USER["user_id"],
        }
        for dataset_id in ("source-dataset", "generated-dataset")
    }
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: datasets.get(dataset_id),
    )

    if public_surface == "upstream":
        response = asyncio.run(
            dataset_routes.get_dataset_upstream(
                "generated-dataset",
                current_user=CURRENT_USER,
            )
        )
    elif public_surface == "downstream":
        response = asyncio.run(
            dataset_routes.get_dataset_downstream(
                "source-dataset",
                current_user=CURRENT_USER,
            )
        )
    else:
        response = asyncio.run(
            dataset_routes.get_dataset_lineage(
                "generated-dataset",
                direction="both",
                depth=2,
                current_user=CURRENT_USER,
            )
        )

    serialized = json.dumps(response, sort_keys=True)
    assert raw_run_token not in serialized
    assert "generation_run_token" not in serialized
    assert response["edges"][0]["op_params"] == {
        "generation_attempt_key": "opaque-attempt-key",
        "threshold": 0.8,
    }
