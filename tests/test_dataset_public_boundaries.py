import asyncio

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import dataset_routes
from train_factory.enums.dataset_status import DatasetStatus
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.services.dataset_service import DatasetService


CURRENT_USER = {"user_id": "user-1", "username": "alice", "is_admin": False}


@pytest.mark.parametrize(
    ("surface", "dataset_id", "edges", "expected_edge_ids"),
    (
        (
            "upstream",
            "owned-root",
            [
                {
                    "edge_id": "foreign-upstream",
                    "from_dataset_id": "foreign-parent",
                    "to_dataset_id": "owned-root",
                    "op_task_id": "foreign-task",
                    "op_params": {"secret": "foreign-upstream-metadata"},
                },
                {
                    "edge_id": "owned-upstream",
                    "from_dataset_id": "owned-parent",
                    "to_dataset_id": "owned-root",
                    "op_task_id": "owned-task",
                    "op_params": {"safe": True},
                },
                {
                    "edge_id": "null-root",
                    "from_dataset_id": None,
                    "to_dataset_id": "owned-root",
                    "op_task_id": None,
                    "op_params": None,
                },
            ],
            ["owned-upstream", "null-root"],
        ),
        (
            "downstream",
            "owned-root",
            [
                {
                    "edge_id": "foreign-downstream",
                    "from_dataset_id": "owned-root",
                    "to_dataset_id": "foreign-child",
                    "op_task_id": "foreign-task",
                    "op_params": {"secret": "foreign-downstream-metadata"},
                },
                {
                    "edge_id": "owned-downstream",
                    "from_dataset_id": "owned-root",
                    "to_dataset_id": "owned-child",
                    "op_task_id": "owned-task",
                    "op_params": {"safe": True},
                },
            ],
            ["owned-downstream"],
        ),
    ),
)
def test_direct_lineage_filters_edges_whose_other_endpoint_is_foreign(
    monkeypatch,
    surface,
    dataset_id,
    edges,
    expected_edge_ids,
):
    datasets = {
        "owned-root": {"dataset_id": "owned-root", "user_id": "user-1"},
        "owned-parent": {"dataset_id": "owned-parent", "user_id": "user-1"},
        "owned-child": {"dataset_id": "owned-child", "user_id": "user-1"},
        "foreign-parent": {"dataset_id": "foreign-parent", "user_id": "user-2"},
        "foreign-child": {"dataset_id": "foreign-child", "user_id": "user-2"},
    }
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "get_dataset",
        lambda candidate_id: datasets.get(candidate_id),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_lineage_service,
        f"get_{surface}",
        lambda _dataset_id: edges,
    )

    route = getattr(dataset_routes, f"get_dataset_{surface}")
    response = asyncio.run(route(dataset_id, current_user=CURRENT_USER))

    assert [edge["edge_id"] for edge in response["edges"]] == expected_edge_ids


def test_public_stats_hide_owner_staging_until_ready(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[DatasetDB.__table__])
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id="visible-ready",
                dataset_name="visible ready",
                dataset_type="custom",
                source_type="uploaded",
                usage="train",
                status=DatasetStatus.READY.value,
                file_size=100,
                num_rows=10,
                user_id=CURRENT_USER["user_id"],
            )
        )
        session.add(
            DatasetDB(
                dataset_id="hidden-staging",
                dataset_name="hidden staging",
                dataset_type="sft_instruct",
                source_type="generated",
                usage="eval",
                status=DatasetStatus.STAGING.value,
                file_size=900,
                num_rows=90,
                user_id=CURRENT_USER["user_id"],
            )
        )
        session.commit()

    service = DatasetService()
    service.engine = engine
    monkeypatch.setattr(dataset_routes, "dataset_service", service)

    before_ready = asyncio.run(dataset_routes.get_stats(current_user=CURRENT_USER))

    assert before_ready == {
        "total": 1,
        "by_type": {"custom": 1},
        "by_source": {"uploaded": 1},
        "by_status": {DatasetStatus.READY.value: 1},
        "by_usage": {"train": 1},
        "total_size_bytes": 100,
        "total_rows": 10,
    }

    with Session(engine) as session:
        staging = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == "hidden-staging")
        ).one()
        staging.update_status(DatasetStatus.READY.value)
        session.add(staging)
        session.commit()

    after_ready = asyncio.run(dataset_routes.get_stats(current_user=CURRENT_USER))

    assert after_ready == {
        "total": 2,
        "by_type": {"custom": 1, "sft_instruct": 1},
        "by_source": {"uploaded": 1, "generated": 1},
        "by_status": {DatasetStatus.READY.value: 2},
        "by_usage": {"train": 1, "eval": 1},
        "total_size_bytes": 1000,
        "total_rows": 100,
    }
