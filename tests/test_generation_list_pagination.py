from __future__ import annotations

import asyncio
import importlib
from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine

from train_factory.api.routes import generation_routes
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)


def test_generation_list_returns_filtered_total_and_global_user_stats(
    tmp_path,
    monkeypatch,
):
    engine = create_engine(f"sqlite:///{tmp_path / 'generation-list.db'}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all([
            GenerationTaskDB(
                task_id=f"own-completed-{index:03d}",
                task_name=f"Own completed {index}",
                input_path="/tmp/input.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.COMPLETED,
                user_id="user-1",
            )
            for index in range(101)
        ])
        session.add(GenerationTaskDB(
            task_id="own-running",
            task_name="Own running",
            input_path="/tmp/input.jsonl",
            llm_config={},
            steps_config={},
            status=GenerationStatus.RUNNING,
            user_id="user-1",
        ))
        session.add(GenerationTaskDB(
            task_id="other-failed",
            task_name="Other failed",
            input_path="/tmp/input.jsonl",
            llm_config={},
            steps_config={},
            status=GenerationStatus.FAILED,
            user_id="user-2",
        ))
        session.commit()

    service_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)

    response = asyncio.run(generation_routes.list_tasks(
        status=GenerationStatus.COMPLETED,
        generation_mode=None,
        limit=10,
        offset=100,
        current_user={"user_id": "user-1"},
    ))

    assert response.total == 101
    assert response.limit == 10
    assert response.offset == 100
    assert len(response.tasks) == 1
    assert response.tasks[0].task_id.startswith("own-completed-")
    assert response.stats.model_dump() == {
        "total": 102,
        "pending": 0,
        "running": 1,
        "stopping": 0,
        "publishing": 0,
        "recovering": 0,
        "restarting": 0,
        "completed": 101,
        "failed": 0,
        "stopped": 0,
    }
