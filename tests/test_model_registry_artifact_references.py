from contextlib import contextmanager
import importlib

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from train_factory.storage.entities.model_registry_entity import ModelRegistryDB


service_module = importlib.import_module(
    "train_factory.storage.services.model_registry_service"
)


def _configure_service(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ModelRegistryDB.__table__.create(engine)

    @contextmanager
    def test_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(service_module, "get_session", test_session)
    return service_module.ModelRegistryService(), engine


def _add_model(
    engine,
    *,
    model_id,
    model_path,
    user_id,
    source_task_id=None,
):
    with Session(engine) as session:
        session.add(
            ModelRegistryDB(
                model_id=model_id,
                model_name=model_id,
                model_type="embedding",
                model_path=model_path,
                source_task_id=source_task_id,
                user_id=user_id,
            )
        )
        session.commit()


def test_registry_finds_exact_and_normalized_descendant_artifact_paths(
    monkeypatch,
    tmp_path,
):
    service, engine = _configure_service(monkeypatch)
    output_dir = tmp_path / "output" / "training-1"
    final_model_path = output_dir / "final_model"
    normalized_descendant = (
        final_model_path / "exports" / ".." / "snapshot"
    )
    _add_model(
        engine,
        model_id="model-exact",
        model_path=str(final_model_path),
        user_id="user-1",
    )
    _add_model(
        engine,
        model_id="model-descendant",
        model_path=str(normalized_descendant),
        user_id="user-1",
    )
    _add_model(
        engine,
        model_id="model-sibling",
        model_path=str(output_dir.parent / "training-10" / "final_model"),
        user_id="user-1",
    )

    models = service.list_models_referencing_artifact_paths(
        [str(output_dir), str(final_model_path)],
        user_id="user-1",
    )

    assert {model["model_id"] for model in models} == {
        "model-exact",
        "model-descendant",
    }


def test_registry_path_lookup_isolates_null_owner_from_other_tenants(
    monkeypatch,
    tmp_path,
):
    service, engine = _configure_service(monkeypatch)
    artifact_path = tmp_path / "output" / "training-1" / "final_model"
    _add_model(
        engine,
        model_id="model-foreign",
        model_path=str(artifact_path),
        user_id="user-2",
    )
    _add_model(
        engine,
        model_id="model-auth-disabled",
        model_path=str(artifact_path),
        user_id=None,
    )

    models = service.list_models_referencing_artifact_paths(
        [str(artifact_path)],
        user_id=None,
    )

    assert [model["model_id"] for model in models] == [
        "model-auth-disabled"
    ]


def test_source_task_lookup_with_null_owner_does_not_return_foreign_model(
    monkeypatch,
    tmp_path,
):
    service, engine = _configure_service(monkeypatch)
    task_id = "training-1"
    _add_model(
        engine,
        model_id="model-foreign",
        model_path=str(tmp_path / "foreign"),
        user_id="user-2",
        source_task_id=task_id,
    )
    _add_model(
        engine,
        model_id="model-auth-disabled",
        model_path=str(tmp_path / "auth-disabled"),
        user_id=None,
        source_task_id=task_id,
    )

    model = service.get_model_by_source_task(task_id, user_id=None)

    assert model is not None
    assert model["model_id"] == "model-auth-disabled"
