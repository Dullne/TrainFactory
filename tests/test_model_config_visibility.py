"""Model configuration visibility must be consistent across read operations."""

from sqlmodel import Session, SQLModel, create_engine

from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.services.model_config_service import ModelConfigService


def test_ownerless_configs_are_visible_to_all_user_scoped_reads(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'model-config-visibility.db'}")
    SQLModel.metadata.create_all(engine)
    service = ModelConfigService()
    service.engine = engine

    with Session(engine) as session:
        session.add_all(
            [
                ModelConfigDB(
                    config_id="shared-config",
                    config_name="shared-public",
                    description="shared public configuration",
                    model_type="llm",
                    provider="openai",
                    api_endpoint="https://api.example.invalid/v1",
                    model_name="shared-model",
                    user_id=None,
                    is_default=True,
                ),
                ModelConfigDB(
                    config_id="owned-config",
                    config_name="owned-private",
                    description="private configuration",
                    model_type="llm",
                    provider="openai",
                    api_endpoint="https://api.example.invalid/v1",
                    model_name="owned-model",
                    user_id="user-1",
                ),
            ]
        )
        session.commit()

    listed, total = service.list_configs(user_id="user-1")
    assert total == 2
    assert {item["config_id"] for item in listed} == {"shared-config", "owned-config"}

    searched = service.search_configs("shared", user_id="user-1")
    assert [item["config_id"] for item in searched] == ["shared-config"]

    default = service.get_default_config("llm", user_id="user-1")
    assert default is not None
    assert default["config_id"] == "shared-config"

    stats = service.get_stats(user_id="user-1")
    assert stats["total"] == 2
    assert stats["by_provider"] == {"openai": 2}
