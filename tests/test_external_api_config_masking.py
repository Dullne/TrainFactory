import json

from train_factory.storage.entities.external_api_config_entity import (
    ExternalApiConfigDB,
)


def test_external_api_config_recursively_masks_sensitive_auth_values():
    marker = "sensitive-marker-that-must-not-leak"
    auth_config = {
        "token": marker,
        "headers": {
            "Authorization": f"Bearer {marker}",
            "X-Trace-Id": "public-trace",
        },
        "profiles": [
            {"cookie": marker, "label": "primary"},
            {"credentials": {"private_key": marker}},
            {"api_key": marker, "password": marker, "enabled": True},
        ],
    }
    config = ExternalApiConfigDB(
        config_name="nested-auth",
        user_id="user-1",
        api_url="https://api.example.test/items",
        auth_config=auth_config,
    )

    public = config.to_dict()

    assert marker not in json.dumps(public, ensure_ascii=False)
    assert public["auth_config"]["token"] == "***"
    assert public["auth_config"]["headers"] == {
        "Authorization": "***",
        "X-Trace-Id": "public-trace",
    }
    assert public["auth_config"]["profiles"] == [
        {"cookie": "***", "label": "primary"},
        {"credentials": "***"},
        {"api_key": "***", "password": "***", "enabled": True},
    ]

    internal = config.to_dict(mask_sensitive=False)
    assert internal["auth_config"] == auth_config
    assert config.auth_config == auth_config
