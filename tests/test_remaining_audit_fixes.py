import asyncio
import ast
import inspect
from pathlib import Path

from fastapi import Response
from fastapi.params import Depends

from train_factory.api.routes import auth_routes, deployment_routes
from train_factory.auth.dependencies import get_current_user
from train_factory.generation.pipeline import (
    PipelineResult,
    _apply_batch_failure_status,
)
from train_factory.storage.services.external_sync_service import (
    external_sync_service,
)
from train_factory.sync import sync_manager as sync_manager_module
from train_factory.sync.sync_manager import SyncManager
from train_factory.utils.training_metrics import extract_final_loss_metrics


def test_get_me_returns_anonymous_profile_when_auth_is_disabled(monkeypatch):
    monkeypatch.setattr(
        auth_routes.user_service,
        "get_user",
        lambda user_id: (_ for _ in ()).throw(
            AssertionError("anonymous profile must not query the users table")
        ),
    )

    response = asyncio.run(
        auth_routes.get_me({"user_id": None, "username": "anonymous"})
    )

    assert response.user_id == "anonymous"
    assert response.username == "anonymous"
    assert response.is_active is True
    assert response.is_admin is False


def test_auth_cookie_security_uses_explicit_setting(monkeypatch):
    response = Response()
    monkeypatch.setattr(auth_routes.settings, "auth_cookie_secure", True)
    auth_routes._set_auth_cookie(response, "token")
    assert "Secure" in response.headers["set-cookie"]

    response = Response()
    monkeypatch.setattr(auth_routes.settings, "auth_cookie_secure", False)
    auth_routes._set_auth_cookie(response, "token")
    assert "Secure" not in response.headers["set-cookie"]


def test_deployment_info_endpoints_require_current_user_dependency():
    for endpoint in (
        deployment_routes.get_gpu_info,
        deployment_routes.get_default_endpoint,
    ):
        default = inspect.signature(endpoint).parameters["current_user"].default
        assert isinstance(default, Depends)
        assert default.dependency is get_current_user


def test_sync_run_once_refreshes_config_after_taking_lock(monkeypatch):
    manager = SyncManager()
    seen_configs = []

    def get_task_raw(task_id):
        assert manager._get_task_lock(task_id).locked()
        return {"task_id": task_id, "revision": "fresh"}

    monkeypatch.setattr(external_sync_service, "get_task_raw", get_task_raw)
    monkeypatch.setattr(
        sync_manager_module,
        "_run_sync_cycle",
        lambda config: seen_configs.append(config) or None,
    )

    asyncio.run(manager.run_once("sync-1"))

    assert seen_configs == [{"task_id": "sync-1", "revision": "fresh"}]


def test_batch_failure_status_preserves_partial_output_and_fails_task():
    result = PipelineResult(
        success=True,
        output_samples=3,
        output_path="partial.jsonl",
        details={"filter_stats": {}},
    )
    failures = {
        "batch_count": 2,
        "record_count": 7,
        "errors": ["first", "second"],
    }

    finalized = _apply_batch_failure_status(result, failures)

    assert finalized.success is False
    assert finalized.output_path == "partial.jsonl"
    assert finalized.output_samples == 3
    assert "2 generation batch(es) failed" in finalized.error
    assert finalized.details["filter_stats"]["batch_failures"] == failures


def test_extract_final_losses_scans_train_and_eval_logs_independently():
    history = [
        {"loss": 0.9, "step": 10},
        {"eval_loss": 0.8, "step": 10},
        {"loss": 0.7, "step": 20},
        {"train_runtime": 12.0},
    ]

    assert extract_final_loss_metrics(history) == {
        "final_train_loss": 0.7,
        "final_eval_loss": 0.8,
    }


def test_all_route_limit_queries_require_positive_values():
    routes_dir = Path(__file__).parents[1] / "train_factory" / "api" / "routes"
    checked_limits = 0

    for route_path in routes_dir.glob("*_routes.py"):
        tree = ast.parse(route_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            positional_defaults = [None] * (
                len(node.args.args) - len(node.args.defaults)
            ) + list(node.args.defaults)
            parameters = list(zip(node.args.args, positional_defaults))
            parameters.extend(zip(node.args.kwonlyargs, node.args.kw_defaults))

            for argument, default in parameters:
                if argument.arg != "limit" or not isinstance(default, ast.Call):
                    continue
                if not isinstance(default.func, ast.Name) or default.func.id != "Query":
                    continue

                checked_limits += 1
                keywords = {keyword.arg: keyword.value for keyword in default.keywords}
                assert "ge" in keywords, f"{route_path.name} has an unbounded limit"
                assert ast.literal_eval(keywords["ge"]) == 1

    assert checked_limits >= 18


def test_all_route_offset_queries_require_nonnegative_values():
    routes_dir = Path(__file__).parents[1] / "train_factory" / "api" / "routes"
    checked_offsets = 0

    for route_path in routes_dir.glob("*_routes.py"):
        tree = ast.parse(route_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            positional_defaults = [None] * (
                len(node.args.args) - len(node.args.defaults)
            ) + list(node.args.defaults)
            parameters = list(zip(node.args.args, positional_defaults))
            parameters.extend(zip(node.args.kwonlyargs, node.args.kw_defaults))

            for argument, default in parameters:
                if argument.arg != "offset" or not isinstance(default, ast.Call):
                    continue
                if not isinstance(default.func, ast.Name) or default.func.id != "Query":
                    continue

                checked_offsets += 1
                keywords = {keyword.arg: keyword.value for keyword in default.keywords}
                assert "ge" in keywords, f"{route_path.name} has a negative offset"
                assert ast.literal_eval(keywords["ge"]) == 0

    assert checked_offsets >= 13
