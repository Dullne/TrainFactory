"""
REST API routes for external API configuration management.

Provides CRUD endpoints for managing reusable external API connections
(URL + Bearer Token) that can be referenced by sync configurations.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...auth.dependencies import get_current_user, verify_resource_ownership
from ...config.settings import get_settings
from ...storage.services.outbound_endpoint_policy import (
    validate_user_outbound_url,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_user_id(current_user: Dict[str, Any], for_query: bool = False) -> Optional[str]:
    if not get_settings().auth_enabled:
        return None if for_query else ""
    return current_user.get("user_id")


# ── Request Models ──

class CreateExternalApiConfigRequest(BaseModel):
    config_name: str = Field(..., min_length=1, max_length=255)
    api_url: str = Field(..., min_length=1)
    auth_config: Dict[str, Any] = Field(..., description="Auth config, e.g. {token: '...'}")
    description: str = Field("", max_length=1024)


class UpdateExternalApiConfigRequest(BaseModel):
    config_name: Optional[str] = Field(None, min_length=1, max_length=255)
    api_url: Optional[str] = Field(None, min_length=1)
    auth_config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    status: Optional[str] = None


class TestConnectionRequest(BaseModel):
    api_url: str = Field(..., min_length=1)
    auth_config: Dict[str, Any] = Field(..., description="Auth config, e.g. {token: '...', auth_method: 'cookie'}")


# ── Test Connection Endpoints ──
# NOTE: These must be registered BEFORE {config_id} routes to avoid path conflicts.


@router.post("/api-configs/test-connection")
async def test_connection_inline(
    request: TestConnectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Test connection to an external API before saving the config."""
    from ...sync.external_client import (
        ExternalApiAuthenticationError,
        ExternalApiClient,
    )

    api_url = validate_user_outbound_url(
        request.api_url,
        current_user.get("user_id"),
    )
    try:
        client = ExternalApiClient(
            api_url,
            request.auth_config,
            timeout=10,
            user_id=current_user.get("user_id"),
        )
        result = await client.fetch_incremental(limit=1)
        return {
            "success": True,
            "message": f"{result['total']}",
            "status_code": 200,
        }
    except ExternalApiAuthenticationError as e:
        return {"success": False, "message": str(e), "status_code": 401}
    except Exception as e:
        return {"success": False, "message": str(e), "status_code": 0}


@router.post("/api-configs/{config_id}/test-connection")
async def test_connection_by_id(
    config_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Test connection for an existing external API config.

    Automatically updates config status: "active" on success, "error" on failure.
    """
    from ...storage.services.external_api_config_service import external_api_config_service
    from ...sync.external_client import (
        ExternalApiAuthenticationError,
        ExternalApiClient,
    )

    config = await run_in_threadpool(
        external_api_config_service.get_config,
        config_id,
    )
    config = verify_resource_ownership(config, current_user, "External API config")

    # Use raw config to get unmasked token
    raw = await run_in_threadpool(
        external_api_config_service.get_config_raw,
        config_id,
    )
    api_url = validate_user_outbound_url(
        raw["api_url"],
        current_user.get("user_id"),
    )

    try:
        client = ExternalApiClient(
            api_url,
            raw["auth_config"],
            timeout=10,
            user_id=current_user.get("user_id"),
        )
        result = await client.fetch_incremental(limit=1)
        await run_in_threadpool(
            external_api_config_service.update_config,
            config_id,
            status="active",
        )
        return {
            "success": True,
            "message": f"{result['total']}",
            "status_code": 200,
        }
    except ExternalApiAuthenticationError as e:
        await run_in_threadpool(
            external_api_config_service.update_config,
            config_id,
            status="error",
        )
        return {"success": False, "message": str(e), "status_code": 401}
    except Exception as e:
        await run_in_threadpool(
            external_api_config_service.update_config,
            config_id,
            status="error",
        )
        return {"success": False, "message": str(e), "status_code": 0}


# ── CRUD Endpoints ──

@router.post("/api-configs", status_code=201)
async def create_api_config(
    request: CreateExternalApiConfigRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Create a new external API configuration.

    After creation, automatically tests the connection and sets status
    to "active" on success or "error" on failure.
    """
    from ...storage.services.external_api_config_service import external_api_config_service
    from ...sync.external_client import ExternalApiClient

    user_id = _resolve_user_id(current_user)
    api_url = validate_user_outbound_url(request.api_url, user_id)
    config = await run_in_threadpool(
        external_api_config_service.create_config,
        config_name=request.config_name,
        user_id=user_id,
        api_url=api_url,
        auth_config=request.auth_config,
        description=request.description,
    )

    # Auto-test connection and update status
    config_id = config["config_id"]
    try:
        client = ExternalApiClient(
            api_url,
            request.auth_config,
            timeout=10,
            user_id=user_id,
        )
        await client.fetch_incremental(limit=1)
        await run_in_threadpool(
            external_api_config_service.update_config,
            config_id,
            status="active",
        )
        config["status"] = "active"
    except Exception as e:
        logger.warning(f"Auto-test failed for new API config {config_id}: {e}")
        await run_in_threadpool(
            external_api_config_service.update_config,
            config_id,
            status="error",
        )
        config["status"] = "error"

    return {"message": "External API config created", "config": config}


@router.get("/api-configs")
async def list_api_configs(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List external API configurations for current user."""
    from ...storage.services.external_api_config_service import external_api_config_service

    user_id = _resolve_user_id(current_user, for_query=True)
    configs, total = await run_in_threadpool(
        external_api_config_service.list_configs,
        user_id=user_id,
    )
    return {"configs": configs, "total": total}


@router.get("/api-configs/{config_id}")
async def get_api_config(
    config_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get a single external API configuration."""
    from ...storage.services.external_api_config_service import external_api_config_service

    config = await run_in_threadpool(
        external_api_config_service.get_config,
        config_id,
    )
    config = verify_resource_ownership(config, current_user, "External API config")
    return {"config": config}


@router.patch("/api-configs/{config_id}")
async def update_api_config(
    config_id: str,
    request: UpdateExternalApiConfigRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Update an external API configuration."""
    from ...storage.services.external_api_config_service import external_api_config_service

    config = await run_in_threadpool(
        external_api_config_service.get_config,
        config_id,
    )
    config = verify_resource_ownership(config, current_user, "External API config")

    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    if "api_url" in updates:
        updates["api_url"] = validate_user_outbound_url(
            updates["api_url"],
            current_user.get("user_id"),
        )

    try:
        updated = await run_in_threadpool(
            external_api_config_service.update_config,
            config_id,
            **updates,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"message": "Config updated", "config": updated}


@router.delete("/api-configs/{config_id}")
async def delete_api_config(
    config_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete an external API configuration."""
    from ...storage.services.external_api_config_service import external_api_config_service

    config = await run_in_threadpool(
        external_api_config_service.get_config,
        config_id,
    )
    config = verify_resource_ownership(config, current_user, "External API config")

    try:
        await run_in_threadpool(
            external_api_config_service.delete_config,
            config_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"message": "Config deleted"}
