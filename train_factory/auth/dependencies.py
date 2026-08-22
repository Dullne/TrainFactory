"""
FastAPI authentication dependencies.

Provides dependency functions for route authentication.

Security:
- Supports both httpOnly cookie (browser) and Authorization header (API clients)
- Cookie is checked first, then Authorization header
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import Depends, HTTPException, status, Request, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config.settings import get_settings
from .jwt_handler import TokenError, decode_token
from .user_service import user_service

# HTTP Bearer token security scheme
security = HTTPBearer(auto_error=False)

# Cookie name (must match auth_routes.py)
COOKIE_NAME = "access_token"


def _get_token_from_request(
    credentials: Optional[HTTPAuthorizationCredentials],
    cookie_token: Optional[str],
    authorization: Optional[str] = None,
) -> Optional[str]:
    """Extract token from either cookie or Authorization header.

    Cookie takes precedence (browser security).
    """
    # Check cookie first (more secure for browsers)
    if cookie_token:
        return cookie_token

    # Fall back to Authorization header (for API clients)
    if credentials:
        return credentials.credentials

    if authorization:
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme.lower() == "bearer" and token.strip():
            return token.strip()

    return None


def get_access_token_from_http_request(request: Request) -> Optional[str]:
    """Extract an access token from a raw request using dependency precedence."""
    return _get_token_from_request(
        credentials=None,
        cookie_token=request.cookies.get(COOKIE_NAME),
        authorization=request.headers.get("authorization"),
    )


def _invalid_token_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def resolve_current_user(token: str) -> Dict[str, Any]:
    """Resolve a token to the user's current database identity."""
    try:
        payload = decode_token(token)
        user_id = payload["sub"]
        token_username = payload["username"]
        token_version = payload["ver"]
    except (TokenError, KeyError, TypeError):
        raise _invalid_token_error() from None

    if (
        not isinstance(user_id, str)
        or not user_id
        or not isinstance(token_username, str)
        or not token_username
        or type(token_version) is not int
    ):
        raise _invalid_token_error()

    user = user_service.get_user(user_id)
    if not user:
        raise _invalid_token_error()

    database_version = user.get("token_version")
    if (
        user.get("is_active") is not True
        or type(database_version) is not int
        or database_version != token_version
    ):
        raise _invalid_token_error()

    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "is_admin": user["is_admin"],
        "is_active": user["is_active"],
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> Dict[str, Any]:
    """Get current authenticated user from JWT token.

    This dependency requires authentication. Raises 401 if no valid token.
    Accepts token from either httpOnly cookie (browsers) or Authorization header (API clients).

    Args:
        credentials: HTTP Authorization credentials
        access_token: Token from httpOnly cookie

    Returns:
        Dict with user_id and username

    Raises:
        HTTPException: 401 if not authenticated or token invalid
    """
    settings = get_settings()

    # Skip auth if disabled (for development/testing)
    # Return None for user_id so queries don't filter by user
    if not settings.auth_enabled:
        return {"user_id": None, "username": "anonymous"}

    token = _get_token_from_request(credentials, access_token)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await resolve_current_user(token)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, None otherwise.

    This dependency allows optional authentication.
    Returns None if no token provided, raises 401 only if token is invalid.
    Accepts token from either httpOnly cookie (browsers) or Authorization header (API clients).

    Args:
        credentials: HTTP Authorization credentials
        access_token: Token from httpOnly cookie

    Returns:
        Dict with user_id and username, or None if no token
    """
    settings = get_settings()

    # Return anonymous user if auth disabled. user_id=None（与 get_current_user
    # 一致）：None 是 falsy，不会触发按 user_id 过滤的查询。
    if not settings.auth_enabled:
        return {"user_id": None, "username": "anonymous"}

    token = _get_token_from_request(credentials, access_token)

    if not token:
        return None

    return await resolve_current_user(token)


async def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Require the current authenticated user to be an administrator."""
    if not current_user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
        )
    return current_user


def verify_resource_ownership(
    resource: Optional[Dict[str, Any]],
    current_user: Dict[str, Any],
    resource_name: str = "Resource"
) -> Dict[str, Any]:
    """Verify that the current user owns the resource.

    Args:
        resource: The resource dict (must have 'user_id' field)
        current_user: Current authenticated user dict
        resource_name: Name of the resource for error messages

    Returns:
        The resource if ownership verified

    Raises:
        HTTPException: 404 if resource not found
        HTTPException: 403 if user doesn't own the resource
    """
    if not resource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource_name} not found"
        )

    # Skip ownership check when auth is disabled (user_id is None or "anonymous")
    if not current_user.get("user_id") or current_user.get("user_id") == "anonymous":
        return resource

    # Check ownership
    resource_user_id = resource.get("user_id")
    if resource_user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized to access this {resource_name.lower()}"
        )

    return resource


def requires_tenant_provenance(current_user: Dict[str, Any]) -> bool:
    """Return whether storage references need user-owned provenance."""
    user_id = current_user.get("user_id")
    return bool(
        get_settings().auth_enabled
        and user_id
        and user_id != "anonymous"
    )


# Default allowed base directories for storage paths
DEFAULT_ALLOWED_DIRS = [
    "/data",
    "/app/data",
    "/app/models",
    "/app/output",
    "/app/datasets",
]


def validate_storage_path(
    path: str,
    allowed_dirs: Optional[List[str]] = None,
    resource_type: str = "resource"
) -> str:
    """
    Validate that a storage path is within allowed directories.

    Prevents path traversal attacks by ensuring the resolved path
    is within the allowed base directories.

    Args:
        path: The path to validate
        allowed_dirs: List of allowed base directories (uses DEFAULT_ALLOWED_DIRS if None)
        resource_type: Type of resource for error messages (e.g., "dataset", "model")

    Returns:
        The validated path

    Raises:
        HTTPException: 400 if path is invalid or outside allowed directories
    """
    if not path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{resource_type.capitalize()} path cannot be empty"
        )

    # Use default allowed dirs if not specified
    if allowed_dirs is None:
        allowed_dirs = DEFAULT_ALLOWED_DIRS

    # Resolve the path to handle .. and symlinks
    try:
        resolved_path = Path(path).resolve()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {resource_type} path: {str(e)}"
        )

    # Check if resolved path is within any allowed directory
    resolved_str = str(resolved_path)
    for allowed_dir in allowed_dirs:
        try:
            allowed_resolved = Path(allowed_dir).resolve()
            # Check if path starts with allowed directory
            if resolved_str.startswith(str(allowed_resolved) + os.sep) or resolved_str == str(allowed_resolved):
                return path
        except Exception:
            continue

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid {resource_type} path: must be within allowed directories ({', '.join(allowed_dirs)})"
    )
