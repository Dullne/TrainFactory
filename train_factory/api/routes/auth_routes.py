"""
Authentication API routes.

Provides endpoints for user registration, login, and profile management.

Security:
- Tokens are set as httpOnly cookies to prevent XSS attacks
- Also returned in response body for backward compatibility / non-browser clients
"""

import logging
from ipaddress import ip_address, ip_network
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from slowapi import Limiter
from starlette.concurrency import run_in_threadpool

from ...config import settings
from ...auth.jwt_handler import decode_token
from ...auth.schemas import (
    RegisterRequest,
    LoginRequest,
    ChangePasswordRequest,
    AccountFlagsUpdateRequest,
    AdminPasswordResetRequest,
    AdminUserCreateRequest,
    AuthConfigResponse,
    MessageResponse,
    TokenResponse,
    UserListResponse,
    UserResponse,
)
from ...auth.dependencies import get_current_user, require_admin
from ...auth.user_service import user_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Cookie settings
COOKIE_NAME = "access_token"
COOKIE_PATH = "/"
COOKIE_SAMESITE = "lax"


def _cookie_max_age() -> int:
    minutes = settings.jwt_access_token_expire_minutes
    if type(minutes) is not int or minutes <= 0:
        raise ValueError(
            "JWT access token expiration minutes must be a positive integer"
        )
    return minutes * 60


def _set_auth_cookie(
    response: Response,
    token: str,
    *,
    max_age: int | None = None,
) -> None:
    """Set httpOnly auth cookie."""
    resolved_max_age = _cookie_max_age() if max_age is None else max_age
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=resolved_max_age,
        path=COOKIE_PATH,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=settings.auth_cookie_secure,
    )


def _clear_auth_cookie(response: Response) -> None:
    """Clear the auth cookie using the same transport attributes."""
    response.delete_cookie(
        key=COOKIE_NAME,
        path=COOKIE_PATH,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=settings.auth_cookie_secure,
    )


def _set_authenticated_audit_identity(request: Request, token: str) -> None:
    """Expose a newly authenticated identity to the audit middleware."""
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        username = payload.get("username")
        if isinstance(user_id, str) and isinstance(username, str):
            request.state.audit_identity = {
                "user_id": user_id,
                "username": username,
            }
    except Exception as exc:
        logger.debug("Could not resolve newly authenticated audit identity: %s", exc)


def get_rate_limit_key(request: Request) -> str:
    """Return the client IP, honoring XFF only from explicitly trusted proxies."""
    remote_host = request.client.host if request.client else "unknown"
    try:
        remote_ip = ip_address(remote_host)
    except ValueError:
        return remote_host

    trusted_networks = [
        ip_network(entry, strict=False)
        for entry in settings.rate_limit_trusted_proxies.split(",")
        if entry
    ]
    if not any(remote_ip in network for network in trusted_networks):
        return str(remote_ip)

    for forwarded_host in request.headers.get("x-forwarded-for", "").split(","):
        try:
            return str(ip_address(forwarded_host.strip()))
        except ValueError:
            continue
    return str(remote_ip)


def _rate_limiting_disabled() -> bool:
    return not settings.rate_limit_enabled


def _registration_rate_limiting_disabled() -> bool:
    return _rate_limiting_disabled() or not settings.self_registration_enabled


# A default limit protects every API route; auth endpoints override it with
# tighter limits where credential guessing or account creation is involved.
limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[settings.rate_limit_default],
    enabled=settings.rate_limit_enabled,
)


@router.get("/config", response_model=AuthConfigResponse)
async def get_auth_config():
    """Return the public authentication configuration."""
    return AuthConfigResponse(
        self_registration_enabled=settings.self_registration_enabled,
        direct_storage_registration_enabled=not settings.auth_enabled,
    )


@router.post("/register", response_model=TokenResponse)
@limiter.limit(
    settings.rate_limit_register,
    exempt_when=_registration_rate_limiting_disabled,
)
async def register(request: Request, response: Response, body: RegisterRequest):
    """Register a new user account.

    Returns an access token upon successful registration.
    Token is also set as an httpOnly cookie for browser security.
    Rate limited to prevent abuse.
    """
    if not settings.self_registration_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-registration is disabled",
        )

    cookie_max_age = _cookie_max_age()
    try:
        result = await run_in_threadpool(
            user_service.register,
            username=body.username,
            password=body.password,
            email=body.email,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists",
        ) from None

    # Cookie configuration errors are server errors, not duplicate accounts.
    _set_auth_cookie(
        response,
        result["access_token"],
        max_age=cookie_max_age,
    )
    _set_authenticated_audit_identity(request, result["access_token"])
    return TokenResponse(
        access_token=result["access_token"],
        token_type=result["token_type"],
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit(
    settings.rate_limit_login,
    exempt_when=_rate_limiting_disabled,
)
async def login(request: Request, response: Response, body: LoginRequest):
    """Authenticate user and get access token.

    Token is also set as an httpOnly cookie for browser security.
    Rate limited to prevent brute force attacks.
    """
    cookie_max_age = _cookie_max_age()
    try:
        result = await run_in_threadpool(
            user_service.authenticate,
            username=body.username,
            password=body.password,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        ) from None

    # Cookie configuration errors are server errors, not credential failures.
    _set_auth_cookie(
        response,
        result["access_token"],
        max_age=cookie_max_age,
    )
    _set_authenticated_audit_identity(request, result["access_token"])
    return TokenResponse(
        access_token=result["access_token"],
        token_type=result["token_type"],
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current authenticated user's profile."""
    if current_user.get("user_id") in (None, "anonymous"):
        return UserResponse(
            user_id="anonymous",
            username=current_user.get("username") or "anonymous",
            email=None,
            is_active=True,
            is_admin=False,
            created_at=None,
        )

    user = await run_in_threadpool(user_service.get_user, current_user["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        user_id=user["user_id"],
        username=user["username"],
        email=user.get("email"),
        is_active=user["is_active"],
        is_admin=user["is_admin"],
        created_at=user.get("created_at"),
        updated_at=user.get("updated_at"),
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Revoke the account's active tokens and clear the auth cookie."""
    user_id = current_user.get("user_id")
    if user_id not in (None, "anonymous"):
        await run_in_threadpool(user_service.revoke_sessions, user_id)
    _clear_auth_cookie(response)
    return {"message": "Logged out successfully"}


@router.post("/change-password", response_model=MessageResponse)
@limiter.limit(
    settings.rate_limit_login,
    exempt_when=_rate_limiting_disabled,
)
async def change_password(
    request: Request,
    response: Response,
    body: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Change current user's password."""
    try:
        await run_in_threadpool(
            user_service.change_password,
            user_id=current_user["user_id"],
            old_password=body.old_password,
            new_password=body.new_password,
        )
        _clear_auth_cookie(response)
        return {
            "message": "Password changed successfully. Please log in again."
        }
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        ) from None


@router.get("/admin/users", response_model=UserListResponse)
async def list_users(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """List user accounts for account administrators."""
    users, total = await run_in_threadpool(
        user_service.list_users,
        limit=limit,
        offset=offset,
    )
    return {
        "users": users,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post(
    "/admin/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    body: AdminUserCreateRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """Create a regular or administrator account."""
    try:
        return await run_in_threadpool(
            user_service.create_user,
            username=body.username,
            password=body.password,
            email=body.email,
            is_admin=body.is_admin,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists",
        ) from None


@router.patch("/admin/users/{user_id}", response_model=UserResponse)
async def update_account_flags(
    user_id: str,
    body: AccountFlagsUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """Update an account's active or administrator flags."""
    try:
        return await run_in_threadpool(
            user_service.set_account_flags,
            user_id,
            is_active=body.is_active,
            is_admin=body.is_admin,
            acting_user_id=current_user["user_id"],
        )
    except ValueError as exc:
        if str(exc) == "User not found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            ) from None
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account update conflicts with safety rules",
        ) from None


@router.post(
    "/admin/users/{user_id}/reset-password",
    response_model=MessageResponse,
)
async def reset_user_password(
    user_id: str,
    body: AdminPasswordResetRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """Reset an account password and revoke its sessions."""
    try:
        await run_in_threadpool(
            user_service.reset_password,
            user_id,
            body.new_password,
        )
    except ValueError as exc:
        if str(exc) == "User not found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            ) from None
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to reset password",
        ) from None
    return {"message": "Password reset successfully"}
