"""
Authentication request/response schemas.
"""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field, model_validator


class AuthConfigResponse(BaseModel):
    """Public authentication configuration."""

    self_registration_enabled: bool
    direct_storage_registration_enabled: bool


class RegisterRequest(BaseModel):
    """User registration request."""

    username: str = Field(..., min_length=3, max_length=64, description="Username")
    password: str = Field(..., min_length=10, max_length=128, description="Password")
    email: Optional[EmailStr] = Field(
        default=None,
        max_length=256,
        description="Email address",
    )


class LoginRequest(BaseModel):
    """User login request."""

    username: str = Field(..., min_length=1, max_length=64, description="Username")
    password: str = Field(..., min_length=1, max_length=128, description="Password")


class ChangePasswordRequest(BaseModel):
    """Change password request."""

    old_password: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Current password",
    )
    new_password: str = Field(
        ...,
        min_length=10,
        max_length=128,
        description="New password",
    )


class AdminUserCreateRequest(BaseModel):
    """Administrator request to create an account."""

    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=10, max_length=128)
    email: Optional[EmailStr] = Field(default=None, max_length=256)
    is_admin: bool = False


class AccountFlagsUpdateRequest(BaseModel):
    """Administrator request to change account status or role."""

    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None

    @model_validator(mode="after")
    def require_at_least_one_flag(self):
        if self.is_active is None and self.is_admin is None:
            raise ValueError("At least one account flag must be provided")
        return self


class AdminPasswordResetRequest(BaseModel):
    """Administrator request to reset an account password."""

    new_password: str = Field(..., min_length=10, max_length=128)


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")


class UserResponse(BaseModel):
    """User information response."""

    user_id: str = Field(..., description="User unique identifier")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(default=None, description="Email address")
    is_active: bool = Field(..., description="Account is active")
    is_admin: bool = Field(..., description="User is admin")
    created_at: Optional[str] = Field(default=None, description="Account creation time")
    updated_at: Optional[str] = Field(default=None, description="Last account update time")


class UserListResponse(BaseModel):
    """Stable paginated administrator user-list response."""

    users: list[UserResponse]
    total: int
    limit: int
    offset: int


class MessageResponse(BaseModel):
    """Stable message-only response."""

    message: str


class CurrentUser(BaseModel):
    """Current authenticated user info (from token)."""
    user_id: str
    username: str
