"""
JWT token handling utilities.

Provides functions for creating and decoding JWT access tokens.
"""

from datetime import timedelta
from train_factory.core.time_utils import now_aware
from typing import Dict, Any

from jose import jwt, JWTError

from ..config.settings import get_settings


class TokenError(Exception):
    """Exception raised for token-related errors."""
    pass


def create_access_token(user_id: str, username: str, token_version: int) -> str:
    """Create a JWT access token.

    Args:
        user_id: User's unique identifier
        username: User's username
        token_version: User's current token version

    Returns:
        Encoded JWT token string
    """
    settings = get_settings()
    expire = now_aware() + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    payload = {
        "sub": user_id,
        "username": username,
        "ver": token_version,
        "exp": expire,
        "iat": now_aware(),
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        TokenError: If token is invalid or expired
    """
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        raise TokenError(f"Invalid token: {e}")
