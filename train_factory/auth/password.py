"""Password hashing and legacy verification utilities."""

import re

import bcrypt
from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError


_ARGON2ID_PREFIX = "$argon2id$"
_BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$")
_BCRYPT_HASH_PATTERN = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")
_PASSWORD_HASHER = PasswordHasher(type=Type.ID)


def hash_password(password: str) -> str:
    """Hash a password using the current Argon2id parameters."""
    return _PASSWORD_HASHER.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify Argon2id or safely supported legacy bcrypt hashes."""
    if not isinstance(plain_password, str) or not isinstance(hashed_password, str):
        return False

    if hashed_password.startswith(_ARGON2ID_PREFIX):
        try:
            return _PASSWORD_HASHER.verify(hashed_password, plain_password)
        except (InvalidHashError, VerificationError, TypeError):
            return False

    if hashed_password.startswith(_BCRYPT_PREFIXES):
        password_bytes = plain_password.encode("utf-8")
        if len(password_bytes) > 72:
            return False
        try:
            return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))
        except (TypeError, ValueError):
            return False

    return False


def password_needs_rehash(hashed_password: str) -> bool:
    """Return whether a recognized valid hash should use current Argon2id."""
    if not isinstance(hashed_password, str):
        return False

    if _BCRYPT_HASH_PATTERN.fullmatch(hashed_password):
        return True

    if hashed_password.startswith(_ARGON2ID_PREFIX):
        try:
            return _PASSWORD_HASHER.check_needs_rehash(hashed_password)
        except (InvalidHashError, TypeError):
            return False

    return False
