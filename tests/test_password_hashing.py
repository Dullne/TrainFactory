import bcrypt
import pytest
from argon2 import PasswordHasher

from train_factory.auth import password as password_module


def _legacy_bcrypt_hash(password_bytes: bytes) -> str:
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def test_new_hash_is_argon2id_and_supports_128_unicode_characters():
    password = "界" * 128
    different_suffix = "界" * 127 + "密"

    hashed = password_module.hash_password(password)

    assert hashed.startswith("$argon2id$")
    assert password_module.verify_password(password, hashed) is True
    assert password_module.verify_password(different_suffix, hashed) is False
    assert password_module.password_needs_rehash(hashed) is False


def test_argon2id_distinguishes_passwords_with_identical_first_72_bytes():
    correct = "a" * 72 + "correct"
    truncated_collision = "a" * 72 + "different"

    hashed = password_module.hash_password(correct)

    assert password_module.verify_password(correct, hashed) is True
    assert password_module.verify_password(truncated_collision, hashed) is False


@pytest.mark.parametrize("prefix", ("$2a$", "$2b$", "$2y$"))
def test_short_legacy_bcrypt_hashes_remain_verifiable_and_need_rehash(prefix):
    password = "LegacyPassword-2026"
    bcrypt_hash = _legacy_bcrypt_hash(password.encode("utf-8"))
    legacy_hash = prefix + bcrypt_hash[4:]

    assert password_module.verify_password(password, legacy_hash) is True
    assert password_module.password_needs_rehash(legacy_hash) is True


@pytest.mark.parametrize(
    "password",
    (
        "a" * 72 + "unsafe-suffix",
        "界" * 25,
    ),
)
def test_legacy_bcrypt_rejects_passwords_longer_than_72_utf8_bytes(password):
    password_bytes = password.encode("utf-8")
    assert len(password_bytes) > 72
    legacy_hash = _legacy_bcrypt_hash(password_bytes[:72])

    assert password_module.verify_password(password, legacy_hash) is False


@pytest.mark.parametrize(
    "malformed_hash",
    (
        None,
        "",
        "not-a-password-hash",
        "$argon2id$malformed",
        "$2b$12$short",
    ),
)
def test_malformed_and_unknown_hashes_are_safely_rejected(malformed_hash):
    assert password_module.verify_password("Password-2026", malformed_hash) is False


def test_unknown_mock_hash_does_not_request_rehash():
    assert password_module.password_needs_rehash("stored-hash") is False


def test_outdated_argon2id_parameters_verify_and_request_rehash():
    password = "ArgonPassword-2026"
    outdated_hash = PasswordHasher(
        time_cost=1,
        memory_cost=8192,
        parallelism=1,
    ).hash(password)

    assert password_module.verify_password(password, outdated_hash) is True
    assert password_module.password_needs_rehash(outdated_hash) is True
