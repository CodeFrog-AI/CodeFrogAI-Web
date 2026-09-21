"""Encrypt and decrypt user-supplied secrets (such as AI provider API keys) at rest.

Uses Fernet with the application's existing TOKEN_ENCRYPTION_KEY. This is deliberately a
separate module from the GitHub token helpers: it never skips silently. A missing or invalid
key, or a value that cannot be decrypted, raises `SecretBoxError`, whose message never contains
a key, a secret, or ciphertext.
"""

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings


class SecretBoxError(RuntimeError):
    """A secret cannot be encrypted or decrypted (no usable encryption key, or corrupt data)."""


def _fernet() -> Fernet:
    key = get_settings().token_encryption_key
    if key is None:
        raise SecretBoxError("TOKEN_ENCRYPTION_KEY is not configured")
    try:
        return Fernet(key.get_secret_value().encode())
    except (ValueError, TypeError):
        raise SecretBoxError("TOKEN_ENCRYPTION_KEY is invalid") from None


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except (InvalidToken, ValueError):
        raise SecretBoxError("The stored secret could not be decrypted") from None
