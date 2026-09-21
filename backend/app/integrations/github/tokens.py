"""Decrypt stored GitHub access tokens for server-side use only."""

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings
from app.integrations.github.contents import GitHubAuthError


def decrypt_access_token(encrypted_token: str | None) -> str | None:
    """Return the plaintext token, or None when the account has no stored token."""

    if not encrypted_token:
        return None
    key = get_settings().token_encryption_key
    if key is None:
        raise GitHubAuthError("GitHub token could not be decrypted")
    try:
        return Fernet(key.get_secret_value().encode()).decrypt(encrypted_token.encode()).decode()
    except (InvalidToken, ValueError):
        raise GitHubAuthError("GitHub token could not be decrypted") from None


class TokenEncryptionError(RuntimeError):
    """GitHub tokens cannot be stored because TOKEN_ENCRYPTION_KEY is missing or invalid."""


def _fernet() -> Fernet:
    key = get_settings().token_encryption_key
    if key is None:
        raise TokenEncryptionError("TOKEN_ENCRYPTION_KEY is not configured")
    try:
        return Fernet(key.get_secret_value().encode())
    except (ValueError, TypeError):
        raise TokenEncryptionError("TOKEN_ENCRYPTION_KEY is invalid") from None


def require_token_encryption() -> None:
    """Raise TokenEncryptionError unless a usable encryption key is configured."""

    _fernet()


def encrypt_access_token(access_token: str) -> str:
    """Encrypt a token for storage. Never silently skips: a missing or invalid key raises."""

    return _fernet().encrypt(access_token.encode()).decode()
