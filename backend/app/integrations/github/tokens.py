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


def encrypt_access_token(access_token: str) -> str | None:
    """Encrypt a token for storage, or return None when no encryption key is configured."""

    key = get_settings().token_encryption_key
    if key is None:
        return None
    return Fernet(key.get_secret_value().encode()).encrypt(access_token.encode()).decode()
