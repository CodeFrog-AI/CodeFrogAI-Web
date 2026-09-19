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
