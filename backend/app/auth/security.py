"""Password hashing and JWT helpers used by local authentication only."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.core.config import get_settings


password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Return an Argon2 hash; callers must never persist the plaintext input."""

    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Safely verify a plaintext password against an Argon2 hash."""

    try:
        return password_hasher.verify(password_hash, password)
    except (InvalidHashError, VerifyMismatchError):
        return False


def create_access_token(user_id: UUID) -> str:
    """Create a short-lived access token containing only the user identifier."""

    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    return jwt.encode(
        {"sub": str(user_id), "exp": expires_at},
        settings.auth_secret_key.get_secret_value(),
        algorithm=settings.auth_algorithm,
    )


def decode_access_token(token: str) -> UUID | None:
    """Return a token subject only when signature and expiry are valid."""

    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.auth_secret_key.get_secret_value(),
            algorithms=[settings.auth_algorithm],
        )
        return UUID(str(payload["sub"]))
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
        return None
