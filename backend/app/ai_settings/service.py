"""A user's own AI provider settings, and how they combine with the server's defaults.

Each user can bring an LLM key/model and an embedding key/model. The two sides are completely
independent (the LLM key is never used for embeddings or the reverse), and the user may enter
the same key on both sides or different ones. A user's own key overrides the server's `.env`
configuration; a side the user has not configured falls back to the server's provider.

Provider base URLs stay server-side (LLM_BASE_URL / EMBEDDING_BASE_URL): users cannot point
the server at another host. Keys are encrypted at rest, never returned, and never logged.
"""

import logging
from collections.abc import Callable
from typing import Literal

from sqlalchemy.orm import Session

from app.agent.llm import LLMNotConfiguredError, LLMProvider, OpenAICompatibleLLM
from app.core.config import get_settings
from app.core.exceptions import ServiceUnavailableError
from app.core.secret_box import SecretBoxError, decrypt_secret, encrypt_secret
from app.db.models import User, UserAISettings
from app.embeddings.provider import EmbeddingNotConfiguredError, EmbeddingProvider, OpenAIEmbeddingProvider
from app.schemas.ai_settings import AISettingsResponse, AISettingsUpdate, ProviderSettingsResponse

logger = logging.getLogger(__name__)

Side = Literal["llm", "embedding"]
HINT_LENGTH = 4
ENCRYPTION_UNAVAILABLE = "AI settings cannot be saved because encryption is not configured on the server"


def get_user_ai_settings(session: Session, user: User) -> UserAISettings | None:
    return session.query(UserAISettings).filter(UserAISettings.user_id == user.id).first()


def _key_hint(key: str) -> str:
    return key[-HINT_LENGTH:]


# ------------------------------------------------------------------ reading (never returns a key)


def _view(row: UserAISettings | None, side: Side) -> ProviderSettingsResponse:
    server = get_settings()
    encrypted = getattr(row, f"{side}_api_key_encrypted", None) if row else None
    hint = getattr(row, f"{side}_api_key_hint", None) if row else None
    model = getattr(row, f"{side}_model", None) if row else None
    server_key = server.llm_api_key if side == "llm" else server.embedding_api_key
    server_model = server.llm_model if side == "llm" else server.embedding_model
    if encrypted:
        source: Literal["user", "server", "none"] = "user"
    elif server_key is not None:
        source = "server"
    else:
        source = "none"
    return ProviderSettingsResponse(
        api_key_configured=bool(encrypted),
        api_key_hint=hint if encrypted else None,
        model=model or server_model,
        source=source,
    )


def build_settings_view(session: Session, user: User) -> AISettingsResponse:
    row = get_user_ai_settings(session, user)
    return AISettingsResponse(llm=_view(row, "llm"), embedding=_view(row, "embedding"))


# ------------------------------------------------------------------ writing


def update_ai_settings(session: Session, user: User, update: AISettingsUpdate) -> AISettingsResponse:
    """Apply a partial update. Omitted = unchanged; a blank key = keep the current key; a blank model = server default."""

    row = get_user_ai_settings(session, user)
    changes: dict[str, str | None] = {}
    for side, key, model in (
        ("llm", update.llm_api_key, update.llm_model),
        ("embedding", update.embedding_api_key, update.embedding_model),
    ):
        if key is not None and key.get_secret_value() != "":
            secret = key.get_secret_value()
            try:
                changes[f"{side}_api_key_encrypted"] = encrypt_secret(secret)
            except SecretBoxError:
                logger.error("AI settings cannot be saved: token encryption is not configured")
                raise ServiceUnavailableError(ENCRYPTION_UNAVAILABLE) from None
            changes[f"{side}_api_key_hint"] = _key_hint(secret)
        if model is not None:
            changes[f"{side}_model"] = model or None
    if row is None:
        row = UserAISettings(user_id=user.id)
        session.add(row)
    for column, value in changes.items():
        setattr(row, column, value)
    session.commit()
    logger.info("AI settings updated user_id=%s fields=%s", user.id, sorted(changes))  # field names only, never values
    return AISettingsResponse(llm=_view(row, "llm"), embedding=_view(row, "embedding"))


def remove_api_key(session: Session, user: User, side: Side) -> AISettingsResponse:
    """Delete one side's stored key (the other side is untouched). Its model setting stays."""

    row = get_user_ai_settings(session, user)
    if row is not None:
        setattr(row, f"{side}_api_key_encrypted", None)
        setattr(row, f"{side}_api_key_hint", None)
        session.commit()
        logger.info("AI settings key removed user_id=%s side=%s", user.id, side)
    return AISettingsResponse(llm=_view(row, "llm"), embedding=_view(row, "embedding"))


# ------------------------------------------------------------------ using the settings


def _user_key(row: UserAISettings | None, side: Side) -> str | None:
    """The user's decrypted key for this side, or None. A key that cannot be decrypted counts as not configured."""

    encrypted = getattr(row, f"{side}_api_key_encrypted", None) if row else None
    if not encrypted:
        return None
    try:
        return decrypt_secret(encrypted)
    except SecretBoxError:
        logger.warning("A stored AI key could not be decrypted side=%s", side)
        raise


def llm_provider_for(session: Session, user: User, fallback: Callable[[], LLMProvider]) -> LLMProvider:
    """The user's LLM provider (their key and/or model), else the server's (`fallback`).

    Raises LLMNotConfiguredError when there is no key to use (existing behavior: a 503).
    """

    row = get_user_ai_settings(session, user)
    model = row.llm_model if row else None
    try:
        key = _user_key(row, "llm")
    except SecretBoxError:
        raise LLMNotConfiguredError("LLM provider is not configured") from None
    if key is None and model is None:
        return fallback()
    server = get_settings()
    if key is None:  # the user chose a model but brought no key: use the server's key with it
        if server.llm_api_key is None:
            raise LLMNotConfiguredError("LLM provider is not configured")
        key = server.llm_api_key.get_secret_value()
    return OpenAICompatibleLLM(key, model or server.llm_model, server.llm_base_url)


def embedding_factory_for(
    session: Session, user: User, fallback: Callable[[], EmbeddingProvider]
) -> Callable[[], EmbeddingProvider]:
    """A factory for the user's embedding provider, else the server's (`fallback`).

    The factory is lazy, like the server's: it raises EmbeddingNotConfiguredError when called with no
    key available, which callers already turn into "not configured". It never uses the LLM key.
    """

    row = get_user_ai_settings(session, user)
    model = row.embedding_model if row else None
    encrypted = row.embedding_api_key_encrypted if row else None
    if not encrypted and model is None:
        return fallback

    def build() -> EmbeddingProvider:
        try:
            key = _user_key(row, "embedding")
        except SecretBoxError:
            raise EmbeddingNotConfiguredError("Embedding provider is not configured") from None
        server = get_settings()
        if key is None:
            if server.embedding_api_key is None:
                raise EmbeddingNotConfiguredError("Embedding provider is not configured")
            key = server.embedding_api_key.get_secret_value()
        return OpenAIEmbeddingProvider(key, model or server.embedding_model, server.embedding_base_url)

    return build
