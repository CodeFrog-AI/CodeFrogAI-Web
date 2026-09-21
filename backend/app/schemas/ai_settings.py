"""Schemas for a user's AI provider settings.

API keys arrive as `SecretStr` (never shown in reprs or logs) and are never returned by any
endpoint: responses carry only whether a key is configured and its last four characters.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

MIN_API_KEY_LENGTH = 8
MAX_API_KEY_LENGTH = 512
_MODEL_PATTERN = re.compile(r"[A-Za-z0-9._:/-]{1,128}")

# Validation messages are fixed strings: they never include the submitted value.
_KEY_MESSAGE = f"API keys must be {MIN_API_KEY_LENGTH}-{MAX_API_KEY_LENGTH} characters with no whitespace or control characters"
_MODEL_MESSAGE = "Model names may only contain letters, digits, and . _ : / - (up to 128 characters)"


def _validate_key(value: SecretStr | None) -> SecretStr | None:
    """None: omitted (unchanged). Empty: keep the existing key. Otherwise a well-formed key."""

    if value is None:
        return None
    secret = value.get_secret_value()
    if secret == "":
        return value
    if (
        not MIN_API_KEY_LENGTH <= len(secret) <= MAX_API_KEY_LENGTH
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in secret)
    ):
        raise ValueError(_KEY_MESSAGE)
    return value


def _validate_model(value: str | None) -> str | None:
    """None: omitted (unchanged). Empty: reset to the server's default model."""

    if value is None or value == "":
        return value
    if _MODEL_PATTERN.fullmatch(value) is None:
        raise ValueError(_MODEL_MESSAGE)
    return value


class AISettingsUpdate(BaseModel):
    """A partial update: an omitted field is unchanged, and the LLM and embedding sides are independent."""

    model_config = ConfigDict(extra="forbid")

    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_model: str | None = None

    _check_llm_key = field_validator("llm_api_key")(_validate_key)
    _check_embedding_key = field_validator("embedding_api_key")(_validate_key)
    _check_llm_model = field_validator("llm_model")(_validate_model)
    _check_embedding_model = field_validator("embedding_model")(_validate_model)


class ProviderSettingsResponse(BaseModel):
    """One provider's settings. `source` says whose key is used: the user's, the server's, or none."""

    api_key_configured: bool
    api_key_hint: str | None
    model: str
    source: Literal["user", "server", "none"]


class AISettingsResponse(BaseModel):
    llm: ProviderSettingsResponse
    embedding: ProviderSettingsResponse
