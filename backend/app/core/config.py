"""Validated, environment-backed application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, PostgresDsn, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigurationError(RuntimeError):
    """Raised when safe application configuration validation fails."""


def _safe_validation_message(error: ValidationError) -> str:
    """Describe invalid setting names without including their submitted values."""

    fields = ", ".join(str(item["loc"][0]).upper() for item in error.errors())
    return f"Invalid application configuration: check {fields}."


class DatabaseSettings(BaseSettings):
    """Only the configuration needed to reach PostgreSQL.

    Database tooling (Alembic, the SQLAlchemy engine) must not depend on auth or
    GitHub OAuth credentials, so it loads this narrower model.
    """

    database_url: PostgresDsn

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings(DatabaseSettings):
    """Full application settings, validated at API startup and on OAuth/auth use."""

    app_name: str = "CodeFrog AI API"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    auth_secret_key: SecretStr = Field(min_length=32)
    auth_algorithm: Literal["HS256"] = "HS256"
    access_token_expire_minutes: int = Field(default=60, ge=1, le=1_440)
    github_client_id: str = Field(min_length=1)
    github_client_secret: SecretStr = Field(min_length=1)
    github_redirect_uri: HttpUrl
    token_encryption_key: SecretStr | None = None

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_app_environment(cls, value: object) -> object:
        """Allow conventional mixed-case environment values."""

        return value.lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, value: object) -> str:
        """Normalize and validate standard Python logging levels."""

        normalized = value.upper() if isinstance(value, str) else value
        if normalized not in VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(VALID_LOG_LEVELS))
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}")
        return normalized


@lru_cache
def get_database_settings() -> DatabaseSettings:
    """Return cached database-only settings, without leaking supplied values."""

    try:
        return DatabaseSettings()
    except ValidationError as error:
        raise ConfigurationError(_safe_validation_message(error)) from None


@lru_cache
def get_settings() -> Settings:
    """Return cached settings, without leaking invalid supplied values."""

    try:
        return Settings()
    except ValidationError as error:
        raise ConfigurationError(_safe_validation_message(error)) from None
