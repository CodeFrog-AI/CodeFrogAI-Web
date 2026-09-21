"""Validated, environment-backed application settings."""

from functools import lru_cache
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, HttpUrl, PostgresDsn, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_SCOPE_PATTERN = re.compile(r"[A-Za-z0-9:_.-]{1,64}")


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
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    workspace_root: Path | None = None
    agent_max_iterations: int = Field(default=6, ge=2, le=15)
    # Scopes requested from GitHub. `read:user` and `user:email` (identity) are always added by the
    # OAuth client; `repo` lets CodeFrog list private repositories, clone, push, and open pull requests.
    github_oauth_scopes: str = "read:user user:email repo"
    # The web frontend's origin: where the OAuth callback sends the browser, and the only CORS origin.
    frontend_url: str = "http://localhost:3000"

    @field_validator("github_oauth_scopes")
    @classmethod
    def validate_github_oauth_scopes(cls, value: str) -> str:
        """Accept space- or comma-separated scope names made of safe characters only."""

        scopes = value.replace(",", " ").split()
        if not scopes or any(_SCOPE_PATTERN.fullmatch(scope) is None for scope in scopes):
            raise ValueError("GITHUB_OAUTH_SCOPES must list valid scope names")
        return " ".join(dict.fromkeys(scopes))

    @field_validator("frontend_url")
    @classmethod
    def validate_frontend_url(cls, value: str) -> str:
        """An origin only (scheme, host, optional port): no credentials, path, query, or fragment."""

        parts = urlsplit(value.strip())
        try:
            port = parts.port
        except ValueError:
            raise ValueError("FRONTEND_URL must be a valid origin") from None
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
        ):
            raise ValueError("FRONTEND_URL must be an origin such as http://localhost:3000")
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        return f"{parts.scheme}://{host}" + (f":{port}" if port else "")

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
