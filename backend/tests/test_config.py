"""Tests for validated, secret-safe application configuration."""

from pathlib import Path

import pytest

from app.core import config
from app.core.config import (
    ConfigurationError,
    DatabaseSettings,
    Settings,
    get_database_settings,
    get_settings,
)
from app.db.database import create_database_engine


VALID_DATABASE_URL = "postgresql+psycopg://test_user:test_password@localhost:5432/test_db"
NON_DATABASE_VARIABLES = (
    "AUTH_SECRET_KEY",
    "GITHUB_CLIENT_ID",
    "GITHUB_CLIENT_SECRET",
    "GITHUB_REDIRECT_URI",
)


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Keep cached process settings isolated between environment tests."""

    get_settings.cache_clear()
    get_database_settings.cache_clear()
    yield
    get_settings.cache_clear()
    get_database_settings.cache_clear()


@pytest.fixture
def database_only_environment(monkeypatch):
    """Provide DATABASE_URL alone: no auth/GitHub variables and no real .env file."""

    for name in NON_DATABASE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    for model in (DatabaseSettings, Settings):
        monkeypatch.setattr(model, "model_config", {**model.model_config, "env_file": None})


def test_settings_load_valid_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    monkeypatch.setenv("APP_NAME", "CodeFrog Test API")
    monkeypatch.setenv("APP_ENV", "TEST")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = get_settings()

    assert settings.app_name == "CodeFrog Test API"
    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert str(settings.database_url) == VALID_DATABASE_URL


def test_missing_database_url_fails_with_safe_startup_error(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        Settings,
        "model_config",
        {**Settings.model_config, "env_file": None},
    )

    with pytest.raises(ConfigurationError, match="DATABASE_URL") as error:
        get_settings()

    assert "postgresql" not in str(error.value).lower()


def test_invalid_database_url_does_not_expose_credentials(monkeypatch):
    secret_url = "not-a-database-url://user:unsafe-password@example.test/database"
    monkeypatch.setenv("DATABASE_URL", secret_url)

    with pytest.raises(ConfigurationError) as error:
        get_settings()

    assert "DATABASE_URL" in str(error.value)
    assert "unsafe-password" not in str(error.value)
    assert secret_url not in str(error.value)


def test_invalid_log_level_fails_validation(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    with pytest.raises(ConfigurationError, match="LOG_LEVEL"):
        get_settings()


def test_environment_variables_override_dotenv_values(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql+psycopg://file_user:file_password@localhost:5432/file_db\n"
        "LOG_LEVEL=ERROR\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    monkeypatch.setenv("LOG_LEVEL", "warning")

    settings = Settings(_env_file=env_file)

    assert str(settings.database_url) == VALID_DATABASE_URL
    assert settings.log_level == "WARNING"


def test_dotenv_file_is_loaded_when_environment_is_unset(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"DATABASE_URL={VALID_DATABASE_URL}\nAPP_ENV=production\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)

    settings = Settings(_env_file=env_file)

    assert str(settings.database_url) == VALID_DATABASE_URL
    assert settings.app_env == "production"


def test_env_file_resolves_to_repository_root_dotenv():
    assert (config.PROJECT_ROOT / "backend" / "alembic.ini").is_file()
    assert Settings.model_config["env_file"] == config.PROJECT_ROOT / ".env"
    assert DatabaseSettings.model_config["env_file"] == config.PROJECT_ROOT / ".env"


def test_database_settings_load_without_auth_or_github_configuration(database_only_environment):
    settings = get_database_settings()

    assert str(settings.database_url) == VALID_DATABASE_URL


def test_database_engine_initializes_without_auth_or_github_configuration(
    database_only_environment,
):
    engine = create_database_engine()

    assert engine.url.database == "test_db"
    engine.dispose()


def test_missing_database_url_is_rejected_for_database_settings(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        DatabaseSettings, "model_config", {**DatabaseSettings.model_config, "env_file": None}
    )

    with pytest.raises(ConfigurationError, match="DATABASE_URL") as error:
        get_database_settings()

    assert "postgresql" not in str(error.value).lower()


def test_invalid_database_url_is_not_leaked_by_database_settings(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "not-a-database-url://user:unsafe-password@example.test/db")

    with pytest.raises(ConfigurationError) as error:
        get_database_settings()

    assert "unsafe-password" not in str(error.value)


def test_full_settings_still_require_auth_and_github_configuration(database_only_environment):
    with pytest.raises(ConfigurationError) as error:
        get_settings()

    message = str(error.value)
    for name in NON_DATABASE_VARIABLES:
        assert name in message
    assert "DATABASE_URL" not in message


def test_full_settings_load_with_complete_valid_configuration(database_only_environment, monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "a" * 32)
    monkeypatch.setenv("GITHUB_CLIENT_ID", "client-id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("GITHUB_REDIRECT_URI", "http://localhost:8000/api/v1/auth/github/callback")

    settings = get_settings()

    assert settings.github_client_secret.get_secret_value() == "client-secret"
    assert "client-secret" not in repr(settings)


def test_short_auth_secret_key_is_rejected_without_leaking_value(
    database_only_environment, monkeypatch
):
    monkeypatch.setenv("AUTH_SECRET_KEY", "too-short-unsafe-secret")
    monkeypatch.setenv("GITHUB_CLIENT_ID", "client-id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "unsafe-client-secret")
    monkeypatch.setenv("GITHUB_REDIRECT_URI", "not a url")

    with pytest.raises(ConfigurationError) as error:
        get_settings()

    message = str(error.value)
    assert "AUTH_SECRET_KEY" in message
    assert "GITHUB_REDIRECT_URI" in message
    assert "too-short-unsafe-secret" not in message
    assert "unsafe-client-secret" not in message
