"""Tests for validated, secret-safe application configuration."""

from pathlib import Path

import pytest

from app.core.config import ConfigurationError, Settings, get_settings


VALID_DATABASE_URL = "postgresql+psycopg://test_user:test_password@localhost:5432/test_db"


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Keep cached process settings isolated between environment tests."""

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
