"""Structural checks for the Alembic migration configuration."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]


def test_alembic_configuration_uses_backend_migration_directory():
    config = Config(str(BACKEND_DIRECTORY / "alembic.ini"))
    script_directory = ScriptDirectory.from_config(config)

    assert Path(script_directory.dir) == BACKEND_DIRECTORY / "alembic"


def test_initial_migration_is_registered():
    config = Config(str(BACKEND_DIRECTORY / "alembic.ini"))
    revisions = list(ScriptDirectory.from_config(config).walk_revisions())

    assert len(revisions) == 1
    assert revisions[0].down_revision is None
