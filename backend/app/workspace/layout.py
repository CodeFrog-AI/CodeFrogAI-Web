"""Where workspaces live on disk: WORKSPACE_ROOT/repositories/<repository-id>/ (the checkout)."""

import uuid
from pathlib import Path

from app.core.config import PROJECT_ROOT, get_settings

DEFAULT_WORKSPACE_DIRECTORY = ".codefrog-workspaces"


def get_workspace_root() -> Path:
    """WORKSPACE_ROOT if configured, else a git-ignored folder in the project directory."""

    configured = get_settings().workspace_root
    return Path(configured) if configured else PROJECT_ROOT / DEFAULT_WORKSPACE_DIRECTORY


def checkout_path(base_directory: Path, repository_id: uuid.UUID) -> Path:
    """The deterministic checkout directory for one repository (its id is a UUID, so it is a safe name)."""

    return Path(base_directory) / "repositories" / str(repository_id)
