"""The persistent local checkout that approved agent edits are written to."""

from app.workspace.layout import checkout_path, get_workspace_root
from app.workspace.lock import exclusive_workspace
from app.workspace.workspace import (
    FileChange,
    FileOperation,
    Workspace,
    WorkspaceError,
    WriteScope,
    validate_path,
)

__all__ = [
    "FileChange",
    "FileOperation",
    "Workspace",
    "WorkspaceError",
    "WriteScope",
    "checkout_path",
    "exclusive_workspace",
    "get_workspace_root",
    "validate_path",
]
